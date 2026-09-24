/*
 * color.legacy.bicolor_grad — content-adaptive two-colour gradient.
 *
 * A port of the shipped NanoGraph "BicolorGrad". Three compute passes:
 *
 *   1. hist    — RGB→YIQ hue histogram over a coarse grid (atomic scatter
 *                into 64 bins, chroma-weighted).
 *   2. analyze — single-thread: pick the dominant "major" hue and an
 *                angularly-isolated "minor" hue, reconstruct saturated
 *                colours (blended toward a neutral by confidence), and locate
 *                the spatial centroids of each region to derive a gradient
 *                Center + Direction. Temporally smoothed across frames.
 *   3. render  — paint the MinorColor→MajorColor gradient (with a neutral
 *                mid-band) and composite it over the input.
 *
 * The analysis buffer persists per-instance, which is both the temporal
 * smoothing memory and why this effect is SeekableApproximate (not
 * TimeIndependent).
 *
 * Per-instance ABI: mutable state in `State`; the three PSOs + sampler-less
 * shared resources are file-static, compiled once in module_init().
 */

#include <gpu.h>
#include <host.h>
#include "bicolor_grad_shaders.h"

#include <cstdint>

namespace bicolor_grad {

static constexpr int   GRID_SN  = 64;     // histogram sample grid (per axis)
static constexpr int   BUCKETS  = 64;     // hue bins
static constexpr int   ANALYSIS_FLOATS = 16;

enum BlendMode : int { MODE_MIX = 0, MODE_MULTIPLY = 1, MODE_SCREEN = 2, MODE_ADD = 3 };

struct HistUniforms { float grid; float weight_scale; float _p0, _p1; };
struct AnalyzeUniforms {
  float nr, ng, nb, smoothing;
  float dir_sign, isolation, color_sat, _p;
};
struct RenderUniforms {
  float scale, blend, neutral_mix, midband;
  float nr, ng, nb, mode;
};

struct State {
  // Per-instance GPU resources.
  gpu::Buffer  hist_buf;       // 64 × int  (atomic histogram)
  gpu::Buffer  analysis_buf;   // 16 × float (persists → smoothing memory)
  gpu::Buffer  hist_uniform;
  gpu::Buffer  analyze_uniform;
  gpu::Buffer  render_uniform;
  gpu::Sampler sampler;
  bool initialized = false;

  // CPU mirrors of schema params.
  float nr = 0.05f, ng = 0.05f, nb = 0.06f;
  float smoothing   = 0.85f;
  float scale       = 1.0f;
  bool  reverse     = false;
  float blend       = 1.0f;
  int   mode        = MODE_MIX;
  float neutral_mix = 0.25f;
  float midband     = 0.2f;
  float isolation   = 0.3f;
  float color_sat   = 0.05f;
};

static gpu::ComputePSO s_pso_hist;
static gpu::ComputePSO s_pso_analyze;
static gpu::ComputePSO s_pso_render;

void module_init() {
  state::init("color.legacy.bicolor_grad", {1, 0, 0},
    state::Schema()
      // ---- Standard ----
      .rgbField  ("neutral",     0.05f, 0.05f, 0.06f,       state::PrimaryInput)
      .floatField("scale",       1.0f,  0.1f,  4.0f,        state::PrimaryInput)
      .floatField("blend",       1.0f,  0.0f,  1.0f,        state::PrimaryInput)
      .floatField("smoothing",   0.85f, 0.0f,  1.0f,        state::PrimaryInput)
      .boolField ("reverse",     false,                     state::PrimaryInput)
      .selectField("mode",       MODE_MIX,                  state::PrimaryInput, {
        {"Mix",      MODE_MIX},
        {"Multiply", MODE_MULTIPLY},
        {"Screen",   MODE_SCREEN},
        {"Add",      MODE_ADD},
      })
      // ---- Tuning ----
      .floatField("neutral_mix", 0.25f, 0.0f,  1.0f,        state::SecondaryInput)
      .floatField("midband",     0.2f,  0.0f,  1.0f,        state::SecondaryInput)
      .floatField("isolation",   0.3f,  0.05f, 1.0f,        state::SecondaryInput)
      .floatField("color_sat",   0.05f, 0.005f, 0.5f,       state::SecondaryInput)
      // ---- I/O ----
      .textureField("tex_in",  state::PrimaryInput)
      .textureField("tex_out", state::PrimaryOutput)
      .capability(state::Capability::SeekableApproximate)
  );

  if (gpu::Device::backend() == gpu::Backend::None) return;

  state::registerShaderSPV("bicolor_grad_hist",    HIST_SPV,    HIST_SPV_SIZE);
  state::registerShaderSPV("bicolor_grad_analyze", ANALYZE_SPV, ANALYZE_SPV_SIZE);
  state::registerShaderSPV("bicolor_grad_render",  RENDER_SPV,  RENDER_SPV_SIZE);

  auto cs_hist    = gpu::Device::createShaderModuleByName("bicolor_grad_hist");
  auto cs_analyze = gpu::Device::createShaderModuleByName("bicolor_grad_analyze");
  auto cs_render  = gpu::Device::createShaderModuleByName("bicolor_grad_render");
  if (!cs_hist || !cs_analyze || !cs_render) return;

  s_pso_hist = gpu::Device::createComputePSO(cs_hist, "main", gpu::Bindings()
      .tex2d(0).sampler(1).storageRW(2).uniform(3));
  s_pso_analyze = gpu::Device::createComputePSO(cs_analyze, "main", gpu::Bindings()
      .tex2d(0).sampler(1).storage(2).storageRW(3).uniform(4));
  s_pso_render = gpu::Device::createComputePSO(cs_render, "main", gpu::Bindings()
      .tex2d(0).sampler(1).storage(2).storageTex2d(3).uniform(4));

  state::log("bicolor_grad: module initialized");
}

void* create() {
  auto* s = new State();
  s->hist_buf        = gpu::Device::createBuffer(BUCKETS * sizeof(int32_t), gpu::BufferUsage::Storage);
  s->analysis_buf    = gpu::Device::createBuffer(ANALYSIS_FLOATS * sizeof(float), gpu::BufferUsage::Storage);
  s->hist_uniform    = gpu::Device::createBuffer(sizeof(HistUniforms), gpu::BufferUsage::Uniform);
  s->analyze_uniform = gpu::Device::createBuffer(sizeof(AnalyzeUniforms), gpu::BufferUsage::Uniform);
  s->render_uniform  = gpu::Device::createBuffer(sizeof(RenderUniforms), gpu::BufferUsage::Uniform);
  s->sampler         = gpu::Device::createSampler(gpu::FilterMode::Linear, gpu::AddressMode::ClampToEdge);
  return s;
}

void destroy(void* self) {
  auto* s = static_cast<State*>(self);
  if (!s) return;
  s->hist_buf.release();
  s->analysis_buf.release();
  s->hist_uniform.release();
  s->analyze_uniform.release();
  s->render_uniform.release();
  s->sampler.release();
  delete s;
}

void init(void* self) {
  auto* s = static_cast<State*>(self);
  if (!s) return;
  if (!s_pso_hist.valid() || !s_pso_analyze.valid() || !s_pso_render.valid()) return;
  if (!s->analysis_buf.valid()) return;

  // Seed the analysis buffer: neutral colours, centred, default direction,
  // initialized flag = 0 so the first analyze snaps instead of smoothing
  // against garbage.
  float seed[ANALYSIS_FLOATS] = {0};
  seed[0] = s->nr; seed[1] = s->ng; seed[2] = s->nb;     // major
  seed[3] = s->nr; seed[4] = s->ng; seed[5] = s->nb;     // minor
  seed[6] = 0.0f;  seed[7] = 0.0f;                       // center
  seed[8] = 1.0f;  seed[9] = 0.0f;                       // direction
  seed[10] = 0.0f;                                       // confidence
  seed[11] = 0.0f;                                       // not initialized
  s->analysis_buf.write(seed, ANALYSIS_FLOATS);
  s->initialized = true;
}

void tick(void* self, double dt) { (void)self; (void)dt; }
void on_resolume_param(void*, long long, double) {}

void on_state_patched(void* self, int n, const char* pb, const int* off,
                      const int* len, const int* ops) {
  auto* s = static_cast<State*>(self);
  if (!s) return;
  for (int i = 0; i < n; i++) {
    if (ops[i] != state::PatchReplace) continue;
    const char* p = pb + off[i];
    int l = len[i];
    if      (state::pathIs(p, l, "neutral"))     { auto v = state::patchVec3(i); s->nr = v.x; s->ng = v.y; s->nb = v.z; }
    else if (state::pathIs(p, l, "scale"))       s->scale = state::patchFloat(i);
    else if (state::pathIs(p, l, "blend"))       s->blend = state::patchFloat(i);
    else if (state::pathIs(p, l, "smoothing"))   s->smoothing = state::patchFloat(i);
    else if (state::pathIs(p, l, "reverse"))     s->reverse = state::patchBool(i);
    else if (state::pathIs(p, l, "mode"))        s->mode = state::patchInt(i);
    else if (state::pathIs(p, l, "neutral_mix")) s->neutral_mix = state::patchFloat(i);
    else if (state::pathIs(p, l, "midband"))     s->midband = state::patchFloat(i);
    else if (state::pathIs(p, l, "isolation"))   s->isolation = state::patchFloat(i);
    else if (state::pathIs(p, l, "color_sat"))   s->color_sat = state::patchFloat(i);
  }
}

void render(void* self, int vp_w, int vp_h) {
  auto* s = static_cast<State*>(self);
  if (!s || !s->initialized || vp_w <= 0 || vp_h <= 0) return;

  auto in  = gpu::Device::textureForField("tex_in");
  auto out = gpu::Device::textureForField("tex_out");
  if (!in.valid() || !out.valid()) return;

  // Clear the histogram (CPU upload, 256 bytes).
  int32_t zeros[BUCKETS] = {0};
  s->hist_buf.write(zeros, BUCKETS);

  HistUniforms hu = { (float)GRID_SN, 256.0f, 0.f, 0.f };
  s->hist_uniform.writeOne(hu);

  AnalyzeUniforms au = {};
  au.nr = s->nr; au.ng = s->ng; au.nb = s->nb;
  au.smoothing = s->smoothing;
  au.dir_sign  = s->reverse ? -1.0f : 1.0f;
  au.isolation = s->isolation;
  au.color_sat = s->color_sat;
  s->analyze_uniform.writeOne(au);

  RenderUniforms ru = {};
  ru.scale = s->scale; ru.blend = s->blend;
  ru.neutral_mix = s->neutral_mix; ru.midband = s->midband;
  ru.nr = s->nr; ru.ng = s->ng; ru.nb = s->nb;
  ru.mode = (float)s->mode;
  s->render_uniform.writeOne(ru);

  // Pass 1 — histogram.
  {
    auto cp = gpu::ComputePass::begin();
    cp.setPSO(s_pso_hist);
    cp.setTexture(in, 0, 0);
    cp.setSampler(s->sampler, 1);
    cp.setBuffer(s->hist_buf, 2);
    cp.setBuffer(s->hist_uniform, 3);
    cp.dispatch((GRID_SN + 7) / 8, (GRID_SN + 7) / 8);
    cp.end();
  }

  // Pass 2 — analyze (single thread).
  {
    auto cp = gpu::ComputePass::begin();
    cp.setPSO(s_pso_analyze);
    cp.setTexture(in, 0, 0);
    cp.setSampler(s->sampler, 1);
    cp.setBuffer(s->hist_buf, 2);
    cp.setBuffer(s->analysis_buf, 3);
    cp.setBuffer(s->analyze_uniform, 4);
    cp.dispatch(1, 1, 1);
    cp.end();
  }

  // Pass 3 — render.
  {
    auto cp = gpu::ComputePass::begin();
    cp.setPSO(s_pso_render);
    cp.setTexture(in, 0, 0);
    cp.setSampler(s->sampler, 1);
    cp.setBuffer(s->analysis_buf, 2);
    cp.setTexture(out, 3, 1);
    cp.setBuffer(s->render_uniform, 4);
    cp.dispatch((vp_w + 7) / 8, (vp_h + 7) / 8);
    cp.end();
  }

  gpu::Device::submit();
}

} // namespace bicolor_grad
