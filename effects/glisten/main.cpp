/*
 * filter.legacy.glisten — image-anchored sparkle fans.
 *
 * Faithful port of the shipped NanoGraph "Glisten" (decoded from
 * dnode/Assets/NanoGraph/Glisten.asset + Glisten.txt / GlistenFindAnchor.txt
 * and the Blur / ExpCurve subgraphs). Pipeline per frame:
 *
 *   1. downsample — input → fixed 64×64 search grid (RGBA16F).
 *   2. blur ×4    — separable weighted blur of the search grid, width
 *                   1−input_chaos, into a coarse (gain 1) and a fine
 *                   (gain 2/pass, saturating) copy. The anchor search runs
 *                   on brightness MASS, not pixel peaks.
 *   3. findanchor — 8×8 coarse + 8×8 fine argmax + gradient/colour taps
 *                   (single thread). Preserves the original's half-cell
 *                   anchor offset, R,B,G channel swap and hot colour adjust
 *                   — see findanchor.hlsl.
 *   4. fan render — levels × (blades−2) triangles: per level an inscribed
 *                   polygon disc fanned from a RIM vertex at the gradient
 *                   direction, additive into a half-res layer.
 *   5. blur ×2    — the sparkle layer itself is blurred (smoothing) with a
 *                   per-pass gain (contrast+1)·mix(ExpCurve(env), 1, sustain)
 *                   — the flicker pulses the layer's gain, not its geometry.
 *   6. composite  — out = input·input_alpha + layer·tint.
 *
 * EVERY internal texture in the original was 8-bit sRGB (GlobalBitDepth is
 * never assigned in dnode → C# default Int8 → BGRA8Unorm_sRGB in the
 * generated program). Two load-bearing consequences:
 *  - unorm clamping: the sparkle layer accumulates in [0,1]; its unclamped
 *    negative vertex colours carve into the layer's OWN glow (the deep-hue
 *    digging) but the layer never goes negative — the composite only ever
 *    ADDS light. Gains saturate instead of amplifying.
 *  - LINEAR-SPACE MATH: sRGB sampling decodes, sRGB writes encode, and
 *    render-target blending happens in linear. All the blur averaging,
 *    gains, gradient extraction and the fan's ×100 colour swings ran on
 *    linear values, re-encoded for display — that encode is what makes the
 *    hue digs pop. The fan target is a real RGBA8_SRGB texture (hardware
 *    linear blending + fine dark-end precision); the storage-written
 *    textures emulate it (WebGPU forbids sRGB storage): they hold codes,
 *    and the shaders decode after sampling / encode before writing.
 *
 * Flicker is a CPU envelope: Poisson-ish trigger (rate Hz), LINEAR release,
 * shaped by ExpCurve. Stateful → SeekableApproximate.
 */

#include <gpu.h>
#include <host.h>
#include "glisten_shaders.h"

#include <cmath>
#include <cstdint>

namespace glisten {

static constexpr int ANCHOR_FLOATS = 16;
static constexpr float PI = 3.14159265358979323846f;
static constexpr int SEARCH_N = 64;          // search grid size
// The original Blur subgraph's step at its baked Quality=0.695:
// step = 1 + 20·(1−q)².
static constexpr float SPARK_BLUR_STEP = 1.0f + 20.0f * (0.305f * 0.305f);

struct BlurUniforms {
  float dir_x, dir_y, half_width, gain;
  float taps, jitter, seed, decode_in;
};
struct FindUniforms {
  float color_grad_soft, color_grad_squash, color_grad_adjust, _p0;
};
struct VsUniforms {
  float aspect_x, blades, levels, size;
  float shape, stretch_grad, stretch_squash, stretch_x;
  float stretch_y, cg_power, intensity, _p0;
};
struct CompositeUniforms { float input_alpha, tint_r, tint_g, tint_b; };

struct State {
  gpu::Buffer  anchor_buf;
  gpu::Buffer  blur_uniform;
  gpu::Buffer  find_uniform;
  gpu::Buffer  vs_uniform;
  gpu::Buffer  comp_uniform;
  gpu::Sampler sampler;
  gpu::Texture t64_in, t64_tmp;          // 64² search grid + blur scratch
  gpu::Texture t64_coarse, t64_fine;     // blurred search copies (gain 1 / 2²)
  gpu::Texture spark_rt;                 // half-res fan target (RGBA8_SRGB)
  gpu::Texture spark_a, spark_b;         // blurred layer ping-pong (codes)
  int spark_w = 0, spark_h = 0;
  bool initialized = false;

  // CPU mirrors of schema params.
  float size          = 0.5f;
  int   blades        = 32;
  int   levels        = 32;
  float intensity     = 1.0f;
  float gradation_shape = 0.2f;
  float stretch_grad  = 1.0f;
  float stretch_grad_squash = 2.0f;
  float stretch_x = 1.0f, stretch_y = 1.0f;
  float color_grad_power = 0.5f;
  float color_grad_saturation = 0.5f;
  float color_grad_soft  = 0.2f;
  float color_grad_sharp = 0.5f;
  float tint_r = 1.0f, tint_g = 1.0f, tint_b = 1.0f;
  float smoothing     = 0.1f;
  float contrast      = 0.0f;
  float jitter        = 0.0f;
  float input_chaos   = 0.125f;
  float input_alpha   = 1.0f;
  float flicker_rate    = 5.0f;   // Hz
  float flicker_sustain = 0.5f;
  float flicker_release = 0.2f;   // seconds
  float flicker_curve   = 1.0f;

  // Flicker state.
  float    env = 0.0f;
  uint32_t rng = 0x9E3779B9u;
  uint32_t frame = 0;
};

static gpu::ComputePSO s_pso_downsample;
static gpu::ComputePSO s_pso_blur;
static gpu::ComputePSO s_pso_find;
static gpu::ComputePSO s_pso_composite;
static gpu::RenderPSO  s_pso_render;

static inline uint32_t lcg_next(uint32_t& s) { s = s * 1664525u + 1013904223u; return s; }
static inline float lcg_unit(uint32_t& s) { return (lcg_next(s) >> 8) * (1.0f / float(1u << 24)); }

// The original ExpCurve subgraph: (p^x − 1) / (sign(p−1)·max(|p−1|, 1e-4)).
static inline float exp_curve(float x, float p) {
  float d = p - 1.0f;
  float dn = (d >= 0.0f ? 1.0f : -1.0f) * std::fmax(std::fabs(d), 1e-4f);
  return (std::pow(p, x) - 1.0f) / dn;
}

// The original Blur subgraph's kernel: length = min(2·⌊1024·s/step⌋+1, 1024),
// width = length·step/1024 (uv). Tap count is capped (the kernel is smooth;
// fewer taps over the same width are visually identical) — the cap rescales
// nothing since weights are a function of normalized x.
static inline void blur_kernel(float strength, float step, int cap,
                               float& half_width, float& taps) {
  float len = 2.0f * std::floor(1024.0f * strength / step) + 1.0f;
  if (len > 1024.0f) len = 1024.0f;
  if (len < 1.0f) len = 1.0f;
  half_width = (len * step / 1024.0f) * 0.5f;
  taps = (len > (float)cap) ? (float)cap : len;
}

void module_init() {
  state::init("filter.legacy.glisten", {2, 0, 0},
    state::Schema()
      .helpField("intro",
        "Finds the brightest region of the image and anchors a layered "
        "sparkle fan there, coloured from the local colour gradient — bright "
        "edges throw deep, saturated glints that flicker and glow.")
      // ---- Sparkle ----
      .group("sparkle", "Sparkle")
        .groupHelp("The fan itself: stacked polygon discs anchored on the "
                   "brightest spot, shaded by the local colour gradient.")
      .floatField("size",       0.5f,  0.01f, 1.5f, state::PrimaryInput).label("Size", "Size")
      .intField  ("blades",     32,    3,     64,   state::PrimaryInput).label("Blades", "Blds")
      .intField  ("levels",     32,    1,     64,   state::PrimaryInput).label("Levels", "Lvls")
      .floatField("intensity",  1.0f,  0.0f,  2.0f, state::PrimaryInput).label("Intensity", "Int")
      .floatField("gradation_shape", 0.2f, 0.0f, 1.0f, state::SecondaryInput)
        .label("Gradation Shape", "Grad")
      // ---- Stretch ----
      .group("stretch", "Stretch")
        .groupHelp("Elongates the fan along the local image gradient.")
      .floatField("stretch_grad",        1.0f, 0.0f, 10.0f, state::SecondaryInput)
        .label("Stretch Grad", "SGrd")
      .floatField("stretch_grad_squash", 2.0f, 0.0f, 10.0f, state::SecondaryInput)
        .label("Stretch Squash", "SSq")
      .floatField("stretch_x",           1.0f, 0.0f, 10.0f, state::SecondaryInput)
        .label("Stretch X", "SX")
      .floatField("stretch_y",           1.0f, 0.0f, 10.0f, state::SecondaryInput)
        .label("Stretch Y", "SY")
      // ---- Colour ----
      .group("color", "Colour")
        .groupHelp("How hard the local colour gradient shades the fan. Power "
                   "runs hot by design — colours go negative and dig "
                   "complementary hues out of the image.")
      .floatField("color_grad_power",      0.5f, 0.0f, 1.0f, state::PrimaryInput)
        .label("Colour Power", "CPow")
      .rgbField  ("tint", 1.0f, 1.0f, 1.0f, state::PrimaryInput).label("Tint", "Tint")
      .floatField("color_grad_saturation", 0.5f, 0.0f, 1.0f, state::SecondaryInput)
        .label("Colour Saturation", "CSat")
      .floatField("color_grad_soft",       0.2f, 0.0f, 1.0f, state::SecondaryInput)
        .label("Colour Soft", "CSft")
      .floatField("color_grad_sharp",      0.5f, 0.0f, 1.0f, state::SecondaryInput)
        .label("Colour Sharp", "CShp")
      // ---- Glow ----
      .group("glow", "Glow")
        .groupHelp("The rendered fan is blurred into a glow layer; contrast "
                   "is the layer's gain (the flicker pulses it).")
      .floatField("smoothing", 0.1f,  0.0f, 1.0f, state::PrimaryInput).label("Smoothing", "Smth")
      .floatField("contrast",  0.0f, -1.0f, 5.0f, state::SecondaryInput).label("Contrast", "Ctr")
      .floatField("jitter",    0.0f,  0.0f, 1.0f, state::SecondaryInput).label("Jitter", "Jit")
      // ---- Flicker ----
      .group("flicker", "Flicker")
        .groupHelp("Random re-triggered envelope that pulses the glow gain. "
                   "Sustain is the resting gain (1 = no flicker).")
      .floatField("flicker_rate",    5.0f, 0.0f, 60.0f, state::PrimaryInput)
        .label("Flicker Rate", "FRat")
      .floatField("flicker_sustain", 0.5f, 0.0f, 1.0f,  state::SecondaryInput)
        .label("Flicker Sustain", "FSus")
      .floatField("flicker_release", 0.2f, 0.0f, 2.0f,  state::SecondaryInput)
        .label("Flicker Release", "FRel")
      .floatField("flicker_curve",   1.0f, 0.0f, 1.0f,  state::SecondaryInput)
        .label("Flicker Curve", "FCrv")
      // ---- Input ----
      .group("input", "Input")
        .groupHelp("Anchor search behaviour and base-image passthrough.")
      .floatField("input_chaos", 0.125f, 0.0f, 1.0f, state::SecondaryInput)
        .label("Input Chaos", "Chaos")
      .floatField("input_alpha", 1.0f,   0.0f, 1.0f, state::SecondaryInput)
        .label("Input Alpha", "InA")
      // ---- I/O ----
      .textureField("tex_in",  state::PrimaryInput)
      .textureField("tex_out", state::PrimaryOutput)
      .capability(state::Capability::SeekableApproximate)
  );

  if (gpu::Device::backend() == gpu::Backend::None) return;

  state::registerShaderSPV("glisten_downsample", DOWNSAMPLE_SPV, DOWNSAMPLE_SPV_SIZE,
                           "rgba8unorm", "write");
  state::registerShaderSPV("glisten_blur",       BLUR_SPV,       BLUR_SPV_SIZE,
                           "rgba8unorm", "write");
  state::registerShaderSPV("glisten_findanchor", FINDANCHOR_SPV, FINDANCHOR_SPV_SIZE);
  state::registerShaderSPV("glisten_composite",  COMPOSITE_SPV,  COMPOSITE_SPV_SIZE);
  state::registerShaderSPV("glisten_vs",         VS_SPV,         VS_SPV_SIZE);
  state::registerShaderSPV("glisten_fs",         FS_SPV,         FS_SPV_SIZE);

  auto cs_down = gpu::Device::createShaderModuleByName("glisten_downsample");
  auto cs_blur = gpu::Device::createShaderModuleByName("glisten_blur");
  auto cs_find = gpu::Device::createShaderModuleByName("glisten_findanchor");
  auto cs_comp = gpu::Device::createShaderModuleByName("glisten_composite");
  auto vs      = gpu::Device::createShaderModuleByName("glisten_vs");
  auto fs      = gpu::Device::createShaderModuleByName("glisten_fs");
  if (!cs_down || !cs_blur || !cs_find || !cs_comp || !vs || !fs) return;

  s_pso_downsample = gpu::Device::createComputePSO(cs_down, "main", gpu::Bindings()
      .tex2d(0).sampler(1).storageTex2d(2, gpu::TextureFormat::RGBA8));
  s_pso_blur = gpu::Device::createComputePSO(cs_blur, "main", gpu::Bindings()
      .tex2d(0).sampler(1).storageTex2d(2, gpu::TextureFormat::RGBA8).uniform(3));
  s_pso_find = gpu::Device::createComputePSO(cs_find, "main", gpu::Bindings()
      .tex2d(0).tex2d(1).sampler(2).storageRW(3).uniform(4));
  s_pso_composite = gpu::Device::createComputePSO(cs_comp, "main", gpu::Bindings()
      .tex2d(0).tex2d(1).sampler(2).storageTex2d(3).uniform(4));
  s_pso_render = gpu::Device::createInstancedRenderPSO(
      vs, "main", fs, "main",
      gpu::TextureFormat::RGBA8_SRGB,
      gpu::Bindings().storage(0).uniform(1),
      gpu::Device::BlendMode::Additive);

  state::log("glisten: module initialized");
}

void* create() {
  auto* s = new State();
  s->anchor_buf   = gpu::Device::createBuffer(ANCHOR_FLOATS * sizeof(float), gpu::BufferUsage::Storage);
  s->blur_uniform = gpu::Device::createBuffer(sizeof(BlurUniforms), gpu::BufferUsage::Uniform);
  s->find_uniform = gpu::Device::createBuffer(sizeof(FindUniforms), gpu::BufferUsage::Uniform);
  s->vs_uniform   = gpu::Device::createBuffer(sizeof(VsUniforms), gpu::BufferUsage::Uniform);
  s->comp_uniform = gpu::Device::createBuffer(sizeof(CompositeUniforms), gpu::BufferUsage::Uniform);
  s->sampler      = gpu::Device::createSampler(gpu::FilterMode::Linear, gpu::AddressMode::ClampToEdge);
  s->t64_in     = gpu::Device::createTexture(SEARCH_N, SEARCH_N, gpu::TextureFormat::RGBA8);
  s->t64_tmp    = gpu::Device::createTexture(SEARCH_N, SEARCH_N, gpu::TextureFormat::RGBA8);
  s->t64_coarse = gpu::Device::createTexture(SEARCH_N, SEARCH_N, gpu::TextureFormat::RGBA8);
  s->t64_fine   = gpu::Device::createTexture(SEARCH_N, SEARCH_N, gpu::TextureFormat::RGBA8);
  return s;
}

void destroy(void* self) {
  auto* s = static_cast<State*>(self);
  if (!s) return;
  s->anchor_buf.release();
  s->blur_uniform.release();
  s->find_uniform.release();
  s->vs_uniform.release();
  s->comp_uniform.release();
  s->sampler.release();
  s->t64_in.release();
  s->t64_tmp.release();
  s->t64_coarse.release();
  s->t64_fine.release();
  s->spark_rt.release();
  s->spark_a.release();
  s->spark_b.release();
  delete s;
}

void init(void* self) {
  auto* s = static_cast<State*>(self);
  if (!s) return;
  if (!s_pso_downsample.valid() || !s_pso_blur.valid() || !s_pso_find.valid() ||
      !s_pso_composite.valid() || !s_pso_render.valid()) return;
  s->env = 0.0f;
  s->rng = 0x9E3779B9u;
  s->frame = 0;
  s->initialized = true;
}

void tick(void* self, double dt) {
  auto* s = static_cast<State*>(self);
  if (!s) return;
  float fdt = (float)dt;
  // Original flicker: trigger with probability rate·dt, LINEAR release.
  if (lcg_unit(s->rng) < s->flicker_rate * fdt) {
    s->env = 1.0f;
  } else {
    s->env = std::fmax(s->env - fdt / std::fmax(s->flicker_release, 1e-3f), 0.0f);
  }
}

void on_resolume_param(void*, long long, double) {}

void on_state_patched(void* self, int n, const char* pb, const int* off,
                      const int* len, const int* ops) {
  auto* s = static_cast<State*>(self);
  if (!s) return;
  for (int i = 0; i < n; i++) {
    if (ops[i] != state::PatchReplace) continue;
    const char* p = pb + off[i];
    int l = len[i];
    if      (state::pathIs(p, l, "size"))            s->size = state::patchFloat(i);
    else if (state::pathIs(p, l, "blades"))          s->blades = state::patchInt(i);
    else if (state::pathIs(p, l, "levels"))          s->levels = state::patchInt(i);
    else if (state::pathIs(p, l, "intensity"))       s->intensity = state::patchFloat(i);
    else if (state::pathIs(p, l, "gradation_shape")) s->gradation_shape = state::patchFloat(i);
    else if (state::pathIs(p, l, "stretch_grad"))    s->stretch_grad = state::patchFloat(i);
    else if (state::pathIs(p, l, "stretch_grad_squash")) s->stretch_grad_squash = state::patchFloat(i);
    else if (state::pathIs(p, l, "stretch_x"))       s->stretch_x = state::patchFloat(i);
    else if (state::pathIs(p, l, "stretch_y"))       s->stretch_y = state::patchFloat(i);
    else if (state::pathIs(p, l, "color_grad_power")) s->color_grad_power = state::patchFloat(i);
    else if (state::pathIs(p, l, "color_grad_saturation")) s->color_grad_saturation = state::patchFloat(i);
    else if (state::pathIs(p, l, "color_grad_soft"))  s->color_grad_soft = state::patchFloat(i);
    else if (state::pathIs(p, l, "color_grad_sharp")) s->color_grad_sharp = state::patchFloat(i);
    else if (state::pathIs(p, l, "tint"))            { auto v = state::patchVec3(i); s->tint_r = v.x; s->tint_g = v.y; s->tint_b = v.z; }
    else if (state::pathIs(p, l, "smoothing"))       s->smoothing = state::patchFloat(i);
    else if (state::pathIs(p, l, "contrast"))        s->contrast = state::patchFloat(i);
    else if (state::pathIs(p, l, "jitter"))          s->jitter = state::patchFloat(i);
    else if (state::pathIs(p, l, "input_chaos"))     s->input_chaos = state::patchFloat(i);
    else if (state::pathIs(p, l, "input_alpha"))     s->input_alpha = state::patchFloat(i);
    else if (state::pathIs(p, l, "flicker_rate"))    s->flicker_rate = state::patchFloat(i);
    else if (state::pathIs(p, l, "flicker_sustain")) s->flicker_sustain = state::patchFloat(i);
    else if (state::pathIs(p, l, "flicker_release")) s->flicker_release = state::patchFloat(i);
    else if (state::pathIs(p, l, "flicker_curve"))   s->flicker_curve = state::patchFloat(i);
  }
}

static void runBlur(State* s, gpu::Texture& src, gpu::Texture& dst,
                    int w, int h, float dx, float dy,
                    float half_width, float gain, float taps, float jit,
                    float decode_in) {
  BlurUniforms bu = { dx, dy, half_width, gain, taps, jit, (float)(s->frame % 1024), decode_in };
  s->blur_uniform.writeOne(bu);
  auto cp = gpu::ComputePass::begin();
  cp.setPSO(s_pso_blur);
  cp.setTexture(src, 0, 0);
  cp.setSampler(s->sampler, 1);
  cp.setTexture(dst, 2, 2);
  cp.setBuffer(s->blur_uniform, 3);
  cp.dispatch((w + 7) / 8, (h + 7) / 8);
  cp.end();
}

void render(void* self, int vp_w, int vp_h) {
  auto* s = static_cast<State*>(self);
  if (!s || !s->initialized || vp_w <= 0 || vp_h <= 0) return;

  auto in  = gpu::Device::textureForField("tex_in");
  auto out = gpu::Device::textureForField("tex_out");
  if (!in.valid() || !out.valid()) return;

  // Half-res sparkle layer (matches the original's Half fragment target).
  int sw = (vp_w + 1) / 2, sh = (vp_h + 1) / 2;
  if (sw != s->spark_w || sh != s->spark_h || !s->spark_rt.valid()) {
    s->spark_rt.release();
    s->spark_a.release();
    s->spark_b.release();
    s->spark_rt = gpu::Device::createTexture(sw, sh, gpu::TextureFormat::RGBA8_SRGB);
    s->spark_a  = gpu::Device::createTexture(sw, sh, gpu::TextureFormat::RGBA8);
    s->spark_b  = gpu::Device::createTexture(sw, sh, gpu::TextureFormat::RGBA8);
    s->spark_w = sw; s->spark_h = sh;
  }

  // ---- derived params (the original's ExpCurve mappings) ----
  float cg_squash = 10.0f * exp_curve(s->color_grad_saturation, 10.0f);
  float cg_adjust = 100.0f * exp_curve(std::fmax(1.0f - s->color_grad_sharp, 2e-4f), 10.0f);
  float shape     = exp_curve(2.5f * s->gradation_shape, 10.0f);
  float cg_power  = s->color_grad_power * 100.0f;

  // Flicker → per-blur-pass gain of the sparkle layer.
  float curve_p  = 32.0f * s->flicker_curve + 0.001f;
  float env_c    = exp_curve(s->env, curve_p);
  float gain     = (s->contrast + 1.0f) *
                   (env_c + (1.0f - env_c) * s->flicker_sustain);
  gain = std::fmax(gain, 0.0f);

  // Blur kernels.
  float search_hw, search_taps;
  blur_kernel(std::fmax(1.0f - s->input_chaos, 0.0f), 1.0f, 129, search_hw, search_taps);
  float spark_hw, spark_taps;
  blur_kernel(std::fmax(s->smoothing, 0.005f), SPARK_BLUR_STEP, 255, spark_hw, spark_taps);

  FindUniforms fu = { s->color_grad_soft, cg_squash, cg_adjust, 0.0f };
  s->find_uniform.writeOne(fu);

  int nBlades = s->blades < 3 ? 3 : s->blades;
  int nLevels = s->levels < 1 ? 1 : s->levels;
  float min_over_w = float(vp_w < vp_h ? vp_w : vp_h) / float(vp_w);  // min(h/w, 1)

  VsUniforms vu = {};
  vu.aspect_x = min_over_w;
  vu.blades = (float)nBlades;
  vu.levels = (float)nLevels;
  vu.size = s->size;
  vu.shape = shape;
  vu.stretch_grad = s->stretch_grad;
  vu.stretch_squash = s->stretch_grad_squash;
  vu.stretch_x = s->stretch_x;
  vu.stretch_y = s->stretch_y;
  vu.cg_power = cg_power;
  vu.intensity = s->intensity;
  s->vs_uniform.writeOne(vu);

  CompositeUniforms cu = { s->input_alpha, s->tint_r, s->tint_g, s->tint_b };
  s->comp_uniform.writeOne(cu);

  // Pass 1 — downsample input to the 64² search grid.
  {
    auto cp = gpu::ComputePass::begin();
    cp.setPSO(s_pso_downsample);
    cp.setTexture(in, 0, 0);
    cp.setSampler(s->sampler, 1);
    cp.setTexture(s->t64_in, 2, 2);
    cp.dispatch(SEARCH_N / 8, SEARCH_N / 8);
    cp.end();
  }

  // Passes 2–5 — blur the search grid twice: coarse copy (gain 1) and fine
  // copy (gain 2 per pass, saturating — the original's BlurF contrast=1).
  // Same blur width for both. All texels are sRGB codes; the blur decodes,
  // averages/gains in linear, re-encodes.
  runBlur(s, s->t64_in,  s->t64_tmp,    SEARCH_N, SEARCH_N, 1, 0, search_hw, 1.0f, search_taps, 0.0f, 1.0f);
  runBlur(s, s->t64_tmp, s->t64_coarse, SEARCH_N, SEARCH_N, 0, 1, search_hw, 1.0f, search_taps, 0.0f, 1.0f);
  runBlur(s, s->t64_in,  s->t64_tmp,    SEARCH_N, SEARCH_N, 1, 0, search_hw, 2.0f, search_taps, 0.0f, 1.0f);
  runBlur(s, s->t64_tmp, s->t64_fine,   SEARCH_N, SEARCH_N, 0, 1, search_hw, 2.0f, search_taps, 0.0f, 1.0f);

  // Pass 6 — find the anchor.
  {
    auto cp = gpu::ComputePass::begin();
    cp.setPSO(s_pso_find);
    cp.setTexture(s->t64_coarse, 0, 0);
    cp.setTexture(s->t64_fine, 1, 1);
    cp.setSampler(s->sampler, 2);
    cp.setBuffer(s->anchor_buf, 3);
    cp.setBuffer(s->find_uniform, 4);
    cp.dispatch(1, 1, 1);
    cp.end();
  }

  // Pass 7 — render the fan layers (additive, half-res, cleared). The
  // RGBA8_SRGB target blends in LINEAR and clamps the accumulation to
  // [0,1]: negatives carve the layer's own glow, never below zero.
  {
    int instances = nLevels * (nBlades - 2);
    auto rp = gpu::RenderPass::begin(s->spark_rt, 0, 0, 0, 0);
    rp.setPSO(s_pso_render);
    rp.setBuffer(s->anchor_buf, 0);
    rp.setBuffer(s->vs_uniform, 1);
    if (s->intensity > 0.0f && instances > 0) rp.draw(3, instances);
    rp.end();
  }

  // Passes 8+9 — blur the sparkle layer with the flicker gain per pass.
  // Pass 8's input is the sRGB target (hardware-decoded → decode_in 0);
  // pass 9 reads the code-holding scratch (decode_in 1).
  runBlur(s, s->spark_rt, s->spark_a, sw, sh, 1, 0, spark_hw, gain, spark_taps, s->jitter, 0.0f);
  runBlur(s, s->spark_a,  s->spark_b, sw, sh, 0, 1, spark_hw, gain, spark_taps, s->jitter, 1.0f);

  // Pass 10 — composite in linear: out = enc(dec(in)·input_alpha +
  // dec(layer)·tint). The layer is ≥ 0, so this only ever brightens.
  {
    auto cp = gpu::ComputePass::begin();
    cp.setPSO(s_pso_composite);
    cp.setTexture(in, 0, 0);
    cp.setTexture(s->spark_b, 1, 1);
    cp.setSampler(s->sampler, 2);
    cp.setTexture(out, 3, 3);
    cp.setBuffer(s->comp_uniform, 4);
    cp.dispatch((vp_w + 7) / 8, (vp_h + 7) / 8);
    cp.end();
  }

  s->frame++;
  gpu::Device::submit();
}

} // namespace glisten
