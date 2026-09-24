/*
 * filter.lights_sim — 4 vertical LED bars sampled from the input (Resolume-style).
 *
 * The input is split into 4 vertical quarters (one LED bar each). Each bar is
 * divided into `segments` LED segments stacked vertically. A segment's colour
 * is sampled from the input at the HORIZONTAL CENTRE of its quarter and the
 * vertical centre of its segment (the standard fixture-sampling point). The
 * bars are then rendered INSET into their quarters (separate horizontal /
 * vertical inset), over the input faded by `input_opacity`.
 *
 * Stateless: a single compute pass, no persistent buffer, no readback. Class-
 * like instance model (module_init compiles the PSO + schema once; each chain
 * entry gets its own State via create()).
 */

#include <gpu.h>
#include <host.h>
#include "lights_sim_shaders.h"

#include <cstdint>

namespace lights_sim {

struct Uniforms {
  int32_t segments;
  float   inset_h;
  float   inset_v;
  float   input_opacity;
};
static_assert(sizeof(Uniforms) == 16, "Uniforms layout mismatch");

struct State {
  gpu::Buffer  uniform_buf;
  gpu::Sampler sampler;
  bool initialized = false;

  // Schema-mirrored params.
  int   segments      = 13;
  float inset_h       = 0.8f;
  float inset_v       = 0.05f;
  float input_opacity = 0.25f;
};

static gpu::ComputePSO s_pso;

static inline float clampf(float v, float lo, float hi) {
  return v < lo ? lo : (v > hi ? hi : v);
}

void module_init() {
  state::init("filter.lights_sim", {1, 0, 1},
    state::Schema()
      .helpField("intro",
        "## Lights Sim\n"
        "Turns your input into a **Resolume-style LED wall** — four vertical bars, each "
        "split into stacked LED *Segments*, sampling their colour from the underlying "
        "image. It's a filter: the source shows through, faded by *Input Opacity*.\n\n"
        "**Try:** drop *Segments* low for chunky pixel-strip bars, or push it high for a "
        "smooth column read; tuck the bars into their quarters with the *Inset* knobs and "
        "lower *Input Opacity* to let the raw footage bleed between them.")
      // --- LED bars ---
      .group("bars", "LED Bars")
        .groupHelp(
          "Four vertical bars sample the input at the centre of each quarter. *Segments* "
          "sets the vertical LED resolution; the *Inset* pair pads the bars inside their "
          "quarters (horizontal gap vs. top/bottom gap); *Input Opacity* fades the "
          "underlying source behind them.")
      .intField  ("segments",      13, 1, 256,         state::PrimaryInput).label("Segments", "Segs")
      .floatField("inset_h",       0.8f,  0.0f, 1.0f,  state::PrimaryInput).label("Horizontal Inset", "InsetH")
      .floatField("inset_v",       0.05f, 0.0f, 1.0f,  state::PrimaryInput).label("Vertical Inset", "InsetV")
      .floatField("input_opacity", 0.25f, 0.0f, 1.0f,  state::PrimaryInput).label("Input Opacity", "InOpac")
      .capability(state::Capability::TimeIndependent)
      .textureField("tex_in",  state::PrimaryInput)
      .textureField("tex_out", state::PrimaryOutput)
  );

  if (gpu::Device::backend() == gpu::Backend::None) return;

  state::registerShaderSPV("lights_sim_render", RENDER_SPV, RENDER_SPV_SIZE);
  auto cs = gpu::Device::createShaderModuleByName("lights_sim_render");
  if (!cs) return;

  s_pso = gpu::Device::createComputePSO(cs, "main", gpu::Bindings()
      .tex2d(0)
      .sampler(1)
      .storageTex2d(2)
      .uniform(3));

  state::log("lights_sim: module initialized");
}

void* create() {
  auto* s = new State();
  s->uniform_buf = gpu::Device::createBuffer(sizeof(Uniforms), gpu::BufferUsage::Uniform);
  s->sampler = gpu::Device::createSampler(gpu::FilterMode::Linear, gpu::AddressMode::ClampToEdge);
  return s;
}

void destroy(void* self) {
  auto* s = static_cast<State*>(self);
  if (!s) return;
  s->uniform_buf.release();
  s->sampler.release();
  delete s;
}

void init(void* self) {
  auto* s = static_cast<State*>(self);
  if (!s) return;
  if (!s_pso.valid() || !s->uniform_buf.valid()) return;
  s->initialized = true;
}

void tick(void*, double) {}

void on_state_patched(void* self, int n, const char* pb, const int* off,
                      const int* len, const int* ops) {
  auto* s = static_cast<State*>(self);
  if (!s) return;
  for (int i = 0; i < n; i++) {
    if (ops[i] != state::PatchReplace) continue;
    const char* path = pb + off[i];
    int plen = len[i];
    if      (state::pathIs(path, plen, "segments"))      s->segments      = (int)state::patchFloat(i);
    else if (state::pathIs(path, plen, "inset_h"))       s->inset_h       = state::patchFloat(i);
    else if (state::pathIs(path, plen, "inset_v"))       s->inset_v       = state::patchFloat(i);
    else if (state::pathIs(path, plen, "input_opacity")) s->input_opacity = state::patchFloat(i);
  }
}

void render(void* self, int vp_w, int vp_h) {
  auto* s = static_cast<State*>(self);
  if (!s || !s->initialized || vp_w <= 0 || vp_h <= 0) return;
  auto in  = gpu::Device::textureForField("tex_in");
  auto out = gpu::Device::textureForField("tex_out");
  if (!in.valid() || !out.valid()) return;

  Uniforms u = {};
  u.segments      = s->segments < 1 ? 1 : s->segments;
  u.inset_h       = clampf(s->inset_h, 0.0f, 1.0f);
  u.inset_v       = clampf(s->inset_v, 0.0f, 1.0f);
  u.input_opacity = clampf(s->input_opacity, 0.0f, 1.0f);
  s->uniform_buf.writeOne(u);

  auto cp = gpu::ComputePass::begin();
  cp.setPSO(s_pso);
  cp.setTexture(in, 0, 0);
  cp.setSampler(s->sampler, 1);
  cp.setTexture(out, 2, 1);
  cp.setBuffer(s->uniform_buf, 3);
  cp.dispatch((vp_w + 7) / 8, (vp_h + 7) / 8);
  cp.end();

  gpu::Device::submit();
}

} // namespace lights_sim
