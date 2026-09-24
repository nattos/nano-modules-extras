/*
 * filter.legacy.sphr_blur — "SPHR Blur" (v2 of the Resolume Wire patch).
 *
 * A seam-correct blur for equirectangular (sphere/dome) content — but the team
 * uses it constantly OFF-sphere too, for its distinctive edge-softening look,
 * so it's a plain filter here (no spherical-display assumption).
 *
 * Two stages, matching the source patch (Texture In → SPHR Expand ISF → Blur):
 *   1. SPHR Expand (expand.hlsl, ported ~verbatim) — a latitude-dependent
 *      HORIZONTAL blur: the horizontal radius grows toward the top/bottom of
 *      the frame (the poles of the implied sphere) and is narrow in the middle
 *      (the equator). This is the "sphere-aware" part.
 *   2. A plain separable Gaussian blur (shared fx::GaussianBlur) on top.
 *
 * v2 re-architecture (flagged per DNODE_MIGRATION_NOTES §3): the original
 * coupled one "Strength" knob into both stages through Wire Map/Curve nodes.
 * Here the two blurs are independent, playable knobs — `strength` drives the
 * sphere-aware horizontal expand, `gaussian` drives the isotropic softening on
 * top — which is friendlier live. `quality` sets the sample density of both.
 * The team's guidance was explicit: the math is approximate, the LOOK is what
 * matters; don't over-invest in correctness.
 *
 * Stateless (pure function of input + params): TimeIndependent, and is_identity
 * when both blurs are off.
 */

#include <gpu.h>
#include <host.h>
#include <effect_blur.h>
#include "sphr_blur_shaders.h"

#include <cstdint>

namespace sphr_blur {

static constexpr float EXPAND_SCALE = 0.6f; // strength=1 → blurSize 0.6 (ISF curve max)

struct Uniforms {
  float blur_size;
  float quality;
  float render_w;
  float _pad;
};
static_assert(sizeof(Uniforms) == 16, "Uniforms layout mismatch");

struct State {
  gpu::Buffer  uniform_buf;
  gpu::Sampler sampler;      // Repeat (equirect x-wrap) / Linear
  gpu::Texture expand_tex;   // per-instance expand output (gaussian reads it)
  int          ex_w = 0, ex_h = 0;
  bool         initialized = false;

  float strength = 0.25f;
  float gaussian = 0.4f;
  float quality  = 0.25f;
};

static gpu::ComputePSO  s_pso;
static fx::GaussianBlur s_blur;

void module_init() {
  state::init("filter.legacy.sphr_blur", {1, 0, 0},
    state::Schema()
      .floatField("strength", 0.25f, 0.0f, 1.0f, state::PrimaryInput, nullptr, 0.01f,
                  nullptr, "Sphere-aware horizontal blur — wider toward the top/bottom edges.")
      .floatField("gaussian", 0.4f,  0.0f, 1.0f, state::PrimaryInput, nullptr, 0.01f,
                  nullptr, "Isotropic Gaussian softening on top (scaled by strength).")
      .floatField("quality",  0.25f, 0.05f, 1.0f, state::PrimaryInput, nullptr, 0.01f,
                  nullptr, "Sample density of both blur stages (tuning).")
      .capability(state::Capability::TimeIndependent)
      .textureField("tex_in",  state::PrimaryInput)
      .textureField("tex_out", state::PrimaryOutput));

  if (gpu::Device::backend() == gpu::Backend::None) return;

  state::registerShaderSPV("sphr_blur_expand", EXPAND_SPV, EXPAND_SPV_SIZE);
  auto cs = gpu::Device::createShaderModuleByName("sphr_blur_expand");
  if (!cs) return;
  s_pso = gpu::Device::createComputePSO(cs, "main", gpu::Bindings()
      .tex2d(0).sampler(1).storageTex2d(2).uniform(3));
  s_blur.init();

  state::log("sphr_blur: module initialized");
}

void* create() {
  auto* s = new State();
  s->uniform_buf = gpu::Device::createBuffer(sizeof(Uniforms), gpu::BufferUsage::Uniform);
  s->sampler = gpu::Device::createSampler(gpu::FilterMode::Linear, gpu::AddressMode::Repeat);
  return s;
}

void destroy(void* self) {
  auto* s = static_cast<State*>(self);
  if (!s) return;
  s->uniform_buf.release();
  s->sampler.release();
  if (s->expand_tex.valid()) s->expand_tex.release();
  delete s;
}

void init(void* self) {
  auto* s = static_cast<State*>(self);
  if (!s) return;
  if (!s_pso.valid() || !s->uniform_buf.valid()) return;
  s->initialized = true;
}

void tick(void* self, double dt) { (void)self; (void)dt; }

void on_state_patched(void* self, int n, const char* pb, const int* off,
                      const int* len, const int* ops) {
  auto* s = static_cast<State*>(self);
  if (!s) return;
  for (int i = 0; i < n; i++) {
    if (ops[i] != state::PatchReplace) continue;
    const char* path = pb + off[i];
    int plen = len[i];
    if      (state::pathIs(path, plen, "strength")) s->strength = state::patchFloat(i);
    else if (state::pathIs(path, plen, "gaussian")) s->gaussian = state::patchFloat(i);
    else if (state::pathIs(path, plen, "quality"))  s->quality  = state::patchFloat(i);
  }
}

void on_resolume_param(void* self, long long, double) { (void)self; }

int32_t is_identity(void* self) {
  auto* s = static_cast<State*>(self);
  if (!s) return 0;
  // Both stages are gated by strength, so strength=0 ⇒ pure passthrough.
  return (s->strength <= 1e-3f) ? 1 : 0;
}

static void ensureExpandTex(State* s, int w, int h) {
  if (s->expand_tex.valid() && s->ex_w == w && s->ex_h == h) return;
  if (s->expand_tex.valid()) s->expand_tex.release();
  s->expand_tex = gpu::Device::createTexture(w, h);
  s->ex_w = w;
  s->ex_h = h;
}

void render(void* self, int vp_w, int vp_h) {
  auto* s = static_cast<State*>(self);
  if (!s || !s->initialized || vp_w <= 0 || vp_h <= 0) return;
  auto in  = gpu::Device::textureForField("tex_in");
  auto out = gpu::Device::textureForField("tex_out");
  if (!in.valid() || !out.valid()) return;

  const bool do_expand = s->strength > 1e-3f;

  gpu::Texture src = in;
  if (do_expand) {
    ensureExpandTex(s, vp_w, vp_h);
    if (!s->expand_tex.valid()) return;
    Uniforms u = {};
    u.blur_size = s->strength * EXPAND_SCALE;
    u.quality   = s->quality;
    u.render_w  = (float)vp_w;
    s->uniform_buf.writeOne(u);

    auto cp = gpu::ComputePass::begin();
    cp.setPSO(s_pso);
    cp.setTexture(in, 0, 0);
    cp.setSampler(s->sampler, 1);
    cp.setTexture(s->expand_tex, 2, 1);
    cp.setBuffer(s->uniform_buf, 3);
    cp.dispatch((vp_w + 7) / 8, (vp_h + 7) / 8);
    cp.end();
    src = s->expand_tex;
  }

  // Gaussian softening on top, gated by strength so strength=0 is a true
  // identity (the `gaussian` knob is the soften AMOUNT within the blur, not an
  // independent always-on blur). radius≈0 → a straight passthrough copy.
  s_blur.applyWithRadius(src, out, vp_w, vp_h, s->strength * s->gaussian, s->quality);
  gpu::Device::submit();
}

} // namespace sphr_blur
