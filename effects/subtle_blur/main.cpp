/*
 * filter.legacy.subtle_blur — "Subtle Blur" (v2 of the Resolume Wire patch).
 *
 * A light Gaussian blur followed by a slowly-drifting chromatic colour offset:
 * a soft bloom with a faint, shifting RGB fringe on edges. Good for breaking
 * up sharp edges and giving a gentle "alive" softness.
 *
 * Source patch (Wire/Patches/Subtle Blur, 24 nodes): Texture In → Blur →
 * Hue Rotate(H) → Color Offset → Hue Rotate(1-H) → Out. The Color Offset
 * resampled R/G/B at three independently-RANDOMIZED 2D directions (re-rolled
 * on a Randomize trigger), each scaled by an "Amount" knob, with the blue
 * channel's X additionally drifting via a Saw whose amplitude was "Movement".
 * Exposed knobs: Hue Rotate, Amount, Movement, Blur.
 *
 * v2 RE-ARCHITECTURE (flagged per DNODE_MIGRATION_NOTES §3):
 *  - The three randomized direction vectors + the inverse Hue-Rotate sandwich
 *    are replaced by ONE rotating basis of three equal-magnitude offsets 120°
 *    apart (prismatic, even split). `hue` biases the basis angle; the basis
 *    DRIFTS continuously via an accumulator (style guide §2.1) at a rate set
 *    by `movement` — the same "slowly shifting" feel without per-frame RNG.
 *  - Blur is the shared fx::GaussianBlur (no-shimmer separable kernel) rather
 *    than the patch's fixed-distance Blur node.
 *
 * Pipeline: blur(tex_in → blur_tex) then chroma(blur_tex → tex_out). When the
 * chroma amount is ~0 the blur writes straight to tex_out; when both blur and
 * amount are ~0 the effect reports is_identity and the executor skips it.
 *
 * Per-instance instance ABI (class-like): module_init() compiles the shared
 * chroma PSO + publishes the schema once per type; each chain entry gets its
 * own State (params, drift accumulator, uniform buffer, blur output texture).
 */

#include <gpu.h>
#include <host.h>
#include <effect_blur.h>
#include "subtle_blur_shaders.h"

#include <cmath>
#include <cstdint>

namespace subtle_blur {

static constexpr float TAU             = 6.28318530717958647692f;
static constexpr float OFFSET_SCALE    = 0.02f; // amount=1 → 2% of short axis (the slide DISTANCE)
static constexpr float MOVE_RATE_SCALE = 2.5f;  // movement=1 → 2.5 sawtooth resets/sec (the RATE)
static constexpr float BLUR_SCALE      = 0.35f; // blur=1 → a fraction of the GaussianBlur ceiling
// Fixed spatial slant of the split (a gentle diagonal). `hue` rotates the
// COLOUR basis, not this direction.
static constexpr float SLANT_X         =  0.70710678f;
static constexpr float SLANT_Y         = -0.70710678f; // tilt diagonally UP (uv y is down)

struct Uniforms {
  float off_x;
  float off_y;
  float hue_shift;
  float _pad;
};
static_assert(sizeof(Uniforms) == 16, "Uniforms layout mismatch");

struct State {
  gpu::Buffer  uniform_buf;
  gpu::Sampler sampler;
  gpu::Texture blur_tex;        // per-instance blur output (chroma reads it)
  int          blur_w = 0, blur_h = 0;
  bool         initialized = false;

  // Schema-mirrored params.
  float blur     = 0.09f;
  float amount   = 0.15f;
  float movement = 1.0f;
  float hue      = 0.22f;  // the Wire patch's exposed Hue Rotate default (the slant)
  float quality  = 0.3f;   // lower = sparser taps = a "harder" (less smooth) blur

  // Runtime: sawtooth phase (ramps then hard-resets) for `movement`.
  double saw_phase = 0.0;
};

// Type-shared, compiled once.
static gpu::ComputePSO   s_pso;
static fx::GaussianBlur  s_blur;

void module_init() {
  state::init("filter.legacy.subtle_blur", {1, 0, 0},
    state::Schema()
      .floatField("blur",     0.09f, 0.0f, 1.0f, state::PrimaryInput, nullptr, 0.01f,
                  nullptr, "Blur amount.")
      .floatField("amount",   0.15f, 0.0f, 1.0f, state::PrimaryInput, nullptr, 0.01f,
                  nullptr, "Chromatic offset distance — RGB fringe width along the slant.")
      .floatField("movement", 1.0f,  0.0f, 1.0f, state::PrimaryInput, nullptr, 0.01f,
                  nullptr, "Sawtooth RATE — how fast the offset sweeps + hard-resets (0 = static).")
      .floatField("hue",      0.22f, 0.0f, 1.0f, state::PrimaryInput, nullptr, 0.01f,
                  nullptr, "Angle of the slanted split axis (0..1 = full turn).")
      .floatField("quality",  0.3f,  0.05f, 1.0f, state::PrimaryInput, nullptr, 0.01f,
                  nullptr, "Blur sample density — lower is harder/less smooth (tuning).")
      .capability(state::Capability::SeekableApproximate)
      .textureField("tex_in",  state::PrimaryInput)
      .textureField("tex_out", state::PrimaryOutput));

  if (gpu::Device::backend() == gpu::Backend::None) return;

  state::registerShaderSPV("subtle_blur_chroma", CHROMA_SPV, CHROMA_SPV_SIZE);
  auto cs = gpu::Device::createShaderModuleByName("subtle_blur_chroma");
  if (!cs) return;
  s_pso = gpu::Device::createComputePSO(cs, "main", gpu::Bindings()
      .tex2d(0).sampler(1).storageTex2d(2).uniform(3));
  s_blur.init();

  state::log("subtle_blur: module initialized");
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
  if (s->blur_tex.valid()) s->blur_tex.release();
  delete s;
}

void init(void* self) {
  auto* s = static_cast<State*>(self);
  if (!s) return;
  s->saw_phase = 0.0;
  if (!s_pso.valid() || !s->uniform_buf.valid()) return;
  s->initialized = true;
}

void tick(void* self, double dt) {
  auto* s = static_cast<State*>(self);
  if (!s) return;
  if (dt < 0.0) dt = 0.0;
  // `movement` is the RATE: it advances the sawtooth phase. frac() in render
  // gives the ramp + hard periodic reset; movement=0 freezes it (static offset).
  s->saw_phase += dt * (double)s->movement * (double)MOVE_RATE_SCALE;
  if (s->saw_phase > 1.0e6) s->saw_phase = std::fmod(s->saw_phase, 1.0);
}

void on_state_patched(void* self, int n, const char* pb, const int* off,
                      const int* len, const int* ops) {
  auto* s = static_cast<State*>(self);
  if (!s) return;
  for (int i = 0; i < n; i++) {
    if (ops[i] != state::PatchReplace) continue;
    const char* path = pb + off[i];
    int plen = len[i];
    if      (state::pathIs(path, plen, "blur"))     s->blur     = state::patchFloat(i);
    else if (state::pathIs(path, plen, "amount"))   s->amount   = state::patchFloat(i);
    else if (state::pathIs(path, plen, "movement")) s->movement = state::patchFloat(i);
    else if (state::pathIs(path, plen, "hue"))      s->hue      = state::patchFloat(i);
    else if (state::pathIs(path, plen, "quality"))  s->quality  = state::patchFloat(i);
  }
}

void on_resolume_param(void* self, long long, double) { (void)self; }

// Pure passthrough when there's no blur AND no chromatic offset. Stateless
// w.r.t. the skip: tick() keeps advancing the drift accumulator regardless,
// so a skipped frame can't desync anything visible.
int32_t is_identity(void* self) {
  auto* s = static_cast<State*>(self);
  if (!s) return 0;
  return (s->blur <= 1e-3f && s->amount <= 1e-3f) ? 1 : 0;
}

static void ensureBlurTex(State* s, int w, int h) {
  if (s->blur_tex.valid() && s->blur_w == w && s->blur_h == h) return;
  if (s->blur_tex.valid()) s->blur_tex.release();
  s->blur_tex = gpu::Device::createTexture(w, h);
  s->blur_w = w;
  s->blur_h = h;
}

void render(void* self, int vp_w, int vp_h) {
  auto* s = static_cast<State*>(self);
  if (!s || !s->initialized || vp_w <= 0 || vp_h <= 0) return;
  auto in  = gpu::Device::textureForField("tex_in");
  auto out = gpu::Device::textureForField("tex_out");
  if (!in.valid() || !out.valid()) return;

  const bool do_blur   = s->blur   > 1e-3f;
  const bool do_chroma = s->amount > 1e-3f;

  if (!do_chroma) {
    // Blur straight to output (applyWithRadius handles radius≈0 as a copy).
    s_blur.applyWithRadius(in, out, vp_w, vp_h, s->blur * BLUR_SCALE, s->quality);
    gpu::Device::submit();
    return;
  }

  gpu::Texture src = in;
  if (do_blur) {
    ensureBlurTex(s, vp_w, vp_h);
    if (!s->blur_tex.valid()) return;
    s_blur.applyWithRadius(in, s->blur_tex, vp_w, vp_h, s->blur * BLUR_SCALE, s->quality);
    src = s->blur_tex;
  }

  // The chroma offset distance is `amount`; the sawtooth sweeps it 0→full then
  // hard-resets at rate `movement` (movement=0 → static at full). `hue` rotates
  // the colour basis the split happens in (not the spatial slant, which is
  // fixed). Offset is aspect-corrected so the slant looks the same on any AR.
  float saw   = (s->movement > 1e-4f)
                  ? (float)(s->saw_phase - std::floor(s->saw_phase)) // frac → 0..1, hard reset
                  : 1.0f;
  float spread = s->amount * OFFSET_SCALE * saw;

  int min_dim = vp_w < vp_h ? vp_w : vp_h;
  Uniforms u = {};
  u.off_x     = SLANT_X * spread * ((float)min_dim / (float)vp_w);
  u.off_y     = SLANT_Y * spread * ((float)min_dim / (float)vp_h);
  u.hue_shift = s->hue * TAU;
  s->uniform_buf.writeOne(u);

  auto cp = gpu::ComputePass::begin();
  cp.setPSO(s_pso);
  cp.setTexture(src, 0, 0);
  cp.setSampler(s->sampler, 1);
  cp.setTexture(out, 2, 1);
  cp.setBuffer(s->uniform_buf, 3);
  cp.dispatch((vp_w + 7) / 8, (vp_h + 7) / 8);
  cp.end();

  gpu::Device::submit();
}

} // namespace subtle_blur
