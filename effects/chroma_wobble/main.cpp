/*
 * warp.legacy.chroma_wobble — "ChromaWobble" (v2 of the Resolume Wire patch).
 *
 * A triggered chromatic-aberration wobble: tap the trigger (or hold the gate)
 * and an animated fractal-noise field warps the image with RGB fringing, then
 * decays via an Attack/Release envelope.
 *
 * Source patch (Wire/Patches/ChromaWobble, 45 nodes): an animated, blurred,
 * temporally-smoothed Fractal Noise field drives per-channel UV Offsets (the
 * ChromaOffset pattern) + a Displace warp, scaled by Intensity and an Attack-
 * Release envelope (release time = "Duration"), wrapped in a Hue-Rotate ±H
 * sandwich. Exposed: Hue, Frequency, Intensity, Duration, Temporal Smoothing,
 * Trigger, Amount, Displace, Threshold/Threshold-Value (auto-trigger).
 *
 * v2 RE-ARCHITECTURE (flagged per DNODE_MIGRATION_NOTES §3, "port as a v2
 * tuned for efficiency"):
 *  - The blurred-feedback noise TEXTURE is replaced by an analytic animated
 *    fbm sampled per pixel (wobble.hlsl) — no ping-pong, no blur passes,
 *    shimmer-free. `speed` drives a drift accumulator (style guide §2.1); the
 *    old "Temporal Smoothing" feedback is dropped (its job was to de-shimmer
 *    the texture noise, which the analytic field doesn't need).
 *  - The chroma split uses the shared nano_chroma_offset() (the YIQ ChromaOffset
 *    "keeper"); `hue` rotates the split.
 *  - The audio Threshold auto-trigger is replaced by the standard gate/trigger
 *    (drive Threshold via a wire/tap if wanted). `amount` is a manual floor.
 *
 * Stateful trigger envelope → no temporal capability. is_identity when the
 * effective wobble is zero.
 */

#include <gpu.h>
#include <host.h>
#include "chroma_wobble_shaders.h"

#include <cmath>
#include <cstdint>

namespace chroma_wobble {

static constexpr float GAIN_SCALE  = 0.15f; // intensity=1,env=1 → ~15% uv displacement
static constexpr float FREQ_SCALE  = 8.0f;  // frequency=1 → fbm freq 8 (Wire ×8)
static constexpr float SPEED_SCALE = 0.4f;  // speed=1 → drift 0.4/sec

struct Uniforms {
  float gain;
  float freq;
  float phase;
  float chroma;
  float warp;
  float hue_shift;
  float aspect_x;
  float aspect_y;
};
static_assert(sizeof(Uniforms) == 32, "Uniforms layout mismatch");

struct State {
  gpu::Buffer  uniform_buf;
  gpu::Sampler sampler;
  bool initialized = false;

  // Schema-mirrored params.
  float amount    = 0.0f;
  float attack    = 0.0f;  // seconds
  float release   = 0.5f;  // seconds
  float intensity = 0.4f;
  float frequency = 0.5f;
  float chroma    = 0.5f;
  float warp      = 0.5f;
  float hue       = 0.2f;
  float speed     = 0.25f;

  // Edge tracking + AR envelope + drift.
  bool   gate_prev = false, trigger_prev = false;
  bool   gate = false, one_shot = false;
  double env = 0.0;
  double drift = 0.0;
  float  gain = 0.0f; // resolved wobble gain (for is_identity)
};

static gpu::ComputePSO s_pso;

static inline float easeWobble(float x) {
  x = x < 0.0f ? 0.0f : (x > 1.0f ? 1.0f : x);
  return x * x * (3.0f - 2.0f * x);
}

void module_init() {
  state::init("warp.legacy.chroma_wobble", {1, 0, 0},
    state::Schema()
      .eventField("trigger", state::PrimaryInput)
      .boolField ("gate", false, state::PrimaryInput,
                  "Hold to sustain the wobble; release to decay.")
      .floatField("amount", 0.0f, 0.0f, 1.0f, state::PrimaryInput, nullptr, 0.01f,
                  nullptr, "Manual wobble level (combines with the trigger envelope).")
      .floatField("intensity", 0.4f, 0.0f, 1.0f, state::PrimaryInput, nullptr, 0.01f,
                  nullptr, "Overall wobble displacement strength.")
      .floatField("chroma", 0.5f, 0.0f, 1.0f, state::PrimaryInput, nullptr, 0.01f,
                  nullptr, "Chromatic (RGB) split magnitude.")
      .floatField("warp", 0.5f, 0.0f, 1.0f, state::PrimaryInput, nullptr, 0.01f,
                  nullptr, "Shared UV warp magnitude.")
      .floatField("frequency", 0.5f, 0.0f, 1.0f, state::PrimaryInput, nullptr, 0.01f,
                  nullptr, "Noise field frequency (wobble scale).")
      .floatField("speed", 0.25f, 0.0f, 1.0f, state::PrimaryInput, nullptr, 0.01f,
                  nullptr, "Animation speed of the wobble field.")
      .floatField("hue", 0.2f, 0.0f, 1.0f, state::PrimaryInput, nullptr, 0.01f,
                  nullptr, "Hue rotation of the colour split.")
      .floatField("attack", 0.0f, 0.0f, 1.0f, state::PrimaryInput, nullptr, 0.01f,
                  "s", "Envelope ramp-up time.")
      .floatField("release", 0.5f, 0.0f, 5.0f, state::PrimaryInput, nullptr, 0.01f,
                  "s", "Envelope decay time.")
      // No temporal capability: stateful trigger AR envelope (a time jump would
      // corrupt it) — conservative default.
      .textureField("tex_in",  state::PrimaryInput)
      .textureField("tex_out", state::PrimaryOutput));

  if (gpu::Device::backend() == gpu::Backend::None) return;

  state::registerShaderSPV("chroma_wobble_wobble", WOBBLE_SPV, WOBBLE_SPV_SIZE);
  auto cs = gpu::Device::createShaderModuleByName("chroma_wobble_wobble");
  if (!cs) return;
  s_pso = gpu::Device::createComputePSO(cs, "main", gpu::Bindings()
      .tex2d(0).sampler(1).storageTex2d(2).uniform(3));

  state::log("chroma_wobble: module initialized");
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
  s->env = 0.0; s->drift = 0.0; s->one_shot = false; s->gain = 0.0f;
  if (!s_pso.valid() || !s->uniform_buf.valid()) return;
  s->initialized = true;
}

void tick(void* self, double dt) {
  auto* s = static_cast<State*>(self);
  if (!s) return;
  if (dt < 0.0) dt = 0.0;

  const bool rising = s->gate || s->one_shot;
  if (rising) {
    if (s->attack <= 1e-4f) s->env = 1.0;
    else s->env += dt / (double)s->attack;
    if (s->env >= 1.0) { s->env = 1.0; s->one_shot = false; }
  } else {
    if (s->release <= 1e-4f) s->env = 0.0;
    else s->env -= dt / (double)s->release;
    if (s->env < 0.0) s->env = 0.0;
  }

  float m = easeWobble((float)s->env);
  float a = s->amount < 0.0f ? 0.0f : (s->amount > 1.0f ? 1.0f : s->amount);
  float B = 1.0f - (1.0f - a) * (1.0f - m);
  s->gain = B * s->intensity;

  s->drift += dt * (double)s->speed * (double)SPEED_SCALE;
  if (s->drift > 1.0e6 || s->drift < -1.0e6) s->drift = std::fmod(s->drift, 1024.0);
}

void on_state_patched(void* self, int n, const char* pb, const int* off,
                      const int* len, const int* ops) {
  auto* s = static_cast<State*>(self);
  if (!s) return;
  for (int i = 0; i < n; i++) {
    if (ops[i] != state::PatchReplace) continue;
    const char* p = pb + off[i];
    int l = len[i];
    if      (state::pathIs(p, l, "amount"))    s->amount    = state::patchFloat(i);
    else if (state::pathIs(p, l, "intensity")) s->intensity = state::patchFloat(i);
    else if (state::pathIs(p, l, "chroma"))    s->chroma    = state::patchFloat(i);
    else if (state::pathIs(p, l, "warp"))      s->warp      = state::patchFloat(i);
    else if (state::pathIs(p, l, "frequency")) s->frequency = state::patchFloat(i);
    else if (state::pathIs(p, l, "speed"))     s->speed     = state::patchFloat(i);
    else if (state::pathIs(p, l, "hue"))       s->hue       = state::patchFloat(i);
    else if (state::pathIs(p, l, "attack"))    s->attack    = state::patchFloat(i);
    else if (state::pathIs(p, l, "release"))   s->release   = state::patchFloat(i);
    else if (state::pathIs(p, l, "gate")) {
      bool g = state::patchBool(i);
      if (g && !s->gate_prev) s->one_shot = false;
      s->gate = g; s->gate_prev = g;
    }
    else if (state::pathIs(p, l, "trigger")) {
      bool t = state::patchFloat(i) != 0.0f;
      if (t && !s->trigger_prev) s->one_shot = true;
      s->trigger_prev = t;
    }
  }
}

void on_resolume_param(void* self, long long, double) { (void)self; }

// NOTE: deliberately NO is_identity. The burn/gain that would make this a
// passthrough is a tick-evolved envelope value, not a config value — and the
// executor can permanently sideline a stage that ever reports identity (it
// stops ticking it), which would freeze the envelope. is_identity must depend
// only on config (cf. subtle_blur/sphr_blur), so a triggerable stateful effect
// simply always runs its (cheap) pass.

void render(void* self, int vp_w, int vp_h) {
  auto* s = static_cast<State*>(self);
  if (!s || !s->initialized || vp_w <= 0 || vp_h <= 0) return;
  auto in  = gpu::Device::textureForField("tex_in");
  auto out = gpu::Device::textureForField("tex_out");
  if (!in.valid() || !out.valid()) return;

  int min_dim = vp_w < vp_h ? vp_w : vp_h;
  Uniforms u = {};
  u.gain      = s->gain * GAIN_SCALE;
  u.freq      = s->frequency * FREQ_SCALE + 0.5f;
  u.phase     = (float)s->drift;
  u.chroma    = s->chroma;
  u.warp      = s->warp;
  u.hue_shift = s->hue * 6.28318530717958647692f;
  u.aspect_x  = (float)min_dim / (float)vp_w;
  u.aspect_y  = (float)min_dim / (float)vp_h;
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

} // namespace chroma_wobble
