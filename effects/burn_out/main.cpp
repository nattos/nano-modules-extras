/*
 * color.legacy.burn_out — "Burn Out" (v2 of the Resolume Wire patch).
 *
 * An AR-envelope-driven exposure-blowout grade, used live for emotional
 * "fade outs": tap the trigger (or hold the gate) and the image blows out —
 * saturation and contrast lift, exposure pushes highlights toward white, and
 * a crossfade carries it to pure white — then it decays back over the release.
 *
 * Source patch (Wire/Patches/Burn Out, 55 nodes): a Momentary button drives a
 * frame-stepped Attack/Release envelope (Metronome + Snapshot + Step + Delay
 * feedback, Target-FPS-normalized) through a curve; the envelope modulates
 * Saturation + Bright.Contrast with atan() soft-clip curves, plus a Cross
 * Fader / Modulate-Alpha output stage. Exposed knobs: Amount, Saturation
 * Boost, Contrast Boost, Momentary, Release Time, Attack Time, Modulate Alpha.
 *
 * v2 RE-ARCHITECTURE (flagged per DNODE_MIGRATION_NOTES §3):
 *  - The frame-stepped Snapshot/Delay envelope is replaced by a clean dt-based
 *    AR envelope (style guide §2.1) — Target-FPS normalization is unnecessary
 *    when we integrate real dt. A `gate` (hold/sustain) and a one-shot
 *    `trigger` both drive it; `amount` is a manual floor (screen-combined with
 *    the envelope, faithful to the Wire "Amount" knob).
 *  - The blow-out toward white is made explicit (`brightness_boost` exposure +
 *    `white_fade` crossfade) rather than emerging from the Cross Fader graph;
 *    the original's exact Wire Curve constants are approximated. `modulate_alpha`
 *    additionally drops alpha for compositing fade-outs (the Wire switch).
 *  - The saturation/contrast boosts keep the atan() soft-clip (analog warmth).
 *
 * Stateful AR/trigger envelope → no temporal capability (a time jump would
 * corrupt the envelope; conservative default). is_identity while the burn is
 * at rest (envelope idle AND amount 0).
 */

#include <gpu.h>
#include <host.h>
#include "burn_out_shaders.h"

#include <cmath>
#include <cstdint>

namespace burn_out {

// Boost scalings (v2 — chosen for a usable sweep; the Wire constants were
// buried in Map/Curve nodes and are approximated here).
static constexpr float SAT_K = 1.6f;  // saturation_boost=1 → up to +160% sat
static constexpr float CON_K = 1.0f;  // contrast_boost=1   → up to +100% contrast

struct Uniforms {
  float sat_amt;
  float con_amt;
  float brightness;
  float fade_black;
  float alpha_fade;
  float _p0, _p1, _p2;
};
static_assert(sizeof(Uniforms) == 32, "Uniforms layout mismatch");

struct State {
  gpu::Buffer uniform_buf;
  bool initialized = false;

  // Schema-mirrored params.
  float amount           = 0.0f;
  float attack           = 0.15f; // seconds
  float release          = 0.0f;  // seconds (instant cut back by default)
  float saturation_boost = 0.5f;
  float contrast_boost   = 0.5f;
  float brightness       = 0.0f; // tone lift/crush (flash), [-1,1]
  float darkness         = 1.0f; // how black the peak gets
  bool  modulate_alpha   = false;

  // Trigger/gate edge tracking.
  bool gate_prev    = false;
  bool trigger_prev = false;

  // AR envelope runtime.
  double env       = 0.0;   // raw envelope 0..1
  bool   gate      = false;
  bool   one_shot  = false; // a trigger tap is ramping up; auto-release at peak
  float  burn      = 0.0f;  // resolved burn intensity B (for is_identity)
};

static gpu::ComputePSO s_pso;

static inline float easeBurn(float x) {           // smoothstep ease (≈ Wire curve3)
  x = x < 0.0f ? 0.0f : (x > 1.0f ? 1.0f : x);
  return x * x * (3.0f - 2.0f * x);
}

void module_init() {
  state::init("color.legacy.burn_out", {1, 0, 0},
    state::Schema()
      .eventField("trigger", state::PrimaryInput)
      .boolField ("gate", false, state::PrimaryInput,
                  "Hold to sustain the burn at its peak; release to decay.")
      .floatField("amount", 0.0f, 0.0f, 1.0f, state::PrimaryInput, nullptr, 0.01f,
                  nullptr, "Manual burn intensity (combines with the trigger envelope).")
      .floatField("attack", 0.15f, 0.0f, 1.0f, state::PrimaryInput, nullptr, 0.01f,
                  "s", "Ramp-up time of the burn envelope.")
      .floatField("release", 0.0f, 0.0f, 5.0f, state::PrimaryInput, nullptr, 0.01f,
                  "s", "Decay time of the burn envelope (0 = instant cut back).")
      .floatField("saturation_boost", 0.5f, 0.0f, 1.0f, state::PrimaryInput, nullptr, 0.01f,
                  nullptr, "Saturation lift on the way down (the burn).")
      .floatField("contrast_boost", 0.5f, 0.0f, 1.0f, state::PrimaryInput, nullptr, 0.01f,
                  nullptr, "Contrast lift on the way down (the burn).")
      .floatField("darkness", 1.0f, 0.0f, 1.0f, state::PrimaryInput, nullptr, 0.01f,
                  nullptr, "How black the peak fades to (1 = full black).")
      .floatField("brightness", 0.0f, -1.0f, 1.0f, state::PrimaryInput, nullptr, 0.01f,
                  nullptr, "Tone lift/crush before the fade (a flash).")
      .boolField ("modulate_alpha", false, state::PrimaryInput,
                  "Also drop alpha with the burn (compositing fade-out).")
      .textureField("tex_in",  state::PrimaryInput)
      .textureField("tex_out", state::PrimaryOutput));

  if (gpu::Device::backend() == gpu::Backend::None) return;

  state::registerShaderSPV("burn_out_grade", BURN_SPV, BURN_SPV_SIZE);
  auto cs = gpu::Device::createShaderModuleByName("burn_out_grade");
  if (!cs) return;
  s_pso = gpu::Device::createComputePSO(cs, "main", gpu::Bindings()
      .tex2d(0).storageTex2d(1).uniform(2));

  state::log("burn_out: module initialized");
}

void* create() {
  auto* s = new State();
  s->uniform_buf = gpu::Device::createBuffer(sizeof(Uniforms), gpu::BufferUsage::Uniform);
  return s;
}

void destroy(void* self) {
  auto* s = static_cast<State*>(self);
  if (!s) return;
  s->uniform_buf.release();
  delete s;
}

void init(void* self) {
  auto* s = static_cast<State*>(self);
  if (!s) return;
  s->env = 0.0;
  s->one_shot = false;
  s->burn = 0.0f;
  if (!s_pso.valid() || !s->uniform_buf.valid()) return;
  s->initialized = true;
}

void tick(void* self, double dt) {
  auto* s = static_cast<State*>(self);
  if (!s) return;
  if (dt < 0.0) dt = 0.0;

  // AR envelope. Rising while the gate is held OR a one-shot trigger is still
  // climbing; falling otherwise. Reaching the peak ends a one-shot (so a tap
  // gives attack→release; a held gate sustains at 1).
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

  // Burn intensity: screen-combine the manual amount with the eased envelope
  // (either alone reaches its own level; both push toward 1) — faithful to the
  // Wire graph's (1 - (1-Amount)(1-curve(env))).
  float m = easeBurn((float)s->env);
  float a = s->amount < 0.0f ? 0.0f : (s->amount > 1.0f ? 1.0f : s->amount);
  s->burn = 1.0f - (1.0f - a) * (1.0f - m);
}

void on_state_patched(void* self, int n, const char* pb, const int* off,
                      const int* len, const int* ops) {
  auto* s = static_cast<State*>(self);
  if (!s) return;
  for (int i = 0; i < n; i++) {
    if (ops[i] != state::PatchReplace) continue;
    const char* p = pb + off[i];
    int l = len[i];
    if      (state::pathIs(p, l, "amount"))           s->amount           = state::patchFloat(i);
    else if (state::pathIs(p, l, "attack"))           s->attack           = state::patchFloat(i);
    else if (state::pathIs(p, l, "release"))          s->release          = state::patchFloat(i);
    else if (state::pathIs(p, l, "saturation_boost")) s->saturation_boost = state::patchFloat(i);
    else if (state::pathIs(p, l, "contrast_boost"))   s->contrast_boost   = state::patchFloat(i);
    else if (state::pathIs(p, l, "darkness"))         s->darkness         = state::patchFloat(i);
    else if (state::pathIs(p, l, "brightness"))       s->brightness       = state::patchFloat(i);
    else if (state::pathIs(p, l, "modulate_alpha"))   s->modulate_alpha   = state::patchBool(i);
    else if (state::pathIs(p, l, "gate")) {
      bool g = state::patchBool(i);
      if (g && !s->gate_prev) s->one_shot = false; // gate takes over from a one-shot
      s->gate = g;
      s->gate_prev = g;
    }
    else if (state::pathIs(p, l, "trigger")) {
      bool t = state::patchFloat(i) != 0.0f;
      if (t && !s->trigger_prev) s->one_shot = true; // rising edge → one-shot AR
      s->trigger_prev = t;
    }
  }
}

void on_resolume_param(void* self, long long, double) { (void)self; }

// NOTE: deliberately NO is_identity. The burn intensity is a tick-evolved
// envelope value, not config — and the executor can permanently sideline a
// stage that reports identity (it stops ticking it), so an armed-but-idle
// burn_out would never fire on a trigger. is_identity must depend only on
// config; a triggerable stateful effect always runs its (cheap) grade pass.

void render(void* self, int vp_w, int vp_h) {
  auto* s = static_cast<State*>(self);
  if (!s || !s->initialized || vp_w <= 0 || vp_h <= 0) return;
  auto in  = gpu::Device::textureForField("tex_in");
  auto out = gpu::Device::textureForField("tex_out");
  if (!in.valid() || !out.valid()) return;

  float B = s->burn;
  // atan soft-clip (analog warmth), normalized to [0,1] over B.
  float soft = std::atan(5.0f * B) / std::atan(5.0f);

  Uniforms u = {};
  u.sat_amt    = soft * s->saturation_boost * SAT_K;
  u.con_amt    = soft * s->contrast_boost   * CON_K;
  u.brightness = s->brightness * soft;
  u.fade_black = B * s->darkness;
  u.alpha_fade = s->modulate_alpha ? B : 0.0f;
  s->uniform_buf.writeOne(u);

  auto cp = gpu::ComputePass::begin();
  cp.setPSO(s_pso);
  cp.setTexture(in, 0, 0);
  cp.setTexture(out, 1, 1);
  cp.setBuffer(s->uniform_buf, 2);
  cp.dispatch((vp_w + 7) / 8, (vp_h + 7) / 8);
  cp.end();

  gpu::Device::submit();
}

} // namespace burn_out
