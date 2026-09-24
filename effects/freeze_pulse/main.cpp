/*
 * warp.legacy.freeze_pulse — "Freeze Pulse" (v2 of the Resolume Wire patch).
 *
 * On a trigger it FREEZES the current frame, then over a short envelope
 * scale-pops + jitters + grades that frozen frame and composites it over the
 * still-running live video using a randomly-chosen blend mode — a stutter-freeze
 * glitch that decays over `time`.
 *
 * Source patch (Wire/Patches/Freeze Pulse, 91 nodes): a Beat/Main/Threshold
 * Trigger latches a frame (Video Mixer + Delay freeze), an Attack-Release env
 * (Time) drives a Transform scale-pop + jitter + Bright.Contrast, a Bag picks a
 * no-repeat random blend mode from 5 Override slots (RGB / Hard Light /
 * Difference / Difference-I / Lighten), and a Transition composites it over the
 * live frame; Switch Probability gates whether the mode re-rolls.
 *
 * v2 RE-ARCHITECTURE (flagged per DNODE_MIGRATION_NOTES §3): the freeze is a
 * per-instance captured texture (capture.hlsl) written on the trigger frame;
 * the AR env, the no-repeat mode bag and the per-pulse random transform live in
 * C++. The beat-clock (Transport Beat / Random Rate) is driven externally via a
 * wire/tap into the trigger; the 5 override-mode slots collapse to a single
 * `blend_mode` select + a `random_mode` bag toggle. Two compute passes:
 * capture (on trigger) then pulse (every frame; blend_phase 0 → passthrough).
 *
 * Stateful (freeze buffer + envelope) → no temporal capability, no is_identity.
 */

#include <gpu.h>
#include <host.h>
#include "freeze_pulse_shaders.h"

#include <cmath>
#include <cstdint>

namespace freeze_pulse {

static constexpr float JITTER_SCALE = 0.3f;
static constexpr int   N_MODES      = 5;

struct Uniforms {
  float scale;
  float trans_x;
  float trans_y;
  float bright;
  float contrast;
  float blend_phase;
  int   mode;
  int   _p0;
};
static_assert(sizeof(Uniforms) == 32, "Uniforms layout mismatch");

struct State {
  gpu::Buffer  uniform_buf;
  gpu::Sampler sampler;
  gpu::Texture frozen_tex;
  int          fz_w = 0, fz_h = 0;
  bool initialized = false;

  // Schema-mirrored params.
  float time       = 0.7f;
  float intensity  = 1.0f;
  float alpha      = 1.0f;
  float start_offset = 0.3f; // initial zoom fraction at the trigger instant
  float max_scale  = 2.0f;
  float jitter     = 0.3f;
  float contrast   = 0.4f;
  int   blend_mode = 0;
  bool  random_mode = true;
  int   seed       = 1234;

  // Trigger/gate edges.
  bool gate_prev = false, trigger_prev = false;
  bool gate = false;

  // Pulse runtime.
  double   env = 0.0;
  bool     pending_capture = false;
  int      cur_mode = 0;
  int      last_mode = -1;
  float    jx = 0.0f, jy = 0.0f;
  uint32_t rng = 0x1234u;
};

static gpu::ComputePSO s_pso_capture;
static gpu::ComputePSO s_pso_pulse;

static inline uint32_t xs(uint32_t& s){ s^=s<<13; s^=s>>17; s^=s<<5; return s; }
static inline float rnd(uint32_t& s){ return (xs(s)>>8)*(1.0f/16777216.0f); }

// Re-roll the per-pulse transform + (optionally) a no-repeat random blend mode.
static void fireTrigger(State* s) {
  s->env = 1.0;
  s->pending_capture = true;
  s->jx = rnd(s->rng) * 2.0f - 1.0f;
  s->jy = rnd(s->rng) * 2.0f - 1.0f;
  if (s->random_mode) {
    int m = (int)(rnd(s->rng) * (float)N_MODES);
    if (m >= N_MODES) m = N_MODES - 1;
    if (m == s->last_mode) m = (m + 1) % N_MODES; // avoid immediate repeat (bag-ish)
    s->cur_mode = m;
  } else {
    s->cur_mode = s->blend_mode;
  }
  s->last_mode = s->cur_mode;
}

void module_init() {
  state::init("warp.legacy.freeze_pulse", {1, 0, 0},
    state::Schema()
      .eventField("trigger", state::PrimaryInput)
      .boolField ("gate", false, state::PrimaryInput,
                  "Hold to keep the frozen frame held at full pulse.")
      .floatField("time", 0.7f, 0.0f, 5.0f, state::PrimaryInput, nullptr, 0.01f,
                  "s", "Freeze-pulse duration (envelope decay).")
      .floatField("intensity", 1.0f, 0.0f, 1.0f, state::PrimaryInput, nullptr, 0.01f,
                  nullptr, "Blend strength of the frozen pulse over the live frame.")
      .floatField("alpha", 1.0f, 0.0f, 1.0f, state::PrimaryInput, nullptr, 0.01f,
                  nullptr, "Overall pulse opacity.")
      .floatField("start_offset", 0.3f, 0.0f, 1.0f, state::PrimaryInput, nullptr, 0.01f,
                  nullptr, "Initial zoom of the frozen frame at the trigger (it grows from here).")
      .floatField("max_scale", 2.0f, 0.0f, 5.0f, state::PrimaryInput, nullptr, 0.01f,
                  nullptr, "How much the frozen frame zooms by the end of the pulse.")
      .floatField("jitter", 0.3f, 0.0f, 1.0f, state::PrimaryInput, nullptr, 0.01f,
                  nullptr, "Random translation of the frozen frame.")
      .floatField("contrast", 0.4f, 0.0f, 1.0f, state::PrimaryInput, nullptr, 0.01f,
                  nullptr, "Contrast pop on the frozen frame.")
      .selectField("blend_mode", 0, state::PrimaryInput,
                   {{"RGB", 0}, {"Hard Light", 1}, {"Difference", 2},
                    {"Difference I", 3}, {"Lighten", 4}}, false,
                   "Blend of the frozen pulse over live (when random is off).")
      .boolField ("random_mode", true, state::PrimaryInput,
                  "Pick a fresh random blend mode (no immediate repeat) each trigger.")
      .intField  ("seed", 1234, 0, 65535, state::PrimaryInput, 0, nullptr, "Random seed.")
      .textureField("tex_in",  state::PrimaryInput)
      .textureField("tex_out", state::PrimaryOutput));

  if (gpu::Device::backend() == gpu::Backend::None) return;

  state::registerShaderSPV("freeze_pulse_capture", CAPTURE_SPV, CAPTURE_SPV_SIZE);
  state::registerShaderSPV("freeze_pulse_pulse", PULSE_SPV, PULSE_SPV_SIZE);
  auto ccap = gpu::Device::createShaderModuleByName("freeze_pulse_capture");
  auto cpul = gpu::Device::createShaderModuleByName("freeze_pulse_pulse");
  if (!ccap || !cpul) return;
  s_pso_capture = gpu::Device::createComputePSO(ccap, "main", gpu::Bindings()
      .tex2d(0).storageTex2d(1));
  s_pso_pulse = gpu::Device::createComputePSO(cpul, "main", gpu::Bindings()
      .tex2d(0).tex2d(1).sampler(2).storageTex2d(3).uniform(4));

  state::log("freeze_pulse: module initialized");
}

void* create() {
  auto* s = new State();
  s->uniform_buf = gpu::Device::createBuffer(sizeof(Uniforms), gpu::BufferUsage::Uniform);
  s->sampler = gpu::Device::createSampler(gpu::FilterMode::Linear, gpu::AddressMode::ClampToEdge);
  s->rng = 0x1234u;
  return s;
}

void destroy(void* self) {
  auto* s = static_cast<State*>(self);
  if (!s) return;
  s->uniform_buf.release();
  s->sampler.release();
  if (s->frozen_tex.valid()) s->frozen_tex.release();
  delete s;
}

void init(void* self) {
  auto* s = static_cast<State*>(self);
  if (!s) return;
  s->env = 0.0; s->pending_capture = false; s->last_mode = -1;
  s->rng = 0x1234u ^ (uint32_t)s->seed;
  if (!s_pso_capture.valid() || !s_pso_pulse.valid() || !s->uniform_buf.valid()) return;
  s->initialized = true;
}

void tick(void* self, double dt) {
  auto* s = static_cast<State*>(self);
  if (!s) return;
  if (dt < 0.0) dt = 0.0;
  if (s->gate) {
    s->env = 1.0;            // sustain while held
  } else {
    if (s->time <= 1e-4f) s->env = 0.0;
    else s->env -= dt / (double)s->time;
    if (s->env < 0.0) s->env = 0.0;
  }
}

void on_state_patched(void* self, int n, const char* pb, const int* off,
                      const int* len, const int* ops) {
  auto* s = static_cast<State*>(self);
  if (!s) return;
  for (int i = 0; i < n; i++) {
    if (ops[i] != state::PatchReplace) continue;
    const char* p = pb + off[i];
    int l = len[i];
    if      (state::pathIs(p, l, "time"))        s->time       = state::patchFloat(i);
    else if (state::pathIs(p, l, "intensity"))   s->intensity  = state::patchFloat(i);
    else if (state::pathIs(p, l, "alpha"))       s->alpha      = state::patchFloat(i);
    else if (state::pathIs(p, l, "start_offset")) s->start_offset = state::patchFloat(i);
    else if (state::pathIs(p, l, "max_scale"))   s->max_scale  = state::patchFloat(i);
    else if (state::pathIs(p, l, "jitter"))      s->jitter     = state::patchFloat(i);
    else if (state::pathIs(p, l, "contrast"))    s->contrast   = state::patchFloat(i);
    else if (state::pathIs(p, l, "blend_mode"))  s->blend_mode = state::patchInt(i);
    else if (state::pathIs(p, l, "random_mode")) s->random_mode = state::patchBool(i);
    else if (state::pathIs(p, l, "seed"))        s->seed       = state::patchInt(i);
    else if (state::pathIs(p, l, "gate")) {
      bool g = state::patchBool(i);
      if (g && !s->gate_prev) fireTrigger(s); // rising edge → freeze + pulse
      s->gate = g; s->gate_prev = g;
    }
    else if (state::pathIs(p, l, "trigger")) {
      bool t = state::patchFloat(i) != 0.0f;
      if (t && !s->trigger_prev) fireTrigger(s);
      s->trigger_prev = t;
    }
  }
}

void on_resolume_param(void* self, long long, double) { (void)self; }

static void ensureFrozen(State* s, int w, int h) {
  if (s->frozen_tex.valid() && s->fz_w == w && s->fz_h == h) return;
  if (s->frozen_tex.valid()) s->frozen_tex.release();
  s->frozen_tex = gpu::Device::createTexture(w, h);
  s->fz_w = w; s->fz_h = h;
}

void render(void* self, int vp_w, int vp_h) {
  auto* s = static_cast<State*>(self);
  if (!s || !s->initialized || vp_w <= 0 || vp_h <= 0) return;
  auto in  = gpu::Device::textureForField("tex_in");
  auto out = gpu::Device::textureForField("tex_out");
  if (!in.valid() || !out.valid()) return;
  ensureFrozen(s, vp_w, vp_h);
  if (!s->frozen_tex.valid()) return;

  // Capture the live frame into the freeze buffer on the trigger frame.
  if (s->pending_capture) {
    auto cp = gpu::ComputePass::begin();
    cp.setPSO(s_pso_capture);
    cp.setTexture(in, 0, 0);
    cp.setTexture(s->frozen_tex, 1, 1);
    cp.dispatch((vp_w + 7) / 8, (vp_h + 7) / 8);
    cp.end();
    s->pending_capture = false;
  }

  float env = (float)s->env;
  // The frozen frame GROWS over the pulse: it pops to `start_offset` of the
  // zoom at the trigger instant, then keeps zooming IN to the full max_scale as
  // the pulse fades out — env runs 1→0, so progress = (1-env).
  float grow = s->start_offset + (1.0f - s->start_offset) * (1.0f - env);
  Uniforms u = {};
  u.scale       = 1.0f + s->max_scale * grow;
  u.trans_x     = s->jx * s->jitter * JITTER_SCALE * grow;
  u.trans_y     = s->jy * s->jitter * JITTER_SCALE * grow;
  u.bright      = 0.0f;
  u.contrast    = s->contrast * env;
  u.blend_phase = s->intensity * s->alpha * env;
  u.mode        = s->cur_mode;
  s->uniform_buf.writeOne(u);

  auto cp = gpu::ComputePass::begin();
  cp.setPSO(s_pso_pulse);
  cp.setTexture(in, 0, 0);
  cp.setTexture(s->frozen_tex, 1, 1);
  cp.setSampler(s->sampler, 2);
  cp.setTexture(out, 3, 1);
  cp.setBuffer(s->uniform_buf, 4);
  cp.dispatch((vp_w + 7) / 8, (vp_h + 7) / 8);
  cp.end();

  gpu::Device::submit();
}

} // namespace freeze_pulse
