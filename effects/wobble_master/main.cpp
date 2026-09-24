/*
 * warp.legacy.wobble_master — "Wobble Master" (v2 of the Resolume Wire family).
 *
 * A pulse clearly emanates from the centre and travels outward, distorting
 * only what's under the wave: each trigger spawns a wave packet — a ring of
 * radial push with a half-sine leading edge and an exponential tail — and the
 * chroma split lingers as an afterglow where the wave has passed.
 *
 * Source: Wire/Patches/Wobble Master 2 (204 nodes). Its heart is a 1-D
 * scrolling feedback buffer: every frame a half-sine bump (amplitude = the
 * pulse envelope, width = Density) is injected at radius 0 and the previous
 * buffer is advected outward by waveSpeed; the buffer is rendered as a radial
 * gradient, multiplied by a conic direction field → the UV displacement, and
 * its blurred magnitude (kept in a ~0.956/frame feedback trail) gates the
 * custom YIQ "ChromaOffset" ISF. v2 keeps that character analytically: up to
 * 4 concurrent pulses tracked as integrated front positions in State,
 * evaluated as packets in wobble.hlsl (tail = the release envelope mapped
 * onto distance-behind-front; chroma trail = its own longer decay). The chroma
 * split reuses the shared ChromaOffset helper (nano_chroma.hlsl) with the
 * Wire graph's snapshotted per-channel shift directions, and the wobble
 * carrier is biased outward (the wave inflates more than it pulls).
 * `amount` keeps a standing concentric sine (manual/wired mode).
 *
 * Stateful pulses → no temporal capability, no is_identity (a triggerable
 * stateful effect must keep ticking — see chroma_wobble).
 */

#include <gpu.h>
#include <host.h>
#include "wobble_master_shaders.h"

#include <cmath>
#include <cstdint>

namespace wobble_master {

static constexpr float AMP_SCALE    = 0.06f;  // amplitude=1 → ~6% short-axis push
// Per-channel split directions: the Wire graph's snapshotted RShift/GShift/
// BShift constants (ISF node 357; the patch re-rolls them per trigger, these
// are the saved values). Y negated: ISF uv is y-up, ours is y-down. GAIN is
// the patch's Gain constant; CHROMA_SCALE stands in for the offsetMap's
// typical magnitude so chroma=0.5 lands at the patch's snapshot strength.
static constexpr float RSHIFT[2] = {  0.21f,  0.33f };
static constexpr float GSHIFT[2] = {  0.03f, -0.57f };
static constexpr float BSHIFT[2] = { -0.43f,  0.24f };
static constexpr float CHROMA_GAIN  = 0.4f;
static constexpr float CHROMA_SCALE = 0.17f;
static constexpr float FREQ_SCALE   = 12.0f;  // frequency=1 → 12 carrier rings
static constexpr float WAVE_SCALE   = 15.0f;  // wave_speed=1 → front travels 15 r-units/s
                                              // (quadratic response: v = SCALE·speed²)
static constexpr float DRIFT_SCALE  = 4.0f;   // carrier drift (rings/s) at ripple_speed=1
static constexpr float CHROMA_TRAIL_SEC = 0.37f; // afterglow decay (Wire's 0.956/frame feedback)
static constexpr int   MAX_PULSES = 4;

struct Uniforms {
  float drift;
  float freq;
  float amp;
  float hue_shift;
  float center_x;
  float center_y;
  float ripple;
  float floor_amt;
  float aspect_x;
  float aspect_y;
  float width;
  float tail_len;
  float fronts[MAX_PULSES];
  float shift_rg[4];  // R.xy, G.xy
  float shift_b[4];   // B.xy, chroma_len, unused
};
static_assert(sizeof(Uniforms) == 96, "Uniforms layout mismatch");

struct State {
  gpu::Buffer  uniform_buf;
  gpu::Sampler sampler;
  bool initialized = false;

  // Schema-mirrored params.
  float amount    = 0.0f;
  float amplitude = 0.5f;
  float frequency = 0.5f;
  float wave_speed = 0.5f;
  float chroma    = 0.5f;
  float hue       = 0.0f;
  float center_x  = 0.0f; // cover-square (0 = centre)
  float center_y  = 0.0f;
  float width     = 0.15f;
  float release   = 0.4f;
  float ripple    = 0.35f;
  float ripple_speed = 0.25f;

  // Trigger edges + the traveling pulses. Fronts are INTEGRATED (front +=
  // v·dt per tick), not derived from age — a wave_speed change accelerates
  // in-flight pulses instead of teleporting them. front < 0 = free slot.
  bool   gate_prev = false, trigger_prev = false;
  bool   gate = false;
  double retrig_t = 0.0;
  double pulse_front[MAX_PULSES] = { -1.0, -1.0, -1.0, -1.0 };
  int    pulse_next = 0;
  double drift = 0.0;
  double last_step = 0.0;  // front travel last tick (r units) — smears the edge
};

static gpu::ComputePSO s_pso;

static inline float clamp01(float x) {
  return x < 0.0f ? 0.0f : (x > 1.0f ? 1.0f : x);
}

static inline float waveVelocity(const State* s) {
  float x = clamp01(s->wave_speed);
  return x * x * WAVE_SCALE;   // r-units/s
}

static inline void spawnPulse(State* s) {
  s->pulse_front[s->pulse_next] = 0.0;
  s->pulse_next = (s->pulse_next + 1) % MAX_PULSES;
}

void module_init() {
  state::init("warp.legacy.wobble_master", {2, 2, 0},
    state::Schema()
      .eventField("trigger", state::PrimaryInput)
      .boolField ("gate", false, state::PrimaryInput,
                  "Hold to keep launching pulses; release to let the last one run out.")
      .floatField("amount", 0.0f, 0.0f, 1.0f, state::PrimaryInput, nullptr, 0.01f,
                  nullptr, "Standing concentric wobble (manual mode; pulses ride on top).")
      .floatField("amplitude", 0.5f, 0.0f, 1.0f, state::PrimaryInput, nullptr, 0.01f,
                  nullptr, "Radial displacement strength under the wave.")
      .floatField("frequency", 0.5f, 0.0f, 1.0f, state::PrimaryInput, nullptr, 0.01f,
                  nullptr, "Ring count of the shimmer carrier.")
      .floatField("wave_speed", 0.5f, 0.0f, 1.0f, state::PrimaryInput, nullptr, 0.01f,
                  nullptr, "Outward travel speed of the pulse front.")
      .floatField("width", 0.15f, 0.01f, 0.5f, state::PrimaryInput, nullptr, 0.01f,
                  nullptr, "Radial thickness of the pulse's leading edge.")
      .floatField("release", 0.4f, 0.0f, 5.0f, state::PrimaryInput, nullptr, 0.01f,
                  "s", "How long the wobble tail lingers behind the front.")
      .floatField("ripple", 0.35f, 0.0f, 1.0f, state::PrimaryInput, nullptr, 0.01f,
                  nullptr, "Texture under the wave: clean push (0) → oscillating shimmer (1).")
      .floatField("ripple_speed", 0.25f, 0.0f, 1.0f, state::PrimaryInput, nullptr, 0.01f,
                  nullptr, "Oscillation speed of the shimmer (independent of the front).")
      .floatField("chroma", 0.5f, 0.0f, 1.0f, state::PrimaryInput, nullptr, 0.01f,
                  nullptr, "Chromatic afterglow left behind the wave (the Wire "
                  "patch's per-channel split directions).")
      .floatField("hue", 0.0f, 0.0f, 1.0f, state::PrimaryInput, nullptr, 0.01f,
                  nullptr, "Hue rotation of the colour split.")
      .vec2Field ("center", 0.0f, 0.0f, state::PrimaryInput)
      .textureField("tex_in",  state::PrimaryInput)
      .textureField("tex_out", state::PrimaryOutput));

  if (gpu::Device::backend() == gpu::Backend::None) return;

  state::registerShaderSPV("wobble_master_wobble", WOBBLE_SPV, WOBBLE_SPV_SIZE);
  auto cs = gpu::Device::createShaderModuleByName("wobble_master_wobble");
  if (!cs) return;
  s_pso = gpu::Device::createComputePSO(cs, "main", gpu::Bindings()
      .tex2d(0).sampler(1).storageTex2d(2).uniform(3));

  state::log("wobble_master: module initialized");
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
  for (int i = 0; i < MAX_PULSES; i++) s->pulse_front[i] = -1.0;
  s->pulse_next = 0;
  s->drift = 0.0;
  s->retrig_t = 0.0;
  s->last_step = 0.0;
  if (!s_pso.valid() || !s->uniform_buf.valid()) return;
  s->initialized = true;
}

void tick(void* self, double dt) {
  auto* s = static_cast<State*>(self);
  if (!s) return;
  if (dt < 0.0) dt = 0.0;

  // Advance the fronts; retire once a front AND both trails are past the far
  // corner (r can reach ~2.3 with an off-centre origin).
  const double v = (double)waveVelocity(s);
  const double trail = 6.0 * v * ((double)s->release * 0.75 > (double)CHROMA_TRAIL_SEC
                                   ? (double)s->release * 0.75 : (double)CHROMA_TRAIL_SEC);
  s->last_step = v * dt;
  for (int i = 0; i < MAX_PULSES; i++) {
    if (s->pulse_front[i] < 0.0) continue;
    s->pulse_front[i] += s->last_step;
    if (s->pulse_front[i] - trail > 2.3 || s->pulse_front[i] > 500.0)
      s->pulse_front[i] = -1.0;
  }

  // Held gate auto-retriggers (the patch's beat-synced Pulse Retrigger loop).
  if (s->gate) {
    s->retrig_t -= dt;
    if (s->retrig_t <= 0.0) {
      spawnPulse(s);
      s->retrig_t = s->release * 0.5 > 0.15 ? s->release * 0.5 : 0.15;
    }
  } else {
    s->retrig_t = 0.0;
  }

  s->drift += dt * (double)s->ripple_speed * (double)DRIFT_SCALE;
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
    if      (state::pathIs(p, l, "amount"))     s->amount     = state::patchFloat(i);
    else if (state::pathIs(p, l, "amplitude"))  s->amplitude  = state::patchFloat(i);
    else if (state::pathIs(p, l, "frequency"))  s->frequency  = state::patchFloat(i);
    else if (state::pathIs(p, l, "wave_speed")) s->wave_speed = state::patchFloat(i);
    else if (state::pathIs(p, l, "width"))      s->width      = state::patchFloat(i);
    else if (state::pathIs(p, l, "release"))    s->release    = state::patchFloat(i);
    else if (state::pathIs(p, l, "ripple"))     s->ripple     = state::patchFloat(i);
    else if (state::pathIs(p, l, "ripple_speed")) s->ripple_speed = state::patchFloat(i);
    else if (state::pathIs(p, l, "chroma"))     s->chroma     = state::patchFloat(i);
    else if (state::pathIs(p, l, "hue"))        s->hue        = state::patchFloat(i);
    else if (state::pathIs(p, l, "center"))     { auto v = state::patchVec2(i); s->center_x = v.x; s->center_y = v.y; }
    else if (state::pathIs(p, l, "gate")) {
      bool g = state::patchBool(i);
      if (g && !s->gate_prev) { spawnPulse(s); s->retrig_t = s->release * 0.5 > 0.15 ? s->release * 0.5 : 0.15; }
      s->gate = g; s->gate_prev = g;
    }
    else if (state::pathIs(p, l, "trigger")) {
      bool t = state::patchFloat(i) != 0.0f;
      if (t && !s->trigger_prev) spawnPulse(s);
      s->trigger_prev = t;
    }
  }
}

void on_resolume_param(void* self, long long, double) { (void)self; }

void render(void* self, int vp_w, int vp_h) {
  auto* s = static_cast<State*>(self);
  if (!s || !s->initialized || vp_w <= 0 || vp_h <= 0) return;
  auto in  = gpu::Device::textureForField("tex_in");
  auto out = gpu::Device::textureForField("tex_out");
  if (!in.valid() || !out.valid()) return;

  const float v = waveVelocity(s);
  // Smear the leading edge over the last frame's travel so a fast front stays
  // continuous frame-to-frame instead of strobing in disconnected bands.
  float w = s->width < 0.01f ? 0.01f : s->width;
  w += (float)s->last_step;

  int min_dim = vp_w < vp_h ? vp_w : vp_h;
  Uniforms u = {};
  u.drift      = (float)s->drift;
  u.freq       = s->frequency * FREQ_SCALE + 0.5f;
  u.amp        = s->amplitude * AMP_SCALE;
  u.hue_shift  = s->hue * 6.28318530717958647692f;
  // center is cover-square (0 = centre); map to uv (0.5 + 0.5*c on the short axis).
  u.center_x   = 0.5f + 0.5f * s->center_x * ((float)min_dim / (float)vp_w);
  u.center_y   = 0.5f + 0.5f * s->center_y * ((float)min_dim / (float)vp_h);
  u.ripple     = clamp01(s->ripple);
  u.floor_amt  = clamp01(s->amount);
  u.aspect_x   = (float)min_dim / (float)vp_w;
  u.aspect_y   = (float)min_dim / (float)vp_h;
  u.width      = w;
  u.tail_len   = v * s->release * 0.75f;
  if (u.tail_len < 1e-3f) u.tail_len = 1e-3f;
  for (int i = 0; i < MAX_PULSES; i++)
    u.fronts[i] = s->pulse_front[i] >= 0.0 ? (float)s->pulse_front[i] : -1000.0f;
  // chroma 0.5 = the patch's snapshot strength (Shift·Gain at typical field).
  const float cs = clamp01(s->chroma) * 2.0f * CHROMA_GAIN * CHROMA_SCALE;
  u.shift_rg[0] = RSHIFT[0] * cs; u.shift_rg[1] = RSHIFT[1] * cs;
  u.shift_rg[2] = GSHIFT[0] * cs; u.shift_rg[3] = GSHIFT[1] * cs;
  u.shift_b[0]  = BSHIFT[0] * cs; u.shift_b[1]  = BSHIFT[1] * cs;
  u.shift_b[2]  = v * CHROMA_TRAIL_SEC > w ? v * CHROMA_TRAIL_SEC : w;
  u.shift_b[3]  = 0.0f;
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

} // namespace wobble_master
