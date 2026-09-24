/*
 * warp.legacy.stutter_scale — "Stutter Scale 2" (v2 of the Resolume Wire patch).
 *
 * A beat-stutter scale/zoom glitch: a phase is quantized into `levels` discrete
 * steps. The zoom sweeps PROGRESSIVELY from `min_scale` to `max_scale` across
 * the steps (quantized to the step grid, so it still stutters), while each
 * step boundary re-rolls a seeded random jitter translation, optional Y-flip
 * and colour inversion, and a hue rotation, plus a contrast/brightness boost —
 * all HELD for that step. The result crossfades with the untouched input by
 * `intensity`. Very useful for stuttering overlays and logos.
 *
 * Source patch (Wire/Patches/Stutter Scale 2, 96 nodes): a Sweep drives a phase
 * floor-quantized by Levels; On-Change fires a Trigger that re-seeds Random
 * nodes for scale (Min/Max Scale), translation/jitter (Max Jitter / Movement),
 * Flip Y, Hue Rotate, Invert RGB, with Bright.Contrast (Cutoff/Boost), an
 * Attack-Release env (Env Time/Trigger) and an Alpha-Mode blend out.
 *
 * This is a MANUALLY-driven effect: `sweep` ∈ [0,1] (a knob or an automation
 * curve) is the sole time source. It stutters through `levels` steps as you
 * sweep — full strength immediately (no ease-in; it's meant to be stuttery).
 * Each step's randoms are seeded by the step index (deterministic). A
 * `deadzone` band at either endpoint turns the effect OFF — showing the
 * `fill`, NOT the input — so it cleanly disappears at the ends of the sweep;
 * the bands toggle independently (`start_deadzone` default OFF,
 * `end_deadzone` default on). `fill` picks what shows both past the scaled
 * image's border and in the deadzones: black, transparent, or edge-smear
 * (the legacy clamp look; deadzones stay transparent).
 *
 * v2 RE-ARCHITECTURE (flagged per DNODE_MIGRATION_NOTES §3): DROPPED for v2
 * (recoverable later): the 22 Alpha blend modes (we crossfade), the optical-flow
 * "Use Motion" motion-blur path, and the explicit Attack-Release env (use
 * `intensity` or a tap). `min/max_scale` are raw scale factors; the rest [0,1].
 *
 * Stateless (pure function of sweep + params) → TimeIndependent, and is_identity
 * (config-only) at sweep in the deadzone or intensity 0.
 */

#include <gpu.h>
#include <host.h>
#include "stutter_scale_shaders.h"

#include <cmath>
#include <cstdint>

namespace stutter_scale {

static constexpr float TAU          = 6.28318530717958647692f;
static constexpr float JITTER_SCALE = 0.5f;  // jitter=1 → up to 0.5 uv translation

// Fill styles: what shows where the transform samples past the source bounds
// AND what the endpoint deadzones show. Lock-step with the branches in
// stutter.hlsl.
enum FillStyle { FillBlack = 0, FillTransparent = 1, FillEdge = 2 };

struct Uniforms {
  float scale;
  float trans_x;
  float trans_y;
  float flip_y;
  float hue_shift;
  float invert;
  float bright;
  float contrast;
  float intensity;
  float fill;
  float dead;
  float _p0;
};
static_assert(sizeof(Uniforms) == 48, "Uniforms layout mismatch");

struct State {
  gpu::Buffer  uniform_buf;
  gpu::Sampler sampler;
  bool initialized = false;

  // Schema-mirrored params.
  float sweep      = 0.0f;
  int   levels     = 10;
  float min_scale  = 1.0f;
  float max_scale  = 6.0f;
  float jitter     = 0.3f;
  float hue        = 0.0f;
  float boost      = 0.25f;
  float intensity  = 1.0f;
  float deadzone   = 0.05f;
  bool  start_deadzone = false;
  bool  end_deadzone = true;
  int   fill       = FillEdge;
  bool  do_flip    = true;
  bool  do_invert  = false;
  int   seed       = 1234;
};

// HARD endpoint deadzone: true when the sweep sits in an ENABLED endpoint band
// ([0, dz] / [1-dz, 1]). In the deadzone the effect goes fully TRANSPARENT —
// no ease-in, the stutter is hard. Outside, full strength.
static inline bool inDeadzone(const State* s) {
  float sw = s->sweep < 0.0f ? 0.0f : (s->sweep > 1.0f ? 1.0f : s->sweep);
  float dz = s->deadzone;
  if (s->start_deadzone && sw <= dz) return true;
  if (s->end_deadzone && sw >= 1.0f - dz) return true;
  return false;
}

static gpu::ComputePSO s_pso;

static inline uint32_t hash_u32(uint32_t x) {
  x ^= x >> 16; x *= 0x7feb352du; x ^= x >> 15; x *= 0x846ca68bu; x ^= x >> 16;
  return x;
}
static inline float rand01(uint32_t h) { return (h >> 8) * (1.0f / 16777216.0f); }

void module_init() {
  state::init("warp.legacy.stutter_scale", {2, 1, 0},
    state::Schema()
      .floatField("sweep", 0.0f, 0.0f, 1.0f, state::PrimaryInput, nullptr, 0.01f,
                  nullptr, "The stutter playhead (knob / automation); the zoom "
                  "sweeps min → max as it advances.")
      .intField  ("levels", 10, 1, 25, state::PrimaryInput, 0, nullptr,
                  "Number of discrete stutter steps per phase unit.")
      .floatField("min_scale", 1.0f, 1.0f, 16.0f, state::PrimaryInput, nullptr, 0.05f,
                  nullptr, "Zoom at the start of the sweep.")
      .floatField("max_scale", 6.0f, 1.0f, 16.0f, state::PrimaryInput, nullptr, 0.05f,
                  nullptr, "Zoom at the end of the sweep.")
      .floatField("jitter", 0.3f, 0.0f, 1.0f, state::PrimaryInput, nullptr, 0.01f,
                  nullptr, "Per-step random translation amount (zero at the "
                  "very start of the sweep).")
      .floatField("hue", 0.0f, 0.0f, 1.0f, state::PrimaryInput, nullptr, 0.01f,
                  nullptr, "Per-step random hue rotation range.")
      .floatField("boost", 0.25f, 0.0f, 1.0f, state::PrimaryInput, nullptr, 0.01f,
                  nullptr, "Contrast/brightness boost.")
      .floatField("intensity", 1.0f, 0.0f, 1.0f, state::PrimaryInput, nullptr, 0.01f,
                  nullptr, "Crossfade with the untouched input.")
      .floatField("deadzone", 0.05f, 0.0f, 0.5f, state::PrimaryInput, nullptr, 0.01f,
                  nullptr, "Endpoint band where the output goes transparent (off).")
      .boolField ("start_deadzone", false, state::PrimaryInput, "Turn off near sweep=0 (shows the fill).")
      .boolField ("end_deadzone", true, state::PrimaryInput, "Turn off near sweep=1 (shows the fill).")
      .selectField("fill", FillEdge, state::PrimaryInput, {
        {"Black", FillBlack}, {"Transparent", FillTransparent}, {"Edge", FillEdge},
      }, false,
        "What shows past the scaled image's border and in the endpoint "
        "deadzones (Edge smears the border; deadzones stay transparent).")
      .boolField ("flip", true, state::PrimaryInput, "Allow random Y-flips per step.")
      .boolField ("color_invert", false, state::PrimaryInput, "Allow random colour inversion per step.")
      .intField  ("seed", 1234, 0, 65535, state::PrimaryInput, 0, nullptr, "Random seed.")
      .capability(state::Capability::TimeIndependent)
      .textureField("tex_in",  state::PrimaryInput)
      .textureField("tex_out", state::PrimaryOutput));

  if (gpu::Device::backend() == gpu::Backend::None) return;

  state::registerShaderSPV("stutter_scale_stutter", STUTTER_SPV, STUTTER_SPV_SIZE);
  auto cs = gpu::Device::createShaderModuleByName("stutter_scale_stutter");
  if (!cs) return;
  s_pso = gpu::Device::createComputePSO(cs, "main", gpu::Bindings()
      .tex2d(0).sampler(1).storageTex2d(2).uniform(3));

  state::log("stutter_scale: module initialized");
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

void tick(void* self, double dt) { (void)self; (void)dt; }

void on_state_patched(void* self, int n, const char* pb, const int* off,
                      const int* len, const int* ops) {
  auto* s = static_cast<State*>(self);
  if (!s) return;
  for (int i = 0; i < n; i++) {
    if (ops[i] != state::PatchReplace) continue;
    const char* p = pb + off[i];
    int l = len[i];
    if      (state::pathIs(p, l, "sweep"))        s->sweep     = state::patchFloat(i);
    else if (state::pathIs(p, l, "levels"))       s->levels    = state::patchInt(i);
    else if (state::pathIs(p, l, "min_scale"))    s->min_scale = state::patchFloat(i);
    else if (state::pathIs(p, l, "max_scale"))    s->max_scale = state::patchFloat(i);
    else if (state::pathIs(p, l, "jitter"))       s->jitter    = state::patchFloat(i);
    else if (state::pathIs(p, l, "hue"))          s->hue       = state::patchFloat(i);
    else if (state::pathIs(p, l, "boost"))        s->boost     = state::patchFloat(i);
    else if (state::pathIs(p, l, "intensity"))    s->intensity = state::patchFloat(i);
    else if (state::pathIs(p, l, "deadzone"))     s->deadzone  = state::patchFloat(i);
    else if (state::pathIs(p, l, "start_deadzone")) s->start_deadzone = state::patchBool(i);
    else if (state::pathIs(p, l, "end_deadzone")) s->end_deadzone = state::patchBool(i);
    else if (state::pathIs(p, l, "fill"))         s->fill      = state::patchInt(i);
    else if (state::pathIs(p, l, "flip"))         s->do_flip   = state::patchBool(i);
    else if (state::pathIs(p, l, "color_invert")) s->do_invert = state::patchBool(i);
    else if (state::pathIs(p, l, "seed"))         s->seed      = state::patchInt(i);
  }
}

void on_resolume_param(void* self, long long, double) { (void)self; }

// Config-only identity: only when intensity is 0 AND the sweep is outside any
// deadzone — the deadzone produces the FILL (transparent/black), which is
// never a passthrough.
int32_t is_identity(void* self) {
  auto* s = static_cast<State*>(self);
  if (!s) return 0;
  return (s->intensity <= 1e-3f && !inDeadzone(s)) ? 1 : 0;
}

void render(void* self, int vp_w, int vp_h) {
  auto* s = static_cast<State*>(self);
  if (!s || !s->initialized || vp_w <= 0 || vp_h <= 0) return;
  auto in  = gpu::Device::textureForField("tex_in");
  auto out = gpu::Device::textureForField("tex_out");
  if (!in.valid() || !out.valid()) return;

  // Quantize the sweep into the current step, then seed this step's randoms.
  int levels = s->levels < 1 ? 1 : s->levels;
  float sweep = s->sweep < 0.0f ? 0.0f : (s->sweep > 1.0f ? 1.0f : s->sweep);
  long step = (long)std::floor((double)sweep * (double)levels);
  if (step > levels - 1) step = levels - 1;
  bool dead = inDeadzone(s);
  uint32_t h = hash_u32((uint32_t)(step * 2654435761u) ^ (uint32_t)s->seed);

  float r1 = rand01(h);              h = hash_u32(h);
  float r2 = rand01(h);              h = hash_u32(h);
  float r3 = rand01(h);              h = hash_u32(h);
  float r4 = rand01(h);              h = hash_u32(h);
  float r5 = rand01(h);

  // The zoom is PROGRESSIVE: it walks min → max across the step grid (held
  // per step, so it still stutters); the randoms only drive jitter/flip/hue.
  float q = levels > 1 ? (float)step / (float)(levels - 1) : 0.0f;
  // The Wire patch gates the jitter by the quantized phase^0.1 (its Power
  // node): exactly ZERO at the start endpoint, near-full almost immediately
  // after — so the sweep's start stays clean.
  float qj = std::pow((float)step / (float)levels, 0.1f);
  float lo = s->min_scale, hi = s->max_scale;
  Uniforms u = {};
  u.scale     = lo + (hi - lo) * q;
  u.trans_x   = (r1 * 2.0f - 1.0f) * s->jitter * JITTER_SCALE * qj;
  u.trans_y   = (r2 * 2.0f - 1.0f) * s->jitter * JITTER_SCALE * qj;
  u.flip_y    = (s->do_flip   && r3 < 0.5f) ? 1.0f : 0.0f;
  u.invert    = (s->do_invert && r4 < 0.5f) ? 1.0f : 0.0f;
  u.hue_shift = (r5 * 2.0f - 1.0f) * s->hue * TAU;
  u.bright    = 0.0f;
  u.contrast  = s->boost;
  u.intensity = s->intensity;
  u.fill      = (float)s->fill;
  // Hard stutter outside the deadzone; the fill (hard off) inside it.
  u.dead      = dead ? 1.0f : 0.0f;
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

} // namespace stutter_scale
