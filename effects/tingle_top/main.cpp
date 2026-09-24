/*
 * source.light.tingle_top — sparkles bundled at the top of each bar while a note is
 * held, released downward as a "wave" when let go. POLYPHONIC: up to 4 voices.
 *
 * A note-on (gate rising / trigger pulse / auto) allocates a SUSTAINING voice
 * (spawns at the top band); note-off converts it to a RELEASE voice — a
 * split-normal gaussian whose window bursts/accelerates downward and drains
 * off screen, then the voice returns to the pool. Multiple releases overlap
 * (the polyphony, since there's only one note to hold). Each voice is a spawn-
 * y distribution; the fixed particle pool samples the weighted MIXTURE, so
 * total density stays constant and splits across voices. Particles live + fade
 * in place (optional per-particle velocity drift on top).
 *
 * GPU-resident particle pool: update compute (samples the voice mixture) →
 * prefill → instanced sparkle quads (additive) → motion passthrough. Trigger
 * surface: gate / trigger / level / auto_mode (the shared Off/Random/Beats
 * self-fire block), default_gate_state as the at-rest fallback (true = a
 * permanently-sustaining voice).
 */

#include <gpu.h>
#include <host.h>
#include <effect_auto_trigger.h>  // fx::AutoTrigger — the shared Off/Random/Beats self-fire
#include "tingle_top_shaders.h"

#include <algorithm>
#include <cmath>
#include <cstdint>

namespace tingle_top {

static constexpr int POOL_HARD_MAX  = 2048;
static constexpr int VOICES_PER_BAR = 4;     // polyphony per bar (16 total)

// Which bars sparkles spawn into.
enum BarTargetMode { BAR_ONE = 0, BAR_RANDOM = 1, BAR_ALL = 2 };

struct GpuParticle { float a[4], b[4], c[4]; };   // matches Particle (48 B)
static_assert(sizeof(GpuParticle) == 48, "GpuParticle layout mismatch");

struct UpdateUniforms {
  uint32_t count, pool_max, frame_index, do_reset;
  float    dt, life_s, respawn_delay_s, size;
  float    size_jitter, life_jitter, hue_jitter, _pad0;
  float    vel_x, vel_y, vel_x_jitter, vel_y_jitter;
  uint32_t respect_bounds, seed, _pad1, _pad2;
  uint32_t bar_nv[4];    // active voice count per bar
  float    voices[64];   // [bar*4 + slot] × (y_peak, sigma_trail, sigma_lead, weight)
};
static_assert(sizeof(UpdateUniforms) == 352, "UpdateUniforms layout mismatch");

struct PrefillUniforms { uint32_t debug_region, _pad0; float _pad1, _pad2; float peaks[4]; };
static_assert(sizeof(PrefillUniforms) == 32, "PrefillUniforms layout mismatch");

struct VsUniforms { float aspect_x, aspect_y, _pad0, _pad1; };
static_assert(sizeof(VsUniforms) == 16, "VsUniforms layout mismatch");

struct FsUniforms {
  float    hue, intensity, frame_alpha_jitter, alpha_curve;
  uint32_t shape_kind, frame_index; float shape_param, _pad;
};
static_assert(sizeof(FsUniforms) == 32, "FsUniforms layout mismatch");

struct State {
  gpu::Buffer  part_buf, update_uniform_buf, prefill_uniform_buf, vs_uniform_buf, fs_uniform_buf;
  gpu::Texture motion_tex, zero_motion_tex;
  int          motion_w = 0, motion_h = 0;
  bool         initialized = false;

  // Standard.
  bool  gate = false;
  float level = 0.0f;
  fx::AutoTrigger auto_trig;   // Off / Random (Poisson) / Beats — see effect_auto_trigger.h
  float top_band_height = 0.1f;
  float release_s = 0.8f;
  float release_curve = 1.5f;   // trailing-edge acceleration exponent
  float release_tilt = 0.0f;    // -1 peak at trailing (top), +1 at leading (bottom)
  float min_sustain_s = 0.3f;
  bool  default_gate_state = false;
  float intensity = 1.0f;
  float hue = 0.12f;
  float hue_jitter = 0.08f;
  int   density = 60;
  // Tuning.
  int   bar_target_mode = BAR_ALL;
  int   one_bar_target = 0;       // only used when bar_target_mode == one_bar
  float particle_life_ms = 200.0f;
  float respawn_delay_ms = 30.0f;
  float life_jitter = 0.4f;
  float size = 0.008f;
  float size_jitter = 0.5f;
  float frame_alpha_jitter = 0.6f;
  int   shape_kind = 2;
  float shape_param = 0.7f;
  float alpha_curve = 1.5f;
  int   pool_max = 1024;
  int   seed = 12345;
  // Velocity.
  float particle_velocity_y = 0.0f;
  float particle_velocity_x = 0.0f;
  float velocity_y_jitter = 0.0f;
  float velocity_x_jitter = 0.0f;
  bool  respect_position_bounds = true;
  bool  debug_show_region = false;

  // Runtime.
  uint32_t frame_index = 0;
  float    frame_dt = 0.0f;
  bool     needs_reset = true;
  int      last_pool_max = -1, last_seed = 0x7FFFFFFF;
  // Polyphonic voices: a note-on allocates a sustaining voice; note-off
  // converts it to a release wave. Up to 4 active at once.
  // Per-bar voice pools: each bar has its own polyphony (VOICES_PER_BAR).
  // sustain_remaining: < 0 = a held voice (gate/level/default, released by the
  // held-edge logic); >= 0 = a discrete timed note that counts down then
  // releases. t = release-elapsed (only once sustaining is false).
  struct Voice { bool active = false; bool sustaining = false; float t = 0.0f; float sustain_remaining = 0.0f; };
  Voice    voices[4][VOICES_PER_BAR];     // [bar][slot]
  int      held_idx[4] = { -1, -1, -1, -1 };   // per-bar held voice
  uint32_t voice_bar_rng = 0xB16B00B5u;   // random bar pick (random_bar mode)
  bool     held_prev = false;
  bool     gate_prev = false;
  float    trigger_prev = 0.0f;
};

static gpu::ComputePSO s_pso_update, s_pso_prefill, s_pso_motion;
static gpu::RenderPSO  s_pso_render_add;

static inline float clampf(float v, float lo, float hi) { return v < lo ? lo : (v > hi ? hi : v); }
static inline int   clampi(int v, int lo, int hi)       { return v < lo ? lo : (v > hi ? hi : v); }

static void apply_mode_visibility(int mode) {
  state::setFieldHidden("one_bar_target", mode != BAR_ONE);
}

// Static (self-less) visibility evaluator — pure over state (see crop). Covers
// BOTH the bar_target_mode knobs and the shared auto-trigger block's.
void eval_visibility(int n, const char* pb, const int* off, const int* len, const int* ops) {
  int mode = BAR_ALL;
  for (int i = 0; i < n; i++) {
    if (ops[i] != state::PatchReplace) continue;
    if (state::pathIs(pb + off[i], len[i], "bar_target_mode")) mode = (int)state::patchFloat(i);
  }
  apply_mode_visibility(mode);
  fx::AutoTrigger::evalVisibility(n, pb, off, len, ops);
}

static void on_state_ready(void* self) {
  auto* s = static_cast<State*>(self);
  if (!s) return;
  apply_mode_visibility(s->bar_target_mode);
  fx::AutoTrigger::applyVisibility(s->auto_trig.mode, s->auto_trig.div);
}

// Allocate a voice in a specific bar's pool (free slot, else steal the oldest
// release wave in THAT bar). Sets it sustaining; the CALLER sets
// sustain_remaining (-1 = held; >= 0 = timed). Returns -1 if the bar is full
// of sustaining voices.
static int alloc_voice(State* s, int bar) {
  int slot = -1;
  for (int k = 0; k < VOICES_PER_BAR; k++) if (!s->voices[bar][k].active) { slot = k; break; }
  if (slot < 0) {
    float best = -1.0f;
    for (int k = 0; k < VOICES_PER_BAR; k++)
      if (s->voices[bar][k].active && !s->voices[bar][k].sustaining && s->voices[bar][k].t > best) { best = s->voices[bar][k].t; slot = k; }
  }
  if (slot < 0) return -1;
  s->voices[bar][slot] = State::Voice{};
  s->voices[bar][slot].active = true;
  s->voices[bar][slot].sustaining = true;
  return slot;
}

// Which bar(s) a new note targets, per bar_target_mode. Fills `bars`, returns
// the count (1 for one_bar/random_bar, 4 for all_bars).
static int target_bars(State* s, int* bars) {
  if (s->bar_target_mode == BAR_ONE) { bars[0] = clampi(s->one_bar_target, 0, 3); return 1; }
  if (s->bar_target_mode == BAR_RANDOM) {
    s->voice_bar_rng = s->voice_bar_rng * 1664525u + 1013904223u;
    uint32_t h = s->voice_bar_rng ^ ((uint32_t)s->seed * 0x9E3779B9u);
    bars[0] = (int)((h >> 13) & 3u);
    return 1;
  }
  bars[0] = 0; bars[1] = 1; bars[2] = 2; bars[3] = 3; return 4;   // all_bars
}

// Fire a note (held or timed) into its target bar(s). `held` voices get -1
// (released by the held-edge logic); timed get a min_sustain countdown.
static void fire_note(State* s, bool held) {
  int bars[4]; int nb = target_bars(s, bars);
  float min_sustain = clampf(s->min_sustain_s, 0.0f, 2.0f);
  for (int j = 0; j < nb; j++) {
    int bar = bars[j];
    int slot = alloc_voice(s, bar);
    if (slot < 0) continue;
    if (held) { s->voices[bar][slot].sustain_remaining = -1.0f; s->held_idx[bar] = slot; }
    else      { s->voices[bar][slot].sustain_remaining = min_sustain; }
  }
}

void module_init() {
  // fx::AutoTrigger::fields() wraps the chain to splice the auto-fire block
  // into the Trigger group (it takes and returns the Schema&).
  state::init("source.light.tingle_top", {1, 0, 1},
    fx::AutoTrigger::fields(
    state::Schema()
      // Top-level manual: high-level "what is this / how to use / what to try".
      .helpField("intro",
        "## Tingle Top\n"
        "Bundles sparkles at the top of each bar while a note is held, then releases "
        "them as a wave that cascades and drains off the bottom when you let go. It's "
        "polyphonic — overlapping releases pile up — and can be driven by a gate, a "
        "trigger pulse, an audio level, or its own auto-fire.\n\n"
        "**Try:** drive *Gate* or *Trigger* from a MIDI note or beat; shape the fall "
        "with *Release Time*, *Release Curve* and *Release Tilt*; turn up *Auto Rate* "
        "for a self-running shimmer. Switch *Default Gate* on for a permanently-"
        "sustaining bed.")
      // --- Trigger ---
      .group("trigger", "Trigger")
        .groupHelp(
          "How sparkles are summoned. Any of *Gate* (held), *Trigger* (a pulse), or "
          "*Level* (≥ 0.5) starts a held cluster at the top of the bar. *Auto Mode* "
          "is **Off** by default — set it to **Random** to fire its own notes on top "
          "(Poisson, at *Auto Rate*), or **Beats** to lock them to the host transport. "
          "Drive these from MIDI, envelopes, or audio to make the sparkles play along.")
      .boolField ("gate",                false,                state::PrimaryInput).label("Gate", "Gate")
      .eventField("trigger",                                   state::PrimaryInput).label("Trigger", "Trig")
      .floatField("level",               0.0f, 0.0f, 1.0f,     state::PrimaryInput).label("Level", "Lvl")
    )   // ← fx::AutoTrigger::fields: auto_mode + auto_rate + auto_beats + custom
      // --- Release Wave ---
      .group("release", "Release Wave")
        .groupHelp(
          "When a note lets go, its cluster drops as a wave. *Release Time* is how "
          "long the fall lasts; *Release Curve* accelerates the trailing edge; "
          "*Release Tilt* slides the dense peak between the top and bottom of the "
          "wave. *Min Sustain* keeps very short notes on screen long enough to read.")
      .floatField("top_band_height",     0.1f, 0.01f, 0.5f,    state::PrimaryInput).label("Top Band Height", "Band")
      .floatField("release_s",           0.8f, 0.05f, 4.0f,    state::PrimaryInput).label("Release Time", "Rel")
      .floatField("release_curve",       1.5f, 0.25f, 4.0f,    state::PrimaryInput).label("Release Curve", "RCurve")
      .floatField("release_tilt",        0.0f, -1.0f, 1.0f,    state::PrimaryInput).label("Release Tilt", "Tilt")
      .floatField("min_sustain_s",       0.3f, 0.0f, 2.0f,     state::PrimaryInput).label("Min Sustain", "Sus")
      .boolField ("default_gate_state",  false,                state::PrimaryInput).label("Default Gate", "DefGt")
      // --- Appearance ---
      .group("appearance", "Appearance")
      .floatField("intensity",           1.0f, 0.0f, 2.0f,     state::PrimaryInput).label("Intensity", "Int")
      .floatField("hue",                 0.12f, 0.0f, 1.0f,    state::PrimaryInput).label("Hue", "Hue")
      .floatField("hue_jitter",          0.08f, 0.0f, 0.5f,    state::PrimaryInput).label("Hue Jitter", "HueJit")
      .intField  ("density",             60, 1, 400,           state::PrimaryInput).label("Density", "Dens")
      // --- Bars ---
      .group("bars", "Bars")
      .selectField("bar_target_mode",    BAR_ALL, state::PrimaryInput,
                   {{"one_bar", BAR_ONE}, {"random_bar", BAR_RANDOM}, {"all_bars", BAR_ALL}}).label("Bar Target", "Bars")
      .intField  ("one_bar_target",      0, 0, 3,              state::PrimaryInput).label("Target Bar", "Bar")
      // --- Particles ---
      .group("particles", "Particles")
        .groupHelp(
          "The sparkle grains themselves — how long each lives, how big it is, and "
          "how it twinkles. *Shape* picks the grain sprite; *Size* / *Size Jitter* set "
          "the footprint; *Alpha Jitter* and *Alpha Curve* control the per-frame "
          "flicker. *Pool Max* caps the total grain budget.")
      .floatField("particle_life_ms",    200.0f, 10.0f, 1000.0f, state::PrimaryInput).label("Particle Life", "Life")
      .floatField("respawn_delay_ms",    30.0f, 0.0f, 500.0f,  state::PrimaryInput).label("Respawn Delay", "Delay")
      .floatField("life_jitter",         0.4f, 0.0f, 1.0f,     state::PrimaryInput).label("Life Jitter", "LfJit")
      .floatField("size",                0.008f, 0.001f, 0.05f, state::PrimaryInput).label("Size", "Size")
      .floatField("size_jitter",         0.5f, 0.0f, 1.0f,     state::PrimaryInput).label("Size Jitter", "SzJit")
      .floatField("frame_alpha_jitter",  0.6f, 0.0f, 1.0f,     state::PrimaryInput).label("Alpha Jitter", "AJit")
      .selectField("shape_kind",         2, state::PrimaryInput,
                   {{"solid", 0}, {"circle", 1}, {"gaussian", 2}}).label("Shape", "Shape")
      .floatField("shape_param",         0.7f, 0.0f, 1.0f,     state::PrimaryInput).label("Shape Param", "Shp")
      .floatField("alpha_curve",         1.5f, 0.25f, 4.0f,    state::PrimaryInput).label("Alpha Curve", "ACurve")
      .intField  ("pool_max",            1024, 8, 2048,        state::PrimaryInput).label("Pool Max", "Pool")
      .intField  ("seed",                12345, 0, 0x7FFFFFFF, state::PrimaryInput).label("Seed", "Seed")
      // --- Velocity ---
      .group("velocity", "Velocity")
        .groupHelp(
          "An optional per-particle drift on top of the fall. *Velocity X/Y* push "
          "every grain in a direction; the *Jitter* knobs randomize it per particle. "
          "*Respect Bounds* keeps drifting grains inside the bar.")
      .floatField("particle_velocity_y", 0.0f, -2.0f, 2.0f,    state::PrimaryInput).label("Velocity Y", "VelY")
      .floatField("particle_velocity_x", 0.0f, -2.0f, 2.0f,    state::PrimaryInput).label("Velocity X", "VelX")
      .floatField("velocity_y_jitter",   0.0f, 0.0f, 1.0f,     state::PrimaryInput).label("Vel Y Jitter", "VYJit")
      .floatField("velocity_x_jitter",   0.0f, 0.0f, 1.0f,     state::PrimaryInput).label("Vel X Jitter", "VXJit")
      .boolField ("respect_position_bounds", true,             state::PrimaryInput).label("Respect Bounds", "Bounds")
      // --- Debug ---
      .group("debug", "Debug")
      .boolField ("debug_show_region",   false,                state::PrimaryInput).label("Show Region", "Region")
      .textureField("tex_in",  state::PrimaryInput)
      .textureField("tex_out", state::PrimaryOutput)
      .renderOutputs(state::PrimaryOutput)
      .renderOutputs(state::PrimaryInput, "render_outputs_in")
        .capability(state::Capability::Generator)
    );
  state::setOnStateReady(&on_state_ready);

  if (gpu::Device::backend() == gpu::Backend::None) return;

  state::registerShaderSPV("tingle_top_update",  UPDATE_SPV,  UPDATE_SPV_SIZE);
  state::registerShaderSPV("tingle_top_prefill", PREFILL_SPV, PREFILL_SPV_SIZE);
  state::registerShaderSPV("tingle_top_vs",      VS_SPV,      VS_SPV_SIZE);
  state::registerShaderSPV("tingle_top_fs",      FS_SPV,      FS_SPV_SIZE);
  state::registerShaderSPV("tingle_top_motion",  MOTION_SPV,  MOTION_SPV_SIZE,
                           "rgba16float", "write");
  auto cs_u  = gpu::Device::createShaderModuleByName("tingle_top_update");
  auto cs_pf = gpu::Device::createShaderModuleByName("tingle_top_prefill");
  auto vs    = gpu::Device::createShaderModuleByName("tingle_top_vs");
  auto fs    = gpu::Device::createShaderModuleByName("tingle_top_fs");
  auto cs_m  = gpu::Device::createShaderModuleByName("tingle_top_motion");
  if (!cs_u || !cs_pf || !vs || !fs || !cs_m) return;

  s_pso_update  = gpu::Device::createComputePSO(cs_u,  "main", gpu::Bindings().storageRW(0).uniform(1));
  s_pso_prefill = gpu::Device::createComputePSO(cs_pf, "main", gpu::Bindings()
      .tex2d(0).storageTex2d(1).uniform(2));
  // Instanced sparkle quads, additive over the pre-filled input.
  s_pso_render_add = gpu::Device::createInstancedRenderPSO(
      vs, "main", fs, "main", gpu::TextureFormat::Surface,
      gpu::Bindings().storage(0).uniform(1).uniform(2),
      gpu::Device::BlendMode::Additive);
  s_pso_motion = gpu::Device::createComputePSO(cs_m, "main", gpu::Bindings()
      .tex2d(0).storageTex2d(1, gpu::TextureFormat::RGBA16F));

  state::log("tingle_top: module initialized");
}

void* create() {
  auto* s = new State();
  s->part_buf            = gpu::Device::createBuffer(sizeof(GpuParticle) * POOL_HARD_MAX, gpu::BufferUsage::Storage);
  s->update_uniform_buf  = gpu::Device::createBuffer(sizeof(UpdateUniforms),  gpu::BufferUsage::Uniform);
  s->prefill_uniform_buf = gpu::Device::createBuffer(sizeof(PrefillUniforms), gpu::BufferUsage::Uniform);
  s->vs_uniform_buf      = gpu::Device::createBuffer(sizeof(VsUniforms),      gpu::BufferUsage::Uniform);
  s->fs_uniform_buf      = gpu::Device::createBuffer(sizeof(FsUniforms),      gpu::BufferUsage::Uniform);
  return s;
}

void destroy(void* self) {
  auto* s = static_cast<State*>(self);
  if (!s) return;
  s->part_buf.release();
  s->update_uniform_buf.release();
  s->prefill_uniform_buf.release();
  s->vs_uniform_buf.release();
  s->fs_uniform_buf.release();
  s->motion_tex.release();
  s->zero_motion_tex.release();
  delete s;
}

void init(void* self) {
  auto* s = static_cast<State*>(self);
  if (!s) return;
  s->frame_index = 0;
  s->frame_dt = 0.0f;
  s->needs_reset = true;
  s->last_pool_max = -1;
  s->last_seed = 0x7FFFFFFF;
  s->motion_w = s->motion_h = 0;
  for (int b = 0; b < 4; b++) {
    for (int k = 0; k < VOICES_PER_BAR; k++) s->voices[b][k] = State::Voice{};
    s->held_idx[b] = -1;
  }
  s->voice_bar_rng = 0xB16B00B5u;
  s->held_prev = false;
  s->gate_prev = false;
  s->trigger_prev = 0.0f;
  if (!s_pso_update.valid() || !s_pso_prefill.valid() || !s_pso_render_add.valid() || !s_pso_motion.valid()) return;
  if (!s->part_buf.valid() || !s->update_uniform_buf.valid() || !s->prefill_uniform_buf.valid()
      || !s->vs_uniform_buf.valid() || !s->fs_uniform_buf.valid()) return;
  s->initialized = true;
}

void tick(void* self, double dt) {
  auto* s = static_cast<State*>(self);
  if (!s || !s->initialized) return;
  float fdt = (float)(dt > 0.1 ? 0.1 : (dt < 0.0 ? 0.0 : dt));
  s->frame_dt = fdt;
  s->frame_index++;

  // Self-fire (Off / Random / Beats — effect_auto_trigger.h) → a DISTINCT timed
  // note (its own sustain countdown), so clustered events form polyphony rather
  // than extending one held note. Loop the count: in Beats a long frame stall
  // can cross several divisions at once.
  for (int i = 0, nf = s->auto_trig.fires(fdt); i < nf; i++) fire_note(s, false);

  // The continuous "held" sources drive the per-bar held voice(s).
  bool held = s->gate || (s->level >= 0.5f) || s->default_gate_state;
  if (held && !s->held_prev) fire_note(s, true);
  if (!held && s->held_prev) {                          // note-off → release held voices
    for (int b = 0; b < 4; b++) if (s->held_idx[b] >= 0) {
      s->voices[b][s->held_idx[b]].sustaining = false;
      s->voices[b][s->held_idx[b]].t = 0.0f;
      s->held_idx[b] = -1;
    }
  }

  // Advance voices: timed notes count down then release; held voices wait for
  // the held-edge logic; release waves run out and retire to the pool.
  float rs = clampf(s->release_s, 0.05f, 4.0f);
  for (int b = 0; b < 4; b++) for (int k = 0; k < VOICES_PER_BAR; k++) {
    State::Voice& v = s->voices[b][k];
    if (!v.active) continue;
    if (v.sustaining) {
      if (v.sustain_remaining >= 0.0f) {               // timed note
        v.sustain_remaining -= fdt;
        if (v.sustain_remaining <= 0.0f) { v.sustaining = false; v.t = 0.0f; }
      }
    } else {
      v.t += fdt;
      if (v.t >= rs) v.active = false;
    }
  }
  s->held_prev = held;
}


void on_state_patched(void* self, int n, const char* pb, const int* off,
                      const int* len, const int* ops) {
  auto* s = static_cast<State*>(self);
  if (!s) return;
  bool vis_changed = false;
  for (int i = 0; i < n; i++) {
    if (ops[i] != state::PatchReplace) continue;
    const char* path = pb + off[i];
    int plen = len[i];
    if (s->auto_trig.patch(path, plen, i, &vis_changed)) continue;
    if (state::pathIs(path, plen, "gate")) {
      s->gate = state::patchFloat(i) != 0.0f; s->gate_prev = s->gate;
    } else if (state::pathIs(path, plen, "trigger")) {
      float v = state::patchFloat(i);
      if (v != 0.0f && s->trigger_prev == 0.0f) fire_note(s, false);   // rising edge → a timed note
      s->trigger_prev = v;
    }
    else if (state::pathIs(path, plen, "level"))               s->level = state::patchFloat(i);
    else if (state::pathIs(path, plen, "top_band_height"))     s->top_band_height = state::patchFloat(i);
    else if (state::pathIs(path, plen, "release_s"))           s->release_s = state::patchFloat(i);
    else if (state::pathIs(path, plen, "release_curve"))       s->release_curve = state::patchFloat(i);
    else if (state::pathIs(path, plen, "release_tilt"))        s->release_tilt = state::patchFloat(i);
    else if (state::pathIs(path, plen, "min_sustain_s"))       s->min_sustain_s = state::patchFloat(i);
    else if (state::pathIs(path, plen, "default_gate_state"))  s->default_gate_state = state::patchFloat(i) != 0.0f;
    else if (state::pathIs(path, plen, "intensity"))           s->intensity = state::patchFloat(i);
    else if (state::pathIs(path, plen, "hue"))                 s->hue = state::patchFloat(i);
    else if (state::pathIs(path, plen, "hue_jitter"))          s->hue_jitter = state::patchFloat(i);
    else if (state::pathIs(path, plen, "density"))             s->density = (int)state::patchFloat(i);
    else if (state::pathIs(path, plen, "bar_target_mode")) {
      s->bar_target_mode = (int)state::patchFloat(i);
      apply_mode_visibility(s->bar_target_mode);
    }
    else if (state::pathIs(path, plen, "one_bar_target"))      s->one_bar_target = (int)state::patchFloat(i);
    else if (state::pathIs(path, plen, "particle_life_ms"))    s->particle_life_ms = state::patchFloat(i);
    else if (state::pathIs(path, plen, "respawn_delay_ms"))    s->respawn_delay_ms = state::patchFloat(i);
    else if (state::pathIs(path, plen, "life_jitter"))         s->life_jitter = state::patchFloat(i);
    else if (state::pathIs(path, plen, "size"))                s->size = state::patchFloat(i);
    else if (state::pathIs(path, plen, "size_jitter"))         s->size_jitter = state::patchFloat(i);
    else if (state::pathIs(path, plen, "frame_alpha_jitter"))  s->frame_alpha_jitter = state::patchFloat(i);
    else if (state::pathIs(path, plen, "shape_kind"))          s->shape_kind = (int)state::patchFloat(i);
    else if (state::pathIs(path, plen, "shape_param"))         s->shape_param = state::patchFloat(i);
    else if (state::pathIs(path, plen, "alpha_curve"))         s->alpha_curve = state::patchFloat(i);
    else if (state::pathIs(path, plen, "pool_max"))            s->pool_max = (int)state::patchFloat(i);
    else if (state::pathIs(path, plen, "seed"))                s->seed = (int)state::patchFloat(i);
    else if (state::pathIs(path, plen, "particle_velocity_y")) s->particle_velocity_y = state::patchFloat(i);
    else if (state::pathIs(path, plen, "particle_velocity_x")) s->particle_velocity_x = state::patchFloat(i);
    else if (state::pathIs(path, plen, "velocity_y_jitter"))   s->velocity_y_jitter = state::patchFloat(i);
    else if (state::pathIs(path, plen, "velocity_x_jitter"))   s->velocity_x_jitter = state::patchFloat(i);
    else if (state::pathIs(path, plen, "respect_position_bounds")) s->respect_position_bounds = state::patchFloat(i) != 0.0f;
    else if (state::pathIs(path, plen, "debug_show_region"))   s->debug_show_region = state::patchFloat(i) != 0.0f;
  }
  if (vis_changed)
    fx::AutoTrigger::applyVisibility(s->auto_trig.mode, s->auto_trig.div);
}

void render(void* self, int vp_w, int vp_h) {
  auto* s = static_cast<State*>(self);
  if (!s || !s->initialized || vp_w <= 0 || vp_h <= 0) return;
  auto in  = gpu::Device::textureForField("tex_in");
  auto out = gpu::Device::textureForField("tex_out");
  if (!in.valid() || !out.valid()) return;

  int pool_max = clampi(s->pool_max, 8, POOL_HARD_MAX);
  // Slot i belongs to bar (i & 3), so the pool is partitioned 1/4 per bar:
  // density particles per bar. (one_bar leaves the other bars' slots idle.)
  int count = clampi(clampi(s->density, 1, 400) * 4, 0, pool_max);

  if (pool_max != s->last_pool_max || s->seed != s->last_seed) {
    s->needs_reset = true; s->last_pool_max = pool_max; s->last_seed = s->seed;
  }

  // Build each active voice's spawn-y distribution (the mixture the pool
  // samples). Sustain → small gaussian at the top band. Release → split-normal
  // gaussian whose window's leading edge bursts down (ease-out) while the
  // trailing edge accelerates (power curve); the peak tilts forward/back.
  const float kEdgeMargin = 0.25f;   // wave overshoots past the bottom
  const float kSigmaFrac  = 0.42f;   // gaussian spread as a fraction of the window
  float tbh    = clampf(s->top_band_height, 0.01f, 0.5f);
  float relS   = clampf(s->release_s, 0.05f, 4.0f);
  float accExp = clampf(s->release_curve, 0.25f, 4.0f);
  float peakPos = clampf(0.5f + 0.5f * clampf(s->release_tilt, -1.0f, 1.0f), 0.0f, 1.0f);
  float vparams[64] = {0};
  float peaks[4] = { -1.0f, -1.0f, -1.0f, -1.0f };
  uint32_t bar_nv[4] = { 0, 0, 0, 0 };
  for (int b = 0; b < 4; b++) {
    int nvb = 0;
    for (int k = 0; k < VOICES_PER_BAR; k++) {
      State::Voice& vc = s->voices[b][k];
      if (!vc.active) continue;
      float yp, st, sl;
      if (vc.sustaining) {
        yp = tbh * 0.5f; st = tbh * 0.5f; sl = tbh * 0.5f;
      } else {
        float p      = clampf(vc.t / relS, 0.0f, 1.0f);
        float accel  = std::pow(p, accExp);             // trailing edge (top), slow→fast
        float burst  = 1.0f - (1.0f - p) * (1.0f - p);  // leading edge (bottom), fast ease-out
        float yTrail = tbh + (1.0f - tbh) * accel;
        float yLead  = tbh + (1.0f - tbh + kEdgeMargin) * burst;
        yp = yTrail + (yLead - yTrail) * peakPos;
        st = std::max((yp - yTrail) * kSigmaFrac, 1e-4f);
        sl = std::max((yLead - yp) * kSigmaFrac, 1e-4f);
      }
      int idx = (b * VOICES_PER_BAR + nvb) * 4;
      vparams[idx + 0] = yp; vparams[idx + 1] = st;
      vparams[idx + 2] = sl; vparams[idx + 3] = 1.0f;     // equal weight → constant density
      if (nvb == 0) peaks[b] = yp;                         // debug: leading voice per bar
      nvb++;
    }
    bar_nv[b] = (uint32_t)nvb;
  }

  // Pass 1 — update the pool.
  UpdateUniforms uu = {};
  uu.count = (uint32_t)count;
  uu.pool_max = (uint32_t)pool_max;
  uu.frame_index = s->frame_index;
  uu.do_reset = s->needs_reset ? 1u : 0u;
  uu.dt = s->frame_dt;
  uu.life_s = clampf(s->particle_life_ms, 10.0f, 1000.0f) * 0.001f;
  uu.respawn_delay_s = clampf(s->respawn_delay_ms, 0.0f, 500.0f) * 0.001f;
  uu.life_jitter = clampf(s->life_jitter, 0.0f, 1.0f);
  uu.size = clampf(s->size, 0.001f, 0.05f);
  uu.size_jitter = clampf(s->size_jitter, 0.0f, 1.0f);
  uu.vel_x = clampf(s->particle_velocity_x, -2.0f, 2.0f);
  uu.vel_y = clampf(s->particle_velocity_y, -2.0f, 2.0f);
  uu.vel_x_jitter = clampf(s->velocity_x_jitter, 0.0f, 1.0f);
  uu.vel_y_jitter = clampf(s->velocity_y_jitter, 0.0f, 1.0f);
  uu.hue_jitter = clampf(s->hue_jitter, 0.0f, 0.5f);
  uu.respect_bounds = s->respect_position_bounds ? 1u : 0u;
  uu.seed = (uint32_t)s->seed;
  for (int b = 0; b < 4; b++) uu.bar_nv[b] = bar_nv[b];
  for (int k = 0; k < 64; k++) uu.voices[k] = vparams[k];
  s->update_uniform_buf.writeOne(uu);
  {
    auto cp = gpu::ComputePass::begin();
    cp.setPSO(s_pso_update);
    cp.setBuffer(s->part_buf, 0);
    cp.setBuffer(s->update_uniform_buf, 1);
    cp.dispatch((pool_max + 63) / 64, 1, 1);
    cp.end();
  }

  // Pass 2 — prefill tex_out with tex_in (so the additive quads blend over it).
  PrefillUniforms pu = {};
  pu.debug_region = s->debug_show_region ? 1u : 0u;
  for (int k = 0; k < 4; k++) pu.peaks[k] = peaks[k];
  s->prefill_uniform_buf.writeOne(pu);
  {
    auto cp = gpu::ComputePass::begin();
    cp.setPSO(s_pso_prefill);
    cp.setTexture(in, 0, 0);
    cp.setTexture(out, 1, 1);
    cp.setBuffer(s->prefill_uniform_buf, 2);
    cp.dispatch((vp_w + 7) / 8, (vp_h + 7) / 8);
    cp.end();
  }

  // Pass 3 — rasterize sparkles as instanced quads, additive over tex_out.
  if (count > 0 && s->intensity > 0.0f) {
    float mn = (float)(vp_w < vp_h ? vp_w : vp_h);
    VsUniforms vu = {}; vu.aspect_x = mn / (float)vp_w; vu.aspect_y = mn / (float)vp_h;
    s->vs_uniform_buf.writeOne(vu);
    FsUniforms fu = {};
    fu.hue = s->hue;
    fu.intensity = clampf(s->intensity, 0.0f, 2.0f);
    fu.frame_alpha_jitter = clampf(s->frame_alpha_jitter, 0.0f, 1.0f);
    fu.alpha_curve = clampf(s->alpha_curve, 0.25f, 4.0f);
    fu.shape_kind = (uint32_t)clampi(s->shape_kind, 0, 2);
    fu.frame_index = s->frame_index;
    fu.shape_param = clampf(s->shape_param, 0.0f, 1.0f);
    s->fs_uniform_buf.writeOne(fu);

    auto rp = gpu::RenderPass::beginLoad(out);
    rp.setPSO(s_pso_render_add);
    rp.setBuffer(s->part_buf, 0);
    rp.setBuffer(s->vs_uniform_buf, 1);
    rp.setBuffer(s->fs_uniform_buf, 2);
    rp.draw(6, count);
    rp.end();
  }

  // Pass 3 — motion passthrough.
  if (state::isOutputConnected("render_outputs")) {
    if (!s->motion_tex.valid() || s->motion_w != vp_w || s->motion_h != vp_h) {
      s->motion_tex = gpu::Device::createTexture(vp_w, vp_h, gpu::TextureFormat::RGBA16F);
      s->motion_w = vp_w; s->motion_h = vp_h;
      if (s->motion_tex.valid()) state::setGpuTexture("render_outputs/motion", s->motion_tex.id);
    }
    if (s->motion_tex.valid()) {
      auto upstream = gpu::Device::textureForField("render_outputs_in/motion");
      if (!upstream.valid()) {
        if (!s->zero_motion_tex.valid())
          s->zero_motion_tex = gpu::Device::createTexture(1, 1, gpu::TextureFormat::RGBA16F);
        upstream = s->zero_motion_tex;
      }
      if (upstream.valid()) {
        auto cp = gpu::ComputePass::begin();
        cp.setPSO(s_pso_motion);
        cp.setTexture(upstream, 0, 0);
        cp.setTexture(s->motion_tex, 1, 1);
        cp.dispatch((vp_w + 7) / 8, (vp_h + 7) / 8);
        cp.end();
      }
    }
  }

  gpu::Device::submit();

  s->needs_reset = false;
  s->frame_dt = 0.0f;
}

} // namespace tingle_top
