/*
 * source.light.chroma_wave — charge-and-burst prismatic wave bloom (polyphonic).
 *
 * A voice is a soft super-gaussian blob that grows from a (jittered) position
 * while gated; as pressure builds the top flattens into a plateau, the blob
 * elongates in X / squishes in Y, and a DoG carve hollows the top so the mass
 * piles into a downward crescent (max pressure). On release it bursts: the
 * radius expands, the crescent opens, contrast shallows, band contrast washes
 * out, and the colour-grade transfer folds so prismatic bands travel down the
 * gradient (dominant) and wash back up the inner edge (secondary).
 *
 * POLYPHONY: up to MAX_VOICES voices run concurrently, each its own CPU-side
 * envelope. A held gate/level holds ONE voice (charges until released); each
 * trigger event and auto-fire event spawns a fresh one-shot voice. The
 * CPU advances every active envelope, packs the active voices' blob geometry +
 * grade params into a storage buffer, and tells the shader how many to render;
 * the shader loops over them and accumulates the band phase (a small per-pixel
 * loop — the §3.5 "precise blend" exception, since the phase-domain interaction
 * can't be expressed by fixed-function blend). The buffer is fixed at
 * MAX_VOICES and we just vary the count — no per-frame GPU resize.
 *
 * Trigger surface (§8.1/§8.2): gate/level/default_gate_state hold a voice;
 * `trigger` is a momentary event fired on the rising edge of its value
 * (replay-safe); `auto_mode` is the shared self-fire block (Off by default —
 * Random is Poisson, Beats locks to the host transport). The blob's colour is
 * GENERATED from the density grade, composited additively over tex_in.
 *
 * Secondary `wave_out` texture output: the wave's additive contribution alone,
 * on opaque black — written by the same render dispatch, produced only while
 * a wire is attached (same gating as the motion-vector output).
 */

#include <gpu.h>
#include <host.h>
#include <effect_utils.h>
#include <effect_auto_trigger.h>  // fx::AutoTrigger — the shared Off/Random/Beats self-fire
#include "chroma_wave_shaders.h"

#include <cmath>
#include <cstdint>
#include <cstring>

namespace chroma_wave {

enum Phase : int {
  PHASE_CHARGE = 0,
  PHASE_BURST  = 1,
};

static constexpr int MAX_VOICES = 32;

// Constant (per-frame, all-voice) tuning pushed in the cbuffer.
struct Uniforms {
  // row 0
  float cres_off;
  float band_tilt;
  float hue_span;
  float saturation;
  // row 1
  float alpha_gamma;
  float intensity;
  float base_hue;
  float debug_field;
  // row 2
  float    color_r;
  float    color_g;
  float    color_b;
  uint32_t voice_count;
  // row 3
  float    hue_interact;
  float    hue_warp_a;   // shift(h) = a + b·cos(2πh) + c·sin(2πh)
  float    hue_warp_b;
  float    hue_warp_c;
  // row 4
  float    write_wave;   // 1 = also write the isolated wave layer (wave_out)
  float    _pw0;
  float    _pw1;
  float    _pw2;
};
static_assert(sizeof(Uniforms) == 80, "Uniforms layout mismatch");

// Per-voice blob geometry + grade, packed as 4 float4 (matches render.hlsl).
struct VoiceGpu {
  float cx, cy, radius, elong;                          // a
  float ycomp, sharp, plateau_p, cres;                  // b
  float grade_freq, grade_phase, band_contrast, overlay_alpha;  // c
  float hue_offset, _p0, _p1, _p2;                      // d
};
static_assert(sizeof(VoiceGpu) == 64, "VoiceGpu layout mismatch");

struct MotionUniforms {
  float    cres_off;
  float    motion_scale;
  float    alpha_gamma;
  float    band_tilt;
  uint32_t voice_count;
  float    motion_warp;
  float    motion_edge_mask;
  float    _p2;
};
static_assert(sizeof(MotionUniforms) == 32, "MotionUniforms layout mismatch");

// One CPU envelope.
struct Voice {
  bool   active   = false;
  bool   held     = false;       // gate-held (charge until release) vs one-shot
  Phase  phase    = PHASE_CHARGE;
  float  charge_t = 0.0f;
  float  burst_t  = 0.0f;
  float  strain_at_release = 0.0f;
  double hold_remaining = 0.0;   // one-shot auto-release timer
  float  smooth_radius  = 0.0f;
  bool   snap_radius    = true;
  float  growth         = 0.0f;  // log radius rate ṙ/r (1/sec), for motion
  float  fold_speed     = 0.0f;  // d(grade_phase)/dt (1/sec), for band motion
  double grade_phase    = 0.0;
  float  pos_x = 0.0f, pos_y = 0.0f;   // signed position captured at spawn
  float  hue_offset = 0.0f;
  uint32_t age = 0;              // spawn order (for stealing)
};

struct State {
  gpu::Buffer  uniform_buf;
  gpu::Buffer  voice_buf;
  gpu::Buffer  motion_uniform_buf;
  gpu::Texture motion_tex;
  gpu::Texture zero_motion_tex;
  int          motion_w = 0;
  int          motion_h = 0;
  gpu::Texture wave_tex;        // wave_out — the isolated wave layer, when wired
  gpu::Texture dummy_wave_tex;  // 1x1 stand-in bound at u4 when wave_out is idle
  int          wave_w = 0;
  int          wave_h = 0;
  bool         initialized = false;

  // --- Standard trigger surface ---
  bool  gate               = false;
  float level              = 0.0f;
  fx::AutoTrigger auto_trig;   // Off / Random (Poisson) / Beats — see effect_auto_trigger.h
  bool  default_gate_state = false;

  // --- Polyphony ---
  int   voice_limit      = 8;
  float voice_pos_jitter = 0.5f;   // per-voice signed position spread (one-shots)
  float voice_hue_jitter = 0.1f;   // per-voice hue offset (one-shots)
  float hue_interact     = 0.8f;   // overlap: 0 avg hue, →2 accumulate/rotate

  // --- Position ---
  float position_x = 0.0f;
  float position_y = -0.7f;        // top-center

  // --- Charge / pressure shape ---
  float charge_s         = 0.6f;
  float base_radius      = 0.12f;
  float charge_expand    = 2.3f;
  float size_smoothing   = 0.06f;
  float gaussian_sharp   = 4.0f;
  float plateau_amount   = 0.6f;
  float squish_amount    = 0.5f;
  float crescent_amount  = 0.7f;
  float crescent_offset  = 0.5f;

  // --- Burst ---
  float release_s      = 0.7f;
  float release_expand = 3.0f;
  float release_curve  = 0.4f;     // signed slider; fx::signedSliderToExp
  float min_sustain_s  = 0.2f;
  float burst_shallow  = 0.7f;

  // --- Colour grade ---
  float base_hue           = 0.55f;
  float hue_span           = 0.18f;
  float saturation         = 0.85f;
  float hue_shift_r        = 0.0f;   // twist the hue wheel at the R/G/B points
  float hue_shift_g        = 0.0f;
  float hue_shift_b        = 0.0f;
  float grade_freq_hold    = 1.5f;
  float grade_freq_burst   = 7.0f;
  float fold_rate          = 1.5f;
  float band_contrast      = 0.6f;
  float band_tilt          = 0.0f;
  float alpha_gamma        = 1.2f;
  float color_r            = 1.0f;
  float color_g            = 1.0f;
  float color_b            = 1.0f;
  float overlay_alpha_hold = 0.3f;
  float overlay_alpha_burst = 0.7f;
  float intensity          = 1.0f;
  float motion_scale       = 1.0f;   // gain on the emitted motion vectors
  float motion_warp        = 0.4f;   // damp lateral spread → coherent wavefront
  float motion_edge_mask   = 0.0f;   // isolate motion to the bands' leading edges

  bool  debug_field = false;

  // --- Runtime ---
  Voice    voices[MAX_VOICES];
  int      held_voice    = -1;     // index of the gate-held voice (-1 = none)
  bool     held_prev     = false;
  bool     trigger_prev  = false;
  uint32_t rng_state     = 0xC0FFEE11u;
  uint32_t jitter_rng    = 0x1234567u;
  uint32_t spawn_counter = 0;
};

static gpu::ComputePSO s_pso;
static gpu::ComputePSO s_pso_motion;

static inline float lerpf(float a, float b, float t) { return a + (b - a) * t; }
static inline float clampf(float v, float lo, float hi) {
  return v < lo ? lo : (v > hi ? hi : v);
}
static inline int clampi(int v, int lo, int hi) {
  return v < lo ? lo : (v > hi ? hi : v);
}
static inline float smooth01(float x) {
  x = clampf(x, 0.0f, 1.0f);
  return x * x * (3.0f - 2.0f * x);
}
static inline float smoothband(float x, float lo, float hi) {
  if (hi <= lo) return x >= hi ? 1.0f : 0.0f;
  return smooth01((x - lo) / (hi - lo));
}
static inline uint32_t lcg_next(uint32_t& s) { s = s * 1664525u + 1013904223u; return s; }
static inline float lcg_unit(uint32_t& s) { return (lcg_next(s) >> 8) * (1.0f / float(1u << 24)); }
static inline float lcg_signed(uint32_t& s) { return lcg_unit(s) * 2.0f - 1.0f; }

// --- Voice pool ----------------------------------------------------------

static int active_count(State* s) {
  int n = 0;
  for (int i = 0; i < MAX_VOICES; i++) if (s->voices[i].active) n++;
  return n;
}

// Find a slot for a new voice: a free slot if we're under the limit, else
// steal the most-progressed (closest-to-dying) BURST voice — never the
// gate-held voice.
static int alloc_slot(State* s) {
  int limit = clampi(s->voice_limit, 1, MAX_VOICES);
  if (active_count(s) < limit) {
    for (int i = 0; i < MAX_VOICES; i++) if (!s->voices[i].active) return i;
  }
  int best = -1; float best_score = -1.0f;
  for (int i = 0; i < MAX_VOICES; i++) {
    if (!s->voices[i].active || i == s->held_voice) continue;
    float score = (s->voices[i].phase == PHASE_BURST) ? (1.0f + s->voices[i].burst_t) : 0.0f;
    if (score > best_score) { best_score = score; best = i; }
  }
  if (best < 0) {                      // fallback: any non-held active slot
    for (int i = 0; i < MAX_VOICES; i++)
      if (s->voices[i].active && i != s->held_voice) { best = i; break; }
  }
  return best < 0 ? 0 : best;
}

static int spawn_voice(State* s, bool held) {
  int idx = alloc_slot(s);
  Voice& v = s->voices[idx];
  v.active = true;
  v.held = held;
  v.phase = PHASE_CHARGE;
  v.charge_t = 0.0f;
  v.burst_t = 0.0f;
  v.strain_at_release = 0.0f;
  v.hold_remaining = held ? 0.0 : (double)(s->charge_s + s->min_sustain_s);
  v.smooth_radius = 0.0f;
  v.snap_radius = true;
  v.grade_phase = 0.0;
  v.age = s->spawn_counter++;
  // Held (gate) voices stay clean at the configured position; one-shots
  // (trigger / auto) scatter so the polyphony reads as distinct waves.
  float jx = 0.0f, jy = 0.0f, jh = 0.0f;
  if (!held) {
    jx = s->voice_pos_jitter * lcg_signed(s->jitter_rng);
    jy = s->voice_pos_jitter * lcg_signed(s->jitter_rng);
    jh = s->voice_hue_jitter * lcg_signed(s->jitter_rng);
  }
  v.pos_x = s->position_x + jx;
  v.pos_y = s->position_y + jy;
  v.hue_offset = jh;
  return idx;
}

// Anisotropy (x elongation / y compression) for the current phase. Shared by
// the GPU packing and the motion extent-rate tracking so they never diverge.
static void voice_elong_ycomp(State* s, const Voice& v, float& elong, float& ycomp) {
  if (v.phase == PHASE_CHARGE) {
    float strain = v.charge_t;
    elong = 1.0f + s->squish_amount * strain;
    ycomp = 1.0f / (1.0f + s->squish_amount * strain * 0.4f);
  } else {
    float bt = clampf(v.burst_t, 0.0f, 1.0f);
    float be = std::pow(bt, fx::signedSliderToExp(s->release_curve));
    float elong_rel = 1.0f + s->squish_amount * v.strain_at_release;
    elong = lerpf(elong_rel, 1.0f, be);
    ycomp = 1.0f;
  }
}

static float voice_target_radius(State* s, const Voice& v) {
  if (v.phase == PHASE_CHARGE)
    return s->base_radius * lerpf(1.0f, s->charge_expand, v.charge_t);
  float bt = clampf(v.burst_t, 0.0f, 1.0f);
  float be = std::pow(bt, fx::signedSliderToExp(s->release_curve));
  float rr = s->base_radius * lerpf(1.0f, s->charge_expand, v.strain_at_release);
  return rr * lerpf(1.0f, s->release_expand, be);
}

static void advance_voice(State* s, Voice& v, double dt) {
  float fdt = (float)dt;
  if (v.held) {
    float rate = (s->charge_s > 1e-4f) ? (fdt / s->charge_s) : 1.0f;
    v.charge_t = clampf(v.charge_t + rate, 0.0f, 1.0f);
  } else if (v.phase == PHASE_CHARGE) {
    v.hold_remaining -= dt;
    float rate = (s->charge_s > 1e-4f) ? (fdt / s->charge_s) : 1.0f;
    v.charge_t = clampf(v.charge_t + rate, 0.0f, 1.0f);
    if (v.hold_remaining <= 0.0) {
      v.strain_at_release = v.charge_t;
      v.phase = PHASE_BURST;
      v.burst_t = 0.0f;
    }
  } else {  // PHASE_BURST
    float rate = (s->release_s > 1e-4f) ? (fdt / s->release_s) : 1.0f;
    v.burst_t += rate;
    if (v.burst_t >= 1.0f) { v.active = false; return; }
  }

  // Smoothed radius + its log growth rate ṙ/r (1/sec). The motion pass uses
  // this as the expansion speed; direction comes from the field gradient. The
  // spawn/retrigger snap frame emits zero rate so there's no velocity spike.
  bool was_snap = v.snap_radius;
  float old_r = v.smooth_radius;
  float tr = voice_target_radius(s, v);
  if (v.snap_radius || s->size_smoothing <= 1e-4f) {
    v.smooth_radius = tr; v.snap_radius = false;
  } else {
    float a = 1.0f - std::exp(-fdt / s->size_smoothing);
    v.smooth_radius += (tr - v.smooth_radius) * a;
  }
  if (was_snap || fdt <= 1e-6f) {
    v.growth = 0.0f;
  } else {
    float r = v.smooth_radius > 1e-5f ? v.smooth_radius : 1e-5f;
    v.growth = ((v.smooth_radius - old_r) / fdt) / r;
  }

  float fold_speed = s->fold_rate * 0.15f;
  if (v.phase == PHASE_BURST) {
    float be = std::pow(clampf(v.burst_t, 0.0f, 1.0f), fx::signedSliderToExp(s->release_curve));
    fold_speed = s->fold_rate * (0.15f + be);
  }
  v.fold_speed = fold_speed;
  v.grade_phase += (double)(fold_speed * fdt);
  if (v.grade_phase > 1024.0) v.grade_phase -= 1024.0;
}

// Fill a voice's GPU record from its envelope state (mirrors the original
// single-blob per-phase geometry; radius comes from the smoothed value).
static void compute_voice_gpu(State* s, const Voice& v, VoiceGpu& o) {
  std::memset(&o, 0, sizeof(o));
  o.cx = v.pos_x * 0.5f + 0.5f;
  o.cy = v.pos_y * 0.5f + 0.5f;
  o.radius = v.smooth_radius;
  o.hue_offset = v.hue_offset;
  o.grade_phase = (float)v.grade_phase;
  // Motion pass reads ṙ/r (_p0) and the grade-phase scroll speed (_p1) to
  // compute the optical flow of the band field.
  o._p0 = v.growth;
  o._p1 = v.fold_speed;

  voice_elong_ycomp(s, v, o.elong, o.ycomp);

  if (v.phase == PHASE_CHARGE) {
    float strain = v.charge_t;
    o.sharp = s->gaussian_sharp;
    o.plateau_p = 1.0f + s->plateau_amount * 1.2f * strain;
    o.cres = s->crescent_amount * smoothband(strain, 0.3f, 1.0f);
    o.grade_freq = s->grade_freq_hold;
    o.band_contrast = s->band_contrast;
    o.overlay_alpha = s->overlay_alpha_hold;
  } else {
    float bt = clampf(v.burst_t, 0.0f, 1.0f);
    float be = std::pow(bt, fx::signedSliderToExp(s->release_curve));
    float strain = v.strain_at_release;
    float p_rel     = 1.0f + s->plateau_amount * 1.2f * strain;
    float cres_rel  = s->crescent_amount * smoothband(strain, 0.3f, 1.0f);
    o.sharp = s->gaussian_sharp * lerpf(1.0f, 1.0f - s->burst_shallow, be);
    o.plateau_p = lerpf(p_rel, 1.0f, be);
    o.cres = lerpf(cres_rel, 0.0f, smoothband(bt, 0.0f, 0.4f));
    o.grade_freq = lerpf(s->grade_freq_hold, s->grade_freq_burst, smoothband(bt, 0.0f, 0.5f));
    o.band_contrast = s->band_contrast * (1.0f - be);
    float rise = smoothband(bt, 0.0f, 0.1f);
    float fade = 1.0f - smoothband(bt, 0.45f, 1.0f);
    o.overlay_alpha = lerpf(s->overlay_alpha_hold, s->overlay_alpha_burst, rise) * fade;
  }
}

// --- ABI -----------------------------------------------------------------

// Static (self-less) visibility evaluator — pure over state. The auto-trigger
// block owns every mode-dependent knob here, so it's the whole evaluator.
void eval_visibility(int n, const char* pb, const int* off, const int* len, const int* ops) {
  fx::AutoTrigger::evalVisibility(n, pb, off, len, ops);
}

static void on_state_ready(void* self) {
  auto* s = static_cast<State*>(self);
  if (s) fx::AutoTrigger::applyVisibility(s->auto_trig.mode, s->auto_trig.div);
}

void module_init() {
  // fx::AutoTrigger::fields() wraps the chain to splice the auto-fire block
  // into the Trigger group (it takes and returns the Schema&).
  state::init("source.light.chroma_wave", {1, 1, 0},
    fx::AutoTrigger::fields(
    state::Schema()
      // Top-level manual: high-level "what is this / how to use / what to try".
      .helpField("intro",
        "## Chroma Wave\n"
        "A charge-and-burst prismatic bloom. Hold the *Gate* to charge a soft blob — "
        "as pressure builds it flattens and curls into a downward crescent; release "
        "and it **bursts**, opening while prismatic colour bands travel down the "
        "gradient. It's **polyphonic**, so many waves can overlap.\n\n"
        "**Try:** play it rhythmically with the trigger for stacked bursts; set *Auto "
        "Mode* to **Beats** to fire it on the transport; push *Intensity* for glowing "
        "additive colour.")
      // --- Standard trigger surface ---
      .group("trigger", "Trigger")
        .groupHelp(
          "How the effect is *played*. Hold **Gate** (or push *Level* past halfway) to "
          "charge one sustained voice; fire **Trigger** for momentary one-shot bursts. "
          "*Auto Mode* is **Off** by default — set it to **Random** (Poisson, at *Auto "
          "Rate*) for hands-off ambience, or **Beats** to lock it to the host transport. "
          "*Default On* keeps a voice charging with no input.")
      .boolField ("gate",               false,                  state::PrimaryInput).label("Gate", "Gate")
      .eventField("trigger",                                    state::PrimaryInput).label("Trigger", "Trig")
      .floatField("level",              0.0f, 0.0f, 1.0f,       state::PrimaryInput).label("Level", "Lvl")
    )   // ← fx::AutoTrigger::fields: auto_mode + auto_rate + auto_beats + custom
      .boolField ("default_gate_state", false,                  state::PrimaryInput).label("Default On", "Def")

      // --- Polyphony ---
      .group("polyphony", "Polyphony")
        .groupHelp(
          "Multiple waves run at once. *Voices* caps how many overlap (older bursts get "
          "stolen past the limit). *Position* and *Hue Jitter* scatter one-shots so they "
          "read as distinct waves. **Hue Interact** decides how overlaps blend: 0 "
          "averages their colour, higher values rotate and compound the bands.")
      .intField  ("voice_limit",        8, 1, MAX_VOICES,       state::PrimaryInput).label("Voices", "Vox")
      .floatField("voice_pos_jitter",   0.5f, 0.0f, 2.0f,       state::PrimaryInput).label("Position Jitter", "PosJit")
      .floatField("voice_hue_jitter",   0.1f, 0.0f, 1.0f,       state::PrimaryInput).label("Hue Jitter", "HueJit")
      // Overlapping voices interact in the band-phase domain: 0 averages their
      // hues, 1 sums them (hue rotates further + bands compound), >1 over-rotates.
      .floatField("hue_interact",       0.8f, 0.0f, 2.0f,       state::PrimaryInput).label("Hue Interact", "HueInt")

      // --- Position ---
      .group("position", "Position")
      .floatField("position_x",         0.0f,  -2.0f, 2.0f,     state::PrimaryInput).label("Position X", "X")
      .floatField("position_y",         -0.7f, -2.0f, 2.0f,     state::PrimaryInput).label("Position Y", "Y")

      // --- Charge / pressure shape ---
      .group("charge", "Charge & Pressure")
        .groupHelp(
          "Sculpts the blob *while it charges*. **Charge Time** is how long a held gate "
          "takes to reach full pressure; **Base Radius** and **Charge Expand** set its "
          "size and growth. As pressure builds, *Plateau*, *Squish* and *Crescent* flatten "
          "the top and curl the mass into a downward hook — push them for a heavier swell.")
      .floatField("charge_s",           0.6f,  0.05f, 3.0f,     state::PrimaryInput).label("Charge Time", "Chrg")
      .floatField("base_radius",        0.12f, 0.01f, 10.0f,    state::PrimaryInput).label("Base Radius", "Rad")
      .floatField("charge_expand",      2.3f,  1.0f, 8.0f,      state::PrimaryInput).label("Charge Expand", "Expand")
      .floatField("size_smoothing",     0.06f, 0.0f, 1.0f,      state::PrimaryInput).label("Size Smoothing", "Smooth")
      .floatField("gaussian_sharpness", 4.0f,  1.0f, 20.0f,     state::PrimaryInput).label("Sharpness", "Sharp")
      .floatField("plateau_amount",     0.6f,  0.0f, 1.0f,      state::PrimaryInput).label("Plateau", "Plat")
      .floatField("squish_amount",      0.5f,  0.0f, 2.0f,      state::PrimaryInput).label("Squish", "Squish")
      .floatField("crescent_amount",    0.7f,  0.0f, 1.0f,      state::PrimaryInput).label("Crescent", "Cres")
      .floatField("crescent_offset",    0.5f,  0.1f, 1.5f,      state::PrimaryInput).label("Crescent Offset", "CresOf")

      // --- Burst ---
      .group("burst", "Burst")
      .floatField("release_s",          0.7f,  0.05f, 20.0f,    state::PrimaryInput).label("Release Time", "Rel")
      .floatField("release_expand",     3.0f,  1.0f, 20.0f,     state::PrimaryInput).label("Release Expand", "RelExp")
      // Signed power curve (style guide §8.3): +1 front-loads the burst,
      // -1 is a gradual swell, 0 linear.
      .floatField("release_curve",      0.4f,  -1.0f, 1.0f,     state::PrimaryInput).label("Release Curve", "RelCrv")
      .floatField("min_sustain_s",      0.2f,  0.0f, 1.0f,      state::PrimaryInput).label("Min Sustain", "MinSus")
      .floatField("burst_shallow",      0.7f,  0.0f, 0.95f,     state::PrimaryInput).label("Burst Shallow", "Shallow")

      // --- Colour grade ---
      .group("grade", "Colour Grade")
        .groupHelp(
          "The prismatic colour. **Base Hue** + **Hue Span** set where the spectrum sits "
          "and how wide it sweeps; the per-channel *Hue Shift* knobs twist the wheel. "
          "**Band Freq** and **Fold Rate** control the travelling stripe density during "
          "charge vs burst. Crank **Intensity** for a glowing additive over the input.")
      .floatField("base_hue",           0.55f, 0.0f, 1.0f,      state::PrimaryInput).label("Base Hue", "Hue")
      .floatField("hue_span",           0.18f, 0.0f, 1.0f,      state::PrimaryInput).label("Hue Span", "Span")
      .floatField("saturation",         0.85f, 0.0f, 1.0f,      state::PrimaryInput).label("Saturation", "Sat")
      // Twist the hue wheel: shift the hue by these amounts where it lands on
      // red / green / blue, smoothly interpolated around the wheel.
      .floatField("hue_shift_r",        0.0f,  -1.0f, 1.0f,     state::PrimaryInput).label("Hue Shift R", "ShiftR")
      .floatField("hue_shift_g",        0.0f,  -1.0f, 1.0f,     state::PrimaryInput).label("Hue Shift G", "ShiftG")
      .floatField("hue_shift_b",        0.0f,  -1.0f, 1.0f,     state::PrimaryInput).label("Hue Shift B", "ShiftB")
      .floatField("grade_freq_hold",    1.5f,  0.0f, 8.0f,      state::PrimaryInput).label("Band Freq Hold", "FrqHld")
      .floatField("grade_freq_burst",   7.0f,  0.0f, 16.0f,     state::PrimaryInput).label("Band Freq Burst", "FrqBst")
      .floatField("fold_rate",          1.5f,  0.0f, 8.0f,      state::PrimaryInput).label("Fold Rate", "Fold")
      .floatField("band_contrast",      0.6f,  0.0f, 1.0f,      state::PrimaryInput).label("Band Contrast", "BndCon")
      .floatField("band_tilt",          0.0f,  -2.0f, 2.0f,     state::PrimaryInput).label("Band Tilt", "Tilt")
      .floatField("alpha_gamma",        1.2f,  0.25f, 4.0f,     state::PrimaryInput).label("Alpha Gamma", "Gamma")
      .rgbField  ("blob_color",         1.0f,  1.0f, 1.0f,      state::PrimaryInput).label("Colour", "Col")
      .floatField("overlay_alpha_hold", 0.3f,  0.0f, 1.0f,      state::PrimaryInput).label("Alpha Hold", "AlpHld")
      .floatField("overlay_alpha_burst",0.7f,  0.0f, 1.0f,      state::PrimaryInput).label("Alpha Burst", "AlpBst")
      // Crankable master gain; soft per-channel rolloff for juicy colour.
      .floatField("intensity",          1.0f,  0.0f, 32.0f,     state::PrimaryInput).label("Intensity", "Int")

      // --- Motion --- (the knob's [0,1] maps to an internal [0,0.25] gain;
      // the raw optical-flow vectors are otherwise quite extreme.)
      .group("motion", "Motion")
        .groupHelp(
          "Emits an *optical-flow* vector field (the expanding bursts' velocity) for "
          "downstream motion-aware effects — it doesn't change this effect's own pixels. "
          "**Motion Scale** sets the strength; **Warp** collapses the spray into a "
          "coherent downward wavefront; **Edge Mask** isolates the bands' leading edges.")
      .floatField("motion_scale",       1.0f,  0.0f, 1.0f,      state::PrimaryInput).label("Motion Scale", "Motion")
      // Perceptual wavefront warp: damp the lateral spread of the motion field
      // so a squat/crescent blob reads as a coherent downward front, not rays
      // fanning from the center. 0 = analytic, 1 = fully vertical.
      .floatField("motion_warp",        0.4f,  0.0f, 1.0f,      state::PrimaryInput).label("Motion Warp", "Warp")
      // Isolate the motion to the bands' OUTWARD (leading) edges: 0 = whole
      // band, 1 = only the leading fronts (rippling-wavefront feel).
      .floatField("motion_edge_mask",   0.0f,  0.0f, 1.0f,      state::PrimaryInput).label("Motion Edge Mask", "Edge")

      // --- Debug ---
      .group("debug", "Debug")
      .boolField ("debug_field",        false,                  state::PrimaryInput).label("Debug Field", "Debug")

      .textureField("tex_in",  state::PrimaryInput)
      .textureField("tex_out", state::PrimaryOutput)
      // The wave's own additive contribution on opaque black — rendered only
      // while a wire is attached (same gating as the motion output).
      .textureField("wave_out", state::SecondaryOutput)
      .renderOutputs(state::PrimaryOutput)
      .renderOutputs(state::PrimaryInput,  "render_outputs_in")
        .capability(state::Capability::Generator)
    );

  if (gpu::Device::backend() == gpu::Backend::None) return;

  state::registerShaderSPV("chroma_wave_render", RENDER_SPV, RENDER_SPV_SIZE);
  state::registerShaderSPV("chroma_wave_motion", MOTION_SPV, MOTION_SPV_SIZE,
                           "rgba16float", "write");
  auto cs = gpu::Device::createShaderModuleByName("chroma_wave_render");
  auto cs_motion = gpu::Device::createShaderModuleByName("chroma_wave_motion");
  if (!cs || !cs_motion) return;

  s_pso = gpu::Device::createComputePSO(cs, "main", gpu::Bindings()
      .tex2d(0)
      .storageTex2d(1)
      .uniform(2)
      .storage(3)
      .storageTex2d(4));
  s_pso_motion = gpu::Device::createComputePSO(cs_motion, "main", gpu::Bindings()
      .tex2d(0)
      .storageTex2d(1, gpu::TextureFormat::RGBA16F)
      .uniform(2)
      .storage(3));

  state::setOnStateReady(&on_state_ready);
  state::log("chroma_wave: module initialized");
}

void* create() {
  auto* s = new State();
  s->uniform_buf = gpu::Device::createBuffer(sizeof(Uniforms), gpu::BufferUsage::Uniform);
  s->voice_buf   = gpu::Device::createBuffer(sizeof(VoiceGpu) * MAX_VOICES, gpu::BufferUsage::Storage);
  s->motion_uniform_buf = gpu::Device::createBuffer(sizeof(MotionUniforms), gpu::BufferUsage::Uniform);
  return s;
}

void destroy(void* self) {
  auto* s = static_cast<State*>(self);
  if (!s) return;
  s->uniform_buf.release();
  s->voice_buf.release();
  s->motion_uniform_buf.release();
  s->motion_tex.release();
  s->zero_motion_tex.release();
  s->wave_tex.release();
  s->dummy_wave_tex.release();
  delete s;
}

void init(void* self) {
  auto* s = static_cast<State*>(self);
  if (!s) return;
  for (int i = 0; i < MAX_VOICES; i++) s->voices[i] = Voice{};
  s->held_voice = -1;
  s->held_prev = false;
  s->trigger_prev = false;
  s->rng_state = 0xC0FFEE11u;
  s->jitter_rng = 0x1234567u;
  s->spawn_counter = 0;
  s->motion_w = 0;
  s->motion_h = 0;
  s->wave_w = 0;
  s->wave_h = 0;
  if (!s_pso.valid() || !s_pso_motion.valid() ||
      !s->uniform_buf.valid() || !s->voice_buf.valid() || !s->motion_uniform_buf.valid()) return;
  s->initialized = true;
  state::log("chroma_wave: initialized");
}

void tick(void* self, double dt) {
  auto* s = static_cast<State*>(self);
  if (!s || !s->initialized) return;
  float fdt = (float)dt;

  // Self-fire (Off / Random / Beats — effect_auto_trigger.h): each event is a
  // fresh one-shot voice (polyphonic). Loop the count — in Beats a long frame
  // stall can cross several divisions at once.
  for (int i = 0, nf = s->auto_trig.fires(dt); i < nf; i++) spawn_voice(s, false);

  // Held gate/level: maintain exactly one held voice across its hold.
  bool held = s->gate || s->level >= 0.5f || s->default_gate_state;
  if (held && !s->held_prev) {
    s->held_voice = spawn_voice(s, true);
  }
  if (!held && s->held_prev) {
    int hv = s->held_voice;
    if (hv >= 0 && s->voices[hv].active && s->voices[hv].held) {
      Voice& v = s->voices[hv];
      v.strain_at_release = v.charge_t;
      v.phase = PHASE_BURST;
      v.burst_t = 0.0f;
      v.held = false;
    }
    s->held_voice = -1;
  }
  s->held_prev = held;

  for (int i = 0; i < MAX_VOICES; i++)
    if (s->voices[i].active) advance_voice(s, s->voices[i], dt);
}


void on_state_patched(void* self, int n, const char* pb, const int* off,
                      const int* len, const int* ops) {
  auto* s = static_cast<State*>(self);
  if (!s) return;
  bool vis_changed = false;
  for (int i = 0; i < n; i++) {
    const char* path = pb + off[i];
    int plen = len[i];
    int op = ops[i];

    if (op == state::PatchReplace) {
      if (s->auto_trig.patch(path, plen, i, &vis_changed)) continue;
      if      (state::pathIs(path, plen, "gate"))               s->gate = state::patchFloat(i) != 0.0f;
      else if (state::pathIs(path, plen, "level"))              s->level = state::patchFloat(i);
      else if (state::pathIs(path, plen, "default_gate_state")) s->default_gate_state = state::patchFloat(i) != 0.0f;
      else if (state::pathIs(path, plen, "voice_limit"))        s->voice_limit = (int)state::patchFloat(i);
      else if (state::pathIs(path, plen, "voice_pos_jitter"))   s->voice_pos_jitter = state::patchFloat(i);
      else if (state::pathIs(path, plen, "voice_hue_jitter"))   s->voice_hue_jitter = state::patchFloat(i);
      else if (state::pathIs(path, plen, "hue_interact"))       s->hue_interact = state::patchFloat(i);
      else if (state::pathIs(path, plen, "position_x"))         s->position_x = state::patchFloat(i);
      else if (state::pathIs(path, plen, "position_y"))         s->position_y = state::patchFloat(i);
      else if (state::pathIs(path, plen, "charge_s"))           s->charge_s = state::patchFloat(i);
      else if (state::pathIs(path, plen, "base_radius"))        s->base_radius = state::patchFloat(i);
      else if (state::pathIs(path, plen, "charge_expand"))      s->charge_expand = state::patchFloat(i);
      else if (state::pathIs(path, plen, "size_smoothing"))     s->size_smoothing = state::patchFloat(i);
      else if (state::pathIs(path, plen, "gaussian_sharpness")) s->gaussian_sharp = state::patchFloat(i);
      else if (state::pathIs(path, plen, "plateau_amount"))     s->plateau_amount = state::patchFloat(i);
      else if (state::pathIs(path, plen, "squish_amount"))      s->squish_amount = state::patchFloat(i);
      else if (state::pathIs(path, plen, "crescent_amount"))    s->crescent_amount = state::patchFloat(i);
      else if (state::pathIs(path, plen, "crescent_offset"))    s->crescent_offset = state::patchFloat(i);
      else if (state::pathIs(path, plen, "release_s"))          s->release_s = state::patchFloat(i);
      else if (state::pathIs(path, plen, "release_expand"))     s->release_expand = state::patchFloat(i);
      else if (state::pathIs(path, plen, "release_curve"))      s->release_curve = state::patchFloat(i);
      else if (state::pathIs(path, plen, "min_sustain_s"))      s->min_sustain_s = state::patchFloat(i);
      else if (state::pathIs(path, plen, "burst_shallow"))      s->burst_shallow = state::patchFloat(i);
      else if (state::pathIs(path, plen, "base_hue"))           s->base_hue = state::patchFloat(i);
      else if (state::pathIs(path, plen, "hue_span"))           s->hue_span = state::patchFloat(i);
      else if (state::pathIs(path, plen, "saturation"))         s->saturation = state::patchFloat(i);
      else if (state::pathIs(path, plen, "hue_shift_r"))        s->hue_shift_r = state::patchFloat(i);
      else if (state::pathIs(path, plen, "hue_shift_g"))        s->hue_shift_g = state::patchFloat(i);
      else if (state::pathIs(path, plen, "hue_shift_b"))        s->hue_shift_b = state::patchFloat(i);
      else if (state::pathIs(path, plen, "grade_freq_hold"))    s->grade_freq_hold = state::patchFloat(i);
      else if (state::pathIs(path, plen, "grade_freq_burst"))   s->grade_freq_burst = state::patchFloat(i);
      else if (state::pathIs(path, plen, "fold_rate"))          s->fold_rate = state::patchFloat(i);
      else if (state::pathIs(path, plen, "band_contrast"))      s->band_contrast = state::patchFloat(i);
      else if (state::pathIs(path, plen, "band_tilt"))          s->band_tilt = state::patchFloat(i);
      else if (state::pathIs(path, plen, "alpha_gamma"))        s->alpha_gamma = state::patchFloat(i);
      else if (state::pathIs(path, plen, "overlay_alpha_hold")) s->overlay_alpha_hold = state::patchFloat(i);
      else if (state::pathIs(path, plen, "overlay_alpha_burst"))s->overlay_alpha_burst = state::patchFloat(i);
      else if (state::pathIs(path, plen, "intensity"))          s->intensity = state::patchFloat(i);
      else if (state::pathIs(path, plen, "motion_scale"))       s->motion_scale = state::patchFloat(i);
      else if (state::pathIs(path, plen, "motion_warp"))        s->motion_warp = state::patchFloat(i);
      else if (state::pathIs(path, plen, "motion_edge_mask"))   s->motion_edge_mask = state::patchFloat(i);
      else if (state::pathIs(path, plen, "debug_field"))        s->debug_field = state::patchFloat(i) != 0.0f;
      else if (state::pathIs(path, plen, "blob_color")) {
        auto v = state::patchVec3(i);
        s->color_r = v.x; s->color_g = v.y; s->color_b = v.z;
      }
    }

    // Event field — momentary (on/off like gate). Fire on the rising edge of
    // the VALUE; the executor replays the stored value every frame, so a
    // value-less guard would re-arm forever (style guide §8.2). Each press
    // spawns a fresh one-shot voice (polyphonic).
    if (state::pathIs(path, plen, "trigger")) {
      bool tval = state::patchFloat(i) != 0.0f;
      if (tval && !s->trigger_prev) spawn_voice(s, false);
      s->trigger_prev = tval;
    }
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

  // Pack the active voices contiguously; the rest stay zeroed (unread).
  VoiceGpu vg[MAX_VOICES];
  std::memset(vg, 0, sizeof(vg));
  int count = 0;
  for (int i = 0; i < MAX_VOICES; i++) {
    if (!s->voices[i].active) continue;
    compute_voice_gpu(s, s->voices[i], vg[count]);
    count++;
  }
  s->voice_buf.writeBytes(vg, (int)sizeof(vg));

  // Isolated wave layer (wave_out) — the additive contribution on opaque
  // black, written by the same render dispatch. Only produced when a wire is
  // attached; otherwise a 1x1 dummy fills the binding and the shader skips
  // the write.
  bool want_wave = state::isOutputConnected("wave_out");
  if (want_wave &&
      (!s->wave_tex.valid() || s->wave_w != vp_w || s->wave_h != vp_h)) {
    s->wave_tex = gpu::Device::createTexture(vp_w, vp_h);
    s->wave_w = vp_w;
    s->wave_h = vp_h;
    if (s->wave_tex.valid()) state::setGpuTexture("wave_out", s->wave_tex.id);
  }
  bool write_wave = want_wave && s->wave_tex.valid();
  if (!write_wave && !s->dummy_wave_tex.valid())
    s->dummy_wave_tex = gpu::Device::createTexture(1, 1);

  Uniforms u = {};
  u.write_wave = write_wave ? 1.0f : 0.0f;
  u.cres_off = s->crescent_offset;
  u.band_tilt = s->band_tilt;
  u.hue_span = s->hue_span;
  u.saturation = s->saturation;
  u.alpha_gamma = s->alpha_gamma;
  u.intensity = s->intensity;
  u.base_hue = s->base_hue;
  u.debug_field = s->debug_field ? 1.0f : 0.0f;
  u.color_r = s->color_r;
  u.color_g = s->color_g;
  u.color_b = s->color_b;
  u.voice_count = (uint32_t)count;
  u.hue_interact = s->hue_interact;
  // Hue-wheel twist: the first Fourier harmonic that interpolates the R/G/B
  // shifts exactly at hue 0, 1/3, 2/3 — smooth and periodic.
  float R = s->hue_shift_r, G = s->hue_shift_g, B = s->hue_shift_b;
  u.hue_warp_a = (R + G + B) / 3.0f;
  u.hue_warp_b = (2.0f * R - G - B) / 3.0f;
  u.hue_warp_c = (G - B) * 0.57735026918962576f;   // 1/sqrt(3)
  s->uniform_buf.writeOne(u);

  {
    auto cp = gpu::ComputePass::begin();
    cp.setPSO(s_pso);
    cp.setTexture(in,  0, 0);
    cp.setTexture(out, 1, 1);
    cp.setBuffer(s->uniform_buf, 2);
    cp.setBuffer(s->voice_buf,   3);
    cp.setTexture(write_wave ? s->wave_tex : s->dummy_wave_tex, 4, 1);
    cp.dispatch((vp_w + 7) / 8, (vp_h + 7) / 8);
    cp.end();
  }

  // Motion pass — radial expansion velocity (uv/sec) of the bursting blobs,
  // blended over upstream motion. Skip entirely when nothing downstream cares.
  if (state::isOutputConnected("render_outputs")) {
    if (!s->motion_tex.valid() || s->motion_w != vp_w || s->motion_h != vp_h) {
      s->motion_tex = gpu::Device::createTexture(vp_w, vp_h, gpu::TextureFormat::RGBA16F);
      s->motion_w = vp_w;
      s->motion_h = vp_h;
      if (s->motion_tex.valid()) state::setGpuTexture("render_outputs/motion", s->motion_tex.id);
    }
    if (s->motion_tex.valid()) {
      auto upstream = gpu::Device::textureForField("render_outputs_in/motion");
      if (!upstream.valid()) {
        if (!s->zero_motion_tex.valid())
          s->zero_motion_tex = gpu::Device::createTexture(1, 1, gpu::TextureFormat::RGBA16F);
        upstream = s->zero_motion_tex;
      }
      MotionUniforms mu = {};
      mu.cres_off = s->crescent_offset;
      mu.motion_scale = s->motion_scale * 0.25f;   // [0,1] knob → [0,0.25] gain
      mu.alpha_gamma = s->alpha_gamma;
      mu.band_tilt = s->band_tilt;
      mu.voice_count = (uint32_t)count;
      mu.motion_warp = s->motion_warp;
      mu.motion_edge_mask = s->motion_edge_mask;
      s->motion_uniform_buf.writeOne(mu);

      auto cp = gpu::ComputePass::begin();
      cp.setPSO(s_pso_motion);
      cp.setTexture(upstream, 0, 0);
      cp.setTexture(s->motion_tex, 1, 1);
      cp.setBuffer(s->motion_uniform_buf, 2);
      cp.setBuffer(s->voice_buf, 3);
      cp.dispatch((vp_w + 7) / 8, (vp_h + 7) / 8);
      cp.end();
    }
  }

  gpu::Device::submit();
}

} // namespace chroma_wave
