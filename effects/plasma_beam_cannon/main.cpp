/*
 * source.light.plasma_beam_cannon — 90s-anime power-up beam.
 *
 * Per-bar vertical beam driven by a linked-across-bars ADSR clock:
 *   - Attack: small seed at `seed_y`, height = `seed_height`.
 *   - Decay:  beam grows from `seed_height` → 1.0.
 *   - Sustain: full bar lit.
 *   - Release: beam stays at full height; per-bar break particles
 *              "eat" the beam, growing/shrinking under a length-target
 *              controller. Near the end, a flicker tail kicks in
 *              (duty cycle decays from `flicker_duty_start` →
 *              `flicker_duty_end`).
 *
 * Triggered by `gate` (rising-edge synthesizes a one-shot pulse) or
 * `trigger` event (momentary, on/off like gate; fires on the rising edge of
 * its value, which is replay-safe — see §8.2).
 *
 * Hard-edge rendering throughout — no alpha, no fades. Break cells
 * either fully eat the beam (revert to input passthrough at that
 * pixel) or don't.
 *
 * Class-like instance model: module_init() compiles the shared color
 * compute PSO + publishes the schema once per type; each chain entry
 * gets its own State (params, ADSR state machine, break-particle pool,
 * RNG streams, per-instance buffers) via create(). All instance
 * callbacks take `self`.
 */

#include <gpu.h>
#include <host.h>
#include <effect_utils.h>
#include <effect_auto_trigger.h>  // fx::AutoTrigger — the shared Off/Random/Beats self-fire
#include "plasma_beam_cannon_shaders.h"

#include <cmath>
#include <cstdint>
#include <cstring>

namespace plasma_beam_cannon {

enum Phase : int {
  PHASE_IDLE    = 0,
  PHASE_ATTACK  = 1,
  PHASE_DECAY   = 2,
  PHASE_SUSTAIN = 3,
  PHASE_RELEASE = 4,
};

enum BreakType : int {
  BREAK_SOLID_ATTRACTOR = 0,
  BREAK_SOLID_REPELLOR  = 1,
  BREAK_SPACER          = 2,  // invisible, only contributes repulsion
};

static constexpr int BARS               = 4;
static constexpr int MAX_BREAKS_PER_BAR = 32;
static constexpr int MAX_BREAKS_TOTAL   = BARS * MAX_BREAKS_PER_BAR;

struct Uniforms {
  // row 0
  float    beam_y_min;
  float    beam_y_max;
  float    intensity;
  uint32_t active;

  // row 1
  float    color_r;
  float    color_g;
  float    color_b;
  uint32_t bar_target_all;

  // row 2
  uint32_t bar_target;
  uint32_t particles_per_bar;
  uint32_t breaks_active;     // 1 during RELEASE; 0 otherwise
  uint32_t flicker_active;    // 1 once release_t ≥ flicker_start_t

  // row 3
  uint32_t flicker_on;        // current on/off state when flicker_active
  uint32_t _pad_0;
  uint32_t _pad_1;
  uint32_t _pad_2;
};
static_assert(sizeof(Uniforms) == 64, "Uniforms layout mismatch");

// GPU layout per break particle: 1 vec4 = 16 bytes.
//   .x = y (uv-space, position along bar height)
//   .y = size (uv-space, full height of the break)
//   .z = type (as float — 0 attractor, 1 repellor, 2 spacer)
//   .w = padding / reserved
struct GpuBreak {
  float y;
  float size;
  float type_f;
  float _pad;
};
static_assert(sizeof(GpuBreak) == 16, "GpuBreak layout mismatch");

struct CpuBreak {
  float y;
  float vy;
  float size;
  int   type;
  // Stagger threshold in release_t ∈ [0, 1]. The break is INACTIVE
  // (invisible, no physics, doesn't contribute forces) until release
  // progress passes this value. Set per-break at reset time from a
  // uniform random draw shaped by `activation_curve` and
  // `activation_min` so breaks don't all pop in simultaneously.
  float threshold;
  // Per-break upper bound on `size`. Drawn at reset from a strongly
  // bimodal distribution — most breaks get either `min_break_size`
  // (stay small forever) or `max_break_size` (free to grow); the
  // population split is controlled by `growth_fraction`. This is what
  // gives release-phase breaks two distinct visual classes instead of
  // every break growing at the same rate under the length controller.
  float personal_max_size;
};

// Per-instance state. One per chain entry. Holds ALL mutable runtime
// state: schema-mirrored params, the ADSR state machine, edge-state
// for the trigger surface, the break-particle pool, RNG streams, the
// flicker accumulator, and the per-instance GPU buffers.
struct State {
  gpu::Buffer uniform_buf;
  gpu::Buffer break_buf;
  bool        initialized = false;

  // --- Schema-mirrored params (standard) ---
  bool  gate            = false;
  float attack_s        = 0.15f;
  float decay_s         = 0.10f;
  float sustain_s       = 0.40f;
  float release_s       = 1.50f;
  float attack_curve    = 0.0f;
  float decay_curve     = 0.0f;
  float release_curve   = 0.0f;
  float seed_y          = 0.50f;
  float seed_height     = 0.06f;
  float color_r         = 1.00f;
  float color_g         = 0.95f;
  float color_b         = 0.80f;
  float intensity       = 1.0f;
  int   bar_target      = 0;
  bool  bar_target_all  = true;
  fx::AutoTrigger auto_trig;   // Off / Random (Poisson) / Beats — see effect_auto_trigger.h

  // --- Break-particle tuning params ---
  int   break_count           = 12;
  float attractor_fraction    = 0.25f;
  float spacer_fraction       = 0.25f;
  float min_break_size        = 0.015f;
  float max_break_size        = 0.12f;
  float force_strength        = 0.4f;
  float spacer_strength       = 0.3f;
  float force_softening       = 0.05f;
  float damping_per_s         = 4.0f;
  float interaction_radius    = 0.3f;
  float teleport_rate         = 0.2f;
  float length_target_start   = 0.1f;
  float length_target_end     = 0.7f;
  float length_target_curve   = 1.0f;
  float grow_response         = 1.0f;
  float activation_curve      = 0.0f;
  float activation_min        = 0.0f;
  float growth_fraction       = 0.4f;
  float flicker_start_t       = 0.7f;
  float flicker_duty_start    = 0.8f;
  float flicker_duty_end      = 0.05f;
  float flicker_freq_hz       = 24.0f;
  int   break_seed            = 0x12345;
  bool  cycle_seed            = true;
  int   seed_cycle_count      = 0;

  // --- State machine runtime state ---
  Phase    phase          = PHASE_IDLE;
  double   time_in_phase  = 0.0;
  bool     gate_prev      = false;
  bool     trigger_prev   = false;   // rising-edge detect for the event field
  bool     trigger_pulse  = false;
  double   trigger_hold_remaining = 0.0;
  uint32_t rng_state      = 0xCAFEBABEu;
  // Separate RNG stream for per-frame break operations (teleport draws,
  // etc). Kept independent of `rng_state` so toggling auto_rate or
  // the trigger-pulse RNG doesn't shift teleport timing in subtle ways.
  uint32_t break_op_rng   = 0xBADDCAFEu;

  // --- Break-particle runtime state ---
  CpuBreak breaks[BARS][MAX_BREAKS_PER_BAR];
  double   flicker_phase  = 0.0;
  bool     flicker_on     = true;
  bool     breaks_seeded  = false;
};

// Type-shared: compiled once in module_init().
static gpu::ComputePSO s_pso;

static inline float lerpf(float a, float b, float t) {
  return a + (b - a) * t;
}
static inline float clampf(float v, float lo, float hi) {
  return v < lo ? lo : (v > hi ? hi : v);
}
static inline uint32_t lcg_next(uint32_t& s) {
  s = s * 1664525u + 1013904223u;
  return s;
}
static inline float lcg_unit(uint32_t& s) {
  return (lcg_next(s) >> 8) * (1.0f / float(1u << 24));
}
static inline float lcg_signed(uint32_t& s) { return lcg_unit(s) * 2.0f - 1.0f; }

static void enter_phase(State& s, Phase p);

// Re-init every break particle for a fresh release cycle.
static void reset_breaks(State& s) {
  // Effective seed combines the user-set seed with the per-trigger
  // cycle counter (when `cycle_seed` is on). Pure deterministic
  // mapping: (break_seed, cycle_count) → break pattern. No drift
  // from elapsed time / auto-trigger Poisson sampling.
  uint32_t effective_seed = (uint32_t)s.break_seed
    + (s.cycle_seed ? (uint32_t)s.seed_cycle_count : 0u);
  uint32_t rng = effective_seed ^ 0xDEAFBEEFu;
  // Mix once so the first lcg_unit() doesn't hand out a near-zero
  // pattern for small seeds.
  lcg_next(rng);

  int per_bar = s.break_count;
  if (per_bar < 1) per_bar = 1;
  if (per_bar > MAX_BREAKS_PER_BAR) per_bar = MAX_BREAKS_PER_BAR;

  // Activation curve via the style-guide power-curve helper. Slider
  // [-1, +1] → exponent [8, 1, 1/8]; we then raise the uniform draw to
  // that exponent to skew thresholds toward 0 (early) or 1 (late).
  float curve_exp = fx::signedSliderToExp(clampf(s.activation_curve, -1.0f, 1.0f));
  float min_t = clampf(s.activation_min, 0.0f, 1.0f);
  float min_span = 1.0f - min_t;

  float growth_fraction = clampf(s.growth_fraction, 0.0f, 1.0f);

  for (int bar = 0; bar < BARS; bar++) {
    for (int i = 0; i < MAX_BREAKS_PER_BAR; i++) {
      CpuBreak& b = s.breaks[bar][i];
      if (i >= per_bar) {
        // Inactive slot — make sure it doesn't get rendered.
        b.y = 0.5f;
        b.vy = 0.0f;
        b.size = 0.0f;
        b.type = BREAK_SPACER;
        b.threshold = 2.0f;          // > 1.0 so it never activates
        b.personal_max_size = s.min_break_size;
        continue;
      }
      b.y = lcg_unit(rng);            // uniformly across [0, 1]
      b.vy = 0.0f;
      // Size starts at 0 — break is "inactive" until release passes
      // its threshold. On first activation we pop it to min size.
      b.size = 0.0f;
      // Type by stochastic draw. Order doesn't matter (no overlap rules).
      float u = lcg_unit(rng);
      if (u < s.attractor_fraction) {
        b.type = BREAK_SOLID_ATTRACTOR;
      } else if (u < s.attractor_fraction + s.spacer_fraction) {
        b.type = BREAK_SPACER;
      } else {
        b.type = BREAK_SOLID_REPELLOR;
      }
      // Activation threshold ∈ [activation_min, 1.0], shaped.
      float thresh_u = lcg_unit(rng);
      float curved = std::pow(thresh_u, curve_exp);
      b.threshold = min_t + min_span * curved;
      // Bimodal personal max: binary growth class. No continuous
      // distribution — either capped at min (stays small) or at max
      // (free to grow under length-controller drive).
      bool wants_to_grow = lcg_unit(rng) < growth_fraction;
      b.personal_max_size = wants_to_grow ? s.max_break_size : s.min_break_size;
    }
  }
  s.breaks_seeded = true;
}

static void enter_phase(State& s, Phase p) {
  s.phase = p;
  s.time_in_phase = 0.0;
  if (p == PHASE_RELEASE) {
    // Bump the cycle counter BEFORE reset_breaks reads it, so the
    // first release after init uses cycle=1 (a non-default seed
    // offset) and each subsequent release advances by one.
    if (s.cycle_seed) s.seed_cycle_count++;
    reset_breaks(s);
    s.flicker_phase = 0.0;
    s.flicker_on = true;
  }
}

// Pairwise 1D N-body step. Solid (attractor) particles pull others
// toward them; solid repellors and spacers push others away. Spacers
// don't paint but their force is felt — they keep solid breaks from
// piling into one big void.
static void update_breaks(State& s, double dt) {
  if (!s.breaks_seeded || s.phase != PHASE_RELEASE) return;
  float fdt = (float)dt;

  int per_bar = s.break_count;
  if (per_bar < 1) per_bar = 1;
  if (per_bar > MAX_BREAKS_PER_BAR) per_bar = MAX_BREAKS_PER_BAR;

  // release_curve warps release_t globally — length target, break
  // activation, flicker onset all see the warped time. Per-system
  // curves (length_target_curve etc.) compose on top.
  float release_t_raw = (s.release_s > 0.0f)
    ? clampf((float)(s.time_in_phase / (double)s.release_s), 0.0f, 1.0f)
    : 1.0f;
  float release_t = std::pow(release_t_raw, fx::signedSliderToExp(s.release_curve));
  float exponent = s.length_target_curve > 0.01f ? s.length_target_curve : 0.01f;
  float target_curve_t = std::pow(release_t, exponent);
  float target_length = lerpf(s.length_target_start, s.length_target_end, target_curve_t);

  for (int bar = 0; bar < BARS; bar++) {
    CpuBreak* bp = s.breaks[bar];

    // Resolve active flags from per-break activation thresholds.
    // Inactive breaks are entirely absent from the simulation this
    // tick — they don't paint, don't exert force, and aren't moved.
    bool is_active[MAX_BREAKS_PER_BAR];
    for (int i = 0; i < per_bar; i++) {
      is_active[i] = (release_t >= bp[i].threshold);
    }

    // Length controller — sum only ACTIVE solid sizes.
    float current_length = 0.0f;
    for (int i = 0; i < per_bar; i++) {
      if (!is_active[i]) continue;
      if (bp[i].type != BREAK_SPACER) current_length += bp[i].size;
    }
    float error = target_length - current_length;

    // Per-pair forces — both endpoints must be active.
    float force[MAX_BREAKS_PER_BAR];
    for (int i = 0; i < per_bar; i++) force[i] = 0.0f;
    float soft_sq = s.force_softening * s.force_softening;
    if (soft_sq < 1e-6f) soft_sq = 1e-6f;     // guard against zero
    for (int i = 0; i < per_bar; i++) {
      if (!is_active[i]) continue;
      for (int j = 0; j < per_bar; j++) {
        if (i == j) continue;
        if (!is_active[j]) continue;
        float dy = bp[j].y - bp[i].y;
        float abs_dy = dy < 0.0f ? -dy : dy;
        if (abs_dy > s.interaction_radius) continue;
        // Plummer-style softened distance: never produces unbounded
        // forces no matter how close two particles seed.
        float d_sq = abs_dy * abs_dy + soft_sq;
        // Attractor: force points TOWARD j → +sign(dy).
        // Repellor / spacer: force points AWAY from j → -sign(dy).
        float dir = (dy > 0.0f) ? 1.0f : -1.0f;
        bool j_is_attractor = (bp[j].type == BREAK_SOLID_ATTRACTOR);
        bool j_is_spacer    = (bp[j].type == BREAK_SPACER);
        float sign_term = j_is_attractor ? dir : -dir;
        // Spacers get their own strength multiplier so they can be
        // dialed down without weakening solid attraction/repulsion.
        float mag = s.force_strength * (j_is_spacer ? s.spacer_strength : 1.0f);
        force[i] += sign_term * mag / d_sq;
      }
    }

    // Integrate (active only).
    float damp = std::exp(-s.damping_per_s * fdt);
    for (int i = 0; i < per_bar; i++) {
      if (!is_active[i]) continue;
      bp[i].vy = bp[i].vy * damp + force[i] * fdt;
      bp[i].y += bp[i].vy * fdt;
      if (bp[i].y < 0.0f) { bp[i].y = 0.0f; bp[i].vy = 0.0f; }
      if (bp[i].y > 1.0f) { bp[i].y = 1.0f; bp[i].vy = 0.0f; }

      // First-frame-of-activation: pop size to min so the break is
      // immediately visible instead of being invisible until the
      // length controller has had a few ticks to grow it from 0.
      if (bp[i].size == 0.0f && bp[i].type != BREAK_SPACER) {
        bp[i].size = s.min_break_size;
      }

      // Length controller — adjust size toward target. Clamp to the
      // per-break personal max so the bimodal split holds: "stays
      // small" breaks never grow beyond min_break_size regardless of
      // how positive the error gets.
      if (bp[i].type != BREAK_SPACER) {
        bp[i].size += error * s.grow_response * fdt;
        bp[i].size = clampf(bp[i].size, s.min_break_size, bp[i].personal_max_size);
      }

      // Per-frame teleport draw. Smaller breaks have higher chance.
      //   size_norm  ∈ [0, 1] from current size in [min, max].
      //   size_factor = 1 - size_norm  → small=1, large=0.
      //   actual_rate = teleport_rate_hz * size_factor.
      // Then standard Poisson per-frame fire: u < 1 - exp(-rate * dt).
      // On fire, pick a fresh random y and reset velocity.
      if (s.teleport_rate > 0.0f) {
        float teleport_rate_hz = std::pow(60.0f, s.teleport_rate) - 1.0f;
        if (teleport_rate_hz > 0.0f) {
          float size_range = s.max_break_size - s.min_break_size;
          float size_norm = (size_range > 1e-4f)
            ? clampf((bp[i].size - s.min_break_size) / size_range, 0.0f, 1.0f)
            : 0.0f;
          float size_factor = 1.0f - size_norm;
          float lambda = teleport_rate_hz * size_factor * fdt;
          float u_fire = lcg_unit(s.break_op_rng);
          if (u_fire < 1.0f - std::exp(-lambda)) {
            bp[i].y  = lcg_unit(s.break_op_rng);
            bp[i].vy = 0.0f;
          }
        }
      }
    }
  }

  // Flicker tail. flicker_t maps 0..1 across the post-flicker_start_t
  // portion of release. Hard-step duty cycle — overrides everything
  // to black during off periods.
  if (release_t >= s.flicker_start_t && s.flicker_start_t < 1.0f) {
    float flicker_t = (release_t - s.flicker_start_t) / (1.0f - s.flicker_start_t);
    flicker_t = clampf(flicker_t, 0.0f, 1.0f);
    float duty = lerpf(s.flicker_duty_start, s.flicker_duty_end, flicker_t);
    s.flicker_phase += dt * (double)s.flicker_freq_hz;
    if (s.flicker_phase >= 1.0) s.flicker_phase -= std::floor(s.flicker_phase);
    s.flicker_on = (float)s.flicker_phase < duty;
  } else {
    s.flicker_on = true;
  }
}

// Static (self-less) visibility evaluator — pure over state. The auto-trigger
// block owns every mode-dependent knob here, so it's the whole evaluator.
void eval_visibility(int n, const char* pb, const int* off, const int* len, const int* ops) {
  fx::AutoTrigger::evalVisibility(n, pb, off, len, ops);
}

static void on_state_ready(void* self) {
  auto* s = static_cast<State*>(self);
  if (s) fx::AutoTrigger::applyVisibility(s->auto_trig.mode, s->auto_trig.div);
}

// Type-level setup: schema + the shared color compute PSO. Runs once
// per type.
void module_init() {
  // fx::AutoTrigger::fields() wraps the chain to splice the auto-fire block
  // into the Trigger group (it takes and returns the Schema&).
  state::init("source.light.plasma_beam_cannon", {1, 0, 1},
    fx::AutoTrigger::fields(
    state::Schema()
      .helpField("intro",
        "## Plasma Beam Cannon\n"
        "A triggered, 90s-anime **power-up beam**. Each shot runs one linked ADSR: a "
        "small seed at *Seed Y* charges up (**Attack**), grows to a full vertical beam "
        "(**Decay**), holds lit (**Sustain**), then in **Release** a swarm of break "
        "particles eats the beam away and a flicker tail sputters it out.\n\n"
        "**Fire it** from the *Gate*/*Trigger* inputs (wire a clock or hit it live), or "
        "raise *Auto Rate* above 0 to let it self-fire on a bar clock. **Try:** shorten "
        "*Release* and raise *Break Count* for a violent dissolve; sculpt how the beam is "
        "eaten with *Length Start/End* and *Grow Response*; add a *Flicker Tail* for the "
        "classic sputter-out.")
      // --- Trigger ---
      .group("trigger", "Trigger")
        .groupHelp(
          "Fire the cannon from *Gate* (a level input — a rising edge starts a shot) or "
          "*Trigger* (a momentary event, replay-safe). Each shot runs the full ADSR once. "
          "*Auto Mode* is **Off** by default — set it to **Random** to self-fire on its "
          "own clock (Poisson, at *Auto Rate*), or **Beats** to lock the shots to the "
          "host transport.")
      .boolField ("gate",            false,                        state::PrimaryInput).label("Gate", "Gate")
      .eventField("trigger",                                       state::PrimaryInput).label("Trigger", "Trig")
    )   // ← fx::AutoTrigger::fields: auto_mode + auto_rate + auto_beats + custom
      // --- Envelope (ADSR) ---
      .group("envelope", "Envelope")
        .groupHelp(
          "The beam's whole life is one ADSR. *Attack* charges the seed in, *Decay* grows "
          "it to the full bar, *Sustain* holds it lit, and *Release* runs the dissolve. "
          "The signed *Curve* sliders bend each phase's shape: -1 = slow-start/fast-finish, "
          "0 = linear, +1 = fast-start/slow-finish.")
      .floatField("seed_y",          0.5f, 0.0f, 1.0f,             state::PrimaryInput).label("Seed Y", "SeedY")
      .floatField("seed_height",     0.06f, 0.0f, 0.5f,            state::PrimaryInput).label("Seed Height", "SeedH")
      .floatField("attack_s",        0.15f, 0.0f, 1.0f,            state::PrimaryInput).label("Attack", "Atk")
      .floatField("decay_s",         0.10f, 0.0f, 0.5f,            state::PrimaryInput).label("Decay", "Dec")
      .floatField("sustain_s",       0.40f, 0.0f, 4.0f,            state::PrimaryInput).label("Sustain", "Sus")
      .floatField("release_s",       1.50f, 0.1f, 5.0f,            state::PrimaryInput).label("Release", "Rel")
      // Per-phase shape curves (signed sliders; style guide §1.3).
      //   -1 → exp 8   → slow start, fast finish
      //    0 → linear
      //   +1 → exp 1/8 → fast start, slow finish
      // Attack curves the seed grow-in. Decay curves the seed → full
      // bar lerp. Release warps release_t globally — all release-
      // phase time-dependent machinery (length target, activation
      // thresholds, flicker onset) sees the warped time.
      .floatField("attack_curve",    0.0f, -1.0f, 1.0f,            state::PrimaryInput).label("Attack Curve", "AtkCrv")
      .floatField("decay_curve",     0.0f, -1.0f, 1.0f,            state::PrimaryInput).label("Decay Curve", "DecCrv")
      .floatField("release_curve",   0.0f, -1.0f, 1.0f,            state::PrimaryInput).label("Release Curve", "RelCrv")
      // --- Beam ---
      .group("beam", "Beam")
      .rgbField  ("beam_color",      1.0f, 0.95f, 0.8f,            state::PrimaryInput).label("Beam Color", "Color")
      .floatField("intensity",       1.0f, 0.0f, 2.0f,             state::PrimaryInput).label("Intensity", "Int")
      .intField  ("bar_target",      0, 0, 3,                      state::PrimaryInput).label("Bar Target", "Bar")
      .boolField ("bar_target_all",  true,                         state::PrimaryInput).label("All Bars", "All")

      // --- Break particles (tuning) ---
      .group("breaks", "Break Particles")
        .groupHelp(
          "During Release, a pool of break particles chews the beam apart. *Break Count* "
          "sets how many; *Attractor*/*Spacer Fraction* set the mix of pulling, pushing "
          "and spacing agents; the force, softening and damping knobs are their physics. "
          "Start with *Length Start/End* and *Grow Response* — they drive how much beam "
          "gets eaten over the release.")
      .intField  ("break_count_per_bar",   12,    1, MAX_BREAKS_PER_BAR, state::PrimaryInput).label("Break Count", "Count")
      .floatField("attractor_fraction",    0.25f, 0.0f, 1.0f,             state::PrimaryInput).label("Attractor Fraction", "Attr")
      .floatField("spacer_fraction",       0.25f, 0.0f, 1.0f,             state::PrimaryInput).label("Spacer Fraction", "Spacer")
      .floatField("min_break_size",        0.015f, 0.001f, 0.2f,          state::PrimaryInput).label("Min Size", "MinSz")
      .floatField("max_break_size",        0.12f,  0.01f, 0.5f,           state::PrimaryInput).label("Max Size", "MaxSz")
      .floatField("force_strength",        0.4f,  0.0f, 2.0f,             state::PrimaryInput).label("Force Strength", "Force")
      // Multiplier on spacer-only force magnitude (relative to
      // `force_strength`). Spacers are repulsion-only "stay apart"
      // markers; at 1.0 they easily dominate and fling solids to the
      // poles. Default 0.3 keeps them a gentle nudge.
      .floatField("spacer_strength",       0.3f,  0.0f, 1.0f,             state::PrimaryInput).label("Spacer Force", "SpcFrc")
      // Plummer softening for the 1/r² pair force — bounds the max
      // close-range force. Without this (i.e. tiny softening) any
      // two particles that seed near each other slingshot themselves
      // and everything around them to the bar edges, even with
      // force_strength near zero. Keep at the 0.05 default unless you
      // explicitly want sharper short-range physics.
      .floatField("force_softening",       0.05f, 0.005f, 0.5f,           state::PrimaryInput).label("Force Softening", "Soften")
      .floatField("damping_per_s",         4.0f,  0.1f, 10.0f,            state::PrimaryInput).label("Damping", "Damp")
      .floatField("interaction_radius",    0.3f,  0.05f, 1.0f,            state::PrimaryInput).label("Interaction Radius", "Radius")
      // Per-break teleport: each active break has a per-frame Poisson
      // chance to jump to a new random y. Rate is biased by current
      // size — smallest breaks teleport at full rate, largest never.
      // Slider [0, 1] maps to rate via §4.1: pow(60, slider) - 1 Hz.
      .floatField("teleport_rate",         0.2f,  0.0f, 1.0f,             state::PrimaryInput).label("Teleport Rate", "Telep")
      .floatField("length_target_start",   0.1f,  0.0f, 1.0f,             state::PrimaryInput).label("Length Start", "LenSt")
      .floatField("length_target_end",     0.7f,  0.0f, 1.0f,             state::PrimaryInput).label("Length End", "LenEnd")
      .floatField("length_target_curve",   1.0f,  0.25f, 4.0f,            state::PrimaryInput).label("Length Curve", "LenCrv")
      .floatField("grow_response",         1.0f,  0.0f, 4.0f,             state::PrimaryInput).label("Grow Response", "Grow")
      // Activation stagger — each break gets a random threshold in
      // release_t ∈ [activation_min, 1.0]. It's invisible (and doesn't
      // exert force) until release passes that threshold. Slider [-1,+1]
      // maps via the style-guide power-curve helper:
      //   -1 → exponent 8   → thresholds cluster near 0 → EARLY
      //    0 → exponent 1   → uniform across [min, 1]
      //   +1 → exponent 1/8 → thresholds cluster near 1 → LATE
      .floatField("activation_curve",      0.0f, -1.0f, 1.0f,             state::PrimaryInput).label("Activation Curve", "ActCrv")
      .floatField("activation_min",        0.0f,  0.0f, 1.0f,             state::PrimaryInput).label("Activation Min", "ActMin")
      // Bimodal break-size population. Each break is binary-classed
      // into "stays at min_break_size" or "free to grow to
      // max_break_size" with probability = growth_fraction. The
      // length-target controller then pushes each break toward its
      // personal cap — small breaks hit theirs immediately and stop,
      // large breaks keep growing.
      .floatField("growth_fraction",       0.4f,  0.0f, 1.0f,             state::PrimaryInput).label("Growth Fraction", "Growth")
      .intField  ("break_seed",            0x12345, 0, 0x7FFFFFFF,        state::PrimaryInput).label("Break Seed", "Seed")
      // When on (default), an internal counter increments per trigger
      // and is folded into the effective seed — every cycle gets a
      // fresh, deterministic break arrangement. Turn off to lock the
      // exact same arrangement every time (useful when staging cues).
      .boolField ("cycle_seed",            true,                          state::PrimaryInput).label("Cycle Seed", "Cycle")

      // --- Flicker tail (tuning) ---
      .group("flicker", "Flicker Tail")
        .groupHelp(
          "Near the end of Release the beam sputters like a dying tube. *Flicker Start* is "
          "when (in release) the flicker begins; the on-time *Duty* ramps from *Start* to "
          "*End* (lower = darker, choppier) and *Flicker Freq* sets the strobe rate. Push "
          "*Duty End* toward 0 for a hard cut-out.")
      .floatField("flicker_start_t",       0.7f,  0.0f, 1.0f,             state::PrimaryInput).label("Flicker Start", "Start")
      .floatField("flicker_duty_start",    0.8f,  0.0f, 1.0f,             state::PrimaryInput).label("Duty Start", "DutySt")
      .floatField("flicker_duty_end",      0.05f, 0.0f, 1.0f,             state::PrimaryInput).label("Duty End", "DtyEnd")
      .floatField("flicker_freq_hz",       24.0f, 1.0f, 60.0f,            state::PrimaryInput).label("Flicker Freq", "Freq")

      .textureField("tex_in",  state::PrimaryInput)
      .textureField("tex_out", state::PrimaryOutput)
        .capability(state::Capability::Generator)
    );

  if (gpu::Device::backend() == gpu::Backend::None) return;

  state::registerShaderSPV("plasma_beam_cannon_render", RENDER_SPV, RENDER_SPV_SIZE);
  auto cs = gpu::Device::createShaderModuleByName("plasma_beam_cannon_render");
  if (!cs) return;

  s_pso = gpu::Device::createComputePSO(cs, "main", gpu::Bindings()
      .tex2d(0)
      .storageTex2d(1)
      .uniform(2)
      .storage(3));

  state::setOnStateReady(&on_state_ready);
  state::log("plasma_beam_cannon: module initialized");
}

// Per-instance construction: allocate State + its own GPU buffers.
void* create() {
  auto* s = new State();
  s->uniform_buf = gpu::Device::createBuffer(sizeof(Uniforms), gpu::BufferUsage::Uniform);
  s->break_buf   = gpu::Device::createBuffer(sizeof(GpuBreak) * MAX_BREAKS_TOTAL,
                                             gpu::BufferUsage::Storage);
  return s;
}

void destroy(void* self) {
  auto* s = static_cast<State*>(self);
  if (!s) return;
  s->uniform_buf.release();
  s->break_buf.release();
  delete s;
}

// Per-instance init tail: reset the ADSR state machine, edge-state,
// RNG streams, flicker accumulator and break pool; mark ready.
void init(void* self) {
  auto* s = static_cast<State*>(self);
  if (!s) return;

  s->phase = PHASE_IDLE;
  s->time_in_phase = 0.0;
  s->gate_prev = false;
  s->trigger_prev = false;
  s->trigger_pulse = false;
  s->trigger_hold_remaining = 0.0;
  s->rng_state = 0xCAFEBABEu;
  s->break_op_rng = 0xBADDCAFEu;
  s->breaks_seeded = false;
  s->flicker_phase = 0.0;
  s->flicker_on = true;
  s->seed_cycle_count = 0;
  std::memset(s->breaks, 0, sizeof(s->breaks));

  if (!s_pso.valid()) return;
  if (!s->uniform_buf.valid() || !s->break_buf.valid()) return;

  s->initialized = true;
  state::log("plasma_beam_cannon: initialized");
}

void tick(void* self, double dt) {
  auto* s = static_cast<State*>(self);
  if (!s || !s->initialized) return;
  s->time_in_phase += dt;

  // Self-fire (Off / Random / Beats — effect_auto_trigger.h). A shot already in
  // flight suppresses it, as before: one beam at a time. fires() must still run
  // so the beat clock keeps advancing across the suppressed shot.
  const int auto_fires = s->auto_trig.fires(dt);
  if (auto_fires > 0 && !s->trigger_pulse) {
    s->trigger_pulse = true;
    s->trigger_hold_remaining = (double)(s->attack_s + s->decay_s + s->sustain_s);
  }

  // Trigger pulse decay.
  if (s->trigger_pulse) {
    s->trigger_hold_remaining -= dt;
    if (s->trigger_hold_remaining <= 0.0) s->trigger_pulse = false;
  }
  bool effective_gate = s->trigger_pulse;

  // Phase transitions.
  switch (s->phase) {
    case PHASE_IDLE:
      if (effective_gate) enter_phase(*s, PHASE_ATTACK);
      break;
    case PHASE_ATTACK:
      if (!effective_gate) { enter_phase(*s, PHASE_RELEASE); break; }
      if (s->time_in_phase >= (double)s->attack_s) enter_phase(*s, PHASE_DECAY);
      break;
    case PHASE_DECAY:
      if (!effective_gate) { enter_phase(*s, PHASE_RELEASE); break; }
      if (s->time_in_phase >= (double)s->decay_s) enter_phase(*s, PHASE_SUSTAIN);
      break;
    case PHASE_SUSTAIN:
      if (!effective_gate) enter_phase(*s, PHASE_RELEASE);
      break;
    case PHASE_RELEASE:
      if (effective_gate) { enter_phase(*s, PHASE_ATTACK); break; }
      if (s->time_in_phase >= (double)s->release_s) enter_phase(*s, PHASE_IDLE);
      break;
  }

  // Update break particle simulation (only during release).
  update_breaks(*s, dt);
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
      if      (state::pathIs(path, plen, "gate")) {
        bool new_gate = state::patchFloat(i) != 0.0f;
        if (new_gate && !s->gate_prev) {
          s->trigger_pulse = true;
          s->trigger_hold_remaining = (double)(s->attack_s + s->decay_s + s->sustain_s);
        }
        s->gate = new_gate;
        s->gate_prev = new_gate;
      }
      else if (state::pathIs(path, plen, "seed_y"))               s->seed_y = state::patchFloat(i);
      else if (state::pathIs(path, plen, "seed_height"))          s->seed_height = state::patchFloat(i);
      else if (state::pathIs(path, plen, "attack_s"))             s->attack_s = state::patchFloat(i);
      else if (state::pathIs(path, plen, "decay_s"))              s->decay_s = state::patchFloat(i);
      else if (state::pathIs(path, plen, "sustain_s"))            s->sustain_s = state::patchFloat(i);
      else if (state::pathIs(path, plen, "release_s"))            s->release_s = state::patchFloat(i);
      else if (state::pathIs(path, plen, "attack_curve"))         s->attack_curve = state::patchFloat(i);
      else if (state::pathIs(path, plen, "decay_curve"))          s->decay_curve = state::patchFloat(i);
      else if (state::pathIs(path, plen, "release_curve"))        s->release_curve = state::patchFloat(i);
      else if (state::pathIs(path, plen, "intensity"))            s->intensity = state::patchFloat(i);
      else if (state::pathIs(path, plen, "bar_target"))           s->bar_target = (int)state::patchFloat(i);
      else if (state::pathIs(path, plen, "bar_target_all"))       s->bar_target_all = state::patchFloat(i) != 0.0f;
      else if (state::pathIs(path, plen, "beam_color")) {
        auto v = state::patchVec3(i);
        s->color_r = v.x; s->color_g = v.y; s->color_b = v.z;
      }

      // Break-particle params.
      else if (state::pathIs(path, plen, "break_count_per_bar"))  s->break_count = (int)state::patchFloat(i);
      else if (state::pathIs(path, plen, "attractor_fraction"))   s->attractor_fraction = state::patchFloat(i);
      else if (state::pathIs(path, plen, "spacer_fraction"))      s->spacer_fraction = state::patchFloat(i);
      else if (state::pathIs(path, plen, "min_break_size"))       s->min_break_size = state::patchFloat(i);
      else if (state::pathIs(path, plen, "max_break_size"))       s->max_break_size = state::patchFloat(i);
      else if (state::pathIs(path, plen, "force_strength"))       s->force_strength = state::patchFloat(i);
      else if (state::pathIs(path, plen, "spacer_strength"))      s->spacer_strength = state::patchFloat(i);
      else if (state::pathIs(path, plen, "force_softening"))      s->force_softening = state::patchFloat(i);
      else if (state::pathIs(path, plen, "damping_per_s"))        s->damping_per_s = state::patchFloat(i);
      else if (state::pathIs(path, plen, "interaction_radius"))   s->interaction_radius = state::patchFloat(i);
      else if (state::pathIs(path, plen, "teleport_rate"))        s->teleport_rate = state::patchFloat(i);
      else if (state::pathIs(path, plen, "length_target_start"))  s->length_target_start = state::patchFloat(i);
      else if (state::pathIs(path, plen, "length_target_end"))    s->length_target_end = state::patchFloat(i);
      else if (state::pathIs(path, plen, "length_target_curve"))  s->length_target_curve = state::patchFloat(i);
      else if (state::pathIs(path, plen, "grow_response"))        s->grow_response = state::patchFloat(i);
      else if (state::pathIs(path, plen, "activation_curve"))     s->activation_curve = state::patchFloat(i);
      else if (state::pathIs(path, plen, "activation_min"))       s->activation_min = state::patchFloat(i);
      else if (state::pathIs(path, plen, "growth_fraction"))      s->growth_fraction = state::patchFloat(i);
      else if (state::pathIs(path, plen, "break_seed"))           s->break_seed = (int)state::patchFloat(i);
      else if (state::pathIs(path, plen, "cycle_seed"))           s->cycle_seed = state::patchFloat(i) != 0.0f;

      // Flicker tail.
      else if (state::pathIs(path, plen, "flicker_start_t"))      s->flicker_start_t = state::patchFloat(i);
      else if (state::pathIs(path, plen, "flicker_duty_start"))   s->flicker_duty_start = state::patchFloat(i);
      else if (state::pathIs(path, plen, "flicker_duty_end"))     s->flicker_duty_end = state::patchFloat(i);
      else if (state::pathIs(path, plen, "flicker_freq_hz"))      s->flicker_freq_hz = state::patchFloat(i);
    }

    // Event field — momentary (on/off like gate; 1 on press, 0 on release).
    // Fire on the rising edge of the VALUE. The executor replays the stored
    // value every frame, so a value-less "any trigger patch fires" check
    // re-arms forever (the "stuck on" loop); the rising edge is what makes it
    // replay-safe (style guide §8.2). Like `gate`, a fresh edge re-triggers
    // mid-cycle.
    if (op == state::PatchReplace && state::pathIs(path, plen, "trigger")) {
      bool tval = state::patchFloat(i) != 0.0f;
      if (tval && !s->trigger_prev) {
        s->trigger_pulse = true;
        s->trigger_hold_remaining = (double)(s->attack_s + s->decay_s + s->sustain_s);
      }
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

  // Beam half-height based on current phase. Release stays at full
  // height — break particles do the visual decay instead.
  float half_height = 0.0f;
  bool active = false;
  switch (s->phase) {
    case PHASE_IDLE:
      half_height = 0.0f;
      break;
    case PHASE_ATTACK: {
      // Seed grows from 0 → seed_height across the attack phase,
      // shaped by attack_curve. Default linear (curve = 0 slider).
      float t = (s->attack_s > 0.0f) ? (float)(s->time_in_phase / (double)s->attack_s) : 1.0f;
      if (t > 1.0f) t = 1.0f;
      float t_curved = std::pow(t, fx::signedSliderToExp(s->attack_curve));
      half_height = s->seed_height * 0.5f * t_curved;
      active = (half_height > 0.0f);
    } break;
    case PHASE_DECAY: {
      float t = (s->decay_s > 0.0f) ? (float)(s->time_in_phase / (double)s->decay_s) : 1.0f;
      if (t > 1.0f) t = 1.0f;
      float t_curved = std::pow(t, fx::signedSliderToExp(s->decay_curve));
      half_height = lerpf(s->seed_height * 0.5f, 0.5f, t_curved);
      active = true;
    } break;
    case PHASE_SUSTAIN:
      half_height = 0.5f;
      active = true;
      break;
    case PHASE_RELEASE:
      // Beam stays full-height; breaks eat it away. Flicker tail
      // can mask everything to black periodically.
      half_height = 0.5f;
      active = true;
      break;
  }

  float y_min = s->seed_y - half_height;
  float y_max = s->seed_y + half_height;
  if (y_min < 0.0f) y_min = 0.0f;
  if (y_max > 1.0f) y_max = 1.0f;

  // Pack the per-bar break buffer (only meaningful during release; we
  // pack regardless because the shader's `breaks_active` flag gates
  // whether to consult them).
  GpuBreak gpu_breaks[MAX_BREAKS_TOTAL];
  std::memset(gpu_breaks, 0, sizeof(gpu_breaks));
  int per_bar = s->break_count;
  if (per_bar < 1) per_bar = 1;
  if (per_bar > MAX_BREAKS_PER_BAR) per_bar = MAX_BREAKS_PER_BAR;
  for (int bar = 0; bar < BARS; bar++) {
    for (int i = 0; i < per_bar; i++) {
      const CpuBreak& src = s->breaks[bar][i];
      GpuBreak& dst = gpu_breaks[bar * per_bar + i];
      dst.y      = src.y;
      dst.size   = src.size;
      dst.type_f = (float)src.type;
      dst._pad   = 0.0f;
    }
  }
  s->break_buf.writeBytes(gpu_breaks, (int)sizeof(GpuBreak) * BARS * per_bar);

  // Flicker is only visible at the tail end of release. Use the
  // release_curve-warped time so a non-zero release_curve shifts
  // flicker onset to match the perceived pace.
  bool flicker_active = (s->phase == PHASE_RELEASE);
  if (flicker_active && s->release_s > 0.0f) {
    float release_t_raw = clampf((float)(s->time_in_phase / (double)s->release_s), 0.0f, 1.0f);
    float release_t = std::pow(release_t_raw, fx::signedSliderToExp(s->release_curve));
    flicker_active = (release_t >= s->flicker_start_t);
  }

  Uniforms u = {};
  u.beam_y_min        = y_min;
  u.beam_y_max        = y_max;
  u.intensity         = s->intensity;
  u.active            = active ? 1u : 0u;
  u.color_r           = s->color_r;
  u.color_g           = s->color_g;
  u.color_b           = s->color_b;
  u.bar_target        = (uint32_t)s->bar_target;
  u.bar_target_all    = s->bar_target_all ? 1u : 0u;
  u.particles_per_bar = (uint32_t)per_bar;
  u.breaks_active     = (s->phase == PHASE_RELEASE) ? 1u : 0u;
  u.flicker_active    = flicker_active ? 1u : 0u;
  u.flicker_on        = s->flicker_on ? 1u : 0u;
  s->uniform_buf.writeOne(u);

  auto cp = gpu::ComputePass::begin();
  cp.setPSO(s_pso);
  cp.setTexture(in,  0, 0);
  cp.setTexture(out, 1, 1);
  cp.setBuffer(s->uniform_buf, 2);
  cp.setBuffer(s->break_buf,   3);
  cp.dispatch((vp_w + 7) / 8, (vp_h + 7) / 8);
  cp.end();

  gpu::Device::submit();
}

} // namespace plasma_beam_cannon
