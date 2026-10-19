#pragma once
/*
 * sweep_knob.h — the performance knob behind the house sweep: how fast it is
 * moving, what level it asks for, the bounce coming back in, and the charge a
 * hard pass through the middle earns and the throw that spends it at the
 * choke.
 *
 * This is three_planes_rig's SweepCore (nano-modules,
 * native/src/sketch/three_planes_rig.h) with the tower taken off. Sweep Choke
 * (sweep_choke_core.h) and Side Racks (side_racks_core.h) both ride it, and
 * they want different pictures out of it — a rolling mask and blinks on the
 * one, a spotlight climbing three bars on the other — but the knob is the
 * same instrument, so there is one copy of it. Sweep Choke's goldens pin it.
 *
 * THE KNOB is bipolar in concept and 0..1 like every MIDI mapping: 0.5 is
 * home, full and still. Each half has a wide DEADZONE that stays at full, and
 * past it the level eases down to `1 - depth` at the end — the CHOKE.
 *
 * THE BOUNCE is a lookahead in time: coming back in, the secant of the level
 * across the rate window drives a stiff spring, so the light leads the hand
 * past full and springs back. Inward only; on the way out it never overshoots.
 *
 * CATCH AND THROW: a pass through the deadzone (or clean over the middle
 * between two samples) charges by how HARD it was; the charge leaks away; the
 * choke spends it all at once as `release`, which rings out linearly over
 * `ring_time`. What a caller does with `release` is its own business.
 *
 * Host-free: no effect ABI, no GPU, no allocation.
 */

#include <cmath>

#include <sketch/knob_rate.h>

namespace sweep_knob {

/// The knob's rest position. An unwired card sits here: full, no motion.
constexpr float kSweepCenter = 0.5f;

/// THE BOUNCE — see three_planes_rig.h, where these were tuned. Seconds of
/// lookahead at Bounce 1, and a stiff spring damped just short of dead: the
/// return has one small dip in it and no wobble after.
constexpr float kBounceLead = 0.05f;
constexpr float kBounceFreq = 60.0f;   ///< rad/s
constexpr float kBounceDamp = 0.30f;   ///< zeta
constexpr float kBounceStep = 0.008f;  ///< integrator sub-step, s

/// Below this the choke counts as reached and a held charge is thrown. Tested
/// against the knob's level, never a rolled or staggered picture of it: where
/// the mute is is a fact about the knob.
constexpr float kMuteGain = 0.06f;

/// Read fresh each tick — a wire may be moving any of it.
struct Params {
  float sweep = kSweepCenter;
  float sense = 2.0f;       ///< full-scale drag speed, knob ranges / second
  float window = 0.09f;     ///< boxcar span the speed is measured over, s
  float decay = 0.18f;      ///< motion envelope release, s
  float deadzone = 0.45f;   ///< fraction of each half that stays fully lit
  float depth = 1.0f;       ///< how far the ends fade; 1 = to black
  float bounce = 0.55f;     ///< how far a fast return overshoots
  float latch_drive = 2.0f; ///< how readily a pass through the middle charges
  float latch_decay = 1.2f; ///< seconds a charge survives before it bleeds away
  float ring_time = 1.4f;   ///< seconds the thrown tail takes to ring out
};

inline float clampf(float v, float lo, float hi) { return v < lo ? lo : (v > hi ? hi : v); }
inline float clamp01(float v) { return clampf(v, 0.0f, 1.0f); }
inline float smooth01(float t) { t = clamp01(t); return t * t * (3.0f - 2.0f * t); }

/// How far into the fade the knob is: 0 anywhere in the deadzone, 1 at the end.
inline float fadeProgress(float deadzone, float sweep) {
  const float d = sweep - kSweepCenter;
  const float mag = clamp01((d < 0.0f ? -d : d) / kSweepCenter);
  const float dz = clamp01(deadzone);
  return clamp01((mag - dz) / (1.0f - dz > 1e-4f ? 1.0f - dz : 1e-4f));
}

/// The level the knob asks for: full across the deadzone, easing to
/// `1 - depth` at either end. Smoothstep, so the shoulder where the fade
/// begins has no corner in it to see as you ride past.
inline float positionLevel(float deadzone, float depth, float sweep) {
  return clamp01(1.0f - smooth01(fadeProgress(deadzone, sweep)) * clamp01(depth));
}

struct Knob {
  knob_rate::KnobRate rate;
  float speed = 0.0f;       ///< 0..1 motion envelope

  float charge = 0.0f;
  float release = 0.0f;     ///< the thrown tail, 1 at a full throw, ringing to 0
  bool armed = false;
  float last = kSweepCenter;

  float bounce = 0.0f;      ///< the spring's lead, in level units (inward only)
  float bounce_v = 0.0f;

  // --- This tick ----------------------------------------------------------
  float sweep = kSweepCenter;  ///< the knob, sanitized
  float level = 1.0f;          ///< what the knob asks for, 0..1
  float progress = 0.0f;       ///< fadeProgress: 0 in the deadzone, 1 choked
  bool fresh_throw = false;    ///< this tick spent a charge

  void reset() { *this = Knob(); }

  /// One tick. `dt` must already be clamped by the caller (0 is a refresh
  /// that ages nothing).
  void tick(const Params& p, float dt) {
    // A NaN patch holds the middle rather than poisoning everything below.
    const float sw = (p.sweep == p.sweep) ? clamp01(p.sweep) : kSweepCenter;
    sweep = sw;

    // --- Speed, off knob_rate's boxcar — see there for why a MIDI knob
    //     cannot be differenced per frame.
    const float r = rate.sample(sw, dt, p.window);
    const float ar = r < 0.0f ? -r : r;
    if (dt > 0.0f) {
      const float v = clamp01(ar / (p.sense > 1e-4f ? p.sense : 1e-4f));
      const float rel = std::exp(-dt / (p.decay > 1e-3f ? p.decay : 1e-3f));
      speed = v > speed * rel ? v : speed * rel;
      if (speed < 1e-4f) speed = 0.0f;
    }

    progress = fadeProgress(p.deadzone, sw);
    const float lvl = positionLevel(p.deadzone, p.depth, sw);
    level = lvl;

    // --- THE BOUNCE: the secant of the dimmer across the rate window drives
    //     a stiff spring, inward only.
    float target = 0.0f;
    if (rate.last_span > 1e-6f) {
      const float from = positionLevel(p.deadzone, p.depth, clamp01(rate.last_from));
      const float climb = (lvl - from) / rate.last_span;
      if (climb > 0.0f) target = kBounceLead * clamp01(p.bounce) * climb;
    }
    if (dt > 0.0f) {
      const int n = (int)(dt / kBounceStep) + 1;
      const float h = dt / (float)n;
      const float k = kBounceFreq * kBounceFreq;
      const float damp = 2.0f * kBounceDamp * kBounceFreq;
      for (int i = 0; i < n; ++i) {
        bounce_v += (k * (target - bounce) - damp * bounce_v) * h;
        bounce += bounce_v * h;
      }
      if (bounce > -1e-5f && bounce < 1e-5f && bounce_v > -1e-4f && bounce_v < 1e-4f) {
        bounce = 0.0f;
        bounce_v = 0.0f;
      }
    }

    // --- CATCH AND THROW. A pass through the middle (or clean over it
    //     between two samples) charges by its VIGOUR; the charge leaks;
    //     reaching the choke spends it all at once, and what was thrown rings
    //     out on its own clock.
    const float mag_dz = clamp01(p.deadzone);
    const float d = sw - kSweepCenter;
    const float mag = clamp01((d < 0.0f ? -d : d) / kSweepCenter);
    const bool crossed = ((last - kSweepCenter) < 0.0f) != (d < 0.0f);
    last = sw;
    if (mag <= mag_dz || crossed) {
      const float vigour = clamp01(speed * (p.latch_drive > 0.0f ? p.latch_drive : 0.0f));
      if (vigour > charge) charge = vigour;
      armed = true;
    }
    if (charge > 0.0f && dt > 0.0f) {
      charge *= std::exp(-dt / (p.latch_decay > 1e-3f ? p.latch_decay : 1e-3f));
      if (charge < 1e-3f) charge = 0.0f;
    }
    fresh_throw = armed && lvl <= kMuteGain && charge > 0.0f;
    if (fresh_throw) {
      if (charge > release) release = charge;
      charge = 0.0f;
      armed = false;
    }
    if (release > 0.0f && !fresh_throw) {
      release -= dt / (p.ring_time > 1e-3f ? p.ring_time : 1e-3f);
      if (release < 0.0f) release = 0.0f;
    }
  }
};

}  // namespace sweep_knob
