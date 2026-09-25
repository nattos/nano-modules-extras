#pragma once
/*
 * sweep_choke_core.h — the logic behind `source.light.sweep_choke`: ONE knob that
 * sweeps and chokes the four LED bars.
 *
 * This is three_planes_rig's SweepCore (nano-modules,
 * native/src/sketch/three_planes_rig.h) lifted off the tower and onto the rig
 * itself. Everything that made that sweep a performance survives — the wide
 * deadzone at full brightness, the bounce coming back in, the charge a hard
 * pass through the middle earns and the throw that spends it at the choke, the
 * stutter that motion wakes and stillness settles — and the constants carry
 * the same values for the same reasons. What changes is what it is dimming.
 * The rig dimmed three floors as one; this dims forty segments, and not as
 * one:
 *
 * THE FADE IS A TILT. Out of the deadzone the grid does not dim evenly — a
 * soft front travels across it at `angle_deg`, so one side is already out
 * while the other is still at full. Park anywhere in the fade and you have a
 * gradient; keep going and the gradient slides off the far edge into black.
 * The shape of the light is a fact about where the knob is, exactly as the
 * rig's single level was, which is what lets you ride it. `tilt` is how much
 * of a front it is: 0 is the rig's even dimmer, 1 a narrow edge.
 *
 * Which edge goes first when the knob goes the OTHER way is a choice
 * (`side_mode`): Ping-Pong mirrors it, so the knob's direction shows up in the
 * light; Forward keeps one direction, so only distance from centre matters;
 * One-Way (below) keeps the front travelling one way in AND out.
 *
 * THE TAIL. Reaching the choke with a charge spends it as `release`, which
 * rings out linearly over `ring_time` — and while it rings, segments BLINK.
 * Blinks are drawn toward the side of the tilt that went out LAST, so the tail
 * reads as what the sweep left behind rather than as noise on the grid. Like
 * three_planes_strobe.h, the tail never dims: blinks arrive at a constant rate
 * and it is their LENGTH the release takes away, until they are shorter than a
 * frame and start missing outright. A thinning tail reads as a thing running
 * down; a dimming one reads as a light going out.
 *
 * A blink TOGGLES what it lands on — dark comes up to full, lit goes to
 * black — so a tail over a lit Base shows as holes punched in it, and is never
 * lost under it. `tail_chunk` sizes them: 0 single segments, 1 whole bars, and
 * in between runs up the bar of a random length, fewer of them, so the same
 * number of segments is lit on average.
 *
 * TWO PICTURES. The tilt and the stutter are `level` — 0..1 per segment, and
 * what the effect publishes as a GRAYSCALE MASK for other effects to be
 * choked and revealed by. Past the front the whole mask also dims with the
 * fade (`fade`), so a sweep leaves gray behind the edge as well as moving it.
 * What the bars themselves show is `glow`: the GESTURE rather than the
 * position —
 *
 *   GLINTS, three_planes_glints.h whole: one launched each time the knob enters
 *   the middle band, travelling at the knob's own speed along the gradient's
 *   `angle_deg`, always the same way whichever way the knob went, gliding on
 *   after it stops for `glint_linger` and puttering out; chaos ones under a
 *   brisk sweep; held and slung by a throw. A glint is pure overdrive — past 1
 *   a segment pushes toward white.
 *
 *   THE TAIL, above.
 *
 *   THE BOUNCE, as a flash: coming back in fast, the whole grid overshoots and
 *   springs back.
 *
 *   A BASE, optional: the mask itself at `base`, which the glints' dark wake
 *   shades as they pass. 0 leaves the bars black between gestures.
 *
 * ONE-WAY. The third `side_mode`, and the tilt's answer to a glint: the front
 * always travels toward +axis. Going out the dark chases it; coming back in,
 * the light does, re-entering from the side that went out FIRST. Which way it
 * runs flips only at the ends — at the choke and back in the deadzone — where
 * both ways agree, so it never jumps. Reverse mid-fade and it retraces.
 *
 * Host-free, like three_planes_glints.h and three_planes_strobe.h: no effect
 * ABI, no GPU, so the Catch2 goldens drive the whole thing at an exact dt.
 */

#include <cmath>

#include <sketch/knob_rate.h>
#include <sketch/led_bars.h>
#include <sketch/three_planes_glints.h>

namespace sweep_choke_core {

using led_bars::kBars;
using led_bars::kSegments;
using led_bars::kCells;

/// A frame longer than this is a transport stall, not slow motion. Same clamp,
/// and the same reason, as the rest of the instrument.
constexpr float kMaxDt = 0.25f;

/// The knob's rest position. Bipolar in concept, 0..1 like every MIDI mapping.
/// An unwired card sits here, which is full intensity and no motion.
constexpr float kSweepCenter = 0.5f;

/// THE BOUNCE — see three_planes_rig.h, where these were tuned. Seconds of
/// lookahead at Bounce 1, and a stiff spring damped just short of dead: the
/// return has one small dip in it and no wobble after.
constexpr float kBounceLead = 0.05f;
constexpr float kBounceFreq = 60.0f;   ///< rad/s
constexpr float kBounceDamp = 0.30f;   ///< zeta
constexpr float kBounceStep = 0.008f;  ///< integrator sub-step, s

/// How far past full a segment's glow is read: from 1 to here it pushes toward
/// white, and beyond it is simply white. What the bounce and the glints
/// overshoot into.
constexpr float kLevelMax = 1.5f;

/// The bounce as the bars show it. It is a lead over the knob in LEVEL units —
/// a hard return is worth a few tenths — and on black that is a timid flash,
/// so it is scaled up to where a hard return reaches full and a slammed one
/// blows out.
constexpr float kBounceFlash = 4.0f;

/// A glint's reach either side of its centre, in half-widths — nano_glint.hlsl's
/// super-Gaussian core has died by 1.5, the wake trailing it by 5.5. Sizes the
/// margins it is born and retired in, as three_planes does.
constexpr float kGlintSkirt = 1.5f;
constexpr float kGlintWakeReach = 5.5f;
/// nano_glint.hlsl's wake: offset behind the core and width, in half-widths.
constexpr float kGlintWake = 1.5f;
constexpr float kGlintWakeW = 1.8f;

/// The stutter's wake, in knob ranges per second: any motion past this re-arms
/// it outright. Well under one MIDI step per rate window. See the rig.
constexpr float kFlickWake = 0.02f;

/// Below this the choke counts as reached and a held charge is thrown. Tested
/// against the BRIGHTEST segment, so a tilt still has to have finished
/// sliding off the grid before it throws.
constexpr float kMuteGain = 0.06f;

/// How wide a front `tilt = 1` is, in axis units: the smoothstep transition
/// spans 1/kSpread of the grid. Larger is a narrower edge.
constexpr float kSpread = 2.0f;

/// Blink slots. Enough for a dense tail and a stutter at once.
constexpr int kMaxBlinks = 16;

enum SideMode { SidePingPong = 0, SideForward = 1, SideOneWay = 2 };

/// Read fresh each tick — a wire may be moving any of it.
struct Params {
  float sweep = kSweepCenter;
  float sense = 2.0f;       ///< full-scale drag speed, knob ranges / second
  float window = 0.09f;     ///< boxcar span the speed is measured over, s
  float decay = 0.18f;      ///< motion envelope release, s
  float deadzone = 0.45f;   ///< fraction of each half that stays fully lit
  float depth = 1.0f;       ///< how far the ends fade; 1 = to black

  float angle_deg = 90.0f;  ///< which way the dark travels, CCW from +x; 90 = upward
  int side_mode = SidePingPong;
  float tilt = 0.6f;        ///< 0 even dimmer .. 1 narrow travelling front
  float fade = 0.5f;        ///< how far the whole mask dims under the front, 0..1

  float bounce = 0.55f;     ///< how far a fast return overshoots
  float flicker = 0.6f;     ///< how hard segments stutter inside the fade
  float settle = 1.2f;      ///< seconds the stutter carries on after the knob stops

  float latch_drive = 2.0f; ///< how readily a pass through the middle charges
  float latch_decay = 1.2f; ///< seconds a charge survives before it bleeds away
  float ring_time = 1.4f;   ///< seconds the thrown tail takes to ring out

  float tail_rate = 18.0f;    ///< blinks launched per second — constant through the tail
  float tail_duty = 2.0f;     ///< blink length at full release, in launch intervals
  float tail_density = 2.5f;  ///< segments per launch
  float tail_bias = 0.75f;    ///< 0 anywhere .. 1 only where the light went out last
  float tail_chunk = 0.0f;    ///< 0 single segments .. 1 whole bars per blink

  // --- The glow: what the bars show ---------------------------------------
  float base = 0.0f;          ///< the mask shown under the glints, 0..1
  float glint_ratio = 1.0f;   ///< crossings of the grid per knob range
  float glint_chaos = 4.0f;   ///< small extra glints per second, sweeping hard
  float glint_gain = 1.6f;    ///< a glint's peak drive; past 1 is toward white
  float glint_shadow = 0.55f; ///< the wake's depth, on the base
  float glint_linger = three_planes_glints::kSpeedRelease;  ///< s a glint glides on
  float glint_width = 0.1f;   ///< half-width, as a fraction of one crossing
  float glint_sling = 1.0f;   ///< how far a throw hurries them off
};

/// A run of segments mid-blink: `len` of them up one bar from `cell`. It
/// toggles them (blended by `strength`) until `t_left` runs out. A TAIL blink
/// lands on the bars; a stutter blink lands on the mask.
struct Blink {
  int cell = -1;
  int len = 1;
  float t_left = 0.0f;
  float strength = 0.0f;
  bool tail = false;
  bool live = false;

  bool covers(int i) const {
    return live && i >= cell && i < cell + len;
  }
};

/// The range of run lengths a tail blink draws from at `chunk`, in segments:
/// one at 0, a whole bar at 1, and in between a spread that widens and climbs.
inline void chunkRange(float chunk, float& lo, float& hi) {
  const float c = chunk < 0.0f ? 0.0f : (chunk > 1.0f ? 1.0f : chunk);
  lo = 1.0f + (float)(kSegments - 1) * c * c;
  hi = 1.0f + (float)(kSegments - 1) * std::pow(c, 0.6f);
}

inline float clampf(float v, float lo, float hi) { return v < lo ? lo : (v > hi ? hi : v); }
inline float clamp01(float v) { return clampf(v, 0.0f, 1.0f); }
inline float smooth01(float t) { t = clamp01(t); return t * t * (3.0f - 2.0f * t); }

/// Where a segment sits along the travel axis, 0 the edge that goes out FIRST
/// and 1 the edge that goes out last, renormalised over the whole grid so any
/// angle uses the full range. Cells are laid out as the bars stand — columns
/// left to right across 0..1, segments bottom to top across 0..1 — so 45 deg
/// runs corner to corner whatever the physical proportions.
inline void axisCoords(float angle_deg, float out[kCells]) {
  const float a = angle_deg * (3.14159265358979323846f / 180.0f);
  const float dx = std::cos(a), dy = std::sin(a);
  float lo = 1e9f, hi = -1e9f;
  for (int b = 0; b < kBars; ++b)
    for (int s = 0; s < kSegments; ++s) {
      const float x = (b + 0.5f) / kBars, y = (s + 0.5f) / kSegments;
      const float v = x * dx + y * dy;
      out[led_bars::cellIndex(b, s)] = v;
      lo = v < lo ? v : lo;
      hi = v > hi ? v : hi;
    }
  const float span = hi - lo;
  for (int i = 0; i < kCells; ++i) out[i] = span > 1e-5f ? (out[i] - lo) / span : 0.5f;
}

/// How far into the fade the knob is: 0 anywhere in the deadzone, 1 at the end.
inline float fadeProgress(const Params& p, float sweep) {
  const float d = sweep - kSweepCenter;
  const float mag = clamp01((d < 0.0f ? -d : d) / kSweepCenter);
  const float dz = clamp01(p.deadzone);
  return clamp01((mag - dz) / (1.0f - dz > 1e-4f ? 1.0f - dz : 1e-4f));
}

/// The position law for ONE segment at axis coordinate `a`. The front: every
/// segment is lit at progress 0 and out at progress 1, and between them the
/// edge at a = 0 leads by `tilt * kSpread` of the fade. Smoothstep, so the
/// shoulder has no corner in it to see as you ride past. Under the front the
/// whole grid also dims with progress by `fade`, so what the front has not
/// reached yet is gray rather than white.
inline float cellGain(const Params& p, float progress, float a) {
  const float k = clamp01(p.tilt) * kSpread;
  const float t = progress * (1.0f + k) - k * a;
  const float lit = (1.0f - smooth01(t)) * (1.0f - clamp01(p.fade) * smooth01(progress));
  return clamp01(1.0f - (1.0f - lit) * clamp01(p.depth));
}

/// Mean positional gain over the grid. Independent of direction — mirroring
/// the axis only reorders the cells — which is what lets the bounce read it
/// across a window that straddles the centre.
inline float meanGain(const Params& p, float sweep, const float axis[kCells]) {
  const float prog = fadeProgress(p, sweep);
  float sum = 0.0f;
  for (int i = 0; i < kCells; ++i) sum += cellGain(p, prog, axis[i]);
  return sum / (float)kCells;
}

struct Core {
  knob_rate::KnobRate rate;
  float speed = 0.0f;       ///< 0..1 motion envelope
  float flick_life = 0.0f;  ///< how much stutter is left; see the rig

  float charge = 0.0f;
  float release = 0.0f;
  bool armed = false;
  float last = kSweepCenter;

  float bounce = 0.0f;
  float bounce_v = 0.0f;

  /// One-Way only: the knob is on its way back IN, so the front is the light
  /// returning rather than the dark leaving. Latched at the ends.
  bool inbound = false;

  three_planes_glints::Core glints;

  float tail_clock = 0.0f;  ///< launches owed, fractional
  float settle_t = 0.0f;    ///< seconds to the next stutter blink
  Blink blinks[kMaxBlinks];
  unsigned rng = 0x9e3779b9u;

  // --- This frame ---------------------------------------------------------
  /// The mask per segment, 0..1: tilt, tail and stutter.
  float level[kCells] = {};
  /// What the bars show per segment, 0.. (past 1 pushes toward white).
  float glow[kCells] = {};
  float glint_axis[kCells] = {};  ///< each segment on the travel axis, unmirrored
  float axis[kCells] = {};  ///< this frame's axis, already mirrored for the side
  float mean_level = 1.0f;
  float sweep_out = kSweepCenter;

  Core() { for (int i = 0; i < kCells; ++i) level[i] = 1.0f; }

  void reset() { *this = Core(); }

  int liveBlinks() const {
    int n = 0;
    for (int i = 0; i < kMaxBlinks; ++i) if (blinks[i].live) ++n;
    return n;
  }

  void tick(const Params& p, float dt) {
    if (!(dt > 0.0f)) dt = 0.0f;
    if (dt > kMaxDt) dt = kMaxDt;

    // A NaN patch holds the middle rather than poisoning everything below.
    const float sw = (p.sweep == p.sweep) ? clamp01(p.sweep) : kSweepCenter;
    sweep_out = sw;

    // --- 1. Speed, and the stutter's life. Both off knob_rate's boxcar — see
    //        there for why a MIDI knob cannot be differenced per frame.
    const float r = rate.sample(sw, dt, p.window);
    const float ar = r < 0.0f ? -r : r;
    if (dt > 0.0f) {
      const float v = clamp01(ar / (p.sense > 1e-4f ? p.sense : 1e-4f));
      const float rel = std::exp(-dt / (p.decay > 1e-3f ? p.decay : 1e-3f));
      speed = v > speed * rel ? v : speed * rel;
      if (speed < 1e-4f) speed = 0.0f;

      const float woke = clamp01(ar / kFlickWake);
      const float srel = std::exp(-dt / (p.settle > 1e-3f ? p.settle : 1e-3f));
      flick_life = woke > flick_life * srel ? woke : flick_life * srel;
      if (flick_life < 1e-4f) flick_life = 0.0f;
    }

    // --- 2. The axis for this side of the knob, and the positional gains.
    float base_axis[kCells];
    axisCoords(p.angle_deg, base_axis);
    const float prog = fadeProgress(p, sw);
    // One-Way turns round only where both directions agree: fully out, or
    // fully home. So it can never jump mid-fade.
    if (prog >= 1.0f - 1e-4f) inbound = true;
    else if (prog <= 0.0f) inbound = false;
    const bool flip = p.side_mode == SideOneWay
                          ? inbound
                          : (p.side_mode == SidePingPong && sw < kSweepCenter);
    for (int i = 0; i < kCells; ++i) axis[i] = flip ? 1.0f - base_axis[i] : base_axis[i];

    float gain[kCells];
    float brightest = 0.0f, sum = 0.0f;
    for (int i = 0; i < kCells; ++i) {
      gain[i] = cellGain(p, prog, axis[i]);
      brightest = gain[i] > brightest ? gain[i] : brightest;
      sum += gain[i];
    }
    const float mean = sum / (float)kCells;

    // --- 2a. THE BOUNCE, as in the rig: the secant of the (mean) dimmer across
    //         the rate window drives a stiff spring, inward only. The bars
    //         show it as a flash over the whole grid that springs back (step
    //         7); the mask never sees it — a mask has nothing past full.
    float target = 0.0f;
    if (rate.last_span > 1e-6f) {
      const float from = meanGain(p, clamp01(rate.last_from), base_axis);
      const float climb = (mean - from) / rate.last_span;
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

    // --- 2b. CATCH AND THROW, as in the rig. A pass through the middle
    //         (or clean over it between two samples) charges by its VIGOUR;
    //         the charge leaks; reaching the choke spends it all at once, and
    //         what was thrown rings out on its own clock.
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
    const bool fresh_throw = armed && brightest <= kMuteGain && charge > 0.0f;
    if (fresh_throw) {
      if (charge > release) release = charge;
      charge = 0.0f;
      armed = false;
      // The tail lands WITH the choke, not one launch interval after it.
      if (tail_clock < 1.0f) tail_clock = 1.0f;
    }
    if (release > 0.0f && !fresh_throw) {
      release -= dt / (p.ring_time > 1e-3f ? p.ring_time : 1e-3f);
      if (release < 0.0f) release = 0.0f;
    }

    // --- 2c. THE GLINTS, three_planes' own particles: launched entering the
    //         middle band (the deadzone, so a glint leaves as the grid comes
    //         back to full), at the knob's speed, always toward +axis. A throw
    //         holds them and, by `glint_sling`, hurries them off.
    {
      three_planes_glints::Params gp;
      gp.sweep = sw;
      gp.band = clamp01(p.deadzone);
      gp.ratio = p.glint_ratio;
      gp.chaos = p.glint_chaos;
      gp.fling = release;
      gp.sling = p.glint_sling;
      gp.linger = p.glint_linger;
      const float w = (p.glint_width > 1e-3f ? p.glint_width : 1e-3f) *
                      three_planes_glints::kMaxWidthFactor;
      gp.lead = kGlintSkirt * w;
      gp.trail = kGlintWakeReach * w;
      glints.tick(gp, dt);
    }

    // --- 3. Age the blinks. One that runs out mid-frame is simply gone.
    for (int i = 0; i < kMaxBlinks; ++i) {
      Blink& b = blinks[i];
      if (!b.live) continue;
      b.t_left -= dt;
      if (b.t_left <= 0.0f) b = Blink();
    }

    // --- 4. THE TAIL. A constant launch clock while the release rings; the
    //        release shortens each blink rather than dimming it. A launch lands
    //        partway through the frame, so a blink shorter than the time left
    //        in it is already over when the frame is sampled — that is the
    //        sputter at the end, for free.
    const float trate = p.tail_rate > 0.0f ? p.tail_rate : 0.0f;
    if (release > 0.0f && trate > 0.0f && dt > 0.0f) {
      tail_clock += trate * dt;
      const float interval = 1.0f / trate;
      const float len = (p.tail_duty > 0.0f ? p.tail_duty : 0.0f) * interval * release;
      for (int guard = 0; tail_clock >= 1.0f && guard < 64; ++guard) {
        tail_clock -= 1.0f;
        const float age = tail_clock * interval;   // how long ago it landed
        // CHUNK trades count for size: runs grow up the bar, stochastically,
        // and fewer launch, so as many segments are lit on average.
        float lo, hi;
        chunkRange(p.tail_chunk, lo, hi);
        float want = (p.tail_density > 0.0f ? p.tail_density : 0.0f) / (0.5f * (lo + hi));
        int count = (int)want;
        if (rand01() < want - (float)count) ++count;
        for (int c = 0; c < count; ++c) {
          const int cell = pickTrailing(p.tail_bias);
          if (cell < 0) continue;
          const float x = lo + (hi - lo) * rand01();
          int run = (int)x;
          if (rand01() < x - (float)run) ++run;
          run = run < 1 ? 1 : (run > kSegments ? kSegments : run);
          // Somewhere over the picked segment, inside its bar.
          const int bar = cell / kSegments, seg = cell % kSegments;
          int s0 = seg - (int)(rand01() * (float)run);
          s0 = s0 < 0 ? 0 : (s0 > kSegments - run ? kSegments - run : s0);
          spawn(led_bars::cellIndex(bar, s0), len - age, 1.0f, run, true);
        }
      }
    } else if (release <= 0.0f) {
      // Only a spent tail resets the clock. A zero-dt tick (the effect
      // refreshing position at render) must leave a launch that is owed.
      tail_clock = 0.0f;
    }

    // --- 5. THE SETTLE STUTTER. The rig's per-floor drive, per segment:
    //        loudest where a segment is halfway out, gated by the life motion
    //        re-arms. One scheduler across the grid, so it reads as one
    //        struggling fixture rather than as static — but a wide fade band
    //        has more places to struggle in, and the gap closes for it.
    float drive[kCells];
    float drive_max = 0.0f, drive_sum = 0.0f;
    int band = 0;
    for (int i = 0; i < kCells; ++i) {
      drive[i] = clamp01(4.0f * gain[i] * (1.0f - gain[i])) * clamp01(p.flicker) * flick_life;
      if (drive[i] > drive_max) drive_max = drive[i];
      drive_sum += drive[i];
      if (drive[i] > 0.05f) ++band;
    }
    if (drive_max <= 1e-3f) {
      settle_t = 0.0f;
    } else if (dt > 0.0f) {
      settle_t -= dt;
      if (settle_t <= 0.0f) {
        const int cell = pickWeighted(drive, drive_sum);
        if (cell >= 0 && !blinking(cell)) spawn(cell, 0.02f + 0.07f * rand01(), drive[cell]);
        const float gap = 0.55f + (0.05f - 0.55f) * drive_max;
        settle_t = gap * (0.4f + 1.2f * rand01()) / (1.0f + 0.25f * (float)band);
      }
    }

    // --- 6. The mask. Position first, then the stutter toggles against it: a
    //        dark segment comes up to full, a lit one goes out. The TAIL is
    //        not here — it is on the bars (step 7).
    float lsum = 0.0f;
    for (int i = 0; i < kCells; ++i) level[i] = gain[i];
    for (int i = 0; i < kMaxBlinks; ++i) {
      const Blink& b = blinks[i];
      if (!b.live || b.tail) continue;
      toggle(level, b);
    }
    for (int i = 0; i < kCells; ++i) lsum += level[i];
    mean_level = lsum / (float)kCells;

    // --- 7. The glow. The base (the mask, shaded by the glints' wakes), the
    //        tail's runs toggling against it, then the bounce's flash and the
    //        glint cores on top — additive, so they land on black as well as
    //        on a lit grid. Glints run along the gradient's own angle, toward
    //        +axis: the way the dark leaves, and the way One-Way brings the
    //        light back.
    for (int i = 0; i < kCells; ++i) glint_axis[i] = base_axis[i];
    const float bw = p.glint_width > 1e-3f ? p.glint_width : 1e-3f;
    const float flash = bounce > 0.0f ? bounce * kBounceFlash : 0.0f;
    const float bl = clamp01(p.base);
    const float gg = p.glint_gain > 0.0f ? p.glint_gain : 0.0f;
    const float gs = clamp01(p.glint_shadow);
    float cores[kCells];
    for (int i = 0; i < kCells; ++i) {
      float core = 0.0f, wake = 0.0f;
      for (int g = 0; g < three_planes_glints::kMaxLive; ++g) {
        const three_planes_glints::Glint& gl = glints.glints[g];
        if (!gl.live) continue;
        const float d = (glint_axis[i] - gl.pos) / (bw * gl.drawWidth());
        const float k = (d + kGlintWake) / kGlintWakeW;
        core += gg * gl.drawGain() * std::exp(-d * d * d * d);
        wake += gs * gl.drawShade() * std::exp(-k * k);
      }
      const float shade = wake < 1.0f ? 1.0f - wake : 0.0f;
      glow[i] = bl * level[i] * shade;
      cores[i] = flash + core;
    }
    for (int i = 0; i < kMaxBlinks; ++i) {
      const Blink& b = blinks[i];
      if (!b.live || !b.tail) continue;
      toggle(glow, b);
    }
    for (int i = 0; i < kCells; ++i) glow[i] += cores[i];
  }

 private:
  float rand01() {
    rng ^= rng << 13;
    rng ^= rng >> 17;
    rng ^= rng << 5;
    return (float)(rng & 0xFFFFFFu) * (1.0f / 16777216.0f);
  }

  bool blinking(int cell) const {
    for (int i = 0; i < kMaxBlinks; ++i)
      if (blinks[i].covers(cell)) return true;
    return false;
  }

  /// A blink toggles every segment it covers: a dark one comes up to full, a
  /// lit one goes out.
  static void toggle(float* v, const Blink& b) {
    for (int k = 0; k < b.len; ++k) {
      const int i = b.cell + k;
      if (i < 0 || i >= kCells) continue;
      const float lit = v[i];
      const float to = lit >= 0.5f ? 0.0f : 1.0f;
      v[i] = lit + (to - lit) * b.strength;
    }
  }

  /// A segment weighted toward where the light went out LAST (axis 1), and
  /// never one already mid-blink — two blinks on one segment are one blink.
  int pickTrailing(float bias) {
    float w[kCells];
    float sum = 0.0f;
    const float b = clamp01(bias);
    for (int i = 0; i < kCells; ++i) {
      w[i] = blinking(i) ? 0.0f : (1.0f + (axis[i] * axis[i] - 1.0f) * b);
      sum += w[i];
    }
    return pickWeighted(w, sum);
  }

  int pickWeighted(const float* w, float sum) {
    if (!(sum > 1e-6f)) return -1;
    float u = rand01() * sum;
    int lastNonZero = -1;
    for (int i = 0; i < kCells; ++i) {
      if (w[i] <= 0.0f) continue;
      lastNonZero = i;
      if (u < w[i]) return i;
      u -= w[i];
    }
    return lastNonZero;
  }

  /// A blink with no time left was over before the frame was sampled: it is
  /// not placed at all, which is how a closing window starts missing frames.
  void spawn(int cell, float t_left, float strength, int len = 1, bool tail = false) {
    if (!(t_left > 0.0f) || strength <= 0.0f) return;
    for (int i = 0; i < kMaxBlinks; ++i) {
      Blink& b = blinks[i];
      if (b.live && b.cell == cell && b.len == len && b.tail == tail) {   // extend rather than double up
        if (t_left > b.t_left) b.t_left = t_left;
        if (strength > b.strength) b.strength = strength;
        return;
      }
    }
    for (int i = 0; i < kMaxBlinks; ++i) {
      Blink& b = blinks[i];
      if (b.live) continue;
      b.live = true;
      b.cell = cell;
      b.len = len;
      b.tail = tail;
      b.t_left = t_left;
      b.strength = strength;
      return;
    }
    // Pool full: drop it. The tail is dense enough to lose one.
  }
};

/// Colour a frame for the LED map. Up to 1 a segment is `rgb * glow`; above
/// it the overdrive pushes toward white by `whiten`, which is what an
/// overdriven LED looks like.
inline led_bars::Cells toCells(const Core& c, const float rgb[3], float intensity,
                               float whiten) {
  led_bars::Cells out;
  const float in = intensity > 0.0f ? intensity : 0.0f;
  const float w = clamp01(whiten);
  for (int i = 0; i < kCells; ++i) {
    const float l = c.glow[i];
    const float base = l < 1.0f ? l : 1.0f;
    const float over = clamp01((l - 1.0f) / (kLevelMax - 1.0f));
    for (int k = 0; k < 3; ++k) {
      const float v = rgb[k] * base;
      out.rgb[i][k] = (v + (1.0f - v) * over * w) * in;
    }
  }
  return out;
}

/// The mask, gray: each segment's level on every channel.
inline led_bars::Cells maskCells(const Core& c) {
  led_bars::Cells out;
  for (int i = 0; i < kCells; ++i)
    for (int k = 0; k < 3; ++k) out.rgb[i][k] = clamp01(c.level[i]);
  return out;
}

}  // namespace sweep_choke_core
