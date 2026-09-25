#pragma once
/*
 * sweep_gradient_core.h — the logic behind `source.light.sweep_gradient`:
 * Sweep Choke's rolling gradient on its own, played by the knob OR fired by a
 * trigger.
 *
 * The gradient is copied from sweep_choke_core.h's mask, unchanged in
 * behaviour. The knob sets ONE level — full across the deadzone, easing to
 * `1 - depth` at either end — and the grid always settles to exactly that,
 * solid. Every change ROLLS across it along `angle_deg`, the near side first
 * and the far side `roll` seconds later, each segment easing on over `soften`.
 * Ping-Pong turns the roll round on the knob's lower half; One-Way keeps it.
 *
 * THE TRIGGER plays a sweep gesture: a virtual knob, out from home toward one
 * end (`reach` of the way) over `attack`, held for `hold`, and back home over
 * `release`. It goes through the same deadzone, depth and Ping-Pong rules as
 * the real knob, so a trigger is a sweep you did not have to perform.
 *
 * MONOPHONIC. While the envelope runs it owns the sweep and the knob is
 * ignored; when it ends the knob takes back over. Either handover is only a
 * change of level, so it rolls and softens like any other — nothing cuts. A
 * retrigger restarts the attack from wherever the envelope already is, so a
 * fast run of triggers never jumps.
 *
 * Host-free: no effect ABI, no GPU, so the Catch2 goldens drive it at an exact
 * dt.
 */

#include <cmath>

#include <sketch/led_bars.h>

namespace sweep_gradient_core {

using led_bars::kBars;
using led_bars::kSegments;
using led_bars::kCells;

/// A frame longer than this is a transport stall, not slow motion.
constexpr float kMaxDt = 0.25f;

/// The knob's rest position: full intensity, no motion.
constexpr float kSweepCenter = 0.5f;

/// The level's history, for the roll to read back into: one entry per tick.
constexpr int kHistory = 1024;
constexpr float kMaxRoll = 4.0f;

/// Ping-Pong mirrors the roll on the knob's lower half; One-Way keeps it.
enum SideMode { SidePingPong = 0, SideOneWay = 1 };

/// Which end a triggered gesture sweeps toward.
enum TrigSide { TrigUpper = 0, TrigLower = 1, TrigAlternate = 2, TrigRandom = 3 };

/// Read fresh each tick — a wire may be moving any of it.
struct Params {
  float sweep = kSweepCenter;
  float deadzone = 0.45f;   ///< fraction of each half that stays fully lit
  float depth = 1.0f;       ///< how far the ends fade; 1 = to black

  float angle_deg = 90.0f;  ///< which way changes roll, CCW from +x; 90 = upward
  int side_mode = SidePingPong;
  float roll = 0.35f;       ///< s for a change to cross the grid; 0 all at once
  float soften = 0.12f;     ///< s each segment eases toward what reaches it

  int trig_side = TrigAlternate;
  float reach = 1.0f;       ///< how far toward the end a trigger goes, 1 = the end
  float attack = 0.08f;     ///< s out
  float hold = 0.15f;       ///< s held there
  float release = 0.35f;    ///< s back home
};

inline float clampf(float v, float lo, float hi) { return v < lo ? lo : (v > hi ? hi : v); }
inline float clamp01(float v) { return clampf(v, 0.0f, 1.0f); }
inline float smooth01(float t) { t = clamp01(t); return t * t * (3.0f - 2.0f * t); }

/// Where a segment sits along the travel axis, 0 the edge a change reaches
/// FIRST and 1 the edge it reaches last, renormalised over the whole grid so
/// any angle uses the full range. Columns left to right across 0..1, segments
/// bottom to top across 0..1.
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

/// How far into the fade the sweep is: 0 anywhere in the deadzone, 1 at the end.
inline float fadeProgress(const Params& p, float sweep) {
  const float d = sweep - kSweepCenter;
  const float mag = clamp01((d < 0.0f ? -d : d) / kSweepCenter);
  const float dz = clamp01(p.deadzone);
  return clamp01((mag - dz) / (1.0f - dz > 1e-4f ? 1.0f - dz : 1e-4f));
}

/// The level the sweep asks for: full across the deadzone, easing to
/// `1 - depth` at either end.
inline float positionLevel(const Params& p, float sweep) {
  return clamp01(1.0f - smooth01(fadeProgress(p, sweep)) * clamp01(p.depth));
}

struct Core {
  // --- The envelope --------------------------------------------------------
  enum Stage { Idle = 0, Attack, Hold, Release };
  int stage = Idle;
  float env = 0.0f;         ///< 0..1, how far out the gesture is
  float stage_t = 0.0f;     ///< s into the current stage
  float side = 1.0f;        ///< +1 toward the upper end, -1 the lower
  int fires = 0;            ///< counts fires, for Alternate
  bool pending = false;     ///< a fire waiting for the next tick
  unsigned rng = 0x9e3779b9u;

  // --- The gradient --------------------------------------------------------
  float hist_t[kHistory] = {};
  float hist_v[kHistory] = {};
  int hist_head = 0;
  int hist_count = 0;
  float clock = 0.0f;
  bool seeded = false;

  // --- This frame ----------------------------------------------------------
  float level[kCells] = {};  ///< per segment, 0..1
  float axis[kCells] = {};   ///< this frame's axis, already mirrored for the side
  float mean_level = 1.0f;
  float sweep_out = kSweepCenter;   ///< the effective sweep: knob or gesture

  Core() { for (int i = 0; i < kCells; ++i) level[i] = 1.0f; }

  void reset() { *this = Core(); }

  bool active() const { return stage != Idle; }

  /// Start the gesture, or restart it from where it already is. Lands on the
  /// next tick, so a fire between frames is never lost to a zero-dt tick.
  void fire() { pending = true; }

  void tick(const Params& p, float dt) {
    if (!(dt > 0.0f)) dt = 0.0f;
    if (dt > kMaxDt) dt = kMaxDt;

    // --- 1. The envelope. A fire re-picks the side and climbs from the
    //        current value, so a retrigger mid-gesture never jumps.
    if (pending && dt > 0.0f) {
      pending = false;
      side = pickSide(p.trig_side);
      ++fires;
      stage = Attack;
      stage_t = 0.0f;
    }
    if (dt > 0.0f) advance(p, dt);

    // A NaN patch holds the middle rather than poisoning everything below.
    const float knob = (p.sweep == p.sweep) ? clamp01(p.sweep) : kSweepCenter;
    const float reach = clamp01(p.reach);
    const float sw = active() ? clamp01(kSweepCenter + side * reach * kSweepCenter * env) : knob;
    sweep_out = sw;

    // --- 2. The level the sweep asks for, and the axis it rolls along on
    //        this side of the knob.
    axisCoords(p.angle_deg, axis);
    if (p.side_mode == SidePingPong && sw < kSweepCenter)
      for (int i = 0; i < kCells; ++i) axis[i] = 1.0f - axis[i];
    const float lvl = positionLevel(p, sw);

    // --- 3. The gradient. The level goes into the history; each segment
    //        reads it back `axis * roll` seconds ago and eases toward that
    //        over `soften`.
    if (!seeded) {
      // A card that comes up parked shows a gradient that settled long ago.
      seeded = true;
      clock = 0.0f;
      hist_head = 0;
      hist_count = 1;
      hist_t[0] = 0.0f;
      hist_v[0] = lvl;
      for (int i = 0; i < kCells; ++i) level[i] = lvl;
    } else if (dt > 0.0f) {
      clock += dt;
      hist_head = (hist_head + 1) % kHistory;
      hist_t[hist_head] = clock;
      hist_v[hist_head] = lvl;
      if (hist_count < kHistory) ++hist_count;
      const float rl = clampf(p.roll, 0.0f, kMaxRoll);
      const float ease = p.soften > 1e-3f ? 1.0f - std::exp(-dt / p.soften) : 1.0f;
      for (int i = 0; i < kCells; ++i) {
        const float want = history(clock - axis[i] * rl);
        level[i] += (want - level[i]) * ease;
      }
    }
    float lsum = 0.0f;
    for (int i = 0; i < kCells; ++i) lsum += level[i];
    mean_level = lsum / (float)kCells;
  }

 private:
  /// Attack climbs at 1/attack per second from wherever it is, Hold sits at
  /// 1, Release falls at 1/release. Time left over at a stage boundary carries
  /// into the next, so a long frame never stalls the envelope.
  void advance(const Params& p, float dt) {
    float t = dt;
    for (int guard = 0; t > 0.0f && stage != Idle && guard < 8; ++guard) {
      if (stage == Attack) {
        const float a = p.attack > 1e-4f ? p.attack : 1e-4f;
        const float need = (1.0f - env) * a;
        if (t < need) { env += t / a; t = 0.0f; }
        else { t -= need; env = 1.0f; stage = Hold; stage_t = 0.0f; }
      } else if (stage == Hold) {
        const float h = p.hold > 0.0f ? p.hold : 0.0f;
        const float need = h - stage_t;
        if (t < need) { stage_t += t; t = 0.0f; }
        else { t -= need > 0.0f ? need : 0.0f; stage = Release; stage_t = 0.0f; }
      } else {   // Release
        const float r = p.release > 1e-4f ? p.release : 1e-4f;
        const float need = env * r;
        if (t < need) { env -= t / r; t = 0.0f; }
        else { t -= need; env = 0.0f; stage = Idle; stage_t = 0.0f; }
      }
    }
  }

  float pickSide(int mode) {
    switch (mode) {
      case TrigLower: return -1.0f;
      case TrigAlternate: return (fires % 2 == 0) ? 1.0f : -1.0f;
      case TrigRandom: return rand01() < 0.5f ? 1.0f : -1.0f;
      default: return 1.0f;
    }
  }

  /// The level as it stood at clock time `t`, interpolated between ticks.
  float history(float t) const {
    int i = hist_head;
    for (int n = 0; n + 1 < hist_count; ++n) {
      const int j = (i - 1 + kHistory) % kHistory;
      if (hist_t[j] <= t) {
        const float span = hist_t[i] - hist_t[j];
        const float u = span > 1e-9f ? clamp01((t - hist_t[j]) / span) : 1.0f;
        return hist_v[j] + (hist_v[i] - hist_v[j]) * u;
      }
      i = j;
    }
    return hist_v[i];
  }

  float rand01() {
    rng ^= rng << 13;
    rng ^= rng >> 17;
    rng ^= rng << 5;
    return (float)(rng & 0xFFFFFFu) * (1.0f / 16777216.0f);
  }
};

/// Colour a frame for the LED map: the colour at each segment's level.
inline led_bars::Cells toCells(const Core& c, const float rgb[3], float intensity) {
  led_bars::Cells out;
  const float in = intensity > 0.0f ? intensity : 0.0f;
  for (int i = 0; i < kCells; ++i) {
    const float l = clamp01(c.level[i]);
    for (int k = 0; k < 3; ++k) out.rgb[i][k] = rgb[k] * l * in;
  }
  return out;
}

}  // namespace sweep_gradient_core
