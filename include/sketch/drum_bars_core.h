#pragma once
/*
 * drum_bars_core.h — the logic behind `source.light.drum_bars`: three drum
 * roles lighting the four LED bars.
 *
 *   ch1  the KICK. Lights a mirror-symmetric pattern — the outer pair, the
 *        inner pair, or all four — at full, decaying over `kick_decay`.
 *   ch2  the BACKBEAT. Lights the CONVERSE of what ch1 would light right now:
 *        outer <-> inner, and all four -> all four again but `under` as bright.
 *        Then it flams: weaker repeats a `flam_gap` apart, each shifted a ring
 *        OUTWARD, so the hit ripples out to the outer bars as it dies.
 *   ch3  DECORATION. One random bar (never the same one twice running) plays
 *        a short staccato strobe burst, each pulse at its own intensity.
 *
 * THE PATTERNS CYCLE. Which pattern a hit fires is a deterministic function
 * of the Cycle position `c` — smooth periodic value noise, cut into crisp
 * choices. Nearby positions give the same pattern, the same position always
 * gives the same one, and c = 0 and c = 1 are the same place, so a ramp or an
 * LFO on Cycle loops without a seam. `drift` walks c on its own.
 *
 * Where channels overlap on a bar they combine XOR-style — fuzzy, a + b - 2ab —
 * so a strobe over a lit kick flickers it DARK rather than adding nothing.
 *
 * Each bar is one level: the LED map shows it on all ten segments.
 *
 * Host-free: no effect ABI, no GPU, no transport. The groove's step clock
 * lives in the effect; the core only says what fires on a given 16th, so the
 * Catch2 goldens drive everything at an exact dt.
 */

#include <cmath>
#include <cstdint>

#include <sketch/led_bars.h>

namespace drum_bars_core {

using led_bars::kBars;
using led_bars::kSegments;
using led_bars::kCells;

/// A frame longer than this is a transport stall, not slow motion.
constexpr float kMaxDt = 0.25f;

constexpr int kChannels = 3;
constexpr int kMaxFlams = 3;
constexpr int kMaxPulses = 6;
constexpr int kPending = 16;

/// What the kick lights. The backbeat lights the converse.
enum Pattern { PatOuter = 0, PatInner = 1, PatAll = 2 };

enum Combine { CombineXor = 0, CombineMax = 1, CombineAdd = 2 };

/// Read fresh each tick — a wire may be moving any of it.
struct Params {
  float cycle = 0.0f;       ///< position in the pattern loop; wraps at 1
  float drift = 0.0f;       ///< cycles per second added on top
  float variety = 4.0f;     ///< noise features per loop: how often picks change
  int seed = 0;

  float kick_decay = 0.35f; ///< s

  float back_level = 0.85f;
  float under = 0.5f;       ///< backbeat level under an all-four kick
  float back_decay = 0.2f;  ///< s
  float flam = 0.6f;        ///< 0 none .. 1 up to three repeats
  float flam_gap = 0.05f;   ///< s between repeats
  float flam_fade = 0.5f;   ///< each repeat this much of the last

  float deco_min = 0.3f;    ///< quietest a strobe pulse gets
  float staccato = 0.35f;   ///< fraction of each pulse period that is lit
  float deco_rate = 1.0f;   ///< strobe speed multiplier

  int combine = CombineXor;
  float auto_deco = 0.2f;   ///< groove: chance of a ch3 hit on each 16th
};

/// NaN reads as `lo`: a dead wire should go dark, not poison the frame.
inline float clampf(float v, float lo, float hi) { return !(v >= lo) ? lo : (v > hi ? hi : v); }
inline float clamp01(float v) { return clampf(v, 0.0f, 1.0f); }
inline float smooth01(float t) { t = clamp01(t); return t * t * (3.0f - 2.0f * t); }
inline float wrap01(float x) {
  if (!std::isfinite(x)) return 0.0f;
  x -= std::floor(x);
  return x >= 1.0f ? 0.0f : x;
}

inline uint32_t hash32(uint32_t x) {
  x ^= x >> 16; x *= 0x7feb352du;
  x ^= x >> 15; x *= 0x846ca68bu;
  x ^= x >> 16;
  return x;
}

inline float hash01(int seed, int stream, int i) {
  const uint32_t h = hash32((uint32_t)seed * 0x9e3779b9u ^ hash32((uint32_t)stream * 0x85ebca6bu + (uint32_t)i));
  return (float)(h >> 8) * (1.0f / 16777216.0f);
}

/// Lattice point `i` of `n` round the loop. STRATIFIED: the n points are a
/// shuffle of n evenly spaced levels (each jittered within its slot), so every
/// loop spans the whole range — a pick cut in thirds meets all three as soon
/// as there are three points, rather than whenever the dice allow.
inline float lattice(int seed, int stream, int i, int n) {
  const float hi = hash01(seed, stream, i);
  int rank = 0;
  for (int j = 0; j < n; ++j) {
    const float hj = hash01(seed, stream, j);
    if (hj < hi || (hj == hi && j < i)) ++rank;
  }
  const float jitter = hash01(seed, stream + 0x100, i) - 0.5f;
  return ((float)rank + 0.5f + 0.8f * jitter) / (float)n;
}

/// Smooth value noise, periodic on [0, 1): `features` lattice points round a
/// ring, eased between. 0..1. Each decision reads its own `stream`.
inline float loopNoise(float x, int seed, int stream, float features) {
  const int n = (int)clampf(std::isfinite(features) ? std::round(features) : 4.0f, 1.0f, 32.0f);
  const float u = wrap01(x) * (float)n;
  int i = (int)std::floor(u);
  if (i >= n) i = n - 1;
  const float f = u - (float)i;
  const float a = lattice(seed, stream, i, n);
  const float b = lattice(seed, stream, (i + 1) % n, n);
  return a + (b - a) * smooth01(f);
}

enum Stream { StreamKick = 1, StreamFlam = 2, StreamCount = 3, StreamPeriod = 4, StreamPickup = 5 };

/// The kick's pattern at Cycle position `c`: the noise cut in thirds.
inline int kickPattern(const Params& p, float c) {
  const float n = loopNoise(c, p.seed, StreamKick, p.variety);
  return n < 1.0f / 3.0f ? PatOuter : (n < 2.0f / 3.0f ? PatInner : PatAll);
}

/// The backbeat's pattern: the converse of the kick's at the same place.
inline int backPattern(const Params& p, float c) {
  const int k = kickPattern(p, c);
  return k == PatOuter ? PatInner : (k == PatInner ? PatOuter : PatAll);
}

/// How many flam repeats a backbeat plays at `c`.
inline int flamCount(const Params& p, float c) {
  const float n = loopNoise(c, p.seed, StreamFlam, p.variety);
  const int k = (int)std::floor(clamp01(p.flam) * (1.0f + 3.0f * n));
  return k < 0 ? 0 : (k > kMaxFlams ? kMaxFlams : k);
}

/// A strobe burst's pulse count (2..6) and period (s) at `c`.
inline int strobeCount(const Params& p, float c) {
  const int k = 2 + (int)std::floor(loopNoise(c, p.seed, StreamCount, p.variety) * 5.0f);
  return k > kMaxPulses ? kMaxPulses : k;
}
inline float strobePeriod(const Params& p, float c) {
  const float rate = std::isfinite(p.deco_rate) ? clampf(p.deco_rate, 0.1f, 8.0f) : 1.0f;
  return (0.035f + 0.04f * loopNoise(c, p.seed, StreamPeriod, p.variety)) / rate;
}

/// Whether bar `bar` is in pattern `pat`. Ring 1 is the outer pair.
inline bool inPattern(int pat, int bar) {
  if (pat == PatAll) return true;
  const int ring = led_bars::barRing(bar);
  return pat == PatOuter ? ring == 1 : ring == 0;
}

/// Groove: what fires on 16th `step` (0..15) of the bar at `c`. Bit n = ch n+1.
/// ch3 is left to the caller's dice (Core::groove) — this is the fixed part.
inline int grooveHits(const Params& p, int step, float c) {
  step = ((step % 16) + 16) % 16;
  int bits = 0;
  if (step == 0 || step == 8) bits |= 1;
  if (step == 4 || step == 12) bits |= 2;
  // A pickup kick at some places in the loop, late in the bar.
  const float n = loopNoise(c, p.seed, StreamPickup, p.variety);
  if (n > 0.55f && step == (n > 0.8f ? 14 : 10)) bits |= 1;
  return bits;
}

struct Hit {
  float a = 0.0f, t = 0.0f, decay = 1.0f;
  float value() const {
    if (a <= 0.0f || t >= decay) return 0.0f;
    const float u = 1.0f - t / decay;
    return a * u * u;
  }
};

struct Pend {
  float delay = 0.0f, a = 0.0f;
  int mask = 0;   ///< bit per bar
  bool live = false;
};

struct Deco {
  bool on = false;
  float t = 0.0f, period = 0.05f;
  int count = 0;
  float amp[kMaxPulses] = {};
  float value(float duty) const {
    if (!on || period <= 0.0f) return 0.0f;
    const float u = t / period;
    const int k = (int)std::floor(u);
    if (k < 0 || k >= count) return 0.0f;
    return (u - (float)k) < duty ? amp[k] : 0.0f;
  }
};

struct Core {
  Hit kick[kBars], back[kBars];
  Pend pend[kPending];
  Deco deco[kBars];
  int queued[kChannels] = {};
  int last_deco = -1;
  float drift_phase = 0.0f;
  uint32_t rng = 0x9e3779b9u;
  int fires[kChannels] = {};

  float level[kBars] = {};
  float mean = 0.0f;
  float cycle_out = 0.0f;

  void reset() { *this = Core(); }

  /// Queue a hit on channel 0..2; it lands on the next tick with dt > 0.
  void fire(int ch) {
    if (ch >= 0 && ch < kChannels) ++queued[ch];
  }

  float position(const Params& p) const {
    const float cyc = std::isfinite(p.cycle) ? p.cycle : 0.0f;
    return wrap01(cyc + drift_phase);
  }

  /// Fire what the groove plays on 16th `step`. Returns the bits fired.
  int groove(const Params& p, int step) {
    const float c = position(p);
    int bits = grooveHits(p, step, c);
    if (rand01() < clamp01(p.auto_deco)) bits |= 4;
    for (int ch = 0; ch < kChannels; ++ch)
      if (bits & (1 << ch)) fire(ch);
    return bits;
  }

  void tick(const Params& p, float dt) {
    if (!(dt > 0.0f)) { compose(p); return; }
    if (dt > kMaxDt) dt = kMaxDt;
    const float c = position(p);

    // Age what is already sounding first, so a fresh hit shows at full.
    for (int b = 0; b < kBars; ++b) {
      kick[b].t += dt;
      back[b].t += dt;
      if (deco[b].on) {
        deco[b].t += dt;
        if (deco[b].t >= deco[b].period * (float)deco[b].count) deco[b].on = false;
      }
    }
    for (Pend& q : pend) {
      if (!q.live) continue;
      q.delay -= dt;
      if (q.delay <= 0.0f) {
        for (int b = 0; b < kBars; ++b)
          if (q.mask & (1 << b)) strike(back[b], q.a, p.back_decay, -q.delay);
        q.live = false;
      }
    }

    for (; queued[0] > 0; --queued[0]) fireKick(p, c);
    for (; queued[1] > 0; --queued[1]) fireBack(p, c);
    for (; queued[2] > 0; --queued[2]) fireDeco(p, c);

    const float dr = std::isfinite(p.drift) ? p.drift : 0.0f;
    drift_phase = wrap01(drift_phase + dr * dt);
    compose(p);
  }

  void compose(const Params& p) {
    const float duty = clamp01(p.staccato);
    float sum = 0.0f;
    for (int b = 0; b < kBars; ++b) {
      const float v[kChannels] = {clamp01(kick[b].value()), clamp01(back[b].value()),
                                  clamp01(deco[b].value(duty))};
      float x = v[0];
      for (int k = 1; k < kChannels; ++k) {
        if (p.combine == CombineMax) x = x > v[k] ? x : v[k];
        else if (p.combine == CombineAdd) x = clamp01(x + v[k]);
        else x = x + v[k] - 2.0f * x * v[k];
      }
      level[b] = clamp01(x);
      sum += level[b];
    }
    mean = sum / kBars;
    cycle_out = position(p);
  }

  float rand01() {
    rng ^= rng << 13;
    rng ^= rng >> 17;
    rng ^= rng << 5;
    return (float)(rng & 0xFFFFFFu) * (1.0f / 16777216.0f);
  }

 private:
  static void strike(Hit& h, float a, float decay, float age) {
    Hit n;
    n.a = a;
    n.decay = decay > 1e-3f ? decay : 1e-3f;
    n.t = age > 0.0f ? age : 0.0f;
    if (n.value() >= h.value()) h = n;
  }

  void fireKick(const Params& p, float c) {
    ++fires[0];
    const int pat = kickPattern(p, c);
    for (int b = 0; b < kBars; ++b)
      if (inPattern(pat, b)) strike(kick[b], 1.0f, p.kick_decay, 0.0f);
  }

  void fireBack(const Params& p, float c) {
    ++fires[1];
    const int pat = backPattern(p, c);
    const float a = clamp01(p.back_level) * (pat == PatAll ? clamp01(p.under) : 1.0f);
    for (int b = 0; b < kBars; ++b)
      if (inPattern(pat, b)) strike(back[b], a, p.back_decay, 0.0f);
    // The flams ripple outward: every repeat is a ring further out, and with
    // two rings that is the outer pair — inner goes out, outer stays out.
    int outer = 0;
    for (int b = 0; b < kBars; ++b)
      if (led_bars::barRing(b) == 1) outer |= 1 << b;
    const int n = flamCount(p, c);
    const float gap = clampf(std::isfinite(p.flam_gap) ? p.flam_gap : 0.05f, 0.005f, 1.0f);
    float amp = a;
    for (int k = 1; k <= n; ++k) {
      amp *= clamp01(p.flam_fade);
      for (Pend& q : pend) {
        if (q.live) continue;
        q.live = true;
        q.delay = gap * (float)k;
        q.a = amp;
        q.mask = outer;
        break;
      }
    }
  }

  void fireDeco(const Params& p, float c) {
    ++fires[2];
    int bar = (int)(rand01() * (last_deco < 0 ? kBars : kBars - 1));
    if (bar >= kBars) bar = kBars - 1;
    if (last_deco >= 0 && bar >= last_deco) ++bar;   // skip the last one
    if (bar >= kBars) bar = kBars - 1;
    last_deco = bar;
    Deco& d = deco[bar];
    d.on = true;
    d.t = 0.0f;
    d.count = strobeCount(p, c);
    d.period = strobePeriod(p, c);
    const float lo = clamp01(p.deco_min);
    for (int k = 0; k < kMaxPulses; ++k) d.amp[k] = lo + (1.0f - lo) * rand01();
  }
};

/// Colour a frame for the LED map: each bar one level, on every segment.
inline led_bars::Cells toCells(const Core& c, const float rgb[3], float intensity) {
  led_bars::Cells out;
  const float in = intensity > 0.0f ? intensity : 0.0f;
  for (int b = 0; b < kBars; ++b)
    for (int s = 0; s < kSegments; ++s)
      for (int k = 0; k < 3; ++k)
        out.rgb[led_bars::cellIndex(b, s)][k] = rgb[k] * clamp01(c.level[b]) * in;
  return out;
}

}  // namespace drum_bars_core
