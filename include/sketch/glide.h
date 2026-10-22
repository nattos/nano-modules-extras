#pragma once
/*
 * glide.h — legato between discrete states. Something that would JUMP (a
 * layout, a preset, a pose) glides there instead, on a stiff spring that
 * visibly overshoots and rings out, optionally roughed up into a glitch.
 *
 * Each element carries an OFFSET from where it is going, never the goal:
 *
 *     shown = goal + offset
 *
 * The goal is the caller's and stays live (it may keep moving — breathing,
 * spinning — while the offset rings out over it). When the goal jumps, the
 * caller hands over the new offset (what is shown now, measured from the new
 * goal) with retargetTo(), or just the jump with retarget(); the velocity is
 * KEPT, so a change mid-glide carries on from where it is and how fast it is
 * going (legato) instead of starting over.
 *
 * THE SPRING is a damped oscillator stepped exactly — its closed form over
 * dt, so any frame rate gives the same curve and no stiffness is unstable —
 * set by two numbers a performer reads:
 *
 *   time       s: the ringing's envelope has fallen to kSettle (2%) by then.
 *              Default 0.4, about a beat.
 *   stiffness  Hz: how fast it rings. At the default 10 the first overshoot
 *              peaks ~50 ms in (60% past the goal) and the first full swing
 *              back is done by ~100 ms. Below ln(1/kSettle) / (2π time) Hz it
 *              no longer overshoots at all and creeps in instead.
 *
 * SMOOTHING (`smooth`, s) calms all of it at once: what is shown follows the
 * spring (and its glitch) through a critically damped follower that catches
 * up in about that long — the overshoot rounds off and shrinks, the ringing
 * dies sooner, a stutter's jump eases on. 0 is off, exactly.
 *
 * THE GLITCH roughs it up per element, deterministically from its seed
 * (Glitch::from maps one 0..1 knob onto all of it):
 *
 *   stagger  each element waits a moment before it lets go, frozen where it
 *            was, so a change ripples through a set instead of moving as one;
 *   skew     within an element, each channel waits its own moment: a block
 *            moves along x, then y, then grows — square, stepped paths;
 *   detune   each element rings at its own rate and decay;
 *   stutter  while it is still energetic, it now and then FREEZES what it
 *            shows for a few frames and then jumps to where it really is:
 *            dropped frames, element by element.
 *
 * Two ways to move it besides a retarget, neither of which starts anything
 * over: carry() puts some of a goal's move through the spring (a value driven
 * every frame twangs after it), and kick() knocks it (jitter: random knocks
 * keep it ringing).
 *
 * And matchNearest(): the greedy pairing a morph between two SETS needs (which
 * old element becomes which new one, which are left over on either side).
 *
 * Host-free: no effect ABI, no GPU, no allocation.
 */

#include <cmath>
#include <cstdint>

namespace glide {

/// What is left of the ringing's envelope at `time`.
constexpr float kSettle = 0.02f;
constexpr float kTau = 6.2831853f;

struct Params {
  float time = 0.4f;        ///< s until the envelope is down to kSettle (0 = snap)
  float stiffness = 10.0f;  ///< Hz, how fast it rings
  float smooth = 0.0f;      ///< s the shown value takes to catch up (0 = off)
};

struct Glitch {
  float stagger = 0.0f;  ///< s, the longest an element waits before letting go
  float skew = 0.0f;     ///< s, the longest one channel lags the element's start
  float detune = 0.0f;   ///< ± fraction each element's rate and decay wander by
  float stutter = 0.0f;  ///< 0..1, the chance an energetic stretch freezes
  float hold = 0.05f;    ///< s, a stutter's (and a free stretch's) mean length

  /// One knob: 0 is a clean spring, 1 a thoroughly broken one. The waits
  /// scale with the glide's time, so a longer glide glitches as long.
  static Glitch from(float g, const Params& p) {
    g = g < 0.0f ? 0.0f : (g > 1.0f ? 1.0f : g);
    Glitch q;
    q.stagger = 0.25f * p.time * g;
    q.skew = 0.15f * p.time * g;
    q.detune = 0.3f * g;
    q.stutter = 0.85f * g;
    q.hold = 0.035f + 0.03f * g;
    return q;
  }
};

/// [0, 1) from two words.
inline float hash01(uint32_t a, uint32_t b) {
  uint32_t h = a * 0x9E3779B9u ^ (b + 0x7F4A7C15u) * 0x85EBCA6Bu;
  h ^= h >> 16; h *= 0x7FEB352Du;
  h ^= h >> 15; h *= 0x846CA68Bu;
  h ^= h >> 16;
  return (float)(h & 0xFFFFFFu) * (1.0f / 16777216.0f);
}

/// The envelope's decay rate (1/s) for a glide of `time`.
inline float decayOf(float time) {
  return std::log(1.0f / kSettle) / (time > 1e-4f ? time : 1e-4f);
}

/// The exact step of e'' + 2σe' + ω²e = 0 over dt, as a matrix on (e, v).
/// Under-, critically and over-damped alike.
struct Step {
  float ee = 1.0f, ev = 0.0f, ve = 0.0f, vv = 1.0f;

  static Step of(float sigma, float omega, float dt) {
    Step s;
    if (!(dt > 0.0f)) return s;
    const float k2 = omega * omega - sigma * sigma;
    float c, sw;   // cos(w dt) and sin(w dt) / w, or their hyperbolic twins
    if (k2 > 1e-6f) {
      const float w = std::sqrt(k2);
      c = std::cos(w * dt);
      sw = std::sin(w * dt) / w;
    } else if (k2 < -1e-6f) {
      const float w = std::sqrt(-k2);
      c = std::cosh(w * dt);
      sw = std::sinh(w * dt) / w;
    } else {
      c = 1.0f;
      sw = dt;
    }
    const float E = std::exp(-sigma * dt);
    s.ee = E * (c + sigma * sw);
    s.ev = E * sw;
    s.ve = -E * omega * omega * sw;
    s.vv = E * (c - sigma * sw);
    return s;
  }
};

/// One element gliding over N channels (any units; each channel is its own
/// spring, sharing the element's rate, decay and stutter).
template <int N>
struct Element {
  float e[N] = {};      ///< offset from the goal
  float v[N] = {};      ///< its velocity
  float held[N] = {};   ///< e, unless a stutter holds it
  float out[N] = {};    ///< what it shows: held, smoothed
  float sv[N] = {};     ///< the smoothing follower's velocity
  float wait[N] = {};   ///< s each channel still waits before it lets go
  float age = 1e9f;     ///< s since it was last retargeted
  float rate = 1.0f;    ///< detune: × stiffness
  float damp = 1.0f;    ///< detune: × decay
  float seg = 0.0f;     ///< s left in this stutter stretch
  bool frozen = false;  ///< the stretch is a stutter
  uint32_t seed = 0;
  uint32_t draws = 0;

  float rnd() { return hash01(seed, ++draws); }

  /// No glide: it shows the goal.
  void snap() {
    for (int c = 0; c < N; c++) e[c] = v[c] = held[c] = out[c] = sv[c] = wait[c] = 0.0f;
    age = 1e9f;
    frozen = false;
    seg = 0.0f;
  }

  /// The goal moved: `offset` is what it shows now, measured from the new
  /// goal. Its velocity (and the smoothing's) carries on, and so does a glide
  /// already under way: only an element at rest waits (stagger, skew) before
  /// it lets go — one still waiting keeps what is left of its wait, one
  /// already moving keeps moving. (Else a goal that moves every frame would
  /// hold it still for good.)
  void retargetTo(const float* offset, const Glitch& g) {
    const bool rest = !active(1e-4f);
    const float st = g.stagger * rnd();
    for (int c = 0; c < N; c++) {
      e[c] = held[c] = out[c] = offset[c];
      const float w = st + g.skew * rnd();
      if (rest) wait[c] = w;
    }
    age = 0.0f;
    rate = 1.0f + g.detune * (2.0f * rnd() - 1.0f);
    damp = 1.0f + g.detune * (2.0f * rnd() - 1.0f);
    frozen = false;
    seg = 0.0f;
  }

  /// The goal jumped by `jump` (new goal - old goal, per channel).
  void retarget(const float* jump, const Glitch& g) {
    float o[N];
    for (int c = 0; c < N; c++) o[c] = out[c] - jump[c];
    retargetTo(o, g);
  }

  /// The goal moved by `jump`, and `k` of that move goes through the spring:
  /// what is shown holds back by k x the jump and springs after it (k = 1: it
  /// stays put, then twangs over; 0: it moves with the goal). Unlike a
  /// retarget nothing starts over — no waits, no glitch, the velocity carries
  /// on — so a goal DRIVEN every frame (a modulated value) can be carried
  /// every frame: a slow drive just lags a little, a sudden one twangs.
  void carry(const float* jump, float k) {
    for (int c = 0; c < N; c++) {
      const float d = k * jump[c];
      e[c] -= d;
      held[c] -= d;
      out[c] -= d;
    }
  }

  /// A knock: `dv` added to the velocity (channel units / s). It rings out
  /// like any glide; at dv = A x 2π stiffness the first swing reaches nearly A
  /// (0.8 A at the default spring).
  void kick(const float* dv) {
    for (int c = 0; c < N; c++) v[c] += dv[c];
  }

  void step(float dt, const Params& p, const Glitch& g) {
    if (!(dt > 0.0f)) return;
    age += dt;
    if (p.time <= 1e-4f) {
      snap();
      return;
    }
    const float sigma = decayOf(p.time) * damp;
    const float omega = kTau * (p.stiffness > 0.0f ? p.stiffness : 0.0f) * rate;
    const Step full = Step::of(sigma, omega, dt);
    for (int c = 0; c < N; c++) {
      Step st = full;
      if (wait[c] > 0.0f) {
        if (wait[c] >= dt) { wait[c] -= dt; continue; }
        st = Step::of(sigma, omega, dt - wait[c]);
        wait[c] = 0.0f;
      }
      const float e1 = st.ee * e[c] + st.ev * v[c];
      v[c] = st.ve * e[c] + st.vv * v[c];
      e[c] = e1;
    }
    // Stutter: stretches of free and frozen, the frozen ones likelier the
    // more energy is left (none once it is past the first ~60% of its time).
    if (g.stutter > 0.0f) {
      const float energy = 1.0f - age / (0.6f * p.time);
      seg -= dt;
      if (seg <= 0.0f) {
        frozen = energy > 0.0f && rnd() < g.stutter * energy;
        seg = g.hold * (0.5f + rnd());
      }
      if (energy <= 0.0f) frozen = false;   // the tail always rings out clean
    } else {
      frozen = false;
    }
    if (!frozen)
      for (int c = 0; c < N; c++) held[c] = e[c];
    // Smoothing: a critically damped follower of what is held, over dt.
    if (p.smooth > 1e-4f) {
      const float w = 2.0f / p.smooth;
      const Step f = Step::of(w, w, dt);
      for (int c = 0; c < N; c++) {
        const float d = out[c] - held[c];
        out[c] = held[c] + f.ee * d + f.ev * sv[c];
        sv[c] = f.ve * d + f.vv * sv[c];
      }
    } else {
      for (int c = 0; c < N; c++) { out[c] = held[c]; sv[c] = 0.0f; }
    }
  }

  /// Is it still moving (or waiting to)? `eps` in the channels' units, and
  /// that much speed over a tenth of a second.
  bool active(float eps) const {
    for (int c = 0; c < N; c++)
      if (wait[c] > 0.0f || std::fabs(e[c]) > eps || std::fabs(v[c]) * 0.1f > eps ||
          std::fabs(out[c]) > eps || std::fabs(sv[c]) * 0.1f > eps)
        return true;
    return false;
  }
};

/// Greedy nearest pairing of `n` sources with `m` targets: it takes the
/// cheapest remaining pair until one side runs out. `cost(i, j)` is any
/// distance-like number; INFINITY forbids a pair. Writes src_to[i] (its
/// target, or -1: left over) and dst_from[j] (its source, or -1). `best` and
/// `bestc` are scratch, n each.
template <class Cost>
void matchNearest(int n, int m, Cost cost, int* src_to, int* dst_from, int* best,
                  float* bestc) {
  for (int i = 0; i < n; i++) src_to[i] = -1;
  for (int j = 0; j < m; j++) dst_from[j] = -1;
  auto refresh = [&](int i) {
    best[i] = -1;
    bestc[i] = INFINITY;
    for (int j = 0; j < m; j++) {
      if (dst_from[j] >= 0) continue;
      const float c = cost(i, j);
      if (c < bestc[i]) { bestc[i] = c; best[i] = j; }
    }
  };
  for (int i = 0; i < n; i++) refresh(i);
  for (;;) {
    int pick = -1;
    for (int i = 0; i < n; i++)
      if (src_to[i] < 0 && best[i] >= 0 && (pick < 0 || bestc[i] < bestc[pick])) pick = i;
    if (pick < 0) return;
    const int j = best[pick];
    src_to[pick] = j;
    dst_from[j] = pick;
    for (int i = 0; i < n; i++)
      if (src_to[i] < 0 && best[i] == j) refresh(i);
  }
}

}  // namespace glide
