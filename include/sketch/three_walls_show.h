#pragma once
/*
 * three_walls_show.h — the moves behind `source.mesh.three_walls`.
 *
 * three_walls is a tunnel: three neon quads rushing at the viewer. Where
 * three_planes is a level you read, this is a hit you feel, so all four moves
 * are about ARRIVAL — how fast, how many, and whether they beat against each
 * other. Kept host-free so the wasm effect and a Catch2 golden run
 * byte-identical code at an exact dt with no GPU and no ABI, the same
 * arrangement as three_planes_rig.h / envelope.h / param_smoothing.h.
 *
 * DEPTH IS A PHASE, NOT A DISTANCE. Each quad carries phase in [0,1) and the
 * depth comes out as z = z_far * (z_near/z_far)^phase — geometric, not linear.
 * Two things fall out of that: apparent size grows at a constant rate, so the
 * approach reads as constant speed rather than a rush at the end; and the wrap
 * from 1 back to 0 is seamless, which is what makes the cycling moves loop
 * without a seam to hide.
 *
 * ONE MOVE AT A TIME. Pulse is a one-shot that plays out; the other three are
 * GATES that run while held. A new press drops whatever was running, wherever
 * it stood — the same monophonic rule the three_planes rig's camera moves use,
 * and for the same reason: two of these at once is noise, not counterpoint.
 */

#include <cmath>

namespace three_walls_show {

constexpr int kQuads = 3;

/// A frame longer than this is a transport stall, not slow motion. Same clamp,
/// and the same reasoning, as three_planes_rig.h.
constexpr float kMaxDt = 0.25f;

/// The band the frame-locked moves will believe. Outside it the dt is a stall
/// or a debug single-step, not a frame rate.
constexpr float kMinFps = 15.0f;
constexpr float kMaxFps = 240.0f;

/// How fast the frame-time estimate follows. ~50 frames of memory: slow enough
/// that per-frame jitter barely moves the step (a brutal 8ms/28ms alternation
/// leaves it inside 4%, where integrating dt directly would swing 110%), fast
/// enough to settle on a genuinely new frame rate inside a second.
constexpr float kFpsSmoothing = 0.02f;

/// Which move is running. Pulse retires on its own; the rest run until release.
enum Move { MoveNone = -1, MovePulse = 0, MoveCycles = 1, MoveResonate = 2,
            MoveResonateRev = 3 };
constexpr int kMoveCount = 4;

inline bool isGate(int m) { return m == MoveCycles || m == MoveResonate ||
                                   m == MoveResonateRev; }
/// Frame-LOCKED moves advance per frame rather than per second (see `advance`).
inline bool isFrameLocked(int m) { return m == MoveResonate || m == MoveResonateRev; }

struct Params {
  /// Depth range, in room units. `z_near` is in front of the near plane, so a
  /// quad at phase 1 has already swallowed the camera.
  float z_far = 8.0f;
  float z_near = 0.15f;

  /// Travel shape, shared by every move that eases. 0 linear, 0.5 smoothstep,
  /// 1 smootherstep — the same family and the same default as the rig's.
  float ease = 0.5f;

  // --- Pulse: a one-shot train, one quad after another ---
  float pulse_time = 0.9f;      ///< seconds for ONE quad to cross the tunnel
  float pulse_stagger = 0.12f;  ///< seconds between successive quads

  // --- Cycles: two counter-rotating quads beating, plus a third on a duty ---
  float cycles_f0 = 0.4f;       ///< Hz at the press
  float cycles_f1 = 2.6f;       ///< Hz at the end of the ramp
  float cycles_ramp = 4.0f;     ///< seconds to get from f0 to f1
  /// The reverse quad's own curve. Deliberately NOT the same numbers — two
  /// identical ramps in opposite directions beat at a constant rate, which is
  /// far less interesting than two that drift apart.
  float cycles_rev_f0 = 0.3f;
  float cycles_rev_f1 = 2.0f;
  float cycles_duty = 0.5f;         ///< fraction of each period quad 1 spends forward
  float cycles_duty_period = 1.5f;  ///< seconds for one forward+backward round

  // --- Resonate: all three evenly spaced, fast enough to alias ---
  float resonate_f0 = 2.0f;
  float resonate_f1 = 14.0f;
  float resonate_ramp = 6.0f;
};

/// One frame of the tunnel. Depths are in room units; a quad that is not `live`
/// is not drawn at all (the card is black at rest).
struct Out {
  float z[kQuads] = {};
  bool live[kQuads] = {};
  float phase[kQuads] = {};  ///< the raw 0..1, for tests and telemetry
  float rate = 0.0f;         ///< the cycling rate in Hz right now, 0 when not cycling
  float fps = 0.0f;          ///< the smoothed frame rate the locked moves are using
};

namespace detail {

inline float clampf(float v, float lo, float hi) { return v < lo ? lo : (v > hi ? hi : v); }
inline float clamp01(float v) { return clampf(v, 0.0f, 1.0f); }
inline float lerpf(float a, float b, float t) { return a + (b - a) * t; }

/// Ease in AND out, at a chosen strength — 0..0.5 fades linear into smoothstep,
/// 0.5..1 fades smoothstep into smootherstep. Shared with the rig so the two
/// cards' motion feels like it came from the same hand.
inline float easeCurve(float t, float ease) {
  t = clamp01(t);
  const float s1 = t * t * (3.0f - 2.0f * t);
  const float s2 = t * t * t * (t * (t * 6.0f - 15.0f) + 10.0f);
  const float e = clamp01(ease);
  return e <= 0.5f ? lerpf(t, s1, e * 2.0f) : lerpf(s1, s2, (e - 0.5f) * 2.0f);
}

/// [0,1) — a cycling quad wraps rather than clamping.
inline float wrap01(float v) {
  v -= (float)(int)v;
  return v < 0.0f ? v + 1.0f : v;
}

}  // namespace detail

/// Depth for a phase. Geometric so apparent speed is constant; see the header.
inline float depthOf(float phase, const Params& p) {
  const float far = p.z_far > 1e-3f ? p.z_far : 1e-3f;
  const float near_ = p.z_near > 1e-3f ? p.z_near : 1e-3f;
  return far * std::pow(near_ / far, detail::clamp01(phase));
}

/// The per-instance state. Every field is an accumulator, which is why the
/// effect declares no temporal capability — it cannot be seeked.
struct Core {
  int move = MoveNone;
  float move_t = 0.0f;      ///< seconds since the press
  float phase[kQuads] = {};
  bool live[kQuads] = {};
  /// Quad 1's duty-cycle clock, in Cycles only.
  float duty_t = 0.0f;

  /// Smoothed FRAME TIME, for the frame-locked moves. Seeded on the first tick
  /// rather than ramped up from a guess, so Resonate is in the right place
  /// immediately instead of sliding into it over the first second.
  ///
  /// Smoothing dt and inverting, rather than smoothing 1/dt: the mean of 1/dt
  /// is not 1/mean(dt), and under jitter the difference is a real bias — an
  /// 8ms/28ms alternation averages 80 fps one way and the true 56 fps the
  /// other. The tunnel would run 40% fast.
  float dt_avg = 0.0f;

  void reset() { *this = Core(); }

  /// Start a move. Monophonic — whatever was running is dropped where it stood.
  void trigger(int m) {
    if (m < 0 || m >= kMoveCount) return;
    move = m;
    move_t = 0.0f;
    duty_t = 0.0f;
    for (int i = 0; i < kQuads; ++i) {
      live[i] = false;
      phase[i] = 0.0f;
    }
    if (m == MoveCycles || m == MoveResonate || m == MoveResonateRev) {
      // Evenly spaced down the tunnel, which is the whole look of these three.
      // Spread rather than stacked: starting them together would put a quad at
      // the far end with nowhere to recede to, and it would wrap to the near
      // end on the first frame and read as a pop.
      //
      // THE HIGHLIGHT ARRIVES FIRST, and which end that puts it at depends on
      // what the other two are doing:
      //
      //   Cycles    — the highlight is the ONLY quad travelling toward you (the
      //               others recede or alternate), so nothing is racing it. Put
      //               it DEEPEST and it still gets there first, having had the
      //               whole tunnel to come down. Spread the other way it starts
      //               on top of the camera and is gone before you see it.
      //   Resonate  — all three run toward you together at one rate, so the
      //               order is fixed by position alone and the leader is simply
      //               whoever is NEAREST. The highlight has to be that one.
      //   Rev       — the same arrangement as Resonate, so it reads as Resonate
      //               running backwards, which is what its name promises.
      //               Putting the highlight at the far end instead would be
      //               worse than useless: going backwards it is already AT the
      //               exit, so it wraps to the near end on the first frame and
      //               never travels at all.
      //
      // So only Cycles wants the highlight at the back.
      const bool highlight_nearest = (m != MoveCycles);
      for (int i = 0; i < kQuads; ++i) {
        const int slot = highlight_nearest ? i : (kQuads - 1 - i);
        phase[i] = (float)slot / (float)kQuads;
        live[i] = true;
      }
    }
    // Pulse leaves every quad dark; they arm one at a time as the train runs.
  }

  /// Release a gate. Ignored unless THAT move is the one still running — a
  /// release arriving after something else took over must not stop it.
  void release(int m) {
    if (move == m && isGate(m)) move = MoveNone;
  }

  /// Advance one frame.
  Out tick(const Params& p, float dt) {
    using namespace detail;
    if (dt < 0.0f) dt = 0.0f;
    if (dt > kMaxDt) dt = kMaxDt;

    // --- The frame-rate estimate. -----------------------------------------
    // The host ABI has no frame counter and no fps, but the executor calls
    // tick() exactly once per frame on every path, so a tick IS a frame. What
    // the locked moves need is not accuracy but STEADINESS: advancing by
    // rate/fps each frame is rate*dt in expectation, yet the per-frame step
    // stays constant while dt jitters — and a constant step is what makes the
    // pattern land on the same depths every frame instead of smearing.
    if (dt > 1e-6f) {
      // Clamp the SAMPLE, not just the result: a 250 ms hitch is a stall, not a
      // 4 fps frame rate, and letting it into the average would drag the tunnel
      // to a crawl for the next second over one bad frame.
      const float sample = clampf(dt, 1.0f / kMaxFps, 1.0f / kMinFps);
      dt_avg = (dt_avg <= 0.0f) ? sample : lerpf(dt_avg, sample, kFpsSmoothing);
    }
    const float fps_now = dt_avg > 1e-6f ? clampf(1.0f / dt_avg, kMinFps, kMaxFps)
                                         : 60.0f;

    Out o;
    o.fps = fps_now;

    if (move != MoveNone) move_t += dt;

    switch (move) {
      case MovePulse:    tickPulse(p, o); break;
      case MoveCycles:   tickCycles(p, dt, fps_now, o); break;
      case MoveResonate:
      case MoveResonateRev: tickResonate(p, fps_now, move == MoveResonateRev, o); break;
      default: break;
    }

    // No move, no quads. Every route back to rest lands here — a release, a
    // Pulse retiring, or never having started — so none of them has to remember
    // to put the lights out on its way.
    if (move == MoveNone) for (int i = 0; i < kQuads; ++i) live[i] = false;

    for (int i = 0; i < kQuads; ++i) {
      o.phase[i] = phase[i];
      o.live[i] = live[i];
      o.z[i] = depthOf(phase[i], p);
    }
    return o;
  }

 private:
  /// The rate right now, ramping f0 -> f1 over `ramp` on the shared ease curve.
  float rampedRate(float f0, float f1, float ramp, const Params& p) const {
    const float r = ramp > 1e-4f ? ramp : 1e-4f;
    const float u = detail::easeCurve(move_t / r, p.ease);
    return detail::lerpf(f0, f1, u);
  }

  /// PULSE — a one-shot train, LED BY THE LAST QUAD.
  ///
  /// The train runs 3, 2, 1 rather than 1, 2, 3, so the HIGHLIGHT arrives first
  /// and the other two follow it in. That is the whole shape of the gesture: the
  /// bright one is the hit and the rest are its tail, which only reads that way
  /// if the hit gets there first. Ordering it the natural way round makes the
  /// highlight a straggler and the pulse lands on the wrong colour.
  ///
  /// Each quad takes `pulse_time` to cross and is dark before and after its own
  /// window, so the move ends by itself once the last one lands.
  void tickPulse(const Params& p, Out& o) {
    using namespace detail;
    const float travel = p.pulse_time > 1e-4f ? p.pulse_time : 1e-4f;
    bool any = false;
    for (int i = 0; i < kQuads; ++i) {
      const float t0 = (float)(kQuads - 1 - i) * p.pulse_stagger;
      const float u = (move_t - t0) / travel;
      if (u < 0.0f || u >= 1.0f) { live[i] = false; continue; }
      live[i] = true;
      phase[i] = easeCurve(u, p.ease);
      any = true;
    }
    // Retire once the whole train has passed, so a held trigger cannot re-arm
    // it and a finished pulse leaves the card black.
    if (!any && move_t > 0.0f) move = MoveNone;
    (void)o;
  }

  /// CYCLES — two quads counter-rotating on separate ramps so they beat, and a
  /// third reversing on a duty cycle. Time-true (dt), not frame-locked: this is
  /// a motion you watch, not an aliasing instrument.
  void tickCycles(const Params& p, float dt, float fps_now, Out& o) {
    using namespace detail;
    const float fwd = rampedRate(p.cycles_f0, p.cycles_f1, p.cycles_ramp, p);
    const float rev = rampedRate(p.cycles_rev_f0, p.cycles_rev_f1, p.cycles_ramp, p);
    o.rate = fwd;
    (void)fps_now;

    // Quad 3 (the highlight) forward, quad 2 (secondary) backward.
    phase[2] = wrap01(phase[2] + fwd * dt);
    phase[1] = wrap01(phase[1] - rev * dt);

    // Quad 1 alternates: forward for `duty` of each period, backward for the
    // rest. It rides the forward rate either way, so it reads as part of the
    // same machine rather than a third independent voice.
    const float period = p.cycles_duty_period > 1e-4f ? p.cycles_duty_period : 1e-4f;
    duty_t += dt;
    while (duty_t >= period) duty_t -= period;
    const bool forward = (duty_t / period) < clamp01(p.cycles_duty);
    phase[0] = wrap01(phase[0] + (forward ? fwd : -fwd) * dt);

    for (int i = 0; i < kQuads; ++i) live[i] = true;
  }

  /// RESONATE — all three evenly spaced, cycling fast enough that the eye reads
  /// the frames rather than the motion. FRAME-LOCKED: the advance is per frame,
  /// which is what makes the aliasing a stable pattern instead of a smear.
  ///
  /// A cycling tunnel always wraps somewhere. Forwards it wraps at the near end,
  /// where the quad is already larger than the frame, so you never see it.
  /// Backwards it wraps the other way and a quad re-enters at the near end —
  /// as a big frame rushing in from off-screen and shrinking away. That is the
  /// character of the reverse move, not a defect in it.
  void tickResonate(const Params& p, float fps_now, bool reverse, Out& o) {
    using namespace detail;
    const float rate = rampedRate(p.resonate_f0, p.resonate_f1, p.resonate_ramp, p);
    o.rate = rate;
    const float step = (reverse ? -rate : rate) / fps_now;
    for (int i = 0; i < kQuads; ++i) {
      phase[i] = wrap01(phase[i] + step);
      live[i] = true;
    }
  }
};

}  // namespace three_walls_show
