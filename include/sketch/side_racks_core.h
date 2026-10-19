#pragma once
/*
 * side_racks_core.h — the logic behind `source.light.side_racks`: the whole
 * show for the six-bar rig (rack_map.h) on one card.
 *
 * Layered in this order, every tick:
 *
 *   1. THE SWEEP. The house knob (sweep_knob.h): full across a wide deadzone,
 *      choking at either end, bouncing coming back in, and throwing a charge
 *      at the choke that rings out as Three Planes' Strobe tail
 *      (three_planes_strobe.h) rolling over the three bars. TILT is how much
 *      the bars roll like a chord rather than strike together:
 *
 *        Sweep Up. The choke is a FRONT climbing the rack, from below the
 *        bottom bar to above the top one as the knob goes from the edge of
 *        the deadzone to the end. While the knob MOVES, a tilted rack lights
 *        only around the front — at full tilt, the one bar it is passing,
 *        and the choke is that spot climbing off the top. At REST the tilt
 *        falls back to `rest_tilt` of itself and the front softens into a
 *        gradient (`rest_blur`): bars the front has passed are darker, bars
 *        it has not reached brighter. Tilt 0 is the plain choke, all three
 *        bars in unison. Inside the deadzone the tilt does nothing at all, so
 *        riding the middle never touches the light.
 *
 *   2. THE PRIMARY MIX-IN, on one gate (bouncy-abode's `strong`):
 *        Flam    — a sidechain duck: a smooth attack to the peak cut, backing
 *                  off to a sustain while the gate holds, a release after it;
 *                  the cut chopped into a shudder, frame-counted.
 *        Wave    — a pulse born at the stage's center line, its front running
 *                  out across the real room in meters, eating the light it
 *                  passes, which recovers behind it. On the front itself the
 *                  cut is bit-crushed — blocky in space and in level — and
 *                  sparkles in the highlight colour on an orthomod-style
 *                  Walsh code, paged fast.
 *        Beating — two oscillators stepped per FRAME (the frame rate is
 *                  estimated from dt), so near-Nyquist rates alias into
 *                  standing patterns; their interference beats up and down
 *                  the rack, each bar one solid colour.
 *
 *   3. THE SECONDARY MIX-IN, on three gates (`low` / `mid` / `high`). A seeded
 *      pattern deals each channel one bar on the left and one on the right,
 *      re-dealt every step. A channel firing alone flashes its bar on ONE side
 *      — a single alternator shared by all three picks which, so two channels
 *      landing together go to opposite sides — and all three up together is a
 *      heavy hit: every channel's bar on BOTH sides.
 *
 *   4. LIFT AND STROBE. Lift adds the accent colour; the strobe flashes it on
 *      and blacks out between, from `strobe_min_hz` to `strobe_max_hz` as its
 *      depth rises, capped at perfect frame alternation.
 *
 *   5. MIX-INS BEFORE OR AFTER. `mix_post` moves the mix-ins from under the
 *      lift and strobe (0) to over them (1), so a duck can cut a strobe.
 *
 * The mix-ins are OPERATORS, not pictures: a multiplier per segment (the cut)
 * and an additive colour per segment (sparkle and flashes), applied around
 * the lift and strobe by `mix_post`.
 *
 * Host-free, like sweep_choke_core.h: no effect ABI, no GPU. The effect feeds
 * it the transport's step crossings (`advanceStep`) and everything else rides
 * in Params, so the Catch2 goldens drive the whole show at an exact dt.
 */

#include <cmath>
#include <cstdint>

#include <sketch/rack_map.h>
#include <sketch/sweep_knob.h>
#include <sketch/three_planes_strobe.h>

namespace side_racks_core {

using rack_map::kSides;
using rack_map::kBars;
using rack_map::kSegments;
using rack_map::kCells;
using sweep_knob::clampf;
using sweep_knob::clamp01;
using sweep_knob::smooth01;

/// A frame longer than this is a transport stall, not slow motion.
constexpr float kMaxDt = 0.25f;

/// How far past full a segment is read: from 1 to here it pushes toward white.
constexpr float kLevelMax = 1.5f;

/// The bounce in level units — a hard return is worth a few tenths of lead,
/// doubled so it reaches past full on the rack.
constexpr float kBounceGain = 2.0f;

/// The moving spot's travel, in bars: it sets out ON the bottom bar — a fast
/// exit from the deadzone lands on it at once rather than in the dark below —
/// and leaves past the top by this many spot widths, so the choke is black.
constexpr float kSpotExit = 2.5f;

/// How far into the fade the tilt takes hold: 0 inside the deadzone, all of it
/// by this much fade progress. Riding the middle never touches the light.
constexpr float kTiltEngage = 0.12f;

/// The gate threshold for every trigger input.
constexpr float kGateOn = 0.5f;

/// Frame-rate estimate for the frame-locked moves: three_walls_show.h's.
constexpr float kMinFps = 15.0f;
constexpr float kMaxFps = 240.0f;
constexpr float kFpsSmoothing = 0.02f;

/// A chop runs free below this many cycles per frame and alternates frames
/// exactly at or above it, with env_lfo's hysteresis so it does not chatter.
constexpr float kLockEnter = 0.5f;
constexpr float kLockExit = 0.45f;

/// Concurrent wave pulses.
constexpr int kMaxPulses = 4;

/// The wave's sparkle code: a 16-point Walsh family.
constexpr int kCode = 16;

/// Steps of the bit-crushed cut on the wavefront.
constexpr float kCrushLevels = 4.0f;

constexpr int kChannels = 3;

enum SweepMode { SweepUp = 0 };
enum PrimaryMode { PrimaryFlam = 0, PrimaryWave = 1, PrimaryBeating = 2 };
enum SecondaryMode { SecondaryPattern = 0 };
/// What advances the secondary pattern. The beat-grid ones are counted by the
/// effect (fx::BeatTick) and handed in through advanceStep.
enum StepMode {
  StepBeat = 0, Step2Beats = 1, StepBar = 2, Step2Bars = 3, Step4Bars = 4,
  StepPrimary = 5,
};

/// Ticks per bar for each beat-grid step mode; 0 for the ones that aren't.
inline float stepsPerBar(int mode) {
  switch (mode) {
    case StepBeat: return 4.0f;
    case Step2Beats: return 2.0f;
    case StepBar: return 1.0f;
    case Step2Bars: return 0.5f;
    case Step4Bars: return 0.25f;
    default: return 0.0f;
  }
}

/// Read fresh each tick — a wire may be moving any of it.
struct Params {
  // --- Sweep ---------------------------------------------------------------
  sweep_knob::Params knob;
  int sweep_mode = SweepUp;
  float tilt = 0.5f;        ///< 0 unison .. 1 one bar at a time while moving
  float rest_tilt = 0.35f;  ///< how much of the tilt is kept at rest
  float focus = 0.45f;      ///< the moving spot's half-width, in bars
  float rest_blur = 1.0f;   ///< the resting front's softness, in bars
  float tail_level = 0.6f;  ///< how bright the thrown tail's roll is
  float tail_rate = 22.0f;  ///< steps per second of the roll
  float tail_duty = 0.55f;  ///< its on-window at full release
  float tail_grace = 0.35f; ///< its grace stroke

  // --- Primary -------------------------------------------------------------
  float primary = 0.0f;     ///< the gate
  int primary_mode = PrimaryFlam;
  float primary_amount = 1.0f;

  float env_attack = 0.03f;   ///< s to the peak cut
  float env_decay = 0.25f;    ///< s from the peak back to the sustain
  float env_sustain = 0.4f;   ///< held while the gate is up
  float env_release = 0.2f;   ///< s back to nothing once it falls

  float flam_amount = 0.85f;
  float flam_rate = 18.0f;    ///< chop cycles per second, 0 a plain duck
  float flam_floor = 0.35f;   ///< how much of the cut the open frames keep

  float wave_amount = 0.9f;
  float wave_speed = 6.0f;    ///< m/s
  float wave_start = 1.0f;    ///< m from the center line the front is born at
  float wave_recover = 0.35f; ///< s for the light to come back behind it
  float wave_width = 0.25f;   ///< m either side of the front that crushes
  int wave_crush = 4;         ///< segments per block on the front
  float wave_sparkle = 1.2f;  ///< the sparkle's brightness
  float wave_sparkle_rate = 30.0f;  ///< code pages per second
  float wave_color[3] = {0.30f, 0.85f, 1.00f};

  float beat_amount = 0.8f;
  float beat_rate = 29.0f;    ///< Hz, stepped per frame
  float beat_detune = 1.5f;   ///< Hz the second oscillator is off by
  float beat_spread_a = 0.0f; ///< cycles per bar up the rack, first oscillator
  float beat_spread_b = 0.33f;///< and the second

  // --- Secondary -----------------------------------------------------------
  float sec[kChannels] = {0.0f, 0.0f, 0.0f};  ///< the gates
  int secondary_mode = SecondaryPattern;
  float secondary_amount = 1.0f;
  int step_mode = StepBar;
  int seed = 0;
  float sec_decay = 0.18f;    ///< s a flash takes to fall
  float sec_flam_rate = 0.0f; ///< chop cycles per second, 0 a plain flash
  float heavy_window = 0.04f; ///< s all three must land within
  float ch_color[kChannels][3] = {
      {1.00f, 0.45f, 0.10f}, {0.30f, 0.85f, 1.00f}, {0.75f, 0.40f, 1.00f}};

  // --- Lift and strobe -----------------------------------------------------
  float lift = 0.0f;
  float strobe = 0.0f;        ///< depth: 0 off, up from there faster
  float strobe_min_hz = 4.0f;
  float strobe_max_hz = 30.0f;
  float lift_color[3] = {1.0f, 1.0f, 1.0f};

  // --- Output --------------------------------------------------------------
  float mix_post = 0.0f;
  float color[3] = {1.00f, 0.22f, 0.62f};
  float intensity = 1.0f;
  float whiten = 0.6f;
};

// --- Small pieces --------------------------------------------------------

inline float lerpf(float a, float b, float t) { return a + (b - a) * t; }

inline uint32_t hash32(uint32_t x) {
  x ^= x >> 16; x *= 0x7feb352du;
  x ^= x >> 15; x *= 0x846ca68bu;
  x ^= x >> 16;
  return x;
}
inline uint32_t hash3(uint32_t a, uint32_t b, uint32_t c) {
  return hash32(a * 0x9e3779b9u ^ hash32(b + 0x632be5abu) ^ hash32(c * 0x85ebca6bu + 0x27d4eb2fu));
}

inline int popcount(uint32_t v) {
  int n = 0;
  while (v) { v &= v - 1; ++n; }
  return n;
}

/// The Walsh code orthomod's Walsh codebook is made of: row `r`, point `c`.
inline bool walshBit(int r, int c) { return ((popcount((uint32_t)(r & c)) & 1) ^ 1) != 0; }

/// The six ways to deal three channels onto three bars.
constexpr int kPerms[6][kChannels] = {
    {0, 1, 2}, {0, 2, 1}, {1, 0, 2}, {1, 2, 0}, {2, 0, 1}, {2, 1, 0}};

/// A chop: free-running square wave below half a cycle per frame, exact frame
/// alternation at or above it. `open` is the first half of each cycle, so a
/// chop restarted by `restart()` opens on its first frame. A zero-dt tick (a
/// refresh at render) changes nothing.
struct Chop {
  float phase = 0.0f;
  bool locked = false;
  bool open = true;
  bool fresh = false;  ///< restarted: the next real frame is the open one

  void restart() { phase = 0.0f; locked = false; open = true; fresh = true; }

  bool tick(float rate, float dt) {
    if (!(dt > 0.0f)) return open;
    const float r = rate > 0.0f ? rate : 0.0f;
    const float cycles = r * dt;
    if (fresh) {
      // The frame it was restarted on opens, locked or not.
      fresh = false;
      locked = cycles >= kLockEnter;
      open = true;
      return open;
    }
    locked = cycles >= (locked ? kLockExit : kLockEnter);
    if (locked) {
      open = !open;
    } else {
      phase += cycles;
      phase -= std::floor(phase);
      open = phase < 0.5f;
    }
    return open;
  }
};

/// The sidechain-duck envelope: a smooth attack from wherever it is to the
/// peak, back off to the sustain, held while the gate is up, a smooth release
/// once it is down. A gate that falls before the peak lets the attack finish,
/// then releases straight from the peak — a short kick is a full duck.
struct DuckEnv {
  enum Stage { Idle, Attack, Decay, Sustain, Release };
  Stage stage = Idle;
  float t = 0.0f;
  float from = 0.0f;
  float value = 0.0f;

  void press() { stage = Attack; t = 0.0f; from = value; }

  void tick(const Params& p, bool gate, float dt) {
    if (stage == Idle) { value = 0.0f; return; }
    t += dt;
    for (int guard = 0; guard < 4; ++guard) {
      if (stage == Attack) {
        const float a = p.env_attack > 1e-4f ? p.env_attack : 1e-4f;
        if (t < a) { value = lerpf(from, 1.0f, smooth01(t / a)); return; }
        t -= a;
        value = 1.0f;
        from = 1.0f;
        stage = gate ? Decay : Release;
        continue;
      }
      if (stage == Decay) {
        if (!gate) { stage = Release; from = value; t = 0.0f; continue; }
        const float d = p.env_decay > 1e-4f ? p.env_decay : 1e-4f;
        const float s = clamp01(p.env_sustain);
        if (t < d) { value = lerpf(1.0f, s, smooth01(t / d)); return; }
        t -= d;
        stage = Sustain;
        continue;
      }
      if (stage == Sustain) {
        value = clamp01(p.env_sustain);
        if (!gate) { stage = Release; from = value; t = 0.0f; continue; }
        return;
      }
      if (stage == Release) {
        const float r = p.env_release > 1e-4f ? p.env_release : 1e-4f;
        if (t < r) { value = from * (1.0f - smooth01(t / r)); return; }
        stage = Idle;
        value = 0.0f;
        return;
      }
      return;
    }
  }
};

struct Pulse {
  bool live = false;
  float age = 0.0f;
};

struct Core {
  sweep_knob::Knob knob;
  three_planes_strobe::Core tail;
  float clock = 0.0f;

  // Frame rate, for the frame-locked moves.
  float dt_avg = 0.0f;
  float fps = 60.0f;

  // Primary.
  bool primary_prev = false;
  DuckEnv env;
  Chop flam_chop;
  Pulse pulses[kMaxPulses];
  float sparkle_clock = 0.0f;
  bool sparkle_locked = false;
  uint32_t sparkle_page = 0;
  double beat_a = 0.0, beat_b = 0.0;

  // Secondary.
  bool sec_prev[kChannels] = {};
  float rise_t[kChannels] = {-1e9f, -1e9f, -1e9f};
  bool heavy_fired = false;
  uint32_t step = 0;
  int perm[kSides] = {0, 0};
  int dealt_seed = 0x7fffffff;
  uint32_t dealt_step = 0xffffffffu;
  int next_side = 0;
  float flash[kSides][kBars] = {};
  int flash_ch[kSides][kBars] = {};
  Chop flash_chop[kSides][kBars];

  // Strobe.
  Chop strobe_chop;
  bool strobing = false;

  // --- This frame ---------------------------------------------------------
  float bar_level[kBars] = {1.0f, 1.0f, 1.0f};  ///< the sweep per bar, past 1 on a bounce
  float tail_gain[kBars] = {};                  ///< the thrown roll per bar, 0..1
  float front = 0.0f;                           ///< the moving spot's centre, in bars
  float tilt_now = 0.0f;                        ///< the tilt in force
  float mul[kCells] = {};                       ///< the mix-ins' cut
  float add[kCells][3] = {};                    ///< the mix-ins' colour
  bool strobe_on = false;                       ///< this frame is a strobe flash
  float primary_env = 0.0f;                     ///< the primary's envelope, for the rail
  float mean_level = 1.0f;

  void reset() { *this = Core(); }

  /// The transport crossed `n` step lines (beat-grid step modes).
  void advanceStep(int n) { if (n > 0) step += (uint32_t)n; }

  /// Which bar channel `ch` lands on, on `side`, this step.
  int barFor(int side, int ch) const { return kPerms[perm[side]][ch]; }

  void tick(const Params& p, float dt) {
    if (!(dt > 0.0f)) dt = 0.0f;
    if (dt > kMaxDt) dt = kMaxDt;
    clock += dt;
    tickFps(dt);
    tickSweep(p, dt);
    tickPrimary(p, dt);
    tickSecondary(p, dt);
    tickStrobe(p, dt);
  }

  /// The frame, coloured, in rack_map's physics order.
  rack_map::Cells cells(const Params& p) const {
    rack_map::Cells out;
    const float post = clamp01(p.mix_post);
    const float pre = 1.0f - post;
    const float lift = clamp01(p.lift);
    const float in = p.intensity > 0.0f ? p.intensity : 0.0f;
    const float wh = clamp01(p.whiten);
    const float tl = clamp01(p.tail_level);
    for (int side = 0; side < kSides; ++side)
      for (int bar = 0; bar < kBars; ++bar) {
        const float lvl = bar_level[bar] + tl * tail_gain[bar];
        for (int seg = 0; seg < kSegments; ++seg) {
          const int i = rack_map::cellIndex(side, bar, seg);
          const float m = mul[i];
          const float m_pre = lerpf(1.0f, m, pre);
          const float m_post = lerpf(1.0f, m, post);
          float rgb[3];
          for (int k = 0; k < 3; ++k) {
            float v = p.color[k] * lvl * m_pre + add[i][k] * pre;
            v += p.lift_color[k] * lift;
            if (strobing) v = strobe_on ? v + p.lift_color[k] : 0.0f;
            v = v * m_post + add[i][k] * post;
            rgb[k] = v > 0.0f ? v : 0.0f;
          }
          overdrive(rgb, wh);
          for (int k = 0; k < 3; ++k) out.rgb[i][k] = rgb[k] * in;
        }
      }
    return out;
  }

  /// Past 1, a segment keeps its hue at full and pushes toward white by
  /// `whiten` — what an overdriven LED looks like.
  static void overdrive(float rgb[3], float whiten) {
    float mx = rgb[0];
    if (rgb[1] > mx) mx = rgb[1];
    if (rgb[2] > mx) mx = rgb[2];
    if (mx <= 1.0f) return;
    const float over = clamp01((mx - 1.0f) / (kLevelMax - 1.0f)) * whiten;
    for (int k = 0; k < 3; ++k) {
      const float v = rgb[k] / mx;
      rgb[k] = v + (1.0f - v) * over;
    }
  }

 private:
  void tickFps(float dt) {
    if (!(dt > 0.0f)) return;
    const float sample = clampf(dt, 1.0f / kMaxFps, 1.0f / kMinFps);
    dt_avg = dt_avg <= 0.0f ? sample : lerpf(dt_avg, sample, kFpsSmoothing);
    fps = clampf(1.0f / dt_avg, kMinFps, kMaxFps);
  }

  // --- 1. The sweep --------------------------------------------------------
  void tickSweep(const Params& p, float dt) {
    knob.tick(p.knob, dt);
    const float m = knob.speed;
    const float q = knob.progress;
    const float lit = knob.level + kBounceGain * knob.bounce;

    const float engage = smooth01(q / kTiltEngage);
    const float tau = clamp01(p.tilt) * lerpf(clamp01(p.rest_tilt), 1.0f, m) * engage;
    tilt_now = tau;
    const float fw = p.focus > 0.05f ? p.focus : 0.05f;
    const float rw = p.rest_blur > 0.05f ? p.rest_blur : 0.05f;
    const float top = (float)(kBars - 1);
    // The spot from the bottom bar to clear of the top; the resting edge from
    // below everything (all lit) to above everything (all choked).
    front = lerpf(0.0f, top + kSpotExit * fw, q);
    const float edge_front = lerpf(-rw, top + rw, q);
    const float depth = clamp01(p.knob.depth);
    float sum = 0.0f;
    for (int b = 0; b < kBars; ++b) {
      // While moving, a spot around the front; at rest, a soft edge with the
      // bars the front has passed darker.
      const float d = (float)b - front;
      const float spot = std::exp(-(d / fw) * (d / fw));
      const float edge = smooth01(0.5f + ((float)b - edge_front) / (2.0f * rw));
      const float shape = lerpf(edge, spot, m);
      // Depth sets how dark the spot's surroundings go, as it does the ends.
      const float tilted = 1.0f - depth * (1.0f - shape);
      // The bounce rides on top whichever picture it is.
      const float over = kBounceGain * knob.bounce;
      bar_level[b] = clampf(lerpf(lit, tilted + over, tau), 0.0f, kLevelMax);
      sum += bar_level[b];
    }
    mean_level = sum / (float)kBars;

    three_planes_strobe::Params tp;
    tp.release = knob.release;
    tp.rate = p.tail_rate;
    tp.duty = p.tail_duty;
    tp.grace = p.tail_grace;
    tail.tick(tp, dt);
    for (int b = 0; b < kBars; ++b) tail_gain[b] = tail.gain[b];
  }

  // --- 2. The primary ------------------------------------------------------
  void tickPrimary(const Params& p, float dt) {
    for (int i = 0; i < kCells; ++i) {
      mul[i] = 1.0f;
      add[i][0] = add[i][1] = add[i][2] = 0.0f;
    }
    const bool gate = p.primary >= kGateOn;
    const bool rose = gate && !primary_prev;
    primary_prev = gate;
    if (rose) {
      env.press();
      flam_chop.restart();
      spawnPulse();
      if (p.step_mode == StepPrimary) ++step;
    }
    env.tick(p, gate, dt);
    const bool chop_open = flam_chop.tick(p.flam_rate, dt);
    ageWave(p, dt);
    if (dt > 0.0f) {
      beat_a += (double)(p.beat_rate / fps);
      beat_b += (double)((p.beat_rate + p.beat_detune) / fps);
      beat_a -= std::floor(beat_a);
      beat_b -= std::floor(beat_b);
    }

    const float trim = clamp01(p.primary_amount);
    primary_env = 0.0f;
    switch (p.primary_mode) {
      case PrimaryWave: applyWave(p, trim); break;
      case PrimaryBeating: applyBeating(p, trim); break;
      default: applyFlam(p, trim, chop_open); break;
    }
  }

  void applyFlam(const Params& p, float trim, bool chop_open) {
    primary_env = env.value;
    const float amt = clamp01(p.flam_amount) * trim * env.value;
    // A rate of 0 is a plain duck: every frame is a cut frame.
    const bool cut = !(p.flam_rate > 0.0f) || !chop_open;
    const float c = cut ? amt : amt * clamp01(p.flam_floor);
    for (int i = 0; i < kCells; ++i) mul[i] = 1.0f - c;
  }

  void applyBeating(const Params& p, float trim) {
    primary_env = env.value;
    const float amt = clamp01(p.beat_amount) * trim * env.value;
    constexpr float kTau = 6.28318530717958647692f;
    for (int b = 0; b < kBars; ++b) {
      const float pa = (float)beat_a + p.beat_spread_a * (float)b;
      const float pb = (float)beat_b + p.beat_spread_b * (float)b;
      const float v = clamp01(0.5f + 0.25f * (std::cos(kTau * pa) + std::cos(kTau * pb)));
      const float m = lerpf(1.0f, v, amt);
      for (int side = 0; side < kSides; ++side)
        for (int seg = 0; seg < kSegments; ++seg) mul[rack_map::cellIndex(side, b, seg)] = m;
    }
  }

  void spawnPulse() {
    int slot = 0;
    float oldest = -1.0f;
    for (int i = 0; i < kMaxPulses; ++i) {
      if (!pulses[i].live) { slot = i; oldest = -1.0f; break; }
      if (pulses[i].age > oldest) { oldest = pulses[i].age; slot = i; }
    }
    pulses[slot].live = true;
    pulses[slot].age = 0.0f;
  }

  float waveSpeed(const Params& p) const { return p.wave_speed > 0.05f ? p.wave_speed : 0.05f; }

  void ageWave(const Params& p, float dt) {
    const float speed = waveSpeed(p);
    const float rec = p.wave_recover > 1e-3f ? p.wave_recover : 1e-3f;
    for (int i = 0; i < kMaxPulses; ++i) {
      Pulse& pu = pulses[i];
      if (!pu.live) continue;
      pu.age += dt;
      const float r = p.wave_start + speed * pu.age;
      // Retired once the light behind it at the outer end is all but back.
      if (r - rack_map::kOuter > speed * rec * 6.0f + p.wave_width) pu = Pulse();
    }
    // The sparkle's page clock: free below half a page per frame, a new page
    // every frame above.
    if (dt > 0.0f) {
      const float pages = (p.wave_sparkle_rate > 0.0f ? p.wave_sparkle_rate : 0.0f) * dt;
      sparkle_locked = pages >= (sparkle_locked ? kLockExit : kLockEnter);
      if (sparkle_locked) {
        ++sparkle_page;
      } else {
        sparkle_clock += pages;
        while (sparkle_clock >= 1.0f) { sparkle_clock -= 1.0f; ++sparkle_page; }
      }
    }
  }

  void applyWave(const Params& p, float trim) {
    const float amt = clamp01(p.wave_amount) * trim;
    const float speed = waveSpeed(p);
    const float rec = p.wave_recover > 1e-3f ? p.wave_recover : 1e-3f;
    const float width = p.wave_width > 1e-3f ? p.wave_width : 1e-3f;
    const int crush = p.wave_crush < 1 ? 1 : (p.wave_crush > kSegments ? kSegments : p.wave_crush);
    const float sparkle = (p.wave_sparkle > 0.0f ? p.wave_sparkle : 0.0f) * trim;
    const uint32_t seed = (uint32_t)p.seed;
    const int rot = (int)(hash32(seed ^ 0x5bd1e995u) % (uint32_t)kCode);

    float env_max = 0.0f;
    for (int i = 0; i < kMaxPulses; ++i) {
      const Pulse& pu = pulses[i];
      if (!pu.live) continue;
      const float r = p.wave_start + speed * pu.age;
      // Count a pulse on the rack only while its front or its wake is there.
      const float rel = std::exp(-pu.age / (rec + rack_map::kBarLength / speed));
      if (rel > env_max) env_max = rel;
      for (int bar = 0; bar < kBars; ++bar) {
        const uint32_t row = 1u + hash3(seed, sparkle_page + (uint32_t)i * 7919u, (uint32_t)bar) %
                                      (uint32_t)(kCode - 1);
        for (int seg = 0; seg < kSegments; ++seg) {
          float reach = rack_map::segmentReach(seg);
          float d = reach - r;
          const bool band = (d < 0.0f ? -d : d) < width;
          int block = 0;
          if (band) {
            // BIT-CRUSHED on the front: the segment reads its block's centre,
            // counted from the inner end so the blocks mirror and line up.
            const int inner = kSegments - 1 - seg;
            block = inner / crush;
            const float centre = ((float)(block * crush) + 0.5f * (float)crush);
            reach = rack_map::kInner + centre * rack_map::kSegmentLength;
            d = reach - r;
          }
          float cut = 0.0f;
          if (d <= 0.0f) cut = amt * std::exp((d / speed) / rec);   // d<=0: time since it passed
          if (band) cut = std::floor(cut * kCrushLevels + 0.5f) / kCrushLevels;
          for (int side = 0; side < kSides; ++side) {
            const int c = rack_map::cellIndex(side, bar, seg);
            const float m = 1.0f - cut;
            if (m < mul[c]) mul[c] = m;
          }
          if (band && sparkle > 0.0f && walshBit((int)row, (block + rot) % kCode)) {
            const float w = 1.0f - (d < 0.0f ? -d : d) / width;
            const float s = sparkle * (w > 0.0f ? w : 0.0f);
            for (int side = 0; side < kSides; ++side) {
              const int c = rack_map::cellIndex(side, bar, seg);
              for (int k = 0; k < 3; ++k) {
                const float v = p.wave_color[k] * s;
                if (v > add[c][k]) add[c][k] = v;
              }
            }
          }
        }
      }
    }
    primary_env = env_max;
  }

  // --- 3. The secondary ----------------------------------------------------
  void deal(const Params& p) {
    if (dealt_step == step && dealt_seed == p.seed) return;
    const uint32_t seed = (uint32_t)p.seed;
    for (int side = 0; side < kSides; ++side)
      perm[side] = (int)(hash3(seed, step, (uint32_t)side + 0x1234u) % 6u);
    dealt_step = step;
    dealt_seed = p.seed;
  }

  void fireFlash(const Params& p, int side, int ch) {
    const int bar = barFor(side, ch);
    flash[side][bar] = 1.0f;
    flash_ch[side][bar] = ch;
    flash_chop[side][bar].restart();
    (void)p;
  }

  void tickSecondary(const Params& p, float dt) {
    deal(p);
    // Age the flashes first, so one fired this frame shows at full.
    const float fall = p.sec_decay > 1e-3f ? dt / p.sec_decay : 1.0f;
    for (int side = 0; side < kSides; ++side)
      for (int bar = 0; bar < kBars; ++bar) {
        if (dt > 0.0f && flash[side][bar] > 0.0f) {
          flash[side][bar] -= fall;
          if (flash[side][bar] < 0.0f) flash[side][bar] = 0.0f;
        }
      }

    bool gate[kChannels], rose[kChannels];
    bool any_rose = false, all_up = true;
    for (int c = 0; c < kChannels; ++c) {
      gate[c] = p.sec[c] >= kGateOn;
      rose[c] = gate[c] && !sec_prev[c];
      sec_prev[c] = gate[c];
      if (rose[c]) { rise_t[c] = clock; any_rose = true; }
      if (!gate[c]) all_up = false;
    }
    if (!all_up) heavy_fired = false;

    bool heavy = false;
    if (any_rose && all_up && !heavy_fired) {
      float lo = rise_t[0], hi = rise_t[0];
      for (int c = 1; c < kChannels; ++c) {
        lo = rise_t[c] < lo ? rise_t[c] : lo;
        hi = rise_t[c] > hi ? rise_t[c] : hi;
      }
      heavy = hi - lo <= (p.heavy_window > 0.0f ? p.heavy_window : 0.0f) + 1e-6f;
    }
    if (heavy) {
      heavy_fired = true;
      for (int c = 0; c < kChannels; ++c)
        for (int side = 0; side < kSides; ++side) fireFlash(p, side, c);
    } else {
      // One shared alternator: two channels landing together take opposite
      // sides, and nothing marches in step.
      for (int c = 0; c < kChannels; ++c) {
        if (!rose[c]) continue;
        fireFlash(p, next_side, c);
        next_side ^= 1;
      }
    }

    const float trim = clamp01(p.secondary_amount);
    for (int side = 0; side < kSides; ++side)
      for (int bar = 0; bar < kBars; ++bar) {
        const bool open = flash_chop[side][bar].tick(p.sec_flam_rate, dt);
        float f = flash[side][bar];
        if (f <= 0.0f) continue;
        if (p.sec_flam_rate > 0.0f && !open) f = 0.0f;
        const float* col = p.ch_color[flash_ch[side][bar]];
        for (int seg = 0; seg < kSegments; ++seg) {
          const int i = rack_map::cellIndex(side, bar, seg);
          for (int k = 0; k < 3; ++k) add[i][k] += col[k] * f * trim;
        }
      }
  }

  // --- 4. The strobe -------------------------------------------------------
  void tickStrobe(const Params& p, float dt) {
    const float depth = clamp01(p.strobe);
    const bool want = depth > 0.0f;
    if (want && !strobing) strobe_chop.restart();
    strobing = want;
    if (!strobing) { strobe_on = false; return; }
    const float lo = p.strobe_min_hz > 0.1f ? p.strobe_min_hz : 0.1f;
    const float hi = p.strobe_max_hz > lo ? p.strobe_max_hz : lo;
    const float hz = lo * std::pow(hi / lo, depth);
    strobe_on = strobe_chop.tick(hz, dt);
  }
};

}  // namespace side_racks_core
