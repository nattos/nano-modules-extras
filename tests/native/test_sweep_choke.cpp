// test_sweep_choke.cpp — goldens for `source.light.sweep_choke`'s core.
//
// Host-free: sweep_choke_core.h carries no effect ABI, so a whole gesture can
// be run out here at an exact dt with no wasm bundle, no executor and no GPU.
// (The web suite covers what the effect wraps around it — that the map lands
// on a real texture the right way up — because that is pixels.)
//
// The brief these pin, in the order the header states it:
//
//   * home is a solid colour at full, and the deadzone does not touch it;
//   * out of it the fade is a TILT along the angle, and the far end is black;
//   * Ping-Pong mirrors the tilt for the other half of the knob, Forward not;
//   * a hard pass into a choke throws a tail; a crawl does not;
//   * the tail thins rather than dims, and ends — on the bars, not the mask;
//   * Chunk grows the tail's blinks up the bar, to whole bars at 1;
//   * Fade grays the mask as well as moving its edge;
//   * motion in the fade stutters, and stillness settles it;
//   * One-Way keeps the front travelling the same way in and out;
//   * the bars show glints — one per pass, always the same way, gliding on
//     after the knob stops — and the bounce as a flash; past full is white.

#include "sketch/sweep_choke_core.h"

#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

#include <cmath>
#include <limits>
#include <utility>
#include <vector>

using Catch::Matchers::WithinAbs;
using namespace sweep_choke_core;

namespace {

constexpr float kDt = 1.0f / 120.0f;

/// Params with every source of motion-driven change switched off, so a test
/// about the position law sees only the position law.
Params still() {
  Params p;
  p.flicker = 0.0f;
  p.bounce = 0.0f;
  p.latch_drive = 0.0f;
  return p;
}

/// Park the knob at `x` and let everything settle.
void park(Core& c, Params& p, float x, float seconds = 1.0f) {
  p.sweep = x;
  for (float t = 0.0f; t < seconds; t += kDt) c.tick(p, kDt);
}

/// Move the knob linearly from its current value to `to` over `seconds`.
void glide(Core& c, Params& p, float to, float seconds) {
  const float from = p.sweep;
  const int n = (int)std::ceil(seconds / kDt);
  for (int i = 1; i <= n; ++i) {
    p.sweep = from + (to - from) * (float)i / (float)n;
    c.tick(p, kDt);
  }
}

float cell(const Core& c, int bar, int seg) { return c.level[led_bars::cellIndex(bar, seg)]; }

float glowRow(const Core& c, int seg) {
  float s = 0.0f;
  for (int b = 0; b < kBars; ++b) s += c.glow[led_bars::cellIndex(b, seg)];
  return s / kBars;
}

float glowMax(const Core& c) {
  float m = 0.0f;
  for (int i = 0; i < kCells; ++i) m = c.glow[i] > m ? c.glow[i] : m;
  return m;
}

/// The segment row the glow is brightest on, or -1 if the bars are dark.
int glowPeakRow(const Core& c) {
  int best = -1;
  float m = 1e-3f;
  for (int k = 0; k < kSegments; ++k)
    if (glowRow(c, k) > m) { m = glowRow(c, k); best = k; }
  return best;
}

float barMean(const Core& c, int bar) {
  float s = 0.0f;
  for (int k = 0; k < kSegments; ++k) s += cell(c, bar, k);
  return s / kSegments;
}

float rowMean(const Core& c, int seg) {
  float s = 0.0f;
  for (int b = 0; b < kBars; ++b) s += cell(c, b, seg);
  return s / kBars;
}

}  // namespace

TEST_CASE("sweep_choke: home is every segment at full", "[sweep_choke]") {
  Core c;
  Params p = still();
  park(c, p, 0.5f);
  for (int i = 0; i < kCells; ++i) REQUIRE_THAT(c.level[i], WithinAbs(1.0f, 1e-6));
  // The bars are dark between gestures by default...
  for (int i = 0; i < kCells; ++i) REQUIRE_THAT(c.glow[i], WithinAbs(0.0f, 1e-6));

  // ...and at full Base they are the solid colour.
  p.base = 1.0f;
  c.tick(p, 0.0f);
  const float rgb[3] = {1.0f, 0.2f, 0.6f};
  const auto cells = toCells(c, rgb, 1.0f, 0.6f);
  for (int i = 0; i < kCells; ++i) {
    REQUIRE_THAT(cells.rgb[i][0], WithinAbs(1.0f, 1e-6));
    REQUIRE_THAT(cells.rgb[i][1], WithinAbs(0.2f, 1e-6));
    REQUIRE_THAT(cells.rgb[i][2], WithinAbs(0.6f, 1e-6));
  }
}

TEST_CASE("sweep_choke: the deadzone does not touch the look", "[sweep_choke]") {
  Core c;
  Params p = still();
  p.deadzone = 0.45f;
  for (float x : {0.3f, 0.4f, 0.6f, 0.72f}) {
    park(c, p, x, 0.2f);
    for (int i = 0; i < kCells; ++i) REQUIRE_THAT(c.level[i], WithinAbs(1.0f, 1e-6));
  }
}

TEST_CASE("sweep_choke: both ends are black", "[sweep_choke]") {
  Core c;
  Params p = still();
  for (float x : {0.0f, 1.0f}) {
    park(c, p, x);
    for (int i = 0; i < kCells; ++i) REQUIRE(c.level[i] < 1e-6f);
  }
}

TEST_CASE("sweep_choke: the fade is a tilt along the angle", "[sweep_choke]") {
  Core c;
  Params p = still();
  p.tilt = 1.0f;

  SECTION("90 deg: the dark rises from the floor") {
    p.angle_deg = 90.0f;
    park(c, p, 0.85f);
    // Every bar sees the same thing; rows go out from the bottom up.
    for (int k = 0; k + 1 < kSegments; ++k) REQUIRE(rowMean(c, k) <= rowMean(c, k + 1) + 1e-6f);
    REQUIRE(rowMean(c, 0) < rowMean(c, kSegments - 1) - 0.5f);
    for (int k = 0; k < kSegments; ++k)
      for (int b = 1; b < kBars; ++b) REQUIRE_THAT(cell(c, b, k), WithinAbs(cell(c, 0, k), 1e-6));
  }

  SECTION("0 deg: the dark sweeps left to right") {
    p.angle_deg = 0.0f;
    park(c, p, 0.85f);
    for (int b = 0; b + 1 < kBars; ++b) REQUIRE(barMean(c, b) <= barMean(c, b + 1) + 1e-6f);
    REQUIRE(barMean(c, 0) < barMean(c, kBars - 1) - 0.5f);
  }

  SECTION("tilt 0 is an even dimmer") {
    p.tilt = 0.0f;
    park(c, p, 0.85f);
    for (int i = 1; i < kCells; ++i) REQUIRE_THAT(c.level[i], WithinAbs(c.level[0], 1e-6));
    REQUIRE(c.level[0] > 0.0f);
    REQUIRE(c.level[0] < 1.0f);
  }
}

TEST_CASE("sweep_choke: Ping-Pong mirrors the other side, Forward does not", "[sweep_choke]") {
  Params p = still();
  p.tilt = 1.0f;
  p.angle_deg = 90.0f;

  auto bottomMinusTop = [&](int mode, float x) {
    Core c;
    Params q = p;
    q.side_mode = mode;
    park(c, q, x);
    return rowMean(c, 0) - rowMean(c, kSegments - 1);
  };

  // Upper half: the floor goes out first in both modes.
  REQUIRE(bottomMinusTop(SidePingPong, 0.85f) < -0.5f);
  REQUIRE(bottomMinusTop(SideForward, 0.85f) < -0.5f);
  // Lower half: Ping-Pong flips it, Forward keeps it.
  REQUIRE(bottomMinusTop(SidePingPong, 0.15f) > 0.5f);
  REQUIRE(bottomMinusTop(SideForward, 0.15f) < -0.5f);
}

TEST_CASE("sweep_choke: a hard pass into a choke throws a tail", "[sweep_choke]") {
  Params p;
  p.flicker = 0.0f;

  SECTION("fast sweep throws") {
    Core c;
    park(c, p, 0.2f);   // start part way out, below the choke
    glide(c, p, 1.0f, 0.2f);
    REQUIRE(c.release > 0.5f);
  }

  SECTION("a crawl barely throws") {
    Core c;
    park(c, p, 0.2f);
    glide(c, p, 1.0f, 8.0f);
    REQUIRE(c.release < 0.05f);
  }

  SECTION("harder sweep, longer tail") {
    auto tailSeconds = [&](float sweep_time) {
      Core c;
      Params q = p;
      q.latch_drive = 1.0f;   // so the fast one does not peg
      park(c, q, 0.2f);
      glide(c, q, 1.0f, sweep_time);
      float t = 0.0f;
      while (c.release > 0.0f && t < 20.0f) { c.tick(q, kDt); t += kDt; }
      return t;
    };
    REQUIRE(tailSeconds(0.25f) > tailSeconds(0.9f) + 0.1f);
  }
}

TEST_CASE("sweep_choke: the tail blinks at full, thins, and ends", "[sweep_choke]") {
  Core c;
  Params p;
  p.flicker = 0.0f;
  p.bounce = 0.0f;
  p.glint_gain = 0.0f;   // the tail alone on the bars
  p.ring_time = 1.5f;
  park(c, p, 0.2f);
  glide(c, p, 1.0f, 0.15f);
  REQUIRE(c.release > 0.5f);

  // With the knob parked in the choke the bars are black, so anything lit is a
  // blink — and a blink on a dark segment is at FULL, never dimmed. The mask
  // stays black throughout: the tail is not in it.
  int early = 0, late = 0, after = 0;
  float t = 0.0f;
  bool bright_only = true, mask_dark = true;
  while (t < 3.0f) {
    c.tick(p, kDt);
    t += kDt;
    int lit = 0;
    for (int i = 0; i < kCells; ++i) {
      if (c.level[i] > 1e-6f) mask_dark = false;
      if (c.glow[i] > 1e-6f) {
        ++lit;
        if (c.glow[i] < 1.0f - 1e-5f) bright_only = false;
      }
    }
    if (t < 0.4f) early += lit;
    else if (t > 1.0f && t < 1.4f) late += lit;
    else if (t > 1.6f) after += lit;
  }
  REQUIRE(bright_only);
  REQUIRE(mask_dark);
  REQUIRE(early > 0);
  REQUIRE(late < early);   // same window length, fewer lit segment-frames
  REQUIRE(after == 0);
  REQUIRE(c.release == 0.0f);
}

TEST_CASE("sweep_choke: the tail gathers where the light went out last", "[sweep_choke]") {
  Core c;
  Params p;
  p.flicker = 0.0f;
  p.bounce = 0.0f;
  p.angle_deg = 90.0f;   // the top goes out last
  p.tail_bias = 1.0f;
  p.ring_time = 3.0f;
  p.glint_gain = 0.0f;
  park(c, p, 0.2f);
  glide(c, p, 1.0f, 0.15f);

  int top = 0, bottom = 0;
  for (float t = 0.0f; t < 2.0f; t += kDt) {
    c.tick(p, kDt);
    for (int b = 0; b < kBars; ++b) {
      for (int k = 0; k < 3; ++k) if (c.glow[led_bars::cellIndex(b, k)] > 0.5f) ++bottom;
      for (int k = kSegments - 3; k < kSegments; ++k)
        if (c.glow[led_bars::cellIndex(b, k)] > 0.5f) ++top;
    }
  }
  REQUIRE(top > 4 * bottom);
}

namespace {
/// Throw a full tail at `chunk` and watch the bars ring for a second. Returns
/// lit segment-frames; `runs` collects every lit run's length up each bar.
int ringTail(float chunk, std::vector<int>& runs) {
  Core c;
  Params p;
  p.flicker = 0.0f;
  p.bounce = 0.0f;
  p.glint_gain = 0.0f;
  p.tail_bias = 0.0f;
  p.ring_time = 2.0f;
  p.tail_chunk = chunk;
  park(c, p, 0.2f);
  glide(c, p, 1.0f, 0.15f);
  int lit = 0;
  for (float t = 0.0f; t < 1.0f; t += kDt) {
    c.tick(p, kDt);
    for (int b = 0; b < kBars; ++b) {
      int run = 0;
      for (int k = 0; k <= kSegments; ++k) {
        const bool on = k < kSegments && c.glow[led_bars::cellIndex(b, k)] > 0.5f;
        if (on) { ++run; ++lit; }
        else if (run > 0) { runs.push_back(run); run = 0; }
      }
    }
  }
  return lit;
}
}  // namespace

TEST_CASE("sweep_choke: Chunk grows the tail's blinks up the bar", "[sweep_choke]") {
  std::vector<int> r0, r5, r1;
  const int lit0 = ringTail(0.0f, r0);
  const int lit5 = ringTail(0.5f, r5);
  const int lit1 = ringTail(1.0f, r1);
  REQUIRE(!r0.empty());
  REQUIRE(!r5.empty());
  REQUIRE(!r1.empty());

  auto mean = [](const std::vector<int>& v) {
    float s = 0.0f;
    for (int x : v) s += (float)x;
    return s / (float)v.size();
  };
  // 0: mostly lone segments. 1: every lit run is a whole bar.
  REQUIRE(mean(r0) < 1.6f);
  for (int x : r1) REQUIRE(x == kSegments);
  // Between: bigger runs, and not all the same size.
  REQUIRE(mean(r5) > 2.0f);
  int lo = kSegments, hi = 0;
  for (int x : r5) { lo = x < lo ? x : lo; hi = x > hi ? x : hi; }
  REQUIRE(hi > lo);
  // Fewer, bigger: about as many segments lit overall.
  REQUIRE(lit5 > lit0 / 3);
  REQUIRE(lit5 < lit0 * 3);
  REQUIRE(lit1 > lit0 / 4);
  REQUIRE(lit1 < lit0 * 4);
}

TEST_CASE("sweep_choke: Fade grays the mask as well as moving its edge", "[sweep_choke]") {
  auto litEnd = [](float fade) {
    Core c;
    Params p = still();
    p.tilt = 1.0f;
    p.angle_deg = 90.0f;
    p.fade = fade;
    park(c, p, 0.86f);
    return std::pair<float, float>(rowMean(c, kSegments - 1), rowMean(c, 0));
  };
  const auto hard = litEnd(0.0f);
  const auto soft = litEnd(1.0f);
  REQUIRE(hard.first > 0.99f);         // the far side untouched without it
  REQUIRE(soft.first < 0.7f);          // gray with it
  REQUIRE(soft.first > 0.3f);
  REQUIRE(soft.second < 0.05f);        // and the edge still took the near side
}

TEST_CASE("sweep_choke: motion in the fade stutters, stillness settles it", "[sweep_choke]") {
  Core c;
  Params p;
  p.latch_drive = 0.0f;
  p.bounce = 0.0f;
  p.flicker = 1.0f;
  p.settle = 0.5f;
  p.tilt = 0.0f;   // every segment half out together: all of them can stutter
  park(c, p, 0.5f);

  // Wobble around the middle of the fade.
  int stutters = 0;
  for (int i = 0; i < 240; ++i) {
    p.sweep = 0.86f + 0.01f * std::sin((float)i * 0.3f);
    c.tick(p, kDt);
    if (c.liveBlinks() > 0) ++stutters;
  }
  REQUIRE(stutters > 0);

  // Park there and wait out the settle many times over.
  park(c, p, 0.86f, 6.0f);
  REQUIRE(c.liveBlinks() == 0);
  for (int i = 1; i < kCells; ++i) REQUIRE_THAT(c.level[i], WithinAbs(c.level[0], 1e-6));
}

TEST_CASE("sweep_choke: a fast return flashes the bars past full", "[sweep_choke]") {
  Core c;
  Params p = still();
  p.bounce = 1.0f;
  p.glint_gain = 0.0f;   // the flash alone, not the glint the return launches
  park(c, p, 1.0f);
  float peak = 0.0f;
  glide(c, p, 0.5f, 0.06f);
  for (int i = 0; i < 60; ++i) {
    c.tick(p, kDt);
    if (glowMax(c) > peak) peak = glowMax(c);
  }
  REQUIRE(peak > 1.0f);
  // And the mask never sees it: nothing past full there.
  for (int i = 0; i < kCells; ++i) REQUIRE(c.level[i] <= 1.0f);
  park(c, p, 0.5f, 1.0f);
  REQUIRE_THAT(glowMax(c), WithinAbs(0.0f, 1e-6));
}

TEST_CASE("sweep_choke: One-Way keeps the front travelling one way", "[sweep_choke]") {
  Params p = still();
  p.tilt = 1.0f;
  p.angle_deg = 90.0f;
  p.side_mode = SideOneWay;
  auto bottomMinusTop = [](const Core& c) { return rowMean(c, 0) - rowMean(c, kSegments - 1); };

  SECTION("going out, either half: the floor goes first") {
    for (float x : {0.86f, 0.14f}) {
      Core c;
      Params q = p;
      park(c, q, 0.5f, 0.1f);
      glide(c, q, x, 0.3f);
      REQUIRE(bottomMinusTop(c) < -0.5f);
    }
  }

  SECTION("coming back in: the floor comes back first") {
    for (float end : {1.0f, 0.0f}) {
      Core c;
      Params q = p;
      park(c, q, 0.5f, 0.1f);
      glide(c, q, end, 0.3f);
      glide(c, q, end > 0.5f ? 0.86f : 0.14f, 0.3f);
      REQUIRE(bottomMinusTop(c) > 0.5f);
      // Home again, and out once more: the floor goes first again.
      glide(c, q, 0.5f, 0.3f);
      glide(c, q, end > 0.5f ? 0.86f : 0.14f, 0.3f);
      REQUIRE(bottomMinusTop(c) < -0.5f);
    }
  }

  SECTION("reversing mid-fade retraces rather than jumping") {
    Core c;
    Params q = p;
    park(c, q, 0.5f, 0.1f);
    glide(c, q, 0.86f, 0.3f);
    float before[kCells];
    for (int i = 0; i < kCells; ++i) before[i] = c.level[i];
    glide(c, q, 0.85f, 0.02f);
    for (int i = 0; i < kCells; ++i) REQUIRE(std::fabs(c.level[i] - before[i]) < 0.2f);
    REQUIRE(bottomMinusTop(c) < -0.5f);
  }
}

TEST_CASE("sweep_choke: a pass throws one glint, always the same way", "[sweep_choke]") {
  for (float from : {1.0f, 0.0f}) {
    Core c;
    Params p = still();
    p.glint_chaos = 0.0f;
    p.angle_deg = 90.0f;
    park(c, p, from);
    REQUIRE(glowMax(c) < 1e-6f);
    // One steady traverse from the end to the far side of the middle.
    glide(c, p, 1.0f - from, 1.0f);
    REQUIRE(c.glints.launches == 1u);
    // Wherever it has got to, it climbed: follow the peak row frame by frame.
    Core d;
    Params q = still();
    q.glint_chaos = 0.0f;
    q.angle_deg = 90.0f;
    park(d, q, from);
    q.sweep = from;
    int last = -1, climbs = 0, falls = 0;
    const int n = (int)std::ceil(1.0f / kDt);
    for (int i = 1; i <= n; ++i) {
      q.sweep = from + ((1.0f - from) - from) * (float)i / (float)n;
      d.tick(q, kDt);
      const int r = glowPeakRow(d);
      if (r >= 0 && last >= 0) {
        if (r > last) ++climbs;
        if (r < last) ++falls;
      }
      if (r >= 0) last = r;
    }
    REQUIRE(climbs >= 5);
    REQUIRE(falls == 0);
  }
}

TEST_CASE("sweep_choke: a glint glides on after the knob stops, then goes out", "[sweep_choke]") {
  Core c;
  Params p = still();
  p.glint_chaos = 0.0f;
  park(c, p, 1.0f);
  glide(c, p, 0.5f, 0.3f);   // stop dead in the middle, just after the launch
  REQUIRE(c.glints.liveCount() == 1);
  park(c, p, 0.5f, 0.1f);
  REQUIRE(c.glints.liveCount() == 1);
  REQUIRE(glowMax(c) > 0.1f);
  park(c, p, 0.5f, 3.0f);
  REQUIRE(c.glints.liveCount() == 0);
  REQUIRE(glowMax(c) < 1e-6f);
}

TEST_CASE("sweep_choke: Linger is how long a glint glides on", "[sweep_choke]") {
  auto lifetime = [](float linger) {
    Core c;
    Params p = still();
    p.glint_chaos = 0.0f;
    p.glint_ratio = 0.25f;   // slow enough that it runs down before it crosses
    p.glint_linger = linger;
    park(c, p, 1.0f);
    glide(c, p, 0.5f, 0.3f);   // launch, then stop dead
    float t = 0.0f;
    while (c.glints.liveCount() > 0 && t < 30.0f) { c.tick(p, kDt); t += kDt; }
    return t;
  };
  const float brief = lifetime(0.1f), dflt = lifetime(0.22f), long_ = lifetime(1.5f);
  REQUIRE(brief < dflt);
  REQUIRE(long_ > 2.0f * dflt);
}

TEST_CASE("sweep_choke: glints follow the gradient's angle", "[sweep_choke]") {
  Core c;
  Params p = still();
  p.glint_chaos = 0.0f;
  p.angle_deg = 0.0f;   // left to right
  p.glint_ratio = 2.0f;  // so it makes it all the way over before running down
  park(c, p, 1.0f);
  int first = -1, lastBar = -1;
  const int n = (int)std::ceil(1.0f / kDt);
  for (int i = 1; i <= 2 * n; ++i) {   // the sweep, then a second to finish crossing
    p.sweep = i < n ? 1.0f - (float)i / (float)n : 0.0f;
    c.tick(p, kDt);
    int best = -1;
    float m = 1e-3f;
    for (int b = 0; b < kBars; ++b) {
      float s = 0.0f;
      for (int k = 0; k < kSegments; ++k) s += c.glow[led_bars::cellIndex(b, k)];
      if (s > m) { m = s; best = b; }
    }
    if (best >= 0) { if (first < 0) first = best; lastBar = best; }
  }
  REQUIRE(first == 0);
  REQUIRE(lastBar == kBars - 1);
}

TEST_CASE("sweep_choke: Base shows the mask under the glints", "[sweep_choke]") {
  Core c;
  Params p = still();
  p.tilt = 1.0f;
  p.base = 1.0f;
  park(c, p, 0.86f);
  for (int i = 0; i < kCells; ++i) REQUIRE_THAT(c.glow[i], WithinAbs(c.level[i], 1e-6));
  p.base = 0.25f;
  c.tick(p, 0.0f);
  for (int i = 0; i < kCells; ++i) REQUIRE_THAT(c.glow[i], WithinAbs(0.25f * c.level[i], 1e-6));
}

TEST_CASE("sweep_choke: past full blows out toward white; the mask is gray", "[sweep_choke]") {
  Core c;
  for (int i = 0; i < kCells; ++i) { c.glow[i] = 0.0f; c.level[i] = 0.25f; }
  c.glow[0] = 0.5f;
  c.glow[1] = 1.0f;
  c.glow[2] = kLevelMax;
  const float rgb[3] = {1.0f, 0.0f, 0.5f};
  const auto out = toCells(c, rgb, 1.0f, 1.0f);
  REQUIRE_THAT(out.rgb[0][1], WithinAbs(0.0f, 1e-6));
  REQUIRE_THAT(out.rgb[0][2], WithinAbs(0.25f, 1e-6));
  REQUIRE_THAT(out.rgb[1][2], WithinAbs(0.5f, 1e-6));
  for (int k = 0; k < 3; ++k) REQUIRE_THAT(out.rgb[2][k], WithinAbs(1.0f, 1e-6));
  const auto m = maskCells(c);
  for (int k = 0; k < 3; ++k) REQUIRE_THAT(m.rgb[5][k], WithinAbs(0.25f, 1e-6));
}

TEST_CASE("sweep_choke: a NaN knob holds the middle", "[sweep_choke]") {
  Core c;
  Params p;
  p.sweep = std::numeric_limits<float>::quiet_NaN();
  for (int i = 0; i < 30; ++i) c.tick(p, kDt);
  for (int i = 0; i < kCells; ++i) REQUIRE_THAT(c.level[i], WithinAbs(1.0f, 1e-6));
}

TEST_CASE("sweep_choke: deterministic for a given dt sequence", "[sweep_choke]") {
  auto run = [] {
    Core c;
    Params p;
    park(c, p, 0.3f, 0.3f);
    glide(c, p, 1.0f, 0.2f);
    glide(c, p, 0.0f, 0.3f);
    glide(c, p, 0.8f, 0.5f);
    for (int i = 0; i < 90; ++i) c.tick(p, kDt);
    return c;
  };
  const Core a = run(), b = run();
  for (int i = 0; i < kCells; ++i) REQUIRE(a.level[i] == b.level[i]);
  REQUIRE(a.release == b.release);
}

TEST_CASE("sweep_choke: a zero-dt tick re-reads position and ages nothing", "[sweep_choke]") {
  // The effect calls tick(p, 0) from render() so a host that renders without
  // ticking still sees where the knob is. That must never advance the tail.
  Core c;
  Params p;
  p.flicker = 0.0f;
  park(c, p, 0.2f);
  glide(c, p, 1.0f, 0.15f);
  const float rel = c.release;
  const float clock = c.tail_clock;
  const int blinks = c.liveBlinks();
  REQUIRE(rel > 0.0f);
  for (int i = 0; i < 10; ++i) c.tick(p, 0.0f);
  REQUIRE(c.release == rel);
  REQUIRE(c.tail_clock == clock);
  REQUIRE(c.liveBlinks() == blinks);

  // ...and a parked position change shows up without any time passing.
  Core d;
  Params q = still();
  d.tick(q, 0.0f);
  q.sweep = 1.0f;
  d.tick(q, 0.0f);
  for (int i = 0; i < kCells; ++i) REQUIRE(d.level[i] < 1e-6f);
}
