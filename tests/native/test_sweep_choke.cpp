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
//   * the tail thins rather than dims, and ends;
//   * motion in the fade stutters, and stillness settles it.

#include "sketch/sweep_choke_core.h"

#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

#include <cmath>
#include <limits>

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
  p.ring_time = 1.5f;
  park(c, p, 0.2f);
  glide(c, p, 1.0f, 0.15f);
  REQUIRE(c.release > 0.5f);

  // With the knob parked in the choke every base segment is black, so anything
  // lit is a blink — and a blink on a dark segment is at FULL, never dimmed.
  int early = 0, late = 0, after = 0;
  float t = 0.0f;
  bool bright_only = true;
  while (t < 3.0f) {
    c.tick(p, kDt);
    t += kDt;
    int lit = 0;
    for (int i = 0; i < kCells; ++i) {
      if (c.level[i] > 1e-6f) {
        ++lit;
        if (c.level[i] < 1.0f - 1e-5f) bright_only = false;
      }
    }
    if (t < 0.4f) early += lit;
    else if (t > 1.0f && t < 1.4f) late += lit;
    else if (t > 1.6f) after += lit;
  }
  REQUIRE(bright_only);
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
  park(c, p, 0.2f);
  glide(c, p, 1.0f, 0.15f);

  int top = 0, bottom = 0;
  for (float t = 0.0f; t < 2.0f; t += kDt) {
    c.tick(p, kDt);
    for (int b = 0; b < kBars; ++b) {
      for (int k = 0; k < 3; ++k) if (cell(c, b, k) > 0.5f) ++bottom;
      for (int k = kSegments - 3; k < kSegments; ++k) if (cell(c, b, k) > 0.5f) ++top;
    }
  }
  REQUIRE(top > 4 * bottom);
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

TEST_CASE("sweep_choke: a fast return overshoots past full", "[sweep_choke]") {
  Core c;
  Params p = still();
  p.bounce = 1.0f;
  park(c, p, 1.0f);
  float peak = 0.0f;
  glide(c, p, 0.5f, 0.06f);
  for (int i = 0; i < 60; ++i) {
    c.tick(p, kDt);
    if (c.mean_level > peak) peak = c.mean_level;
  }
  REQUIRE(peak > 1.0f);
  park(c, p, 0.5f, 1.0f);
  REQUIRE_THAT(c.mean_level, WithinAbs(1.0f, 1e-6));
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
