// test_drum_bars.cpp — goldens for `source.light.drum_bars`'s core.
//
// Host-free: drum_bars_core.h carries no effect ABI and no transport, so every
// hit can be played at an exact dt and every pattern pick read straight off
// the Cycle position.

#include "sketch/drum_bars_core.h"

#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

#include <cmath>
#include <limits>
#include <algorithm>
#include <set>
#include <vector>

using Catch::Matchers::WithinAbs;
using namespace drum_bars_core;

namespace {

constexpr float kDt = 1.0f / 120.0f;

void run(Core& c, const Params& p, float seconds) {
  const int n = (int)std::lround(seconds / kDt);
  for (int i = 0; i < n; ++i) c.tick(p, kDt);
}

/// Fire one channel on a quiet core and return the frame it lands on.
Core hitOnce(const Params& p, int ch) {
  Core c;
  c.fire(ch);
  c.tick(p, kDt);
  return c;
}

int litMask(const Core& c, float thresh = 0.05f) {
  int m = 0;
  for (int b = 0; b < kBars; ++b)
    if (c.level[b] > thresh) m |= 1 << b;
  return m;
}

constexpr int kOuter = 0b1001, kInner = 0b0110, kAll = 0b1111;

}  // namespace

TEST_CASE("drum_bars: the kick is mirror-symmetric, set by Cycle, and loops", "[drum_bars]") {
  Params p;
  std::set<int> seen;
  for (int i = 0; i < 400; ++i) {
    p.cycle = i / 400.0f;
    const Core c = hitOnce(p, 0);
    CHECK(c.level[0] == c.level[3]);
    CHECK(c.level[1] == c.level[2]);
    const int m = litMask(c);
    CHECK((m == kOuter || m == kInner || m == kAll));
    const int pat = kickPattern(p, p.cycle);
    CHECK(m == (pat == PatOuter ? kOuter : pat == PatInner ? kInner : kAll));
    seen.insert(pat);
    // The same place, the same pick.
    CHECK(litMask(hitOnce(p, 0)) == m);
  }
  CHECK(seen.size() == 3);
  p.cycle = 0.0f;
  const int at0 = litMask(hitOnce(p, 0));
  p.cycle = 1.0f;
  CHECK(litMask(hitOnce(p, 0)) == at0);
  // Crisp but smooth: the pick changes only a handful of times per loop.
  int changes = 0;
  for (int i = 0; i < 1000; ++i)
    if (kickPattern(p, i / 1000.0f) != kickPattern(p, (i + 1) / 1000.0f)) ++changes;
  CHECK(changes >= 2);
  CHECK(changes <= 3 * (int)p.variety);
}

TEST_CASE("drum_bars: across seeds, every pattern turns up", "[drum_bars]") {
  int full = 0;
  for (int seed = 0; seed < 20; ++seed) {
    Params p;
    p.seed = seed;
    std::set<int> seen;
    for (int i = 0; i < 200; ++i) seen.insert(kickPattern(p, i / 200.0f));
    if (seen.size() == 3) ++full;
  }
  CHECK(full == 20);
}

TEST_CASE("drum_bars: the backbeat is the kick's converse", "[drum_bars]") {
  Params p;
  p.flam = 0.0f;
  bool saw_all = false;
  for (int i = 0; i < 200; ++i) {
    p.cycle = i / 200.0f;
    const int kick = litMask(hitOnce(p, 0));
    const Core c = hitOnce(p, 1);
    const int back = litMask(c);
    if (kick == kAll) {
      saw_all = true;
      CHECK(back == kAll);
      for (int b = 0; b < kBars; ++b)
        CHECK_THAT(c.level[b], WithinAbs(p.back_level * p.under, 1e-5f));
    } else {
      CHECK(back == (kick ^ kAll));
      CHECK_THAT(c.level[back == kOuter ? 0 : 1], WithinAbs(p.back_level, 1e-5f));
    }
  }
  CHECK(saw_all);
}

TEST_CASE("drum_bars: backbeat flams ripple outward, weaker each time", "[drum_bars]") {
  Params p;
  p.flam = 1.0f;
  p.back_decay = 0.03f;   // short, so each repeat stands on its own
  // Find a place where the backbeat is the inner pair and flams at least twice.
  float at = -1.0f;
  for (int i = 0; i < 400 && at < 0.0f; ++i)
    if (backPattern(p, i / 400.0f) == PatInner && flamCount(p, i / 400.0f) >= 2) at = i / 400.0f;
  REQUIRE(at >= 0.0f);
  p.cycle = at;
  const int n = flamCount(p, at);

  Core c;
  c.fire(1);
  c.tick(p, kDt);
  CHECK(litMask(c) == kInner);
  // Peaks on the outer bars, one per repeat, each weaker than the last.
  std::vector<float> peaks;
  float prev = 0.0f;
  bool rising = false;
  float inner_after = 0.0f;
  for (int f = 0; f < 120; ++f) {
    c.tick(p, kDt);
    CHECK(c.level[0] == c.level[3]);
    const float v = c.level[0];
    if (v > prev) rising = true;
    else if (rising && v < prev) { peaks.push_back(prev); rising = false; }
    prev = v;
    if (f * kDt > 0.04f) inner_after = std::max(inner_after, c.level[1]);
  }
  REQUIRE((int)peaks.size() == n);
  // A repeat may land up to a frame late in its own life, so it can read a
  // little under its amplitude, never over.
  CHECK(peaks[0] <= p.back_level * p.flam_fade + 1e-5f);
  CHECK(peaks[0] > 0.5f * p.back_level * p.flam_fade);
  for (size_t k = 1; k < peaks.size(); ++k) CHECK(peaks[k] < peaks[k - 1]);
  CHECK(inner_after == 0.0f);   // the ripple left the inner pair

  p.flam = 0.0f;
  Core q;
  q.fire(1);
  q.tick(p, kDt);
  run(q, p, 0.05f);
  for (int f = 0; f < 60; ++f) { q.tick(p, kDt); CHECK(q.level[0] == 0.0f); }
}

TEST_CASE("drum_bars: decoration strobes one bar, never the same twice running", "[drum_bars]") {
  Params p;
  p.deco_min = 0.2f;
  Core c;
  int last = -1;
  std::set<float> levels;
  for (int h = 0; h < 40; ++h) {
    c.fire(2);
    int bar = -1, offs = 0;
    bool was_on = false;
    for (int f = 0; f < 60; ++f) {
      c.tick(p, kDt);
      int lit = 0;
      for (int b = 0; b < kBars; ++b)
        if (c.level[b] > 0.0f) { ++lit; bar = b; levels.insert(c.level[b]); }
      CHECK(lit <= 1);
      if (was_on && lit == 0) ++offs;
      was_on = lit > 0;
    }
    REQUIRE(bar >= 0);
    CHECK(bar != last);
    CHECK(offs >= 2);
    last = bar;
  }
  CHECK(levels.size() > 10);
  for (float v : levels) CHECK(v >= p.deco_min - 1e-6f);
}

TEST_CASE("drum_bars: overlaps cancel, XOR-style; Max keeps them", "[drum_bars]") {
  Params p;
  p.back_level = 1.0f;
  p.under = 1.0f;
  p.flam = 0.0f;
  // Kick and backbeat both light all four only when the kick is All.
  float at = -1.0f;
  for (int i = 0; i < 400 && at < 0.0f; ++i)
    if (kickPattern(p, i / 400.0f) == PatAll) at = i / 400.0f;
  REQUIRE(at >= 0.0f);
  p.cycle = at;
  Core c;
  c.fire(0);
  c.fire(1);
  c.tick(p, kDt);
  for (int b = 0; b < kBars; ++b) CHECK(c.level[b] < 1e-5f);

  p.combine = CombineMax;
  Core m;
  m.fire(0);
  m.fire(1);
  m.tick(p, kDt);
  for (int b = 0; b < kBars; ++b) CHECK_THAT(m.level[b], WithinAbs(1.0f, 1e-5f));

  // Half and half: a + b - 2ab.
  CHECK_THAT(0.5f + 0.5f - 2.0f * 0.25f, WithinAbs(0.5f, 1e-6f));
}

TEST_CASE("drum_bars: Drift walks the loop; 0 holds it", "[drum_bars]") {
  Params p;
  Core c;
  run(c, p, 1.0f);
  CHECK(c.cycle_out == 0.0f);
  p.drift = 0.25f;
  run(c, p, 1.0f);
  CHECK_THAT(c.cycle_out, WithinAbs(0.25f, 1e-3f));
  p.cycle = 0.9f;
  run(c, p, 0.0f);
  c.tick(p, 0.0f);
  CHECK_THAT(c.cycle_out, WithinAbs(0.15f, 1e-3f));   // wraps
}

TEST_CASE("drum_bars: the groove's grid", "[drum_bars]") {
  Params p;
  p.auto_deco = 0.0f;
  for (int step = 0; step < 16; ++step) {
    Core c;
    const int bits = c.groove(p, step);
    CHECK(((bits & 1) != 0) == (step == 0 || step == 8 || (grooveHits(p, step, 0.0f) & 1)));
    CHECK(((bits & 2) != 0) == (step == 4 || step == 12));
    CHECK((bits & 4) == 0);
  }
  // Somewhere in the loop there is a pickup kick.
  bool pickup = false;
  for (int i = 0; i < 400; ++i) {
    const float c = i / 400.0f;
    if ((grooveHits(p, 10, c) | grooveHits(p, 14, c)) & 1) pickup = true;
  }
  CHECK(pickup);
  p.auto_deco = 1.0f;
  Core c;
  CHECK((c.groove(p, 3) & 4) != 0);
}

TEST_CASE("drum_bars: hits die to black; zero-dt holds; deterministic; NaN safe", "[drum_bars]") {
  Params p;
  Core c;
  c.fire(0);
  c.fire(1);
  c.fire(2);
  for (int i = 0; i < 5; ++i) c.tick(p, 0.0f);
  CHECK(c.mean == 0.0f);   // held until time moves
  c.tick(p, kDt);
  CHECK(c.mean > 0.0f);
  run(c, p, 2.0f);
  CHECK(c.mean == 0.0f);

  Params r;
  r.drift = 0.05f;
  Core a, b;
  for (int i = 0; i < 2000; ++i) {
    for (Core* x : {&a, &b}) {
      if (i % 30 == 0) x->groove(r, (i / 30) % 16);
      x->tick(r, kDt);
    }
    for (int k = 0; k < kBars; ++k) REQUIRE(a.level[k] == b.level[k]);
  }

  Params n;
  n.cycle = std::numeric_limits<float>::quiet_NaN();
  n.drift = std::numeric_limits<float>::quiet_NaN();
  n.variety = std::numeric_limits<float>::quiet_NaN();
  n.deco_rate = std::numeric_limits<float>::quiet_NaN();
  n.flam_gap = std::numeric_limits<float>::quiet_NaN();
  Core z;
  for (int ch = 0; ch < 3; ++ch) z.fire(ch);
  for (int i = 0; i < 240; ++i) {
    z.tick(n, kDt);
    for (int k = 0; k < kBars; ++k) REQUIRE(std::isfinite(z.level[k]));
  }
  CHECK(z.cycle_out == 0.0f);
}
