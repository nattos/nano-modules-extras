// test_sweep_gradient.cpp — goldens for `source.light.sweep_gradient`'s core.
//
// Host-free: sweep_gradient_core.h carries no effect ABI, so a whole gesture
// can be played at an exact dt. The knob half is Sweep Choke's mask law, and is
// pinned here again in brief; the trigger half is new — a sweep gesture on an
// attack / hold / release envelope that owns the sweep while it runs.

#include "sketch/sweep_gradient_core.h"

#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

#include <cmath>
#include <limits>

using Catch::Matchers::WithinAbs;
using namespace sweep_gradient_core;

namespace {

constexpr float kDt = 1.0f / 120.0f;

void run(Core& c, const Params& p, float seconds) {
  const int n = (int)std::lround(seconds / kDt);
  for (int i = 0; i < n; ++i) c.tick(p, kDt);
}

float cell(const Core& c, int bar, int seg) { return c.level[led_bars::cellIndex(bar, seg)]; }

float rowMean(const Core& c, int seg) {
  float s = 0.0f;
  for (int b = 0; b < kBars; ++b) s += cell(c, b, seg);
  return s / kBars;
}

float minLevel(const Core& c) {
  float m = 1.0f;
  for (int i = 0; i < kCells; ++i) m = c.level[i] < m ? c.level[i] : m;
  return m;
}

bool flat(const Core& c, float tol) {
  for (int i = 0; i < kCells; ++i)
    if (std::fabs(c.level[i] - c.level[0]) > tol) return false;
  return true;
}

}  // namespace

TEST_CASE("sweep_gradient: the knob sets one level, settled flat", "[sweep_gradient]") {
  Params p;
  Core c;
  c.tick(p, kDt);
  for (int i = 0; i < kCells; ++i) CHECK_THAT(c.level[i], WithinAbs(1.0f, 1e-6f));

  p.sweep = 0.0f;
  run(c, p, 2.0f);
  CHECK(c.level[0] < 1e-3f);
  CHECK(flat(c, 1e-3f));

  p.sweep = 1.0f;
  run(c, p, 2.0f);
  CHECK(c.mean_level < 1e-3f);

  p.sweep = 0.86f;
  run(c, p, 2.0f);
  CHECK(c.level[0] > 0.1f);
  CHECK(c.level[0] < 0.9f);
  CHECK(flat(c, 1e-3f));
  CHECK_THAT(c.level[0], WithinAbs(positionLevel(p, 0.86f), 1e-3f));
}

TEST_CASE("sweep_gradient: a knob change rolls along the angle", "[sweep_gradient]") {
  Params p;   // angle 90: bottom first
  Core c;
  c.tick(p, kDt);
  p.sweep = 1.0f;
  run(c, p, 0.2f);
  CHECK(rowMean(c, 0) < rowMean(c, 9) - 0.3f);
  run(c, p, 2.0f);
  CHECK(flat(c, 1e-3f));
}

TEST_CASE("sweep_gradient: a trigger plays a choke that rolls in and back out", "[sweep_gradient]") {
  Params p;
  p.trig_side = TrigUpper;
  Core c;
  c.tick(p, kDt);
  c.fire();
  run(c, p, p.attack + 2 * kDt);
  CHECK(c.stage == Core::Hold);
  CHECK_THAT(c.env, WithinAbs(1.0f, 1e-6f));
  CHECK_THAT(c.sweep_out, WithinAbs(1.0f, 1e-6f));

  // Upper side, One-Way or not: bottom goes out first.
  run(c, p, 0.12f);
  CHECK(rowMean(c, 0) < 0.5f);
  CHECK(rowMean(c, 9) > rowMean(c, 0) + 0.3f);

  // Through release, the roll and the soften: all back to full, idle.
  run(c, p, 2.0f);
  CHECK(c.stage == Core::Idle);
  CHECK(c.env == 0.0f);
  CHECK(minLevel(c) > 0.99f);
}

TEST_CASE("sweep_gradient: the gesture owns the sweep; the knob takes back over", "[sweep_gradient]") {
  Params p;
  p.trig_side = TrigUpper;
  p.reach = 0.4f;   // inside the deadzone: the gesture is full light
  p.sweep = 0.0f;   // the knob parked at black
  Core c;
  run(c, p, 1.0f);
  CHECK(c.mean_level < 1e-3f);

  c.fire();
  run(c, p, 0.3f);
  // The knob moving does nothing while the gesture runs.
  p.sweep = 0.02f;
  run(c, p, 0.2f);
  CHECK(c.active());
  CHECK(c.sweep_out > 0.5f);
  CHECK(c.mean_level > 0.7f);

  run(c, p, 2.0f);
  CHECK(!c.active());
  CHECK_THAT(c.sweep_out, WithinAbs(0.02f, 1e-6f));
  CHECK_THAT(c.mean_level, WithinAbs(positionLevel(p, 0.02f), 1e-3f));
}

TEST_CASE("sweep_gradient: a retrigger picks up from where the envelope is", "[sweep_gradient]") {
  Params p;
  p.trig_side = TrigUpper;
  Core c;
  c.tick(p, kDt);
  c.fire();
  run(c, p, 0.4f);   // into the release
  REQUIRE(c.stage == Core::Release);
  REQUIRE(c.env > 0.2f);
  REQUIRE(c.env < 0.8f);

  c.fire();
  float prev = c.env, max_step = 0.0f, peak = 0.0f;
  for (int i = 0; i < 240; ++i) {
    c.tick(p, kDt);
    const float d = std::fabs(c.env - prev);
    max_step = d > max_step ? d : max_step;
    peak = c.env > peak ? c.env : peak;
    prev = c.env;
  }
  // Never faster than the attack's own slope: no jump back to 0.
  CHECK(max_step <= kDt / p.attack + 1e-4f);
  CHECK_THAT(peak, WithinAbs(1.0f, 1e-6f));
  CHECK(!c.active());
}

TEST_CASE("sweep_gradient: Alternate flips the side, and the roll with it", "[sweep_gradient]") {
  Params p;   // Alternate, Ping-Pong, angle 90
  Core c;
  c.tick(p, kDt);

  c.fire();
  run(c, p, 0.2f);
  CHECK(c.sweep_out > 0.5f);
  CHECK(rowMean(c, 0) < rowMean(c, 9));   // upper end: bottom first
  run(c, p, 2.0f);

  c.fire();
  run(c, p, 0.2f);
  CHECK(c.sweep_out < 0.5f);
  CHECK(rowMean(c, 9) < rowMean(c, 0));   // lower end, mirrored: top first
  run(c, p, 2.0f);

  p.trig_side = TrigLower;
  for (int k = 0; k < 2; ++k) {
    c.fire();
    run(c, p, 0.2f);
    CHECK(c.sweep_out < 0.5f);
    run(c, p, 2.0f);
  }
}

TEST_CASE("sweep_gradient: Reach under the deadzone barely dims", "[sweep_gradient]") {
  Params p;
  p.reach = 0.5f;
  Core c;
  c.tick(p, kDt);
  c.fire();
  float lo = 1.0f;
  for (int i = 0; i < 240; ++i) {
    c.tick(p, kDt);
    lo = minLevel(c) < lo ? minLevel(c) : lo;
  }
  CHECK(lo > 0.9f);
  CHECK(lo < 1.0f);
}

TEST_CASE("sweep_gradient: a zero-dt tick ages nothing and holds a fire", "[sweep_gradient]") {
  Params p;
  Core c;
  c.tick(p, kDt);
  c.fire();
  for (int i = 0; i < 10; ++i) c.tick(p, 0.0f);
  CHECK(!c.active());
  CHECK(c.env == 0.0f);
  CHECK(minLevel(c) == 1.0f);
  c.tick(p, kDt);
  CHECK(c.active());
  CHECK(c.env > 0.0f);
}

TEST_CASE("sweep_gradient: deterministic, and a NaN knob holds the middle", "[sweep_gradient]") {
  Params p;
  p.trig_side = TrigRandom;
  Core a, b;
  for (int i = 0; i < 600; ++i) {
    if (i % 50 == 3) { a.fire(); b.fire(); }
    p.sweep = 0.5f + 0.5f * std::sin(i * 0.01f);
    a.tick(p, kDt);
    b.tick(p, kDt);
  }
  for (int i = 0; i < kCells; ++i) CHECK(a.level[i] == b.level[i]);

  Params n;
  n.sweep = std::numeric_limits<float>::quiet_NaN();
  Core c;
  run(c, n, 1.0f);
  CHECK_THAT(c.sweep_out, WithinAbs(0.5f, 1e-6f));
  CHECK(minLevel(c) > 0.99f);
}

TEST_CASE("sweep_gradient: the knob's level never shapes the wave", "[sweep_gradient]") {
  // The same gesture fired over a knob at home, dimmed, and at either end:
  // frame for frame, the whole envelope, the grid is identical.
  Params p;
  p.trig_side = TrigUpper;
  const float knobs[] = {0.5f, 0.8f, 0.0f, 1.0f};
  Core ref;
  Core c[3];
  p.sweep = knobs[0];
  run(ref, p, 1.0f);
  for (int k = 0; k < 3; ++k) { Params q = p; q.sweep = knobs[k + 1]; run(c[k], q, 1.0f); }
  ref.fire();
  for (Core& x : c) x.fire();
  int frames = 0;
  while (true) {
    p.sweep = knobs[0];
    ref.tick(p, kDt);
    for (int k = 0; k < 3; ++k) { Params q = p; q.sweep = knobs[k + 1]; c[k].tick(q, kDt); }
    if (!ref.active()) break;
    ++frames;
    for (int k = 0; k < 3; ++k)
      for (int i = 0; i < kCells; ++i) REQUIRE_THAT(c[k].level[i], WithinAbs(ref.level[i], 1e-6f));
  }
  CHECK(frames > 60);
}

TEST_CASE("sweep_gradient: a trigger shows its whole wave from either end", "[sweep_gradient]") {
  for (const float knob : {0.0f, 1.0f}) {
    Params p;
    p.trig_side = TrigUpper;
    p.sweep = knob;
    Core c;
    run(c, p, 1.0f);
    REQUIRE(c.mean_level < 1e-3f);

    c.fire();
    c.tick(p, kDt);
    // It takes the grid over: lit, from the first frame.
    CHECK(minLevel(c) > 0.99f);
    // The choke rolls up the bars...
    run(c, p, 0.2f);
    CHECK(rowMean(c, 0) < 0.5f);
    CHECK(rowMean(c, 9) > rowMean(c, 0) + 0.3f);
    // ...and back: full again before the knob takes over.
    // The top goes out behind the bottom, then both come back up — the top
    // too, before the knob's black is let back in.
    float bottom = 0.0f, top = 0.0f;
    bool top_dark = false;
    while (c.active()) {
      c.tick(p, kDt);
      bottom = rowMean(c, 0) > bottom ? rowMean(c, 0) : bottom;
      if (rowMean(c, 9) < 0.3f) top_dark = true;
      if (top_dark) top = rowMean(c, 9) > top ? rowMean(c, 9) : top;
    }
    CHECK(top_dark);
    CHECK(bottom > 0.95f);
    CHECK(top > 0.8f);
    // Then the knob's black rolls back in.
    run(c, p, 2.0f);
    CHECK(c.mean_level < 1e-3f);
  }
}
