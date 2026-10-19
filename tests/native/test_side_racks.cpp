// test_side_racks.cpp — goldens for `source.light.side_racks`'s core.
//
// Host-free: side_racks_core.h carries no effect ABI and no transport (the
// effect hands it step crossings), so every gesture and every hit is played at
// an exact dt.

#include "sketch/side_racks_core.h"

#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

#include <cmath>
#include <set>

using Catch::Matchers::WithinAbs;
using namespace side_racks_core;

namespace {

constexpr float kDt = 1.0f / 120.0f;

void run(Core& c, const Params& p, float seconds, float dt = kDt) {
  const int n = (int)std::lround(seconds / dt);
  for (int i = 0; i < n; ++i) c.tick(p, dt);
}

/// Move the knob linearly from where it is to `to` over `seconds`.
void glide(Core& c, Params& p, float to, float seconds) {
  const float from = p.knob.sweep;
  const int n = (int)std::lround(seconds / kDt);
  for (int i = 1; i <= n; ++i) {
    p.knob.sweep = from + (to - from) * (float)i / (float)n;
    c.tick(p, kDt);
  }
}

float cellMul(const Core& c, int side, int bar, int seg) {
  return c.mul[rack_map::cellIndex(side, bar, seg)];
}

}  // namespace

// --- The map ---------------------------------------------------------------

TEST_CASE("side_racks: the rig's geometry and the compact grid", "[side_racks]") {
  // Segment 0 is the outer end on both sides; reach is a distance from center.
  CHECK_THAT(rack_map::segmentReach(0), WithinAbs(2.0f - 0.5f / 60.0f, 1e-5));
  CHECK_THAT(rack_map::segmentReach(59), WithinAbs(1.0f + 0.5f / 60.0f, 1e-5));
  CHECK(rack_map::segmentX(0, 0) < -1.9f);
  CHECK(rack_map::segmentX(1, 0) > 1.9f);
  // The grid reads like the stage: left outer end top-left for the top bar,
  // right outer end at the far right, inner ends meeting in the middle.
  CHECK(rack_map::gridIndex(0, 2, 0) == 0);
  CHECK(rack_map::gridIndex(1, 2, 0) == 119);
  CHECK(rack_map::gridIndex(0, 2, 59) == 59);
  CHECK(rack_map::gridIndex(1, 2, 59) == 60);
  CHECK(rack_map::gridIndex(0, 0, 0) == 2 * 120);
  std::set<int> seen;
  for (int s = 0; s < 2; ++s)
    for (int b = 0; b < 3; ++b)
      for (int g = 0; g < 60; ++g) seen.insert(rack_map::gridIndex(s, b, g));
  CHECK(seen.size() == 360);
}

// --- 1. The sweep ----------------------------------------------------------

TEST_CASE("side_racks: tilt 0 keeps the bars in unison through a whole sweep", "[side_racks]") {
  Params p;
  p.tilt = 0.0f;
  Core c;
  const float path[] = {1.0f, 0.5f, 0.0f, 0.5f, 0.8f, 0.5f};
  for (float to : path) {
    const float from = p.knob.sweep;
    for (int i = 1; i <= 40; ++i) {
      p.knob.sweep = from + (to - from) * i / 40.0f;
      c.tick(p, kDt);
      CHECK(c.bar_level[0] == c.bar_level[1]);
      CHECK(c.bar_level[1] == c.bar_level[2]);
    }
  }
}

TEST_CASE("side_racks: full tilt in motion lights the bar the front is passing", "[side_racks]") {
  Params p;
  p.tilt = 1.0f;
  p.knob.sense = 0.5f;   // a steady 1 range/s reads as full speed
  Core c;
  run(c, p, 0.3f);
  // Where in the fade the spot reaches the middle bar.
  const float q = 1.0f / (2.0f + kSpotExit * p.focus);
  const float target = 0.5f + 0.5f * (p.knob.deadzone + q * (1.0f - p.knob.deadzone));
  glide(c, p, target, (target - 0.5f) / 1.0f);
  CHECK_THAT(c.front, WithinAbs(1.0f, 0.15f));
  CHECK(c.tilt_now > 0.9f);
  CHECK(c.bar_level[1] > 0.85f);
  CHECK(c.bar_level[0] < 0.25f);
  CHECK(c.bar_level[2] < 0.25f);
}

TEST_CASE("side_racks: at rest the tilt falls back to a soft gradient", "[side_racks]") {
  Params p;
  p.tilt = 1.0f;
  Core c;
  const float target = 0.5f + 0.5f * (p.knob.deadzone + 0.5f * (1.0f - p.knob.deadzone));
  glide(c, p, target, 0.4f);
  run(c, p, 2.0f);
  CHECK(c.knob.speed == 0.0f);
  CHECK_THAT(c.tilt_now, WithinAbs(p.rest_tilt, 1e-4));
  // The bars the front has passed are darker; none is out and none at full.
  CHECK(c.bar_level[0] < c.bar_level[1]);
  CHECK(c.bar_level[1] < c.bar_level[2]);
  CHECK(c.bar_level[0] > 0.2f);
  CHECK(c.bar_level[2] < 1.0f);
}

TEST_CASE("side_racks: the deadzone ignores tilt, the ends choke", "[side_racks]") {
  Params p;
  p.tilt = 1.0f;
  Core c;
  // Riding inside the deadzone, briskly: full, all three.
  for (int k = 0; k < 4; ++k) {
    glide(c, p, 0.5f + 0.2f, 0.1f);
    for (float v : c.bar_level) CHECK_THAT(v, WithinAbs(1.0f, 0.02f));
    glide(c, p, 0.5f - 0.2f, 0.2f);
    for (float v : c.bar_level) CHECK_THAT(v, WithinAbs(1.0f, 0.02f));
  }
  // Out to either end and parked: black, tilted or not.
  for (float tilt : {0.0f, 1.0f}) {
    p.tilt = tilt;
    glide(c, p, 1.0f, 1.0f);
    run(c, p, 1.0f);
    for (float v : c.bar_level) CHECK(v < 0.02f);
    glide(c, p, 0.0f, 2.0f);
    run(c, p, 1.0f);
    for (float v : c.bar_level) CHECK(v < 0.02f);
    glide(c, p, 0.5f, 1.0f);
  }
}

TEST_CASE("side_racks: a fast return bounces past full, a hard pass throws a tail", "[side_racks]") {
  Params p;
  p.tilt = 0.0f;
  Core c;
  glide(c, p, 1.0f, 1.0f);
  run(c, p, 1.0f);
  float peak = 0.0f;
  glide(c, p, 0.5f, 0.08f);
  for (int i = 0; i < 30; ++i) {
    c.tick(p, kDt);
    peak = std::max(peak, c.bar_level[0]);
  }
  CHECK(peak > 1.05f);
  run(c, p, 1.0f);
  CHECK_THAT(c.bar_level[0], WithinAbs(1.0f, 1e-3));

  // A hard pass through the middle into the choke: the roll lands on all
  // three at once, then rolls one at a time and dies out.
  glide(c, p, 0.0f, 0.12f);
  glide(c, p, 1.0f, 0.15f);
  CHECK(c.knob.release > 0.3f);
  bool all = true;
  for (float g : c.tail_gain) all = all && g > 0.0f;
  CHECK(all);
  int lit_frames = 0;
  for (int i = 0; i < 60; ++i) {
    c.tick(p, kDt);
    int n = 0;
    for (float g : c.tail_gain) n += g >= 1.0f ? 1 : 0;
    CHECK(n <= 3);
    lit_frames += n > 0 ? 1 : 0;
  }
  CHECK(lit_frames > 5);
  run(c, p, p.knob.ring_time + 0.2f);
  CHECK(c.knob.release == 0.0f);
  for (float g : c.tail_gain) CHECK(g == 0.0f);
}

// --- 2. The primary --------------------------------------------------------

TEST_CASE("side_racks: Flam ducks with a smooth attack, a sustain and a release", "[side_racks]") {
  Params p;
  p.primary_mode = PrimaryFlam;
  p.flam_rate = 0.0f;  // a plain duck to read the envelope
  p.flam_amount = 1.0f;
  Core c;
  run(c, p, 0.1f);
  CHECK(c.mul[0] == 1.0f);

  p.primary = 1.0f;
  float prev = 0.0f;
  const int attack_frames = (int)std::lround(p.env_attack / kDt);
  for (int i = 0; i < attack_frames; ++i) {
    c.tick(p, kDt);
    CHECK(c.primary_env >= prev);
    prev = c.primary_env;
  }
  c.tick(p, kDt);
  CHECK(c.primary_env > 0.95f);
  run(c, p, p.env_decay + 0.2f);
  CHECK_THAT(c.primary_env, WithinAbs(p.env_sustain, 1e-4));
  CHECK_THAT(c.mul[0], WithinAbs(1.0f - p.env_sustain, 1e-4));

  p.primary = 0.0f;
  c.tick(p, kDt);
  CHECK(c.primary_env > 0.0f);
  run(c, p, p.env_release + 0.05f);
  CHECK(c.primary_env == 0.0f);
  CHECK(c.mul[0] == 1.0f);

  // A gate shorter than the attack is still a full duck.
  p.primary = 1.0f;
  c.tick(p, kDt);
  p.primary = 0.0f;
  float peak = 0.0f;
  for (int i = 0; i < 20; ++i) {
    c.tick(p, kDt);
    peak = std::max(peak, c.primary_env);
  }
  CHECK(peak > 0.95f);
}

TEST_CASE("side_racks: the flam's shudder caps at one frame on, one frame off", "[side_racks]") {
  Params p;
  p.primary_mode = PrimaryFlam;
  p.flam_amount = 1.0f;
  p.env_sustain = 1.0f;
  p.flam_floor = 0.25f;
  p.flam_rate = 200.0f;  // past the cap
  Core c;
  const float dt = 1.0f / 60.0f;
  run(c, p, 0.2f, dt);
  p.primary = 1.0f;
  run(c, p, 0.5f, dt);
  float last = cellMul(c, 0, 0, 0);
  for (int i = 0; i < 12; ++i) {
    c.tick(p, dt);
    const float m = cellMul(c, 0, 0, 0);
    CHECK(m != last);
    CHECK((std::fabs(m - 0.0f) < 1e-4f || std::fabs(m - 0.75f) < 1e-4f));
    last = m;
  }
}

TEST_CASE("side_racks: the Wave crosses the gap, eats, crushes, and recovers", "[side_racks]") {
  Params p;
  p.primary_mode = PrimaryWave;
  p.wave_start = 0.0f;
  p.wave_speed = 4.0f;
  p.wave_amount = 1.0f;
  Core c;
  run(c, p, 0.1f);
  p.primary = 1.0f;
  c.tick(p, kDt);   // born this frame
  p.primary = 0.0f;
  // The inner end is 1 m out: nothing there for the first 0.25 s.
  const int inner = 59, outer = 0;
  run(c, p, 0.25f - 0.05f);
  CHECK(cellMul(c, 0, 1, inner) == 1.0f);
  CHECK(cellMul(c, 1, 1, inner) == 1.0f);
  run(c, p, 0.1f);
  CHECK(cellMul(c, 0, 1, inner) < 0.6f);
  CHECK(cellMul(c, 0, 1, outer) == 1.0f);   // not there yet
  // Left and right are mirror images, cut and sparkle alike.
  for (int b = 0; b < 3; ++b)
    for (int s = 0; s < 60; ++s) {
      const int l = rack_map::cellIndex(0, b, s), r = rack_map::cellIndex(1, b, s);
      CHECK(c.mul[l] == c.mul[r]);
      for (int k = 0; k < 3; ++k) CHECK(c.add[l][k] == c.add[r][k]);
    }
  // On the front it is blocky: a crush block reads one value.
  run(c, p, 0.12f);
  const float r_front = p.wave_start + p.wave_speed * c.pulses[0].age;
  int blocks_checked = 0;
  for (int blk = 0; blk < 60 / p.wave_crush; ++blk) {
    const int s0 = 59 - blk * p.wave_crush;
    const float reach = rack_map::segmentReach(s0);
    if (std::fabs(reach - r_front) > p.wave_width * 0.5f) continue;
    for (int j = 1; j < p.wave_crush; ++j)
      CHECK(cellMul(c, 0, 0, s0 - j) == cellMul(c, 0, 0, s0));
    ++blocks_checked;
  }
  CHECK(blocks_checked > 0);
  // Long after it has passed, the light is back.
  run(c, p, 3.0f);
  for (int s = 0; s < 60; ++s) CHECK(cellMul(c, 0, 1, s) > 0.999f);
  for (const Pulse& pu : c.pulses) CHECK(!pu.live);
}

TEST_CASE("side_racks: the Wave's sparkle is seeded and deterministic", "[side_racks]") {
  auto sparkleAt = [](int seed) {
    Params p;
    p.primary_mode = PrimaryWave;
    p.seed = seed;
    p.wave_sparkle_rate = 60.0f;
    Core c;
    c.tick(p, kDt);
    p.primary = 1.0f;
    c.tick(p, kDt);
    p.primary = 0.0f;
    run(c, p, 0.08f);
    float sum = 0.0f;
    std::set<int> lit;
    for (int i = 0; i < kCells; ++i)
      if (c.add[i][1] > 0.0f) { sum += c.add[i][1]; lit.insert(i); }
    return std::make_pair(sum, lit);
  };
  const auto a = sparkleAt(3), b = sparkleAt(3), d = sparkleAt(11);
  CHECK(a.first > 0.0f);
  CHECK(a.first == b.first);
  CHECK(a.second == b.second);
  CHECK(a.second != d.second);
}

TEST_CASE("side_racks: Beating keeps each bar one colour and steps per frame", "[side_racks]") {
  Params p;
  p.primary_mode = PrimaryBeating;
  p.beat_amount = 1.0f;
  p.env_sustain = 1.0f;
  p.beat_spread_b = 0.33f;
  Core c;
  p.primary = 1.0f;
  run(c, p, 0.5f);
  for (int b = 0; b < 3; ++b) {
    const float m = cellMul(c, 0, b, 0);
    for (int side = 0; side < 2; ++side)
      for (int s = 0; s < 60; ++s) CHECK(cellMul(c, side, b, s) == m);
  }

  // Frame-locked: at exactly the frame rate each oscillator advances a whole
  // cycle per frame, so with no detune the picture stands still.
  Params q = p;
  q.beat_rate = 60.0f;
  q.beat_detune = 0.0f;
  Core f;
  const float dt = 1.0f / 60.0f;
  run(f, q, 0.6f, dt);
  const float m0 = cellMul(f, 0, 1, 0);
  for (int i = 0; i < 20; ++i) {
    f.tick(q, dt);
    CHECK_THAT(cellMul(f, 0, 1, 0), WithinAbs(m0, 1e-3));
  }
  // At Nyquist, detuned by 1 Hz: frame-to-frame flicker whose depth beats
  // once a second.
  q.beat_rate = 30.0f;
  q.beat_detune = 1.0f;
  q.beat_spread_a = q.beat_spread_b = 0.0f;
  Core g;
  run(g, q, 0.6f, dt);
  float lo = 1.0f, hi = 0.0f;
  for (int i = 0; i < 60; ++i) {
    g.tick(q, dt);
    lo = std::min(lo, cellMul(g, 0, 0, 0));
    hi = std::max(hi, cellMul(g, 0, 0, 0));
  }
  CHECK(lo < 0.1f);
  CHECK(hi > 0.9f);
  const float at = cellMul(g, 0, 0, 0);
  run(g, q, 1.0f, dt);
  CHECK_THAT(cellMul(g, 0, 0, 0), WithinAbs(at, 0.02f));
}

// --- 3. The secondary ------------------------------------------------------

TEST_CASE("side_racks: each step deals every channel a bar on each side", "[side_racks]") {
  Params p;
  std::set<int> deals;
  for (int seed : {0, 1, 7}) {
    p.seed = seed;
    Core c;
    for (int k = 0; k < 24; ++k) {
      c.tick(p, kDt);
      for (int side = 0; side < 2; ++side) {
        std::set<int> bars;
        for (int ch = 0; ch < 3; ++ch) bars.insert(c.barFor(side, ch));
        CHECK(bars.size() == 3);
      }
      deals.insert(c.perm[0] * 6 + c.perm[1]);
      c.advanceStep(1);
    }
  }
  CHECK(deals.size() > 6);
  // The same seed and step, the same deal.
  Core a, b;
  p.seed = 5;
  a.advanceStep(9);
  b.advanceStep(9);
  a.tick(p, kDt);
  b.tick(p, kDt);
  CHECK(a.perm[0] == b.perm[0]);
  CHECK(a.perm[1] == b.perm[1]);
}

TEST_CASE("side_racks: single hits alternate sides on one shared alternator", "[side_racks]") {
  Params p;
  Core c;
  c.tick(p, kDt);
  auto hit = [&](std::initializer_list<int> chs) {
    for (int ch : chs) p.sec[ch] = 1.0f;
    c.tick(p, kDt);
    for (int ch = 0; ch < 3; ++ch) p.sec[ch] = 0.0f;
  };
  auto litSide = [&](int ch) {
    int sides = 0;
    for (int side = 0; side < 2; ++side)
      if (c.flash[side][c.barFor(side, ch)] > 0.99f && c.flash_ch[side][c.barFor(side, ch)] == ch)
        sides |= 1 << side;
    return sides;
  };
  hit({0});
  CHECK(litSide(0) == 0b01);
  run(c, p, 0.5f);
  hit({0});
  CHECK(litSide(0) == 0b10);
  run(c, p, 0.5f);
  hit({1});   // shared: the next single hit goes left whatever the channel
  CHECK(litSide(1) == 0b01);
  run(c, p, 0.5f);
  // Two landing together take opposite sides.
  hit({0, 2});
  CHECK(litSide(0) != 0);
  CHECK(litSide(2) != 0);
  CHECK((litSide(0) | litSide(2)) == 0b11);
}

TEST_CASE("side_racks: all three together is a heavy hit on both sides", "[side_racks]") {
  Params p;
  Core c;
  c.tick(p, kDt);
  p.sec[0] = p.sec[1] = p.sec[2] = 1.0f;
  c.tick(p, kDt);
  for (int side = 0; side < 2; ++side)
    for (int bar = 0; bar < 3; ++bar) CHECK(c.flash[side][bar] > 0.99f);
  // Staggered past the window, it is not.
  Core d;
  p.sec[0] = p.sec[1] = p.sec[2] = 0.0f;
  d.tick(p, kDt);
  p.sec[0] = 1.0f;
  d.tick(p, kDt);
  run(d, p, 0.1f);
  p.sec[1] = p.sec[2] = 1.0f;
  d.tick(p, kDt);
  int lit = 0;
  for (int side = 0; side < 2; ++side)
    for (int bar = 0; bar < 3; ++bar) lit += d.flash[side][bar] > 0.99f ? 1 : 0;
  CHECK(lit == 2);
  // The flashes fall over the decay.
  run(c, p, p.sec_decay + 0.02f);
  for (int side = 0; side < 2; ++side)
    for (int bar = 0; bar < 3; ++bar) CHECK(c.flash[side][bar] == 0.0f);
}

// --- 4. Lift and strobe ----------------------------------------------------

TEST_CASE("side_racks: lift adds the accent; a full strobe alternates frames", "[side_racks]") {
  Params p;
  p.color[0] = p.color[1] = p.color[2] = 0.0f;
  p.lift_color[0] = 0.2f; p.lift_color[1] = 0.4f; p.lift_color[2] = 0.6f;
  p.lift = 0.5f;
  Core c;
  c.tick(p, kDt);
  auto cells = c.cells(p);
  CHECK_THAT(cells.rgb[0][0], WithinAbs(0.1f, 1e-5));
  CHECK_THAT(cells.rgb[0][2], WithinAbs(0.3f, 1e-5));

  p.lift = 0.0f;
  p.strobe = 1.0f;
  const float dt = 1.0f / 60.0f;
  Core s;
  bool expect = true;
  for (int i = 0; i < 20; ++i) {
    s.tick(p, dt);
    CHECK(s.strobe_on == expect);
    const auto px = s.cells(p);
    CHECK((px.rgb[100][2] > 0.5f) == expect);
    expect = !expect;
  }
  // Half depth is slower: a free-running square wave between min and max.
  p.strobe = 0.5f;
  Core h;
  int flips = 0;
  bool last = true;
  for (int i = 0; i < 600; ++i) {
    h.tick(p, kDt);
    if (h.strobe_on != last) ++flips;
    last = h.strobe_on;
  }
  const float hz = p.strobe_min_hz * std::pow(p.strobe_max_hz / p.strobe_min_hz, 0.5f);
  CHECK(std::fabs(flips - 2.0f * hz * 5.0f) <= 3.0f);
}

TEST_CASE("side_racks: Mix Post moves the mix-ins over the strobe", "[side_racks]") {
  Params p;
  p.primary_mode = PrimaryFlam;
  p.flam_rate = 0.0f;
  p.flam_amount = 1.0f;
  p.env_sustain = 1.0f;
  p.strobe = 1.0f;
  const float dt = 1.0f / 60.0f;
  for (float post : {0.0f, 1.0f}) {
    p.mix_post = post;
    p.primary = 0.0f;
    Core c;
    c.tick(p, dt);
    p.primary = 1.0f;
    run(c, p, 0.5f, dt);
    if (!c.strobe_on) c.tick(p, dt);
    REQUIRE(c.strobe_on);
    const auto px = c.cells(p);
    if (post == 0.0f) CHECK(px.rgb[0][1] > 0.5f);   // the strobe flash survives the duck
    else CHECK(px.rgb[0][1] == 0.0f);              // the duck cuts it
  }
}

TEST_CASE("side_racks: a refresh at zero dt ages nothing", "[side_racks]") {
  Params p;
  p.strobe = 1.0f;
  p.primary_mode = PrimaryFlam;
  p.flam_rate = 200.0f;
  p.primary = 1.0f;
  Core c;
  run(c, p, 0.3f, 1.0f / 60.0f);
  const bool on = c.strobe_on;
  const float m = c.mul[0];
  const float env = c.primary_env;
  for (int i = 0; i < 5; ++i) c.tick(p, 0.0f);
  CHECK(c.strobe_on == on);
  CHECK(c.mul[0] == m);
  CHECK(c.primary_env == env);
}
