// test_led_bars.cpp — goldens for the house rig's pixel map.
//
// Host-free, like the show headers next to it: led_bars.h knows nothing about
// the effect ABI, so a frame can be walked past the bars here at an exact `a`
// and an exact dt, with no wasm bundle, no executor and no GPU.
//
// That matters more than usual for this one. The map's whole job is to be
// EXACT — a segment is a physical LED and either shows the colour it was asked
// for or does not — and nothing about a rendered frame can tell you whether
// segment 6 belongs to the middle floor or the top one.

#include "sketch/led_bars.h"

#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

#include <vector>

using Catch::Matchers::WithinAbs;
using namespace led_bars;

namespace {

/// The colour of one cell, as a triple.
struct Rgb { float r, g, b; };

Rgb cellOf(const Cells& c, int bar, int seg) {
  const int i = cellIndex(bar, seg);
  return {c.rgb[i][0], c.rgb[i][1], c.rgb[i][2]};
}

Rgb barOf(const Bars& b, int bar) {
  return {b.rgb[bar][0], b.rgb[bar][1], b.rgb[bar][2]};
}

void expectRgb(const Rgb& got, const Rgb& want, float eps = 1e-5f) {
  CHECK_THAT(got.r, WithinAbs(want.r, eps));
  CHECK_THAT(got.g, WithinAbs(want.g, eps));
  CHECK_THAT(got.b, WithinAbs(want.b, eps));
}

/// The three_planes tower, in the three primaries so a floor is unmistakable.
const float kTowerRgb[kLayers][3] = {{1.f, 0.f, 0.f},    // bottom
                                     {0.f, 1.f, 0.f},    // middle
                                     {0.f, 0.f, 1.f}};   // top

/// A tunnel with one frame in it, at apparent size `a`.
WallScene oneFrame(float a, float level = 1.0f) {
  WallScene s;
  s.a[0] = a;
  s.level[0] = level;
  s.live[0] = true;
  s.rgb[0][0] = 1.f; s.rgb[0][1] = 1.f; s.rgb[0][2] = 1.f;
  return s;
}

constexpr float kSpacing = 1.5f;
constexpr float kWidth = 0.45f;

}  // namespace

// --- The tower's segments -------------------------------------------------

TEST_CASE("ten segments split three / three / four, bottom to top", "[led_bars]") {
  // The spec, spelled out rather than derived: segment 0 is the LOWEST LED.
  const int want[kSegments] = {0, 0, 0, 1, 1, 1, 2, 2, 2, 2};
  for (int seg = 0; seg < kSegments; ++seg)
    CHECK(layerOfSegment(seg) == want[seg]);
}

TEST_CASE("the spare segment goes to the TOP floor", "[led_bars]") {
  int count[kLayers] = {};
  for (int seg = 0; seg < kSegments; ++seg) count[layerOfSegment(seg)]++;
  CHECK(count[0] == 3);
  CHECK(count[1] == 3);
  CHECK(count[2] == 4);
  // Ten over three leaves one over, and it lands at the top — the part of a
  // meter you actually read. If this ever flips to the bottom, the tower will
  // still look plausible on the rig and will be wrong.
  CHECK(count[2] > count[0]);
}

TEST_CASE("a segment index off either end clamps rather than reading out", "[led_bars]") {
  CHECK(layerOfSegment(-1) == 0);
  CHECK(layerOfSegment(kSegments) == kLayers - 1);
  CHECK(layerOfSegment(9999) == kLayers - 1);
}

// --- The tower on the bars ------------------------------------------------

TEST_CASE("every bar shows the same tower", "[led_bars]") {
  const float level[kLayers] = {1.0f, 0.5f, 0.25f};
  const Cells c = planeCells(kTowerRgb, level);
  for (int bar = 1; bar < kBars; ++bar)
    for (int seg = 0; seg < kSegments; ++seg)
      expectRgb(cellOf(c, bar, seg), cellOf(c, 0, seg));
}

TEST_CASE("each floor lights its own segments at its own level", "[led_bars]") {
  const float level[kLayers] = {1.0f, 0.5f, 0.25f};
  const Cells c = planeCells(kTowerRgb, level);

  // Bottom three: red at full.
  for (int seg = 0; seg <= 2; ++seg) expectRgb(cellOf(c, 0, seg), {1.f, 0.f, 0.f});
  // Middle three: green at half.
  for (int seg = 3; seg <= 5; ++seg) expectRgb(cellOf(c, 0, seg), {0.f, 0.5f, 0.f});
  // Top FOUR: blue at a quarter.
  for (int seg = 6; seg <= 9; ++seg) expectRgb(cellOf(c, 0, seg), {0.f, 0.f, 0.25f});
}

TEST_CASE("a dark floor is dark, and takes no colour from its neighbours",
          "[led_bars]") {
  const float level[kLayers] = {1.0f, 0.0f, 1.0f};
  const Cells c = planeCells(kTowerRgb, level);
  for (int seg = 3; seg <= 5; ++seg) expectRgb(cellOf(c, 0, seg), {0.f, 0.f, 0.f});
  // And the floors either side of the hole are untouched — the cells are
  // independent, so nothing bleeds across the boundary the way it would if
  // this were a picture being filtered.
  expectRgb(cellOf(c, 0, 2), {1.f, 0.f, 0.f});
  expectRgb(cellOf(c, 0, 6), {0.f, 0.f, 1.f});
}

// --- Where the bars stand -------------------------------------------------

TEST_CASE("the columns run left to right across the room", "[led_bars]") {
  // bar 0 = left outer, 1 = left inner | screen | 2 = right inner, 3 = right outer
  CHECK(barSide(0) == 0);
  CHECK(barSide(1) == 0);
  CHECK(barSide(2) == 1);
  CHECK(barSide(3) == 1);

  CHECK(barRing(0) == 1);
  CHECK(barRing(1) == 0);
  CHECK(barRing(2) == 0);
  CHECK(barRing(3) == 1);

  // Mirrored about the centre line, which is the whole reason the ordering is
  // outer-inner | inner-outer rather than inner-outer twice.
  CHECK(barRing(0) == barRing(kBars - 1));
  CHECK(barRing(1) == barRing(kBars - 2));
}

TEST_CASE("both rings stand beyond the back wall's edge", "[led_bars]") {
  const float inner = barApparentSize(0, kSpacing);
  const float outer = barApparentSize(1, kSpacing);
  // a = 1 IS the wall's edge. A bar at or inside it would be lit while the
  // frame was still a rectangle on the screen, which is not where it stands.
  CHECK(inner > 1.0f);
  CHECK(outer > inner);
  CHECK_THAT(outer - inner, WithinAbs(kSpacing, 1e-5f));
}

TEST_CASE("the spacing knob moves the rings and nothing else", "[led_bars]") {
  CHECK(barApparentSize(0, 3.0f) > barApparentSize(0, 1.0f));
  // Degenerate spacing still yields an ordering rather than a division by zero.
  CHECK(barApparentSize(1, 0.0f) > barApparentSize(0, 0.0f));
}

// --- The pass -------------------------------------------------------------

TEST_CASE("a bar is brightest when the frame is exactly on it", "[led_bars]") {
  const float a_bar = barApparentSize(0, kSpacing);
  const float on = passResponse(a_bar, a_bar, 0.5f);
  CHECK_THAT(on, WithinAbs(1.0f, 1e-5f));
  CHECK(passResponse(a_bar + 0.3f, a_bar, 0.5f) < on);
  CHECK(passResponse(a_bar - 0.3f, a_bar, 0.5f) < on);
  // Symmetric: approaching and leaving look the same, so the pass reads as one
  // event rather than as an attack with a different release.
  CHECK_THAT(passResponse(a_bar + 0.3f, a_bar, 0.5f),
             WithinAbs(passResponse(a_bar - 0.3f, a_bar, 0.5f), 1e-6f));
}

TEST_CASE("a wider pass keeps the bar lit further out", "[led_bars]") {
  const float a_bar = 2.5f;
  CHECK(passResponse(a_bar + 1.0f, a_bar, 1.0f) >
        passResponse(a_bar + 1.0f, a_bar, 0.3f));
}

TEST_CASE("the rig fires as a pair opening outward", "[led_bars]") {
  // Walk one frame out through the room and note where each ring peaks.
  int inner_peak = -1, outer_peak = -1;
  float inner_best = 0.f, outer_best = 0.f;
  const int kSteps = 400;
  for (int i = 0; i < kSteps; ++i) {
    const float a = 1.0f + 8.0f * float(i) / float(kSteps - 1);
    const Bars b = wallBars(oneFrame(a), kSpacing, kWidth);
    if (b.rgb[1][0] > inner_best) { inner_best = b.rgb[1][0]; inner_peak = i; }
    if (b.rgb[0][0] > outer_best) { outer_best = b.rgb[0][0]; outer_peak = i; }
  }
  REQUIRE(inner_peak >= 0);
  REQUIRE(outer_peak >= 0);
  // Inner first, then outer. This is the whole gesture: the frame passes the
  // pair nearest the screen and then the pair beyond them, so on the rig the
  // lights open away from the picture.
  CHECK(inner_peak < outer_peak);
  CHECK_THAT(inner_best, WithinAbs(1.0f, 0.02f));
  CHECK_THAT(outer_best, WithinAbs(1.0f, 0.02f));
}

TEST_CASE("the two sides always agree", "[led_bars]") {
  // The room is symmetric about its centre line and the frames are axis
  // aligned, so there is nothing that could tell the left half from the right.
  for (int i = 0; i <= 40; ++i) {
    const float a = 1.0f + 0.25f * float(i);
    const Bars b = wallBars(oneFrame(a), kSpacing, kWidth);
    CHECK_THAT(b.rgb[0][0], WithinAbs(b.rgb[3][0], 1e-6f));
    CHECK_THAT(b.rgb[1][0], WithinAbs(b.rgb[2][0], 1e-6f));
  }
}

TEST_CASE("a frame that is not live is not in the room", "[led_bars]") {
  WallScene s = oneFrame(barApparentSize(0, kSpacing));
  REQUIRE(wallBars(s, kSpacing, kWidth).rgb[1][0] > 0.5f);
  s.live[0] = false;
  const Bars b = wallBars(s, kSpacing, kWidth);
  for (int bar = 0; bar < kBars; ++bar) expectRgb(barOf(b, bar), {0.f, 0.f, 0.f});
}

TEST_CASE("two frames on one bar add", "[led_bars]") {
  // LEDs are lights. There is no occlusion out here and no fill: a bar has
  // nothing behind it, so a frame set to cut a hole in the picture lights the
  // rig like any other.
  const float a_bar = barApparentSize(0, kSpacing);
  WallScene s;
  for (int q = 0; q < 2; ++q) {
    s.a[q] = a_bar;
    s.level[q] = 0.5f;
    s.live[q] = true;
    s.rgb[q][0] = 1.f;
  }
  CHECK_THAT(wallBars(s, kSpacing, kWidth).rgb[1][0], WithinAbs(1.0f, 1e-5f));
}

TEST_CASE("a dim frame lights the bar dimly", "[led_bars]") {
  const float a_bar = barApparentSize(0, kSpacing);
  const Bars b = wallBars(oneFrame(a_bar, 0.25f), kSpacing, kWidth);
  CHECK_THAT(b.rgb[1][0], WithinAbs(0.25f, 1e-5f));
}

TEST_CASE("a bar takes the frame's own colour", "[led_bars]") {
  WallScene s = oneFrame(barApparentSize(0, kSpacing));
  s.rgb[0][0] = 0.2f; s.rgb[0][1] = 0.8f; s.rgb[0][2] = 0.4f;
  expectRgb(barOf(wallBars(s, kSpacing, kWidth), 1), {0.2f, 0.8f, 0.4f}, 1e-4f);
}

// --- The envelope ---------------------------------------------------------

TEST_CASE("a hit lands instantly and falls back afterwards", "[led_bars]") {
  const float dt = 1.0f / 60.0f;
  float v = peakHold(0.f, 1.f, dt, 0.12f);
  CHECK_THAT(v, WithinAbs(1.0f, 1e-6f));   // no attack: a hit is a hit
  const float after_one = peakHold(v, 0.f, dt, 0.12f);
  CHECK(after_one < 1.0f);
  CHECK(after_one > 0.5f);
  // Monotone down, and it gets there.
  float prev = after_one;
  for (int i = 0; i < 120; ++i) {
    const float next = peakHold(prev, 0.f, dt, 0.12f);
    CHECK(next <= prev);
    prev = next;
  }
  CHECK(prev < 0.01f);
}

TEST_CASE("the envelope never sits below what the bar sees right now",
          "[led_bars]") {
  CHECK_THAT(peakHold(0.1f, 0.8f, 1.0f, 0.12f), WithinAbs(0.8f, 1e-6f));
}

TEST_CASE("no decay means the bar follows the frame exactly", "[led_bars]") {
  CHECK_THAT(peakHold(1.0f, 0.f, 1.f / 60.f, 0.f), WithinAbs(0.f, 1e-6f));
}

TEST_CASE("a pass too fast to sample still lights the bar", "[led_bars]") {
  // The reason the envelope exists. Walk a frame out in strides big enough
  // that no single sample lands anywhere near a bar, and the rig must still
  // show the pass rather than nothing at all.
  const float stride = 2.2f;   // well past a 0.675 pass width
  float held[kBars] = {};
  float peak = 0.f;
  for (int i = 0; i < 12; ++i) {
    const float a = 1.05f + stride * float(i);
    const Bars b = wallBars(oneFrame(a), kSpacing, kWidth);
    for (int bar = 0; bar < kBars; ++bar) {
      held[bar] = peakHold(held[bar], b.rgb[bar][0], 1.f / 60.f, 0.25f);
      if (held[bar] > peak) peak = held[bar];
    }
  }
  CHECK(peak > 0.05f);
}

// --- Spreading down the bar -----------------------------------------------

TEST_CASE("a bar is one colour all the way up", "[led_bars]") {
  // The frames are axis aligned and so are the bars, so a vertical bar sees
  // the same thing at every height. The ten segments are there because the rig
  // has ten, not because there is vertical content to carry.
  Bars b;
  for (int bar = 0; bar < kBars; ++bar) {
    b.rgb[bar][0] = 0.1f * float(bar + 1);
    b.rgb[bar][1] = 0.2f;
    b.rgb[bar][2] = 0.3f;
  }
  const Cells c = barsToCells(b);
  for (int bar = 0; bar < kBars; ++bar)
    for (int seg = 0; seg < kSegments; ++seg)
      expectRgb(cellOf(c, bar, seg), barOf(b, bar));
}

TEST_CASE("cells are bar-major, segment 0 at the bottom", "[led_bars]") {
  // The packing the shader indexes with. Getting this wrong transposes the rig.
  CHECK(cellIndex(0, 0) == 0);
  CHECK(cellIndex(0, kSegments - 1) == kSegments - 1);
  CHECK(cellIndex(1, 0) == kSegments);
  CHECK(cellIndex(kBars - 1, kSegments - 1) == kCells - 1);
}
