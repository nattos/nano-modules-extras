#pragma once
/*
 * led_bars.h — the pixel map for the house rig: four vertical LED bars,
 * ten segments each.
 *
 * Both neon-quad instruments publish one of these alongside their picture, so
 * the bars standing in the room do the same thing the screen does. This header
 * owns the whole mapping — which segment belongs to what, where the bars are
 * in the scene, how brightly a passing frame lights one — and it is host-free
 * on purpose: <native/tests/test_led_bars.cpp> drives every case here at an
 * exact dt with no wasm and no GPU.
 *
 * THE GRID IS 4 WIDE AND 10 TALL, and it is a CONTROL SIGNAL, not a picture.
 * Each cell is one physical LED segment, so it is emitted as a flat block of
 * colour: anything downstream that resamples the texture — a pixel mapper
 * picking a point inside the block, a scaler averaging across it — has to land
 * on the exact colour the segment wants. A gradient would be averaged into
 * mud, which is why quantising is the default rather than an option.
 *
 * Columns run LEFT TO RIGHT across the room, the way the bars actually stand:
 *
 *      bar 0        bar 1     |  back wall  |    bar 2        bar 3
 *   (left, outer) (left, in)  |             |  (right, in) (right, outer)
 *
 * The room is symmetric about its centre line and the frames are axis-aligned,
 * so bars 0 and 3 always agree, and so do 1 and 2. That is not a shortcut — it
 * is what the geometry says, and it is what makes a frame blowing past read on
 * the rig as a pair of lights opening outward from the screen.
 */

#include <cmath>

namespace led_bars {

constexpr int kBars = 4;
constexpr int kSegments = 10;
constexpr int kCells = kBars * kSegments;
constexpr int kBarsPerSide = kBars / 2;
static_assert(kBars % 2 == 0, "the bars mirror about the room's centre line");

/// The tower three_planes shows: three floors, bottom to top.
constexpr int kLayers = 3;

/// Segments per floor, bottom to top. Ten does not divide by three, and the
/// spare segment goes to the TOP floor — the top of a meter is the part you
/// actually read, so it gets the extra resolution, and a peak that overshoots
/// has somewhere to go.
constexpr int kSegmentsPerLayer[kLayers] = {3, 3, 4};
static_assert(kSegmentsPerLayer[0] + kSegmentsPerLayer[1] + kSegmentsPerLayer[2]
                  == kSegments,
              "the floors have to tile the bar exactly");

/// Which floor segment `seg` belongs to, counted from the BOTTOM (segment 0 is
/// the lowest LED). Out-of-range segments clamp rather than read off the end.
inline int layerOfSegment(int seg) {
  if (seg < 0) return 0;
  int acc = 0;
  for (int i = 0; i < kLayers; ++i) {
    acc += kSegmentsPerLayer[i];
    if (seg < acc) return i;
  }
  return kLayers - 1;
}

/// Which side of the room bar `bar` stands on: 0 = left, 1 = right.
inline int barSide(int bar) { return bar < kBarsPerSide ? 0 : 1; }

/// How far out bar `bar` stands, 0 being the one nearest the screen. The two
/// halves mirror, so the columns read outer-inner | inner-outer across.
inline int barRing(int bar) {
  const int i = bar < kBarsPerSide ? bar : (kBars - 1 - bar);
  return kBarsPerSide - 1 - i;
}

// --- The three_walls scene ------------------------------------------------
//
// three_walls measures everything in `a`, a frame's apparent half-size in
// back-wall half-widths (see the room note in that effect). a = 1 is exactly
// the wall's edge — the instant the frame stops being a rectangle on the back
// wall and becomes two bars running down the side walls.
//
// The LED bars stand BEYOND the side walls, so they live at a > 1, and
// `spacing` is the gap between rings measured in that same unit. Anchoring
// them to `a` and nothing else is deliberate: keystone and stretch are how the
// side-wall PICTURES are drawn, and a light standing in the room does not move
// because you changed a projector's mapping.

/// Where ring `ring` stands, as an apparent size. Ring 0 is the pair closest to
/// the screen.
inline float barApparentSize(int ring, float spacing) {
  const float s = spacing > 1e-3f ? spacing : 1e-3f;
  return 1.0f + s * float(ring + 1);
}

/// How brightly a frame at apparent size `a` lights a bar standing at `a_bar`.
///
/// A Gaussian, not a hard threshold: the frame tears through this region — it
/// is moving at a constant APPARENT rate, so `a` grows exponentially and the
/// outer rings go past fastest — and a bar that only lit while the edge was
/// exactly on it would flash for less than a frame and alias into nothing. The
/// width is what gives the pass a duration.
inline float passResponse(float a, float a_bar, float width) {
  const float w = width > 1e-4f ? width : 1e-4f;
  const float t = (a - a_bar) / w;
  return std::exp(-t * t);
}

/// A peak-hold envelope: jumps straight to `inst` and falls back with time
/// constant `decay`. Without it a fast move puts the whole pass between two
/// frames and the bar never lights at all; with it, whatever the bar saw is
/// still on its way down when the next frame samples it.
inline float peakHold(float prev, float inst, float dt, float decay) {
  float v = prev;
  if (decay > 1e-4f && dt > 0.0f) v *= std::exp(-dt / decay);
  else if (dt > 0.0f) v = 0.0f;
  return v > inst ? v : inst;
}

// --- The grids ------------------------------------------------------------

/// One colour per bar, before it is spread down the segments.
struct Bars {
  float rgb[kBars][3] = {};
};

/// The whole map, bar-major: cell `bar * kSegments + seg`, segment 0 at the
/// BOTTOM of the bar.
struct Cells {
  float rgb[kCells][3] = {};
};

inline int cellIndex(int bar, int seg) { return bar * kSegments + seg; }

/// How many frames the tunnel carries. Its own constant, not the tower's
/// kLayers: the two instruments happen to have three of everything, but a
/// floor and a frame are different things and one changing must not silently
/// move the other.
constexpr int kFrames = 3;

/// three_walls' scene, as the bars see it. `level` is the normalised 0..1
/// brightness of each frame and `a` its apparent size; a frame that is not
/// `live` is not in the room at all.
struct WallScene {
  float a[kFrames] = {};
  float level[kFrames] = {};
  float rgb[kFrames][3] = {};
  bool live[kFrames] = {};
};

/// ADDITIVE across the frames, because LEDs are lights and two frames passing
/// one bar together is brighter than either alone. There is no occlusion here
/// and no fill: a bar has nothing behind it to mask, so a frame set to cut a
/// hole in the picture simply lights the rig like any other.
inline Bars wallBars(const WallScene& s, float spacing, float width) {
  Bars out;
  const float w = width * (spacing > 1e-3f ? spacing : 1e-3f);
  for (int b = 0; b < kBars; ++b) {
    const float a_bar = barApparentSize(barRing(b), spacing);
    for (int q = 0; q < kFrames; ++q) {
      if (!s.live[q]) continue;
      const float g = s.level[q] * passResponse(s.a[q], a_bar, w);
      for (int c = 0; c < 3; ++c) out.rgb[b][c] += s.rgb[q][c] * g;
    }
  }
  return out;
}

/// Spread one colour per bar down all ten of its segments.
///
/// The frames are axis-aligned and so are the bars, so a vertical bar sees the
/// same thing at every height and the segments carry no vertical content of
/// their own. They exist because the rig has ten of them — the map has to have
/// the shape of the thing it is driving, even where the picture is flat.
inline Cells barsToCells(const Bars& b) {
  Cells out;
  for (int bar = 0; bar < kBars; ++bar)
    for (int seg = 0; seg < kSegments; ++seg)
      for (int c = 0; c < 3; ++c)
        out.rgb[cellIndex(bar, seg)][c] = b.rgb[bar][c];
  return out;
}

/// three_planes' tower, as the bars see it: every bar shows the same three
/// floors stacked bottom to top, at each floor's own colour and brightness.
/// `level` is normalised 0..1 — the plane's emission through the same dimmer
/// curve the picture uses, scaled so a fully-lit plane is a fully-lit segment.
inline Cells planeCells(const float rgb[kLayers][3], const float level[kLayers]) {
  Cells out;
  for (int bar = 0; bar < kBars; ++bar) {
    for (int seg = 0; seg < kSegments; ++seg) {
      const int i = layerOfSegment(seg);
      for (int c = 0; c < 3; ++c)
        out.rgb[cellIndex(bar, seg)][c] = rgb[i][c] * level[i];
    }
  }
  return out;
}

}  // namespace led_bars
