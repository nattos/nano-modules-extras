#pragma once
/*
 * rack_map.h — the pixel map for the six-bar rig: two racks of three
 * horizontal LED bars standing either side of the stage, sixty segments a bar.
 *
 * THE ROOM, seen from the audience, in meters from the stage's center line:
 *
 *        left rack                               right rack
 *   x = -2 ........ -1         (2 m gap)        +1 ........ +2
 *   [==== bar 2 ====]                            [==== bar 2 ====]   top
 *   [==== bar 1 ====]                            [==== bar 1 ====]
 *   [==== bar 0 ====]                            [==== bar 0 ====]   bottom
 *   seg 0 ....... 59                             59 ....... seg 0
 *   (outer)   (inner)                         (inner)   (outer)
 *
 * Segment 0 is a bar's OUTER end on both sides, so a segment index is a
 * distance from the stage: the rig is mirror-symmetric and anything that
 * radiates from the middle reads the same index on both sides.
 *
 * THE TEXTURE is the compact grid, a control signal and not a picture: the
 * left half of the frame is the left rack and the right half the right rack,
 * three rows with the top bar on top, and within each half the segments run
 * left to right as they stand — so the left half starts at the left rack's
 * outer end and the right half ends at the right rack's outer end. Each cell
 * is 1/120 of the width and 1/3 of the height and is painted as a flat block,
 * so a fixture footprint anywhere inside it lands on the exact colour.
 *
 * Host-free: no effect ABI, no GPU.
 */

#include <cmath>

namespace rack_map {

constexpr int kSides = 2;      ///< 0 = left, 1 = right
constexpr int kBars = 3;       ///< per rack, 0 = bottom
constexpr int kSegments = 60;  ///< per bar, 0 = outer end
constexpr int kCells = kSides * kBars * kSegments;

/// The compact grid the texture carries: 120 columns by 3 rows.
constexpr int kGridCols = kSides * kSegments;
constexpr int kGridRows = kBars;

/// The rig as it stands, in meters. Everything that travels through the room
/// measures against these and nothing else.
constexpr float kBarLength = 1.0f;
constexpr float kGap = 2.0f;           ///< inner end to inner end
constexpr float kInner = kGap * 0.5f;  ///< a bar's inner end, from the center line
constexpr float kOuter = kInner + kBarLength;
constexpr float kSegmentLength = kBarLength / (float)kSegments;

/// The physics cell order: side-major, then bar, then segment.
inline int cellIndex(int side, int bar, int seg) {
  return (side * kBars + bar) * kSegments + seg;
}

/// How far a segment's center stands from the stage's center line, in meters.
/// The same on both sides.
inline float segmentReach(int seg) {
  return kOuter - ((float)seg + 0.5f) * kSegmentLength;
}

/// Its signed position across the room: left negative.
inline float segmentX(int side, int seg) {
  const float r = segmentReach(seg);
  return side == 0 ? -r : r;
}

/// Where a cell lands in the compact grid. Row 0 is the TOP of the frame.
inline int gridColumn(int side, int seg) {
  return side == 0 ? seg : kSegments + (kSegments - 1 - seg);
}
inline int gridRow(int bar) { return kBars - 1 - bar; }
inline int gridIndex(int side, int bar, int seg) {
  return gridRow(bar) * kGridCols + gridColumn(side, seg);
}

/// The whole rig, one RGB per segment, in physics order (cellIndex).
struct Cells {
  float rgb[kCells][3] = {};
};

}  // namespace rack_map
