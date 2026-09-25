// filter.light.bar_scatter — shared constants + GPU struct layouts.
//
// Mirrored byte-for-byte by the C++ structs in main.cpp (static_asserts
// there). Cells are ROW-MAJOR: idx = row * BS_BARS + bar, row 0 the BOTTOM
// segment (the led_bars convention). Groups — one per bar per zone — are
// idx = zone * BS_BARS + bar, so one zone row (the four bars that co-trigger
// together) is contiguous.

#ifndef BAR_SCATTER_COMMON_HLSL
#define BAR_SCATTER_COMMON_HLSL

#define BS_BARS      4
#define BS_MAX_ROWS  16
#define BS_MAX_ZONES 16
#define BS_MAX_CELLS (BS_BARS * BS_MAX_ROWS)
#define BS_MAX_GROUPS (BS_BARS * BS_MAX_ZONES)
#define BS_HIST      64   // history slots (~1 s at 60 fps)

#define BS_PASS  0
#define BS_CHOKE 1
#define BS_DELAY 2

// Per-cell input reduction (rewritten every frame by reduce.hlsl).
struct CellStat {
  float r; float g; float b;
  float luma;
};

// Per-cell state (PERSISTS across frames).
struct CellState {
  float ref;                       // floor follower of luma: the onset reference
  float prev_r; float prev_g; float prev_b;   // last frame's live colour
  float snap_r; float snap_g; float snap_b;   // pre-flash colour, taken at a choke
  float pad;
};

// Per bar-and-zone state (PERSISTS).
struct GroupState {
  int   mode;      // BS_PASS / BS_CHOKE / BS_DELAY
  int   armed;     // 1 = may fire; re-arms below half the sensitivity
  float t_fire;    // when it last fired (s, the sim clock)
  float t_event;   // when its current choke/delay began
  float lag;       // delay, s
  float gain;      // choke gain shown this frame (for inspection)
  float pad0; float pad1;
};

// Cell (bar, row) → uv of its bottom-left corner, and its uv extent.
float2 bs_cell_size(int rows) { return float2(1.0 / float(BS_BARS), 1.0 / float(rows)); }
float2 bs_cell_origin(int bar, int row, int rows) {
  float2 sz = bs_cell_size(rows);
  // uv.y runs top-down; row 0 is at the bottom.
  return float2(float(bar) * sz.x, 1.0 - float(row + 1) * sz.y);
}
int bs_cell_of(float2 uv, int rows) {
  int bar = clamp(int(uv.x * float(BS_BARS)), 0, BS_BARS - 1);
  int row = clamp(int((1.0 - uv.y) * float(rows)), 0, rows - 1);
  return row * BS_BARS + bar;
}

#endif  // BAR_SCATTER_COMMON_HLSL
