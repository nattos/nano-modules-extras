// filter.light.bounce_resonator — shared constants + GPU struct layouts.
//
// Mirrored byte-for-byte by the C++ structs in main.cpp (static_asserts
// there). Cells are ROW-MAJOR: idx = row * BR_BARS + bar, so one row — one
// copy of the 4-bar network — is contiguous for the sim's per-row hop.
// Row 0 is the BOTTOM segment (the led_bars convention); reduce and render
// share the mapping through br_cell_uv / br_cell_of.

#ifndef BOUNCE_RESONATOR_FX_COMMON_HLSL
#define BOUNCE_RESONATOR_FX_COMMON_HLSL

#define BR_BARS     4
#define BR_MAX_ROWS 16
#define BR_MAX_CELLS (BR_BARS * BR_MAX_ROWS)

// Per-cell input reduction (rewritten every frame by reduce.hlsl).
struct CellStat {
  float r; float g; float b;   // box-sampled mean colour
  float luma;                  // mean Rec.601 luma
};

// Per-cell resonator state (PERSISTS across frames).
struct CellState {
  float v;           // signed amplitude in the waveguide
  float h;           // hue, turns
  float s;           // saturation
  float mean;        // running mean of v (the A/C reference)
  float luma_prev;   // last frame's luma, for the up-edge excitation
  float swing;       // v - ac_amount * mean: what render shows
  float pad0; float pad1;
};

// Whole-network state (PERSISTS).
struct NetState {
  float env;    // decay-phase envelope follower (Σ|v|)
  float rms2;   // running mean-square swing over the grid (Contrast's level)
  float pad0; float pad1;
};

// Cell (bar, row) → uv of its bottom-left corner, and its uv extent.
float2 br_cell_size(int rows) { return float2(1.0 / float(BR_BARS), 1.0 / float(rows)); }
float2 br_cell_origin(int bar, int row, int rows) {
  float2 sz = br_cell_size(rows);
  // uv.y runs top-down; row 0 is at the bottom.
  return float2(float(bar) * sz.x, 1.0 - float(row + 1) * sz.y);
}
int br_cell_of(float2 uv, int rows) {
  int bar = clamp(int(uv.x * float(BR_BARS)), 0, BR_BARS - 1);
  int row = clamp(int((1.0 - uv.y) * float(rows)), 0, rows - 1);
  return row * BR_BARS + bar;
}

#endif  // BOUNCE_RESONATOR_FX_COMMON_HLSL
