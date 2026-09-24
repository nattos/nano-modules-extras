// filter.light.flicker_grid — shared constants + GPU struct layouts.
//
// Mirrored byte-for-byte by the C++ structs in main.cpp (static_asserts
// there). Cell indexing is COLUMN-MAJOR: idx = col * FG_MAX_ROWS + row,
// so a column's cells are contiguous for the sim's serial per-column loop.

#ifndef FLICKER_GRID_COMMON_HLSL
#define FLICKER_GRID_COMMON_HLSL

#define FG_MAX_COLS 16
#define FG_MAX_ROWS 32

// Per-cell input reduction (rewritten every frame by reduce.hlsl).
struct CellStat {
  float avg_r; float avg_g; float avg_b;   // box-sampled mean color
  float avg_luma;                          // mean Rec.601 luma
  float max_luma;                          // max sample luma
  float pad0; float pad1; float pad2;
};

// Per-column flicker state (PERSISTS across frames; only `acc` is truly
// cross-frame — gate/fill/level_target are last frame's outputs, kept so a
// re-render without a tick can hold the frame).
struct ColState {
  float acc;            // phase accumulator, [0,1)
  float gate;           // this frame: 1 = pulse on, 0 = off
  float fill;           // this frame: off-frame brightness 0..1
  float level_target;   // column max post-pull HSL lightness (leveling target)
};

#endif  // FLICKER_GRID_COMMON_HLSL
