// source.brutal_fold — edge/variance reduce over the rendered tex_out.
//
// A GPU flatness detector for the "skip empty" feature. One thread per output
// pixel: read the 3x3 luma neighbourhood of the COMPOSITED frame, run a Sobel
// operator, and scatter four running sums into a small int stats buffer via
// atomics. The CPU reads these back (gpu_poll_readback) and forms the flatness
// metric = blend(luminance std-dev, mean edge energy). This measures the ACTUAL
// rendered output (colour grade + fog + receding layers), unlike the analytic
// CPU proxy it supersedes.
//
// Fixed-point: all three summed quantities are normalized to [0,1] in-shader and
// scaled by 128 for InterlockedAdd (int-only). Count is unscaled. Worst-case slot
// at 4K ≈ N*128 ≈ 1.06e9 < int32 max — see main.cpp kStatsScale.

#include "common.hlsl"   // cbuffer U (res_x/res_y) + nano_luminance

Texture2D<float4>    tex_out : register(t1);
// Per-TILE stats: a kTileGrid×kTileGrid grid, kSlots ints each
// [edge_sum, luma_sum, luma2_sum, motion_sum, pixel_count]. The CPU forms each
// tile's local variance + edge + motion and takes the weighted MAX over tiles, so
// any single structured region reads as non-flat — a global stat would dilute a
// small busy area away.
RWStructuredBuffer<int>   stats     : register(u2);
// Persistent previous-frame luma at each sample point (motion = |L - prevL|). Sized
// to the fixed sample grid, so it never reallocs on viewport resize. Sentinel <0 on
// the first frame → motion 0 (no spurious spike). Each thread owns one slot (no race).
RWStructuredBuffer<float> prevLuma  : register(u3);

static const int   kSlots       = 5;            // ints per tile (must match main.cpp)
static const int   kTileGrid    = 16;           // must match main.cpp kTileGrid
// Sample on a fixed grid, NOT per output pixel: bounded, resolution-independent
// work (256² threads regardless of viewport) + far less atomic contention. Each
// 16×16 tile gets 16×16 = 256 samples, plenty for per-tile stats. Sample spacing
// is res/256 (~4-8px), so region/shelf edges are still captured; raise if features
// thinner than that get missed at high res.
static const int   kSampleGrid  = 256;          // must match main.cpp kSampleGrid
// Fixed-point scale for the atomic sums. Must be LARGE: the variance is
// E[L²]−E[L]², and quantizing luma_sum and luma²_sum independently leaves a
// residual ~1/scale that shows up as spurious variance on a UNIFORM frame (at
// scale 128 that was ~0.03 std — enough to hide flat frames from the detector).
// 65536 drops it to <0.005; per-tile sums (≤256 samples) stay far under int32.
static const float kStatsScale = 65536.0;
static const float kSobelNorm  = 5.65685425;    // sqrt(32): max Sobel |grad| for luma in [0,1]
// Per-pixel edge weight is a CONTRAST-SQUASHED soft measure, not a binary count.
//   floor — deadzone: gradients below this (noise, AA fuzz, smooth shading/fog
//           gradients that spread over many px) contribute nothing.
//   span  — the gradient that maps to a "full" edge (a strong gray/black step).
//   gamma — <1 squash that lifts the low end so a subtle GRAY-ON-GRAY step still
//           registers strongly instead of being dwarfed by high-contrast edges.
static const float kEdgeFloor  = 0.015;
static const float kEdgeSpan   = 0.20;
static const float kEdgeGamma  = 0.5;

float lumAt(int2 p, int2 res) {
  p = clamp(p, int2(0, 0), res - 1);
  return nano_luminance(tex_out.Load(int3(p, 0)).rgb);
}

[numthreads(8, 8, 1)]
void main(uint3 gid : SV_DispatchThreadID) {
  if ((int)gid.x >= kSampleGrid || (int)gid.y >= kSampleGrid) return;
  int2 res = int2((int)res_x, (int)res_y);
  // Map this sample to a pixel centre; the Sobel reads its 3x3 neighbourhood spaced
  // `cell` pixels apart (a coarse full-frame Sobel at the sampling resolution).
  int2 cell = max(res / kSampleGrid, int2(1, 1));
  int2 p = clamp(int2(gid.xy) * res / kSampleGrid + cell / 2, int2(0, 0), res - 1);

  float l00 = lumAt(p + int2(-cell.x, -cell.y), res), l10 = lumAt(p + int2(0, -cell.y), res), l20 = lumAt(p + int2(cell.x, -cell.y), res);
  float l01 = lumAt(p + int2(-cell.x,  0), res),      l11 = lumAt(p,                       res), l21 = lumAt(p + int2(cell.x,  0), res);
  float l02 = lumAt(p + int2(-cell.x, cell.y), res),  l12 = lumAt(p + int2(0, cell.y), res),   l22 = lumAt(p + int2(cell.x, cell.y), res);

  float gx = (l20 + 2.0 * l21 + l22) - (l00 + 2.0 * l01 + l02);
  float gy = (l02 + 2.0 * l12 + l22) - (l00 + 2.0 * l10 + l20);
  float g = saturate(sqrt(gx * gx + gy * gy) / kSobelNorm);   // edge magnitude [0,1]
  float e = pow(saturate((g - kEdgeFloor) / kEdgeSpan), kEdgeGamma);  // contrast-squashed
  float L = l11;

  // Motion = temporal frame difference at this sample, same contrast-squash as edges.
  int sidx = (int)gid.y * kSampleGrid + (int)gid.x;
  float pl = prevLuma[sidx];
  float m = (pl < 0.0) ? 0.0 : abs(L - pl);
  float me = pow(saturate((m - kEdgeFloor) / kEdgeSpan), kEdgeGamma);
  prevLuma[sidx] = L;

  // Route this sample into its tile's slots (tile from the SAMPLE grid).
  int2 tile = clamp(int2(gid.xy) * kTileGrid / kSampleGrid, int2(0, 0), int2(kTileGrid - 1, kTileGrid - 1));
  int ti = (tile.y * kTileGrid + tile.x) * kSlots;

  int prev;   // round (not truncate) to halve quantization error
  InterlockedAdd(stats[ti + 0], (int)(e * kStatsScale + 0.5),     prev);  // soft edge sum
  InterlockedAdd(stats[ti + 1], (int)(L * kStatsScale + 0.5),     prev);
  InterlockedAdd(stats[ti + 2], (int)(L * L * kStatsScale + 0.5), prev);
  InterlockedAdd(stats[ti + 3], (int)(me * kStatsScale + 0.5),    prev);  // soft motion sum
  InterlockedAdd(stats[ti + 4], 1,                                prev);
}
