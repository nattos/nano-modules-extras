// source.shape_fold — motion/variance reduce over the RAW FIELD (pre-grade).
//
// The skip-static detector measures frame-to-frame change. Reading it off the
// composited tex_out was wrong: shape_fold histogram-equalizes every frame, so the
// auto-levels LUT periodically reshapes and the whole image flashes even when the
// field is barely moving — the detector read those flashes as motion. So we evaluate
// the field DIRECTLY here (the same sf_field_at the present pass uses), no
// equalization LUT, no grade — a static field → identical values → zero motion.
//
// Motion is the change in the leveled value (F-lo)/range. Three subtleties make it
// correct:
//   • Floor clamp only (max(·,0), NOT saturate). Field below lo displays as black, so
//     sub-black wiggle mustn't count — the floor clamp drops it. But do NOT clamp the
//     top: when the range SHRINKS (a shape dimming, whose interior was last frame's
//     max) the previous value re-levels above 1; a top clamp would pin it to 1 and
//     silently drop the whole dimming (half of every fill pulse).
//   • Level BOTH frames against the CURRENT lo/hi. prevField stores the RAW previous
//     field and we re-level it here, so a drifting lo/hi cancels — a static pixel (or
//     one below the black floor) shows no phantom motion when the range shifts — while
//     a real level/fill change within the visible range still registers. (The naïve
//     (F-lo)/(hi-lo) diff, which levels each frame against ITS OWN lo/hi, gets both
//     wrong: static pixels flash when the range drifts, floor-pinned pixels move with
//     the floor.)
//
// One thread per fixed-grid sample: evaluate the field over a 3×3 neighbourhood
// (Sobel edge + local stats), diff the visible centre value (motion), and scatter
// running sums into the per-tile int stats buffer. Slots 5-10 are the SIGNED Lucas-
// Kanade flow sums for the uniform-drift penalty (range-normalized gradient vs signed
// dv). The CPU reads these back.

#include "common.hlsl"   // cbuffer U (terms, res, domain_scale) + sf_field_at + SF_NB

StructuredBuffer<float>   lut       : register(t1);   // auto-levels: [SF_NB]=lo, [SF_NB+1]=hi
// Per-TILE stats: kTileGrid×kTileGrid grid, kSlots ints each.
//   [0] edge  [1] v_sum  [2] v²_sum  [3] motion  [4] count
//   [5] Σnx²  [6] Σny²   [7] Σnx·ny  [8] Σnx·dv  [9] Σny·dv  [10] Σdv²
RWStructuredBuffer<int>   stats     : register(u2);
// Persistent previous-frame LEVELED field per sample (motion = |v - prevV|). Sized
// to the fixed sample grid; sentinel <0 on the first frame → motion 0.
RWStructuredBuffer<float> prevField : register(u3);

static const int   kSlots       = 11;           // ints per tile (must match main.cpp)
static const int   kTileGrid    = 16;
static const int   kSampleGrid  = 256;
static const float kStatsScale  = 65536.0;
static const float kSobelNorm   = 5.65685425;   // sqrt(32): max Sobel |grad| for v in [0,1]
// Edge squash (available for tuning; edge weight defaults off).
static const float kEdgeFloor  = 0.015;
static const float kEdgeSpan   = 0.20;
static const float kEdgeGamma  = 0.5;
// Motion squash — sensitive (leveled-field drift is small but genuine).
static const float kMotionFloor = 0.003;
static const float kMotionSpan  = 0.05;
static const float kMotionGamma = 0.5;

// Raw analytic field at a pixel (matches present's cover-square mapping). No lo/hi,
// no equalization LUT, no grade.
float field_at(int2 p_px, float2 vp) {
  float mx = max(vp.x, vp.y);
  float2 sq = (float2(p_px) + 0.5 - 0.5 * vp) / (0.5 * mx);
  float2 p = sq * domain_scale;
  return sf_field_at(p);
}

[numthreads(8, 8, 1)]
void main(uint3 gid : SV_DispatchThreadID) {
  if ((int)gid.x >= kSampleGrid || (int)gid.y >= kSampleGrid) return;
  int2 res = int2((int)res_x, (int)res_y);
  float2 vp = float2(res);
  float lo  = lut[SF_NB + 0];
  float hi  = lut[SF_NB + 1];
  float rng = hi - lo;
  // Normalize by the linear range for perceptual scale (bold shapes ride a big DC
  // offset). Near-flat / blank frame → 0 (no spurious motion). Crucially the range
  // divides the DIFF, it does NOT re-level each frame — so a pixel whose LEVEL
  // shifts (a shape brightening / fading / its fill pulsing) still reads as motion,
  // unlike (F-lo)/(hi-lo) which cancels level changes when lo/hi move with them.
  float invR = (rng < 1e-4) ? 0.0 : 1.0 / rng;

  int2 cell = max(res / kSampleGrid, int2(1, 1));
  int2 p = clamp(int2(gid.xy) * res / kSampleGrid + cell / 2, int2(0, 0), res - 1);

  float f00 = field_at(p + int2(-cell.x, -cell.y), vp), f10 = field_at(p + int2(0, -cell.y), vp), f20 = field_at(p + int2(cell.x, -cell.y), vp);
  float f01 = field_at(p + int2(-cell.x,  0), vp),      f11 = field_at(p,                    vp), f21 = field_at(p + int2(cell.x,  0), vp);
  float f02 = field_at(p + int2(-cell.x, cell.y), vp),  f12 = field_at(p + int2(0, cell.y), vp), f22 = field_at(p + int2(cell.x, cell.y), vp);

  float f11raw = f11;   // raw centre field, kept for storage + re-leveling next frame

  // Clamp the neighbourhood to the VISIBLE range [lo,hi] for the spatial features:
  // field below lo displays as black, above hi as peak.
  f00 = clamp(f00, lo, hi); f10 = clamp(f10, lo, hi); f20 = clamp(f20, lo, hi);
  f01 = clamp(f01, lo, hi); f11 = clamp(f11, lo, hi); f21 = clamp(f21, lo, hi);
  f02 = clamp(f02, lo, hi); f12 = clamp(f12, lo, hi); f22 = clamp(f22, lo, hi);

  // Range-normalized Sobel (the lo offset cancels in a difference), for edge + LK.
  float gx = ((f20 + 2.0 * f21 + f22) - (f00 + 2.0 * f01 + f02)) * invR;
  float gy = ((f02 + 2.0 * f12 + f22) - (f00 + 2.0 * f10 + f20)) * invR;
  float g = saturate(sqrt(gx * gx + gy * gy) / kSobelNorm);              // edge magnitude [0,1]
  float e = pow(saturate((g - kEdgeFloor) / kEdgeSpan), kEdgeGamma);     // contrast-squashed
  float V = (f11 - lo) * invR;   // visible leveled value ∈ [0,1], for the variance feature

  // Motion = change in the leveled value, both frames on the CURRENT lo/hi, clamped
  // ONLY at the black floor (max, not saturate). prevField holds the RAW previous
  // field, re-leveled here, so a drifting lo/hi cancels (static / below-black pixels
  // stay put). Crucially we do NOT clamp the TOP: when the range SHRINKS (a shape
  // dimming, whose bright interior was last frame's max) the previous value re-levels
  // above 1 — saturate would pin it to 1 and match V, dropping the whole dimming. The
  // floor clamp still ignores sub-black wiggle. `dts` signed for the LK flow sums.
  int sidx = (int)gid.y * kSampleGrid + (int)gid.x;
  float pf = prevField[sidx];
  float mot_now  = max((f11raw - lo) * invR, 0.0);
  float mot_prev = (pf < -1e29) ? mot_now : max((pf - lo) * invR, 0.0);   // sentinel: first frame → 0
  float dts = mot_now - mot_prev;
  float m = abs(dts);
  float me = pow(saturate((m - kMotionFloor) / kMotionSpan), kMotionGamma);
  prevField[sidx] = f11raw;                              // store RAW field (re-leveled next frame)

  float nx = gx / kSobelNorm;   // normalized gradient (‖(nx,ny)‖ ≤ 1)
  float ny = gy / kSobelNorm;

  int2 tile = clamp(int2(gid.xy) * kTileGrid / kSampleGrid, int2(0, 0), int2(kTileGrid - 1, kTileGrid - 1));
  int ti = (tile.y * kTileGrid + tile.x) * kSlots;

  int prev;   // round (not truncate) to halve quantization error
  InterlockedAdd(stats[ti + 0], (int)(e * kStatsScale + 0.5),     prev);  // soft edge sum
  InterlockedAdd(stats[ti + 1], (int)(V * kStatsScale + 0.5),     prev);
  InterlockedAdd(stats[ti + 2], (int)(V * V * kStatsScale + 0.5), prev);
  InterlockedAdd(stats[ti + 3], (int)(me * kStatsScale + 0.5),    prev);  // soft motion sum
  InterlockedAdd(stats[ti + 4], 1,                                prev);
  // Flow sums (SIGNED; round handles negatives). Per-tile |Σ| ≤ 256 samples → fits.
  InterlockedAdd(stats[ti + 5], (int)round(nx * nx  * kStatsScale), prev);
  InterlockedAdd(stats[ti + 6], (int)round(ny * ny  * kStatsScale), prev);
  InterlockedAdd(stats[ti + 7], (int)round(nx * ny  * kStatsScale), prev);
  InterlockedAdd(stats[ti + 8], (int)round(nx * dts * kStatsScale), prev);
  InterlockedAdd(stats[ti + 9], (int)round(ny * dts * kStatsScale), prev);
  InterlockedAdd(stats[ti + 10], (int)round(dts * dts * kStatsScale), prev);
}
