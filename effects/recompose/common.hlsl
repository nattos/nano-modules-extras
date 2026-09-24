// warp.recompose — shared shader definitions.
//
// The effect analyzes the input for compositional balance against a pattern
// (currently the rule of thirds), then slices the frame into the nine
// rule-of-thirds cells and translates each cell to push the composition toward
// that pattern. Four passes: `accumulate` gathers global normalizers on a
// coarse grid, `weigh` re-samples that grid to build the saliency centroid and
// the nine cell masses, a single-thread `solve` reduces them into a latched set
// of per-cell correction vectors, and `render` inverse-maps each output pixel
// against the nine translated cells.
//
// All geometry is in aspect-aware cover-square coords (nano_coords.hlsl):
// (0,0) = viewport center, ±1 along the long axis = the viewport edge.

#ifndef RECOMPOSE_COMMON_HLSL
#define RECOMPOSE_COMMON_HLSL

#include "nano_coords.hlsl"
#include "nano_color.hlsl"

// ---- Analysis grid (keep in sync with main.cpp) ----
#define RC_GRID_SN 128                              // sample grid, per axis
#define RC_GRID_N  (RC_GRID_SN * RC_GRID_SN)        // 16384 samples

// ---- stats buffer (ints) ----
// Pass A (`accumulate`) writes [0..4]; pass B (`weigh`) writes [8..19].
#define RC_A_L      0   // Σ luma                       (scale RC_SCALE_A)
#define RC_A_L2     1   // Σ luma²                      (scale RC_SCALE_A)
#define RC_A_G      2   // Σ saturate(|∇luma|)          (scale RC_SCALE_A)
#define RC_A_S      3   // Σ chroma                     (scale RC_SCALE_A)
#define RC_A_N      4   // sample count — UNSCALED int
                        // 5..7 pad, keeping pass B 8-aligned
#define RC_B_W      8   // Σ w                          (scale RC_SCALE_W)
#define RC_B_WX     9   // Σ w·x                        (scale RC_SCALE_W, signed)
#define RC_B_WY    10   // Σ w·y                        (scale RC_SCALE_W, signed)
#define RC_B_M     11   // [11..19] Σ w per cell, index RC_B_M + k, k = row*3+col
#define RC_STATS_INTS 20

// Fixed-point scales (there is no atomic float add; scale → round → int add).
//
// RC_SCALE_A must stay large: pass A derives a VARIANCE (ΣL²/N − (ΣL/N)²) from
// two independently-quantized sums, and brutal_fold/edge.hlsl:37-42 documents
// that a small scale leaves a residual reading as spurious variance on a
// uniform frame (at 128 that was ~0.03 std). 65536 drops it below 0.005.
//
// Overflow bound is N · maxValue · SCALE ≤ INT32_MAX:
//   pass A: 16384 · 1.0      · 65536 = 1.07e9   (2.0× headroom)
//   pass B: 16384 · RC_W_MAX ·  8192 = 1.07e9   (2.0× headroom)
// Σw·x shares pass B's bound because |x| ≤ E.x ≤ 1 by construction.
//
// HARD CONSTRAINT: RC_GRID_N · RC_SCALE_A ≤ 2^31−1. Raising RC_GRID_SN to 256
// would force RC_SCALE_A down to 16384, which is below the variance bar above.
// Stay at 128.
#define RC_SCALE_A 65536.0
#define RC_SCALE_W  8192.0
#define RC_W_MAX       8.0    // per-sample saliency clamp (outlier + overflow guard)

// The |∇luma| term is saturate()d before accumulating: rc_grad_cover returns an
// aspect-scaled cover-metric gradient whose magnitude can exceed 1 on extreme
// aspects, which would blow the pass A bound. Clamping also stops a single hard
// edge from dominating the frame's own normalizer.

// ---- solve buffer (floats) — PERSISTENT across frames, the latched analysis ----
//
// Slots 0..3 are FIRST deliberately: both readback backends copy `byteCount`
// bytes from OFFSET 0 (native readBuffer(buf, 0, …); web copyBufferToBuffer(src,
// 0, …)), so a 16-byte readback of the published scalars requires them here.
#define RC_S_BAL_X    0   // published, [-1,1] signed
#define RC_S_BAL_Y    1   // published, [-1,1] signed
#define RC_S_CELLERR  2   // published, [0,1] unsigned
#define RC_S_VALID    3   // 1.0 once a solve has run against real analysis
#define RC_S_CX       4   // smoothed saliency centroid (cover-square)
#define RC_S_CY       5
#define RC_S_PX       6   // nearest power point (derived each frame)
#define RC_S_PY       7
#define RC_S_GX       8   // G = P* − centroid (derived each frame)
#define RC_S_GY       9
#define RC_S_SCALE   10   // the global clamp scale s (debug overlay)
#define RC_S_INIT    11   // 0 = never solved; the first solve snaps (alpha = 1)
#define RC_S_M       12   // [12..20] smoothed normalized cell masses
#define RC_S_D       21   // [21..38] D_k.xy — D_k.x at RC_S_D+2k, .y at +2k+1
#define RC_SOLVE_FLOATS 40

// Overlap modes (match the selectField enum in main.cpp).
#define RC_OM_HEAVIEST 0
#define RC_OM_BLEND    1
#define RC_OM_ADDITIVE 2

// ---- Cell geometry ----
//
// nano_coords maps uv → (uv − 0.5)/aspect, so the visible frame's HALF-EXTENT
// in cover-square units is 0.5/aspect — NOT `aspect` itself. At 1920×1080
// (ax = 0.5, ay = 0.8889) this is (1.0, 0.5625): x spans [-1, 1], y spans
// [-0.5625, 0.5625]; the long axis is always exactly 1. Every third line, power
// point and cell rect derives from it, and because `weigh` and `render` both
// call these same helpers they cannot disagree about where the cells are.
float2 rc_extent(float2 aspect) { return 0.5 / max(aspect, 1e-6); }

// Cell side lengths (a third of the frame on each axis).
float2 rc_cell_size(float2 E) { return (2.0 / 3.0) * E; }

// Normalized frame coords: n ∈ [-1,1]² over the visible frame.
float2 rc_norm(float2 p, float2 E) { return p / max(E, 1e-6); }

// Nearest of the four power points at (±E.x/3, ±E.y/3). Separable and exact:
// the two candidates per axis are symmetric about 0, so the Euclidean nearest
// is the componentwise nearest.
float2 rc_nearest_power(float2 p, float2 E) {
  return float2((p.x >= 0.0) ? (E.x / 3.0) : (-E.x / 3.0),
                (p.y >= 0.0) ? (E.y / 3.0) : (-E.y / 3.0));
}

// Cell index (row*3 + col), or -1 when p is outside the visible frame. The
// out-of-frame answer is what lets `render` distinguish a rift from an edge
// reveal, so this must NOT clamp.
int rc_cell_index(float2 p, float2 E) {
  float2 n = rc_norm(p, E);
  if (abs(n.x) > 1.0 || abs(n.y) > 1.0) return -1;
  int2 c = clamp((int2)floor((n + 1.0) * 1.5), int2(0, 0), int2(2, 2));
  return c.y * 3 + c.x;
}

// Cell centre. col 0/1/2 → -2E.x/3, 0, +2E.x/3 (likewise rows). NOTE these are
// NOT the power points — the centres sit at ±2E/3, the power points at ±E/3.
float2 rc_cell_center(int k, float2 E) {
  return float2((float(k % 3) - 1.0) * (2.0 / 3.0),
                (float(k / 3) - 1.0) * (2.0 / 3.0)) * E;
}

// Cell rect [lo, hi). The nine rects tile the visible frame exactly, so a point
// inside a cell rect is always inside the frame.
void rc_cell_rect(int k, float2 E, out float2 lo, out float2 hi) {
  float2 sz = rc_cell_size(E);
  lo = float2(-E.x + float(k % 3) * sz.x, -E.y + float(k / 3) * sz.y);
  hi = lo + sz;
}

// Ideal rule-of-thirds mass distribution — separable, and sums to exactly 1 for
// every bias so no renormalization is needed:
//   u(b) = ( (1+b/2)/3, (1-b)/3, (1+b/2)/3 ),  ideal[j][i] = u_i·u_j
// b = 0   → uniform 1/9 (no compositional preference)
// b = 0.5 → corners 0.174, edges 0.069, centre 0.028
// b = 1   → all mass in the four corner cells
// De-emphasising the centre while favouring the corners IS the rule of thirds.
float rc_ideal(int k, float bias) {
  float outer = (1.0 + bias * 0.5) / 3.0;
  float inner = (1.0 - bias) / 3.0;
  int i = k % 3, j = k / 3;
  float ux = (i == 1) ? inner : outer;
  float uy = (j == 1) ? inner : outer;
  return ux * uy;
}

// Cover-space luma gradient at viewport uv, sampled a few texels apart. The uv
// derivative is rescaled by `aspect` so the result is a gradient in cover-square
// metric (geometric), not uv metric — style guide §1.4/§1.5.
float2 rc_grad_cover(Texture2D<float4> tex, SamplerState samp, float2 uv,
                     float2 res, float2 aspect) {
  float2 step = 1.5 / max(res, 1.0);
  float lxp = nano_luminance(tex.SampleLevel(samp, uv + float2(step.x, 0), 0).rgb);
  float lxn = nano_luminance(tex.SampleLevel(samp, uv - float2(step.x, 0), 0).rgb);
  float lyp = nano_luminance(tex.SampleLevel(samp, uv + float2(0, step.y), 0).rgb);
  float lyn = nano_luminance(tex.SampleLevel(samp, uv - float2(0, step.y), 0).rgb);
  return float2((lxp - lxn) * aspect.x, (lyp - lyn) * aspect.y);
}

// Chroma = max(rgb) − min(rgb). Preferred over HSL/HSV saturation as a saliency
// term: normalized saturation would let a near-black pixel score a full 1.0.
float rc_chroma(float3 rgb) {
  return max(rgb.r, max(rgb.g, rgb.b)) - min(rgb.r, min(rgb.g, rgb.b));
}

// Per-sample saliency weight, shared by `weigh` and nothing else — but kept here
// beside the normalizer slots it depends on. Each term is divided by its own
// frame mean (or std-dev), so all three read as RELATIVE weights on any footage
// and `w` averages ~1 per frame. The floors make the fixed-point residual
// structurally inert and remove the Σw == 0 degenerate case: a black frame with
// all weights at 0 reports a perfectly uniform image rather than a divide by 0.
float rc_weight(float lum, float grad, float chroma,
                float meanL, float sdL, float meanG, float meanS,
                float w_grad, float w_dev, float w_sat) {
  return w_grad * (grad             / max(meanG, 0.02))
       + w_dev  * (abs(lum - meanL) / max(sdL,   0.02))
       + w_sat  * (chroma           / max(meanS, 0.02))
       + 1e-3;
}

#endif
