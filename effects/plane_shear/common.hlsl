// warp.plane_shear — shared shader definitions.
//
// The effect analyzes the input to pick a "natural" dividing line (a plane in
// 2D) — one of four algorithms — then shears the two halves on either side of
// that line. Analysis is on-GPU (no CPU readback): a coarse-grid `accumulate`
// pass scatters image statistics into a stats buffer, a single-thread `solve`
// pass reduces them into a latched line (center + unit normal), and the
// `render` pass warps the image by the signed distance to that line.
//
// All geometry is in aspect-aware cover-square coords (nano_coords.hlsl):
// (0,0) = viewport center, ±1 along the long axis = the viewport edge.

#ifndef PLANE_SHEAR_COMMON_HLSL
#define PLANE_SHEAR_COMMON_HLSL

#include "nano_coords.hlsl"
#include "nano_color.hlsl"

// ---- Analysis grid dimensions (keep in sync with main.cpp) ----
#define PS_GRID_SN 128      // accumulate sample grid (per axis)
#define PS_NA      64       // Hough / seam angle bins  (theta in [0, PI))
#define PS_NO      64       // Hough / seam offset bins  (rho in [-OFF_MAX, OFF_MAX])

// stats buffer (ints) layout:
//   [0..11]  moment sums (fixed-point, scale PS_SCALE_MOM)
//   [16..16 + PS_NA*PS_NO - 1]  (angle,offset) energy grid (scale PS_SCALE_E)
#define PS_MOM_BASE  0
#define PS_GRID_BASE 16
#define PS_STATS_INTS (PS_GRID_BASE + PS_NA * PS_NO)

// Moment slot indices.
#define PS_M_JXX 0    // structure tensor  Sum gx*gx
#define PS_M_JXY 1    //                   Sum gx*gy
#define PS_M_JYY 2    //                   Sum gy*gy
#define PS_M_GW  3    // Sum |g|            (gradient-energy weight)
#define PS_M_GWX 4    // Sum |g|*x
#define PS_M_GWY 5    // Sum |g|*y
#define PS_M_W   6    // Sum w              (luma weight, for PCA)
#define PS_M_WX  7    // Sum w*x
#define PS_M_WY  8    // Sum w*y
#define PS_M_WXX 9    // Sum w*x*x
#define PS_M_WXY 10   // Sum w*x*y
#define PS_M_WYY 11   // Sum w*y*y

// Fixed-point scales (atomic float-add isn't available; scale->round->int add).
#define PS_SCALE_MOM 1024.0
#define PS_SCALE_E   4096.0

// Algorithm selector (matches the selectField enum in main.cpp).
#define PS_ALG_DOMINANT 0
#define PS_ALG_HOUGH    1
#define PS_ALG_SEAM     2
#define PS_ALG_PCA      3

static const float PS_PI = 3.14159265358979323846;

// Cover-space luma gradient at viewport uv, sampled a few texels apart. The uv
// derivative is rescaled by `aspect` so the returned vector is the gradient in
// cover-square metric (geometric), not uv metric — see the aspect note in the
// style guide §1.4/§1.5.
float2 ps_grad_cover(Texture2D<float4> tex, SamplerState samp, float2 uv,
                     float2 res, float2 aspect) {
  float2 step = 1.5 / max(res, 1.0);
  float lxp = nano_luminance(tex.SampleLevel(samp, uv + float2(step.x, 0), 0).rgb);
  float lxn = nano_luminance(tex.SampleLevel(samp, uv - float2(step.x, 0), 0).rgb);
  float lyp = nano_luminance(tex.SampleLevel(samp, uv + float2(0, step.y), 0).rgb);
  float lyn = nano_luminance(tex.SampleLevel(samp, uv - float2(0, step.y), 0).rgb);
  // d(luma)/d(cover) = d(luma)/d(uv) * d(uv)/d(cover) = (lx,ly) * aspect.
  return float2((lxp - lxn) * aspect.x, (lyp - lyn) * aspect.y);
}

// Fold a line normal angle into [0, PI) and flip the offset sign to match, so
// n and -n map to the same (theta, rho) bin.
void ps_fold_line(inout float theta, inout float rho) {
  if (theta < 0.0)   { theta += PS_PI; rho = -rho; }
  if (theta >= PS_PI){ theta -= PS_PI; rho = -rho; }
}

#endif
