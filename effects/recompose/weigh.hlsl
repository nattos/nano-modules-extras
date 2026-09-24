// warp.recompose — weigh pass (pass B of the analysis).
//
// Re-samples the same coarse grid as `accumulate`, now that the global
// normalizers are known, and scatters the saliency-weighted moments: the total
// weight, the weighted centroid numerators, and the per-cell mass for each of
// the nine rule-of-thirds cells.
//
// Runs in the same submit as `accumulate`, after it — so pass A's sums are
// visible here. Same idiom as auto_level's minmax → hist (hist.hlsl: "Runs
// after minmax in the same submit, so the lo/hi writes are visible").

#include "common.hlsl"

Texture2D<float4>       inputTex : register(t0);
SamplerState            samp     : register(s1);
RWStructuredBuffer<int> stats    : register(u2);

cbuffer U : register(b3) {
  float res_x, res_y;        // viewport resolution
  float aspect_x, aspect_y;  // cover-square aspect
  float w_grad;              // weight on edge / local detail
  float w_dev;               // weight on deviation from the frame's mean luma
  float w_sat;               // weight on colour saturation
  float _pad0;
};

void addFixed(int slot, float v, float scale) {
  int prev;
  InterlockedAdd(stats[slot], (int)round(v * scale), prev);
}

[numthreads(8, 8, 1)]
void main(uint3 gid : SV_DispatchThreadID) {
  if (gid.x >= RC_GRID_SN || gid.y >= RC_GRID_SN) return;

  float2 res    = float2(res_x, res_y);
  float2 aspect = float2(aspect_x, aspect_y);
  float2 E      = rc_extent(aspect);

  float2 uv = (float2(gid.xy) + 0.5) / float(RC_GRID_SN);
  float2 p  = nano_uv_to_cover_square(uv, aspect);   // cover-square position

  // --- Global normalizers from pass A ---
  float N     = max(float(stats[RC_A_N]), 1.0);
  float meanL = (float(stats[RC_A_L])  / RC_SCALE_A) / N;
  float meanQ = (float(stats[RC_A_L2]) / RC_SCALE_A) / N;
  float meanG = (float(stats[RC_A_G])  / RC_SCALE_A) / N;
  float meanS = (float(stats[RC_A_S])  / RC_SCALE_A) / N;
  float sdL   = sqrt(max(meanQ - meanL * meanL, 0.0));

  // --- This sample's saliency ---
  float3 rgb  = inputTex.SampleLevel(samp, uv, 0).rgb;
  float  lum  = max(nano_luminance(rgb), 0.0);
  float  chrm = rc_chroma(rgb);
  float  grad = saturate(length(rc_grad_cover(inputTex, samp, uv, res, aspect)));

  float w = rc_weight(lum, grad, chrm, meanL, sdL, meanG, meanS,
                      w_grad, w_dev, w_sat);
  w = clamp(w, 0.0, RC_W_MAX);   // outlier + overflow guard

  addFixed(RC_B_W,  w,       RC_SCALE_W);
  addFixed(RC_B_WX, w * p.x, RC_SCALE_W);
  addFixed(RC_B_WY, w * p.y, RC_SCALE_W);

  // Per-cell mass. The grid samples strictly inside the frame, so the index is
  // always valid; the guard is belt-and-braces against a degenerate aspect.
  int k = rc_cell_index(p, E);
  if (k >= 0) addFixed(RC_B_M + k, w, RC_SCALE_W);
}
