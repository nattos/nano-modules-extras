// warp.recompose — accumulate pass (pass A of the analysis).
//
// One thread per coarse-grid sample. Gathers the GLOBAL NORMALIZERS the
// saliency weight needs but can't know per-sample: the mean luma, the luma
// standard deviation (via Σluma²), and the mean gradient / mean chroma. The
// `weigh` pass re-samples the same grid with these in hand.
//
// Runs only on analysis-update frames (host-gated), like plane_shear's
// accumulate. The stats buffer is zeroed CPU-side immediately before this pass.

#include "common.hlsl"

Texture2D<float4>       inputTex : register(t0);
SamplerState            samp     : register(s1);
RWStructuredBuffer<int> stats    : register(u2);

cbuffer U : register(b3) {
  float res_x, res_y;        // viewport resolution
  float aspect_x, aspect_y;  // cover-square aspect
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
  float2 uv     = (float2(gid.xy) + 0.5) / float(RC_GRID_SN);

  float3 rgb  = inputTex.SampleLevel(samp, uv, 0).rgb;
  float  lum  = max(nano_luminance(rgb), 0.0);
  float  chrm = rc_chroma(rgb);

  // Saturated so one hard edge can't dominate the frame's own normalizer — and
  // so the pass A overflow bound in common.hlsl holds on extreme aspects.
  float grad = saturate(length(rc_grad_cover(inputTex, samp, uv, res, aspect)));

  addFixed(RC_A_L,  lum,       RC_SCALE_A);
  addFixed(RC_A_L2, lum * lum, RC_SCALE_A);
  addFixed(RC_A_G,  grad,      RC_SCALE_A);
  addFixed(RC_A_S,  chrm,      RC_SCALE_A);

  int prev;
  InterlockedAdd(stats[RC_A_N], 1, prev);   // unscaled count
}
