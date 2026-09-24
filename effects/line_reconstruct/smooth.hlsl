// filter.reconstruct.line — pass 5d: combine the blurred products into the
// smoothed feature fields. (Port of pipeline.pass_smooth_weights, minus the
// shared centerline which is its own fp16-safe pass.)
//   - polarity coherence: |blur(pol*w²)| / blur(w²) — the step-edge rejector
//     (a blurred edge's two ridge lobes have opposite polarity and cancel).
//   - orientation coherence: |blur(dir*wl)| / blur(wl) — vetoes crossings the
//     tensor junction measure misses. Exported, applied as a veto in pass 6.
//   - confidence-weighted smoothing of w_est / orientation (a plain blur bleeds
//     the flat region's meaningless sigma* into line centers).
//   S0 = (cos2t_s, sin2t_s, w_est_s, 0)      S1 = (wl_s, wp_s, wg_s, ori_coh)

#include "common.hlsl"

Texture2D<float4>   g1  : register(t0);   // blur@1 (wl, cos2t*wl, sin2t*wl, w_est*wl)
Texture2D<float4>   g1p : register(t1);   // blur@1 (w_point, w_grad, -, -)
Texture2D<float4>   g2  : register(t2);   // blur@2 (pol*w2, w2, cos2t*wl, sin2t*wl)
Texture2D<float4>   g2b : register(t3);   // blur@2 (wl, -, -, -)
Texture2D<float4>   m0  : register(t4);   // raw (cos2t, sin2t, w_est, delta)
RWTexture2D<float4> s0  : register(u5);
RWTexture2D<float4> s1  : register(u6);

[numthreads(8, 8, 1)]
void main(uint3 gid : SV_DispatchThreadID) {
  uint w, h;
  s0.GetDimensions(w, h);
  if (gid.x >= w || gid.y >= h) return;

  float4 G1  = g1[gid.xy];
  float4 G1p = g1p[gid.xy];
  float4 G2  = g2[gid.xy];
  float  bwl2 = g2b[gid.xy].x;
  float4 M0  = m0[gid.xy];

  float pol_coh = abs(G2.x) / (G2.y + 1e-9);
  float ori_coh = sqrt(G2.z * G2.z + G2.w * G2.w) / (bwl2 + 1e-9);

  // Confidence-weighted smoothing (+0.05 keeps flat regions from blowing up).
  float wsum = G1.x + 0.05;
  float cos2t_s = (G1.y + 0.05 * M0.x) / wsum;
  float sin2t_s = (G1.z + 0.05 * M0.y) / wsum;
  float west_s  = (G1.w + 0.05 * M0.z) / wsum;

  float wl_s = G1.x * lr_smoothstep(0.15, 0.45, pol_coh);

  s0[gid.xy] = float4(cos2t_s, sin2t_s, west_s, 0.0);
  s1[gid.xy] = float4(wl_s, G1p.x, G1p.y, ori_coh);
}
