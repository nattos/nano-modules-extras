// filter.reconstruct.line — pass 5c-prep: shared-centerline vote weights.
// Materializes the fp16-safe centroid inputs so the gather (centerline.hlsl)
// reads ONE texture per tap. Only majority-polarity, high-purity pixels vote.
//   Wc = (w_ctr, w_ctr*delta*nxr, w_ctr*delta*nyr, 0)
// where w_ctr = wl * rho² * agree,  agree = clamp(polarity * mean_pol * 4, 0, 1),
// mean_pol = blur(pol*w²,2) / blur(w²,2), and (nxr,nyr) is the RAW line normal.

#include "common.hlsl"

Texture2D<float4>   m0 : register(t0);   // (cos2t, sin2t, w_est, delta)
Texture2D<float4>   m1 : register(t1);   // (w_line, w_point, w_grad, polarity)
Texture2D<float4>   m3 : register(t2);   // (rho, -, -, -)
Texture2D<float4>   g2 : register(t3);   // blur@2 (pol*w2, w2, -, -)
RWTexture2D<float4> wc : register(u4);

[numthreads(8, 8, 1)]
void main(uint3 gid : SV_DispatchThreadID) {
  uint w, h;
  wc.GetDimensions(w, h);
  if (gid.x >= w || gid.y >= h) return;

  float4 M0 = m0[gid.xy];
  float4 M1 = m1[gid.xy];
  float rho = m3[gid.xy].x;
  float4 G2 = g2[gid.xy];

  float wl = M1.x, pol = M1.w;
  float c2 = M0.x, s2 = M0.y, delta = M0.w;

  float mean_pol = G2.x / (G2.y + 1e-9);
  float agree = clamp(pol * mean_pol * 4.0, 0.0, 1.0);
  float w_ctr = wl * rho * rho * agree;

  // RAW line normal pairs with the raw delta.
  float nxr = sqrt(clamp(0.5 * (1.0 + c2), 0.0, 1.0));
  float nyr = sign(s2) * sqrt(clamp(0.5 * (1.0 - c2), 0.0, 1.0));

  wc[gid.xy] = float4(w_ctr, w_ctr * delta * nxr, w_ctr * delta * nyr, 0.0);
}
