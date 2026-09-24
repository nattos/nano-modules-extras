// filter.reconstruct.line — pass 5a: materialize the blur-input products.
// pass_smooth_weights needs several confidence-weighted fields blurred at
// sigma 1 and sigma 2. We pack the products into textures grouped BY sigma so a
// single separable RGBA16F blur (Blur16, in-place) handles each group.
//   G1  @1 : (wl, cos2t*wl, sin2t*wl, w_est*wl)
//   G1p @1 : (w_point, w_grad, 0, 0)
//   G2  @2 : (polarity*w2, w2, cos2t*wl, sin2t*wl)   [w2 = wl*wl]
//   G2b @2 : (wl, 0, 0, 0)                            [bwl = blur(wl,2) normalizer]

#include "common.hlsl"

Texture2D<float4>   m0  : register(t0);   // (cos2t, sin2t, w_est, delta)
Texture2D<float4>   m1  : register(t1);   // (w_line, w_point, w_grad, polarity)
RWTexture2D<float4> g1  : register(u2);
RWTexture2D<float4> g1p : register(u3);
RWTexture2D<float4> g2  : register(u4);
RWTexture2D<float4> g2b : register(u5);

[numthreads(8, 8, 1)]
void main(uint3 gid : SV_DispatchThreadID) {
  uint w, h;
  g1.GetDimensions(w, h);
  if (gid.x >= w || gid.y >= h) return;

  float4 M0 = m0[gid.xy];
  float4 M1 = m1[gid.xy];
  float wl = M1.x, wp = M1.y, wg = M1.z, pol = M1.w;
  float c2 = M0.x, s2 = M0.y, west = M0.z;
  float w2 = wl * wl;

  g1[gid.xy]  = float4(wl, c2 * wl, s2 * wl, west * wl);
  g1p[gid.xy] = float4(wp, wg, 0.0, 0.0);
  g2[gid.xy]  = float4(pol * w2, w2, c2 * wl, s2 * wl);
  g2b[gid.xy] = float4(wl, 0.0, 0.0, 0.0);
}
