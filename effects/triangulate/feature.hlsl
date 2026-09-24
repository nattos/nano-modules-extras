// triangulate — feature pass. Reads the pre-blurred input and builds the
// importance field from spatial derivatives of luma:
//   ridge  = strong NEGATIVE Laplacian (a bright crest running through a pixel)
//   corner = positive Hessian determinant where the Laplacian is negative (a peak)
//   density= blurred luma (general coverage of lit regions)
// Output (rgba16f):  r=density  g=ridge  b=corner  a=importance W.
#include "nano_color.hlsl"

Texture2D<float4>   blurTex : register(t0);
RWTexture2D<float4> featTex : register(u1);

cbuffer FeatureUniforms : register(b2) {
  float u_ridge_w;
  float u_corner_w;
  float u_void_w;
  float u_stencil;      // neighbour step in pixels for the derivative stencil
  float u_ridge_gain;
  float u_corner_gain;
  float u_pad0;
  float u_pad1;
};

float luma_at(int2 p, int2 dim) {
  p = clamp(p, int2(0, 0), dim - int2(1, 1));
  return nano_luminance(blurTex[p].rgb);
}

[numthreads(8, 8, 1)]
void main(uint3 gid : SV_DispatchThreadID) {
  uint w, h;
  featTex.GetDimensions(w, h);
  if (gid.x >= w || gid.y >= h) return;
  int2 dim = int2(w, h);
  int2 p = int2(gid.xy);
  int s = max(1, (int)round(u_stencil));

  float v  = luma_at(p, dim);
  float l  = luma_at(p + int2(-s,  0), dim);
  float r  = luma_at(p + int2( s,  0), dim);
  float u  = luma_at(p + int2( 0, -s), dim);
  float d  = luma_at(p + int2( 0,  s), dim);
  float tl = luma_at(p + int2(-s, -s), dim);
  float tr = luma_at(p + int2( s, -s), dim);
  float bl = luma_at(p + int2(-s,  s), dim);
  float br = luma_at(p + int2( s,  s), dim);

  // Second derivatives (finite-difference Hessian) on the smoothed field.
  float dxx = l - 2.0 * v + r;
  float dyy = u - 2.0 * v + d;
  float dxy = (br - bl - tr + tl) * 0.25;

  float laplacian = dxx + dyy;
  float ridge_raw = max(0.0, -laplacian);                       // bright crest
  float det = dxx * dyy - dxy * dxy;
  float corner_raw = (det > 0.0 && laplacian < 0.0) ? det : 0.0; // peak / corner

  // Rough soft-normalize into [0,1). Deliberately gentle (no saturation) — the
  // histogram/percentile pass (hist → cdf → remap) does the real per-frame
  // auto-leveling and the ridge/corner/void blend. Output = raw-ish maps only.
  float density = saturate(v);
  float ridge   = 1.0 - exp(-u_ridge_gain  * ridge_raw);
  float corner  = 1.0 - exp(-u_corner_gain * corner_raw);

  featTex[gid.xy] = float4(density, ridge, corner, 0.0);
}
