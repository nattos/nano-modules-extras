// source.mesh.lantern — what the final composite and the aux outputs share:
// the half-res media brought up to a full-res pixel, and the AgX grade.

#ifndef LANTERN_GRADE_HLSL
#define LANTERN_GRADE_HLSL

#include "common.hlsl"

// AgX (Blender's base transform), minimal fit: inset matrix, log2 encoding
// over [-12.47, 4.03] EV, a polynomial sigmoid, outset matrix. The result is
// display-encoded. `contrast` steepens about a log-space pivot; with the
// output power below, contrast 0.2 matches Blender's "AgX - Medium High
// Contrast" to ~0.01 (fitted to a Cycles render's linear/display pairs).
float3 ltn_agx(float3 c, float contrast) {
  const float3x3 inset = float3x3(
      0.842479062253094, 0.0423282422610123, 0.0423756549057051,
      0.0784335999999992, 0.878468636469772, 0.0784336,
      0.0792237451477643, 0.0791661274605434, 0.879142973793104);
  const float3x3 outset = float3x3(
      1.19687900512017, -0.0528968517574562, -0.0529716355144438,
      -0.0980208811401368, 1.15190312990417, -0.0980434501171241,
      -0.0990297440797205, -0.0989611768448433, 1.15107367264116);
  const float min_ev = -12.47393, max_ev = 4.026069;
  c = mul(max(c, 1e-10), inset);
  c = clamp(log2(c), min_ev, max_ev);
  c = (c - min_ev) / (max_ev - min_ev);
  const float pivot = 0.475;
  c = saturate(pivot + (c - pivot) * (1.0 + contrast));
  float3 x2 = c * c, x4 = x2 * x2;
  c = 15.5 * x4 * x2 - 40.14 * x4 * c + 31.96 * x4 - 6.868 * x2 * c +
      0.4298 * x2 + 0.1191 * c - 0.00232;
  return pow(saturate(mul(c, outset)), 1.3);
}

// The Depth picture (Aux output and Debug View): a ramp over 5 m either side
// of the structure's center (aux.w, its camera-space depth), white nearest;
// black where rays miss. `z` is camera-space depth in metres.
float ltn_depth_ramp(bool hit, float z) {
  return hit ? saturate(0.5 - (z - aux.w) / 10.0) : 0.0;
}

float3 ltn_srgb(float3 c) {
  c = saturate(c);
  return select(c <= 0.0031308, 12.92 * c, 1.055 * pow(c, 1.0 / 2.4) - 0.055);
}

// The half-res media (in-scatter I, transmittance T) at full-res pixel `px`
// in front of depth `d`: a depth-aware bilinear upsample — of the four
// nearest half-res texels, those that marched to a depth like this pixel's
// count most, so a dark block's silhouette doesn't pick up the glow marched
// past it.
void ltn_media(Texture2D<float4> fogI, Texture2D<float4> fogT, uint2 px, float d,
               out float3 I, out float3 T) {
  uint W = (uint)vp.x, H = (uint)vp.y;
  int2 hs = int2((int(W) + 1) / 2, (int(H) + 1) / 2);
  float2 hp = (float2(px) + 0.5) * 0.5 - 0.5;
  int2 h0 = int2(floor(hp));
  float2 f = hp - float2(h0);
  I = float3(0.0, 0.0, 0.0);
  T = float3(0.0, 0.0, 0.0);
  float wsum = 0.0;
  [unroll] for (int j = 0; j < 4; j++) {
    int2 o = int2(j & 1, j >> 1);
    int2 q = clamp(h0 + o, int2(0, 0), hs - 1);
    float wb = (o.x ? f.x : 1.0 - f.x) * (o.y ? f.y : 1.0 - f.y);
    float4 fi = fogI.Load(int3(q, 0));
    float dd = abs(fi.a - d) / (0.02 * min(d, fi.a) + 0.05);
    float w = (wb + 1e-3) / (1.0 + dd * dd);
    I += fi.rgb * w;
    T += fogT.Load(int3(q, 0)).rgb * w;
    wsum += w;
  }
  I /= wsum;
  T /= wsum;
}

#endif
