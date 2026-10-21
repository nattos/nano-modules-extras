// source.mesh.lantern — final composite (full res).
//
// Upsamples the half-res media (depth-aware: of the four nearest half-res
// texels, those that marched to a depth like this pixel's count most — so a
// dark block's silhouette doesn't pick up the glow marched past it), puts the
// surface behind it, exposes, tonemaps with AgX and writes over the input.

#include "common.hlsl"
#include "nano_hash.hlsl"

Texture2D<float4>   sceneTex : register(t2);
Texture2D<float4>   fogI     : register(t3);
Texture2D<float4>   fogT     : register(t4);
Texture2D<float4>   bgTex    : register(t5);
RWTexture2D<float4> outTex   : register(u6);
Texture2D<float4>   sparkTex : register(t7);   // the flakes (spark_fs): radiance x cov, cov

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

float3 ltn_srgb(float3 c) {
  c = saturate(c);
  return select(c <= 0.0031308, 12.92 * c, 1.055 * pow(c, 1.0 / 2.4) - 0.055);
}

[numthreads(8, 8, 1)]
void main(uint3 gid : SV_DispatchThreadID) {
  uint W = (uint)vp.x, H = (uint)vp.y;
  if (gid.x >= W || gid.y >= H) return;

  float4 scene = sceneTex.Load(int3(int2(gid.xy), 0));
  float d = scene.a;

  // Depth-aware bilinear upsample of the half-res media.
  int2 hs = int2((int(W) + 1) / 2, (int(H) + 1) / 2);
  float2 hp = (float2(gid.xy) + 0.5) * 0.5 - 0.5;
  int2 h0 = int2(floor(hp));
  float2 f = hp - float2(h0);
  float3 I = float3(0.0, 0.0, 0.0), T = float3(0.0, 0.0, 0.0);
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

  bool hit = d < LTN_MISS * 0.5;
  float3 surface = hit ? scene.rgb : float3(0.0002, 0.0004, 0.0016);
  float3 c = I + T * surface;
  int dbg = (int)shade.w;
  // The flakes, over what's behind them (their own water in front is already
  // folded into what they carry).
  if (dust.w > 0.5) {
    float4 sp = sparkTex.Load(int3(int2(gid.xy), 0));
    c = c * (1.0 - saturate(sp.a)) + sp.rgb;
    // Debug: the flakes alone, their coverage a faint gray under them.
    if (dbg >= 5) c = sp.rgb + 0.02 * saturate(sp.a);
  }
  if (dbg == 0 || dbg >= 5) {
    c = ltn_agx(c * grade.x, grade.w);
  } else {
    // Debug views: LINEAR values (times exposure), sRGB-encoded, no tonemap —
    // for calibrating against a renderer's passes.
    float3 v = dbg == 1 ? c : dbg == 2 ? surface : dbg == 3 ? I : T;
    c = ltn_srgb(v * (dbg == 4 ? 1.0 : grade.x));
  }

  float4 bg = grade.z > 0.5 ? bgTex.Load(int3(int2(gid.xy), 0)) : float4(0.0, 0.0, 0.0, 0.0);
  float3 outc = lerp(bg.rgb, c, grade.y);
  // Triangular dither at the 8-bit quantizer (the haze is all shallow gradients).
  float dn = nano_ign(float2(gid.xy)) + nano_hash21(float2(gid.xy) + float2(97.0, 71.0)) - 1.0;
  outc += dn * (1.0 / 255.0);
  outTex[gid.xy] = float4(outc, max(bg.a, grade.y));
}
