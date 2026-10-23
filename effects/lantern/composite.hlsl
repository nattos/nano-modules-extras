// source.mesh.lantern — final composite (full res).
//
// Upsamples the half-res media (depth-aware: of the four nearest half-res
// texels, those that marched to a depth like this pixel's count most — so a
// dark block's silhouette doesn't pick up the glow marched past it), puts the
// surface behind it, exposes, tonemaps with AgX and writes over the input.

#include "grade.hlsl"
#include "nano_hash.hlsl"

Texture2D<float4>   sceneTex : register(t2);
Texture2D<float4>   fogI     : register(t3);
Texture2D<float4>   fogT     : register(t4);
Texture2D<float4>   bgTex    : register(t5);
RWTexture2D<float4> outTex   : register(u6);
Texture2D<float4>   sparkTex : register(t7);   // the flakes (spark_fs): radiance x cov, cov

[numthreads(8, 8, 1)]
void main(uint3 gid : SV_DispatchThreadID) {
  uint W = (uint)vp.x, H = (uint)vp.y;
  if (gid.x >= W || gid.y >= H) return;

  float4 scene = sceneTex.Load(int3(int2(gid.xy), 0));
  float d = scene.a;

  float3 I, T;
  ltn_media(fogI, fogT, gid.xy, d, I, T);

  bool hit = d < LTN_MISS * 0.5;
  float3 surface = hit ? scene.rgb : float3(0.0002, 0.0004, 0.0016);
  float3 c = I + T * surface;
  int dbg = (int)shade.w;
  // The flakes, over what's behind them (their own water in front is already
  // folded into what they carry).
  float4 sp = float4(0.0, 0.0, 0.0, 0.0);
  if (dust.w > 0.5) {
    sp = sparkTex.Load(int3(int2(gid.xy), 0));
    c = c * (1.0 - saturate(sp.a)) + sp.rgb;
  }
  // Debug: the flakes alone, their coverage a faint gray under them.
  if (dust.w > 0.5 && dbg >= 5 && dbg <= 7) c = sp.rgb + 0.02 * saturate(sp.a);
  if (dbg == 8) {
    // Glint: the blocks' charge alone, flat, on black.
    c = ltn_srgb(hit ? surface : float3(0.0, 0.0, 0.0));
  } else if (dbg == 9) {
    // Depth: the Depth aux output's ramp, as data (no grade).
    float3 ro, rd;
    ltn_camera_ray(float2(gid.xy), vp.xy, ro, rd);
    c = ltn_depth_ramp(hit, d * dot(rd, cam_f.xyz));
  } else if (dbg == 0 || dbg >= 5) {
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
