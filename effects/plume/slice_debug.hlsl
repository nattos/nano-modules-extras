// source.sdf.plume — debug views (replaces tex_out when enabled).
//
// mode 0: tier-0 SDF slice at a chosen z — signed distance as a
//         blue(outside)/white(zero)/orange(inside) diverging map, with the
//         density band's edge visible as the white line.
// mode 1: shell_full displacement map (the raw octahedral field).
// mode 2: shell residual (full − coarse), mid-gray anchored — the data the
//         detail tier will consume; seams or aliasing show up here first.
// mode 3: GI radiance slice — watch the wave field slosh and ring.

#include "common.hlsl"

Texture3D<float4>   sdfVol      : register(t0);
Texture2D<float4>   shellFull   : register(t1);
Texture2D<float4>   shellCoarse : register(t2);
SamplerState        linearSamp  : register(s3);
RWTexture2D<float4> outTex      : register(u4);
Texture3D<float4>   radVol      : register(t6);

cbuffer DebugUniforms : register(b5) {
  float mode;     // 0 sdf slice, 1 shell, 2 residual
  float slice;    // [0,1] -> volume z
  float scale;    // value gain
  float _pad0;
};

[numthreads(8, 8, 1)]
void main(uint3 gid : SV_DispatchThreadID) {
  uint W, H;
  outTex.GetDimensions(W, H);
  if (gid.x >= W || gid.y >= H) return;

  // Center-square uv so the square data isn't stretched by the viewport.
  float2 uv = (float2(gid.xy) + 0.5) / float2(W, H);
  float aspect = float(W) / max(float(H), 1.0);
  float2 sq = float2((uv.x - 0.5) * aspect + 0.5, uv.y);
  if (sq.x < 0.0 || sq.x > 1.0) {
    outTex[gid.xy] = float4(0.02, 0.02, 0.02, 1.0);
    return;
  }

  float3 col;
  int m = int(mode);
  if (m == 0) {
    float d = sdfVol.SampleLevel(linearSamp, float3(sq, slice), 0).r * scale;
    float band = exp2(-abs(d) * 60.0);            // white line at the surface
    float3 inside = float3(1.0, 0.55, 0.15) * saturate(-d * 4.0);
    float3 outside = float3(0.15, 0.35, 1.0) * saturate(d * 4.0);
    col = inside + outside + band.xxx;
  } else if (m == 1) {
    float h = shellFull.SampleLevel(linearSamp, sq, 0).g * scale;
    col = h.xxx;
  } else if (m == 2) {
    float r = (shellFull.SampleLevel(linearSamp, sq, 0).r -
               shellCoarse.SampleLevel(linearSamp, sq, 0).r) * scale;
    col = saturate(0.5 + r * 4.0).xxx;
  } else {
    // Self-normalizing view: the field's equilibrium level varies a lot
    // with the decay/speed knobs, so compress instead of scaling.
    float3 gi = radVol.SampleLevel(linearSamp, float3(sq, slice), 0).rgb * scale;
    col = gi / (gi + 0.12);
  }
  outTex[gid.xy] = float4(col, 1.0);
}
