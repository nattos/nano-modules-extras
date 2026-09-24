// filter.blur.lens — highlight-energy extract (shared by veiling-glare + glow).
// hi = max(luma(src) − threshold, 0), replicated to RGB so a single Blur16 pass
// blurs it. (pipeline.pass_hood :257 / pass_glow :438 — the threshold differs.)

#include "common.hlsl"

Texture2D<float4>   inputTex  : register(t0);
RWTexture2D<float4> outputTex : register(u1);
cbuffer Uniforms : register(b2) { float u_threshold; float _p0, _p1, _p2; };

[numthreads(8, 8, 1)]
void main(uint3 gid : SV_DispatchThreadID) {
  uint w, h;
  outputTex.GetDimensions(w, h);
  if (gid.x >= w || gid.y >= h) return;
  float hi = max(lens_luma(inputTex[gid.xy].rgb) - u_threshold, 0.0);
  outputTex[gid.xy] = float4(hi, hi, hi, 1.0);
}
