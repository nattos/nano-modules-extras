// filter.blur.lens — pass 6 (halation + bloom). Highlights bleed a soft glow — a
// neutral gentle bloom + a wider warm halation ring (pipeline.pass_glow :434).
// Composites two pre-blurred highlight maps (σ 0.02·md and 0.05·md) onto the image.

#include "common.hlsl"

Texture2D<float4>   srcTex    : register(t0);   // current linear-HDR image (full res)
Texture2D<float4>   bloomTex  : register(t1);   // Blur16(hi, σ_bloom) at flare res
Texture2D<float4>   halTex    : register(t2);   // Blur16(hi, σ_hal) at flare res
SamplerState        samp      : register(s3);   // Linear + ClampToEdge (upsample)
RWTexture2D<float4> outputTex : register(u4);
cbuffer Uniforms : register(b5) {
  float u_bloom;
  float u_halation;
  float2 _p;
  float3 u_hal_color;
  float _p2;
};

[numthreads(8, 8, 1)]
void main(uint3 gid : SV_DispatchThreadID) {
  uint w, h;
  outputTex.GetDimensions(w, h);
  if (gid.x >= w || gid.y >= h) return;
  float2 uv = (float2(gid.xy) + 0.5) / float2(w, h);   // bilinear-upsample the glow
  float4 src = srcTex[gid.xy];
  float3 rgb = src.rgb
             + bloomTex.SampleLevel(samp, uv, 0.0).rgb * u_bloom
             + halTex.SampleLevel(samp, uv, 0.0).rgb * u_hal_color * u_halation;
  outputTex[gid.xy] = float4(rgb, src.a);
}
