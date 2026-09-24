// source.particles.sweep_chamber — field debug view.
// Replaces tex_out with a composite readout of the coarse field:
//   hue        = composed velocity direction (noise + image terms)
//   saturation = velocity magnitude
//   value      = swept luma L' (mean), with the ridge detector (L'max)
//                lifted toward white so trapped ridge lines are visible.
// The intra-cell peak offset shows as a faint green shift.

#include "common.hlsl"

Texture2D<float4>   fieldA  : register(t0);
Texture2D<float4>   fieldB  : register(t1);
SamplerState        lin     : register(s2);
RWTexture2D<float4> outTex  : register(u3);
Texture2D<float4>   fieldC  : register(t5);   // .rg ∇luma (curl dir)

cbuffer Uniforms : register(b4) {
  float to_image;
  float to_image_curl;
  float _p0, _p1;
}

[numthreads(8, 8, 1)]
void main(uint3 gid : SV_DispatchThreadID) {
  uint W, H;
  outTex.GetDimensions(W, H);
  if (gid.x >= W || gid.y >= H) return;

  float2 uv = (float2(gid.xy) + 0.5) / float2(W, H);
  float4 fa = fieldA.SampleLevel(lin, uv, 0);
  float4 fb = fieldB.SampleLevel(lin, uv, 0);

  float2 gl = fieldC.SampleLevel(lin, uv, 0).rg;
  float2 v = fb.xy + fb.zw * to_image
           + swc_undertow(fb.zw, gl, fa.a) * to_image_curl;
  float mag = length(v);
  float hue = frac(atan2(v.y, v.x) / 6.28318530718 + 0.5);
  float3 flow = swc_hsv_to_rgb(float3(hue, saturate(mag * 2.0), saturate(0.2 + mag)));

  float ridge = saturate(fa.a);          // L'max — the ridge detector
  float body  = saturate(fa.r);          // L'mean
  float3 col = flow * 0.5 + float3(body, body, body) * 0.35
             + float3(ridge, ridge, ridge) * 0.35;
  // Peak offsets tint green where non-zero (sub-cell ridge localization).
  col.g += saturate(length(fa.gb) * 1.5) * 0.25;

  outTex[gid.xy] = float4(saturate(col), 1.0);
}
