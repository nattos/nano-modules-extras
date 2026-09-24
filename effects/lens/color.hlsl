// filter.blur.lens — pass 3 (coating colour grade). Linear-space tint × warmth ×
// transmission + micro-contrast around 18% grey (coatings.apply_color :70). The
// combined multiplier (coat.tint · warmth_tint · coat.transmission · transmission)
// and the coating contrast are precomputed host-side. Per-pixel map.

#include "common.hlsl"

Texture2D<float4>   inputTex  : register(t0);
RWTexture2D<float4> outputTex : register(u1);
cbuffer Uniforms : register(b2) {
  float3 u_color_mul;   // coat.tint · warmth_tint · coat.transmission · transmission
  float  u_contrast;    // coat.contrast
};

[numthreads(8, 8, 1)]
void main(uint3 gid : SV_DispatchThreadID) {
  uint w, h;
  outputTex.GetDimensions(w, h);
  if (gid.x >= w || gid.y >= h) return;
  float4 c = inputTex[gid.xy];
  float3 rgb = c.rgb * u_color_mul;
  rgb = (rgb - 0.18) * u_contrast + 0.18;   // micro-contrast around 18% grey
  outputTex[gid.xy] = float4(max(rgb, 0.0.xxx), c.a);
}
