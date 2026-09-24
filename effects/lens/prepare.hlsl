// filter.blur.lens — pass 1 (prepare). Input level conditioning + highlight boost
// (pipeline.pass_prepare :147).
//
// First a Resolume-style brightness/contrast on the incoming display-space signal
// (contrast pivots BLACK, not gray: in·(1+contrast) + brightness) — the whole
// downstream pipeline (filmic toe, tonemap) tends to crush blacks, so a small
// default lift keeps already-processed footage from clipping. Then linearise the
// sRGB input (REPORT §1) and multiply highlights by (1 + hl_boost·mask) so bright
// points survive the energy-normalised bokeh gather as discs. Writes linear HDR.

#include "common.hlsl"

Texture2D<float4>   inputTex  : register(t0);
RWTexture2D<float4> outputTex : register(u1);
cbuffer Uniforms : register(b2) {
  float u_hl_threshold;
  float u_hl_boost;      // already ×8 host-side
  float u_in_brightness; // additive lift (display space)
  float u_in_contrast;   // Resolume contrast: pivots black, factor (1+contrast)
};

[numthreads(8, 8, 1)]
void main(uint3 gid : SV_DispatchThreadID) {
  uint w, h;
  outputTex.GetDimensions(w, h);
  if (gid.x >= w || gid.y >= h) return;

  float4 c = inputTex[gid.xy];
  // Resolume-style brightness/contrast (pivot black), on the display signal.
  float3 disp = max(c.rgb * (1.0 + u_in_contrast) + u_in_brightness, 0.0.xxx);
  float3 lin = lens_srgb_to_linear(disp);
  float  lum = lens_luma(lin);
  float  mask = lens_smoothstep(u_hl_threshold, u_hl_threshold * 2.0, lum);
  outputTex[gid.xy] = float4(lin * (1.0 + u_hl_boost * mask), c.a);
}
