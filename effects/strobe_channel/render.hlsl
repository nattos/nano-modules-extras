// source.light.strobe_channel — render pass.
//
// Lights up exactly one bar at full brightness; all others stay black.
// The "which bar" decision is made CPU-side via the logistic map and
// passed in as a uniform. Per pixel we just compare bar_index to the
// chosen one.

#include "nano_bars.hlsl"

Texture2D<float4>   inputTex  : register(t0);
RWTexture2D<float4> outputTex : register(u1);

cbuffer Uniforms : register(b2) {
  uint  active_bar;
  uint  bar_count;
  float intensity;
  float _pad_h0;

  float color_r;
  float color_g;
  float color_b;
  float _pad_h1;
};

[numthreads(8, 8, 1)]
void main(uint3 gid : SV_DispatchThreadID) {
  uint w, h;
  outputTex.GetDimensions(w, h);
  if (gid.x >= w || gid.y >= h) return;

  float u = float(gid.x) / float(w);
  uint bar = uint(clamp(floor(u * float(bar_count)),
                        0.0,
                        float(bar_count) - 1.0));

  float4 base = inputTex[gid.xy];
  if (bar == active_bar) {
    float3 c = float3(color_r, color_g, color_b) * intensity;
    base.rgb = saturate(base.rgb + c);
  }
  outputTex[gid.xy] = base;
}
