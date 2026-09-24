// filter.blur.lens — bilinear upsample of a proc-res texture back to full res
// (the bokeh gather runs at reduced res; the disc is low-frequency so the
// upsample is invisible — pipeline.pass_bokeh :235). Uniform-free: the output
// dims come from GetDimensions, and the sampler does the [0,1] bilinear resize.

#include "common.hlsl"

Texture2D<float4>   inputTex  : register(t0);   // proc-res
SamplerState        samp      : register(s1);   // Linear + ClampToEdge
RWTexture2D<float4> outputTex : register(u2);   // full-res

[numthreads(8, 8, 1)]
void main(uint3 gid : SV_DispatchThreadID) {
  uint w, h;
  outputTex.GetDimensions(w, h);
  if (gid.x >= w || gid.y >= h) return;
  float2 uv = (float2(gid.xy) + 0.5) / float2(w, h);
  outputTex[gid.xy] = inputTex.SampleLevel(samp, uv, 0.0);
}
