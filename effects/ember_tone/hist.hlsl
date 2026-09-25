// filter.color.ember_tone — Pass 1: hue histogram.
//
// A coarse grid of the input, each sample's YIQ hue voted into 64 bins,
// weighted by its chroma (so grey votes nothing). The same histogram as
// color.legacy.bicolor_grad's.

#include "common.hlsl"

Texture2D<float4>       inputTex : register(t0);
SamplerState           samp     : register(s1);
RWStructuredBuffer<int> hist     : register(u2);   // 64 bins

cbuffer U : register(b3) {
  float grid;          // samples per axis
  float weight_scale;  // chroma -> integer vote
  float _p0;
  float _p1;
};

[numthreads(8, 8, 1)]
void main(uint3 gid : SV_DispatchThreadID) {
  int g = (int)grid;
  if ((int)gid.x >= g || (int)gid.y >= g) return;

  float2 uv = (float2(gid.xy) + 0.5) / grid;
  float3 rgb = max(inputTex.SampleLevel(samp, uv, 0).rgb, 0.0);
  float2 hc = et_yiq_hue(rgb);

  int bin = clamp((int)(hc.x * ET_BUCKETS), 0, ET_BUCKETS - 1);
  int w = (int)(hc.y * weight_scale);
  if (w > 0) {
    int prev;
    InterlockedAdd(hist[bin], w, prev);
  }
}
