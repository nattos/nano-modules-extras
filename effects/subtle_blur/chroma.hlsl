// filter.legacy.subtle_blur — chromatic-offset pass.
//
// Reads the (already blurred) image and splits the colour channels spatially
// along a fixed slanted axis (red one way, blue the other, green anchored),
// where the split is done in a HUE-ROTATED colour basis: rotate hue by +H,
// take the per-channel spatial samples, recombine, rotate back by −H. So `hue`
// controls WHICH colours separate (H=0 → plain RGB; the patch's ~0.22 →
// magenta/cyan-ish) rather than the spatial direction. The split WIDTH (the
// `off` vector) is animated by the sawtooth in main.cpp.

#include "nano_chroma.hlsl"

Texture2D<float4>   inputTex      : register(t0);
SamplerState        linearSampler : register(s1);
RWTexture2D<float4> outputTex     : register(u2);

cbuffer Uniforms : register(b3) {
  float off_x;      // spatial offset for the red sample (uv, aspect-corrected)
  float off_y;
  float hue_shift;  // YIQ colour-basis rotation (radians)
  float _pad;
};

[numthreads(8, 8, 1)]
void main(uint3 gid : SV_DispatchThreadID) {
  uint W, H;
  outputTex.GetDimensions(W, H);
  if (gid.x >= W || gid.y >= H) return;

  float2 uv  = (float2(gid.xy) + 0.5) / float2(W, H);
  float2 off = float2(off_x, off_y);

  // R sampled at +off, B at −off, G anchored — recombined in a hue-rotated
  // basis so `hue_shift` chooses the separated colour pair.
  float4 col = nano_chroma_offset(inputTex, linearSampler, uv,
                                  off, float2(0.0, 0.0), -off, hue_shift);
  outputTex[gid.xy] = col;
}
