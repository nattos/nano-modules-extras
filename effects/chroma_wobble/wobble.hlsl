// warp.legacy.chroma_wobble — the wobble + chroma-split pass.
//
// An animated fractal-noise field drives a UV WOBBLE (warp) and a per-channel
// chromatic offset. The whole thing is scaled by an intensity gain that the
// host gates with a triggered Attack/Release envelope (+ a manual floor), so
// it pulses on a trigger and decays. v2 of the Resolume Wire "ChromaWobble".
//
// The displacement field is computed analytically here (two fbm samples of an
// animated coordinate) rather than via the original's blurred/feedback noise
// texture — cheaper and shimmer-free, the "v2 tuned for efficiency" the team
// asked for.

#include "nano_chroma.hlsl"
#include "nano_hash.hlsl"

Texture2D<float4>   inputTex    : register(t0);
SamplerState        linearSampler : register(s1);
RWTexture2D<float4> outputTex   : register(u2);

cbuffer Uniforms : register(b3) {
  float gain;       // resolved wobble intensity (envelope × Intensity)
  float freq;       // noise frequency
  float phase;      // animation drift (accumulated)
  float chroma;     // chromatic-split magnitude
  float warp;       // shared UV warp magnitude
  float hue_shift;  // YIQ hue rotation of the split (radians)
  float aspect_x;   // min(W,H)/W
  float aspect_y;   // min(W,H)/H
};

static const float TAU = 6.28318530717958647692;

// Per-channel offset directions (from the original's randomized channel
// multipliers — kept as a fixed prismatic-ish set).
static const float2 DIR_R = float2(-0.82,  0.86);
static const float2 DIR_G = float2(-0.19, -0.51);
static const float2 DIR_B = float2( 0.67, -0.12);

[numthreads(8, 8, 1)]
void main(uint3 gid : SV_DispatchThreadID) {
  uint W, H;
  outputTex.GetDimensions(W, H);
  if (gid.x >= W || gid.y >= H) return;

  float2 uv  = (float2(gid.xy) + 0.5) / float2(W, H);
  float2 asp = float2(aspect_x, aspect_y);

  // Animated 2D fbm displacement in ~[-1,1]^2 (two decorrelated samples).
  float2 np = uv * freq + float2(phase, phase * 0.7);
  float nx = nano_fbm2(np, 4) * 2.0 - 1.0;
  float ny = nano_fbm2(np + float2(37.2, 11.7), 4) * 2.0 - 1.0;
  float2 disp = float2(nx, ny) * gain;

  // Shared warp + per-channel chromatic shift (aspect-corrected).
  float2 w  = disp * warp * asp;
  float2 sR = w + disp * DIR_R * chroma * asp;
  float2 sG = w + disp * DIR_G * chroma * asp;
  float2 sB = w + disp * DIR_B * chroma * asp;

  float4 col = nano_chroma_offset(inputTex, linearSampler, uv, sR, sG, sB, hue_shift);
  outputTex[gid.xy] = col;
}
