// filter.legacy.glisten — separable weighted blur with per-pass gain.
//
// Port of the NanoGraph "Blur" subgraph: taps span ±half_width (uv units, so
// the extent is aspect-relative like the original), weighted 41^(-x²) over
// x ∈ [-1, 1], normalized by the actual weight sum, then multiplied by a gain.
// The gain is where the flicker lives — the sparkle layer's blur passes pulse
// (contrast+1)·mix(env^curve, 1, sustain) per pass. Jitter displaces each tap
// by a 2D random offset (the original jittered its offset array per frame).
//
// Used twice on the 64² search grid (width 1−input_chaos, gain 1) and twice
// on the half-res sparkle layer (width from smoothing, flicker gain).
//
// The original's textures were all 8-bit sRGB: sampling decoded to linear,
// writes encoded. The blur MATH therefore ran in linear space with sRGB
// storage precision. We reproduce that manually because WebGPU forbids sRGB
// storage textures: `decode_in` says whether the input holds sRGB codes
// (plain unorm — decode each tap) or decodes in hardware (an RGBA8_SRGB
// render target); the output is always encoded back to codes.

#include "nano_color.hlsl"   // nano_srgb_to_linear / nano_linear_to_srgb

Texture2D<float4>   inputTex : register(t0);
SamplerState        samp     : register(s1);
RWTexture2D<float4> outTex   : register(u2);

cbuffer U : register(b3) {
  float dir_x, dir_y;        // pass axis (1,0) or (0,1)
  float half_width;          // tap extent in uv
  float gain;                // per-pass output multiplier (applied in linear)
  float taps;                // kernel length (odd; 1 = passthrough)
  float jitter;              // 0..1 — random 2D tap displacement × half_width
  float seed;
  float decode_in;           // 1 = input texels are sRGB codes (decode them)
};

float2 hash2(float2 p) {
  float3 q = frac(float3(p.xyx) * float3(0.1031, 0.1030, 0.0973));
  q += dot(q, q.yzx + 33.33);
  return frac((q.xx + q.yz) * q.zy);
}

float4 tapLinear(float2 uv) {
  float4 t = inputTex.SampleLevel(samp, uv, 0);
  if (decode_in > 0.5) t.rgb = nano_srgb_to_linear(t.rgb);
  return t;
}

[numthreads(8, 8, 1)]
void main(uint3 gid : SV_DispatchThreadID) {
  uint w, h;
  outTex.GetDimensions(w, h);
  if (gid.x >= w || gid.y >= h) return;
  float2 uv = (float2(gid.xy) + 0.5) / float2(w, h);

  int n = (int)taps;
  float4 o;
  if (n <= 1) {
    o = tapLinear(uv) * gain;
  } else {
    float2 dir = float2(dir_x, dir_y);
    float4 acc = 0.0;
    float wsum = 0.0;
    for (int i = 0; i < n; ++i) {
      float x = (float(i) / float(n - 1)) * 2.0 - 1.0;    // -1..1
      float wt = pow(41.0, -x * x);
      float2 off = dir * (x * half_width);
      if (jitter > 0.0) {
        off += (hash2(float2(float(i) * 0.731, seed)) - 0.5) * half_width * jitter;
      }
      acc += tapLinear(uv + off) * wt;
      wsum += wt;
    }
    o = (acc / max(wsum, 1e-6)) * gain;
  }
  o.rgb = nano_linear_to_srgb(saturate(o.rgb));
  outTex[gid.xy] = o;
}
