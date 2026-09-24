// filter.blur.smear (Blur mode) — one axis of a separable directional smear.
//
// Run TWICE from the host with different uniforms:
//   pass 1 (major axis) — asymmetric reach [-reach_back, +reach_fwd] so a bright
//                         point trails a TAIL behind the head (not a symmetric
//                         blob); tilt = 0 (perspective is a minor-axis effect).
//   pass 2 (minor axis) — symmetric reach ±width, scaled per-pixel by a GLOBAL
//                         perspective gradient (tilt): the minor-blur width ramps
//                         across the whole frame along the major axis — narrow on
//                         the forward/head side, wide on the rear (flips with sign).
//
// Aspect-correct: the host bakes min_dim/vp into (axis_x, axis_y) so a unit of
// reach is equal SCREEN distance in u and v. Linear sampler + SampleLevel so
// diagonal axes don't stair-step. Reach ~0 on both sides ⇒ every tap lands on the
// centre ⇒ passthrough.

Texture2D<float4>   inputTex      : register(t0);
SamplerState        linearSampler : register(s1);
RWTexture2D<float4> outputTex     : register(u2);

cbuffer Uniforms : register(b3) {
  float axis_x, axis_y;       // aspect-scaled UV step per unit of reach (this pass's axis)
  float reach_fwd, reach_back; // asymmetric reach, in short-axis fractions
  float major_x, major_y;     // screen-unit major direction (for perspective proj)
  float tilt;                 // perspective gradient amount (0 on the major pass)
  float falloff_k;            // gaussian tail sharpness: exp(-k·norm²). low = boxy, high = soft
  float exposure;             // output gain (final pass only; 1.0 on the major pass)
  int   samples;              // taps across the span
  float _pad0, _pad1;
};

// Interleaved-gradient noise — a cheap, stable (per-pixel, frame-independent)
// dither so long diagonal spans don't band.
float ign(float2 p) {
  return frac(52.9829189 * frac(dot(p, float2(0.06711056, 0.00583715))));
}

[numthreads(8, 8, 1)]
void main(uint3 gid : SV_DispatchThreadID) {
  uint W, H;
  outputTex.GetDimensions(W, H);
  if (gid.x >= W || gid.y >= H) return;

  float2 vp = float2(W, H);
  float2 uv = (float2(gid.xy) + 0.5) / vp;

  // Global perspective gradient along the major axis (tilt=0 ⇒ f=1 everywhere).
  float proj = clamp(dot(uv - 0.5, float2(major_x, major_y)) * 2.0, -1.0, 1.0);
  float f = clamp(1.0 - tilt * proj, 0.05, 3.0);
  float fwd  = reach_fwd  * f;
  float back = reach_back * f;

  int N = samples < 1 ? 1 : samples;
  float span = fwd + back;
  float jitter = (ign(float2(gid.xy)) - 0.5) * (span / (float)max(N - 1, 1));

  float4 acc = 0.0;
  float  wsum = 0.0;
  for (int i = 0; i < N; i++) {
    float t = (N == 1) ? 0.5 : (float)i / (float)(N - 1);
    float off = lerp(-back, fwd, t) + jitter;      // reach units, signed
    // Normalise against this side's reach so BOTH ends fade smoothly; the longer
    // (back) side stretches the falloff → the tail.
    float norm = off >= 0.0 ? (fwd  > 1e-6 ? off / fwd  : 0.0)
                            : (back > 1e-6 ? off / back : 0.0);
    float w = exp(-falloff_k * norm * norm);
    float2 duv = off * float2(axis_x, axis_y);
    acc  += w * inputTex.SampleLevel(linearSampler, uv + duv, 0.0);
    wsum += w;
  }

  float4 col = wsum > 1e-6 ? acc / wsum : inputTex.SampleLevel(linearSampler, uv, 0.0);
  col.rgb *= exposure;                 // recover thin bright lines diluted by the blur
  outputTex[gid.xy] = col;
}
