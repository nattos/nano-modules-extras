// vcr_halo prefilter — full res in, half res out.
//
// Two jobs in one pass, so the glow pyramid's level 0 costs nothing extra:
// the 13-tap downsample that seeds the chain, and the emitter extraction
// (deciding what is allowed to glow at all).
//
// Note what is NOT here: the outline band-pass. That happens per-octave in
// up.hlsl, so its band tracks the halo radius instead of being pinned to
// whatever a fixed tap offset happens to measure. What IS here is the
// brightness gate, and it reads the LOW-PASS value on purpose — "only bright
// things glow" has to mean the same thing whether the glow ends up coming
// from a body or from an edge.

Texture2D<float4>   srcTex : register(t0);
RWTexture2D<float4> dstTex : register(u1);
SamplerState        samp   : register(s2);

cbuffer Uniforms : register(b3) {
  float src_texel_x;   // 1 / input width
  float src_texel_y;   // 1 / input height
  float threshold;     // luma below this contributes nothing
  float knee;          // width of the soft shoulder around the threshold

  float saturation;    // chroma boost on the emitter
  float _pad0;
  float _pad1;
  float _pad2;

  float4 tint;         // rgb multiplier on the emitter, w unused
};

// Soft-knee threshold (the standard bloom prefilter). Below `threshold`
// nothing survives, above it the pixel passes at full strength, and the
// `knee` window between is a quadratic so a slow fade-up does not pop the
// glow on. Driven by the channel PEAK and applied as a scalar, so hue is
// preserved exactly.
float3 soft_threshold(float3 c, float thr, float kn) {
  float br = max(c.r, max(c.g, c.b));
  float k  = max(kn, 1e-4);
  float soft = clamp(br - thr + k, 0.0, 2.0 * k);
  soft = soft * soft / (4.0 * k);
  return c * (max(soft, br - thr) / max(br, 1e-4));
}

[numthreads(8, 8, 1)]
void main(uint3 gid : SV_DispatchThreadID) {
  uint dw, dh;
  dstTex.GetDimensions(dw, dh);
  if (gid.x >= dw || gid.y >= dh) return;

  float2 uv = (float2(gid.xy) + 0.5) / float2(dw, dh);
  float2 t  = float2(src_texel_x, src_texel_y);

  // 13-tap downsample (Jorge Jimenez, "Next Generation Post Processing in
  // Call of Duty: Advanced Warfare", SIGGRAPH 2014) — 9 outer taps at +-2
  // source texels plus 4 inner at +-1. Bilinear makes each tap a 2x2 average,
  // so the kernel covers 5x5 smoothly with no banding.
  float3 a = srcTex.SampleLevel(samp, uv + t * float2(-2,  2), 0).rgb;
  float3 b = srcTex.SampleLevel(samp, uv + t * float2( 0,  2), 0).rgb;
  float3 c = srcTex.SampleLevel(samp, uv + t * float2( 2,  2), 0).rgb;
  float3 d = srcTex.SampleLevel(samp, uv + t * float2(-2,  0), 0).rgb;
  float3 e = srcTex.SampleLevel(samp, uv,                      0).rgb;
  float3 f = srcTex.SampleLevel(samp, uv + t * float2( 2,  0), 0).rgb;
  float3 g = srcTex.SampleLevel(samp, uv + t * float2(-2, -2), 0).rgb;
  float3 h = srcTex.SampleLevel(samp, uv + t * float2( 0, -2), 0).rgb;
  float3 i = srcTex.SampleLevel(samp, uv + t * float2( 2, -2), 0).rgb;
  float3 j = srcTex.SampleLevel(samp, uv + t * float2(-1,  1), 0).rgb;
  float3 k = srcTex.SampleLevel(samp, uv + t * float2( 1,  1), 0).rgb;
  float3 l = srcTex.SampleLevel(samp, uv + t * float2(-1, -1), 0).rgb;
  float3 m = srcTex.SampleLevel(samp, uv + t * float2( 1, -1), 0).rgb;

  float3 emit = (j + k + l + m) * 0.125
              + e * 0.125
              + (b + d + f + h) * 0.0625
              + (a + c + g + i) * 0.03125;

  emit = soft_threshold(emit, threshold, knee);

  // Chroma boost, luma-preserving. A tape halo is more saturated than the
  // light that made it; a desaturated glow reads as fog rather than neon.
  float lum = dot(emit, float3(0.2126, 0.7152, 0.0722));
  emit = max(lerp(float3(lum, lum, lum), emit, saturation), 0.0);

  dstTex[gid.xy] = float4(emit * tint.rgb, 1.0);
}
