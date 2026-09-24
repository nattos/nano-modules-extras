// vcr_halo composite — full res. Adds the glow pyramid back over the source
// and runs the shared VCR grade.
//
// Everything here is deliberately the SAME code path source.mesh.three_planes
// takes after its resolve: accumulate in linear HDR, split the channels
// horizontally, then one nano_vcr_grade() call. The only difference between
// the two effects is where the halo came from — analytic distance there, a
// convolution pyramid here.

#include "nano_vcr.hlsl"
#include "nano_coords.hlsl"

Texture2D<float4>   inputTex : register(t0);
Texture2D<float4>   glowTex  : register(t1);   // S[0], half res
Texture2D<float4>   emitTex  : register(t2);   // D[0], the raw emitter (debug)
RWTexture2D<float4> outputTex: register(u3);
SamplerState        samp     : register(s4);

cbuffer Uniforms : register(b5) {
  float vp_w;
  float vp_h;
  float chroma_uv_off;   // per-channel horizontal displacement, in uv
  float debug_mode;      // 0 none, 1 halo only, 2 emitter only

  float input_gain;
  float halo_gain;
  float _pad0;
  float _pad1;

  VcrGrade grade;
};

// The linear HDR accumulator, sampled at one position. Called three times
// when the chroma split is on — the split then separates the source AND its
// halo, which is what a tape transport actually does to both.
float3 accumulate(float2 uv) {
  float3 base = inputTex.SampleLevel(samp, uv, 0).rgb * input_gain;
  float3 glow = glowTex.SampleLevel(samp, uv, 0).rgb * halo_gain;
  return max(base + glow, 0.0);
}

[numthreads(8, 8, 1)]
void main(uint3 gid : SV_DispatchThreadID) {
  uint W, H;
  outputTex.GetDimensions(W, H);
  if (gid.x >= W || gid.y >= H) return;

  float2 uv = nano_pixel_to_uv(float2(gid.xy), float2(vp_w, vp_h));

  int dbg = int(debug_mode + 0.5);
  if (dbg == 1) {
    // Halo alone, ungraded, on black — the only honest way to judge radius
    // and falloff while the grade is busy crushing everything.
    float3 g = glowTex.SampleLevel(samp, uv, 0).rgb * halo_gain;
    outputTex[gid.xy] = float4(g, 1.0);
    return;
  }
  if (dbg == 2) {
    // What was allowed to glow in the first place — check Threshold, Knee
    // and Outline here, not in the finished image.
    float3 e = emitTex.SampleLevel(samp, uv, 0).rgb;
    outputTex[gid.xy] = float4(e, 1.0);
    return;
  }

  float3 c;
  if (chroma_uv_off > 0.0) {
    c.r = accumulate(uv + float2(-chroma_uv_off, 0.0)).r;
    c.g = accumulate(uv).g;
    c.b = accumulate(uv + float2( chroma_uv_off, 0.0)).b;
  } else {
    c = accumulate(uv);
  }

  float3 graded = nano_vcr_grade(c, uv, grade);
  float  src_a  = inputTex.SampleLevel(samp, uv, 0).a;
  // Stay layerable: alpha is whatever came in, plus whatever the halo added
  // outside it (a glow that spills past the source's silhouette has to carry
  // its own coverage or it vanishes the moment this is composited).
  float  a = saturate(max(src_a, max(graded.r, max(graded.g, graded.b))));
  outputTex[gid.xy] = float4(graded, a);
}
