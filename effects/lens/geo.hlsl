// filter.blur.lens — pass 7 (image-plane geometry). One per-channel radial
// resample: geometric distortion (1 + k1·r² + k2·r⁴) warps all channels together,
// and transverse chromatic aberration magnifies R/B oppositely ∝ r² so high-
// contrast edges fringe red/cyan toward the corners. (pipeline.pass_geo :454.)
// Full-res; border (clamp-to-edge) address to avoid black creep at the frame edge.

#include "common.hlsl"

Texture2D<float4>   inputTex  : register(t0);
SamplerState        samp      : register(s1);   // Linear + ClampToEdge (≈ border)
RWTexture2D<float4> outputTex : register(u2);
cbuffer Uniforms : register(b3) {
  float u_half, u_dimw, u_dimh, u_distortion;
  float u_wave, u_tca, _p0, _p1;
};

float sampleChan(float2 px, int ch) {
  float4 c = inputTex.SampleLevel(samp, (px + 0.5) / float2(u_dimw, u_dimh), 0.0);
  return ch == 0 ? c.r : (ch == 1 ? c.g : c.b);
}

[numthreads(8, 8, 1)]
void main(uint3 gid : SV_DispatchThreadID) {
  uint w, h;
  outputTex.GetDimensions(w, h);
  if (gid.x >= w || gid.y >= h) return;
  float2 pf = float2(gid.xy);

  float gx = (pf.x + 0.5 - u_dimw * 0.5) / u_half;
  float gy = (pf.y + 0.5 - u_dimh * 0.5) / u_half;
  float rr2 = gx * gx + gy * gy;
  float distort = 1.0 + u_distortion * rr2 + u_wave * rr2 * rr2;
  float sx = gx * distort, sy = gy * distort;

  float3 res;
  // R: +tca, G: 0, B: −tca.
  [unroll] for (int ch = 0; ch < 3; ch++) {
    float tc = ch == 0 ? u_tca : (ch == 2 ? -u_tca : 0.0);
    float cs = 1.0 + tc * rr2;
    float px = sx * cs * u_half + u_dimw * 0.5 - 0.5;
    float py = sy * cs * u_half + u_dimh * 0.5 - 0.5;
    res[ch] = sampleChan(float2(px, py), ch);
  }
  outputTex[gid.xy] = float4(res, inputTex[gid.xy].a);
}
