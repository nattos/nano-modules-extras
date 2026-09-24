// filter.blur.lens — pass 8 (finish). Exposure → mechanical vignette → highlight
// desaturation → filmic tonemap (Reinhard→ACES) → toe crush → sRGB → film grain.
// (pipeline.pass_finish :503.) Reads the linear-HDR pipeline output, writes the
// display-space RGBA8 tex_out. This stage always runs — it produces the image.

#include "common.hlsl"

Texture2D<float4>   inputTex  : register(t0);   // linear HDR
RWTexture2D<float4> outputTex : register(u1);
cbuffer Uniforms : register(b2) {
  float u_half;         // max(W,H)*0.5
  float u_dimw, u_dimh;
  float u_exposure;     // linear multiplier (2^(3·stops)) host-side
  float u_vignette;
  float u_hl_desat;
  float u_tone;
  float u_tone_black;
  float u_grain;
  float _p0, _p1, _p2;
};

[numthreads(8, 8, 1)]
void main(uint3 gid : SV_DispatchThreadID) {
  uint w, h;
  outputTex.GetDimensions(w, h);
  if (gid.x >= w || gid.y >= h) return;
  float2 pf = float2(gid.xy);

  float4 cin = inputTex[gid.xy];
  float3 x = cin.rgb * u_exposure;

  // mechanical vignette (cos^4-ish via r^2), cover-square from centre.
  float gx = (pf.x + 0.5 - u_dimw * 0.5) / u_half;
  float gy = (pf.y + 0.5 - u_dimh * 0.5) / u_half;
  float rf2 = gx * gx + gy * gy;
  float vig = saturate(1.0 - u_vignette * clamp(rf2, 0.0, 1.5));
  x *= vig;

  // highlight desaturation → roll bright chroma toward the peak channel (white).
  if (u_hl_desat > 1e-4) {
    float mx = max(x.r, max(x.g, x.b));
    float d = u_hl_desat * lens_smoothstep(1.0, 5.0, mx);
    x = lerp(x, mx.xxx, d);
  }

  // tonemap: extended-Reinhard base blended toward ACES.
  float3 reinhard = x * (1.0 + x / 36.0) / (1.0 + x);
  x = lerp(reinhard, lens_aces(x), u_tone);
  if (u_tone_black > 1e-4)
    x = saturate((x - u_tone_black) / (1.0 - u_tone_black));

  float3 outc = lens_linear_to_srgb(x);

  // fine film grain, strongest in the mids, in display space.
  if (u_grain > 1e-4) {
    float n = sin(pf.x * 12.9898 + pf.y * 78.233) * 43758.5453;
    float g = frac(n) * 2.0 - 1.0;
    float ymid = (outc.r + outc.g + outc.b) / 3.0;
    float weight = saturate(4.0 * ymid * (1.0 - ymid));
    outc = saturate(outc + u_grain * 0.25 * g * weight);
  }

  outputTex[gid.xy] = float4(outc, cin.a);
}
