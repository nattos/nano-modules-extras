// filter.blur.lens — pass 2 (bokeh gather). The single-plane DOF cost centre
// (pipeline.pass_bokeh :157). For each output pixel, gather K fixed Vogel-disc
// taps, per-pixel warped (cat's-eye pupil clip, swirl, anamorphic, field-curvature
// CoC, per-channel LoCA) and energy-normalised. Runs on the linear-HDR buffer.
//
// STAGE 2: full-resolution gather (no downsample yet — added with the quality
// tiers). The tap set O[k] + base_w[k] (aperture·rim·apodization) is a per-frame
// uniform precompute uploaded as a storage buffer; only the pupil clip is
// per-pixel. K, the mono/LoCA branch, and the warp are runtime here.

#include "common.hlsl"

Texture2D<float4>          inputTex  : register(t0);   // prepared linear HDR
StructuredBuffer<float4>   taps      : register(t1);   // (ox, oy, base_w, _)
SamplerState               samp      : register(s2);
RWTexture2D<float4>        outputTex : register(u3);
cbuffer Uniforms : register(b4) {
  float u_half;         // max(W,H)*0.5  (working res)
  float u_dimw, u_dimh; // working W,H
  float u_coc_px;       // effective CoC radius (px, working res)
  float u_field_curv;
  float u_focus_cx, u_focus_cy;
  float u_cats_eye;
  float u_swirl, u_anamorphic, u_loca_scale;
  uint  u_taps;
};

float3 sampleRGB(float2 pix) {
  return inputTex.SampleLevel(samp, (pix + 0.5) / float2(u_dimw, u_dimh), 0.0).rgb;
}

[numthreads(8, 8, 1)]
void main(uint3 gid : SV_DispatchThreadID) {
  uint w, h;
  outputTex.GetDimensions(w, h);
  if (gid.x >= w || gid.y >= h) return;
  float2 pf = float2(gid.xy);

  // cover-square from the optical centre (0,0): pupil / swirl geometry.
  float gx = (pf.x + 0.5 - u_dimw * 0.5) / u_half;
  float gy = (pf.y + 0.5 - u_dimh * 0.5) / u_half;
  float2 xy0 = float2(gx, gy);
  float  rf  = max(length(xy0), 1e-6);
  float2 rhat = xy0 / rf;

  // cover-square from the focus centre: CoC radius field.
  float2 fxy = float2(gx - u_focus_cx, gy - u_focus_cy);
  float  rF  = length(fxy);
  float  R   = u_coc_px * (1.0 + u_field_curv * rF * rF);   // px

  float  s2   = u_cats_eye * saturate(rf);
  float2 disp = s2 * rhat;                                  // pupil displacement
  float  ang  = u_swirl * rf;
  float  ca = cos(ang), sa = sin(ang);
  float  ax = 1.0 + u_anamorphic, ay = 1.0 - u_anamorphic;

  bool   mono = u_loca_scale < 1e-4;
  float3 sc   = float3(1.0 + u_loca_scale, 1.0, 1.0 - u_loca_scale);

  float3 num = 0.0.xxx;
  float  den = 0.0;
  uint K = u_taps;
  for (uint k = 0; k < K; k++) {
    float4 t = taps[k];
    float2 o = t.xy;
    float  bw = t.z;
    float ox = o.x * ax, oy = o.y * ay;
    float rx = ca * ox - sa * oy;
    float ry = sa * ox + ca * oy;
    // per-pixel pupil clip (cat's-eye) — uses the UNSCALED tap.
    float dlen = length(o - disp);
    float pw = lens_smoothstep_down(1.0, dlen, 0.06);
    float wk = bw * pw;
    if (mono) {
      num += sampleRGB(pf + float2(rx * R, ry * R)) * wk;
    } else {
      float rc = sampleRGB(pf + float2(rx * R * sc.r, ry * R * sc.r)).r;
      float gc = sampleRGB(pf + float2(rx * R,        ry * R)).g;
      float bc = sampleRGB(pf + float2(rx * R * sc.b, ry * R * sc.b)).b;
      num += float3(rc, gc, bc) * wk;
    }
    den += wk;
  }

  outputTex[gid.xy] = float4(num / max(den, 1e-6), 1.0);
}
