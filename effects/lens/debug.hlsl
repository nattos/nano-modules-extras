// filter.blur.lens — debug views. Replaces the finish pass when debug_view != 0,
// rendering an internal stage to the display output for inspection:
//   1 Highlight Mask — what the effect treats as a highlight (input luma vs
//     threshold), as a grayscale mask.
//   2 CoC Field — the per-pixel circle-of-confusion radius (the same analytic
//     field bokeh.hlsl gathers over), grayscale: black = in focus, white = max blur.
//   3 Bokeh Only — the isolated depth-of-field result (host skips coating/flare/
//     geo), tonemapped.
//   4 Flare Only — the flare/glow contribution alone: the post-flare image minus a
//     pre-flare snapshot (host passes the snapshot as auxTex), tonemapped.
// All ALU + two loads — no gathers. Writes display-space RGBA8.

#include "common.hlsl"

Texture2D<float4>   srcTex    : register(t0);   // composited / selected linear image
Texture2D<float4>   auxTex    : register(t1);   // tex_in (mask) OR pre-flare snapshot (flare)
RWTexture2D<float4> outputTex : register(u2);
cbuffer Uniforms : register(b3) {
  float u_mode;          // 1 mask, 2 coc, 3 bokeh, 4 flare
  float u_hl_threshold;
  float u_half;          // max(W,H)*0.5
  float u_dimw, u_dimh;
  float u_coc_px;        // baseline CoC radius at the focus centre (px)
  float u_field_curv;
  float u_focus_cx, u_focus_cy;
  float _p0, _p1, _p2;
};

// Extended-Reinhard + sRGB, matching finish.hlsl's low end (enough to make the
// linear-HDR debug buffers legible without the full filmic curve).
float3 dbg_tonemap(float3 x) {
  float3 r = x * (1.0 + x / 36.0) / (1.0 + x);
  return lens_linear_to_srgb(saturate(r));
}

[numthreads(8, 8, 1)]
void main(uint3 gid : SV_DispatchThreadID) {
  uint w, h;
  outputTex.GetDimensions(w, h);
  if (gid.x >= w || gid.y >= h) return;
  float2 pf = float2(gid.xy);
  int mode = (int)(u_mode + 0.5);

  float3 outc;
  if (mode == 1) {
    // highlight mask from the ORIGINAL input (sRGB → linear luma vs threshold).
    float luma = lens_luma(lens_srgb_to_linear(auxTex[gid.xy].rgb));
    outc = lens_smoothstep(u_hl_threshold, 2.0 * u_hl_threshold, luma).xxx;
  } else if (mode == 2) {
    // CoC field — matches bokeh.hlsl: R = coc_px·(1 + field_curv·rF²), rF from the
    // focus centre in cover-square coords. Normalised to the max blur radius.
    float gx = (pf.x + 0.5 - u_dimw * 0.5) / u_half;
    float gy = (pf.y + 0.5 - u_dimh * 0.5) / u_half;
    float2 fxy = float2(gx - u_focus_cx, gy - u_focus_cy);
    float rF = length(fxy);
    float R = u_coc_px * (1.0 + u_field_curv * rF * rF);
    float md = min(u_dimw, u_dimh);
    float n = md > 1e-4 ? saturate(R / (0.25 * md)) : 0.0;   // 0.25·md = max coc_px
    outc = n.xxx;
  } else if (mode == 4) {
    // flare only = post-flare image − pre-flare snapshot (clamped ≥ 0).
    float3 fl = max(srcTex[gid.xy].rgb - auxTex[gid.xy].rgb, 0.0.xxx);
    outc = dbg_tonemap(fl);
  } else {
    // bokeh only (3) + fallback: tonemap the selected linear buffer.
    outc = dbg_tonemap(srcTex[gid.xy].rgb);
  }
  outputTex[gid.xy] = float4(outc, 1.0);
}
