// warp.legacy.zoom_scroller — render pass.
//
// Two jobs in one compute shader (the original used a Transform + a Shape
// Render rectangle outline + a Video Mixer):
//   1. Pan/zoom resample of the input: sample tex_in at a window centred on
//      `pan` (cover-square coords) and zoomed by `scale`, using a selectable
//      reconstruction filter (Crisp/Linear/Smooth-bicubic).
//   2. Gizmo overlay: an analytic rectangle OUTLINE box, centred and nudged in
//      the direction of motion (`giz_off`), composited over the result.
//
// On the zoom: a plain bilinear tap at LOD 0 over a single-mip texture leaves
// magnified high-contrast detail "crunchy" (you see the source texel grid; thin
// lines alias). Catmull-Rom bicubic (the "Smooth" default) reconstructs it
// smoothly with the standard 4-bilinear-tap trick — the usual upgrade for a
// zoom/pan resampler.
//
// All positions are in cover-square coords (see nano_coords.hlsl): (0,0) =
// centre, ±1 along the long axis = the viewport edge. main.cpp feeds the
// already-resolved per-frame quantities; this shader is pure.

#include "nano_coords.hlsl"

Texture2D<float4>   inputTex  : register(t0);
SamplerState        samp      : register(s2);   // Linear / ClampToEdge
RWTexture2D<float4> outputTex : register(u1);

cbuffer Uniforms : register(b3) {
  float pan_x, pan_y;        // source-window centre (cover-square)
  float scale;               // zoom factor (>1 = zoomed in)
  float aspect_x;            // cover-square aspect (ax)
  float aspect_y;            // cover-square aspect (ay)
  float giz_off_x, giz_off_y;// gizmo box centre offset (cover-square)
  float giz_hw, giz_hh;      // gizmo box half-extents (cover-square)
  float giz_thick;           // outline half-thickness (cover-square)
  float giz_show;            // 0/1
  float giz_alpha;           // overlay opacity
  float giz_r, giz_g, giz_b; // outline colour
  float filter_mode;         // 0 = crisp/nearest, 1 = linear, 2 = smooth/bicubic
  float _pad1, _pad2, _pad3;
};

// B-spline bicubic via 4 bilinear taps (Sigg & Hadwiger, GPU Gems 2). B-spline
// weights are all non-negative, so the s0/s1 = wA+wB denominators are always
// positive — unlike Catmull-Rom, whose negative lobes make this 4-tap trick
// divide by ~0 (→ NaN). B-spline is the *smoothing* cubic (slightly soft),
// which is exactly what we want to de-"crunch" a magnified image.
float4 sampleBicubic(float2 uv, float2 texSize) {
  float2 invTex = 1.0 / texSize;
  float2 tc = uv * texSize - 0.5;
  float2 c  = floor(tc) + 0.5;     // centre of the texel we're in (texel space)
  float2 f  = tc + 0.5 - c;        // fractional position in [0,1)

  // Cubic B-spline basis weights (all ≥ 0, sum to 1).
  float2 f2 = f * f, f3 = f2 * f;
  float2 w0 = (1.0 / 6.0) * (1.0 - 3.0 * f + 3.0 * f2 - f3);
  float2 w1 = (1.0 / 6.0) * (4.0 - 6.0 * f2 + 3.0 * f3);
  float2 w2 = (1.0 / 6.0) * (1.0 + 3.0 * f + 3.0 * f2 - 3.0 * f3);
  float2 w3 = (1.0 / 6.0) * f3;

  float2 s0 = w0 + w1;
  float2 s1 = w2 + w3;
  float2 o0 = w1 / s0;             // sub-texel offset for the first linear tap
  float2 o1 = w3 / s1;            // ... and the second

  float2 t0 = (c - 1.0 + o0) * invTex;
  float2 t1 = (c + 1.0 + o1) * invTex;

  float4 a = inputTex.SampleLevel(samp, float2(t0.x, t0.y), 0.0);
  float4 b = inputTex.SampleLevel(samp, float2(t1.x, t0.y), 0.0);
  float4 cc= inputTex.SampleLevel(samp, float2(t0.x, t1.y), 0.0);
  float4 d = inputTex.SampleLevel(samp, float2(t1.x, t1.y), 0.0);

  return (a * s0.x + b * s1.x) * s0.y + (cc * s0.x + d * s1.x) * s1.y;
}

[numthreads(8, 8, 1)]
void main(uint3 gid : SV_DispatchThreadID) {
  uint w, h;
  outputTex.GetDimensions(w, h);
  if (gid.x >= w || gid.y >= h) return;
  uint sw, sh;
  inputTex.GetDimensions(sw, sh);

  float2 aspect = float2(aspect_x, aspect_y);
  float2 sq = nano_pixel_to_cover_square(float2(gid.xy), float2(w, h), aspect);

  // Pan/zoom: the output pixel at cover-square `sq` reads the source at a point
  // scaled toward the window centre `pan`. scale>1 magnifies (smaller window).
  float invScale = (scale > 1e-4) ? (1.0 / scale) : 1.0;
  float2 src_sq = sq * invScale + float2(pan_x, pan_y);
  float2 src_uv = nano_cover_square_to_uv(src_sq, aspect);

  float4 color;
  if (filter_mode < 0.5) {
    // Crisp / nearest.
    float2 texel = (floor(src_uv * float2(sw, sh)) + 0.5) / float2(sw, sh);
    color = inputTex.SampleLevel(samp, texel, 0.0);
  } else if (filter_mode < 1.5) {
    // Plain bilinear.
    color = inputTex.SampleLevel(samp, src_uv, 0.0);
  } else {
    // Smooth bicubic (default).
    color = sampleBicubic(src_uv, float2(sw, sh));
  }

  // Gizmo outline box, centred at the frame centre, nudged by motion.
  // The outline is drawn from the axis-aligned rectangle "box value"
  // b = max(|x|-Hx, |y|-Hy): b == 0 on the outline, and its level sets are
  // concentric rectangles with SQUARE corners (a euclidean SDF would round the
  // outer corners). The band |b| < giz_thick is the frame.
  // Analytic AA: the +0.5 half-pixel term keeps an infinitely thin line a 1px
  // footprint (hairline still shows, just dimmer) instead of vanishing between
  // pixel centres; giz_alpha carries the width-fade so it disappears at width 0.
  if (giz_show > 0.5 && giz_alpha > 0.0) {
    float2 q = abs(sq - float2(giz_off_x, giz_off_y));
    float b = max(q.x - giz_hw, q.y - giz_hh);
    float aa = max(1.5 * aspect_y * 2.0 / float(h), 1e-5);
    float cov = clamp((giz_thick - abs(b)) / aa + 0.5, 0.0, 1.0);
    float a = cov * giz_alpha;
    color.rgb = lerp(color.rgb, float3(giz_r, giz_g, giz_b), a);
  }

  outputTex[gid.xy] = color;
}
