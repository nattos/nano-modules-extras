// filter.height_from_gradient — present / visualization pass.
//
// Turns the reconstructed full-res height into the output image. Four modes:
//   Hillshade (default) — Lambertian-shade the surface normal (built from the
//                         height's local slope) against an adjustable light.
//                         DC-invariant, so the Poisson null-space (height is
//                         defined up to a constant) doesn't matter here.
//   Grayscale           — the height as brightness, normalized to its global
//                         (min,max) range, with scale/offset fine-tuning.
//   Normals             — RG-encoded surface normal (data/producer view).
//   Contours            — iso-lines of the reconstructed (normalized) height.
// `mix_amount` cross-fades the visualization back toward the input image.
// `debug_show_gradient` overrides everything with the source gradient field.

#include "common.hlsl"

Texture2D<float4>   heightTex : register(t0);   // reconstructed height (R)
Texture2D<float4>   gradTex   : register(t1);   // source gradient (RG) — debug
Texture2D<float4>   inputTex  : register(t2);   // original input (mix)
Texture2D<float4>   minmaxTex : register(t3);   // 1x1 global height (min, max)
RWTexture2D<float4> outTex    : register(u4);   // rgba8 output

cbuffer Uniforms : register(b5) { HFG_UNIFORMS };

[numthreads(8, 8, 1)]
void main(uint3 gid : SV_DispatchThreadID) {
  uint w, h;
  outTex.GetDimensions(w, h);
  if (gid.x >= w || gid.y >= h) return;

  int2 hi = int2(int(w) - 1, int(h) - 1);
  int2 p  = int2(gid.xy);

  float hC = heightTex[gid.xy].x;
  float hL = heightTex[uint2(clamp(p + int2(-1, 0), int2(0, 0), hi))].x;
  float hR = heightTex[uint2(clamp(p + int2( 1, 0), int2(0, 0), hi))].x;
  float hD = heightTex[uint2(clamp(p + int2(0, -1), int2(0, 0), hi))].x;
  float hU = heightTex[uint2(clamp(p + int2(0,  1), int2(0, 0), hi))].x;

  // Raw per-pixel slope from central differences (used by the normal AND the
  // contour spacing); relief_scale (0..1) steepens the normal. Typical
  // reconstructed-height slopes are small, so map through a gain.
  float2 rawSlope = float2((hR - hL) * 0.5, (hU - hD) * 0.5);
  float s = relief_scale * 16.0;
  float3 n = normalize(float3(-rawSlope.x * s, -rawSlope.y * s, 1.0));

  // Normalize the height to [0,1] by its global range — the Poisson solution
  // has an arbitrary scale, so grayscale and contours work off the normalized
  // value (hillshade/normals use slope and are scale/DC-invariant).
  float2 mm = minmaxTex[uint2(0, 0)].xy;
  float range = max(mm.y - mm.x, 1e-5);
  float hN = (hC - mm.x) / range;

  float3 rgb;
  if (present_mode > 2.5) {
    // Contours — iso-lines of OUR reconstructed height. A constant-width line
    // wherever the NORMALIZED height crosses a level. No fragment derivatives
    // in a compute pass, so the per-pixel rate of change is the manual height
    // gradient: levels-per-pixel = |grad hN| * density. Distance (in levels) to
    // the nearest level / that rate = distance in pixels → a crisp line. Where
    // levels pack tighter than they can be drawn (steep spots / high density),
    // FADE the line so contours degrade to nothing rather than smearing solid.
    float density = contour_density * 32.0;
    float hs = hN * density;
    float dist_lvl = min(frac(hs), 1.0 - frac(hs));       // [0,0.5] level units
    float lpp = (length(rawSlope) / range) * density;     // levels per pixel
    float dist_px = dist_lvl / max(lpp, 1e-6);
    // Exponential line half-width (px): razor-thin (sub-pixel, aliasing
    // accepted) at the low end → a few px at the top, no floor. The AA span
    // shrinks with W so thin lines actually stay thin instead of being held to
    // a ~1px ramp.
    float W = 0.02 * pow(200.0, line_width);
    float aa = min(W, 0.5);
    float lineMask = 1.0 - smoothstep(W, W + aa, dist_px);
    float cell_px = 1.0 / max(lpp, 1e-6);                 // pixels between levels
    // Fade only where levels pack tighter than the pixel grid can resolve (the
    // steep-spot / over-dense artifact) — independent of line width, so thick
    // lines still read as thick rather than vanishing.
    float fade = saturate(cell_px - 1.0);
    rgb = lineMask * fade * float3(tint_r, tint_g, tint_b);
  } else if (present_mode > 1.5) {
    // Normals
    rgb = n * 0.5 + 0.5;
  } else if (present_mode > 0.5) {
    // Grayscale height (normalized; height_scale/offset fine-tune)
    float v = saturate(hN * height_scale + height_offset);
    rgb = v * float3(tint_r, tint_g, tint_b);
  } else {
    // Hillshade relief (default)
    float3 light = float3(light_x, light_y, light_z);
    float shade = saturate(dot(n, light) * light_gain + ambient);
    rgb = shade * float3(tint_r, tint_g, tint_b);
  }

  rgb = lerp(rgb, inputTex[gid.xy].rgb, saturate(mix_amount));

  if (debug_show_gradient > 0.5) {
    float2 g = gradTex[gid.xy].xy;
    rgb = float3(g * 2.0 + 0.5, 0.5);
  }

  outTex[gid.xy] = float4(saturate(rgb), 1.0);
}
