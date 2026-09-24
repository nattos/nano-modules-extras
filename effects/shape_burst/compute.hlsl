// shape_burst/compute.hlsl — color pass. Draws up to N concentric expanding
// rings (circle / square / triangle) over a background, one compute dispatch.
//
// Each active "voice" contributes a ring at its own scale (cover-square units).
// Strokes have sharp corners (see sb_ring_cov) and are either hard-cut solid
// or shaded by a smooth across-stroke bell (sb_shade, tiltable inner<->outer);
// antialiased with a fixed-width smoothstep band (house style — no fwidth in
// this tree). Composited alpha-over onto a background chosen by
// composite_mode (black / transparent / custom / input).

#include "common.hlsl"

Texture2D<float4>   inputTex  : register(t0);
RWTexture2D<float4> outputTex : register(u1);

[numthreads(8, 8, 1)]
void main(uint3 gid : SV_DispatchThreadID) {
  uint w, h;
  outputTex.GetDimensions(w, h);
  if (gid.x >= w || gid.y >= h) return;

  // Background per composite mode.
  float4 acc;
  if      (u_composite == 0u) acc = float4(0.0, 0.0, 0.0, 1.0);          // black
  else if (u_composite == 1u) acc = float4(0.0, 0.0, 0.0, 0.0);          // transparent
  else if (u_composite == 2u) acc = u_bg;                               // custom
  else                        acc = inputTex.Load(int3(gid.xy, 0));      // input

  float2 sq = nano_pixel_to_cover_square(float2(gid.xy), float2(w, h), u_aspect);
  float2 p  = sq - u_center;

  const float half_t = max(u_thickness * 0.5, 1e-5);
  const float aa     = max(u_px, 1e-5);

  for (uint i = 0u; i < u_count; ++i) {
    float s = u_scales[i / 4u][i % 4u];
    if (s <= 0.0) continue;
    // Canonical perimeter position (angle-only, ignores stroke width).
    float2 dir = (dot(p, p) > 1e-12) ? normalize(p) : float2(1.0, 0.0);
    float disp = sb_distort(dir * s, u_dist_seeds[i / 4u][i % 4u]);
    float2 pr = sb_unrotate(p, u_rotations[i / 4u][i % 4u]);
    float se  = s + disp;
    float cov = sb_ring_cov(pr, se, half_t, aa, u_shape_kind);
    if (cov <= 0.0) continue;
    cov *= sb_shade(sb_ring_band(pr, se, half_t, u_shape_kind));
    float a = u_color.a * cov;                        // alpha-over
    acc.rgb = lerp(acc.rgb, u_color.rgb, a);
    acc.a   = a + acc.a * (1.0 - a);
  }

  outputTex[gid.xy] = acc;
}
