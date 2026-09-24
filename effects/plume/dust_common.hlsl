// source.sdf.plume — dust splat shared half (uniforms + projection).
//
// Dust primary visibility is a two-pass SOFTWARE SPLAT (the render-pass
// ABI has no depth attachment, and a per-ray traversal would tax every
// pixel): pass 1 (dust_depth) resolves the nearest visible particle per
// pixel by InterlockedMin of the ray distance into a screen-sized uint
// buffer; pass 2 (dust_shade) lets exactly the winning particle shade
// its pixel. Both passes iterate one thread per PARTICLE over the same
// tiny disc footprint, so the work scales with particle count, not
// resolution, and dust-vs-dust AND dust-vs-surface depth are exact.
//
// The including file must #define DUST_UB_REG (cbuffer register) before
// including — binding indices are register numbers shared across
// resource types, so the two passes place it differently.

#ifndef PLUME_DUST_COMMON_HLSL
#define PLUME_DUST_COMMON_HLSL

// Two float4 rows per particle: (pos.xyz, radius) / (normal.xyz, seed) —
// the sdf_field rail's dust layout (effect_sdf_field.h).
StructuredBuffer<float4> dust : register(t0);

cbuffer DustSplatUniforms : register(DUST_UB_REG) {
  float4 cam_row0;   // view right (world), w = cam_pos.x
  float4 cam_row1;   // view up (world),    w = cam_pos.y
  float4 cam_row2;   // view fwd (world),   w = cam_pos.z
  float4 cam_p;      // focal, cover_ax, cover_ay, dust_count
  float4 sun_p;      // sun dir (world, toward light), w = intensity
  float4 albedo;     // dust material rgb, w = exposure gain
  float4 vp;         // w, h, 1/w, 1/h
  float4 shade_p;    // shadow, ambient, bounce, inv_lip
  float4 misc;       // px_world (per unit t), dust metallic, 0, 0
};

// Footprint radius cap, pixels — bounds the per-thread pixel loop.
// There is no radius floor: sub-pixel presence is carried by COVERAGE
// (du_cov below), not by inflating the disc.
static const float DUST_MAX_PX = 8.0;

// Feather reach beyond the disc edge, pixels. At half a pixel-diagonal
// the nearest pixel center is always inside the footprint, so a mote
// never blinks out crossing a pixel corner — presence is guaranteed by
// a fading alpha instead of a full-brightness pixel.
static const float DUST_FEATHER = 0.7072;

// Per-pixel coverage of the disc at center-distance `dist`: an area term
// (a sub-pixel disc truly covers ~pi*rp^2 of its one pixel) times a
// feathered edge (larger discs get an anti-aliased rim instead of a
// pixel-center step). Both passes and composite share this contract:
// pass 1 accumulates it per pixel, composite uses the clamped sum as the
// dust layer's alpha.
float du_cov(float dist, float rp) {
  float area = saturate(3.14159265 * rp * rp);
  return area * saturate(rp + DUST_FEATHER - dist);
}

// Particle -> screen. ctr is the footprint center in PIXEL coordinates,
// t the camera distance (the march's ray parameter — rays are unit
// length, so scene .a compares directly), rp the disc radius in pixels.
// False = behind / at the camera plane.
bool du_project(uint tid, out float3 pos, out float3 nrm, out float seed,
                out float2 ctr, out float t, out float rp) {
  float4 r0 = dust[tid * 2u];
  float4 r1 = dust[tid * 2u + 1u];
  pos = r0.xyz;
  nrm = r1.xyz;
  seed = r1.w;
  float3 ro = float3(cam_row0.w, cam_row1.w, cam_row2.w);
  float3 rel = pos - ro;
  float w = dot(rel, cam_row2.xyz);
  ctr = float2(0.0, 0.0);
  t = 0.0;
  rp = 0.0;
  if (w < 0.05) return false;
  t = length(rel);
  // Inverse of the march's ray construction: V = (ndc.x/(2·ax·f),
  // ndc.y/(2·ay·f), 1) before normalization, so ndc = (x/w)·2f·(ax, ay).
  float2 ndc = float2(dot(rel, cam_row0.xyz), dot(rel, cam_row1.xyz))
             / w * (2.0 * cam_p.x) * float2(cam_p.y, cam_p.z);
  ctr = float2(0.5 * (ndc.x + 1.0), 0.5 * (1.0 - ndc.y)) * vp.xy;
  rp = min(r0.w / (misc.x * t), DUST_MAX_PX);
  return true;
}

#endif  // PLUME_DUST_COMMON_HLSL
