// motion.local_delay — shared math for all passes.
//
// One header so the passes agree byte-for-byte on the colinear vector
// alignment, the stochastic + vignette mask, the power-squashed blend
// weight, and the manual bilinear sampling used by the pyramidal
// Lucas-Kanade flow estimator. Drift between passes would be invisible
// at a glance and miserable to debug — keep the math here.

#ifndef LOCAL_DELAY_COMMON_HLSL
#define LOCAL_DELAY_COMMON_HLSL

#include "nano_curves.hlsl"   // nano_apply_curve (power-curve squash, §1.3)
#include "nano_coords.hlsl"   // nano_pixel_to_cover_square (vignette, §1.5)
#include "nano_twitch.hlsl"   // nano_twitch_mask (roaming twitch vignette)

// PCG-style integer hash + pixel hash (from debug.motion_static) — the
// stochastic mask is stable and seed-reproducible, no frac-sin banding.
uint ld_pcg_hash(uint x) {
  x = x * 747796405u + 2891336453u;
  uint word = ((x >> ((x >> 28u) + 4u)) ^ x) * 277803737u;
  return (word >> 22u) ^ word;
}
float ld_hash(uint2 p, uint seed) {
  uint h = ld_pcg_hash(p.x + ld_pcg_hash(p.y + ld_pcg_hash(seed)));
  return float(h) * (1.0 / 4294967296.0);
}

// Rec.601 luma — the scalar the flow estimator tracks.
float ld_luma(float3 c) { return dot(c, float3(0.299, 0.587, 0.114)); }

// Manual bilinear sampling (no GPU sampler — keeps every pass on plain
// integer loads, and avoids the R32F-not-filterable restriction by keeping
// luma in the R channel of a filterable RGBA16F texture). `p` is in
// texel-index space: integer coordinate = texel center.
float ld_bil_r(Texture2D<float4> tex, float2 p, int2 dims) {
  int2 i0 = (int2)floor(p);
  float2 f = p - floor(p);
  int2 a = clamp(i0,                 int2(0, 0), dims - 1);
  int2 b = clamp(i0 + int2(1, 0),    int2(0, 0), dims - 1);
  int2 c = clamp(i0 + int2(0, 1),    int2(0, 0), dims - 1);
  int2 d = clamp(i0 + int2(1, 1),    int2(0, 0), dims - 1);
  float c00 = tex[uint2(a)].x, c10 = tex[uint2(b)].x, c01 = tex[uint2(c)].x, c11 = tex[uint2(d)].x;
  return lerp(lerp(c00, c10, f.x), lerp(c01, c11, f.x), f.y);
}
float2 ld_bil_flow(Texture2D<float4> tex, float2 p, int2 dims) {
  int2 i0 = (int2)floor(p);
  float2 f = p - floor(p);
  int2 a = clamp(i0,                 int2(0, 0), dims - 1);
  int2 b = clamp(i0 + int2(1, 0),    int2(0, 0), dims - 1);
  int2 c = clamp(i0 + int2(0, 1),    int2(0, 0), dims - 1);
  int2 d = clamp(i0 + int2(1, 1),    int2(0, 0), dims - 1);
  float2 c00 = tex[uint2(a)].xy, c10 = tex[uint2(b)].xy;
  float2 c01 = tex[uint2(c)].xy, c11 = tex[uint2(d)].xy;
  return lerp(lerp(c00, c10, f.x), lerp(c01, c11, f.x), f.y);
}
float3 ld_bil_rgb(Texture2D<float4> tex, float2 p, int2 dims) {
  int2 i0 = (int2)floor(p);
  float2 f = p - floor(p);
  int2 a = clamp(i0,                 int2(0, 0), dims - 1);
  int2 b = clamp(i0 + int2(1, 0),    int2(0, 0), dims - 1);
  int2 c = clamp(i0 + int2(0, 1),    int2(0, 0), dims - 1);
  int2 d = clamp(i0 + int2(1, 1),    int2(0, 0), dims - 1);
  float3 c00 = tex[uint2(a)].rgb, c10 = tex[uint2(b)].rgb;
  float3 c01 = tex[uint2(c)].rgb, c11 = tex[uint2(d)].rgb;
  return lerp(lerp(c00, c10, f.x), lerp(c01, c11, f.x), f.y);
}

// Schema mirror shared by the align/color/motion passes' cbuffers.
struct LdParams {
  float delay_amount;       // 0..1: blend toward the history frame
  float noise_weight;       // 0..1: how strongly the stochastic mask suppresses
  float seed;               // effective seed = user_seed*17 + step_count
  float weight_gain;        // masked-magnitude → pre-curve weight scale

  float vignette;           // -1..1: + suppresses outside radius, - suppresses inside
  float vignette_radius;    // 0..1: cover-square distance where falloff starts
  float vignette_softness;  // 0..1: falloff width
  float squash;             // -1..1: power-curve slider applied to the weight

  float max_flow;           // uv/frame ceiling on the flow magnitude
  float align_amount;       // 0..1: raw↔colinear-aligned flow lerp
  float align_sharpness;    // colinearity exponent k (higher = stricter)
  float have_history;       // 0/1: first frame has no valid history → flow forced 0

  float2 aspect;            // cover-square half-extents (fx::coverSquare)
  float debug_show_motion;  // 0/1: show ONLY the motion field in the color pass
};

// Build an LdParams from the loose cbuffer scalars. One place defines the
// field order so the passes can't drift.
LdParams ld_make(float delay_amount, float noise_weight, float seed, float weight_gain,
                 float vignette, float vignette_radius, float vignette_softness, float squash,
                 float max_flow, float align_amount, float align_sharpness, float have_history,
                 float aspect_x, float aspect_y, float debug_show_motion) {
  LdParams P;
  P.delay_amount = delay_amount;
  P.noise_weight = noise_weight;
  P.seed = seed;
  P.weight_gain = weight_gain;
  P.vignette = vignette;
  P.vignette_radius = vignette_radius;
  P.vignette_softness = vignette_softness;
  P.squash = squash;
  P.max_flow = max_flow;
  P.align_amount = align_amount;
  P.align_sharpness = align_sharpness;
  P.have_history = have_history;
  P.aspect = float2(aspect_x, aspect_y);
  P.debug_show_motion = debug_show_motion;
  return P;
}

// Colinearity-weighted 3x3 blur of the (already smooth) flow field — a
// cheap final polish on top of the LK estimate. Each neighbor contributes
// in proportion to how parallel it is to the center vector.
float2 ld_align_at(Texture2D<float4> flowTex, uint2 gid, uint w, uint h, LdParams P) {
  float2 c = flowTex[gid].xy;
  float clen = length(c);
  if (clen < 1e-7 || P.align_amount <= 0.0) return c;
  float2 cdir = c / clen;
  int2 p = int2(gid);
  int2 hi = int2(int(w) - 1, int(h) - 1);
  float2 acc = float2(0.0, 0.0);
  float wsum = 0.0;
  [unroll] for (int dy = -1; dy <= 1; dy++) {
    [unroll] for (int dx = -1; dx <= 1; dx++) {
      int2 q = clamp(p + int2(dx, dy), int2(0, 0), hi);
      float2 nv = flowTex[uint2(q)].xy;
      float nlen = length(nv);
      float wgt = (nlen < 1e-7) ? 0.0
                : pow(max(dot(cdir, nv / nlen), 0.0), P.align_sharpness);
      acc += nv * wgt;
      wsum += wgt;
    }
  }
  float2 aligned = (wsum > 1e-6) ? acc / wsum : c;
  return lerp(c, aligned, saturate(P.align_amount));
}

// Twitch mask now lives in shaders_common/nano_twitch.hlsl as
// nano_twitch_mask() (CPU side: fx::TwitchMask in effect_twitch_mask.h).

// Spatial/stochastic mask — WHERE (and how much) the effect acts.
//   noise term  : `noise_weight` is the PROBABILITY a pixel is affected by
//                 noise (a clean binary selection). Affected pixels get a
//                 BALANCED multiplier in [0,2] — ~half boost, ~half cut,
//                 averaging 1 — so sweeping noise_weight adds stochastic
//                 variation without dimming the average effect.
//                 Pixels re-roll INDEPENDENTLY (not all at once): a time-
//                 invariant draw splits into a base seed (integer part) and a
//                 phase offset (fractional part); the per-pixel phase =
//                 floor(noise_time + offset) ticks at staggered moments, so
//                 each pixel flips on its own clock.
//   vignette term: signed cover-square falloff. + suppresses OUTSIDE the
//                 radius, - suppresses INSIDE it, 0 = no mask.
//   twitch term   : a second vignette at a per-frame random `twitch_anchor`,
//                 scaled by `twitch_strength` (amount × random intensity).
// Gates both the color blend and the published motion vectors consistently.
float ld_mask_at(uint2 gid, uint w, uint h, LdParams P, float noise_time,
                 float twitch_shape, float twitch_radius, float twitch_softness,
                 float2 twitch_anchor, float twitch_strength) {
  float r    = ld_hash(gid, uint(P.seed)) * 4096.0;   // time-invariant per-pixel draw
  float toff = frac(r);                                // phase offset (staggers the flip)
  uint  base = (uint)r;                                // integer part = per-pixel base seed
  uint  pseed = base + (uint)floor(noise_time + toff); // per-pixel, ticks at its own time

  float noise_term = 1.0;
  if (ld_hash(gid, pseed) < saturate(P.noise_weight)) {
    noise_term = ld_hash(gid, pseed + 0x9E3779B9u) * 2.0;
  }

  float2 sq = nano_pixel_to_cover_square(float2(gid), float2(w, h), P.aspect);
  float t = smoothstep(P.vignette_radius,
                       P.vignette_radius + max(P.vignette_softness, 1e-4),
                       length(sq));
  float suppress = (P.vignette >= 0.0) ? t : (1.0 - t);
  float vign_term = lerp(1.0, 1.0 - suppress, abs(P.vignette));

  float twitch_term = nano_twitch_mask(sq, twitch_anchor, twitch_radius, twitch_softness,
                                 twitch_shape, twitch_strength);
  return noise_term * vign_term * twitch_term;
}

// Per-pixel temporal-lookup driver in [0,1] — how far (as a fraction of the
// configured depth) this pixel reaches back in time. Motion magnitude is
// scaled by `weight_gain` (sensitivity) into [0,1], shaped by `squash`
// (response curve), and ONLY THEN gated by the mask. Keeping the mask OUTSIDE
// the saturate is the fix for the binary-vignette bug: idx = motion * mask, so
// a soft mask gradient maps straight through even when motion saturates to 1.
float ld_index_at(float2 flow, float mask, LdParams P) {
  float motion = saturate(length(flow) * P.weight_gain);
  return nano_apply_curve(motion, P.squash) * mask;
}

// HSV → RGB (debug viz only).
float3 ld_hsv_to_rgb(float h, float s, float v) {
  float h6 = h * 6.0;
  float c = v * s;
  float x = c * (1.0 - abs(fmod(h6, 2.0) - 1.0));
  float m = v - c;
  float3 rgb;
  if      (h6 < 1.0) rgb = float3(c, x, 0);
  else if (h6 < 2.0) rgb = float3(x, c, 0);
  else if (h6 < 3.0) rgb = float3(0, c, x);
  else if (h6 < 4.0) rgb = float3(0, x, c);
  else if (h6 < 5.0) rgb = float3(x, 0, c);
  else               rgb = float3(c, 0, x);
  return rgb + m;
}

#endif
