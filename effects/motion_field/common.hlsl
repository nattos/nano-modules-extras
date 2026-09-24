// motion.field — shared math for color + motion passes.
//
// Both passes need to compute the per-pixel velocity. Defining it
// once here keeps the two shaders byte-for-byte consistent — any
// drift between the visual overlay and the actual motion vectors
// written for downstream consumers would be invisible at a glance
// and a nightmare to debug.

#ifndef MOTION_FIELD_COMMON_HLSL
#define MOTION_FIELD_COMMON_HLSL

// PCG bit-mix hash. Integer-only, no float-precision artifacts.
uint mf_pcg_hash(uint x) {
  x = x * 747796405u + 2891336453u;
  uint word = ((x >> ((x >> 28u) + 4u)) ^ x) * 277803737u;
  return (word >> 22u) ^ word;
}

uint mf_pcg_hash3(uint a, uint b, uint c) {
  return mf_pcg_hash(a + mf_pcg_hash(b + mf_pcg_hash(c)));
}

// Random in [0, 1] for an integer cell coord + seed.
float mf_cell_rand(int2 cell, uint seed) {
  uint h = mf_pcg_hash3(uint(cell.x + 1000000), uint(cell.y + 1000000), seed);
  return float(h) * (1.0 / 4294967296.0);
}

// 2D value noise: hash-based random per integer cell, smoothed
// bilinearly between cells with a quintic ease curve. Output in
// [0, 1]. `cell_size` is the period of the noise in pixels — bigger
// = smoother (Perlin-ish) variation.
float mf_value_noise(float2 p_pixels, float cell_size, uint seed) {
  float2 p = p_pixels / max(cell_size, 1.0);
  int2 i = int2(floor(p));
  float2 f = frac(p);
  // Quintic ease: smoother than smoothstep, common Perlin choice.
  float2 u = f * f * f * (f * (f * 6.0 - 15.0) + 10.0);
  float a = mf_cell_rand(i,                      seed);
  float b = mf_cell_rand(i + int2(1, 0),         seed);
  float c = mf_cell_rand(i + int2(0, 1),         seed);
  float d = mf_cell_rand(i + int2(1, 1),         seed);
  return lerp(lerp(a, b, u.x), lerp(c, d, u.x), u.y);
}

// Per-cell value at a continuous `time_index`, with staggered ticks.
//
// The naive "snapshot N → snapshot N+1 cross-fade" approach pulses
// globally — every cell transitions at the same time, producing a
// visible breathing rhythm. We avoid that by giving each cell its
// own phase offset:
//
//   sig         = pcg(cell, base_seed XOR mix_prime)
//   phase       = lower 16 bits of sig, normalized to [0, 1)
//   step_index  = floor(phase + time_index)
//   value       = mf_cell_rand(cell, base_seed + step_index)
//
// Because `phase` differs per cell, each cell crosses its own
// integer threshold at a different time. At any moment some cells
// just ticked, others are about to tick — the transitions are spread
// out and look stochastic rather than synchronous. Each cell still
// ticks once per unit `time_index`.
//
// The step is hard (no temporal lerp) — adjacent cells' values
// disagree across the cell boundary anyway, and the spatial bilinear
// interpolation in mf_value_noise_phased smooths the result over
// space. Per-cell hard steps are perceptually preferable to a global
// pulse even with quintic temporal ease.
float mf_cell_rand_phased(int2 cell, uint base_seed, float time_index) {
  // Sig: phase for this cell. XOR by golden-ratio prime decouples
  // sig from any other hash that uses (cell, base_seed) directly,
  // so the cell's value at step 0 isn't correlated with its phase.
  uint sig = mf_pcg_hash3(uint(cell.x + 1000000),
                          uint(cell.y + 1000000),
                          base_seed ^ 0x9E3779B1u);
  float phase = float(sig & 0xFFFFu) * (1.0 / 65536.0);
  uint step_index = uint(floor(phase + time_index));
  return mf_cell_rand(cell, base_seed + step_index);
}

// Spatial bilinear interpolation across 4 corners, each evaluated
// via the per-cell phased rand. Cells in this composition tick
// independently, so the noise field mutates over time without the
// pulsing artifact of a global cross-snapshot lerp.
float mf_value_noise_phased(float2 p_pixels, float cell_size,
                            uint base_seed, float time_index) {
  float2 p = p_pixels / max(cell_size, 1.0);
  int2 i = int2(floor(p));
  float2 f = frac(p);
  float2 u = f * f * f * (f * (f * 6.0 - 15.0) + 10.0);
  float a = mf_cell_rand_phased(i,                base_seed, time_index);
  float b = mf_cell_rand_phased(i + int2(1, 0),   base_seed, time_index);
  float c = mf_cell_rand_phased(i + int2(0, 1),   base_seed, time_index);
  float d = mf_cell_rand_phased(i + int2(1, 1),   base_seed, time_index);
  return lerp(lerp(a, b, u.x), lerp(c, d, u.x), u.y);
}

float mf_lum(float3 c) {
  return dot(c, float3(0.299, 0.587, 0.114));
}

// Central-difference luma gradient at integer pixel `gid`. Returns a
// 2D vector pointing towards higher luma (uphill). Magnitude scales
// with how steep the gradient is.
float2 mf_luma_gradient(Texture2D<float4> tex, uint2 gid, uint w, uint h) {
  // Edge clamping — central difference becomes a half-step at the
  // border, which is fine; we don't care about the exact magnitude
  // for direction-only use, only that the result isn't NaN.
  uint xl = (gid.x > 0u) ? gid.x - 1u : 0u;
  uint xr = (gid.x + 1u < w) ? gid.x + 1u : w - 1u;
  uint yu = (gid.y > 0u) ? gid.y - 1u : 0u;
  uint yd = (gid.y + 1u < h) ? gid.y + 1u : h - 1u;
  float ll = mf_lum(tex[uint2(xl, gid.y)].rgb);
  float lr = mf_lum(tex[uint2(xr, gid.y)].rgb);
  float lu = mf_lum(tex[uint2(gid.x, yu)].rgb);
  float ld = mf_lum(tex[uint2(gid.x, yd)].rgb);
  return float2(lr - ll, ld - lu) * 0.5;
}

// Rotate a 2D vector by `angle_rad`.
float2 mf_rotate(float2 v, float angle_rad) {
  float ca = cos(angle_rad);
  float sa = sin(angle_rad);
  return float2(ca * v.x - sa * v.y, sa * v.x + ca * v.y);
}

// Convert HSV → RGB. Used for the visualization overlay only.
float3 mf_hsv_to_rgb(float h, float s, float v) {
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

// =====================================================================
//  All schema parameters threaded through one struct for readability.
// =====================================================================

struct MfParams {
  // Activation
  float threshold;        // 0..1 luma cutoff
  float softness;         // 0..0.5 soft-transition half-width

  // Magnitude
  float magnitude;        // base velocity (uv per frame)
  float mag_jitter;       // 0..1 — fraction of magnitude varied per pixel
  float mag_noise_scale;  // pixels per noise cell (Perlin-like)

  // Direction components (each weighted; final direction is a
  // normalised sum). Weights don't need to add to 1 — anything
  // greater than zero contributes. All zero → fallback to (1, 0).
  float rotation_rad;     // static direction angle
  float rotation_weight;
  float radial_weight;
  float2 radial_anchor;   // uv-space anchor for the outward direction
  float gradient_weight;
  float gradient_bias_rad;  // rotation applied to the luma gradient
                            // before contributing to the direction sum

  // Per-pixel angular jitter (Perlin-ish locality).
  float angle_jitter;       // ±jitter * π radians of jitter
  float angle_noise_scale;  // pixels per cell

  // Continuous time index into the noise field (host-side
  // `accumulated_time * evolution_rate`). Drives smooth interpolation
  // between adjacent integer-seed snapshots, so the noise field
  // mutates over time when the user sets evolution_rate > 0 and
  // freezes when evolution_rate == 0. Two phase offsets below
  // decorrelate the magnitude and angle noise streams.
  float noise_time;
};

// Phase offsets that drop the magnitude and angle noise streams in
// totally separate regions of the pcg seed space — the two streams
// stay uncorrelated even at the exact same `noise_time`.
static const uint MF_PHASE_MAG   = 0xA1B2C3D4u;
static const uint MF_PHASE_ANGLE = 0x9E3779B1u;  // golden-ratio mix prime

// Returns the per-pixel velocity in uv-space. Pixels with zero
// activation (luma below threshold) get (0, 0).
//
// `inputTex` must be the texture being thresholded — it's also used
// as the source for the luma gradient direction component.
float2 mf_velocity_at(Texture2D<float4> inputTex,
                      uint2 gid, uint w, uint h,
                      MfParams P) {
  float3 c = inputTex[gid].rgb;
  float luma = mf_lum(c);

  // Soft threshold. softness controls how fast we ramp from 0 to 1
  // around the threshold; with softness=0 it would be a hard step
  // (we clamp to a tiny minimum below to avoid that exact case
  // producing NaNs in smoothstep).
  float t_soft = max(P.softness, 1e-4);
  float t_lo = P.threshold - t_soft;
  float t_hi = P.threshold + t_soft;
  float activation = smoothstep(t_lo, t_hi, luma);
  if (activation <= 0.0) return float2(0.0, 0.0);

  // ---- Direction ----
  float2 uv = (float2(gid) + 0.5) / float2(w, h);

  // Static rotation: a unit vector at `rotation_rad`.
  float2 d_static = float2(cos(P.rotation_rad), sin(P.rotation_rad));

  // Radial: outward from the anchor. Normalised; falls back to a
  // tangent of the static direction at the exact anchor pixel.
  float2 r = uv - P.radial_anchor;
  float r_len = length(r);
  float2 d_radial = (r_len > 1e-4) ? (r / r_len) : d_static;

  // Luma gradient direction, rotated by gradient_bias. A bias of 0
  // points uphill (towards brighter pixels); ±π/2 gives the
  // gradient's perpendicular (along iso-luma contours), which is
  // typically what edge-flow effects want.
  float2 grad = mf_luma_gradient(inputTex, gid, w, h);
  float g_len = length(grad);
  float2 d_gradient;
  if (g_len > 1e-4) {
    d_gradient = mf_rotate(grad / g_len, P.gradient_bias_rad);
  } else {
    // Flat region — no gradient direction available, so this term
    // contributes nothing. Setting it to zero keeps the weight sum
    // honest (a flat region with all weight on `gradient` would
    // still fall back to the static direction below).
    d_gradient = float2(0.0, 0.0);
  }

  float2 dir_sum = P.rotation_weight  * d_static
                 + P.radial_weight    * d_radial
                 + P.gradient_weight  * d_gradient;
  float dir_len = length(dir_sum);
  float2 dir = (dir_len > 1e-4) ? (dir_sum / dir_len) : float2(1.0, 0.0);

  // Angular jitter. Per-pixel value-noise sample, mapped to [-π, +π]
  // and scaled by jitter strength. Noise is locally smooth (cell
  // size in pixels controls correlation length) so neighbouring
  // pixels rotate together — patches of the field swirl coherently
  // instead of becoming random salt-and-pepper. The evolving variant
  // makes the field mutate over time when evolution_rate > 0.
  if (P.angle_jitter > 0.0) {
    float n = mf_value_noise_phased(
        float2(gid), P.angle_noise_scale, MF_PHASE_ANGLE, P.noise_time) - 0.5;
    float angle_off = n * P.angle_jitter * 6.2832;
    dir = mf_rotate(dir, angle_off);
  }

  // ---- Magnitude ----
  // Local mag jitter via value noise: range [1 - mag_jitter, 1 + mag_jitter]
  // times `magnitude`. With mag_jitter=0 the magnitude is uniform.
  float mag_n = mf_value_noise_phased(
      float2(gid), P.mag_noise_scale, MF_PHASE_MAG, P.noise_time);
  float mag_scale = 1.0 + P.mag_jitter * (mag_n * 2.0 - 1.0);
  float mag = P.magnitude * max(0.0, mag_scale);

  return dir * mag * activation;
}

#endif
