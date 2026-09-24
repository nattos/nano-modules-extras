// source.sdf.dust_halo — the halo generator: stateless shaped dust.
//
// One dispatch fills the outgoing rail dust buffer. Threads below
// gen_count SYNTHESIZE a halo mote as a pure function of (tid, knobs,
// two accumulated clocks) — no per-particle state, no pool, no respawn
// logic; drift and tumble are absolute phases, so rate 0 freezes the
// cloud exactly. Threads above gen_count COPY the upstream provider's
// motes into the tail (the merge). Halo motes live at the HEAD so the
// density accumulate pass can bind this same buffer and count only
// them (upstream motes already carry their density in the upstream
// grid's .a).
//
// Shape: a soft latitude band on a tiltable axis + a radial profile
// above the provider's surface. Band center 0 = a cap (the "beret"),
// width up = spherical shell, center at the equator + thin band + fat
// radial thickness = planetary ring. Motes orbit the axis with
// Keplerian r^-1.5 shear (inner motes lap outer ones); the ecc
// envelope is evaluated at the CURRENT azimuth, so the elliptical
// shape stays put while motes stream through it, breathing in polar
// angle as they cross the narrow side.

#include "nano_hash.hlsl"

StructuredBuffer<float4>   up_dust : register(t0);  // upstream rail motes
RWStructuredBuffer<float4> parts   : register(u1);  // outgoing rail motes

cbuffer HaloGenUniforms : register(b2) {
  float4 axis_a;   // halo axis (unit), w = gen count
  float4 axis_b1;  // band basis 1,    w = total count (gen + upstream)
  float4 axis_b2;  // band basis 2,    w = provider radius R
  float4 band;     // theta_c, theta_w, soft, ecc
  float4 radial;   // gap, thick, gap_soft, mote size
  float4 motion;   // T_drift (rad at inner radius), T_tumble (rad), 0, 0
};

float hg_rand(uint i, uint ch) {
  return float(nano_uhash(i * 16u + ch + 77771u)) * (1.0 / 4294967296.0);
}

// Density-shaping warp: 0 = uniform (hard slab edges), 1 = triangular
// (dense middle, density ramping linearly to zero at both edges). A
// smoothstep warp would do the OPPOSITE — sample density goes as
// 1/warp', which vanishes mid-band and spikes at the edges.
float hg_shape(float u, float u2, float k) {
  return lerp(u, 0.5 * (u + u2), k);
}

[numthreads(64, 1, 1)]
void main(uint3 gid : SV_DispatchThreadID) {
  uint tid = gid.x;
  uint gen = (uint)axis_a.w;
  uint total = (uint)axis_b1.w;
  if (tid >= total) return;

  if (tid >= gen) {
    // Merge: relay an upstream mote untouched.
    uint src = tid - gen;
    parts[tid * 2u] = up_dust[src * 2u];
    parts[tid * 2u + 1u] = up_dust[src * 2u + 1u];
    return;
  }

  // --- Synthesize halo mote `tid` ---
  float R = axis_b2.w;
  float r = R + radial.x
          + hg_shape(hg_rand(tid, 0u), hg_rand(tid, 12u), radial.z) * radial.y;

  float w_kep = pow(max((R + radial.x) / max(r, 1e-4), 0.05), 1.5);
  float az = 6.2831853 * hg_rand(tid, 1u) + motion.x * w_kep;

  float w_eff = band.y * (1.0 - band.w * 0.85 * sin(az) * sin(az));
  float sp = hg_shape(hg_rand(tid, 2u), hg_rand(tid, 13u), band.z) * 2.0 - 1.0;
  float th = band.x + sp * w_eff;

  float3 dir = axis_a.xyz * cos(th)
             + (axis_b1.xyz * cos(az) + axis_b2.xyz * sin(az)) * sin(th);

  // Facet normal: hashed unit vector, Rodrigues-tumbled by absolute
  // phase around a stable per-mote axis (per-mote rate jitter so the
  // field twinkles instead of strobing).
  float3 n0 = float3(hg_rand(tid, 3u), hg_rand(tid, 4u),
                     hg_rand(tid, 5u)) * 2.0 - 1.0;
  float nl = length(n0);
  float3 N = nl > 1e-3 ? n0 / nl : float3(0.0, 1.0, 0.0);
  float3 tax = normalize(float3(hg_rand(tid, 6u), hg_rand(tid, 7u),
                                hg_rand(tid, 8u)) * 2.0 - 1.0 +
                         float3(1e-4, 0.0, 0.0));
  float wt = (0.5 + 1.5 * hg_rand(tid, 9u)) * motion.y;
  float cw = cos(wt), sw = sin(wt);
  N = N * cw + cross(tax, N) * sw + tax * dot(tax, N) * (1.0 - cw);

  float sz = radial.w * (0.6 + 0.8 * hg_rand(tid, 10u));
  parts[tid * 2u] = float4(dir * r, sz);
  parts[tid * 2u + 1u] = float4(normalize(N), hg_rand(tid, 11u));
}
