// source.mesh.lantern — the water and the halo (half res).
//
// Two media: WATER everywhere (thin, blue-scattering, red-absorbing) and the
// HALO, a dense wispy ball around the structure that only the core lights.
// Light: the core (shadowed per step by TWO of the light samples' atlas
// tiles — they cycle across steps and pixels, and blur.hlsl averages them
// into soft beams) and the ambient sun (unshadowed).
//
// Inside a sphere around the core the ray is marched with steps spaced
// EQUIANGULARLY about the core (t = D + h tan(theta), theta uniform): dense
// where the ray passes nearest the light, where the beams are, sparse far
// away — the same 64 steps resolve both a ray grazing the structure and one
// passing 30 m off. Water outside the sphere integrates analytically.
//
// Output: fogI = (in-scattered radiance, the depth this pixel marched to),
//         fogT = (transmittance rgb, 1).

#include "common.hlsl"
#include "nano_hash.hlsl"
#include "nano_noise3.hlsl"

Texture2D<float4>   atlas    : register(t2);
Texture2D<float4>   sceneTex : register(t3);
RWTexture2D<float4> fogI     : register(u4);
RWTexture2D<float4> fogT     : register(u5);

// Homogeneous water lit by the ambient sun over [0, len]: in-scatter and
// transmittance of the segment.
void ltn_water_segment(float len, float3 rd, inout float3 I, inout float3 T) {
  if (len <= 0.0) return;
  float3 st = water_t.rgb;
  float3 seg = exp(-st * len);
  float3 src = water_s.rgb * ltn_hg(water_s.w, dot(-amb_d.xyz, -rd)) * amb_e.rgb * vol_k.y;
  I += T * src * (1.0 - seg) / max(st, 1e-6);
  T *= seg;
}

[numthreads(8, 8, 1)]
void main(uint3 gid : SV_DispatchThreadID) {
  float2 hs = floor((vp.xy + 1.0) * 0.5);
  if (gid.x >= (uint)hs.x || gid.y >= (uint)hs.y) return;

  // The half-res ray goes through the center of its 2x2 full-res block.
  float3 ro, rd;
  ltn_camera_ray(float2(gid.xy) * 2.0 + 0.5, vp.xy, ro, rd);
  int2 fpx = min(int2(gid.xy) * 2, int2(vp.xy) - 1);
  float depth = sceneTex.Load(int3(fpx, 0)).a;

  float3 I = float3(0.0, 0.0, 0.0);
  float3 T = float3(1.0, 1.0, 1.0);

  // March sphere around the core.
  float R = water_t.w;
  float3 oc = ro - core_p.xyz;
  float b = dot(oc, rd);
  float cc = dot(oc, oc) - R * R;
  float disc = b * b - cc;
  float t0 = depth, t1 = depth;
  if (disc > 0.0) {
    float sq = sqrt(disc);
    t0 = clamp(-b - sq, 0.0, depth);
    t1 = clamp(-b + sq, 0.0, depth);
  }

  ltn_water_segment(t0, rd, I, T);

  if (t1 > t0) {
    // Equiangular parameterisation about the core.
    float D = -b;                                   // closest approach along the ray
    float h = max(length(oc + rd * D), 0.05);
    float ta = atan((t0 - D) / h), tb = atan((t1 - D) / h);
    int steps = (int)misc.w;
    float jitter = nano_ign(float2(gid.xy) + misc.x * float2(5.588238, 5.588238));
    int ns = (int)core_i.w;
    int kbase = (int)(nano_hash21(float2(gid.xy) * 0.37 + misc.x) * float(ns));
    float cos_amb = dot(-amb_d.xyz, -rd);
    float3 amb_w = ltn_hg(water_s.w, cos_amb) * amb_e.rgb * vol_k.y;
    float3 amb_h = ltn_hg(halo_p.x, cos_amb) * amb_e.rgb * vol_k.y;
    const float3 halo_alb = float3(0.9, 0.9, 0.88);
    [loop] for (int i = 0; i < steps; i++) {
      float u = (float(i) + jitter) / float(steps);
      float th = lerp(ta, tb, u);
      float t = D + h * tan(th);
      float ct = cos(th);
      float dt = h * (tb - ta) / (float(steps) * ct * ct);
      float3 p = ro + rd * t;

      float wn = 0.5 + 0.5 * nano_gfbm3((p - halo_c.xyz) * halo_k.xyz * halo_p.y +
                                        float3(0.0, 0.0, halo_p.z), 3, 0.6);
      float dh = ltn_halo_density(p, wn);
      float3 sig_t = water_t.rgb + dh * halo_alb;

      float3 dl = p - core_p.xyz;
      float r2 = dot(dl, dl);
      float cosv = dot(dl * rsqrt(max(r2, 1e-6)), -rd);
      // Two light samples a step, half the samples apart (stratified).
      int k = (kbase + i) % ns;
      float2 jit = frac(float2(jitter * 7.31 + float(i) * 0.6180340,
                               jitter * 3.17 + float(i) * 0.7548777));
      float vis = 0.5 * (ltn_core_vis_soft(atlas, k, p, jit) +
                         ltn_core_vis_soft(atlas, (k + ns / 2) % ns, p, frac(jit + 0.5)));
      float3 Lc = ltn_core_irr_vol(r2) * vis;
      float3 src = water_s.rgb * (Lc * ltn_hg(water_s.w, cosv) + amb_w) +
                   dh * halo_alb * (Lc * ltn_hg(halo_p.x, cosv) + amb_h);

      float3 seg = exp(-sig_t * dt);
      I += T * src * (1.0 - seg) / max(sig_t, 1e-6);
      T *= seg;
    }
  }

  // Beyond the sphere: water again, to the surface or to infinity (where the
  // near-black world is entirely absorbed).
  if (depth < LTN_MISS * 0.5) ltn_water_segment(depth - t1, rd, I, T);
  else                        ltn_water_segment(1.0e4, rd, I, T);

  fogI[gid.xy] = float4(I, depth);
  fogT[gid.xy] = float4(T, 1.0);
}
