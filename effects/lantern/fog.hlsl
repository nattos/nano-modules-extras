// source.mesh.lantern — the water and the halo (half res).
//
// Two media: WATER everywhere (thin, blue-scattering, red-absorbing) and the
// HALO, a dense wispy ball around the structure that only the core lights.
// Light: the core, shadowed per step by its smooth, prefiltered visibility
// (vis.hlsl — a few filtered samples, no random shadow lookups, so the
// beams hold still under motion), and the ambient sun (unshadowed).
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

Texture2D<float4>   esc      : register(t2);
Texture2D<float4>   sceneTex : register(t3);
RWTexture2D<float4> fogI     : register(u4);
RWTexture2D<float4> fogT     : register(u5);
Texture2D<float4>   visF     : register(t6);
Texture2D<float4>   visD     : register(t7);
SamplerState        samp     : register(s8);

// Lit fraction of the core at r, given an escaped fraction F and the mean
// depth D of what blocked the rest: water in front of the blockers sees it all.
float ltn_lit(float F, float D, float r) {
  return F + (1.0 - F) * (1.0 - smoothstep(D - 0.2, D + 0.2, r));
}

// The core's smooth visibility at offset dl (length r) from its center: the
// prefiltered escape map at the blur radius r calls for (vis.hlsl), between
// its levels; the unblurred map near the shell.
float ltn_fog_vis(float3 dl, float r) {
  float2 uv = nano_oct_encode(dl / max(r, 1e-5));
  float2 e = esc.SampleLevel(samp, uv, 0.0).rg;
  float4 F = visF.SampleLevel(samp, uv, 0.0);
  float4 D = visD.SampleLevel(samp, uv, 0.0);
  float v0 = ltn_lit(e.x, e.y / max(1.0 - e.x, 1e-3), r);
  float v1 = ltn_lit(F.x, D.x, r), v2 = ltn_lit(F.y, D.y, r);
  float v3 = ltn_lit(F.z, D.z, r), v4 = ltn_lit(F.w, D.w, r);
  // Levels sit at rho_max * (0, 1/8, 1/4, 1/2, 1).
  float x = ltn_fog_rho(r) / ltn_fog_rho_max();
  float pos = x <= 0.125 ? x * 8.0 : 1.0 + log2(x * 8.0);
  if (pos < 1.0) return lerp(v0, v1, pos);
  if (pos < 2.0) return lerp(v1, v2, pos - 1.0);
  if (pos < 3.0) return lerp(v2, v3, pos - 2.0);
  return lerp(v3, v4, saturate(pos - 3.0));
}

// Water lit by the ambient sun over [ta, tb]: in-scatter and transmittance
// of the segment. The source is uniform per unit density, so the air wall
// only changes the segment's integrated length (ltn_air_len).
void ltn_water_segment(float ta, float tb, float3 rd, float c, inout float3 I, inout float3 T) {
  float len = ltn_air_len(ta, tb, c);
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

  float cv = dot(rd, cam_f.xyz);
  ltn_water_segment(0.0, t0, rd, cv, I, T);

  if (t1 > t0) {
    // Equiangular parameterisation about the core.
    float D = -b;                                   // closest approach along the ray
    float h = max(length(oc + rd * D), 0.05);
    float ta = atan((t0 - D) / h), tb = atan((t1 - D) / h);
    int steps = (int)misc.w;
    float jitter = nano_ign(float2(gid.xy) + misc.x * float2(5.588238, 5.588238));
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
      float fa = ltn_air_f(t, cv);   // the air wall thins both media
      float3 sig_t = (water_t.rgb + dh * halo_alb) * fa;

      float3 dl = p - core_p.xyz;
      float r2 = dot(dl, dl);
      float cosv = dot(dl * rsqrt(max(r2, 1e-6)), -rd);
      float3 Lc = ltn_core_irr_vol(r2) * ltn_fog_vis(dl, sqrt(r2));
      float3 src = (water_s.rgb * (Lc * ltn_hg(water_s.w, cosv) + amb_w) +
                    dh * halo_alb * (Lc * ltn_hg(halo_p.x, cosv) + amb_h)) * fa;

      float3 seg = exp(-sig_t * dt);
      I += T * src * (1.0 - seg) / max(sig_t, 1e-6);
      T *= seg;
    }
  }

  // Beyond the sphere: water again, to the surface or to infinity (where the
  // near-black world is entirely absorbed).
  if (depth < LTN_MISS * 0.5) ltn_water_segment(t1, depth, rd, cv, I, T);
  else                        ltn_water_segment(t1, t1 + 1.0e4, rd, cv, I, T);

  fogI[gid.xy] = float4(I, depth);
  fogT[gid.xy] = float4(T, 1.0);
}
