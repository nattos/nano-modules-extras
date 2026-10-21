// source.mesh.lantern — each flake's look this frame (one thread per flake).
//
// Projects the flake (plus the tail's vibration, a render-only offset), tests
// it against the scene's depth, lights it as a tiny flat disc — polished mica
// (the poster's metallic, roughness 0.12), near-black grit or dull silt —
// attenuates it by the water and halo in front, and writes what the spark
// raster needs: its center, the inverse covariance of its projected disc
// (an EWA splat — a sub-pixel flake keeps its size as coverage), its
// radiance, and its screen motion for the motion-vector output.

#include "particles.hlsl"

StructuredBuffer<float4>   parts    : register(t2);
Texture2D<float4>          atlas    : register(t5);
Texture2D<float4>          sceneTex : register(t6);
RWStructuredBuffer<float4> recs     : register(u7);

float ltn_fl_d(float ndh, float a) {
  float a2 = a * a;
  float d = ndh * ndh * (a2 - 1.0) + 1.0;
  return a2 / (LTN_PI * d * d);
}

float ltn_fl_vis(float ndl, float ndv, float a) {
  float a2 = a * a;
  float gv = ndl * sqrt(ndv * ndv * (1.0 - a2) + a2);
  float gl = ndv * sqrt(ndl * ndl * (1.0 - a2) + a2);
  return 0.5 / max(gv + gl, 1e-5);
}

// One light on a two-sided flake (n already faces the eye): irradiance E at
// normal incidence from direction l. Lit from behind, a little comes through.
float3 ltn_fl_light(float3 n, float3 v, float3 l, float3 E, float a, float3 f0,
                    float3 diff, float trans) {
  float ndl = dot(n, l);
  if (ndl <= 0.0) return E * (-ndl * trans / LTN_PI) * diff;
  float ndv = max(dot(n, v), 1e-3);
  float3 h = normalize(l + v);
  float ndh = saturate(dot(n, h));
  float vdh = saturate(dot(v, h));
  float3 F = f0 + (1.0 - f0) * pow(1.0 - vdh, 5.0);
  float3 spec = ltn_fl_d(ndh, a) * ltn_fl_vis(ndl, ndv, a) * F;
  return E * ndl * (diff / LTN_PI + spec);
}

// Transmittance from the eye to t along rd: water (with the air wall) and the
// halo's smooth body (no wisps), a few midpoint steps.
float3 ltn_fl_trans(float3 ro, float3 rd, float t) {
  float cv = dot(rd, cam_f.xyz);
  float3 tau = water_t.rgb * ltn_air_len(0.0, t, cv);
  float halo = 0.0;
  const int N = 6;
  float dt = t / float(N);
  [unroll] for (int i = 0; i < N; i++) {
    float ts = (float(i) + 0.5) * dt;
    halo += ltn_halo_density(ro + rd * ts, 0.5) * ltn_air_f(ts, cv) * dt;
  }
  tau += halo * float3(0.9, 0.9, 0.88);
  return exp(-tau);
}

void ltn_cull(uint id) {
  uint b = id * (uint)LTN_R_ROWS;
  recs[b + 0u] = float4(-1.0e4, -1.0e4, 0.0, 0.0);
  recs[b + 1u] = float4(0.0, 0.0, 0.0, 0.0);
  recs[b + 2u] = float4(0.0, 0.0, 0.0, 0.0);
  recs[b + 3u] = float4(0.0, 0.0, 0.0, 0.0);
}

[numthreads(64, 1, 1)]
void main(uint3 gid : SV_DispatchThreadID) {
  uint id = gid.x;
  if (id >= (uint)pa.x) return;
  uint sb = id * (uint)LTN_P_ROWS;
  float4 s0 = parts[sb + 0u], s1 = parts[sb + 1u], s2 = parts[sb + 2u];
  float4 s4 = parts[sb + 4u];
  float3 pos = s0.xyz, vel = s1.xyz;

  // The tail's shake: a shared frequency (falling as the tail rings out), so
  // the shell rings together; Spread detunes, dephases and turns each
  // flake's direction away from radial. Strongest on the shell.
  float r = max(length(pos), 1e-3);
  float3 er = pos / r;
  float3 vib = float3(0.0, 0.0, 0.0), vibd = float3(0.0, 0.0, 0.0);
  if (vibe.x > 0.0) {
    float sw = (r - shell.x) / 1.5;
    float A = vibe.x * exp(-sw * sw);
    float spread = vibe.z;
    float3 dir = normalize(lerp(er, ltn_prand_dir(id, 70u), spread));
    float det = 1.0 + spread * 0.35 * (ltn_prand(id, 72u) - 0.5);
    float ph = vibe.y * det + spread * 6.2831853 * ltn_prand(id, 73u);
    vib = dir * (A * sin(ph));
    // What it moves over this frame, as the motion blur should draw it: the
    // shake's own speed, never more than its full swing.
    float sp = A * cos(ph) * vibe.w * det * pframe.x;
    vibd = dir * clamp(sp, -2.0 * A, 2.0 * A);
  }
  float3 p = pos + vib;

  float2 px;
  float w;
  if (!ltn_project(cam_r, cam_u, cam_f, p, px, w)) { ltn_cull(id); return; }
  if (any(px < -32.0) || any(px > vp.xy + 32.0)) { ltn_cull(id); return; }
  float3 ro = float3(cam_r.w, cam_u.w, cam_f.w);
  float t = length(p - ro);
  float3 rd = (p - ro) / t;
  // Behind a block? (Point test at its center: the flakes are a few pixels.)
  int2 ip = clamp(int2(px), int2(0, 0), int2(vp.xy) - 1);
  if (t > sceneTex.Load(int3(ip, 0)).a + 0.02) { ltn_cull(id); return; }

  // --- Footprint: the disc's two radii projected to pixels.
  float3 n = normalize(s2.xyz);
  float3 tu = normalize(cross(n, abs(n.z) < 0.9 ? float3(0.0, 0.0, 1.0) : float3(1.0, 0.0, 0.0)));
  float3 tv = cross(n, tu);
  float rf = s4.x * flake.x;
  float kx = vp.x * cam_p.x * cam_p.y / w, ky = -vp.y * cam_p.x * cam_p.z / w;
  float2 a2 = float2(dot(tu, cam_r.xyz) * kx, dot(tu, cam_u.xyz) * ky) * rf;
  float2 b2 = float2(dot(tv, cam_r.xyz) * kx, dot(tv, cam_u.xyz) * ky) * rf;
  // A disc of radius rho has variance rho^2/4 along each of its axes; the
  // pixel's own footprint is added so a sub-pixel disc is a smooth blob
  // whose coverage integrates to its true projected area.
  float sxx = 0.25 * (a2.x * a2.x + b2.x * b2.x) + 0.1;
  float syy = 0.25 * (a2.y * a2.y + b2.y * b2.y) + 0.1;
  float sxy = 0.25 * (a2.x * a2.y + b2.x * b2.y);
  float det = sxx * syy - sxy * sxy;
  float area = LTN_PI * abs(a2.x * b2.y - a2.y * b2.x);
  float norm = area / (2.0 * LTN_PI * sqrt(det));
  float lmax = 0.5 * (sxx + syy) + sqrt(0.25 * (sxx - syy) * (sxx - syy) + sxy * sxy);
  float extent = clamp(3.0 * sqrt(lmax), 1.0, 24.0);

  // --- Shading.
  float3 v = -rd;
  float3 nf = dot(n, v) < 0.0 ? -n : n;
  float kind = s4.y;
  float3 f0, diff;
  float rough, trans;
  if (kind < flake.y) {            // mica: polished, metallic, a little translucent
    f0 = float3(0.85, 0.8, 0.7); diff = float3(0.02, 0.019, 0.016); rough = 0.12; trans = 0.6;
  } else if (kind > 1.0 - 0.48 * (1.0 - flake.y)) {   // grit: near black
    f0 = float3(0.006, 0.006, 0.006); diff = float3(0.004, 0.004, 0.005); rough = 0.55; trans = 0.0;
  } else {                         // silt: dull
    f0 = float3(0.04, 0.04, 0.04); diff = float3(0.05, 0.045, 0.035); rough = 0.7; trans = 0.2;
  }
  float r2 = rough * rough;
  float3 L = float3(0.0, 0.0, 0.0);
  {
    float3 dl = core_p.xyz - p;
    float d2 = dot(dl, dl);
    float d = sqrt(d2);
    int ns = (int)core_i.w;
    float vis = 0.0;
    [loop] for (int k = 0; k < ns; k++) vis += ltn_core_vis(atlas, k, p);
    vis /= float(max(ns, 1));
    L += ltn_fl_light(nf, v, dl / d, ltn_core_irr(d2) * vis, saturate(r2 + 0.5 * core_p.w / d),
                      f0, diff, trans);
  }
  L += ltn_fl_light(nf, v, amb_d.xyz, amb_e.rgb, saturate(r2 + 0.35), f0, diff, trans);
  L += ltn_fl_light(nf, v, rim_d.xyz, rim_e.rgb, saturate(r2 + 0.03), f0, diff, trans);
  {
    float3 dl = kick_p.xyz - p;
    float d2 = dot(dl, dl);
    float d = sqrt(d2);
    float3 l = dl / d;
    float3 E = kick_i.rgb * saturate(dot(kick_d.xyz, -l)) / d2;
    L += ltn_fl_light(nf, v, l, E, saturate(r2 + 0.5 * kick_p.w / d), f0, diff, trans);
  }
  L *= flake.w * surf.z;
  float3 T = ltn_fl_trans(ro, rd, t);

  // --- Motion this frame (pixels): where it was, through last frame's camera.
  float2 mv = float2(0.0, 0.0);
  if (pframe.z > 0.5) {
    float3 pp = pos - vel * pframe.x + vib - vibd;
    float2 ppx;
    float pw;
    if (ltn_project(prev_r, prev_u, prev_f, pp, ppx, pw)) mv = px - ppx;
  }

  uint b = id * (uint)LTN_R_ROWS;
  // Coverage folds in how much of the scene behind the water in front lets
  // through, so a flake deep in the haze darkens it less.
  float Tm = dot(T, float3(0.2126, 0.7152, 0.0722));
  recs[b + 0u] = float4(px, t, Tm);
  recs[b + 1u] = float4(syy / det, -sxy / det, sxx / det, norm);
  recs[b + 2u] = float4(L * T / max(Tm, 1e-4), extent);
  recs[b + 3u] = float4(mv * pview.zw, 0.0, 0.0);
}
