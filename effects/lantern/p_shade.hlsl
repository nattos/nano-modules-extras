// source.mesh.lantern — each flake's look this frame (one thread per flake).
//
// Projects the flake (plus the tail's vibration, a render-only offset), tests
// it against the scene's depth, lights it as a tiny flat disc — polished mica
// (the poster's metallic, roughness 0.12), near-black grit or dull silt —
// attenuates it by the water and halo in front, and writes what the spark
// raster needs: an ember capsule (its head, the way back to its tail, its
// width — a sub-pixel flake keeps its size as coverage), its radiance, and
// its screen motion for the motion-vector output.

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
  // Debug view Anchors: each flake pinned to a block's center, which always
  // projects inside that block — a check that the raster puts flakes where
  // the scene puts the blocks.
  const bool anchors = shade.w > 5.5 && shade.w < 6.5;
  if (anchors) {
    p = boxes[(int)(id % (uint)max(ltn_box_count(), 1)) * LTN_BOX_ROWS].xyz;
    vel = float3(0.0, 0.0, 0.0);
    vibd = float3(0.0, 0.0, 0.0);
  }

  float2 px;
  float w;
  if (!ltn_project(cam_r, cam_u, cam_f, p, px, w)) { ltn_cull(id); return; }
  if (any(px < -32.0) || any(px > vp.xy + 32.0)) { ltn_cull(id); return; }
  float3 ro = float3(cam_r.w, cam_u.w, cam_f.w);
  float t = length(p - ro);
  float3 rd = (p - ro) / t;
  // Behind a block? (Point test at its center: the flakes are a few pixels.)
  int2 ip = clamp(int2(px), int2(0, 0), int2(vp.xy) - 1);
  // (Debug view X-Ray, 7: the flakes behind the blocks too.)
  if (!anchors && shade.w < 6.5 && t > sceneTex.Load(int3(ip, 0)).a + 0.02) { ltn_cull(id); return; }

  // --- Footprint: an ember. The disc's size in pixels (at least a hairline;
  // a smaller flake keeps its size as coverage), stretched from where it is
  // back along where it came from over `streak` frames, tapering to its tail:
  // back through the corner it turned last frame (p_update's kink) to where
  // it set off, and on back the way it set off for a streak over a frame.
  float3 n = normalize(s2.xyz);
  float kpx = vp.y * cam_p.x * cam_p.z / w;          // pixels per metre here
  float wt = s4.x * flake.x * kpx;                    // true half-width, px
  float wd = max(wt, 0.55);
  float peak = min(1.0, (wt * wt) / (wd * wd));
  float2 tail = px, kn = px;
  if (pframe.w > 0.0 && !anchors) {
    float3 kw = parts[sb + 5u].xyz, fw = parts[sb + 6u].xyz;
    float la = length(pos - kw), lb = length(kw - fw);
    // (The shake's own travel spread along it.)
    kw += vib - vibd * (la / max(la + lb, 1e-6));
    fw += vib - vibd;
    la = length(p - kw);
    lb = length(kw - fw);
    float want = pframe.w * (la + lb);
    float3 te;
    if (want <= la) {                 // short of the corner: straight
      te = p + (kw - p) * (want / max(la, 1e-6));
      kw = 0.5 * (p + te);
    } else if (want <= la + lb) {
      te = kw + (fw - kw) * ((want - la) / max(lb, 1e-6));
    } else {                          // past where it set off: on back that way
      float3 bd = lb > 1e-6 ? (fw - kw) / lb : (kw - p) / max(la, 1e-6);
      te = fw + bd * (want - la - lb);
    }
    float2 tpx, kp;
    float tw, kz;
    if (ltn_project(cam_r, cam_u, cam_f, te, tpx, tw) &&
        ltn_project(cam_r, cam_u, cam_f, kw, kp, kz)) { tail = tpx; kn = kp; }
  }
  // Its frame: along the chord from the tail to the head, the corner at
  // (ku, kv) in it.
  float lp = length(px - kn) + length(kn - tail);
  if (lp > 256.0) {
    tail = px + (tail - px) * (256.0 / lp);
    kn = px + (kn - px) * (256.0 / lp);
  }
  float2 dpx = px - tail;
  float len = length(dpx);
  float2 axis = len > 1e-3 ? dpx / len : float2(1.0, 0.0);
  float2 ko = kn - tail;
  float2 kink = float2(dot(ko, axis), dot(ko, float2(-axis.y, axis.x)));

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
  // Heat: a fast flake glows like a spark off a grinder, in the core's color.
  float hs = saturate(length(vel) / 16.0);
  L += float3(egg_r0.w, egg_r1.w, egg_r2.w) * (shell.w * 6.0 * hs * hs * flake.w);
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
  recs[b + 1u] = float4(axis, len, wd);
  recs[b + 2u] = float4(L * T / max(Tm, 1e-4), peak);
  recs[b + 3u] = float4(mv * pview.zw, kink);
}
