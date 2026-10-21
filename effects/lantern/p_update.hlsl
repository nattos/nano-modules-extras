// source.mesh.lantern — the flakes' simulation (one thread per flake).
//
// Substepped semi-implicit Euler over the regimes in particles.hlsl, then a
// trilinear splat of the flake into the presence grid (p_resolve turns the
// counts into the dust the fog reads). Flakes at or past `first index to
// (re)initialize` are seeded on the ring first.

#include "particles.hlsl"
#include "nano_noise3.hlsl"

RWStructuredBuffer<float4> parts  : register(u2);
RWStructuredBuffer<uint>   gcount : register(u3);

// Into the strata frame (scene.hlsl's ltn_to_egg).
float3 ltn_p_egg(float3 p) {
  return float3(dot(egg_r0.xyz, p), dot(egg_r1.xyz, p), dot(egg_r2.xyz, p));
}

// How far p sits from the nearest EMBER GROOVE along the strata axis (m), and
// how strongly that groove shows there: the stone's groove bands (scene.hlsl
// ltn_material), whose peaks are the glowing cracks of straining stone.
float ltn_groove_offset(float3 p, out float mask) {
  float3 q = ltn_p_egg(p);
  float3 fold = float3(nano_gnoise3(q * 0.35),
                       nano_gnoise3(q * 0.35 + float3(17.3, 5.1, 9.7)),
                       nano_gnoise3(q * 0.35 + float3(41.9, 23.3, 3.1)));
  float3 f = q + 0.45 * fold;
  // ltn_sine_bands(f, 2, 4, (1, 9, 4)): a groove where sin(n) peaks.
  float n = f.z * 40.0 + 4.0 * nano_gfbm3(f * 2.0 + float3(1.0, 9.0, 4.0), 3, 0.5);
  float ph = n - 1.5707963;
  ph -= 6.2831853 * round(ph / 6.2831853);
  mask = saturate((0.5 + 0.5 * nano_gnoise3(f * 1.2) - 0.4) / 0.2);
  return ph / 40.0;
}

void ltn_seed(uint id, out float4 s0, out float4 s1, out float4 s2, out float4 s3,
              out float4 s4) {
  float jr = ltn_pgauss(id, 10u), jz = ltn_pgauss(id, 20u);
  float th = ltn_prand(id, 1u) * 6.2831853;
  float rho = ring.x * (1.0 + ring.y * 0.3 * jr);
  float z = ring.y * ring.x * 0.1 * jz;
  float2 er = float2(cos(th), sin(th));
  float2 et = float2(-er.y, er.x);
  s0 = float4(er * rho, z, -1.0);
  s1 = float4(et * ring.z * (0.8 + 0.4 * ltn_prand(id, 2u)) +
              (ltn_prand_dir(id, 3u).xy * 0.15), ltn_prand_dir(id, 5u).z * 0.1, 0.0);
  s2 = float4(ltn_prand_dir(id, 30u), 0.5 + ltn_prand(id, 32u));
  s3 = float4(ltn_prand_dir(id, 40u), 0.5 + ltn_prand(id, 42u));
  // The poster's flakes: radius 4-13 mm, one in 25 a 24 mm one.
  float size = ltn_prand(id, 50u) < 0.04 ? 0.024 : lerp(0.004, 0.013, ltn_prand(id, 51u));
  s4 = float4(size, ltn_prand(id, 52u), jr, jz);
}

// The choked regime: ballistic orbits about the vertical through the
// structure, returned to a loose ring by a field that is soft inside a band
// around it and grows outside.
float3 ltn_ring_accel(float3 pos, float3 vel, float jr, float jz) {
  float rho = max(length(pos.xy), 0.3);
  float2 er = pos.xy / rho;
  float2 et = float2(-er.y, er.x);
  float rho0 = ring.x * (1.0 + ring.y * 0.3 * jr);
  float z0 = ring.y * ring.x * 0.1 * jz;
  float2 d2 = float2(rho - rho0, pos.z - z0);
  float band = 0.15 * ring.x * ring.y + 0.15;
  float dd = dot(d2, d2);
  float2 f = -ring.w * d2 * dd / (dd + band * band);
  float vt = dot(vel.xy, et);
  float3 a = float3(er * f.x, f.y);
  a.xy -= er * (0.9 * vt * vt / rho);            // most of the centripetal pull
  a.xy += et * (ring.z - vt) * 0.25;             // a light push to orbit speed
  a -= (vel - float3(et * vt, 0.0)) * 0.08;      // and a light drag off it
  return a;
}

// The sustained regime's jet: the flow of a SMOKE RING about the jet axis,
// centered on the core (Biot-Savart over a ring filament, softened at its
// core). Every streamline threads the hole — drawn in from a wide intake,
// blown through the core as a fast bore, fanned out the far side — and loops
// back round the outside to the intake, near ones tight and fast, far ones
// wide and slow. A swirl about the axis turns the loops into helices, as
// through a turbine. Flakes are dragged toward this flow.
static const int LTN_JET_SEGS = 24;

float3 ltn_jet_accel(float3 pos, float3 vel) {
  float3 J = jet.xyz;
  float3 b1 = normalize(cross(J, float3(0.0, 0.0, 1.0)));
  float3 b2 = cross(J, b1);
  float R = 0.45 * ring.x;                  // the vortex ring's radius
  float gam = 2.0 * R * jet2.x;             // circulation: jet2.x through the center
  const float d2c = 0.36;                   // its core, squared (m^2)
  float dth = 6.2831853 / float(LTN_JET_SEGS);
  float cd = cos(dth), sd = sin(dth);
  float c = 1.0, sn = 0.0;
  float3 u = float3(0.0, 0.0, 0.0);
  [loop] for (int k = 0; k < LTN_JET_SEGS; k++) {
    float3 rpt = (b1 * c + b2 * sn) * R;
    float3 dl = (b2 * c - b1 * sn) * (R * dth);
    float3 r = pos - rpt;
    float d2 = dot(r, r) + d2c;
    u += cross(dl, r) / (d2 * sqrt(d2));
    float cn = c * cd - sn * sd;
    sn = sn * cd + c * sd;
    c = cn;
  }
  u *= gam / (4.0 * 3.14159265);
  float s = dot(pos, J);
  float3 rp = pos - s * J;
  u += cross(J, rp) * (0.35 * jet2.x / R * exp(-dot(rp, rp) / 9.0));
  return (u - vel) * (4.0 * jet.w);
}

[numthreads(64, 1, 1)]
void main(uint3 gid : SV_DispatchThreadID) {
  uint id = gid.x;
  if (id >= (uint)pa.x) return;
  uint base = id * (uint)LTN_P_ROWS;
  float4 s0, s1, s2, s3, s4;
  if (id >= (uint)pa.y) {
    ltn_seed(id, s0, s1, s2, s3, s4);
  } else {
    s0 = parts[base + 0u]; s1 = parts[base + 1u]; s2 = parts[base + 2u];
    s3 = parts[base + 3u]; s4 = parts[base + 4u];
  }
  float3 pos = s0.xyz, vel = s1.xyz, nrm = s2.xyz;
  bool bound = s1.w > 0.5;
  float susc = s3.w;
  int nbox = ltn_box_count();

  const float S = pb.x, R = pb.y;
  const float Scap = S * (1.0 - R) * (1.0 - R);
  const float3 Z = egg_r2.xyz;     // the strata axis: the streams run round it
  // Each flake's lane: how far above the surface it streams (a spread of
  // lanes, like the smoke lines over a wing), its direction (a few run
  // against the rest) and its pace.
  const float lane = ltn_prand(id, 80u);
  const float sgn = ltn_prand(id, 81u) < 0.2 ? -1.0 : 1.0;
  const float pace = 0.8 + 0.4 * ltn_prand(id, 82u);

  // The throw: everything lets go and is blown out from the center.
  if (pb.z > 0.0) {
    float r = length(pos);
    float3 er = r > 1e-3 ? pos / r : ltn_prand_dir(id, 60u);
    vel += er * (shell.z * pb.z * (0.7 + 0.6 * ltn_prand(id, 61u)));
    bound = false;
  }

  int steps = (int)pa.w;
  float dt = pa.z;
  [loop] for (int k = 0; k < steps; k++) {
    // The blocks' surface as one (the nearest box), and the strongest magnet:
    // a squared Lorentzian of the gap to its lane (the magneto envelope's
    // falloff, sharpened so only a flake that comes within about Reach is
    // caught) scaled by the block's magnetism.
    float dmin = 1.0e9, d2nd = 1.0e9, mnear = 1.0, pull = 0.0;
    float3 nmin = float3(0.0, 0.0, 1.0), n2nd = nmin;
    const float h0 = orbit.x * lerp(0.35, 2.2, lane);
    [loop] for (int i = 0; i < nbox; i++) {
      float m = boxes[i * LTN_BOX_ROWS + 4].z;
      if (m <= 0.0) continue;    // not a magnet (a loose fragment): not streamed over
      float3 n;
      float d = ltn_box_sdf(i, pos, n);
      if (d < dmin) { d2nd = dmin; n2nd = nmin; dmin = d; nmin = n; mnear = m; }
      else if (d < d2nd) { d2nd = d; n2nd = n; }
      if (!bound && Scap > 1e-4) {
        float eps = cap.y * cap.y * m;
        float dl = max(d - h0, 0.0);
        float lz = eps / (dl * dl + eps);
        pull = max(pull, m * lz * lz);
      }
    }
    // A glint's band pulls the lanes in closer and drives them faster.
    float boost = max(mnear - 1.0, 0.0);
    float h = h0 / (1.0 + 0.6 * boost);
    // Where two blocks meet in a hollow, their normals blend over about a
    // lane's width: the crease becomes a fillet the streams can run along.
    {
      float kb = saturate(1.0 - (d2nd - dmin) / max(h, 0.05));
      nmin = normalize(nmin + n2nd * (kb * kb));
    }
    float g = dmin - h;

    // Held: snapped back onto its lane each step, normal velocity killed —
    // for as long as that asks less than the magnet holds. Along a face it
    // asks nothing; round an edge it asks the centripetal v^2/lane, so the
    // inner lanes peel off sharp corners at speed while the outer ones hug.
    if (bound) {
      float need = abs(g) / (dt * dt);
      if (Scap < 0.02 || need > cap.z * sqrt(Scap) * max(mnear, 0.3)) {
        bound = false;
      } else {
        pos -= nmin * g;
        vel -= nmin * dot(vel, nmin);
      }
    } else if (Scap > 1e-4 && saturate(Scap * susc * pull * orbit.w) > 0.15 &&
               abs(g) < 0.05 + abs(dot(vel, nmin)) * dt) {
      bound = true;
      pos -= nmin * g;
      vel -= nmin * dot(vel, nmin);
    }

    if (bound) {
      // Streaming: along the surface, round the strata axis, settling into
      // the ember grooves — steered, not pushed, so it turns hard.
      float3 W = cross(Z, pos) * sgn;
      float rho = length(W);              // distance from the strata axis
      W /= max(rho, 1e-4);
      float3 t = W - nmin * dot(W, nmin);
      float tl = length(t);
      if (tl < 0.3) {
        // Its way is blocked head-on: it climbs over, along the strata axis,
        // the way it was already heading (or, lane by lane, either way).
        float3 Zt = Z - nmin * dot(Z, nmin);
        float zs = dot(vel, Zt);
        t = Zt * (abs(zs) > 0.1 ? sign(zs) : (lane < 0.5 ? -1.0 : 1.0)) + t;
        tl = length(t);
      }
      t = tl > 1e-4 ? t / tl : vel * rsqrt(max(dot(vel, vel), 1e-8));
      // Slower toward the axis, as a turning body is, so the poles don't spin.
      float V = orbit.y * pace * (1.0 + 0.5 * boost) * saturate(rho / 1.5);
      float3 target = t * V;
      if (cap.w > 0.0) {
        float gm;
        float off = ltn_groove_offset(pos, gm);
        float3 Zt = Z - nmin * dot(Z, nmin);
        target -= Zt * (off * 30.0 * cap.w * lerp(0.4, 1.0, gm));
      }
      vel = lerp(vel, target, 1.0 - exp(-dt * 40.0));
    } else {
      float w = saturate(Scap * susc * pull * orbit.w);
      float freew = 1.0 - saturate(2.0 * w);
      float3 a = float3(0.0, 0.0, 0.0);
      a += ltn_ring_accel(pos, vel, s4.z, s4.w) * (freew * (1.0 - 0.7 * R) * (1.0 - saturate(1.6 * jet.w)));
      a += ltn_jet_accel(pos, vel) * freew * (1.0 - R);
      // A slow wander, so the ring isn't clockwork.
      float3 nq = pos * 0.22 + float3(0.0, 0.0, pb.w * 0.04);
      a += float3(nano_gnoise3(nq), nano_gnoise3(nq + float3(31.7, 0.0, 0.0)),
                  nano_gnoise3(nq + float3(0.0, 47.3, 0.0))) * (0.2 * freew * (1.0 - jet.w));
      // The tail: onto a sphere around the structure.
      if (R > 0.0) {
        float r = max(length(pos), 1e-3);
        float3 er = pos / r;
        float rs = shell.x * (1.0 + 0.08 * s4.z);
        float vr = dot(vel, er);
        a += er * (-shell.y * (r - rs) - 1.2 * sqrt(shell.y) * vr) * R;
        a -= (vel - er * vr) * (0.4 * R);
      }
      // Drawn toward its lane by a nearby magnet, thrown back out from under it.
      if (w > 1e-3) {
        float vn = dot(vel, nmin);
        a += nmin * (g > 0.0 ? -cap.x * w : cap.x * 40.0 * w * saturate(-g / 0.1));
        a -= nmin * (vn * 4.0 * w);
      }
      // Off a surface the streams' speed bleeds away to what the free flow
      // carries, quickly but not at once (a released stream flies on).
      float sp = length(vel);
      float vmax = max(ring.z * 2.0, jet2.x * jet.w * 1.5);
      if (sp > vmax) a -= vel * ((sp - vmax) / sp * 1.6);
      vel += a * dt;
    }
    pos += vel * dt;

    // Tumble — faster the faster it flies; a bound flake settles toward the
    // surface's orientation, so a stream's flakes glint together.
    float spin = s2.w * flake.z * (2.0 + 1.5 * length(vel)) * dt;
    float3 ax = s3.xyz;
    float cs = cos(spin), sn = sin(spin);
    nrm = nrm * cs + cross(ax, nrm) * sn + ax * dot(ax, nrm) * (1.0 - cs);
    if (bound) nrm = lerp(nrm, nmin * (dot(nrm, nmin) < 0.0 ? -1.0 : 1.0), 0.02);
    nrm = normalize(nrm);
  }

  // Never let a runaway poison the grid or the raster.
  if (any(pos != pos) || any(vel != vel) || dot(pos, pos) > 1.0e4) {
    float4 a0, a1, a2, a3, a4;
    ltn_seed(id, a0, a1, a2, a3, a4);
    pos = a0.xyz; vel = a1.xyz; nrm = a2.xyz;
    bound = false;
  }

  parts[base + 0u] = float4(pos, -1.0);
  parts[base + 1u] = float4(vel, bound ? 1.0 : 0.0);
  parts[base + 2u] = float4(nrm, s2.w);
  parts[base + 3u] = s3;
  parts[base + 4u] = s4;

  // Splat into the presence grid (8 cells, fixed-point weights).
  float res = grid.y;
  float3 gp = ltn_grid_uvw(pos) * res - 0.5;
  int3 g0 = int3(floor(gp));
  float3 f = gp - float3(g0);
  [unroll] for (int c = 0; c < 8; c++) {
    int3 o = int3(c & 1, (c >> 1) & 1, c >> 2);
    int3 q = g0 + o;
    if (any(q < 0) || any(q >= int3(res, res, res))) continue;
    float3 ww = select(o == 1, f, 1.0 - f);
    uint wv = (uint)(ww.x * ww.y * ww.z * grid.w + 0.5);
    if (wv == 0u) continue;
    uint idx = ((uint)q.z * (uint)res + (uint)q.y) * (uint)res + (uint)q.x;
    InterlockedAdd(gcount[idx], wv);
  }
}
