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

// What scales a magnet's pull on a flake in flight: the sustain, the block's
// magnetism, the flake's susceptibility.
float msc_free(float Scap, float m, float susc) {
  return sqrt(max(Scap, 0.0)) * max(m, 0.3) * susc;
}

// The circulation round block i: about its own axis r4.w (one of the two it
// does not jut along, picked per block on the CPU), its sense set per block.
// Unit direction; rho is the distance from that axis.
float3 ltn_block_swirl(int i, float3 p, out float rho) {
  float4 r0 = boxes[i * LTN_BOX_ROWS + 0];
  float4 r4 = boxes[i * LTN_BOX_ROWS + 4];
  float3 A = boxes[i * LTN_BOX_ROWS + 1 + clamp((int)r4.w, 0, 2)].xyz;
  float3 W = cross(A, p - r0.xyz) * (r4.y < 0.5 ? -1.0 : 1.0);
  rho = length(W);
  return W / max(rho, 1e-4);
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
  return (u - vel) * (1.5 * jet.w);
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
  int captor = (int)s0.w;
  float susc = s3.w;
  int nbox = ltn_box_count();
  if (captor >= nbox) { captor = -1; bound = false; }

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
    captor = -1;
  }

  int steps = (int)pa.w;
  float dt = pa.z;
  [loop] for (int k = 0; k < steps; k++) {
    // The strongest magnet: a squared Lorentzian of the gap to its lane (the
    // magneto envelope's falloff, sharpened so only a flake that comes within
    // about Reach is caught) scaled by the block's magnetism. And, for a
    // bound flake, its captor's own surface.
    const float h0 = orbit.x * lerp(0.35, 2.2, lane);
    float pull = 0.0, dcap = 1.0e9, dnear = 1.0e9;
    float3 ncap = float3(0.0, 0.0, 1.0);
    int ibest = -1, inear = -1;
    [loop] for (int i = 0; i < nbox; i++) {
      float m = boxes[i * LTN_BOX_ROWS + 4].z;
      if (m <= 0.0) continue;    // not a magnet (a loose fragment)
      float3 n;
      float d = ltn_box_sdf(i, pos, n);
      if (i == captor) { dcap = d; ncap = n; }
      if (d < dnear) { dnear = d; inear = i; }
      if (!bound && Scap > 1e-4) {
        float eps = cap.y * cap.y * m;
        float dl = max(d - h0, 0.0);
        float lz = eps / (dl * dl + eps);
        float pi = m * lz * lz;
        if (pi > pull) { pull = pi; ibest = i; }
      }
    }
    if (bound && dcap > 1.0e8) bound = false;
    if (!bound) {
      captor = ibest;
      if (ibest >= 0) dcap = ltn_box_sdf(ibest, pos, ncap);
    }
    // A glint's band pulls the lanes in closer and drives them faster.
    float mcap = captor >= 0 ? boxes[captor * LTN_BOX_ROWS + 4].z : 1.0;
    float boost = max(mcap - 1.0, 0.0);
    float h = h0 / (1.0 + 0.6 * boost);
    float g = dcap - h;

    // The magnet: how stiffly a bound flake is held at its lane's altitude —
    // a spring, slightly under-damped and hardening as it is pushed off its
    // lane, so it rides up over an edge under the load and springs back with
    // a little wobble. Soft and flexible deep in the sweep, nearly pinching in
    // the middle (stiffer still on an overshoot, a strained block, a glint's
    // band); scaled by Hold. Only a big excursion breaks it free.
    const float x = saturate(pb.x);
    const float omega = cap.z * lerp(5.0, 70.0, x * x) * (1.0 + 0.6 * max(pb.x - 1.0, 0.0)) *
                        sqrt(max(mcap, 0.3)) * lerp(0.75, 1.25, susc - 0.5);
    const float w = bound ? 1.0 : saturate(Scap * susc * pull * orbit.w);
    const float hs = max(h, 0.05);

    if (bound && (Scap < 0.02 || g > 6.0 * hs + 0.3)) bound = false;   // torn loose
    // Captured: a free flake drawn in close enough to its lane is held by it.
    if (!bound && captor >= 0 && Scap > 1e-4 && w > 0.15 && g < 1.5 * hs) bound = true;

    float3 a = float3(0.0, 0.0, 0.0);
    if (bound) {
      // Its altitude: the spring.
      float vn = dot(vel, ncap);
      float hard = min(1.0 + (g / hs) * (g / hs), 8.0);
      a -= ncap * (omega * omega * g * hard + 2.0 * 0.12 * omega * vn);
      // Streaming round its captor, about the block's own axis (one of the
      // two it does not jut along, set per block) — or, as much as Local is
      // short of 1, round the whole structure. A push toward the stream's
      // speed, not a steer: it keeps its momentum, and only slowly gives up
      // going any other way across the face.
      float rho;
      float3 W = ltn_block_swirl(captor, pos, rho);
      {
        float3 Wg = cross(Z, pos);
        float rg = length(Wg);
        W = lerp(Wg / max(rg, 1e-4), W, jet2.y);
        rho = lerp(rg, rho, jet2.y);
        W *= rsqrt(max(dot(W, W), 1e-8)) * sgn;
      }
      float3 t = W - ncap * dot(W, ncap);
      float tl = length(t);
      t = tl > 1e-4 ? t / tl : vel * rsqrt(max(dot(vel, vel), 1e-8));
      // Slower toward the axis, as a turning body is, so a cap's center
      // doesn't spin.
      float V = orbit.y * pace * (1.0 + 0.5 * boost) * saturate(rho / lerp(1.5, 0.3, jet2.y));
      float vt = dot(vel, t);
      float3 side = vel - t * vt - ncap * vn;
      a += t * ((V - vt) * 2.0);
      a -= side * 1.5;
      if (cap.w > 0.0) {
        // Drawn sideways into the ember grooves (never along its way, so a
        // stream across them isn't held back).
        float gm;
        float off = ltn_groove_offset(pos, gm);
        float3 Zt = Z - ncap * dot(Z, ncap);
        Zt -= t * dot(Zt, t);
        a -= Zt * (off * 400.0 * cap.w * lerp(0.4, 1.0, gm));
      }
    } else {
      float freew = 1.0 - saturate(2.0 * w);
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
      // In flight near a magnet: a pull toward its lane that is weak a lane
      // or two out and ramps up hard as it closes (the magneto envelope's
      // eps/(d^2+eps)), so a flake swings in and is caught.
      if (w > 1e-3 && captor >= 0 && g > 0.0) {
        float eps = (0.6 * h + 0.05) * (0.6 * h + 0.05);
        a -= ncap * (cap.x * omega * omega * hs * msc_free(Scap, mcap, susc) * w * eps / (g * g + eps));
      }
      // Off a surface a stream's speed bleeds slowly toward what the free
      // flow carries: released, it flies on almost ballistically.
      float sp = length(vel);
      float vmax = max(ring.z * 2.0, jet2.x * jet.w * 1.5);
      if (sp > vmax) a -= vel * ((sp - vmax) / sp * 0.5);
    }
    vel += a * dt;
    pos += vel * dt;

    // Ricochet: a flake held by (or swinging in to) a magnet that strikes a
    // block bounces off it — Ricochet is how much of its speed into the face it
    // keeps — scattered a little. (Free flakes, the jet's, pass through.)
    if (inear >= 0 && (bound || w > 0.15)) {
      float3 nb;
      float db = ltn_box_sdf(inear, pos, nb);
      const float rad = 0.01;
      if (db < rad) {
        pos += nb * (rad - db);
        float vn = dot(vel, nb);
        if (vn < 0.0) {
          float3 kick = ltn_prand_dir(id, 90u + (uint)k + (uint)pb.w * 7u);
          kick -= nb * dot(kick, nb);
          vel -= nb * (vn * (1.0 + jet2.z));
          vel += kick * (-vn * 0.35 * jet2.z);
        }
      }
    }

    // Tumble — faster the faster it flies; a bound flake settles toward the
    // surface's orientation, so a stream's flakes glint together.
    float spin = s2.w * flake.z * (2.0 + 1.5 * length(vel)) * dt;
    float3 ax = s3.xyz;
    float cs = cos(spin), sn = sin(spin);
    nrm = nrm * cs + cross(ax, nrm) * sn + ax * dot(ax, nrm) * (1.0 - cs);
    if (bound) nrm = lerp(nrm, ncap * (dot(nrm, ncap) < 0.0 ? -1.0 : 1.0), 0.02);
    nrm = normalize(nrm);
  }

  // Never let a runaway poison the grid or the raster.
  if (any(pos != pos) || any(vel != vel) || dot(pos, pos) > 1.0e4) {
    float4 a0, a1, a2, a3, a4;
    ltn_seed(id, a0, a1, a2, a3, a4);
    pos = a0.xyz; vel = a1.xyz; nrm = a2.xyz;
    bound = false;
    captor = -1;
  }

  parts[base + 0u] = float4(pos, (float)captor);
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
