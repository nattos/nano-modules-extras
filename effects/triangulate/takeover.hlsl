// triangulate — stochastic confidence-gated takeover. Seeds are LOCKED in place
// and only TELEPORT to a candidate when it is confidently a better match, with a
// Poisson-gated acceptance (no continuous drift → no swim). Only mismatched
// vertices churn; well-placed seeds stay put indefinitely.
#include "common.hlsl"

StructuredBuffer<uint>  accum : register(t0);
Texture2D<float4>       feat  : register(t1);   // a = importance W
RWStructuredBuffer<Seed> seeds : register(u2);

cbuffer TakeoverUniforms : register(b3) {
  uint  u_count;
  uint  u_w;
  uint  u_h;
  uint  u_frame;
  float u_dt;
  float u_churn;        // 0..1 → Poisson rate
  float u_confidence;   // 0..1 → takeover margin (deadband)
  float u_aspect;       // proc_w / proc_h
  uint  u_mode;         // 0 ridge-protect, 1 cell-residual, 2 feature-weight, 3 blue-noise
  float u_decimation;   // 0..1 slope-merge strength (Ridge Protect)
  uint  u_bnd;          // # of reserved frame-anchor seeds [0, u_bnd)
  uint  u_pad2;
};

// Distribute seed i (< bnd) evenly around the frame perimeter; corners land on
// the segment boundaries when bnd is a multiple of 4.
float2 tri_perimeter(uint i, uint bnd) {
  float f = (float)i / (float)max(bnd, 1u) * 4.0;
  uint seg = min((uint)f, 3u);
  float t = f - (float)seg;
  if (seg == 0u) return float2(t, 0.0);
  if (seg == 1u) return float2(1.0, t);
  if (seg == 2u) return float2(1.0 - t, 1.0);
  return float2(0.0, 1.0 - t);
}

static const float DEC_GAMMA   = 10.0;  // decimation → survival-importance exponent
static const float DEC_TOPCULL = 0.95;  // extra global thinning at high decimation
static const float DEC_HYST    = 0.06;  // activation hysteresis (stickiness)

float feat_at(float2 pos) {
  int2 p = int2(clamp(pos, float2(0.0, 0.0), float2(0.99999, 0.99999)) * float2(u_w, u_h));
  return max(0.0, feat.Load(int3(p, 0)).a);
}

// Rejection-sample a position with probability ∝ the balanced importance W, so a
// respawned / reactivated seed lands on a feature (ridge or maximum), not a void.
float2 importance_sample(uint id) {
  [loop]
  for (uint k = 0u; k < 24u; ++k) {
    float2 p = tri_hash2(id * 17u + k, u_frame * 5u + 1u);
    if (tri_hash_f(id * 4099u + k * 131u + u_frame) < feat_at(p)) return p;
  }
  return tri_hash2(id, u_frame);   // fallback: uniform
}

[numthreads(64, 1, 1)]
void main(uint3 gid : SV_DispatchThreadID) {
  uint i = gid.x;
  if (i >= u_count) return;
  Seed s = seeds[i];

  // NaN guard.
  if (s.pos.x != s.pos.x || s.pos.y != s.pos.y) s.pos = tri_hash2(i, u_frame);

  // Frame anchors: reserved seeds pinned to the perimeter, always active, never
  // relaxed/decimated — the Delaunay then triangulates out to the border.
  if (i < u_bnd) {
    s.pos = tri_perimeter(i, u_bnd);
    s.flags = 1.0;
    seeds[i] = s;
    return;
  }

  uint b = i * TRI_ACCUM_STRIDE;
  uint m0 = accum[b + 0];
  uint m1 = accum[b + 1];
  uint m2 = accum[b + 2];
  float massFx = float(max(m0, 1u));
  float mass = float(m0) / TRI_FX;
  float2 ctr = float2(float(m1), float(m2)) / massFx;
  uint cand = accum[b + 3];
  float candW = tri_cand_w(cand);
  float2 candPos = (float2(tri_cand_px(cand), tri_cand_py(cand)) + 0.5) / float2(u_w, u_h);

  float rate_hz = pow(60.0, u_churn) - 1.0;
  float pflip = 1.0 - exp(-rate_hz * u_dt);

  // --- DECIMATION via sticky stochastic ACTIVATION (ALL modes) ---
  // Each seed survives with a STICKY, stochastic probability that falls off with
  // its importance: keep = W^(decimation·γ) · (1 - topcull·decimation²). A fixed
  // per-seed threshold + hysteresis makes the survivor set stable (sticky) and
  // random (stochastic); flips happen at the Poisson rate. Deactivated seeds drop
  // out of the Voronoi entirely → fewer vertices → coarser, coalesced triangles
  // (still a full Delaunay of the survivors → no holes). The scoring_mode only
  // decides WHERE the surviving seeds sit; decimation thins them the same way in
  // every mode.
  float my_w  = s.score;                              // importance at this seed
  bool active = s.flags > 0.5;
  float r_i   = tri_hash_f(i * 2246822519u) * 0.9;    // fixed survival threshold
  float keep  = pow(max(my_w, 1e-3), u_decimation * DEC_GAMMA);
  keep *= (1.0 - DEC_TOPCULL * u_decimation * u_decimation);
  if (tri_hash_f(i * 3266489917u + u_frame) < pflip) {
    if (active) { if (keep < r_i - DEC_HYST) active = false; }
    else        { if (keep > r_i + DEC_HYST) { active = true; s.pos = importance_sample(i); } }
  }
  if (!active) { s.flags = 0.0; seeds[i] = s; return; }   // inactive → frozen, out of the Voronoi
  s.flags = 1.0;

  // Active but homeless (no cell) → respawn onto a feature.
  if (mass < 1e-4) {
    if (tri_hash_f(i * 9781u + u_frame) < pflip) s.pos = importance_sample(i);
    seeds[i] = s;
    return;
  }

  // --- Mode 0: relax toward the W-weighted centroid (density ∝ W) ---
  if (u_mode == 0u) {
    float2 dd = (s.pos - ctr) * float2(u_aspect, 1.0);
    float conf = saturate(length(dd) / 0.10);
    if (tri_hash_f(i * 2654435761u + u_frame) < 1.0 - exp(-rate_hz * u_dt * conf))
      s.pos = saturate(ctr);
    seeds[i] = s;
    return;
  }

  float2 target;
  float confidence;   // 0..1 margin-normalized merge strength
  if (u_mode == 1u) {
    // Cell residual: how far the seed sits from its cell's weighted centre.
    target = ctr;
    float2 d = (s.pos - ctr) * float2(u_aspect, 1.0);
    float residual = length(d);
    float margin = 0.01 + u_confidence * 0.15;
    confidence = saturate((residual - margin) / max(margin, 1e-4));
  } else {
    // Feature weight: climb toward the cell's argmax-importance pixel.
    target = candPos;
    float w_inc = feat_at(s.pos);
    float improvement = candW - w_inc;
    if (u_mode == 3u) improvement *= saturate(mass * 4.0);  // blue-noise: damp in sparse cells
    float margin = 0.02 + u_confidence * 0.4;
    confidence = saturate((improvement - margin) / max(margin, 1e-4));
  }

  float lambda = rate_hz * u_dt * confidence;
  float p = 1.0 - exp(-lambda);
  if (tri_hash_f(i * 2654435761u + u_frame) < p) {
    s.pos = saturate(target);
  }

  seeds[i] = s;
}
