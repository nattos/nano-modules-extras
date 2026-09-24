// source.sdf.helio_field — dust pass: granule chemistry in the flatlands.
//
// A PASSIVE detail layer for the quiet regions between the field lines:
// a two-chemical Gray-Scott medium (a = substrate, b = granules) that is
// advected by the SAME velocity field as the magnetic potential — the
// granules curl along the eddies and shear into streaks — while the
// reaction keeps re-forming them into discrete, SELF-SPACING spots (the
// soliton regime: spots repel, split, and tile whatever quiet space
// they're given). Nothing here feeds back into vel or A, so the storm
// criticality band is untouched.
//
// The feed rate is gated by local field strength |∇A|: granules starve
// and dissolve near the lines and thrive in the flatlands — like real
// solar granulation, which strong field suppresses. A slow drifting
// noise nucleation re-seeds quiet regions whose population was swept
// into a line and killed.
//
// Two clocks: ADVECTION runs on the sim dt (granules must move with the
// fluid they live in), the REACTION on its own accelerated step `gs`
// (classic Gray-Scott needs ~unit steps to pattern in seconds). Both are
// zero when dt is zero, so Sim Rate 0 freezes this layer exactly too.
//
// State: 2 × RGBA16F ping-pong, (a, b, 0, 0). Seeded blobs on reset.

#include "../plume/common.hlsl"
#include "nano_noise3.hlsl"

Texture2D<float4>   dynTex   : register(t0);  // THIS frame's velocity
Texture2D<float4>   auxTex   : register(t1);  // THIS frame's (A, A_smooth)
Texture2D<float4>   dustPrev : register(t2);
SamplerState        samp     : register(s3);
RWTexture2D<float4> dustNext : register(u4);

cbuffer DustUniforms : register(b5) {
  float dt;        // sim-scaled frame delta, seconds (advection clock)
  float reset;     // 1 = write seeded initial conditions
  float sim_res;
  float seed;      // variation

  float feed;      // Gray-Scott F in the quiet regions
  float kill;      // Gray-Scott k
  float diff;      // substrate diffusion gain (granule chemical runs at /2)
  float gs;        // reaction step this frame (min(rate·dt, 1) — stable)

  float eps;       // diffusion ring half-step, radians (sets grain size)
  float line_kill; // extra kill rate under strong field (gate strength)
  float nucleate;  // re-seed rate for cleared quiet regions
  float drift;     // nucleation noise domain phase (stir clock)

  float gate_eps;  // |∇A| gate stencil half-step, radians
  float _p0, _p1, _p2;
};

void du_frame(float3 dir, out float3 t1, out float3 t2) {
  float3 a = abs(dir.y) < 0.92 ? float3(0.0, 1.0, 0.0)
                               : float3(1.0, 0.0, 0.0);
  t1 = normalize(cross(a, dir));
  t2 = cross(dir, t1);
}

float2 dust_at(float3 d) {
  return dustPrev.SampleLevel(samp, nano_oct_encode(d), 0).xy;
}
// A_smooth (.y) like every other gradient estimate in the effect.
float A_at(float3 d) { return auxTex.SampleLevel(samp, nano_oct_encode(d), 0).y; }

// Sparse seed blobs from the variation-keyed noise (also the reset state).
float seed_b(float3 dir) {
  float3 so = float3(seed * 23.9, seed * 5.3, seed * 17.1);
  return saturate((nano_gnoise3(dir * 6.0 + so) - 0.45) * 3.0) * 0.35;
}

[numthreads(8, 8, 1)]
void main(uint3 gid : SV_DispatchThreadID) {
  int r = int(sim_res);
  if (gid.x >= (uint)r || gid.y >= (uint)r) return;
  float2 uv = (float2(gid.xy) + 0.5) / sim_res;
  float3 dir = nano_oct_decode(uv);

  if (reset > 0.5) {
    float b0 = seed_b(dir);
    dustNext[gid.xy] = float4(1.0 - b0, b0, 0.0, 0.0);
    return;
  }

  // --- Semi-Lagrangian advection (midpoint, same scheme as dynamics) ---
  float3 v0 = dynTex.Load(int3(gid.xy, 0)).xyz;
  float3 dm = normalize(dir - v0 * (0.5 * dt));
  float3 vm = dynTex.SampleLevel(samp, nano_oct_encode(dm), 0).xyz;
  float3 db = normalize(dir - vm * dt);

  // Diffusion ring around the BACKTRACED point — reaction and diffusion
  // are evaluated where the parcel came from, so they ride along with it.
  float3 t1, t2;
  du_frame(db, t1, t2);
  float2 c  = dust_at(db);
  float2 avg = (dust_at(normalize(db + eps * t1)) +
                dust_at(normalize(db - eps * t1)) +
                dust_at(normalize(db + eps * t2)) +
                dust_at(normalize(db - eps * t2))) * 0.25;

  // --- Field-strength gate. NOT a feed gate: scaling F down shifts the
  // whole medium out of its pattern regime and the population starves
  // (measured: 3% coverage). Instead the quiet sun keeps the EXACT
  // mitosis-corner parameters, and strong field adds an extra kill term
  // — granules colonize the flats and dissolve on approach to a line. ---
  float3 g1, g2;
  du_frame(dir, g1, g2);
  float Apu = A_at(normalize(dir + gate_eps * g1));
  float Amu = A_at(normalize(dir - gate_eps * g1));
  float Apv = A_at(normalize(dir + gate_eps * g2));
  float Amv = A_at(normalize(dir - gate_eps * g2));
  float gAl = length(g1 * (Apu - Amu) + g2 * (Apv - Amv)) / (2.0 * gate_eps);
  float gate = 1.0 - smoothstep(0.6, 1.6, gAl);

  // --- Gray-Scott step. `diff` is the classic Du·4 (ring-average form of
  // the 5-point laplacian); granule chemical diffuses at half rate —
  // that 2:1 ratio is what makes spots instead of soup. gs ≤ 1 keeps the
  // explicit step in the standard stable regime. ---
  // b runs well BELOW the classic Dv = Du/2: the per-frame bilinear
  // resample of the advection adds its own diffusion (~half of Dv at
  // this ring radius), which would melt the pattern — the chemical
  // coefficient is lowered so the TOTAL stays near the 2:1 ratio.
  float a = c.x, b = c.y;
  float ab2 = a * b * b;
  float k_eff = kill + line_kill * (1.0 - gate);
  a += gs * (diff * (avg.x - a) - ab2 + feed * (1.0 - a));
  b += gs * (0.3 * diff * (avg.y - b) + ab2 - (feed + k_eff) * b);

  // --- Nucleation: a slowly drifting sparse noise mask drops granule
  // seeds where the substrate is charged and the field is quiet. ---
  float3 np = dir * 5.0 + float3(seed * 31.7, seed * 7.9, seed * 13.3)
            + float3(0.23, -0.31, 0.17) * drift;
  float m = saturate((nano_gnoise3(np) - 0.45) * 5.0);
  b += gs * nucleate * m * gate * a;

  dustNext[gid.xy] = float4(saturate(a), saturate(b), 0.0, 0.0);
}
