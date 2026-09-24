// source.sdf.helio_field — dust motes: a persistent advected pool on
// the rail's particle channel.
//
// One thread owns one particle for its whole life (no races, no
// compaction). The rail buffer (two float4 rows: pos+radius /
// normal+seed — effect_sdf_field.h) holds the RENDERABLE state; a
// private side buffer holds (age, life, salt, hover).
//
// Life cycle: born on a live granule (hash candidate direction,
// accepted where the chemistry's b chemical exceeds the threshold),
// advected by the SAME velocity field as everything else (the motes
// ride the eddies), hovering just above the local shell, tumbling
// around a per-particle axis (the glint twinkles), dying of old age —
// or early, when the granule under it starves (near lines) — and
// respawning at a fresh candidate. Rejected/dead motes park at the
// ORIGIN: inside the body every camera ray meets the surface first, so
// a parked mote can never win a splat pixel.
//
// Determinism: every motion term rides dt (the SIM clock) and respawn
// retries are gated on dt > 0, so Sim Rate 0 freezes the set exactly —
// except the reset frame, which places the initial population even at
// dt 0 (a frozen sun still wears its dust).

#include "../plume/common.hlsl"
#include "nano_hash.hlsl"

Texture2D<float4>          dustTex  : register(t0);  // chemistry (a, b)
Texture2D<float4>          shellTex : register(t1);  // (h, crest, ...)
Texture2D<float4>          dynTex   : register(t2);  // velocity (xyz)
SamplerState               samp     : register(s3);
RWStructuredBuffer<float4> parts    : register(u4);  // rail dust layout
RWStructuredBuffer<float4> pstate   : register(u5);  // age, life, salt, hover

cbuffer DustSimUniforms : register(b6) {
  float count;    // live particle slots
  float seed;     // variation key
  float R;        // body radius, world units
  float lift;     // hover height above the local surface, world units

  float size;     // mote radius, world units (jittered per mote)
  float thresh;   // chemistry b acceptance threshold
  float dt;       // sim-scaled frame delta, seconds
  float reset;    // 1 = seed the pool from scratch

  float tumble;   // tumble rate, rad per sim-second
  float life0;    // mean lifetime, sim-seconds
  float _p0, _p1;
};

float du_rand(uint tid, float salt, uint ch) {
  uint h = nano_uhash(tid * 16u + ch + (uint)salt * 197u
                      + (uint)(seed * 1013.0) * 2654435761u);
  return float(h) * (1.0 / 4294967296.0);
}

float du_b(float3 dir) {
  return dustTex.SampleLevel(samp, nano_oct_encode(dir), 0).y;
}

void du_park(uint tid, inout float4 st) {
  parts[tid * 2u] = float4(0.0, 0.0, 0.0, 0.0);
  parts[tid * 2u + 1u] = float4(0.0, 1.0, 0.0, 0.0);
  st.x = st.y;   // age = life: dead, retries while the sim runs
}

[numthreads(64, 1, 1)]
void main(uint3 gid : SV_DispatchThreadID) {
  uint tid = gid.x;
  if (tid >= (uint)count) return;

  float4 st = pstate[tid];
  bool fresh = reset > 0.5;
  if (fresh) st = float4(0.0, 1.0, float(tid & 1023u), 0.0);

  bool dead = st.x >= st.y;
  if (dead && !fresh && dt <= 0.0) return;   // frozen: stay parked

  if (dead || fresh) {
    // --- (Re)spawn at a hash candidate; accept only on live granules.
    st.z += 1.0;   // new candidate every attempt
    float z = 1.0 - 2.0 * du_rand(tid, st.z, 0u);
    float ph = 6.2831853 * du_rand(tid, st.z, 1u);
    float rxy = sqrt(saturate(1.0 - z * z));
    float3 dir = float3(rxy * cos(ph), z, rxy * sin(ph));
    if (du_b(dir) < thresh) {
      du_park(tid, st);
      pstate[tid] = st;
      return;
    }
    st.y = life0 * (0.5 + du_rand(tid, st.z, 2u));
    // Reset staggers ages so the initial population doesn't die (and
    // respawn) in one synchronized wave.
    st.x = fresh ? st.y * 0.8 * du_rand(tid, st.z, 3u) : 0.0;
    st.w = lift * (0.6 + 0.8 * du_rand(tid, st.z, 4u));

    float h = shellTex.SampleLevel(samp, nano_oct_encode(dir), 0).x;
    float3 n = float3(du_rand(tid, st.z, 5u), du_rand(tid, st.z, 6u),
                      du_rand(tid, st.z, 7u)) * 2.0 - 1.0;
    float nl = length(n);
    n = nl > 1e-3 ? n / nl : float3(0.0, 1.0, 0.0);
    float sz = size * (0.6 + 0.8 * du_rand(tid, st.z, 8u));
    parts[tid * 2u] = float4(dir * (R + h + st.w), sz);
    parts[tid * 2u + 1u] = float4(n, du_rand(tid, st.z, 9u));
    pstate[tid] = st;
    return;
  }

  // --- Alive: ride the fluid, hug the moving shell, tumble, age. ---
  float4 r0 = parts[tid * 2u];
  float4 r1 = parts[tid * 2u + 1u];
  float3 dir = normalize(r0.xyz);
  float3 v = dynTex.SampleLevel(samp, nano_oct_encode(dir), 0).xyz;
  dir = normalize(dir + v * dt);

  // Starved granule under the mote (swept against a line): die early.
  if (du_b(dir) < 0.3 * thresh) {
    du_park(tid, st);
    pstate[tid] = st;
    return;
  }

  float h = shellTex.SampleLevel(samp, nano_oct_encode(dir), 0).x;

  // Tumble: Rodrigues rotation of the facet normal around a stable
  // per-particle axis; rate jitters per mote so the field twinkles
  // instead of strobing.
  float3 axis = normalize(float3(du_rand(tid, st.z, 10u),
                                 du_rand(tid, st.z, 11u),
                                 du_rand(tid, st.z, 12u)) * 2.0 - 1.0);
  float w = tumble * (0.5 + 1.5 * du_rand(tid, st.z, 13u)) * dt;
  float cw = cos(w), sw = sin(w);
  float3 N = r1.xyz;
  N = N * cw + cross(axis, N) * sw + axis * dot(axis, N) * (1.0 - cw);

  st.x += dt;
  parts[tid * 2u] = float4(dir * (R + h + st.w), r0.w);
  parts[tid * 2u + 1u] = float4(normalize(N), r1.w);
  pstate[tid] = st;
}
