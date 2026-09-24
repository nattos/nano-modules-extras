// source.sdf.helio_field — dust density accumulate: particles → a
// fixed-point 128³ count volume (InterlockedAdd; float16 texture
// atomics don't exist, so the accumulation runs in a uint buffer and
// dust_fold normalizes it into the published grid's .a channel).
//
// Each accumulated mote deposits 256·gain total, trilinearly split over
// its 2×2×2 voxel neighborhood — the consumer's bilinear reads then see
// a smooth aggregate medium (fog scattering, sun extinction), which is
// exactly the licensed approximation: individual motes are SHARP only
// in the splat pass; their lighting influence is a soft cloud.
//
// DECIMATED: the caller may stride through the particle buffer and
// weight each sample by live/accumulated (gain) — the mean density is
// exactly unchanged, only per-voxel variance rises, and the field
// saturates at ~20 effective motes/voxel anyway. This is the same
// softness license again: a dense clump serializes trilinear atomics
// on a handful of voxels, so sampling the clump beats enumerating it.

#include "../plume/common.hlsl"

StructuredBuffer<float4>   parts : register(t0);
RWStructuredBuffer<uint>   accum : register(u1);

cbuffer DustAccumUniforms : register(b2) {
  float count;     // accumulated samples (≤ live particles)
  float stride;    // particle-index stride (1 = every mote)
  float gain;      // deposit weight: live / accumulated (≥ 1)
  float _p2;
};

[numthreads(64, 1, 1)]
void main(uint3 gid : SV_DispatchThreadID) {
  if (gid.x >= (uint)count) return;
  uint pi = gid.x * (uint)stride;
  float4 r0 = parts[pi * 2u];
  if (r0.w <= 0.0) return;   // parked

  const int RES = PLM_VOL_RES;
  float3 f = (r0.xyz * (0.5 / PLM_EXT0) + 0.5) * float(RES) - 0.5;
  int3 v0 = int3(floor(f));
  float3 t = f - float3(v0);
  [unroll] for (int k = 0; k < 8; k++) {
    int3 o = int3(k & 1, (k >> 1) & 1, (k >> 2) & 1);
    int3 v = v0 + o;
    if (any(v < 0) || any(v >= RES)) continue;
    float w = (o.x != 0 ? t.x : 1.0 - t.x)
            * (o.y != 0 ? t.y : 1.0 - t.y)
            * (o.z != 0 ? t.z : 1.0 - t.z);
    uint dep = (uint)(w * 256.0 * gain + 0.5);
    if (dep == 0u) continue;
    uint prev;
    InterlockedAdd(accum[(v.z * RES + v.y) * RES + v.x], dep, prev);
  }
}
