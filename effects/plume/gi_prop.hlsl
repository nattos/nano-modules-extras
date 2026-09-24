// source.sdf.plume — resonant wave GI propagation (one leapfrog step).
//
// The radiance field obeys a damped wave equation (telegrapher):
//
//   ∂²L/∂t² + γ·∂L/∂t = c²·∇²L + S
//
// discretized on the GI grid (units: voxels and steps):
//
//   L⁺ = (2·L − (1−g)·L⁻ + c²·lap(L) + S) / (1+g),   g = γ·Δt/2
//
// Heavy damping (g→1) collapses this toward diffusion — calm, classic
// soft GI. Light damping lets light SLOSH: it overshoots, rings, and
// decays like a reverb tail — the `resonance` knob morphs between the
// two with this ONE shader. Absorption: a global decay multiplier (the
// `decay` knob) plus extra loss inside the body (density) and a soft
// fade at the volume boundary so waves leave instead of reflecting.
//
// Persistent state across frames: sanitize on load (nano_sanitize —
// a single f16 NaN would otherwise stick forever) and clamp the result
// (f16 blowup guard). CFL: stable for c ≤ 1/√3 voxels/step.

#include "common.hlsl"
#include "nano_sanitize.hlsl"

Texture3D<float4>   radCur     : register(t0);
Texture3D<float4>   radPrev    : register(t1);
Texture3D<float4>   injectVol  : register(t2);
Texture3D<float4>   sdfVol     : register(t3);
SamplerState        linearSamp : register(s4);
RWTexture3D<float4> radNext    : register(u5);

cbuffer PropUniforms : register(b6) {
  float c2;          // c², (voxels/step)²
  float damp;        // g
  float decay_mul;   // global per-step survival
  float inject_gain; // source scale
};

float3 plm_load_rad(Texture3D<float4> t, int3 v) {
  v = clamp(v, int3(0, 0, 0), int3(PLM_GI_RES - 1, PLM_GI_RES - 1, PLM_GI_RES - 1));
  float3 r = t.Load(int4(v, 0)).rgb;
  return float3(nano_sanitize(r.x, 0.0, 0.0, 64.0),
                nano_sanitize(r.y, 0.0, 0.0, 64.0),
                nano_sanitize(r.z, 0.0, 0.0, 64.0));
}

[numthreads(4, 4, 4)]
void main(uint3 gid : SV_DispatchThreadID) {
  if (gid.x >= (uint)PLM_GI_RES || gid.y >= (uint)PLM_GI_RES ||
      gid.z >= (uint)PLM_GI_RES) return;
  int3 v = int3(gid);

  float3 L = plm_load_rad(radCur, v);
  float3 Lp = plm_load_rad(radPrev, v);
  float3 lap = plm_load_rad(radCur, v + int3( 1, 0, 0))
             + plm_load_rad(radCur, v + int3(-1, 0, 0))
             + plm_load_rad(radCur, v + int3(0,  1, 0))
             + plm_load_rad(radCur, v + int3(0, -1, 0))
             + plm_load_rad(radCur, v + int3(0, 0,  1))
             + plm_load_rad(radCur, v + int3(0, 0, -1))
             - 6.0 * L;

  float3 S = injectVol.Load(int4(v, 0)).rgb * inject_gain;

  float3 Ln = (2.0 * L - (1.0 - damp) * Lp + c2 * lap + S) / (1.0 + damp);

  // Absorption inside the body + the global decay knob.
  float3 p = ((float3(gid) + 0.5) * (1.0 / float(PLM_GI_RES)) - 0.5)
             * (2.0 * PLM_EXT0);
  float dens = sdfVol.SampleLevel(linearSamp, plm_world_to_uvw(p), 0).g;
  Ln *= decay_mul * (1.0 - 0.35 * dens);

  // Soft absorbing boundary: waves fade out over the outer voxels.
  int edge = min(min(min(v.x, PLM_GI_RES - 1 - v.x), min(v.y, PLM_GI_RES - 1 - v.y)),
                 min(v.z, PLM_GI_RES - 1 - v.z));
  if (edge < 3) Ln *= 0.55 + 0.15 * float(edge);

  Ln = clamp(Ln, 0.0, 64.0);
  radNext[gid] = float4(Ln, 0.0);
}
