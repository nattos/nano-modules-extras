// source.particles.sweep_chamber — swept-image stats → calm↔intense response.
// ONE dispatch of a single 256-thread group per frame.
//
// The calm↔intense axis is DERIVED FROM THE SWEPT IMAGE ITSELF, not a knob:
// each thread serially reduces a slice of field_a/field_b (65 536 texels /
// 256 threads = 256 loads each), the group tree-reduces in groupshared
// memory, then thread 0 performs the temporal update IN PLACE — the previous
// frame's smoothed values live in the same persistent stats buffer. No CPU
// readback anywhere.
//
// statsBuf layout (2 × float4, persistent):
//   [0] raw:      x = E  (mean L': captured energy — the primary driver;
//                          exactly 0 at both sweep endpoints)
//                 y = C  (coverage: fraction of cells with any ridge signal)
//                 z = S  (mean |∇L'| / GAIN)
//                 w = dt used
//   [1] response: x = E_smooth   (attack/decay-smoothed energy)
//                 y = E_fast     (τ = 50 ms tracker, feeds the release detector)
//                 z = intensity  ∈ [0,1)   — x/(x+0.15) of E_smooth·sens
//                 w = release_env (1/s)    — peak-held rate of energy LOSS;
//                                            drives the fling
//
// Consumers (p_update) read statsBuf[1] the same frame (scheduled after this
// pass); the raw reduction itself is of THIS frame's field.

#include "common.hlsl"
#include "nano_sanitize.hlsl"

Texture2D<float4>            fieldA   : register(t0);
Texture2D<float4>            fieldB   : register(t1);
RWStructuredBuffer<float4>   statsBuf : register(u2);

cbuffer Uniforms : register(b3) {
  float field_res;
  float dt;
  float intensity_attack;   // smoothing time constants (s)
  float intensity_decay;

  float intensity_sens;     // E gain before the x/(x+0.15) saturation
  float release_gain;       // release envelope gain
  float release_decay;      // release envelope hold time (s)
  float _pad0;
}

groupshared float3 acc[256];

[numthreads(256, 1, 1)]
void main(uint tid : SV_GroupThreadID) {
  uint res = (uint)field_res;
  uint total = res * res;
  uint per = total / 256u;

  float3 a = float3(0.0, 0.0, 0.0);
  for (uint k = 0u; k < per; k++) {
    uint idx = tid * per + k;
    int3 tc = int3(int(idx % res), int(idx / res), 0);
    float4 fa = fieldA.Load(tc);
    a.x += fa.r;
    a.y += (fa.a > 0.02) ? 1.0 : 0.0;
    a.z += length(fieldB.Load(tc).ba);
  }
  acc[tid] = a;
  GroupMemoryBarrierWithGroupSync();
  for (uint s2 = 128u; s2 > 0u; s2 >>= 1u) {
    if (tid < s2) acc[tid] += acc[tid + s2];
    GroupMemoryBarrierWithGroupSync();
  }

  if (tid == 0u) {
    float inv = 1.0 / max((float)total, 1.0);
    float E = acc[0].x * inv;
    float C = acc[0].y * inv;
    float S = acc[0].z * inv * (1.0 / 6.0);

    // Previous response — sanitize on load (persistent state).
    float4 prev = statsBuf[1];
    float Es  = nano_sanitize(prev.x, 0.0, 0.0, 4.0);
    float Ef  = nano_sanitize(prev.y, 0.0, 0.0, 4.0);
    float env = nano_sanitize(prev.w, 0.0, 0.0, 64.0);
    float dts = max(dt, 1e-4);

    // Attack/decay smoothing (framerate-independent).
    float r = (E > Es) ? (1.0 - exp(-dts / max(intensity_attack, 1e-3)))
                       : (1.0 - exp(-dts / max(intensity_decay,  1e-3)));
    Es += (E - Es) * r;

    // Fast tracker → release = rate of energy LOSS (1/s), peak-held envelope.
    float EfN = Ef + (E - Ef) * (1.0 - exp(-dts / 0.05));
    float rel = max(Ef - EfN, 0.0) / dts;
    env = max(env * exp(-dts / max(release_decay, 1e-3)), rel * release_gain);

    // Saturating intensity shape (no pow).
    float x = Es * intensity_sens;
    float inten = x / (x + 0.15);

    statsBuf[0] = float4(E, C, S, dts);
    statsBuf[1] = float4(Es, EfN, inten, env);
  }
}
