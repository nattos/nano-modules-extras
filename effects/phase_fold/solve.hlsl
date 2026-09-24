// source.phase_fold — limit-cycle solver (compute, STATEFUL, double-buffered).
//
// N persistent particles form a ring on the limit cycle. Each carries POSITION
// (xy) and VELOCITY (zw); the update is a little force/momentum integrator so
// the ring WOBBLES around the cycle instead of snapping onto it. Each frame a
// particle either RESPAWNS onto the nearest cell's resting-cycle point (timer /
// first frame), or is integrated for `solve_steps` sub-steps as a real 2D mass:
//
//     v = v*momentum + F*step_size;   p += v;
//
// ALL forces feed the SAME 2D velocity, so momentum applies FREELY IN XY (not
// just radially). F = radial relaxation toward the wind-corrected cycle (a
// Newton-scaled spring) + tangential (random WALK back-and-forth + a SPRING
// toward the midpoint of the two neighbours' OLD positions, for even spacing).
// Underdamped (momentum < 1) → the particle overshoots and wobbles like a mass
// on springs. step_size scales how far each step pushes; momentum sets the
// wobble. Velocity is clamped (PF_VEL_CAP) for stability.
//
// Reads the previous frame's ring (particles_prev) and writes the next
// (particles_next), ping-ponged on the CPU, so neighbour reads are race-free.
// Sanitised on load (a stuck NaN would persist forever).

#include "field.hlsl"
#include "nano_hash.hlsl"

RWStructuredBuffer<float4> particles_next : register(u2);   // write (xy=pos, zw=vel)
StructuredBuffer<float>    curve          : register(t3);   // PF_CURVE resting cycles
StructuredBuffer<float4>   particles_prev : register(t4);   // read (last frame)
StructuredBuffer<float4>   good_ring      : register(t5);   // last "good" cycle (respawn source)
StructuredBuffer<float>    status         : register(t6);   // cycle health (PF_ST_*)

[numthreads(64, 1, 1)]
void main(uint3 tid : SV_DispatchThreadID) {
  uint i = tid.x;
  if (i >= (uint)PF_PARTICLES) return;
  uint N = (uint)PF_PARTICLES;

  float4 prev = particles_prev[i];
  float2 p = prev.xy;
  float2 v = prev.zw;
  // Respawn on the CPU timer / first frame, OR when the GPU flagged a broken
  // short cycle last frame (PF_ST_RESPAWN), OR a stuck NaN.
  bool reseed = (respawn > 0.5) || (status[PF_ST_RESPAWN] > 0.5) ||
                nano_is_nan(p.x) || nano_is_nan(p.y) ||
                nano_is_nan(v.x) || nano_is_nan(v.y);

  if (reseed) {
    // Clone the last good cycle; on the first frame (good not yet built) fall
    // back to the cell's resting cycle.
    if (good_init > 0.5) {
      p = good_ring[i].xy;
    } else {
      uint co = ((uint)nearest_cell * (uint)PF_NOUT + i) * 2u;
      p = float2(curve[co], curve[co + 1u]);
    }
    v = float2(0.0, 0.0);
  } else if (pf_weight_sum() >= 1e-4) {
    // Per-frame random value + the old-neighbour midpoint (for the spacing
    // spring). Both fixed across the sub-steps.
    float rnd = nano_hash21(float2(float(i), rand_seed)) * 2.0 - 1.0;
    float2 mid = 0.5 * (particles_prev[(i + N - 1u) % N].xy +
                        particles_prev[(i + 1u) % N].xy);

    // ALL forces feed the single 2D velocity, so momentum applies freely in XY
    // (not just radially): a radial relaxation spring toward the cycle, plus a
    // tangential walk + spacing — the particle wobbles like a real mass.
    uint steps = (uint)max(solve_steps, 1.0);
    for (uint s = 0u; s < steps; s++) {
      PfField f = pf_field(p);
      float gl2 = dot(f.grad, f.grad);
      float2 F = float2(0.0, 0.0);
      if (gl2 >= 1e-6) {
        float gl = sqrt(gl2);
        float2 n = f.grad / gl;            // contour normal
        float2 t = float2(-n.y, n.x);      // contour tangent
        float target = f.lev + bias + clamp(dot(f.W, n) / (f.mu * gl + 1e-3), -0.5, 0.5);
        // Radial: relaxation toward the cycle (Newton-scaled), clamped.
        float2 Fr = -(f.H - target) * f.grad / gl2;
        float fl = length(Fr);
        if (fl > PF_RELAX_CAP) Fr *= PF_RELAX_CAP / fl;
        // Tangential: random walk + spring toward even spacing.
        float along = dot(mid - p, t);
        float2 Ft = t * (rnd * explore * 0.04 + along * spread * 0.4);
        F = Fr + Ft;
      }
      v = v * momentum + F * step_size;
      float vl = length(v);
      if (vl > PF_VEL_CAP) v *= PF_VEL_CAP / vl;
      p += v;
    }
  }

  particles_next[i] = float4(p, v);
}
