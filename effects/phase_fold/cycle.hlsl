// source.phase_fold — limit-cycle build + break detection (compute).
//
// Reads the relaxed particle ring (from solve.hlsl) and emits one line segment
// per consecutive pair — EXCEPT where the cycle is broken. A break is detected
// two ways between particle i and i+1:
//   • the gap is too large (break_dist) — relaxation pulled them apart, or
//   • the field gradient ∇H FLIPS along the segment (sampled a few times): on a
//     clean cycle the contour normal rotates smoothly, so a sign flip means the
//     two particles landed on different branches / across a critical structure.
// Broken pairs are written dead (the VS culls them), so a cycle that fails to
// close — e.g. killed by wind past the SNIC — visibly opens up. This is the
// "ok to fail for things that aren't cycles" path. Segment.c.x carries the seed
// angle so the FS rides a moving highlight around the cycle.

#include "field.hlsl"

RWStructuredBuffer<Segment> segs : register(u2);
StructuredBuffer<float4> particles : register(t3);   // xy = relaxed positions

void cy_dead(uint i) {
  Segment s; s.a = float4(0, 0, 0, 0); s.b = float4(0, 0, 0, 1); s.c = float4(0, 0, 0, 0);
  segs[i] = s;
}
void cy_seg(uint i, float2 p0, float2 p1, float arc) {
  Segment s;
  s.a = float4(p0, p1);
  s.b = float4(PF_CODE_CYCLE, 0.95, cycle_width, 0);
  s.c = float4(arc, 0, 0, 0);
  segs[i] = s;
}

[numthreads(64, 1, 1)]
void main(uint3 tid : SV_DispatchThreadID) {
  uint i = tid.x;
  if (i >= (uint)PF_PARTICLES) return;

  if (pf_weight_sum() < 1e-4) { cy_dead(i); return; }

  float2 pa = particles[i].xy;
  float2 pb = particles[(i + 1u) % (uint)PF_PARTICLES].xy;
  if (nano_is_nan(pa.x) || nano_is_nan(pa.y) || nano_is_nan(pb.x) || nano_is_nan(pb.y)) {
    cy_dead(i); return;
  }

  // Break 1: the pair drifted too far apart (the cycle opened here).
  if (length(pb - pa) > break_dist) { cy_dead(i); return; }

  // Break 2: the polyline DOUBLES BACK here. If a particle fell out of order the
  // index-connected line reverses sharply at pa; on a clean cycle the turn from
  // the incoming edge to the outgoing edge is gentle. Cull the reversal so the
  // longest-run select drops the spur.
  float2 pprev = particles[(i + (uint)PF_PARTICLES - 1u) % (uint)PF_PARTICLES].xy;
  float2 indir = pa - pprev;
  float2 outdir = pb - pa;
  float il = length(indir), ol = length(outdir);
  if (il > 1e-5 && ol > 1e-5 && dot(indir / il, outdir / ol) < break_turn_cos) {
    cy_dead(i); return;
  }

  // Break 2: the gradient direction flips along the segment. On a clean cycle
  // the contour normal rotates smoothly between adjacent particles.
  float2 n0 = normalize(pf_field(pa).grad);
  for (uint k = 1u; k <= (uint)PF_BREAK_SAMPLES; k++) {
    float2 ps = lerp(pa, pb, float(k) / float(PF_BREAK_SAMPLES + 1u));
    float2 nk = normalize(pf_field(ps).grad);
    if (dot(nk, n0) < 0.3) { cy_dead(i); return; }   // ~72° flip → break
  }

  cy_seg(i, pa, pb, float(i) / float(PF_PARTICLES));
}
