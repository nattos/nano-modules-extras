// source.phase_fold — streamline tracer (compute).
//
// One thread per streamline seed (an NS×NS grid over the phase window). Each
// thread integrates the blended vector field for SL_STEPS RK2 steps and writes
// its polyline as a run of line SEGMENTS. The flow animation is NOT a discrete
// arrowhead anymore — each segment carries its arc-length position along the
// line (Segment.c.x) and a per-streamline stagger (c.y), and the fragment
// shader rides a continuous glow down the line from flow_phase. No barb math,
// no quantized stepping. PF_SL_STEPS segments are reserved per streamline;
// segments past the (possibly early-terminated) polyline are written dead.

#include "field.hlsl"

RWStructuredBuffer<Segment> segs : register(u2);

void write_dead(uint i) {
  Segment s;
  s.a = float4(0, 0, 0, 0);
  s.b = float4(0, 0, 0, 1);   // dead = 1
  s.c = float4(0, 0, 0, 0);
  segs[i] = s;
}

void write_seg(uint i, float2 p0, float2 p1, float code, float arc, float stagger) {
  Segment s;
  s.a = float4(p0, p1);
  s.b = float4(code, stream_alpha, stream_width, 0);
  s.c = float4(arc, stagger, 0, 0);
  segs[i] = s;
}

[numthreads(64, 1, 1)]
void main(uint3 tid : SV_DispatchThreadID) {
  uint idx = tid.x;
  if (idx >= (uint)(PF_NS * PF_NS)) return;
  uint base = idx * (uint)PF_SL_STEPS;

  // Hole → nothing to draw.
  if (pf_weight_sum() < 1e-4) {
    for (uint d = 0u; d < (uint)PF_SL_STEPS; d++) write_dead(base + d);
    return;
  }

  uint gi = idx / (uint)PF_NS;
  uint gj = idx % (uint)PF_NS;
  float stagger = frac(float(gi) * 0.618 + float(gj) * 0.382);
  // Seed grid spans ±(extent * stream_spread): >1 starts streamlines OUTSIDE the
  // window so they flow inward and keep the frame dense (the flow converges).
  float ext = extent * stream_spread;
  float x = -ext + (float(gi) + 0.5) / float(PF_NS) * 2.0 * ext;
  float y = -ext + (float(gj) + 0.5) / float(PF_NS) * 2.0 * ext;

  float spx[PF_SL_STEPS + 1];
  float spy[PF_SL_STEPS + 1];
  spx[0] = x; spy[0] = y;
  uint pn = 1u;
  float lim = ext * 1.05;   // allow roaming over the (possibly wider) seed region
  for (uint s = 0u; s < (uint)PF_SL_STEPS; s++) {
    float2 q = pf_step(float2(x, y), PF_SL_DT);
    // NaN via nano_is_nan; ±Inf and out-of-bounds via the magnitude check
    // (naga rejects the isfinite/IsNan intrinsic — see nano_sanitize.hlsl).
    if (nano_is_nan(q.x) || nano_is_nan(q.y) || abs(q.x) > lim || abs(q.y) > lim) break;
    x = q.x; y = q.y;
    spx[pn] = x; spy[pn] = y; pn++;
  }

  // Streamline segments, coloured by local speed (faint); arc = fractional
  // position along the line so the FS can ride a continuous glow down it.
  for (uint k = 0u; k < (uint)PF_SL_STEPS; k++) {
    if (k + 1u < pn) {
      float dx = spx[k + 1u] - spx[k];
      float dy = spy[k + 1u] - spy[k];
      float c = min(0.55, (length(float2(dx, dy)) / PF_SL_DT) * 0.16);
      float arc = (float(k) + 0.5) / float(PF_SL_STEPS);
      write_seg(base + k, float2(spx[k], spy[k]), float2(spx[k + 1u], spy[k + 1u]),
                c, arc, stagger);
    } else {
      write_dead(base + k);
    }
  }
}
