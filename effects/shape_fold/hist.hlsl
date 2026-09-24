// source.shape_fold — auto-levels pass 2: histogram over the SN×SN grid.
//
// Buckets the field into NB bins across [lo, hi] (from pass 1) via atomic add.
// Runs after minmax in the same submit, so the lo/hi writes are visible.

#include "common.hlsl"

RWStructuredBuffer<int> stats : register(u1);   // [0]=lo, [1]=hi, [2..]=hist[NB]

[numthreads(8, 8, 1)]
void main(uint3 gid : SV_DispatchThreadID) {
  if (gid.x >= SF_SN || gid.y >= SF_SN) return;
  float F = sf_field_at(sf_levels_p(gid.xy));
  float lo = asfloat(stats[0]);
  float hi = asfloat(stats[1]);
  float t = clamp((F - lo) / max(hi - lo, 1e-5), 0.0, 0.99999);
  int bin = (int)(t * float(SF_NB));
  int prev;
  InterlockedAdd(stats[2 + bin], 1, prev);
}
