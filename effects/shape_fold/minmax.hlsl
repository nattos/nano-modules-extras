// source.shape_fold — auto-levels pass 1: field min/max over the SN×SN grid.
//
// Atomic min/max of the field into a stats buffer. F ≥ 0 always, so the IEEE
// bit pattern (asint) orders the same as the float — the testbed's trick.
// Reset on the CPU each frame to (lo = +INF-ish, hi = 0).

#include "common.hlsl"

RWStructuredBuffer<int> stats : register(u1);   // [0]=lo(asint), [1]=hi(asint), [2..]=hist

[numthreads(8, 8, 1)]
void main(uint3 gid : SV_DispatchThreadID) {
  if (gid.x >= SF_SN || gid.y >= SF_SN) return;
  float F = sf_field_at(sf_levels_p(gid.xy));
  int bits = asint(F);
  int prev;
  InterlockedMin(stats[0], bits, prev);
  InterlockedMax(stats[1], bits, prev);
}
