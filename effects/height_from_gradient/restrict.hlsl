// filter.height_from_gradient — divergence pyramid restriction.
//
// Coarsens the pre-scaled divergence one level: F_{k+1}[p] = SUM of the 2x2
// children of F_k. The sum (not the average) is what bakes the squared-grid-
// spacing factor into the RHS — F_{k+1} = 4 * mean(F_k) = (2^{k+1})^2 * div —
// so the Jacobi stencil stays spacing-agnostic at every level (see
// common.hlsl). Out-of-range children clamp to the edge.

#include "common.hlsl"

Texture2D<float4>   fineDiv   : register(t0);   // F_k   (finer)
RWTexture2D<float4> coarseDiv : register(u1);   // F_{k+1} (coarser)

[numthreads(8, 8, 1)]
void main(uint3 gid : SV_DispatchThreadID) {
  uint cw, ch;
  coarseDiv.GetDimensions(cw, ch);
  if (gid.x >= cw || gid.y >= ch) return;

  uint fw, fh;
  fineDiv.GetDimensions(fw, fh);
  int2 hi   = int2(int(fw) - 1, int(fh) - 1);
  int2 base = int2(gid.xy) * 2;

  float sum = 0.0;
  [unroll] for (int dy = 0; dy < 2; dy++)
  [unroll] for (int dx = 0; dx < 2; dx++) {
    int2 s = clamp(base + int2(dx, dy), int2(0, 0), hi);
    sum += fineDiv[uint2(s)].x;
  }

  coarseDiv[gid.xy] = float4(sum, 0.0, 0.0, 1.0);
}
