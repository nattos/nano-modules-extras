// filter.height_from_gradient — prolongation (coarse → fine initial guess).
//
// Bilinearly upsamples the solved coarse height onto the next finer level as
// its initial guess, before that level's Jacobi sweeps refine it. This is the
// coarse-to-fine cascade (FMG-lite): the coarse solve propagates the low-
// frequency shape globally in a handful of iterations, so the fine level only
// has to add detail — far cheaper than relaxing the whole thing at full res.

#include "common.hlsl"

Texture2D<float4>   coarseH : register(t0);   // solved coarse height (R)
RWTexture2D<float4> fineH   : register(u1);   // fine initial guess (R)

[numthreads(8, 8, 1)]
void main(uint3 gid : SV_DispatchThreadID) {
  uint fw, fh;
  fineH.GetDimensions(fw, fh);
  if (gid.x >= fw || gid.y >= fh) return;

  uint cw, ch;
  coarseH.GetDimensions(cw, ch);

  // Fine pixel center → coarse texel-index space. With half_up coarsening the
  // coarse grid is ~half the fine grid; map centers so texel (2i,2i+1) both
  // straddle coarse texel i.
  float2 cp = (float2(gid.xy) + 0.5) * 0.5 - 0.5;
  float v = hfg_bil_r(coarseH, cp, int2(int(cw), int(ch)));

  fineH[gid.xy] = float4(v, 0.0, 0.0, 1.0);
}
