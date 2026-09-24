// filter.height_from_gradient — divergence pass (finest level).
//
// F_0 = div(g) = d(gx)/dx + d(gy)/dy by central differences in grid units
// (dx = 1), Neumann (clamped) boundaries. This is the right-hand side of the
// Poisson equation laplacian(h) = div(g). Level-0 spacing is 1, so F_0 is
// stored unscaled; restrict.hlsl builds the pre-scaled coarser levels.

#include "common.hlsl"

Texture2D<float4>   gradTex : register(t0);   // gradient field (RG)
RWTexture2D<float4> divOut  : register(u1);   // divergence (R)

[numthreads(8, 8, 1)]
void main(uint3 gid : SV_DispatchThreadID) {
  uint w, h;
  divOut.GetDimensions(w, h);
  if (gid.x >= w || gid.y >= h) return;

  int2 hi = int2(int(w) - 1, int(h) - 1);
  int2 p  = int2(gid.xy);
  int2 xl = clamp(p + int2(-1, 0), int2(0, 0), hi);
  int2 xr = clamp(p + int2( 1, 0), int2(0, 0), hi);
  int2 yd = clamp(p + int2(0, -1), int2(0, 0), hi);
  int2 yu = clamp(p + int2(0,  1), int2(0, 0), hi);

  float dgx = (gradTex[uint2(xr)].x - gradTex[uint2(xl)].x) * 0.5;
  float dgy = (gradTex[uint2(yu)].y - gradTex[uint2(yd)].y) * 0.5;

  divOut[gid.xy] = float4(dgx + dgy, 0.0, 0.0, 1.0);
}
