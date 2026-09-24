// filter.height_from_gradient — Jacobi relaxation sweep (one iteration).
//
// One Jacobi step of laplacian(h) = F at a single pyramid level:
//   h' = (hL + hR + hD + hU - F) / 4
// with Neumann (clamped-coordinate) boundaries. F is the PRE-SCALED
// divergence for this level (common.hlsl), so the stencil needs no per-level
// grid-spacing factor and the same PSO is reused for every level. The host
// runs this `iterations` times per level, ping-ponging the height texture.

#include "common.hlsl"

Texture2D<float4>   hIn   : register(t0);   // current height estimate (R)
Texture2D<float4>   divIn : register(t1);   // pre-scaled divergence F (R)
RWTexture2D<float4> hOut  : register(u2);   // next height estimate (R)

[numthreads(8, 8, 1)]
void main(uint3 gid : SV_DispatchThreadID) {
  uint w, h;
  hOut.GetDimensions(w, h);
  if (gid.x >= w || gid.y >= h) return;

  int2 hi = int2(int(w) - 1, int(h) - 1);
  int2 p  = int2(gid.xy);
  float hL = hIn[uint2(clamp(p + int2(-1, 0), int2(0, 0), hi))].x;
  float hR = hIn[uint2(clamp(p + int2( 1, 0), int2(0, 0), hi))].x;
  float hD = hIn[uint2(clamp(p + int2(0, -1), int2(0, 0), hi))].x;
  float hU = hIn[uint2(clamp(p + int2(0,  1), int2(0, 0), hi))].x;
  float F  = divIn[gid.xy].x;

  hOut[gid.xy] = float4((hL + hR + hD + hU - F) * 0.25, 0.0, 0.0, 1.0);
}
