// filter.height_from_gradient — min/max reduction step.
//
// Folds the (min, max) range one level coarser: each coarse texel takes the
// min of the mins and max of the maxes over its 2x2 children. Run to 1x1 to get
// the global height range, which the present pass uses to normalize.

#include "common.hlsl"

Texture2D<float4>   fineMM   : register(t0);   // (min, max) finer
RWTexture2D<float4> coarseMM : register(u1);   // (min, max) coarser

[numthreads(8, 8, 1)]
void main(uint3 gid : SV_DispatchThreadID) {
  uint cw, ch;
  coarseMM.GetDimensions(cw, ch);
  if (gid.x >= cw || gid.y >= ch) return;

  uint fw, fh;
  fineMM.GetDimensions(fw, fh);
  int2 hi   = int2(int(fw) - 1, int(fh) - 1);
  int2 base = int2(gid.xy) * 2;

  float mn =  1e30;
  float mx = -1e30;
  [unroll] for (int dy = 0; dy < 2; dy++)
  [unroll] for (int dx = 0; dx < 2; dx++) {
    int2 s = clamp(base + int2(dx, dy), int2(0, 0), hi);
    float2 m = fineMM[uint2(s)].xy;
    mn = min(mn, m.x);
    mx = max(mx, m.y);
  }
  coarseMM[gid.xy] = float4(mn, mx, 0.0, 1.0);
}
