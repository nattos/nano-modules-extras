// motion.local_delay — luma pyramid downsample.
//
// 2x2 box-average of an R32F luma level into the next-coarser level.
// Used twice: half→quarter and quarter→eighth.

#include "common.hlsl"

Texture2D<float4>   srcLuma : register(t0);   // finer level (R channel = luma)
RWTexture2D<float4> dstLuma : register(u1);   // coarser level (half the size)

[numthreads(8, 8, 1)]
void main(uint3 gid : SV_DispatchThreadID) {
  uint w, h;
  dstLuma.GetDimensions(w, h);
  if (gid.x >= w || gid.y >= h) return;

  uint sw, sh;
  srcLuma.GetDimensions(sw, sh);
  int2 hi = int2(int(sw) - 1, int(sh) - 1);
  int2 base = int2(gid.xy) * 2;

  float l = 0.0;
  [unroll] for (int dy = 0; dy < 2; dy++)
  [unroll] for (int dx = 0; dx < 2; dx++) {
    int2 s = clamp(base + int2(dx, dy), int2(0, 0), hi);
    l += srcLuma[uint2(s)].x;
  }
  dstLuma[gid.xy] = float4(l * 0.25, 0.0, 0.0, 1.0);
}
