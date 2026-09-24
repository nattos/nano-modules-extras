// filter.reconstruct.line — pass 1: stats. Rec.709 luma + 3x3 min/max/contrast.
// Writes (Y, min3, max3, c=max-min) as RGBA16F. (Port of pipeline.pass_stats; the
// 9x9-max CAS normalizer c* is a separate pass so this stays a tight 3x3.)

#include "common.hlsl"

Texture2D<float4>   inputTex : register(t0);
RWTexture2D<float4> statsTex : register(u1);   // (Y, min3, max3, c)

[numthreads(8, 8, 1)]
void main(uint3 gid : SV_DispatchThreadID) {
  uint w, h;
  statsTex.GetDimensions(w, h);
  if (gid.x >= w || gid.y >= h) return;
  int2 p  = int2(gid.xy);
  int2 hi = int2(w - 1, h - 1);

  float y0 = lr_luma709(inputTex[p].rgb);
  float mn = y0, mx = y0;
  [unroll] for (int dy = -1; dy <= 1; dy++)
    [unroll] for (int dx = -1; dx <= 1; dx++) {
      int2 q = clamp(p + int2(dx, dy), int2(0, 0), hi);
      float yy = lr_luma709(inputTex[q].rgb);
      mn = min(mn, yy); mx = max(mx, yy);
    }
  statsTex[gid.xy] = float4(y0, mn, mx, mx - mn);
}
