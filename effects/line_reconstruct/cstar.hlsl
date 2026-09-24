// filter.reconstruct.line — pass 1b: CAS contrast normalizer c*.
// c* = 9x9 MAX of the 3x3 contrast c, floored at c_floor. A detector response at
// a stroke's sidelobe is CAUSED by the stroke a few px away, so it must be
// normalized by the STROKE's contrast — dividing by the sidelobe pixel's own
// (near-floor) contrast would amplify sidelobes ~10x and invert the ranking.
// (Port of pipeline.pass_stats' cstar; study 5.)

#include "common.hlsl"

Texture2D<float4>   statsTex : register(t0);   // (Y, min3, max3, c)
RWTexture2D<float4> cstarTex : register(u1);   // (c*, -, -, -)
cbuffer Uniforms : register(b2) { LRUniforms u; };

[numthreads(8, 8, 1)]
void main(uint3 gid : SV_DispatchThreadID) {
  uint w, h;
  cstarTex.GetDimensions(w, h);
  if (gid.x >= w || gid.y >= h) return;
  int2 p  = int2(gid.xy);
  int2 hi = int2(w - 1, h - 1);

  float c9 = 0.0;
  [unroll] for (int dy = -4; dy <= 4; dy++)
    [unroll] for (int dx = -4; dx <= 4; dx++) {
      int2 q = clamp(p + int2(dx, dy), int2(0, 0), hi);
      c9 = max(c9, statsTex[q].a);
    }
  cstarTex[gid.xy] = float4(max(c9, u.c_floor), 0.0, 0.0, 0.0);
}
