// filter.blur.lens — box-average downsample by an integer factor `ds` (matches
// the prototype's F.avg_pool2d before the bokeh gather, pipeline.pass_bokeh :177).
// Runs the gather at a reduced "working" resolution so a FIXED tap count covers
// the circle-of-confusion densely (no stipple) — the standard GPU-DOF cost lever.

#include "common.hlsl"

Texture2D<float4>   inputTex  : register(t0);   // full-res linear HDR
RWTexture2D<float4> outputTex : register(u1);   // proc-res
cbuffer Uniforms : register(b2) { uint u_ds; uint _p0, _p1, _p2; };

[numthreads(8, 8, 1)]
void main(uint3 gid : SV_DispatchThreadID) {
  uint pw, ph;
  outputTex.GetDimensions(pw, ph);
  if (gid.x >= pw || gid.y >= ph) return;
  uint sw, sh;
  inputTex.GetDimensions(sw, sh);
  int2 hi = int2(sw - 1, sh - 1);

  float4 sum = 0.0.xxxx;
  int2 base = int2(gid.xy) * (int)u_ds;
  for (uint j = 0; j < u_ds; j++)
    for (uint i = 0; i < u_ds; i++)
      sum += inputTex[clamp(base + int2(i, j), int2(0, 0), hi)];
  outputTex[gid.xy] = sum / (float)(u_ds * u_ds);
}
