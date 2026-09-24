// motion.local_delay — luma + initial downsample pass.
//
// Reads the full-res input and writes HALF-res Rec.601 luma (a 2x2 box
// average). This is the "downsample first" step: the flow estimator never
// runs at full res, so the input texture noise is averaged out up front
// and every later pass is cheap. Luma lives in the R channel of an
// RGBA16F texture (filterable — R32F can't be sampled as Float on WebGPU).

#include "common.hlsl"

Texture2D<float4>   inputTex : register(t0);   // full-res input
RWTexture2D<float4> lumaOut  : register(u1);   // half-res luma (R channel)

[numthreads(8, 8, 1)]
void main(uint3 gid : SV_DispatchThreadID) {
  uint w, h;
  lumaOut.GetDimensions(w, h);                // half-res dims
  if (gid.x >= w || gid.y >= h) return;

  uint sw, sh;
  inputTex.GetDimensions(sw, sh);
  int2 hi = int2(int(sw) - 1, int(sh) - 1);
  int2 base = int2(gid.xy) * 2;

  float l = 0.0;
  [unroll] for (int dy = 0; dy < 2; dy++)
  [unroll] for (int dx = 0; dx < 2; dx++) {
    int2 s = clamp(base + int2(dx, dy), int2(0, 0), hi);
    l += ld_luma(inputTex[uint2(s)].rgb);
  }
  lumaOut[gid.xy] = float4(l * 0.25, 0.0, 0.0, 1.0);
}
