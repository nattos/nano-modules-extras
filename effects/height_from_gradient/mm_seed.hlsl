// filter.height_from_gradient — min/max reduction seed.
//
// The reconstructed height has an arbitrary scale (the Poisson solution is
// defined up to a constant and its magnitude depends on the gradient field and
// image size). To contour it — or show it as grayscale — meaningfully, we need
// its global range. This seeds a min/max reduction pyramid: each texel starts
// as (min, max) = (h, h). mm_reduce then folds it down to a 1x1 global range.

#include "common.hlsl"

Texture2D<float4>   heightTex : register(t0);   // reconstructed height (R)
RWTexture2D<float4> mmOut     : register(u1);   // (min, max) in RG

[numthreads(8, 8, 1)]
void main(uint3 gid : SV_DispatchThreadID) {
  uint w, h;
  mmOut.GetDimensions(w, h);
  if (gid.x >= w || gid.y >= h) return;
  float v = heightTex[gid.xy].x;
  mmOut[gid.xy] = float4(v, v, 0.0, 1.0);
}
