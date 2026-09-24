// motion.local_delay — upsample the half-res LK flow to full res.
//
// The flow is estimated on the downsampled pyramid (never at full res);
// this bilinearly upsamples the finest (half-res) flow to the full output
// resolution so the align/color/motion passes have a per-pixel field.
// Flow is in uv/frame, so upsampling is just a smooth bilinear fetch.

#include "common.hlsl"

Texture2D<float4>   flowIn  : register(t0);   // half-res flow (uv)
RWTexture2D<float4> flowOut : register(u1);   // full-res flow (uv)

[numthreads(8, 8, 1)]
void main(uint3 gid : SV_DispatchThreadID) {
  uint w, h;
  flowOut.GetDimensions(w, h);
  if (gid.x >= w || gid.y >= h) return;

  uint sw, sh;
  flowIn.GetDimensions(sw, sh);
  float2 uv = (float2(gid.xy) + 0.5) / float2(w, h);
  float2 p  = uv * float2(sw, sh) - 0.5;
  float2 flow = ld_bil_flow(flowIn, p, int2(int(sw), int(sh)));
  flowOut[gid.xy] = float4(flow, 0.0, 0.0);
}
