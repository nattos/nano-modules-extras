// filter.reconstruct.line — pass 6-prep: per-channel RGB 3x3 min/max of the
// input. The line repaint clamps its energy-gained foreground to the union of
// the pixel's and the centerline's local colour range (so it can't invent colour
// past local evidence). That clamp is per-channel RGB (not luma), so we need the
// RGB 3x3 min/max as textures the reconstruct pass can bilinear-sample.

#include "common.hlsl"

Texture2D<float4>   inputTex : register(t0);
RWTexture2D<float4> cmnTex   : register(u1);   // per-channel 3x3 min (rgb)
RWTexture2D<float4> cmxTex   : register(u2);   // per-channel 3x3 max (rgb)

[numthreads(8, 8, 1)]
void main(uint3 gid : SV_DispatchThreadID) {
  uint w, h;
  cmnTex.GetDimensions(w, h);
  if (gid.x >= w || gid.y >= h) return;
  int2 p  = int2(gid.xy);
  int2 hi = int2(w - 1, h - 1);

  float3 mn = inputTex[p].rgb, mx = mn;
  [unroll] for (int dy = -1; dy <= 1; dy++)
    [unroll] for (int dx = -1; dx <= 1; dx++) {
      float3 v = inputTex[clamp(p + int2(dx, dy), int2(0, 0), hi)].rgb;
      mn = min(mn, v); mx = max(mx, v);
    }
  cmnTex[gid.xy] = float4(mn, 0.0);
  cmxTex[gid.xy] = float4(mx, 0.0);
}
