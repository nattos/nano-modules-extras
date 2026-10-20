// source.mesh.lantern — depth-aware blur of the half-res in-scatter.
//
// The fog march dithers its step positions per pixel; one 9-tap pass along `blur.xy`, run horizontally then vertically,
// averages that into the soft haze it stands for. Taps that marched to a
// different depth (the other side of a block's silhouette) are left out, so
// glow doesn't bleed onto dark blocks in front of it.

#include "common.hlsl"

Texture2D<float4>   src : register(t2);
RWTexture2D<float4> dst : register(u3);

cbuffer Blur : register(b4) {
  float4 blur;   // direction (x, y), tap spacing (texels), 0
};

[numthreads(8, 8, 1)]
void main(uint3 gid : SV_DispatchThreadID) {
  int2 hs = int2((int(vp.x) + 1) / 2, (int(vp.y) + 1) / 2);
  if (gid.x >= (uint)hs.x || gid.y >= (uint)hs.y) return;
  float4 c = src.Load(int3(int2(gid.xy), 0));
  float d0 = c.a;
  float3 sum = float3(0.0, 0.0, 0.0);
  float wsum = 0.0;
  [unroll] for (int i = -4; i <= 4; i++) {
    int2 q = clamp(int2(gid.xy) + int2(blur.xy * (float(i) * blur.z)), int2(0, 0), hs - 1);
    float4 s = src.Load(int3(q, 0));
    float dd = (s.a - d0) / (0.02 * min(s.a, d0) + 0.05);
    float w = exp(-0.5 * float(i * i) / 6.25) / (1.0 + dd * dd);
    sum += s.rgb * w;
    wsum += w;
  }
  dst[gid.xy] = float4(sum / wsum, d0);
}
