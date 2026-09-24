// triangulate — per-cell scoring scatter. Each pixel adds its importance to its
// Voronoi cell's accumulator (mass + weighted-centroid sums) and competes to be
// the cell's argmax-importance candidate.
#include "common.hlsl"

[[vk::image_format("r32f")]] RWTexture2D<float> idTex : register(u0);
Texture2D<float4>        feat  : register(t1);   // a = importance W
RWStructuredBuffer<uint> accum : register(u2);

cbuffer ScoreUniforms : register(b3) {
  uint  u_w;
  uint  u_h;
  float u_pad0, u_pad1;
};

[numthreads(8, 8, 1)]
void main(uint3 gid : SV_DispatchThreadID) {
  if (gid.x >= u_w || gid.y >= u_h) return;
  float cid = idTex[gid.xy];
  if (cid < 0.0) return;
  uint i = (uint)cid;

  float W = max(0.0, feat.Load(int3(gid.xy, 0)).a);
  float2 uv = (float2(gid.xy) + 0.5) / float2(u_w, u_h);

  uint b = i * TRI_ACCUM_STRIDE;
  uint prev;
  InterlockedAdd(accum[b + 0], (uint)(W * TRI_FX), prev);
  InterlockedAdd(accum[b + 1], (uint)(W * uv.x * TRI_FX), prev);
  InterlockedAdd(accum[b + 2], (uint)(W * uv.y * TRI_FX), prev);
  uint packed = tri_pack_cand(W, gid.x, gid.y);
  InterlockedMax(accum[b + 3], packed, prev);
}
