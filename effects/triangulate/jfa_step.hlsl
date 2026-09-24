// triangulate — one Jump-Flood pass. Each pixel adopts, among its 8 neighbours
// at the current step (and itself), the seed whose position is nearest.
#include "common.hlsl"

[[vk::image_format("r32f")]] RWTexture2D<float> srcId : register(u0);
StructuredBuffer<Seed> seeds : register(t1);
[[vk::image_format("r32f")]] RWTexture2D<float> dstId : register(u2);

cbuffer StepUniforms : register(b3) {
  int   u_step;
  uint  u_w;
  uint  u_h;
  float u_aspect;   // proc_w / proc_h — isotropic screen-space distance
};

[numthreads(8, 8, 1)]
void main(uint3 gid : SV_DispatchThreadID) {
  if (gid.x >= u_w || gid.y >= u_h) return;
  int2 dim = int2(u_w, u_h);
  float2 uv = (float2(gid.xy) + 0.5) / float2(dim);

  float best_id = -1.0;
  float best_d2 = 1e30;

  [unroll]
  for (int oy = -1; oy <= 1; ++oy) {
    for (int ox = -1; ox <= 1; ++ox) {
      int2 q = int2(gid.xy) + int2(ox, oy) * u_step;
      if (q.x < 0 || q.y < 0 || q.x >= dim.x || q.y >= dim.y) continue;
      float cid = srcId[q];
      if (cid < 0.0) continue;
      float2 sp = seeds[(uint)cid].pos;
      float2 d = (uv - sp) * float2(u_aspect, 1.0);
      float d2 = dot(d, d);
      if (d2 < best_d2) { best_d2 = d2; best_id = cid; }
    }
  }
  dstId[gid.xy] = best_id;
}
