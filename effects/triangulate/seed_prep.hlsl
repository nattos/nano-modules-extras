// triangulate — per-seed prep for the feature-protect dynamics. Stamps each
// seed's own importance weight into seed.score (read by the adjacency pass and
// takeover) and clears its neighbour-max accumulator for this frame.
#include "common.hlsl"

RWStructuredBuffer<Seed> seeds : register(u0);
Texture2D<float4>        feat  : register(t1);   // a = importance W

cbuffer PrepUniforms : register(b2) {
  uint u_count;
  uint u_w;
  uint u_h;
  uint u_pad;
};

[numthreads(64, 1, 1)]
void main(uint3 gid : SV_DispatchThreadID) {
  uint i = gid.x;
  if (i >= u_count) return;
  Seed s = seeds[i];
  int2 p = int2(clamp(s.pos, float2(0.0, 0.0), float2(0.99999, 0.99999)) * float2(u_w, u_h));
  s.score = max(0.0, feat.Load(int3(p, 0)).a);
  seeds[i] = s;
}
