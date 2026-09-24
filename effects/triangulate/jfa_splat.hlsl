// triangulate — splat each seed's index into the JFA id texture at its pixel.
// Runs one thread per seed (collisions on a shared pixel are harmless — last
// writer wins).
#include "common.hlsl"

StructuredBuffer<Seed> seeds   : register(t0);
[[vk::image_format("r32f")]] RWTexture2D<float> idTex : register(u1);

cbuffer SplatUniforms : register(b2) {
  uint  u_count;
  uint  u_w;
  uint  u_h;
  uint  u_pad;
};

[numthreads(64, 1, 1)]
void main(uint3 gid : SV_DispatchThreadID) {
  uint i = gid.x;
  if (i >= u_count) return;
  Seed s = seeds[i];
  if (s.flags < 0.5) return;            // inactive (decimated) → not a Voronoi cell
  int2 p = int2(clamp(s.pos, float2(0.0, 0.0), float2(0.99999, 0.99999)) * float2(u_w, u_h));
  idTex[p] = (float)i;
}
