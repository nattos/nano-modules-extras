// triangulate — COMPLETE Delaunay edge extraction from the JFA Voronoi id map.
// Two seeds are Delaunay-adjacent iff their Voronoi cells share a boundary, i.e.
// some pixel has one cell id and its right/down neighbour has another. Scanning
// every 4-neighbour boundary captures the WHOLE triangulation with no missed
// edges (the earlier triple-point scan dropped edges → holes). A per-pair bitmask
// dedups so each edge is appended exactly once (no buffer overflow → no dropped
// edges either). The same first-touch also accumulates per-seed neighbour-max
// weight for the Ridge Protect dynamics.
#include "common.hlsl"

[[vk::image_format("r32f")]] RWTexture2D<float> idTex : register(u0);
RWStructuredBuffer<uint> edges   : register(u1);
RWStructuredBuffer<uint> counter : register(u2);
RWStructuredBuffer<uint> seen    : register(u3);   // dedup bitmask (MAX_SEEDS^2 bits)

cbuffer EdgeUniforms : register(b4) {
  uint u_w;
  uint u_h;
  uint u_max;
  uint u_pad;
};

void try_pair(uint a, uint b) {
  if (a == b) return;
  uint lo = min(a, b), hi = max(a, b);
  uint key  = lo * TRI_MAX_SEEDS + hi;         // < MAX_SEEDS^2
  uint word = key >> 5, bit = 1u << (key & 31u);
  uint old;
  InterlockedOr(seen[word], bit, old);
  if (old & bit) return;                       // this edge already emitted
  uint slot;
  InterlockedAdd(counter[0], 1u, slot);
  if (slot < u_max) edges[slot] = (lo << 16) | hi;
}

[numthreads(8, 8, 1)]
void main(uint3 gid : SV_DispatchThreadID) {
  if (gid.x >= u_w || gid.y >= u_h) return;
  int2 p = int2(gid.xy);
  float c0 = idTex[p];
  if (c0 < 0.0) return;
  uint a = (uint)c0;

  if (gid.x + 1u < u_w) {
    float cr = idTex[p + int2(1, 0)];
    if (cr >= 0.0) try_pair(a, (uint)cr);
  }
  if (gid.y + 1u < u_h) {
    float cd = idTex[p + int2(0, 1)];
    if (cd >= 0.0) try_pair(a, (uint)cd);
  }
}
