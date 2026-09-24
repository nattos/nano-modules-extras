// triangulate — reset the edge buffer to the empty sentinel, zero the append
// counter, and clear the per-pair dedup bitmask, once per frame before edge
// extraction. Dispatched over the bitmask word count (the largest of the three).
RWStructuredBuffer<uint> edges   : register(u0);
RWStructuredBuffer<uint> counter : register(u1);
RWStructuredBuffer<uint> seen    : register(u2);

cbuffer ClearEdgeUniforms : register(b3) {
  uint u_max_edges;
  uint u_seen_words;
  uint u_p1, u_p2;
};

[numthreads(64, 1, 1)]
void main(uint3 gid : SV_DispatchThreadID) {
  uint i = gid.x;
  if (i >= u_seen_words) return;
  if (i < u_max_edges) edges[i] = 0xFFFFFFFFu;
  seen[i] = 0u;
  if (i == 0u) counter[0] = 0u;
}
