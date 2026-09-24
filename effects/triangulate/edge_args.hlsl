// triangulate — convert the edge append counter into indirect draw args for
// the mesh line pass: {6 verts, min(count, max_edges) instances, 0, 0}. One
// thread; runs right after edge extraction so the line draw renders EXACTLY
// the edges found this frame instead of a worst-case MAX_EDGES instances.
// (counter[0] can exceed max_edges — the edges kernel increments before the
// slot bound check — hence the clamp.)
RWStructuredBuffer<uint> counter : register(u0);
RWStructuredBuffer<uint> args    : register(u1);

// Reuses the ClearEdgeUniforms layout (u_max_edges lives at offset 0), so the
// host binds the SAME uniform buffer the edge_clear pass already writes.
cbuffer ClearEdgeUniforms : register(b2) {
  uint u_max_edges;
  uint u_seen_words;
  uint u_p1, u_p2;
};

[numthreads(1, 1, 1)]
void main() {
  args[0] = 6u;                              // vertex_count: one quad per edge
  args[1] = min(counter[0], u_max_edges);    // instance_count: REAL edge count
  args[2] = 0u;                              // first_vertex
  args[3] = 0u;                              // first_instance
}
