// source.mesh.monolith — G-buffer vertex pull.
//
// The CPU emits TRUE homogeneous clip coordinates (pos = (clip.xy*z,
// 0.5*z, z)) so world_y / view_z interpolate perspective-correct across
// the big faces. Normals are per-face constants riding the vertex stream.
//
// NDC note (same contract as flash_particles/vs.hlsl): the CPU's single
// y-flip lives in its projection; DXC emits Vulkan NDC and naga inserts
// the WebGPU y-negation — do not touch y here.

struct Vtx {
  float4 pos;    // homogeneous clip: (clip.x*z, clip.y*z, 0.5*z, z)
  float4 nrm;    // xyz = view-space outward face normal, w = 1 (coverage)
  float4 misc;   // x = world_y, y = view_z
};

StructuredBuffer<Vtx> verts : register(t0);

struct VsOut {
  float4 pos  : SV_Position;
  float4 nrm  : TEXCOORD0;
  float4 misc : TEXCOORD1;
};

[shader("vertex")]
VsOut main(uint vid : SV_VertexID) {
  Vtx v = verts[vid];
  VsOut o;
  o.pos = v.pos;
  o.nrm = v.nrm;
  o.misc = v.misc;
  return o;
}
