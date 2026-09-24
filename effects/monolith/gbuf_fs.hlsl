// source.mesh.monolith — G-buffer fragment (MRT, Replace semantics).
//
//   target 0 (gbufA): view-space normal xyz, coverage (1 = covered)
//   target 1 (gbufB): world_y, view_z, world_x, world_z (for caustics)
//
// Both targets are cleared to zero each round; front faces of one convex
// solid never overlap on screen, so plain Replace writes are exact.

struct VsOut {
  float4 pos  : SV_Position;
  float4 nrm  : TEXCOORD0;
  float4 misc : TEXCOORD1;
};

struct PsOut {
  float4 a : SV_Target0;
  float4 b : SV_Target1;
};

[shader("pixel")]
PsOut main(VsOut i) {
  PsOut o;
  o.a = float4(i.nrm.xyz, 1.0);
  o.b = i.misc;
  return o;
}
