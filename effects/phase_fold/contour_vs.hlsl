// source.phase_fold — contour mode vertex shader (full-screen triangle).
//
// The Contour limit-cycle mode draws the zero level-set of the blended height
// field directly — no particles. This VS emits one oversized triangle covering
// the viewport; the fragment shader does all the work per pixel.

#include "common.hlsl"

struct VsOut { float4 pos : SV_Position; };

[shader("vertex")]
VsOut main(uint vid : SV_VertexID) {
  float2 verts[3] = { float2(-1.0, -1.0), float2(3.0, -1.0), float2(-1.0, 3.0) };
  VsOut o;
  o.pos = float4(verts[vid % 3u], 0.0, 1.0);
  return o;
}
