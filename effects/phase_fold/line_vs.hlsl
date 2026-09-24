// source.phase_fold — line raster vertex shader (instanced).
//
// One instance per traced Segment (from stream.hlsl / cycle.hlsl), six vertices
// per instance forming a quad. The quad spans the segment p0→p1 with a half-
// width offset along the segment normal; the fragment shader fades it to the
// rim for a soft anti-aliased line. Dead segments collapse to a degenerate
// triangle (no rasterised pixels). The segment buffer is bound at register(t1)
// and the shared uniforms at b0 (see field passes for the b0 layout) — the line
// raster doesn't touch the atlas cells, so it includes common.hlsl only.

#include "common.hlsl"

StructuredBuffer<Segment> segs : register(t1);

struct VsOut {
  float4 pos    : SV_Position;
  // x unused, y = signed across-position ∈ [-1,1] for the rim falloff.
  float2 local  : TEXCOORD0;
  nointerpolation float2 meta : TEXCOORD1;  // code, alpha
  nointerpolation float2 flow : TEXCOORD2;  // arc, stagger (continuous flow glow)
};

[shader("vertex")]
VsOut main(uint vid : SV_VertexID, uint iid : SV_InstanceID) {
  // Quad as a 6-vertex triangle list: x = along [0,1], y = across [-1,1].
  static const float2 quad[6] = {
    float2(0.0, -1.0), float2(1.0, -1.0), float2(0.0, 1.0),
    float2(1.0, -1.0), float2(1.0,  1.0), float2(0.0, 1.0),
  };

  Segment s = segs[iid];
  VsOut o;

  float2 p0 = s.a.xy, p1 = s.a.zw;
  float2 d = p1 - p0;
  float len = length(d);
  if (s.b.w > 0.5 || len < 1e-7) {            // dead / degenerate → cull
    o.pos = float4(2.0, 2.0, 2.0, 1.0);
    o.local = float2(0.0, 0.0);
    o.meta = float2(0.0, 0.0);
    o.flow = float2(0.0, 0.0);
    return o;
  }

  float2 dir = d / len;
  float2 nrm = float2(-dir.y, dir.x);
  float half_w = s.b.z * 0.5;                 // width in phase-space

  float2 c = quad[vid % 6u];
  float2 along = lerp(p0, p1, c.x);
  float2 world = along + nrm * (c.y * half_w);

  float2 vp = float2(res_x, res_y);
  float2 uv = pf_p_to_uv(world, vp);          // y-down uv; naga flips to WebGPU NDC
  o.pos = float4(uv * 2.0 - 1.0, 0.0, 1.0);
  o.local = float2(c.x, c.y);
  o.meta = float2(s.b.x, s.b.y);
  o.flow = float2(s.c.x, s.c.y);   // arc, stagger
  return o;
}
