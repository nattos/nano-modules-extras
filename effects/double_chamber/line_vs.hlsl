// source.legacy.double_chamber — tracer line vertex shader.
// One oriented quad per segment (6 verts). Reads the segment buffer by
// instance id; degenerate (collapsed) for zeroed/dead segments.

#include "common.hlsl"

StructuredBuffer<Seg> segs : register(t0);

cbuffer VsUniforms : register(b1) {
  float aspect_x, aspect_y, width, _pad;
};

[shader("vertex")]
LineVsOut main(uint vid : SV_VertexID, uint iid : SV_InstanceID) {
  // (t, w): t along [0,1], w across {-1,+1}.
  static const float2 corners[6] = {
    float2(0, -1), float2(1, -1), float2(0, 1),
    float2(1, -1), float2(1, 1), float2(0, 1),
  };
  float2 c = corners[vid % 6u];

  Seg s = segs[iid];
  float2 p0 = s.a.xy, p1 = s.a.zw;
  float4 col = s.b;

  LineVsOut o;
  if (col.a <= 0.0 || (p0.x == p1.x && p0.y == p1.y)) {
    o.pos = float4(2, 2, 2, 1);
    o.local = float2(0, 0);
    o.col = float4(0, 0, 0, 0);
    return o;
  }

  float2 d = p1 - p0;
  float len = length(d);
  float2 dir = (len > 1e-6) ? d / len : float2(1, 0);
  float2 perp = float2(-dir.y, dir.x);

  float2 along = lerp(p0, p1, c.x);
  float2 off = perp * (c.y * width * 0.5);
  off = float2(off.x * aspect_x, off.y * aspect_y);   // round on any aspect
  float2 world = along + off;

  o.pos = float4(world * 2.0 - 1.0, 0.0, 1.0);
  o.local = c;
  o.col = col;
  return o;
}
