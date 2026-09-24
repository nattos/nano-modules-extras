// source.particles.sweep_chamber — motion-vector vertex shader for lines.
// Same oriented-quad geometry as line_vs.hlsl, but the per-pixel motion
// vector points ALONG the segment (its tangent) scaled by `line_speed` — a
// line "moves" lengthwise, so downstream motion blur smears it along its
// body rather than across it. double_chamber parity.

#include "common.hlsl"

StructuredBuffer<Seg> segs : register(t0);

cbuffer LineMotionVsUniforms : register(b1) {
  float aspect_x, aspect_y, width, line_speed;
};

[shader("vertex")]
LineMotionVsOut main(uint vid : SV_VertexID, uint iid : SV_InstanceID) {
  static const float2 corners[6] = {
    float2(0, -1), float2(1, -1), float2(0, 1),
    float2(1, -1), float2(1, 1), float2(0, 1),
  };
  float2 c = corners[vid % 6u];

  Seg s = segs[iid];
  float2 p0 = s.a.xy, p1 = s.a.zw;
  float4 col = s.b;

  LineMotionVsOut o;
  if (col.a <= 0.0 || (p0.x == p1.x && p0.y == p1.y)) {
    o.pos = float4(2, 2, 2, 1);
    o.local = float2(0, 0);
    o.motion = float2(0, 0);
    return o;
  }

  float2 d = p1 - p0;
  float len = length(d);
  float2 dir = (len > 1e-6) ? d / len : float2(1, 0);
  float2 perp = float2(-dir.y, dir.x);

  float2 along = lerp(p0, p1, c.x);
  float2 off = perp * (c.y * width * 0.5);
  off = float2(off.x * aspect_x, off.y * aspect_y);
  float2 world = along + off;

  o.pos = float4(world * 2.0 - 1.0, 0.0, 1.0);
  o.local = c;
  o.motion = dir * line_speed;          // uv/frame along the segment
  return o;
}
