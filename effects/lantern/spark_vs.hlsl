// source.mesh.lantern — the flakes' raster: one quad per flake around its
// ember capsule (p_shade), read from the render records by instance. Culled
// flakes collapse outside clip space.

#include "part_ub.hlsl"
#include "spark_common.hlsl"

StructuredBuffer<float4> recs : register(t7);

[shader("vertex")]
SparkOut main(uint vid : SV_VertexID, uint iid : SV_InstanceID) {
  static const float2 corners[6] = {
    float2(0.0, -1.0), float2(1.0, -1.0), float2(0.0, 1.0),
    float2(1.0, -1.0), float2(1.0,  1.0), float2(0.0, 1.0),
  };
  float4 r0 = recs[iid * 4u + 0u];
  float4 r1 = recs[iid * 4u + 1u];
  float4 r2 = recs[iid * 4u + 2u];
  float4 r3 = recs[iid * 4u + 3u];
  SparkOut o;
  float len = r1.z, w = r1.w;
  o.shp = float4(len, w, r2.w, r0.w);
  o.col = float4(r2.rgb, 0.0);
  o.mv = r3.xy;
  float2 c = corners[vid % 6u];
  float pad = w + 1.0;
  // Along: from behind the tail to past the head; across: either side.
  float u = lerp(-pad, len + pad, c.x);
  float v = c.y * pad;
  o.off = float2(u, v);
  if (r0.w <= 0.0) {
    o.pos = float4(2.0, 2.0, 2.0, 1.0);
    return o;
  }
  float2 ax = r1.xy;
  float2 tail = r0.xy - ax * len;
  float2 p = tail + ax * u + float2(-ax.y, ax.x) * v;
  o.pos = float4(p.x * pview.z * 2.0 - 1.0, 1.0 - p.y * pview.w * 2.0, 0.0, 1.0);
  return o;
}
