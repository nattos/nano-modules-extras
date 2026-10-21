// source.mesh.lantern — the flakes' raster: one quad per flake, sized to its
// EWA footprint (p_shade), read from the render records by instance. Culled
// flakes collapse outside clip space.

#include "part_ub.hlsl"

StructuredBuffer<float4> recs : register(t7);

#include "spark_common.hlsl"

[shader("vertex")]
SparkOut main(uint vid : SV_VertexID, uint iid : SV_InstanceID) {
  static const float2 corners[6] = {
    float2(-1.0, -1.0), float2( 1.0, -1.0), float2(-1.0,  1.0),
    float2( 1.0, -1.0), float2( 1.0,  1.0), float2(-1.0,  1.0),
  };
  float4 r0 = recs[iid * 4u + 0u];
  float4 r1 = recs[iid * 4u + 1u];
  float4 r2 = recs[iid * 4u + 2u];
  float4 r3 = recs[iid * 4u + 3u];
  SparkOut o;
  o.ctr = float4(r0.xy, r0.w, 0.0);
  o.q = r1;
  o.col = float4(r2.rgb, 0.0);
  o.mv = r3.xy;
  o.off = corners[vid % 6u] * r2.w;
  if (r0.w <= 0.0) {
    o.pos = float4(2.0, 2.0, 2.0, 1.0);
    return o;
  }
  float2 p = r0.xy + o.off;
  o.pos = float4(p.x * pview.z * 2.0 - 1.0, 1.0 - p.y * pview.w * 2.0, 0.0, 1.0);
  return o;
}
