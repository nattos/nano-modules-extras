// source.particles.sweep_chamber — instanced particle quad vertex shader.
//
// Six vertices per instance = one quad, read from the pool by SV_InstanceID.
// Dead particles collapse to a degenerate triangle outside clip space (no
// compaction pass). Size is isotropic uv (one unit = min(W,H) px) so a
// particle is a true pixel square on any aspect. flow_swarm parity.

#include "common.hlsl"

StructuredBuffer<Particle> particles : register(t0);

cbuffer VsUniforms : register(b1) {
  float aspect_x;       // min(W,H)/W
  float aspect_y;       // min(W,H)/H
  float point_size;     // fixed isotropic-uv size for the Point shape (~1px)
  float shape_kind;     // 0 point · 1 gaussian · 2 circle · 3 solid
};

[shader("vertex")]
VsOut main(uint vid : SV_VertexID, uint iid : SV_InstanceID) {
  static const float2 corners[6] = {
    float2(-1.0, -1.0), float2( 1.0, -1.0), float2(-1.0,  1.0),
    float2( 1.0, -1.0), float2( 1.0,  1.0), float2(-1.0,  1.0),
  };
  float2 c = corners[vid % 6u];

  Particle p = particles[iid];
  VsOut o;

  // Dead / zero-size → degenerate triangle (rasterizer culls).
  if (p.a.z <= 0.0 || p.b.z <= 0.0) {
    o.pos      = float4(2.0, 2.0, 2.0, 1.0);
    o.corner   = float2(0.0, 0.0);
    o.col_life = float4(0.0, 0.0, 0.0, 0.0);
    o.vel      = float4(0.0, 0.0, 0.0, 0.0);
    return o;
  }

  // Point shape ignores the per-particle size and draws a fixed ~1px quad.
  float sz = (shape_kind < 0.5) ? point_size : p.b.z;
  float2 half_iso  = (sz * 0.5).xx;
  float2 local     = c * half_iso;
  float2 offset_uv = float2(local.x * aspect_x, local.y * aspect_y);
  float2 world_uv  = p.a.xy + offset_uv;
  float2 clip = world_uv * 2.0 - 1.0;

  float life_norm = saturate(p.a.z / max(p.a.w, 1e-5));
  uint packed = asuint(p.b.w);
  float3 col = swc_unpack_rgb(packed);
  float spd  = length(p.b.xy);

  o.pos      = float4(clip, 0.0, 1.0);
  o.corner   = c;
  o.col_life = float4(col, life_norm);
  o.vel      = float4(p.b.xy, spd, 0.0);
  return o;
}
