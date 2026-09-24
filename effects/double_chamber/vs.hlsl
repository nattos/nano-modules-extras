// source.legacy.double_chamber — instanced point vertex shader (shared by the
// P pool and the Big pool; the two draws differ only in buffer + point_size).
// Six verts per instance = one round quad. Dead particles collapse to a
// degenerate triangle so the rasterizer skips them.

#include "common.hlsl"

StructuredBuffer<Particle> particles : register(t0);

cbuffer VsUniforms : register(b1) {
  float aspect_x;     // min(W,H)/W
  float aspect_y;     // min(W,H)/H
  float point_size;   // isotropic-uv full size
  float _pad0;
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
  if (p.a.z <= 0.0) {
    o.pos = float4(2.0, 2.0, 2.0, 1.0);
    o.corner = float2(0.0, 0.0);
    o.col_life = float4(0.0, 0.0, 0.0, 0.0);
    o.extra = float4(0.0, 0.0, 0.0, 0.0);
    return o;
  }

  float2 half_iso  = (point_size * 0.5).xx;
  float2 local     = c * half_iso;
  float2 offset_uv = float2(local.x * aspect_x, local.y * aspect_y);
  float2 world_uv  = p.a.xy + offset_uv;
  float2 clip = world_uv * 2.0 - 1.0;

  float life_norm = saturate(p.a.z / max(p.a.w, 1e-5));
  float3 col = dc_unpack_rgb(asuint(p.b.w));

  o.pos = float4(clip, 0.0, 1.0);
  o.corner = c;
  o.col_life = float4(col, life_norm);
  o.extra = float4(length(p.b.xy), 0.0, 0.0, 0.0);
  return o;
}
