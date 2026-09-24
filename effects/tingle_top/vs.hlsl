// source.light.tingle_top — instanced sparkle vertex shader. 6 verts per particle form
// one quad; dead particles collapse to a degenerate triangle outside clip
// space so the rasteriser skips them (no compaction pass). Size is in
// isotropic uv (min(W,H) units) so a sparkle is round on any aspect.

#include "common.hlsl"

StructuredBuffer<Particle> parts : register(t0);

cbuffer VsUniforms : register(b1) {
  float aspect_x;   // min(W,H) / W
  float aspect_y;   // min(W,H) / H
  float _pad0;
  float _pad1;
};

struct VsOut {
  float4 pos    : SV_Position;
  float2 corner : TEXCOORD0;                 // particle-local ∈ [-1,1]² (mask coord)
  nointerpolation float4 data : TEXCOORD1;   // x life_norm, y hue_offset, z instance id
};

[shader("vertex")]
VsOut main(uint vid : SV_VertexID, uint iid : SV_InstanceID) {
  static const float2 corners[6] = {
    float2(-1, -1), float2(1, -1), float2(-1, 1),
    float2( 1, -1), float2(1,  1), float2(-1, 1),
  };
  float2 c = corners[vid % 6u];

  Particle p = parts[iid];
  VsOut o;
  float life_remain = p.a.w, life_total = max(p.b.x, 1e-5), sz = p.a.z;
  if (life_remain <= 0.0 || sz <= 0.0) {     // dead → degenerate
    o.pos = float4(2.0, 2.0, 2.0, 1.0);
    o.corner = float2(0.0, 0.0);
    o.data = float4(0.0, 0.0, 0.0, 0.0);
    return o;
  }

  float2 off   = float2(c.x * sz * aspect_x, c.y * sz * aspect_y);
  float2 world = p.a.xy + off;
  float2 clip  = world * 2.0 - 1.0;          // uv→clip (naga handles the y flip)

  o.pos    = float4(clip, 0.0, 1.0);
  o.corner = c;
  o.data   = float4(saturate(life_remain / life_total), p.b.z, float(iid), 0.0);
  return o;
}
