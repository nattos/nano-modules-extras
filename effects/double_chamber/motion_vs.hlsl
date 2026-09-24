// source.legacy.double_chamber — motion-vector vertex shader (P + Big points).
// Mirrors vs.hlsl's quad construction, but instead of colour it carries the
// particle's per-frame screen-space displacement: the sim integrates
// s += vel·dt·motion_rate and uv = 0.5 + s·aspect, so the uv velocity this
// frame is exactly vel·dt·motion_rate·aspect. Respawns zero the velocity, so
// there's no teleport spike when a particle recycles.

#include "common.hlsl"

StructuredBuffer<Particle> particles : register(t0);

cbuffer MotionVsUniforms : register(b1) {
  float aspect_x;
  float aspect_y;
  float point_size;   // isotropic-uv full size (footprint of the motion)
  float dt;
  float motion_rate;
  float scale;        // extra multiplier on the emitted particle velocity
  float _m1, _m2;
};

[shader("vertex")]
MotionVsOut main(uint vid : SV_VertexID, uint iid : SV_InstanceID) {
  static const float2 corners[6] = {
    float2(-1.0, -1.0), float2( 1.0, -1.0), float2(-1.0,  1.0),
    float2( 1.0, -1.0), float2( 1.0,  1.0), float2(-1.0,  1.0),
  };
  float2 c = corners[vid % 6u];

  Particle p = particles[iid];
  MotionVsOut o;
  if (p.a.z <= 0.0) {                       // dead → collapse
    o.pos = float4(2.0, 2.0, 2.0, 1.0);
    o.corner = float2(0.0, 0.0);
    o.motion = float2(0.0, 0.0);
    return o;
  }

  float2 half_iso  = (point_size * 0.5).xx;
  float2 local     = c * half_iso;
  float2 offset_uv = float2(local.x * aspect_x, local.y * aspect_y);
  float2 world_uv  = p.a.xy + offset_uv;
  float2 clip = world_uv * 2.0 - 1.0;

  o.pos = float4(clip, 0.0, 1.0);
  o.corner = c;
  o.motion = p.b.xy * dt * motion_rate * scale * float2(aspect_x, aspect_y);
  return o;
}
