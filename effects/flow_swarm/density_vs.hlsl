// source.particles.flow_swarm — density splat vertex shader.
//
// Splats each live particle as a small quad into the (square, fixed-res)
// density buffer so the next frame's update pass can read local crowding. The
// density buffer is an ABSTRACT proximity field — its resolution and the splat
// radius are unrelated to the visual viewport / particle size. Particle uv is
// used directly as density uv (the buffer is square, so no aspect handling),
// which keeps splat-write and density-read self-consistent.

#include "common.hlsl"

StructuredBuffer<Particle> particles : register(t0);

cbuffer DensityUniforms : register(b1) {
  float radius;     // splat half-size (isotropic uv: 1 unit = min(W,H) px)
  float aspect_x;   // min/W
  float aspect_y;   // min/H
  float _pad;
};

struct DOut {
  float4 pos    : SV_Position;
  float2 corner : TEXCOORD0;   // quad-local [-1,1]² → halo falloff in the FS
  nointerpolation float2 vel : TEXCOORD1;   // particle velocity → motion channels
};

[shader("vertex")]
DOut main(uint vid : SV_VertexID, uint iid : SV_InstanceID) {
  static const float2 corners[6] = {
    float2(-1.0, -1.0), float2( 1.0, -1.0), float2(-1.0,  1.0),
    float2( 1.0, -1.0), float2( 1.0,  1.0), float2(-1.0,  1.0),
  };
  float2 c = corners[vid % 6u];

  Particle p = particles[iid];
  DOut o;
  if (p.a.z <= 0.0) {   // dead → degenerate triangle (culled)
    o.pos = float4(2.0, 2.0, 2.0, 1.0);
    o.corner = float2(0.0, 0.0);
    o.vel = float2(0.0, 0.0);
    return o;
  }
  o.vel = p.b.xy;   // forward this particle's velocity to the motion channels

  // Aspect-correct the half-extent so the (square texture) halo is ROUND in
  // screen pixels: a uv radius of radius*aspect_x covers radius*min px on x, and
  // radius*aspect_y covers radius*min px on y — equal. Without this the splat is
  // circular in uv but elliptical in pixels (the density map looked squashed).
  float2 world = p.a.xy + c * float2(radius * aspect_x, radius * aspect_y);
  o.pos    = float4(world * 2.0 - 1.0, 0.0, 1.0);
  o.corner = c;
  return o;
}
