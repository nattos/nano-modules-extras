// source.particles.sweep_chamber — density splat vertex shader.
//
// Scatters each live particle as a UNIT MASS into the (square, fixed-res)
// density buffer; the interaction radius is applied afterwards by the
// separable blur pass (density_blur), NOT by the splat footprint.
//
// Why: the old flow_swarm-style splat drew a quad of half-extent
// `interaction_radius`, i.e. ~(2·r·RES)² texels per particle. At r=0.04 on a
// 256² buffer that is ~200 blended fragments EACH — 700k particles then cost
// >100M additive RGBA16F fragments per frame, all landing on 65k texels, so
// clustered particles serialised the ROP and the frame rate collapsed.
// A gaussian is separable, so splat mass once (4 fragments) and convolve the
// whole buffer instead: cost becomes O(N + RES²·taps) instead of O(N·r²).
//
// The quad is exactly 2×2 texels of the density target, so the fragment
// shader's bilinear tent deposits a total weight of 1 per particle (the four
// covered texel centres carry the bilinear weights, which sum to unity).

#include "common.hlsl"

StructuredBuffer<Particle> particles : register(t0);

cbuffer DensityUniforms : register(b1) {
  float res;         // density buffer resolution (texels per axis)
  float dens_scale;  // screen uv → density uv (scale about the centre)
  float dens_off;    // = (1 - dens_scale) / 2
  float _pad;
};

struct DOut {
  float4 pos    : SV_Position;
  float2 corner : TEXCOORD0;   // quad-local [-1,1]² = texel offset from centre
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
  o.vel = p.b.xy;

  // The buffer covers a margin of screen uv beyond the frame, so an off-frame
  // particle still deposits mass its in-frame neighbours can feel.
  // One texel of half-extent in each axis → the 2×2 bilinear footprint.
  float2 world = p.a.xy * dens_scale + dens_off + c * (1.0 / max(res, 1.0));
  o.pos    = float4(world * 2.0 - 1.0, 0.0, 1.0);
  o.corner = c;
  return o;
}
