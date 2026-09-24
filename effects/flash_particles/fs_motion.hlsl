// source.particles.flash_particles — motion fragment shader.
//
// Outputs the per-particle motion vector along its rotation, with the
// mask value as the alpha channel. The render pipeline runs in
// alpha-over blend mode for this pass, so:
//   result.xy = motion.xy * mask + upstream.xy * (1 - mask)
// — exactly the lerp we want: fully replace where the particle is
// solid, smoothly fade to upstream where the mask softens, and leave
// upstream untouched where mask=0 (we discard those fragments before
// the blend stage).
//
// The destination texture is pre-filled with the upstream motion
// content (rgba16f) by a compute pass; this raster pass uses
// loadOp='load' so that pre-fill survives until the quads draw.
//
// Last-particle-wins under overlap (the user opted out of ordering
// concerns; this matches the same iteration order the color pass
// would expose).

#include "common.hlsl"

// Binding 0 (particles storage buf) and 1 (VsUniforms) live in the
// vertex shader's bind group; the fragment-stage uniform takes 2.
cbuffer Uniforms : register(b2) {
  float motion_strength;
  uint  shape_kind;
  float shape_param;
  float alpha_curve;   // shared with the color pass — scales motion mag by
                       // pow(life_norm, alpha_curve) so dying particles
                       // write proportionally weaker velocity, matching
                       // the way they fade visually.
};

struct VsOut {
  float4 pos       : SV_Position;
  float2 corner    : TEXCOORD0;
  nointerpolation float4 captured  : TEXCOORD1;
  // x = rotation, ...
  nointerpolation float4 state     : TEXCOORD2;
  nointerpolation float4 jitters   : TEXCOORD3;
};

[shader("pixel")]
float4 main(VsOut i) : SV_Target0 {
  float mask = pf_mask(i.corner, shape_kind, shape_param);
  if (mask <= 0.0) discard;

  float rot       = i.state.x;
  float life_norm = saturate(i.state.y);
  float decay     = pow(life_norm, max(alpha_curve, 1e-3));

  float2 motion = float2(cos(rot), sin(rot)) * motion_strength * decay;
  // .xy = velocity, .z reserved, .w = mask used as the blend alpha so
  // covered pixels overwrite upstream and edge fade is smooth.
  return float4(motion, 0.0, mask);
}
