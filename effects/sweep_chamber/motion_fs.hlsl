// source.particles.sweep_chamber — motion-vector fragment shader (points).
// Shape mask is the coverage alpha; AlphaOver composes the velocity over the
// existing motion target by coverage. double_chamber parity.

#include "common.hlsl"

cbuffer MotionFsUniforms : register(b2) {
  uint  shape_kind;     // 0 point · 1 gaussian · 2 circle · 3 solid
  float shape_param;
  float _f0, _f1;
};

[shader("pixel")]
float4 main(MotionVsOut i) : SV_Target0 {
  float cov = swc_mask(i.corner, shape_kind, shape_param);
  if (cov <= 0.0) discard;
  return float4(i.motion, 0.0, cov);   // .xy = velocity, .a = coverage
}
