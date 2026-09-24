// source.legacy.double_chamber — motion-vector fragment shader (points).
// Writes the particle's uv/frame velocity into render_outputs/motion. The
// shape mask is the coverage alpha; the AlphaOver PSO composes the velocity
// over whatever motion is already in the target (upstream / earlier particle),
// so overlapping footprints blend by coverage instead of summing.

#include "common.hlsl"

cbuffer MotionFsUniforms : register(b2) {
  uint  shape_kind;     // 0 point/solid · 1 gaussian · 2 circle
  float shape_param;
  float _f0, _f1;
};

[shader("pixel")]
float4 main(MotionVsOut i) : SV_Target0 {
  float cov = dc_mask(i.corner, shape_kind, shape_param);
  if (cov <= 0.0) discard;
  return float4(i.motion, 0.0, cov);   // .xy = velocity, .a = coverage
}
