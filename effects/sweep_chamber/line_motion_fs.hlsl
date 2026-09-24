// source.particles.sweep_chamber — motion-vector fragment shader for lines.
// Coverage = the same soft cross-line falloff as line_fs.hlsl. dc parity.

#include "common.hlsl"

cbuffer LineMotionFsUniforms : register(b2) {
  float soft, _a, _b, _c;
};

[shader("pixel")]
float4 main(LineMotionVsOut i) : SV_Target0 {
  float across = abs(i.local.y);
  float cov = pow(saturate(1.0 - across), max(soft, 0.1));
  if (cov <= 0.0) discard;
  return float4(i.motion, 0.0, cov);
}
