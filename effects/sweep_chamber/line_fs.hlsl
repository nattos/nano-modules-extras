// source.particles.sweep_chamber — tracer line fragment shader.
// Soft falloff across the line width. double_chamber parity.

#include "common.hlsl"

cbuffer Uniforms : register(b2) {
  float soft, _a, _b, _c;
};

[shader("pixel")]
float4 main(LineVsOut i) : SV_Target0 {
  float across = abs(i.local.y);                 // 0 centre → 1 edge
  float edge = pow(saturate(1.0 - across), max(soft, 0.1));
  float a = i.col.a * edge;
  if (a <= 0.0) discard;
  return float4(i.col.rgb, a);
}
