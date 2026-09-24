// source.particles.sweep_chamber — particle fragment shader.
//
// Color = captured-input ↔ solid blend, optionally tinted by flow direction.
// Alpha = life fade × opacity × shape mask. Straight (non-premultiplied)
// output; the PSO's blend factors multiply rgb by alpha, so this one shader
// serves both the alpha-over and additive PSOs. flow_swarm parity.

#include "common.hlsl"

cbuffer Uniforms : register(b2) {
  float color_blend;     // 0 = solid_color, 1 = captured input color (dc parity:
                         // dc points were ALWAYS input-colored; 1 = that look)
  float solid_r;
  float solid_g;
  float solid_b;

  float tint_by_flow;    // 0 = off, 1 = full hue-by-direction tint
  float opacity;         // global alpha multiplier
  float alpha_curve;     // life-fade exponent
  float shape_param;

  uint  shape_kind;      // 0 point · 1 gaussian · 2 circle · 3 solid
  float exposure;        // rgb multiplier
  float _pad0;
  float _pad1;
};

[shader("pixel")]
float4 main(VsOut i) : SV_Target0 {
  float mask = swc_mask(i.corner, shape_kind, shape_param);
  if (mask <= 0.0) discard;

  float life_norm = i.col_life.w;
  float alpha = pow(saturate(life_norm), max(alpha_curve, 1e-3)) * saturate(opacity) * mask;
  if (alpha <= 0.0) discard;

  float3 base = lerp(float3(solid_r, solid_g, solid_b), i.col_life.rgb,
                     saturate(color_blend));

  // Optional flow tint: hue from velocity direction, brightness from speed.
  if (tint_by_flow > 1e-3) {
    float hue = frac(atan2(i.vel.y, i.vel.x) / 6.28318530718 + 0.5);
    float val = saturate(i.vel.z * 4.0);
    float3 flow_col = swc_hsv_to_rgb(float3(hue, 0.85, max(val, 0.3)));
    base = lerp(base, flow_col, saturate(tint_by_flow));
  }

  base *= max(exposure, 0.0);
  return float4(base, alpha);
}
