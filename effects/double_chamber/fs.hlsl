// source.legacy.double_chamber — point fragment shader (additive).
//
// Colour = white ↔ captured-input blend (Color Contrib), hue-shifted by
// render_hue, tinted, exposed. Alpha = life-fade × opacity × shape mask.
// Straight output; the additive PSO multiplies rgb by alpha.

#include "common.hlsl"

cbuffer Uniforms : register(b2) {
  float color_contrib;   // 0 = white, 1 = captured input colour
  float render_hue;      // hue offset added to the colour
  float opacity;
  float alpha_curve;

  float tint_r, tint_g, tint_b;
  float exposure;

  uint  shape_kind;      // 0 point/solid · 1 gaussian · 2 circle
  float shape_param;
  float _p0, _p1;
};

[shader("pixel")]
float4 main(VsOut i) : SV_Target0 {
  float mask = dc_mask(i.corner, shape_kind, shape_param);
  if (mask <= 0.0) discard;

  float alpha = pow(saturate(i.col_life.w), max(alpha_curve, 1e-3))
                * saturate(opacity) * mask;
  if (alpha <= 0.0) discard;

  float3 base = lerp(float3(1.0, 1.0, 1.0), i.col_life.rgb, saturate(color_contrib));
  if (render_hue != 0.0) {
    float3 hsv = dc_rgb_to_hsv(base);
    hsv.x = frac(hsv.x + render_hue);
    base = dc_hsv_to_rgb(hsv);
  }
  base *= float3(tint_r, tint_g, tint_b) * max(exposure, 0.0);
  return float4(base, alpha);
}
