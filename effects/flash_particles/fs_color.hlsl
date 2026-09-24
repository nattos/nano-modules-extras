// source.particles.flash_particles — color fragment shader.
//
// Outputs a per-fragment premultiplied color contribution. The render
// pipeline's blend state takes it from there:
//   - alpha-over PSO: src * src.a + dst * (1 - src.a)
//   - additive PSO : src * src.a + dst
//
// Mask (solid / squircle / gaussian) is evaluated from the
// interpolated quad-local corner coord. Life decay + per-frame
// jitter + captured per-particle alpha jitter all multiply into the
// final alpha; out-of-mask fragments are discarded so they don't
// even reach the blend stage.

#include "common.hlsl"

// Binding 0 (particles storage buf) and 1 (VsUniforms) live in the
// vertex shader's bind group; the fragment-stage uniform takes 2.
cbuffer Uniforms : register(b2) {
  float input_alpha;
  float color_blend;
  float global_color_r;
  float global_color_g;

  float global_color_b;
  float alpha_curve;
  float frame_alpha_jitter;
  uint  frame_index;

  uint  shape_kind;
  float shape_param;
  float exposure;     // RGB multiplier; alpha untouched, clips naturally
  float color_alpha;  // global alpha multiplier; gated to 0 by C++ skip
};

struct VsOut {
  float4 pos       : SV_Position;
  float2 corner    : TEXCOORD0;
  nointerpolation float4 captured  : TEXCOORD1;
  // x = rotation, y = life_norm, z = alpha_jitter_mult, w = frame_seed_as_float
  nointerpolation float4 state     : TEXCOORD2;
  nointerpolation float4 jitters   : TEXCOORD3;
};

[shader("pixel")]
float4 main(VsOut i) : SV_Target0 {
  // Mask. Out-of-shape pixels skip the rest entirely.
  float mask = pf_mask(i.corner, shape_kind, shape_param);
  if (mask <= 0.0) discard;

  // Life decay alpha + power curve.
  float alpha_decay = pow(saturate(i.state.y), max(alpha_curve, 1e-3));

  // Per-particle, per-frame alpha wobble. Stable seed lives on the
  // particle so it doesn't change every respawn; XOR with frame_index
  // gives a per-frame draw without storing per-frame state.
  uint  frame_seed = asuint(i.state.w);
  float fjit  = pf_signed(pf_pcg_hash(frame_seed ^ frame_index));
  float alpha = alpha_decay * (1.0 + frame_alpha_jitter * fjit);

  // Captured per-particle alpha jitter (multiplicative).
  alpha *= max(i.state.z, 0.0);
  alpha = saturate(alpha) * mask;
  // Global color-alpha gate. C++ also skips the whole pass at <= 0 so
  // we never reach this branch in that case, but keep the multiply in
  // for fractional values.
  alpha *= saturate(color_alpha);
  if (alpha <= 0.0) discard;

  // Color: captured ↔ global blend, then HSV jitters.
  float3 captured_rgb = i.captured.rgb;
  float3 global_color = float3(global_color_r, global_color_g, global_color_b);
  float3 base_rgb = lerp(captured_rgb, global_color, saturate(color_blend));

  float3 hsv = pf_rgb_to_hsv(max(base_rgb, 0.0));
  hsv.x = frac(hsv.x + i.jitters.x);
  hsv.z = saturate(hsv.z * max(i.jitters.y, 0.0));
  hsv.y = saturate(hsv.y * max(i.jitters.z, 0.0));
  float3 final_color = pf_hsv_to_rgb(hsv);

  // Exposure boost. Multiplied AFTER HSV jitters so a value > 1 can
  // legitimately push channels past 1.0 (the rgba8 surface format
  // clips to white on store, which is the requested behavior).
  final_color *= max(exposure, 0.0);

  // Straight (non-premultiplied) output. The pipeline's blend factors
  // multiply src.rgb by src.a, so:
  //   alpha-over PSO → final_color*alpha + dst*(1-alpha)
  //   additive  PSO → final_color*alpha + dst
  // Both expressions are exactly what we want for the two compositing
  // modes; the ONLY difference is which PSO the C++ side bound.
  return float4(final_color, alpha);
}
