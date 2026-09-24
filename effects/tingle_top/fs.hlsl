// source.light.tingle_top — sparkle fragment shader. Mask (solid/circle/gaussian) ×
// life-fade × per-frame shimmer → alpha; single hue + per-particle jitter →
// colour × intensity. Output is straight colour + alpha; the additive PSO
// composites it as colour*alpha over the pre-filled input.

#include "common.hlsl"

cbuffer FsUniforms : register(b2) {
  float hue;
  float intensity;
  float frame_alpha_jitter;
  float alpha_curve;
  uint  shape_kind;
  uint  frame_index;
  float shape_param;
  float _pad;
};

struct VsOut {
  float4 pos    : SV_Position;
  float2 corner : TEXCOORD0;
  nointerpolation float4 data : TEXCOORD1;
};

[shader("pixel")]
float4 main(VsOut i) : SV_Target0 {
  float m = tt_mask(i.corner, shape_kind, shape_param);
  if (m <= 0.0) discard;

  float life_norm = i.data.x, hue_off = i.data.y;
  uint  iid = uint(i.data.z + 0.5);

  float aLife   = pow(saturate(life_norm), max(alpha_curve, 0.01));
  float shimmer = 1.0 - frame_alpha_jitter * tt_unit(tt_pcg2(iid + 0x5A5Au, frame_index));
  float alpha   = saturate(aLife * shimmer) * m;
  if (alpha <= 0.0) discard;

  float3 col = tt_hsv_to_rgb(float3(hue + hue_off, 0.7, 1.0)) * max(intensity, 0.0);
  return float4(col, alpha);
}
