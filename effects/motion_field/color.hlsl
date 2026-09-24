// motion.field — color pass.
//
// By default this is an identity copy (tex_in → tex_out). When
// `vis_opacity > 0` we additionally blend an HSV-polar visualization
// of the motion vectors over the input — useful while tuning
// thresholds and direction weights, off in production.

#include "common.hlsl"

Texture2D<float4>   inputTex  : register(t0);
RWTexture2D<float4> outputTex : register(u1);

// Schema mirror — keep in sync with motion.hlsl and main.cpp's
// Uniforms struct. Lay out as float4-aligned rows.
cbuffer Uniforms : register(b2) {
  // row 0: activation
  float threshold;
  float softness;
  float magnitude;
  float mag_jitter;

  // row 1: noise scales + static rotation
  float mag_noise_scale;
  float rotation_rad;
  float rotation_weight;
  float radial_weight;

  // row 2: radial anchor + gradient
  float radial_anchor_x;
  float radial_anchor_y;
  float gradient_weight;
  float gradient_bias_rad;

  // row 3: angle jitter + viz
  float angle_jitter;
  float angle_noise_scale;
  float vis_opacity;
  float vis_scale;

  // row 4: temporal evolution
  float noise_time;
  float _pad_t0;
  float _pad_t1;
  float _pad_t2;
};

[numthreads(8, 8, 1)]
void main(uint3 gid : SV_DispatchThreadID) {
  uint w, h;
  outputTex.GetDimensions(w, h);
  if (gid.x >= w || gid.y >= h) return;

  MfParams P;
  P.threshold         = threshold;
  P.softness          = softness;
  P.magnitude         = magnitude;
  P.mag_jitter        = mag_jitter;
  P.mag_noise_scale   = mag_noise_scale;
  P.rotation_rad      = rotation_rad;
  P.rotation_weight   = rotation_weight;
  P.radial_weight     = radial_weight;
  P.radial_anchor     = float2(radial_anchor_x, radial_anchor_y);
  P.gradient_weight   = gradient_weight;
  P.gradient_bias_rad = gradient_bias_rad;
  P.angle_jitter      = angle_jitter;
  P.angle_noise_scale = angle_noise_scale;
  P.noise_time        = noise_time;

  float4 base = inputTex[gid.xy];
  float3 rgb = base.rgb;

  if (vis_opacity > 0.0) {
    float2 v = mf_velocity_at(inputTex, gid.xy, w, h, P);
    float vlen = length(v);
    // Hue from direction angle, value from magnitude.
    float angle = atan2(v.y, v.x);
    float hue = angle / 6.2832 + 0.5;
    float val = saturate(vlen * vis_scale);
    float3 vis = mf_hsv_to_rgb(hue, 1.0, val);
    // Local alpha gates by per-pixel magnitude so non-moving pixels
    // stay visible (input shows through) at any opacity.
    float vis_alpha = saturate(vis_opacity * saturate(vlen * vis_scale));
    rgb = lerp(rgb, vis, vis_alpha);
  }

  outputTex[gid.xy] = float4(rgb, base.a);
}
