// motion.field — motion pass.
//
// Computes the per-pixel velocity via the shared `mf_velocity_at`
// and writes it into the rgba16float motion texture. Pixels with zero
// activation (sub-threshold luma) inherit `upstreamMotion` — the
// upstream effect's render_outputs/motion when our render_outputs_in
// is connected, or zero otherwise (1x1 fallback texture; out-of-bounds
// loads return zero per WebGPU spec). Active pixels override with this
// stage's local velocity.

#include "common.hlsl"

Texture2D<float4>   inputTex       : register(t0);
RWTexture2D<float4> motionTex      : register(u1);
Texture2D<float4>   upstreamMotion : register(t3);

// Layout MUST match color.hlsl exactly so the same CPU-side uniform
// buffer binds cleanly into both shaders.
cbuffer Uniforms : register(b2) {
  float threshold;
  float softness;
  float magnitude;
  float mag_jitter;

  float mag_noise_scale;
  float rotation_rad;
  float rotation_weight;
  float radial_weight;

  float radial_anchor_x;
  float radial_anchor_y;
  float gradient_weight;
  float gradient_bias_rad;

  float angle_jitter;
  float angle_noise_scale;
  float vis_opacity;       // unused here (mirror only)
  float vis_scale;         // unused here

  float noise_time;
  float _pad_t0;
  float _pad_t1;
  float _pad_t2;
};

[numthreads(8, 8, 1)]
void main(uint3 gid : SV_DispatchThreadID) {
  uint w, h;
  motionTex.GetDimensions(w, h);
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

  float2 v = mf_velocity_at(inputTex, gid.xy, w, h, P);
  // mf_velocity_at returns exactly (0, 0) for sub-threshold pixels.
  // Any non-zero magnitude means this stage is "active" here and
  // should override upstream; zero means pass upstream through.
  float2 upstream = upstreamMotion[gid.xy].xy;
  float2 out_vel = (length(v) > 0.0) ? v : upstream;
  motionTex[gid.xy] = float4(out_vel.x, out_vel.y, 0.0, 0.0);
}
