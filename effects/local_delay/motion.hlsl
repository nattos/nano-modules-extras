// motion.local_delay — motion pass.
//
// Publishes the motion field on render_outputs/motion so a downstream
// motion.blur can smooth residual imperfections.
//
// The written vector is the SMOOTHED FLOW (the real per-frame velocity in
// uv/frame — exactly what motion_blur consumes), gated by the spatial mask
// and scaled by motion_gain. Crucially it is NOT scaled by the blend
// weight: the weight is itself ∝ |flow|, so flow*weight collapses to ~flow²
// and the vectors come out far too small (you'd need motion_blur strength
// ~50). flow*mask keeps a usable magnitude; motion_gain tunes it on this
// node instead. Pixels we don't drive inherit the upstream motion field.

#include "common.hlsl"

Texture2D<float4>   flowTex        : register(t0);   // RG = flow, B = weight, A = mask
RWTexture2D<float4> motionTex      : register(u1);   // rgba16f output rail
Texture2D<float4>   upstreamMotion : register(t2);

cbuffer Uniforms : register(b3) {
  float delay_amount;  float noise_weight;     float seed;            float weight_gain;
  float vignette;      float vignette_radius;  float vignette_softness; float squash;
  float max_flow;      float align_amount;     float align_sharpness; float have_history;
  float aspect_x;      float aspect_y;         float debug_show_motion; float history_alpha;
  float motion_gain;   float _pad1;            float _pad2;           float _pad3;
};

[numthreads(8, 8, 1)]
void main(uint3 gid : SV_DispatchThreadID) {
  uint w, h;
  motionTex.GetDimensions(w, h);
  if (gid.x >= w || gid.y >= h) return;

  // motion_gain == 0 is handled on the host (the dispatch is skipped and the
  // rail texture cleared), so this pass only runs for motion_gain > 0 — no
  // special-casing needed here for the "publish nothing" state.
  float4 f = flowTex[gid.xy];
  float2 v = f.rg * f.a * motion_gain;         // real velocity, mask-gated, scaled
  float2 upstream = upstreamMotion[gid.xy].xy;
  // Where we drive motion, override; elsewhere pass upstream through.
  float2 out_vel = (length(v) > 0.0) ? v : upstream;
  motionTex[gid.xy] = float4(out_vel.x, out_vel.y, 0.0, 0.0);
}
