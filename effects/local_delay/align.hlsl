// motion.local_delay — align + temporal-smooth + weight pass.
//
// Three steps, producing the final per-pixel flow + blend weight the color
// and motion passes consume:
//   1. spatial colinear polish of the current LK flow (ld_align_at).
//   2. TEMPORAL exponential moving average against last frame's smoothed
//      flow — this is what kills the flicker. The LK estimate is noisy
//      frame-to-frame (and at render-fps > source-fps it zeroes on
//      duplicate frames then spikes at each new source frame); EMA-ing the
//      flow holds the weight steady. Same dt-based, frame-rate-independent
//      alpha as the history (one `smoothing` knob drives both).
//   3. mask + the per-pixel temporal-lookup index from the SMOOTHED flow.
// Output: RG = smoothed flow (uv/frame), B = lookup index [0,1], A = mask.
// (The color pass scales B by delay_amount*delay_frames to pick a fractional
// point in the delay ring; the motion pass uses RG*A for its vectors.)

#include "common.hlsl"

Texture2D<float4>   flowIn   : register(t0);   // current LK flow (full res)
RWTexture2D<float4> flowOut  : register(u1);   // smoothed flow (RG) + weight (B)
Texture2D<float4>   flowPrev : register(t2);   // last frame's smoothed flow

cbuffer Uniforms : register(b3) {
  float delay_amount;  float noise_weight;     float seed;            float weight_gain;
  float vignette;      float vignette_radius;  float vignette_softness; float squash;
  float max_flow;      float align_amount;     float align_sharpness; float have_history;
  float aspect_x;      float aspect_y;         float debug_show_motion; float history_alpha;
  float motion_gain;   float delay_steps;      float noise_time;      float _pad3;
  float twitch_shape;    float twitch_radius;      float twitch_softness;   float twitch_strength;
  float twitch_anchor_x; float twitch_anchor_y;    float _pad5;           float _pad6;
};

[numthreads(8, 8, 1)]
void main(uint3 gid : SV_DispatchThreadID) {
  uint w, h;
  flowOut.GetDimensions(w, h);
  if (gid.x >= w || gid.y >= h) return;

  LdParams P = ld_make(delay_amount, noise_weight, seed, weight_gain,
                       vignette, vignette_radius, vignette_softness, squash,
                       max_flow, align_amount, align_sharpness, have_history,
                       aspect_x, aspect_y, debug_show_motion);

  float2 aligned  = ld_align_at(flowIn, gid.xy, w, h, P);
  float2 smoothed = lerp(flowPrev[gid.xy].xy, aligned, saturate(history_alpha));
  float mask = ld_mask_at(gid.xy, w, h, P, noise_time,
                          twitch_shape, twitch_radius, twitch_softness,
                          float2(twitch_anchor_x, twitch_anchor_y), twitch_strength);
  float idx  = ld_index_at(smoothed, mask, P);
  flowOut[gid.xy] = float4(smoothed, idx, mask);
}
