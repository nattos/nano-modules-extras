// motion.local_delay — color / forward-advection pass.
//
// Instead of looking BACK into smeary history buffers (which lose all detail
// exactly where motion peaks), we trace FORWARD along the (temporally-
// smoothed) flow field and sample the ORIGINAL input at the endpoint — like
// motion_blur's line integral, but a single solid tap rather than an average.
//
//   pos = this pixel
//   repeat `delay_steps` times:   pos += flow(pos) * (delay_amount * mask)
//   out = input(pos)
//
// The smoothed flow already carries a trail of motion for isolated objects,
// so a trailing pixel's flow points toward where the object is now; advecting
// forward lands on it and the pixel reads the object — a clean, solid echo.
// Step size scales with delay_amount and the spatial mask (vignette/noise),
// so the trail length is modulated per pixel. Masked pixels (mask 0) don't
// advect and just show the present.

#include "common.hlsl"

Texture2D<float4>   inputTex : register(t0);   // original input (sampled at the endpoint)
Texture2D<float4>   flowTex  : register(t1);   // RG = smoothed flow, B = index, A = mask
RWTexture2D<float4> outTex   : register(u2);

cbuffer Uniforms : register(b3) {
  float delay_amount;  float noise_weight;     float seed;            float weight_gain;
  float vignette;      float vignette_radius;  float vignette_softness; float squash;
  float max_flow;      float align_amount;     float align_sharpness; float have_history;
  float aspect_x;      float aspect_y;         float debug_show_motion; float history_alpha;
  float motion_gain;   float delay_steps;      float _pad2;           float delay_dir;
};

#define LD_MAX_STEPS   32
#define LD_ADVECT_GAIN 3.0   // per-step ≈ this many frames of motion (× delay_amount)

[numthreads(8, 8, 1)]
void main(uint3 gid : SV_DispatchThreadID) {
  uint w, h;
  outTex.GetDimensions(w, h);
  if (gid.x >= w || gid.y >= h) return;
  int2 dims = int2(int(w), int(h));
  float2 vp = float2(w, h);

  float4 flow0 = flowTex[gid.xy];   // rg = flow, b = index, a = mask

  // Debug: motion field — hue = flow direction, value = index (motion gated
  // by the vignette + noise mask, so the noise shows as a darkening overlay
  // wherever there's motion).
  if (debug_show_motion > 0.5) {
    float hue = atan2(flow0.g, flow0.r) / 6.2832 + 0.5;
    outTex[gid.xy] = float4(ld_hsv_to_rgb(hue, 1.0, saturate(flow0.b)), 1.0);
    return;
  }

  // Flow-passthrough / pure-vector-conditioner mode: delay_amount = 0 makes the
  // advection step `scale` 0, so `head` never leaves this pixel and the echo
  // collapses to an exact copy of the input. Early-out to a straight passthrough
  // — skips the whole advection loop (the real cost) — while align/mask/motion
  // still run, so local_delay acts as a pure motion-vector processor (smooth /
  // mask / curve the field) feeding a downstream consumer.
  if (delay_amount <= 0.0) {
    outTex[gid.xy] = float4(inputTex[gid.xy].rgb, 1.0);
    return;
  }

  // Forward advection along the flow streamline (re-sampling the flow each
  // step, so curved trails are followed), tracking where the flow magnitude
  // PEAKS — the head: the object's current position, where motion is freshest.
  //
  // PATH MODULATION: the per-pixel step `scale` rides on the INDEX (B) =
  // motion_response(weight_gain, squash) × mask(vignette, noise). So how FAR a
  // pixel reaches back is set by its own motion (and the noise/vignette): a
  // high-index pixel advects to the object → solid echo; a low/masked pixel
  // barely moves → head stays near home → shows the present. The depth of the
  // delay is per-pixel, which is the original intent.
  // delay_dir = +1 walks WITH the flow → echo trails behind (Past/causal);
  // -1 walks against it → echo leads ahead (Future).
  float scale = flow0.b * delay_amount * LD_ADVECT_GAIN * delay_dir;
  float2 pos  = float2(gid.xy);
  float2 head = pos;
  float  best = length(flow0.rg);                    // flow magnitude at the start
  [loop] for (int i = 0; i < LD_MAX_STEPS; i++) {
    if ((float)i >= delay_steps) break;
    float2 v_uv = ld_bil_flow(flowTex, pos, dims);   // uv/frame at the current pos
    float  m    = length(v_uv);
    if (m > best) { best = m; head = pos; }          // strongest-flow point so far = head
    pos += v_uv * vp * scale;                         // advance, in pixels
  }

  // Single tap at the head (no path blending; the reach above is the delay).
  outTex[gid.xy] = float4(ld_bil_rgb(inputTex, head, dims), 1.0);
}
