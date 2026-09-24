// warp.legacy.d_wave — motion-vector output (render_outputs/motion).
//
// Only run when a downstream sink reads the rail. Emits per-pixel screen-space
// velocity in uv/frame (.xy of an RGBA16F), the convention shared with
// motion_swarm / double_chamber. For a warp the velocity is the per-frame
// CHANGE in the radial displacement: disp(p) = c·(warpFactor − 1), and
// warpFactor = 1 − GAIN·LI·distortion, so the screen velocity of the content at
// p ≈ −(disp_now − disp_prev) = c·GAIN·distortion·(LI_now − LI_prev). We read
// the wave + damp fields at both this frame and last (ping-ponged) to get the
// strength delta, so both the propagating wave AND the fast dampening flashes
// contribute motion automatically. Upstream motion (the input's own) passes
// through, sampled at the warped position so it rides with the content.

Texture2D<float4>   waveNow  : register(t0);
Texture2D<float4>   wavePrev : register(t1);
Texture2D<float4>   dampNow  : register(t2);
Texture2D<float4>   dampPrev : register(t3);
Texture2D<float4>   upstream : register(t4);   // render_outputs_in/motion (or 1×1 zero)
SamplerState        sampF    : register(s5);   // Linear + Repeat (angle wraps)
SamplerState        sampIn   : register(s6);   // Linear + ClampToEdge
RWTexture2D<float4> motionOut : register(u7);

cbuffer Uniforms : register(b8) {
  float aspect, distortion, scale, squeeze;     // distortion = the curved/scaled value
  float damp_amount, center_x, center_y, motion_scale;
}

static const float DW_PI   = 3.14159265358979323846;
static const float DW_GAIN = 5.0;

[numthreads(8, 8, 1)]
void main(uint3 gid : SV_DispatchThreadID) {
  uint W, H;
  motionOut.GetDimensions(W, H);
  if (gid.x >= W || gid.y >= H) return;

  float2 uv = (float2(gid.xy) + 0.5) / float2(W, H);
  float2 anchor = 0.5 + float2(center_x, center_y);
  float2 c = uv - anchor;
  float  ax = c.x * aspect;

  float ang = atan2(c.y, ax) / (2.0 * DW_PI) + 0.5;
  float r   = length(float2(ax, c.y));
  float row = clamp((r + squeeze * 0.5) / max(scale, 1e-3), 0.002, 0.998);
  float2 polar = float2(frac(ang), row);

  float liNow  = max(waveNow.SampleLevel(sampF, polar, 0).r  - damp_amount * dampNow.SampleLevel(sampF, polar, 0).r,  0.0);
  float liPrev = max(wavePrev.SampleLevel(sampF, polar, 0).r - damp_amount * dampPrev.SampleLevel(sampF, polar, 0).r, 0.0);

  // Per-frame velocity the warp induces at this pixel.
  float2 warpMotion = c * DW_GAIN * distortion * (liNow - liPrev) * motion_scale;

  // Upstream motion rides with the content: sample it at where this pixel reads.
  float warpFactor = clamp(1.0 - DW_GAIN * liNow * distortion, 0.15, 2.0);
  float2 sampleUV = c * warpFactor + anchor;
  float2 up = upstream.SampleLevel(sampIn, sampleUV, 0).xy;

  motionOut[gid.xy] = float4(up + warpMotion, 0.0, 0.0);
}
