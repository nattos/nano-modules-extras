// filter.sim.simulant — Pass A: the difference-blend feedback injection.
//
// Faithful port of the original Simulant.wire feedback core (Video Mixers 23+20,
// both mode 11 = DIFFERENCE). Resolume's mixer is out = lerp(A, blend(A,B), op2),
// and the DIFFERENCE quirk (abs(A-B) never goes pure black, leaves halos) is the
// load-bearing behavior of the whole Simulant/Pixulant family — reproduced here.
//
//   fadedPrev = prev * (1 - choke)                       # node 23 (A = black)
//   inject    = colorize(transform(input))               # nodes 36,159,167
//   accumRaw  = lerp(fadedPrev, abs(fadedPrev-inject), injectAmount)   # node 20
//
// injectAmount is split into a STEADY part (Const Alpha) and the time-dependent
// FLICKER part (base + envelope). Only the flicker is gated by an optional
// roaming spatial MASK (a nano_twitch shape: size / softness / position, re-
// anchored on each flicker pulse), so pulses can light a shaped region instead
// of the whole frame. mask_amount blends global(1) → shaped selector; 0 = off,
// which reproduces the original global flicker exactly.

#include "nano_color.hlsl"
#include "nano_coords.hlsl"
#include "nano_twitch.hlsl"

Texture2D<float4>   delayPrev : register(t0);   // feedback buffer (blurred+decayed)
Texture2D<float4>   inputTex  : register(t1);   // incoming video
SamplerState        samp      : register(s2);   // Linear + ClampToEdge
RWTexture2D<float4> accumRaw  : register(u3);

cbuffer Uniforms : register(b4) {
  float choke, inject_const, inject_flicker, scale;
  float pos_x, pos_y, color_alpha, color_contrast;
  float color_r, color_g, color_b, _p0;
  float mask_amount, mask_anchor_x, mask_anchor_y, mask_radius;
  float mask_softness, mask_shape, aspect_x, aspect_y;
};

[numthreads(8, 8, 1)]
void main(uint3 gid : SV_DispatchThreadID) {
  uint W, H;
  accumRaw.GetDimensions(W, H);
  if (gid.x >= W || gid.y >= H) return;
  float2 uv = (float2(gid.xy) + 0.5) / float2(W, H);

  // Feedback, faded by (1 - choke) — node 23 (difference vs black = itself).
  float3 prev      = delayPrev.SampleLevel(samp, uv, 0).rgb;
  float3 fadedPrev = prev * (1.0 - choke);

  // Transform the input (scale about centre + translate) — node 36. Gather:
  // map this accum pixel back into input space. Off-frame reads inject nothing.
  float2 iuv = (uv - 0.5 - float2(pos_x, pos_y)) / max(scale, 1e-4) + 0.5;
  float3 inj = float3(0.0, 0.0, 0.0);
  if (all(iuv >= 0.0) && all(iuv <= 1.0)) {
    inj = inputTex.SampleLevel(samp, iuv, 0).rgb;
    // Colorize (node 159) — grey the input and tint by the colour, with a small
    // contrast, then crossfade against the raw input (Transition 167 by alpha).
    float g = nano_luminance(inj);
    g = saturate((g - 0.5) * (1.0 + color_contrast) + 0.5);
    float3 colorized = g * float3(color_r, color_g, color_b);
    inj = lerp(inj, colorized, color_alpha);
  }

  // Spatial flicker mask: gate the time-dependent flicker part by a roaming
  // twitch shape; the steady Const Alpha stays global. mask_amount blends the
  // global weight (1) toward the shaped selector (1 inside the shape).
  float w = 1.0;
  if (mask_amount > 1e-4) {
    float2 sq = nano_uv_to_cover_square(uv, float2(aspect_x, aspect_y));
    float sel = 1.0 - nano_twitch_mask(sq, float2(mask_anchor_x, mask_anchor_y),
                                       mask_radius, mask_softness, mask_shape, 1.0);
    w = lerp(1.0, sel, mask_amount);
  }
  float inject_amount = clamp(inject_const + inject_flicker * w, 0.0, 1.0);

  // DIFFERENCE blend feedback — node 20.
  float3 diff  = abs(fadedPrev - inj);
  float3 accum = lerp(fadedPrev, diff, inject_amount);
  accum = clamp(accum, 0.0, 1.0);
  accum = select(accum != accum, float3(0.0, 0.0, 0.0), accum);   // NaN sanitize

  accumRaw[gid.xy] = float4(accum, 1.0);
}
