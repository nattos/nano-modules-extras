// source.light.plasma_beam_cannon — render pass.
//
// Per pixel:
//   1. If effect is idle → passthrough.
//   2. If bar isn't targeted → passthrough.
//   3. If outside beam extent → passthrough.
//   4. If flicker tail is active AND currently OFF → passthrough
//      (whole-bar black flicker; the beam stays computed beneath but
//      isn't drawn).
//   5. If breaks_active (release phase) AND this pixel falls inside
//      any SOLID break's range → passthrough (the break eats the beam).
//   6. Otherwise → paint the beam.
//
// Hard edges throughout; no alpha. A "break" looks like a clean
// horizontal slice of black through the beam.

#include "nano_bars.hlsl"

// One vec4 per break particle:
//   .x = y center (uv-space)
//   .y = size (uv-space, full height)
//   .z = type (0 attractor, 1 repellor, 2 spacer; spacer is invisible)
//   .w = pad
StructuredBuffer<float4> breaks    : register(t3);

Texture2D<float4>   inputTex  : register(t0);
RWTexture2D<float4> outputTex : register(u1);

cbuffer Uniforms : register(b2) {
  float beam_y_min;
  float beam_y_max;
  float intensity;
  uint  active;

  float color_r;
  float color_g;
  float color_b;
  uint  bar_target_all;

  uint  bar_target;
  uint  particles_per_bar;
  uint  breaks_active;
  uint  flicker_active;

  uint  flicker_on;
  uint  _pad_0;
  uint  _pad_1;
  uint  _pad_2;
};

[numthreads(8, 8, 1)]
void main(uint3 gid : SV_DispatchThreadID) {
  uint W, H;
  outputTex.GetDimensions(W, H);
  if (gid.x >= W || gid.y >= H) return;

  float4 base = inputTex[gid.xy];
  if (active == 0u) {
    outputTex[gid.xy] = base;
    return;
  }

  uint bar = nano_bar_index(float(gid.x) / float(W));
  if (bar_target_all == 0u && bar != bar_target) {
    outputTex[gid.xy] = base;
    return;
  }

  float uv_y = (float(gid.y) + 0.5) / float(H);
  if (uv_y < beam_y_min || uv_y > beam_y_max) {
    outputTex[gid.xy] = base;
    return;
  }

  // Flicker tail: when active and currently OFF, the whole beam is
  // suppressed for this frame.
  if (flicker_active != 0u && flicker_on == 0u) {
    outputTex[gid.xy] = base;
    return;
  }

  // Break particle check — only during release. Iterate this bar's
  // particle slice; if any SOLID particle's range covers uv_y, the
  // beam is "eaten" → passthrough at this pixel.
  if (breaks_active != 0u) {
    uint base_idx = bar * particles_per_bar;
    for (uint i = 0u; i < particles_per_bar; i++) {
      float4 p = breaks[base_idx + i];
      // type 2 = spacer (invisible). Solid (0, 1) paints the break.
      if (p.z > 1.5) continue;
      if (p.y <= 0.0) continue;     // inactive slot
      float half_size = p.y * 0.5;
      if (uv_y > p.x - half_size && uv_y < p.x + half_size) {
        outputTex[gid.xy] = base;
        return;
      }
    }
  }

  float3 c = float3(color_r, color_g, color_b) * intensity;
  outputTex[gid.xy] = float4(saturate(base.rgb + c), base.a);
}
