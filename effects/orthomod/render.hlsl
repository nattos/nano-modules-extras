// source.light.orthomod — render pass.
//
// Per pixel:
//   1. Find (bar, segment) from uv + inset geometry.
//   2. Look up the System B bit at (bar, segment).
//      - off → passthrough input
//      - on  → colored value (env-modulated hsv with hue scatter)
//   3. Outside the inset vertical range → passthrough input.
//
// All Hadamard / page / channel logic lives on the CPU; the shader gets:
//   - the current page bits (4 bars × hadamard_size, packed as a uint array)
//   - per-bar channel env values ch[0..3] (already waveform-modulated)
//   - the global env scalar + scatter seed inputs.

#include "nano_bars.hlsl"
#include "nano_color.hlsl"
#include "nano_hash.hlsl"

Texture2D<float4>   inputTex  : register(t0);
RWTexture2D<float4> outputTex : register(u1);

cbuffer Uniforms : register(b2) {
  // row 0
  float ch0;
  float ch1;
  float ch2;
  float ch3;

  // row 1
  float env;
  float primary_hue;
  float saturation;
  float intensity;

  // row 2
  float scatter_max;
  float channel_brightness_mod;
  float inset_top;
  float inset_bottom;

  // row 3
  uint  hadamard_size;     // M — segment dimension of System B
  uint  render_bits;       // number of segments shown in the inset range
  uint  page_idx;          // current System B page (used as hash key)
  uint  seed;              // user seed (used as hash key)

  // row 4
  float env_brightness;    // env shaped by the brightness power curve (CPU-side)
  float _pad_r4_0;
  float _pad_r4_1;
  float _pad_r4_2;
};

// Bits for the current page only — 4 rows × hadamard_size cols, one
// bit per uint (0 or 1). Layout: pageBits[bar * hadamard_size + col].
StructuredBuffer<uint> pageBits : register(t3);

[numthreads(8, 8, 1)]
void main(uint3 gid : SV_DispatchThreadID) {
  uint w, h;
  outputTex.GetDimensions(w, h);
  if (gid.x >= w || gid.y >= h) return;

  float2 uv = (float2(gid.xy) + 0.5) / float2(w, h);
  float4 base = inputTex[gid.xy];

  // Outside inset vertical range — passthrough.
  float y_lo = inset_top;
  float y_hi = 1.0 - inset_bottom;
  if (uv.y < y_lo || uv.y >= y_hi) {
    outputTex[gid.xy] = base;
    return;
  }
  float y_norm = (uv.y - y_lo) / max(y_hi - y_lo, 1e-5);

  // Map to segment in [0, render_bits).
  uint rb = render_bits < 1u ? 1u : render_bits;
  uint segment = uint(clamp(floor(y_norm * float(rb)), 0.0, float(rb - 1u)));
  uint bar = nano_bar_index(uv.x);
  uint M = hadamard_size < 1u ? 1u : hadamard_size;
  uint col = segment % M;

  uint bit = pageBits[bar * M + col];
  if (bit == 0u) {
    outputTex[gid.xy] = base;
    return;
  }

  // Per-bar channel envelope (already shaped — square/sin/on/off — on CPU).
  float chs[4] = { ch0, ch1, ch2, ch3 };
  float ch = chs[bar];

  // brightness = lerp(env_b, env_b * ch, channel_brightness_mod), where
  // env_b is the envelope shaped by the brightness power curve. The raw
  // env (not env_b) still drives scatter below so hue spread is unaffected.
  float brightness = lerp(env_brightness, env_brightness * ch, channel_brightness_mod);

  // tent(env) = 4 * env * (1 - env) — peaks 1 at env=0.5, 0 at 0/1.
  float tent = 4.0 * env * (1.0 - env);
  float scatter_amount = scatter_max * tent * ch;

  // Per-cell stable hash → signed [-1, 1].
  float hu = nano_hash21(float2(
    float(bar) * 31.7 + float(col) * 7.13 + float(seed) * 1.91,
    float(page_idx) * 13.37 + float(seed) * 0.71
  ));
  float hash_signed = hu * 2.0 - 1.0;
  float hue = primary_hue + hash_signed * scatter_amount;

  // Lit bars are additively blended over the input. The bar's emissive colour
  // (value V = envelope-driven brightness × intensity, so it's effectively
  // premultiplied by its own coverage) adds to the input RGB, and that same
  // intensity adds to the input alpha — so the generator only ever INCREASES
  // coverage (alpha goes up, never down) and contributes nothing as the
  // envelope decays. Off / outside-inset pixels above already passthrough base.
  float v = saturate(brightness * intensity);
  float3 rgb = nano_hsv_to_rgb(float3(frac(hue), saturate(saturation), v));
  outputTex[gid.xy] = float4(saturate(base.rgb + rgb), saturate(base.a + v));
}
