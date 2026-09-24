// source.light.bounce_resonator — color pass.
//
// Each bar IS its whole 1/4-width column: fill the entire vertical strip
// with the bar's colour. Brightness = its diffusion value; hue AND
// saturation = its diffused colour state (band_color's value supplies the
// rest). The per-bar state is read from the GPU sim buffer (sim.hlsl).

#include "nano_bars.hlsl"
#include "nano_color.hlsl"

Texture2D<float4>   inputTex  : register(t0);
RWTexture2D<float4> outputTex : register(u1);

cbuffer Uniforms : register(b2) {
  float band_val; float intensity; float input_opacity; float chroma_hold;
};

struct SimState { float v[4]; float h[4]; float s[4]; float env; float pad[3]; };
StructuredBuffer<SimState> simState : register(t3);

[numthreads(8, 8, 1)]
void main(uint3 gid : SV_DispatchThreadID) {
  uint w, h;
  outputTex.GetDimensions(w, h);
  if (gid.x >= w || gid.y >= h) return;

  float2 uv = (float2(gid.xy) + 0.5) / float2(w, h);
  float4 base = inputTex[gid.xy];

  uint bar = nano_bar_index(uv.x);
  SimState st = simState[0];
  float vi = st.v[bar];

  float3 col = nano_hsv_to_rgb(float3(st.h[bar], st.s[bar], band_val));
  float3 lin = col * (intensity * vi);

  // Warm bloom: channel energy beyond 1 (a saturated hue can't hold more)
  // spills as warm white, so a hot bar glows white-hot instead of clipping
  // to a flat, oversaturated colour. chroma_hold fades the spill out.
  float peak = max(lin.r, max(lin.g, lin.b));
  float spill = max(peak - 1.0, 0.0);
  lin += float3(1.0, 0.85, 0.6) * (spill * (1.0 - chroma_hold));

  // Soft exposure rolloff (1 - e^-x): warm highlight shoulder, no hard clip.
  // Per-channel it washes hot colours to white (each channel saturates
  // toward 1 on its own). The held variant maps only the peak channel
  // through the curve and scales rgb by that one ratio, so the r:g:b ratio
  // — hue AND saturation — survives any drive. chroma_hold blends between
  // them: 0 = classic white-hot, 1 = fully hue-preserving limiter.
  float3 washed = 1.0 - exp(-lin);
  float pk = max(lin.r, max(lin.g, lin.b));
  float3 held = (pk > 1e-6) ? lin * ((1.0 - exp(-pk)) / pk) : float3(0.0, 0.0, 0.0);
  float3 shaped = lerp(washed, held, chroma_hold);

  // input_opacity fades the passed-through input (0 → render bars on black).
  float3 outc = base.rgb * input_opacity + shaped;
  outputTex[gid.xy] = float4(saturate(outc), base.a);
}
