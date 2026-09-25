// filter.light.bounce_resonator — Pass 3: composite over the input.
//
// Each pixel reads its cell's A/C output d = intensity·out:
//   d > 0 — light: the cell's resonating colour, shaped by the same warm
//           bloom / chroma-hold limiter as source.light.bounce_resonator,
//           added over the image;
//   d < 0 — the overshoot into black: the dry image is choked by
//           exp(-black_depth·|d|), so a swinging cell strobes bright/black.

#include "nano_color.hlsl"
#include "common.hlsl"

Texture2D<float4>   inputTex  : register(t0);
RWTexture2D<float4> outputTex : register(u1);

cbuffer RenderU : register(b2) {
  float band_val; float intensity; float input_opacity; float chroma_hold;
  float black_depth; int rows; float rpad0; float rpad1;
};

StructuredBuffer<CellState> cells : register(t3);

[numthreads(8, 8, 1)]
void main(uint3 gid : SV_DispatchThreadID) {
  uint w, h;
  outputTex.GetDimensions(w, h);
  if (gid.x >= w || gid.y >= h) return;

  float2 uv = (float2(gid.xy) + 0.5) / float2(w, h);
  float4 base = inputTex[gid.xy];
  CellState c = cells[br_cell_of(uv, clamp(rows, 1, BR_MAX_ROWS))];
  float d = intensity * c.swing;

  float3 col = nano_hsv_to_rgb(float3(c.h, c.s, band_val));
  float3 lin = col * max(d, 0.0);

  // Warm bloom: energy past what a saturated hue can hold spills as warm
  // white; chroma_hold fades the spill and blends to a hue-preserving
  // limiter (see source.light.bounce_resonator's color.hlsl).
  float peak = max(lin.r, max(lin.g, lin.b));
  lin += float3(1.0, 0.85, 0.6) * (max(peak - 1.0, 0.0) * (1.0 - chroma_hold));
  float3 washed = 1.0 - exp(-lin);
  float pk = max(lin.r, max(lin.g, lin.b));
  float3 held = (pk > 1e-6) ? lin * ((1.0 - exp(-pk)) / pk) : float3(0.0, 0.0, 0.0);
  float3 shaped = lerp(washed, held, chroma_hold);

  float choke = exp(-black_depth * max(-d, 0.0));
  float3 outc = base.rgb * (input_opacity * choke) + shaped;
  outputTex[gid.xy] = float4(saturate(outc), base.a);
}
