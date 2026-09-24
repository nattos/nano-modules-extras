// filter.light.flicker_grid — Pass 3: paint the gated grid.
//
// Per pixel: find the hosting cell, take its flat box-averaged color
// (one flat color per cell — the LED semantics), shape it in HSL —
//   * neutral pull: drag lightness toward 0.5 so the temporal pattern, not
//     the pixel luma, carries the brightness (avoid double-applying luma);
//   * column leveling: lift lightness toward the column's max on a curve
//     whose weight fades to 0 near black, so leveling brightens dim colour
//     without inventing light from nothing (lift-only, never darkens) —
// then multiply by the column's gate: 1 when the pulse is on, the fill
// amount when off (0 unless overflow fill is engaged), so a dark column
// renders plain black (LED off).

#include "nano_color.hlsl"   // nano_rgb_to_hsl / nano_hsl_to_rgb
#include "common.hlsl"

Texture2D<float4> inputTex : register(t0);
RWTexture2D<float4> outputTex : register(u1);

cbuffer RenderU : register(b2) {
  int cols; int rows; float neutral_pull; float level_strength;
};

StructuredBuffer<CellStat> stats : register(t3);
StructuredBuffer<ColState> colstate : register(t4);

static const float LEVEL_GAMMA = 0.5;

[numthreads(8, 8, 1)]
void main(uint3 gid : SV_DispatchThreadID) {
  uint W, H;
  outputTex.GetDimensions(W, H);
  if (gid.x >= W || gid.y >= H) return;

  float2 uv = (float2(gid.xy) + 0.5) / float2(W, H);
  uint col = min(uint(uv.x * float(cols)), uint(max(cols - 1, 0)));
  uint row = min(uint(uv.y * float(rows)), uint(max(rows - 1, 0)));

  CellStat st = stats[col * FG_MAX_ROWS + row];
  float3 c = saturate(float3(st.avg_r, st.avg_g, st.avg_b));
  float3 hsl = nano_rgb_to_hsl(c);
  hsl.z = lerp(hsl.z, 0.5, saturate(neutral_pull));

  ColState cs = colstate[col];
  float target = saturate(cs.level_target);
  if (hsl.z < target) {
    float w = saturate(level_strength) * pow(saturate(hsl.z / max(target, 1e-4)), LEVEL_GAMMA);
    hsl.z = lerp(hsl.z, target, w);
  }

  float3 rgb = nano_hsl_to_rgb(hsl);
  float b = (cs.gate > 0.5) ? 1.0 : saturate(cs.fill);
  outputTex[gid.xy] = float4(rgb * b, inputTex[gid.xy].a);
}
