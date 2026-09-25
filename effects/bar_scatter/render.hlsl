// filter.light.bar_scatter — Pass 3: composite over the input.
//
// Each pixel moves by its cell's change, out = in + (proc − live): a cell the
// sim left alone passes through EXACTLY, and whatever detail sits inside a
// cell survives a choke or a delay.

#include "common.hlsl"

Texture2D<float4>   inputTex  : register(t0);
RWTexture2D<float4> outputTex : register(u1);

cbuffer RenderU : register(b2) {
  int rows; int rpad0; int rpad1; int rpad2;
};

StructuredBuffer<CellStat> stats : register(t3);
StructuredBuffer<float4>   proc  : register(t4);

[numthreads(8, 8, 1)]
void main(uint3 gid : SV_DispatchThreadID) {
  uint w, h;
  outputTex.GetDimensions(w, h);
  if (gid.x >= w || gid.y >= h) return;

  float2 uv = (float2(gid.xy) + 0.5) / float2(w, h);
  float4 base = inputTex[gid.xy];
  int cell = bs_cell_of(uv, clamp(rows, 1, BS_MAX_ROWS));
  CellStat live = stats[cell];
  float3 d = proc[cell].rgb - float3(live.r, live.g, live.b);
  outputTex[gid.xy] = float4(max(base.rgb + d, 0.0), base.a);
}
