// filter.light.bar_scatter — Pass 1: per-cell input reduction + history.
//
// One thread per cell (guarded 8×8 dispatch, x = bar, y = row). Box-samples
// its cell on a fixed SAMPLE_N² Load grid → mean colour + luma, and files
// the colour in this frame's history slot for the delay to read back.

#include "nano_color.hlsl"   // nano_luminance
#include "common.hlsl"

Texture2D<float4>          inputTex : register(t0);
RWStructuredBuffer<CellStat> stats  : register(u1);
RWStructuredBuffer<float4>   hist   : register(u2);   // BS_HIST × BS_MAX_CELLS

cbuffer ReduceU : register(b3) {
  int rows; int tex_w; int tex_h; int head;
};

static const int SAMPLE_N = 8;

[numthreads(8, 8, 1)]
void main(uint3 gid : SV_DispatchThreadID) {
  if (gid.x >= (uint)BS_BARS || gid.y >= (uint)rows) return;
  int bar = int(gid.x), row = int(gid.y);
  float2 o = bs_cell_origin(bar, row, rows), sz = bs_cell_size(rows);

  float3 sum_rgb = float3(0.0, 0.0, 0.0);
  float sum_luma = 0.0;
  [loop] for (int sy = 0; sy < SAMPLE_N; sy++) {
    for (int sx = 0; sx < SAMPLE_N; sx++) {
      float2 uv = o + (float2(sx, sy) + 0.5) / float(SAMPLE_N) * sz;
      int px = clamp(int(uv.x * float(tex_w)), 0, tex_w - 1);
      int py = clamp(int(uv.y * float(tex_h)), 0, tex_h - 1);
      float3 rgb = inputTex.Load(int3(px, py, 0)).rgb;
      sum_rgb += rgb;
      sum_luma += nano_luminance(rgb);
    }
  }
  float inv = 1.0 / float(SAMPLE_N * SAMPLE_N);
  int cell = row * BS_BARS + bar;
  CellStat cs;
  cs.r = sum_rgb.r * inv; cs.g = sum_rgb.g * inv; cs.b = sum_rgb.b * inv;
  cs.luma = sum_luma * inv;
  stats[cell] = cs;
  hist[clamp(head, 0, BS_HIST - 1) * BS_MAX_CELLS + cell] = float4(cs.r, cs.g, cs.b, 1.0);
}
