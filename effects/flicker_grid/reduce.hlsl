// filter.light.flicker_grid — Pass 1: per-cell input reduction.
//
// One thread per grid CELL (guarded 8×8 dispatch). Each thread works alone —
// no cross-thread sync — so the native Metal hardcoded-8×8-threadgroup
// constraint is harmless. A thread box-samples its cell with a fixed
// SAMPLE_N × SAMPLE_N Load grid and writes the cell's mean color, mean luma
// and max sample luma into the stats buffer.

#include "nano_color.hlsl"   // nano_luminance
#include "common.hlsl"

Texture2D<float4> inputTex : register(t0);
RWStructuredBuffer<CellStat> stats : register(u1);

cbuffer ReduceU : register(b2) {
  int cols; int rows; int tex_w; int tex_h;
};

static const int SAMPLE_N = 8;

[numthreads(8, 8, 1)]
void main(uint3 gid : SV_DispatchThreadID) {
  if (gid.x >= (uint)cols || gid.y >= (uint)rows) return;
  int cx = int(gid.x), cy = int(gid.y);

  float3 sum_rgb = float3(0.0, 0.0, 0.0);
  float sum_luma = 0.0;
  float max_luma = 0.0;
  [loop] for (int sy = 0; sy < SAMPLE_N; sy++) {
    for (int sx = 0; sx < SAMPLE_N; sx++) {
      float ux = (float(cx) + (float(sx) + 0.5) / float(SAMPLE_N)) / float(cols);
      float uy = (float(cy) + (float(sy) + 0.5) / float(SAMPLE_N)) / float(rows);
      int px = clamp(int(ux * float(tex_w)), 0, tex_w - 1);
      int py = clamp(int(uy * float(tex_h)), 0, tex_h - 1);
      float3 rgb = inputTex.Load(int3(px, py, 0)).rgb;
      sum_rgb += rgb;
      float l = nano_luminance(rgb);
      sum_luma += l;
      max_luma = max(max_luma, l);
    }
  }

  float inv = 1.0 / float(SAMPLE_N * SAMPLE_N);
  CellStat st;
  st.avg_r = sum_rgb.r * inv;
  st.avg_g = sum_rgb.g * inv;
  st.avg_b = sum_rgb.b * inv;
  st.avg_luma = sum_luma * inv;
  st.max_luma = max_luma;
  st.pad0 = 0.0; st.pad1 = 0.0; st.pad2 = 0.0;
  stats[cx * FG_MAX_ROWS + cy] = st;
}
