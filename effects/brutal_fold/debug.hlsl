// source.brutal_fold — skip-empty debug visualizer.
//
// Overwrites tex_out with a per-tile heatmap of one of the flatness features
// (variance / edge / motion / combined), so the weights can be tuned by eye. Reads
// the per-tile stats the edge pass just wrote (same submit) and replicates the CPU
// reduce math per tile — no tex_out read, so no read/write hazard. Only dispatched
// when the debug mode is non-zero (a tuning aid, not the normal path).

cbuffer DU : register(b0) {
  float res_x, res_y, dbg_mode, _pad0;   // dbg_mode: 1=var 2=edge 3=motion 4=combined
  float w_var, w_edge, w_motion, _pad1;  // feature weights (for the combined view)
};
StructuredBuffer<int> stats  : register(t1);   // per-tile [edge,luma,luma2,motion,count]
RWTexture2D<float4>   outTex : register(u2);

// Must match main.cpp / edge.hlsl.
static const int   kTileGrid     = 16;
static const int   kSlots        = 5;
static const float kStatsScale   = 65536.0;
static const float kEdgeNormGain = 2.0;
static const float kVarFloor     = 0.008;

[numthreads(8, 8, 1)]
void main(uint3 gid : SV_DispatchThreadID) {
  int2 res = int2((int)res_x, (int)res_y);
  if ((int)gid.x >= res.x || (int)gid.y >= res.y) return;

  int2 tile = clamp(int2(gid.xy) * kTileGrid / res, int2(0, 0), int2(kTileGrid - 1, kTileGrid - 1));
  int ti = (tile.y * kTileGrid + tile.x) * kSlots;

  int cnt = stats[ti + 4];
  float v = 0.0;
  if (cnt > 0) {
    float ne = (float)cnt;
    float lu = ((float)stats[ti + 1] / kStatsScale) / ne;
    float lq = ((float)stats[ti + 2] / kStatsScale) / ne;
    float var = max(lq - lu * lu, 0.0);
    float sd = sqrt(var); sd = max(sd - kVarFloor, 0.0);
    float edge   = saturate((float)stats[ti + 0] / kStatsScale / sqrt(ne) * kEdgeNormGain);
    float motion = saturate((float)stats[ti + 3] / kStatsScale / sqrt(ne) * kEdgeNormGain);
    int mode = (int)(dbg_mode + 0.5);
    if      (mode == 1) v = sd;
    else if (mode == 2) v = edge;
    else if (mode == 3) v = motion;
    else                v = max(sd * w_var, max(edge * w_edge, motion * w_motion));
  }
  outTex[gid.xy] = float4(v, v, v, 1.0);
}
