// warp.plane_shear — accumulate pass.
//
// One thread per coarse-grid sample. Computes the local cover-space luma
// gradient once, then scatters (via fixed-point atomic add) into the stats
// buffer whatever the SELECTED algorithm needs. Only the active algorithm's
// accumulation runs (a uniform branch), so an update is cheap. Runs only on
// plane-update frames (host-gated).

#include "common.hlsl"

Texture2D<float4>       inputTex : register(t0);
SamplerState            samp     : register(s1);
RWStructuredBuffer<int> stats    : register(u2);

cbuffer U : register(b3) {
  float res_x, res_y;      // viewport resolution
  float algorithm;         // PS_ALG_*
  float off_max;           // grid offset half-range (cover-square units)
  float aspect_x, aspect_y;
  float _pad0, _pad1;
};

void addFixed(int slot, float v, float scale) {
  int prev;
  InterlockedAdd(stats[slot], (int)round(v * scale), prev);
}

[numthreads(8, 8, 1)]
void main(uint3 gid : SV_DispatchThreadID) {
  if (gid.x >= PS_GRID_SN || gid.y >= PS_GRID_SN) return;

  float2 res    = float2(res_x, res_y);
  float2 aspect = float2(aspect_x, aspect_y);
  float2 uv = (float2(gid.xy) + 0.5) / float(PS_GRID_SN);
  float2 p  = nano_uv_to_cover_square(uv, aspect);   // cover-square position

  int alg = (int)(algorithm + 0.5);

  if (alg == PS_ALG_PCA) {
    // Luma-weighted moments for a principal-axis split.
    float w = max(nano_luminance(inputTex.SampleLevel(samp, uv, 0).rgb), 0.0);
    addFixed(PS_M_W,   w,           PS_SCALE_MOM);
    addFixed(PS_M_WX,  w * p.x,     PS_SCALE_MOM);
    addFixed(PS_M_WY,  w * p.y,     PS_SCALE_MOM);
    addFixed(PS_M_WXX, w * p.x*p.x, PS_SCALE_MOM);
    addFixed(PS_M_WXY, w * p.x*p.y, PS_SCALE_MOM);
    addFixed(PS_M_WYY, w * p.y*p.y, PS_SCALE_MOM);
    return;
  }

  float2 g   = ps_grad_cover(inputTex, samp, uv, res, aspect);
  float  mag = length(g);

  if (alg == PS_ALG_DOMINANT) {
    // Global structure tensor + gradient-energy-weighted centroid.
    addFixed(PS_M_JXX, g.x * g.x, PS_SCALE_MOM);
    addFixed(PS_M_JXY, g.x * g.y, PS_SCALE_MOM);
    addFixed(PS_M_JYY, g.y * g.y, PS_SCALE_MOM);
    addFixed(PS_M_GW,  mag,        PS_SCALE_MOM);
    addFixed(PS_M_GWX, mag * p.x,  PS_SCALE_MOM);
    addFixed(PS_M_GWY, mag * p.y,  PS_SCALE_MOM);
    return;
  }

  if (alg == PS_ALG_HOUGH) {
    // Vote the sample's edge line (normal = gradient dir) into (theta, rho).
    if (mag < 1e-5) return;
    float2 n = g / mag;
    float theta = atan2(n.y, n.x);
    float rho   = dot(p, n);
    ps_fold_line(theta, rho);
    int ai = clamp((int)(theta / PS_PI * PS_NA), 0, PS_NA - 1);
    int oi = clamp((int)((rho / off_max * 0.5 + 0.5) * PS_NO), 0, PS_NO - 1);
    addFixed(PS_GRID_BASE + ai * PS_NO + oi, mag, PS_SCALE_E);
    return;
  }

  // PS_ALG_SEAM — bin this sample's edge energy into EVERY candidate cut angle
  // at its offset, so a later argmin finds the lowest-energy straight cut.
  [loop] for (int a = 0; a < PS_NA; ++a) {
    float psi = (a + 0.5) / PS_NA * PS_PI;
    float2 m  = float2(cos(psi), sin(psi));
    float rho = dot(p, m);
    int oi = clamp((int)((rho / off_max * 0.5 + 0.5) * PS_NO), 0, PS_NO - 1);
    addFixed(PS_GRID_BASE + a * PS_NO + oi, mag, PS_SCALE_E);
  }
}
