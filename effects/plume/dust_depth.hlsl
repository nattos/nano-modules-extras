// source.sdf.plume — dust splat pass 1: per-pixel nearest-particle depth
// + aggregate coverage.
//
// One thread per particle: project, then over the feathered footprint
// InterlockedMin(asuint(t)) — positive-float bit patterns order like the
// floats, so the uint min IS the depth min — and InterlockedAdd the
// pixel's coverage (8.8 fixed point). The winner supplies the COLOR
// (pass 2); the coverage sum supplies the ALPHA (composite): two
// half-covering specks on one pixel build real opacity instead of the
// farther one vanishing. Surface occlusion is resolved HERE (particles
// behind the surface contribute neither depth nor coverage), so a
// winning entry is a visible entry and the sum only counts visible dust.

#define DUST_UB_REG b3
#include "dust_common.hlsl"

Texture2D<float4>        sceneTex : register(t1);   // .a = surface hit t
RWStructuredBuffer<uint> depthBuf : register(u2);
RWStructuredBuffer<uint> covBuf   : register(u4);   // 8.8 coverage sum

[numthreads(64, 1, 1)]
void main(uint3 gid : SV_DispatchThreadID) {
  if (gid.x >= (uint)cam_p.w) return;
  float3 pos, nrm;
  float seed, t, rp;
  float2 ctr;
  if (!du_project(gid.x, pos, nrm, seed, ctr, t, rp)) return;

  int W = (int)vp.x, H = (int)vp.y;
  float reach = rp + DUST_FEATHER;
  int x0 = max((int)floor(ctr.x - reach), 0);
  int x1 = min((int)floor(ctr.x + reach), W - 1);
  int y0 = max((int)floor(ctr.y - reach), 0);
  int y1 = min((int)floor(ctr.y + reach), H - 1);
  uint du = asuint(t);
  for (int py = y0; py <= y1; py++) {
    for (int px = x0; px <= x1; px++) {
      float2 d = float2(px, py) + 0.5 - ctr;
      uint cq = (uint)(du_cov(length(d), rp) * 256.0 + 0.5);
      if (cq == 0u) continue;
      if (t >= sceneTex.Load(int3(px, py, 0)).a) continue;
      uint prev;
      InterlockedMin(depthBuf[py * W + px], du, prev);
      InterlockedAdd(covBuf[py * W + px], cq, prev);
    }
  }
}
