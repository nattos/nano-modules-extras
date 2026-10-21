// source.mesh.lantern — the flakes' presence grid -> the dust the fog reads.
//
// p_update splats each flake into `gcount` (fixed point); here each cell's
// count becomes flakes per cubic metre, eased into a running average (so a
// fast flake leaves a fading wake of dust behind it), and the count is
// cleared for the next frame. One thread per cell.

#include "particles.hlsl"

RWStructuredBuffer<uint>  gcount   : register(u3);
RWStructuredBuffer<float> gaccum   : register(u5);
RWTexture3D<float4>       presence : register(u6);

[numthreads(4, 4, 4)]
void main(uint3 gid : SV_DispatchThreadID) {
  uint res = (uint)grid.y;
  if (any(gid >= uint3(res, res, res))) return;
  uint idx = (gid.z * res + gid.y) * res + gid.x;
  float cell = 2.0 * grid.x / float(res);
  float dens = float(gcount[idx]) / (grid.w * cell * cell * cell);
  gcount[idx] = 0u;
  float acc = gaccum[idx];
  if (acc != acc) acc = 0.0;
  acc = lerp(acc, dens, grid.z);
  gaccum[idx] = acc;
  presence[gid] = float4(acc, 0.0, 0.0, 0.0);
}
