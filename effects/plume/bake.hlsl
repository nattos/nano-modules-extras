// source.sdf.plume — bake: shell map -> tier-0 cartesian SDF volume.
//
// Per voxel: radial distance to the displaced sphere
//   d = |p| − R − h(normalize(p))
// compressed by a Lipschitz factor (a radial displacement field's true
// distance bound shrinks with the angular slope of h; the CPU derives the
// factor from ridge amplitude/frequency) so trilinear sphere-tracing never
// overshoots. Channels: .r distance (world units), .g soft density
// (1 inside → 0 outside over a small band — the GI/fog passes read this),
// .b crest emphasis carried from the shell, .a spare.

#include "common.hlsl"

Texture2D<float4>   shellCoarse : register(t0);
SamplerState        linearSamp  : register(s1);
RWTexture3D<float4> sdfVol      : register(u2);

cbuffer BakeUniforms : register(b3) {
  float radius;      // base sphere radius, world units
  float lipschitz;   // distance compression, (0, 1]
  float dens_soft;   // half-width of the density band, world units
  float _pad0;
};

[numthreads(4, 4, 4)]
void main(uint3 gid : SV_DispatchThreadID) {
  if (gid.x >= (uint)PLM_VOL_RES || gid.y >= (uint)PLM_VOL_RES ||
      gid.z >= (uint)PLM_VOL_RES) return;

  float3 p = plm_voxel_to_world(int3(gid));
  float r = length(p);
  float3 dir = r > 1e-5 ? p / r : float3(0.0, 1.0, 0.0);

  float2 uv = nano_oct_encode(dir);
  float2 sh = shellCoarse.SampleLevel(linearSamp, uv, 0).rg;

  float d = (r - radius - sh.x) * lipschitz;
  float dens = saturate(0.5 - d / max(dens_soft, 1e-4));

  sdfVol[gid] = float4(d, dens, sh.y, 0.0);
}
