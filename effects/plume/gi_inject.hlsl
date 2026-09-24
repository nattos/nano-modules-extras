// source.sdf.plume — GI light injection (the wave equation's source term).
//
// Per GI voxel: sample the SDF grid's density band; where the voxel sits
// ON the surface shell (dens rising 0→1 across the band), deposit the
// shadowed direct sunlight the surface reflects there — albedo × sun ×
// a short shadow march through the SDF grid. The band weight
// dens·(1−dens)·4 peaks AT the surface and vanishes both in open air and
// deep inside the body, so light enters the wave field exactly where a
// bounce is born. The propagation pass then carries it into the air (and
// the fog, later) with momentum.

#include "common.hlsl"

Texture3D<float4>   sdfVol     : register(t0);
SamplerState        linearSamp : register(s1);
RWTexture3D<float4> injectVol  : register(u2);

cbuffer InjectUniforms : register(b3) {
  float4 sun_p;      // sun dir (world, toward light), w = intensity
  float4 albedo;     // rgb bounce color, w = inv_lip (grid decompression)
};

float plmg_sdf(float3 p) {
  return sdfVol.SampleLevel(linearSamp, plm_world_to_uvw(p), 0).r;
}

[numthreads(4, 4, 4)]
void main(uint3 gid : SV_DispatchThreadID) {
  if (gid.x >= (uint)PLM_GI_RES || gid.y >= (uint)PLM_GI_RES ||
      gid.z >= (uint)PLM_GI_RES) return;

  float3 p = ((float3(gid) + 0.5) * (1.0 / float(PLM_GI_RES)) - 0.5)
             * (2.0 * PLM_EXT0);
  float dens = sdfVol.SampleLevel(linearSamp, plm_world_to_uvw(p), 0).g;
  float band = saturate(dens * (1.0 - dens) * 4.0);
  if (band < 0.01) {
    injectVol[gid] = float4(0.0, 0.0, 0.0, 0.0);
    return;
  }

  // Short shadow march toward the sun through the (compressed) SDF grid.
  const float voxel = 2.0 * PLM_EXT0 / float(PLM_VOL_RES);
  float sh = 1.0;
  float st = 3.0 * voxel;
  [loop] for (int m = 0; m < 10; m++) {
    float3 sp = p + sun_p.xyz * st;
    if (abs(sp.x) > PLM_EXT0 || abs(sp.y) > PLM_EXT0 ||
        abs(sp.z) > PLM_EXT0) break;
    float d = plmg_sdf(sp) * albedo.w;
    sh = min(sh, 5.0 * d / st);
    if (sh < 0.02) break;
    st += clamp(d, 1.0 * voxel, 6.0 * voxel);
  }
  sh = saturate(sh);

  float3 e = albedo.rgb * (sun_p.w * sh * band);
  injectVol[gid] = float4(e, band);
}
