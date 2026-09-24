// source.sdf.helio_field — dust density fold: baked SDF volume + count
// buffer → the PUBLISHED volume, with .a = normalized dust density.
// (A separate output texture because a storage texture can't be read
// and rewritten portably in one pass; .rgb passes through the bake's
// distance/density/crest untouched.)

#include "../plume/common.hlsl"

Texture3D<float4>        bakedVol : register(t0);
StructuredBuffer<uint>   accum    : register(t1);
RWTexture3D<float4>      outVol   : register(u2);

cbuffer DustFoldUniforms : register(b3) {
  float norm;      // count → density gain (1 at "norm_count" motes/voxel)
  float _p0, _p1, _p2;
};

[numthreads(4, 4, 4)]
void main(uint3 gid : SV_DispatchThreadID) {
  const int RES = PLM_VOL_RES;
  if (gid.x >= (uint)RES || gid.y >= (uint)RES || gid.z >= (uint)RES) return;
  float4 v = bakedVol.Load(int4(int3(gid), 0));
  uint c = accum[(gid.z * RES + gid.y) * RES + gid.x];
  outVol[gid] = float4(v.rgb, saturate(float(c) * norm));
}
