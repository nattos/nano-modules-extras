// color.legacy.lut_collection — pass 1: upload a baked LUT cube into a 3D
// texture. Run once per preset at bake time (first frame). Reads the preset's
// RGBA8 bytes from a storage buffer (one uint per texel, little-endian
// R|G<<8|B<<16|A<<24 — exactly the byte order in lut_data.h) and writes them
// into a texture_storage_3d<rgba8unorm, write>. The format is supplied at
// registerShaderSPV time so the naga bridge emits the right storage type on
// web; native takes it from the bound texture.
RWTexture3D<float4>    lut  : register(u0);
StructuredBuffer<uint> data : register(t1);

// LUT side length (matches lut_data.h LUT_DIM). Querying a write-access 3D
// storage texture's dimensions isn't portable, so it's a compile-time const
// and the dispatch covers exactly N^3 threads.
static const uint N = 64u;

[numthreads(4, 4, 4)]
void main(uint3 gid : SV_DispatchThreadID) {
  if (gid.x >= N || gid.y >= N || gid.z >= N) return;
  uint idx = gid.x + gid.y * N + gid.z * N * N;
  uint v = data[idx];
  float4 c = float4(float(v & 0xffu),
                    float((v >> 8) & 0xffu),
                    float((v >> 16) & 0xffu),
                    float((v >> 24) & 0xffu)) / 255.0;
  lut[gid] = c;
}
