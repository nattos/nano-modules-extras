// source.sdf.plume — uint-buffer fill. Two users: the dust splat's
// depth-resolve reset (all ones — InterlockedMin with asuint(t) of any
// positive float beats it) and, re-registered by dust-publishing
// providers, the density-accumulator reset (zeros).

RWStructuredBuffer<uint> depthBuf : register(u0);

cbuffer DustClearUniforms : register(b1) {
  float count;   // element count (exact in float below 2^24)
  float ones;    // fill: > 0.5 = 0xFFFFFFFF, else 0
  float _p0, _p1;
};

[numthreads(64, 1, 1)]
void main(uint3 gid : SV_DispatchThreadID) {
  if (gid.x >= (uint)count) return;
  depthBuf[gid.x] = ones > 0.5 ? 0xFFFFFFFFu : 0u;
}
