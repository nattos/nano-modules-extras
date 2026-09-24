// source.sdf.plume — pre-fill compute pass (monolith parity).
//
// Copies tex_in into tex_out for the idle/passthrough path. A compute copy
// (not gpu::Device::copy) because the web executor's mid-chain
// intermediates are not COPY_DST. Out-of-bounds Loads return zero, so an
// unwired input degenerates to a clear.

Texture2D<float4>   srcTex : register(t0);
RWTexture2D<float4> dstTex : register(u1);

[numthreads(8, 8, 1)]
void main(uint3 gid : SV_DispatchThreadID) {
  uint W, H;
  dstTex.GetDimensions(W, H);
  if (gid.x >= W || gid.y >= H) return;
  dstTex[gid.xy] = srcTex.Load(int3(int(gid.x), int(gid.y), 0));
}
