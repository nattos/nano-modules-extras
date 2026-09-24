// source.mesh.monolith — pre-fill compute pass.
//
// Copies tex_in into tex_out so the raster pass can alpha-blend the
// shape on top. A compute copy (not gpu::Device::copy) because the
// web executor's mid-chain intermediates are not COPY_DST. For an
// unwired input the host binds a 1x1 zero fallback; out-of-bounds
// Loads return zero per WebGPU spec, so the unconditional read is
// safe (degenerates to a clear).

Texture2D<float4>   srcTex : register(t0);
RWTexture2D<float4> dstTex : register(u1);

[numthreads(8, 8, 1)]
void main(uint3 gid : SV_DispatchThreadID) {
  uint W, H;
  dstTex.GetDimensions(W, H);
  if (gid.x >= W || gid.y >= H) return;
  dstTex[gid.xy] = srcTex.Load(int3(int(gid.x), int(gid.y), 0));
}
