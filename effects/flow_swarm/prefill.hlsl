// source.particles.flow_swarm — pre-fill pass. Copies tex_in × input_alpha into tex_out
// (rgba8) so the instanced particle raster blends on top of the (optionally
// dimmed) input. Same shape as flash_particles' prefill.

Texture2D<float4>   srcTex : register(t0);
RWTexture2D<float4> dstTex : register(u1);

cbuffer Uniforms : register(b2) {
  float4 scale;   // (input_alpha, input_alpha, input_alpha, 1)
};

[numthreads(8, 8, 1)]
void main(uint3 gid : SV_DispatchThreadID) {
  uint W, H;
  dstTex.GetDimensions(W, H);
  if (gid.x >= W || gid.y >= H) return;
  float4 src = srcTex.Load(int3(int(gid.x), int(gid.y), 0));
  dstTex[gid.xy] = src * scale;
}
