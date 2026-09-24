// filter.glitch.block_dehance — motion-vector pass: pure passthrough of any upstream
// motion (block_dehance generates no motion of its own). Kept so the effect
// is a transparent link in a render_outputs/motion chain.

Texture2D<float4>   upstreamTex : register(t0);
RWTexture2D<float4> motionTex   : register(u1);

[numthreads(8, 8, 1)]
void main(uint3 gid : SV_DispatchThreadID) {
  uint w, h;
  motionTex.GetDimensions(w, h);
  if (gid.x >= w || gid.y >= h) return;
  float4 up = upstreamTex[gid.xy];
  motionTex[gid.xy] = float4(up.xy, 0.0, 0.0);
}
