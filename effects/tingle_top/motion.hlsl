// source.light.tingle_top — motion-vector pass: passthrough of upstream motion (v1).
// TODO: emit per-particle (velocity_x, velocity_y) where the canvas is
// covered by a moving sparkle, so a downstream motion.blur streaks the
// downward_sparkle preset into trails.

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
