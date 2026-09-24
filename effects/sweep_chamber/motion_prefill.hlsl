// source.particles.sweep_chamber — motion-vector prefill.
// Seeds render_outputs/motion with the upstream motion field
// (render_outputs_in) so the particle/line motion passes compose OVER it.
// With nothing wired the host binds a 1×1 zero texture; out-of-bounds loads
// return zero, so this doubles as a clear. double_chamber parity.

Texture2D<float4>   upstreamMotion : register(t0);
RWTexture2D<float4> motionTex      : register(u1);

[numthreads(8, 8, 1)]
void main(uint3 gid : SV_DispatchThreadID) {
  uint w, h;
  motionTex.GetDimensions(w, h);
  if (gid.x >= w || gid.y >= h) return;
  motionTex[gid.xy] = float4(upstreamMotion[gid.xy].xy, 0.0, 0.0);
}
