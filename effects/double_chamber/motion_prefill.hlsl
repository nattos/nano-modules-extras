// source.legacy.double_chamber — motion-vector prefill.
// Seeds render_outputs/motion with the upstream motion field (render_outputs_in)
// so the particle/line motion passes compose OVER it. When nothing is wired in
// the host binds a 1x1 zero texture; out-of-bounds loads return zero per the
// WebGPU spec, so this also doubles as a clear-to-zero.

Texture2D<float4>   upstreamMotion : register(t0);
RWTexture2D<float4> motionTex      : register(u1);

[numthreads(8, 8, 1)]
void main(uint3 gid : SV_DispatchThreadID) {
  uint w, h;
  motionTex.GetDimensions(w, h);
  if (gid.x >= w || gid.y >= h) return;
  motionTex[gid.xy] = float4(upstreamMotion[gid.xy].xy, 0.0, 0.0);
}
