// warp.legacy.freeze_pulse — frame capture pass.
//
// Copies the current live frame into the per-instance "frozen" texture on a
// trigger, so the pulse pass can stutter the held frame while the live video
// keeps moving underneath. A compute copy (not gpu::copy) for web portability
// — the executor's mid-chain intermediates aren't COPY_DST.

Texture2D<float4>   inputTex  : register(t0);
RWTexture2D<float4> outputTex : register(u1);

[numthreads(8, 8, 1)]
void main(uint3 gid : SV_DispatchThreadID) {
  uint W, H;
  outputTex.GetDimensions(W, H);
  if (gid.x >= W || gid.y >= H) return;
  outputTex[gid.xy] = inputTex[gid.xy];
}
