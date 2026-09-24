// triangulate — auto-level the raw feature maps by their percentile divisors and
// blend into the balanced importance field W the dynamics consume. ridge and
// corner are each normalized to their own per-frame distribution, so the
// ridge/corner/void weights are a FAIR mix (a rare corner isn't drowned by an
// extended ridge just because the ridge covers more pixels at a bigger raw
// magnitude).
Texture2D<float4>        featRaw : register(t0);   // r=density g=ridge_s b=corner_s
StructuredBuffer<float>  pct     : register(t1);   // [0]=ridge divisor, [1]=corner divisor
RWTexture2D<float4>      featOut : register(u2);   // r=density g=ridge b=corner a=W

cbuffer RemapUniforms : register(b3) {
  float u_ridge_w;
  float u_corner_w;
  float u_void_w;
  float u_pad;
};

[numthreads(8, 8, 1)]
void main(uint3 gid : SV_DispatchThreadID) {
  uint w, h;
  featOut.GetDimensions(w, h);
  if (gid.x >= w || gid.y >= h) return;
  float4 f = featRaw.Load(int3(gid.xy, 0));
  float density    = f.r;
  float ridge_bal  = saturate(f.g / pct[0]);
  float corner_bal = saturate(f.b / pct[1]);
  float wsum = u_ridge_w + u_corner_w + u_void_w;
  float W = (u_ridge_w * ridge_bal + u_corner_w * corner_bal + u_void_w * density) / max(wsum, 1e-3);
  featOut[gid.xy] = float4(density, ridge_bal, corner_bal, W);
}
