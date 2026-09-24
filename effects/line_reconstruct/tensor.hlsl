// filter.reconstruct.line — pass 3c: structure tensor → coherence + junction.
// Eigen-decomposes the sigma-1.5-blurred gradient products. Emits kappa (tensor
// coherence) and the junction measure (both eigenvalues large relative to local
// contrast). Orientation is re-derived from the Hessian in pass 4, so cos2g/sin2g
// aren't needed downstream. (Port of pipeline.pass_tensor eigen half.)

#include "common.hlsl"

Texture2D<float4>   jblur    : register(t0);   // (Jxx, Jxy, Jyy) blurred at sigma 1.5
Texture2D<float4>   cstarTex : register(t1);   // (c*, -, -, -)
RWTexture2D<float4> tensor   : register(u2);   // (kappa, junction, 0, 0)

[numthreads(8, 8, 1)]
void main(uint3 gid : SV_DispatchThreadID) {
  uint w, h;
  tensor.GetDimensions(w, h);
  if (gid.x >= w || gid.y >= h) return;

  float3 j = jblur[gid.xy].rgb;
  float jxx = j.x, jxy = j.y, jyy = j.z;
  float tr  = jxx + jyy;
  float d   = sqrt((jxx - jyy) * (jxx - jyy) + 4.0 * jxy * jxy + 1e-20);
  float l1  = 0.5 * (tr + d);
  float l2  = max(0.5 * (tr - d), 0.0);
  float kappa = d / (tr + 1e-9);

  float cs = cstarTex[gid.xy].r;
  float jn = lr_smoothstep(J0, J1, l2 / (l1 + 1e-9))
           * lr_smoothstep(0.05, 0.15, sqrt(l2) / cs);

  tensor[gid.xy] = float4(kappa, jn, 0.0, 0.0);
}
