// source.brutal_fold — present: composite the receding prism layers to grayscale.
//
// Single compute pass, one thread per output pixel. The square [-1,1]² field
// COVERS the (possibly non-square) viewport uniformly (long axis spans the frame,
// short axis cropped — no bars), then bf_fieldVal composites both structures'
// depth layers with fog. Output is grayscale (the fog already lifts distant
// layers toward the light sky tone). No auto-levels: the solid threshold `level`
// is resolved on the CPU (build_params) and baked into P.

#include "common.hlsl"

RWTexture2D<float4> tex_out : register(u1);

[numthreads(8, 8, 1)]
void main(uint3 gid : SV_DispatchThreadID) {
  float2 vp = float2(res_x, res_y);
  if (gid.x >= (uint)res_x || gid.y >= (uint)res_y) return;

  // Cover the viewport: uniform scale by the LONG axis so the square fills the
  // frame (short axis cropped) with no bars. The field is periodic, so the
  // cropped overflow is just more pattern.
  float mx = max(vp.x, vp.y);
  float2 p0 = (float2(gid.xy) + 0.5 - 0.5 * vp) / (0.5 * mx);

  float3 col = bf_fieldVal(p0);
  tex_out[gid.xy] = float4(col, 1.0);
}
