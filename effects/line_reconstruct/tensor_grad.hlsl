// filter.reconstruct.line — pass 3a: structure-tensor gradient products.
// Scharr gradient (normalized to unit response on a unit ramp, /32) of the
// sigma-1.4 luma level, then the outer products (Jxx, Jxy, Jyy). A separable
// sigma-1.5 blur of these (host-side fx::GaussianBlur) forms the tensor, which
// pass 3c eigen-decomposes. (Port of pipeline.pass_tensor gradient half.)

#include "common.hlsl"

Texture2D<float4>   y1Tex  : register(t0);   // sigma-1.4 luma level (.r)
RWTexture2D<float4> jraw   : register(u1);   // (Jxx, Jxy, Jyy, 0)

[numthreads(8, 8, 1)]
void main(uint3 gid : SV_DispatchThreadID) {
  uint w, h;
  jraw.GetDimensions(w, h);
  if (gid.x >= w || gid.y >= h) return;
  int2 hi = int2(w - 1, h - 1);
  int2 p  = int2(gid.xy);

  // Clamped .r taps. Image coords: x = col, y = row. N = row-1, S = row+1.
  #define YT(cx, cy) (y1Tex[clamp(int2(cx, cy), int2(0, 0), hi)].r)
  float nw = YT(p.x - 1, p.y - 1), n = YT(p.x, p.y - 1), ne = YT(p.x + 1, p.y - 1);
  float ww = YT(p.x - 1, p.y),                          ee = YT(p.x + 1, p.y);
  float sw = YT(p.x - 1, p.y + 1), s = YT(p.x, p.y + 1), se = YT(p.x + 1, p.y + 1);
  #undef YT

  float gx = (3.0 * (ne - nw) + 10.0 * (ee - ww) + 3.0 * (se - sw)) / 32.0;
  float gy = (3.0 * (sw - nw) + 10.0 * (s  - n ) + 3.0 * (se - ne)) / 32.0;

  jraw[gid.xy] = float4(gx * gx, gx * gy, gy * gy, 0.0);
}
