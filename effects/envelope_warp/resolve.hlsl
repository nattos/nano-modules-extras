// warp.envelope — resolve: compose the 1D coordinate maps and sample the input
// ONCE per output pixel.
//
// Axis modes: the X and Y warps are separable and independent, so the two maps
// compose coordinate-wise — src_uv = (mapX[x].r, mapY[y].r) — and the image is
// resampled a single time (no intermediate scratch image, no double blur).
// Each map is read at the pixel's own column/row texel: the rasterized value
// there IS the exact analytic source coordinate for this pixel.
//
// Radial: the pixel's radius (1 = half the LONGER viewport axis, from the
// authored center) samples the radial map with a manual two-tap lerp — the
// rgba32float map binds as unfilterable-float on WebGPU, so no sampler may
// filter it. The source position scales the center offset by src_r / r. Map R
// rides premultiplied by coverage G, so the lerp across a coverage edge
// reconstructs the radius as R/G.

Texture2D<float4>   inputTex : register(t0);
SamplerState        samp     : register(s1);
Texture2D<float4>   mapX     : register(t2);   // X-axis map, or the radial map
Texture2D<float4>   mapY     : register(t3);
RWTexture2D<float4> outTex   : register(u4);

cbuffer Uniforms : register(b5) {
  float radial;     // 0 = axis mode, 1 = radial
  float warp_x;     // axis: read mapX for u (else identity)
  float warp_y;     // axis: read mapY for v (else identity)
  float dom;        // radial map domain length (radius units)
  float center_x;   // radial center, pixels
  float center_y;
  float r_scale;    // pixels -> radius units: 2 / max(W, H)
  float _pad0;
};

[numthreads(8, 8, 1)]
void main(uint3 gid : SV_DispatchThreadID) {
  uint w, h;
  outTex.GetDimensions(w, h);
  if (gid.x >= w || gid.y >= h) return;
  float2 vp = float2(w, h);
  float2 p  = float2(gid.xy) + 0.5;

  float2 src_uv;
  bool covered = true;

  if (radial > 0.5) {
    float2 c   = float2(center_x, center_y);
    float2 off = p - c;
    float  r   = length(off) * r_scale;
    uint mw, mh;
    mapX.GetDimensions(mw, mh);
    float tx = clamp(r / dom * (float)mw - 0.5, 0.0, (float)mw - 1.0);
    uint  i0 = (uint)tx;
    uint  i1 = min(i0 + 1u, mw - 1u);
    float4 m = lerp(mapX[uint2(i0, 0)], mapX[uint2(i1, 0)], frac(tx));
    if (m.g < 0.5) {
      covered = false;
      src_uv = p / vp;
    } else {
      float src_r = m.r / m.g;
      float scale = (r > 1e-6) ? (src_r / r) : 1.0;
      src_uv = (c + off * scale) / vp;
    }
  } else {
    float2 s = p / vp;
    if (warp_x > 0.5) {
      float4 mx = mapX[uint2(gid.x, 0)];
      if (mx.g < 0.5) covered = false;
      s.x = mx.r;
    }
    if (warp_y > 0.5) {
      float4 my = mapY[uint2(gid.y, 0)];
      if (my.g < 0.5) covered = false;
      s.y = my.r;
    }
    src_uv = s;
  }

  float4 col = covered ? inputTex.SampleLevel(samp, src_uv, 0.0)
                       : float4(0.0, 0.0, 0.0, 0.0);
  outTex[gid.xy] = col;
}
