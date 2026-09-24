// warp.envelope — map raster fragment: exact analytic inverse of the segment's
// exponential ease.
//
// Within a segment the forward map is
//   d = d0 + t^e * (d1 - d0),   t = (s - s0) / (s1 - s0),   e = 2^(-3*ease)
// so given the interpolated destination fraction td = (d - d0) / (d1 - d0) the
// source coordinate is
//   s = s0 + td^(1/e) * (s1 - s0)
// with inv_e = 1/e = 2^(3*ease) carried per instance.
//
// Output texel: R = source coordinate (map-domain units), G = coverage. The
// map clears to 0, so texels no segment covers stay G = 0 — the resolve pass
// treats those as transparent. R rides premultiplied by coverage so a linear
// sample across a coverage boundary reconstructs the source value as R/G.

struct VsOut {
  float4 pos : SV_Position;
  float  td  : TEXCOORD0;
  nointerpolation float3 seg : TEXCOORD1;   // s0, s1, inv_e
};

[shader("pixel")]
float4 main(VsOut i) : SV_Target {
  float t = pow(saturate(i.td), i.seg.z);
  float src = lerp(i.seg.x, i.seg.y, t);
  return float4(src, 1.0, 0.0, 1.0);
}
