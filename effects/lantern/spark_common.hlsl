// source.mesh.lantern — the flakes' fragment: an EMBER's coverage — a capsule
// from the flake's tail (where it was `streak` frames ago) to its head,
// tapering and dimming toward the tail, with a hard anti-aliased edge.

#ifndef LANTERN_SPARK_COMMON_HLSL
#define LANTERN_SPARK_COMMON_HLSL

// The fragment's position in the ember's own frame rides as an interpolated
// varying (`off`) rather than being read off SV_Position, whose y origin is
// not the same on every backend.
struct SparkOut {
  float4 pos : SV_Position;
  float2 off : TEXCOORD4;                   // px: along (tail 0 -> head len), across
  nointerpolation float4 shp : TEXCOORD0;   // length, half-width, peak coverage, coverage scale
  nointerpolation float4 col : TEXCOORD2;   // radiance, 0
  nointerpolation float2 mv  : TEXCOORD3;   // motion (uv / frame)
};

static const float LTN_TAPER = 0.45;   // the tail's width, of the head's

// How much of this pixel the ember covers (0..1), already dimmed toward its
// tail.
float ltn_spark_cov(SparkOut i) {
  float len = i.shp.x, w = i.shp.y;
  float u = i.off.x, v = i.off.y;
  float t = len > 1e-3 ? saturate(u / len) : 1.0;
  float wt = w * lerp(LTN_TAPER, 1.0, t);
  float du = u - clamp(u, 0.0, len);
  float r = sqrt(du * du + v * v);
  float c = saturate((wt - r) / 0.7 + 0.5);
  return c * i.shp.z * lerp(0.3, 1.0, t * t);
}

#endif  // LANTERN_SPARK_COMMON_HLSL
