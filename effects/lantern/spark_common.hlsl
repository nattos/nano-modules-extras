// source.mesh.lantern — the flakes' fragment: an EMBER's coverage — a capsule
// from the flake's tail (where it was `streak` frames ago) to its head, bent
// at the corner it turned on the way (two capsules joined there), tapering
// and dimming toward the tail, with a hard anti-aliased edge.

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
  nointerpolation float2 kink : TEXCOORD5;  // the corner, in the same frame as `off`
};

static const float LTN_TAPER = 0.45;   // the tail's width, of the head's

// One leg, from a to b, a fraction t0..t1 of the way from tail to head.
float ltn_spark_leg(float2 q, float2 a, float2 b, float t0, float t1, float w) {
  float2 ab = b - a;
  float l2 = dot(ab, ab);
  float h = l2 > 1e-6 ? saturate(dot(q - a, ab) / l2) : 0.0;
  float t = lerp(t0, t1, h);
  float r = length(q - a - ab * h);
  float c = saturate((w * lerp(LTN_TAPER, 1.0, t) - r) / 0.7 + 0.5);
  return c * lerp(0.3, 1.0, t * t);
}

// How much of this pixel the ember covers (0..1), already dimmed toward its
// tail.
float ltn_spark_cov(SparkOut i) {
  float len = i.shp.x, w = i.shp.y;
  float2 k = i.kink, hd = float2(len, 0.0);
  float la = length(k), lb = length(hd - k);
  float L = la + lb;
  // (A flake that has not moved is all head.)
  float tk = L > 1e-3 ? la / L : 1.0, t0 = L > 1e-3 ? 0.0 : 1.0;
  float c = max(ltn_spark_leg(i.off, float2(0.0, 0.0), k, t0, tk, w),
                ltn_spark_leg(i.off, k, hd, tk, 1.0, w));
  return c * i.shp.z;
}

#endif  // LANTERN_SPARK_COMMON_HLSL
