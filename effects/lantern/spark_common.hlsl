// source.mesh.lantern — the flakes' fragment: the EWA splat's coverage.

#ifndef LANTERN_SPARK_COMMON_HLSL
#define LANTERN_SPARK_COMMON_HLSL

// The fragment's offset from the flake's center rides as an interpolated
// varying (`off`) rather than being read off SV_Position, whose y origin is
// not the same on every backend.
struct SparkOut {
  float4 pos : SV_Position;
  float2 off : TEXCOORD4;                   // pixels from the center (y down)
  nointerpolation float4 ctr : TEXCOORD0;
  nointerpolation float4 q   : TEXCOORD1;
  nointerpolation float4 col : TEXCOORD2;
  nointerpolation float2 mv  : TEXCOORD3;
};

// How much of this pixel the flake covers (0..1).
float ltn_spark_cov(SparkOut i) {
  float2 d = i.off;
  float e = i.q.x * d.x * d.x + 2.0 * i.q.y * d.x * d.y + i.q.z * d.y * d.y;
  return min(i.q.w * exp(-0.5 * e), 1.0);
}

#endif  // LANTERN_SPARK_COMMON_HLSL
