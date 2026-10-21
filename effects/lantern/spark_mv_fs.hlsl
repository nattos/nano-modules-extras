// source.mesh.lantern — a flake's fragment into the spark layer AND the
// motion vectors: target 0 as spark_fs (additive), target 1 its motion laid
// over the blocks' by its coverage (alpha-over).

#include "spark_common.hlsl"

struct SparkMrt {
  float4 spark  : SV_Target0;
  float4 motion : SV_Target1;
};

[shader("pixel")]
SparkMrt main(SparkOut i) {
  float c = ltn_spark_cov(i);
  if (c < 1e-4) discard;
  SparkMrt o;
  o.spark = float4(i.col.rgb, c * i.shp.w);
  o.motion = float4(i.mv, 0.0, c);
  return o;
}
