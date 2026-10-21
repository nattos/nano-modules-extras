// source.mesh.lantern — a flake's fragment into the spark layer (additive):
// rgb accumulates radiance x coverage, alpha the coverage the composite lets
// it hide the scene behind by.

#include "spark_common.hlsl"

[shader("pixel")]
float4 main(SparkOut i) : SV_Target0 {
  float c = ltn_spark_cov(i);
  if (c < 1e-4) discard;
  return float4(i.col.rgb, c * i.ctr.z);
}
