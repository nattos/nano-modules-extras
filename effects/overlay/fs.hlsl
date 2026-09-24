// overlay — instanced solid-quad fragment shader. Emits the flat straight-alpha
// colour; the render PSO's AlphaOver blend (src.rgb*src.a + dst*(1-src.a)) does
// the compositing onto the target.

#include "common.hlsl"

struct VsOut {
  float4 pos : SV_Position;
  nointerpolation float4 color : TEXCOORD0;
};

[shader("pixel")]
float4 main(VsOut i) : SV_Target0 {
  if (i.color.a <= 0.0) discard;
  return i.color;
}
