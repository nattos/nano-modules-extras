// triangulate — line-quad fragment shader. Soft anti-aliased edge; straight
// (non-premultiplied) colour for the alpha-over render PSO.
struct VsOut {
  float4 pos   : SV_Position;
  float2 local : TEXCOORD0;
  nointerpolation float3 color : TEXCOORD1;
};

float4 main(VsOut i) : SV_Target0 {
  float a = smoothstep(1.0, 0.35, abs(i.local.y));   // soft rim falloff
  if (a <= 0.0) discard;
  return float4(i.color, a);
}
