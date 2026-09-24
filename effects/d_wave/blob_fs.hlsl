// warp.legacy.d_wave — dampening-flash fragment shader.
//
// Soft elongated profile (thin × short) written ADDITIVELY (positive) into the
// damp texture; the warp pass subtracts this accumulated damp from the wave
// field, so overlapping flashes dampen more. Additive blend is src*src.a + dst,
// so alpha MUST be 1 for the strength to accumulate.

struct In {
  float4 pos      : SV_Position;
  float2 local    : TEXCOORD0;                 // quad-local [-1,1]
  nointerpolation float strength : TEXCOORD1;
};

[shader("pixel")]
float4 main(In i) : SV_Target {
  float ax = i.local.x, ay = i.local.y;
  float mask = exp(-ax * ax * 8.0) * exp(-ay * ay * 4.0);   // thin × short
  return float4(i.strength * mask, 0.0, 0.0, 1.0);
}
