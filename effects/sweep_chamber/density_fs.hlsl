// source.particles.sweep_chamber — density splat fragment shader.
//
// Bilinear scatter of ONE unit of mass per particle, drawn ADDITIVELY:
//   .r  = Σ mass            — particle count per texel
//   .gb = Σ mass · velocity — mass-weighted motion (stream align/diverge)
// `corner` is the texel-space offset from the particle centre (the quad is
// 2×2 texels), so the tent weight (1-|dx|)(1-|dy|) over the four covered
// texel centres sums to exactly 1 — no mass gain or loss as a particle
// drifts between texels. The interaction halo is applied by density_blur.

struct DOut {
  float4 pos    : SV_Position;
  float2 corner : TEXCOORD0;
  nointerpolation float2 vel : TEXCOORD1;
};

[shader("pixel")]
float4 main(DOut i) : SV_Target0 {
  float2 t = saturate(1.0 - abs(i.corner));
  float w = t.x * t.y;
  return float4(w, w * i.vel.x, w * i.vel.y, w);
}
