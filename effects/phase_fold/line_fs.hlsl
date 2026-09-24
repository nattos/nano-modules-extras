// source.phase_fold — line raster fragment shader.
//
// Soft anti-aliased line with a CONTINUOUS flow animation (no quantized
// arrowheads). Each segment carries its arc-length position along the line
// (flow.x) and a per-line stagger (flow.y); a glow rides down the line wherever
// frac(arc - flow_phase + stagger) ≈ 0, advancing smoothly as flow_phase grows.
//   code < 0.58 → streamline: faint speed-graded blue + a flowing white comet
//   code > 0.80 → gold limit cycle + a moving bright highlight
// Straight (non-premultiplied) output; the alpha-over render PSO does
// src.rgb*src.a + dst*(1-src.a). flow_phase rides in the shared uniform (b0).

#include "common.hlsl"

struct VsOut {
  float4 pos    : SV_Position;
  float2 local  : TEXCOORD0;
  nointerpolation float2 meta : TEXCOORD1;  // code, alpha
  nointerpolation float2 flow : TEXCOORD2;  // arc, stagger
};

// Symmetric, wrapped distance of `arc` from the travelling front — 0 at the
// front, growing to 0.5 on the far side. Smaller = closer to the bright head.
float pf_flow_dist(float arc, float stagger) {
  float d = frac(arc - flow_phase + stagger);
  return min(d, 1.0 - d);
}

[shader("pixel")]
float4 main(VsOut i) : SV_Target0 {
  float edge = smoothstep(1.0, 0.35, abs(i.local.y));   // soft rim falloff
  float code = i.meta.x;
  float baseAlpha = i.meta.y * edge;
  if (baseAlpha <= 0.0) discard;

  if (code > 0.80) {
    // Gold limit cycle + a moving bright highlight riding around it.
    float hot = smoothstep(0.05, 0.0, pf_flow_dist(i.flow.x, 0.0));
    float3 col = lerp(float3(1.0, 0.92, 0.38), float3(1.0, 1.0, 0.92), hot);
    float alpha = saturate(baseAlpha * (1.0 + hot * 0.6));
    return float4(col, alpha);
  }

  // Streamline: faint speed-graded blue with a flowing white comet head.
  float t = saturate(code / 0.55);
  float3 base = lerp(float3(0.22, 0.40, 0.78), float3(0.78, 0.88, 1.0), t);
  float comet = smoothstep(0.12, 0.0, pf_flow_dist(i.flow.x, i.flow.y));
  float3 col = lerp(base, float3(0.85, 0.93, 1.0), comet);
  float alpha = saturate(baseAlpha * (1.0 + comet * 1.6));
  return float4(col, alpha);
}
