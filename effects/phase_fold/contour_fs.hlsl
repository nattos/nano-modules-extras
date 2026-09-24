// source.phase_fold — contour mode fragment shader.
//
// Draws the zero level-set of the blended height field as a gold line — the
// "limit cycle" taken straight from the approximate height map, no particles or
// tracer. The contour is where f(p) = pf_blended_height(p) - bias == 0, i.e. the
// white band-centre of the Bands backdrop. We turn that into a constant-width
// line via the first-order distance-to-contour estimate dist ≈ |f| / |∇f|, with
// ∇f = ∇H + W (the wind tilt's gradient is the constant wind force). Blended
// alpha-over the backdrop. Cheap: one pf_field eval per pixel, like the backdrop.

#include "field.hlsl"

struct VsOut { float4 pos : SV_Position; };

[shader("pixel")]
float4 main(VsOut i) : SV_Target0 {
  if (pf_weight_sum() < 1e-4) discard;     // hole — no cycle

  float2 vp = float2(res_x, res_y);
  float2 p = pf_pixel_to_p(i.pos.xy, vp);

  PfField f = pf_field(p);
  // pf_blended_height(p) = H - level + W·p; the cycle/zero is that minus bias.
  float fval = f.H - f.lev + dot(f.W, p) - bias;
  float2 gradf = f.grad + f.W;
  float gl = max(length(gradf), 1e-4);
  float dist = abs(fval) / gl;             // ≈ distance to the zero contour (phase-space)

  float halfw = max(cycle_width, 1e-4) * 0.5;
  float a = 1.0 - smoothstep(halfw * 0.6, halfw, dist);
  if (a <= 0.0) discard;

  return float4(1.0, 0.92, 0.38, a * 0.95);  // gold, alpha-over
}
