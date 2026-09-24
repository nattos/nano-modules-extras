// source.phase_fold — flow_field bake pass.
//
// Bakes the induced vector field v = level-set flow + wind (the same
// pf_velocity the streamlines integrate) into a screen-resolution
// rgba16float texture so a downstream consumer (a particle swarm, a flow
// modifier, …) can sample it without re-evaluating the atlas. Only runs
// when the `flow_field` output is wired (gated host-side).
//
// Output convention (the canonical `flow_field/velocity` leaf):
//   .xy = screen-uv velocity per SECOND (advect uv by v.xy * dt),
//   .z  = speed |xy|,
//   .w  = validity (1 inside the field, 0 over a hole).
//
// phase_fold's field is in phase-space; the map phase→uv is the constant
// jacobian below (a uniform scale + the screen y-flip), so we convert the
// phase-space velocity to a screen-uv velocity once per texel.

#include "field.hlsl"

RWTexture2D<float4> flowTex : register(u2);   // b0 uniforms, t1 cells (field.hlsl)

[numthreads(8, 8, 1)]
void main(uint3 gid : SV_DispatchThreadID) {
  float2 vp = float2(res_x, res_y);
  if (gid.x >= (uint)res_x || gid.y >= (uint)res_y) return;

  // Hole (all four corners invalid): no field here.
  if (pf_weight_sum() < 1e-4) {
    flowTex[gid.xy] = float4(0.0, 0.0, 0.0, 0.0);
    return;
  }

  float2 p = pf_pixel_to_p(float2(gid.xy), vp);
  float2 v = pf_velocity(p);

  // phase-space velocity → screen-uv velocity/sec. uv = pf_p_to_uv(p), whose
  // jacobian is d(uv.x)/d(p.x) = k/vp.x and d(uv.y)/d(p.y) = -k/vp.y, with
  // k = 0.5*max(vp)/extent (the y axis flips because uv is y-down).
  float mx = max(vp.x, vp.y);
  float k = 0.5 * mx / extent;
  float2 uvv = float2(v.x * k / vp.x, -v.y * k / vp.y);

  flowTex[gid.xy] = float4(uvv, length(uvv), 1.0);
}
