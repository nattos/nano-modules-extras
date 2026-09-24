// source.legacy.double_chamber — bridger pass. One thread per bridger.
//
// A bridger is a stochastic chord between two P particles (the original's
// "Bridger" block, Latch 2176: a per-bridger endpoint that re-targets to a
// random particle with prob "Bridger Rate" and lerps toward it by a motion
// rate). Here each bridger holds TWO endpoints, each tracking a live particle
// by index. Per frame, per endpoint: with prob `rate` (or if its particle
// died) re-pick a random target, then GLIDE the rendered endpoint toward the
// live target position by `motion` (so re-targets sweep instead of snapping;
// a fresh endpoint snaps on its first appearance). The endpoints are written
// out as a Seg and drawn by the shared line vs/fs (additive/alpha).

#include "common.hlsl"

RWStructuredBuffer<BridgerState> bridgers  : register(u0);
StructuredBuffer<Particle>       particles : register(t1);
RWStructuredBuffer<Seg>          segs       : register(u2);

cbuffer Uniforms : register(b3) {
  uint  count;          // active bridger count
  uint  p_count;        // particle pool size to sample from
  uint  frame_index;
  float dt;             // reserved

  float rate;           // P(re-target) per endpoint per frame
  float motion;         // glide lerp factor toward the live target
  float opacity;        // line alpha
  float color_contrib;  // 0 = pure hue colour, 1 = captured particle colour

  float hue;            // base hue when color_contrib is low
  float tint_r;
  float tint_g;
  float tint_b;
}

// Resolve one endpoint: re-target if firing / dead, glide toward the live
// target, and report the target index back (for colour capture).
float2 resolveEnd(inout uint tgt, inout float fresh, float2 end, uint pc, uint salt) {
  uint h = dc_hash3(salt, frame_index, 0x5BD1u);
  bool dead = (tgt >= pc) || (particles[min(tgt, pc - 1u)].a.z <= 0.0);
  bool fire = (fresh < 0.5) || dead || (rate > 0.0 && dc_unit(dc_hash(h)) < rate);
  if (fire) {
    tgt = min((uint)(dc_unit(dc_hash(h ^ 0x77u)) * (float)pc), pc - 1u);
  }
  float2 tpos = particles[min(tgt, pc - 1u)].a.xy;
  float2 outEnd = (fresh < 0.5) ? tpos : lerp(end, tpos, saturate(motion));
  fresh = 1.0;
  return outEnd;
}

[numthreads(64, 1, 1)]
void main(uint3 gid : SV_DispatchThreadID) {
  uint i = gid.x;
  if (i >= count) return;

  uint pc = max(p_count, 1u);
  BridgerState st = bridgers[i];

  uint  tA = asuint(st.b.x), tB = asuint(st.b.y);
  float fA = st.b.z,         fB = st.b.w;
  float2 endA = resolveEnd(tA, fA, st.a.xy, pc, i * 2u + 0u);
  float2 endB = resolveEnd(tB, fB, st.a.zw, pc, i * 2u + 1u);

  st.a = float4(endA, endB);
  st.b = float4(asfloat(tA), asfloat(tB), fA, fB);
  bridgers[i] = st;

  // Colour: hue-generated base blended toward the mean captured particle colour.
  float3 cA = dc_unpack_rgb(asuint(particles[min(tA, pc - 1u)].b.w));
  float3 cB = dc_unpack_rgb(asuint(particles[min(tB, pc - 1u)].b.w));
  float3 partCol = (cA + cB) * 0.5;
  float3 hueCol  = dc_hsv_to_rgb(float3(hue, 1.0, 1.0));
  float3 col = lerp(hueCol, partCol, saturate(color_contrib))
             * float3(tint_r, tint_g, tint_b);

  Seg sg;
  sg.a = float4(endA, endB);
  sg.b = float4(max(col, 0.0), max(opacity, 0.0));
  segs[i] = sg;
}
