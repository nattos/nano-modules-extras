// source.sdf.helio_field — storm pass: excitable medium on the field lines.
//
// The self-organized-criticality release half of the effect. The
// dynamics pass slowly LOADS the system (differential rotation winds A,
// gradients steepen); this pass watches a dimensionless KINK diagnostic
//   kink = |∇²A| · eps / (|∇A| + k0)
// (line curvature at the force scale — big where lines bend harder than
// they are strong) and runs an excitable medium over it:
//
//   u (activation)  ignites where kink exceeds the Excitability
//                   threshold, and PROPAGATES anisotropically along the
//                   line direction perp(∇A) — storms travel the lines.
//   v (refractory)  charges while burning and gates re-ignition: the
//                   dead time that separates storms. Its recovery rate
//                   is the Calm knob.
//   heat            slow afterglow accumulator (crest glow downstream).
//
// The quench is physical, not just refractory: the dynamics pass reads
// u and locally reconnects A (blend toward the neighborhood average)
// where the storm burns — the storm ERASES the kink that ignited it,
// so the region needs re-winding before it can fire again. Excitability
// × Rotation spans quiet → intermittent storms → self-resonant.
//
// State: 2 × RGBA16F ping-pong, (u, v, heat, 0). Zeroed on reset.

#include "../plume/common.hlsl"

Texture2D<float4>   auxTex    : register(t0);  // (A, ...) — THIS frame's A
Texture2D<float4>   stormPrev : register(t1);
SamplerState        samp      : register(s2);
RWTexture2D<float4> stormNext : register(u3);

cbuffer StormUniforms : register(b4) {
  float dt;         // sim-scaled frame delta
  float thresh;     // kink ignition threshold (Excitability, inverted)
  float prop;       // along-line propagation gain
  float burn;       // ignition rate toward u=1

  float cool;       // u decay rate, 1/s
  float charge;     // v charge rate while burning
  float recover;    // v recovery rate, 1/s (Calm, inverted)
  float kink_gain;  // scales the drive above threshold

  float force_eps;  // stencil half-step, radians (matches dynamics)
  float sim_res;
  float reset;      // 1 = zero the storm state
  float _pad0;
};

void st_frame(float3 dir, out float3 t1, out float3 t2) {
  float3 a = abs(dir.y) < 0.92 ? float3(0.0, 1.0, 0.0)
                               : float3(1.0, 0.0, 0.0);
  t1 = normalize(cross(a, dir));
  t2 = cross(dir, t1);
}

// Reads A_smooth (.y) — every curvature estimate here must see the
// low-passed field, or per-texel bilinear phase noise reads as kinks.
float A_at(float3 d) { return auxTex.SampleLevel(samp, nano_oct_encode(d), 0).y; }
float4 storm_at(float3 d) { return stormPrev.SampleLevel(samp, nano_oct_encode(d), 0); }

[numthreads(8, 8, 1)]
void main(uint3 gid : SV_DispatchThreadID) {
  int r = int(sim_res);
  if (gid.x >= (uint)r || gid.y >= (uint)r) return;
  // Born refractory (v = 1): a fresh sun's initial field is full of
  // sheets, and without the dead time it flash-burns globally on frame
  // one. This way the field loads first and the first storms arrive on
  // the natural rhythm.
  if (reset > 0.5) { stormNext[gid.xy] = float4(0.0, 1.0, 0.0, 0.0); return; }
  float2 uv = (float2(gid.xy) + 0.5) / sim_res;
  float3 dir = nano_oct_decode(uv);

  float4 st = stormPrev.Load(int3(gid.xy, 0));
  float u = st.x, v = st.y, heat = st.z;

  // --- Kink diagnostic at the force scale. Curvature is a difference
  // of ring averages (outer at force_eps, inner at force_eps/2) — a
  // band-pass that only sees bending at the force scale. Referencing
  // the raw center texel would amplify texel noise by 1/eps² and read
  // the whole sphere as one giant kink (the everything-burns bug). ---
  float3 t1, t2;
  st_frame(dir, t1, t2);
  float heps = 0.5 * force_eps;
  float Apu = A_at(normalize(dir + force_eps * t1));
  float Amu = A_at(normalize(dir - force_eps * t1));
  float Apv = A_at(normalize(dir + force_eps * t2));
  float Amv = A_at(normalize(dir - force_eps * t2));
  float Mpu = A_at(normalize(dir + heps * t1));
  float Mmu = A_at(normalize(dir - heps * t1));
  float Mpv = A_at(normalize(dir + heps * t2));
  float Mmv = A_at(normalize(dir - heps * t2));
  float3 gA = (t1 * (Apu - Amu) + t2 * (Apv - Amv)) / (2.0 * force_eps);
  float gAl = max(length(gA), 1e-5);
  // (outer − inner) ≈ κ·(3/16)·eps² for curvature κ.
  float lapA = (Apu + Amu + Apv + Amv - Mpu - Mmu - Mpv - Mmv) * 0.25
             * (16.0 / 3.0) / (force_eps * force_eps);
  float kink = abs(lapA) * force_eps / (gAl + 0.3);

  // --- Along-line neighborhood (storms travel the lines): two taps
  // along perp(∇A), plus a weak isotropic leak so a storm can jump a
  // reconnecting junction. ---
  float3 ldir = normalize(cross(dir, gA / gAl));
  float u_line = max(storm_at(normalize(dir + force_eps * ldir)).x,
                     storm_at(normalize(dir - force_eps * ldir)).x);
  float u_iso = max(max(storm_at(normalize(dir + force_eps * t1)).x,
                        storm_at(normalize(dir - force_eps * t1)).x),
                    max(storm_at(normalize(dir + force_eps * t2)).x,
                        storm_at(normalize(dir - force_eps * t2)).x));

  // --- Excitable update. The threshold sits deep in the kink
  // distribution's tail (calibrated: background turbulence ~0.2, p90
  // ~0.5, real sheets 1+), so ignition SEEDS are rare and localized —
  // the storm's spatial extent comes from propagation along the line,
  // not from broad ignition. No field-strength gate: a current sheet's
  // field REVERSES at its center (∇A → 0 exactly where |∇²A| peaks),
  // so gating on strength would mask the true reconnection sites. ---
  float ign = saturate((kink - thresh) * kink_gain);
  // ACTIVATION BARRIER (the excitable medium's middle root): neighbor
  // activity below ~0.25 must not propagate, or u = 0 is unstable —
  // the linear gain burn·prop exceeds cool, so any residual tail would
  // regrow into a global self-sustaining simmer. With the barrier only
  // genuine fronts spread and burned-out tails die.
  float apl = saturate((u_line - 0.25) * 1.333);
  float api = saturate((u_iso - 0.25) * 1.333);
  float input = saturate(ign + prop * apl + 0.25 * prop * api);
  input *= saturate(1.0 - 1.5 * v);          // refractory gate
  u = saturate(u + dt * (burn * input * (1.0 - u) - cool * u));
  v = saturate(v + dt * (charge * u - recover * v));
  heat = saturate(heat * exp(-0.8 * dt) + 3.0 * u * dt);

  stormNext[gid.xy] = float4(u, v, heat, 0.0);
}
