// source.sdf.helio_field — dynamics pass: 2D MHD-lite on the sphere.
//
// One step of the coupled fluid + magnetic-potential system on an
// octahedral map. State (two RGBA32F ping-pong pairs):
//   dyn = (vel.xyz, spare)  — tangent velocity as a WORLD 3D vector
//   aux = (A, A_smooth, 0, 0) — magnetic scalar potential + a low-passed
//         copy (0.5·center + 0.5·sim-ring — near-zero response at texel
//         Nyquist). ALL curvature/gradient estimates downstream read
//         A_smooth: ring averages of the raw bilinear interpolant carry
//         O(1) per-texel phase error at small radii, which reads as
//         zebra-stripe "kinks" everywhere and feeds back into the force.
// (storm state u/v/heat lives in the storm pass's own ping-pong; this
// pass only READS last frame's u for reconnection)
//
// The magnetic model is the classic 2D reduction: contours of A ARE the
// field lines (frozen-in flux — advecting A transports the lines with
// the fluid), line direction = perp(∇A), current density j = −∇²A, and
// the Lorentz force reduces to F = −∇²A·∇A — a tension that pulls kinked
// lines straight and combs the flow into filaments and current sheets.
//
// Forces, in order: semi-Lagrangian advection (midpoint backtrace in
// DIRECTION space: dir' = normalize(dir − v·dt) then re-encode — this is
// seam-correct by construction), Lorentz tension (saturated), vorticity
// confinement (re-sharpens the eddies bilinear advection smears out),
// zonal differential-rotation drive (the Ω-effect: equator-fast shear
// that winds the lines up — the energy input of the storm cycle), curl-
// noise granulation stirring, drag, speed clamp.
//
// There is deliberately NO pressure projection: at this resolution an
// explicit divergence solve is CFL-starved into irrelevance, the driven
// forces are almost entirely rotational, and the one compressive term
// (Lorentz) bunching A-contours into filaments is the wanted look.
// Drag + vmax bound the compressible part.
//
// Seam handling: every stencil tap goes through a 3D direction and
// nano_oct_encode (always in-range), so no uv fold rules exist anywhere.
// The tangent frame used for taps is arbitrary — every derived quantity
// (div, curl, gradient-as-3D-vector, laplacian) is frame-invariant.

#include "../plume/common.hlsl"
#include "nano_noise3.hlsl"

Texture2D<float4>   dynPrev   : register(t0);
Texture2D<float4>   auxPrev   : register(t1);
SamplerState        samp      : register(s2);
RWTexture2D<float4> dynNext   : register(u3);
RWTexture2D<float4> auxNext   : register(u4);
Texture2D<float4>   stormPrev : register(t6);  // (u, v, heat) — last frame

cbuffer DynUniforms : register(b5) {
  float dt;         // sim-scaled frame delta, seconds
  float reset;      // 1 = write initial conditions this frame
  float sim_res;    // map resolution
  float seed;       // variation

  float rot_rate;   // zonal equator angular rate, rad/s
  float rot_relax;  // relaxation toward the zonal profile, 1/s
  float stir_gain;  // granulation stirring acceleration
  float stir_phase; // drifting noise-domain phase (CPU-accumulated)

  float mag_gain;   // Lorentz tension gain
  float conf_gain;  // vorticity confinement gain
  float drag;       // 1/s
  float vmax;       // speed clamp, rad/s

  float sim_eps;    // stencil half-step, radians (~1.5 texel arcs)
  float recon;      // reconnection lerp fraction where storms burn
  float emerge;     // flux-emergence relax rate toward target_A, 1/s
  float emerge_phase; // slow drift of the emergence target's noise domain

  float resist;     // resistivity: A -> neighbor-avg blend fraction [0,0.25]
  float force_eps;  // FORCE stencil half-step, radians (several texels:
                    // grid-scale force estimates are pure noise — Lorentz
                    // whiskers, confinement box-cells — so forces only see
                    // scales the sim can actually resolve)
  float resist_w;   // wide-band resistivity: blend toward the FORCE-ring
                    // average. The Lorentz cascade pumps low-amplitude,
                    // high-curvature corrugation into the 3–8 texel band
                    // (invisible in the lines, dominant in curvature) —
                    // without this it saturates the storm kink diagnostic
                    // sphere-wide. Ring-average blending is scale-selective:
                    // linear-across-the-ring content passes untouched.
  float visc;       // band-limited VISCOSITY: blend vel toward its force-
                    // ring average. The tension force + induction form an
                    // Alfvén oscillator whose band-edge modes have ω·dt ≫ 1
                    // — explicitly integrated they explode into standing
                    // zebra. Nothing legitimate drives velocity below the
                    // force scale (stirring is large-scale, zonal global),
                    // so hard-dissipating it there kills the oscillator
                    // while the large eddies pass through the ring average.
};

// Arbitrary smooth-enough tangent frame. Frame choice cancels out of
// every quantity we compute (all are expressed as scalars or 3D vectors).
void helio_frame(float3 dir, out float3 t1, out float3 t2) {
  float3 a = abs(dir.y) < 0.92 ? float3(0.0, 1.0, 0.0)
                               : float3(1.0, 0.0, 0.0);
  t1 = normalize(cross(a, dir));
  t2 = cross(dir, t1);
}

float4 dyn_at(float3 d) { return dynPrev.SampleLevel(samp, nano_oct_encode(d), 0); }
float4 aux_at(float3 d) { return auxPrev.SampleLevel(samp, nano_oct_encode(d), 0); }

// Surface vorticity zeta = dir · (∇ × v) at an arbitrary direction,
// measured at the FORCE scale.
float zeta_at(float3 d) {
  float3 t1, t2;
  helio_frame(d, t1, t2);
  float3 vpu = dyn_at(normalize(d + force_eps * t1)).xyz;
  float3 vmu = dyn_at(normalize(d - force_eps * t1)).xyz;
  float3 vpv = dyn_at(normalize(d + force_eps * t2)).xyz;
  float3 vmv = dyn_at(normalize(d - force_eps * t2)).xyz;
  return (dot(vpu - vmu, t2) - dot(vpv - vmv, t1)) / (2.0 * force_eps);
}

// Flux-emergence target: a tilted global dipole + two octaves of large-
// scale noise whose domain drifts slowly. This is both the initial
// condition and the field A relaxes toward — semi-Lagrangian advection
// numerically diffuses A, and without replenishment the sun would decay
// to a few giant cells and go quiet. Emergence keeps it alive forever
// (new flux surfacing) while staying far slower than the advection, so
// the fluid's combing/bunching remains what you see.
float target_A(float3 dir) {
  float3 so = float3(seed * 13.1, seed * 7.7, seed * 3.3);
  float3 dr = float3(0.11, -0.07, 0.13) * emerge_phase;
  float3 m = normalize(float3(0.35, 1.0, 0.2 + 0.1 * seed));
  return 0.5 * dot(dir, m)
       + 0.75 * nano_gnoise3(dir * 1.7 + so + dr)
       + 0.35 * nano_gnoise3(dir * 3.4 + so * 1.7 + 5.0 + dr * 1.7);
}

[numthreads(8, 8, 1)]
void main(uint3 gid : SV_DispatchThreadID) {
  int r = int(sim_res);
  if (gid.x >= (uint)r || gid.y >= (uint)r) return;
  float2 uv = (float2(gid.xy) + 0.5) / sim_res;
  float3 dir = nano_oct_decode(uv);

  if (reset > 0.5) {
    dynNext[gid.xy] = float4(0.0, 0.0, 0.0, 0.0);
    float A0 = target_A(dir);
    auxNext[gid.xy] = float4(A0, A0, 0.0, 0.0);
    return;
  }

  // --- Semi-Lagrangian advection (midpoint) ---
  float3 v0 = dynPrev.Load(int3(gid.xy, 0)).xyz;
  float3 dm = normalize(dir - v0 * (0.5 * dt));
  float3 vm = dyn_at(dm).xyz;
  float3 db = normalize(dir - vm * dt);
  float3 vel = dyn_at(db).xyz;
  float A = aux_at(db).x;
  vel -= dir * dot(dir, vel);   // parallel-transport approx: reproject

  // --- Stencil taps around the CENTER point. Two scales: sim_eps for
  // the local resistivity blur, force_eps for every force estimate. ---
  float3 t1, t2;
  helio_frame(dir, t1, t2);
  float Apu = aux_at(normalize(dir + sim_eps * t1)).x;
  float Amu = aux_at(normalize(dir - sim_eps * t1)).x;
  float Apv = aux_at(normalize(dir + sim_eps * t2)).x;
  float Amv = aux_at(normalize(dir - sim_eps * t2)).x;

  float3 dpu = normalize(dir + force_eps * t1);
  float3 dmu = normalize(dir - force_eps * t1);
  float3 dpv = normalize(dir + force_eps * t2);
  float3 dmv = normalize(dir - force_eps * t2);
  float2 Fpu = aux_at(dpu).xy, Fmu = aux_at(dmu).xy;
  float2 Fpv = aux_at(dpv).xy, Fmv = aux_at(dmv).xy;
  float heps = 0.5 * force_eps;
  float Mpu = aux_at(normalize(dir + heps * t1)).y;
  float Mmu = aux_at(normalize(dir - heps * t1)).y;
  float Mpv = aux_at(normalize(dir + heps * t2)).y;
  float Mmv = aux_at(normalize(dir - heps * t2)).y;

  // --- Lorentz tension: F = −∇²A · ∇A (kinked lines pull straight).
  // Gradient and laplacian both read A_smooth (.y — see header): ring
  // averages of the raw field carry per-texel phase noise that this
  // force would amplify into standing zebra. The laplacian is a
  // difference of ring averages (force ring minus half ring) — a
  // band-pass seeing only curvature at the force scale. Saturated so a
  // forming current sheet can't spike the integrator. ---
  float3 gA = (t1 * (Fpu.y - Fmu.y) + t2 * (Fpv.y - Fmv.y)) /
              (2.0 * force_eps);
  float avgN4 = (Apu + Amu + Apv + Amv) * 0.25;
  float avgF4 = (Fpu.x + Fmu.x + Fpv.x + Fmv.x) * 0.25;
  float avgFy = (Fpu.y + Fmu.y + Fpv.y + Fmv.y) * 0.25;
  float avgMy = (Mpu + Mmu + Mpv + Mmv) * 0.25;
  float lapA = (avgFy - avgMy) * (16.0 / 3.0) / (force_eps * force_eps);
  float3 fmag = (-lapA) * gA;
  fmag = fmag / (1.0 + 0.25 * length(fmag));
  vel += fmag * (mag_gain * dt);

  // Cross-line friction (anisotropic drag): the tension force + lagged
  // advection form an explicit Alfvén oscillator — plasma sloshing
  // ACROSS the lines — which explicit integration destabilizes into
  // standing-wave zebra at the force-stencil band. Damping only the
  // across-line velocity kills the oscillation while leaving the
  // along-line streaming (the look) untouched. Scaled with the tension
  // gain, since that's what drives the oscillator.
  float gAl_f = length(gA);
  if (gAl_f > 0.2) {
    float3 nperp = gA / gAl_f;
    float k_x = (1.0 + 0.6 * mag_gain) * dt;
    vel -= nperp * (dot(vel, nperp) * (1.0 - exp(-k_x)));
  }

  // --- Vorticity confinement: push existing swirls toward their own
  // cores — counteracts the bilinear advection's smearing. ---
  if (conf_gain > 1e-4) {
    float zc = zeta_at(dir);
    float3 gz = (t1 * (abs(zeta_at(dpu)) - abs(zeta_at(dmu))) +
                 t2 * (abs(zeta_at(dpv)) - abs(zeta_at(dmv)))) /
                (2.0 * force_eps);
    float gzl = length(gz);
    if (gzl > 1e-4)
      vel += cross(gz / gzl, dir) * (zc * conf_gain * force_eps * dt);
  }

  // --- Zonal differential rotation (Ω-effect drive): equator-fast
  // solid-body-ish profile the velocity relaxes toward. This is the
  // slow energy input that winds the field lines up. ---
  float om = rot_rate * (1.0 - 0.6 * dir.y * dir.y);
  float3 vzn = cross(float3(0.0, 1.0, 0.0), dir) * om;
  vel = lerp(vel, vzn, 1.0 - exp(-rot_relax * dt));

  // --- Granulation stirring: surface curl of a drifting noise
  // streamfunction — divergence-free by construction. ---
  if (stir_gain > 1e-4) {
    float3 sp = dir * 3.1 + float3(seed * 37.7, seed * 11.3, seed * 5.1)
              + float3(0.31, 0.17, -0.23) * stir_phase;
    float e = 0.12;
    float ps0 = nano_gnoise3(sp);
    float3 gpsi = float3(nano_gnoise3(sp + float3(e, 0.0, 0.0)) - ps0,
                         nano_gnoise3(sp + float3(0.0, e, 0.0)) - ps0,
                         nano_gnoise3(sp + float3(0.0, 0.0, e)) - ps0) / e;
    vel += cross(dir, gpsi) * (stir_gain * dt);
  }

  // --- Band-limited viscosity (see `visc` above) ---
  {
    float3 vr = (dyn_at(dpu).xyz + dyn_at(dmu).xyz +
                 dyn_at(dpv).xyz + dyn_at(dmv).xyz) * 0.25;
    vr -= dir * dot(dir, vr);
    vel = lerp(vel, vr, visc);
  }

  // --- Drag + clamp ---
  vel *= exp(-drag * dt);
  vel -= dir * dot(dir, vel);
  float spd = length(vel);
  if (spd > vmax) vel *= vmax / spd;

  // --- A transport: resistivity, flux emergence, reconnection.
  // Resistivity (dt-scaled neighbor-average blend — unconditionally
  // stable): without it the Lorentz tension filaments A all the way to
  // grid scale and every line grows herringbone fuzz. This is the η∇²A
  // of real 2D MHD, sized to eat wiggles the stencil can't resolve
  // while leaving the large-scale combing alone. ---
  A = lerp(A, avgN4, resist);
  A = lerp(A, avgF4, resist_w);
  A += (target_A(dir) - A) * (emerge * dt);
  // Reconnection: where a storm actively burns (u from last frame's
  // storm pass), A blends toward its FORCE-scale neighborhood average —
  // the storm erases the kink that ignited it. Same stable lerp form as
  // the resistivity, so no diffusion CFL to violate.
  float storm_u = stormPrev.Load(int3(gid.xy, 0)).x;
  A = lerp(A, avgF4, saturate(recon * storm_u));

  dynNext[gid.xy] = float4(vel, 0.0);
  // A_smooth: half center, half sim-ring — a real low-pass (near-zero
  // gain at texel Nyquist), recomputed fresh each frame.
  auxNext[gid.xy] = float4(A, 0.5 * A + 0.5 * avgN4, 0.0, 0.0);
}
