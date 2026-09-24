// source.light.side_jet — Stage 1: 1D axial plume solver.
//
// The engine under test is FIXED at the left edge (cell 0 = nozzle). The
// plume is quasi-1D along the axis; this pass evolves a small row of cells
// that every downstream synthesis pass reads.
//
// SINGLE-THREAD design: the entire stiff solve runs on one thread with
// serial loops over substeps and cells, operating on threadgroup memory.
// This is deliberate — the process is stiff (fast pressure relaxation
// against slow material advection), so we want many in-shader substeps,
// and a single serial sweep is backend-agnostic (it does not depend on the
// host's threadgroup-size convention, which varies). The cell row is tiny,
// so one thread looping is cheap.
//
// Two propagation speeds, deliberately decoupled (this is the whole point):
//   * pressure / structure  → transported at `wavespeed` (fast): a throttle
//     change re-shapes the shock-cell train across the jet within ~2 frames.
//   * luminous material `b`  → advected at the flow velocity `u` (slow):
//     the visible gas takes its physical transit time to reach the far field.
//
// Operator split: the hyperbolic (advection / wave) parts run explicit at a
// CFL-bounded sub-dt; the stiff relaxation terms use an exponential
// integrator  x += (target-x)(1-e^(-dt/tau))  which is exact for linear
// relaxation and unconditionally stable regardless of tau.
//
// Double buffering without a second array: the update is pure upwind (each
// cell reads only itself and its upstream neighbour i-1), so a forward
// sweep that carries the previous cell's pre-update value in registers
// (old_*L) is exactly a synchronous double-buffered step.

#include "nano_sanitize.hlsl"

struct Cell {
  float u;     // axial velocity (canvas-uv / sec)
  float p;     // pressure ratio (>1 over-expanded → shock diamonds)
  float b;     // luminous density (advected brightness)
  float m;     // maturity: 0 coherent core → 1 turbulent breakdown
  float kappa; // local shock-cell wavenumber (2pi / cell spacing)
  float phi;   // integrated shock phase along the axis
  float lit;   // ignition front 0..1 (races down the axis on light-up)
  float _pad;
};

RWStructuredBuffer<Cell> cells : register(u0);

cbuffer SimUniforms : register(b1) {
  float dt;             // frame delta (sec)
  uint  substeps;       // in-shader sub-iterations
  uint  W;              // active cell count (<= 256)
  float cell_dx;        // axial cell spacing (canvas-uv)

  float chamberP;       // chamber pressure (spool-integrated, CPU side)
  float exitVel;        // nozzle exit velocity (BC for u)
  float pressureRatio;  // nozzle over-expansion (BC for p)
  float litTarget;      // 1 firing / 0 off (BC for the ignition front)

  float wavespeed;      // pressure transport speed (>> u → snappy)
  float maturityGrowth; // turbulent-breakdown growth rate
  float coreDecay;      // luminous-density decay along the axis
  float flameSpeed;     // extra ignition-front advance over the flow

  float diamondSpacing; // base shock-cell spacing (canvas-uv)
  float velRelax;       // velocity relaxation time constant
  float _pad0;
  float _pad1;
};

#define MAXW 256
groupshared float s_u[MAXW];
groupshared float s_p[MAXW];
groupshared float s_b[MAXW];
groupshared float s_m[MAXW];
groupshared float s_lit[MAXW];

[numthreads(64, 1, 1)]
void main(uint3 gid : SV_GroupThreadID) {
  // Single thread does all the work — robust across host threadgroup-size
  // conventions (some hosts launch a fixed tile regardless of [numthreads]).
  if (gid.x != 0u || gid.y != 0u || gid.z != 0u) return;

  uint w = (W > 256u) ? 256u : W;

  // Load persistent state from last frame into threadgroup memory, SANITIZED.
  // The buffer persists across frames, so a stray NaN (e.g. a pre-fix CFL
  // blow-up) or uninitialized garbage would otherwise stick forever and
  // freeze the tail. nano_sanitize maps NaN -> ambient and folds ±Inf/garbage
  // back into range; the field then re-propagates from the nozzle in a frame
  // or two.
  for (uint k = 0u; k < w; ++k) {
    Cell c = cells[k];
    s_u[k]   = nano_sanitize(c.u,   0.0, 0.0, 1000.0);
    s_p[k]   = nano_sanitize(c.p,   1.0, 0.0, 100.0);
    s_b[k]   = nano_sanitize(c.b,   0.0, 0.0, 100.0);
    s_m[k]   = nano_sanitize(c.m,   0.0, 0.0, 1.0);
    s_lit[k] = nano_sanitize(c.lit, 0.0, 0.0, 1.0);
  }

  float dts = dt / max((float)substeps, 1.0);

  // --- Substep loop (stiff). Each substep is a forward upwind sweep. ---
  for (uint s = 0u; s < substeps; ++s) {
    // Nozzle (cell 0) — boundary condition from the CPU control integrator.
    float old_uL  = s_u[0];   // pre-update values feed cell 1 (double buffer)
    float old_pL  = s_p[0];
    float old_bL  = s_b[0];
    float old_mL  = s_m[0];
    float old_litL = s_lit[0];
    s_lit[0] = lerp(s_lit[0], litTarget, 1.0 - exp(-dts / 0.02));
    s_u[0]   = exitVel;
    s_p[0]   = pressureRatio;
    s_b[0]   = chamberP * s_lit[0];
    s_m[0]   = 0.0;

    for (uint i = 1u; i < w; ++i) {
      float u = s_u[i], p = s_p[i], b = s_b[i], m = s_m[i], lit = s_lit[i];

      float cfl_u = clamp(u * dts / cell_dx, 0.0, 0.95);  // material Courant #

      // Velocity: upwind self-advection + relax to a maturity-decayed target.
      float u_target = exitVel * (1.0 - 0.3 * m);
      float nu = u - cfl_u * (u - old_uL);
      nu = lerp(nu, u_target, 1.0 - exp(-dts / max(velRelax, 1e-3)));

      // Pressure: transported FAST at wavespeed, relaxes toward ambient (1).
      float cfl_p = min(wavespeed * dts / cell_dx, 0.95);
      float np = p - cfl_p * (p - old_pL);
      np = lerp(np, 1.0, 1.0 - exp(-dts / 0.25));

      // Maturity: advected + grows (faster where the flow is over-expanded).
      float nm = m - cfl_u * (m - old_mL);
      nm = saturate(nm + dts * maturityGrowth * (0.4 + 0.6 * saturate(p - 1.0)));

      // Ignition front: advected at (u + flameSpeed) — races out on light-up,
      // recedes from the root on flame-out.
      float cfl_f = min((u + flameSpeed) * dts / cell_dx, 0.95);
      float nlit = saturate(lit - cfl_f * (lit - old_litL));

      // Luminous density: advected from the nozzle source, decays downstream.
      float nb = b - cfl_u * (b - old_bL);
      nb = max(nb - dts * coreDecay * b, 0.0);

      // Carry this cell's PRE-update values upstream to the next cell, then
      // commit. (= synchronous double-buffered upwind.)
      old_uL = u; old_pL = p; old_bL = b; old_mL = m; old_litL = lit;
      s_u[i] = nu; s_p[i] = np; s_b[i] = nb; s_m[i] = nm; s_lit[i] = nlit;
    }
  }

  // --- Shock-cell wavenumber + integrated phase (serial prefix sum). ---
  float acc = 0.0;
  for (uint j = 0u; j < w; ++j) {
    float Ls = diamondSpacing * (0.5 + 1.5 * max(s_p[j], 1.0));
    float kappa = (s_m[j] < 0.99) ? (6.28318530718 / max(Ls, 1e-4)) : 0.0;
    acc += kappa * cell_dx;

    Cell c;
    c.u = s_u[j]; c.p = s_p[j]; c.b = s_b[j]; c.m = s_m[j];
    c.kappa = kappa; c.phi = acc; c.lit = s_lit[j]; c._pad = 0.0;
    cells[j] = c;
  }
}
