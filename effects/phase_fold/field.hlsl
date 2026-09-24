// source.phase_fold — the atlas field + induced vector field.
//
// Declares the GPU-uploaded atlas cell buffer (register t1) and evaluates the
// blended scalar field H (backdrop) and the vector field v = level-set flow +
// wind (the tracers). Split out of common.hlsl so the line raster can include
// common.hlsl alone — it binds the segment buffer at t1 and never touches the
// cells. Included by backdrop.hlsl, stream.hlsl, cycle.hlsl.

#ifndef PHASE_FOLD_FIELD_HLSL
#define PHASE_FOLD_FIELD_HLSL

#include "common.hlsl"
#include "nano_sanitize.hlsl"   // nano_is_nan / nano_sanitize — never the isnan/isinf intrinsics

// The atlas cell buffer (read-only). One cell = PF_STRIDE floats.
StructuredBuffer<float> cells : register(t1);

// --- Scalar field H (for the backdrop bands) -------------------------------

float pf_cellH(uint ci, float2 p) {
  uint base = ci * PF_STRIDE;
  float H = -0.5 * cells[base] * dot(p, p);
  for (uint k = 0u; k < PF_K; k++) {
    uint o = base + 4u + k * 4u;
    float2 d = p - float2(cells[o], cells[o + 1u]);
    float s2 = max(cells[o + 2u] * cells[o + 2u], 1e-6);
    H = H + cells[o + 3u] * exp(-dot(d, d) / (2.0 * s2));
  }
  float rho = length(p);
  for (uint j = 0u; j < PF_R; j++) {
    uint o = base + 4u + PF_K * 4u + j * 3u;
    float dr = rho - cells[o];
    float rs2 = max(cells[o + 1u] * cells[o + 1u], 1e-6);
    H = H + cells[o + 2u] * exp(-dr * dr / (2.0 * rs2));
  }
  return H;
}

// Height above each cell's OWN cycle level, blended over the 4 corners. The
// zero crossing of this is the limit cycle; the backdrop bands it.
//
// Wind awareness: wind is a non-potential FORCE, so no scalar height reproduces
// the wind-distorted flow exactly. But the real attractor is held near H=level
// by the strong mu normal-pull, and that orbit BULGES DOWNWIND under wind (as
// the integrated gold cycle shows). So tilt the banded terrace ALONG the wind by
// the linear force-potential ramp W·p — the zero band then shifts downwind with
// the cycle. (The pure streamfunction Wx*y - Wy*x would drift perpendicular —
// the guiding-centre drift of the rotational flow alone — which reads as the
// wrong direction once the normal-pull dominates.) Blended over the 4 corners;
// zero at wind=0 so the calm look is unchanged.
float pf_blended_height(float2 p) {
  float d = 0.0;
  float Wx = 0.0, Wy = 0.0;
  [unroll] for (uint i = 0u; i < 4u; i++) {
    float w = weights[i];
    if (w <= 0.0) continue;
    uint ci = (uint)corners[i];
    d += w * (pf_cellH(ci, p) - cells[ci * PF_STRIDE + 1u]);
    uint wb = ci * PF_STRIDE + PF_WIND_OFF;
    Wx += w * wind * cells[wb + 2u] * cells[wb + 0u];
    Wy += w * wind * cells[wb + 2u] * cells[wb + 1u];
  }
  return d + (Wx * p.x + Wy * p.y);   // + along-wind force-potential tilt
}

// --- Blended field sample: H, ∇H, level, mu, wind force (all 4-corner blended).
// The shared core for the vector field AND the limit-cycle solver / break check.
struct PfField {
  float H;       // blended scalar field
  float2 grad;   // ∇H
  float lev;     // blended cycle level
  float mu;      // blended attraction rate
  float2 W;      // blended wind force (z * wmax * wdir)
};

PfField pf_field(float2 p) {
  PfField f;
  f.H = 0.0; f.grad = float2(0, 0); f.lev = 0.0; f.mu = 0.0; f.W = float2(0, 0);
  [unroll] for (uint i = 0u; i < 4u; i++) {
    float wi = weights[i];
    if (wi <= 0.0) continue;
    uint ci = (uint)corners[i];
    uint base = ci * PF_STRIDE;
    float well = cells[base];
    float h = -0.5 * well * dot(p, p);
    float cgx = -well * p.x, cgy = -well * p.y;
    for (uint k = 0u; k < PF_K; k++) {
      uint o = base + 4u + k * 4u;
      float2 d = p - float2(cells[o], cells[o + 1u]);
      float s2 = max(cells[o + 2u] * cells[o + 2u], 1e-6);
      float e = cells[o + 3u] * exp(-dot(d, d) / (2.0 * s2));
      h += e; cgx -= e * d.x / s2; cgy -= e * d.y / s2;
    }
    float rho = max(length(p), 1e-6);
    for (uint j = 0u; j < PF_R; j++) {
      uint o = base + 4u + PF_K * 4u + j * 3u;
      float dr = rho - cells[o];
      float rs2 = max(cells[o + 1u] * cells[o + 1u], 1e-6);
      float e = cells[o + 2u] * exp(-dr * dr / (2.0 * rs2));
      h += e;
      float dHdrho = e * (-dr / rs2);
      cgx += dHdrho * p.x / rho; cgy += dHdrho * p.y / rho;
    }
    uint wb = base + PF_WIND_OFF;
    f.H += wi * h; f.grad += wi * float2(cgx, cgy);
    f.lev += wi * cells[base + 1u]; f.mu += wi * cells[base + 2u];
    f.W += wi * wind * cells[wb + 2u] * float2(cells[wb + 0u], cells[wb + 1u]);
  }
  return f;
}

// --- Vector field v = level-set flow + wind (for the streamlines) -----------

float2 pf_velocity(float2 p) {
  PfField f = pf_field(p);
  // bias shifts the cycle level → moves the limit cycle to a different contour.
  float s = -f.mu * (f.H - f.lev - bias);
  return float2(-f.grad.y + s * f.grad.x + f.W.x,
                 f.grad.x + s * f.grad.y + f.W.y);
}

// RK2 (midpoint) with a capped displacement so the integration stays smooth.
float2 pf_step(float2 q, float dt) {
  float2 v0 = pf_velocity(q);
  float2 v1 = pf_velocity(q + 0.5 * dt * v0);
  float2 s = v1 * dt;
  float m = length(s);
  if (m > PF_STEP_CAP) s *= PF_STEP_CAP / m;
  return q + s;
}

#endif // PHASE_FOLD_FIELD_HLSL
