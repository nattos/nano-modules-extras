#pragma once
/*
 * spectral_curve.h — Shared spectral-morph CURVE generation.
 *
 * Factors the atlas lookup + spectral morph (atlas triangle → FFT → barycentric
 * blend → IFFT → geometric straighten) out of mod.source.spectral_lfo so it can be
 * reused. mod.source.spectral_lfo samples the curve over TIME (a phase accumulator);
 * mod.shaper.spectral samples it at an INPUT value — the same morphed curve becomes a
 * remap envelope. The heavy ~1.2MB atlas lives in ONE translation unit
 * (spectral_curve.cpp), so the second consumer costs no extra data.
 *
 * The inline FFT/morph kernels come from spectral_morph.h; only the curve-build
 * functions + the atlas data are out-of-line here.
 */

#include "spectral_morph.h"   // SPEC_N + the inline morph kernels

namespace spectral_lfo {

constexpr int SPECTRAL_NUM_METRICS = 5;   // mirrors SL_NUM_METRICS in the atlas

// A rasterized curve plus the key it was computed for — recomputed only when
// (x, y, metric, interpolation) change.
struct CurveCache {
  bool  valid = false;
  float x = -1.0f, y = -1.0f;
  int   metric = -1;
  bool  interp = true;
  float curve[SPEC_N];
};

// Build the morphed curve for a manifold position into `out` (SPEC_N samples).
void computeCurve(int metric, double ex, double ey, bool interp, float* out);

// Recompute a cache only when its (x, y, metric, interpolation) key changes.
void ensureCurve(CurveCache& c, int metric, float ex, float ey, bool interp);

// Sample a curve at a normalized index/phase (linear, looping over [0,1)).
float sampleCurveAt(const float* curve, double phase);

}  // namespace spectral_lfo
