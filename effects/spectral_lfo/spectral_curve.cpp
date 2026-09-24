/*
 * spectral_curve.cpp — out-of-line spectral-morph curve generation + the atlas.
 *
 * The ONLY translation unit that includes spectral_lfo_atlas.h, so the ~1.2MB of
 * baked atlas data exists exactly once no matter how many effects consume the
 * curve (mod.source.spectral_lfo, mod.shaper.spectral, ...). See spectral_curve.h.
 */

#include "spectral_curve.h"
#include "spectral_lfo_atlas.h"

#include <algorithm>
#include <cmath>

namespace spectral_lfo {

static inline float clampf(float v, float lo, float hi) {
  return v < lo ? lo : (v > hi ? hi : v);
}

// ─── Triangle lookup (port of findTriangle) ────────────────────────────
struct TriHit { int verts[3]; double weights[3]; };

static bool findTriangle(int metric, double tx, double ty, TriHit& hit) {
  const float* coords = SL_COORDS[metric];
  const int*   tris   = SL_TRIS[metric];
  const int*   toData = SL_TRITODATA[metric];
  const int    ntris  = SL_NTRIS[metric];
  for (int t = 0; t < ntris; t++) {
    const int a = tris[t * 3], b = tris[t * 3 + 1], c = tris[t * 3 + 2];
    double bary[3];
    if (!barycentric(tx, ty,
                     coords[a * 2], coords[a * 2 + 1],
                     coords[b * 2], coords[b * 2 + 1],
                     coords[c * 2], coords[c * 2 + 1], bary))
      continue;
    if (bary[0] >= -0.001 && bary[1] >= -0.001 && bary[2] >= -0.001) {
      const double w0 = std::max(0.0, bary[0]);
      const double w1 = std::max(0.0, bary[1]);
      const double w2 = std::max(0.0, bary[2]);
      const double sum = w0 + w1 + w2;
      hit.verts[0] = toData[a]; hit.verts[1] = toData[b]; hit.verts[2] = toData[c];
      hit.weights[0] = w0 / sum; hit.weights[1] = w1 / sum; hit.weights[2] = w2 / sum;
      return true;
    }
  }
  return false;
}

// Nearest real data point — fallback when no triangle contains the query.
static int nearestData(int metric, double tx, double ty) {
  const float* coords = SL_COORDS[metric];
  double best = 1e300; int bestIdx = 0;
  for (int i = 0; i < SL_NUM_ENTRIES; i++) {
    const double dx = coords[i * 2] - tx, dy = coords[i * 2 + 1] - ty;
    const double d = dx * dx + dy * dy;
    if (d < best) { best = d; bestIdx = i; }
  }
  return bestIdx;
}

// Rasterize a single shape (entry) into `out` (SPEC_N samples).
static void evalEntry(int entry, float* out) {
  const int off = SL_ENTRY_OFFSET[entry];
  const int np  = SL_ENTRY_NCP[entry];
  evaluateCurve(SL_CP_X + off, SL_CP_Y + off, SL_CP_F + off, np, out, SPEC_N);
}

void computeCurve(int metric, double ex, double ey, bool interp, float* out) {
  TriHit hit;
  if (!findTriangle(metric, ex, ey, hit)) {
    const int n = nearestData(metric, ex, ey);
    hit.verts[0] = hit.verts[1] = hit.verts[2] = n;
    hit.weights[0] = 1.0; hit.weights[1] = hit.weights[2] = 0.0;
  }

  if (!interp) {
    // Snap to the dominant (max-weight) shape.
    int best = 0;
    if (hit.weights[1] > hit.weights[best]) best = 1;
    if (hit.weights[2] > hit.weights[best]) best = 2;
    evalEntry(hit.verts[best], out);
    return;
  }

  // Spectral morph of the 3 shapes.
  static float c0[SPEC_N], c1[SPEC_N], c2[SPEC_N];
  static double m0[SPEC_N], p0[SPEC_N], m1[SPEC_N], p1[SPEC_N], m2[SPEC_N], p2[SPEC_N];
  evalEntry(hit.verts[0], c0);
  evalEntry(hit.verts[1], c1);
  evalEntry(hit.verts[2], c2);
  curveToSpectrum(c0, m0, p0);
  curveToSpectrum(c1, m1, p1);
  curveToSpectrum(c2, m2, p2);
  const double* mags[3]   = { m0, m1, m2 };
  const double* phases[3] = { p0, p1, p2 };

  float raw[SPEC_N];
  // Web defaults: sigma=0, phaseCoherence=1, geoStraighten=1.
  blendSpectra(mags, phases, hit.weights, /*sigma=*/0.0, /*phaseCoherence=*/1.0, raw);
  geometricStraighten(raw, SPEC_N, out, /*strength=*/1.0);
  for (int i = 0; i < SPEC_N; i++) out[i] = clampf(out[i], 0.0f, 1.0f);
}

void ensureCurve(CurveCache& c, int metric, float ex, float ey, bool interp) {
  if (c.valid && ex == c.x && ey == c.y && metric == c.metric && interp == c.interp) return;
  computeCurve(metric, ex, ey, interp, c.curve);
  c.valid = true; c.x = ex; c.y = ey; c.metric = metric; c.interp = interp;
}

float sampleCurveAt(const float* curve, double phase) {
  const double p = phase * SPEC_N;
  int i0 = (int)std::floor(p);
  const double frac = p - i0;
  i0 = ((i0 % SPEC_N) + SPEC_N) % SPEC_N;
  const int i1 = (i0 + 1) % SPEC_N;
  return (float)(curve[i0] + (curve[i1] - curve[i0]) * frac);
}

}  // namespace spectral_lfo
