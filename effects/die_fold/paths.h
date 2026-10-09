// source.die_fold — closed parametric curves through the atlas, and the atlas
// sample itself. The C++ twin of the prototype's `web/src/paths.ts` (itself the
// twin of `diefold/paths.py`, gated there at max |Δ| 5e-10 over 512 phases)
// and of `web/src/atlas.ts` `sampleAtlas`.
//
// EVERYTHING CLOSES. Harmonics are integers, the phase warp is monotone, and the
// camera path returns to the same super-cell AND the same fraction, so phase 0
// and phase 1 are the same frame. Chaos comes from raising the harmonic order,
// never from letting the curve wander.
//
// The family table, knob ranges, defaults and presets are NOT retyped here: they
// come from the prototype's atlas.json via gen_atlas.py (die_fold_atlas.h).
#pragma once

#include "die_fold_atlas.h"

#include <cmath>
#include <cstdint>

namespace die_fold {

constexpr double kTau = 6.283185307179586;

/** A path: its family plus the DF_NMACRO knobs (M_* indices). */
struct Spec {
  int family = FAM_LISSAJOUS;
  double m[DF_NMACRO] = {};
  double operator[](int i) const { return m[i]; }
};

inline Spec preset_spec(int i) {
  if (i < 0) i = 0;
  if (i >= DF_NPRESET) i = DF_NPRESET - 1;
  Spec s;
  s.family = DF_PRESET_FAMILY[i];
  for (int k = 0; k < DF_NMACRO; k++) s.m[k] = DF_PRESET[i][k];
  return s;
}

// JS Math.round: half up (toward +inf). The prototype's knobs are integers or
// near enough that this only matters for parity, so it is spelled exactly.
inline double jsround(double x) { return std::floor(x + 0.5); }

// --- the same lowbias32 finalizer as the shader -----------------------------
inline uint32_t lb_mix(uint32_t h) {
  h ^= h >> 16;
  h *= 0x7feb352du;
  h ^= h >> 15;
  h *= 0x846ca68bu;
  h ^= h >> 16;
  return h;
}

inline double hashF3(uint32_t a, uint32_t b, uint32_t c) {
  uint32_t h = 0x2545f491u;
  h = lb_mix(h ^ (a * 0x9e3779b1u));
  h = lb_mix(h ^ (b * 0x9e3779b1u));
  h = lb_mix(h ^ (c * 0x9e3779b1u));
  return (double)(h >> 8) * (1.0 / 16777216.0);
}

inline int gcd_i(int a, int b) {
  while (b) { int t = b; b = a % b; a = t; }
  return a;
}

// --- the families: p in [0,1) -> a curve in roughly [-1,1]^2 ----------------

inline void fam_eval(const Spec& s, double p, double& ox, double& oy) {
  switch (s.family) {
    case FAM_LISSAJOUS: {
      double a = jsround(s[M_A]), b = jsround(s[M_B]);
      ox = std::sin(kTau * a * p + kTau * s[M_DELTA]);
      oy = std::sin(kTau * b * p);
      return;
    }
    case FAM_ROSE: {
      // An even-petalled rose needs two turns to close.
      int k = (int)jsround(s[M_K]);
      double turns = (((k % 2) + 2) % 2 == 1) ? 1.0 : 2.0;
      double th = kTau * turns * p;
      double r = s[M_FILL] + (1.0 - s[M_FILL]) * std::cos(k * th);
      ox = r * std::cos(th);
      oy = r * std::sin(th);
      return;
    }
    case FAM_TROCHOID: {
      const double R = 8.0;
      int r = (int)std::fmax(1.0, jsround(s[M_RR]));
      double t = kTau * ((double)r / gcd_i(8, r)) * p;
      double q = (R - r) / r;
      double x = (R - r) * std::cos(t) + s[M_DD] * r * std::cos(q * t);
      double y = (R - r) * std::sin(t) - s[M_DD] * r * std::sin(q * t);
      double sc = std::fmax(std::fabs(R - r) + std::fabs(s[M_DD] * r), 1e-6);
      ox = x / sc;
      oy = y / sc;
      return;
    }
    case FAM_KNOT: {
      double a = jsround(s[M_A]), b = jsround(s[M_B]);
      double t = kTau * p;
      double rad = 2.0 + std::cos(b * t);
      double x = rad * std::cos(a * t);
      double y = rad * std::sin(a * t);
      ox = x / 3.0;
      oy = (y + s[M_DELTA] * x) / 3.0;
      return;
    }
    case FAM_HARMONOGRAPH: {
      int n = (int)jsround(s[M_N_TERMS]);
      double x = 0, y = 0, tot = 0;
      for (int i = 0; i < n; i++) {
        double f = std::fmax(1.0, jsround((i + 1) * (1.0 + s[M_SPREAD] * (i + 1))));
        double g = std::fmax(1.0, jsround((i + 1) * (1.0 + s[M_SPREAD] * (i + 2))));
        double amp = 1.0 / (i + 1);
        x += amp * std::sin(kTau * f * p + kTau * (s[M_PHASE] + 0.13 * i));
        y += amp * std::cos(kTau * g * p + kTau * (s[M_PHASE] * 0.7 + 0.31 * i));
        tot += amp;
      }
      ox = tot > 0 ? x / tot : 0.0;
      oy = tot > 0 ? y / tot : 0.0;
      return;
    }
    default: {   // FAM_FOURIER: hashed phases, a k^-alpha spectrum, exactly closed
      int K = (int)jsround(s[M_KMAX]);
      uint32_t c = (uint32_t)(int32_t)jsround(s[M_CSEED]);
      double x = 0, y = 0;
      for (int k = 1; k <= K; k++) {
        double amp = std::pow((double)k, -s[M_ALPHA]);
        x += amp * std::sin(kTau * k * p + kTau * hashF3(c, (uint32_t)k, 1u));
        y += amp * std::sin(kTau * k * p + kTau * hashF3(c, (uint32_t)k, 2u));
      }
      ox = x;
      oy = y;
      return;
    }
  }
}

/** `fourier` normalizes by its own peak, off a fixed 512-point grid. */
inline double fourier_scale(const Spec& s) {
  double mx = 1e-6;
  for (int i = 0; i < 512; i++) {
    double x, y;
    fam_eval(s, i / 512.0, x, y);
    mx = std::fmax(mx, std::fmax(std::fabs(x), std::fabs(y)));
  }
  return mx;
}

// --- the modifiers ----------------------------------------------------------

/** A monotone phase warp: linger in some places, sweep past others. */
inline double dwell(double p, const Spec& s) {
  double k = jsround(s[M_DWELL_K]);
  if (k < 1 || s[M_DWELL_AMP] <= 0) return p;
  double a = std::fmin(s[M_DWELL_AMP], 0.95);
  double q = p + (a * std::sin(kTau * k * p + kTau * s[M_DWELL_PHASE])) / (kTau * k);
  return q - std::floor(q);
}

/** Fold into [-1,1] rather than clamping. */
inline double reflect(double v) {
  double m = std::fmod(v + 1.0, 4.0);       // JS `%`: sign of the dividend
  m = std::fmod(m + 4.0, 4.0);
  return -(std::fabs(m - 2.0) - 1.0);
}

inline void modify(double x, double y, double p, const Spec& s, double& ox, double& oy) {
  double r = std::hypot(x, y);
  double ang = kTau * s[M_ROT] + s[M_SWIRL] * r;
  double c = std::cos(ang), sn = std::sin(ang);
  double px = c * x - sn * y;
  double py = sn * x + c * y;
  if (std::fabs(s[M_PINCH] - 1.0) > 1e-6) {
    double rr = std::fmax(std::hypot(px, py), 1e-9);
    double f = std::pow(rr, s[M_PINCH]) / rr;
    px *= f;
    py *= f;
  }
  double k = jsround(s[M_WOB_K]);
  if (k >= 1 && s[M_WOB_AMP] > 0) {
    px += s[M_WOB_AMP] * std::cos(kTau * k * p);
    py += s[M_WOB_AMP] * std::sin(kTau * k * p);
  }
  ox = reflect(px * s[M_SCALE] + s[M_CX]);
  oy = reflect(py * s[M_SCALE] + s[M_CY]);
}

// --- the orbit: a warp of the CAMERA, the anchor at the FOCUS ----------------

/** Solve M = E - e sin E: Newton from E0 = M + e sin M, a FIXED 24 times. */
inline double kepler_E(double M, double e) {
  double E = M + e * std::sin(M);
  for (int i = 0; i < 24; i++) E = E - (E - e * std::sin(E) - M) / (1.0 - e * std::cos(E));
  return E;
}

/** -> (dx, dy) in super-cells and dzoom in octaves. orb_span <= 0 is off. */
inline void orbit_offset(const Spec& s, double p, double& dx, double& dy, double& dz) {
  double span = s[M_ORB_SPAN];
  if (!(span > 0)) { dx = dy = dz = 0; return; }
  double e = std::fmin(std::fmax(s[M_ORB_E], 0.0), 0.9);
  double k = std::fmax(1.0, jsround(s[M_ORB_K]));
  double M = std::fmod(p * k, 1.0);
  M -= std::floor(M);
  double E = kepler_E(kTau * M, e);
  double x = std::cos(E) - e;                          // focus at the origin
  double y = std::sqrt(std::fmax(1 - e * e, 0.0)) * std::sin(E);
  double a = kTau * s[M_ORB_ROT];
  double c = std::cos(a), sn = std::sin(a);
  dx = span * (c * x - sn * y);
  dy = span * (sn * x + c * y);
  // The zoom breathes with the radius, or the fall is invisible.
  dz = s[M_ORB_ZOOM] * e * std::cos(E);
}

// --- the path ----------------------------------------------------------------

/** p in [0,1) -> a position on the pad, in [0,1]^2. */
inline void path_xy(const Spec& s, double p, double fscale, double& ox, double& oy) {
  double q = dwell(p, s);
  double x, y;
  fam_eval(s, q, x, y);
  if (s.family == FAM_FOURIER) { x /= fscale; y /= fscale; }
  double mx, my;
  modify(x, y, q, s, mx, my);
  ox = 0.5 * (mx + 1);
  oy = 0.5 * (my + 1);
}

/** Where the camera is on the same phase, in super-cells (and its zoom). The
 *  camera flies the SAME curve, turned, scaled and phase-shifted: one move,
 *  not two. The orbit rides on top. */
inline void path_camera(const Spec& s, double p, double fscale, double base_x, double base_y,
                        double& wx, double& wy, double& zoom) {
  double q = dwell(p, s);
  double ph = std::fmod(q + s[M_CAM_PHASE], 1.0);
  double x, y;
  fam_eval(s, ph, x, y);
  if (s.family == FAM_FOURIER) { x /= fscale; y /= fscale; }
  double a = kTau * s[M_CAM_ROT];
  double c = std::cos(a), sn = std::sin(a);
  double dx, dy, dz;
  orbit_offset(s, p, dx, dy, dz);
  wx = base_x + s[M_CAM_SPAN] * (c * x - sn * y) + dx;
  wy = base_y + s[M_CAM_SPAN] * (sn * x + c * y) + dy;
  double k = jsround(s[M_CAM_ZK]);
  double u = k >= 1 ? 0.5 - 0.5 * std::cos(kTau * k * q) : 0.0;
  zoom = s[M_ZOOM_LO] + (s[M_ZOOM_HI] - s[M_ZOOM_LO]) * u + dz;
}

// --- the atlas ----------------------------------------------------------------

/** Bilinear over the four surrounding cells into the parameter slots of `prm`
 *  (buffer layout), then the pin. Cell centers sit at k/(G-1). The seed is NOT
 *  among the four: one value for the whole atlas, supplied by the caller. */
inline void sample_atlas(double x, double y, float* prm) {
  const int G = DF_GRID, K = DF_NFOLD;
  double fx = std::fmin(std::fmax(x, 0.0), 1.0) * (G - 1);
  double fy = std::fmin(std::fmax(y, 0.0), 1.0) * (G - 1);
  int j0 = (int)std::fmin(std::floor(fx), G - 1);
  int i0 = (int)std::fmin(std::floor(fy), G - 1);
  int j1 = j0 + 1 < G ? j0 + 1 : G - 1;
  int i1 = i0 + 1 < G ? i0 + 1 : G - 1;
  float tx = (float)(fx - j0), ty = (float)(fy - i0);
  const float* b00 = DF_CELLS + (i0 * G + j0) * K;
  const float* b01 = DF_CELLS + (i0 * G + j1) * K;
  const float* b10 = DF_CELLS + (i1 * G + j0) * K;
  const float* b11 = DF_CELLS + (i1 * G + j1) * K;
  for (int k = 0; k < K; k++) {
    float top = b00[k] * (1 - tx) + b01[k] * tx;
    float bot = b10[k] * (1 - tx) + b11[k] * tx;
    prm[DF_FOLD_SLOT[k]] = top * (1 - ty) + bot * ty;
  }
  prm[S_ZOOM] = DF_PIN_ZOOM;
  prm[S_SPAN_CELLS] = DF_PIN_SPAN_CELLS;
}

}  // namespace die_fold
