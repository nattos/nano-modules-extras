/*
 * spectral_morph.h — C++ port of the default-on path of nano-lfo's
 * lfo_spectral_morph.ts. Pure CPU math, no GPU, no host deps.
 *
 * Only the passes that are ON by the web app's defaults are ported:
 *   blendSpectra (Lanczos sigma + phase-coherence) + geometricStraighten + clamp.
 * The default-OFF passes (straightenWobbles, deringFeatures) are intentionally
 * omitted; add them here if those toggles are ever exposed.
 */
#ifndef SPECTRAL_MORPH_H
#define SPECTRAL_MORPH_H

#include <cmath>
#include <vector>
#include <algorithm>

namespace spectral_lfo {

static const int SPEC_N = 2048;

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

// ─── FFT (radix-2 in-place) — port of fft() ────────────────────────────
inline void fft(double* re, double* im, int N, bool inverse) {
  for (int i = 1, j = 0; i < N; i++) {
    int bit = N >> 1;
    for (; j & bit; bit >>= 1) j ^= bit;
    j ^= bit;
    if (i < j) { std::swap(re[i], re[j]); std::swap(im[i], im[j]); }
  }
  const double sign = inverse ? 1.0 : -1.0;
  for (int len = 2; len <= N; len <<= 1) {
    const int half = len >> 1;
    const double angle = (sign * 2.0 * M_PI) / len;
    const double wRe = std::cos(angle), wIm = std::sin(angle);
    for (int i = 0; i < N; i += len) {
      double curRe = 1.0, curIm = 0.0;
      for (int j = 0; j < half; j++) {
        const int a = i + j, b = a + half;
        const double tRe = curRe * re[b] - curIm * im[b];
        const double tIm = curRe * im[b] + curIm * re[b];
        re[b] = re[a] - tRe;
        im[b] = im[a] - tIm;
        re[a] += tRe;
        im[a] += tIm;
        const double nextRe = curRe * wRe - curIm * wIm;
        curIm = curRe * wIm + curIm * wRe;
        curRe = nextRe;
      }
    }
  }
  if (inverse) {
    for (int i = 0; i < N; i++) { re[i] /= N; im[i] /= N; }
  }
}

// ─── Curve evaluation — port of evaluateCurve() ────────────────────────
// Rasterize control points (Serum power-curve easing) to `n` samples.
inline void evaluateCurve(const float* px, const float* py, const float* pf,
                          int np, float* out, int n) {
  if (np < 2) { for (int i = 0; i < n; i++) out[i] = 0.5f; return; }
  for (int s = 0; s < n; s++) {
    const double t = (double)s / (n - 1);
    int seg = 0;
    for (int i = 0; i < np - 1; i++) {
      if (px[i] <= t) seg = i;
    }
    const int i0 = seg;
    const int i1 = std::min(seg + 1, np - 1);
    const double dx = px[i1] - px[i0];
    if (dx < 1e-10) { out[s] = py[i0]; continue; }
    double lt = (t - px[i0]) / dx;
    if (lt < 0) lt = 0; else if (lt > 1) lt = 1;
    if (std::fabs(pf[i0] - 0.5) > 0.001) {
      const double power = std::pow(2.0, 2.0 * (1.0 - pf[i0]) - 1.0);
      lt = lt > 0 ? std::pow(lt, power) : 0.0;
    }
    out[s] = (float)(py[i0] + (py[i1] - py[i0]) * lt);
  }
}

// ─── curveToSpectrum ───────────────────────────────────────────────────
inline void curveToSpectrum(const float* curve, double* mag, double* phase) {
  std::vector<double> re(SPEC_N), im(SPEC_N, 0.0);
  for (int i = 0; i < SPEC_N; i++) re[i] = curve[i];
  fft(re.data(), im.data(), SPEC_N, false);
  for (int i = 0; i < SPEC_N; i++) {
    mag[i] = std::sqrt(re[i] * re[i] + im[i] * im[i]);
    phase[i] = std::atan2(im[i], re[i]);
  }
}

// ─── blendSpectra — barycentric spectral blend + IFFT ──────────────────
// Port of blendSpectra(). `sigma`/`phaseCoherence` in [0,1].
inline void blendSpectra(const double* const mags[3], const double* const phases[3],
                         const double weights[3], double sigma, double phaseCoherence,
                         float* out) {
  std::vector<double> re(SPEC_N), im(SPEC_N);
  const int half = SPEC_N / 2;
  for (int k = 0; k < SPEC_N; k++) {
    const double mag = weights[0] * mags[0][k]
                     + weights[1] * mags[1][k]
                     + weights[2] * mags[2][k];
    double pRe = 0, pIm = 0;
    for (int v = 0; v < 3; v++) {
      pRe += weights[v] * std::cos(phases[v][k]);
      pIm += weights[v] * std::sin(phases[v][k]);
    }
    const double ph = std::atan2(pIm, pRe);

    double sf = 1.0;
    if (sigma > 0 && k > 0) {
      const int kn = k <= half ? k : SPEC_N - k;
      const double x = M_PI * kn / half;
      const double sinc = std::sin(x) / x;
      sf = 1.0 - sigma * (1.0 - sinc);
    }
    if (phaseCoherence > 0 && k > 0) {
      const double coherence = std::sqrt(pRe * pRe + pIm * pIm);
      sf *= 1.0 - phaseCoherence * (1.0 - coherence);
    }
    re[k] = sf * mag * std::cos(ph);
    im[k] = sf * mag * std::sin(ph);
  }
  fft(re.data(), im.data(), SPEC_N, true);
  for (int i = 0; i < SPEC_N; i++) out[i] = (float)re[i];
}

// ─── Geometric straightening — port of geometricStraighten() ───────────

inline void boxSmooth(const float* curve, int N, int w, std::vector<float>& out) {
  std::vector<double> prefix(N + 1, 0.0);
  for (int i = 0; i < N; i++) prefix[i + 1] = prefix[i] + curve[i];
  out.resize(N);
  for (int i = 0; i < N; i++) {
    const int lo = std::max(0, i - w);
    const int hi = std::min(N, i + w + 1);
    out[i] = (float)((prefix[hi] - prefix[lo]) / (hi - lo));
  }
}

// Tuning knobs (geometric straightening).
static const int    GEO_SMOOTH_W     = 12;
static const double GEO_EXTREMA_TOL  = 0.015;
static const double GEO_SLOPE_THRESH = 0.25;
static const double GEO_STEP_THRESH  = 0.08;
static const int    GEO_STEP_SPAN    = 12;
static const int    GEO_MIN_NODE_DIST = 16;

inline const std::vector<double>& candidatePowers() {
  static std::vector<double> ps = [] {
    std::vector<double> v;
    for (int k = 0; k <= 28; k++) {            // f = 0.15 .. 0.85 step 0.025
      const double f = 0.15 + k * 0.025;
      v.push_back(std::pow(2.0, 2.0 * (1.0 - f) - 1.0));
    }
    std::sort(v.begin(), v.end());
    return v;
  }();
  return ps;
}

inline double fitSegmentPower(const float* curve, int a, int b, double ya, double yb) {
  const int len = b - a;
  if (len < 4) return 1.0;
  const double dy = yb - ya;
  if (std::fabs(dy) < 1e-6) return 1.0;
  double bestPower = 1.0, bestErr = 1e300;
  for (double power : candidatePowers()) {
    double err = 0;
    for (int i = a; i <= b; i++) {
      const double t = (double)(i - a) / len;
      const double predicted = ya + dy * (t > 0 ? std::pow(t, power) : 0.0);
      const double d = curve[i] - predicted;
      err += d * d;
    }
    if (err < bestErr) { bestErr = err; bestPower = power; }
  }
  return bestPower;
}

// Detect structural nodes (extrema / step edges / slope breaks + endpoints).
inline void detectNodes(const float* curve, int N, std::vector<int>& nodesOut) {
  std::vector<unsigned char> mark(N, 0);
  mark[0] = 1; mark[N - 1] = 1;

  std::vector<float> smooth;
  boxSmooth(curve, N, GEO_SMOOTH_W, smooth);

  // Local extrema with prominence filter.
  for (int i = 1; i < N - 1; i++) {
    const bool isPeak   = smooth[i] >= smooth[i - 1] && smooth[i] >= smooth[i + 1] && smooth[i] > smooth[i - 1] + 1e-6f;
    const bool isValley = smooth[i] <= smooth[i - 1] && smooth[i] <= smooth[i + 1] && smooth[i] < smooth[i - 1] - 1e-6f;
    if (!isPeak && !isValley) continue;
    double leftDepth = 0, rightDepth = 0;
    if (isPeak) {
      double minL = smooth[i];
      for (int j = i - 1; j >= 0; j--) { minL = std::min(minL, (double)smooth[j]); if (smooth[j] > smooth[i]) break; }
      leftDepth = smooth[i] - minL;
      double minR = smooth[i];
      for (int j = i + 1; j < N; j++) { minR = std::min(minR, (double)smooth[j]); if (smooth[j] > smooth[i]) break; }
      rightDepth = smooth[i] - minR;
    } else {
      double maxL = smooth[i];
      for (int j = i - 1; j >= 0; j--) { maxL = std::max(maxL, (double)smooth[j]); if (smooth[j] < smooth[i]) break; }
      leftDepth = maxL - smooth[i];
      double maxR = smooth[i];
      for (int j = i + 1; j < N; j++) { maxR = std::max(maxR, (double)smooth[j]); if (smooth[j] < smooth[i]) break; }
      rightDepth = maxR - smooth[i];
    }
    if (std::min(leftDepth, rightDepth) >= GEO_EXTREMA_TOL) mark[i] = 1;
  }

  // Step edges.
  for (int i = GEO_STEP_SPAN; i < N - GEO_STEP_SPAN; i++) {
    const double jump = std::fabs(smooth[i + GEO_STEP_SPAN] - smooth[i - GEO_STEP_SPAN]);
    if (jump > GEO_STEP_THRESH) {
      double maxSlope = 0; int maxJ = i;
      for (int j = i - GEO_STEP_SPAN; j < i + GEO_STEP_SPAN; j++) {
        const double s = std::fabs(smooth[std::min(N - 1, j + 1)] - smooth[std::max(0, j)]);
        if (s > maxSlope) { maxSlope = s; maxJ = j; }
      }
      mark[std::max(0, maxJ - 2)] = 1;
      mark[std::min(N - 1, maxJ + 2)] = 1;
    }
  }

  // Slope-break points.
  std::vector<float> d1(N, 0.0f);
  for (int i = 1; i < N; i++) d1[i] = smooth[i] - smooth[i - 1];
  std::vector<float> d1smooth;
  boxSmooth(d1.data(), N, GEO_SMOOTH_W, d1smooth);
  for (int i = GEO_SMOOTH_W + 1; i < N - GEO_SMOOTH_W - 1; i++) {
    const double slopeChange = std::fabs(d1smooth[i + 1] - d1smooth[i - 1]);
    if (slopeChange > GEO_SLOPE_THRESH / N) mark[i] = 1;
  }

  // Gather sorted, enforce min distance, always include last sample.
  nodesOut.clear();
  for (int i = 0; i < N; i++) {
    if (!mark[i]) continue;
    if (nodesOut.empty()) { nodesOut.push_back(i); continue; }
    if (i - nodesOut.back() >= GEO_MIN_NODE_DIST) nodesOut.push_back(i);
  }
  if (nodesOut.empty() || nodesOut.back() != N - 1) nodesOut.push_back(N - 1);
}

inline void geometricStraighten(const float* spectral, int N, float* out, double strength) {
  if (strength <= 0) { for (int i = 0; i < N; i++) out[i] = spectral[i]; return; }
  std::vector<int> nodes;
  detectNodes(spectral, N, nodes);
  std::vector<float> eased(N, 0.0f);
  for (size_t seg = 0; seg + 1 < nodes.size(); seg++) {
    const int a = nodes[seg], b = nodes[seg + 1];
    const double ya = spectral[a], yb = spectral[b];
    const int len = b - a;
    const double power = fitSegmentPower(spectral, a, b, ya, yb);
    const double dy = yb - ya;
    for (int i = a; i <= b; i++) {
      const double t = (double)(i - a) / len;
      eased[i] = (float)(ya + dy * (t > 0 ? std::pow(t, power) : 0.0));
    }
  }
  for (int i = 0; i < N; i++) out[i] = (float)(spectral[i] * (1 - strength) + eased[i] * strength);
}

// ─── Barycentric coordinates — port of barycentric() ───────────────────
// Returns false on degenerate triangle.
inline bool barycentric(double px, double py,
                        double ax, double ay, double bx, double by, double cx, double cy,
                        double out[3]) {
  const double v0x = bx - ax, v0y = by - ay;
  const double v1x = cx - ax, v1y = cy - ay;
  const double v2x = px - ax, v2y = py - ay;
  const double d00 = v0x * v0x + v0y * v0y;
  const double d01 = v0x * v1x + v0y * v1y;
  const double d11 = v1x * v1x + v1y * v1y;
  const double d20 = v2x * v0x + v2y * v0y;
  const double d21 = v2x * v1x + v2y * v1y;
  const double denom = d00 * d11 - d01 * d01;
  if (std::fabs(denom) < 1e-12) return false;
  const double v = (d11 * d20 - d01 * d21) / denom;
  const double w = (d00 * d21 - d01 * d20) / denom;
  out[0] = 1 - v - w; out[1] = v; out[2] = w;
  return true;
}

} // namespace spectral_lfo

#endif // SPECTRAL_MORPH_H
