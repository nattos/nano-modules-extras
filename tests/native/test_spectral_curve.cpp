// test_spectral_curve.cpp — the shared spectral-morph curve generator
// (wasm_modules/spectral_lfo/spectral_curve.{h,cpp}), factored out of
// mod.source.spectral_lfo so mod.shaper.spectral can reuse it. Verifies the curve build is
// in-range, deterministic, position-sensitive, and properly cached.

#include "spectral_curve.h"

#include <catch2/catch_test_macros.hpp>
#include <cmath>

using namespace spectral_lfo;

TEST_CASE("spectral curve build is in range and deterministic", "[spectral_curve]") {
  CurveCache a;
  ensureCurve(a, 0, 0.5f, 0.5f, true);
  REQUIRE(a.valid);
  for (int i = 0; i < SPEC_N; i++) {
    CHECK(a.curve[i] >= -1e-3f);
    CHECK(a.curve[i] <= 1.0f + 1e-3f);
  }
  // Same key → identical curve.
  CurveCache b;
  ensureCurve(b, 0, 0.5f, 0.5f, true);
  bool identical = true;
  for (int i = 0; i < SPEC_N; i++) if (a.curve[i] != b.curve[i]) { identical = false; break; }
  CHECK(identical);
}

TEST_CASE("different manifold positions give different curves", "[spectral_curve]") {
  CurveCache a, b;
  ensureCurve(a, 0, 0.2f, 0.2f, true);
  ensureCurve(b, 0, 0.8f, 0.8f, true);
  bool differ = false;
  for (int i = 0; i < SPEC_N; i++)
    if (std::fabs(a.curve[i] - b.curve[i]) > 1e-3f) { differ = true; break; }
  CHECK(differ);
}

TEST_CASE("ensureCurve only recomputes when the key changes", "[spectral_curve]") {
  CurveCache c;
  ensureCurve(c, 0, 0.5f, 0.5f, true);
  c.curve[0] = -999.0f;                 // poison
  ensureCurve(c, 0, 0.5f, 0.5f, true);  // unchanged key → cache hit, no recompute
  CHECK(c.curve[0] == -999.0f);
  ensureCurve(c, 0, 0.6f, 0.5f, true);  // changed x → recompute
  CHECK(c.curve[0] != -999.0f);
}

TEST_CASE("sampleCurveAt interpolates and loops", "[spectral_curve]") {
  CurveCache c;
  ensureCurve(c, 0, 0.5f, 0.5f, true);
  // In range across the index domain.
  for (double p = 0.0; p < 1.0; p += 0.05) {
    const float v = sampleCurveAt(c.curve, p);
    CHECK(v >= -1e-3f);
    CHECK(v <= 1.0f + 1e-3f);
  }
  // index 0 hits sample 0 exactly; wraps at 1.0.
  CHECK(sampleCurveAt(c.curve, 0.0) == c.curve[0]);
  CHECK(sampleCurveAt(c.curve, 1.0) == c.curve[0]);
}
