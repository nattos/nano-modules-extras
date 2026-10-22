// test_glide.cpp — goldens for sketch/glide.h, the stiff-spring legato a
// discrete change (Lantern's layouts) glides through. Host-free: driven at an
// exact dt with no wasm bundle, no executor and no GPU.
//
// The brief these pin:
//
//   * it overshoots visibly and fast — the first swing past the goal peaks
//     ~50 ms in at the default 10 Hz, well past halfway;
//   * it rings out by `time`: the envelope is down to kSettle (2%) there;
//   * the step is exact — frame rate doesn't change the curve;
//   * too soft a spring for its time creeps in without overshooting;
//   * a retarget mid-glide keeps the velocity and the picture (legato);
//   * the glitch: stagger and skew hold channels still before they let go,
//     a stutter freezes what is shown while the rest moves on, and none of it
//     happens at glitch 0;
//   * smoothing calms it — a smaller overshoot, softer turns, no new
//     wobble — and still lands; 0 is off exactly;
//   * carry: a goal's move put through the spring holds the picture still
//     that instant and twangs after it, overshooting; a slow drive only
//     lags; k 0 follows at once;
//   * kick: a knock of A x 2π stiffness swings out nearly A and rings out;
//   * matchNearest pairs nearest first and leaves the rest over.

#include "sketch/glide.h"

#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

#include <cmath>
#include <vector>

using Catch::Matchers::WithinAbs;
using namespace glide;

namespace {

constexpr float kDt = 1.0f / 240.0f;

// One channel released from offset 1 at rest; e(t) sampled every kDt.
std::vector<float> ringOut(const Params& p, float seconds, float dt = kDt) {
  Element<1> el;
  const float one = 1.0f;
  el.retargetTo(&one, Glitch{});
  std::vector<float> e;
  for (float t = 0.0f; t < seconds; t += dt) {
    el.step(dt, p, Glitch{});
    e.push_back(el.out[0]);
  }
  return e;
}

}  // namespace

TEST_CASE("glide overshoots visibly, within the first ~50 ms", "[glide]") {
  const auto e = ringOut(Params{}, 0.2f);
  int imin = 0;
  for (int i = 0; i < (int)e.size(); i++)
    if (e[i] < e[imin]) imin = i;
  const float tmin = (imin + 1) * kDt;
  CHECK(e[imin] < -0.5f);              // well past the goal
  CHECK(tmin > 0.04f);
  CHECK(tmin < 0.06f);
  // ...and back over to the start's side by ~100 ms.
  CHECK(e[(int)(0.1f / kDt)] > 0.2f);
}

TEST_CASE("glide rings out by its time", "[glide]") {
  for (float time : {0.2f, 0.4f, 1.0f}) {
    Params p;
    p.time = time;
    const auto e = ringOut(p, time * 2.5f);
    float late = 0.0f, early = 0.0f;
    for (int i = 0; i < (int)e.size(); i++) {
      const float t = (i + 1) * kDt;
      if (t >= time) late = std::fmax(late, std::fabs(e[i]));
      else if (t >= 0.5f * time) early = std::fmax(early, std::fabs(e[i]));
    }
    INFO("time " << time << " early " << early << " late " << late);
    CHECK(late < kSettle * 1.05f);
    CHECK(early > kSettle * 2.0f);   // still visibly moving halfway there
  }
}

TEST_CASE("glide's step is exact: the frame rate doesn't change the curve", "[glide]") {
  const auto fine = ringOut(Params{}, 0.25f, 1.0f / 240.0f);
  const auto coarse = ringOut(Params{}, 0.25f, 1.0f / 30.0f);
  for (int k = 1; k <= 7; k++)   // the coarse samples, against the fine at the same time
    CHECK_THAT(coarse[k - 1], WithinAbs(fine[k * 8 - 1], 1e-4));
}

TEST_CASE("glide: too soft a spring creeps in without overshooting", "[glide]") {
  Params p;
  p.stiffness = 1.0f;   // below ln(50) / (2π 0.4) ≈ 1.56 Hz
  const auto e = ringOut(p, 2.0f);
  for (float x : e) CHECK(x > -1e-5f);
  CHECK(e.back() < 0.05f);
}

TEST_CASE("glide: a retarget mid-glide keeps the velocity and the picture", "[glide]") {
  Element<2> el;
  const float start[2] = {1.0f, -0.5f};
  el.retargetTo(start, Glitch{});
  for (int i = 0; i < 7; i++) el.step(kDt, Params{}, Glitch{});
  const float v0 = el.v[0], v1 = el.v[1];
  const float o0 = el.out[0], o1 = el.out[1];
  CHECK(std::fabs(v0) > 1.0f);
  const float jump[2] = {0.3f, 0.2f};
  el.retarget(jump, Glitch{});
  CHECK(el.v[0] == v0);
  CHECK(el.v[1] == v1);
  CHECK_THAT(el.out[0], WithinAbs(o0 - 0.3f, 1e-6));
  CHECK_THAT(el.out[1], WithinAbs(o1 - 0.2f, 1e-6));
  CHECK(el.age == 0.0f);
}

TEST_CASE("glide: a goal that moves every frame still gets glided to", "[glide]") {
  // Retargeted every frame (a wired Layout sweeping), it must not restart its
  // stagger each time and so never let go.
  const Glitch g = Glitch::from(1.0f, Params{});
  Element<2> el;
  el.seed = 3;
  const float start[2] = {1.0f, -1.0f};
  el.retargetTo(start, g);
  const float none[2] = {0.0f, 0.0f};
  for (int i = 0; i < 60; i++) {
    el.step(1.0f / 60.0f, Params{}, g);
    el.retarget(none, g);
  }
  CHECK(std::fabs(el.e[0]) < 0.05f);
  CHECK(std::fabs(el.e[1]) < 0.05f);
}

TEST_CASE("glide: stagger and skew hold channels still before they let go", "[glide]") {
  Glitch g;
  g.stagger = 0.1f;
  g.skew = 0.05f;
  Element<3> el;
  el.seed = 7;
  const float start[3] = {1.0f, 1.0f, 1.0f};
  el.retargetTo(start, g);
  float wmin = 1e9f, wmax = 0.0f;
  for (float w : el.wait) { wmin = std::fmin(wmin, w); wmax = std::fmax(wmax, w); }
  CHECK(wmax <= 0.15f + 1e-6f);
  CHECK(wmax > wmin);                  // the channels let go apart
  // Every channel holds exactly still until its own wait is up.
  float t = 0.0f;
  while (t + kDt < wmin) {
    el.step(kDt, Params{}, g);
    t += kDt;
    for (float e : el.e) CHECK(e == 1.0f);
  }
  for (int i = 0; i < 240; i++) el.step(kDt, Params{}, g);
  for (float e : el.e) CHECK(std::fabs(e) < 0.05f);
}

TEST_CASE("glide: a stutter freezes the picture while the spring moves on", "[glide]") {
  Glitch g = Glitch::from(1.0f, Params{});
  g.stagger = g.skew = 0.0f;
  int frozen = 0, late_frozen = 0;
  for (uint32_t seed = 1; seed <= 32; seed++) {
    Element<1> el;
    el.seed = seed;
    const float one = 1.0f;
    el.retargetTo(&one, g);
    for (float t = 0.0f; t < 0.4f; t += 1.0f / 60.0f) {
      el.step(1.0f / 60.0f, Params{}, g);
      if (el.out[0] != el.e[0]) {
        frozen++;
        if (t > 0.6f * 0.4f) late_frozen++;
      }
    }
  }
  CHECK(frozen > 20);        // it stutters...
  CHECK(late_frozen == 0);   // ...only while it is energetic

  // Glitch 0: what is shown is always the spring.
  Element<1> el;
  const float one = 1.0f;
  el.retargetTo(&one, Glitch::from(0.0f, Params{}));
  for (int i = 0; i < 60; i++) {
    el.step(1.0f / 60.0f, Params{}, Glitch::from(0.0f, Params{}));
    CHECK(el.out[0] == el.e[0]);
  }
}

TEST_CASE("glide: the same seed glitches the same way", "[glide]") {
  const Glitch g = Glitch::from(0.7f, Params{});
  auto run = [&](uint32_t seed) {
    Element<3> el;
    el.seed = seed;
    const float s[3] = {1.0f, -2.0f, 0.5f};
    el.retargetTo(s, g);
    std::vector<float> out;
    for (int i = 0; i < 30; i++) {
      el.step(1.0f / 60.0f, Params{}, g);
      out.insert(out.end(), el.out, el.out + 3);
    }
    return out;
  };
  CHECK(run(5) == run(5));
  CHECK(run(5) != run(6));
}

TEST_CASE("glide: smoothing calms the overshoot and still lands", "[glide]") {
  Params raw;
  Params calm;
  calm.smooth = 0.03f;
  const auto a = ringOut(raw, 1.0f);
  const auto b = ringOut(calm, 1.0f);
  float amin = 0.0f, bmin = 0.0f, ajerk = 0.0f, bjerk = 0.0f;
  for (int i = 0; i < (int)a.size(); i++) {
    amin = std::fmin(amin, a[i]);
    bmin = std::fmin(bmin, b[i]);
    if (i >= 2) {   // the sharpest turn: the largest second difference
      ajerk = std::fmax(ajerk, std::fabs(a[i] - 2 * a[i - 1] + a[i - 2]));
      bjerk = std::fmax(bjerk, std::fabs(b[i] - 2 * b[i - 1] + b[i - 2]));
    }
  }
  INFO("overshoot raw " << amin << " smoothed " << bmin);
  CHECK(bmin > 0.5f * amin);          // under half the overshoot...
  CHECK(bmin < -0.05f);               // ...but still a visible one
  CHECK(bjerk < 0.5f * ajerk);        // and softer turns
  CHECK(std::fabs(b.back()) < 1e-3f); // it still lands
  // The follower itself never overshoots: a held step stays one-sided.
  Element<1> el;
  const float one = 1.0f;
  el.retargetTo(&one, Glitch{});
  Params hold = calm;
  hold.stiffness = 0.5f;              // a spring too soft to swing back
  float lo = 1.0f;
  for (int i = 0; i < 480; i++) {
    el.step(kDt, hold, Glitch{});
    lo = std::fmin(lo, el.out[0]);
  }
  CHECK(lo > -1e-5f);

  // 0 is exactly off.
  Params off;
  off.smooth = 0.0f;
  CHECK(ringOut(off, 0.3f) == ringOut(Params{}, 0.3f));
}

TEST_CASE("glide: Time 0 is a cut", "[glide]") {
  Params p;
  p.time = 0.0f;
  Element<2> el;
  const float s[2] = {1.0f, 2.0f};
  el.retargetTo(s, Glitch{});
  el.step(1.0f / 60.0f, p, Glitch{});
  CHECK(el.out[0] == 0.0f);
  CHECK(el.out[1] == 0.0f);
  CHECK_FALSE(el.active(1e-6f));
}

TEST_CASE("glide: carry twangs after a sudden move and only lags a slow one", "[glide]") {
  const Params p;
  // A goal that steps from 0 to 1, carried in full: shown = goal + out.
  Element<1> el;
  const float jump = 1.0f;
  el.carry(&jump, 1.0f);
  CHECK(el.out[0] == -1.0f);           // the picture hasn't moved yet
  float peak = 0.0f;
  for (int i = 0; i < (int)(0.5f / kDt); i++) {
    el.step(kDt, p, Glitch{});
    peak = std::fmax(peak, 1.0f + el.out[0]);
  }
  CHECK(peak > 1.4f);                  // it twangs past the new goal...
  CHECK(std::fabs(el.out[0]) < 0.02f); // ...and lands on it
  // k 0: no spring at all.
  Element<1> none;
  none.carry(&jump, 0.0f);
  CHECK(none.out[0] == 0.0f);
  CHECK_FALSE(none.active(1e-6f));
  // A ramp over a second, carried a frame at a time: it only lags, a little.
  Element<1> ramp;
  float worst = 0.0f;
  const float step = kDt / 1.0f;
  for (int i = 0; i < (int)(1.0f / kDt); i++) {
    ramp.carry(&step, 1.0f);
    ramp.step(kDt, p, Glitch{});
    worst = std::fmax(worst, std::fabs(ramp.out[0]));
  }
  // (The worst is the ramp's start, itself a kink: rate / 2π stiffness.)
  INFO("ramp lag " << worst << ", at the end " << ramp.out[0]);
  CHECK(worst < 0.02f);
  CHECK(std::fabs(ramp.out[0]) < 0.008f);   // steady: 2σ/ω² x the rate
}

TEST_CASE("glide: a kick swings out nearly its amplitude and rings out", "[glide]") {
  const Params p;
  Element<1> el;
  const float A = 0.2f, dv = A * kTau * p.stiffness;
  el.kick(&dv);
  float peak = 0.0f;
  for (int i = 0; i < (int)(p.time / kDt); i++) {
    el.step(kDt, p, Glitch{});
    peak = std::fmax(peak, std::fabs(el.out[0]));
  }
  INFO("peak " << peak);
  CHECK(peak > 0.7f * A);
  CHECK(peak < 1.0f * A);
  CHECK(std::fabs(el.out[0]) < kSettle * A);
}

TEST_CASE("matchNearest pairs nearest first and leaves the rest over", "[glide]") {
  int to[8], from[8], best[8];
  float bestc[8];

  SECTION("a shuffled set finds itself") {
    const float a[5] = {0.0f, 1.0f, 2.0f, 3.0f, 4.0f};
    const float b[5] = {3.1f, 0.1f, 4.1f, 1.1f, 2.1f};
    matchNearest(5, 5, [&](int i, int j) { return std::fabs(a[i] - b[j]); }, to, from, best, bestc);
    CHECK(to[0] == 1);
    CHECK(to[1] == 3);
    CHECK(to[2] == 4);
    CHECK(to[3] == 0);
    CHECK(to[4] == 2);
    for (int j = 0; j < 5; j++) CHECK(to[from[j]] == j);
  }
  SECTION("the closest pair goes first, the far one is left over") {
    const float a[2] = {0.0f, 0.4f};
    const float b[1] = {0.5f};
    matchNearest(2, 1, [&](int i, int j) { return std::fabs(a[i] - b[j]); }, to, from, best, bestc);
    CHECK(to[0] == -1);
    CHECK(to[1] == 0);
    CHECK(from[0] == 1);
  }
  SECTION("more targets than sources: the far targets are left over") {
    const float a[1] = {1.0f};
    const float b[3] = {5.0f, 1.2f, -3.0f};
    matchNearest(1, 3, [&](int i, int j) { return std::fabs(a[i] - b[j]); }, to, from, best, bestc);
    CHECK(to[0] == 1);
    CHECK(from[0] == -1);
    CHECK(from[2] == -1);
  }
  SECTION("a forbidden pair is never made") {
    const float a[2] = {0.0f, 10.0f};
    const float b[2] = {0.1f, 10.1f};
    matchNearest(2, 2,
                 [&](int i, int j) { return i == 1 ? (float)INFINITY : std::fabs(a[i] - b[j]); },
                 to, from, best, bestc);
    CHECK(to[0] == 0);
    CHECK(to[1] == -1);
    CHECK(from[1] == -1);
  }
}
