// test_three_planes_strobe.cpp — goldens for Strobe, the other thing a throw
// can do to `source.mesh.three_planes`.
//
// Host-free: three_planes_strobe.h carries no effect ABI, so a whole tail can
// be run out here at an exact dt with no wasm bundle, no executor and no GPU.
// (The web suite covers what the effect wraps around it — the wireframe look,
// the one-frame hold and the local-contrast damping — because those are
// pixels.)
//
// The brief these pin, in the order the header states it:
//
//   * the arrival: one step with the whole stack lit, then it breaks up;
//   * the roll: strictly one floor at a time, and always a neighbour;
//   * the flam: a quieter grace stroke on the next floor just before it lands;
//   * the putter: the RATE never changes and the WINDOW closes, until the hits
//     start missing frames outright.

#include "sketch/three_planes_strobe.h"

#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

#include <cmath>
#include <vector>

using Catch::Matchers::WithinAbs;
using namespace three_planes_strobe;

namespace {

constexpr float kDt = 1.0f / 120.0f;

/// One frame at a given release. Returns the gains for that frame.
struct Frame { float g[kPlanes]; };

/// How many frames in a run light floor `i` at all, and how bright the
/// brightest of them was.
struct Played { int frames; float peak; };

Frame step(Core& c, Params& p, float release, float dt = kDt) {
  p.release = release;
  c.tick(p, dt);
  Frame f{};
  for (int i = 0; i < kPlanes; ++i) f.g[i] = c.gain[i];
  return f;
}

int litCount(const Frame& f, float thresh = 1e-4f) {
  int n = 0;
  for (int i = 0; i < kPlanes; ++i) if (f.g[i] > thresh) ++n;
  return n;
}

int brightest(const Frame& f) {
  int k = -1;
  float best = 1e-4f;
  for (int i = 0; i < kPlanes; ++i) if (f.g[i] > best) { best = f.g[i]; k = i; }
  return k;
}

/// Run a whole tail: a throw of `charge` falling linearly to 0 over
/// `ring_time`, exactly the way the rig spends it. Hands every frame back.
/// `steps_run` (if given) collects how far the roll got before the release
/// reached 0 and rewound the clock.
std::vector<Frame> tail(Core& c, Params& p, float charge, float ring_time,
                        float dt = kDt, float* steps_run = nullptr) {
  std::vector<Frame> out;
  float rel = charge;
  // The throw itself: the frame the release rises.
  out.push_back(step(c, p, rel, dt));
  while (rel > 0.0f) {
    if (steps_run) *steps_run = c.clock;
    rel -= dt / ring_time;
    if (rel < 0.0f) rel = 0.0f;
    out.push_back(step(c, p, rel, dt));
  }
  return out;
}

/// How many frames in [a, b) have anything lit at all.
int litFrames(const std::vector<Frame>& fs, size_t a, size_t b) {
  int n = 0;
  for (size_t i = a; i < b && i < fs.size(); ++i) if (litCount(fs[i]) > 0) ++n;
  return n;
}

}  // namespace

TEST_CASE("nothing at rest") {
  Core c;
  Params p;
  for (int i = 0; i < 200; ++i) {
    Frame f = step(c, p, 0.0f);
    REQUIRE(litCount(f) == 0);
  }
  // And the clock has not crept, so the next throw starts on the arrival
  // rather than partway through a bounce.
  REQUIRE_THAT(c.clock, WithinAbs(0.0f, 1e-6f));
}

TEST_CASE("the throw arrives as the whole stack") {
  Core c;
  Params p;
  Frame f = step(c, p, 1.0f);
  REQUIRE(litCount(f) == kPlanes);
  for (int i = 0; i < kPlanes; ++i) REQUIRE_THAT(f.g[i], WithinAbs(1.0f, 1e-6f));
}

TEST_CASE("the arrival lasts exactly one step, whatever the duty") {
  // It is the hit the tail hangs off, so it is NOT cut short by the window —
  // a duty that would leave half a frame of it would leave a dropout.
  Params p;
  p.rate = 20.0f;
  p.duty = 0.05f;   // far shorter than a step
  Core c;

  const float step_s = 1.0f / p.rate;
  float t = 0.0f;
  int full = 0;
  for (int i = 0; i < 40; ++i) {
    Frame f = step(c, p, 1.0f);
    if (t < step_s - kDt) REQUIRE(litCount(f) == kPlanes);
    if (litCount(f) == kPlanes) ++full;
    t += kDt;
  }
  // One step at 20/s is 6 frames at 120fps; the roll takes over right after.
  REQUIRE(full >= 5);
  REQUIRE(full <= 7);
}

TEST_CASE("the roll lights one floor at a time") {
  Params p;
  p.rate = 20.0f;
  p.duty = 0.5f;
  p.grace = 0.0f;   // the flam is tested on its own below
  Core c;

  step(c, p, 1.0f);
  const int arrival = (int)std::ceil(120.0f / p.rate);
  for (int i = 0; i < 400; ++i) {
    Frame f = step(c, p, 1.0f);
    if (i < arrival) continue;              // still the arrival
    REQUIRE(litCount(f) <= 1);
  }
}

TEST_CASE("the roll bounces, so consecutive hits are neighbours") {
  Params p;
  p.rate = 20.0f;
  p.duty = 0.5f;
  p.grace = 0.0f;
  Core c;

  step(c, p, 1.0f);
  const int arrival = (int)std::ceil(120.0f / p.rate);
  int prev = -1;
  int changes = 0;
  for (int i = 0; i < 400; ++i) {
    Frame f = step(c, p, 1.0f);
    if (i < arrival) continue;
    const int k = brightest(f);
    if (k < 0) continue;
    if (prev >= 0 && k != prev) { REQUIRE(std::abs(k - prev) == 1); ++changes; }
    prev = k;
  }
  REQUIRE(changes > 10);   // it really did move around
}

TEST_CASE("every floor gets played") {
  Params p;
  p.rate = 20.0f;
  p.duty = 0.5f;
  p.grace = 0.0f;
  Core c;

  bool seen[kPlanes] = {};
  step(c, p, 1.0f);
  for (int i = 0; i < 400; ++i) {
    Frame f = step(c, p, 1.0f);
    const int k = brightest(f);
    if (k >= 0 && litCount(f) == 1) seen[k] = true;
  }
  for (int i = 0; i < kPlanes; ++i) REQUIRE(seen[i]);
}

TEST_CASE("the flam is a quieter stroke just before the change") {
  Params p;
  p.rate = 12.0f;    // slow enough that a step is many frames
  p.duty = 0.5f;
  p.grace = 0.5f;
  Core c;

  step(c, p, 1.0f);
  const int arrival = (int)std::ceil(120.0f / p.rate);
  for (int i = 0; i < arrival; ++i) step(c, p, 1.0f);

  // Somewhere in the run there is a frame with TWO floors lit at different
  // levels — the grace under the main stroke — and the quiet one is the floor
  // that plays next.
  int graces = 0;
  int prev_main = -1;
  int grace_floor = -1;
  for (int i = 0; i < 400; ++i) {
    Frame f = step(c, p, 1.0f);
    if (litCount(f) == 2) {
      ++graces;
      // exactly one of them is the grace
      int quiet = -1, loud = -1;
      for (int k = 0; k < kPlanes; ++k) {
        if (f.g[k] > 0.9f) loud = k;
        else if (f.g[k] > 1e-4f) quiet = k;
      }
      REQUIRE(loud == -1);              // the main window has already closed
      REQUIRE(quiet >= 0);
    }
    const int k = brightest(f);
    if (k >= 0 && f.g[k] > 0.9f) {
      if (grace_floor >= 0 && k != prev_main) {
        REQUIRE(k == grace_floor);      // the grace named the next floor
        grace_floor = -1;
      }
      prev_main = k;
    } else if (k >= 0) {
      grace_floor = k;
      REQUIRE_THAT(f.g[k], WithinAbs(kGraceGain, 1e-6f));
    }
  }
  REQUIRE(graces == 0);   // grace and main never overlap at this duty
}

TEST_CASE("the grace is wide enough to land at display rates") {
  // Why it is measured against the STEP and not against the on-window: at the
  // default 22 steps/s a 60 Hz frame is more than a third of a step, so a
  // grace scaled by the window too would be half a frame wide and would land
  // on some steps and miss others.
  Params p;
  p.rate = 22.0f;
  p.duty = 0.55f;
  p.grace = 0.35f;
  Core c;

  const float dt60 = 1.0f / 60.0f;
  const int frames = 240;
  step(c, p, 1.0f, dt60);
  int graces = 0;
  for (int i = 0; i < frames; ++i) {
    Frame f = step(c, p, 1.0f, dt60);
    const int k = brightest(f);
    if (k >= 0 && f.g[k] < 0.9f) ++graces;
  }
  const float steps = p.rate * (float)frames * dt60;
  REQUIRE((float)graces > steps * 0.6f);
}

TEST_CASE("no flam at all when the grace is dialled out") {
  Params p;
  p.rate = 12.0f;
  p.duty = 0.5f;
  p.grace = 0.0f;
  Core c;

  step(c, p, 1.0f);
  for (int i = 0; i < 400; ++i) {
    Frame f = step(c, p, 1.0f);
    const int k = brightest(f);
    if (k >= 0) REQUIRE_THAT(f.g[k], WithinAbs(1.0f, 1e-6f));   // never the quiet one
  }
}

TEST_CASE("the rate never changes as the tail runs down") {
  // The decay is the window closing, not the roll slowing. So the number of
  // STEPS in a tail is exactly the rate times its length, whatever the duty is
  // doing.
  Params p;
  p.rate = 20.0f;
  p.duty = 0.9f;
  p.grace = 0.0f;
  Core c;

  const float ring = 1.5f;
  float steps = 0.0f;
  auto fs = tail(c, p, 1.0f, ring, kDt, &steps);
  REQUIRE_THAT(steps, WithinAbs(p.rate * ring, 0.5f));
}

TEST_CASE("the duty putters out") {
  // Split the tail in half and count lit frames. The rate is identical in both
  // halves, so anything the second half loses it lost to the window.
  Params p;
  p.rate = 20.0f;
  p.duty = 0.9f;
  p.grace = 0.0f;
  Core c;

  auto fs = tail(c, p, 1.0f, 2.0f);
  const size_t half = fs.size() / 2;
  const int early = litFrames(fs, 0, half);
  const int late  = litFrames(fs, half, fs.size());
  REQUIRE(early > late * 2);
  // ...and it SPUTTERS at the end rather than settling on a floor: by then the
  // window is a fraction of a frame, so the last tenth of the tail catches
  // barely any of it, and the frame the release finally spends is dark.
  REQUIRE(litFrames(fs, fs.size() * 9 / 10, fs.size()) <= 4);
  REQUIRE(litCount(fs.back()) == 0);
}

TEST_CASE("the hits stay as hard as the first one") {
  // Brightness is deliberately NOT part of the decay: a late hit is as bright
  // as an early one, there are simply fewer of them.
  Params p;
  p.rate = 20.0f;
  p.duty = 0.9f;
  p.grace = 0.0f;
  Core c;

  auto fs = tail(c, p, 1.0f, 2.0f);
  float last_lit = 0.0f;
  for (const auto& f : fs) {
    const int k = brightest(f);
    if (k >= 0) {
      REQUIRE_THAT(f.g[k], WithinAbs(1.0f, 1e-6f));
      last_lit = f.g[k];
    }
  }
  REQUIRE_THAT(last_lit, WithinAbs(1.0f, 1e-6f));
}

TEST_CASE("a lit floor loses its window, not its brightness") {
  // Local contrast, which the effect hands in as a per-floor weight. The point
  // of doing it here rather than on the gain: a floor that is already lit
  // underneath gets FEWER hits, at exactly the same strength as everyone
  // else's, instead of the same hits turned down.
  Params p;
  p.rate = 20.0f;
  p.duty = 0.9f;
  p.grace = 0.0f;
  p.weight[1] = 0.2f;    // the middle floor is blazing underneath

  // Measured against the same roll with nothing damped, because the floors are
  // not interchangeable: the bounce visits the middle one twice a cycle, so
  // comparing it to its neighbours would be comparing two different jobs.
  auto run = [](Params q) {
    Core c;
    Played out[kPlanes] = {};
    step(c, q, 1.0f);
    for (int i = 0; i < 600; ++i) {
      Frame f = step(c, q, 1.0f);
      for (int k = 0; k < kPlanes; ++k) {
        if (f.g[k] > 1e-4f) {
          ++out[k].frames;
          if (f.g[k] > out[k].peak) out[k].peak = f.g[k];
        }
      }
    }
    return std::vector<Played>(out, out + kPlanes);
  };

  Params open = p;
  open.weight[1] = 1.0f;
  const auto damped = run(p);
  const auto full = run(open);

  // Far fewer frames on the damped floor...
  REQUIRE(damped[1].frames > 0);
  REQUIRE(damped[1].frames * 2 < full[1].frames);
  // ...its neighbours untouched...
  REQUIRE(damped[0].frames == full[0].frames);
  REQUIRE(damped[2].frames == full[2].frames);
  // ...and every hit that did land, on any floor, at exactly full strength.
  for (int k = 0; k < kPlanes; ++k) {
    REQUIRE_THAT(damped[k].peak, WithinAbs(1.0f, 1e-6f));
    REQUIRE_THAT(full[k].peak, WithinAbs(1.0f, 1e-6f));
  }
}

TEST_CASE("a floor damped to nothing drops out entirely") {
  Params p;
  p.rate = 20.0f;
  p.duty = 0.9f;
  p.weight[2] = 0.0f;

  Core c;
  step(c, p, 1.0f);
  for (int i = 0; i < 600; ++i) {
    Frame f = step(c, p, 1.0f);
    REQUIRE_THAT(f.g[2], WithinAbs(0.0f, 1e-6f));
  }
}

TEST_CASE("the weight shortens the arrival too") {
  // The arrival is the whole step for an untouched floor. A damped one gets
  // its share of it and no more — cut off early, never dimmed.
  Params p;
  p.rate = 20.0f;
  p.weight[0] = 0.25f;
  Core c;

  int frames[kPlanes] = {};
  const int arrival = (int)std::ceil(120.0f / p.rate);   // one step, in frames
  for (int i = 0; i < arrival; ++i) {
    Frame f = step(c, p, 1.0f);
    for (int k = 0; k < kPlanes; ++k) {
      if (f.g[k] > 1e-4f) { ++frames[k]; REQUIRE_THAT(f.g[k], WithinAbs(1.0f, 1e-6f)); }
    }
  }
  REQUIRE(frames[1] == arrival);
  REQUIRE(frames[2] == arrival);
  REQUIRE(frames[0] > 0);
  REQUIRE(frames[0] * 2 < arrival);
}

TEST_CASE("a soft throw is a short tail, not a dim one") {
  Params p;
  p.rate = 20.0f;
  p.duty = 0.9f;
  p.grace = 0.0f;

  Core hard, soft;
  Params ph = p, ps = p;
  auto fh = tail(hard, ph, 1.0f, 2.0f);
  auto fs = tail(soft, ps, 0.3f, 2.0f);

  REQUIRE(fs.size() * 3 < fh.size() * 2);          // markedly shorter
  REQUIRE(litFrames(fs, 0, fs.size()) > 0);        // but it did play
  for (const auto& f : fs) {
    const int k = brightest(f);
    if (k >= 0) REQUIRE_THAT(f.g[k], WithinAbs(1.0f, 1e-6f));
  }
}

TEST_CASE("a fresh throw restarts the roll on the arrival") {
  Params p;
  p.rate = 20.0f;
  p.duty = 0.5f;
  Core c;

  step(c, p, 1.0f);
  float rel = 1.0f;
  for (int i = 0; i < 40; ++i) { rel -= kDt / 1.0f; step(c, p, rel); }
  REQUIRE(c.clock > 1.0f);              // well into the bounce

  Frame f = step(c, p, 1.0f);           // hit it again
  REQUIRE(litCount(f) == kPlanes);      // the whole stack, again
  REQUIRE_THAT(c.clock, WithinAbs(p.rate * kDt, 1e-4f));
}

TEST_CASE("a dropped frame does not teleport the roll") {
  Params p;
  p.rate = 20.0f;
  Core c;
  step(c, p, 1.0f);
  const float before = c.clock;
  step(c, p, 1.0f, 5.0f);               // a five-second stall
  REQUIRE(c.clock - before <= p.rate * kMaxDt + 1e-4f);
}

TEST_CASE("a zero window silences the roll but not the arrival") {
  Params p;
  p.rate = 20.0f;
  p.duty = 0.0f;
  Core c;

  Frame f = step(c, p, 1.0f);
  REQUIRE(litCount(f) == kPlanes);
  const int arrival = (int)std::ceil(120.0f / p.rate);
  for (int i = 0; i < arrival; ++i) step(c, p, 1.0f);
  for (int i = 0; i < 200; ++i) REQUIRE(litCount(step(c, p, 1.0f)) == 0);
}

TEST_CASE("reset puts it back to silence") {
  Params p;
  Core c;
  step(c, p, 1.0f);
  for (int i = 0; i < 30; ++i) step(c, p, 1.0f);
  c.reset();
  REQUIRE_THAT(c.clock, WithinAbs(0.0f, 1e-6f));
  for (int i = 0; i < kPlanes; ++i) REQUIRE_THAT(c.gain[i], WithinAbs(0.0f, 1e-6f));
}
