// Unit tests for the reworked NanoLooper sequencer core (wasm_modules/nanolooper
// /core.cpp), compiled natively (pure C++, cmath only). Covers the rework:
//   - a note records HOW LONG it was held (begin/end), and repeats that gate;
//   - onset quantization and length quantization are each independently optional;
//   - playback windows wrap the loop seam and diff cleanly per frame;
//   - a still-held (pending) note does not play back (the live press covers it).

#include <catch2/catch_test_macros.hpp>
#include <catch2/catch_approx.hpp>

#include "nanolooper/core.h"

namespace {
bool active_ch(const LooperCore& c, double phase, int ch) {
  int a[NUM_CHANNELS];
  looper_active_channels(&c, phase, a);
  return a[ch] != 0;
}
constexpr double LOOP = (double)NUM_STEPS;  // 16
}  // namespace

TEST_CASE("note records its gate duration", "[looper_core]") {
  LooperCore c;
  looper_init(&c, LOOP);

  looper_begin_note(&c, 0, 2.0);
  looper_end_note(&c, 0, 5.0);

  REQUIRE(c.event_count == 1);
  CHECK(c.events[0].channel == 0);
  CHECK(c.events[0].start == 2.0);
  CHECK(c.events[0].length == 3.0);

  // The gate is on across [2, 5) and off elsewhere — repeats what was played.
  CHECK(active_ch(c, 2.0, 0));
  CHECK(active_ch(c, 4.99, 0));
  CHECK_FALSE(active_ch(c, 5.01, 0));
  CHECK_FALSE(active_ch(c, 1.0, 0));
  CHECK_FALSE(active_ch(c, 6.0, 0));
}

TEST_CASE("a held note plays back only after release (pending excluded)",
          "[looper_core]") {
  LooperCore c;
  looper_init(&c, LOOP);

  looper_begin_note(&c, 1, 4.0);  // still held — pending
  CHECK_FALSE(active_ch(c, 4.5, 1));  // live press covers it, not the sequencer

  looper_end_note(&c, 1, 7.0);
  CHECK(active_ch(c, 4.5, 1));  // now the recorded window plays back
  CHECK(active_ch(c, 6.9, 1));
  CHECK_FALSE(active_ch(c, 7.5, 1));
}

TEST_CASE("quantize_start floors the onset but keeps the real hold length",
          "[looper_core]") {
  LooperCore c;
  looper_init(&c, LOOP);
  looper_set_quantize(&c, /*q_start=*/1, /*q_length=*/0);

  looper_begin_note(&c, 0, 2.7);
  looper_end_note(&c, 0, 5.3);

  REQUIRE(c.event_count == 1);
  CHECK(c.events[0].start == 2.0);              // snapped down to the step grid
  CHECK(c.events[0].length == Catch::Approx(2.6));  // real hold (5.3 - 2.7)
}

TEST_CASE("quantize_length snaps the gate to whole steps, min one",
          "[looper_core]") {
  LooperCore c;
  looper_init(&c, LOOP);
  looper_set_quantize(&c, /*q_start=*/0, /*q_length=*/1);

  // A short tap still yields a one-step gate.
  looper_begin_note(&c, 0, 1.0);
  looper_end_note(&c, 0, 1.2);
  CHECK(c.events[0].length == 1.0);

  // 2.6 steps rounds up to 3.
  looper_begin_note(&c, 1, 1.0);
  looper_end_note(&c, 1, 3.6);
  CHECK(c.events[1].length == 3.0);
}

TEST_CASE("gate window wraps the loop seam", "[looper_core]") {
  LooperCore c;
  looper_init(&c, LOOP);

  looper_begin_note(&c, 2, 15.0);
  looper_end_note(&c, 2, 2.0);  // held across the seam: length 3 → [15, 18)

  REQUIRE(c.event_count == 1);
  CHECK(c.events[2 - 2].length == 3.0);  // events[0]
  CHECK(active_ch(c, 15.5, 2));
  CHECK(active_ch(c, 0.5, 2));   // wrapped tail
  CHECK(active_ch(c, 1.9, 2));
  CHECK_FALSE(active_ch(c, 2.5, 2));
  CHECK_FALSE(active_ch(c, 10.0, 2));
}

TEST_CASE("overdub at the same onset reuses the slot", "[looper_core]") {
  LooperCore c;
  looper_init(&c, LOOP);
  looper_set_quantize(&c, 1, 0);

  looper_begin_note(&c, 0, 3.0);
  looper_end_note(&c, 0, 4.0);
  looper_begin_note(&c, 0, 3.4);   // same step (3) — should not pile up
  looper_end_note(&c, 0, 7.0);

  CHECK(c.event_count == 1);
  CHECK(c.events[0].start == 3.0);
}

TEST_CASE("onset and coverage overlay queries", "[looper_core]") {
  LooperCore c;
  looper_init(&c, LOOP);

  looper_begin_note(&c, 0, 2.0);
  looper_end_note(&c, 0, 5.0);  // [2, 5): onset step 2, covers 2,3,4

  CHECK(looper_has_event(&c, 0, 2));
  CHECK_FALSE(looper_has_event(&c, 0, 3));

  CHECK(looper_step_covered(&c, 0, 2));
  CHECK(looper_step_covered(&c, 0, 3));
  CHECK(looper_step_covered(&c, 0, 4));
  CHECK_FALSE(looper_step_covered(&c, 0, 5));
  CHECK_FALSE(looper_step_covered(&c, 0, 1));
}

TEST_CASE("a new onset inside an old note's body truncates the old note",
          "[looper_core]") {
  LooperCore c;
  looper_init(&c, LOOP);

  looper_begin_note(&c, 0, 2.0);
  looper_end_note(&c, 0, 6.0);     // old = [2, 6)

  looper_begin_note(&c, 0, 4.0);   // press inside [2,6) → cut old to [2, 4)
  looper_end_note(&c, 0, 8.0);     // new = [4, 8)

  REQUIRE(c.event_count == 2);
  // Old note truncated to end at the new onset; new note recorded after it.
  CHECK(c.events[0].start == 2.0);
  CHECK(c.events[0].length == Catch::Approx(2.0));
  CHECK(c.events[1].start == 4.0);
  CHECK(c.events[1].length == Catch::Approx(4.0));
  // No double coverage at the boundary.
  CHECK(active_ch(c, 3.5, 0));
  CHECK(active_ch(c, 4.5, 0));
  CHECK_FALSE(active_ch(c, 8.5, 0));
}

TEST_CASE("truncating an old note below the grace period deletes it",
          "[looper_core]") {
  LooperCore c;
  looper_init(&c, LOOP);        // default grace = LOOP/64 = 0.25

  looper_begin_note(&c, 0, 2.0);
  looper_end_note(&c, 0, 6.0);  // old = [2, 6)

  // Press a hair past the old onset: the sliver [2, 2.1) is below grace → gone.
  looper_begin_note(&c, 0, 2.1);
  looper_end_note(&c, 0, 5.0);

  REQUIRE(c.event_count == 1);
  CHECK(c.events[0].start == Catch::Approx(2.1));
  CHECK(c.events[0].length == Catch::Approx(2.9));
}

TEST_CASE("a new note's body swallows an old onset it grows well past",
          "[looper_core]") {
  LooperCore c;
  looper_init(&c, LOOP);

  looper_begin_note(&c, 0, 8.0);
  looper_end_note(&c, 0, 10.0);   // old = [8, 10)

  looper_begin_note(&c, 0, 5.0);
  looper_end_note(&c, 0, 12.0);   // new = [5, 12) grows well past onset 8 → swallow

  REQUIRE(c.event_count == 1);
  CHECK(c.events[0].start == 5.0);
  CHECK(c.events[0].length == Catch::Approx(7.0));
}

TEST_CASE("releasing within grace of an old onset truncates the new note instead",
          "[looper_core]") {
  LooperCore c;
  looper_init(&c, LOOP);          // grace = 0.25

  looper_begin_note(&c, 0, 8.0);
  looper_end_note(&c, 0, 10.0);   // old = [8, 10)

  looper_begin_note(&c, 0, 5.0);
  looper_end_note(&c, 0, 8.1);    // released 0.1 past onset 8 (< grace) → butt up

  REQUIRE(c.event_count == 2);
  // New note truncated right up to the old onset; old note untouched.
  CHECK(c.events[0].start == 8.0);
  CHECK(c.events[0].length == Catch::Approx(2.0));
  CHECK(c.events[1].start == 5.0);
  CHECK(c.events[1].length == Catch::Approx(3.0));   // [5, 8), not [5, 8.1)
}

TEST_CASE("a wider grace makes the swallow window looser", "[looper_core]") {
  LooperCore c;
  looper_init(&c, LOOP);
  looper_set_grace(&c, 1.0);      // a full step of grace

  looper_begin_note(&c, 0, 8.0);
  looper_end_note(&c, 0, 10.0);   // old = [8, 10)

  looper_begin_note(&c, 0, 5.0);
  looper_end_note(&c, 0, 8.5);    // 0.5 past onset 8, now within the wider grace

  REQUIRE(c.event_count == 2);
  CHECK(c.events[1].length == Catch::Approx(3.0));   // new truncated to [5, 8)
}

TEST_CASE("a held note grows to the current time for the overlay",
          "[looper_core]") {
  LooperCore c;
  looper_init(&c, LOOP);

  looper_begin_note(&c, 0, 2.0);
  CHECK(c.events[0].length == Catch::Approx(0.0));   // provisional at press

  looper_tick_pending(&c, 5.0);
  CHECK(c.events[0].length == Catch::Approx(3.0));    // extends to now
  CHECK_FALSE(active_ch(c, 4.0, 0));                  // still pending → no playback

  looper_tick_pending(&c, 7.0);
  CHECK(c.events[0].length == Catch::Approx(5.0));    // keeps growing

  looper_end_note(&c, 0, 8.0);                        // release finalizes from raw
  CHECK(c.events[0].length == Catch::Approx(6.0));    // [2, 8)
  CHECK(active_ch(c, 4.0, 0));                        // now it plays
}

TEST_CASE("latch: first trigger starts a capture, in-window triggers accumulate",
          "[looper_core]") {
  int capturing = 0;
  double start = 0;
  const double INV = 0.25;   // inverse grace (a 1/64 note)

  // First trigger at abs=3 → start capture (returns 1 = clear the pattern).
  CHECK(looper_latch_press(&capturing, &start, 3.0, LOOP, INV) == 1);
  CHECK(capturing == 1);
  CHECK(start == 3.0);

  // Later triggers within the window just add (return 0, state unchanged).
  CHECK(looper_latch_press(&capturing, &start, 8.0, LOOP, INV) == 0);
  CHECK(looper_latch_press(&capturing, &start, 18.0, LOOP, INV) == 0);  // 15 in
  CHECK(start == 3.0);
}

TEST_CASE("latch: a trigger past the window restarts (and aligns to the grid)",
          "[looper_core]") {
  int capturing = 0;
  double start = 0;
  const double INV = 0.25;

  looper_latch_press(&capturing, &start, 3.0, LOOP, INV);   // start at 3

  // A trigger ~1 bar later (abs 19.1) is past the window → restart, aligned to
  // the previous bar boundary (3 + 16 = 19), NOT to the jittered tap time.
  CHECK(looper_latch_press(&capturing, &start, 19.1, LOOP, INV) == 1);
  CHECK(start == Catch::Approx(19.0));   // snapped to the grid, no drift
}

TEST_CASE("latch: inverse grace treats a near-end trigger as a new bar",
          "[looper_core]") {
  int capturing = 0;
  double start = 0;
  const double INV = 0.25;

  looper_latch_press(&capturing, &start, 3.0, LOOP, INV);   // window [3, 3+16)

  // A tap at abs 18.8 (elapsed 15.8) lands inside the trailing inverse-grace
  // zone [15.75, 16) → treated as OUTSIDE → restart (a fresh repeat), aligned.
  CHECK(looper_latch_press(&capturing, &start, 18.8, LOOP, INV) == 1);
  CHECK(start == Catch::Approx(19.0));

  // Whereas a tap just before that zone (elapsed 15.5) still accumulates.
  int cap2 = 0; double st2 = 0;
  looper_latch_press(&cap2, &st2, 3.0, LOOP, INV);
  CHECK(looper_latch_press(&cap2, &st2, 18.5, LOOP, INV) == 0);
}

TEST_CASE("latch: a retrigger well after a bar starts fresh (unaligned)",
          "[looper_core]") {
  int capturing = 0;
  double start = 0;
  const double INV = 0.25;

  looper_latch_press(&capturing, &start, 3.0, LOOP, INV);   // start 3

  // Waited more than a full bar past the boundary (abs 43, boundary 19, 24 > 16)
  // → not aligned; a brand-new capture anchored at the tap.
  CHECK(looper_latch_press(&capturing, &start, 43.0, LOOP, INV) == 1);
  CHECK(start == Catch::Approx(43.0));
}

TEST_CASE("clear and undo restore the pattern", "[looper_core]") {
  LooperCore c;
  looper_init(&c, LOOP);

  looper_begin_note(&c, 0, 1.0);
  looper_end_note(&c, 0, 2.0);
  looper_begin_note(&c, 1, 4.0);
  looper_end_note(&c, 1, 6.0);
  REQUIRE(c.event_count == 2);

  looper_clear_channel(&c, 0);
  CHECK(c.event_count == 1);
  CHECK(c.events[0].channel == 1);

  looper_undo(&c);
  CHECK(c.event_count == 2);

  looper_clear_all(&c);
  CHECK(c.event_count == 0);
  looper_undo(&c);
  CHECK(c.event_count == 2);
}

TEST_CASE("loop length change keeps the pattern: grow in place, shrink folds",
          "[looper_core]") {
  LooperCore c;
  looper_init(&c, LOOP);  // 1 bar (16)

  looper_begin_note(&c, 0, 2.0);
  looper_end_note(&c, 0, 5.0);

  // Grow 1 → 4 bars: the note stays where it was, and does NOT repeat in the
  // later bars (it plays once per 4-bar loop now).
  looper_set_loop_length(&c, LOOP * 4);  // 64
  CHECK(c.loop_length == LOOP * 4);
  CHECK(c.events[0].start == 2.0);
  CHECK(active_ch(c, 3.0, 0));
  CHECK_FALSE(active_ch(c, 16.0 + 3.0, 0));   // bar 2 is empty

  // Record into bar 3 of the long loop.
  looper_begin_note(&c, 1, 34.0);
  looper_end_note(&c, 1, 36.0);
  CHECK(active_ch(c, 35.0, 1));

  // Shrink back to 1 bar: the bar-3 note FOLDS to 34 mod 16 = 2, not dropped.
  looper_set_loop_length(&c, LOOP);
  CHECK(c.loop_length == LOOP);
  REQUIRE(c.event_count == 2);
  CHECK(active_ch(c, 2.5, 1));                // folded into the first bar
  CHECK(active_ch(c, 3.0, 0));                // original note untouched
}

TEST_CASE("undo across a loop-length shrink folds restored notes into range",
          "[looper_core]") {
  LooperCore c;
  looper_init(&c, LOOP * 2);  // 2 bars (32)

  looper_begin_note(&c, 0, 20.0);  // bar 2
  looper_end_note(&c, 0, 22.0);
  looper_clear_all(&c);
  looper_set_loop_length(&c, LOOP);  // shrink with an empty pattern

  looper_undo(&c);  // restores the bar-2 note into a 1-bar loop
  REQUIRE(c.event_count == 1);
  CHECK(c.events[0].start == 4.0);   // 20 mod 16 — folded, so it still plays
  CHECK(active_ch(c, 4.5, 0));
}

TEST_CASE("quantize_start_amount pulls the onset partway to the grid",
          "[looper_core]") {
  LooperCore c;
  looper_init(&c, LOOP);
  looper_set_quantize(&c, /*q_start=*/1, /*q_length=*/0);

  // Full snap (default amount = 1): identical to classic quantize.
  looper_begin_note(&c, 0, 2.7);
  looper_end_note(&c, 0, 5.3);
  REQUIRE(c.event_count == 1);
  CHECK(c.events[0].start == Catch::Approx(2.0));

  // Half snap: the onset moves half of the 0.7 it would have snapped.
  looper_set_quantize_start_amount(&c, 0.5);
  looper_begin_note(&c, 1, 2.7);
  looper_end_note(&c, 1, 5.3);
  REQUIRE(c.event_count == 2);
  CHECK(c.events[1].start == Catch::Approx(2.35));
  CHECK(c.events[1].length == Catch::Approx(2.6));  // real hold, unaffected

  // Zero: quantize_start on but the onset is untouched.
  looper_set_quantize_start_amount(&c, 0.0);
  looper_begin_note(&c, 2, 2.7);
  looper_end_note(&c, 2, 5.3);
  REQUIRE(c.event_count == 3);
  CHECK(c.events[2].start == Catch::Approx(2.7));

  // Out-of-range amounts clamp.
  looper_set_quantize_start_amount(&c, 1.7);
  CHECK(c.quantize_start_amount == 1.0);
  looper_set_quantize_start_amount(&c, -0.3);
  CHECK(c.quantize_start_amount == 0.0);
}
