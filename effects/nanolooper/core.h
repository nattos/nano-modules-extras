#ifndef LOOPER_CORE_H
#define LOOPER_CORE_H

#define MAX_EVENTS 256
#define MAX_UNDO 16
#define NUM_CHANNELS 4
#define NUM_STEPS 16

/* A recorded note: an onset plus the GATE DURATION the user held it for. Both
 * are in loop units (steps), where the loop spans [0, loop_length) == one bar of
 * NUM_STEPS. length > 0; a note may wrap past loop_length (start+length capped at
 * one full loop). This is the core of the rework: the looper repeats not just
 * WHEN a channel fired but HOW LONG it stayed on. */
typedef struct {
  double start;   /* onset within loop [0, loop_length) */
  double length;  /* gate duration in loop units (0, loop_length] */
  int channel;    /* 0-3 */
} Event;

typedef struct {
  Event events[MAX_EVENTS];
  int count;
} EventSnapshot;

typedef struct {
  double loop_length;

  /* Quantization is per-axis and independently optional (both default OFF — a
   * free looper that repeats exactly what you played). quantize_start snaps the
   * onset to the step grid; quantize_length snaps the gate duration to a whole
   * number of steps (min 1). Grid spacing is one step (1.0).
   *
   * quantize_start_amount (0..1, default 1) makes the start snap PARTIAL: the
   * onset moves that fraction of the way to its grid position. 1 = full snap
   * (classic quantize); 0.5 = halfway there (tightened but still human); 0 =
   * untouched. Only applies while quantize_start is on. */
  int quantize_start;
  int quantize_length;
  double quantize_start_amount;

  /* Overwrite "grace period", in LOOP UNITS (steps). When a new note truncates
   * an existing one (its onset lands inside the old note's body), the old note
   * is deleted outright if the truncation leaves it shorter than this. And when
   * a new note's body grows over an existing note's onset, the old note is
   * swallowed (deleted) UNLESS the release lands within `grace` of the body
   * reaching that onset — then the new note is truncated to butt against it and
   * the old note survives. Default = a 1/64 note (loop_length / 64). */
  double grace;

  Event events[MAX_EVENTS];
  int event_count;

  /* Recording: the note currently held open per channel (index into events, or
   * -1). pending_raw_start is the UNSNAPPED press time, so the finalized gate
   * length reflects the real hold duration even when the onset is quantized. */
  int pending_index[NUM_CHANNELS];
  double pending_raw_start[NUM_CHANNELS];

  EventSnapshot undo_stack[MAX_UNDO];
  int undo_count;
  EventSnapshot redo_stack[MAX_UNDO];
  int redo_count;

  int destructive_recording;
  EventSnapshot pre_record_snapshot;
} LooperCore;

void looper_init(LooperCore* c, double loop_length);
/* Change the loop length in place, keeping the recorded pattern: note onsets
 * are FOLDED into the new loop (start mod length) and gate lengths clamped, so
 * growing 1→4 bars keeps the pattern in bar 1 and shrinking folds everything
 * back down instead of silently dropping notes. Grace/quantize are untouched
 * (both are step-based; a step stays one 16th regardless of loop length). */
void looper_set_loop_length(LooperCore* c, double loop_length);
void looper_set_quantize(LooperCore* c, int q_start, int q_length);
/* Set how far starts snap (0..1, clamped; see quantize_start_amount above). */
void looper_set_quantize_start_amount(LooperCore* c, double amount);
/* Set the overwrite grace period, in loop units (steps). Clamped to >= 0. */
void looper_set_grace(LooperCore* c, double grace_units);

/* Recording a note is two-phase: begin_note on the press (adds/re-opens the
 * event and marks it pending), end_note on the release (finalizes its length
 * from the real hold duration). If a channel is still pending when the loop is
 * torn down, its length keeps the provisional value from the last end_note or 0. */
int  looper_begin_note(LooperCore* c, int channel, double current_time);
void looper_end_note(LooperCore* c, int channel, double current_time);

/* Grow the in-flight (held) note(s) to the current time so the overlay shows a
 * note extending from its onset to `current_time` while the trigger is down.
 * Call once per frame. No-op for channels with nothing pending. Playback
 * (looper_active_channels) still ignores pending notes — this is display state
 * that end_note finalizes (and quantizes) on release. */
void looper_tick_pending(LooperCore* c, double current_time);

/* Latch-mode capture-window state machine (pure; the module owns the transport
 * timer and does the actual clear). Given a trigger at monotonic time `now_abs`
 * (accumulated transport phase, never wrapping), the bar length `loop`, and an
 * inverse-grace `inv_grace` at the bar's TRAILING edge:
 *   - the first trigger, or one AFTER the 1-bar window (including the trailing
 *     inverse-grace zone, so the first tap of a repeated phrase reads as a NEW
 *     bar), starts/restarts a capture: returns 1 (the caller clears the pattern)
 *     and updates *start_abs / *capturing;
 *   - a trigger inside the window just adds to it: returns 0, state unchanged.
 * A restart within one bar of the previous bar boundary snaps *start_abs to that
 * boundary (grid-aligned) so a repeatedly-tapped phrase's start doesn't drift. */
int looper_latch_press(int* capturing, double* start_abs,
                       double now_abs, double loop, double inv_grace);

/* Playback gate: fills active[NUM_CHANNELS] with 1 where some recorded note's
 * [start, start+length) window (wrapping) covers `phase`. The module diffs this
 * against the previous frame to emit gate on/off — frame-rate independent and
 * wrap-safe. Pending (still-held) notes do not play back; the live press does. */
void looper_active_channels(const LooperCore* c, double phase, int* active);

void looper_clear_channel(LooperCore* c, int channel);
void looper_clear_all(LooperCore* c);
void looper_clear_at(LooperCore* c, int channel, int step);
void looper_begin_destructive_record(LooperCore* c);
void looper_end_destructive_record(LooperCore* c);
void looper_undo(LooperCore* c);
void looper_redo(LooperCore* c);

/* Overlay queries. has_event: some note's ONSET floors to `step` (the bright
 * leading edge). step_covered: `step` lies within some note's gate window (the
 * sustained bar showing how long it was held). */
int  looper_has_event(const LooperCore* c, int channel, int step);
int  looper_step_covered(const LooperCore* c, int channel, int step);

#endif
