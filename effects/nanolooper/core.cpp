#include "core.h"

#include <cmath>

#define EPSILON 1e-6
#define STEP 1.0  /* grid spacing in loop units (one step) */

static double wrap(double time, double len) {
  double t = fmod(time, len);
  if (t < 0) t += len;
  return t;
}

/* Forward (circular) distance from `from` to `to` in [0, loop): how far you
 * advance from `from` before reaching `to`, wrapping the loop seam. */
static double fwd(double from, double to, double loop) {
  double d = fmod(to - from, loop);
  if (d < 0) d += loop;
  return d;
}

/* Compact the event array, dropping every index i where del[i] != 0, and remap
 * all pending indices through the move so in-flight recordings stay valid. */
static void remove_marked(LooperCore* c, const char* del) {
  int newidx[MAX_EVENTS];
  int j = 0;
  for (int i = 0; i < c->event_count; i++) {
    if (del[i]) { newidx[i] = -1; }
    else { newidx[i] = j; c->events[j++] = c->events[i]; }
  }
  c->event_count = j;
  for (int ch = 0; ch < NUM_CHANNELS; ch++) {
    if (c->pending_index[ch] >= 0)
      c->pending_index[ch] = newidx[c->pending_index[ch]];
  }
}

/* Snap an onset to the step grid (floor into the current step). */
static double quantize_start_val(double time, double len) {
  double t = wrap(time, len);
  return floor(t / STEP) * STEP;
}

/* Snap a gate duration to a whole number of steps, at least one. */
static double quantize_length_val(double length) {
  double q = floor(length / STEP + 0.5) * STEP;
  if (q < STEP) q = STEP;
  return q;
}

static void save_snapshot(const LooperCore* c, EventSnapshot* s) {
  s->count = c->event_count;
  for (int i = 0; i < c->event_count; i++)
    s->events[i] = c->events[i];
}

static void load_snapshot(LooperCore* c, const EventSnapshot* s) {
  c->event_count = s->count;
  for (int i = 0; i < s->count; i++) {
    c->events[i] = s->events[i];
    /* The snapshot may predate a loop-length change — fold it in like
     * looper_set_loop_length so restored notes always play back. */
    c->events[i].start = wrap(c->events[i].start, c->loop_length);
    if (c->events[i].length > c->loop_length)
      c->events[i].length = c->loop_length;
  }
  /* Any snapshot restore invalidates in-flight recording. */
  for (int ch = 0; ch < NUM_CHANNELS; ch++) c->pending_index[ch] = -1;
}

static void push_undo(LooperCore* c) {
  if (c->undo_count < MAX_UNDO) {
    save_snapshot(c, &c->undo_stack[c->undo_count++]);
  } else {
    /* Shift stack down, drop oldest */
    for (int i = 0; i < MAX_UNDO - 1; i++)
      c->undo_stack[i] = c->undo_stack[i + 1];
    save_snapshot(c, &c->undo_stack[MAX_UNDO - 1]);
  }
  c->redo_count = 0;
}

void looper_init(LooperCore* c, double loop_length) {
  c->loop_length = loop_length;
  c->quantize_start = 0;
  c->quantize_length = 0;
  c->quantize_start_amount = 1.0;
  c->grace = loop_length / 64.0;   /* a 1/64 note (1 bar / 64) */
  c->event_count = 0;
  c->undo_count = 0;
  c->redo_count = 0;
  c->destructive_recording = 0;
  c->pre_record_snapshot.count = 0;
  for (int ch = 0; ch < NUM_CHANNELS; ch++) {
    c->pending_index[ch] = -1;
    c->pending_raw_start[ch] = 0.0;
  }
}

void looper_set_loop_length(LooperCore* c, double loop_length) {
  if (!(loop_length > 0) || loop_length == c->loop_length) return;
  c->loop_length = loop_length;
  for (int i = 0; i < c->event_count; i++) {
    c->events[i].start = wrap(c->events[i].start, loop_length);
    if (c->events[i].length > loop_length) c->events[i].length = loop_length;
  }
  for (int ch = 0; ch < NUM_CHANNELS; ch++)
    c->pending_raw_start[ch] = wrap(c->pending_raw_start[ch], loop_length);
}

void looper_set_quantize(LooperCore* c, int q_start, int q_length) {
  c->quantize_start = q_start ? 1 : 0;
  c->quantize_length = q_length ? 1 : 0;
}

void looper_set_quantize_start_amount(LooperCore* c, double amount) {
  c->quantize_start_amount = amount < 0 ? 0 : (amount > 1 ? 1 : amount);
}

void looper_set_grace(LooperCore* c, double grace_units) {
  c->grace = grace_units > 0 ? grace_units : 0;
}

void looper_tick_pending(LooperCore* c, double current_time) {
  double now = wrap(current_time, c->loop_length);
  for (int ch = 0; ch < NUM_CHANNELS; ch++) {
    int idx = c->pending_index[ch];
    if (idx < 0) continue;
    /* Grow the held note from its (unsnapped) press to now — display length,
     * unsnapped; end_note applies quantize_length + overwrite on release. */
    double len = now - c->pending_raw_start[ch];
    if (len < 0) len += c->loop_length;
    if (len < EPSILON) len = EPSILON;
    if (len > c->loop_length) len = c->loop_length;
    c->events[idx].length = len;
  }
}

/* Find a finalized (non-pending) event on `channel` whose onset matches `start`
 * (same step when quantizing starts, else within epsilon). Returns index or -1. */
static int find_event_at(const LooperCore* c, int channel, double start) {
  for (int i = 0; i < c->event_count; i++) {
    if (c->events[i].channel != channel) continue;
    int is_pending = 0;
    for (int ch = 0; ch < NUM_CHANNELS; ch++)
      if (c->pending_index[ch] == i) { is_pending = 1; break; }
    if (is_pending) continue;
    if (c->quantize_start) {
      if ((int)floor(c->events[i].start / STEP) == (int)floor(start / STEP))
        return i;
    } else if (fabs(c->events[i].start - start) < EPSILON) {
      return i;
    }
  }
  return -1;
}

/* Truncate finalized notes on `channel` whose BODY strictly contains `at` back
 * to end at `at`; delete any left shorter than the grace period. Onsets exactly
 * at `at` (fwd == 0, an overdub target) are left alone. */
static void truncate_overlapping_start(LooperCore* c, int channel, double at) {
  char del[MAX_EVENTS];
  for (int i = 0; i < c->event_count; i++) del[i] = 0;
  int any = 0;
  for (int i = 0; i < c->event_count; i++) {
    Event* e = &c->events[i];
    if (e->channel != channel) continue;
    double d = fwd(e->start, at, c->loop_length);
    if (d > EPSILON && d < e->length - EPSILON) {   /* `at` strictly inside body */
      e->length = d;
      if (e->length < c->grace) { del[i] = 1; any = 1; }
    }
  }
  if (any) remove_marked(c, del);
}

int looper_begin_note(LooperCore* c, int channel, double current_time) {
  if (channel < 0 || channel >= NUM_CHANNELS) return -1;

  /* A press without a matching release: finalize the stale note first. */
  if (c->pending_index[channel] >= 0)
    looper_end_note(c, channel, current_time);

  double raw = wrap(current_time, c->loop_length);
  double start = raw;
  if (c->quantize_start) {
    /* Partial snap: move the onset quantize_start_amount of the way to its
     * grid position. The snap FLOORS within the onset's own step, so the lerp
     * never crosses the loop seam — plain interpolation is wrap-safe here. */
    double snapped = quantize_start_val(raw, c->loop_length);
    start = raw + (snapped - raw) * c->quantize_start_amount;
  }

  /* One undo checkpoint for the whole press action — the start-overwrite here,
   * the new note, and any release-time body-overwrite all share it. */
  if (!c->destructive_recording) push_undo(c);

  /* A new onset inside an existing note's body cuts that note short (deleting
   * it outright if only a grace-sized sliver would remain). */
  truncate_overlapping_start(c, channel, start);

  /* Overdub onto an existing onset at this position rather than piling up. */
  int idx = find_event_at(c, channel, start);
  if (idx < 0) {
    if (c->event_count >= MAX_EVENTS) return -1;
    idx = c->event_count++;
    c->events[idx].channel = channel;
    c->events[idx].start = start;
    c->events[idx].length = 0.0;  /* provisional until release */
  }

  c->pending_index[channel] = idx;
  c->pending_raw_start[channel] = raw;
  return idx;
}

void looper_end_note(LooperCore* c, int channel, double current_time) {
  if (channel < 0 || channel >= NUM_CHANNELS) return;
  int idx = c->pending_index[channel];
  if (idx < 0) return;

  /* Gate duration = real hold time (measured from the unsnapped press), so
   * quantizing the onset never distorts how long the note is held. */
  double length = wrap(current_time, c->loop_length) - c->pending_raw_start[channel];
  if (length < 0) length += c->loop_length;      /* held across the loop seam */
  if (length < EPSILON) length = EPSILON;         /* a tap still has a tiny gate */
  if (c->quantize_length) length = quantize_length_val(length);
  if (length > c->loop_length) length = c->loop_length;
  c->events[idx].length = length;

  /* Overwrite the far side: any note whose ONSET this note's body grew over is
   * swallowed (deleted) — UNLESS the release landed within `grace` of the body
   * reaching the LAST such onset, in which case butt the new note right up to
   * that onset and spare it. Notes reached earlier (held past grace) still go. */
  double n_start = c->events[idx].start;
  int gi = -1;                   /* last-reached covered onset (largest fwd) */
  double gd = -1.0;
  for (int i = 0; i < c->event_count; i++) {
    if (i == idx || c->events[i].channel != channel) continue;
    double d = fwd(n_start, c->events[i].start, c->loop_length);
    if (d > EPSILON && d < c->events[idx].length - EPSILON) {   /* onset in body */
      if (d > gd) { gd = d; gi = i; }
    }
  }

  if (gi >= 0) {
    int spare = (c->events[idx].length - gd < c->grace) ? gi : -1;
    if (spare >= 0) c->events[idx].length = gd;   /* butt up to the spared onset */

    char del[MAX_EVENTS];
    for (int i = 0; i < c->event_count; i++) del[i] = 0;
    double body = c->events[idx].length;          /* possibly truncated */
    for (int i = 0; i < c->event_count; i++) {
      if (i == idx || i == spare || c->events[i].channel != channel) continue;
      double d = fwd(n_start, c->events[i].start, c->loop_length);
      if (d > EPSILON && d < body - EPSILON) del[i] = 1;
    }
    c->pending_index[channel] = -1;               /* clear before compaction */
    remove_marked(c, del);
  } else {
    c->pending_index[channel] = -1;
  }
}

int looper_latch_press(int* capturing, double* start_abs,
                       double now_abs, double loop, double inv_grace) {
  if (!*capturing) {
    *capturing = 1;
    *start_abs = now_abs;
    return 1;   /* first trigger — start capture, clear the pattern */
  }
  double elapsed = now_abs - *start_abs;
  if (elapsed < loop - inv_grace) {
    return 0;   /* inside the 1-bar window (before the inverse-grace edge) — add */
  }
  /* Past the window (or in its trailing inverse-grace zone) — restart. Align to
   * the previous bar boundary when the retrigger lands within one bar of it, so
   * a repeatedly-tapped phrase stays on a stable grid instead of drifting. */
  double next_boundary = *start_abs + loop;
  if (now_abs - next_boundary < loop) *start_abs = next_boundary;
  else                                *start_abs = now_abs;
  return 1;
}

void looper_active_channels(const LooperCore* c, double phase, int* active) {
  for (int ch = 0; ch < NUM_CHANNELS; ch++) active[ch] = 0;
  double p = wrap(phase, c->loop_length);
  for (int i = 0; i < c->event_count; i++) {
    /* Skip the note currently being held — the live press covers it. */
    int is_pending = 0;
    for (int ch = 0; ch < NUM_CHANNELS; ch++)
      if (c->pending_index[ch] == i) { is_pending = 1; break; }
    if (is_pending) continue;

    const Event* e = &c->events[i];
    if (e->length <= EPSILON) continue;
    double s = e->start;
    double end = s + e->length;
    int on;
    if (end <= c->loop_length + EPSILON) {
      on = (p >= s - EPSILON && p < end - EPSILON);
    } else {
      /* Window wraps past the loop seam. */
      double wrapped_end = end - c->loop_length;
      on = (p >= s - EPSILON) || (p < wrapped_end - EPSILON);
    }
    if (on) active[e->channel] = 1;
  }
}

/* Drop the pending marker for any event index being removed, and compact the
 * remaining pending indices after an erase. */
static void repending_after_erase(LooperCore* c, int erased_index) {
  for (int ch = 0; ch < NUM_CHANNELS; ch++) {
    if (c->pending_index[ch] == erased_index) c->pending_index[ch] = -1;
    else if (c->pending_index[ch] > erased_index) c->pending_index[ch]--;
  }
}

void looper_clear_channel(LooperCore* c, int channel) {
  int has = 0;
  for (int i = 0; i < c->event_count; i++)
    if (c->events[i].channel == channel) { has = 1; break; }
  if (!has) return;

  push_undo(c);
  int j = 0;
  for (int i = 0; i < c->event_count; i++) {
    if (c->events[i].channel != channel)
      c->events[j++] = c->events[i];
    else
      repending_after_erase(c, j);  /* i collapses onto slot j */
  }
  c->event_count = j;
  c->pending_index[channel] = -1;
}

void looper_clear_all(LooperCore* c) {
  if (c->event_count == 0) return;
  push_undo(c);
  c->event_count = 0;
  for (int ch = 0; ch < NUM_CHANNELS; ch++) c->pending_index[ch] = -1;
}

void looper_clear_at(LooperCore* c, int channel, int step) {
  int has = 0;
  for (int i = 0; i < c->event_count; i++) {
    if (c->events[i].channel == channel &&
        (int)floor(c->events[i].start) == step) { has = 1; break; }
  }
  if (!has) return;
  if (!c->destructive_recording)
    push_undo(c);
  int j = 0;
  for (int i = 0; i < c->event_count; i++) {
    if (!(c->events[i].channel == channel &&
          (int)floor(c->events[i].start) == step))
      c->events[j++] = c->events[i];
    else
      repending_after_erase(c, j);
  }
  c->event_count = j;
}

void looper_begin_destructive_record(LooperCore* c) {
  save_snapshot(c, &c->pre_record_snapshot);
  c->destructive_recording = 1;
}

void looper_end_destructive_record(LooperCore* c) {
  if (!c->destructive_recording) return;
  c->destructive_recording = 0;
  if (c->undo_count < MAX_UNDO) {
    c->undo_stack[c->undo_count++] = c->pre_record_snapshot;
  }
  c->redo_count = 0;
}

void looper_undo(LooperCore* c) {
  if (c->undo_count == 0) return;
  if (c->redo_count < MAX_UNDO) {
    save_snapshot(c, &c->redo_stack[c->redo_count++]);
  }
  c->undo_count--;
  load_snapshot(c, &c->undo_stack[c->undo_count]);
}

void looper_redo(LooperCore* c) {
  if (c->redo_count == 0) return;
  if (c->undo_count < MAX_UNDO) {
    save_snapshot(c, &c->undo_stack[c->undo_count++]);
  }
  c->redo_count--;
  load_snapshot(c, &c->redo_stack[c->redo_count]);
}

int looper_has_event(const LooperCore* c, int channel, int step) {
  for (int i = 0; i < c->event_count; i++) {
    if (c->events[i].channel == channel &&
        (int)floor(c->events[i].start) == step)
      return 1;
  }
  return 0;
}

int looper_step_covered(const LooperCore* c, int channel, int step) {
  /* A step is covered if the note's gate window overlaps [step, step+1). */
  for (int i = 0; i < c->event_count; i++) {
    const Event* e = &c->events[i];
    if (e->channel != channel) continue;
    if (e->length <= EPSILON) continue;
    double s = e->start;
    double end = s + e->length;
    double cell0 = (double)step;
    double cell1 = (double)step + STEP;
    if (end <= c->loop_length + EPSILON) {
      if (s < cell1 - EPSILON && end > cell0 + EPSILON) return 1;
    } else {
      double wrapped_end = end - c->loop_length;
      if (s < cell1 - EPSILON) return 1;                 /* [s, loop_end) */
      if (wrapped_end > cell0 + EPSILON) return 1;       /* [0, wrapped_end) */
    }
  }
  return 0;
}
