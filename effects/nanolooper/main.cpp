/*
 * NanoLooper WASM Module
 *
 * A 4-channel looper sequencer with visual overlay. The loop spans 1, 2 or 4
 * bars (`bars` param) of 16th-note steps — 16/32/64 steps. The host only
 * reports a per-bar phase, so the module counts bar wraps locally to know
 * which bar of the multi-bar loop it's in; changing `bars` folds/keeps the
 * recorded pattern (see looper_set_loop_length).
 * Uses shared host API headers for all host function imports.
 *
 * Class-like instance model (v2 ABI): module_init() publishes the schema
 * once per type; each chain entry gets its own State (sequencer core,
 * transport, edge-state, timers, channel mapping) via create(). All
 * instance callbacks take `self`.
 *
 * The visual overlay is drawn with the in-effect overlay toolbox (overlay.h):
 * solid rects/borders via an instanced GPU pass + labels via the host text
 * engine, composited onto tex_out over tex_in. (The old host `canvas_*` ABI is
 * a no-op in the sketch-executor / barrel render path, so the overlay never
 * showed there.) Declaring tex_out is what makes the executor bind a writable
 * render target and call render() at all — without it the looper is a
 * modulation-source passthrough and render() never runs.
 */

#include <gpu.h>
#include <host.h>
#include <overlay.h>
#include <val.h>
#include "core.h"
#include <json/json_doc_client.h>

#include <cmath>
#include <cstring>

namespace nanolooper {

/* ======================================================================
 * Constants
 * ====================================================================== */

/* Channel → clip mapping */
#define MAX_CHANNEL_CLIPS 8

/* Trigger-rail event ring (mirrors mod.trigger.beat). The executor drains the
 * published "triggers" ring onto the process-global trigger rail, which the
 * shared server launches Resolume clips from. */
#define TRIG_RING_CAP 16
struct LoopEv {
  long long seq = 0;
  bool on = false;
  int channel = 1;   /* 1-based rail channel */
  float velocity = 1.0f;
};

/* Channel colors (matching original) */
static const float CH_R[4] = {1.0f, 0.33f, 1.0f, 0.33f};
static const float CH_G[4] = {0.33f, 1.0f, 1.0f, 1.0f};
static const float CH_B[4] = {0.33f, 0.33f, 0.33f, 1.0f};

/* Parameter types (matching FFGL / state_document.h ParamType) */
#define PARAM_BOOLEAN  0
#define PARAM_STANDARD 10

/* Param IDs (must match LooperParamID enum) */
#define PID_TRIGGER_1    0
#define PID_TRIGGER_2    1
#define PID_TRIGGER_3    2
#define PID_TRIGGER_4    3
#define PID_DELETE       4
#define PID_MUTE         5
#define PID_UNDO         6
#define PID_REDO         7
#define PID_RECORD       8
#define PID_SHOW_OVERLAY 9
#define PID_SYNTH        10
#define PID_SYNTH_GAIN   11
#define PID_SEND_TO_RAIL 12
#define PID_QUANTIZE_START  13
#define PID_QUANTIZE_LENGTH 14
#define PID_GRACE           15
#define PID_LATCH           16
#define PID_ANCHOR          17
#define PID_OVERLAY_OPACITY 18
#define PID_LOOP_MODE       19
#define PID_STRICT_DEADLINE 20
#define PID_BARS            21
#define PID_QUANTIZE_START_AMOUNT 22

/* Loop length limits, in bars (4 beats of 4 steps each). */
#define MAX_BARS 4
#define MAX_LOOP_STEPS (MAX_BARS * NUM_STEPS)

/* Loop mode (single enum replacing the old record/latch bools):
 *   Off     — the recorded pattern is DISABLED (kept in memory, not played);
 *             live triggers still play + show as transient overlay notes.
 *   Overdub — pattern plays; triggers overdub into it (the old record-arm).
 *   Latch   — pattern plays; latch capture (first tap clears + captures a bar). */
#define LOOP_OFF     0
#define LOOP_OVERDUB 1
#define LOOP_LATCH   2

/* ======================================================================
 * State
 * ====================================================================== */

/* Per-instance state. One per chain entry. Holds every mutable var that
 * used to live as a file-static in this module. */
struct State {
  LooperCore looper{};
  double phase = 0.0;
  double prev_phase = 0.0;
  double elapsed = 0.0;

  /* Loop length in bars (1/2/4). The host clock only gives a per-bar phase, so
   * bar_in_loop tracks which bar of the loop we're in by counting bar wraps
   * (prev_bar_phase detects them). phase = (bar_in_loop + bar_phase) * 16. */
  int bars = 1;
  int bar_in_loop = 0;
  double prev_bar_phase = 0.0;

  /* Per-channel state. trigger_held = physical press; gate_state = the combined
   * (live press OR played-back window) on/off we've actually emitted, diffed each
   * frame so on/off fire exactly on transitions. No fixed gate timer any more —
   * a played note's gate lasts exactly as long as it was recorded. */
  int trigger_held[NUM_CHANNELS] = {0};
  int gate_state[NUM_CHANNELS] = {0};

  /* Modifier keys */
  int delete_held = 0;
  int delete_acted = 0;           /* did delete+trigger happen during this press? */
  int last_action_was_clear = 0;  /* was the last standalone delete a clear-all? */
  int mute_held = 0;
  /* loop_mode is the authority (Off/Overdub/Latch); record_held + latch are
   * derived mirrors kept in sync so the trigger/latch logic reads naturally.
   * record_held||latch ⟺ "pattern enabled" (plays back). */
  int loop_mode = LOOP_OVERDUB;
  int record_held = 1;
  int show_overlay = 0;

  /* Off-mode transient live notes: a held trigger shows a growing overlay bar
   * that vanishes on release, WITHOUT touching the (disabled) recorded pattern. */
  int live_held[NUM_CHANNELS] = {0};
  double live_start[NUM_CHANNELS] = {0};
  int anchor = 0;               /* overlay corner: 0=top-left, 1=bottom-left, 2=top-right, 3=bottom-right */
  float overlay_opacity = 1.0f; /* overall overlay alpha multiplier */
  int quantize_start = 0;
  int quantize_length = 0;
  float quantize_start_amount = 1.0f;  /* 0..1 partial start snap (1 = full) */
  float grace_beats = 0.0625f;   /* overwrite grace, in beats (1/64 note) */

  /* Latch mode: the first trigger clears the pattern and opens a 1-bar capture
   * window; triggers inside it accumulate; a trigger past it (incl. the trailing
   * inverse-grace zone) restarts. abs_phase is the monotonic transport clock the
   * window is measured against (a repeatedly-tapped phrase stays grid-aligned). */
  int latch = 0;
  int latch_capturing = 0;
  double latch_start_abs = 0.0;
  double abs_phase = 0.0;

  /* Connection state */
  int ws_connected = 0;

  /* Trigger-rail output. `send_to_rail` gates emission; the ring carries on/off
   * events (channel = ch+1); ch_out[] is a per-channel EXACT gate (1 while
   * gated, 0 otherwise — no decay) exposed as the out_N modulation outputs. */
  int send_to_rail = 1;
  /* Strict-precision deadline, in ms. 0 → precision "any" (immediate dispatch,
   * today's behavior). >0 → each emitted trigger carries precision
   * {mode:"strict", deadline:<this>}, asking the barrel pump to hold the clip
   * launch until a rendered frame reflecting it reaches the display (bounded by
   * the deadline). Range clamps to [0,250]. */
  float strict_deadline = 0.0f;
  LoopEv trig_ring[TRIG_RING_CAP] = {};
  int trig_ring_len = 0;
  long long trig_seq = 0;
  float ch_out[NUM_CHANNELS] = {0};

  /* Channel → clip mapping */
  long long channel_clip_ids[NUM_CHANNELS][MAX_CHANNEL_CLIPS] = {{0}};
  int channel_clip_count[NUM_CHANNELS] = {0};
  char channel_names[NUM_CHANNELS][64] = {{0}};
  int channel_thumb_tex[NUM_CHANNELS] = {0};
  int channel_connected[NUM_CHANNELS] = {0};

  /* Debug overlay drawer (solid-quad GPU rects + host-text labels). */
  overlay::Canvas ov;

  bool initialized = false;
};

/* ======================================================================
 * Pure helpers (no state)
 * ====================================================================== */

static int str_len(const char* s) {
  int n = 0;
  while (s[n]) n++;
  return n;
}

/* Log levels */
#define LOG_INFO  0
#define LOG_WARN  1
#define LOG_ERROR 2

static void log_msg(int level, const char* msg) {
  state_console_log(level, msg, str_len(msg));
}

// Removed: decl_param — using schema-based declaration now

static void log_structured(int level, const char* msg, const char* json) {
  state_console_log_structured(level, msg, str_len(msg), json, str_len(json));
}

/* Quick JSON snippet builder for structured logs */
static const char* json_ch_step(int ch, int step) {
  static char _jbuf[128];
  int p = 0;
  _jbuf[p++] = '{'; _jbuf[p++] = '"'; _jbuf[p++] = 'c'; _jbuf[p++] = 'h'; _jbuf[p++] = '"'; _jbuf[p++] = ':';
  _jbuf[p++] = '0' + ch;
  _jbuf[p++] = ','; _jbuf[p++] = '"'; _jbuf[p++] = 's'; _jbuf[p++] = 't'; _jbuf[p++] = 'e'; _jbuf[p++] = 'p'; _jbuf[p++] = '"'; _jbuf[p++] = ':';
  if (step >= 10) { _jbuf[p++] = '0' + step / 10; }
  _jbuf[p++] = '0' + step % 10;
  _jbuf[p++] = '}'; _jbuf[p] = 0;
  return _jbuf;
}

/* ======================================================================
 * State-touching helpers
 * ====================================================================== */

static bool live_note_for(const State& s, int ch, Event& out);  /* defined below */

/* Publish the current sequencer grid state as JSON */
static void publish_state(State& s) {
  /* Build a JSON string representing the grid and playback state.
   * Format: {"phase":N,"recording":B,"grid":[[steps],[steps],[steps],[steps]]}
   * Keep it compact since this runs every tick. */
  const int loop_steps = s.bars * NUM_STEPS;
  auto state = val::object();
  val::set(state, "phase", val::number(s.phase));
  val::set(state, "recording", val::boolean(s.record_held != 0));
  val::set(state, "loop_mode", val::number(s.loop_mode));  /* 0=Off 1=Overdub 2=Latch */
  /* Loop length in steps (16/32/64) — the web editor sizes its grid from this. */
  val::set(state, "loop_steps", val::number(loop_steps));
  val::set(state, "event_count", val::number(s.looper.event_count));

  /* Latch capture progress (0..1 over the add-window) while a phrase is being
   * captured, else -1. Drives the web editor's green capture bar, matching the
   * on-video overlay's. See the render() latch indicator for the same math. */
  double latch_capture = -1.0;
  if (s.latch && s.latch_capturing) {
    double inv = (double)s.grace_beats * (NUM_STEPS / 4.0);
    double add_window = (double)loop_steps - inv;
    if (add_window < 1e-4) add_window = (double)loop_steps;
    double elapsed = s.abs_phase - s.latch_start_abs;
    if (elapsed >= 0 && elapsed < add_window) latch_capture = elapsed / add_window;
  }
  val::set(state, "latch_capture", val::number(latch_capture));

  auto grid = val::array();
  for (int ch = 0; ch < NUM_CHANNELS; ch++) {
    auto channel = val::array();
    for (int st = 0; st < loop_steps; st++) {
      if (looper_has_event(&s.looper, ch, st)) {
        val::push(channel, val::number(st));
      }
    }
    val::push(grid, channel);
  }
  val::set(state, "grid", grid);

  /* Full note events (onset + gate length + channel) — the SAME data the debug
   * overlay draws its continuous bars from, so a web custom editor can mirror it
   * exactly (the `grid` above is onset-only / lossy). A held note appears here
   * too, growing, because looper_tick_pending extends its length in-place. */
  auto notes = val::array();
  for (int i = 0; i < s.looper.event_count; i++) {
    const Event& e = s.looper.events[i];
    auto n = val::object();
    val::set(n, "ch", val::number(e.channel));
    val::set(n, "start", val::number(e.start));
    val::set(n, "length", val::number(e.length));
    val::push(notes, n);
  }
  val::set(state, "notes", notes);

  /* Off-mode transient live notes (display only; the recorded `notes` above are
   * shown DISABLED in Off). Empty in Overdub/Latch (held notes live in `notes`). */
  auto live_notes = val::array();
  for (int ch = 0; ch < NUM_CHANNELS; ch++) {
    Event lv;
    if (live_note_for(s, ch, lv)) {
      auto n = val::object();
      val::set(n, "ch", val::number(lv.channel));
      val::set(n, "start", val::number(lv.start));
      val::set(n, "length", val::number(lv.length));
      val::push(live_notes, n);
    }
  }
  val::set(state, "live_notes", live_notes);

  /* Per-channel live gate (1 while sounding) — for lane highlight + the trigger
   * dots. Exact (no fade). */
  auto gates = val::array();
  for (int ch = 0; ch < NUM_CHANNELS; ch++)
    val::push(gates, val::number(s.gate_state[ch]));
  val::set(state, "gates", gates);

  /* Per-channel modulation outputs (out_1..out_4): the exact gate (1/0). */
  val::set(state, "out_1", val::number(s.ch_out[0]));
  val::set(state, "out_2", val::number(s.ch_out[1]));
  val::set(state, "out_3", val::number(s.ch_out[2]));
  val::set(state, "out_4", val::number(s.ch_out[3]));

  /* Trigger-rail event ring — {seq,on,channel,velocity}. The executor drains it
   * onto the global rail (drainTriggerRing). Empty unless send_to_rail is on. */
  auto triggers = val::array();
  for (int i = 0; i < s.trig_ring_len; i++) {
    auto e = val::object();
    val::set(e, "seq", val::number((double)s.trig_ring[i].seq));
    val::set(e, "on", val::boolean(s.trig_ring[i].on));
    val::set(e, "channel", val::number(s.trig_ring[i].channel));
    val::set(e, "velocity", val::number(s.trig_ring[i].velocity));
    /* Optional precision subtree — uniform per instance (the current param), so
     * read here rather than widening LoopEv. >0 → strict with that deadline;
     * 0 → omit entirely (drain treats absence as "any"). */
    if (s.strict_deadline > 0.0f) {
      auto prec = val::object();
      val::set(prec, "mode", val::string("strict"));
      val::set(prec, "deadline", val::number((double)s.strict_deadline));
      val::set(e, "precision", prec);
    }
    val::push(triggers, e);
  }
  val::set(state, "triggers", triggers);

  state::setVal(state);
  val::release(state);
}

/* Push an on/off trigger event onto the ring (channel is 1-based). Only when
 * send_to_rail is enabled — otherwise the ring stays empty and the executor has
 * nothing to drain. */
static void push_trigger(State& s, bool on, int ch /*0-based*/) {
  if (!s.send_to_rail) return;
  LoopEv e;
  e.seq = ++s.trig_seq;
  e.on = on;
  e.channel = ch + 1;
  e.velocity = 1.0f;
  if (s.trig_ring_len == TRIG_RING_CAP) {
    for (int i = 1; i < TRIG_RING_CAP; i++) s.trig_ring[i - 1] = s.trig_ring[i];
    s.trig_ring_len--;
  }
  s.trig_ring[s.trig_ring_len++] = e;
}

/* Emit a gate transition for one channel (idempotent — no-op if already in the
 * requested state). ON fires the modulation output + audio + rail event; OFF
 * fires the rail off event. The modulation output (ch_out) is an EXACT gate
 * (1 while gated, 0 otherwise) — no flash/decay envelope. */
static void set_gate(State& s, int ch, bool on) {
  if (on && !s.gate_state[ch]) {
    s.gate_state[ch] = 1;
    s.ch_out[ch] = 1.0f;              /* per-channel modulation output = gate */
    push_trigger(s, /*on=*/true, ch); /* → global trigger rail */
    // Legacy direct-launch imports are no-op stubs; the rail is the live path.
    for (int i = 0; i < s.channel_clip_count[ch]; i++)
      resolume_trigger_clip(s.channel_clip_ids[ch][i], 1);
    host_trigger_audio(ch);
  } else if (!on && s.gate_state[ch]) {
    s.gate_state[ch] = 0;
    s.ch_out[ch] = 0.0f;              /* gate off → output off immediately */
    push_trigger(s, /*on=*/false, ch);
    for (int i = 0; i < s.channel_clip_count[ch]; i++)
      resolume_trigger_clip(s.channel_clip_ids[ch][i], 0);
  }
}

/* True if some recorded note on `ch` has its ONSET in (prev, phase] modulo the
 * loop — i.e. a note started this frame. Used to RETRIGGER abutting notes: when
 * one note ends exactly as the next begins, playback coverage never lapses (the
 * gate would otherwise stay continuously on and emit no edge), but the boundary
 * is a real new hit, so we force an off→on. Wrap-aware. */
static bool onset_crossed(const LooperCore* c, int ch, double prev, double phase,
                          double loop) {
  double advance = phase - prev;
  if (advance < 0) advance += loop;           /* frame's forward phase travel */
  if (advance <= 0) return false;             /* no advance (e.g. param edit) */
  for (int i = 0; i < c->event_count; i++) {
    if (c->events[i].channel != ch) continue;
    double d = c->events[i].start - prev;     /* forward distance prev → onset */
    d = std::fmod(d, loop);
    if (d < 0) d += loop;
    if (d > 0.0 && d <= advance) return true; /* onset landed in (prev, phase] */
  }
  return false;
}

/* Single source of truth for every channel's gate: the live press OR a
 * played-back note window covering the current phase. Called after any input
 * edge and every tick (as the phase moves through recorded windows). Mute
 * silences a channel you're actively holding (mute + that trigger). In Off mode
 * the recorded pattern is DISABLED, so only live presses gate. */
static void recompute_gates(State& s) {
  int active[NUM_CHANNELS];
  looper_active_channels(&s.looper, s.phase, active);
  bool pattern_on = (s.loop_mode != LOOP_OFF);
  double loop = s.looper.loop_length > 0 ? s.looper.loop_length : (double)NUM_STEPS;
  for (int ch = 0; ch < NUM_CHANNELS; ch++) {
    bool live = s.trigger_held[ch] && !s.mute_held && !s.delete_held;
    bool played = pattern_on && active[ch] && !(s.mute_held && s.trigger_held[ch]);
    bool want = live || played;
    /* Abutting-note retrigger: coverage stays on across a note boundary, but a
     * new playback onset was crossed → emit an off then on so downstream sees a
     * distinct new hit (and strict mode can hold a frame of "off" between). */
    if (want && played && s.gate_state[ch] &&
        onset_crossed(&s.looper, ch, s.prev_phase, s.phase, loop)) {
      set_gate(s, ch, false);
      set_gate(s, ch, true);
    } else {
      set_gate(s, ch, want);
    }
  }
}

/* Off-mode transient live note for a held channel: the note's onset + its length
 * grown to the current phase (with a small visible minimum so a fresh press
 * shows even when the transport is paused). Returns false when the channel has
 * no live note. Wrap-aware. */
static bool live_note_for(const State& s, int ch, Event& out) {
  if (!s.live_held[ch]) return false;
  const double loop = (double)(s.bars * NUM_STEPS);
  double len = s.phase - s.live_start[ch];
  if (len < 0) len += loop;
  if (len < 0.35) len = 0.35;
  if (len > loop) len = loop;
  out.channel = ch;
  out.start = s.live_start[ch];
  out.length = len;
  return true;
}

static void refresh_channels(State& s) {
  int clip_count = resolume_get_clip_count();
  for (int ch = 0; ch < NUM_CHANNELS; ch++) {
    s.channel_clip_count[ch] = 0;
    s.channel_names[ch][0] = 0;
    s.channel_thumb_tex[ch] = -1;
    s.channel_connected[ch] = 0;
  }
  for (int i = 0; i < clip_count; i++) {
    int ch = resolume_get_clip_channel(i);
    if (ch < 0 || ch >= NUM_CHANNELS) continue;
    if (s.channel_clip_count[ch] < MAX_CHANNEL_CLIPS) {
      s.channel_clip_ids[ch][s.channel_clip_count[ch]++] = resolume_get_clip_id(i);
    }
    if (s.channel_clip_count[ch] == 1) {
      resolume_get_clip_name(i, s.channel_names[ch], 64);
      s.channel_connected[ch] = resolume_get_clip_connected(i);
      s.channel_thumb_tex[ch] = resolume_load_thumbnail(i);
    }
  }
}

static void on_param_change(State& s, int index, double value) {
  int pressed = (value >= 0.5);

  if (index >= PID_TRIGGER_1 && index <= PID_TRIGGER_4) {
    int ch = index - PID_TRIGGER_1;
    int was = s.trigger_held[ch];
    s.trigger_held[ch] = pressed;

    if (pressed && !was) {
      /* Rising edge — open a note (records onset + starts measuring the hold) */
      if (s.delete_held) {
        looper_clear_channel(&s.looper, ch);
        s.delete_acted = 1;
        s.last_action_was_clear = 0;
        log_structured(LOG_INFO, "Clear channel", json_ch_step(ch + 1, -1));
      } else if (s.mute_held) {
        /* Muted press: silence only, no recording. */
      } else {
        int step = (int)s.phase % (s.bars * NUM_STEPS);
        s.last_action_was_clear = 0;
        /* Overdub/Latch RECORD the tap into the pattern; Off doesn't — but Off
         * still plays live (recompute_gates gates it) and shows a TRANSIENT
         * overlay note that vanishes on release. */
        bool recording = s.record_held || s.latch;
        if (recording) {
          /* Latch: the first trigger (or one past the one-LOOP window — 1, 2
           * or 4 bars) clears the pattern and (re)opens a capture window. */
          if (s.latch) {
            double inv_grace = (double)s.grace_beats * (NUM_STEPS / 4.0);
            if (looper_latch_press(&s.latch_capturing, &s.latch_start_abs,
                                   s.abs_phase, (double)(s.bars * NUM_STEPS), inv_grace))
              looper_clear_all(&s.looper);
          }
          looper_begin_note(&s.looper, ch, s.phase);
          log_structured(LOG_INFO, "Note on", json_ch_step(ch + 1, step));
        } else {
          /* Off mode: transient live note (display only, pattern untouched). */
          s.live_held[ch] = 1;
          s.live_start[ch] = s.phase;
        }
      }
    } else if (!pressed && was) {
      /* Falling edge — finalize a recorded note, or drop the transient one so it
       * disappears the instant the trigger releases. */
      if (s.record_held || s.latch) looper_end_note(&s.looper, ch, s.phase);
      else s.live_held[ch] = 0;
    }
  } else if (index == PID_DELETE) {
    if (pressed) {
      s.delete_held = 1;
      s.delete_acted = 0;
    } else if (s.delete_held) {
      /* Release: if no trigger was pressed during hold, do clear or undo */
      if (!s.delete_acted) {
        if (s.last_action_was_clear && s.looper.undo_count > 0) {
          /* Double-tap delete = undo */
          looper_undo(&s.looper);
          s.last_action_was_clear = 0;
          log_msg(LOG_INFO, "Undo (double-tap delete)");
        } else {
          looper_clear_all(&s.looper);
          s.last_action_was_clear = 1;
          log_msg(LOG_INFO, "Clear all");
        }
      }
      s.delete_held = 0;
    }
  } else if (index == PID_MUTE) {
    s.mute_held = pressed;
  } else if (index == PID_UNDO) {
    if (pressed) {
      looper_undo(&s.looper);
      s.last_action_was_clear = 0;
      log_msg(LOG_INFO, "Undo");
    }
  } else if (index == PID_REDO) {
    if (pressed) {
      looper_redo(&s.looper);
      s.last_action_was_clear = 0;
      log_msg(LOG_INFO, "Redo");
    }
  } else if (index == PID_LOOP_MODE) {
    /* The single Loop enum. Off disables the recorded pattern (kept in memory);
     * Overdub/Latch play + record. record_held/latch are derived mirrors. */
    int m = (int)(value + 0.5);
    m = m < LOOP_OFF ? LOOP_OFF : (m > LOOP_LATCH ? LOOP_LATCH : m);
    if (m != s.loop_mode) {
      /* Finalize/drop any in-flight notes so a switch mid-hold doesn't dangle. */
      for (int ch = 0; ch < NUM_CHANNELS; ch++) {
        looper_end_note(&s.looper, ch, s.phase);  /* no-op if not pending */
        s.live_held[ch] = 0;
      }
      s.loop_mode = m;
      s.record_held = (m != LOOP_OFF);     /* Overdub/Latch record */
      s.latch = (m == LOOP_LATCH);
      s.latch_capturing = 0;               /* re-arm latch capture */
      /* Entering Off with triggers physically held → show them as transient. */
      if (m == LOOP_OFF) {
        for (int ch = 0; ch < NUM_CHANNELS; ch++)
          if (s.trigger_held[ch] && !s.mute_held && !s.delete_held) {
            s.live_held[ch] = 1;
            s.live_start[ch] = s.phase;
          }
      }
      log_msg(LOG_INFO, m == LOOP_OFF ? "Loop: off (pattern disabled)"
                        : m == LOOP_LATCH ? "Loop: latch" : "Loop: overdub");
    }
  } else if (index == PID_BARS) {
    int nb = (int)(value + 0.5);
    nb = nb >= 3 ? MAX_BARS : (nb == 2 ? 2 : 1);   /* select values: 1 / 2 / 4 */
    if (nb != s.bars) {
      /* Finalize/drop in-flight notes before the clock re-maps under them. */
      for (int ch = 0; ch < NUM_CHANNELS; ch++) {
        looper_end_note(&s.looper, ch, s.phase);   /* no-op if not pending */
        s.live_held[ch] = 0;
      }
      s.bars = nb;
      s.bar_in_loop %= nb;
      looper_set_loop_length(&s.looper, (double)(nb * NUM_STEPS));
      /* Keep phase inside the new loop this frame; tick() re-derives it. */
      s.phase = std::fmod(s.phase, (double)(nb * NUM_STEPS));
      s.prev_phase = s.phase;
      s.latch_capturing = 0;                       /* re-arm against the new length */
      log_msg(LOG_INFO, nb == 1 ? "Loop length: 1 bar"
                        : nb == 2 ? "Loop length: 2 bars" : "Loop length: 4 bars");
    }
  } else if (index == PID_SHOW_OVERLAY) {
    s.show_overlay = pressed;
  } else if (index == PID_ANCHOR) {
    int a = (int)(value + 0.5);
    s.anchor = a < 0 ? 0 : (a > 3 ? 3 : a);
  } else if (index == PID_OVERLAY_OPACITY) {
    float o = (float)value;
    s.overlay_opacity = o < 0.0f ? 0.0f : (o > 1.0f ? 1.0f : o);
  } else if (index == PID_SEND_TO_RAIL) {
    s.send_to_rail = pressed;
  } else if (index == PID_QUANTIZE_START) {
    s.quantize_start = pressed;
    looper_set_quantize(&s.looper, s.quantize_start, s.quantize_length);
  } else if (index == PID_QUANTIZE_LENGTH) {
    s.quantize_length = pressed;
    looper_set_quantize(&s.looper, s.quantize_start, s.quantize_length);
  } else if (index == PID_QUANTIZE_START_AMOUNT) {
    float a = (float)value;
    s.quantize_start_amount = a < 0.0f ? 0.0f : (a > 1.0f ? 1.0f : a);
    looper_set_quantize_start_amount(&s.looper, (double)s.quantize_start_amount);
  } else if (index == PID_GRACE) {
    /* Param is in fractions of a beat; core wants loop units. 4 beats / bar
     * (NUM_STEPS steps), so 1 beat == NUM_STEPS/4 units. */
    s.grace_beats = (float)value;
    looper_set_grace(&s.looper, (double)s.grace_beats * (NUM_STEPS / 4.0));
  } else if (index == PID_STRICT_DEADLINE) {
    float d = (float)value;
    s.strict_deadline = d < 0.0f ? 0.0f : (d > 250.0f ? 250.0f : d);
  }

  /* One authority for gate emission — reflect the new input state immediately. */
  recompute_gates(s);
}

/* --- State change handler (called by host when canonical state is modified) --- */

/* Buffer layout for reading grid from state document. Sized for the LONGEST
 * loop (4 bars = 64 steps) so a saved multi-bar grid restores whole. */
struct GridReadBuf {
  /* 4 channels, each: [i32 count][i32 steps[64]] */
  int32_t ch0_count; int32_t ch0_steps[MAX_LOOP_STEPS];
  int32_t ch1_count; int32_t ch1_steps[MAX_LOOP_STEPS];
  int32_t ch2_count; int32_t ch2_steps[MAX_LOOP_STEPS];
  int32_t ch3_count; int32_t ch3_steps[MAX_LOOP_STEPS];
};

/* Paths for state.read layout (packed, null-separated) */
static const char grid_paths[] =
  "/grid/0\0"   /* 0: offset 0, len 7 */
  "/grid/1\0"   /* 1: offset 8, len 7 */
  "/grid/2\0"   /* 2: offset 16, len 7 */
  "/grid/3\0";  /* 3: offset 24, len 7 */

#define GRID_CH_SIZE (4 + MAX_LOOP_STEPS * 4)  /* i32 count + i32[64] */

static JDocField grid_layout[NUM_CHANNELS] = {
  { 0,  7, JDOC_TYPE_ARRAY_I32, 0 * GRID_CH_SIZE, MAX_LOOP_STEPS },
  { 8,  7, JDOC_TYPE_ARRAY_I32, 1 * GRID_CH_SIZE, MAX_LOOP_STEPS },
  { 16, 7, JDOC_TYPE_ARRAY_I32, 2 * GRID_CH_SIZE, MAX_LOOP_STEPS },
  { 24, 7, JDOC_TYPE_ARRAY_I32, 3 * GRID_CH_SIZE, MAX_LOOP_STEPS },
};

static void load_grid_from_state(State& s) {
  struct GridReadBuf buf;
  JDocResult results[NUM_CHANNELS];

  int overflow = state_read(
    (const char*)grid_layout, NUM_CHANNELS,
    grid_paths,
    (char*)&buf, (int)sizeof(buf),
    (char*)results);
  (void)overflow;

  /* Only update if we actually got grid data */
  int any_found = 0;
  for (int i = 0; i < NUM_CHANNELS; i++) {
    if (results[i].found) any_found = 1;
  }
  if (!any_found) return;

  /* Rebuild looper events from the grid arrays */
  s.looper.event_count = 0;
  s.looper.undo_count = 0;
  s.looper.redo_count = 0;

  int32_t* channel_data[NUM_CHANNELS] = {
    buf.ch0_steps, buf.ch1_steps, buf.ch2_steps, buf.ch3_steps
  };
  int32_t channel_counts[NUM_CHANNELS] = {
    buf.ch0_count, buf.ch1_count, buf.ch2_count, buf.ch3_count
  };

  for (int ch = 0; ch < NUM_CHANNELS; ch++) {
    if (!results[ch].found) continue;
    int count = channel_counts[ch];
    if (count > MAX_LOOP_STEPS) count = MAX_LOOP_STEPS;
    for (int j = 0; j < count; j++) {
      int step = channel_data[ch][j];
      /* Accept up to the MAX loop; fold into the CURRENT loop length so a
       * multi-bar grid restored into a shorter loop still plays (mirrors
       * looper_set_loop_length — order-independent vs the `bars` patch). */
      if (step >= 0 && step < MAX_LOOP_STEPS && s.looper.event_count < MAX_EVENTS) {
        /* The persisted grid only carries onsets; restore each as a one-step
         * gate. Live-recorded gate lengths have full fidelity — this is the
         * lossy reload path (host save/restore of the onset grid only). */
        s.looper.events[s.looper.event_count].start =
            std::fmod((double)step, s.looper.loop_length);
        s.looper.events[s.looper.event_count].length = 1.0;
        s.looper.events[s.looper.event_count].channel = ch;
        s.looper.event_count++;
      }
    }
  }

  for (int ch = 0; ch < NUM_CHANNELS; ch++) s.looper.pending_index[ch] = -1;
  log_msg(LOG_INFO, "Grid loaded from state");
}

static int field_to_pid(const char* path, int pathLen) {
  struct { const char* name; int pid; } map[] = {
    {"trigger_1", PID_TRIGGER_1}, {"trigger_2", PID_TRIGGER_2},
    {"trigger_3", PID_TRIGGER_3}, {"trigger_4", PID_TRIGGER_4},
    {"delete", PID_DELETE}, {"mute", PID_MUTE},
    {"undo", PID_UNDO}, {"redo", PID_REDO},
    {"loop_mode", PID_LOOP_MODE}, {"bars", PID_BARS},
    {"show_overlay", PID_SHOW_OVERLAY},
    {"anchor", PID_ANCHOR}, {"overlay_opacity", PID_OVERLAY_OPACITY},
    {"send_to_rail", PID_SEND_TO_RAIL},
    {"quantize_start", PID_QUANTIZE_START},
    {"quantize_length", PID_QUANTIZE_LENGTH},
    {"quantize_start_amount", PID_QUANTIZE_START_AMOUNT},
    {"grace", PID_GRACE},
    {"strict_deadline", PID_STRICT_DEADLINE},
  };
  for (auto& m : map) {
    int mlen = std::strlen(m.name);
    if (pathLen == mlen && std::memcmp(path, m.name, mlen) == 0) return m.pid;
  }
  return -1;
}

/* ======================================================================
 * Exports (v2 instance ABI)
 * ====================================================================== */

/* Type-level setup: schema registration. Runs once per type. No GPU work.
 *
 * Built with the state::Schema builder so every param carries a proper display
 * name (+ compact short name), lives in a labelled GROUP, and has help text
 * (per-group markdown in the inspector's "?" mode + per-field tooltips). The
 * declaration ORDER below is the inspector layout order — it mirrors the
 * NanoLooper FFGL plugin's param panel (looper_plugin.mm). Field names/types/io
 * and defaults are unchanged from the original raw-JSON schema; only the
 * presentation metadata and ordering are new.
 *
 * io 5 = PrimaryInput (controls + tex_in), 6 = PrimaryOutput (out_N + tex_out).
 * The 4 per-channel `out_N` fields are modulation outputs — an exact 0/1 gate
 * (no decay); the module declares trigger_source so the executor drains its
 * "triggers" ring onto the global trigger rail (see drainTriggerRing). */
void module_init() {
  static const char id[] = "control.nanolooper";

  state::Schema schema;

  schema.helpField("intro",
    "## NanoLooper\n"
    "A 4-channel performance looper — 1, 2 or 4 **Bars** of 16th-note steps. Tap "
    "the **triggers** in time and it records what you play, looping it against "
    "the host's beat clock. Each channel can launch Resolume clips through the "
    "trigger rail and — in the dedicated NanoLooper plugin — pluck a built-in "
    "synth. Everything is live; nothing is saved between sessions.");

  // --- Triggers -------------------------------------------------------
  schema.group("triggers", "Triggers")
    .groupHelp(
      "The four performance pads. Tap one to play — and, when **Record** is armed, "
      "record — a note on that channel at the current beat; the note lasts as long "
      "as you hold it. Hold **Delete** and tap a pad to clear just that channel. "
      "Each channel maps to a color in the overlay and to clips tagged with its "
      "NanoLooper Ch marker.");
  schema.eventField("trigger_1", state::PrimaryInput).label("Trigger 1", "T1");
  schema.eventField("trigger_2", state::PrimaryInput).label("Trigger 2", "T2");
  schema.eventField("trigger_3", state::PrimaryInput).label("Trigger 3", "T3");
  schema.eventField("trigger_4", state::PrimaryInput).label("Trigger 4", "T4");

  // --- Editing --------------------------------------------------------
  schema.group("editing", "Editing")
    .groupHelp(
      "Modifiers that reshape what you've recorded. **Mute** and **Delete** are held "
      "while you tap a trigger; **Undo**/**Redo** step through edit history.");
  schema.eventField("delete", state::PrimaryInput).label("Delete", "Del");
  schema.boolField("mute", false, state::PrimaryInput,
    "Hold and tap a trigger to silence that channel without erasing it. Tapped alone, "
    "Delete clears everything; a double-tap undoes.").label("Mute", "Mute");
  schema.eventField("undo", state::PrimaryInput).label("Undo", "Undo");
  schema.eventField("redo", state::PrimaryInput).label("Redo", "Redo");

  // --- Loop -----------------------------------------------------------
  schema.group("loop", "Loop")
    .groupHelp(
      "What your taps do to the loop, and how long it is. **Off** disables the "
      "recorded pattern (it stops playing but is kept — switch back to restore it); "
      "triggers still play live. **Overdub** plays the loop and records your taps "
      "into it. **Latch** turns the first tap into a one-loop capture window for "
      "building a phrase hands-free. **Bars** sets the loop length — growing keeps "
      "the pattern where it is; shrinking folds it back into the shorter loop.");
  schema.selectField("loop_mode", LOOP_OVERDUB, state::PrimaryInput,
    { {"Off", LOOP_OFF}, {"Overdub", LOOP_OVERDUB}, {"Latch", LOOP_LATCH} },
    /*wrap=*/false,
    "Off = pattern disabled (kept, live triggers only); Overdub = play + record; "
    "Latch = one-loop capture.")
    .label("Loop", "Loop");
  schema.selectField("bars", 1, state::PrimaryInput,
    { {"1 Bar", 1}, {"2 Bars", 2}, {"4 Bars", 4} },
    /*wrap=*/false,
    "Loop length in bars (16/32/64 steps — the grid stays 16th notes). Growing "
    "keeps the recorded pattern in place; shrinking folds notes back into the "
    "shorter loop instead of dropping them.")
    .label("Bars", "Bars");

  // --- Quantize -------------------------------------------------------
  schema.group("quantize", "Quantize")
    .groupHelp(
      "Snap recorded notes to the 16th-note step grid. **Start** aligns each note's "
      "onset — fully, or partially via **Amount** (1 = hard snap, lower values pull "
      "the onset only that fraction of the way to the grid, tightening the feel while "
      "keeping it human). **Length** aligns its duration. **Grace** sets how forgiving "
      "overwrite and truncation are near a step boundary.");
  schema.boolField("quantize_start", false, state::PrimaryInput,
    "Snap each recorded note's start to the step grid (see Amount for partial snap).")
    .label("Quantize Start", "Q Start");
  schema.floatField("quantize_start_amount", 1.0f, 0.0f, 1.0f, state::PrimaryInput,
    /*magnitude=*/nullptr, /*step=*/0.f, /*units=*/nullptr,
    "How far a quantized onset moves toward its grid position. 1 = full snap "
    "(classic quantize); 0.5 = halfway there; 0 = untouched. Only applies while "
    "Quantize Start is on.")
    .label("Quantize Amount", "Q Amt");
  schema.boolField("quantize_length", false, state::PrimaryInput,
    "Snap each recorded note's length to whole grid steps.")
    .label("Quantize Length", "Q Len");
  schema.floatField("grace", 0.0625f, 0.0f, 1.0f, state::PrimaryInput,
    /*magnitude=*/nullptr, /*step=*/0.f, /*units=*/"beats",
    "Overwrite grace, in fractions of a beat (default 1/64 note). A note truncated "
    "below this is deleted; it also decides swallow-vs-truncate when a new note grows "
    "over an old onset, and doubles as the trailing zone that ends a Latch capture.")
    .label("Grace", "Grace");

  // --- Output ---------------------------------------------------------
  schema.group("output", "Output")
    .groupHelp(
      "Where the looper sends its gates. **Send To Rail** publishes each channel's "
      "on/off onto the global trigger rail, launching clips tagged with that channel's "
      "NanoLooper Ch marker.");
  schema.boolField("send_to_rail", true, state::PrimaryInput,
    "Emit each channel's gate onto the trigger rail so it can launch Resolume clips. "
    "Off = the looper runs as a modulation/overlay source only, launching nothing.")
    .label("Send To Rail", "Rail");
  schema.floatField("strict_deadline", 0.0f, 0.0f, 250.0f, state::PrimaryInput,
    /*magnitude=*/nullptr, /*step=*/0.0f, /*units=*/"ms",
    "Strict on-screen coordination for launched clips, in milliseconds. **0 = Any** "
    "(default): triggers launch immediately, as usual. **Above 0 = Strict**: the "
    "launch is held until a rendered frame reflecting the trigger reaches the display "
    "(so the clip lands in sync and isn't lost to a dropped frame or the flaky "
    "connect/disconnect pipe). This value is the deadline — if that long elapses "
    "without confirmation the pipe is assumed borked and all queued triggers flush "
    "through, fully reconciling only the most recent.")
    .label("Strict Deadline", "Strict");

  // --- Display --------------------------------------------------------
  schema.group("display", "Display")
    .groupHelp(
      "The on-video sequencer overlay. It's sized to the smaller of the viewport's "
      "width/height and **Anchor**ed to a corner, at **Overlay Opacity**.");
  schema.boolField("show_overlay", true, state::PrimaryInput,
    "Draw the looper's lanes, playhead and status overlay on top of the video. Off = "
    "the image passes through untouched (the looper keeps running).")
    .label("Show Overlay", "Overlay");
  schema.selectField("anchor", 0, state::PrimaryInput,
    { {"Top Left", 0}, {"Bottom Left", 1}, {"Top Right", 2}, {"Bottom Right", 3} },
    /*wrap=*/false,
    "Which corner of the video the overlay panel sits in.")
    .label("Anchor", "Anchor");
  schema.floatField("overlay_opacity", 1.0f, 0.0f, 1.0f, state::PrimaryInput,
    /*magnitude=*/nullptr, /*step=*/0.f, /*units=*/nullptr,
    "Overall opacity of the overlay. 0 hides it (image passes through); 1 is fully opaque.")
    .label("Overlay Opacity", "Opacity");

  // --- Synth ----------------------------------------------------------
  schema.group("synth", "Synth")
    .groupHelp(
      "The built-in audio voice — dedicated NanoLooper plugin only. When enabled, each "
      "channel gate-on plucks an audible tone through the host audio bus. Inside a "
      "barrel sketch these fields have no effect (no synth is attached).");
  schema.boolField("synth", false, state::PrimaryInput,
    "Enable the built-in synth: each trigger gate-on plucks an audible note.")
    .label("Synth", "Synth");
  schema.floatField("synth_gain", 0.5f, 0.0f, 1.0f, state::PrimaryInput,
    /*magnitude=*/nullptr, /*step=*/0.f, /*units=*/nullptr,
    "Output level of the built-in synth.")
    .label("Synth Gain", "Gain");

  // --- System fields (ungrouped): modulation outputs + image passthrough.
  // out_N are the per-channel exact gates (wireable modulation outs; no decay).
  // Declaring tex_out (PrimaryOutput texture) is what makes the executor treat
  // the looper as a rendering stage — binding a writable output + calling
  // render() — instead of a modulation-source passthrough (which skips render()).
  schema.endGroup();
  schema.floatField("out_1", 0.f, 0.f, 1.f, state::PrimaryOutput).label("Out 1", "O1");
  schema.floatField("out_2", 0.f, 0.f, 1.f, state::PrimaryOutput).label("Out 2", "O2");
  schema.floatField("out_3", 0.f, 0.f, 1.f, state::PrimaryOutput).label("Out 3", "O3");
  schema.floatField("out_4", 0.f, 0.f, 1.f, state::PrimaryOutput).label("Out 4", "O4");
  schema.textureField("tex_in",  state::PrimaryInput);
  schema.textureField("tex_out", state::PrimaryOutput);

  schema.capability(state::Capability::TriggerSource)
        .capability(state::Capability::ModulationSource)
        .capability(state::Capability::ModulationSourceMulti);

  state::init(id, {1, 2, 0}, schema);

  /* Compile the overlay toolbox's solid-quad shader up front (idempotent; also
   * retried lazily on first render if no GPU backend exists yet). */
  overlay::initShaders();
}

/* Per-instance construction: allocate State + one-time looper init. */
void* create() {
  auto* s = new State();
  looper_init(&s->looper, (double)NUM_STEPS);
  return s;
}

void destroy(void* self) {
  auto* s = static_cast<State*>(self);
  if (!s) return;
  s->ov.dispose();
  delete s;
}

/* Per-instance init tail: reset the looper + transport / edge-state /
 * timers and channel mapping, then publish initial state. */
void init(void* self) {
  auto* s = static_cast<State*>(self);
  if (!s) return;

  looper_init(&s->looper, (double)NUM_STEPS);
  s->bars = 1;          /* schema default; on_state_patched syncs */
  s->bar_in_loop = 0;
  s->prev_bar_phase = 0.0;
  s->phase = 0;
  s->prev_phase = 0;
  s->elapsed = 0;
  s->show_overlay = 1;
  s->anchor = 0;
  s->overlay_opacity = 1.0f;
  s->ws_connected = 0;

  for (int i = 0; i < NUM_CHANNELS; i++) {
    s->trigger_held[i] = 0;
    s->gate_state[i] = 0;
    s->channel_clip_count[i] = 0;
    s->channel_names[i][0] = 0;
    s->channel_thumb_tex[i] = -1;
    s->channel_connected[i] = 0;
    s->ch_out[i] = 0;
  }
  s->quantize_start = 0;
  s->quantize_length = 0;
  s->quantize_start_amount = 1.0f;
  looper_set_quantize(&s->looper, 0, 0);
  looper_set_quantize_start_amount(&s->looper, 1.0);
  s->grace_beats = 0.0625f;
  looper_set_grace(&s->looper, (double)s->grace_beats * (NUM_STEPS / 4.0));
  s->loop_mode = LOOP_OVERDUB;   /* default (see schema); on_state_patched syncs */
  s->latch = 0;
  s->latch_capturing = 0;
  s->latch_start_abs = 0.0;
  s->abs_phase = 0.0;
  for (int i = 0; i < NUM_CHANNELS; i++) { s->live_held[i] = 0; s->live_start[i] = 0.0; }
  /* send_to_rail keeps its schema default (on) via on_state_patched; reset the
   * ring so a re-init never replays stale events. */
  s->strict_deadline = 0.0f;  /* schema default: "any" (immediate) */
  s->trig_ring_len = 0;
  s->trig_seq = 0;
  s->delete_held = 0;
  s->delete_acted = 0;
  s->last_action_was_clear = 0;
  s->mute_held = 0;
  s->record_held = 1;   /* Overdub records by default; on_state_patched syncs */

  char key_buf[64];
  int key_len = state_get_key(key_buf, sizeof(key_buf) - 1);
  key_buf[key_len] = 0;

  /* Build: "NanoLooper initialized as <key>" */
  static char init_msg[128];
  int p = 0;
  const char* prefix = "NanoLooper initialized as ";
  while (*prefix) init_msg[p++] = *prefix++;
  for (int i = 0; i < key_len && p < 127; i++) init_msg[p++] = key_buf[i];
  init_msg[p] = 0;

  log_msg(LOG_INFO, init_msg);
  publish_state(*s);

  s->initialized = true;
}

void tick(void* self, double dt) {
  auto* s = static_cast<State*>(self);
  if (!s) return;

  s->elapsed += dt;

  /* Advance phase from the host bar phase. The host clock wraps every BAR, so
   * for multi-bar loops we count the wraps to know which bar of the loop we're
   * in (backward jumps are wraps; the loop's bar-0 alignment is arbitrary —
   * the host doesn't report a bar index). */
  double bar = host_get_bar_phase();
  if (bar < s->prev_bar_phase - 0.5)
    s->bar_in_loop = (s->bar_in_loop + 1) % s->bars;
  s->prev_bar_phase = bar;
  s->prev_phase = s->phase;
  s->phase = ((double)s->bar_in_loop + bar) * NUM_STEPS;

  /* Monotonic transport clock (never wraps) — the reference the latch capture
   * window is measured against. */
  double loop_steps = (double)(s->bars * NUM_STEPS);
  double dphase = s->phase - s->prev_phase;
  if (dphase < 0) dphase += loop_steps;          /* crossed the loop seam */
  s->abs_phase += dphase;

  /* Gate emission is fully driven by recompute_gates: as the phase sweeps
   * through each recorded note's [start, start+length) window, gates turn on at
   * the onset and off exactly when the recorded gate length elapses. Wrap-safe
   * and frame-rate independent — no per-frame edge scan or fixed timer. */
  recompute_gates(*s);

  /* Grow any held (pending) note to the current time so the overlay shows it
   * extending from its onset while the trigger is down. Playback still ignores
   * pending notes; end_note finalizes + overwrites on release. */
  looper_tick_pending(&s->looper, s->phase);

  /* Modulation output tracks the gate EXACTLY — no flash/decay envelope. (Kept
   * pinned to gate_state here too so it's correct even if a gate was set outside
   * a transition.) */
  for (int ch = 0; ch < NUM_CHANNELS; ch++)
    s->ch_out[ch] = s->gate_state[ch] ? 1.0f : 0.0f;

  publish_state(*s);
}

void on_state_patched(void* self, int n, const char* pb, const int* off,
                      const int* len, const int* ops) {
  auto* s = static_cast<State*>(self);
  if (!s) return;

  bool grid_changed = false;
  for (int i = 0; i < n; i++) {
    if (ops[i] != state::PatchReplace) continue;
    int pid = field_to_pid(pb + off[i], len[i]);
    if (pid >= 0) {
      on_param_change(*s, pid, state::patchFloat(i));
    } else if (state::pathIs(pb + off[i], len[i], "grid")) {
      grid_changed = true;
    }
  }
  if (grid_changed) load_grid_from_state(*s);
}


/* Is this note sounding at `phase`? True when phase falls in the note's
 * [start, start+length) window (wrap-aware). Per-NOTE, so only the step actually
 * under the playhead lights up — not every recorded note in the channel. */
static bool note_active(const Event& e, double phase, double loop) {
  double len = e.length;
  if (len <= 0.0) return false;
  if (len > loop) len = loop;
  double d = phase - e.start;
  d = std::fmod(d, loop);
  if (d < 0) d += loop;
  return d < len;
}

/* Draw one recorded note as a continuous bar on a lane's timeline. start/length
 * are in loop units [0, NUM_STEPS); the bar may wrap the loop seam, so it's
 * drawn as up to two segments. A bright leading edge marks the true onset.
 * `playing` is per-NOTE (this note is under the playhead); `op` is the overlay
 * opacity multiplier. */
static void draw_note_bar(overlay::Canvas& ov, const Event& e, double loop,
                          float track_x, float track_w, float bar_y, float bar_h,
                          float cr, float cg, float cb,
                          bool muted, bool playing, float scale, float op,
                          bool disabled = false) {
  double s0 = e.start;
  if (s0 < 0) s0 = 0; else if (s0 >= loop) s0 -= loop;
  double rem = e.length;
  if (rem > loop) rem = loop;

  // `disabled` (Off mode): the recorded pattern is kept but inactive — draw it
  // dim + desaturated so it reads as "there but off".
  const float body_a = disabled ? 0.26f * op
                     : (muted ? 0.30f : (playing ? 0.95f : 0.72f)) * op;
  const float edge_a = disabled ? 0.40f * op : (muted ? 0.45f : 1.0f) * op;
  const float bf = disabled ? 0.42f : (playing ? 0.85f : 0.55f);   // body brightness

  /* Minimum on-screen width so even a genuinely tiny gate stays visible (wide
   * enough to show body past the onset marker). */
  const float min_w = 7.0f * scale;

  bool first = true;
  while (rem > 1e-4) {
    double seg = rem;
    if (s0 + seg > loop) seg = loop - s0;      // clip at the seam → wrap

    float x = track_x + (float)(s0 / loop) * track_w;
    float w = (float)(seg / loop) * track_w;
    if (w < min_w) w = min_w;

    ov.fillRect(x, bar_y, w, bar_h,
                overlay::rgba(cr * bf, cg * bf, cb * bf, body_a));
    if (first)   // onset marker (only on the leading segment)
      ov.fillRect(x, bar_y, 3.5f * scale, bar_h, overlay::rgba(cr, cg, cb, edge_a));

    rem -= seg;
    s0 += seg;
    if (s0 >= loop) s0 -= loop;
    first = false;
  }
}

void render(void* self, int vp_w, int vp_h) {
  auto* s = static_cast<State*>(self);
  if (!s) return;

  /* The looper now owns its output texture (tex_out), so it must always write
   * it — even when the overlay is hidden, forward the input so downstream sees
   * the image (this replaces the executor's old alias-passthrough). */
  if (!s->show_overlay || vp_w <= 0 || vp_h <= 0) {
    int out = gpu::Device::textureForField("tex_out").id;
    if (out >= 0) {
      int in = gpu::Device::textureForField("tex_in").id;
      if (in >= 0) gpu::Device::copy(gpu::Texture{in}, gpu::Texture{out});
      else         gpu::Device::clear(gpu::Texture{out}, 0, 0, 0, 0);
      gpu::Device::submit();
    }
    return;
  }

  /* Opacity: 0 → nothing to draw, forward the input like an off overlay. */
  const float op = s->overlay_opacity;
  if (op <= 0.001f) {
    int out = gpu::Device::textureForField("tex_out").id;
    if (out >= 0) {
      int in = gpu::Device::textureForField("tex_in").id;
      if (in >= 0) gpu::Device::copy(gpu::Texture{in}, gpu::Texture{out});
      else         gpu::Device::clear(gpu::Texture{out}, 0, 0, 0, 0);
      gpu::Device::submit();
    }
    return;
  }

  overlay::Canvas& ov = s->ov;
  ov.begin(vp_w, vp_h);

  /* Every alpha runs through the overlay-opacity multiplier. */
  auto C = [op](float r, float g, float b, float a) {
    return overlay::rgba(r, g, b, a * op);
  };

  /* Design at 1080 reference units, scaled by the SMALLER viewport dimension so
   * the panel stays a compact block (never a full-width bar) and reads the same
   * on wide, tall, or square outputs. */
  const int   S = vp_w < vp_h ? vp_w : vp_h;
  const float scale = (float)S / 1080.0f;
  const float pad = 24.0f * scale;         // gap from the viewport corner
  const float m = 22.0f * scale;           // inner content margin

  const float title_sz = 30.0f * scale;
  const float label_sz = 22.0f * scale;
  const float small_sz = 17.0f * scale;
  const float lane_h = 46.0f * scale;
  const float lane_gap = 10.0f * scale;

  /* Content layout in box-LOCAL coords (origin = panel top-left). */
  const float title_y = m;
  const float lanes_top = title_y + title_sz + 20.0f * scale;
  const float lanes_h = NUM_CHANNELS * (lane_h + lane_gap) - lane_gap;
  const float row_y = lanes_top + lanes_h + 16.0f * scale;   // bottom modifier row
  const float boxH = row_y + small_sz + 14.0f * scale + m;
  const float boxW = (float)S - 2.0f * pad;

  const float number_x = m + 6.0f * scale;
  const float name_x = m + 34.0f * scale;
  const float track_x = m + 200.0f * scale;
  float track_w = boxW - m - track_x;
  if (track_w < 60.0f * scale) track_w = 60.0f * scale;

  /* Anchor the box to the chosen corner (0=TL, 1=BL, 2=TR, 3=BR). */
  const bool left = (s->anchor == 0 || s->anchor == 1);
  const bool top  = (s->anchor == 0 || s->anchor == 2);
  const float ox = left ? pad : ((float)vp_w - pad - boxW);
  const float oy = top  ? pad : ((float)vp_h - pad - boxH);

  /* --- Panel background (first rect → behind everything else) --- */
  ov.fillRect(ox, oy, boxW, boxH, C(0.04f, 0.05f, 0.07f, 0.72f));

  /* --- Title + loop-mode badge --- */
  ov.text("LOOPER", ox + m, oy + title_y, title_sz, C(0.90f, 0.92f, 0.95f, 0.95f), 800);
  {
    const float bx = ox + m + title_sz * 5.6f, by = oy + title_y + 4.0f * scale;
    if (s->loop_mode == LOOP_OVERDUB)
      ov.text("\xe2\x97\x8f OVERDUB", bx, by, label_sz, C(1.0f, 0.28f, 0.28f, 1.0f), 700);
    else if (s->loop_mode == LOOP_LATCH)
      ov.text("\xe2\x97\x89 LATCH", bx, by, label_sz, C(0.45f, 1.0f, 0.6f, 0.95f), 700);
    else  /* Off — pattern disabled */
      ov.text("\xe2\x96\xa0 OFF", bx, by, label_sz, C(0.62f, 0.66f, 0.72f, 0.8f), 700);
  }

  /* --- Connection status (top-right of the panel, pulsing dot) --- */
  {
    float t = (float)s->elapsed;
    float pulse = 0.35f + 0.65f * (0.5f + 0.5f * sinf(t * 6.0f));
    float dot = 12.0f * scale;
    float cx = ox + boxW - m - 144.0f * scale;
    ov.fillRect(cx, oy + title_y + 5.0f * scale, dot, dot, C(0.30f, 0.55f, 1.0f, pulse));
    ov.text("connecting", cx + dot + 8.0f * scale, oy + title_y + 1.0f * scale, small_sz,
            C(0.60f, 0.72f, 1.0f, 0.80f));
  }

  /* --- Beat gridlines behind the lanes (4 beats / bar; bar lines brighter) --- */
  const double loop = (double)(s->bars * NUM_STEPS);
  const int n_beats = 4 * s->bars;
  for (int beat = 0; beat <= n_beats; beat++) {
    float gx = ox + track_x + ((float)beat / (float)n_beats) * track_w;
    float a = (beat % 4 == 0) ? 0.34f : 0.15f;
    ov.fillRect(gx, oy + lanes_top, 1.5f * scale, lanes_h, C(0.60f, 0.66f, 0.78f, a));
  }

  /* --- Lanes: continuous note bars per channel --- */
  for (int ch = 0; ch < NUM_CHANNELS; ch++) {
    float ly = oy + lanes_top + ch * (lane_h + lane_gap);
    float cr = CH_R[ch], cg = CH_G[ch], cb = CH_B[ch];
    bool muted = s->mute_held && s->trigger_held[ch];
    float dim = muted ? 0.35f : 1.0f;
    bool gated = s->gate_state[ch] != 0;

    /* Lane track background — the CHANNEL flash: a steady highlight while the
     * channel is gated, off the instant it releases (no fade/decay envelope).
     * (The individual note bars light per-step, below.) */
    float track_a = gated ? 0.16f : 0.05f;
    ov.fillRect(ox + track_x, ly, track_w, lane_h, C(cr, cg, cb, track_a));

    /* Channel number + optional clip name. */
    char num[2] = { char('1' + ch), 0 };
    ov.text(num, ox + number_x, ly + lane_h * 0.5f - label_sz * 0.55f, label_sz,
            C(cr * dim, cg * dim, cb * dim, 1.0f), 700);
    if (s->channel_names[ch][0])
      ov.text(s->channel_names[ch], ox + name_x, ly + lane_h * 0.5f - small_sz * 0.55f,
              small_sz, C(0.70f, 0.72f, 0.76f, 0.80f));

    /* Note bars (unquantized: positioned/sized by real onset + gate length).
     * Only the note under the playhead flashes bright. In Off mode the recorded
     * pattern is drawn DISABLED (dim); a held trigger's TRANSIENT note draws
     * bright and vanishes on release. */
    const bool disabled = (s->loop_mode == LOOP_OFF);
    float bar_pad = 6.0f * scale;
    for (int i = 0; i < s->looper.event_count; i++) {
      const Event& e = s->looper.events[i];
      if (e.channel != ch) continue;
      bool note_playing = !disabled && note_active(e, s->phase, loop);
      draw_note_bar(ov, e, loop, ox + track_x, track_w, ly + bar_pad, lane_h - 2.0f * bar_pad,
                    cr, cg, cb, muted, note_playing, scale, op, disabled);
    }
    Event lv;
    if (live_note_for(*s, ch, lv))
      draw_note_bar(ov, lv, loop, ox + track_x, track_w, ly + bar_pad, lane_h - 2.0f * bar_pad,
                    cr, cg, cb, /*muted=*/false, /*playing=*/true, scale, op, /*disabled=*/false);
  }

  /* --- Playhead: continuous position across all lanes --- */
  {
    float ph = (float)(s->phase / loop);
    if (ph < 0) ph = 0; else if (ph > 1) ph = 1;
    float px = ox + track_x + ph * track_w;
    ov.fillRect(px - 1.5f * scale, oy + lanes_top - 4.0f * scale, 3.0f * scale,
                lanes_h + 8.0f * scale, C(1.0f, 1.0f, 1.0f, 0.88f));
  }

  /* --- Trigger-state dots + modifier state (bottom row). The dots track the
   * gate EXACTLY — on while the channel is gated, off the instant it releases
   * (no hold/decay). --- */
  float row_abs = oy + row_y;
  for (int i = 0; i < NUM_CHANNELS; i++) {
    float x = ox + m + i * 44.0f * scale;
    float a = s->gate_state[i] ? 1.0f : 0.28f;
    ov.fillRect(x, row_abs, 16.0f * scale, 16.0f * scale, C(CH_R[i], CH_G[i], CH_B[i], a));
  }
  float mod_x = ox + m + NUM_CHANNELS * 44.0f * scale + 16.0f * scale;
  ov.text("DEL", mod_x, row_abs - 2.0f * scale, small_sz,
          C(1.0f, 0.30f, 0.30f, s->delete_held ? 1.0f : 0.30f), 700);
  ov.text("MUTE", mod_x + 56.0f * scale, row_abs - 2.0f * scale, small_sz,
          C(1.0f, 0.85f, 0.30f, s->mute_held ? 1.0f : 0.30f), 700);
  ov.text("Q.start", mod_x + 140.0f * scale, row_abs - 2.0f * scale, small_sz,
          C(0.55f, 0.80f, 1.0f, s->quantize_start ? 1.0f : 0.32f), 700);
  ov.text("Q.len", mod_x + 232.0f * scale, row_abs - 2.0f * scale, small_sz,
          C(0.55f, 0.80f, 1.0f, s->quantize_length ? 1.0f : 0.32f), 700);
  /* (Loop mode is shown as the header badge; the latch capture bar is below.) */

  /* Latch capture indicator: a thin green bar above the lanes that fills over
   * the window in which a trigger ADDS to the current phrase. It vanishes the
   * moment a trigger would instead CLEAR + restart (past the add-window, i.e.
   * inside the trailing inverse-grace zone or beyond) — its presence is the
   * direct "a press now adds / a press now wipes" cue. */
  if (s->latch && s->latch_capturing) {
    double inv = (double)s->grace_beats * (NUM_STEPS / 4.0);
    double add_window = loop - inv;
    if (add_window < 1e-4) add_window = loop;
    double elapsed = s->abs_phase - s->latch_start_abs;
    if (elapsed >= 0 && elapsed < add_window) {   /* a press right now ADDS */
      float prog = (float)(elapsed / add_window);
      float by = oy + lanes_top - 9.0f * scale;
      ov.fillRect(ox + track_x, by, track_w, 3.0f * scale, C(0.2f, 0.4f, 0.25f, 0.4f));
      ov.fillRect(ox + track_x, by, track_w * prog, 3.0f * scale, C(0.45f, 1.0f, 0.6f, 0.9f));
    }
  }

  ov.end();
}

} // namespace nanolooper
