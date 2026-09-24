/*
 * warp.legacy.zoom_scroller — "Zoom Scroller"
 *
 * v2 port of the Resolume Wire "ZoomScroller" patch (259 nodes): a procedural
 * pan-and-zoom "scroller" camera over the input texture, with a choppy,
 * quantized stepping aesthetic and an on-screen white box "gizmo" that juts in
 * the direction of motion.
 *
 * The original is a sprawling dataflow state machine (Counters, Delays,
 * Snapshots, Pack/Unpack, an 8-wide array path builder, Shuffle/Sort). State
 * machines are miserable in a node graph and trivial in C++, so this is a
 * behavioural reimplementation, not a node translation.
 *
 * ── Two nested loops + dwell ──────────────────────────────────────────────
 *  OUTER loop = a "sequence". On Retrigger (or automatically, `sequence_delay`
 *  after the previous sequence ends) we:
 *    1. pick a far TARGET by polar placement from centre
 *       (target_min/max_angle × target_min/max_radius),
 *    2. pick an ORIGIN set back toward centre behind it
 *       (origin_center_bias ± origin_variance), plus a small jitter,
 *    3. pick a ZOOM level via a wrapping random-walk accumulator
 *       (accum += Normal(0, scale_variance); scale = lerp(min,max, frac(accum))),
 *    4. build an L-shaped Manhattan staircase of N grid waypoints from origin→
 *       target (N ← target_step_distance / scale, clamped 1..7; x/y moves
 *       interleaved in a random order — the "turning corners" travel),
 *    5. pan through each waypoint; after the last, dwell `sequence_delay`, then
 *       start a brand-new sequence.
 *
 *  INNER loop = one "pan" (waypoint→waypoint), clocked by a metronome at
 *  `flicker_rate` Hz. It tweens start→target over sub_steps·(sub_step_frames+1)
 *  ticks, but the DISPLAYED position is snapshot-held at each sub-step boundary
 *  → the signature quantized stepping. At the end it dwells `sub_delay` before
 *  advancing to the next waypoint.
 *
 * ── Render ────────────────────────────────────────────────────────────────
 *  apply.hlsl scale+translates a sample of tex_in (the "Transform") and draws
 *  the gizmo box outline over it (the "Shape Render" + "Video Mixer"). The zoom
 *  reconstruction filter is selectable (`filter_mode`): Crisp (nearest), Linear,
 *  or Smooth (B-spline bicubic, default) — bicubic de-"crunches" the magnified
 *  pixel grid that a plain LOD-0 bilinear tap leaves on a single-mip texture.
 *
 * Re-architecture notes (flagged per DNODE_MIGRATION_NOTES §3/§4):
 *  - Timing is real-time (dt seconds); `flicker_rate` is the metronome Hz that
 *    sets the stepping cadence. The original's quantizer divided by a literal 3;
 *    here the hold cadence is `sub_step_frames+1` so `sub_steps` genuinely = the
 *    number of quantized steps per pan (the original's clear intent).
 *  - The Wire "Curve" easings (curve 12/10) are approximated with simple eases;
 *    they only reshape the radius / line-thickness distributions (cosmetic).
 *  - Positions live in cover-square coords; PAN_GAIN calibrates the roam.
 *  - The gizmo is shown for the whole pan (and its directional offset animates),
 *    rather than the original's per-metronome-tick blink ("flicker"). The blink
 *    is a subtlety we can restore if wanted; the always-on box reads as the same
 *    "animates to indicate direction" cue and is steadier.
 */

#include <gpu.h>
#include <host.h>
#include <effect_utils.h>
#include "zoom_scroller_shaders.h"

#include <cmath>
#include <cstdint>

namespace zoom_scroller {

static constexpr int    MAX_WAYPOINTS = 8;   // matches the original's 8-wide array
static constexpr float  TWO_PI        = 6.28318530717958647692f;
static constexpr float  PAN_GAIN      = 1.0f;// cover-square units per position unit
static constexpr double MAX_TICK_DT   = 0.1; // clamp starved frames (see tick())

struct Uniforms {
  float pan_x, pan_y;
  float scale;
  float aspect_x, aspect_y;
  float giz_off_x, giz_off_y;
  float giz_hw, giz_hh;
  float giz_thick;
  float giz_show;
  float giz_alpha;
  float giz_r, giz_g, giz_b;
  float filter_mode;          // 0 crisp / 1 linear / 2 smooth(bicubic)
  float _pad1, _pad2, _pad3;  // pad to 80 bytes (cbuffer 16-float alignment)
};

enum FilterMode { FILTER_CRISP = 0, FILTER_LINEAR = 1, FILTER_SMOOTH = 2 };

enum Phase { PHASE_PAN, PHASE_SUB_DWELL, PHASE_SEQ_DWELL };

struct State {
  // ---- schema-mirrored params ----
  float min_scale = 2.0f, max_scale = 5.0f;
  float scale_variance = 0.1f;
  float scale_override = 0.5f;
  float target_min_angle = -1.0f, target_max_angle = 1.0f;
  float target_min_radius = 0.3f, target_max_radius = 0.6f;
  float target_step_distance = 0.3f;
  float sub_delay = 0.3f, sequence_delay = 0.6f;
  float origin_center_bias = 0.3f, origin_variance = 0.1f;
  float flicker_rate = 15.0f;
  int   sub_steps = 6, sub_step_frames = 3;
  bool  show_gizmo = true;
  float gizmo_size = 0.9f, gizmo_squash = 0.7f, gizmo_width = 0.3f;
  float gizmo_r = 1.0f, gizmo_g = 1.0f, gizmo_b = 1.0f, gizmo_a = 1.0f;
  float gizmo_alpha = 0.75f, gizmo_motion_scale = 1.0f;
  int   filter_mode = FILTER_SMOOTH;   // bicubic by default (smooth zoom)

  // ---- runtime state machine ----
  bool   initialized = false;
  bool   seq_started = false;
  bool   pending_retrigger = false;
  bool   pending_override   = false;   // scale_override changed → pin next seq
  Phase  phase = PHASE_PAN;
  double tick_accum = 0.0;             // fractional metronome ticks
  double dwell_timer = 0.0;            // seconds remaining (sub/seq dwell)

  int    frame_counter = 0;           // metronome ticks into the current pan
  int    last_hold = -1;              // sub-step index of the held position

  float  px = 0.f, py = 0.f;          // displayed (held) pan position
  float  sx = 0.f, sy = 0.f;          // smooth tween position (for gizmo velocity)

  // current pan endpoints
  float  start_x = 0.f, start_y = 0.f;
  float  targ_x = 0.f,  targ_y = 0.f;

  // current sequence
  float  wp_x[MAX_WAYPOINTS] = {0};
  float  wp_y[MAX_WAYPOINTS] = {0};
  int    n_waypoints = 1;
  int    wp_index = 0;
  float  cur_scale = 3.0f;
  float  scale_accum = 0.0f;

  uint32_t rng = 0x1234567u;

  gpu::Buffer  uniform_buf;
  gpu::Sampler sampler;
};

static gpu::ComputePSO s_pso;

// ── small deterministic RNG (xorshift32) ──────────────────────────────────
static inline uint32_t xs(uint32_t& s) {
  s ^= s << 13; s ^= s >> 17; s ^= s << 5; return s;
}
static inline float rand01(uint32_t& s) { return (xs(s) >> 8) * (1.0f / 16777216.0f); }
static inline float rand_range(uint32_t& s, float a, float b) { return a + (b - a) * rand01(s); }
// Approximate a unit normal via the central-limit of 4 uniforms (mean 0, ~unit).
static inline float rand_normal(uint32_t& s, float stddev) {
  float u = rand01(s) + rand01(s) + rand01(s) + rand01(s) - 2.0f; // ~N(0,1/3)
  return u * 1.732050808f * stddev;
}
static inline float fracf(float x) { return x - std::floor(x); }
static inline float clampf(float x, float a, float b) { return x < a ? a : (x > b ? b : x); }
static inline float lerpf(float a, float b, float t) { return a + (b - a) * t; }

void module_init() {
  state::init("warp.legacy.zoom_scroller", {1, 0, 0},
    state::Schema()
      // ---- motion / zoom (primary) ----
      .floatField("min_scale",        2.0f, 0.0f, 10.0f, state::PrimaryInput, nullptr, 0.0f, nullptr,
                  "Lower bound of the random zoom level.")
      .floatField("max_scale",        5.0f, 0.0f, 10.0f, state::PrimaryInput, nullptr, 0.0f, nullptr,
                  "Upper bound of the random zoom level.")
      .floatField("scale_variance",   0.1f, 0.0f, 1.0f,  state::PrimaryInput, nullptr, 0.0f, nullptr,
                  "How far the zoom random-walks each new sequence.")
      .floatField("target_min_radius",0.3f, 0.0f, 2.0f,  state::PrimaryInput, nullptr, 0.0f, nullptr,
                  "Closest a sequence target lands from centre.")
      .floatField("target_max_radius",0.6f, 0.0f, 2.0f,  state::PrimaryInput, nullptr, 0.0f, nullptr,
                  "Farthest a sequence target lands from centre.")
      .intField  ("sub_steps",        6, 1, 30,          state::PrimaryInput, 0, nullptr,
                  "Quantized steps taken per pan (the choppiness).")
      .intField  ("sub_step_frames",  3, 0, 30,          state::PrimaryInput, 0, nullptr,
                  "Metronome ticks held per sub-step (dwell between steps).")
      .floatField("flicker_rate",     15.0f, 0.0f, 60.0f,state::PrimaryInput, nullptr, 0.0f, nullptr,
                  "Stepping metronome rate (Hz) — sets overall speed/choppiness.")
      .floatField("sub_delay",        0.3f, 0.0f, 1.0f,  state::PrimaryInput, nullptr, 0.0f, nullptr,
                  "Dwell (s) at the end of each pan before the next waypoint.")
      .floatField("sequence_delay",   0.6f, 0.0f, 1.0f,  state::PrimaryInput, nullptr, 0.0f, nullptr,
                  "Dwell (s) after a sequence ends before a new one starts.")
      .selectField("filter_mode", FILTER_SMOOTH, state::PrimaryInput, {
          {"Crisp (nearest)", FILTER_CRISP},
          {"Linear",          FILTER_LINEAR},
          {"Smooth (bicubic)",FILTER_SMOOTH},
      }, /*wrap=*/false, "Image reconstruction filter for the zoom (Smooth = bicubic, "
                         "softens the magnified pixel grid).")
      .boolField ("show_gizmo",       true,              state::PrimaryInput,
                  "Draw the motion-indicator box.")
      .eventField("retrigger",        state::PrimaryInput)
      // ---- tuning (secondary) ----
      .floatField("scale_override",   0.5f, 0.0f, 1.0f,  state::SecondaryInput, nullptr, 0.0f, nullptr,
                  "Pin the zoom to this fraction of [min,max] (resets the walk).")
      .floatField("target_min_angle", -1.0f, -1.0f, 1.0f,state::SecondaryInput, nullptr, 0.0f, nullptr,
                  "Lower bound of the target direction (turns).")
      .floatField("target_max_angle",  1.0f, -1.0f, 1.0f,state::SecondaryInput, nullptr, 0.0f, nullptr,
                  "Upper bound of the target direction (turns).")
      .floatField("target_step_distance", 0.3f, 0.0f, 1.0f, state::SecondaryInput, nullptr, 0.0f, nullptr,
                  "Grid step size: smaller → more waypoints per sequence.")
      .floatField("origin_center_bias", 0.3f, 0.0f, 1.0f, state::SecondaryInput, nullptr, 0.0f, nullptr,
                  "How far back toward centre the origin sits behind the target.")
      .floatField("origin_variance",   0.1f, 0.0f, 1.0f,  state::SecondaryInput, nullptr, 0.0f, nullptr,
                  "Randomness of the origin set-back.")
      .floatField("gizmo_size",   0.9f, 0.0f, 2.0f, state::SecondaryInput, nullptr, 0.0f, nullptr,
                  "Gizmo box size.")
      .floatField("gizmo_squash", 0.7f, 0.0f, 1.0f, state::SecondaryInput, nullptr, 0.0f, nullptr,
                  "Gizmo box vertical squash.")
      .floatField("gizmo_width",  0.3f, 0.0f, 1.0f, state::SecondaryInput, nullptr, 0.0f, nullptr,
                  "Gizmo outline thickness.")
      .rgbaField ("gizmo_color",  1.0f, 1.0f, 1.0f, 1.0f, state::SecondaryInput)
      .floatField("gizmo_alpha",  0.75f, 0.0f, 1.0f, state::SecondaryInput, nullptr, 0.0f, nullptr,
                  "Gizmo overlay opacity.")
      .floatField("gizmo_motion_scale", 1.0f, 0.0f, 5.0f, state::SecondaryInput, nullptr, 0.0f, nullptr,
                  "How far the gizmo juts per unit of motion.")
      // ---- I/O ----
      .textureField("tex_in",  state::PrimaryInput)
      .textureField("tex_out", state::PrimaryOutput)
      .capability(state::Capability::SeekableApproximate));

  if (gpu::Device::backend() == gpu::Backend::None) return;

  state::registerShaderSPV("zoom_scroller_apply", APPLY_SPV, APPLY_SPV_SIZE);
  auto cs = gpu::Device::createShaderModuleByName("zoom_scroller_apply");
  if (!cs) return;
  s_pso = gpu::Device::createComputePSO(cs, "main", gpu::Bindings()
      .tex2d(0).storageTex2d(1).sampler(2).uniform(3));

  state::log("zoom_scroller: module initialized");
}

void* create() {
  auto* s = new State();
  s->uniform_buf = gpu::Device::createBuffer(sizeof(Uniforms), gpu::BufferUsage::Uniform);
  s->sampler = gpu::Device::createSampler(gpu::FilterMode::Linear, gpu::AddressMode::ClampToEdge);
  return s;
}

void destroy(void* self) {
  auto* s = static_cast<State*>(self);
  if (!s) return;
  s->uniform_buf.release();
  s->sampler.release();
  delete s;
}

// Cosmetic easings approximating the Wire "Curve" presets (radius shaping).
static inline float ease_radius(float x) { return x * x; }   // bias toward min

// ── sequence / pan construction ───────────────────────────────────────────
static void start_pan(State* s, float sx0, float sy0, float tx, float ty) {
  s->start_x = sx0; s->start_y = sy0;
  s->targ_x = tx;   s->targ_y = ty;
  s->frame_counter = 0;
  s->last_hold = -1;
  s->px = sx0; s->py = sy0;
  s->sx = sx0; s->sy = sy0;
  s->phase = PHASE_PAN;
}

static void build_waypoints(State* s, float ox, float oy, float tx, float ty) {
  float dx = tx - ox, dy = ty - oy;
  float adx = std::fabs(dx), ady = std::fabs(dy);
  float manh = adx + ady;

  float tsd = clampf(s->target_step_distance, 0.001f, 1.0f);
  float step_dist = tsd / (s->cur_scale > 1e-3f ? s->cur_scale : 1e-3f);
  int steps = 1;
  if (step_dist > 1e-5f) steps = (int)std::lround(manh / step_dist);
  if (steps < 1) steps = 1;
  if (steps > MAX_WAYPOINTS - 1) steps = MAX_WAYPOINTS - 1;  // clamp 1..7

  int xsteps = 0;
  if (manh > 1e-6f) xsteps = (int)std::lround(steps * (adx / manh));
  if (xsteps > steps) xsteps = steps;
  int ysteps = steps - xsteps;

  // Per-axis move sizes: random weights (0.9..1.1) normalized so they sum to
  // the full delta — near-uniform spacing with a touch of jitter.
  float mvx[MAX_WAYPOINTS], mvy[MAX_WAYPOINTS];
  int mvn = 0;
  struct Mv { float vx, vy, t; } moves[MAX_WAYPOINTS];
  if (xsteps > 0) {
    float wsum = 0.f; for (int i = 0; i < xsteps; i++) { mvx[i] = rand_range(s->rng, 0.9f, 1.1f); wsum += mvx[i]; }
    for (int i = 0; i < xsteps; i++) moves[mvn++] = { dx * mvx[i] / wsum, 0.f, rand01(s->rng) };
  }
  if (ysteps > 0) {
    float wsum = 0.f; for (int j = 0; j < ysteps; j++) { mvy[j] = rand_range(s->rng, 0.9f, 1.1f); wsum += mvy[j]; }
    for (int j = 0; j < ysteps; j++) moves[mvn++] = { 0.f, dy * mvy[j] / wsum, rand01(s->rng) };
  }
  // Interleave x/y moves in a random order (Shuffle + Sort by timing) → the
  // staircase turns corners instead of doing all-x-then-all-y.
  for (int a = 0; a < mvn; a++)
    for (int b = a + 1; b < mvn; b++)
      if (moves[b].t > moves[a].t) { Mv tmp = moves[a]; moves[a] = moves[b]; moves[b] = tmp; }

  float cx = ox, cy = oy;
  for (int k = 0; k < mvn; k++) {
    cx += moves[k].vx; cy += moves[k].vy;
    s->wp_x[k] = cx; s->wp_y[k] = cy;
  }
  if (mvn < 1) { s->wp_x[0] = tx; s->wp_y[0] = ty; mvn = 1; }
  s->n_waypoints = mvn;
}

static void start_new_sequence(State* s) {
  // Zoom: pin to override if it just changed, else random-walk + frac wrap.
  if (s->pending_override) { s->scale_accum = clampf(s->scale_override, 0.0f, 1.0f); s->pending_override = false; }
  else s->scale_accum += rand_normal(s->rng, s->scale_variance);
  float frac = fracf(s->scale_accum);
  s->cur_scale = lerpf(s->min_scale, s->max_scale, frac);
  if (s->cur_scale < 0.01f) s->cur_scale = 0.01f;

  // Target: polar placement from centre.
  float ang = rand_range(s->rng, s->target_min_angle, s->target_max_angle) * TWO_PI;
  float rseed = rand01(s->rng);
  float radius = lerpf(s->target_min_radius, s->target_max_radius, ease_radius(rseed));
  float dirx = std::sin(ang), diry = std::cos(ang);
  float tx = dirx * radius * PAN_GAIN;
  float ty = diry * radius * PAN_GAIN;

  // Origin: set back toward centre behind the target + a small jitter.
  float back = rand_range(s->rng, s->origin_center_bias - s->origin_variance,
                                  s->origin_center_bias + s->origin_variance);
  float jang = rand01(s->rng) * TWO_PI;
  float jr = lerpf(0.05f, 0.10f, ease_radius(rand01(s->rng)));
  float ox = tx - dirx * back * PAN_GAIN + std::sin(jang) * jr * PAN_GAIN;
  float oy = ty - diry * back * PAN_GAIN + std::cos(jang) * jr * PAN_GAIN;

  build_waypoints(s, ox, oy, tx, ty);
  s->wp_index = 0;
  s->seq_started = true;
  start_pan(s, ox, oy, s->wp_x[0], s->wp_y[0]);
}

static void advance_waypoint(State* s) {
  s->wp_index++;
  if (s->wp_index >= s->n_waypoints) {
    s->phase = PHASE_SEQ_DWELL;
    s->dwell_timer = s->sequence_delay;
  } else {
    start_pan(s, s->targ_x, s->targ_y, s->wp_x[s->wp_index], s->wp_y[s->wp_index]);
  }
}

// One metronome tick of the active pan.
static void step_tick(State* s) {
  int fps = s->sub_step_frames + 1; if (fps < 1) fps = 1;
  int total = s->sub_steps * fps; if (total < 1) total = 1;
  s->frame_counter++;

  float t = clampf((float)s->frame_counter / (float)total, 0.0f, 1.0f);
  s->sx = lerpf(s->start_x, s->targ_x, t);   // smooth tween (skew 0.5 = linear)
  s->sy = lerpf(s->start_y, s->targ_y, t);

  int hold = s->frame_counter / fps;          // quantized sub-step index
  if (hold != s->last_hold) { s->px = s->sx; s->py = s->sy; s->last_hold = hold; }

  if (s->frame_counter >= total) {
    s->px = s->targ_x; s->py = s->targ_y;
    s->sx = s->targ_x; s->sy = s->targ_y;
    s->phase = PHASE_SUB_DWELL;
    s->dwell_timer = s->sub_delay;
  }
}

void init(void* self) {
  auto* s = static_cast<State*>(self);
  if (!s) return;
  s->initialized = (s_pso.valid() && s->uniform_buf.valid());
}

void tick(void* self, double dt) {
  auto* s = static_cast<State*>(self);
  if (!s || !s->initialized) return;
  if (dt < 0.0) dt = 0.0;
  // Clamp a starved frame's delta. When the tab is backgrounded / the display
  // sleeps / requestAnimationFrame throttles to sub-1fps, the runtime hands us a
  // multi-second dt; the metronome catch-up below would then fast-forward an
  // ENTIRE pan in a single frame — the quantized stepping collapses to a
  // teleport ("motion incorrect") and the motion gizmo, which draws the
  // held-vs-smooth lag (zero at a pan's start and end), never gets a frame to
  // show, so it just blinks once per sequence. Capping dt makes the effect run
  // in slow-motion under starvation but stay visually correct — the right call
  // for a live/"playable" effect. (Mirrors engine-worker's own reset of
  // `lastTime` "so the next frame's dt isn't a giant" on resume.)
  if (dt > MAX_TICK_DT) dt = MAX_TICK_DT;

  if (s->pending_retrigger) { s->pending_retrigger = false; start_new_sequence(s); }
  if (!s->seq_started)      { start_new_sequence(s); }

  switch (s->phase) {
    case PHASE_SEQ_DWELL:
      s->dwell_timer -= dt;
      if (s->dwell_timer <= 0.0) start_new_sequence(s);
      break;
    case PHASE_SUB_DWELL:
      s->dwell_timer -= dt;
      if (s->dwell_timer <= 0.0) advance_waypoint(s);
      break;
    case PHASE_PAN: {
      if (s->flicker_rate <= 0.0f) break;
      s->tick_accum += dt * (double)s->flicker_rate;
      // Cap the catch-up so a long stall (HMR/seek) can't spin thousands of ticks.
      int guard = 0;
      while (s->tick_accum >= 1.0 && s->phase == PHASE_PAN && guard++ < 256) {
        s->tick_accum -= 1.0;
        step_tick(s);
      }
      if (guard >= 256) s->tick_accum = 0.0;
      break;
    }
  }
}

void on_state_patched(void* self, int n, const char* pb, const int* off,
                      const int* len, const int* ops) {
  auto* s = static_cast<State*>(self);
  if (!s) return;
  for (int i = 0; i < n; i++) {
    if (ops[i] != state::PatchReplace) continue;
    const char* p = pb + off[i]; int l = len[i];
    if      (state::pathIs(p, l, "min_scale"))           s->min_scale = state::patchFloat(i);
    else if (state::pathIs(p, l, "max_scale"))           s->max_scale = state::patchFloat(i);
    else if (state::pathIs(p, l, "scale_variance"))      s->scale_variance = state::patchFloat(i);
    else if (state::pathIs(p, l, "scale_override"))    { s->scale_override = state::patchFloat(i); s->pending_override = true; }
    else if (state::pathIs(p, l, "target_min_angle"))    s->target_min_angle = state::patchFloat(i);
    else if (state::pathIs(p, l, "target_max_angle"))    s->target_max_angle = state::patchFloat(i);
    else if (state::pathIs(p, l, "target_min_radius"))   s->target_min_radius = state::patchFloat(i);
    else if (state::pathIs(p, l, "target_max_radius"))   s->target_max_radius = state::patchFloat(i);
    else if (state::pathIs(p, l, "target_step_distance"))s->target_step_distance = state::patchFloat(i);
    else if (state::pathIs(p, l, "sub_delay"))           s->sub_delay = state::patchFloat(i);
    else if (state::pathIs(p, l, "sequence_delay"))      s->sequence_delay = state::patchFloat(i);
    else if (state::pathIs(p, l, "origin_center_bias"))  s->origin_center_bias = state::patchFloat(i);
    else if (state::pathIs(p, l, "origin_variance"))     s->origin_variance = state::patchFloat(i);
    else if (state::pathIs(p, l, "flicker_rate"))        s->flicker_rate = state::patchFloat(i);
    else if (state::pathIs(p, l, "sub_steps"))           s->sub_steps = state::patchInt(i);
    else if (state::pathIs(p, l, "sub_step_frames"))     s->sub_step_frames = state::patchInt(i);
    else if (state::pathIs(p, l, "filter_mode"))         s->filter_mode = state::patchInt(i);
    else if (state::pathIs(p, l, "show_gizmo"))          s->show_gizmo = state::patchBool(i);
    else if (state::pathIs(p, l, "retrigger"))         { if (state::patchEvent(i)) s->pending_retrigger = true; }
    else if (state::pathIs(p, l, "gizmo_size"))          s->gizmo_size = state::patchFloat(i);
    else if (state::pathIs(p, l, "gizmo_squash"))        s->gizmo_squash = state::patchFloat(i);
    else if (state::pathIs(p, l, "gizmo_width"))         s->gizmo_width = state::patchFloat(i);
    else if (state::pathIs(p, l, "gizmo_color"))       { auto c = state::patchVec4(i); s->gizmo_r=c.x; s->gizmo_g=c.y; s->gizmo_b=c.z; s->gizmo_a=c.w; }
    else if (state::pathIs(p, l, "gizmo_alpha"))         s->gizmo_alpha = state::patchFloat(i);
    else if (state::pathIs(p, l, "gizmo_motion_scale"))  s->gizmo_motion_scale = state::patchFloat(i);
  }
}

void render(void* self, int vp_w, int vp_h) {
  auto* s = static_cast<State*>(self);
  if (!s || !s->initialized || vp_w <= 0 || vp_h <= 0) return;
  auto in  = gpu::Device::textureForField("tex_in");
  auto out = gpu::Device::textureForField("tex_out");
  if (!in.valid() || !out.valid()) return;

  auto [ax, ay] = fx::coverSquare(vp_w, vp_h);

  // Gizmo box geometry (cover-square half-extents) + motion offset.
  float box_hw = 0.5f * s->gizmo_size;
  float box_hh = 0.5f * s->gizmo_size * s->gizmo_squash;
  // Outline half-thickness (cover-square), linear from 0 (~1.5px at the 0.3
  // default, up to ~5px). The shader's +0.5 AA keeps a sub-pixel line visible as
  // a 1px hairline; a width-fade folded into the alpha takes it cleanly to zero
  // over the bottom of the range so the box can be dialled out entirely.
  float thick = 0.04f * s->gizmo_width;
  float width_fade = clampf(s->gizmo_width / 0.04f, 0.0f, 1.0f);  // 0 at 0 → hairline → gone
  // Motion velocity = (held − smooth) × 2 × scale × motion_scale (the original).
  float gox = (s->px - s->sx) * 2.0f * s->cur_scale * s->gizmo_motion_scale;
  float goy = (s->py - s->sy) * 2.0f * s->cur_scale * s->gizmo_motion_scale;

  Uniforms u = {};
  u.pan_x = s->px; u.pan_y = s->py;
  u.scale = s->cur_scale;
  u.aspect_x = ax; u.aspect_y = ay;
  u.giz_off_x = gox; u.giz_off_y = goy;
  u.giz_hw = box_hw; u.giz_hh = box_hh;
  u.giz_thick = thick;
  // Shown only while actively panning (hidden during sub/sequence dwells —
  // faithful to the original's "motion frame" gate, minus the per-tick blink).
  u.giz_show = (s->show_gizmo && s->phase == PHASE_PAN) ? 1.0f : 0.0f;
  u.giz_alpha = s->gizmo_alpha * s->gizmo_a * width_fade;
  u.giz_r = s->gizmo_r; u.giz_g = s->gizmo_g; u.giz_b = s->gizmo_b;
  u.filter_mode = (float)s->filter_mode;
  s->uniform_buf.writeOne(u);

  auto cp = gpu::ComputePass::begin();
  cp.setPSO(s_pso);
  cp.setTexture(in, 0, 0);
  cp.setTexture(out, 1, 1);
  cp.setSampler(s->sampler, 2);
  cp.setBuffer(s->uniform_buf, 3);
  cp.dispatch((vp_w + 7) / 8, (vp_h + 7) / 8);
  cp.end();
  gpu::Device::submit();
}

} // namespace zoom_scroller
