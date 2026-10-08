/*
 * filter.light.bar_scatter — contrast BETWEEN the four LED bars.
 *
 * When several bars flash together (the whole screen flashing, say) they read
 * as one blob. Bar Scatter breaks the blob up: each co-triggering bar is, at
 * random, passed, CHOKED (its new light removed) or DELAYED (shown from a
 * moment ago, the delayed bars staggered one step apart), so one flash lands
 * as a spread of separate hits. Strength scales the odds; Tilt leans them
 * toward the outer pair (+1, the centre passes) or the centre pair (−1). One
 * bar always survives, so the flash is never lost outright.
 *
 * Nothing latches. The input excites every band continuously, to whatever
 * degree; each flash that beats the band's recent peak rolls fresh dice, and
 * each frame the fate is re-derived from those dice against the live knobs.
 * So a fast, chaotic strobe gets a new scatter on every hit, and turning
 * Strength down releases a choked bar on the next frame.
 *
 * Bars have ~10 cells, so a "bar" here is a BAND of cells: Zones cuts each
 * bar into bands, and each band across the four bars is its own co-trigger
 * group (1 = the whole bar decides together). A band's onset is the mean
 * rise over its cells, and a choke only removes light that is new since the
 * flash, so cells that didn't flash pass untouched.
 *
 * Three passes, state GPU-resident — a readback's frame or two of latency
 * would let every flash leak through before its choke landed:
 *   reduce — one thread per cell: mean colour + luma, filed into a history ring
 *   sim    — single thread: onsets, co-triggers, the roll, choke and delay
 *   render — full-res: each pixel moves by its cell's change, so untouched
 *            cells pass through exactly
 */

#include <gpu.h>
#include <preview_scenario.h>
#include <host.h>
#include <val.h>
#include <effect_utils.h>
#include "bar_scatter_shaders.h"

#include <cmath>
#include <cstdint>

namespace bar_scatter {

static constexpr int BARS      = 4;
static constexpr int MAX_ROWS  = 16;
static constexpr int MAX_ZONES = 16;
static constexpr int MAX_CELLS = BARS * MAX_ROWS;
static constexpr int MAX_GROUPS = BARS * MAX_ZONES;
static constexpr int HIST      = 64;

// --- GPU-shared layouts (must match common.hlsl / the HLSL cbuffers) ---

struct CellStat   { float r, g, b, luma; };
static_assert(sizeof(CellStat) == 16, "CellStat layout mismatch");

struct CellState  { float ref, prev_r, prev_g, prev_b, snap_r, snap_g, snap_b, pad; };
static_assert(sizeof(CellState) == 32, "CellState layout mismatch");

struct GroupState {
  float det, hot, t_event, co, amp, u_hit, u_mode, u_order, u_jit, pad0, pad1, pad2;
};
static_assert(sizeof(GroupState) == 48, "GroupState layout mismatch");

struct ReduceUniforms { int32_t rows, tex_w, tex_h, head; };
static_assert(sizeof(ReduceUniforms) == 16, "ReduceUniforms layout mismatch");

struct SimUniforms {
  float    now, dt; int32_t rows, zones;
  int32_t  head, min_bars, do_reset; uint32_t seed;
  uint32_t frame; float strength, tilt, delay_share;
  float    sensitivity, window, choke_depth, choke_length;
  float    choke_release, delay_step, delay_jitter, delay_hold;
  float    return_time, rise_time, pad0, pad1;
};
static_assert(sizeof(SimUniforms) == 96, "SimUniforms layout mismatch");

struct RenderUniforms { int32_t rows, pad0, pad1, pad2; };
static_assert(sizeof(RenderUniforms) == 16, "RenderUniforms layout mismatch");

struct SlotTimes { float t[HIST]; };

// How fast a cell's onset reference climbs back after it drops: a jump
// faster than this reads as an onset, a fade slower than it doesn't.
static constexpr float kRiseTime = 0.15f;

struct State {
  gpu::Buffer stats_buf;      // transient per-cell input stats
  gpu::Buffer hist_buf;       // PERSISTENT colour history ring
  gpu::Buffer times_buf;      // the ring's slot times (CPU-owned, uploaded)
  gpu::Buffer cells_buf;      // PERSISTENT per-cell state
  gpu::Buffer groups_buf;     // PERSISTENT per bar-and-zone state
  gpu::Buffer proc_buf;       // per-cell processed colour
  gpu::Buffer reduce_uniform_buf;
  gpu::Buffer sim_uniform_buf;
  gpu::Buffer render_uniform_buf;
  bool initialized = false;

  // --- Schema-mirrored params ---
  float strength      = 0.6f;
  float tilt          = 0.0f;
  float delay_share   = 0.5f;
  int   seed          = 0;
  float sensitivity   = 0.25f;
  int   min_bars      = 2;
  float window        = 0.06f;
  float choke_depth   = 1.0f;
  float choke_length  = 0.25f;
  float choke_release = 0.15f;
  float delay_step    = 0.08f;
  float delay_jitter  = 0.3f;
  float delay_hold    = 0.3f;
  float return_time   = 0.12f;
  int   rows          = 10;
  int   zones         = 1;

  // --- Runtime state ---
  double    pending_dt  = 0.0;   // accumulated tick dt, consumed by render()
  double    now         = 0.0;   // the sim clock, s
  int       head        = 0;     // this frame's history slot
  uint32_t  frame       = 0;
  SlotTimes times       = {};
  bool      needs_reset = true;
  int       last_rows   = -1;
  int       last_zones  = -1;
};

static gpu::ComputePSO s_pso_reduce;
static gpu::ComputePSO s_pso_sim;
static gpu::ComputePSO s_pso_render;

static inline float clampf(float v, float lo, float hi) {
  return v < lo ? lo : (v > hi ? hi : v);
}
static inline int clampi(int v, int lo, int hi) {
  return v < lo ? lo : (v > hi ? hi : v);
}

void module_init() {
  state::init("filter.light.bar_scatter", {1, 0, 0},
    state::Schema()
      .helpField("intro",
        "## Bar Scatter\n"
        "Breaks up flashes that hit several bars at once. When bars light up "
        "together, each one is — at random — let through, **choked** (its new "
        "light held back) or **delayed** (shown a moment later, the delayed "
        "bars staggered one step apart), so one big flash lands as a spread "
        "of separate hits. One bar always gets through.\n\n"
        "**Try:** raise **Strength** over footage with big full-frame hits; "
        "**Tilt** +1 keeps the center pair and scatters the outer pair; "
        "**Delay Share** 1 turns every hit into a ripple across the rig.")
      // --- Scatter ---
      .group("scatter", "Scatter")
        .groupHelp(
          "**Strength** is the chance each co-triggering bar is hit, re-read "
          "every frame, so a turn of the knob lands at once. **Tilt** "
          "leans it: +1 always lets the center pair through and hits the outer "
          "pair twice as often, −1 the reverse. **Delay Share** splits the "
          "hits between choking (0) and delaying (1). **Seed** picks a "
          "different run of dice.")
      .floatField("strength",     0.6f,  0.0f, 1.0f, state::PrimaryInput,
                  nullptr, 0.f, nullptr,
                  "The chance each co-triggering bar is choked or delayed.")
        .label("Strength", "Str")
      .floatField("tilt",         0.0f, -1.0f, 1.0f, state::PrimaryInput,
                  nullptr, 0.f, nullptr,
                  "+1 protects the center pair and hits the outer pair; −1 "
                  "protects the outer pair.")
        .label("Tilt", "Tilt")
      .floatField("delay_share",  0.5f,  0.0f, 1.0f, state::PrimaryInput,
                  nullptr, 0.f, nullptr,
                  "Of the bars hit, the share delayed rather than choked.")
        .label("Delay Share", "Dly%")
      .intField  ("seed",         0, 0, 0x7FFFFFFF,  state::PrimaryInput).label("Seed", "Seed")
      // --- Detect ---
      .group("detect", "Detect")
        .groupHelp(
          "What counts as a flash. A bar (or a band of it — see *Zones*) is "
          "excited by how far its cells jump up at once, on average: fully at "
          "**Sensitivity**, not at all under half of it, and a slow fade-in "
          "barely at all. Every jump that beats the bar's own recent peak is "
          "a fresh flash with fresh dice, so a fast strobe re-scatters on "
          "every hit. The odds scale with how hard the OTHER bars flash within "
          "**Window** seconds — fully once **Min Bars** are flashing together; "
          "a bar flashing on its own always passes.")
      .floatField("sensitivity",  0.25f, 0.02f, 1.0f, state::PrimaryInput).label("Sensitivity", "Sens")
      .intField  ("min_bars",     2, 2, 4,            state::PrimaryInput).label("Min Bars", "MinB")
      .floatField("window",       0.06f, 0.0f, 0.25f, state::PrimaryInput,
                  nullptr, 0.f, "s").label("Window", "Wind")
      // --- Choke ---
      .group("choke", "Choke")
        .groupHelp(
          "A choked bar loses only the light that is new since the flash, by "
          "**Depth**; what was already lit stays. It is held for **Length**, "
          "then fades back in over **Release**.")
      .floatField("choke_depth",   1.0f,  0.0f, 1.0f, state::PrimaryInput).label("Depth", "Depth")
      .floatField("choke_length",  0.25f, 0.0f, 2.0f, state::PrimaryInput,
                  nullptr, 0.f, "s").label("Length", "Len")
      .floatField("choke_release", 0.15f, 0.0f, 1.0f, state::PrimaryInput,
                  nullptr, 0.f, "s").label("Release", "Rel")
      // --- Delay ---
      .group("delay", "Delay")
        .groupHelp(
          "Delayed bars show the picture from a moment ago, staggered: the "
          "first by one **Step**, the next by two, and so on, each nudged by "
          "**Jitter**. **Hold** after its flash has played, it fast-forwards "
          "back to live over **Return**. A new flash re-rolls it.")
      .floatField("delay_step",    0.08f, 0.02f, 0.4f, state::PrimaryInput,
                  nullptr, 0.f, "s").label("Step", "Step")
      .floatField("delay_jitter",  0.3f,  0.0f,  1.0f, state::PrimaryInput).label("Jitter", "Jit")
      .floatField("delay_hold",    0.3f,  0.0f,  2.0f, state::PrimaryInput,
                  nullptr, 0.f, "s").label("Hold", "Hold")
      .floatField("return_time",   0.12f, 0.01f, 1.0f, state::PrimaryInput,
                  nullptr, 0.f, "s").label("Return", "Ret")
      // --- Grid ---
      .group("grid", "Grid")
        .groupHelp(
          "**Rows** is cells per bar — one per LED segment. **Zones** cuts "
          "each bar into bands that decide separately: 1 treats each bar as "
          "one, and at Rows every segment rolls its own dice.")
      .intField  ("rows",  10, 1, MAX_ROWS,  state::PrimaryInput).label("Rows", "Rows")
      .intField  ("zones",  1, 1, MAX_ZONES, state::PrimaryInput).label("Zones", "Zones")
      .textureField("tex_in",  state::PrimaryInput)
      .textureField("tex_out", state::PrimaryOutput)
  );

  if (gpu::Device::backend() == gpu::Backend::None) return;

  state::registerShaderSPV("bar_scatter_reduce", REDUCE_SPV, REDUCE_SPV_SIZE);
  state::registerShaderSPV("bar_scatter_sim",    SIM_SPV,    SIM_SPV_SIZE);
  state::registerShaderSPV("bar_scatter_render", RENDER_SPV, RENDER_SPV_SIZE);
  auto cs_reduce = gpu::Device::createShaderModuleByName("bar_scatter_reduce");
  auto cs_sim    = gpu::Device::createShaderModuleByName("bar_scatter_sim");
  auto cs_render = gpu::Device::createShaderModuleByName("bar_scatter_render");
  if (!cs_reduce || !cs_sim || !cs_render) return;

  s_pso_reduce = gpu::Device::createComputePSO(cs_reduce, "main", gpu::Bindings()
      .tex2d(0)
      .storageRW(1)
      .storageRW(2)
      .uniform(3));
  s_pso_sim = gpu::Device::createComputePSO(cs_sim, "main", gpu::Bindings()
      .storage(0)
      .storage(1)
      .storage(2)
      .storageRW(3)
      .storageRW(4)
      .storageRW(5)
      .uniform(6));
  s_pso_render = gpu::Device::createComputePSO(cs_render, "main", gpu::Bindings()
      .tex2d(0)
      .storageTex2d(1)
      .uniform(2)
      .storage(3)
      .storage(4));

  state::log("bar_scatter: module initialized");
}

void* create() {
  auto* s = new State();
  s->stats_buf          = gpu::Device::createBuffer(sizeof(CellStat) * MAX_CELLS,        gpu::BufferUsage::Storage);
  s->hist_buf           = gpu::Device::createBuffer(sizeof(float) * 4 * HIST * MAX_CELLS, gpu::BufferUsage::Storage);
  s->times_buf          = gpu::Device::createBuffer(sizeof(SlotTimes),                   gpu::BufferUsage::Storage);
  s->cells_buf          = gpu::Device::createBuffer(sizeof(CellState) * MAX_CELLS,       gpu::BufferUsage::Storage);
  s->groups_buf         = gpu::Device::createBuffer(sizeof(GroupState) * MAX_GROUPS,     gpu::BufferUsage::Storage);
  s->proc_buf           = gpu::Device::createBuffer(sizeof(float) * 4 * MAX_CELLS,       gpu::BufferUsage::Storage);
  s->reduce_uniform_buf = gpu::Device::createBuffer(sizeof(ReduceUniforms),              gpu::BufferUsage::Uniform);
  s->sim_uniform_buf    = gpu::Device::createBuffer(sizeof(SimUniforms),                 gpu::BufferUsage::Uniform);
  s->render_uniform_buf = gpu::Device::createBuffer(sizeof(RenderUniforms),              gpu::BufferUsage::Uniform);
  return s;
}

void destroy(void* self) {
  auto* s = static_cast<State*>(self);
  if (!s) return;
  s->stats_buf.release();
  s->hist_buf.release();
  s->times_buf.release();
  s->cells_buf.release();
  s->groups_buf.release();
  s->proc_buf.release();
  s->reduce_uniform_buf.release();
  s->sim_uniform_buf.release();
  s->render_uniform_buf.release();
  delete s;
}

void init(void* self) {
  auto* s = static_cast<State*>(self);
  if (!s) return;
  s->initialized = false;
  s->pending_dt = 0.0;
  s->needs_reset = true;

  if (!s_pso_reduce.valid() || !s_pso_sim.valid() || !s_pso_render.valid()) return;
  if (!s->stats_buf.valid() || !s->hist_buf.valid() || !s->times_buf.valid()
      || !s->cells_buf.valid() || !s->groups_buf.valid() || !s->proc_buf.valid()
      || !s->reduce_uniform_buf.valid() || !s->sim_uniform_buf.valid()
      || !s->render_uniform_buf.valid()) return;

  s->initialized = true;
  state::log("bar_scatter: initialized");
}

void tick(void* self, double dt) {
  auto* s = static_cast<State*>(self);
  if (!s || !s->initialized) return;
  // Accumulated here, consumed in render(): one history slot per RENDERED
  // frame. Clamped so a stall can't open a hole in the ring.
  if (dt > 0.0) s->pending_dt += dt;
  if (s->pending_dt > 0.25) s->pending_dt = 0.25;
}

void on_state_patched(void* self, int n, const char* pb, const int* off,
                      const int* len, const int* ops) {
  auto* s = static_cast<State*>(self);
  if (!s) return;
  for (int i = 0; i < n; i++) {
    if (ops[i] != state::PatchReplace) continue;
    const char* path = pb + off[i];
    int plen = len[i];
    if      (state::pathIs(path, plen, "strength"))      s->strength      = state::patchFloat(i);
    else if (state::pathIs(path, plen, "tilt"))          s->tilt          = state::patchFloat(i);
    else if (state::pathIs(path, plen, "delay_share"))   s->delay_share   = state::patchFloat(i);
    else if (state::pathIs(path, plen, "seed"))          s->seed          = (int)state::patchFloat(i);
    else if (state::pathIs(path, plen, "sensitivity"))   s->sensitivity   = state::patchFloat(i);
    else if (state::pathIs(path, plen, "min_bars"))      s->min_bars      = (int)state::patchFloat(i);
    else if (state::pathIs(path, plen, "window"))        s->window        = state::patchFloat(i);
    else if (state::pathIs(path, plen, "choke_depth"))   s->choke_depth   = state::patchFloat(i);
    else if (state::pathIs(path, plen, "choke_length"))  s->choke_length  = state::patchFloat(i);
    else if (state::pathIs(path, plen, "choke_release")) s->choke_release = state::patchFloat(i);
    else if (state::pathIs(path, plen, "delay_step"))    s->delay_step    = state::patchFloat(i);
    else if (state::pathIs(path, plen, "delay_jitter"))  s->delay_jitter  = state::patchFloat(i);
    else if (state::pathIs(path, plen, "delay_hold"))    s->delay_hold    = state::patchFloat(i);
    else if (state::pathIs(path, plen, "return_time"))   s->return_time   = state::patchFloat(i);
    else if (state::pathIs(path, plen, "rows"))          s->rows          = (int)state::patchFloat(i);
    else if (state::pathIs(path, plen, "zones"))         s->zones         = (int)state::patchFloat(i);
  }
}

void render(void* self, int vp_w, int vp_h) {
  auto* s = static_cast<State*>(self);
  if (!s || !s->initialized || vp_w <= 0 || vp_h <= 0) return;
  auto in  = gpu::Device::textureForField("tex_in");
  auto out = gpu::Device::textureForField("tex_out");
  if (!in.valid() || !out.valid()) return;

  int rows  = clampi(s->rows, 1, MAX_ROWS);
  int zones = clampi(s->zones, 1, rows);
  // A new grid is a new set of cells and bands: start them clean.
  if (rows != s->last_rows || zones != s->last_zones) s->needs_reset = true;
  s->last_rows = rows;
  s->last_zones = zones;

  if (s->needs_reset) {
    s->now = 0.0;
    s->head = 0;
    s->frame = 0;
    for (int i = 0; i < HIST; i++) s->times.t[i] = -1e9f;   // never written
  }

  // One history slot per stepped frame; a re-render refiles the same one.
  float dt = (float)s->pending_dt;
  if (dt > 0.0f) {
    s->now += dt;
    s->head = (s->head + 1) % HIST;
    s->times.t[s->head] = (float)s->now;
  }
  s->times_buf.writeOne(s->times);

  ReduceUniforms ru = { rows, vp_w, vp_h, s->head };
  s->reduce_uniform_buf.writeOne(ru);

  SimUniforms su = {};
  su.now           = (float)s->now;
  su.dt            = dt;
  su.rows          = rows;
  su.zones         = zones;
  su.head          = s->head;
  su.min_bars      = clampi(s->min_bars, 2, BARS);
  su.do_reset      = s->needs_reset ? 1 : 0;
  su.seed          = (uint32_t)s->seed;
  su.frame         = s->frame;
  su.strength      = clampf(s->strength, 0.0f, 1.0f);
  su.tilt          = clampf(s->tilt, -1.0f, 1.0f);
  su.delay_share   = clampf(s->delay_share, 0.0f, 1.0f);
  su.sensitivity   = clampf(s->sensitivity, 0.02f, 1.0f);
  su.window        = clampf(s->window, 0.0f, 0.25f);
  su.choke_depth   = clampf(s->choke_depth, 0.0f, 1.0f);
  su.choke_length  = clampf(s->choke_length, 0.0f, 2.0f);
  su.choke_release = clampf(s->choke_release, 0.0f, 1.0f);
  su.delay_step    = clampf(s->delay_step, 0.02f, 0.4f);
  su.delay_jitter  = clampf(s->delay_jitter, 0.0f, 1.0f);
  su.delay_hold    = clampf(s->delay_hold, 0.0f, 2.0f);
  su.return_time   = clampf(s->return_time, 0.01f, 1.0f);
  su.rise_time     = kRiseTime;
  s->sim_uniform_buf.writeOne(su);

  RenderUniforms cu = { rows, 0, 0, 0 };
  s->render_uniform_buf.writeOne(cu);

  // Pass 1 — reduce: one thread per cell → colour + luma, into the ring.
  {
    auto cp = gpu::ComputePass::begin();
    cp.setPSO(s_pso_reduce);
    cp.setTexture(in, 0, 0);
    cp.setBuffer(s->stats_buf,          1);
    cp.setBuffer(s->hist_buf,           2);
    cp.setBuffer(s->reduce_uniform_buf, 3);
    cp.dispatch((BARS + 7) / 8, (rows + 7) / 8);
    cp.end();
  }

  // Pass 2 — sim (single-thread): onsets, the roll, choke and delay.
  {
    auto cp = gpu::ComputePass::begin();
    cp.setPSO(s_pso_sim);
    cp.setBuffer(s->stats_buf,       0);
    cp.setBuffer(s->hist_buf,        1);
    cp.setBuffer(s->times_buf,       2);
    cp.setBuffer(s->cells_buf,       3);
    cp.setBuffer(s->groups_buf,      4);
    cp.setBuffer(s->proc_buf,        5);
    cp.setBuffer(s->sim_uniform_buf, 6);
    cp.dispatch(1, 1, 1);
    cp.end();
  }

  // Pass 3 — render: each pixel moves by its cell's change.
  {
    auto cp = gpu::ComputePass::begin();
    cp.setPSO(s_pso_render);
    cp.setTexture(in,  0, 0);
    cp.setTexture(out, 1, 1);
    cp.setBuffer(s->render_uniform_buf, 2);
    cp.setBuffer(s->stats_buf,          3);
    cp.setBuffer(s->proc_buf,           4);
    cp.dispatch((vp_w + 7) / 8, (vp_h + 7) / 8);
    cp.end();
  }

  gpu::Device::submit();

  if (dt > 0.0f) s->frame++;
  s->pending_dt = 0.0;
  s->needs_reset = false;
}

} // namespace bar_scatter

namespace bar_scatter {

// Effect store preview: scatters co-triggering bars, so its input is the
// drum-bars groove.
void preview(nano::PreviewScenario& s) {
  s.input("black")
      .param("strength", 1.0f)
      .pre("src", "source.light.drum_bars")
      .auxParam("src", "auto", 1.0f)
      .capture(2.0f);
}

}  // namespace bar_scatter
