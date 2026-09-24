/*
 * filter.light.flicker_grid — per-column luma→flicker-rate LED grid.
 *
 * The input is reduced to a grid (default 4 columns × 10 rows) of flat,
 * box-averaged cells. Each COLUMN's luma — Peak (max, default) or Average
 * across its cells — sets that column's flicker rate:
 *
 *   * brighter → faster pulses, CAPPED at on/off alternating every frame
 *     (0.5 cycles/frame). With `fill` on, demanded rate beyond the cap pours
 *     into the off frames (fill = overflow / cap), ramping continuously from
 *     50%-duty flicker to solid at 2× the cap.
 *   * dimmer → pulses stay exactly 1 frame while the gaps between them grow
 *     (the duty cycle shrinks) — implemented as a phase accumulator that
 *     wraps at 1 with a per-frame increment ≤ 0.5, so two consecutive
 *     on-frames are impossible.
 *   * below `min_thr` the column is plain black; at/above `max_thr` it's
 *     solid on.
 *
 * Color shaping (both per cell, in HSL): `neutral_pull` drags lightness
 * toward 0.5 so the temporal pattern — not the pixel luma — carries the
 * brightness (the flicker would otherwise double-apply it), and
 * `level_strength` lifts cells toward their column's max lightness on a
 * curve that fades out near black (dim colour gets lifted, black stays
 * black). Built for driving LEDs: works around their weakness at low
 * brightness and adds visible temporal contrast.
 *
 * Three passes, flicker state GPU-resident (no readback):
 *   reduce — one thread per cell: box-sample input → per-cell stats buffer
 *   sim    — single thread, serial columns: luma reduce + the pulse
 *            accumulator in a persistent ColState buffer
 *   render — full-res: flat cell color, HSL shaping, gate multiply
 */

#include <gpu.h>
#include <host.h>
#include "flicker_grid_shaders.h"

#include <cstdint>

namespace flicker_grid {

static constexpr int FG_MAX_COLS = 16;
static constexpr int FG_MAX_ROWS = 32;

// --- GPU-shared layouts (must match common.hlsl) ---

struct CellStat {
  float avg_r, avg_g, avg_b, avg_luma;
  float max_luma, pad0, pad1, pad2;
};
static_assert(sizeof(CellStat) == 32, "CellStat layout mismatch");

struct ColState {
  float acc, gate, fill, level_target;
};
static_assert(sizeof(ColState) == 16, "ColState layout mismatch");

struct ReduceUniforms {
  int32_t cols, rows, tex_w, tex_h;
};
static_assert(sizeof(ReduceUniforms) == 16, "ReduceUniforms layout mismatch");

struct SimUniforms {
  int32_t cols, rows, mode, do_reset;   // mode: 0 = peak, 1 = average
  float   dt, rate_max, min_thr, max_thr;
  int32_t fill_enable; float neutral_pull, pad0, pad1;
};
static_assert(sizeof(SimUniforms) == 48, "SimUniforms layout mismatch");

struct RenderUniforms {
  int32_t cols, rows;
  float   neutral_pull, level_strength;
};
static_assert(sizeof(RenderUniforms) == 16, "RenderUniforms layout mismatch");

enum LumaMode { MODE_PEAK = 0, MODE_AVERAGE = 1 };

// Per-instance state. One per chain entry.
struct State {
  // --- Per-instance GPU resources ---
  gpu::Buffer stats_buf;      // transient per-cell stats (rewritten each frame)
  gpu::Buffer colstate_buf;   // PERSISTENT per-column flicker state
  gpu::Buffer reduce_uniform_buf;
  gpu::Buffer sim_uniform_buf;
  gpu::Buffer render_uniform_buf;
  bool initialized = false;

  // --- Schema-mirrored params ---
  int   columns        = 4;
  int   rows           = 10;
  int   mode           = MODE_PEAK;
  float rate_max       = 60.0f;
  bool  fill           = false;
  float min_thr        = 0.05f;
  float max_thr        = 0.9f;
  float neutral_pull   = 0.5f;
  float level_strength = 0.5f;

  // --- Runtime state ---
  double pending_dt  = 0.0;   // accumulated tick dt, consumed by render()
  bool   needs_reset = true;  // zero the GPU colstate on next dispatch
};

// Type-shared: compiled once in module_init().
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
  state::init("filter.light.flicker_grid", {1, 0, 0},
    state::Schema()
      .helpField("intro",
        "## Flicker Grid\n"
        "Reduces the input to a coarse grid of flat cells and turns each "
        "**column's brightness into a flicker rate** — bright columns pulse "
        "fast (capped at on/off every frame), dim ones keep short 1-frame "
        "pulses with growing gaps. Built for driving LEDs: the temporal "
        "pattern carries the brightness, sidestepping LED weakness at low "
        "levels and adding punchy temporal contrast.\n\n"
        "**Try:** turn on **Overflow Fill** and push **Max Rate** past your "
        "frame rate so hot columns glide from strobe into solid; raise "
        "**Neutral Pull** so the cell colours stay vivid while the flicker "
        "does the dimming; use **Column Level** to pull a column's dim cells "
        "up to its brightest without lifting black.")
      // --- Grid ---
      .group("grid", "Grid")
        .groupHelp(
          "The sampling grid. The input is box-averaged into *Columns* × "
          "*Rows* flat cells — one cell per LED. Each column flickers as one "
          "unit; cells keep their own colour.")
      .intField  ("columns",        4, 1, FG_MAX_COLS,  state::PrimaryInput).label("Columns", "Cols")
      .intField  ("rows",           10, 1, FG_MAX_ROWS, state::PrimaryInput).label("Rows", "Rows")
      // --- Flicker ---
      .group("flicker", "Flicker")
        .groupHelp(
          "The engine. Each column's luma — **Peak** (its brightest cell) or "
          "**Average** — maps between the thresholds to a pulse rate, up to "
          "*Max Rate* (Hz) at the *High Threshold*. Pulses are 1 frame; the "
          "rate is capped at on/off every frame, and above the cap **Overflow "
          "Fill** pours the excess into the off frames (2× the cap reads "
          "solid). Below the *Low Threshold* a column is plain black; at/above "
          "the *High Threshold* it holds solid on. Rates past ~half your frame "
          "rate are what engage Fill.")
      .selectField("mode",          MODE_PEAK,          state::PrimaryInput,
                   {{"peak", MODE_PEAK}, {"average", MODE_AVERAGE}}, /*wrap=*/true).label("Luma Mode", "Mode")
      .floatField("rate_max",       60.0f, 0.0f, 240.0f, state::PrimaryInput).label("Max Rate", "Rate")
      .boolField ("fill",           false,               state::PrimaryInput).label("Overflow Fill", "Fill")
      .floatField("min_thr",        0.05f, 0.0f, 1.0f,   state::PrimaryInput).label("Low Threshold", "LoThr")
      .floatField("max_thr",        0.9f,  0.0f, 1.0f,   state::PrimaryInput).label("High Threshold", "HiThr")
      // --- Color shaping ---
      .group("shape", "Color Shaping")
        .groupHelp(
          "Per-cell colour, in HSL. **Neutral Pull** drags lightness toward "
          "0.5 so the flicker — not the pixel luma — carries the brightness "
          "(otherwise dim content gets dimmed twice). **Column Level** lifts "
          "cells toward their column's brightest on a curve that fades out "
          "near black, so dim colour brightens but black stays black.")
      .floatField("neutral_pull",   0.5f, 0.0f, 1.0f,    state::PrimaryInput).label("Neutral Pull", "Neut")
      .floatField("level_strength", 0.5f, 0.0f, 1.0f,    state::PrimaryInput).label("Column Level", "Level")
      .capability(state::Capability::SeekableApproximate)
      .textureField("tex_in",  state::PrimaryInput)
      .textureField("tex_out", state::PrimaryOutput)
  );

  if (gpu::Device::backend() == gpu::Backend::None) return;

  state::registerShaderSPV("flicker_grid_reduce", REDUCE_SPV, REDUCE_SPV_SIZE);
  state::registerShaderSPV("flicker_grid_sim",    SIM_SPV,    SIM_SPV_SIZE);
  state::registerShaderSPV("flicker_grid_render", RENDER_SPV, RENDER_SPV_SIZE);
  auto cs_reduce = gpu::Device::createShaderModuleByName("flicker_grid_reduce");
  auto cs_sim    = gpu::Device::createShaderModuleByName("flicker_grid_sim");
  auto cs_render = gpu::Device::createShaderModuleByName("flicker_grid_render");
  if (!cs_reduce || !cs_sim || !cs_render) return;

  s_pso_reduce = gpu::Device::createComputePSO(cs_reduce, "main", gpu::Bindings()
      .tex2d(0)
      .storageRW(1)
      .uniform(2));
  s_pso_sim = gpu::Device::createComputePSO(cs_sim, "main", gpu::Bindings()
      .storage(0)
      .storageRW(1)
      .uniform(2));
  s_pso_render = gpu::Device::createComputePSO(cs_render, "main", gpu::Bindings()
      .tex2d(0)
      .storageTex2d(1)
      .uniform(2)
      .storage(3)
      .storage(4));

  state::log("flicker_grid: module initialized");
}

void* create() {
  auto* s = new State();
  s->stats_buf          = gpu::Device::createBuffer(sizeof(CellStat) * FG_MAX_COLS * FG_MAX_ROWS,
                                                    gpu::BufferUsage::Storage);
  s->colstate_buf       = gpu::Device::createBuffer(sizeof(ColState) * FG_MAX_COLS,
                                                    gpu::BufferUsage::Storage);
  s->reduce_uniform_buf = gpu::Device::createBuffer(sizeof(ReduceUniforms), gpu::BufferUsage::Uniform);
  s->sim_uniform_buf    = gpu::Device::createBuffer(sizeof(SimUniforms),    gpu::BufferUsage::Uniform);
  s->render_uniform_buf = gpu::Device::createBuffer(sizeof(RenderUniforms), gpu::BufferUsage::Uniform);
  return s;
}

void destroy(void* self) {
  auto* s = static_cast<State*>(self);
  if (!s) return;
  s->stats_buf.release();
  s->colstate_buf.release();
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
  if (!s->stats_buf.valid() || !s->colstate_buf.valid()
      || !s->reduce_uniform_buf.valid() || !s->sim_uniform_buf.valid()
      || !s->render_uniform_buf.valid()) return;

  s->initialized = true;
  state::log("flicker_grid: initialized");
}

void tick(void* self, double dt) {
  auto* s = static_cast<State*>(self);
  if (!s || !s->initialized) return;
  // Accumulated here, consumed in render() — the runner may tick several
  // times per render. Clamp so a stall can't slam the accumulator (the
  // 0.5-cycle cap bounds the visual effect anyway).
  s->pending_dt += dt;
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
    if      (state::pathIs(path, plen, "columns"))        s->columns        = (int)state::patchFloat(i);
    else if (state::pathIs(path, plen, "rows"))           s->rows           = (int)state::patchFloat(i);
    else if (state::pathIs(path, plen, "mode"))           s->mode           = (int)state::patchFloat(i);
    else if (state::pathIs(path, plen, "rate_max"))       s->rate_max       = state::patchFloat(i);
    else if (state::pathIs(path, plen, "fill"))           s->fill           = state::patchFloat(i) != 0.0f;
    else if (state::pathIs(path, plen, "min_thr"))        s->min_thr        = state::patchFloat(i);
    else if (state::pathIs(path, plen, "max_thr"))        s->max_thr        = state::patchFloat(i);
    else if (state::pathIs(path, plen, "neutral_pull"))   s->neutral_pull   = state::patchFloat(i);
    else if (state::pathIs(path, plen, "level_strength")) s->level_strength = state::patchFloat(i);
  }
}

void render(void* self, int vp_w, int vp_h) {
  auto* s = static_cast<State*>(self);
  if (!s || !s->initialized || vp_w <= 0 || vp_h <= 0) return;
  auto in  = gpu::Device::textureForField("tex_in");
  auto out = gpu::Device::textureForField("tex_out");
  if (!in.valid() || !out.valid()) return;

  int cols = clampi(s->columns, 1, FG_MAX_COLS);
  int rows = clampi(s->rows,    1, FG_MAX_ROWS);

  ReduceUniforms ru = {};
  ru.cols = cols; ru.rows = rows;
  ru.tex_w = vp_w; ru.tex_h = vp_h;
  s->reduce_uniform_buf.writeOne(ru);

  SimUniforms su = {};
  su.cols         = cols;
  su.rows         = rows;
  su.mode         = (s->mode == MODE_AVERAGE) ? MODE_AVERAGE : MODE_PEAK;
  su.do_reset     = s->needs_reset ? 1 : 0;
  su.dt           = (float)s->pending_dt;
  su.rate_max     = clampf(s->rate_max, 0.0f, 240.0f);
  su.min_thr      = clampf(s->min_thr, 0.0f, 1.0f);
  su.max_thr      = clampf(s->max_thr, 0.0f, 1.0f);
  su.fill_enable  = s->fill ? 1 : 0;
  su.neutral_pull = clampf(s->neutral_pull, 0.0f, 1.0f);
  s->sim_uniform_buf.writeOne(su);

  RenderUniforms cu = {};
  cu.cols           = cols;
  cu.rows           = rows;
  cu.neutral_pull   = clampf(s->neutral_pull, 0.0f, 1.0f);
  cu.level_strength = clampf(s->level_strength, 0.0f, 1.0f);
  s->render_uniform_buf.writeOne(cu);

  // Pass 1 — reduce: one thread per cell → per-cell stats.
  {
    auto cp = gpu::ComputePass::begin();
    cp.setPSO(s_pso_reduce);
    cp.setTexture(in, 0, 0);
    cp.setBuffer(s->stats_buf,          1);
    cp.setBuffer(s->reduce_uniform_buf, 2);
    cp.dispatch((cols + 7) / 8, (rows + 7) / 8);
    cp.end();
  }

  // Pass 2 — sim (single-thread): per-column luma → flicker state.
  {
    auto cp = gpu::ComputePass::begin();
    cp.setPSO(s_pso_sim);
    cp.setBuffer(s->stats_buf,       0);
    cp.setBuffer(s->colstate_buf,    1);
    cp.setBuffer(s->sim_uniform_buf, 2);
    cp.dispatch(1, 1, 1);
    cp.end();
  }

  // Pass 3 — render: flat cells, HSL shaping, gate multiply.
  {
    auto cp = gpu::ComputePass::begin();
    cp.setPSO(s_pso_render);
    cp.setTexture(in,  0, 0);
    cp.setTexture(out, 1, 1);
    cp.setBuffer(s->render_uniform_buf, 2);
    cp.setBuffer(s->stats_buf,          3);
    cp.setBuffer(s->colstate_buf,       4);
    cp.dispatch((vp_w + 7) / 8, (vp_h + 7) / 8);
    cp.end();
  }

  gpu::Device::submit();

  // Consume per-frame inputs so a re-render without a tick holds the frame
  // (the sim's dt<=0 branch).
  s->pending_dt = 0.0;
  s->needs_reset = false;
}

} // namespace flicker_grid
