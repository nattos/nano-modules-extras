/*
 * warp.recompose — rule-of-thirds compositional rebalancer.
 *
 * Measures where the frame's visual weight actually sits (a saliency field
 * blending edge detail, deviation from the frame's mean brightness, and colour
 * saturation), compares that against the rule of thirds, then slices the frame
 * into the nine thirds cells and translates each one to push the composition
 * toward balance. The measured imbalance is also published as three scalar
 * modulation outputs.
 *
 * Four GPU passes, all on-GPU for the warp itself (zero latency):
 *   accumulate — coarse-grid global normalizers (mean luma, sd, mean gradient,
 *                mean chroma). Host-gated to the update rate.
 *   weigh      — re-samples the grid with the normalizers known; scatters the
 *                saliency centroid numerators + the nine cell masses. Same gate.
 *   solve      — one thread; smooths the centroid/masses and re-derives the
 *                nine correction vectors. Runs EVERY frame so param modulation
 *                stays instant while the analysis still eases.
 *   render     — per-pixel inverse map against the nine translated cells.
 *
 * The only CPU round trip is a 16-byte readback of the three published scalars
 * (~1-2 frames late on web, ~0 native) — the warp never waits on it.
 *
 * Deliberately NOT declaring is_identity at correct == 0: the effect doubles as
 * a measurement device, and letting the executor skip it would silently freeze
 * the modulation outputs for anyone using it purely as an analyzer.
 *
 * Per-instance ABI: mutable state in `State`; the four PSOs are file-static,
 * compiled once in module_init().
 */

#include <gpu.h>
#include <host.h>
#include <effect_utils.h>
#include "recompose_shaders.h"

#include <cstdint>
#include <cmath>

namespace recompose {

// Keep in sync with common.hlsl.
static constexpr int GRID_SN      = 128;
static constexpr int STATS_INTS   = 20;
static constexpr int SOLVE_FLOATS = 40;

enum Axis    : int { AXIS_BOTH = 0, AXIS_X = 1, AXIS_Y = 2 };
enum Overlap : int { OM_HEAVIEST = 0, OM_BLEND = 1, OM_ADDITIVE = 2 };
enum Debug   : int { DBG_OFF = 0, DBG_GRID = 1, DBG_MASS = 2 };

struct AccumU  { float res_x, res_y, aspect_x, aspect_y; };
struct WeighU  { float res_x, res_y, aspect_x, aspect_y,
                       w_grad, w_dev, w_sat, _p0; };
struct SolveU  { float aspect_x, aspect_y, correct, overshoot,
                       spread, distance, axis, center_bias,
                       alpha, _p0, _p1, _p2; };
struct RenderU { float aspect_x, aspect_y, rift_fill, edge_fill,
                       overlap_mode, debug_show, center_bias, _p0; };
static_assert(sizeof(AccumU)  ==  4 * 4, "AccumU layout");
static_assert(sizeof(WeighU)  ==  8 * 4, "WeighU layout");
static_assert(sizeof(SolveU)  == 12 * 4, "SolveU layout");
static_assert(sizeof(RenderU) ==  8 * 4, "RenderU layout");

struct State {
  // Per-instance GPU resources.
  gpu::Buffer  accum_uniform;
  gpu::Buffer  weigh_uniform;
  gpu::Buffer  solve_uniform;
  gpu::Buffer  render_uniform;
  gpu::Buffer  stats_buf;     // int[STATS_INTS]
  gpu::Buffer  solve_buf;     // float[SOLVE_FLOATS], persists → latched analysis
  gpu::Sampler sampler;
  bool initialized = false;

  // --- Schema-mirrored params ---
  float correct     = 0.6f;
  float spread      = 0.5f;
  int   axis        = AXIS_BOTH;
  float w_grad      = 1.0f;
  float w_dev       = 0.5f;
  float w_sat       = 0.3f;
  float center_bias = 0.5f;
  float update_rate = 0.35f;
  float smooth      = 0.4f;
  bool  trigger_prev = false;
  int   rift_fill   = 1;       // original
  int   edge_fill   = 2;       // edge stretch
  int   overlap_mode = OM_HEAVIEST;
  float distance    = 0.5f;
  float overshoot   = 1.0f;
  int   debug_show  = DBG_OFF;

  // --- Runtime (CPU accumulators, style guide §2.1) ---
  double solve_timer = 0.0;
  float  alpha       = 1.0f;   // temporal EMA coefficient for the solve pass
  bool   do_update   = true;   // force analysis on the first frame

  // --- Published scalars (via readback) ---
  float bal_x = 0.0f, bal_y = 0.0f, cell_err = 0.0f;
};

static gpu::ComputePSO s_pso_accum;
static gpu::ComputePSO s_pso_weigh;
static gpu::ComputePSO s_pso_solve;
static gpu::ComputePSO s_pso_render;

void module_init() {
  state::init("warp.recompose", {1, 0, 0},
    state::Schema()
      .helpField("intro",
        "## Recompose\n"
        "Measures where the image's **visual weight** actually sits, compares it to the "
        "**rule of thirds**, then slices the frame into the 3×3 thirds grid and slides each "
        "cell to push the composition toward balance. The imbalance is also published as "
        "modulation outputs (*Balance X/Y*, *Cell Error*) you can wire anywhere.\n\n"
        "**Try:** leave *Correction* around 0.6 for a natural re-framing; push *Spread* up to "
        "break the frame into a collage that still lands on the thirds; drive *Correction* "
        "**negative** to deliberately un-balance a too-tidy shot. Set *Update Rate* to 0 and "
        "drive **Trigger** on the beat so the slice snaps to the music.\n\n"
        "*Correction 0 is an exact passthrough*, so a modulation wire at rest does nothing — "
        "but the analysis keeps running, so the outputs stay live.")

      .group("standard", "Recompose")
        .groupHelp(
          "*Correction* is the whole effect: 0 leaves the frame alone, 1 lands the centre of "
          "visual mass on the nearest rule-of-thirds intersection, and negative values push it "
          "away. *Spread* redistributes the individual cells around that correction — the "
          "overall balance is guaranteed either way, so Spread changes the character without "
          "changing where the weight ends up. *Axis* restricts the motion to one direction.")
      .floatField("correct", 0.6f, -1.0f, 1.0f, state::PrimaryInput, "signed", 0.01f, nullptr,
                  "How hard to correct. 0 = untouched, 1 = fully balanced, negative = deliberately un-balanced.")
        .label("Correction", "Corr")
      .floatField("spread", 0.5f, 0.0f, 1.0f, state::PrimaryInput, nullptr, 0.01f, nullptr,
                  "How much each cell moves on its own toward the thirds, on top of the shared correction.")
        .label("Spread", "Sprd")
      .selectField("axis", AXIS_BOTH, state::PrimaryInput, {
        {"Both", AXIS_BOTH}, {"X Only", AXIS_X}, {"Y Only", AXIS_Y},
      }, false, "Restrict the correction to one axis.").label("Axis", "Axis")

      .group("saliency", "Visual Weight")
        .groupHelp(
          "What counts as \"visually heavy\". *Detail* follows edges and local contrast, "
          "*Contrast* follows how far a region sits from the frame's average brightness, and "
          "*Colour* follows saturation. Only the RATIO between the three matters — scaling all "
          "three together changes nothing. *Centre Bias* says how empty the middle cell should "
          "ideally be: 0 treats all nine cells equally, 1 wants everything in the corners.")
      .floatField("w_grad", 1.0f, 0.0f, 1.0f, state::PrimaryInput, nullptr, 0.01f, nullptr,
                  "Weight on edges and local detail.").label("Detail", "Detl")
      .floatField("w_dev", 0.5f, 0.0f, 1.0f, state::PrimaryInput, nullptr, 0.01f, nullptr,
                  "Weight on deviation from the frame's average brightness.").label("Contrast", "Cntr")
      .floatField("w_sat", 0.3f, 0.0f, 1.0f, state::PrimaryInput, nullptr, 0.01f, nullptr,
                  "Weight on colour saturation.").label("Colour", "Col")
      .floatField("center_bias", 0.5f, 0.0f, 1.0f, state::PrimaryInput, nullptr, 0.01f, nullptr,
                  "How strongly the ideal layout empties the centre cell. 0 = even, 1 = corners only.")
        .label("Centre Bias", "CBias")

      .group("timing", "Timing")
        .groupHelp(
          "*Update Rate* is how often the image is re-analysed — 0 freezes on the current "
          "solve and waits for **Trigger**. *Smoothing* eases the solved offsets so a retarget "
          "glides instead of snapping. **Trigger** forces a re-analysis now.")
      .floatField("update_rate", 0.35f, 0.0f, 1.0f, state::PrimaryInput, nullptr, 0.01f, nullptr,
                  "How often the composition is re-analysed (0 = frozen, manual Trigger only; higher = faster).")
        .label("Update Rate", "Rate")
      .floatField("smooth", 0.4f, 0.0f, 1.0f, state::PrimaryInput, nullptr, 0.01f, nullptr,
                  "Temporal easing of the solved offsets. 0 = snap instantly.").label("Smoothing", "Smth")
      .eventField("trigger", state::PrimaryInput).label("Trigger", "Trig")

      .group("fills", "Fills")
        .groupHelp(
          "What fills the exposed regions. *Rift Fill* is the gap between cells that separated; "
          "*Edge Fill* is the viewport border a cell sliding inward reveals; *Overlap* is how "
          "cells combine where they cover each other.")
      .selectField("rift_fill", 1, state::SecondaryInput, {
        {"Transparent", 0}, {"Original", 1}, {"Edge Stretch", 2}, {"Mirror", 3}, {"Black", 4},
      }, false, "Fills the gaps between cells that pulled apart.").label("Rift Fill", "Rift")
      .selectField("edge_fill", 2, state::SecondaryInput, {
        {"Transparent", 0}, {"Original", 1}, {"Edge Stretch", 2}, {"Mirror", 3}, {"Black", 4},
      }, false, "Fills the viewport border a cell sliding inward reveals.").label("Edge Fill", "Edge")
      .selectField("overlap_mode", OM_HEAVIEST, state::SecondaryInput, {
        {"Heaviest On Top", OM_HEAVIEST}, {"Blend", OM_BLEND}, {"Additive", OM_ADDITIVE},
      }, false, "How cells combine where they overlap.").label("Overlap", "Over")

      .group("tuning", "Tuning")
        .groupHelp(
          "*Max Shift* caps how far any one cell may travel — the whole correction scales down "
          "uniformly to respect it, so the balance still lands, just nearer. *Overshoot* "
          "multiplies the correction past 1 for a deliberately over-corrected look; the pair "
          "can never exceed a perfect mirror, so the result cannot run away.")
      .floatField("distance", 0.5f, 0.0f, 1.0f, state::SecondaryInput, nullptr, 0.01f, nullptr,
                  "Cap on any one cell's travel, as a fraction of the frame's short half-axis.")
        .label("Max Shift", "Shift")
      .floatField("overshoot", 1.0f, 1.0f, 2.0f, state::SecondaryInput, nullptr, 0.01f, nullptr,
                  "Multiplies the correction. 1 = exact balance, 2 = twice past it.")
        .label("Overshoot", "Over+")

      .group("debug", "Debug")
        .groupHelp("Inspection aids. *Show* overlays the thirds grid with the measured centre "
                   "of mass and its target, or a per-cell surplus/deficit heat map.")
      .selectField("debug_show", DBG_OFF, state::SecondaryInput, {
        {"Off", DBG_OFF}, {"Grid + Centroid", DBG_GRID}, {"Cell Mass", DBG_MASS},
      }, false, "Overlay the analysis.").label("Show", "Show")
      .endGroup()

      // ---- I/O ----
      .textureField("tex_in",  state::PrimaryInput)
      .textureField("tex_out", state::PrimaryOutput)
      // Published imbalance — modulation channels. min/max IS the modulation
      // contract, so these are declared on the ranges they actually span.
      .floatField("balance_x", 0.0f, -1.0f, 1.0f, state::PrimaryOutput, "signed", 0.0f, nullptr,
                  "Horizontal correction the frame needs, signed. 0 = balanced.")
        .label("Balance X", "BalX")
      .floatField("balance_y", 0.0f, -1.0f, 1.0f, state::PrimaryOutput, "signed", 0.0f, nullptr,
                  "Vertical correction the frame needs, signed. 0 = balanced.")
        .label("Balance Y", "BalY")
      .floatField("cell_error", 0.0f, 0.0f, 1.0f, state::PrimaryOutput, "unsigned", 0.0f, nullptr,
                  "How far the 3x3 mass layout is from the ideal. 0 = ideal, 1 = maximally lopsided.")
        .label("Cell Error", "CErr")

      .capability(state::Capability::SeekableApproximate)
      .capability(state::Capability::ModulationSource)
      .capability(state::Capability::ModulationSourceMulti)
  );

  if (gpu::Device::backend() == gpu::Backend::None) return;

  state::registerShaderSPV("recompose_accumulate", ACCUMULATE_SPV, ACCUMULATE_SPV_SIZE);
  state::registerShaderSPV("recompose_weigh",      WEIGH_SPV,      WEIGH_SPV_SIZE);
  state::registerShaderSPV("recompose_solve",      SOLVE_SPV,      SOLVE_SPV_SIZE);
  state::registerShaderSPV("recompose_render",     RENDER_SPV,     RENDER_SPV_SIZE);

  auto cs_accum  = gpu::Device::createShaderModuleByName("recompose_accumulate");
  auto cs_weigh  = gpu::Device::createShaderModuleByName("recompose_weigh");
  auto cs_solve  = gpu::Device::createShaderModuleByName("recompose_solve");
  auto cs_render = gpu::Device::createShaderModuleByName("recompose_render");
  if (!cs_accum || !cs_weigh || !cs_solve || !cs_render) return;

  s_pso_accum  = gpu::Device::createComputePSO(cs_accum, "main", gpu::Bindings()
      .tex2d(0).sampler(1).storageRW(2).uniform(3));
  s_pso_weigh  = gpu::Device::createComputePSO(cs_weigh, "main", gpu::Bindings()
      .tex2d(0).sampler(1).storageRW(2).uniform(3));
  s_pso_solve  = gpu::Device::createComputePSO(cs_solve, "main", gpu::Bindings()
      .storage(0).storageRW(1).uniform(2));
  s_pso_render = gpu::Device::createComputePSO(cs_render, "main", gpu::Bindings()
      .tex2d(0).sampler(1).storage(2).storageTex2d(3).uniform(4));

  state::log("recompose: module initialized");
}

void* create() {
  auto* s = new State();
  s->accum_uniform  = gpu::Device::createBuffer(sizeof(AccumU),  gpu::BufferUsage::Uniform);
  s->weigh_uniform  = gpu::Device::createBuffer(sizeof(WeighU),  gpu::BufferUsage::Uniform);
  s->solve_uniform  = gpu::Device::createBuffer(sizeof(SolveU),  gpu::BufferUsage::Uniform);
  s->render_uniform = gpu::Device::createBuffer(sizeof(RenderU), gpu::BufferUsage::Uniform);
  s->stats_buf      = gpu::Device::createBuffer(STATS_INTS   * sizeof(int32_t), gpu::BufferUsage::Storage);
  s->solve_buf      = gpu::Device::createBuffer(SOLVE_FLOATS * sizeof(float),   gpu::BufferUsage::Storage);
  s->sampler        = gpu::Device::createSampler(gpu::FilterMode::Linear, gpu::AddressMode::ClampToEdge);
  return s;
}

void destroy(void* self) {
  auto* s = static_cast<State*>(self);
  if (!s) return;
  s->accum_uniform.release();
  s->weigh_uniform.release();
  s->solve_uniform.release();
  s->render_uniform.release();
  s->stats_buf.release();
  s->solve_buf.release();
  s->sampler.release();
  delete s;
}

void init(void* self) {
  auto* s = static_cast<State*>(self);
  if (!s) return;
  if (!s_pso_accum.valid() || !s_pso_weigh.valid() ||
      !s_pso_solve.valid() || !s_pso_render.valid()) return;
  if (!s->solve_buf.valid()) return;

  // Seed a neutral analysis: centroid at the origin, uniform cell masses, all
  // correction vectors zero, and INIT/VALID clear. A render before the first
  // solve is therefore an exact passthrough with zero published imbalance.
  float seed[SOLVE_FLOATS] = {0};
  for (int k = 0; k < 9; ++k) seed[12 + k] = 1.0f / 9.0f;   // RC_S_M
  s->solve_buf.write(seed, SOLVE_FLOATS);

  s->solve_timer  = 0.0;
  s->alpha        = 1.0f;
  s->do_update    = true;
  s->trigger_prev = false;
  s->bal_x = s->bal_y = s->cell_err = 0.0f;
  s->initialized  = true;
}

static float updateInterval(float slider) {
  // Slider 0 → 8 s (slow), 1 → 0.05 s (fast); exp-mapped (style guide §1.3).
  return 8.0f * std::pow(0.05f / 8.0f, slider);
}
static float smoothTau(float slider) {
  // 0 → snap; quadratic feel up to ~2 s.
  return slider * slider * 2.0f;
}

void tick(void* self, double dt) {
  auto* s = static_cast<State*>(self);
  if (!s || !s->initialized) return;
  if (dt < 0.0) dt = 0.0;
  if (dt > 0.25) dt = 0.25;   // clamp starved frames

  // --- Analysis metronome. update_rate 0 = never auto-update (manual trigger). ---
  s->solve_timer += dt;
  float interval = updateInterval(s->update_rate);
  if (!s->do_update && s->update_rate > 1e-4f && s->solve_timer >= interval) {
    s->do_update   = true;
    s->solve_timer = 0.0;
  }

  // --- Temporal easing coefficient consumed by the solve pass. ---
  float tau = smoothTau(s->smooth);
  s->alpha = (tau <= 1e-4f) ? 1.0f
                            : (1.0f - std::exp(-(float)dt / tau));

  // --- Drain the published-scalar readback requested last frame. Polling here
  //     (not in render) waits only on ALREADY-submitted work, so it is not a
  //     pipeline stall. 0 bytes = not ready; keep the previous values. ---
  float raw[4] = {0.0f, 0.0f, 0.0f, 0.0f};
  if (s->solve_buf.valid() &&
      s->solve_buf.pollReadback(raw, sizeof(raw)) == (int)sizeof(raw) &&
      raw[3] > 0.5f) {                       // RC_S_VALID
    s->bal_x    = raw[0];
    s->bal_y    = raw[1];
    s->cell_err = raw[2];
  }

  // Publish every tick — before the first readback these are 0, which is the
  // neutral value for both the signed and the unsigned channels.
  { auto v = val::number(s->bal_x);    state::setValPath("balance_x",  v); val::release(v); }
  { auto v = val::number(s->bal_y);    state::setValPath("balance_y",  v); val::release(v); }
  { auto v = val::number(s->cell_err); state::setValPath("cell_error", v); val::release(v); }
}

void on_state_patched(void* self, int n, const char* pb, const int* off,
                      const int* len, const int* ops) {
  auto* s = static_cast<State*>(self);
  if (!s) return;
  for (int i = 0; i < n; i++) {
    if (ops[i] != state::PatchReplace) continue;
    const char* p = pb + off[i]; int l = len[i];
    if      (state::pathIs(p, l, "correct"))      s->correct = state::patchFloat(i);
    else if (state::pathIs(p, l, "spread"))       s->spread = state::patchFloat(i);
    else if (state::pathIs(p, l, "axis"))         s->axis = state::patchInt(i);
    // The saliency weights and the ideal template change what is MEASURED, so
    // they force a re-analysis rather than waiting for the metronome.
    else if (state::pathIs(p, l, "w_grad"))       { s->w_grad = state::patchFloat(i); s->do_update = true; }
    else if (state::pathIs(p, l, "w_dev"))        { s->w_dev = state::patchFloat(i); s->do_update = true; }
    else if (state::pathIs(p, l, "w_sat"))        { s->w_sat = state::patchFloat(i); s->do_update = true; }
    else if (state::pathIs(p, l, "center_bias"))  s->center_bias = state::patchFloat(i);
    else if (state::pathIs(p, l, "update_rate"))  s->update_rate = state::patchFloat(i);
    else if (state::pathIs(p, l, "smooth"))       s->smooth = state::patchFloat(i);
    else if (state::pathIs(p, l, "trigger")) {
      // Rising-edge (0→1) forces a re-analysis. Replay-safe: constant replays
      // never re-fire (§8.2).
      bool tval = state::patchFloat(i) != 0.0f;
      if (tval && !s->trigger_prev) s->do_update = true;
      s->trigger_prev = tval;
    }
    else if (state::pathIs(p, l, "rift_fill"))    s->rift_fill = state::patchInt(i);
    else if (state::pathIs(p, l, "edge_fill"))    s->edge_fill = state::patchInt(i);
    else if (state::pathIs(p, l, "overlap_mode")) s->overlap_mode = state::patchInt(i);
    else if (state::pathIs(p, l, "distance"))     s->distance = state::patchFloat(i);
    else if (state::pathIs(p, l, "overshoot"))    s->overshoot = state::patchFloat(i);
    else if (state::pathIs(p, l, "debug_show"))   s->debug_show = state::patchInt(i);
  }
}

void render(void* self, int vp_w, int vp_h) {
  auto* s = static_cast<State*>(self);
  if (!s || !s->initialized || vp_w <= 0 || vp_h <= 0) return;

  auto in  = gpu::Device::textureForField("tex_in");
  auto out = gpu::Device::textureForField("tex_out");
  if (!in.valid() || !out.valid()) return;

  auto cover = fx::coverSquare(vp_w, vp_h);
  int  sg    = (GRID_SN + 7) / 8;

  // --- Analysis (only on update frames) — the measurement is stiff. ---
  if (s->do_update) {
    int32_t zeros[STATS_INTS] = {0};
    s->stats_buf.write(zeros, STATS_INTS);

    AccumU au = {};
    au.res_x = (float)vp_w; au.res_y = (float)vp_h;
    au.aspect_x = cover.ax; au.aspect_y = cover.ay;
    s->accum_uniform.writeOne(au);

    WeighU wu = {};
    wu.res_x = (float)vp_w; wu.res_y = (float)vp_h;
    wu.aspect_x = cover.ax; wu.aspect_y = cover.ay;
    wu.w_grad = s->w_grad; wu.w_dev = s->w_dev; wu.w_sat = s->w_sat;
    s->weigh_uniform.writeOne(wu);

    {
      auto cp = gpu::ComputePass::begin();
      cp.setPSO(s_pso_accum);
      cp.setTexture(in, 0, 0);
      cp.setSampler(s->sampler, 1);
      cp.setBuffer(s->stats_buf, 2);
      cp.setBuffer(s->accum_uniform, 3);
      cp.dispatch(sg, sg);
      cp.end();
    }
    {
      // Reads the normalizers the accumulate pass just wrote (same submit).
      auto cp = gpu::ComputePass::begin();
      cp.setPSO(s_pso_weigh);
      cp.setTexture(in, 0, 0);
      cp.setSampler(s->sampler, 1);
      cp.setBuffer(s->stats_buf, 2);
      cp.setBuffer(s->weigh_uniform, 3);
      cp.dispatch(sg, sg);
      cp.end();
    }
    s->do_update = false;
  }

  // --- Solve every frame: the analysis eases, the params respond instantly. ---
  {
    SolveU su = {};
    su.aspect_x = cover.ax; su.aspect_y = cover.ay;
    su.correct = s->correct;
    su.overshoot = s->overshoot;
    su.spread = s->spread;
    su.distance = s->distance;
    su.axis = (float)s->axis;
    su.center_bias = s->center_bias;
    su.alpha = s->alpha;
    s->solve_uniform.writeOne(su);

    auto cp = gpu::ComputePass::begin();
    cp.setPSO(s_pso_solve);
    cp.setBuffer(s->stats_buf, 0);
    cp.setBuffer(s->solve_buf, 1);
    cp.setBuffer(s->solve_uniform, 2);
    cp.dispatch(1, 1, 1);
    cp.end();
  }

  // --- Cell translation warp. ---
  {
    RenderU ru = {};
    ru.aspect_x = cover.ax; ru.aspect_y = cover.ay;
    ru.rift_fill = (float)s->rift_fill;
    ru.edge_fill = (float)s->edge_fill;
    ru.overlap_mode = (float)s->overlap_mode;
    ru.debug_show = (float)s->debug_show;
    ru.center_bias = s->center_bias;
    s->render_uniform.writeOne(ru);

    auto cp = gpu::ComputePass::begin();
    cp.setPSO(s_pso_render);
    cp.setTexture(in, 0, 0);
    cp.setSampler(s->sampler, 1);
    cp.setBuffer(s->solve_buf, 2);
    cp.setTexture(out, 3, 1);
    cp.setBuffer(s->render_uniform, 4);
    cp.dispatch((vp_w + 7) / 8, (vp_h + 7) / 8);
    cp.end();
  }

  // Only the first 4 floats (the published scalars + the valid flag) — both
  // backends copy from offset 0, which is why those slots come first.
  s->solve_buf.requestReadback(4 * sizeof(float));

  gpu::Device::submit();
}

} // namespace recompose
