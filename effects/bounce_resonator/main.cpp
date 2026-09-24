/*
 * source.light.bounce_resonator — 4-bar scalar diffusion network (GPU-resident).
 *
 * Each bar holds a value + hue. A cycling diffusion matrix (v ← M·v)
 * exchanges value (and carries hue) between bars on each hop; the matrix
 * has no spatial structure (seeded permutations + random spread). The sim
 * runs in a single-threaded compute shader (sim.hlsl) with state living in
 * a persistent GPU buffer — the CPU only builds + uploads the matrices and
 * the per-frame uniforms, so the sampled-input mode needs no readback.
 *
 * Impulse source (impulse_mode enum):
 *   one_bar / random_bar / all_bars → impulse at band_color's hue, amount =
 *     impulse_strength, into the chosen bar(s).
 *   tex_in → on trigger, each bar samples its slice of tex_in: impulse amount
 *     = mean luminance (scaled by tex_in_boost), impulse colour = the slice's
 *     dominant hue/saturation blended toward band_color by tex_in_color.
 *
 * Trigger semantics (style guide §8.1): gate (bool) + trigger (event) fire
 * on a 0→1 rising edge; `auto_mode` is the shared self-fire block (Off by
 * default — Random is Poisson, Beats locks to the host transport). Impulses
 * are injected AFTER the diffusion hops so a fresh trigger is a solid flash.
 *
 * Outputs:
 *   tex_out                  — each bar fills its 1/4 column (hsv2rgb)
 *   render_outputs/motion    — rgba16f motion vectors (passthrough for now)
 */

#include <gpu.h>
#include <host.h>
#include <val.h>
#include <effect_utils.h>
#include <effect_diffusion_network.h>
#include <effect_auto_trigger.h>  // fx::AutoTrigger — the shared Off/Random/Beats self-fire
#include "bounce_resonator_shaders.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>

namespace bounce_resonator {

static constexpr int BARS       = 4;
static constexpr int SAMPLE_NX  = 24;     // per-bar tex_in sample grid (x)
static constexpr int SAMPLE_NY  = 6;      // per-bar tex_in sample grid (y)

// Where a trigger's impulse comes from / lands.
enum ImpulseMode { MODE_TEX_IN = 0, MODE_ONE_BAR = 1, MODE_RANDOM = 2, MODE_ALL = 3 };

// --- GPU-shared layouts (must match the HLSL structs) ---

struct SimState {                         // persistent GPU sim state
  float v[4];
  float h[4];
  float s[4];
  float env;
  float pad[3];
};
static_assert(sizeof(SimState) == 64, "SimState layout mismatch");

struct SimUniforms {
  float   feedback, decay_shaping, hue_converge, home_hue;
  int32_t pattern_count, hop_idx_start, n_hops, mode;
  float   pending0, pending1, pending2, pending3;
  float   band_hue, tex_in_boost; int32_t trigger_fired, do_reset;
  int32_t tex_w, tex_h, sample_nx, sample_ny;
  float   band_sat, tex_in_color, upad0, upad1;
};
static_assert(sizeof(SimUniforms) == 96, "SimUniforms layout mismatch");

struct ColorUniforms { float band_val, intensity, input_opacity, chroma_hold; };
static_assert(sizeof(ColorUniforms) == 16, "ColorUniforms layout mismatch");

struct MatsBuf { float data[fx::DiffusionNetwork4::kMaxN * 48]; };

// Per-instance state. One per chain entry.
struct State {
  // --- Per-instance GPU resources ---
  gpu::Buffer  sim_state_buf;
  gpu::Buffer  sim_uniform_buf;
  gpu::Buffer  mats_buf;
  gpu::Buffer  color_uniform_buf;
  gpu::Texture motion_tex;
  gpu::Texture zero_motion_tex;
  int          motion_w = 0;
  int          motion_h = 0;
  bool         initialized = false;

  // --- Schema-mirrored params ---
  bool  gate              = false;
  int   impulse_mode      = MODE_ONE_BAR;
  int   one_bar_target    = 0;     // only used when impulse_mode == one_bar
  float tex_in_boost      = 1.0f;  // only used when impulse_mode == tex_in
  float tex_in_color      = 1.0f;  // only used when impulse_mode == tex_in
  float feedback          = 0.90f;
  float spread            = 0.30f;
  float spread_contrast   = 0.0f;
  float decay_shaping     = 0.0f;
  float hue_spread        = 0.0f;
  float hue_converge      = 0.0f;
  int   seed              = 0;
  int   pattern_count     = 4;
  float cycle_rate        = 6.0f;
  float impulse_strength  = 1.0f;
  float color_r           = 1.0f;
  float color_g           = 0.92f;
  float color_b           = 0.78f;
  float band_hue          = 0.0f;
  float band_sat          = 0.0f;
  float band_val          = 1.0f;
  float intensity         = 1.0f;
  float chroma_hold       = 0.0f;  // 0 = white-hot spill, 1 = hue-preserving limit
  float input_opacity     = 1.0f;
  fx::AutoTrigger auto_trig;   // Off / Random (Poisson) / Beats — see effect_auto_trigger.h

  // --- Runtime state ---
  fx::DiffusionNetwork4 net;     // CPU: builds + exports the cycling matrices
  float    accum         = 0.0f; // hop accumulator
  int      hop_idx       = 0;
  int      n_hops        = 0;    // computed in tick(), consumed in render()
  int      hop_idx_start = 0;
  bool     needs_reset   = true; // zero the GPU sim state on next dispatch
  // Per-frame impulse queue (gen mode) + trigger flag (both modes).
  float    pending[BARS] = {0.0f, 0.0f, 0.0f, 0.0f};
  bool     trigger_fired = false;
  // Rising-edge detection.
  bool     gate_prev      = false;
  float    trigger_prev   = 0.0f;
  uint32_t target_rng      = 0x1357BD13u;
};

// Type-shared: compiled once in module_init().
static gpu::ComputePSO s_pso_sim;
static gpu::ComputePSO s_pso_color;
static gpu::ComputePSO s_pso_motion;

static inline float clampf(float v, float lo, float hi) {
  return v < lo ? lo : (v > hi ? hi : v);
}

// RGB → (hue turns, saturation, value). Matches nano_color.hlsl's HSV.
static void rgb_to_hsv(float r, float g, float b, float& h, float& s, float& v) {
  float mx = std::max(r, std::max(g, b));
  float mn = std::min(r, std::min(g, b));
  float d = mx - mn;
  v = mx;
  s = (mx > 1e-6f) ? d / mx : 0.0f;
  h = 0.0f;
  if (d > 1e-6f) {
    if      (mx == r) h = (g - b) / d + (g < b ? 6.0f : 0.0f);
    else if (mx == g) h = (b - r) / d + 2.0f;
    else              h = (r - g) / d + 4.0f;
    h /= 6.0f;
  }
}

static void update_band_hsv(State& s) {
  rgb_to_hsv(s.color_r, s.color_g, s.color_b, s.band_hue, s.band_sat, s.band_val);
}

// Show only the parameters relevant to the active impulse_mode.
static void apply_mode_visibility(int mode) {
  state::setFieldHidden("one_bar_target", mode != MODE_ONE_BAR);
  state::setFieldHidden("tex_in_boost",   mode != MODE_TEX_IN);
  state::setFieldHidden("tex_in_color",   mode != MODE_TEX_IN);
}

// Static (self-less) visibility evaluator — pure over state (see crop). Covers
// BOTH the impulse_mode knobs and the shared auto-trigger block's.
void eval_visibility(int n, const char* pb, const int* off, const int* len, const int* ops) {
  int mode = MODE_ONE_BAR;
  for (int i = 0; i < n; i++) {
    if (ops[i] != state::PatchReplace) continue;
    if (state::pathIs(pb + off[i], len[i], "impulse_mode")) mode = (int)state::patchFloat(i);
  }
  apply_mode_visibility(mode);
  fx::AutoTrigger::evalVisibility(n, pb, off, len, ops);
}

// On any trigger: flag it (tex_in mode samples on the GPU), and in the bar
// modes queue the per-bar amount per impulse_mode.
static void fire_impulse(State& s) {
  s.trigger_fired = true;
  if (s.impulse_mode == MODE_TEX_IN) return;   // sampled on the GPU
  float amt = clampf(s.impulse_strength, 0.0f, 8.0f);
  if (s.impulse_mode == MODE_ALL) {
    for (int b = 0; b < BARS; b++) s.pending[b] += amt;
  } else if (s.impulse_mode == MODE_RANDOM) {
    s.target_rng = s.target_rng * 1664525u + 1013904223u;
    int b = (int)((s.target_rng >> 8) % (uint32_t)BARS);
    s.pending[b] += amt;
  } else {   // MODE_ONE_BAR
    int b = s.one_bar_target < 0 ? 0 : (s.one_bar_target > BARS - 1 ? BARS - 1 : s.one_bar_target);
    s.pending[b] += amt;
  }
}

static void on_state_ready(void* self) {
  auto* s = static_cast<State*>(self);
  if (!s) return;
  apply_mode_visibility(s->impulse_mode);
  fx::AutoTrigger::applyVisibility(s->auto_trig.mode, s->auto_trig.div);
}

void module_init() {
  // fx::AutoTrigger::fields() wraps the chain to splice the auto-fire block
  // into the Trigger group (it takes and returns the Schema&).
  state::init("source.light.bounce_resonator", {1, 1, 1},
    fx::AutoTrigger::fields(
    state::Schema()
      // Top-level manual: high-level "what is this / how to use / what to try".
      .helpField("intro",
        "## Bounce Resonator\n"
        "Four light bars wired into a diffusion network: fire an impulse into one "
        "bar and its energy bounces and bleeds between the others, ringing out and "
        "decaying like a plucked resonator. Great for reactive LED-bar chases and "
        "glowing accents.\n\n"
        "**How to play it:** pick an **Impulse Mode** (one bar, a random bar, all "
        "bars, or sample from the incoming video), then drive it with **Gate** / "
        "**Trigger**, or let **Auto Rate** self-fire. **Try:** push **Feedback** "
        "near 1 for a long, humming decay; raise **Cycle Rate** so the bouncing "
        "pattern shimmers; and add **Hue Spread** to let each bounce drift the "
        "colour across the bars.")
      // --- Standard trigger surface ---
      .group("trigger", "Trigger")
        .groupHelp(
          "How and where impulses enter the network. **Gate** and **Trigger** both "
          "fire on a rising edge. *Auto Mode* is **Off** by default — set it to "
          "**Random** (Poisson, at *Auto Rate*) so it lives on its own, or **Beats** "
          "to lock the impulses to the host transport. **Impulse Mode** chooses the "
          "target — a fixed bar, a random bar, all four at once, or sampling colour "
          "straight from the incoming video (**Input Color** sets how strongly the "
          "sampled hue takes over from **Bar Color**). **Try:** hold a steady auto rate and "
          "modulate the mode for an evolving, hands-off light show.")
      .boolField ("gate",                false,                  state::PrimaryInput).label("Gate", "Gate")
      .eventField("trigger",                                     state::PrimaryInput).label("Trigger", "Trig")
    )   // ← fx::AutoTrigger::fields: auto_mode + auto_rate + auto_beats + custom
      .selectField("impulse_mode",       MODE_ONE_BAR,           state::PrimaryInput,
                   {{"tex_in", MODE_TEX_IN}, {"one_bar", MODE_ONE_BAR},
                    {"random_bar", MODE_RANDOM}, {"all_bars", MODE_ALL}}, /*wrap=*/true).label("Impulse Mode", "Mode")
      .intField  ("one_bar_target",      0, 0, 3,                state::PrimaryInput).label("Target Bar", "Bar")
      .floatField("tex_in_boost",        1.0f, 0.0f, 10.0f,      state::PrimaryInput).label("Input Boost", "Boost")
      .floatField("tex_in_color",        1.0f, 0.0f, 1.0f,       state::PrimaryInput).label("Input Color", "InCol")
      // --- Diffusion network ---
      .group("network", "Diffusion Network")
        .groupHelp(
          "The bouncing engine. **Feedback** sets how long energy rings before it "
          "dies (near 1 = a long resonant tail); **Spread** and **Contrast** shape "
          "how much each hop bleeds into its neighbours. **Cycle Rate** and "
          "**Pattern Count** drive how fast and how variedly the exchange matrix "
          "shuffles, while the **Hue** knobs let colour wander or converge on the "
          "bar colour. **Bar Color**, **Intensity** and **Input Opacity** set the "
          "final look; **Chroma Hold** keeps overdriven bars saturated at their "
          "hue instead of blooming white-hot.")
      .floatField("feedback",            0.90f, 0.0f, 1.2f,      state::PrimaryInput).label("Feedback", "Fbk")
      .floatField("spread",              0.30f, 0.0f, 1.0f,      state::PrimaryInput).label("Spread", "Sprd")
      .floatField("spread_contrast",     0.0f, 0.0f, 1.0f,       state::PrimaryInput).label("Spread Contrast", "SprCon")
      .floatField("decay_shaping",       0.0f, -1.0f, 1.0f,      state::PrimaryInput).label("Decay Shaping", "Decay")
      .floatField("hue_spread",          0.0f, 0.0f, 1.0f,       state::PrimaryInput).label("Hue Spread", "HueSp")
      .floatField("hue_converge",        0.0f, 0.0f, 1.0f,       state::PrimaryInput).label("Hue Converge", "HueCv")
      .intField  ("seed",                0, 0, 0x7FFFFFFF,       state::PrimaryInput).label("Seed", "Seed")
      .intField  ("pattern_count",       4, 1, 16,               state::PrimaryInput).label("Pattern Count", "Ptn")
      .floatField("cycle_rate",          6.0f, 0.0f, 60.0f,      state::PrimaryInput).label("Cycle Rate", "Cycle")
      .floatField("impulse_strength",    1.0f,  0.0f, 2.0f,      state::PrimaryInput).label("Impulse Strength", "Impls")
      .rgbField  ("band_color",          1.0f, 0.92f, 0.78f,     state::PrimaryInput).label("Bar Color", "Color")
      .floatField("intensity",           1.0f, 0.0f, 10.0f,      state::PrimaryInput).label("Intensity", "Int")
      .floatField("chroma_hold",         0.0f, 0.0f, 1.0f,       state::PrimaryInput).label("Chroma Hold", "Chroma")
      .floatField("input_opacity",       1.0f, 0.0f, 1.0f,       state::PrimaryInput).label("Input Opacity", "Opac")
      // --- I/O ---
      .textureField("tex_in",  state::PrimaryInput)
      .textureField("tex_out", state::PrimaryOutput)
      .renderOutputs(state::PrimaryOutput)
      .renderOutputs(state::PrimaryInput,  "render_outputs_in")
        .capability(state::Capability::Generator)
    );
  state::setOnStateReady(&on_state_ready);

  if (gpu::Device::backend() == gpu::Backend::None) return;

  state::registerShaderSPV("bounce_resonator_sim",   SIM_SPV,   SIM_SPV_SIZE);
  state::registerShaderSPV("bounce_resonator_color", COLOR_SPV, COLOR_SPV_SIZE);
  state::registerShaderSPV("bounce_resonator_motion", MOTION_SPV, MOTION_SPV_SIZE,
                           "rgba16float", "write");
  auto cs_sim    = gpu::Device::createShaderModuleByName("bounce_resonator_sim");
  auto cs_color  = gpu::Device::createShaderModuleByName("bounce_resonator_color");
  auto cs_motion = gpu::Device::createShaderModuleByName("bounce_resonator_motion");
  if (!cs_sim || !cs_color || !cs_motion) return;

  s_pso_sim = gpu::Device::createComputePSO(cs_sim, "main", gpu::Bindings()
      .tex2d(0)
      .storageRW(1)
      .storage(2)
      .uniform(3));
  s_pso_color = gpu::Device::createComputePSO(cs_color, "main", gpu::Bindings()
      .tex2d(0)
      .storageTex2d(1)
      .uniform(2)
      .storage(3));
  s_pso_motion = gpu::Device::createComputePSO(cs_motion, "main", gpu::Bindings()
      .tex2d(0)
      .storageTex2d(1, gpu::TextureFormat::RGBA16F)
      .uniform(2));

  state::log("bounce_resonator: module initialized");
}

void* create() {
  auto* s = new State();
  s->sim_state_buf     = gpu::Device::createBuffer(sizeof(SimState),     gpu::BufferUsage::Storage);
  s->sim_uniform_buf   = gpu::Device::createBuffer(sizeof(SimUniforms),  gpu::BufferUsage::Uniform);
  s->mats_buf          = gpu::Device::createBuffer(sizeof(MatsBuf),      gpu::BufferUsage::Storage);
  s->color_uniform_buf = gpu::Device::createBuffer(sizeof(ColorUniforms), gpu::BufferUsage::Uniform);
  return s;
}

void destroy(void* self) {
  auto* s = static_cast<State*>(self);
  if (!s) return;
  s->sim_state_buf.release();
  s->sim_uniform_buf.release();
  s->mats_buf.release();
  s->color_uniform_buf.release();
  s->motion_tex.release();
  s->zero_motion_tex.release();
  delete s;
}

void init(void* self) {
  auto* s = static_cast<State*>(self);
  if (!s) return;
  s->initialized = false;
  s->gate = false;
  s->gate_prev = false;
  s->trigger_prev = 0.0f;
  s->target_rng = 0x1357BD13u;
  s->accum = 0.0f;
  s->hop_idx = 0;
  s->n_hops = 0;
  s->needs_reset = true;
  s->trigger_fired = false;
  for (int b = 0; b < BARS; b++) s->pending[b] = 0.0f;
  update_band_hsv(*s);
  s->motion_w = 0;
  s->motion_h = 0;
  s->net.reset();

  if (!s_pso_sim.valid() || !s_pso_color.valid() || !s_pso_motion.valid()) return;
  if (!s->sim_state_buf.valid() || !s->sim_uniform_buf.valid()
      || !s->mats_buf.valid() || !s->color_uniform_buf.valid()) return;

  s->initialized = true;
  state::log("bounce_resonator: initialized");
}

void tick(void* self, double dt) {
  auto* s = static_cast<State*>(self);
  if (!s) return;
  if (!s->initialized) return;

  // Self-fire (Off / Random / Beats — effect_auto_trigger.h). Loop the count:
  // in Beats a long frame stall can cross several divisions at once.
  for (int i = 0, nf = s->auto_trig.fires(dt); i < nf; i++) fire_impulse(*s);

  // Build the cycling matrices on the CPU (cheap; uploaded in render()).
  fx::DiffusionNetwork4::Params p;
  p.feedback        = clampf(s->feedback, 0.0f, 1.2f);
  p.spread          = clampf(s->spread,   0.0f, 1.0f);
  p.spread_contrast = clampf(s->spread_contrast, 0.0f, 1.0f);
  p.decay_shaping   = clampf(s->decay_shaping, -1.0f, 1.0f);
  p.hue_spread      = clampf(s->hue_spread, 0.0f, 1.0f);
  p.hue_converge    = clampf(s->hue_converge, 0.0f, 1.0f);
  p.home_hue        = s->band_hue;
  p.seed            = (uint32_t)s->seed;
  p.pattern_count   = s->pattern_count;
  p.rate            = clampf(s->cycle_rate, 0.0f, 60.0f);
  s->net.setParams(p);

  // Advance the hop accumulator (the GPU sim does the actual stepping).
  int nc = s->pattern_count < 1 ? 1 : s->pattern_count;
  float rate = clampf(s->cycle_rate, 0.0f, 60.0f);
  int hops = 0;
  if (rate > 0.0f) {
    s->accum += (float)dt;
    hops = (int)std::floor(s->accum * rate);
    if (hops > fx::DiffusionNetwork4::kMaxHops) { hops = fx::DiffusionNetwork4::kMaxHops; s->accum = 0.0f; }
    else if (hops > 0) s->accum -= (float)hops / rate;
  }
  s->hop_idx_start = s->hop_idx;
  s->n_hops = hops;
  s->hop_idx = (s->hop_idx + hops) % nc;
}


void on_state_patched(void* self, int n, const char* pb, const int* off, const int* len, const int* ops) {
  auto* s = static_cast<State*>(self);
  if (!s) return;
  bool vis_changed = false;
  for (int i = 0; i < n; i++) {
    const char* path = pb + off[i];
    int plen = len[i];
    if (ops[i] != state::PatchReplace) continue;

    if (s->auto_trig.patch(path, plen, i, &vis_changed)) continue;
    if (state::pathIs(path, plen, "gate")) {
      bool new_gate = state::patchFloat(i) != 0.0f;
      if (new_gate && !s->gate_prev) fire_impulse(*s);
      s->gate = new_gate;
      s->gate_prev = new_gate;
    }
    else if (state::pathIs(path, plen, "trigger")) {
      float v = state::patchFloat(i);
      if (v != 0.0f && s->trigger_prev == 0.0f) fire_impulse(*s);
      s->trigger_prev = v;
    }
    else if (state::pathIs(path, plen, "impulse_mode")) {
      s->impulse_mode = (int)state::patchFloat(i);
      apply_mode_visibility(s->impulse_mode);
    }
    else if (state::pathIs(path, plen, "one_bar_target"))      s->one_bar_target     = (int)state::patchFloat(i);
    else if (state::pathIs(path, plen, "tex_in_boost"))        s->tex_in_boost       = state::patchFloat(i);
    else if (state::pathIs(path, plen, "tex_in_color"))        s->tex_in_color       = state::patchFloat(i);
    else if (state::pathIs(path, plen, "feedback"))            s->feedback           = state::patchFloat(i);
    else if (state::pathIs(path, plen, "spread"))              s->spread             = state::patchFloat(i);
    else if (state::pathIs(path, plen, "spread_contrast"))     s->spread_contrast    = state::patchFloat(i);
    else if (state::pathIs(path, plen, "decay_shaping"))       s->decay_shaping      = state::patchFloat(i);
    else if (state::pathIs(path, plen, "hue_spread"))          s->hue_spread         = state::patchFloat(i);
    else if (state::pathIs(path, plen, "hue_converge"))        s->hue_converge       = state::patchFloat(i);
    else if (state::pathIs(path, plen, "seed"))                s->seed               = (int)state::patchFloat(i);
    else if (state::pathIs(path, plen, "pattern_count"))       s->pattern_count      = (int)state::patchFloat(i);
    else if (state::pathIs(path, plen, "cycle_rate"))          s->cycle_rate         = state::patchFloat(i);
    else if (state::pathIs(path, plen, "impulse_strength"))    s->impulse_strength   = state::patchFloat(i);
    else if (state::pathIs(path, plen, "band_color")) {
      auto v = state::patchVec3(i);
      s->color_r = v.x; s->color_g = v.y; s->color_b = v.z;
      update_band_hsv(*s);
    }
    else if (state::pathIs(path, plen, "intensity"))           s->intensity          = state::patchFloat(i);
    else if (state::pathIs(path, plen, "chroma_hold"))         s->chroma_hold        = state::patchFloat(i);
    else if (state::pathIs(path, plen, "input_opacity"))       s->input_opacity      = state::patchFloat(i);
  }
  if (vis_changed)
    fx::AutoTrigger::applyVisibility(s->auto_trig.mode, s->auto_trig.div);
}

void render(void* self, int vp_w, int vp_h) {
  auto* s = static_cast<State*>(self);
  if (!s) return;
  if (!s->initialized || vp_w <= 0 || vp_h <= 0) return;
  auto in  = gpu::Device::textureForField("tex_in");
  auto out = gpu::Device::textureForField("tex_out");
  if (!in.valid() || !out.valid()) return;

  // Upload the CPU-built cycling matrices.
  MatsBuf mats = {};
  s->net.exportMatrices(mats.data);
  s->mats_buf.writeOne(mats);

  // Pack sim uniforms.
  SimUniforms su = {};
  su.feedback         = clampf(s->feedback, 0.0f, 1.2f);
  su.decay_shaping    = clampf(s->decay_shaping, -1.0f, 1.0f);
  su.hue_converge     = clampf(s->hue_converge, 0.0f, 1.0f);
  su.home_hue         = s->band_hue;
  su.pattern_count    = s->pattern_count < 1 ? 1 : s->pattern_count;
  su.hop_idx_start    = s->hop_idx_start;
  su.n_hops           = s->n_hops;
  su.mode             = (s->impulse_mode == MODE_TEX_IN) ? 1 : 0;
  su.pending0 = s->pending[0]; su.pending1 = s->pending[1];
  su.pending2 = s->pending[2]; su.pending3 = s->pending[3];
  su.band_hue         = s->band_hue;
  su.tex_in_boost     = clampf(s->tex_in_boost, 0.0f, 10.0f);
  su.trigger_fired    = s->trigger_fired ? 1 : 0;
  su.do_reset         = s->needs_reset ? 1 : 0;
  su.tex_w            = vp_w;
  su.tex_h            = vp_h;
  su.sample_nx        = SAMPLE_NX;
  su.sample_ny        = SAMPLE_NY;
  su.band_sat         = s->band_sat;
  su.tex_in_color     = clampf(s->tex_in_color, 0.0f, 1.0f);
  s->sim_uniform_buf.writeOne(su);

  // Pass 1 — sim (single-thread): step the hops + inject impulses.
  {
    auto cp = gpu::ComputePass::begin();
    cp.setPSO(s_pso_sim);
    cp.setTexture(in, 0, 0);
    cp.setBuffer(s->sim_state_buf,   1);
    cp.setBuffer(s->mats_buf,        2);
    cp.setBuffer(s->sim_uniform_buf, 3);
    cp.dispatch(1, 1, 1);
    cp.end();
  }

  // Pass 2 — color: read the post-step state, fill each bar's column.
  ColorUniforms cu = { s->band_val, clampf(s->intensity, 0.0f, 10.0f),
                       clampf(s->input_opacity, 0.0f, 1.0f),
                       clampf(s->chroma_hold, 0.0f, 1.0f) };
  s->color_uniform_buf.writeOne(cu);
  {
    auto cp = gpu::ComputePass::begin();
    cp.setPSO(s_pso_color);
    cp.setTexture(in,  0, 0);
    cp.setTexture(out, 1, 1);
    cp.setBuffer(s->color_uniform_buf, 2);
    cp.setBuffer(s->sim_state_buf,     3);
    cp.dispatch((vp_w + 7) / 8, (vp_h + 7) / 8);
    cp.end();
  }

  // Pass 3 — motion (passthrough). Skip when no downstream consumer.
  if (state::isOutputConnected("render_outputs")) {
    if (!s->motion_tex.valid() || s->motion_w != vp_w || s->motion_h != vp_h) {
      s->motion_tex = gpu::Device::createTexture(vp_w, vp_h, gpu::TextureFormat::RGBA16F);
      s->motion_w = vp_w;
      s->motion_h = vp_h;
      if (s->motion_tex.valid()) state::setGpuTexture("render_outputs/motion", s->motion_tex.id);
    }
    if (s->motion_tex.valid()) {
      auto upstream = gpu::Device::textureForField("render_outputs_in/motion");
      if (!upstream.valid()) {
        if (!s->zero_motion_tex.valid())
          s->zero_motion_tex = gpu::Device::createTexture(1, 1, gpu::TextureFormat::RGBA16F);
        upstream = s->zero_motion_tex;
      }
      if (upstream.valid()) {
        auto cp = gpu::ComputePass::begin();
        cp.setPSO(s_pso_motion);
        cp.setTexture(upstream,      0, 0);
        cp.setTexture(s->motion_tex, 1, 1);
        cp.setBuffer(s->color_uniform_buf, 2);
        cp.dispatch((vp_w + 7) / 8, (vp_h + 7) / 8);
        cp.end();
      }
    }
  }

  gpu::Device::submit();

  // Consume per-frame inputs so a re-render without a tick is a no-op step.
  for (int b = 0; b < BARS; b++) s->pending[b] = 0.0f;
  s->trigger_fired = false;
  s->needs_reset = false;
  s->n_hops = 0;
}

} // namespace bounce_resonator
