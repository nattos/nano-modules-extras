/*
 * filter.sim.simulant — "Simulant": a faithful port of the original Resolume
 * Wire patch (Wire/Patches/Simulant, 78 nodes), quirks intact.
 *
 * The original was NOT a wave equation and NOT a zoom-feedback drift — it is a
 * DIFFERENCE-BLEND + BLUR-DIFFUSION feedback loop whose accumulator is thresholded
 * into Sobel lines. Reconstructed node-by-node from the patch:
 *
 *   FEEDBACK LOOP (the "wave"), per frame:
 *     fadedPrev = prev * (1 - Choke)                       // mixer 23 (A=black)
 *     inject    = colorize(transform(input, scale .5))     // 36 / 159 / 167
 *     accumRaw  = lerp(fadedPrev, abs(fadedPrev-inject),   // mixer 20 = DIFFERENCE
 *                      injectAmount)                        //   ← load-bearing quirk
 *     delay    <- contrast(blur(accumRaw, 200px*Speed))    // 30 blur → 31 decay → 24
 *
 *   LINE EXTRACTION (reads accumRaw, node 20 out):
 *     smooth blur → Levels(bias/contrast) → Posterize → Sobel(strength,width)
 *       → crop 1px → replace-alpha → out.
 *
 *   FLICKER (CPU control rate): a Poisson-ish auto-fire (random > 1/2^(rate/120))
 *   and a manual Trigger restart an Attack/Release envelope; injectAmount =
 *   flickerBase + env*amount*sign + constAlpha.  THE STOCK QUIRK: the env sign is
 *   -1 (invert off) and flickerMin/Max = 0, so with default knobs injectAmount
 *   clamps to 0 and a fresh drop just decays — exactly the "never happy with it"
 *   out-of-box behavior. Bring it alive with Const Alpha / Flicker Max / Invert.
 *
 * Passes/frame (all one submit): inject → wave-blur(H,V+decay) → smooth-blur(H,V)
 *   → lines. Feedback runs at a reduced sim res; lines are extracted at viewport
 *   res. Stateful feedback: NO is_identity, NO temporal capability tag.
 */

#include <gpu.h>
#include <host.h>
#include <effect_utils.h>
#include <effect_twitch_mask.h>
#include "simulant_shaders.h"

#include <cmath>
#include <cstdint>

namespace simulant {

// --- Field / sim tuning constants ---
static constexpr int   SIM_MIN = 128;   // sim_scale=0 → chunky + cheap
static constexpr int   SIM_MAX = 720;   // sim_scale=1 → fine + detailed
static constexpr float WAVE_MAX_SIGMA = 0.15f; // wave_speed=1 → screen-crossing blur
static constexpr float WAVE_CONTRAST  = -0.016f; // node 31: contrast = ws * this
static constexpr float SMOOTH_MAX_SIGMA = 0.03f; // smoothing=1 → soft pre-edge blur
static constexpr float DILATE_SLOPE   = 0.7f;  // dilate front dims to 1-SLOPE at reach

// Pass A — difference-blend injection. inject split into steady (const) +
// time-dependent (flicker); the flicker is gated by an optional roaming mask.
struct InjectUniforms {
  float choke, inject_const, inject_flicker, scale;
  float pos_x, pos_y, color_alpha, color_contrast;
  float color_r, color_g, color_b, _p0;
  float mask_amount, mask_anchor_x, mask_anchor_y, mask_radius;
  float mask_softness, mask_shape, aspect_x, aspect_y;
};

// Spread (H/V) — the wave spread (Gaussian diffuse OR parabolic dilate) and the
// smoothing blur (always diffuse).
struct BlurUniforms {
  float dir_x, dir_y, step_uv, sigma_uv;
  float contrast, mode, parab, post_mult;
  float taps, _q0, _q1, _q2;
};

// Pass D — line extractor.
struct LinesUniforms {
  float zoom, level_bias, level_contrast, posterize_levels;
  float edge_strength, line_width_px, crop_right, crop_bottom;
  float line_r, line_g, line_b, _p1;
};

struct State {
  gpu::Texture delay[2];   // feedback buffer (blurred + decayed accum), ping-pong
  gpu::Texture accum;      // node 20 output (difference-blend result)
  gpu::Texture scratch;    // separable-blur intermediate
  gpu::Texture smoothed;   // pre-line smoothing-blur result
  int   cur = 0;
  int   sim_w = 0, sim_h = 0;
  bool  cleared = false;

  gpu::Buffer  inject_uniform;
  gpu::Buffer  blur_wh, blur_wv;   // wave blur H / V (+decay)
  gpu::Buffer  blur_sh, blur_sv;   // smoothing blur H / V
  gpu::Buffer  lines_uniform;
  gpu::Sampler samp_lin;
  bool initialized = false;

  // --- Params (faithful node names; defaults tuned to a lively preset) ---
  float wave_speed        = 0.41f;  // Wave Speed
  int   spread_mode       = 0;      // 0 = diffuse (Gaussian), 1 = dilate (max)
  float spread_decay      = 0.3f;   // dilate per-frame fade
  int   quality           = 16;     // samples each side per spread pass
  float choke             = 0.0f;   // Choke
  float input_scale       = 1.0f;   // A Scale (full-res injection, node 36)
  float pos_x             = 0.0f;   // A X
  float pos_y             = 0.0f;   // A Y
  float zoom              = 1.0f;   // Zoom (full-res injection → no compensation)
  float smoothing         = 0.19f;  // Smoothing (pre-edge blur)
  float sim_scale         = 0.76f;
  float levels            = 0.31f;  // Levels → posterize fineness
  float level_bias        = 0.01f;  // Level Bias
  float level_contrast    = -0.49f; // Level Contrast
  float line_strength     = 0.42f;  // Line Strength
  float line_width        = 0.0f;   // Line Width (px offset)
  // Flicker
  float flicker_rate      = 0.58f;  // A Flicker Rate
  float flicker_min       = 1.0f;   // A Flicker Min
  float flicker_max       = 0.32f;  // A Flicker Max
  float flicker_release   = 0.41f;  // A Flicker Release
  float flicker_env_amount= 1.0f;   // A Flicker Env Amount
  float flicker_invert    = 0.0f;   // A Flicker Invert (false → env SUBTRACTED)
  float const_alpha       = 0.86f;  // A Const Alpha (steady injection → alive)
  // Flicker spatial mask (roaming nano_twitch shape gating the flicker part)
  float mask_amount       = 0.0f;   // 0 = global flicker (off), 1 = fully shaped
  float mask_shape        = -0.5f;  // bipolar (see nano_twitch)
  float mask_radius       = 0.3f;
  float mask_softness     = 0.3f;
  float mask_position     = 0.0f;   // -1 outer ring → +1 centre
  float color_r = 1.0f, color_g = 1.0f, color_b = 1.0f;  // Color Filter
  float color_alpha       = 1.0f;   // Color Filter Alpha
  float color_contrast    = 0.2f;   // Colorize contrast (node 159)
  float line_cr = 1.0f, line_cg = 1.0f, line_cb = 1.0f;  // Line Colour (node 65 out)

  // --- Flicker envelope state ---
  bool     fire_prev     = false;
  bool     trigger_prev  = false;
  float    flicker_env   = 0.0f;   // linear-release Attack/Release env
  float    flicker_base  = 0.0f;   // rerolled base level on each fire
  float    inject_const  = 0.0f;   // steady (const_alpha), resolved per tick
  float    inject_flicker= 0.0f;   // time-dependent (base + env), resolved per tick
  uint32_t rng           = 0x1234567u;

  // Roaming flicker-mask anchor, re-picked on each pulse (own PRNG).
  fx::TwitchMask mask_twitch;
  float mask_anchor_x = 0.0f, mask_anchor_y = 0.0f;

  float dt = 1.0f / 60.0f;
};

static gpu::ComputePSO s_pso_inject;
static gpu::ComputePSO s_pso_blur;
static gpu::ComputePSO s_pso_lines;

// Context-dependent fields: Dilate Decay only in Dilate mode; the flicker-mask
// shape controls only when the mask is engaged.
static void apply_visibility(const State* s) {
  state::setFieldHidden("spread_decay", s->spread_mode != 1);
  bool mask_off = s->mask_amount <= 0.0f;
  state::setFieldHidden("mask_shape",    mask_off);
  state::setFieldHidden("mask_radius",   mask_off);
  state::setFieldHidden("mask_softness", mask_off);
  state::setFieldHidden("mask_position", mask_off);
}

// Fires once after init + the initial state replay: set visibility from the
// restored state so the inspector never flashes the wrong fields.
static void on_state_ready(void* self) {
  auto* s = static_cast<State*>(self);
  if (s) apply_visibility(s);
}

void module_init() {
  state::init("filter.sim.simulant", {1, 0, 0},
    state::Schema()
      .helpField("intro",
        "## Simulant\n"
        "A re-creation of the original *Simulant* Wire patch, tuned to a lively "
        "preset. It is a **difference-blend feedback** loop: each frame the image "
        "is *differenced* against a blurred copy of the previous frame and the "
        "result diffuses outward — that blur **is** the propagation. The churning "
        "accumulator is then traced into **Sobel lines**.\n\n"
        "**Try:** sweep *Wave Speed* (blur diffusion) and *Choke* (feedback "
        "retention); shape the lines with *Levels* / *Line Strength* / *Line "
        "Width*. On a static image the **Flicker** pulses keep it moving; **Const "
        "Alpha** is a steady injection. (Faithful quirk still in the engine: turn "
        "*Const Alpha*, *Flicker Min/Max* and the added base all to 0 with *Flicker "
        "Invert* off and it decays to nothing — the original's dead default.)")

      // ---- Feedback: the accumulator ----
      .group("feedback", "Feedback")
        .groupHelp("The core difference-blend loop. *Wave Speed* is the per-frame "
                   "spread — bigger pushes the structure outward faster (this is the "
                   "propagation). *Spread* picks the technique: *Diffuse* (soft "
                   "Gaussian rings) or *Dilate* (sharp expanding fronts, faded by "
                   "*Dilate Decay*). *Choke* fades the retained feedback each frame.")
      .floatField("wave_speed", 0.41f, 0.f, 1.f, state::PrimaryInput, nullptr, 0.01f, nullptr,
                  "Per-frame feedback spread — the outward 'wave' speed.")
        .label("Wave Speed", "Speed")
      .selectField("spread_mode", 0, state::PrimaryInput, {
        {"Diffuse", 0}, {"Dilate", 1} })
        .label("Spread Kernel", "Spread")
      .floatField("spread_decay", 0.3f, 0.f, 1.f, state::PrimaryInput, nullptr, 0.01f, nullptr,
                  "Dilate only: per-frame fade so fronts die instead of filling (0 = slow).")
        .label("Dilate Decay", "Decay")
      .intField("quality", 16, 4, 48, state::SecondaryInput)
        .label("Quality (Samples)", "Quality")
      .floatField("choke", 0.0f, 0.f, 1.f, state::PrimaryInput, nullptr, 0.01f, nullptr,
                  "Fades the retained feedback each frame (0 = full retention).")
        .label("Choke", "Choke")
      .floatField("sim_scale", 0.76f, 0.f, 1.f, state::SecondaryInput, nullptr, 0.01f, nullptr,
                  "Sim grid resolution — high retains detail, low is chunky + cheap.")
        .label("Detail Scale", "Scale")

      // ---- Injection: how the input enters ----
      .group("inject", "Injection")
        .groupHelp("How the input is placed into the accumulator. *Const Alpha* is "
                   "a steady injection amount — the simplest way to bring the stock "
                   "patch to life. *Scale* / *Pos X* / *Pos Y* place the (halved) "
                   "input; *Colour* / *Colour Mix* tint it (grey by default).")
      .floatField("const_alpha", 0.86f, 0.f, 1.f, state::PrimaryInput, nullptr, 0.01f, nullptr,
                  "Steady injection opacity added to the flicker (bring it alive).")
        .label("Const Alpha", "Const")
      .floatField("input_scale", 1.0f, 0.f, 2.f, state::SecondaryInput, nullptr, 0.01f, nullptr,
                  "Scale of the injected input (1 = full-res; Wire injected at 0.5).")
        .label("Input Scale", "InScale")
      .floatField("pos_x", 0.0f, -1.f, 1.f, state::SecondaryInput, nullptr, 0.01f, nullptr,
                  "Horizontal placement of the injected input.")
        .label("Pos X", "PosX")
      .floatField("pos_y", 0.0f, -1.f, 1.f, state::SecondaryInput, nullptr, 0.01f, nullptr,
                  "Vertical placement of the injected input.")
        .label("Pos Y", "PosY")
      .rgbField("color", 1.f, 1.f, 1.f, state::SecondaryInput)
        .label("Colour Filter", "Colour")
      .floatField("color_alpha", 1.0f, 0.f, 1.f, state::SecondaryInput, nullptr, 0.01f, nullptr,
                  "Crossfade the injected input toward the colour-tinted grey.")
        .label("Colour Mix", "ColMix")

      // ---- Flicker: pulse the injection ----
      .group("flicker", "Flicker")
        .groupHelp("Randomly pulses the injection to churn the field. Auto-fires on "
                   "a Poisson *Rate*; each fire restarts an *Attack/Release* env and "
                   "rerolls a base between *Min* and *Max*. NOTE: with *Invert* off "
                   "the env is SUBTRACTED (the original's default) — flip it, or "
                   "raise *Max* / *Const Alpha*, to see anything.")
      .floatField("flicker_rate", 0.58f, 0.f, 1.f, state::PrimaryInput, nullptr, 0.01f, nullptr,
                  "How often flicker pulses fire (Poisson; 0.25 ≈ the original 15).")
        .label("Flicker Rate", "Rate")
      .floatField("flicker_max", 0.32f, 0.f, 1.f, state::PrimaryInput, nullptr, 0.01f, nullptr,
                  "Upper bound of the random per-pulse injection base.")
        .label("Flicker Max", "FMax")
      .floatField("flicker_min", 1.0f, 0.f, 1.f, state::SecondaryInput, nullptr, 0.01f, nullptr,
                  "Lower bound of the random per-pulse injection base.")
        .label("Flicker Min", "FMin")
      .floatField("flicker_release", 0.41f, 0.01f, 2.f, state::SecondaryInput, nullptr, 0.01f, nullptr,
                  "Release time of the flicker envelope (seconds).")
        .label("Flicker Release", "FRel")
      .floatField("flicker_env_amount", 1.0f, 0.f, 2.f, state::SecondaryInput, nullptr, 0.01f, nullptr,
                  "How strongly the flicker envelope drives the injection.")
        .label("Flicker Env Amount", "FEnv")
      .boolField("flicker_invert", false, state::SecondaryInput,
                 "Add the flicker envelope instead of subtracting it (default off = subtract).")
        .label("Flicker Invert", "FInv")
      .eventField("trigger", state::PrimaryInput)

      // ---- Flicker Mask: confine the flicker to a roaming shape ----
      .group("flicker_mask", "Flicker Mask")
        .groupHelp("Spatially confine the flicker (only the time-dependent part — "
                   "the steady *Const Alpha* stays global) to a roaming shape that "
                   "jumps to a new spot on each pulse. *Amount* 0 = the whole frame "
                   "flickers as before; 1 = pulses land only inside the shape. "
                   "*Shape* is bipolar (radial ↔ linear ↔ solid, sign flips inside/"
                   "outside); *Size* / *Softness* / *Position* place it.")
      .floatField("mask_amount", 0.0f, 0.f, 1.f, state::PrimaryInput, nullptr, 0.01f, nullptr,
                  "How much the flicker is confined to the shape (0 = global).")
        .label("Mask Amount", "Mask")
      .floatField("mask_shape", -0.5f, -1.f, 1.f, state::PrimaryInput, nullptr, 0.01f, nullptr,
                  "Bipolar shape: |1| radial, |0.5| linear, |0| solid; sign flips in/out.")
        .label("Mask Shape", "MShape")
      .floatField("mask_radius", 0.3f, 0.f, 1.f, state::SecondaryInput, nullptr, 0.01f, nullptr,
                  "Size of the flicker region.")
        .label("Mask Size", "MSize")
      .floatField("mask_softness", 0.3f, 0.f, 1.f, state::SecondaryInput, nullptr, 0.01f, nullptr,
                  "Feather of the region's edge.")
        .label("Mask Softness", "MSoft")
      .floatField("mask_position", 0.0f, -1.f, 1.f, state::SecondaryInput, nullptr, 0.01f, nullptr,
                  "Spawn bias: -1 outer ring → +1 centre.")
        .label("Mask Position", "MPos")

      // ---- Line: threshold the accumulator ----
      .group("line", "Line")
        .groupHelp("Trace the accumulator into lines. *Smoothing* softens before the "
                   "edge detector; *Levels* sets posterize fineness; *Bias* / "
                   "*Contrast* pre-adjust; *Line Strength* is the edge gain and "
                   "*Line Width* the sampling offset; *Zoom* scales the field.")
      .floatField("line_strength", 0.42f, 0.f, 1.f, state::PrimaryInput, nullptr, 0.01f, nullptr,
                  "Edge (Sobel) gain — how strongly contours are drawn.")
        .label("Line Strength", "LineStr")
      .floatField("line_width", 0.0f, 0.f, 1.f, state::PrimaryInput, nullptr, 0.01f, nullptr,
                  "Edge sampling offset — thicker lines.")
        .label("Line Width", "LineW")
      .floatField("levels", 0.31f, 0.f, 1.f, state::PrimaryInput, nullptr, 0.01f, nullptr,
                  "Posterize fineness (low = hard bands, high = smooth).")
        .label("Levels", "Levels")
      .floatField("smoothing", 0.19f, 0.f, 1.f, state::SecondaryInput, nullptr, 0.01f, nullptr,
                  "Pre-edge blur so the contours come out clean.")
        .label("Smoothing", "Smooth")
      .floatField("level_bias", 0.01f, -1.f, 1.f, state::SecondaryInput, nullptr, 0.01f, nullptr,
                  "Brightness bias before posterize.")
        .label("Level Bias", "Bias")
      .floatField("level_contrast", -0.49f, -1.f, 1.f, state::SecondaryInput, nullptr, 0.01f, nullptr,
                  "Contrast before posterize.")
        .label("Level Contrast", "LvlCon")
      .floatField("zoom", 1.0f, 0.25f, 4.f, state::SecondaryInput, nullptr, 0.01f, nullptr,
                  "Scale the accumulator field for the line pass.")
        .label("Zoom", "Zoom")
      .rgbField("line_color", 1.f, 1.f, 1.f, state::SecondaryInput)
        .label("Line Colour", "LineCol")

      // ---- I/O ----
      .textureField("tex_in",  state::PrimaryInput)
      .textureField("tex_out", state::PrimaryOutput)
  );

  if (gpu::Device::backend() == gpu::Backend::None) return;

  state::registerShaderSPV("simulant_inject", INJECT_SPV, INJECT_SPV_SIZE, "rgba16float", "write");
  state::registerShaderSPV("simulant_blur",   BLUR_SPV,   BLUR_SPV_SIZE,   "rgba16float", "write");
  state::registerShaderSPV("simulant_lines",  LINES_SPV,  LINES_SPV_SIZE);

  auto cs_inject = gpu::Device::createShaderModuleByName("simulant_inject");
  auto cs_blur   = gpu::Device::createShaderModuleByName("simulant_blur");
  auto cs_lines  = gpu::Device::createShaderModuleByName("simulant_lines");
  if (!cs_inject || !cs_blur || !cs_lines) return;

  s_pso_inject = gpu::Device::createComputePSO(cs_inject, "main", gpu::Bindings()
      .tex2d(0).tex2d(1).sampler(2).storageTex2d(3, gpu::TextureFormat::RGBA16F).uniform(4));
  s_pso_blur = gpu::Device::createComputePSO(cs_blur, "main", gpu::Bindings()
      .tex2d(0).sampler(1).storageTex2d(2, gpu::TextureFormat::RGBA16F).uniform(3));
  s_pso_lines = gpu::Device::createComputePSO(cs_lines, "main", gpu::Bindings()
      .tex2d(0).sampler(1).storageTex2d(2).uniform(3));

  state::setOnStateReady(&on_state_ready);
  state::log("simulant: module initialized");
}

static uint32_t s_seed_counter = 0x9E3779B9u;   // distinct mask PRNG per instance

void* create() {
  auto* s = new State();
  s_seed_counter = s_seed_counter * 1664525u + 1013904223u;
  s->mask_twitch.seed(s_seed_counter ^ 0x51EDCA7u);
  s->inject_uniform = gpu::Device::createBuffer(sizeof(InjectUniforms), gpu::BufferUsage::Uniform);
  s->blur_wh        = gpu::Device::createBuffer(sizeof(BlurUniforms),   gpu::BufferUsage::Uniform);
  s->blur_wv        = gpu::Device::createBuffer(sizeof(BlurUniforms),   gpu::BufferUsage::Uniform);
  s->blur_sh        = gpu::Device::createBuffer(sizeof(BlurUniforms),   gpu::BufferUsage::Uniform);
  s->blur_sv        = gpu::Device::createBuffer(sizeof(BlurUniforms),   gpu::BufferUsage::Uniform);
  s->lines_uniform  = gpu::Device::createBuffer(sizeof(LinesUniforms),  gpu::BufferUsage::Uniform);
  s->samp_lin       = gpu::Device::createSampler(gpu::FilterMode::Linear, gpu::AddressMode::ClampToEdge);
  return s;
}

void destroy(void* self) {
  auto* s = static_cast<State*>(self);
  if (!s) return;
  s->delay[0].release();
  s->delay[1].release();
  s->accum.release();
  s->scratch.release();
  s->smoothed.release();
  s->inject_uniform.release();
  s->blur_wh.release();
  s->blur_wv.release();
  s->blur_sh.release();
  s->blur_sv.release();
  s->lines_uniform.release();
  s->samp_lin.release();
  delete s;
}

void init(void* self) {
  auto* s = static_cast<State*>(self);
  if (!s) return;
  if (!s_pso_inject.valid() || !s_pso_blur.valid() || !s_pso_lines.valid()) return;
  s->cur = 0;
  s->cleared = false;
  s->fire_prev = false;
  s->trigger_prev = false;
  s->flicker_env = 0.0f;
  s->flicker_base = 0.0f;
  s->inject_const = 0.0f;
  s->inject_flicker = 0.0f;
  s->rng = 0x1234567u;
  s->initialized = true;
}

static inline float rng_next(uint32_t& r) {
  r = r * 1664525u + 1013904223u;
  return (r >> 8) * (1.0f / 16777216.0f);
}

// Re-pick the roaming flicker-mask anchor (called on each flicker pulse).
static void advance_mask_anchor(State* s) {
  auto f = s->mask_twitch.update({ 1.0f, s->mask_shape, s->mask_radius,
                                   s->mask_softness, s->mask_position });
  s->mask_anchor_x = f.anchor_x;
  s->mask_anchor_y = f.anchor_y;
}

void tick(void* self, double dt) {
  auto* s = static_cast<State*>(self);
  if (!s) return;
  s->dt = (float)dt;

  // Poisson-ish auto-fire (nodes 71-74, 46, 47): random > 1/2^(rate/120).
  // flicker_rate 0.25 → rate 15 → threshold 0.917 (the original default).
  float rate_num  = s->flicker_rate * 60.0f;
  float threshold = 1.0f / std::pow(2.0f, rate_num / 120.0f);
  float u = rng_next(s->rng);
  bool fire = (u > threshold);
  if (fire && !s->fire_prev) {
    // Reroll the base level and restart the env (Attack 0, restart-at-zero).
    float r = rng_next(s->rng);
    s->flicker_base = s->flicker_min + (s->flicker_max - s->flicker_min) * r;
    s->flicker_env = 1.0f;
    advance_mask_anchor(s);   // each pulse lights a new shaped region
  }
  s->fire_prev = fire;

  // Linear release (Attack Release, node 109).
  s->flicker_env -= (float)dt / (s->flicker_release > 1e-3f ? s->flicker_release : 1e-3f);
  if (s->flicker_env < 0.0f) s->flicker_env = 0.0f;

  // Resolve the injection (node 110 Add → Hub → mixer 20 opacity2), split so the
  // spatial mask can gate only the FLICKER part. The env is SUBTRACTED unless
  // Invert is on — the stock quirk that makes defaults decay.
  float sign = (s->flicker_invert > 0.5f) ? 1.0f : -1.0f;
  s->inject_const   = s->const_alpha;
  s->inject_flicker = s->flicker_base + s->flicker_env * s->flicker_env_amount * sign;
}

void on_resolume_param(void*, long long, double) {}

void on_state_patched(void* self, int n, const char* pb, const int* off,
                      const int* len, const int* ops) {
  auto* s = static_cast<State*>(self);
  if (!s) return;
  bool vis_dirty = false;
  for (int i = 0; i < n; i++) {
    if (ops[i] != state::PatchReplace) continue;
    const char* p = pb + off[i];
    int l = len[i];
    if      (state::pathIs(p, l, "wave_speed"))         s->wave_speed = state::patchFloat(i);
    else if (state::pathIs(p, l, "spread_mode"))      { s->spread_mode = state::patchInt(i); vis_dirty = true; }
    else if (state::pathIs(p, l, "spread_decay"))       s->spread_decay = state::patchFloat(i);
    else if (state::pathIs(p, l, "quality"))            s->quality = state::patchInt(i);
    else if (state::pathIs(p, l, "choke"))              s->choke = state::patchFloat(i);
    else if (state::pathIs(p, l, "sim_scale"))          s->sim_scale = state::patchFloat(i);
    else if (state::pathIs(p, l, "const_alpha"))        s->const_alpha = state::patchFloat(i);
    else if (state::pathIs(p, l, "input_scale"))        s->input_scale = state::patchFloat(i);
    else if (state::pathIs(p, l, "pos_x"))              s->pos_x = state::patchFloat(i);
    else if (state::pathIs(p, l, "pos_y"))              s->pos_y = state::patchFloat(i);
    else if (state::pathIs(p, l, "color_alpha"))        s->color_alpha = state::patchFloat(i);
    else if (state::pathIs(p, l, "flicker_rate"))       s->flicker_rate = state::patchFloat(i);
    else if (state::pathIs(p, l, "flicker_min"))        s->flicker_min = state::patchFloat(i);
    else if (state::pathIs(p, l, "flicker_max"))        s->flicker_max = state::patchFloat(i);
    else if (state::pathIs(p, l, "flicker_release"))    s->flicker_release = state::patchFloat(i);
    else if (state::pathIs(p, l, "flicker_env_amount")) s->flicker_env_amount = state::patchFloat(i);
    else if (state::pathIs(p, l, "flicker_invert"))     s->flicker_invert = state::patchBool(i) ? 1.0f : 0.0f;
    else if (state::pathIs(p, l, "mask_amount"))      { s->mask_amount = state::patchFloat(i); vis_dirty = true; }
    else if (state::pathIs(p, l, "mask_shape"))         s->mask_shape = state::patchFloat(i);
    else if (state::pathIs(p, l, "mask_radius"))        s->mask_radius = state::patchFloat(i);
    else if (state::pathIs(p, l, "mask_softness"))      s->mask_softness = state::patchFloat(i);
    else if (state::pathIs(p, l, "mask_position"))      s->mask_position = state::patchFloat(i);
    else if (state::pathIs(p, l, "line_strength"))      s->line_strength = state::patchFloat(i);
    else if (state::pathIs(p, l, "line_width"))         s->line_width = state::patchFloat(i);
    else if (state::pathIs(p, l, "levels"))             s->levels = state::patchFloat(i);
    else if (state::pathIs(p, l, "smoothing"))          s->smoothing = state::patchFloat(i);
    else if (state::pathIs(p, l, "level_bias"))         s->level_bias = state::patchFloat(i);
    else if (state::pathIs(p, l, "level_contrast"))     s->level_contrast = state::patchFloat(i);
    else if (state::pathIs(p, l, "zoom"))               s->zoom = state::patchFloat(i);
    else if (state::pathIs(p, l, "color")) {
      auto v = state::patchVec3(i); s->color_r = v.x; s->color_g = v.y; s->color_b = v.z;
    }
    else if (state::pathIs(p, l, "line_color")) {
      auto v = state::patchVec3(i); s->line_cr = v.x; s->line_cg = v.y; s->line_cb = v.z;
    }
    else if (state::pathIs(p, l, "trigger")) {
      bool t = state::patchFloat(i) != 0.0f;
      if (t && !s->trigger_prev) { s->flicker_env = 1.0f; advance_mask_anchor(s); }  // rising edge
      s->trigger_prev = t;
    }
  }
  if (vis_dirty) apply_visibility(s);
}

static bool ensure_field(State* s, int vp_w, int vp_h) {
  float t = s->sim_scale; if (t < 0.f) t = 0.f; if (t > 1.f) t = 1.f;
  int longSide = (int)std::lround(SIM_MIN + (SIM_MAX - SIM_MIN) * t);
  if (longSide < 16) longSide = 16;
  int sw, sh;
  if (vp_w >= vp_h) { sw = longSide; sh = (int)std::lround((float)longSide * vp_h / (float)vp_w); }
  else              { sh = longSide; sw = (int)std::lround((float)longSide * vp_w / (float)vp_h); }
  if (sw < 8) sw = 8; if (sh < 8) sh = 8;

  if (s->delay[0].valid() && s->delay[1].valid() && s->sim_w == sw && s->sim_h == sh)
    return true;

  s->delay[0].release(); s->delay[1].release();
  s->accum.release(); s->scratch.release(); s->smoothed.release();
  s->delay[0] = gpu::Device::createTexture(sw, sh, gpu::TextureFormat::RGBA16F);
  s->delay[1] = gpu::Device::createTexture(sw, sh, gpu::TextureFormat::RGBA16F);
  s->accum    = gpu::Device::createTexture(sw, sh, gpu::TextureFormat::RGBA16F);
  s->scratch  = gpu::Device::createTexture(sw, sh, gpu::TextureFormat::RGBA16F);
  s->smoothed = gpu::Device::createTexture(sw, sh, gpu::TextureFormat::RGBA16F);
  s->sim_w = sw; s->sim_h = sh;
  s->cur = 0;
  s->cleared = false;
  return s->delay[0].valid() && s->delay[1].valid();
}

static void dispatch_blur(State* s, gpu::Texture src, gpu::Texture dst,
                          gpu::Buffer buf, float dx, float dy, float sigma,
                          float contrast, float mode, float parab, float post_mult, int taps) {
  // Fixed reach = 4·sigma; step = reach/taps so Quality changes tap DENSITY, not
  // the blur's scale (taps=16 reproduces the old sigma*0.25 spacing).
  float step = (sigma > 1e-6f) ? sigma * (4.0f / (float)taps) : 0.0f;
  BlurUniforms b = { dx, dy, step, sigma, contrast, mode, parab, post_mult,
                     (float)taps, 0.f, 0.f, 0.f };
  buf.writeOne(b);
  auto cp = gpu::ComputePass::begin();
  cp.setPSO(s_pso_blur);
  cp.setTexture(src, 0, 0);
  cp.setSampler(s->samp_lin, 1);
  cp.setTexture(dst, 2, 1);
  cp.setBuffer(buf, 3);
  cp.dispatch((s->sim_w + 7) / 8, (s->sim_h + 7) / 8);
  cp.end();
}

void render(void* self, int vp_w, int vp_h) {
  auto* s = static_cast<State*>(self);
  if (!s || !s->initialized || vp_w <= 0 || vp_h <= 0) return;

  auto in  = gpu::Device::textureForField("tex_in");
  auto out = gpu::Device::textureForField("tex_out");
  if (!in.valid() || !out.valid()) return;
  if (!ensure_field(s, vp_w, vp_h)) return;

  if (!s->cleared) {
    gpu::Device::clear(s->delay[0], 0.f, 0.f, 0.f, 1.f);
    gpu::Device::clear(s->delay[1], 0.f, 0.f, 0.f, 1.f);
    s->cleared = true;
  }

  int rd = s->cur, wr = s->cur ^ 1;

  // --- Pass A: difference-blend injection → accumRaw (node 20 output) ---
  auto [ax, ay] = fx::coverSquare(vp_w, vp_h);
  InjectUniforms iu = {};
  iu.choke          = s->choke;
  iu.inject_const   = s->inject_const;    // steady Const Alpha (global)
  iu.inject_flicker = s->inject_flicker;  // time-dependent flicker (maskable)
  iu.scale          = s->input_scale;   // full-res injection — Wire used A Scale
                                        // × 0.5 (node 148); our pipeline is more
                                        // capable, so we inject at 1:1.
  iu.pos_x          = s->pos_x * 0.5f;          // node 115 × 0.5
  iu.pos_y          = s->pos_y * 0.5f;
  iu.color_alpha    = s->color_alpha;
  iu.color_contrast = s->color_contrast;
  iu.color_r        = s->color_r;
  iu.color_g        = s->color_g;
  iu.color_b        = s->color_b;
  iu.mask_amount    = s->mask_amount;
  iu.mask_anchor_x  = s->mask_anchor_x;
  iu.mask_anchor_y  = s->mask_anchor_y;
  iu.mask_radius    = s->mask_radius;
  iu.mask_softness  = s->mask_softness;
  iu.mask_shape     = s->mask_shape;
  iu.aspect_x       = ax;
  iu.aspect_y       = ay;
  s->inject_uniform.writeOne(iu);
  {
    auto cp = gpu::ComputePass::begin();
    cp.setPSO(s_pso_inject);
    cp.setTexture(s->delay[rd], 0, 0);   // feedback (previous, blurred)
    cp.setTexture(in, 1, 0);             // input video
    cp.setSampler(s->samp_lin, 2);
    cp.setTexture(s->accum, 3, 1);       // node 20 output
    cp.setBuffer(s->inject_uniform, 4);
    cp.dispatch((s->sim_w + 7) / 8, (s->sim_h + 7) / 8);
    cp.end();
  }

  // Per-axis sigma so the diffusion is ISOTROPIC in pixels. The sim aspect
  // matches the viewport, so equal reach in sim-pixels = equal reach on screen;
  // a single uv sigma would blur ~(longSide/shortSide)× wider along the long
  // axis and — compounded every feedback frame — stretch the rings into
  // ellipses. Convert a normalized reach (fraction of the long side) to a
  // per-axis uv sigma: sigma_axis = reach_px / axis_dim.
  float longSide = (float)(s->sim_w > s->sim_h ? s->sim_w : s->sim_h);

  // --- Wave spread (nodes 30 + 31): accumRaw → delay[wr], H then V ---
  // Diffuse = Gaussian (contrast decay in V). Dilate = parabolic max: the k²
  // penalty is in TAP units (isotropic in px thanks to per-axis step), scaled so
  // a bright front dims to (1-DILATE_SLOPE) at the window edge (k=N); post_mult
  // (V pass) is the per-frame fade.
  float wave_reach = s->wave_speed * WAVE_MAX_SIGMA * longSide;   // sim px
  float wsig_h = wave_reach / (float)s->sim_w;
  float wsig_v = wave_reach / (float)s->sim_h;
  int   N      = s->quality; if (N < 4) N = 4; if (N > 48) N = 48;
  bool  dilate = (s->spread_mode == 1);
  float mode   = dilate ? 1.0f : 0.0f;
  float parab  = DILATE_SLOPE / (float)(N * N);          // edge (k=N) penalty = SLOPE
  float decay  = 1.0f - s->spread_decay * 0.2f;          // 1.0 → 0.8 per frame
  float wave_contrast = dilate ? 0.0f : s->wave_speed * WAVE_CONTRAST;
  dispatch_blur(s, s->accum,   s->scratch,   s->blur_wh, 1.f, 0.f, wsig_h, 0.0f,          mode, parab, 1.0f,                   N);
  dispatch_blur(s, s->scratch, s->delay[wr], s->blur_wv, 0.f, 1.f, wsig_v, wave_contrast, mode, parab, dilate ? decay : 1.0f, N);

  // --- Smoothing blur (node 67): accumRaw → smoothed, H then V (always diffuse) ---
  float sm_reach = s->smoothing * SMOOTH_MAX_SIGMA * longSide;    // sim px
  float ssig_h = sm_reach / (float)s->sim_w;
  float ssig_v = sm_reach / (float)s->sim_h;
  dispatch_blur(s, s->accum,   s->scratch,  s->blur_sh, 1.f, 0.f, ssig_h, 0.0f, 0.0f, 0.0f, 1.0f, N);
  dispatch_blur(s, s->scratch, s->smoothed, s->blur_sv, 0.f, 1.f, ssig_v, 0.0f, 0.0f, 0.0f, 1.0f, N);

  // --- Pass D: line extraction (viewport res) ---
  LinesUniforms lu = {};
  lu.zoom             = s->zoom;
  lu.level_bias       = s->level_bias;
  lu.level_contrast   = s->level_contrast;
  lu.posterize_levels = std::round(2.0f + s->levels * 46.0f);  // low=hard bands
  lu.edge_strength    = 0.5f + s->line_strength * 8.0f;
  lu.line_width_px    = 1.0f + s->line_width * 6.0f;
  lu.crop_right       = 1.0f / (float)vp_w;   // 1px crop (node 149)
  lu.crop_bottom      = 1.0f / (float)vp_h;
  lu.line_r           = s->line_cr;
  lu.line_g           = s->line_cg;
  lu.line_b           = s->line_cb;
  s->lines_uniform.writeOne(lu);
  {
    auto cp = gpu::ComputePass::begin();
    cp.setPSO(s_pso_lines);
    cp.setTexture(s->smoothed, 0, 0);
    cp.setSampler(s->samp_lin, 1);
    cp.setTexture(out, 2, 1);
    cp.setBuffer(s->lines_uniform, 3);
    cp.dispatch((vp_w + 7) / 8, (vp_h + 7) / 8);
    cp.end();
  }

  s->cur = wr;
  gpu::Device::submit();
}

} // namespace simulant
