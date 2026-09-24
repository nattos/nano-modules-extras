/*
 * motion.local_delay — stylized motion-driven local delay.
 *
 * Estimates dense optical flow between the current and previous frame with a
 * pyramidal Lucas-Kanade solver (windowed structure tensor, coarse-to-fine on
 * a downsampled luma pyramid — smooth and coherent, not the old spiky single-
 * pixel normal flow), temporally smooths it, masks it by a stochastic-noise
 * field + a signed center vignette, then in the color pass FORWARD-ADVECTS
 * each pixel along that flow streamline and samples the ORIGINAL input at the
 * endpoint. Sampling the live input (rather than looking back into smeary
 * history) gives clean solid pixels: a trailing pixel's smoothed flow points
 * toward the object, so advecting forward lands on it — a solid motion echo.
 * The flow is also published on render_outputs/motion for a downstream
 * motion.blur.
 *
 * Pass pipeline (shared common.hlsl):
 *   luma     — input → HALF-res Rec.601 luma (the "downsample first" step).
 *   down x2  — luma pyramid: half → quarter → eighth.
 *   lk x3    — coarse-to-fine Lucas-Kanade (eighth → quarter → half),
 *              warping the previous luma by the upsampled coarse flow.
 *   upsample — half-res flow → full res.
 *   align    — colinear flow polish + TEMPORAL flow EMA (kills flicker;
 *              `smoothing` knob) + mask + index
 *              (RG = smoothed flow, B = index, A = mask).
 *   color    — forward-advect along the flow (delay_steps steps), sample the
 *              original input at the endpoint.
 *   motion   — write aligned_flow*mask*gain to render_outputs/motion (gated).
 *
 * Stateful (history + luma pyramid + stochastic step) → NO is_identity.
 * Freeform (multi-pass, neighbor reads, struct rail) → NO fusion.
 *
 * Class-like instance model: module_init() compiles the shared compute
 * PSOs + publishes the schema once per type; each chain entry gets its own
 * State (params, ping-pong luma pyramid + history, flow textures, step
 * state) via create(). All instance callbacks take `self`.
 */

#include <gpu.h>
#include <host.h>
#include <effect_utils.h>
#include <effect_twitch_mask.h>
#include "local_delay_shaders.h"

#include <cmath>
#include <cstdint>

namespace local_delay {

// 4 float4 rows = 64 bytes. std140-style alignment; every pass shader that
// reads it declares the same layout (one uniform buffer, bound into each).
struct Uniforms {
  // row 0: blend + noise
  float delay_amount;
  float noise_weight;
  float seed;            // time-invariant base seed (user seed*17)
  float weight_gain;

  // row 1: vignette + squash
  float vignette;
  float vignette_radius;
  float vignette_softness;
  float squash;

  // row 2: flow / align
  float max_flow;
  float align_amount;
  float align_sharpness;
  float have_history;    // 0/1

  // row 3: aspect + debug + history EMA
  float aspect_x;
  float aspect_y;
  float debug_show_motion;  // 0/1
  float history_alpha;      // dt-based EMA rate: 1 - 0.5^(dt/half_life)

  // row 4: motion output + advection + noise clock
  float motion_gain;        // published-velocity scale (render_outputs/motion)
  float delay_steps;        // number of forward-advection steps (color pass)
  float noise_time;         // float noise clock (per-pixel staggered re-roll)
  float delay_dir;          // +1 = Past (advect with flow), -1 = Future (against)

  // row 5: twitch (a roaming vignette — random anchor + intensity/frame)
  float twitch_shape;         // -1..1: + blacks the rim, - blacks the centre
  float twitch_radius;        // 0..1 cover-square radius of the twitch
  float twitch_softness;      // 0..1 falloff width
  float twitch_strength;      // amount × this frame's random intensity (0..1)

  // row 6: twitch anchor (cover-square coords, chosen on the CPU each frame)
  float twitch_anchor_x;
  float twitch_anchor_y;
  float _pad5;
  float _pad6;
};
static_assert(sizeof(Uniforms) == 112, "Uniforms layout mismatch");

static constexpr int NUM_LEVELS = 3;   // luma pyramid: half, quarter, eighth

// Per-instance state. One per chain entry.
struct State {
  gpu::Buffer  uniform_buf;
  // Ping-pong luma pyramid (RGBA16F, luma in R). luma[cur] is built this
  // frame; luma[1-cur] is the previous frame's pyramid the LK passes warp against.
  gpu::Texture luma[2][NUM_LEVELS];
  // LK per-level flow outputs (rgba16f, RG = uv/frame flow).
  gpu::Texture flow_e;     // eighth res  (coarsest)
  gpu::Texture flow_q;     // quarter res
  gpu::Texture flow_h;     // half res    (finest LK level)
  gpu::Texture flow_zero;  // 1x1 rgba16f zero — incoming flow for the coarsest level
  gpu::Texture flow_a;     // full-res upsampled flow → align
  gpu::Texture flow_ema[2];// full-res ping-pong: smoothed flow (RG) + index (B) + mask (A)
  gpu::Texture motion_tex; // published render_outputs/motion
  gpu::Texture zero_motion_tex; // 1x1 rgba16f fallback when no upstream
  int  tex_w = 0;
  int  tex_h = 0;
  int  motion_w = 0;
  int  motion_h = 0;
  int  luma_idx = 0;
  int  flow_idx = 0;       // ping-pong index for flow_ema (temporal flow smoothing)
  bool have_history = false;
  bool initialized = false;

  // --- Schema-mirrored params ---
  float delay_amount      = 0.7f;   // advection step scale
  float delay_steps       = 12.0f;  // number of forward-advection steps (integer)
  int   delay_direction   = 0;      // 0 = Past (causal echo), 1 = Future (echo ahead)
  int   flow_source       = 0;      // 0 = Estimate (LK), 1 = Incoming render_outputs motion
  float smoothing         = 0.4f;   // dt-based flow/index EMA (flicker stability)
  float noise_weight      = 0.0f;
  float noise_motion      = 0.0f;   // noise temporal re-roll rate (0..1)
  float vignette          = 0.0f;
  float vignette_radius   = 0.5f;
  float vignette_softness = 0.3f;
  // Twitch — a vignette-like mask at a per-frame RANDOM anchor + intensity.
  // (Designed to be lifted into a standalone effect; keep the math self-contained.)
  float twitch_amount       = 0.0f;   // 0 = off; modulation depth into the mask
  float twitch_shape        = -0.5f;  // -1..1 bipolar; |1| radial → |.5| linear → |0| solid
  float twitch_radius       = 0.3f;
  float twitch_softness     = 0.3f;
  float twitch_position     = 0.0f;   // -1 → outer-oval spawn, +1 → centre spawn
  float squash            = 0.0f;
  float max_flow          = 0.03f;
  float weight_gain       = 0.5f;   // motion sensitivity (0..1 → ×2048 effective)
  float align_amount      = 0.5f;
  float align_sharpness   = 4.0f;
  float motion_gain       = 0.03f;  // published-motion scale (0..1 → ×128 effective)
  int   seed              = 0;
  bool  debug_show_motion = false;

  // Noise clock: a float time advancing at the `motion` rate (pow(60, motion)
  // - 1 per second). The shader builds a per-pixel staggered phase from it, so
  // pixels re-roll independently rather than all at once.
  float        noise_time = 0.0f;

  // Most recent frame dt (from tick) → frame-rate-independent history EMA.
  float        last_dt    = 1.0f / 60.0f;

  // Roaming twitch vignette — owns its own per-instance PRNG.
  fx::TwitchMask twitch;
};

// Type-shared: compiled once in module_init().
static gpu::ComputePSO s_pso_luma;
static gpu::ComputePSO s_pso_down;
static gpu::ComputePSO s_pso_lk;
static gpu::ComputePSO s_pso_upsample;
static gpu::ComputePSO s_pso_align;
static gpu::ComputePSO s_pso_color;
static gpu::ComputePSO s_pso_motion;

void module_init() {
  state::init("motion.local_delay", {1, 0, 1},
    state::Schema()
      // Top-level manual: high-level "what is this / how to use / what to try".
      .helpField("intro",
        "## Local Delay\n"
        "A motion-driven echo. It estimates the **optical flow** of the incoming "
        "video and smears each pixel *along* that flow — sampling the live image at "
        "the endpoint, so the trails stay solid instead of ghosting. The result is a "
        "clean directional motion echo that only appears where things move.\n\n"
        "**Try:** raise *Steps* + *Delay Amount* together for a longer trail; flip "
        "*Direction* to Future to throw the echo **ahead** of the motion; add a little "
        "*Twitch* for a nervous, roaming cut-out.")
      // --- Standard (live) ---
      .group("standard", "Standard")
        .groupHelp(
          "The everyday controls. *Delay Amount* × *Steps* set how long the echo "
          "trails; *Direction* points it behind (Past) or ahead (Future) of the "
          "motion. *Flow Source* picks whether the flow is estimated here or read "
          "from an upstream motion rail. The *Noise*, *Vignette* and *Twitch* knobs "
          "mask **where** the delay is allowed to act — useful for keeping the echo "
          "off a subject or roaming it around the frame.")
      // Advection step scale: how far each step walks along the flow (× the
      // per-step gain and the spatial mask). Larger = longer trail / more delay.
      .floatField("delay_amount",      0.7f,  0.0f, 1.0f, state::PrimaryInput).label("Delay Amount", "Delay")
      // Number of forward-advection steps along the flow streamline (integer).
      // More steps = longer, smoother trail (and more cost). Total reach ≈
      // delay_steps × delay_amount frames of motion.
      .floatField("delay_steps",       12.0f, 1.0f, 32.0f, state::PrimaryInput).label("Steps", "Steps")
      // Which way the echo trails. Past = causal (behind the motion, where it
      // came from); Future = anti-causal (ahead, where it's going).
      .selectField("delay_direction",  0, state::PrimaryInput, {{"Past", 0}, {"Future", 1}}).label("Direction", "Dir")
      // Where the flow comes from. Estimate = our pyramidal Lucas-Kanade on the
      // input image; Incoming = use the render_outputs_in/motion vectors fed in
      // upstream (e.g. from motion_field / a renderer), skipping the estimator.
      .selectField("flow_source",      0, state::PrimaryInput, {{"Estimate", 0}, {"Incoming", 1}}).label("Flow Source", "Flow")
      // Temporal smoothing of the FLOW/index field (dt-based EMA, half-life =
      // smoothing^2 * 1.5s, frame-rate independent). Removes flicker, since the
      // per-frame flow estimate is otherwise noisy and zeroes out on duplicate
      // frames when render fps outpaces the source. Independent of the color
      // tails — it only stabilizes which temporal level each pixel reaches.
      // 0 = no smoothing (instant, snappy but flickery).
      .floatField("smoothing",         0.4f,  0.0f, 1.0f, state::PrimaryInput).label("Smoothing", "Smooth")
      // Probability a pixel is affected by noise (clean binary selection).
      // Affected pixels get a balanced ×[0,2] multiplier (boost some, cut
      // others, averaging 1), so sweeping this adds stochastic variation
      // without dimming the average effect. Re-rolls over time via noise_motion.
      .floatField("noise_weight",      0.0f,  0.0f, 1.0f, state::PrimaryInput).label("Noise Weight", "NseWt")
      // Noise re-roll rate, Hz = pow(60, noise_motion) - 1 (0 = frozen field).
      // Each pixel re-rolls on its own staggered clock.
      .floatField("noise_motion",      0.0f,  0.0f, 1.0f, state::PrimaryInput).label("Noise Motion", "NseMot")
      // Signed center vignette: + suppresses motion OUTSIDE the radius,
      // - suppresses INSIDE it, 0 = no spatial mask.
      .floatField("vignette",          0.0f, -1.0f, 1.0f, state::PrimaryInput).label("Vignette", "Vig")
      .floatField("vignette_radius",   0.5f,  0.0f, 1.0f, state::PrimaryInput).label("Vignette Radius", "VigRad")
      .floatField("vignette_softness", 0.3f,  0.0f, 1.0f, state::PrimaryInput).label("Vignette Softness", "VigSft")
      // Twitch: a roaming vignette that picks a new random anchor +
      // intensity EACH FRAME. `twitch_amount` (0 = off): 0..0.5 ramps the
      // modulation depth in; 0.5..1 holds depth full and boosts the random
      // per-frame weight toward 1.0 (cuts motion harder, more often).
      // `twitch_shape` is bipolar: sign sets polarity (+ blacks the rim, -
      // the centre); magnitude morphs |1| radial → |0.5| linear → |0| solid.
      // `twitch_position` biases WHERE it spawns: -1 = an oval around the
      // outside, +1 = an oval in the centre.
      .floatField("twitch_amount",       0.0f,  0.0f, 1.0f, state::PrimaryInput).label("Twitch Amount", "Twitch")
      .floatField("twitch_shape",       -0.5f, -1.0f, 1.0f, state::PrimaryInput).label("Twitch Shape", "TwShp")
      .floatField("twitch_radius",       0.3f,  0.0f, 1.0f, state::PrimaryInput).label("Twitch Radius", "TwRad")
      .floatField("twitch_softness",     0.3f,  0.0f, 1.0f, state::PrimaryInput).label("Twitch Softness", "TwSft")
      .floatField("twitch_position",     0.0f, -1.0f, 1.0f, state::PrimaryInput).label("Twitch Position", "TwPos")
      // Power curve on the blend weight (-1 crush / +1 lift, §1.3).
      .floatField("squash",            0.0f, -1.0f, 1.0f, state::PrimaryInput).label("Squash", "Squash")
      // --- Tuning ---
      .group("tuning", "Tuning")
        .groupHelp(
          "Under-the-hood knobs for the flow estimator and the published motion "
          "rail. *Sensitivity* and *Max Flow* set how much motion registers and its "
          "ceiling; *Align* polishes the raw estimate; *Motion Gain* scales the "
          "vectors written to `render_outputs/motion` for a downstream motion blur. "
          "The defaults are well-tuned — reach here only when the echo over- or "
          "under-reacts.")
      // uv/frame ceiling on the flow magnitude.
      .floatField("max_flow",          0.03f, 0.0f, 0.1f, state::PrimaryInput).label("Max Flow", "MaxFlw")
      // Motion sensitivity: scales |flow| into the temporal-lookup index
      // (higher = more delay/reach for the same motion). Normalized 0..1, with
      // an effective range of 0..2048 internally (typical flow is tiny).
      .floatField("weight_gain",       0.5f, 0.0f, 1.0f, state::PrimaryInput).label("Sensitivity", "Sens")
      // Raw↔colinear-aligned flow lerp (a polish on the LK estimate).
      .floatField("align_amount",      0.5f,  0.0f, 1.0f, state::PrimaryInput).label("Align Amount", "Align")
      .floatField("align_sharpness",   4.0f,  1.0f, 16.0f, state::PrimaryInput).label("Align Sharpness", "AlgnSh")
      // Scale of the motion vectors published on render_outputs/motion.
      // Normalized 0..1. In Estimate mode the effective range is 0..128 (the
      // estimated flow is tiny); in Incoming mode the base is ×1 so 1.0 = unit
      // gain (re-publishes the incoming vectors unchanged). Also boosted by
      // `smoothing` (×1 at 0 → ×5 at 1) since heavier smoothing averages the
      // per-frame flow magnitude down.
      .floatField("motion_gain",       0.03f,  0.0f, 1.0f, state::PrimaryInput).label("Motion Gain", "MotGn")
      .intField  ("seed",              0,     0,    1000,  state::PrimaryInput).label("Seed", "Seed")
      // --- Debug (last) ---
      .group("debug", "Debug")
      .boolField ("debug_show_motion", false,              state::PrimaryInput).label("Show Motion", "ShowMo")
      // --- I/O ---
      .textureField("tex_in",  state::PrimaryInput)
      .textureField("tex_out", state::PrimaryOutput)
      .renderOutputs(state::PrimaryOutput)
      .renderOutputs(state::PrimaryInput, "render_outputs_in")
  );

  if (gpu::Device::backend() == gpu::Backend::None) return;

  // Storage-format hints: everything writes RGBA16F (luma lives in the R
  // channel — R32F can't be sampled as Float on WebGPU); color writes tex_out
  // and follows the sketch-default working format.
  state::registerShaderSPV("local_delay_luma",     LUMA_SPV,     LUMA_SPV_SIZE,     "rgba16float", "write");
  state::registerShaderSPV("local_delay_down",     DOWN_SPV,     DOWN_SPV_SIZE,     "rgba16float", "write");
  state::registerShaderSPV("local_delay_lk",       LK_SPV,       LK_SPV_SIZE,       "rgba16float", "write");
  state::registerShaderSPV("local_delay_upsample", UPSAMPLE_SPV, UPSAMPLE_SPV_SIZE, "rgba16float", "write");
  state::registerShaderSPV("local_delay_align",    ALIGN_SPV,    ALIGN_SPV_SIZE,    "rgba16float", "write");
  state::registerShaderSPV("local_delay_color",    COLOR_SPV,    COLOR_SPV_SIZE);
  state::registerShaderSPV("local_delay_motion",   MOTION_SPV,   MOTION_SPV_SIZE,   "rgba16float", "write");

  auto cs_luma     = gpu::Device::createShaderModuleByName("local_delay_luma");
  auto cs_down     = gpu::Device::createShaderModuleByName("local_delay_down");
  auto cs_lk       = gpu::Device::createShaderModuleByName("local_delay_lk");
  auto cs_upsample = gpu::Device::createShaderModuleByName("local_delay_upsample");
  auto cs_align    = gpu::Device::createShaderModuleByName("local_delay_align");
  auto cs_color    = gpu::Device::createShaderModuleByName("local_delay_color");
  auto cs_motion   = gpu::Device::createShaderModuleByName("local_delay_motion");
  if (!cs_luma || !cs_down || !cs_lk || !cs_upsample ||
      !cs_align || !cs_color || !cs_motion) return;

  s_pso_luma = gpu::Device::createComputePSO(cs_luma, "main", gpu::Bindings()
      .tex2d(0)                                       // input
      .storageTex2d(1, gpu::TextureFormat::RGBA16F)); // half-res luma (R channel)

  s_pso_down = gpu::Device::createComputePSO(cs_down, "main", gpu::Bindings()
      .tex2d(0)                                       // finer luma
      .storageTex2d(1, gpu::TextureFormat::RGBA16F)); // coarser luma

  s_pso_lk = gpu::Device::createComputePSO(cs_lk, "main", gpu::Bindings()
      .tex2d(0)                                       // current luma (this level)
      .tex2d(1)                                       // previous luma (this level)
      .tex2d(2)                                       // incoming (coarser) flow / 1x1 zero
      .storageTex2d(3, gpu::TextureFormat::RGBA16F)   // flow out
      .uniform(4));

  s_pso_upsample = gpu::Device::createComputePSO(cs_upsample, "main", gpu::Bindings()
      .tex2d(0)                                       // half-res flow
      .storageTex2d(1, gpu::TextureFormat::RGBA16F)); // full-res flow

  s_pso_align = gpu::Device::createComputePSO(cs_align, "main", gpu::Bindings()
      .tex2d(0)                                       // current LK flow
      .storageTex2d(1, gpu::TextureFormat::RGBA16F)   // smoothed flow + weight
      .tex2d(2)                                       // last frame's smoothed flow
      .uniform(3));

  s_pso_color = gpu::Device::createComputePSO(cs_color, "main", gpu::Bindings()
      .tex2d(0)                                        // original input (advected sample)
      .tex2d(1)                                        // flow (RG flow, B index, A mask)
      .storageTex2d(2)                                 // tex_out
      .uniform(3));

  s_pso_motion = gpu::Device::createComputePSO(cs_motion, "main", gpu::Bindings()
      .tex2d(0)                                       // aligned flow + weight
      .storageTex2d(1, gpu::TextureFormat::RGBA16F)   // motion out
      .tex2d(2)                                       // upstream motion
      .uniform(3));

  state::log("local_delay: module initialized");
}

// Distinct PRNG seed per instance (no wall-clock / RNG primitive needed).
static uint32_t s_seed_counter = 0x9E3779B9u;

void* create() {
  auto* s = new State();
  s_seed_counter = s_seed_counter * 1664525u + 1013904223u;
  s->twitch.seed(s_seed_counter ^ 0xC0FFEEu);
  s->uniform_buf = gpu::Device::createBuffer(sizeof(Uniforms), gpu::BufferUsage::Uniform);
  return s;
}

void destroy(void* self) {
  auto* s = static_cast<State*>(self);
  if (!s) return;
  s->uniform_buf.release();
  for (int i = 0; i < 2; i++)
    for (int l = 0; l < NUM_LEVELS; l++) s->luma[i][l].release();
  s->flow_e.release();
  s->flow_q.release();
  s->flow_h.release();
  s->flow_zero.release();
  s->flow_a.release();
  s->flow_ema[0].release();
  s->flow_ema[1].release();
  s->motion_tex.release();
  s->zero_motion_tex.release();
  delete s;
}

void init(void* self) {
  auto* s = static_cast<State*>(self);
  if (!s) return;
  s->initialized = false;
  s->tex_w = 0;
  s->tex_h = 0;
  s->motion_w = 0;
  s->motion_h = 0;
  s->luma_idx = 0;
  s->flow_idx = 0;
  s->have_history = false;
  s->noise_time = 0.0f;
  s->last_dt = 1.0f / 60.0f;

  if (!s_pso_luma.valid() || !s_pso_down.valid() ||
      !s_pso_lk.valid() || !s_pso_upsample.valid() || !s_pso_align.valid() ||
      !s_pso_color.valid() || !s_pso_motion.valid()) return;
  if (!s->uniform_buf.valid()) return;
  s->initialized = true;
}

void tick(void* self, double dt) {
  auto* s = static_cast<State*>(self);
  if (!s) return;
  if (dt > 0.0) s->last_dt = float(dt);   // drive the frame-rate-independent history EMA
  if (s->noise_motion <= 0.0f) return;
  // Exponential rate mapping (style guide §4.1): noise_motion 0..1 → 0..59 Hz.
  // Each pixel's phase ticks ~rate times/sec, staggered by its time-invariant
  // offset. Wrap to keep float precision of the fractional offset (the wrap is
  // a rare, imperceptible global re-roll).
  float rate = std::pow(60.0f, s->noise_motion) - 1.0f;
  if (rate <= 0.0f) return;
  s->noise_time += float(dt) * rate;
  if (s->noise_time > 65536.0f) s->noise_time -= 65536.0f;
}


void on_state_patched(void* self, int n, const char* pb, const int* off,
                      const int* len, const int* ops) {
  auto* s = static_cast<State*>(self);
  if (!s) return;
  for (int i = 0; i < n; i++) {
    if (ops[i] != state::PatchReplace) continue;
    const char* path = pb + off[i];
    int plen = len[i];
    if      (state::pathIs(path, plen, "delay_amount"))      s->delay_amount = state::patchFloat(i);
    else if (state::pathIs(path, plen, "delay_steps"))       s->delay_steps = state::patchFloat(i);
    else if (state::pathIs(path, plen, "delay_direction"))   s->delay_direction = (int)state::patchFloat(i);
    else if (state::pathIs(path, plen, "flow_source"))       s->flow_source = (int)state::patchFloat(i);
    else if (state::pathIs(path, plen, "smoothing"))         s->smoothing = state::patchFloat(i);
    else if (state::pathIs(path, plen, "noise_weight"))      s->noise_weight = state::patchFloat(i);
    // noise_motion just sets the rate — do NOT reset the noise clock here. The
    // executor replays every state value as a PatchReplace EVERY frame (style
    // guide §8.2), so resetting noise_time on this patch would wipe it every
    // frame and the noise would never advance.
    else if (state::pathIs(path, plen, "noise_motion"))     s->noise_motion = state::patchFloat(i);
    else if (state::pathIs(path, plen, "vignette"))          s->vignette = state::patchFloat(i);
    else if (state::pathIs(path, plen, "vignette_radius"))   s->vignette_radius = state::patchFloat(i);
    else if (state::pathIs(path, plen, "vignette_softness")) s->vignette_softness = state::patchFloat(i);
    else if (state::pathIs(path, plen, "twitch_amount"))       s->twitch_amount = state::patchFloat(i);
    else if (state::pathIs(path, plen, "twitch_shape"))        s->twitch_shape = state::patchFloat(i);
    else if (state::pathIs(path, plen, "twitch_radius"))       s->twitch_radius = state::patchFloat(i);
    else if (state::pathIs(path, plen, "twitch_softness"))     s->twitch_softness = state::patchFloat(i);
    else if (state::pathIs(path, plen, "twitch_position"))     s->twitch_position = state::patchFloat(i);
    else if (state::pathIs(path, plen, "squash"))            s->squash = state::patchFloat(i);
    else if (state::pathIs(path, plen, "max_flow"))          s->max_flow = state::patchFloat(i);
    else if (state::pathIs(path, plen, "weight_gain"))       s->weight_gain = state::patchFloat(i);
    else if (state::pathIs(path, plen, "align_amount"))      s->align_amount = state::patchFloat(i);
    else if (state::pathIs(path, plen, "align_sharpness"))   s->align_sharpness = state::patchFloat(i);
    else if (state::pathIs(path, plen, "motion_gain"))       s->motion_gain = state::patchFloat(i);
    else if (state::pathIs(path, plen, "seed"))              s->seed = (int)state::patchFloat(i);
    else if (state::pathIs(path, plen, "debug_show_motion")) s->debug_show_motion = state::patchFloat(i) != 0.0f;
  }
}

static inline int half_up(int x) { return (x + 1) / 2; }

// (Re)allocate the working set when the viewport changes. The luma pyramid
// is rebuilt every frame (LK guards the first frame), so it needs no clear;
// history is cleared so the first cross-fade never reads NaN, and the 1x1
// zero flow is cleared once (it's the coarsest level's incoming flow).
static bool ensure_textures(State* s, int vp_w, int vp_h) {
  if (s->tex_w == vp_w && s->tex_h == vp_h &&
      s->flow_a.valid() && s->flow_ema[0].valid() && s->flow_ema[1].valid() &&
      s->flow_zero.valid())
    return true;

  for (int i = 0; i < 2; i++)
    for (int l = 0; l < NUM_LEVELS; l++) s->luma[i][l].release();
  s->flow_e.release();
  s->flow_q.release();
  s->flow_h.release();
  s->flow_a.release();
  s->flow_ema[0].release();
  s->flow_ema[1].release();

  int hw = half_up(vp_w),  hh = half_up(vp_h);   // half
  int qw = half_up(hw),    qh = half_up(hh);     // quarter
  int ew = half_up(qw),    eh = half_up(qh);     // eighth
  int dimsW[NUM_LEVELS] = { hw, qw, ew };
  int dimsH[NUM_LEVELS] = { hh, qh, eh };

  for (int i = 0; i < 2; i++)
    for (int l = 0; l < NUM_LEVELS; l++)
      s->luma[i][l] = gpu::Device::createTexture(dimsW[l], dimsH[l], gpu::TextureFormat::RGBA16F);
  s->flow_h  = gpu::Device::createTexture(hw, hh, gpu::TextureFormat::RGBA16F);
  s->flow_q  = gpu::Device::createTexture(qw, qh, gpu::TextureFormat::RGBA16F);
  s->flow_e  = gpu::Device::createTexture(ew, eh, gpu::TextureFormat::RGBA16F);
  s->flow_a     = gpu::Device::createTexture(vp_w, vp_h, gpu::TextureFormat::RGBA16F);
  s->flow_ema[0] = gpu::Device::createTexture(vp_w, vp_h, gpu::TextureFormat::RGBA16F);
  s->flow_ema[1] = gpu::Device::createTexture(vp_w, vp_h, gpu::TextureFormat::RGBA16F);
  if (!s->flow_zero.valid())
    s->flow_zero = gpu::Device::createTexture(1, 1, gpu::TextureFormat::RGBA16F);

  if (!s->flow_a.valid() || !s->flow_ema[0].valid() || !s->flow_ema[1].valid() ||
      !s->flow_zero.valid() ||
      !s->flow_h.valid() || !s->flow_q.valid() || !s->flow_e.valid())
    return false;

  gpu::Device::clear(s->flow_ema[0], 0.0f, 0.0f, 0.0f, 0.0f);
  gpu::Device::clear(s->flow_ema[1], 0.0f, 0.0f, 0.0f, 0.0f);
  gpu::Device::clear(s->flow_zero, 0.0f, 0.0f, 0.0f, 0.0f);

  s->tex_w = vp_w;
  s->tex_h = vp_h;
  s->luma_idx = 0;
  s->flow_idx = 0;
  s->have_history = false;   // re-prime after a resize
  return true;
}

static inline void dispatch2(const gpu::ComputePSO& pso, int w, int h,
                             const gpu::Texture* t0, int a0,
                             const gpu::Texture* t1, int a1,
                             const gpu::Texture* t2, int a2,
                             const gpu::Texture* t3, int a3,
                             const gpu::Buffer* ub, int ubslot) {
  auto cp = gpu::ComputePass::begin();
  cp.setPSO(pso);
  if (t0) cp.setTexture(*t0, 0, a0);
  if (t1) cp.setTexture(*t1, 1, a1);
  if (t2) cp.setTexture(*t2, 2, a2);
  if (t3) cp.setTexture(*t3, 3, a3);
  if (ub) cp.setBuffer(*ub, ubslot);
  cp.dispatch((w + 7) / 8, (h + 7) / 8);
  cp.end();
}

void render(void* self, int vp_w, int vp_h) {
  auto* s = static_cast<State*>(self);
  if (!s || !s->initialized || vp_w <= 0 || vp_h <= 0) return;
  auto in  = gpu::Device::textureForField("tex_in");
  auto out = gpu::Device::textureForField("tex_out");
  if (!in.valid() || !out.valid()) return;
  if (!ensure_textures(s, vp_w, vp_h)) return;

  auto cs = fx::coverSquare(vp_w, vp_h);
  // Time-INVARIANT seed (the per-pixel staggered phase comes from noise_time
  // in the shader, not from baking a step counter into the seed here).
  float noise_seed = float(s->seed * 17);

  // Frame-rate-independent history EMA rate. First frame (no valid history)
  // → alpha 1 so the EMA seeds to the input.
  float half_life = s->smoothing * s->smoothing * 1.5f;
  float history_alpha = 1.0f;
  if (s->have_history && half_life > 1e-4f)
    history_alpha = 1.0f - std::pow(0.5f, s->last_dt / half_life);

  // Map the normalized [0,1] knobs to their effective ranges (the shaders see
  // the effective values). weight_gain → ×2048 (typical flow is tiny so it
  // needs a large multiplier); motion_gain → ×128.
  float eff_weight_gain = s->weight_gain * 2048.0f;

  // Compensate the published motion magnitude for the flow EMA: higher
  // `smoothing` averages the per-frame flow down (the spike-train flow at
  // render-fps > source-fps gets spread over time), so scale motion_gain up
  // to keep the exported velocity usable. Low smoothing = spiky but already
  // strong → ~no boost; smoothing=1 → 5x. Done here so the shader is unchanged.
  // The base multiplier is mode-dependent: Estimate flow is tiny (uv/frame) so
  // it needs ×128; Incoming vectors are already real-scale, so ×1 → motion_gain
  // = 1.0 is unit gain (re-publishes the incoming vectors unchanged).
  float sm = std::fmin(std::fmax(s->smoothing, 0.0f), 1.0f);
  float gain_base = (s->flow_source == 0) ? 128.0f : 1.0f;
  float eff_motion_gain = (s->motion_gain * gain_base) / (1.0f - 0.8f * sm);

  // Twitch: pick a NEW random anchor + intensity every frame (frame-rate-
  // dependent for now). `twitch_position` biases the spawn radius: +1 → an oval
  // near the centre, -1 → an oval ring around the outside. The whole spawn range
  // scales out with `twitch_radius`, so a big twitch roams further — at max
  // radius + position -1 the anchor is usually off-screen (its falloff still
  // creeps in from the edges). Anchor is in cover-square coords (isotropic there
  // → an oval in the viewport via the aspect).
  auto tw = s->twitch.update({ s->twitch_amount, s->twitch_shape, s->twitch_radius,
                               s->twitch_softness, s->twitch_position });
  float twitch_ax = tw.anchor_x, twitch_ay = tw.anchor_y, twitch_strength = tw.strength;

  Uniforms u = {
    s->delay_amount, s->noise_weight, noise_seed, eff_weight_gain,
    s->vignette, s->vignette_radius, s->vignette_softness, s->squash,
    s->max_flow, s->align_amount, s->align_sharpness, s->have_history ? 1.0f : 0.0f,
    cs.ax, cs.ay, s->debug_show_motion ? 1.0f : 0.0f, history_alpha,
    eff_motion_gain, s->delay_steps, s->noise_time, s->delay_direction == 0 ? 1.0f : -1.0f,
    s->twitch_shape, s->twitch_radius, s->twitch_softness, twitch_strength,
    twitch_ax, twitch_ay, 0.0f, 0.0f,
  };
  s->uniform_buf.writeOne(u);

  const int cc = s->luma_idx;       // build current luma into this set
  const int pc = 1 - cc;            // previous frame's luma
  const int fp = s->flow_idx;       // last frame's smoothed flow (read by align)
  const int fc = 1 - fp;            // this frame's smoothed flow (write target)

  // The flow that feeds align/advect: either our LK estimate or the incoming
  // render_outputs motion vectors (which skip the estimator entirely).
  gpu::Texture flow_src;
  if (s->flow_source == 0) {
    // ----- Estimate flow with pyramidal Lucas-Kanade -----
    int hw = half_up(vp_w), hh = half_up(vp_h);
    int qw = half_up(hw),   qh = half_up(hh);
    int ew = half_up(qw),   eh = half_up(qh);

    // 1 — luma: input → half-res luma.
    dispatch2(s_pso_luma, hw, hh, &in, 0, &s->luma[cc][0], 1,
              nullptr, 0, nullptr, 0, nullptr, 0);
    // 2 — downsample: half → quarter → eighth.
    dispatch2(s_pso_down, qw, qh, &s->luma[cc][0], 0, &s->luma[cc][1], 1,
              nullptr, 0, nullptr, 0, nullptr, 0);
    dispatch2(s_pso_down, ew, eh, &s->luma[cc][1], 0, &s->luma[cc][2], 1,
              nullptr, 0, nullptr, 0, nullptr, 0);
    // 3 — coarse-to-fine Lucas-Kanade (incoming flow per level; 1x1 zero for
    // the coarsest), warping the previous frame's luma.
    dispatch2(s_pso_lk, ew, eh, &s->luma[cc][2], 0, &s->luma[pc][2], 0,
              &s->flow_zero, 0, &s->flow_e, 1, &s->uniform_buf, 4);
    dispatch2(s_pso_lk, qw, qh, &s->luma[cc][1], 0, &s->luma[pc][1], 0,
              &s->flow_e, 0, &s->flow_q, 1, &s->uniform_buf, 4);
    dispatch2(s_pso_lk, hw, hh, &s->luma[cc][0], 0, &s->luma[pc][0], 0,
              &s->flow_q, 0, &s->flow_h, 1, &s->uniform_buf, 4);
    // 4 — upsample finest LK flow (half) → full res.
    dispatch2(s_pso_upsample, vp_w, vp_h, &s->flow_h, 0, &s->flow_a, 1,
              nullptr, 0, nullptr, 0, nullptr, 0);
    flow_src = s->flow_a;
  } else {
    // ----- Use the incoming render_outputs motion vectors as the flow -----
    flow_src = gpu::Device::textureForField("render_outputs_in/motion");
    if (!flow_src.valid()) {
      // No upstream wired → 1x1 zero (cleared) so the effect is a passthrough.
      if (!s->zero_motion_tex.valid()) {
        s->zero_motion_tex = gpu::Device::createTexture(1, 1, gpu::TextureFormat::RGBA16F);
        gpu::Device::clear(s->zero_motion_tex, 0.0f, 0.0f, 0.0f, 0.0f);
      }
      flow_src = s->zero_motion_tex;
    }
  }

  // 5 — align: colinear polish + temporal flow EMA + mask + index.
  dispatch2(s_pso_align, vp_w, vp_h, &flow_src, 0, &s->flow_ema[fc], 1,
            &s->flow_ema[fp], 0, nullptr, 0, &s->uniform_buf, 3);

  // 6 — color: forward-advect along the smoothed flow, sample the original
  // input at the endpoint (solid pixels, no smeary history buffers). At
  // delay_amount = 0 the shader early-outs to a passthrough copy of the input
  // (flow-conditioner mode): the image is untouched while align/mask/motion
  // still run and publish the conditioned vectors. We keep the dispatch rather
  // than a host-side copy because the web executor's intermediate textures are
  // allocated COPY_SRC-only (no COPY_DST), so copyTextureToTexture into tex_out
  // is invalid there; the early-out skips the expensive advection loop, which is
  // the actual cost — the dispatch launch itself is negligible.
  dispatch2(s_pso_color, vp_w, vp_h, &in, 0, &s->flow_ema[fc], 0,
            &out, 1, nullptr, 0, &s->uniform_buf, 3);

  // 8 — motion: only when something downstream reads the rail.
  if (state::isOutputConnected("render_outputs")) {
    if (!s->motion_tex.valid() || s->motion_w != vp_w || s->motion_h != vp_h) {
      s->motion_tex = gpu::Device::createTexture(vp_w, vp_h, gpu::TextureFormat::RGBA16F);
      s->motion_w = vp_w;
      s->motion_h = vp_h;
      if (s->motion_tex.valid())
        state::setGpuTexture("render_outputs/motion", s->motion_tex.id);
    }
    if (s->motion_tex.valid()) {
      if (s->motion_gain <= 0.0f) {
        // motion_gain 0 → publish nothing. Skip the dispatch entirely and clear
        // the rail to zero (a clear is free vs. a full-frame compute pass) — no
        // contribution, no upstream passthrough. This also makes the published
        // magnitude continuous as motion_gain → 0 (the vectors fade to zero
        // rather than snapping back to the unscaled upstream field).
        gpu::Device::clear(s->motion_tex, 0.0f, 0.0f, 0.0f, 0.0f);
      } else {
        auto upstream = gpu::Device::textureForField("render_outputs_in/motion");
        if (!upstream.valid()) {
          if (!s->zero_motion_tex.valid())
            s->zero_motion_tex = gpu::Device::createTexture(1, 1, gpu::TextureFormat::RGBA16F);
          upstream = s->zero_motion_tex;
        }
        dispatch2(s_pso_motion, vp_w, vp_h, &s->flow_ema[fc], 0, &s->motion_tex, 1,
                  &upstream, 0, nullptr, 0, &s->uniform_buf, 3);
      }
    }
  }

  // Advance ping-pong state; this frame's luma and smoothed flow are now the
  // "previous".
  s->luma_idx = pc;
  s->flow_idx = fc;
  s->have_history = true;
  gpu::Device::submit();
}

} // namespace local_delay
