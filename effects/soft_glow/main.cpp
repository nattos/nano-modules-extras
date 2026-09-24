/*
 * source.light.soft_glow — Continuous warm-blob atmosphere bed (v1 scaffold).
 *
 * A pool of soft Gaussian blobs drifting at constant velocity (toroidal
 * wrap). Per pixel sums blob contributions, looks up a hue-shifting
 * ramp, additively blends over input.
 *
 * Class-like instance model: module_init() compiles the two shared
 * compute PSOs + publishes the schema once per type; each chain entry
 * gets its own State (params, blob pool, per-instance buffers/textures)
 * via create(). All instance callbacks take `self`.
 */

#include <effect_utils.h>
#include <gpu.h>
#include <host.h>
#include "soft_glow_shaders.h"

#include <cmath>
#include <cstdint>
#include <cstring>

namespace soft_glow {

static constexpr int   MAX_BLOBS = 32;
static constexpr float TAU       = 6.28318530717958647692f;

struct GpuBlob {
  // .xyz = (x, y, radius); .w = per-blob amplitude (1.0 = nominal).
  // Both color and motion shaders multiply their per-blob contribution
  // by this so a dimmed blob also drives less motion (visible footprint
  // ↔ motion footprint stay coupled).
  float pos_size[4];
  // .x = hue_offset (color shader only); .yz = (vx, vy) — velocity
  // in canvas-uv / sec, consumed by the motion shader; .w = pad.
  float jitters[4];
};
static_assert(sizeof(GpuBlob) == 32, "Blob layout mismatch");

struct Uniforms {
  uint32_t blob_count;
  float    intensity;
  float    ramp_curve;
  float    white_point;

  float    hue;            // hue at max amplitude
  float    hue_shift;      // hue offset added as amp → 0; |shift|>1 bands
  float    saturation;
  float    aspect_x;

  float    aspect_y;
  float    intensity_skew; // 0=isotropic, 1=wavefront-only contribution
  float    hue_curve;      // power on (1-ramp_t) — shapes where hue transition lives
  float    overflow_band;  // 0=soft-clip hue at peak; >0 keeps rotating → banding

  float    color_strength; // final multiplier on emitted glow rgb (no effect on motion)
  float    _pad_a;
  float    _pad_b;
  float    _pad_c;
};
static_assert(sizeof(Uniforms) == 64, "Uniforms layout mismatch");

struct MotionUniforms {
  uint32_t blob_count;
  float    motion_strength;
  float    motion_skew;
  float    aspect_x;

  float    aspect_y;
  float    motion_curl;     // signed [-1,+1] → rotation by curl·π radians
  float    intensity;       // scales coverage mask AND velocity magnitude
  float    motion_extent;   // 1 = full blob footprint; <1 shrinks toward centers
};
static_assert(sizeof(MotionUniforms) == 32, "MotionUniforms layout mismatch");

struct BlobState {
  // Position is derived each tick from the elliptical orbit:
  //   x = cx + wobble_rx * cos(wobble_phase)
  //   y = cy + wobble_ry * sin(wobble_phase)
  // rx > ry makes the orbit elliptical → tangential velocity is
  // minimum near the long-axis apexes, so the blob naturally
  // "weights" the tightest curvature points.
  float cx, cy;          // wobble center; drifts with drift_x/y_bias
  float wobble_rx;       // ellipse semi-axis x (cover-square units)
  float wobble_ry;       // ellipse semi-axis y; rx > ry → corners
  float wobble_phase;    // orbit phase accumulator
  float wobble_omega;    // per-blob angular-rate jitter in [0.5, 1.5]

  float x, y;            // derived this tick — read by render
  float vx, vy;          // instantaneous velocity (orbit + bias) for motion

  float size_jit;        // signed [-1,+1] — radius is recomputed each tick
                         // from (blob_size, blob_size_jitter, size_jit)
                         // so size changes don't need a reseed
  float hue_offset;
  // Two-octave amplitude oscillator. phase_a fast (breathing); phase_b
  // slow (drift envelope). freq_jit decorrelates blob-to-blob.
  float phase_a, phase_b;
  float freq_jit;
  float amp;

  // Fade state — current_alive smoothly tracks target_alive at rate
  // 1/fade_time. amp is multiplied by current_alive at pack time, so
  // a slot's full contribution to BOTH color and motion ramps with
  // its visibility. target_alive is derived each tick from
  // (i < blob_count) AND the `wants_respawn` override.
  float target_alive;    // 0 or 1
  float current_alive;   // [0, 1]
  // Set when the orbit has drifted fully off-screen — forces target to
  // 0 until current_alive reaches 0, then triggers respawn at a fresh
  // location. Lets bias-driven drift wander off-screen cleanly via the
  // fade machinery (no wrap teleport).
  bool  wants_respawn;
};

// Per-instance state. One per chain entry.
struct State {
  gpu::Buffer  uniform_buf;
  gpu::Buffer  motion_uniform_buf;
  gpu::Buffer  blob_buf;
  gpu::Texture motion_tex;
  gpu::Texture zero_motion_tex;   // 1x1 rgba16f fallback (no upstream)
  int          motion_w = 0;
  int          motion_h = 0;
  BlobState    blobs[MAX_BLOBS];
  bool         initialized = false;

  // Schema-mirrored params
  int   blob_count        = 12;
  float blob_size         = 0.4f;
  float blob_size_jitter  = 0.3f;
  float drift_rate        = 0.2f;
  float drift_x_bias      = 0.0f;
  float drift_y_bias      = 0.0f;
  float hue               = 0.05f;  // hue at peak amplitude
  float hue_shift         = 0.30f;  // amount added as amp → 0
  float hue_curve         = 0.0f;   // signed [-1..+1] → exp via signedSliderToExp(2.0)
  float saturation        = 0.95f;
  float intensity_skew    = 0.0f;
  float overflow_band     = 0.0f;  // 0 = soft-clip hue at peak; >0 → banding
  float color_strength    = 1.0f;  // final multiplier on emitted glow (color only)
  float ramp_curve        = 0.0f;   // signed [-1..+1] → exp via signedSliderToExp(2.0)
  float white_point       = 1.5f;
  float intensity         = 1.0f;
  float intensity_mod     = 0.0f;
  float motion_strength   = 1.0f;  // boost — blobs drift slowly
  float motion_skew       = 0.0f;  // 0=isotropic, 1=wavefront-only
  float motion_curl       = 0.0f;  // signed [-1,+1] → rotate local motion by curl·π
  float motion_extent     = 1.0f;  // 1→0: shrink emitted motion footprint toward centers
  float pulse_depth       = 0.4f;
  float pulse_rate        = 0.6f;   // Hz, fast breathing rate
  float amp_drift_depth   = 0.9f;   // depth of slow envelope (multiplicative)
  float amp_drift_rate    = 0.08f;  // Hz, slow drift rate (~12s period)
  float fade_time         = 3.0f;   // sec — blob_count fade in/out
  uint32_t seed           = 0xA17F2B91u;

  // RNG stream used for *respawn* of revived slots. Advances on each
  // respawn so toggling count up and down yields fresh positions on
  // every spawn (rather than re-drawing the deterministic seed_all
  // sequence). Reset from seed whenever the user changes seed.
  uint32_t spawn_rng      = 0xDEADBEEFu;

  // Set once after init()'s seed_all finishes.
  bool blobs_seeded = false;
};

// Type-shared: compiled once in module_init().
static gpu::ComputePSO s_pso_color;
static gpu::ComputePSO s_pso_motion;

static uint32_t lcg_next(uint32_t& s) {
  s = s * 1664525u + 1013904223u;
  return s;
}
static float lcg_unit(uint32_t& s) {
  return (lcg_next(s) >> 8) * (1.0f / float(1u << 24));
}
static float lcg_signed(uint32_t& s) { return lcg_unit(s) * 2.0f - 1.0f; }

static void seed_blob(State& st, int i, uint32_t& rng) {
  BlobState& b = st.blobs[i];
  // Wobble center anywhere in uv-space.
  b.cx = lcg_unit(rng);
  b.cy = lcg_unit(rng);
  // Ellipse semi-axes. rx is the long axis; ry is shorter so the orbit
  // is definitively elliptical → tangential speed slows near the
  // long-axis apexes ("weight at the tightest corners").
  b.wobble_rx = 0.10f + lcg_unit(rng) * 0.25f;          // [0.10, 0.35]
  b.wobble_ry = b.wobble_rx * (0.25f + lcg_unit(rng) * 0.45f); // [0.25, 0.70] of rx
  // Rotate the ellipse to a random orientation by phase-shifting:
  // start phase = random ∈ [0, TAU).
  b.wobble_phase = lcg_unit(rng) * TAU;
  b.wobble_omega = 0.5f + lcg_unit(rng);                 // [0.5, 1.5]
  // Initial position derived from wobble.
  b.x = b.cx + b.wobble_rx * std::cos(b.wobble_phase);
  b.y = b.cy + b.wobble_ry * std::sin(b.wobble_phase);
  b.vx = 0.0f;
  b.vy = 0.0f;
  b.size_jit   = lcg_signed(rng);
  b.hue_offset = lcg_signed(rng) * 0.05f;  // small per-blob hue jitter
  // Amplitude oscillator seeds. Phases get a full TAU spread so blobs
  // start out of sync; freq_jit decorrelates pulse rates blob-to-blob
  // so the bed doesn't pump as one.
  b.phase_a  = lcg_unit(rng) * TAU;
  b.phase_b  = lcg_unit(rng) * TAU;
  b.freq_jit = 0.5f + lcg_unit(rng);            // [0.5, 1.5]
  b.amp      = 1.0f;
  b.wants_respawn = false;
}

static void respawn_blob(State& st, int i) {
  seed_blob(st, i, st.spawn_rng);
  BlobState& b = st.blobs[i];
  // If a drift bias is set, respawn upwind (just off the trailing
  // edge) so blobs sweep across the viewport like wind-blown clouds
  // instead of clustering at the leading edge. With no bias, the
  // seed_blob random position in [0, 1) stands.
  if (st.drift_x_bias > 0.0f) {
    b.cx = -b.wobble_rx - 0.05f - lcg_unit(st.spawn_rng) * 0.15f;
  } else if (st.drift_x_bias < 0.0f) {
    b.cx = 1.0f + b.wobble_rx + 0.05f + lcg_unit(st.spawn_rng) * 0.15f;
  }
  if (st.drift_y_bias > 0.0f) {
    b.cy = -b.wobble_ry - 0.05f - lcg_unit(st.spawn_rng) * 0.15f;
  } else if (st.drift_y_bias < 0.0f) {
    b.cy = 1.0f + b.wobble_ry + 0.05f + lcg_unit(st.spawn_rng) * 0.15f;
  }
  // Recompute initial position from (possibly relocated) center.
  b.x = b.cx + b.wobble_rx * std::cos(b.wobble_phase);
  b.y = b.cy + b.wobble_ry * std::sin(b.wobble_phase);
  // New blobs fade IN from invisible — current_alive will rise to
  // target_alive at fade rate in tick().
  b.target_alive  = 1.0f;
  b.current_alive = 0.0f;
}

static void seed_all(State& st) {
  uint32_t rng = st.seed;
  for (int i = 0; i < MAX_BLOBS; i++) {
    seed_blob(st, i, rng);
    bool alive = (i < st.blob_count);
    // Initial state: blobs within count start fully alive (no fade-in
    // on first frame), blobs beyond count start fully dead.
    st.blobs[i].target_alive  = alive ? 1.0f : 0.0f;
    st.blobs[i].current_alive = alive ? 1.0f : 0.0f;
  }
  // Continue the spawn-rng stream from wherever seed_all left off, so
  // subsequent respawns get fresh randomness rather than re-drawing
  // the initial sequence.
  st.spawn_rng = rng;
  st.blobs_seeded = true;
}

// Type-level setup: schema + the two shared compute PSOs.
void module_init() {
  state::init("source.light.soft_glow", {1, 0, 1},
    state::Schema()
      // Top-level manual: high-level "what is this / how to use / what to try".
      .helpField("intro",
        "## Soft Glow\n"
        "Lays a bed of soft, drifting Gaussian blobs over the image — a warm "
        "atmospheric glow that breathes and wanders. Each blob pulses in brightness "
        "and shifts hue as it fades, then respawns to keep the field alive.\n\n"
        "**Try:** raise *Blob Count* and *Blob Size* for a dense wash; set a *Drift X/Y* "
        "bias to make the bed sweep across the frame like wind-blown cloud; push *Hue "
        "Shift* and *Overflow Band* for rolling colour bands. Wire the motion output "
        "downstream to let the glow drive optical-flow effects.")
      // --- Output ---
      .group("output", "Output")
      .floatField("intensity",        1.0f, 0.0f, 2.0f, state::PrimaryInput).label("Intensity", "Int")
      .floatField("intensity_mod",    0.0f, -1.0f, 1.0f, state::PrimaryInput).label("Intensity Mod", "Mod")
      // --- Blobs ---
      .group("blobs", "Blobs")
        .groupHelp(
          "The glow is a pool of soft blobs orbiting on their own little ellipses. "
          "*Count* sets how many are alive at once; *Size* and *Size Jitter* set their "
          "footprint and how much it varies blob-to-blob. *Fade Time* smooths how "
          "gently blobs appear and disappear when the count changes.")
      .intField  ("blob_count",       12,   0,    MAX_BLOBS, state::PrimaryInput).label("Blob Count", "Count")
      .floatField("fade_time",        3.0f, 0.05f, 10.0f, state::PrimaryInput).label("Fade Time", "Fade")
      .floatField("blob_size",        0.4f, 0.05f, 1.0f, state::PrimaryInput).label("Blob Size", "Size")
      .floatField("blob_size_jitter", 0.3f, 0.0f, 1.0f, state::PrimaryInput).label("Size Jitter", "SzJit")
      // --- Drift ---
      .group("drift", "Drift")
        .groupHelp(
          "How the whole bed wanders. *Rate* sets the orbital speed; the *X/Y Bias* "
          "knobs push a steady wind so blobs sweep across the frame and respawn "
          "upwind. Leave the biases at 0 to let the field breathe in place.")
      .floatField("drift_rate",       0.2f, 0.0f, 1.0f, state::PrimaryInput).label("Drift Rate", "Rate")
      .floatField("drift_x_bias",     0.0f, -1.0f, 1.0f, state::PrimaryInput).label("Drift X", "X")
      .floatField("drift_y_bias",     0.0f, -1.0f, 1.0f, state::PrimaryInput).label("Drift Y", "Y")
      // --- Colour ---
      .group("color", "Colour")
        .groupHelp(
          "Blobs are tinted from a hue ramp keyed to their brightness. *Hue* is the "
          "colour at full amplitude; *Hue Shift* rotates it as a blob fades, and "
          "*Overflow Band* keeps that rotation going past the peak for hard colour "
          "banding. *Saturation*, *White Point* and the two curves shape richness "
          "and how the tone rolls off.")
      .floatField("hue",              0.05f, 0.0f, 1.0f,  state::PrimaryInput).label("Hue", "Hue")
      .floatField("hue_shift",        0.30f, -3.0f, 3.0f, state::PrimaryInput).label("Hue Shift", "Shift")
      .floatField("hue_curve",        0.0f,  -1.0f, 1.0f, state::PrimaryInput).label("Hue Curve", "HCurve")
      .floatField("overflow_band",    0.0f,  0.0f,  3.0f, state::PrimaryInput).label("Overflow Band", "Band")
      .floatField("color_strength",   1.0f,  0.0f,  4.0f, state::PrimaryInput).label("Colour Strength", "Str")
      .floatField("saturation",       0.95f, 0.0f, 1.0f,  state::PrimaryInput).label("Saturation", "Sat")
      .floatField("intensity_skew",   0.0f,  0.0f, 1.0f,  state::PrimaryInput).label("Intensity Skew", "Skew")
      .floatField("ramp_curve",       0.0f, -1.0f, 1.0f, state::PrimaryInput).label("Ramp Curve", "Ramp")
      .floatField("white_point",      1.5f, 0.5f, 3.0f, state::PrimaryInput).label("White Point", "White")
      // --- Pulse & Breathing ---
      .group("pulse", "Pulse & Breathing")
      .floatField("pulse_depth",      0.4f, 0.0f, 1.0f, state::PrimaryInput).label("Pulse Depth", "Depth")
      .floatField("pulse_rate",       0.6f, 0.0f, 3.0f, state::PrimaryInput).label("Pulse Rate", "Rate")
      .floatField("amp_drift_depth",  0.9f,  0.0f, 1.0f, state::PrimaryInput).label("Drift Depth", "DrDep")
      .floatField("amp_drift_rate",   0.08f, 0.0f, 1.0f, state::PrimaryInput).label("Drift Rate", "DrRat")
      // --- Motion Output ---
      .group("motion", "Motion Output")
        .groupHelp(
          "Emits an optical-flow field matching the blobs' movement, for downstream "
          "motion-driven effects (it doesn't change the glow itself). *Strength* "
          "scales the emitted velocity; *Skew* biases it toward the leading edge; "
          "*Curl* rotates it for a swirling feel; *Extent* shrinks the footprint "
          "toward blob centres.")
      .floatField("motion_strength",  1.0f,  0.0f, 8.0f,  state::PrimaryInput).label("Motion Strength", "Str")
      .floatField("motion_skew",      0.0f,  0.0f, 1.0f,  state::PrimaryInput).label("Motion Skew", "Skew")
      .floatField("motion_curl",      0.0f, -1.0f, 1.0f,  state::PrimaryInput).label("Motion Curl", "Curl")
      .floatField("motion_extent",    1.0f,  0.0f, 1.0f,  state::PrimaryInput).label("Motion Extent", "Ext")
      // --- Seed ---
      .group("seed", "Seed")
      .intField  ("seed",             0,    0,    65535, state::PrimaryInput).label("Seed", "Seed")
      .textureField("tex_in",  state::PrimaryInput)
      .textureField("tex_out", state::PrimaryOutput)
      .renderOutputs(state::PrimaryOutput)
      // Upstream auxiliary inputs (e.g. motion) — when wired the motion
      // pass lerps our local contribution on top of the upstream field.
      .renderOutputs(state::PrimaryInput, "render_outputs_in")
        .capability(state::Capability::Generator)
    );

  if (gpu::Device::backend() == gpu::Backend::None) return;

  state::registerShaderSPV("soft_glow_color", COLOR_SPV, COLOR_SPV_SIZE);
  state::registerShaderSPV("soft_glow_motion", MOTION_SPV, MOTION_SPV_SIZE,
                           "rgba16float", "write");
  auto cs_color  = gpu::Device::createShaderModuleByName("soft_glow_color");
  auto cs_motion = gpu::Device::createShaderModuleByName("soft_glow_motion");
  if (!cs_color || !cs_motion) return;

  s_pso_color = gpu::Device::createComputePSO(cs_color, "main", gpu::Bindings()
      .storage(0)
      .tex2d(1)
      .storageTex2d(2)
      .uniform(3));

  s_pso_motion = gpu::Device::createComputePSO(cs_motion, "main", gpu::Bindings()
      .storage(0)                                       // blobs (with velocity)
      .tex2d(1)                                         // upstream motion
      .storageTex2d(2, gpu::TextureFormat::RGBA16F)     // motionTex
      .uniform(3));                                     // MotionUniforms

  state::log("soft_glow: module initialized");
}

// Per-instance construction: allocate State + its own GPU buffers.
void* create() {
  auto* st = new State();
  st->blob_buf = gpu::Device::createBuffer(
      sizeof(GpuBlob) * MAX_BLOBS, gpu::BufferUsage::Storage);
  st->uniform_buf = gpu::Device::createBuffer(sizeof(Uniforms), gpu::BufferUsage::Uniform);
  st->motion_uniform_buf = gpu::Device::createBuffer(
      sizeof(MotionUniforms), gpu::BufferUsage::Uniform);
  return st;
}

void destroy(void* self) {
  auto* st = static_cast<State*>(self);
  if (!st) return;
  st->blob_buf.release();
  st->uniform_buf.release();
  st->motion_uniform_buf.release();
  st->motion_tex.release();
  st->zero_motion_tex.release();
  delete st;
}

// Per-instance init tail: mark ready + seed the blob pool.
void init(void* self) {
  auto* st = static_cast<State*>(self);
  if (!st) return;
  if (!s_pso_color.valid() || !s_pso_motion.valid()) return;
  if (!st->blob_buf.valid() || !st->uniform_buf.valid() ||
      !st->motion_uniform_buf.valid()) return;
  st->initialized = true;
  seed_all(*st);
}

void tick(void* self, double dt) {
  auto* st = static_cast<State*>(self);
  if (!st || !st->initialized || !st->blobs_seeded) return;
  float fdt = (float)dt;

  // Per-second fade rate (current_alive moves at this rate toward
  // target_alive). fade_time → 0 means instantaneous.
  float fade_t = st->fade_time > 0.001f ? st->fade_time : 0.001f;
  float fade_rate = 1.0f / fade_t;

  float omega_fast = st->pulse_rate * TAU;
  float omega_slow = st->amp_drift_rate * TAU;

  for (int i = 0; i < MAX_BLOBS; i++) {
    BlobState& b = st->blobs[i];

    // ---- Fade machine ----
    // Priority order: wants_respawn (off-screen drift) → count-derived
    // target. wants_respawn forces fade-out; once current_alive hits
    // 0 we respawn at a fresh location and clear the flag.
    float count_target = (i < st->blob_count) ? 1.0f : 0.0f;
    if (b.wants_respawn) {
      if (b.current_alive == 0.0f) {
        respawn_blob(*st, i);
        b.wants_respawn = false;
        // respawn_blob set target=1, current=0 — fade-in proceeds.
      } else {
        b.target_alive = 0.0f;
      }
    } else {
      bool fresh_spawn = (count_target > 0.0f
                          && b.target_alive == 0.0f
                          && b.current_alive == 0.0f);
      if (fresh_spawn) {
        respawn_blob(*st, i);
      } else {
        b.target_alive = count_target;
      }
    }

    // Advance current_alive toward target_alive at fade_rate, clamped.
    float diff = b.target_alive - b.current_alive;
    if (diff != 0.0f) {
      float step = (diff > 0.0f ? fade_rate : -fade_rate) * fdt;
      if ((step >= 0.0f && step >= diff) ||
          (step <  0.0f && step <= diff)) {
        b.current_alive = b.target_alive;
      } else {
        b.current_alive += step;
      }
    }

    // ---- Orbit: position derived from elliptical parameterization.
    // drift_x/y_bias drifts the orbit *center*; the blob still wobbles
    // around it. No wrap — a wrap on cx/cy would teleport the visible
    // position by ~1 unit (popping). Instead, we detect "fully off-
    // screen" below and route the blob through the fade-out + respawn
    // path, which is invisible to the viewer since the blob is already
    // off-screen.
    b.cx += st->drift_x_bias * fdt;
    b.cy += st->drift_y_bias * fdt;

    float omega_eff = st->drift_rate * b.wobble_omega;
    b.wobble_phase += omega_eff * fdt;

    float c = std::cos(b.wobble_phase);
    float sn = std::sin(b.wobble_phase);
    b.x = b.cx + b.wobble_rx * c;
    b.y = b.cy + b.wobble_ry * sn;
    // Instantaneous orbital velocity (analytic derivative of position).
    // With rx ≠ ry, |v| varies around the orbit and is smallest near
    // the long-axis apexes — exactly the "weight at corners" feel.
    b.vx = -b.wobble_rx * sn * omega_eff + st->drift_x_bias;
    b.vy =  b.wobble_ry * c  * omega_eff + st->drift_y_bias;

    // Per-blob off-screen check: only check the *leading* edge along
    // each axis (the edge the drift bias is carrying the blob toward).
    // Checking the trailing edge would immediately re-flag the
    // freshly-respawned upwind blob for another fade-out — it would
    // never get to actually enter the viewport. With bias=0, both
    // edges are checked; without drift the blob never reaches an
    // edge anyway (initial cx ∈ [0, 1)).
    if (!b.wants_respawn) {
      bool oob = false;
      if (st->drift_x_bias >= 0.0f && b.cx - b.wobble_rx > 1.0f) oob = true;
      if (st->drift_x_bias <= 0.0f && b.cx + b.wobble_rx < 0.0f) oob = true;
      if (st->drift_y_bias >= 0.0f && b.cy - b.wobble_ry > 1.0f) oob = true;
      if (st->drift_y_bias <= 0.0f && b.cy + b.wobble_ry < 0.0f) oob = true;
      if (oob) b.wants_respawn = true;
    }

    // ---- Amplitude oscillator (unchanged) ----
    b.phase_a += omega_fast * b.freq_jit * fdt;
    b.phase_b += omega_slow * b.freq_jit * fdt;
    float slow_unit = 0.5f + 0.5f * std::sin(b.phase_b);
    float slow_env  = (1.0f - st->amp_drift_depth) + st->amp_drift_depth * slow_unit;
    float breath    = 1.0f + st->pulse_depth * std::sin(b.phase_a);
    float amp       = slow_env * breath;
    b.amp = amp > 0.0f ? amp : 0.0f;
  }
}


void on_state_patched(void* self, int n, const char* pb, const int* off,
                      const int* len, const int* ops) {
  auto* st = static_cast<State*>(self);
  if (!st) return;
  // Only `seed` change triggers a reseed now — blob_size, blob_size_jitter,
  // drift_rate, drift_x/y_bias all take effect per-tick via the derived
  // wobble physics and radius packing. State-replay defense (§8.2) lives
  // on the `seed` field's change-detect.
  bool seed_changed = false;
  for (int i = 0; i < n; i++) {
    if (ops[i] != state::PatchReplace) continue;
    const char* path = pb + off[i];
    int plen = len[i];
    if      (state::pathIs(path, plen, "intensity"))        st->intensity = state::patchFloat(i);
    else if (state::pathIs(path, plen, "intensity_mod"))    st->intensity_mod = state::patchFloat(i);
    else if (state::pathIs(path, plen, "blob_count"))       st->blob_count = (int)state::patchFloat(i);
    else if (state::pathIs(path, plen, "fade_time"))        st->fade_time = state::patchFloat(i);
    else if (state::pathIs(path, plen, "blob_size"))        st->blob_size = state::patchFloat(i);
    else if (state::pathIs(path, plen, "blob_size_jitter")) st->blob_size_jitter = state::patchFloat(i);
    else if (state::pathIs(path, plen, "drift_rate"))       st->drift_rate = state::patchFloat(i);
    else if (state::pathIs(path, plen, "drift_x_bias"))     st->drift_x_bias = state::patchFloat(i);
    else if (state::pathIs(path, plen, "drift_y_bias"))     st->drift_y_bias = state::patchFloat(i);
    else if (state::pathIs(path, plen, "hue"))              st->hue = state::patchFloat(i);
    else if (state::pathIs(path, plen, "hue_shift"))        st->hue_shift = state::patchFloat(i);
    else if (state::pathIs(path, plen, "hue_curve"))        st->hue_curve = state::patchFloat(i);
    else if (state::pathIs(path, plen, "overflow_band"))    st->overflow_band = state::patchFloat(i);
    else if (state::pathIs(path, plen, "color_strength"))   st->color_strength = state::patchFloat(i);
    else if (state::pathIs(path, plen, "saturation"))       st->saturation = state::patchFloat(i);
    else if (state::pathIs(path, plen, "intensity_skew"))   st->intensity_skew = state::patchFloat(i);
    else if (state::pathIs(path, plen, "ramp_curve"))       st->ramp_curve = state::patchFloat(i);
    else if (state::pathIs(path, plen, "white_point"))      st->white_point = state::patchFloat(i);
    else if (state::pathIs(path, plen, "motion_strength"))  st->motion_strength = state::patchFloat(i);
    else if (state::pathIs(path, plen, "motion_skew"))      st->motion_skew = state::patchFloat(i);
    else if (state::pathIs(path, plen, "motion_curl"))      st->motion_curl = state::patchFloat(i);
    else if (state::pathIs(path, plen, "motion_extent"))    st->motion_extent = state::patchFloat(i);
    else if (state::pathIs(path, plen, "pulse_depth"))      st->pulse_depth = state::patchFloat(i);
    else if (state::pathIs(path, plen, "pulse_rate"))       st->pulse_rate = state::patchFloat(i);
    else if (state::pathIs(path, plen, "amp_drift_depth"))  st->amp_drift_depth = state::patchFloat(i);
    else if (state::pathIs(path, plen, "amp_drift_rate"))   st->amp_drift_rate = state::patchFloat(i);
    else if (state::pathIs(path, plen, "seed")) {
      uint32_t new_seed = (uint32_t)((int)state::patchFloat(i)) ^ 0xA17F2B91u;
      if (new_seed != st->seed) {
        st->seed = new_seed;
        seed_changed = true;
      }
    }
  }
  // Full re-seed on seed change — fresh deterministic positions/orbits
  // for all slots. The spawn-rng stream is also rebound to the new
  // seed so subsequent respawn-on-revival hits new sequences.
  if (seed_changed && st->initialized) {
    seed_all(*st);
  }
}

void render(void* self, int vp_w, int vp_h) {
  auto* st = static_cast<State*>(self);
  if (!st || !st->initialized || vp_w <= 0 || vp_h <= 0) return;
  auto in  = gpu::Device::textureForField("tex_in");
  auto out = gpu::Device::textureForField("tex_out");
  if (!in.valid() || !out.valid()) return;

  // Cover-square aspect so the blobs are isotropic.
  float min_dim = (float)(vp_w < vp_h ? vp_w : vp_h);
  float aspect_x = min_dim / (float)vp_w;
  float aspect_y = min_dim / (float)vp_h;

  // Pack every slot. Slots fading in/out (current_alive < 1) and
  // slots outside the active count (current_alive = 0) all flow
  // through the same path — amp * current_alive gates their
  // contribution. Cost is negligible at MAX_BLOBS = 32.
  GpuBlob gpu_blobs[MAX_BLOBS];
  std::memset(gpu_blobs, 0, sizeof(gpu_blobs));
  for (int i = 0; i < MAX_BLOBS; i++) {
    const BlobState& b = st->blobs[i];
    // Radius is derived each tick from base + per-blob jitter factor,
    // so blob_size / blob_size_jitter changes take effect without a
    // reseed.
    float radius = st->blob_size * (1.0f + b.size_jit * st->blob_size_jitter);
    if (radius < 0.02f) radius = 0.02f;
    // Smoothstep on the linear fade — zero slope at both endpoints so
    // the contribution easily eases away from 0 and into 1 instead of
    // crossing the boundary with a step in derivative. Human vision is
    // ~logarithmic, so a linear brightness ramp feels like a pop at
    // exactly the moments it touches off/on; smoothstep removes those.
    float ca = b.current_alive;
    float fade_eased = ca * ca * (3.0f - 2.0f * ca);
    gpu_blobs[i].pos_size[0] = b.x;
    gpu_blobs[i].pos_size[1] = b.y;
    gpu_blobs[i].pos_size[2] = radius;
    gpu_blobs[i].pos_size[3] = b.amp * fade_eased;       // fade rolls into amp
    gpu_blobs[i].jitters[0]  = b.hue_offset;
    gpu_blobs[i].jitters[1]  = b.vx;
    gpu_blobs[i].jitters[2]  = b.vy;
  }
  st->blob_buf.writeBytes(gpu_blobs, sizeof(GpuBlob) * MAX_BLOBS);
  uint32_t cnt = (uint32_t)MAX_BLOBS;

  float intensity = st->intensity + st->intensity_mod;
  if (intensity < 0.0f) intensity = 0.0f;

  Uniforms u = {};
  u.blob_count     = cnt;
  u.intensity      = intensity;
  // Signed slider [-1,+1] → exp in [1/4, 4] via the canonical helper,
  // matching hue_curve. 0 = linear; +1 = fast initial response (peak
  // saturates early); -1 = slow initial response (peak takes more
  // accum to reach).
  u.ramp_curve     = fx::signedSliderToExp(st->ramp_curve, 2.0f);
  u.white_point    = st->white_point;
  u.hue       = st->hue;
  u.hue_shift      = st->hue_shift;
  u.saturation     = st->saturation;
  u.aspect_x       = aspect_x;
  u.aspect_y       = aspect_y;
  u.intensity_skew = st->intensity_skew;
  // Signed slider [-1,+1] mapped to an exponent in [1/4, 4] via the
  // canonical curve helper (style guide §8.3 / §1.3). 0 = linear; +1 =
  // contrast at the peak (fast initial shift); -1 = contrast at the rim
  // (clings to peak, then races at low amp).
  u.hue_curve      = fx::signedSliderToExp(st->hue_curve, 2.0f);
  u.overflow_band  = st->overflow_band;
  u.color_strength = st->color_strength;
  st->uniform_buf.writeOne(u);

  // Pass 1 — color.
  {
    auto cp = gpu::ComputePass::begin();
    cp.setPSO(s_pso_color);
    cp.setBuffer(st->blob_buf, 0);
    cp.setTexture(in,  1, 0);
    cp.setTexture(out, 2, 1);
    cp.setBuffer(st->uniform_buf, 3);
    cp.dispatch((vp_w + 7) / 8, (vp_h + 7) / 8);
    cp.end();
  }

  // Pass 2 — motion. Skip when no downstream consumer.
  if (!state::isOutputConnected("render_outputs")) {
    gpu::Device::submit();
    return;
  }

  if (!st->motion_tex.valid() || st->motion_w != vp_w || st->motion_h != vp_h) {
    st->motion_tex = gpu::Device::createTexture(vp_w, vp_h, gpu::TextureFormat::RGBA16F);
    st->motion_w = vp_w;
    st->motion_h = vp_h;
    if (st->motion_tex.valid()) {
      state::setGpuTexture("render_outputs/motion", st->motion_tex.id);
    }
  }
  if (!st->motion_tex.valid()) {
    gpu::Device::submit();
    return;
  }

  // Resolve upstream motion; bind a 1x1 zero fallback when no upstream
  // is wired so the shader's unconditional sample is safe.
  auto upstream = gpu::Device::textureForField("render_outputs_in/motion");
  if (!upstream.valid()) {
    if (!st->zero_motion_tex.valid()) {
      st->zero_motion_tex = gpu::Device::createTexture(1, 1, gpu::TextureFormat::RGBA16F);
    }
    upstream = st->zero_motion_tex;
  }

  MotionUniforms mu = {};
  mu.blob_count      = cnt;
  mu.motion_strength = st->motion_strength;
  mu.motion_skew     = st->motion_skew;
  mu.aspect_x        = aspect_x;
  mu.aspect_y        = aspect_y;
  mu.motion_curl     = st->motion_curl;
  mu.intensity       = intensity;        // ties motion to the visible blob's brightness
  mu.motion_extent   = st->motion_extent < 0.0f ? 0.0f : (st->motion_extent > 1.0f ? 1.0f : st->motion_extent);
  st->motion_uniform_buf.writeOne(mu);

  {
    auto cp = gpu::ComputePass::begin();
    cp.setPSO(s_pso_motion);
    cp.setBuffer(st->blob_buf, 0);
    cp.setTexture(upstream, 1, 0);
    cp.setTexture(st->motion_tex, 2, 1);
    cp.setBuffer(st->motion_uniform_buf, 3);
    cp.dispatch((vp_w + 7) / 8, (vp_h + 7) / 8);
    cp.end();
  }

  gpu::Device::submit();
}

} // namespace soft_glow
