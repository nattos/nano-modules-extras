/*
 * source.legacy.double_chamber — v2 of the shipped NanoGraph "DoubleChamber".
 *
 * DoubleChamber was a 613-node graph fusing three coupled particle systems and
 * emitting laser (PONK) lines. This is a clean re-architecture of the SUBSET
 * the team actually used: the "P" field-particles (a PetriDish polynomial
 * vector field), the "Big" attractors they orbit, curl steering, and optional
 * image coupling. The charged-collision "K accelerator" block and the PONK
 * output are intentionally dropped; output is a normal texture. Tracers (L
 * block, field streamlines) and bridgers (stochastic chords between particles)
 * are ported as additional default-off line systems sharing the line raster.
 *
 * Motion vectors: when a downstream sink reads the canonical render_outputs
 * rail, a final pass rasterizes each element's screen-space velocity into
 * render_outputs/motion (RGBA16F) — particles emit their integrated per-frame
 * displacement, lines emit a tangent-along-segment velocity — composed over an
 * optional upstream motion field (render_outputs_in). Skipped entirely when no
 * sink is attached (state::isOutputConnected).
 *
 * Interactions (ported from flow_swarm, off by default): P particles splat soft
 * halos into a persistent square density buffer, and the NEXT frame's update
 * reads local crowding back out of it — giving particle-vs-particle behaviour
 * (density death, avoidance, streaming) for one extra raster pass, with no
 * neighbour search. See the "Interactions" schema block.
 *
 * Two persistent GPU storage buffers replace the old graph's LatchNode feedback
 * registers — a clean read-prev / write-next pool per system:
 *   1. big_update (compute) — drift / image-ride / boundary the few attractors.
 *   2. p_update   (compute) — crowding + field + Big pull + sink + jitter +
 *                             boundary + image; integrate; respawn + colour-capture.
 *   3. prefill    (compute) — tex_in × input_alpha → tex_out base.
 *   4. render     (instanced, additive) — P points then Big points.
 *   5. density    (instanced, additive) — P halos → crowding buffer (next frame).
 *
 * Per-instance ABI: mutable state in `State`; PSOs file-static (compiled once).
 */

#include <gpu.h>
#include <host.h>
#include <effect_blur.h>
#include "double_chamber_shaders.h"

#include <cstdint>
#include <cstring>
#include <cmath>

namespace double_chamber {

static constexpr int MAX_P   = 1000000;   // 1M × 32 B = 32 MB pool
static constexpr int MAX_BIG = 64;
static constexpr int SEED_CHUNK = 512;
static constexpr int MAX_TRACERS = 96;
static constexpr int MAX_SEG     = 96;   // segment slots per tracer (half each dir)
static constexpr int MAX_BRIDGERS = 1024;  // chords between particles (one Seg each)
// p_point_size slider [0,1] → effective isotropic-uv size [0, 0.01].
static constexpr float POINT_SIZE_SCALE = 0.01f;
// l_width slider [0,1] → effective line half-extent uv [0, 0.02].
static constexpr float LINE_WIDTH_SCALE = 0.02f;
// motion_line_speed slider [0,1] → effective uv/frame along-line motion [0, 0.05].
static constexpr float MOTION_LINE_SCALE = 0.05f;
// Interaction density buffer — an abstract square proximity field, resolution
// unrelated to the viewport. P particles splat soft halos into it (additive) and
// the next frame's p_update reads local crowding back from it.
static constexpr int DENSITY_RES = 256;

struct GpuParticle { float a[4]; float b[4]; };
static_assert(sizeof(GpuParticle) == 32, "particle must be 32 bytes");

struct PUpdateUniforms {
  uint32_t count, big_count, frame_index; float dt;
  float motion_rate, momentum, momentum_decay, field_speed;
  float field_scale, field_skew, field_squash, jitter;
  float to_big, to_big_curl, curl_dir, sink;
  float boundary, boundary_size, boundary_stiffness, boundary_speed;
  float to_image, to_image_curl, undertow_skew, undertow_squash;
  float ttl, spawn_size, aspect_x, aspect_y;
  float to_big_range, image_smoothing, to_line_rate, seg_total;
  float boundary_death, l_count_f, seg_stride, seg_live;
  uint32_t interactions; float density_threshold, density_death, avoid;
  float avoid_curl, avoid_noise, stream, stream_density;
  float density_res, _pd0, _pd1, _pd2;
};
struct DensityUniforms { float radius, aspect_x, aspect_y, _pad; };
struct TraceUniforms {
  uint32_t count, max_seg, frame_index; float dt;
  float field_scale, field_skew, field_squash, field_speed;
  float to_image, momentum, step_speed, length01;
  float time_decay, adv_step, color_contrib, l_opacity;
  float aspect_x, aspect_y, tint_r, tint_g;
  float tint_b, reseed_spread, image_smoothing, gradient_descent;
  float value_stop, grad_stop, time_stop_decay, _tp0;
};
struct LineVsUniforms { float aspect_x, aspect_y, width, _pad; };
struct LineFsUniforms { float soft, _a, _b, _c; };
struct BridgerUniforms {
  uint32_t count, p_count, frame_index; float dt;
  float rate, motion, opacity, color_contrib;
  float hue, tint_r, tint_g, tint_b;
};
struct MotionVsUniforms { float aspect_x, aspect_y, point_size, dt; float motion_rate, scale, _m1, _m2; };
struct MotionFsUniforms { uint32_t shape_kind; float shape_param, _f0, _f1; };
struct LineMotionVsUniforms { float aspect_x, aspect_y, width, line_speed; };
struct BigUpdateUniforms {
  uint32_t count, frame_index; float dt, motion_rate;
  float big_speed, big_momentum, big_momentum_decay, drift;
  float repel, direction, curl, curl_dir;
  float sink, boundary, boundary_size, boundary_stiffness;
  float boundary_speed, spread, ttl, aspect_x;
  float aspect_y, image_smoothing, _p1, _p2;
};
struct PrefillUniforms { float sr, sg, sb, sa; };
struct VsUniforms { float aspect_x, aspect_y, point_size, _pad; };
struct FsUniforms {
  float color_contrib, render_hue, opacity, alpha_curve;
  float tint_r, tint_g, tint_b, exposure;
  uint32_t shape_kind; float shape_param, _p0, _p1;
};

enum BlendMode : int { BLEND_ADD = 0, BLEND_ALPHA = 1 };

struct State {
  gpu::Buffer  p_buf, big_buf, tracer_buf, seg_buf;
  gpu::Buffer  bridger_buf, bridger_seg_buf;
  gpu::Buffer  p_uniform, big_uniform, prefill_uniform, trace_uniform;
  gpu::Buffer  vs_uniform_p, vs_uniform_big, fs_uniform_p, fs_uniform_big;
  gpu::Buffer  line_vs_uniform, line_fs_uniform;
  gpu::Buffer  bridger_uniform, bridger_vs_uniform, bridger_fs_uniform;
  gpu::Buffer  motion_vs_uniform_p, motion_vs_uniform_big, motion_fs_uniform;
  gpu::Buffer  lm_vs_uniform_l, lm_vs_uniform_b;
  gpu::Buffer  density_uniform;
  gpu::Sampler sampler;
  gpu::Texture black_tex;   // 1×1 fallback when no input is wired (true generator)
  gpu::Texture field_tex;   // smoothed input for gradient/colour sampling
  gpu::Texture motion_tex;       // render_outputs/motion (RGBA16F, when a sink reads it)
  gpu::Texture zero_motion_tex;  // 1×1 fallback for an unwired upstream motion input
  gpu::Texture density_tex;       // persistent crowding buffer (1-frame delayed)
  gpu::Texture zero_density_tex;  // 1×1 fallback when interactions are off
  int field_w = 0, field_h = 0;
  int motion_w = 0, motion_h = 0;
  bool initialized = false;
  uint32_t frame_index = 0;
  int  inited_count = 0;   // P slots seeded so far (lazy growth to p_count)

  // CPU param mirrors.
  int   p_count = 12000;
  float motion_rate = 1.0f;
  float field_speed = 0.25f, field_scale = 1.0f, field_skew = 0.0f, field_squash = 0.5f;
  float momentum = 0.6f, momentum_decay = 0.98f;
  float to_big = 0.3f, to_big_curl = 0.2f, to_big_range = 0.4f, curl_dir = 1.0f, sink = 0.0f;
  float jitter = 0.04f;
  float boundary = 1.0f, boundary_size = 0.42f, boundary_stiffness = 8.0f, boundary_speed = 1.2f;
  float boundary_death = 0.0f;  // P(die+respawn) at the boundary, ∝ overshoot (0 = off)
  float to_image = 0.0f, to_image_curl = 0.0f, image_smoothing = 0.3f;
  float undertow_skew = 0.0f, undertow_squash = 1.0f;
  float ttl = 0.4f, spawn_size = 0.5f;
  // p_point_size is a [0,1] slider (2-decimal IDE clipping needs the wide
  // range for fine control) → effective uv size = slider * POINT_SIZE_SCALE.
  float p_point_size = 0.5f, p_opacity = 0.25f, p_alpha_curve = 1.0f, render_hue = 0.0f;
  float tint_r = 1.0f, tint_g = 1.0f, tint_b = 1.0f, exposure = 1.0f;
  float color_contrib = 0.5f;
  int   p_shape = 1;  // gaussian
  // Big.
  int   big_count = 6;
  float big_speed = 0.5f, big_drift = 0.3f, big_repel = 0.2f, big_curl = 0.3f;
  float big_momentum = 0.85f, big_momentum_decay = 0.99f;
  float big_spread = 0.5f, big_ttl = 0.6f;
  float big_point_size = 0.04f, big_opacity = 0.25f, big_render_hue = 0.0f;
  // Tracers (L block).
  int   l_count = 24;
  float spawn_on_line = 0.0f;   // P(respawn onto an existing tracer line)
  float l_length = 0.5f, l_step_speed = 0.02f, l_momentum = 0.5f, l_opacity = 0.5f;
  float l_width = 0.2f, l_soft = 1.0f, l_time_decay = 0.1f, l_adv_step = 0.1f, l_reseed_spread = 0.4f;
  float l_gradient_descent = 0.2f;  // 0 = level-curve/tangent, 1 = down-gradient
  float l_value_stop = 0.0f, l_grad_stop = 0.0f, l_time_stop_decay = 0.0f;  // early-death (0=off)
  // Bridgers (stochastic chords between P particles). Off by default — the
  // original shipped with Bridger Count 12, but defaulting off keeps the base
  // look clean; raise the count to enable (see DNODE_MIGRATION_NOTES §4.2).
  int   bridger_count = 0;
  float bridger_rate = 0.05f;      // P(re-target) per endpoint per frame
  float bridger_motion = 0.4f;     // glide lerp toward the live target
  float bridger_opacity = 0.4f;
  float bridger_width = 0.15f, bridger_soft = 1.0f;
  float bridger_hue = 0.0f, bridger_color_contrib = 0.5f;
  // Motion-vector output (render_outputs/motion) — only produced when a sink
  // is wired. Particles emit their integrated velocity; lines emit a tangent.
  float motion_line_speed = 0.3f;  // [0,1] slider → uv/frame via MOTION_LINE_SCALE
  float motion_particle_scale = 1.0f;  // extra multiplier on emitted particle velocity
  // Interactions (a 1-frame-delayed density buffer drives these). Off by
  // default — enabling costs one extra raster pass over the pool.
  bool  interactions       = false;
  float interaction_radius = 0.015f;  // splat halo radius in density uv (the RANGE)
  float density_threshold  = 4.0f;    // crowding (≈ neighbours) before death
  float density_death      = 0.0f;    // death-rate scale over threshold
  float avoid              = 0.0f;    // push away from neighbours
  float avoid_curl         = 0.0f;    // rotate the avoidance ±90°
  float avoid_noise        = 0.08f;   // random jitter on avoidance (breaks clumps)
  float stream             = 0.0f;    // +align / -diverge velocity vs the group
  float stream_density     = 3.0f;    // neighbour density for ~max stream effect
  bool  debug_density      = false;   // render the density buffer as a heat map
  // Composite.
  float input_alpha = 1.0f;
  int   blend_mode = BLEND_ADD;
};

static fx::GaussianBlur s_blur;   // shared image-smoothing helper
static gpu::ComputePSO s_pso_big_update;
static gpu::ComputePSO s_pso_p_update;
static gpu::ComputePSO s_pso_prefill;
static gpu::ComputePSO s_pso_trace;
static gpu::ComputePSO s_pso_bridger;
static gpu::RenderPSO  s_pso_render_add;
static gpu::RenderPSO  s_pso_render_alpha;
static gpu::RenderPSO  s_pso_line_add;
static gpu::RenderPSO  s_pso_line_alpha;
static gpu::ComputePSO s_pso_motion_prefill;
static gpu::RenderPSO  s_pso_motion_point;   // particle velocity → motion (RGBA16F)
static gpu::RenderPSO  s_pso_motion_line;    // line tangent → motion (RGBA16F)
static gpu::RenderPSO  s_pso_density;        // soft-halo additive splat → density buffer
static gpu::ComputePSO s_pso_density_debug;  // heat-map blit of the density buffer

static inline uint32_t lcg(uint32_t& s) { s = s * 1664525u + 1013904223u; return s; }
static inline float lcgf(uint32_t& s) { return (lcg(s) >> 8) * (1.0f / float(1u << 24)); }
static inline float as_float(uint32_t u) { float f; std::memcpy(&f, &u, 4); return f; }

// Seed a pool's slots [from, to) to match the STEADY STATE so there's no
// startup "all at once" flash: positions on a centred disc (≈ the respawn
// distribution), and a staggered remaining life against a FULL life_total so
// life_norm starts spread across [0,1] (varied brightness) and deaths/respawns
// are spread in time rather than synchronized. Chunked to keep the wasm stack
// small; seeded lazily up to the live count (the pool can be 1M).
static void seed_pool(gpu::Buffer& buf, int from, int to, float lifetime, uint32_t salt) {
  if (from >= to) return;
  uint32_t rng = (0x12345678u ^ salt) + (uint32_t)from * 2654435761u;
  GpuParticle chunk[SEED_CHUNK];
  for (int start = from; start < to; start += SEED_CHUNK) {
    int end = start + SEED_CHUNK; if (end > to) end = to;
    int m = end - start;
    for (int k = 0; k < m; k++) {
      // Centred disc in uv (aspect-corrected by the first update); ~steady state.
      float ang = lcgf(rng) * 6.2831853f;
      float rad = 0.33f * std::sqrt(lcgf(rng));
      float ux = 0.5f + std::cos(ang) * rad;
      float uy = 0.5f + std::sin(ang) * rad;
      float life_remain = lcgf(rng) * lifetime;   // staggered against full total
      uint32_t zbyte = (uint32_t)(lcgf(rng) * 255.0f) & 0xFFu;
      uint32_t packed = 0x00FFFFFFu | (zbyte << 24);
      GpuParticle& p = chunk[k];
      p.a[0] = ux; p.a[1] = uy; p.a[2] = life_remain; p.a[3] = lifetime;
      p.b[0] = 0.f; p.b[1] = 0.f; p.b[2] = 1.0f; p.b[3] = as_float(packed);
    }
    buf.writeBytes(chunk, int(sizeof(GpuParticle)) * m, int(sizeof(GpuParticle)) * start);
  }
}

// Bridger state layout matches the HLSL BridgerState (two float4 = 32 B).
struct GpuBridger { float a[4]; float b[4]; };
static_assert(sizeof(GpuBridger) == 32, "bridger must be 32 bytes");

// Seed all MAX_BRIDGERS slots with random initial targets (fresh flag 0 → the
// shader snaps each endpoint onto its particle on the first frame). Stale
// targets past the current p_count self-heal in-shader (re-pick), so no lazy
// re-seed is needed when the count grows. 32 KB total — one cheap write.
static void seed_bridgers(gpu::Buffer& buf, int p_count) {
  uint32_t rng = 0xB12D9E37u;
  GpuBridger chunk[SEED_CHUNK];
  uint32_t pc = (uint32_t)(p_count < 1 ? 1 : p_count);
  for (int start = 0; start < MAX_BRIDGERS; start += SEED_CHUNK) {
    int end = start + SEED_CHUNK; if (end > MAX_BRIDGERS) end = MAX_BRIDGERS;
    int m = end - start;
    for (int k = 0; k < m; k++) {
      uint32_t ta = (uint32_t)(lcgf(rng) * float(pc)); if (ta >= pc) ta = pc - 1u;
      uint32_t tb = (uint32_t)(lcgf(rng) * float(pc)); if (tb >= pc) tb = pc - 1u;
      GpuBridger& g = chunk[k];
      g.a[0] = 0.5f; g.a[1] = 0.5f; g.a[2] = 0.5f; g.a[3] = 0.5f;
      g.b[0] = as_float(ta); g.b[1] = as_float(tb); g.b[2] = 0.f; g.b[3] = 0.f;
    }
    buf.writeBytes(chunk, int(sizeof(GpuBridger)) * m, int(sizeof(GpuBridger)) * start);
  }
}

// Hide the interaction knobs when interactions are off (§0) — the block is a
// dozen fields that do nothing until the bool is on.
static void apply_interaction_visibility(bool interactions) {
  bool ix = !interactions;
  state::setFieldHidden("interaction_radius", ix);
  state::setFieldHidden("density_threshold",  ix);
  state::setFieldHidden("density_death",      ix);
  state::setFieldHidden("avoid",              ix);
  state::setFieldHidden("avoid_curl",         ix);
  state::setFieldHidden("avoid_noise",        ix);
  state::setFieldHidden("stream",             ix);
  state::setFieldHidden("stream_density",     ix);
  state::setFieldHidden("debug_density",      ix);
}

// Static (self-less) visibility evaluator — pure over the patch list.
void eval_visibility(int n, const char* pb, const int* off, const int* len, const int* ops) {
  bool interactions = false;
  for (int i = 0; i < n; i++) {
    if (ops[i] != state::PatchReplace) continue;
    const char* p = pb + off[i]; int l = len[i];
    if (state::pathIs(p, l, "interactions")) interactions = state::patchFloat(i) != 0.0f;
  }
  apply_interaction_visibility(interactions);
}

static void on_state_ready(void* self) {
  auto* s = static_cast<State*>(self);
  if (s) apply_interaction_visibility(s->interactions);
}

void module_init() {
  state::init("source.legacy.double_chamber", {1, 4, 7},
    state::Schema()
      // ---- P system (standard) ----
      .intField  ("p_count",        12000, 1, MAX_P,      state::PrimaryInput)
      .floatField("motion_rate",    1.0f,  0.0f, 4.0f,    state::PrimaryInput)
      .floatField("field_speed",    0.25f, 0.0f, 2.0f,    state::PrimaryInput)
      .floatField("momentum",       0.6f,  0.0f, 1.0f,    state::PrimaryInput)
      .floatField("to_big",         0.3f,  0.0f, 3.0f,    state::PrimaryInput)
      .floatField("to_big_curl",    0.2f, -3.0f, 3.0f,    state::PrimaryInput)
      .floatField("to_big_range",   0.4f,  0.0f, 2.0f,    state::PrimaryInput)
      .floatField("jitter",         0.04f, 0.0f, 1.0f,    state::PrimaryInput)
      .floatField("color_contrib",  0.5f,  0.0f, 1.0f,    state::PrimaryInput)
      .floatField("to_image",       0.0f, -2.0f, 2.0f,    state::PrimaryInput)
      .floatField("image_smoothing", 0.3f, 0.0f, 1.0f,    state::PrimaryInput)
      // ---- P tuning ----
      .floatField("field_scale",    1.0f,  0.1f, 4.0f,    state::SecondaryInput)
      .floatField("field_skew",     0.0f, -2.0f, 2.0f,    state::SecondaryInput)
      .floatField("field_squash",   0.5f,  0.0f, 2.0f,    state::SecondaryInput)
      .floatField("momentum_decay", 0.98f, 0.8f, 1.0f,    state::SecondaryInput)
      .floatField("curl_dir",       1.0f, -1.0f, 1.0f,    state::SecondaryInput)
      .floatField("sink",           0.0f, -1.0f, 1.0f,    state::SecondaryInput)
      .floatField("to_image_curl",  0.0f, -2.0f, 2.0f,    state::SecondaryInput)
      .floatField("undertow_skew",  0.0f, -1.0f, 1.0f,    state::SecondaryInput)
      .floatField("undertow_squash",1.0f,  0.0f, 8.0f,    state::SecondaryInput)
      .floatField("boundary",       1.0f,  0.0f, 1.0f,    state::SecondaryInput)
      .floatField("boundary_size",  0.42f, 0.05f, 0.7f,   state::SecondaryInput)
      .floatField("boundary_stiffness", 8.0f, 0.5f, 32.0f, state::SecondaryInput)
      .floatField("boundary_speed", 1.2f,  0.0f, 6.0f,    state::SecondaryInput)
      .floatField("boundary_death", 0.0f,  0.0f, 1.0f,    state::SecondaryInput)
      .floatField("ttl",            0.4f,  0.02f, 1.0f,   state::SecondaryInput)
      .floatField("spawn_size",     0.5f,  0.0f, 1.0f,    state::SecondaryInput)
      // ---- P render ----
      .floatField("p_point_size",   0.5f,  0.0f,  1.0f,    state::SecondaryInput)
      .floatField("p_opacity",      0.25f, 0.0f, 1.0f,    state::SecondaryInput)
      .floatField("p_alpha_curve",  1.0f,  0.25f, 4.0f,   state::SecondaryInput)
      .floatField("render_hue",     0.0f,  0.0f, 1.0f,    state::SecondaryInput)
      .selectField("p_shape",       1,                    state::SecondaryInput, {
        {"Point", 0}, {"Gaussian", 1}, {"Circle", 2} })
      .rgbField  ("tint",           1.0f, 1.0f, 1.0f,     state::SecondaryInput)
      .floatField("exposure",       1.0f,  0.0f, 4.0f,    state::SecondaryInput)
      // ---- Big attractors ----
      .intField  ("big_count",      6,     0, MAX_BIG,    state::SecondaryInput)
      .floatField("big_speed",      0.5f,  0.0f, 4.0f,    state::SecondaryInput)
      .floatField("big_drift",      0.3f, -2.0f, 2.0f,    state::SecondaryInput)
      .floatField("big_repel",      0.2f, -2.0f, 2.0f,    state::SecondaryInput)
      .floatField("big_curl",       0.3f, -2.0f, 2.0f,    state::SecondaryInput)
      .floatField("big_momentum",   0.85f, 0.0f, 1.0f,    state::SecondaryInput)
      .floatField("big_spread",     0.5f,  0.0f, 1.0f,    state::SecondaryInput)
      .floatField("big_ttl",        0.6f,  0.05f, 1.0f,   state::SecondaryInput)
      .floatField("big_point_size", 0.04f, 0.002f, 0.2f,  state::SecondaryInput)
      .floatField("big_opacity",    0.25f, 0.0f, 1.0f,    state::SecondaryInput)
      .floatField("big_render_hue", 0.0f,  0.0f, 1.0f,    state::SecondaryInput)
      // ---- Tracers (L block) ----
      .intField  ("l_count",        24,    0, MAX_TRACERS, state::PrimaryInput)
      .floatField("spawn_on_line",  0.0f,  0.0f, 1.0f,    state::PrimaryInput)
      .floatField("l_length",       0.5f,  0.0f, 1.0f,    state::PrimaryInput)
      .floatField("l_opacity",      0.5f,  0.0f, 1.0f,    state::PrimaryInput)
      .floatField("l_step_speed",   0.02f, 0.005f, 0.1f,  state::SecondaryInput)
      .floatField("l_momentum",     0.5f,  0.0f, 1.0f,    state::SecondaryInput)
      .floatField("l_gradient_descent", 0.2f, 0.0f, 1.0f, state::PrimaryInput)
      .floatField("l_width",        0.2f,  0.0f, 1.0f,    state::SecondaryInput)
      .floatField("l_soft",         1.0f,  0.1f, 4.0f,    state::SecondaryInput)
      .floatField("l_time_decay",   0.1f,  0.0f, 2.0f,    state::SecondaryInput)
      .floatField("l_adv_step",     0.1f,  0.0f, 1.0f,    state::SecondaryInput)
      .floatField("l_reseed_spread", 0.4f, 0.0f, 1.0f,    state::SecondaryInput)
      .floatField("l_value_stop",    0.0f, 0.0f, 1.0f,    state::SecondaryInput)
      .floatField("l_grad_stop",     0.0f, 0.0f, 1.0f,    state::SecondaryInput)
      .floatField("l_time_stop_decay", 0.0f, 0.0f, 2.0f,  state::SecondaryInput)
      // ---- Bridgers (chords between particles) ----
      .intField  ("bridger_count",  0,     0, MAX_BRIDGERS, state::PrimaryInput)
      .floatField("bridger_rate",   0.05f, 0.0f, 1.0f,    state::PrimaryInput)
      .floatField("bridger_motion", 0.4f,  0.0f, 1.0f,    state::PrimaryInput)
      .floatField("bridger_opacity", 0.4f, 0.0f, 1.0f,    state::PrimaryInput)
      .floatField("bridger_width",  0.15f, 0.0f, 1.0f,    state::SecondaryInput)
      .floatField("bridger_soft",   1.0f,  0.1f, 4.0f,    state::SecondaryInput)
      .floatField("bridger_hue",    0.0f,  0.0f, 1.0f,    state::SecondaryInput)
      .floatField("bridger_color_contrib", 0.5f, 0.0f, 1.0f, state::SecondaryInput)
      // ---- Interactions (particle-vs-particle, via a density buffer) ----
      // P particles splat into a 1-frame-delayed crowding buffer so they can
      // react to each other. Off by default (an extra splat pass over the pool).
      .boolField ("interactions",       false,                state::PrimaryInput)
      // Halo radius in the density buffer — the interaction RANGE.
      .floatField("interaction_radius", 0.015f, 0.002f, 0.08f, state::PrimaryInput)
      // Density death: over `threshold` crowding (≈ neighbour count, soft knee)
      // particles get a `death`-scaled chance to die & respawn → uniform density.
      .floatField("density_threshold",  4.0f,   0.0f,  32.0f, state::PrimaryInput)
      .floatField("density_death",      0.0f,   0.0f,  1.0f,  state::PrimaryInput)
      // Avoidance: a force down the density gradient (away from neighbours);
      // curl rotates that push ±90° for a swirling avoidance.
      .floatField("avoid",              0.0f,   0.0f,  1.0f,  state::PrimaryInput)
      .floatField("avoid_curl",         0.0f,  -1.0f,  1.0f,  state::SecondaryInput)
      // Random jitter on the avoidance so particles still scatter where the
      // density gradient is flat (a symmetric clump's centre).
      .floatField("avoid_noise",        0.08f,  0.0f,  1.0f,  state::SecondaryInput)
      // Stream: align (+) or diverge (-) each particle's velocity DIRECTION with
      // the local group's mean motion (the density buffer's .gb channels).
      .floatField("stream",             0.0f,  -1.0f,  1.0f,  state::PrimaryInput)
      .floatField("stream_density",     3.0f,   0.5f,  32.0f, state::SecondaryInput)
      // Debug: render the density buffer itself (heat map) instead of the cloud.
      .boolField ("debug_density",      false,                state::SecondaryInput)
      // ---- Motion-vector output ----
      .floatField("motion_line_speed", 0.3f, 0.0f, 1.0f,  state::SecondaryInput)
      .floatField("motion_particle_scale", 1.0f, 0.0f, 4.0f, state::SecondaryInput)
      // ---- Composite ----
      .selectField("blend_mode",    BLEND_ADD,            state::SecondaryInput, {
        {"Add", BLEND_ADD}, {"Alpha", BLEND_ALPHA} })
      .floatField("input_alpha",    1.0f,  0.0f, 1.0f,    state::SecondaryInput)
      // ---- I/O ----
      .textureField("tex_in",  state::PrimaryInput)
      .textureField("tex_out", state::PrimaryOutput)
      // Canonical depth+motion rail. The motion pass only runs when a
      // downstream sink reads it (state::isOutputConnected). render_outputs_in
      // lets this compose its motion OVER an upstream motion field.
      .renderOutputs(state::PrimaryOutput)
      .renderOutputs(state::PrimaryInput, "render_outputs_in")
      .capability(state::Capability::Generator)
  );

  if (gpu::Device::backend() == gpu::Backend::None) return;

  state::registerShaderSPV("dc_big_update", BIG_UPDATE_SPV, BIG_UPDATE_SPV_SIZE);
  state::registerShaderSPV("dc_p_update",   P_UPDATE_SPV,   P_UPDATE_SPV_SIZE);
  state::registerShaderSPV("dc_prefill",    PREFILL_SPV,    PREFILL_SPV_SIZE);
  state::registerShaderSPV("dc_vs",         VS_SPV,         VS_SPV_SIZE);
  state::registerShaderSPV("dc_fs",         FS_SPV,         FS_SPV_SIZE);
  state::registerShaderSPV("dc_trace",      TRACE_SPV,      TRACE_SPV_SIZE);
  state::registerShaderSPV("dc_bridger",    BRIDGER_SPV,    BRIDGER_SPV_SIZE);
  state::registerShaderSPV("dc_line_vs",    LINE_VS_SPV,    LINE_VS_SPV_SIZE);
  state::registerShaderSPV("dc_line_fs",    LINE_FS_SPV,    LINE_FS_SPV_SIZE);
  state::registerShaderSPV("dc_motion_prefill", MOTION_PREFILL_SPV, MOTION_PREFILL_SPV_SIZE,
                           "rgba16float", "write");
  state::registerShaderSPV("dc_motion_vs",  MOTION_VS_SPV,  MOTION_VS_SPV_SIZE);
  state::registerShaderSPV("dc_motion_fs",  MOTION_FS_SPV,  MOTION_FS_SPV_SIZE);
  state::registerShaderSPV("dc_line_motion_vs", LINE_MOTION_VS_SPV, LINE_MOTION_VS_SPV_SIZE);
  state::registerShaderSPV("dc_line_motion_fs", LINE_MOTION_FS_SPV, LINE_MOTION_FS_SPV_SIZE);
  state::registerShaderSPV("dc_density_vs",    DENSITY_VS_SPV,    DENSITY_VS_SPV_SIZE);
  state::registerShaderSPV("dc_density_fs",    DENSITY_FS_SPV,    DENSITY_FS_SPV_SIZE);
  state::registerShaderSPV("dc_density_debug", DENSITY_DEBUG_SPV, DENSITY_DEBUG_SPV_SIZE);

  auto cs_big = gpu::Device::createShaderModuleByName("dc_big_update");
  auto cs_p   = gpu::Device::createShaderModuleByName("dc_p_update");
  auto cs_pre = gpu::Device::createShaderModuleByName("dc_prefill");
  auto cs_tr  = gpu::Device::createShaderModuleByName("dc_trace");
  auto cs_br  = gpu::Device::createShaderModuleByName("dc_bridger");
  auto vs     = gpu::Device::createShaderModuleByName("dc_vs");
  auto fs     = gpu::Device::createShaderModuleByName("dc_fs");
  auto lvs    = gpu::Device::createShaderModuleByName("dc_line_vs");
  auto lfs    = gpu::Device::createShaderModuleByName("dc_line_fs");
  auto cs_mpf = gpu::Device::createShaderModuleByName("dc_motion_prefill");
  auto mvs    = gpu::Device::createShaderModuleByName("dc_motion_vs");
  auto mfs    = gpu::Device::createShaderModuleByName("dc_motion_fs");
  auto lmvs   = gpu::Device::createShaderModuleByName("dc_line_motion_vs");
  auto lmfs   = gpu::Device::createShaderModuleByName("dc_line_motion_fs");
  auto dvs    = gpu::Device::createShaderModuleByName("dc_density_vs");
  auto dfs    = gpu::Device::createShaderModuleByName("dc_density_fs");
  auto cs_dbg = gpu::Device::createShaderModuleByName("dc_density_debug");
  if (!cs_big || !cs_p || !cs_pre || !cs_tr || !cs_br || !vs || !fs || !lvs || !lfs) return;
  if (!cs_mpf || !mvs || !mfs || !lmvs || !lmfs) return;
  if (!dvs || !dfs || !cs_dbg) return;

  s_pso_big_update = gpu::Device::createComputePSO(cs_big, "main", gpu::Bindings()
      .storageRW(0).tex2d(1).sampler(2).uniform(3));
  s_pso_p_update = gpu::Device::createComputePSO(cs_p, "main", gpu::Bindings()
      .storageRW(0).storage(1).tex2d(2).sampler(3).uniform(4).storage(5).tex2d(6));
  s_pso_prefill = gpu::Device::createComputePSO(cs_pre, "main", gpu::Bindings()
      .tex2d(0).storageTex2d(1).uniform(2));
  s_pso_trace = gpu::Device::createComputePSO(cs_tr, "main", gpu::Bindings()
      .storageRW(0).storageRW(1).tex2d(2).sampler(3).uniform(4));
  s_pso_bridger = gpu::Device::createComputePSO(cs_br, "main", gpu::Bindings()
      .storageRW(0).storage(1).storageRW(2).uniform(3));
  s_pso_render_add = gpu::Device::createInstancedRenderPSO(vs, "main", fs, "main",
      gpu::TextureFormat::Surface,
      gpu::Bindings().storage(0).uniform(1).uniform(2),
      gpu::Device::BlendMode::Additive);
  s_pso_render_alpha = gpu::Device::createInstancedRenderPSO(vs, "main", fs, "main",
      gpu::TextureFormat::Surface,
      gpu::Bindings().storage(0).uniform(1).uniform(2),
      gpu::Device::BlendMode::AlphaOver);
  s_pso_line_add = gpu::Device::createInstancedRenderPSO(lvs, "main", lfs, "main",
      gpu::TextureFormat::Surface,
      gpu::Bindings().storage(0).uniform(1).uniform(2),
      gpu::Device::BlendMode::Additive);
  s_pso_line_alpha = gpu::Device::createInstancedRenderPSO(lvs, "main", lfs, "main",
      gpu::TextureFormat::Surface,
      gpu::Bindings().storage(0).uniform(1).uniform(2),
      gpu::Device::BlendMode::AlphaOver);

  // Motion-vector PSOs. The render targets are RGBA16F; AlphaOver composes each
  // footprint's velocity over the (upstream-seeded) motion field by coverage.
  s_pso_motion_prefill = gpu::Device::createComputePSO(cs_mpf, "main", gpu::Bindings()
      .tex2d(0).storageTex2d(1, gpu::TextureFormat::RGBA16F));
  s_pso_motion_point = gpu::Device::createInstancedRenderPSO(mvs, "main", mfs, "main",
      gpu::TextureFormat::RGBA16F,
      gpu::Bindings().storage(0).uniform(1).uniform(2),
      gpu::Device::BlendMode::AlphaOver);
  s_pso_motion_line = gpu::Device::createInstancedRenderPSO(lmvs, "main", lmfs, "main",
      gpu::TextureFormat::RGBA16F,
      gpu::Bindings().storage(0).uniform(1).uniform(2),
      gpu::Device::BlendMode::AlphaOver);

  // Interactions: soft halos summed into the RGBA16F crowding buffer, plus a
  // heat-map debug blit of it.
  s_pso_density = gpu::Device::createInstancedRenderPSO(dvs, "main", dfs, "main",
      gpu::TextureFormat::RGBA16F,
      gpu::Bindings().storage(0).uniform(1),
      gpu::Device::BlendMode::Additive);
  s_pso_density_debug = gpu::Device::createComputePSO(cs_dbg, "main", gpu::Bindings()
      .tex2d(0).sampler(1).storageTex2d(2));

  s_blur.init();
  state::log("double_chamber: module initialized");
}

void* create() {
  auto* s = new State();
  s->p_buf   = gpu::Device::createBuffer(sizeof(GpuParticle) * MAX_P, gpu::BufferUsage::Storage);
  s->big_buf = gpu::Device::createBuffer(sizeof(GpuParticle) * MAX_BIG, gpu::BufferUsage::Storage);
  s->tracer_buf = gpu::Device::createBuffer(16 * MAX_TRACERS, gpu::BufferUsage::Storage);
  s->seg_buf = gpu::Device::createBuffer(32 * MAX_TRACERS * MAX_SEG, gpu::BufferUsage::Storage);
  s->bridger_buf     = gpu::Device::createBuffer(sizeof(GpuBridger) * MAX_BRIDGERS, gpu::BufferUsage::Storage);
  s->bridger_seg_buf = gpu::Device::createBuffer(32 * MAX_BRIDGERS, gpu::BufferUsage::Storage);
  s->p_uniform       = gpu::Device::createBuffer(sizeof(PUpdateUniforms), gpu::BufferUsage::Uniform);
  s->big_uniform     = gpu::Device::createBuffer(sizeof(BigUpdateUniforms), gpu::BufferUsage::Uniform);
  s->prefill_uniform = gpu::Device::createBuffer(sizeof(PrefillUniforms), gpu::BufferUsage::Uniform);
  s->trace_uniform   = gpu::Device::createBuffer(sizeof(TraceUniforms), gpu::BufferUsage::Uniform);
  s->vs_uniform_p    = gpu::Device::createBuffer(sizeof(VsUniforms), gpu::BufferUsage::Uniform);
  s->vs_uniform_big  = gpu::Device::createBuffer(sizeof(VsUniforms), gpu::BufferUsage::Uniform);
  s->fs_uniform_p    = gpu::Device::createBuffer(sizeof(FsUniforms), gpu::BufferUsage::Uniform);
  s->fs_uniform_big  = gpu::Device::createBuffer(sizeof(FsUniforms), gpu::BufferUsage::Uniform);
  s->line_vs_uniform = gpu::Device::createBuffer(sizeof(LineVsUniforms), gpu::BufferUsage::Uniform);
  s->line_fs_uniform = gpu::Device::createBuffer(sizeof(LineFsUniforms), gpu::BufferUsage::Uniform);
  s->bridger_uniform    = gpu::Device::createBuffer(sizeof(BridgerUniforms), gpu::BufferUsage::Uniform);
  s->bridger_vs_uniform = gpu::Device::createBuffer(sizeof(LineVsUniforms), gpu::BufferUsage::Uniform);
  s->bridger_fs_uniform = gpu::Device::createBuffer(sizeof(LineFsUniforms), gpu::BufferUsage::Uniform);
  s->motion_vs_uniform_p   = gpu::Device::createBuffer(sizeof(MotionVsUniforms), gpu::BufferUsage::Uniform);
  s->motion_vs_uniform_big = gpu::Device::createBuffer(sizeof(MotionVsUniforms), gpu::BufferUsage::Uniform);
  s->motion_fs_uniform     = gpu::Device::createBuffer(sizeof(MotionFsUniforms), gpu::BufferUsage::Uniform);
  s->lm_vs_uniform_l       = gpu::Device::createBuffer(sizeof(LineMotionVsUniforms), gpu::BufferUsage::Uniform);
  s->lm_vs_uniform_b       = gpu::Device::createBuffer(sizeof(LineMotionVsUniforms), gpu::BufferUsage::Uniform);
  s->density_uniform       = gpu::Device::createBuffer(sizeof(DensityUniforms), gpu::BufferUsage::Uniform);
  s->sampler         = gpu::Device::createSampler(gpu::FilterMode::Linear, gpu::AddressMode::ClampToEdge);
  s->black_tex       = gpu::Device::createTexture(1, 1, gpu::TextureFormat::RGBA8);
  if (s->black_tex.valid()) gpu::Device::clear(s->black_tex, 0.f, 0.f, 0.f, 1.f);
  return s;
}

void destroy(void* self) {
  auto* s = static_cast<State*>(self);
  if (!s) return;
  s->p_buf.release(); s->big_buf.release(); s->tracer_buf.release(); s->seg_buf.release();
  s->bridger_buf.release(); s->bridger_seg_buf.release();
  s->p_uniform.release(); s->big_uniform.release(); s->prefill_uniform.release(); s->trace_uniform.release();
  s->vs_uniform_p.release(); s->vs_uniform_big.release();
  s->fs_uniform_p.release(); s->fs_uniform_big.release();
  s->line_vs_uniform.release(); s->line_fs_uniform.release();
  s->bridger_uniform.release(); s->bridger_vs_uniform.release(); s->bridger_fs_uniform.release();
  s->motion_vs_uniform_p.release(); s->motion_vs_uniform_big.release(); s->motion_fs_uniform.release();
  s->lm_vs_uniform_l.release(); s->lm_vs_uniform_b.release();
  s->density_uniform.release();
  s->sampler.release();
  s->black_tex.release();
  s->field_tex.release();
  s->motion_tex.release();
  s->zero_motion_tex.release();
  s->density_tex.release();
  s->zero_density_tex.release();
  delete s;
}

void init(void* self) {
  auto* s = static_cast<State*>(self);
  if (!s) return;
  if (!s_pso_p_update.valid() || !s_pso_big_update.valid() || !s_pso_render_add.valid()) return;
  if (!s->p_buf.valid() || !s->big_buf.valid()) return;
  s->frame_index = 0;
  seed_pool(s->p_buf,   0, s->p_count,  s->ttl * 8.0f,      0x1111u);
  seed_pool(s->big_buf, 0, MAX_BIG,     s->big_ttl * 20.0f, 0x2222u);
  s->inited_count = s->p_count;
  // Zero tracers (time=0 → each seeds itself on the first trace).
  {
    float zeros[4 * MAX_TRACERS] = {0};
    s->tracer_buf.write(zeros, 4 * MAX_TRACERS);
  }
  if (s->bridger_buf.valid()) seed_bridgers(s->bridger_buf, s->p_count);
  s->initialized = true;
  state::setOnStateReady(&on_state_ready);
}

void tick(void* self, double dt) { (void)self; (void)dt; }
void on_resolume_param(void*, long long, double) {}

void on_state_patched(void* self, int n, const char* pb, const int* off,
                      const int* len, const int* ops) {
  auto* s = static_cast<State*>(self);
  if (!s) return;
  for (int i = 0; i < n; i++) {
    if (ops[i] != state::PatchReplace) continue;
    const char* p = pb + off[i]; int l = len[i];
    if      (state::pathIs(p, l, "p_count"))        s->p_count = state::patchInt(i);
    else if (state::pathIs(p, l, "motion_rate"))    s->motion_rate = state::patchFloat(i);
    else if (state::pathIs(p, l, "field_speed"))    s->field_speed = state::patchFloat(i);
    else if (state::pathIs(p, l, "momentum"))       s->momentum = state::patchFloat(i);
    else if (state::pathIs(p, l, "to_big"))         s->to_big = state::patchFloat(i);
    else if (state::pathIs(p, l, "to_big_curl"))    s->to_big_curl = state::patchFloat(i);
    else if (state::pathIs(p, l, "to_big_range"))   s->to_big_range = state::patchFloat(i);
    else if (state::pathIs(p, l, "jitter"))         s->jitter = state::patchFloat(i);
    else if (state::pathIs(p, l, "color_contrib"))  s->color_contrib = state::patchFloat(i);
    else if (state::pathIs(p, l, "to_image"))       s->to_image = state::patchFloat(i);
    else if (state::pathIs(p, l, "image_smoothing")) s->image_smoothing = state::patchFloat(i);
    else if (state::pathIs(p, l, "field_scale"))    s->field_scale = state::patchFloat(i);
    else if (state::pathIs(p, l, "field_skew"))     s->field_skew = state::patchFloat(i);
    else if (state::pathIs(p, l, "field_squash"))   s->field_squash = state::patchFloat(i);
    else if (state::pathIs(p, l, "momentum_decay")) s->momentum_decay = state::patchFloat(i);
    else if (state::pathIs(p, l, "curl_dir"))       s->curl_dir = state::patchFloat(i);
    else if (state::pathIs(p, l, "sink"))           s->sink = state::patchFloat(i);
    else if (state::pathIs(p, l, "to_image_curl"))  s->to_image_curl = state::patchFloat(i);
    else if (state::pathIs(p, l, "undertow_skew"))  s->undertow_skew = state::patchFloat(i);
    else if (state::pathIs(p, l, "undertow_squash")) s->undertow_squash = state::patchFloat(i);
    else if (state::pathIs(p, l, "boundary"))       s->boundary = state::patchFloat(i);
    else if (state::pathIs(p, l, "boundary_size"))  s->boundary_size = state::patchFloat(i);
    else if (state::pathIs(p, l, "boundary_stiffness")) s->boundary_stiffness = state::patchFloat(i);
    else if (state::pathIs(p, l, "boundary_speed")) s->boundary_speed = state::patchFloat(i);
    else if (state::pathIs(p, l, "boundary_death")) s->boundary_death = state::patchFloat(i);
    else if (state::pathIs(p, l, "ttl"))            s->ttl = state::patchFloat(i);
    else if (state::pathIs(p, l, "spawn_size"))     s->spawn_size = state::patchFloat(i);
    else if (state::pathIs(p, l, "p_point_size"))   s->p_point_size = state::patchFloat(i);
    else if (state::pathIs(p, l, "p_opacity"))      s->p_opacity = state::patchFloat(i);
    else if (state::pathIs(p, l, "p_alpha_curve"))  s->p_alpha_curve = state::patchFloat(i);
    else if (state::pathIs(p, l, "render_hue"))     s->render_hue = state::patchFloat(i);
    else if (state::pathIs(p, l, "p_shape"))        s->p_shape = state::patchInt(i);
    else if (state::pathIs(p, l, "tint"))           { auto v = state::patchVec3(i); s->tint_r = v.x; s->tint_g = v.y; s->tint_b = v.z; }
    else if (state::pathIs(p, l, "exposure"))       s->exposure = state::patchFloat(i);
    else if (state::pathIs(p, l, "big_count"))      s->big_count = state::patchInt(i);
    else if (state::pathIs(p, l, "big_speed"))      s->big_speed = state::patchFloat(i);
    else if (state::pathIs(p, l, "big_drift"))      s->big_drift = state::patchFloat(i);
    else if (state::pathIs(p, l, "big_repel"))      s->big_repel = state::patchFloat(i);
    else if (state::pathIs(p, l, "big_curl"))       s->big_curl = state::patchFloat(i);
    else if (state::pathIs(p, l, "big_momentum"))   s->big_momentum = state::patchFloat(i);
    else if (state::pathIs(p, l, "big_spread"))     s->big_spread = state::patchFloat(i);
    else if (state::pathIs(p, l, "big_ttl"))        s->big_ttl = state::patchFloat(i);
    else if (state::pathIs(p, l, "big_point_size")) s->big_point_size = state::patchFloat(i);
    else if (state::pathIs(p, l, "big_opacity"))    s->big_opacity = state::patchFloat(i);
    else if (state::pathIs(p, l, "big_render_hue")) s->big_render_hue = state::patchFloat(i);
    else if (state::pathIs(p, l, "l_count"))        s->l_count = state::patchInt(i);
    else if (state::pathIs(p, l, "spawn_on_line"))  s->spawn_on_line = state::patchFloat(i);
    else if (state::pathIs(p, l, "l_length"))       s->l_length = state::patchFloat(i);
    else if (state::pathIs(p, l, "l_opacity"))      s->l_opacity = state::patchFloat(i);
    else if (state::pathIs(p, l, "l_step_speed"))   s->l_step_speed = state::patchFloat(i);
    else if (state::pathIs(p, l, "l_momentum"))     s->l_momentum = state::patchFloat(i);
    else if (state::pathIs(p, l, "l_gradient_descent")) s->l_gradient_descent = state::patchFloat(i);
    else if (state::pathIs(p, l, "l_width"))        s->l_width = state::patchFloat(i);
    else if (state::pathIs(p, l, "l_soft"))         s->l_soft = state::patchFloat(i);
    else if (state::pathIs(p, l, "l_time_decay"))   s->l_time_decay = state::patchFloat(i);
    else if (state::pathIs(p, l, "l_adv_step"))     s->l_adv_step = state::patchFloat(i);
    else if (state::pathIs(p, l, "l_reseed_spread")) s->l_reseed_spread = state::patchFloat(i);
    else if (state::pathIs(p, l, "l_value_stop"))    s->l_value_stop = state::patchFloat(i);
    else if (state::pathIs(p, l, "l_grad_stop"))     s->l_grad_stop = state::patchFloat(i);
    else if (state::pathIs(p, l, "l_time_stop_decay")) s->l_time_stop_decay = state::patchFloat(i);
    else if (state::pathIs(p, l, "bridger_count"))  s->bridger_count = state::patchInt(i);
    else if (state::pathIs(p, l, "bridger_rate"))   s->bridger_rate = state::patchFloat(i);
    else if (state::pathIs(p, l, "bridger_motion")) s->bridger_motion = state::patchFloat(i);
    else if (state::pathIs(p, l, "bridger_opacity")) s->bridger_opacity = state::patchFloat(i);
    else if (state::pathIs(p, l, "bridger_width"))  s->bridger_width = state::patchFloat(i);
    else if (state::pathIs(p, l, "bridger_soft"))   s->bridger_soft = state::patchFloat(i);
    else if (state::pathIs(p, l, "bridger_hue"))    s->bridger_hue = state::patchFloat(i);
    else if (state::pathIs(p, l, "bridger_color_contrib")) s->bridger_color_contrib = state::patchFloat(i);
    else if (state::pathIs(p, l, "interactions"))   { bool v = state::patchBool(i); if (v != s->interactions) { s->interactions = v; apply_interaction_visibility(v); } }
    else if (state::pathIs(p, l, "interaction_radius")) s->interaction_radius = state::patchFloat(i);
    else if (state::pathIs(p, l, "density_threshold")) s->density_threshold = state::patchFloat(i);
    else if (state::pathIs(p, l, "density_death"))  s->density_death = state::patchFloat(i);
    else if (state::pathIs(p, l, "avoid"))          s->avoid = state::patchFloat(i);
    else if (state::pathIs(p, l, "avoid_curl"))     s->avoid_curl = state::patchFloat(i);
    else if (state::pathIs(p, l, "avoid_noise"))    s->avoid_noise = state::patchFloat(i);
    else if (state::pathIs(p, l, "stream"))         s->stream = state::patchFloat(i);
    else if (state::pathIs(p, l, "stream_density")) s->stream_density = state::patchFloat(i);
    else if (state::pathIs(p, l, "debug_density"))  s->debug_density = state::patchBool(i);
    else if (state::pathIs(p, l, "motion_line_speed")) s->motion_line_speed = state::patchFloat(i);
    else if (state::pathIs(p, l, "motion_particle_scale")) s->motion_particle_scale = state::patchFloat(i);
    else if (state::pathIs(p, l, "blend_mode"))     s->blend_mode = state::patchInt(i);
    else if (state::pathIs(p, l, "input_alpha"))    s->input_alpha = state::patchFloat(i);
  }
  if (s->l_count < 0) s->l_count = 0; if (s->l_count > MAX_TRACERS) s->l_count = MAX_TRACERS;
  if (s->p_count < 1) s->p_count = 1; if (s->p_count > MAX_P) s->p_count = MAX_P;
  if (s->big_count < 0) s->big_count = 0; if (s->big_count > MAX_BIG) s->big_count = MAX_BIG;
  if (s->bridger_count < 0) s->bridger_count = 0; if (s->bridger_count > MAX_BRIDGERS) s->bridger_count = MAX_BRIDGERS;
  // Lazily seed slots newly brought into play by a count increase.
  if (s->initialized && s->p_count > s->inited_count) {
    seed_pool(s->p_buf, s->inited_count, s->p_count, s->ttl * 8.0f, 0x1111u);
    s->inited_count = s->p_count;
  }
}

void render(void* self, int vp_w, int vp_h) {
  auto* s = static_cast<State*>(self);
  if (!s || !s->initialized || vp_w <= 0 || vp_h <= 0) return;
  auto out = gpu::Device::textureForField("tex_out");
  if (!out.valid()) return;
  // tex_in is optional — a true source renders on black when nothing is wired.
  auto in = gpu::Device::textureForField("tex_in");
  bool has_in = in.valid();
  auto sample_tex = has_in ? in : s->black_tex;
  if (!sample_tex.valid()) return;

  s->frame_index++;
  float dt = (float)host::deltaTime();
  float min_dim = float(vp_w < vp_h ? vp_w : vp_h);
  float ax = min_dim / float(vp_w), ay = min_dim / float(vp_h);

  // Interactions: a persistent density buffer the P update reads (built from
  // LAST frame's particles → 1-frame delay). When off, bind a 1×1 zero buffer
  // so the update's texture slot is always valid and the read costs nothing.
  bool ix = s->interactions;
  gpu::Texture density_in;
  if (ix) {
    if (!s->density_tex.valid()) {
      s->density_tex = gpu::Device::createTexture(DENSITY_RES, DENSITY_RES,
                                                  gpu::TextureFormat::RGBA16F);
      gpu::Device::clear(s->density_tex, 0.f, 0.f, 0.f, 0.f);   // first read = empty
    }
    if (!s->density_tex.valid()) ix = false;
  }
  if (ix) {
    density_in = s->density_tex;
  } else {
    if (!s->zero_density_tex.valid()) {
      s->zero_density_tex = gpu::Device::createTexture(1, 1, gpu::TextureFormat::RGBA16F);
      gpu::Device::clear(s->zero_density_tex, 0.f, 0.f, 0.f, 0.f);
    }
    density_in = s->zero_density_tex;
  }

  // Image smoothing: blur the input into field_tex so the steering gradients
  // read broad structure, not per-pixel noise. Only when image steering is
  // actually active (else the blur is wasted and we keep sharp colour capture).
  bool image_coupled = s->to_image != 0.0f || s->to_image_curl != 0.0f
                       || s->big_repel != 0.0f || s->big_curl != 0.0f;
  if (has_in && image_coupled && s->image_smoothing > 0.001f && s_blur.valid()) {
    if (!s->field_tex.valid() || s->field_w != vp_w || s->field_h != vp_h) {
      s->field_tex.release();
      s->field_tex = gpu::Device::createTexture(vp_w, vp_h);
      s->field_w = vp_w; s->field_h = vp_h;
    }
    if (s->field_tex.valid()) {
      s_blur.applyWithRadius(in, s->field_tex, vp_w, vp_h, s->image_smoothing, 1.0f);
      sample_tex = s->field_tex;
    }
  }

  BigUpdateUniforms bu = {};
  bu.count = (uint32_t)s->big_count; bu.frame_index = s->frame_index; bu.dt = dt; bu.motion_rate = s->motion_rate;
  bu.big_speed = s->big_speed; bu.big_momentum = s->big_momentum; bu.big_momentum_decay = s->big_momentum_decay; bu.drift = s->big_drift;
  bu.repel = s->big_repel; bu.direction = 1.0f; bu.curl = s->big_curl; bu.curl_dir = s->curl_dir;
  bu.sink = s->sink; bu.boundary = s->boundary; bu.boundary_size = s->boundary_size; bu.boundary_stiffness = s->boundary_stiffness;
  bu.boundary_speed = s->boundary_speed; bu.spread = s->big_spread; bu.ttl = s->big_ttl; bu.aspect_x = ax; bu.aspect_y = ay;
  bu.image_smoothing = s->image_smoothing;
  s->big_uniform.writeOne(bu);

  PUpdateUniforms pu = {};
  pu.count = (uint32_t)s->p_count; pu.big_count = (uint32_t)s->big_count; pu.frame_index = s->frame_index; pu.dt = dt;
  pu.motion_rate = s->motion_rate; pu.momentum = s->momentum; pu.momentum_decay = s->momentum_decay; pu.field_speed = s->field_speed;
  pu.field_scale = s->field_scale; pu.field_skew = s->field_skew; pu.field_squash = s->field_squash; pu.jitter = s->jitter;
  pu.to_big = s->to_big; pu.to_big_curl = s->to_big_curl; pu.curl_dir = s->curl_dir; pu.sink = s->sink;
  pu.boundary = s->boundary; pu.boundary_size = s->boundary_size; pu.boundary_stiffness = s->boundary_stiffness; pu.boundary_speed = s->boundary_speed;
  pu.to_image = s->to_image; pu.to_image_curl = s->to_image_curl; pu.undertow_skew = s->undertow_skew; pu.undertow_squash = s->undertow_squash;
  pu.ttl = s->ttl; pu.spawn_size = s->spawn_size; pu.aspect_x = ax; pu.aspect_y = ay;
  pu.to_big_range = s->to_big_range;
  pu.image_smoothing = s->image_smoothing;
  int seg_total = (s->l_count > 0) ? s->l_count * MAX_SEG : 0;
  pu.to_line_rate = (seg_total > 0) ? s->spawn_on_line : 0.0f;
  pu.seg_total = (float)seg_total;
  // Spawn-on-line needs the segment buffer's BLOCK geometry, not just its size:
  // each tracer owns [i*MAX_SEG, i*MAX_SEG + live), and only `live` of those
  // slots are written (the rest are zeroed). Mirrors trace.hlsl's `steps`.
  float l01 = s->l_length < 0.0f ? 0.0f : (s->l_length > 1.0f ? 1.0f : s->l_length);
  int l_steps = (int)(l01 * (float)(MAX_SEG / 2));
  if (l_steps < 2) l_steps = 2;
  pu.l_count_f = (float)s->l_count;
  pu.seg_stride = (float)MAX_SEG;
  pu.seg_live = (float)(2 * l_steps > MAX_SEG ? MAX_SEG : 2 * l_steps);
  pu.boundary_death = s->boundary_death;
  pu.interactions = ix ? 1u : 0u;
  pu.density_threshold = s->density_threshold;
  pu.density_death = s->density_death;
  pu.avoid = s->avoid;
  pu.avoid_curl = s->avoid_curl;
  pu.avoid_noise = s->avoid_noise;
  pu.stream = s->stream;
  pu.stream_density = s->stream_density;
  pu.density_res = (float)DENSITY_RES;
  s->p_uniform.writeOne(pu);

  PrefillUniforms pf = { s->input_alpha, s->input_alpha, s->input_alpha, 1.0f };
  s->prefill_uniform.writeOne(pf);

  VsUniforms vp = { ax, ay, s->p_point_size * POINT_SIZE_SCALE, 0.f }; s->vs_uniform_p.writeOne(vp);
  VsUniforms vb = { ax, ay, s->big_point_size, 0.f }; s->vs_uniform_big.writeOne(vb);

  FsUniforms fp = {};
  fp.color_contrib = s->color_contrib; fp.render_hue = s->render_hue; fp.opacity = s->p_opacity; fp.alpha_curve = s->p_alpha_curve;
  fp.tint_r = s->tint_r; fp.tint_g = s->tint_g; fp.tint_b = s->tint_b; fp.exposure = s->exposure;
  fp.shape_kind = (uint32_t)s->p_shape; fp.shape_param = 0.5f;
  s->fs_uniform_p.writeOne(fp);

  FsUniforms fb = fp;
  fb.render_hue = s->big_render_hue; fb.opacity = s->big_opacity; fb.shape_kind = 1u; // gaussian
  s->fs_uniform_big.writeOne(fb);

  TraceUniforms tu = {};
  tu.count = (uint32_t)s->l_count; tu.max_seg = (uint32_t)MAX_SEG; tu.frame_index = s->frame_index; tu.dt = dt;
  tu.field_scale = s->field_scale; tu.field_skew = s->field_skew; tu.field_squash = s->field_squash; tu.field_speed = s->field_speed;
  tu.to_image = s->to_image; tu.momentum = s->l_momentum; tu.step_speed = s->l_step_speed; tu.length01 = s->l_length;
  tu.time_decay = s->l_time_decay; tu.adv_step = s->l_adv_step; tu.color_contrib = s->color_contrib; tu.l_opacity = s->l_opacity;
  tu.aspect_x = ax; tu.aspect_y = ay; tu.tint_r = s->tint_r; tu.tint_g = s->tint_g;
  tu.tint_b = s->tint_b; tu.reseed_spread = s->l_reseed_spread; tu.image_smoothing = s->image_smoothing;
  tu.gradient_descent = s->l_gradient_descent;
  tu.value_stop = s->l_value_stop; tu.grad_stop = s->l_grad_stop; tu.time_stop_decay = s->l_time_stop_decay;
  s->trace_uniform.writeOne(tu);

  LineVsUniforms lv = { ax, ay, s->l_width * LINE_WIDTH_SCALE, 0.f };
  s->line_vs_uniform.writeOne(lv);
  LineFsUniforms lf = { s->l_soft, 0.f, 0.f, 0.f };
  s->line_fs_uniform.writeOne(lf);

  BridgerUniforms br = {};
  br.count = (uint32_t)s->bridger_count; br.p_count = (uint32_t)s->p_count;
  br.frame_index = s->frame_index; br.dt = dt;
  br.rate = s->bridger_rate; br.motion = s->bridger_motion;
  br.opacity = s->bridger_opacity; br.color_contrib = s->bridger_color_contrib;
  br.hue = s->bridger_hue; br.tint_r = s->tint_r; br.tint_g = s->tint_g; br.tint_b = s->tint_b;
  s->bridger_uniform.writeOne(br);
  LineVsUniforms bv = { ax, ay, s->bridger_width * LINE_WIDTH_SCALE, 0.f };
  s->bridger_vs_uniform.writeOne(bv);
  LineFsUniforms bf = { s->bridger_soft, 0.f, 0.f, 0.f };
  s->bridger_fs_uniform.writeOne(bf);

  // Pass 1 — Big update (must precede P so P reads fresh attractors).
  if (s->big_count > 0) {
    auto cp = gpu::ComputePass::begin();
    cp.setPSO(s_pso_big_update);
    cp.setBuffer(s->big_buf, 0);
    cp.setTexture(sample_tex, 1, 0);
    cp.setSampler(s->sampler, 2);
    cp.setBuffer(s->big_uniform, 3);
    cp.dispatch((s->big_count + 63) / 64, 1, 1);
    cp.end();
  }

  // Pass 1b — tracers (before P so P can spawn onto the current lines).
  if (s->l_count > 0) {
    auto cp = gpu::ComputePass::begin();
    cp.setPSO(s_pso_trace);
    cp.setBuffer(s->tracer_buf, 0);
    cp.setBuffer(s->seg_buf, 1);
    cp.setTexture(sample_tex, 2, 0);
    cp.setSampler(s->sampler, 3);
    cp.setBuffer(s->trace_uniform, 4);
    cp.dispatch((s->l_count + 63) / 64, 1, 1);
    cp.end();
  }

  // Pass 2 — P update.
  {
    auto cp = gpu::ComputePass::begin();
    cp.setPSO(s_pso_p_update);
    cp.setBuffer(s->p_buf, 0);
    cp.setBuffer(s->big_buf, 1);
    cp.setTexture(sample_tex, 2, 0);
    cp.setSampler(s->sampler, 3);
    cp.setBuffer(s->p_uniform, 4);
    cp.setBuffer(s->seg_buf, 5);
    cp.setTexture(density_in, 6, 0);
    cp.dispatch((s->p_count + 63) / 64, 1, 1);
    cp.end();
  }

  // Pass 2b — bridgers (after P so endpoints track the freshly-updated
  // particle positions). Emits one Seg per bridger into bridger_seg_buf.
  if (s->bridger_count > 0) {
    auto cp = gpu::ComputePass::begin();
    cp.setPSO(s_pso_bridger);
    cp.setBuffer(s->bridger_buf, 0);
    cp.setBuffer(s->p_buf, 1);
    cp.setBuffer(s->bridger_seg_buf, 2);
    cp.setBuffer(s->bridger_uniform, 3);
    cp.dispatch((s->bridger_count + 63) / 64, 1, 1);
    cp.end();
  }

  // Pass 3 — base: composite over the input, or clear to black (no input).
  if (has_in) {
    auto cp = gpu::ComputePass::begin();
    cp.setPSO(s_pso_prefill);
    cp.setTexture(in, 0, 0);
    cp.setTexture(out, 1, 1);
    cp.setBuffer(s->prefill_uniform, 2);
    cp.dispatch((vp_w + 7) / 8, (vp_h + 7) / 8);
    cp.end();
  } else {
    gpu::Device::clear(out, 0.f, 0.f, 0.f, 1.f);
  }

  // Pass 4 — render P points then Big points.
  auto pso = (s->blend_mode == BLEND_ALPHA) ? s_pso_render_alpha : s_pso_render_add;
  {
    auto rp = gpu::RenderPass::beginLoad(out);
    rp.setPSO(pso);
    rp.setBuffer(s->p_buf, 0);
    rp.setBuffer(s->vs_uniform_p, 1);
    rp.setBuffer(s->fs_uniform_p, 2);
    rp.draw(6, s->p_count);
    rp.end();
  }
  if (s->big_count > 0 && s->big_opacity > 0.0f) {
    auto rp = gpu::RenderPass::beginLoad(out);
    rp.setPSO(pso);
    rp.setBuffer(s->big_buf, 0);
    rp.setBuffer(s->vs_uniform_big, 1);
    rp.setBuffer(s->fs_uniform_big, 2);
    rp.draw(6, s->big_count);
    rp.end();
  }

  // Pass 5 — tracer lines (one instanced quad per segment slot).
  if (s->l_count > 0 && s->l_opacity > 0.0f) {
    auto lpso = (s->blend_mode == BLEND_ALPHA) ? s_pso_line_alpha : s_pso_line_add;
    auto rp = gpu::RenderPass::beginLoad(out);
    rp.setPSO(lpso);
    rp.setBuffer(s->seg_buf, 0);
    rp.setBuffer(s->line_vs_uniform, 1);
    rp.setBuffer(s->line_fs_uniform, 2);
    rp.draw(6, s->l_count * MAX_SEG);
    rp.end();
  }

  // Pass 6 — bridger chords (reuse the line PSO; one quad per bridger).
  if (s->bridger_count > 0 && s->bridger_opacity > 0.0f) {
    auto lpso = (s->blend_mode == BLEND_ALPHA) ? s_pso_line_alpha : s_pso_line_add;
    auto rp = gpu::RenderPass::beginLoad(out);
    rp.setPSO(lpso);
    rp.setBuffer(s->bridger_seg_buf, 0);
    rp.setBuffer(s->bridger_vs_uniform, 1);
    rp.setBuffer(s->bridger_fs_uniform, 2);
    rp.draw(6, s->bridger_count);
    rp.end();
  }

  // Pass 6b — density splat (after p_update moved them) → next frame's read.
  // RenderPass::begin clears the buffer to zero, then additive halos accumulate.
  // The splat is the expensive part of interactions (a draw over the whole pool,
  // with overdraw), so only run it when something actually READS the buffer this
  // frame — death, avoidance, streaming, or the debug view. Interactions-on-but-
  // unused then costs nothing extra.
  bool debug = ix && s->debug_density;
  bool need_density = ix && (s->density_death > 0.0f || s->avoid > 0.0f ||
                             s->stream != 0.0f || s->debug_density);
  if (need_density) {
    DensityUniforms du = { s->interaction_radius, ax, ay, 0.f };
    s->density_uniform.writeOne(du);
    auto rp = gpu::RenderPass::begin(s->density_tex, 0.f, 0.f, 0.f, 0.f);
    rp.setPSO(s_pso_density);
    rp.setBuffer(s->p_buf, 0);
    rp.setBuffer(s->density_uniform, 1);
    rp.draw(6, s->p_count);
    rp.end();
  }

  // Pass 6c (debug) — heat-map this frame's density buffer over tex_out.
  if (debug) {
    auto cp = gpu::ComputePass::begin();
    cp.setPSO(s_pso_density_debug);
    cp.setTexture(s->density_tex, 0, 0);
    cp.setSampler(s->sampler, 1);
    cp.setTexture(out, 2, 1);
    cp.dispatch((vp_w + 7) / 8, (vp_h + 7) / 8);
    cp.end();
  }

  // Pass 7 — motion vectors (render_outputs/motion). Only run when a downstream
  // SINK reads the rail; pure overhead otherwise. Particles emit their
  // per-frame integrated velocity; lines emit a tangent-along-segment velocity.
  // All footprints compose (AlphaOver) over an upstream-seeded motion field.
  if (state::isOutputConnected("render_outputs")) {
    if (!s->motion_tex.valid() || s->motion_w != vp_w || s->motion_h != vp_h) {
      s->motion_tex.release();
      s->motion_tex = gpu::Device::createTexture(vp_w, vp_h, gpu::TextureFormat::RGBA16F);
      s->motion_w = vp_w; s->motion_h = vp_h;
      if (s->motion_tex.valid()) state::setGpuTexture("render_outputs/motion", s->motion_tex.id);
    }
    if (s->motion_tex.valid()) {
      // Seed from upstream motion (1×1 zero fallback → effectively a clear).
      auto up = gpu::Device::textureForField("render_outputs_in/motion");
      if (!up.valid()) {
        if (!s->zero_motion_tex.valid())
          s->zero_motion_tex = gpu::Device::createTexture(1, 1, gpu::TextureFormat::RGBA16F);
        up = s->zero_motion_tex;
      }
      if (up.valid()) {
        auto cp = gpu::ComputePass::begin();
        cp.setPSO(s_pso_motion_prefill);
        cp.setTexture(up, 0, 0);
        cp.setTexture(s->motion_tex, 1, 1);
        cp.dispatch((vp_w + 7) / 8, (vp_h + 7) / 8);
        cp.end();
      }

      MotionVsUniforms mvp = { ax, ay, s->p_point_size * POINT_SIZE_SCALE, dt, s->motion_rate, s->motion_particle_scale, 0.f, 0.f };
      s->motion_vs_uniform_p.writeOne(mvp);
      MotionVsUniforms mvb = { ax, ay, s->big_point_size, dt, s->motion_rate, s->motion_particle_scale, 0.f, 0.f };
      s->motion_vs_uniform_big.writeOne(mvb);
      MotionFsUniforms mfu = { (uint32_t)s->p_shape, 0.5f, 0.f, 0.f };
      s->motion_fs_uniform.writeOne(mfu);
      float line_spd = s->motion_line_speed * MOTION_LINE_SCALE;
      LineMotionVsUniforms lml = { ax, ay, s->l_width * LINE_WIDTH_SCALE, line_spd };
      s->lm_vs_uniform_l.writeOne(lml);
      LineMotionVsUniforms lmb = { ax, ay, s->bridger_width * LINE_WIDTH_SCALE, line_spd };
      s->lm_vs_uniform_b.writeOne(lmb);

      // Motion mirrors VISIBILITY: an element only writes motion when it's
      // actually drawn in the colour pass (same opacity gates). Otherwise an
      // invisible element (e.g. big_opacity == 0) would still overwrite the
      // particle motion in its footprint — reading as "holes" in the motion
      // field once a downstream motion blur smears around them.
      bool draw_p   = s->p_opacity > 0.0f;
      bool draw_big = s->big_count > 0 && s->big_opacity > 0.0f;
      bool draw_l   = line_spd > 0.0f && s->l_count > 0 && s->l_opacity > 0.0f;
      bool draw_b   = line_spd > 0.0f && s->bridger_count > 0 && s->bridger_opacity > 0.0f;

      auto rp = gpu::RenderPass::beginLoad(s->motion_tex);
      if (draw_p || draw_big) {
        rp.setPSO(s_pso_motion_point);
        rp.setBuffer(s->motion_fs_uniform, 2);
        if (draw_p) {
          rp.setBuffer(s->p_buf, 0);
          rp.setBuffer(s->motion_vs_uniform_p, 1);
          rp.draw(6, s->p_count);
        }
        if (draw_big) {
          rp.setBuffer(s->big_buf, 0);
          rp.setBuffer(s->motion_vs_uniform_big, 1);
          rp.draw(6, s->big_count);
        }
      }
      if (draw_l || draw_b) {
        rp.setPSO(s_pso_motion_line);
        if (draw_l) {
          rp.setBuffer(s->seg_buf, 0);
          rp.setBuffer(s->lm_vs_uniform_l, 1);
          rp.setBuffer(s->line_fs_uniform, 2);   // reuse tracer soft
          rp.draw(6, s->l_count * MAX_SEG);
        }
        if (draw_b) {
          rp.setBuffer(s->bridger_seg_buf, 0);
          rp.setBuffer(s->lm_vs_uniform_b, 1);
          rp.setBuffer(s->bridger_fs_uniform, 2);  // reuse bridger soft
          rp.draw(6, s->bridger_count);
        }
      }
      rp.end();
    }
  }

  gpu::Device::submit();
}

} // namespace double_chamber
