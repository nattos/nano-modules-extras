/*
 * source.particles.flash_particles — Mask-driven particle compositor.
 *
 * Maintains a GPU-resident pool of up to MAX_PARTICLES particles. On
 * spawn, a compute shader random-samples the (optional) mask texture
 * for the brightest of K candidates, captures the input color at that
 * uv, and re-rolls per-particle geometry / color / alpha jitters from
 * the current uniforms. While alive, particles render as oriented
 * quads via vertex/fragment instanced draw — one storage-buffered
 * particle per quad, the VS reads it, the FS shapes it (solid /
 * squircle / gaussian) and outputs color * alpha.
 *
 * Composition modes (alpha-over vs additive) are TWO PSOs against the
 * SAME fragment shader, picked at dispatch time by the user's
 * blend_mode param. The render pass loads the pre-filled framebuffer
 * (`tex_in × input_alpha`) so the blend equation does the right thing
 * on top of the dimmed input.
 *
 * Motion vectors share the same VS but a different FS that emits
 * `(velocity, 0, mask)`. Pre-filled with upstream render_outputs/motion
 * (or zero when no upstream is wired); the alpha-over blend then acts
 * as a mask-controlled lerp — covered pixels get the particle's motion,
 * uncovered keep upstream.
 *
 * Aspect: width/height are stored in "isotropic uv" — one unit
 * corresponds to min(W, H) pixels. So size.x == size.y always renders
 * as a real pixel square, regardless of viewport aspect.
 *
 * Class-like instance model: module_init() compiles the shared
 * compute/render PSOs + publishes the schema once per type; each chain
 * entry gets its own State (params, particle pool buffer, per-instance
 * uniform buffers/textures) via create(). All instance callbacks take
 * `self`.
 */

#include <gpu.h>
#include <host.h>
#include "flash_particles_shaders.h"

#include <cmath>
#include <cstdint>
#include <cstring>

namespace flash_particles {

// Hard cap. The GPU buffer is sized for the maximum so the user can
// dial `count` up at runtime without us reallocating; only fresh
// (count - prev_count) slots are seeded.
static constexpr int   MAX_PARTICLES   = 100000;
static constexpr float DEG2RAD         = 3.14159265358979323846f / 180.0f;

// 5 vec4 = 80 bytes per particle. Mirror of `Particle` in common.hlsl.
struct GpuParticle {
  float pos_size[4];
  float captured[4];
  float state[4];
  float jitters[4];
  float meta[4];
};
static_assert(sizeof(GpuParticle) == 80, "Particle GPU struct must be 80 bytes");

struct UpdateUniforms {
  // row 0
  uint32_t count;
  uint32_t frame_index;
  float    dt;
  float    mask_temperature;  // 0 = argmax (greedy); higher = softmax(luma / T)

  // row 1 — lifetime
  float    life;
  float    respawn_delay;
  float    life_jitter;
  float    _pad_l0;

  // row 2 — geometry
  float    width;
  float    height;
  float    global_scale;
  float    width_jitter;

  // row 3 — geometry cont.
  float    height_jitter;
  float    rotation_rad;
  float    rotation_jitter_rad;
  float    _pad_g0;

  // row 4 — color jitters
  float    hue_jitter;
  float    brightness_jitter;
  float    saturation_jitter;
  float    alpha_jitter;
};
static_assert(sizeof(UpdateUniforms) == 80, "UpdateUniforms layout mismatch");

struct PrefillUniforms {
  // Per-channel scale applied to the source tex during the copy. For
  // the color pass we use (input_alpha, input_alpha, input_alpha, 1);
  // for the motion pass (1, 1, 1, 1).
  float scale_r;
  float scale_g;
  float scale_b;
  float scale_a;
};
static_assert(sizeof(PrefillUniforms) == 16, "PrefillUniforms layout mismatch");

struct VsUniforms {
  // Aspect conversion factors: isotropic-uv → true-uv on each axis.
  // For W==H both are 1; otherwise the longer axis gets compressed
  // so a particle with size.x == size.y renders as a pixel square.
  float aspect_x;
  float aspect_y;
  float _pad0;
  float _pad1;
};
static_assert(sizeof(VsUniforms) == 16, "VsUniforms layout mismatch");

struct ColorUniforms {
  // row 0 — composite + color blend
  float    input_alpha;
  float    color_blend;
  float    global_color_r;
  float    global_color_g;

  // row 1 — color cont. + curve
  float    global_color_b;
  float    alpha_curve;
  float    frame_alpha_jitter;
  uint32_t frame_index;

  // row 2 — shape + exposure/color-alpha gate
  uint32_t shape_kind;
  float    shape_param;
  float    exposure;     // RGB multiplier (alpha untouched; clips to white)
  float    color_alpha;  // global multiplier on final color alpha
};
static_assert(sizeof(ColorUniforms) == 48, "ColorUniforms layout mismatch");

struct MotionUniforms {
  float    motion_strength;
  uint32_t shape_kind;
  float    shape_param;
  float    alpha_curve;  // mirrors ColorUniforms.alpha_curve — applied to motion magnitude
};
static_assert(sizeof(MotionUniforms) == 16, "MotionUniforms layout mismatch");

enum BlendMode : int { BLEND_ALPHA = 0, BLEND_ADD = 1 };
enum ShapeKind : int { SHAPE_SOLID = 0, SHAPE_CIRCLE = 1, SHAPE_GAUSSIAN = 2 };

// Type-shared: compiled once in module_init().
static gpu::ComputePSO s_pso_update;
static gpu::ComputePSO s_pso_prefill_color;
static gpu::ComputePSO s_pso_prefill_motion;
static gpu::RenderPSO  s_pso_render_alpha;   // color, alpha-over blend
static gpu::RenderPSO  s_pso_render_add;     // color, additive blend
static gpu::RenderPSO  s_pso_render_motion;  // motion, alpha-over (mask-as-blend-alpha)

// Per-instance state. One per chain entry.
struct State {
  // Per-instance GPU buffers.
  gpu::Buffer  particle_buf;
  gpu::Buffer  update_uniforms;
  gpu::Buffer  prefill_color_uniforms;
  gpu::Buffer  prefill_motion_uniforms;
  gpu::Buffer  vs_uniforms;
  gpu::Buffer  color_uniforms;
  gpu::Buffer  motion_uniforms;
  gpu::Sampler sampler;

  // Per-instance, viewport-sized motion target (+ 1×1 zero fallback).
  gpu::Texture motion_tex;
  gpu::Texture zero_motion_tex;  // 1×1 fallback when no upstream
  int  motion_w = 0;
  int  motion_h = 0;

  bool initialized = false;

  // CPU mirrors of all schema params.
  int   count               = 64;
  float life                = 1.5f;
  float respawn_delay       = 0.3f;
  float life_jitter         = 0.3f;
  float width               = 0.05f;
  float height              = 0.05f;
  float width_jitter        = 0.3f;
  float height_jitter       = 0.3f;
  float global_scale        = 1.0f;
  float rotation_deg        = 0.0f;
  float rotation_jitter_deg = 30.0f;
  int   shape_kind          = SHAPE_GAUSSIAN;
  float shape_param         = 0.5f;
  float alpha_curve         = 1.5f;
  float frame_alpha_jitter  = 0.0f;
  float global_color_r      = 1.0f;
  float global_color_g      = 1.0f;
  float global_color_b      = 1.0f;
  float color_blend         = 0.5f;
  float hue_jitter          = 0.1f;
  float brightness_jitter   = 0.1f;
  float saturation_jitter   = 0.1f;
  float alpha_jitter        = 0.1f;
  int   blend_mode          = BLEND_ALPHA;
  float input_alpha         = 1.0f;
  float motion_strength     = 0.5f;
  float exposure            = 1.0f;
  float color_alpha         = 1.0f;
  float mask_temperature    = 0.0f;

  // Spawn/seed accumulators.
  int      inited_count = 0;
  uint32_t frame_index  = 0;
  uint32_t init_lcg     = 0x12345678u;
};

static inline uint32_t lcg_next(uint32_t& s) {
  s = s * 1664525u + 1013904223u;
  return s;
}
static inline float lcg_unit(uint32_t& s) {
  return (lcg_next(s) >> 8) * (1.0f / float(1u << 24));
}

// We chunk seed_initial_slots into INIT_CHUNK-sized writes so a 100k-
// particle pool doesn't blow the wasm stack with an 80*100000 = 8 MB
// local array. 256 entries × 80 bytes = 20 KB per chunk fits well
// under any reasonable stack budget.
static constexpr int INIT_CHUNK = 256;

// Seed slots [from, to) with a "ready to respawn" state, randomised
// across [0, life + respawn] so the pool spawns staggered rather than
// in lockstep on the first frame.
static void seed_initial_slots(State& s, int from, int to) {
  if (s.initialized == false) return;
  if (from >= to) return;
  GpuParticle entries[INIT_CHUNK];
  float cycle = s.life + s.respawn_delay;
  for (int chunk_start = from; chunk_start < to; chunk_start += INIT_CHUNK) {
    int chunk_end = chunk_start + INIT_CHUNK;
    if (chunk_end > to) chunk_end = to;
    int n = chunk_end - chunk_start;
    for (int i = 0; i < n; i++) {
      GpuParticle& p = entries[i];
      std::memset(&p, 0, sizeof(p));
      p.pos_size[2] = 1e-3f;        // tiny size avoids div-by-zero before first respawn
      p.pos_size[3] = 1e-3f;
      p.state[2]    = 1.0f;          // life_total placeholder
      p.state[1]    = 0.0f;          // life_remain (start invisible)
      p.state[3]    = lcg_unit(s.init_lcg) * cycle;  // staggered respawn
    }
    s.particle_buf.writeBytes(
        entries, int(sizeof(GpuParticle)) * n,
        int(sizeof(GpuParticle)) * chunk_start);
  }
}

static void apply_count_change(State& s) {
  if (s.count > MAX_PARTICLES) s.count = MAX_PARTICLES;
  if (s.count < 1)             s.count = 1;
  if (s.count > s.inited_count) {
    seed_initial_slots(s, s.inited_count, s.count);
    s.inited_count = s.count;
  }
}

// Type-level setup: schema + shared compute/render PSOs.
void module_init() {
  state::init("source.particles.flash_particles", {1, 0, 1},
    state::Schema()
      // Top-level manual: high-level "what is this / how to use / what to try".
      .helpField("intro",
        "## Flash Particles\n"
        "A **mask-driven particle compositor**. It keeps a GPU pool of "
        "short-lived particles, spawning each where a **mask** is brightest and "
        "capturing the underlying image colour there. Particles live, fade on an "
        "alpha curve, then respawn — a continuous shimmer keyed to image "
        "content, composited straight over the input.\n\n"
        "**Try:** wire a *mask* input (a luma, edge, or motion pass) to steer "
        "where particles land — with none wired it falls back to the image "
        "itself. Push *Count* up and *Lifetime* down for a fast sparkle, dial "
        "*Global Color* + *Color Blend* to tint between captured and chosen "
        "colour, and switch *Composite* to **Add** for glowing light bursts.")
      // ---- Pool ----
      .group("pool", "Pool")
      .intField  ("count",            64,    1,    MAX_PARTICLES, state::PrimaryInput).label("Count", "Count")
      // ---- Lifetime ----
      .group("lifetime", "Lifetime & Spawn")
        .groupHelp(
          "How long each particle lives and how it respawns. *Lifetime* is the "
          "visible span; *Respawn Delay* is the dark gap before a slot fires "
          "again; *Lifetime Jitter* randomizes per particle so they don't pulse "
          "in lockstep. *Mask Temperature* controls spawn adherence: 0 always "
          "picks the brightest mask candidate; higher values scatter toward "
          "darker areas (very high approaches uniform random).")
      .floatField("life",             1.5f,  0.05f, 10.0f,        state::PrimaryInput).label("Lifetime", "Life")
      .floatField("respawn_delay",    0.3f,  0.0f,  10.0f,        state::PrimaryInput).label("Respawn Delay", "Respwn")
      .floatField("life_jitter",      0.3f,  0.0f,  1.0f,         state::PrimaryInput).label("Lifetime Jitter", "L Jit")
      // Temperature on the mask-sample softmax. 0 = greedy / argmax
      // (always pick the brightest of K candidates — sharpest
      // adherence to the mask). Increasing it spreads the spawn
      // probability across darker candidates too; very large values
      // → uniform random.
      .floatField("mask_temperature", 0.0f,  0.0f,  4.0f,         state::PrimaryInput).label("Mask Temperature", "M Temp")
      // ---- Geometry (isotropic uv — equal w/h is a pixel square) ----
      .group("geometry", "Geometry")
        .groupHelp(
          "Particle size and orientation. Sizes are in *isotropic uv* — equal "
          "*Width* and *Height* always render as a true pixel square regardless "
          "of viewport aspect. *Global Scale* multiplies everything at once; the "
          "*Jitter* knobs (width / height / rotation) randomize each particle so "
          "the spray looks organic rather than stamped.")
      .floatField("width",            0.05f, 0.001f, 1.0f,        state::PrimaryInput).label("Width", "Width")
      .floatField("height",           0.05f, 0.001f, 1.0f,        state::PrimaryInput).label("Height", "Height")
      .floatField("width_jitter",     0.3f,  0.0f,  1.0f,         state::PrimaryInput).label("Width Jitter", "W Jit")
      .floatField("height_jitter",    0.3f,  0.0f,  1.0f,         state::PrimaryInput).label("Height Jitter", "H Jit")
      .floatField("global_scale",     1.0f,  0.01f, 4.0f,         state::PrimaryInput).label("Global Scale", "Scale")
      .floatField("rotation",         0.0f,  -360.0f, 360.0f,     state::PrimaryInput).label("Rotation", "Rot")
      .floatField("rotation_jitter",  30.0f, 0.0f,  360.0f,       state::PrimaryInput).label("Rotation Jitter", "R Jit")
      // ---- Mask shape ----
      .group("shape", "Shape")
      .selectField("shape_kind",      SHAPE_GAUSSIAN,             state::PrimaryInput, {
        {"Solid",    SHAPE_SOLID},
        {"Circle",   SHAPE_CIRCLE},
        {"Gaussian", SHAPE_GAUSSIAN},
      }).label("Shape", "Shape")
      .floatField("shape_param",      0.5f,  0.0f,  1.0f,         state::PrimaryInput).label("Shape Param", "Param")
      // ---- Alpha decay + frame jitter ----
      .group("alpha", "Alpha")
      .floatField("alpha_curve",      1.5f,  0.25f, 4.0f,         state::PrimaryInput).label("Alpha Curve", "Curve")
      .floatField("frame_alpha_jitter", 0.0f, 0.0f, 1.0f,         state::PrimaryInput).label("Frame Alpha Jitter", "FA Jit")
      // ---- Color ----
      .group("color", "Color")
        .groupHelp(
          "Each particle can wear the colour it captured from the image at spawn "
          "or a chosen *Global Color* — *Color Blend* cross-fades between them "
          "(0 = captured, 1 = global). The *Jitter* knobs (hue / brightness / "
          "saturation / alpha) spread each particle around that colour for a "
          "livelier, less flat spray.")
      .rgbField  ("global_color",     1.0f, 1.0f, 1.0f,           state::PrimaryInput).label("Global Color", "Color")
      .floatField("color_blend",      0.5f,  0.0f,  1.0f,         state::PrimaryInput).label("Color Blend", "Blend")
      .floatField("hue_jitter",       0.1f,  0.0f,  1.0f,         state::PrimaryInput).label("Hue Jitter", "Hue")
      .floatField("brightness_jitter", 0.1f, 0.0f,  1.0f,         state::PrimaryInput).label("Brightness Jitter", "Bright")
      .floatField("saturation_jitter", 0.1f, 0.0f,  1.0f,         state::PrimaryInput).label("Saturation Jitter", "Sat")
      .floatField("alpha_jitter",     0.1f,  0.0f,  1.0f,         state::PrimaryInput).label("Alpha Jitter", "A Jit")
      // ---- Composite ----
      .group("composite", "Composite")
        .groupHelp(
          "How the particles land on the frame. **Alpha** blends over the input; "
          "**Add** accumulates light for glowing bursts. *Input Alpha* dims the "
          "underlying image (0 = particles on black), *Exposure* boosts particle "
          "intensity (clips to white), and *Color Alpha* fades the whole colour "
          "pass out — at 0 it's skipped entirely, yet the motion rail still "
          "emits for downstream effects.")
      .selectField("blend_mode",      BLEND_ALPHA,                state::PrimaryInput, {
        {"Alpha", BLEND_ALPHA},
        {"Add",   BLEND_ADD},
      }).label("Blend Mode", "Blend")
      .floatField("input_alpha",      1.0f,  0.0f,  1.0f,         state::PrimaryInput).label("Input Alpha", "In A")
      // RGB multiplier applied per-fragment after the captured/global
      // color blend. Alpha is untouched, so a >1 value boosts intensity
      // and naturally clips to white in the rgba8 surface format.
      .floatField("exposure",         1.0f,  0.0f,  8.0f,         state::PrimaryInput).label("Exposure", "Expo")
      // Global multiplier on the color render's output alpha. At 0 the
      // color raster pass is skipped entirely (motion vectors still
      // emit), letting the user fade particles out without losing the
      // motion-vector channel a downstream effect may consume.
      .floatField("color_alpha",      1.0f,  0.0f,  1.0f,         state::PrimaryInput).label("Color Alpha", "Col A")
      // ---- Motion ----
      .group("motion", "Motion")
        .groupHelp(
          "Particles can also write a **motion-vector** rail for downstream "
          "motion-blur or optical-flow effects. *Motion Strength* scales the "
          "emitted velocity; pixels no particle covers inherit the upstream "
          "motion. This pass only runs when something downstream consumes the "
          "rail.")
      .floatField("motion_strength",  0.5f,  0.0f,  1.0f,         state::PrimaryInput).label("Motion Strength", "Motion")
      // ---- Standard I/O ----
      .textureField("tex_in",  state::PrimaryInput)
      // Optional secondary mask. Falls back to tex_in C++-side when
      // no tap is wired.
      .textureField("mask_in", state::SecondaryInput)
      .textureField("tex_out", state::PrimaryOutput)
      .renderOutputs(state::PrimaryOutput)
      // Upstream render-outputs (auxiliary input). Same chaining
      // contract as the other motion writers — pixels not covered by
      // any particle inherit upstream motion.
      .renderOutputs(state::PrimaryInput, "render_outputs_in")
        .capability(state::Capability::Generator)
    );

  if (gpu::Device::backend() == gpu::Backend::None) return;

  // ---- Compute shaders ----
  state::registerShaderSPV("flash_particles_update", UPDATE_SPV, UPDATE_SPV_SIZE);
  // Same prefill SPV registered twice with different storage formats —
  // the naga-bridge substitutes the format per registration, so each
  // PSO gets the right `texture_storage_2d<...>` declaration.
  state::registerShaderSPV("flash_particles_prefill_color",  PREFILL_SPV, PREFILL_SPV_SIZE);
  state::registerShaderSPV("flash_particles_prefill_motion", PREFILL_SPV, PREFILL_SPV_SIZE,
                            "rgba16float", "write");

  // ---- Vertex / fragment shaders ----
  state::registerShaderSPV("flash_particles_vs",        VS_SPV,        VS_SPV_SIZE);
  state::registerShaderSPV("flash_particles_fs_color",  FS_COLOR_SPV,  FS_COLOR_SPV_SIZE);
  state::registerShaderSPV("flash_particles_fs_motion", FS_MOTION_SPV, FS_MOTION_SPV_SIZE);

  auto cs_update         = gpu::Device::createShaderModuleByName("flash_particles_update");
  auto cs_prefill_color  = gpu::Device::createShaderModuleByName("flash_particles_prefill_color");
  auto cs_prefill_motion = gpu::Device::createShaderModuleByName("flash_particles_prefill_motion");
  auto vs_module         = gpu::Device::createShaderModuleByName("flash_particles_vs");
  auto fs_color          = gpu::Device::createShaderModuleByName("flash_particles_fs_color");
  auto fs_motion         = gpu::Device::createShaderModuleByName("flash_particles_fs_motion");
  if (!cs_update || !cs_prefill_color || !cs_prefill_motion
      || !vs_module || !fs_color || !fs_motion) return;

  s_pso_update = gpu::Device::createComputePSO(cs_update, "main", gpu::Bindings()
      .storageRW(0)         // particles[] (read+write)
      .tex2d(1)             // maskTex
      .tex2d(2)             // inputTex
      .sampler(3)           // linear sampler
      .uniform(4));         // UpdateUniforms

  // Both prefill PSOs share the same binding shape; only the storage-
  // tex format substitution differs.
  s_pso_prefill_color = gpu::Device::createComputePSO(cs_prefill_color, "main", gpu::Bindings()
      .tex2d(0)
      .storageTex2d(1)
      .uniform(2));
  s_pso_prefill_motion = gpu::Device::createComputePSO(cs_prefill_motion, "main", gpu::Bindings()
      .tex2d(0)
      .storageTex2d(1, gpu::TextureFormat::RGBA16F)
      .uniform(2));

  // ---- Color render PSOs (alpha + additive variants of same shader) ----
  s_pso_render_alpha = gpu::Device::createInstancedRenderPSO(
      vs_module, "main", fs_color, "main",
      gpu::TextureFormat::Surface,
      gpu::Bindings()
          .storage(0)       // particles[] (vertex stage reads)
          .uniform(1)       // VsUniforms (vertex)
          .uniform(2),      // ColorUniforms (fragment)
      gpu::Device::BlendMode::AlphaOver);
  s_pso_render_add = gpu::Device::createInstancedRenderPSO(
      vs_module, "main", fs_color, "main",
      gpu::TextureFormat::Surface,
      gpu::Bindings()
          .storage(0)
          .uniform(1)
          .uniform(2),
      gpu::Device::BlendMode::Additive);

  // ---- Motion render PSO (alpha-over acts as mask-controlled overwrite) ----
  s_pso_render_motion = gpu::Device::createInstancedRenderPSO(
      vs_module, "main", fs_motion, "main",
      gpu::TextureFormat::RGBA16F,
      gpu::Bindings()
          .storage(0)
          .uniform(1)       // VsUniforms (vertex)
          .uniform(2),      // MotionUniforms (fragment)
      gpu::Device::BlendMode::AlphaOver);

  state::log("flash_particles: module initialized");
}

// Per-instance construction: allocate State + its own GPU buffers.
void* create() {
  auto* s = new State();
  s->particle_buf = gpu::Device::createBuffer(
      sizeof(GpuParticle) * MAX_PARTICLES, gpu::BufferUsage::Storage);
  s->update_uniforms = gpu::Device::createBuffer(
      sizeof(UpdateUniforms), gpu::BufferUsage::Uniform);
  s->prefill_color_uniforms = gpu::Device::createBuffer(
      sizeof(PrefillUniforms), gpu::BufferUsage::Uniform);
  s->prefill_motion_uniforms = gpu::Device::createBuffer(
      sizeof(PrefillUniforms), gpu::BufferUsage::Uniform);
  s->vs_uniforms = gpu::Device::createBuffer(
      sizeof(VsUniforms), gpu::BufferUsage::Uniform);
  s->color_uniforms = gpu::Device::createBuffer(
      sizeof(ColorUniforms), gpu::BufferUsage::Uniform);
  s->motion_uniforms = gpu::Device::createBuffer(
      sizeof(MotionUniforms), gpu::BufferUsage::Uniform);
  s->sampler = gpu::Device::createSampler(gpu::FilterMode::Linear,
                                          gpu::AddressMode::ClampToEdge);
  return s;
}

void destroy(void* self) {
  auto* s = static_cast<State*>(self);
  if (!s) return;
  s->particle_buf.release();
  s->update_uniforms.release();
  s->prefill_color_uniforms.release();
  s->prefill_motion_uniforms.release();
  s->vs_uniforms.release();
  s->color_uniforms.release();
  s->motion_uniforms.release();
  s->sampler.release();
  s->motion_tex.release();
  s->zero_motion_tex.release();
  delete s;
}

// Per-instance init tail: reset accumulators + seed the initial pool.
void init(void* self) {
  auto* s = static_cast<State*>(self);
  if (!s) return;
  if (!s_pso_update.valid() || !s_pso_prefill_color.valid() ||
      !s_pso_prefill_motion.valid() || !s_pso_render_alpha.valid() ||
      !s_pso_render_add.valid() || !s_pso_render_motion.valid()) return;
  if (!s->particle_buf.valid()) return;

  s->inited_count = 0;
  s->frame_index  = 0;
  s->init_lcg     = 0x12345678u;
  s->motion_w     = 0;
  s->motion_h     = 0;

  s->initialized = true;
  apply_count_change(*s);   // seed the initial pool
}

void tick(void* self, double dt) {
  auto* s = static_cast<State*>(self);
  if (!s) return;
  (void)dt;  // all timing is GPU-side via dt uniform
}


void on_state_patched(void* self, int n, const char* pb, const int* off,
                      const int* len, const int* ops) {
  auto* s = static_cast<State*>(self);
  if (!s) return;
  for (int i = 0; i < n; i++) {
    if (ops[i] != state::PatchReplace) continue;
    const char* path = pb + off[i];
    int plen = len[i];
    if      (state::pathIs(path, plen, "count"))            { s->count = (int)state::patchFloat(i); apply_count_change(*s); }
    else if (state::pathIs(path, plen, "life"))             s->life = state::patchFloat(i);
    else if (state::pathIs(path, plen, "respawn_delay"))    s->respawn_delay = state::patchFloat(i);
    else if (state::pathIs(path, plen, "life_jitter"))      s->life_jitter = state::patchFloat(i);
    else if (state::pathIs(path, plen, "mask_temperature")) s->mask_temperature = state::patchFloat(i);
    else if (state::pathIs(path, plen, "width"))            s->width = state::patchFloat(i);
    else if (state::pathIs(path, plen, "height"))           s->height = state::patchFloat(i);
    else if (state::pathIs(path, plen, "width_jitter"))     s->width_jitter = state::patchFloat(i);
    else if (state::pathIs(path, plen, "height_jitter"))    s->height_jitter = state::patchFloat(i);
    else if (state::pathIs(path, plen, "global_scale"))     s->global_scale = state::patchFloat(i);
    else if (state::pathIs(path, plen, "rotation"))         s->rotation_deg = state::patchFloat(i);
    else if (state::pathIs(path, plen, "rotation_jitter"))  s->rotation_jitter_deg = state::patchFloat(i);
    else if (state::pathIs(path, plen, "shape_kind"))       s->shape_kind = (int)state::patchFloat(i);
    else if (state::pathIs(path, plen, "shape_param"))      s->shape_param = state::patchFloat(i);
    else if (state::pathIs(path, plen, "alpha_curve"))      s->alpha_curve = state::patchFloat(i);
    else if (state::pathIs(path, plen, "frame_alpha_jitter")) s->frame_alpha_jitter = state::patchFloat(i);
    else if (state::pathIs(path, plen, "global_color")) {
      auto v = state::patchVec3(i);
      s->global_color_r = v.x; s->global_color_g = v.y; s->global_color_b = v.z;
    }
    else if (state::pathIs(path, plen, "color_blend"))        s->color_blend = state::patchFloat(i);
    else if (state::pathIs(path, plen, "hue_jitter"))         s->hue_jitter = state::patchFloat(i);
    else if (state::pathIs(path, plen, "brightness_jitter"))  s->brightness_jitter = state::patchFloat(i);
    else if (state::pathIs(path, plen, "saturation_jitter"))  s->saturation_jitter = state::patchFloat(i);
    else if (state::pathIs(path, plen, "alpha_jitter"))       s->alpha_jitter = state::patchFloat(i);
    else if (state::pathIs(path, plen, "blend_mode"))         s->blend_mode = (int)state::patchFloat(i);
    else if (state::pathIs(path, plen, "input_alpha"))        s->input_alpha = state::patchFloat(i);
    else if (state::pathIs(path, plen, "exposure"))           s->exposure = state::patchFloat(i);
    else if (state::pathIs(path, plen, "color_alpha"))        s->color_alpha = state::patchFloat(i);
    else if (state::pathIs(path, plen, "motion_strength"))    s->motion_strength = state::patchFloat(i);
  }
}

void render(void* self, int vp_w, int vp_h) {
  auto* s = static_cast<State*>(self);
  if (!s || !s->initialized || vp_w <= 0 || vp_h <= 0) return;

  auto in  = gpu::Device::textureForField("tex_in");
  auto out = gpu::Device::textureForField("tex_out");
  if (!in.valid() || !out.valid()) return;

  // Mask falls back to the color input when no secondary mask is wired.
  auto mask = gpu::Device::textureForField("mask_in");
  if (!mask.valid()) mask = in;

  bool emit_motion = state::isOutputConnected("render_outputs");
  if (emit_motion) {
    if (!s->motion_tex.valid() || s->motion_w != vp_w || s->motion_h != vp_h) {
      s->motion_tex = gpu::Device::createTexture(vp_w, vp_h, gpu::TextureFormat::RGBA16F);
      s->motion_w = vp_w;
      s->motion_h = vp_h;
      if (s->motion_tex.valid()) {
        state::setGpuTexture("render_outputs/motion", s->motion_tex.id);
      }
    }
    if (!s->motion_tex.valid()) emit_motion = false;
  }

  auto upstream = gpu::Device::textureForField("render_outputs_in/motion");
  if (!upstream.valid()) {
    if (!s->zero_motion_tex.valid()) {
      s->zero_motion_tex = gpu::Device::createTexture(1, 1, gpu::TextureFormat::RGBA16F);
    }
    upstream = s->zero_motion_tex;
  }

  s->frame_index++;

  // ---- Update uniforms ----
  UpdateUniforms uu = {};
  uu.count               = (uint32_t)s->count;
  uu.frame_index         = s->frame_index;
  uu.dt                  = (float)host::deltaTime();
  uu.mask_temperature    = s->mask_temperature;
  uu.life                = s->life;
  uu.respawn_delay       = s->respawn_delay;
  uu.life_jitter         = s->life_jitter;
  uu.width               = s->width;
  uu.height              = s->height;
  uu.global_scale        = s->global_scale;
  uu.width_jitter        = s->width_jitter;
  uu.height_jitter       = s->height_jitter;
  uu.rotation_rad        = s->rotation_deg * DEG2RAD;
  uu.rotation_jitter_rad = s->rotation_jitter_deg * DEG2RAD;
  uu.hue_jitter          = s->hue_jitter;
  uu.brightness_jitter   = s->brightness_jitter;
  uu.saturation_jitter   = s->saturation_jitter;
  uu.alpha_jitter        = s->alpha_jitter;
  s->update_uniforms.writeOne(uu);

  PrefillUniforms pu_color  = { s->input_alpha, s->input_alpha, s->input_alpha, 1.0f };
  PrefillUniforms pu_motion = { 1.0f, 1.0f, 1.0f, 1.0f };
  s->prefill_color_uniforms.writeOne(pu_color);
  s->prefill_motion_uniforms.writeOne(pu_motion);

  // Aspect normalization for "isotropic uv" rendering.
  float min_dim = float(vp_w < vp_h ? vp_w : vp_h);
  VsUniforms vu = { min_dim / float(vp_w), min_dim / float(vp_h), 0.f, 0.f };
  s->vs_uniforms.writeOne(vu);

  ColorUniforms cu = {};
  cu.input_alpha        = s->input_alpha;
  cu.color_blend        = s->color_blend;
  cu.global_color_r     = s->global_color_r;
  cu.global_color_g     = s->global_color_g;
  cu.global_color_b     = s->global_color_b;
  cu.alpha_curve        = s->alpha_curve;
  cu.frame_alpha_jitter = s->frame_alpha_jitter;
  cu.frame_index        = s->frame_index;
  cu.shape_kind         = (uint32_t)s->shape_kind;
  cu.shape_param        = s->shape_param;
  cu.exposure           = s->exposure;
  cu.color_alpha        = s->color_alpha;
  s->color_uniforms.writeOne(cu);

  MotionUniforms mu = {};
  mu.motion_strength = s->motion_strength;
  mu.shape_kind      = (uint32_t)s->shape_kind;
  mu.shape_param     = s->shape_param;
  mu.alpha_curve     = s->alpha_curve;
  s->motion_uniforms.writeOne(mu);

  // ---- Pass 1: update particles ----
  {
    auto cp = gpu::ComputePass::begin();
    cp.setPSO(s_pso_update);
    cp.setBuffer(s->particle_buf, 0);
    cp.setTexture(mask, 1, 0);
    cp.setTexture(in,   2, 0);
    cp.setSampler(s->sampler, 3);
    cp.setBuffer(s->update_uniforms, 4);
    int groups = (s->count + 63) / 64;
    cp.dispatch(groups, 1, 1);
    cp.end();
  }

  // ---- Pass 2: pre-fill color (tex_in × input_alpha → tex_out) ----
  {
    auto cp = gpu::ComputePass::begin();
    cp.setPSO(s_pso_prefill_color);
    cp.setTexture(in,  0, 0);
    cp.setTexture(out, 1, 1);
    cp.setBuffer(s->prefill_color_uniforms, 2);
    cp.dispatch((vp_w + 7) / 8, (vp_h + 7) / 8);
    cp.end();
  }

  // ---- Pass 3: pre-fill motion (upstream → motionTex) ----
  if (emit_motion) {
    auto cp = gpu::ComputePass::begin();
    cp.setPSO(s_pso_prefill_motion);
    cp.setTexture(upstream,    0, 0);
    cp.setTexture(s->motion_tex, 1, 1);
    cp.setBuffer(s->prefill_motion_uniforms, 2);
    cp.dispatch((vp_w + 7) / 8, (vp_h + 7) / 8);
    cp.end();
  }

  // ---- Pass 4: color raster (instanced quads, blend over pre-filled tex_out) ----
  // Skipped at color_alpha=0 — every fragment would output alpha=0 and
  // contribute nothing under either blend mode, so save the draw.
  // tex_out keeps the pre-filled `tex_in × input_alpha` content.
  if (s->color_alpha > 0.0f) {
    auto rp = gpu::RenderPass::beginLoad(out);
    auto pso = (s->blend_mode == BLEND_ADD) ? s_pso_render_add : s_pso_render_alpha;
    rp.setPSO(pso);
    rp.setBuffer(s->particle_buf, 0);
    rp.setBuffer(s->vs_uniforms,  1);
    rp.setBuffer(s->color_uniforms, 2);
    rp.draw(6, s->count);
    rp.end();
  }

  // ---- Pass 5: motion raster (only when downstream consumes the rail) ----
  if (emit_motion) {
    auto rp = gpu::RenderPass::beginLoad(s->motion_tex);
    rp.setPSO(s_pso_render_motion);
    rp.setBuffer(s->particle_buf,   0);
    rp.setBuffer(s->vs_uniforms,    1);
    rp.setBuffer(s->motion_uniforms, 2);
    rp.draw(6, s->count);
    rp.end();
  }

  gpu::Device::submit();
}

} // namespace flash_particles
