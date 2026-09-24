/*
 * source.particles.flow_swarm — flow-field-driven GPU particle swarm.
 *
 * Consumes a `flow_field` rail (the canonical velocity texture produced by
 * phase_fold or any flow generator/modifier) and advects a GPU-resident pool
 * of up to 1,000,000 particles along it. Each particle chases the sampled
 * field velocity with momentum (inertia), captures the input color where it
 * spawns, and respawns at a fresh random uv when its lifetime expires or it
 * drifts off-field — keeping visual density steady.
 *
 * The split this enables: field GENERATION (phase_fold) is separate from
 * field RENDERING (this swarm), and flow→flow modifiers can sit between.
 *
 * Pipeline (style guide §3.5 — rasterize geometry, don't loop per pixel):
 *   1. update   (compute)  — advect / age / respawn the pool.
 *   2. prefill  (compute)  — tex_in × input_alpha → tex_out.
 *   3. raster   (instanced)— 6 verts × count quads, additive/alpha-over,
 *                            dead particles collapse to a degenerate triangle.
 *
 * Class-like instance ABI: module_init() compiles shared PSOs + publishes the
 * schema once; each chain entry gets its own State (params + pool buffer).
 */

#include <gpu.h>
#include <host.h>
#include "flow_swarm_shaders.h"

#include <cstdint>
#include <cstring>

namespace flow_swarm {

// 1M particles × 32 bytes = 32 MB. The pool is sized for the max so `count`
// can be dialed up at runtime without reallocating; only fresh slots seed.
static constexpr int MAX_PARTICLES = 1000000;

// Quadratic size mapping: the schema `size` is a [0,1] slider (the IDE clips
// floats to 2 decimals, so a small linear range is unusable); the on-GPU size
// is SIZE_SCALE · slider² — fine control at the small end where the swarm
// usually lives. Slider 1 → SIZE_SCALE uv; default ~0.3 → ~0.003 uv.
static constexpr float SIZE_SCALE = 0.035f;
// Point shape draws a fixed quad this many pixels across (size slider ignored).
static constexpr float POINT_PX = 1.5f;

// Interaction density buffer — an abstract square proximity field, resolution
// unrelated to the viewport. Particles splat soft halos into it (additive) and
// the next frame's update reads local crowding back from it.
static constexpr int DENSITY_RES = 256;

// 2 vec4 = 32 bytes. Mirror of `Particle` in common.hlsl.
struct GpuParticle {
  float a[4];   // a.xy=pos, a.z=life_remain, a.w=life_total
  float b[4];   // b.xy=vel, b.z=size, b.w=asfloat(packed rgba8 color)
};
static_assert(sizeof(GpuParticle) == 32, "Particle GPU struct must be 32 bytes");

struct UpdateUniforms {
  uint32_t count;
  uint32_t frame_index;
  float    dt;
  float    speed;

  float    momentum;
  float    jitter;
  float    drag;
  float    life;

  float    life_jitter;
  float    size;
  float    size_jitter;
  uint32_t seed;

  uint32_t mode;
  float    weight;
  float    undertow_split;
  float    undertow_polarity;

  float    undertow_curl;
  float    pull;
  uint32_t interactions;
  float    density_threshold;

  float    density_death;
  float    avoid;
  float    avoid_curl;
  float    density_res;

  float    avoid_noise;
  uint32_t substeps;
  float    dens_aspect_x;   // min/W, min/H — make the density isotropic in pixels
  float    dens_aspect_y;

  float    stream;          // +align / -diverge velocity vs the group
  float    stream_density;  // neighbour density for ~max stream effect
  float    _pad_s0;
  float    _pad_s1;
};
static_assert(sizeof(UpdateUniforms) == 128, "UpdateUniforms layout mismatch");

struct PrefillUniforms { float scale_r, scale_g, scale_b, scale_a; };
static_assert(sizeof(PrefillUniforms) == 16, "PrefillUniforms layout mismatch");

struct DensityUniforms { float radius, aspect_x, aspect_y, _pad; };
static_assert(sizeof(DensityUniforms) == 16, "DensityUniforms layout mismatch");

struct VsUniforms {
  float aspect_x, aspect_y, point_size, shape_kind;
  float undertow_split, _pad0, _pad1, _pad2;
};
static_assert(sizeof(VsUniforms) == 32, "VsUniforms layout mismatch");

struct ColorUniforms {
  float    color_blend;
  float    solid_r;
  float    solid_g;
  float    solid_b;

  float    tint_by_flow;
  float    opacity;
  float    alpha_curve;
  float    shape_param;

  uint32_t shape_kind;
  float    exposure;
  float    undertow_alpha;
  float    _pad0;

  float    undertow_tint_r;
  float    undertow_tint_g;
  float    undertow_tint_b;
  float    _pad1;
};
static_assert(sizeof(ColorUniforms) == 64, "ColorUniforms layout mismatch");

enum BlendMode : int { BLEND_ALPHA = 0, BLEND_ADD = 1 };
enum ShapeKind : int { SHAPE_POINT = 0, SHAPE_GAUSSIAN = 1, SHAPE_CIRCLE = 2, SHAPE_SOLID = 3 };
enum Mode      : int { MODE_VELOCITY = 0, MODE_FORCE = 1 };

// Type-shared: compiled once in module_init().
static gpu::ComputePSO s_pso_update;
static gpu::ComputePSO s_pso_prefill;
static gpu::RenderPSO  s_pso_render_alpha;
static gpu::RenderPSO  s_pso_render_add;
static gpu::RenderPSO  s_pso_density;     // soft-halo additive splat → density buffer
static gpu::ComputePSO s_pso_density_debug; // heat-map blit of the density buffer

struct State {
  gpu::Buffer  particle_buf;
  gpu::Buffer  update_uniforms;
  gpu::Buffer  prefill_uniforms;
  gpu::Buffer  vs_uniforms;
  gpu::Buffer  color_uniforms;
  gpu::Buffer  density_uniforms;
  gpu::Sampler sampler;
  gpu::Texture zero_flow_tex;     // 1×1 fallback when no flow upstream
  gpu::Texture density_tex;       // persistent crowding buffer (1-frame delayed)
  gpu::Texture zero_density_tex;  // 1×1 fallback when interactions are off

  bool initialized = false;

  // CPU mirrors of schema params.
  int   count        = 150000;
  int   mode         = MODE_VELOCITY;
  float speed        = 1.5f;
  float momentum     = 0.0f;    // velocity mode: 0 = clean sim of the field
  float weight       = 1.0f;    // force mode: particle mass
  int   substeps     = 1;       // integration substeps per frame (finer = stabler)
  float pull         = 0.0f;    // settle velocity toward the field flow (both modes)
  float life         = 4.0f;
  float life_jitter  = 0.4f;
  float size         = 0.3f;    // [0,1] slider, quadratic → uv (SIZE_SCALE·size²)
  float size_jitter  = 0.5f;
  float jitter       = 0.0f;
  float drag         = 0.1f;
  float color_blend  = 0.3f;
  float solid_r      = 1.0f;
  float solid_g      = 1.0f;
  float solid_b      = 1.0f;
  float tint_by_flow = 0.0f;
  // Undertow: a depth-gated secondary flow behaviour.
  float undertow_split    = 0.0f;   // 0 = none undertow, 1 = all
  float undertow_polarity = 1.0f;   // 1 normal, -1 reverse, 2 = 2× speed
  float undertow_curl     = 0.0f;   // -1 turn 90° left, +1 right
  float undertow_alpha    = 1.0f;
  float undertow_tint_r   = 0.2f;
  float undertow_tint_g   = 0.45f;
  float undertow_tint_b   = 1.0f;
  // Interactions (a 1-frame-delayed density buffer drives these).
  bool  interactions       = false;
  float interaction_radius = 0.015f;  // splat halo radius in density uv
  float density_threshold  = 4.0f;    // crowding (≈ neighbours) before death
  float density_death      = 0.0f;    // death-rate scale over threshold
  float avoid              = 0.0f;    // push away from neighbours
  float avoid_curl         = 0.0f;    // rotate the avoidance ±90°
  float avoid_noise        = 0.08f;   // random jitter on avoidance (breaks clumps)
  float stream             = 0.0f;    // +align / -diverge velocity vs the group
  float stream_density     = 3.0f;    // neighbour density for ~max stream effect
  bool  debug_density      = false;   // render the density buffer as a heat map
  float opacity      = 1.0f;
  float alpha_curve  = 0.6f;
  float exposure     = 1.0f;
  int   shape_kind   = SHAPE_POINT;
  float shape_param  = 0.5f;
  int   blend_mode   = BLEND_ADD;
  float input_alpha  = 1.0f;
  int   seed         = 0;

  // Seed/accumulator bookkeeping.
  int      inited_count = 0;
  uint32_t frame_index  = 0;
  uint32_t init_lcg     = 0x12345678u;
};

static inline uint32_t lcg_next(uint32_t& s) { s = s * 1664525u + 1013904223u; return s; }
static inline float    lcg_unit(uint32_t& s) { return (lcg_next(s) >> 8) * (1.0f / float(1u << 24)); }

// Pack white rgb + an 8-bit depth into the particle's color slot (matches
// fsw_pack_rgbd in common.hlsl). Only bit-reinterpreted, never float math.
static inline float pack_white_depth(float depth) {
  uint32_t d = (uint32_t)(depth * 255.0f + 0.5f);
  if (d > 255u) d = 255u;
  uint32_t packed = 0xFFFFFFu | (d << 24);
  float f;
  std::memcpy(&f, &packed, sizeof(f));
  return f;
}

// Chunked seeding so a 1M pool doesn't blow the wasm stack (256 × 32 B = 8 KB).
static constexpr int INIT_CHUNK = 256;

static void seed_initial_slots(State& s, int from, int to) {
  if (!s.initialized || from >= to) return;
  GpuParticle entries[INIT_CHUNK];
  float size_uv = SIZE_SCALE * s.size * s.size;   // quadratic mapping
  for (int chunk_start = from; chunk_start < to; chunk_start += INIT_CHUNK) {
    int chunk_end = chunk_start + INIT_CHUNK;
    if (chunk_end > to) chunk_end = to;
    int n = chunk_end - chunk_start;
    for (int i = 0; i < n; i++) {
      GpuParticle& p = entries[i];
      float ux = lcg_unit(s.init_lcg);
      float uy = lcg_unit(s.init_lcg);
      float life_remain = lcg_unit(s.init_lcg) * s.life;   // staggered start
      float depth = lcg_unit(s.init_lcg);
      p.a[0] = ux; p.a[1] = uy; p.a[2] = life_remain; p.a[3] = s.life;
      p.b[0] = 0.0f; p.b[1] = 0.0f; p.b[2] = size_uv; p.b[3] = pack_white_depth(depth);
    }
    s.particle_buf.writeBytes(entries, int(sizeof(GpuParticle)) * n,
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

// Hide params the inactive mode / disabled interactions don't use (§0).
static void apply_mode_visibility(int mode, bool interactions) {
  state::setFieldHidden("momentum", mode != MODE_VELOCITY);
  state::setFieldHidden("weight",   mode != MODE_FORCE);
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

// Static (self-less) visibility evaluator — pure over state (see crop).
void eval_visibility(int n, const char* pb, const int* off, const int* len, const int* ops) {
  int mode = MODE_VELOCITY; bool interactions = false;
  for (int i = 0; i < n; i++) {
    if (ops[i] != state::PatchReplace) continue;
    const char* p = pb + off[i]; int l = len[i];
    if      (state::pathIs(p, l, "mode"))         mode = (int)state::patchFloat(i);
    else if (state::pathIs(p, l, "interactions")) interactions = state::patchFloat(i) != 0.0f;
  }
  apply_mode_visibility(mode, interactions);
}

static void on_state_ready(void* self) {
  auto* s = static_cast<State*>(self);
  if (s) apply_mode_visibility(s->mode, s->interactions);
}

void module_init() {
  state::init("source.particles.flow_swarm", {1, 0, 1},
    state::Schema()
      // Top-level manual: high-level "what is this / how to use / what to try".
      .helpField("intro",
        "## Flow Swarm\n"
        "A GPU particle swarm that **rides a flow field**. Wire a `flow_field` "
        "rail (from *phase_fold* or any flow generator) into it and up to a "
        "million particles are advected along the sampled velocity, each capturing "
        "the input colour where it spawns and respawning to keep density steady.\n\n"
        "**Try:** start with a modest *Count* and low *Speed* to read the field, "
        "then raise *Momentum* for silky trailing inertia; switch *Mode* to **Force** "
        "and add *Weight* for overshoot; enable *Interactions* to let the swarm "
        "spread itself evenly or stream in packs.")
      // ---- Pool / advection (the live controls) ----
      .group("advection", "Pool & Advection")
        .groupHelp(
          "The core motion. *Count* sets how many particles are live (the pool is "
          "pre-sized to 1M, so dial it freely). **Velocity** mode treats the field "
          "as a target velocity — add *Momentum* for inertia; **Force** mode drives "
          "a mass (*Weight*) so the field becomes a hint and overshoot emerges. "
          "*Settle* glues motion back toward the flow, *Substeps* stabilises fast "
          "flow, and *Jitter* sprays a little life along the direction of travel.")
      .intField  ("count",        150000, 1, MAX_PARTICLES, state::PrimaryInput).label("Count", "Count")
      // Acceleration mode: Velocity treats the field as a velocity (momentum
      // blends inertia in); Force treats it as a force/acceleration on a mass
      // (weight), so the field becomes a hint and overshoot emerges.
      .selectField("mode",        MODE_VELOCITY, state::PrimaryInput, {
        {"Velocity", MODE_VELOCITY},
        {"Force",    MODE_FORCE},
      }).label("Mode", "Mode")
      .floatField("speed",        1.5f,  0.0f,  8.0f,  state::PrimaryInput).label("Speed", "Spd")
      // Velocity mode only: 0 = clean sim of the field, →1 = heavy inertia.
      .floatField("momentum",     0.0f,  0.0f,  0.99f, state::PrimaryInput).label("Momentum", "Mom")
      // Force mode only: particle mass (accel = field / weight).
      .floatField("weight",       1.0f,  0.05f, 8.0f,  state::PrimaryInput).label("Weight", "Wt")
      // Integration substeps per frame: the motion is integrated this many times
      // with dt/substeps, re-sampling the field each time. Higher = stabler /
      // less overshoot for force mode & fast flow (at proportional GPU cost).
      .intField  ("substeps",     1,     1,     16,    state::PrimaryInput).label("Substeps", "Sub")
      // Settle (both modes): pull each particle's velocity back toward the
      // field flow, keeping it in the stable zone (the limit cycle) and damping
      // force-mode overshoot. 0 = free, 1 = strongly glued to the field.
      .floatField("pull",         0.0f,  0.0f,  1.0f,  state::PrimaryInput).label("Settle", "Pull")
      // Jitter: a forward SPRAY — ±wobble on the forward speed plus a slight
      // directional spread, scaled by the particle's own speed (rides the flow,
      // not an isotropic cloud).
      .floatField("jitter",       0.0f,  0.0f,  1.0f,  state::PrimaryInput).label("Jitter", "Jit")
      .floatField("drag",         0.1f,  0.0f,  4.0f,  state::PrimaryInput).label("Drag", "Drag")
      // ---- Geometry / lifetime ----
      .group("geometry", "Geometry & Lifetime")
      // size is a [0,1] slider mapped quadratically to a (small) uv size — the
      // IDE clips to 2 decimals so a raw uv range was too coarse at the bottom.
      .floatField("size",         0.3f,  0.0f,  1.0f,  state::PrimaryInput).label("Size", "Size")
      .floatField("size_jitter",  0.5f,  0.0f,  1.0f,  state::PrimaryInput).label("Size Jitter", "SzJit")
      .floatField("life",         4.0f,  0.1f,  30.0f, state::PrimaryInput).label("Lifetime", "Life")
      .floatField("life_jitter",  0.4f,  0.0f,  1.0f,  state::PrimaryInput).label("Life Jitter", "LfJit")
      // ---- Color ----
      .group("color", "Colour")
      .floatField("color_blend",  0.3f,  0.0f,  1.0f,  state::PrimaryInput).label("Colour Blend", "Blend")
      .rgbField  ("solid_color",  1.0f, 1.0f, 1.0f,    state::PrimaryInput).label("Solid Colour", "Colour")
      .floatField("tint_by_flow", 0.0f,  0.0f,  1.0f,  state::PrimaryInput).label("Tint By Flow", "Tint")
      // ---- Undertow: a depth-gated secondary flow ----
      .group("undertow", "Undertow")
        .groupHelp(
          "A **second current** hidden inside the swarm. Every particle is born "
          "with a secret depth; *Split* softly selects which depths peel off into "
          "the undertow (0 = none, 1 = all). Those members run at *Polarity* × the "
          "field (negative to flow **backwards**, >1 to race ahead) and can *Curl* "
          "90° off-axis, tinted by *Tint*/*Alpha* — great for counter-flowing layers.")
      // Each particle gets a hidden depth ∈[0,1] at spawn. `split` softly
      // selects which depths join the undertow stream (0 none → 1 all).
      .floatField("undertow_split",    0.0f,  0.0f,  1.0f,  state::PrimaryInput).label("Undertow Split", "Split")
      // Members travel at `polarity` × the field (1 normal, -1 reverse, 2 = 2×).
      .floatField("undertow_polarity", 1.0f, -2.0f,  2.0f,  state::PrimaryInput).label("Undertow Polarity", "Polar")
      // Curl rotates the undertow direction: -1 = 90° left, +1 = 90° right.
      .floatField("undertow_curl",     0.0f, -1.0f,  1.0f,  state::PrimaryInput).label("Undertow Curl", "Curl")
      // Members blend toward this tint by membership, with this alpha multiplier.
      .rgbField  ("undertow_tint",     0.2f, 0.45f, 1.0f,   state::PrimaryInput).label("Undertow Tint", "Tint")
      .floatField("undertow_alpha",    1.0f,  0.0f,  2.0f,  state::PrimaryInput).label("Undertow Alpha", "Alpha")
      // ---- Interactions (particle-vs-particle, via a density buffer) ----
      .group("interactions", "Interactions")
        .groupHelp(
          "Lets particles feel their neighbours through a 1-frame crowding buffer "
          "(off by default — it adds a splat pass). *Radius* is the sensing range. "
          "**Density Death** thins over-packed areas back toward even coverage; "
          "**Avoidance** pushes down the crowd gradient (add *Curl* to swirl, "
          "*Noise* to unstick symmetric clumps); **Streaming** aligns (+) or "
          "scatters (−) each particle with its local group. Flip **Debug Density** "
          "to see the buffer as a heat map while tuning.")
      // Splats particles to a 1-frame-delayed crowding buffer so they can react
      // to each other. Off by default (an extra splat pass).
      .boolField ("interactions",        false,                state::PrimaryInput).label("Interactions", "Inter")
      // Halo radius in the density buffer — the interaction RANGE.
      .floatField("interaction_radius",  0.015f, 0.002f, 0.08f, state::PrimaryInput).label("Interaction Radius", "Radius")
      // Density death: over `threshold` crowding (≈ neighbour count, soft knee)
      // particles get a `death`-scaled chance to die & respawn → uniform density.
      .floatField("density_threshold",   4.0f,   0.0f,  32.0f, state::PrimaryInput).label("Density Threshold", "Thresh")
      .floatField("density_death",       0.0f,   0.0f,  1.0f,  state::PrimaryInput).label("Density Death", "Death")
      // Avoidance: push down the density gradient (away from neighbours); curl
      // rotates that push ±90° for a swirling avoidance.
      .floatField("avoid",               0.0f,   0.0f,  1.0f,  state::PrimaryInput).label("Avoidance", "Avoid")
      .floatField("avoid_curl",          0.0f,  -1.0f,  1.0f,  state::PrimaryInput).label("Avoid Curl", "Curl")
      // Random jitter on the avoidance so particles still scatter where the
      // density gradient is flat (a symmetric clump's centre). Default on.
      .floatField("avoid_noise",         0.08f,  0.0f,  1.0f,  state::PrimaryInput).label("Avoid Noise", "Noise")
      // Stream: align (+) or diverge (-) each particle's velocity DIRECTION with
      // the local group's mean motion (read from the proximity texture's motion
      // channels). stream_density scales how many proximate, motion-contributing
      // neighbours are needed to reach ~maximum effect.
      .floatField("stream",              0.0f,  -1.0f,  1.0f,  state::PrimaryInput).label("Streaming", "Stream")
      .floatField("stream_density",      3.0f,   0.5f,  32.0f, state::PrimaryInput).label("Stream Density", "StrDen")
      // Debug: render the density buffer itself (heat map) instead of the swarm.
      .boolField ("debug_density",       false,                state::PrimaryInput).label("Debug Density", "Debug")
      // ---- Composite ----
      .group("composite", "Composite")
      .selectField("blend_mode",  BLEND_ADD, state::PrimaryInput, {
        {"Add",   BLEND_ADD},
        {"Alpha", BLEND_ALPHA},
      }).label("Blend Mode", "Blend")
      .floatField("opacity",      1.0f,  0.0f,  1.0f,  state::PrimaryInput).label("Opacity", "Opac")
      .floatField("input_alpha",  1.0f,  0.0f,  1.0f,  state::PrimaryInput).label("Input Alpha", "InAlph")
      // ---- Tuning / shape ----
      .group("shape", "Shape & Tuning")
      .selectField("shape_kind",  SHAPE_POINT, state::PrimaryInput, {
        {"Point",    SHAPE_POINT},
        {"Gaussian", SHAPE_GAUSSIAN},
        {"Circle",   SHAPE_CIRCLE},
        {"Solid",    SHAPE_SOLID},
      }).label("Shape", "Shape")
      .floatField("shape_param",  0.5f,  0.0f,  1.0f,  state::PrimaryInput).label("Shape Param", "Param")
      .floatField("alpha_curve",  0.6f,  0.25f, 4.0f,  state::PrimaryInput).label("Alpha Curve", "Curve")
      .floatField("exposure",     1.0f,  0.0f,  8.0f,  state::PrimaryInput).label("Exposure", "Exp")
      .intField  ("seed",         0,     0,     65535, state::PrimaryInput).label("Seed", "Seed")
      // ---- I/O ----
      .textureField("tex_in",  state::PrimaryInput)
      .flowField(state::PrimaryInput, "flow_field_in")
      .textureField("tex_out", state::PrimaryOutput)
        .capability(state::Capability::Generator)
    );

  if (gpu::Device::backend() == gpu::Backend::None) return;

  state::registerShaderSPV("flow_swarm_update",     UPDATE_SPV,     UPDATE_SPV_SIZE);
  state::registerShaderSPV("flow_swarm_prefill",    PREFILL_SPV,    PREFILL_SPV_SIZE);
  state::registerShaderSPV("flow_swarm_vs",         VS_SPV,         VS_SPV_SIZE);
  state::registerShaderSPV("flow_swarm_fs",         FS_SPV,         FS_SPV_SIZE);
  state::registerShaderSPV("flow_swarm_density_vs", DENSITY_VS_SPV, DENSITY_VS_SPV_SIZE);
  state::registerShaderSPV("flow_swarm_density_fs", DENSITY_FS_SPV, DENSITY_FS_SPV_SIZE);
  state::registerShaderSPV("flow_swarm_density_debug", DENSITY_DEBUG_SPV, DENSITY_DEBUG_SPV_SIZE);

  auto cs_update   = gpu::Device::createShaderModuleByName("flow_swarm_update");
  auto cs_prefill  = gpu::Device::createShaderModuleByName("flow_swarm_prefill");
  auto vs_module   = gpu::Device::createShaderModuleByName("flow_swarm_vs");
  auto fs_module   = gpu::Device::createShaderModuleByName("flow_swarm_fs");
  auto vs_density  = gpu::Device::createShaderModuleByName("flow_swarm_density_vs");
  auto fs_density  = gpu::Device::createShaderModuleByName("flow_swarm_density_fs");
  auto cs_dbg      = gpu::Device::createShaderModuleByName("flow_swarm_density_debug");
  if (!cs_update || !cs_prefill || !vs_module || !fs_module || !vs_density || !fs_density || !cs_dbg) return;

  s_pso_update = gpu::Device::createComputePSO(cs_update, "main", gpu::Bindings()
      .storageRW(0)   // particles[]
      .tex2d(1)       // flow velocity
      .tex2d(2)       // input (color capture)
      .sampler(3)
      .uniform(4)
      .tex2d(5));     // density (last frame's crowding)

  s_pso_prefill = gpu::Device::createComputePSO(cs_prefill, "main", gpu::Bindings()
      .tex2d(0)
      .storageTex2d(1)
      .uniform(2));

  s_pso_render_alpha = gpu::Device::createInstancedRenderPSO(
      vs_module, "main", fs_module, "main", gpu::TextureFormat::Surface,
      gpu::Bindings().storage(0).uniform(1).uniform(2),
      gpu::Device::BlendMode::AlphaOver);
  s_pso_render_add = gpu::Device::createInstancedRenderPSO(
      vs_module, "main", fs_module, "main", gpu::TextureFormat::Surface,
      gpu::Bindings().storage(0).uniform(1).uniform(2),
      gpu::Device::BlendMode::Additive);

  // Density splat: soft halos summed into the RGBA16F crowding buffer.
  s_pso_density = gpu::Device::createInstancedRenderPSO(
      vs_density, "main", fs_density, "main", gpu::TextureFormat::RGBA16F,
      gpu::Bindings().storage(0).uniform(1),
      gpu::Device::BlendMode::Additive);

  // Debug: heat-map the density buffer into tex_out.
  s_pso_density_debug = gpu::Device::createComputePSO(cs_dbg, "main", gpu::Bindings()
      .tex2d(0)
      .sampler(1)
      .storageTex2d(2));

  state::log("flow_swarm: module initialized");
}

void* create() {
  auto* s = new State();
  s->particle_buf = gpu::Device::createBuffer(
      sizeof(GpuParticle) * MAX_PARTICLES, gpu::BufferUsage::Storage);
  s->update_uniforms  = gpu::Device::createBuffer(sizeof(UpdateUniforms),  gpu::BufferUsage::Uniform);
  s->prefill_uniforms = gpu::Device::createBuffer(sizeof(PrefillUniforms), gpu::BufferUsage::Uniform);
  s->vs_uniforms      = gpu::Device::createBuffer(sizeof(VsUniforms),      gpu::BufferUsage::Uniform);
  s->color_uniforms   = gpu::Device::createBuffer(sizeof(ColorUniforms),   gpu::BufferUsage::Uniform);
  s->density_uniforms = gpu::Device::createBuffer(sizeof(DensityUniforms), gpu::BufferUsage::Uniform);
  s->sampler = gpu::Device::createSampler(gpu::FilterMode::Linear, gpu::AddressMode::ClampToEdge);
  return s;
}

void destroy(void* self) {
  auto* s = static_cast<State*>(self);
  if (!s) return;
  s->particle_buf.release();
  s->update_uniforms.release();
  s->prefill_uniforms.release();
  s->vs_uniforms.release();
  s->color_uniforms.release();
  s->density_uniforms.release();
  s->sampler.release();
  s->zero_flow_tex.release();
  s->density_tex.release();
  s->zero_density_tex.release();
  delete s;
}

void init(void* self) {
  auto* s = static_cast<State*>(self);
  if (!s) return;
  if (!s_pso_update.valid() || !s_pso_prefill.valid() ||
      !s_pso_render_alpha.valid() || !s_pso_render_add.valid() ||
      !s_pso_density.valid() || !s_pso_density_debug.valid()) return;
  if (!s->particle_buf.valid()) return;

  s->inited_count = 0;
  s->frame_index  = 0;
  s->init_lcg     = 0x12345678u;
  s->initialized  = true;
  apply_count_change(*s);   // seed the initial pool
  state::setOnStateReady(&on_state_ready);
}

void tick(void* self, double dt) { (void)self; (void)dt; }   // timing is GPU-side via dt uniform

void on_state_patched(void* self, int n, const char* pb, const int* off,
                      const int* len, const int* ops) {
  auto* s = static_cast<State*>(self);
  if (!s) return;
  for (int i = 0; i < n; i++) {
    if (ops[i] != state::PatchReplace) continue;
    const char* path = pb + off[i];
    int plen = len[i];
    if      (state::pathIs(path, plen, "count"))        { s->count = (int)state::patchFloat(i); apply_count_change(*s); }
    else if (state::pathIs(path, plen, "mode"))         { int v = (int)state::patchFloat(i); if (v != s->mode) { s->mode = v; apply_mode_visibility(s->mode, s->interactions); } }
    else if (state::pathIs(path, plen, "speed"))        s->speed = state::patchFloat(i);
    else if (state::pathIs(path, plen, "momentum"))     s->momentum = state::patchFloat(i);
    else if (state::pathIs(path, plen, "weight"))       s->weight = state::patchFloat(i);
    else if (state::pathIs(path, plen, "substeps"))     s->substeps = (int)state::patchFloat(i);
    else if (state::pathIs(path, plen, "pull"))         s->pull = state::patchFloat(i);
    else if (state::pathIs(path, plen, "jitter"))       s->jitter = state::patchFloat(i);
    else if (state::pathIs(path, plen, "drag"))         s->drag = state::patchFloat(i);
    else if (state::pathIs(path, plen, "size"))         s->size = state::patchFloat(i);
    else if (state::pathIs(path, plen, "size_jitter"))  s->size_jitter = state::patchFloat(i);
    else if (state::pathIs(path, plen, "life"))         s->life = state::patchFloat(i);
    else if (state::pathIs(path, plen, "life_jitter"))  s->life_jitter = state::patchFloat(i);
    else if (state::pathIs(path, plen, "color_blend"))  s->color_blend = state::patchFloat(i);
    else if (state::pathIs(path, plen, "solid_color")) {
      auto v = state::patchVec3(i);
      s->solid_r = v.x; s->solid_g = v.y; s->solid_b = v.z;
    }
    else if (state::pathIs(path, plen, "tint_by_flow"))      s->tint_by_flow = state::patchFloat(i);
    else if (state::pathIs(path, plen, "undertow_split"))    s->undertow_split = state::patchFloat(i);
    else if (state::pathIs(path, plen, "undertow_polarity")) s->undertow_polarity = state::patchFloat(i);
    else if (state::pathIs(path, plen, "undertow_curl"))     s->undertow_curl = state::patchFloat(i);
    else if (state::pathIs(path, plen, "undertow_tint")) {
      auto v = state::patchVec3(i);
      s->undertow_tint_r = v.x; s->undertow_tint_g = v.y; s->undertow_tint_b = v.z;
    }
    else if (state::pathIs(path, plen, "undertow_alpha"))    s->undertow_alpha = state::patchFloat(i);
    else if (state::pathIs(path, plen, "interactions"))      { bool v = state::patchFloat(i) != 0.0f; if (v != s->interactions) { s->interactions = v; apply_mode_visibility(s->mode, s->interactions); } }
    else if (state::pathIs(path, plen, "interaction_radius")) s->interaction_radius = state::patchFloat(i);
    else if (state::pathIs(path, plen, "density_threshold")) s->density_threshold = state::patchFloat(i);
    else if (state::pathIs(path, plen, "density_death"))     s->density_death = state::patchFloat(i);
    else if (state::pathIs(path, plen, "avoid"))            s->avoid = state::patchFloat(i);
    else if (state::pathIs(path, plen, "avoid_curl"))       s->avoid_curl = state::patchFloat(i);
    else if (state::pathIs(path, plen, "avoid_noise"))      s->avoid_noise = state::patchFloat(i);
    else if (state::pathIs(path, plen, "stream"))           s->stream = state::patchFloat(i);
    else if (state::pathIs(path, plen, "stream_density"))   s->stream_density = state::patchFloat(i);
    else if (state::pathIs(path, plen, "debug_density"))    s->debug_density = state::patchFloat(i) != 0.0f;
    else if (state::pathIs(path, plen, "blend_mode"))   s->blend_mode = (int)state::patchFloat(i);
    else if (state::pathIs(path, plen, "opacity"))      s->opacity = state::patchFloat(i);
    else if (state::pathIs(path, plen, "input_alpha"))  s->input_alpha = state::patchFloat(i);
    else if (state::pathIs(path, plen, "shape_kind"))   s->shape_kind = (int)state::patchFloat(i);
    else if (state::pathIs(path, plen, "shape_param"))  s->shape_param = state::patchFloat(i);
    else if (state::pathIs(path, plen, "alpha_curve"))  s->alpha_curve = state::patchFloat(i);
    else if (state::pathIs(path, plen, "exposure"))     s->exposure = state::patchFloat(i);
    else if (state::pathIs(path, plen, "seed"))         s->seed = (int)state::patchFloat(i);
  }
}

void render(void* self, int vp_w, int vp_h) {
  auto* s = static_cast<State*>(self);
  if (!s || !s->initialized || vp_w <= 0 || vp_h <= 0) return;

  auto in  = gpu::Device::textureForField("tex_in");
  auto out = gpu::Device::textureForField("tex_out");
  if (!in.valid() || !out.valid()) return;

  // Flow field input. Fall back to a 1×1 zero (still) field when unwired —
  // particles then drift only by jitter and still render.
  auto flow = gpu::Device::textureForField("flow_field_in/velocity");
  if (!flow.valid()) {
    if (!s->zero_flow_tex.valid()) {
      s->zero_flow_tex = gpu::Device::createTexture(1, 1, gpu::TextureFormat::RGBA16F);
      gpu::Device::clear(s->zero_flow_tex, 0.0f, 0.0f, 0.0f, 0.0f);
    }
    flow = s->zero_flow_tex;
  }

  // Interactions: a persistent density buffer the update pass reads (built from
  // LAST frame's particles → 1-frame delay). When off, bind a 1×1 zero buffer.
  bool ix = s->interactions;
  gpu::Texture density_in;
  if (ix) {
    if (!s->density_tex.valid()) {
      s->density_tex = gpu::Device::createTexture(DENSITY_RES, DENSITY_RES,
                                                  gpu::TextureFormat::RGBA16F);
      gpu::Device::clear(s->density_tex, 0.0f, 0.0f, 0.0f, 0.0f);  // first read = empty
    }
    if (!s->density_tex.valid()) ix = false;
  }
  if (ix) {
    density_in = s->density_tex;
  } else {
    if (!s->zero_density_tex.valid()) {
      s->zero_density_tex = gpu::Device::createTexture(1, 1, gpu::TextureFormat::RGBA16F);
      gpu::Device::clear(s->zero_density_tex, 0.0f, 0.0f, 0.0f, 0.0f);
    }
    density_in = s->zero_density_tex;
  }

  s->frame_index++;

  float size_uv = SIZE_SCALE * s->size * s->size;   // quadratic mapping

  UpdateUniforms uu = {};
  uu.count             = (uint32_t)s->count;
  uu.frame_index       = s->frame_index;
  uu.dt                = (float)host::deltaTime();
  uu.speed             = s->speed;
  uu.momentum          = s->momentum;
  uu.jitter            = s->jitter;
  uu.drag              = s->drag;
  uu.life              = s->life;
  uu.life_jitter       = s->life_jitter;
  uu.size              = size_uv;
  uu.size_jitter       = s->size_jitter;
  uu.seed              = (uint32_t)s->seed;
  uu.mode              = (uint32_t)s->mode;
  uu.weight            = s->weight;
  uu.undertow_split    = s->undertow_split;
  uu.undertow_polarity = s->undertow_polarity;
  uu.undertow_curl     = s->undertow_curl;
  uu.pull              = s->pull;
  uu.interactions      = ix ? 1u : 0u;
  uu.density_threshold = s->density_threshold;
  uu.density_death     = s->density_death;
  uu.avoid             = s->avoid;
  uu.avoid_curl        = s->avoid_curl;
  uu.density_res       = (float)DENSITY_RES;
  uu.avoid_noise       = s->avoid_noise;
  uu.substeps          = (uint32_t)(s->substeps < 1 ? 1 : (s->substeps > 16 ? 16 : s->substeps));
  // Isotropic-uv aspect (1 unit = min(W,H) px) — same factors the visual VS uses.
  float min_dim = float(vp_w < vp_h ? vp_w : vp_h);
  float aspect_x = min_dim / float(vp_w);
  float aspect_y = min_dim / float(vp_h);
  uu.dens_aspect_x     = aspect_x;
  uu.dens_aspect_y     = aspect_y;
  uu.stream            = s->stream;
  uu.stream_density    = s->stream_density;
  s->update_uniforms.writeOne(uu);

  PrefillUniforms pu = { s->input_alpha, s->input_alpha, s->input_alpha, 1.0f };
  s->prefill_uniforms.writeOne(pu);

  VsUniforms vu = {};
  vu.aspect_x        = aspect_x;
  vu.aspect_y        = aspect_y;
  vu.point_size      = POINT_PX / min_dim;          // ~1.5px in isotropic uv
  vu.shape_kind      = (float)s->shape_kind;
  vu.undertow_split  = s->undertow_split;
  s->vs_uniforms.writeOne(vu);

  ColorUniforms cu = {};
  cu.color_blend     = s->color_blend;
  cu.solid_r         = s->solid_r;
  cu.solid_g         = s->solid_g;
  cu.solid_b         = s->solid_b;
  cu.tint_by_flow    = s->tint_by_flow;
  cu.opacity         = s->opacity;
  cu.alpha_curve     = s->alpha_curve;
  cu.shape_param     = s->shape_param;
  cu.shape_kind      = (uint32_t)s->shape_kind;
  cu.exposure        = s->exposure;
  cu.undertow_alpha  = s->undertow_alpha;
  cu.undertow_tint_r = s->undertow_tint_r;
  cu.undertow_tint_g = s->undertow_tint_g;
  cu.undertow_tint_b = s->undertow_tint_b;
  s->color_uniforms.writeOne(cu);

  // ---- Pass 1: update particles ----
  {
    auto cp = gpu::ComputePass::begin();
    cp.setPSO(s_pso_update);
    cp.setBuffer(s->particle_buf, 0);
    cp.setTexture(flow, 1, 0);
    cp.setTexture(in,   2, 0);
    cp.setSampler(s->sampler, 3);
    cp.setBuffer(s->update_uniforms, 4);
    cp.setTexture(density_in, 5, 0);
    int groups = (s->count + 63) / 64;
    cp.dispatch(groups, 1, 1);
    cp.end();
  }

  // ---- Pass 2: pre-fill (tex_in × input_alpha → tex_out) ----
  {
    auto cp = gpu::ComputePass::begin();
    cp.setPSO(s_pso_prefill);
    cp.setTexture(in,  0, 0);
    cp.setTexture(out, 1, 1);
    cp.setBuffer(s->prefill_uniforms, 2);
    cp.dispatch((vp_w + 7) / 8, (vp_h + 7) / 8);
    cp.end();
  }

  bool debug = ix && s->debug_density;
  // The density splat is the most expensive interaction pass (a draw over every
  // particle into the crowding buffer, with overdraw). Only run it when
  // something actually reads the buffer this frame — death, avoid, or the debug
  // view. interactions-on-but-unused then costs nothing extra.
  bool need_density = ix && (s->density_death > 0.0f || s->avoid > 0.0f ||
                             s->stream != 0.0f || s->debug_density);

  // ---- Pass 3: instanced raster (blend over pre-filled tex_out) ----
  // Skipped when debugging the density buffer (the blit below overwrites it).
  if (s->opacity > 0.0f && !debug) {
    auto rp = gpu::RenderPass::beginLoad(out);
    auto pso = (s->blend_mode == BLEND_ADD) ? s_pso_render_add : s_pso_render_alpha;
    rp.setPSO(pso);
    rp.setBuffer(s->particle_buf, 0);
    rp.setBuffer(s->vs_uniforms,  1);
    rp.setBuffer(s->color_uniforms, 2);
    rp.draw(6, s->count);
    rp.end();
  }

  // ---- Pass 4: density splat (after update moved them) → next frame's read.
  // RenderPass::begin clears the buffer to zero, then additive halos accumulate.
  if (need_density) {
    DensityUniforms du = { s->interaction_radius, aspect_x, aspect_y, 0.f };
    s->density_uniforms.writeOne(du);
    auto rp = gpu::RenderPass::begin(s->density_tex, 0.0f, 0.0f, 0.0f, 0.0f);
    rp.setPSO(s_pso_density);
    rp.setBuffer(s->particle_buf, 0);
    rp.setBuffer(s->density_uniforms, 1);
    rp.draw(6, s->count);
    rp.end();
  }

  // ---- Pass 5 (debug): heat-map this frame's density buffer into tex_out. ----
  if (debug) {
    auto cp = gpu::ComputePass::begin();
    cp.setPSO(s_pso_density_debug);
    cp.setTexture(s->density_tex, 0, 0);
    cp.setSampler(s->sampler, 1);
    cp.setTexture(out, 2, 1);
    cp.dispatch((vp_w + 7) / 8, (vp_h + 7) / 8);
    cp.end();
  }

  gpu::Device::submit();
}

} // namespace flow_swarm
