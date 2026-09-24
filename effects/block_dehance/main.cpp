/*
 * filter.glitch.block_dehance — glitch rectangles that "dehance" the input.
 *
 * A GPU-resident pool of rectangles cycles continuously (no trigger). Each
 * rect, on respawn, bright-seeks the mask (K samples, softmax by temperature)
 * for its position and samples a dehance MODE by weight — black-fill, mosaic
 * downres, or noise — sticky for its lifetime. The update compute pass owns
 * the pool (lifecycle + respawn + mask sampling, so no CPU readback); the
 * render pass paints the covered pixels per mode; a motion pass passes
 * upstream motion through.
 *
 * Stateful (persistent GPU buffer) + instance ABI. Continuous, so no
 * gate/trigger surface.
 */

#include <gpu.h>
#include <host.h>
#include <effect_utils.h>
#include "block_dehance_shaders.h"

#include <cmath>
#include <cstdint>

namespace block_dehance {

static constexpr int POOL_HARD_MAX = 128;

struct GpuRect {                          // matches Rect in common.hlsl (48 B)
  float pos_size[4];
  float state[4];
  float params[4];
};
static_assert(sizeof(GpuRect) == 48, "GpuRect layout mismatch");

struct UpdateUniforms {
  uint32_t count, pool_max, frame_index, do_reset;
  float    dt, mask_temperature, life_s, respawn_delay_s;
  float    life_jitter, rect_width, rect_height, rect_size_jitter;
  float    mode_black_w, mode_mosaic_w, mode_noise_w, mosaic_cell_size;
  float    mosaic_cell_jitter; uint32_t mask_samples, seed; float move_chance;
  float    move_amount, move_delay_max, spawn_amount, spawn_radius;
  float    spawn_softness, aspect_x, aspect_y, _pad0;
};
static_assert(sizeof(UpdateUniforms) == 112, "UpdateUniforms layout mismatch");

struct RenderUniforms {
  uint32_t count, pool_max, tick_index, debug_show;
  float    time, flicker_rate_hz, flicker_duty, noise_intensity;
  float    fill_r, fill_g, fill_b, fill_a;
  uint32_t noise_temporal, noise_color_mode; float _pad0, _pad1;
};
static_assert(sizeof(RenderUniforms) == 64, "RenderUniforms layout mismatch");

struct State {
  gpu::Buffer  rect_buf;
  gpu::Buffer  update_uniform_buf;
  gpu::Buffer  render_uniform_buf;
  gpu::Sampler sampler;
  gpu::Texture motion_tex;
  gpu::Texture zero_motion_tex;
  int          motion_w = 0, motion_h = 0;
  bool         initialized = false;

  // Schema-mirrored params.
  int   count                = 6;
  float life_s               = 1.5f;
  float respawn_delay_s       = 1.0f;
  float life_jitter          = 0.3f;
  float rect_width           = 0.18f;
  float rect_height          = 0.06f;
  float rect_size_jitter     = 0.4f;
  float move_chance          = 0.0f;
  float move_amount          = 0.03f;
  float move_delay_max       = 0.3f;
  float mask_temperature     = 0.5f;
  float spawn_amount         = 0.0f;
  float spawn_radius         = 0.5f;
  float spawn_softness       = 0.25f;
  float mode_black_weight    = 0.33f;
  float mode_mosaic_weight   = 0.33f;
  float mode_noise_weight    = 0.33f;
  float fill_r = 0.0f, fill_g = 0.0f, fill_b = 0.0f, fill_a = 1.0f;
  float mosaic_cell_size     = 0.02f;
  float mosaic_cell_jitter   = 0.5f;
  bool  noise_temporal       = true;
  int   noise_color_mode     = 0;
  float noise_intensity      = 1.0f;
  int   pool_max             = 32;
  int   mask_samples         = 8;
  float flicker_rate_hz      = 0.0f;
  float flicker_duty         = 0.5f;
  int   seed                 = 12345;
  bool  debug_show_rects     = false;

  // Runtime.
  uint32_t frame_index = 0;
  float    time        = 0.0f;
  float    frame_dt    = 0.0f;
  bool     needs_reset = true;
  int      last_pool_max = -1;
  int      last_seed     = 0x7FFFFFFF;
};

static gpu::ComputePSO s_pso_update;
static gpu::ComputePSO s_pso_render;
static gpu::ComputePSO s_pso_motion;

static inline float clampf(float v, float lo, float hi) { return v < lo ? lo : (v > hi ? hi : v); }
static inline int   clampi(int v, int lo, int hi)       { return v < lo ? lo : (v > hi ? hi : v); }

void module_init() {
  state::init("filter.glitch.block_dehance", {1, 1, 0},
    state::Schema()
      // Top-level manual: high-level "what is this / how to use / what to try".
      .helpField("intro",
        "## Block Dehance\n"
        "Scatters glitch rectangles across the image that *dehance* whatever they "
        "cover — crushing it to a fill colour, blocking it into a coarse mosaic, or "
        "replacing it with noise. Rectangles continuously bright-seek the picture (or "
        "a wired mask), so they cluster on the brightest, most eye-catching regions.\n\n"
        "**Try:** balance the three *Mode Weights* to mix black / mosaic / noise "
        "blocks; raise *Count* and drop *Lifetime* for a frantic strobe; wire a *Mask* "
        "input to steer the blocks onto specific areas; bias the *Spawn Area* to pin "
        "blocks to (or away from) the centre; add *Flicker* for a broken-signal "
        "stutter.")
      // --- Rectangles ---
      .group("rects", "Rectangles")
        .groupHelp(
          "The pool of glitch rectangles. *Count* is how many are live at once; "
          "*Lifetime* and *Respawn Delay* set how fast they cycle. *Width* / *Height* "
          "size them and the *Jitter* knobs vary each one. A rectangle picks its "
          "dehance mode on spawn and keeps it for its whole life.")
      .intField  ("count",                 6, 0, 64,            state::PrimaryInput).label("Count", "Count")
      .floatField("life_s",                1.5f, 0.05f, 10.0f,  state::PrimaryInput).label("Lifetime", "Life")
      .floatField("respawn_delay_s",       1.0f, 0.0f, 10.0f,   state::PrimaryInput).label("Respawn Delay", "Delay")
      .floatField("life_jitter",           0.3f, 0.0f, 1.0f,    state::PrimaryInput).label("Life Jitter", "LfJit")
      .floatField("rect_width",            0.18f, 0.005f, 1.0f, state::PrimaryInput).label("Width", "Width")
      .floatField("rect_height",           0.06f, 0.005f, 1.0f, state::PrimaryInput).label("Height", "Height")
      .floatField("rect_size_jitter",      0.4f, 0.0f, 1.0f,    state::PrimaryInput).label("Size Jitter", "SzJit")
      // --- Movement ---
      .group("movement", "Movement")
        .groupHelp(
          "Optional per-rectangle jumps. *Move Chance* sets how often a rect hops to "
          "a new spot; *Move Amount* is how far, and *Move Delay* spaces the hops out "
          "in time. Leave *Move Chance* at 0 to keep blocks pinned where they spawn.")
      .floatField("move_chance",           0.0f, 0.0f, 1.0f,    state::PrimaryInput).label("Move Chance", "Move")
      .floatField("move_amount",           0.03f, 0.0f, 0.5f,   state::PrimaryInput).label("Move Amount", "Amt")
      .floatField("move_delay_max",        0.3f, 0.0f, 5.0f,    state::PrimaryInput).label("Move Delay", "MvDly")
      // --- Mask Targeting ---
      .group("mask", "Mask Targeting")
        .groupHelp(
          "Rectangles bright-seek the image (or a wired *Mask* input) to place "
          "themselves on the most prominent regions. *Mask Temperature* controls how "
          "greedy that search is — low values snap hard to the brightest spot, high "
          "values spread the placement out.")
      .floatField("mask_temperature",      0.5f, 0.0f, 4.0f,    state::PrimaryInput).label("Mask Temperature", "Temp")
      // --- Spawn Area ---
      .group("spawn", "Spawn Area")
        .groupHelp(
          "A vignette-like radial gate on where new rectangles may spawn. "
          "*Spawn Bias* is bipolar: positive confines spawning to the centre, "
          "negative pushes it out to the rim; 0 leaves the whole frame open. "
          "*Radius* sets where the falloff starts (0 = the centre, 1 = the "
          "frame edge) and *Softness* feathers it from a hard border into a "
          "gradual probability fade.")
      .floatField("spawn_amount",          0.0f, -1.0f, 1.0f,   state::PrimaryInput).label("Spawn Bias", "Bias")
      .floatField("spawn_radius",          0.5f, 0.0f, 1.0f,    state::PrimaryInput).label("Spawn Radius", "Rad")
      .floatField("spawn_softness",        0.25f, 0.0f, 1.0f,   state::PrimaryInput).label("Spawn Softness", "Soft")
      // --- Dehance Modes ---
      .group("modes", "Dehance Modes")
        .groupHelp(
          "Each rectangle randomly picks one dehance style, weighted by these knobs: "
          "*Black* crushes to the fill colour, *Mosaic* blocks the area into big "
          "pixels, *Noise* replaces it with static. Set a weight to 0 to drop that "
          "style. *Fill Colour* is what the black mode paints.")
      .floatField("mode_black_weight",     0.33f, 0.0f, 1.0f,   state::PrimaryInput).label("Black Weight", "Black")
      .floatField("mode_mosaic_weight",    0.33f, 0.0f, 1.0f,   state::PrimaryInput).label("Mosaic Weight", "Mosaic")
      .floatField("mode_noise_weight",     0.33f, 0.0f, 1.0f,   state::PrimaryInput).label("Noise Weight", "Noise")
      .rgbaField ("fill_color",            0.0f, 0.0f, 0.0f, 1.0f, state::SecondaryInput).label("Fill Colour", "Fill")
      // --- Mosaic ---
      .group("mosaic", "Mosaic")
      .floatField("mosaic_cell_size",      0.02f, 0.001f, 0.2f, state::PrimaryInput).label("Cell Size", "Cell")
      .floatField("mosaic_cell_size_jitter", 0.5f, 0.0f, 1.0f,  state::PrimaryInput).label("Cell Jitter", "CJit")
      // --- Noise ---
      .group("noise", "Noise")
      .boolField ("noise_temporal",        true,                state::PrimaryInput).label("Temporal Noise", "Temp")
      .selectField("noise_color_mode",     0, state::PrimaryInput,
                   {{"rgb", 0}, {"grayscale", 1}, {"luma_preserve", 2}}).label("Noise Colour", "Colour")
      .floatField("noise_intensity",       1.0f, 0.0f, 1.0f,    state::PrimaryInput).label("Noise Intensity", "Int")
      // --- Pool ---
      .group("pool", "Pool")
      .intField  ("pool_max",              32, 8, 128,          state::PrimaryInput).label("Pool Max", "Pool")
      .intField  ("mask_samples",          8, 4, 16,            state::PrimaryInput).label("Mask Samples", "Smpls")
      // --- Flicker ---
      .group("flicker", "Flicker")
      .floatField("flicker_rate_hz",       0.0f, 0.0f, 60.0f,   state::PrimaryInput).label("Flicker Rate", "Rate")
      .floatField("flicker_duty",          0.5f, 0.0f, 1.0f,    state::PrimaryInput).label("Flicker Duty", "Duty")
      // --- Seed ---
      .group("seed", "Seed")
      .intField  ("seed",                  12345, 0, 0x7FFFFFFF, state::PrimaryInput).label("Seed", "Seed")
      // --- Debug ---
      .group("debug", "Debug")
      .boolField ("debug_show_rects",      false,               state::PrimaryInput).label("Show Rects", "Rects")
      .capability(state::Capability::SeekableApproximate)
      .textureField("tex_in",  state::PrimaryInput)
      .textureField("mask_in", state::SecondaryInput)
      .textureField("tex_out", state::PrimaryOutput)
      .renderOutputs(state::PrimaryOutput)
      .renderOutputs(state::PrimaryInput, "render_outputs_in")
  );

  if (gpu::Device::backend() == gpu::Backend::None) return;

  state::registerShaderSPV("block_dehance_update", UPDATE_SPV, UPDATE_SPV_SIZE);
  state::registerShaderSPV("block_dehance_render", RENDER_SPV, RENDER_SPV_SIZE);
  state::registerShaderSPV("block_dehance_motion", MOTION_SPV, MOTION_SPV_SIZE,
                           "rgba16float", "write");
  auto cs_update = gpu::Device::createShaderModuleByName("block_dehance_update");
  auto cs_render = gpu::Device::createShaderModuleByName("block_dehance_render");
  auto cs_motion = gpu::Device::createShaderModuleByName("block_dehance_motion");
  if (!cs_update || !cs_render || !cs_motion) return;

  s_pso_update = gpu::Device::createComputePSO(cs_update, "main", gpu::Bindings()
      .storageRW(0).tex2d(1).sampler(2).uniform(3));
  s_pso_render = gpu::Device::createComputePSO(cs_render, "main", gpu::Bindings()
      .tex2d(0).sampler(1).storageTex2d(2).uniform(3).storage(4));
  s_pso_motion = gpu::Device::createComputePSO(cs_motion, "main", gpu::Bindings()
      .tex2d(0).storageTex2d(1, gpu::TextureFormat::RGBA16F));

  state::log("block_dehance: module initialized");
}

void* create() {
  auto* s = new State();
  s->rect_buf           = gpu::Device::createBuffer(sizeof(GpuRect) * POOL_HARD_MAX, gpu::BufferUsage::Storage);
  s->update_uniform_buf = gpu::Device::createBuffer(sizeof(UpdateUniforms), gpu::BufferUsage::Uniform);
  s->render_uniform_buf = gpu::Device::createBuffer(sizeof(RenderUniforms), gpu::BufferUsage::Uniform);
  s->sampler = gpu::Device::createSampler(gpu::FilterMode::Linear, gpu::AddressMode::ClampToEdge);
  return s;
}

void destroy(void* self) {
  auto* s = static_cast<State*>(self);
  if (!s) return;
  s->rect_buf.release();
  s->update_uniform_buf.release();
  s->render_uniform_buf.release();
  s->sampler.release();
  s->motion_tex.release();
  s->zero_motion_tex.release();
  delete s;
}

void init(void* self) {
  auto* s = static_cast<State*>(self);
  if (!s) return;
  s->frame_index = 0;
  s->time = 0.0f;
  s->frame_dt = 0.0f;
  s->needs_reset = true;
  s->last_pool_max = -1;
  s->last_seed = 0x7FFFFFFF;
  s->motion_w = s->motion_h = 0;
  if (!s_pso_update.valid() || !s_pso_render.valid() || !s_pso_motion.valid()) return;
  if (!s->rect_buf.valid() || !s->update_uniform_buf.valid() || !s->render_uniform_buf.valid()) return;
  s->initialized = true;
}

void tick(void* self, double dt) {
  auto* s = static_cast<State*>(self);
  if (!s || !s->initialized) return;
  float fdt = (float)(dt > 0.1 ? 0.1 : (dt < 0.0 ? 0.0 : dt));
  s->frame_dt = fdt;
  s->time += fdt;
  s->frame_index++;
}


void on_state_patched(void* self, int n, const char* pb, const int* off,
                      const int* len, const int* ops) {
  auto* s = static_cast<State*>(self);
  if (!s) return;
  for (int i = 0; i < n; i++) {
    if (ops[i] != state::PatchReplace) continue;
    const char* path = pb + off[i];
    int plen = len[i];
    if      (state::pathIs(path, plen, "count"))                 s->count = (int)state::patchFloat(i);
    else if (state::pathIs(path, plen, "life_s"))               s->life_s = state::patchFloat(i);
    else if (state::pathIs(path, plen, "respawn_delay_s"))      s->respawn_delay_s = state::patchFloat(i);
    else if (state::pathIs(path, plen, "life_jitter"))          s->life_jitter = state::patchFloat(i);
    else if (state::pathIs(path, plen, "rect_width"))           s->rect_width = state::patchFloat(i);
    else if (state::pathIs(path, plen, "rect_height"))          s->rect_height = state::patchFloat(i);
    else if (state::pathIs(path, plen, "rect_size_jitter"))     s->rect_size_jitter = state::patchFloat(i);
    else if (state::pathIs(path, plen, "move_chance"))          s->move_chance = state::patchFloat(i);
    else if (state::pathIs(path, plen, "move_amount"))          s->move_amount = state::patchFloat(i);
    else if (state::pathIs(path, plen, "move_delay_max"))       s->move_delay_max = state::patchFloat(i);
    else if (state::pathIs(path, plen, "mask_temperature"))     s->mask_temperature = state::patchFloat(i);
    else if (state::pathIs(path, plen, "spawn_amount"))         s->spawn_amount = state::patchFloat(i);
    else if (state::pathIs(path, plen, "spawn_radius"))         s->spawn_radius = state::patchFloat(i);
    else if (state::pathIs(path, plen, "spawn_softness"))       s->spawn_softness = state::patchFloat(i);
    else if (state::pathIs(path, plen, "mode_black_weight"))    s->mode_black_weight = state::patchFloat(i);
    else if (state::pathIs(path, plen, "mode_mosaic_weight"))   s->mode_mosaic_weight = state::patchFloat(i);
    else if (state::pathIs(path, plen, "mode_noise_weight"))    s->mode_noise_weight = state::patchFloat(i);
    else if (state::pathIs(path, plen, "fill_color")) {
      auto v = state::patchVec4(i);
      s->fill_r = v.x; s->fill_g = v.y; s->fill_b = v.z; s->fill_a = v.w;
    }
    else if (state::pathIs(path, plen, "mosaic_cell_size"))     s->mosaic_cell_size = state::patchFloat(i);
    else if (state::pathIs(path, plen, "mosaic_cell_size_jitter")) s->mosaic_cell_jitter = state::patchFloat(i);
    else if (state::pathIs(path, plen, "noise_temporal"))       s->noise_temporal = state::patchFloat(i) != 0.0f;
    else if (state::pathIs(path, plen, "noise_color_mode"))     s->noise_color_mode = (int)state::patchFloat(i);
    else if (state::pathIs(path, plen, "noise_intensity"))      s->noise_intensity = state::patchFloat(i);
    else if (state::pathIs(path, plen, "pool_max"))             s->pool_max = (int)state::patchFloat(i);
    else if (state::pathIs(path, plen, "mask_samples"))         s->mask_samples = (int)state::patchFloat(i);
    else if (state::pathIs(path, plen, "flicker_rate_hz"))      s->flicker_rate_hz = state::patchFloat(i);
    else if (state::pathIs(path, plen, "flicker_duty"))         s->flicker_duty = state::patchFloat(i);
    else if (state::pathIs(path, plen, "seed"))                 s->seed = (int)state::patchFloat(i);
    else if (state::pathIs(path, plen, "debug_show_rects"))     s->debug_show_rects = state::patchFloat(i) != 0.0f;
  }
}

void render(void* self, int vp_w, int vp_h) {
  auto* s = static_cast<State*>(self);
  if (!s || !s->initialized || vp_w <= 0 || vp_h <= 0) return;
  auto in  = gpu::Device::textureForField("tex_in");
  auto out = gpu::Device::textureForField("tex_out");
  if (!in.valid() || !out.valid()) return;
  auto mask = gpu::Device::textureForField("mask_in");
  if (!mask.valid()) mask = in;                 // fall back to tex_in

  int pool_max = clampi(s->pool_max, 8, POOL_HARD_MAX);
  int count    = clampi(s->count, 0, pool_max);

  // Re-stagger the pool when the cap or seed changes (or on first frame).
  if (pool_max != s->last_pool_max || s->seed != s->last_seed) {
    s->needs_reset = true;
    s->last_pool_max = pool_max;
    s->last_seed = s->seed;
  }

  // --- Pass 1: update the rect pool. ---
  UpdateUniforms uu = {};
  uu.count = (uint32_t)count;
  uu.pool_max = (uint32_t)pool_max;
  uu.frame_index = s->frame_index;
  uu.do_reset = s->needs_reset ? 1u : 0u;
  uu.dt = s->frame_dt;
  uu.mask_temperature = clampf(s->mask_temperature, 0.0f, 4.0f);
  uu.life_s = clampf(s->life_s, 0.05f, 10.0f);
  uu.respawn_delay_s = clampf(s->respawn_delay_s, 0.0f, 10.0f);
  uu.life_jitter = clampf(s->life_jitter, 0.0f, 1.0f);
  uu.rect_width = clampf(s->rect_width, 0.005f, 1.0f);
  uu.rect_height = clampf(s->rect_height, 0.005f, 1.0f);
  uu.rect_size_jitter = clampf(s->rect_size_jitter, 0.0f, 1.0f);
  uu.mode_black_w = clampf(s->mode_black_weight, 0.0f, 1.0f);
  uu.mode_mosaic_w = clampf(s->mode_mosaic_weight, 0.0f, 1.0f);
  uu.mode_noise_w = clampf(s->mode_noise_weight, 0.0f, 1.0f);
  uu.mosaic_cell_size = clampf(s->mosaic_cell_size, 0.001f, 0.2f);
  uu.mosaic_cell_jitter = clampf(s->mosaic_cell_jitter, 0.0f, 1.0f);
  uu.mask_samples = (uint32_t)clampi(s->mask_samples, 4, 16);
  uu.seed = (uint32_t)s->seed;
  uu.move_chance = clampf(s->move_chance, 0.0f, 1.0f);
  uu.move_amount = clampf(s->move_amount, 0.0f, 0.5f);
  uu.move_delay_max = clampf(s->move_delay_max, 0.0f, 5.0f);
  uu.spawn_amount = clampf(s->spawn_amount, -1.0f, 1.0f);
  uu.spawn_radius = clampf(s->spawn_radius, 0.0f, 1.0f);
  uu.spawn_softness = clampf(s->spawn_softness, 0.0f, 1.0f);
  auto [ax, ay] = fx::coverSquare(vp_w, vp_h);
  uu.aspect_x = ax;
  uu.aspect_y = ay;
  s->update_uniform_buf.writeOne(uu);
  {
    auto cp = gpu::ComputePass::begin();
    cp.setPSO(s_pso_update);
    cp.setBuffer(s->rect_buf, 0);
    cp.setTexture(mask, 1, 0);
    cp.setSampler(s->sampler, 2);
    cp.setBuffer(s->update_uniform_buf, 3);
    cp.dispatch((pool_max + 63) / 64, 1, 1);
    cp.end();
  }

  // --- Pass 2: render. ---
  RenderUniforms ru = {};
  ru.count = (uint32_t)count;
  ru.pool_max = (uint32_t)pool_max;
  ru.tick_index = s->frame_index;
  ru.debug_show = s->debug_show_rects ? 1u : 0u;
  ru.time = s->time;
  ru.flicker_rate_hz = clampf(s->flicker_rate_hz, 0.0f, 60.0f);
  ru.flicker_duty = clampf(s->flicker_duty, 0.0f, 1.0f);
  ru.noise_intensity = clampf(s->noise_intensity, 0.0f, 1.0f);
  ru.fill_r = s->fill_r; ru.fill_g = s->fill_g; ru.fill_b = s->fill_b; ru.fill_a = s->fill_a;
  ru.noise_temporal = s->noise_temporal ? 1u : 0u;
  ru.noise_color_mode = (uint32_t)clampi(s->noise_color_mode, 0, 2);
  s->render_uniform_buf.writeOne(ru);
  {
    auto cp = gpu::ComputePass::begin();
    cp.setPSO(s_pso_render);
    cp.setTexture(in, 0, 0);
    cp.setSampler(s->sampler, 1);
    cp.setTexture(out, 2, 1);
    cp.setBuffer(s->render_uniform_buf, 3);
    cp.setBuffer(s->rect_buf, 4);
    cp.dispatch((vp_w + 7) / 8, (vp_h + 7) / 8);
    cp.end();
  }

  // --- Pass 3: motion passthrough. ---
  if (state::isOutputConnected("render_outputs")) {
    if (!s->motion_tex.valid() || s->motion_w != vp_w || s->motion_h != vp_h) {
      s->motion_tex = gpu::Device::createTexture(vp_w, vp_h, gpu::TextureFormat::RGBA16F);
      s->motion_w = vp_w; s->motion_h = vp_h;
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
        cp.setTexture(upstream, 0, 0);
        cp.setTexture(s->motion_tex, 1, 1);
        cp.dispatch((vp_w + 7) / 8, (vp_h + 7) / 8);
        cp.end();
      }
    }
  }

  gpu::Device::submit();

  s->needs_reset = false;
  s->frame_dt = 0.0f;
}

} // namespace block_dehance
