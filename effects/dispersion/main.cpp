/*
 * warp.dispersion — Block-quantized UV-jitter sampler (v1 scaffold).
 *
 * Block sizes are internally quantized to a discrete ladder so a
 * sliding slider doesn't visibly sweep boundaries — when the slider
 * crosses a quantization step we ALSO re-roll the start offset so the
 * layout snaps to a fresh arrangement.
 *
 * Deferred for v2: noise-distribution shape (uniform/Gaussian),
 * per-axis temporal rate split, mosaic vs noise vs solid block modes.
 *
 * Class-like instance model: module_init() compiles the shared compute
 * PSO + publishes the schema once per type; each chain entry gets its
 * own State (params, runtime accumulators, RNG, uniform buffer) via
 * create(). All instance callbacks take `self`.
 */

#include <gpu.h>
#include <host.h>
#include "dispersion_shaders.h"

#include <cmath>
#include <cstdint>
#include <cstring>

namespace dispersion {

struct Uniforms {
  int32_t block_w;
  int32_t block_h;
  int32_t start_x;
  int32_t start_y;

  int32_t tick_index;
  float   offset_max;
  float   intensity;
  int32_t seed;
};
static_assert(sizeof(Uniforms) == 32, "Uniforms layout mismatch");

// Per-instance state. One per chain entry.
struct State {
  gpu::Buffer  uniform_buf;
  gpu::Sampler sampler;
  bool initialized = false;

  // Schema-mirrored params
  float vertical_block_norm   = 0.1f;
  float horizontal_block_norm = 0.1f;
  float offset_max            = 0.08f;
  float intensity             = 1.0f;
  float temporal_rate_hz      = 60.0f;
  int   quant_v               = 16;
  int   quant_h               = 16;
  int   block_max_v           = 64;
  int   block_max_h           = 64;
  int   seed                  = 12345;

  // Runtime state
  int      last_block_w = -1;
  int      last_block_h = -1;
  int      start_x      = 0;
  int      start_y      = 0;
  int      tick_index   = 0;
  double   tick_accum   = 0.0;
  uint32_t init_lcg     = 0xDEAFBEEFu;
};

// Type-shared: compiled once in module_init(), reused by every instance.
static gpu::ComputePSO s_pso;

static inline uint32_t lcg_next(uint32_t& s) {
  s = s * 1664525u + 1013904223u;
  return s;
}

// Quantize a [0, 1] norm slider to an integer in [1, max_block] using
// `levels` discrete steps.
static int quantize_block(float norm, int levels, int max_block) {
  if (norm < 0.0f) norm = 0.0f;
  if (norm > 1.0f) norm = 1.0f;
  if (levels < 2) levels = 2;
  if (max_block < 1) max_block = 1;
  int step = (int)std::floor(norm * (float)(levels - 1) + 0.5f);
  // Map step ∈ [0, levels-1] linearly to [1, max_block].
  float t = (float)step / (float)(levels - 1);
  int block = 1 + (int)std::floor(t * (float)(max_block - 1) + 0.5f);
  if (block < 1) block = 1;
  if (block > max_block) block = max_block;
  return block;
}

// Type-level setup: schema + the shared compute PSO. Runs once per type.
void module_init() {
  state::init("warp.dispersion", {1, 0, 1},
    state::Schema()
      // Top-level manual: high-level "what is this / how to use / what to try".
      .helpField("intro",
        "## Dispersion\n"
        "Chops the image into a grid of blocks and jitters each block's sampling "
        "offset, for a blocky *displacement / datamosh* feel. Block sizes snap to a "
        "discrete ladder so nudging a slider re-rolls the whole layout instead of "
        "smoothly sliding.\n\n"
        "**Try:** big blocks + high *Offset* for chunky glitch tears; tiny blocks + "
        "low *Offset* for a shimmering grain; drop *Rate* to freeze the pattern.")
      // --- Blocks ---
      .group("blocks", "Blocks")
        .groupHelp(
          "Sets how the frame is diced up. *Vertical* and *Horizontal Blocks* pick the "
          "cell size along each axis independently — pull them apart for tall streaks "
          "or wide bands rather than square tiles.")
      .floatField("vertical_block_norm",   0.1f,  0.0f, 1.0f, state::PrimaryInput).label("Vertical Blocks", "V Size")
      .floatField("horizontal_block_norm", 0.1f,  0.0f, 1.0f, state::PrimaryInput).label("Horizontal Blocks", "H Size")
      // --- Displacement ---
      .group("displacement", "Displacement")
      .floatField("offset_max",            0.08f, 0.0f, 0.5f, state::PrimaryInput).label("Offset", "Off")
      .floatField("intensity",             1.0f,  0.0f, 1.0f, state::PrimaryInput).label("Intensity", "Amt")
      .floatField("temporal_rate_hz",      60.0f, 0.0f, 60.0f, state::PrimaryInput).label("Rate", "Rate")
      // --- Block ladder ---
      .group("ladder", "Block Ladder")
        .groupHelp(
          "The *discrete* machinery behind the blocks. *Steps* set how many quantized "
          "sizes a slider can land on (fewer = coarser jumps); *Max Pixels* caps the "
          "largest block per axis. *Seed* re-rolls every block's random arrangement.")
      .intField  ("quantization_levels_vertical",   16, 4, 64, state::PrimaryInput).label("Vertical Steps", "V Stp")
      .intField  ("quantization_levels_horizontal", 16, 4, 64, state::PrimaryInput).label("Horizontal Steps", "H Stp")
      .intField  ("block_max_pixels_vertical",      64, 1, 512, state::PrimaryInput).label("Max Vertical Px", "V Max")
      .intField  ("block_max_pixels_horizontal",    64, 1, 512, state::PrimaryInput).label("Max Horizontal Px", "H Max")
      .intField  ("seed",                  12345, 0,    65535, state::PrimaryInput).label("Seed", "Seed")
      .capability(state::Capability::SeekableApproximate)
      .textureField("tex_in",  state::PrimaryInput)
      .textureField("tex_out", state::PrimaryOutput)
  );

  if (gpu::Device::backend() == gpu::Backend::None) return;

  state::registerShaderSPV("dispersion_render", RENDER_SPV, RENDER_SPV_SIZE);
  auto cs = gpu::Device::createShaderModuleByName("dispersion_render");
  if (!cs) return;

  s_pso = gpu::Device::createComputePSO(cs, "main", gpu::Bindings()
      .tex2d(0)
      .sampler(1)
      .storageTex2d(2)
      .uniform(3));

  state::log("dispersion: module initialized");
}

// Per-instance construction: allocate State + its own uniform buffer + sampler.
void* create() {
  auto* s = new State();
  s->uniform_buf = gpu::Device::createBuffer(sizeof(Uniforms), gpu::BufferUsage::Uniform);
  s->sampler = gpu::Device::createSampler(gpu::FilterMode::Linear, gpu::AddressMode::ClampToEdge);
  return s;
}

void destroy(void* self) {
  auto* s = static_cast<State*>(self);
  if (!s) return;
  s->uniform_buf.release();
  s->sampler.release();
  delete s;
}

// Per-instance init tail: reset runtime accumulators + mark ready.
void init(void* self) {
  auto* s = static_cast<State*>(self);
  if (!s) return;
  s->last_block_w  = -1;
  s->last_block_h  = -1;
  s->start_x       = 0;
  s->start_y       = 0;
  s->tick_index    = 0;
  s->tick_accum    = 0.0;
  s->init_lcg      = 0xDEAFBEEFu;
  if (!s_pso.valid()) return;
  if (!s->uniform_buf.valid()) return;
  s->initialized = true;
}

void tick(void* self, double dt) {
  auto* s = static_cast<State*>(self);
  if (!s) return;
  if (s->temporal_rate_hz > 0.0f) {
    s->tick_accum += dt * (double)s->temporal_rate_hz;
    while (s->tick_accum >= 1.0) {
      s->tick_index++;
      s->tick_accum -= 1.0;
    }
  }
}


void on_state_patched(void* self, int n, const char* pb, const int* off,
                      const int* len, const int* ops) {
  auto* s = static_cast<State*>(self);
  if (!s) return;
  for (int i = 0; i < n; i++) {
    if (ops[i] != state::PatchReplace) continue;
    const char* path = pb + off[i];
    int plen = len[i];
    if      (state::pathIs(path, plen, "vertical_block_norm"))   s->vertical_block_norm = state::patchFloat(i);
    else if (state::pathIs(path, plen, "horizontal_block_norm")) s->horizontal_block_norm = state::patchFloat(i);
    else if (state::pathIs(path, plen, "offset_max"))            s->offset_max = state::patchFloat(i);
    else if (state::pathIs(path, plen, "intensity"))             s->intensity = state::patchFloat(i);
    else if (state::pathIs(path, plen, "temporal_rate_hz"))      s->temporal_rate_hz = state::patchFloat(i);
    else if (state::pathIs(path, plen, "quantization_levels_vertical"))   s->quant_v = (int)state::patchFloat(i);
    else if (state::pathIs(path, plen, "quantization_levels_horizontal")) s->quant_h = (int)state::patchFloat(i);
    else if (state::pathIs(path, plen, "block_max_pixels_vertical"))      s->block_max_v = (int)state::patchFloat(i);
    else if (state::pathIs(path, plen, "block_max_pixels_horizontal"))    s->block_max_h = (int)state::patchFloat(i);
    else if (state::pathIs(path, plen, "seed"))                  s->seed = (int)state::patchFloat(i);
  }
}

void render(void* self, int vp_w, int vp_h) {
  auto* s = static_cast<State*>(self);
  if (!s || !s->initialized || vp_w <= 0 || vp_h <= 0) return;
  auto in  = gpu::Device::textureForField("tex_in");
  auto out = gpu::Device::textureForField("tex_out");
  if (!in.valid() || !out.valid()) return;

  // Resolve block sizes via the discrete ladder. Anchor block_max to
  // the viewport so the slider scales sensibly when the canvas is small.
  int max_v = (s->block_max_v < vp_h) ? s->block_max_v : vp_h;
  int max_h = (s->block_max_h < vp_w) ? s->block_max_h : vp_w;
  int block_h = quantize_block(s->vertical_block_norm,   s->quant_v, max_v);
  int block_w = quantize_block(s->horizontal_block_norm, s->quant_h, max_h);

  // When block size changes across a quantization step, reroll the
  // start offset so the layout snaps fresh instead of sliding.
  if (block_w != s->last_block_w) {
    s->start_x = (int)(lcg_next(s->init_lcg) % (uint32_t)block_w);
    s->last_block_w = block_w;
  }
  if (block_h != s->last_block_h) {
    s->start_y = (int)(lcg_next(s->init_lcg) % (uint32_t)block_h);
    s->last_block_h = block_h;
  }

  Uniforms u = {};
  u.block_w    = block_w;
  u.block_h    = block_h;
  u.start_x    = s->start_x;
  u.start_y    = s->start_y;
  u.tick_index = s->tick_index;
  u.offset_max = s->offset_max;
  u.intensity  = s->intensity;
  u.seed       = s->seed;
  s->uniform_buf.writeOne(u);

  auto cp = gpu::ComputePass::begin();
  cp.setPSO(s_pso);
  cp.setTexture(in, 0, 0);
  cp.setSampler(s->sampler, 1);
  cp.setTexture(out, 2, 1);
  cp.setBuffer(s->uniform_buf, 3);
  cp.dispatch((vp_w + 7) / 8, (vp_h + 7) / 8);
  cp.end();

  gpu::Device::submit();
}

} // namespace dispersion
