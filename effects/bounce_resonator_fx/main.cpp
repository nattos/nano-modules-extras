/*
 * filter.light.bounce_resonator — the passive, input-excited "fx" version of
 * source.light.bounce_resonator, for the LED bars.
 *
 * The grid is 4 bars × `rows` cells (default 10 — one per segment). Every
 * row is its own copy of the 4-bar network, and all rows step through the
 * SAME cycling matrix at the same hop, so the impulse-response pattern
 * travels across the whole grid at once while each cell keeps its own
 * energy — the image survives into the tail.
 *
 * The waveguide is SIGNED and its mixing ORTHOGONAL, as in an audio
 * feedback-delay-network reverb: fx::DiffusionNetwork4 supplies the seeded
 * shuffles and per-edge hue shifts, and orthogonalize() folds Spread in as
 * rotations. Energy is spread around rather than averaged into a flat level,
 * so a long, wide tail doesn't wash out; its negative half is the overshoot.
 * Decay is separate and per second (Tail).
 *
 * No triggers: the input excites it continuously.
 *   grains — brightness sets a per-cell Poisson rate of fixed-size kicks, so
 *            a bright picture fizzes rather than floods;
 *   edges  — a rise in a cell's luma kicks it by the rise.
 *
 * A/C output: each cell subtracts its own running mean, so steady energy
 * cancels and the swing is what shows — positive lights, negative chokes the
 * image toward black. Contrast then rides one gain for the whole grid (swing
 * ÷ its running RMS, floored) so a fading tail keeps swinging at full scale. Hops cap at one per frame, so a Cycle
 * Rate at or past the frame rate flips energy cell to cell every frame:
 * solid strobing.
 *
 * Three passes, state GPU-resident (no readback):
 *   reduce — one thread per cell: box-sample the input → mean colour + luma
 *   sim    — single thread: decay, hop, excite, A/C, contrast
 *   render — full-res composite over the input
 */

#include <gpu.h>
#include <host.h>
#include <val.h>
#include <effect_utils.h>
#include <effect_diffusion_network.h>
#include "bounce_resonator_fx_shaders.h"

#include <algorithm>
#include <cmath>
#include <cstdint>

namespace bounce_resonator_fx {

static constexpr int BARS     = 4;
static constexpr int MAX_ROWS = 16;
static constexpr int MAX_CELLS = BARS * MAX_ROWS;

// --- GPU-shared layouts (must match common.hlsl / the HLSL cbuffers) ---

struct CellStat  { float r, g, b, luma; };
static_assert(sizeof(CellStat) == 16, "CellStat layout mismatch");

struct CellState { float v, h, s, mean, luma_prev, swing, pad0, pad1; };
static_assert(sizeof(CellState) == 32, "CellState layout mismatch");

struct NetState  { float env, rms2, pad0, pad1; };
static_assert(sizeof(NetState) == 16, "NetState layout mismatch");

struct ReduceUniforms { int32_t rows, tex_w, tex_h, pad; };
static_assert(sizeof(ReduceUniforms) == 16, "ReduceUniforms layout mismatch");

struct SimUniforms {
  float    decay_k, decay_shaping, hue_converge, home_hue;
  int32_t  pattern_count, hop_idx, do_hop, rows;
  float    grain_rate, grain_size, edge_excite, input_color;
  float    band_hue, band_sat, ac_amount, ac_time;
  float    dt; uint32_t frame, seed; int32_t do_reset;
  float    contrast, contrast_floor, pad0, pad1;
};
static_assert(sizeof(SimUniforms) == 96, "SimUniforms layout mismatch");

struct RenderUniforms {
  float   band_val, intensity, input_opacity, chroma_hold;
  float   black_depth; int32_t rows; float pad0, pad1;
};
static_assert(sizeof(RenderUniforms) == 32, "RenderUniforms layout mismatch");

struct MatsBuf { float data[fx::DiffusionNetwork4::kMaxN * 48]; };

// Per-instance state. One per chain entry.
struct State {
  // --- Per-instance GPU resources ---
  gpu::Buffer stats_buf;      // transient per-cell input stats
  gpu::Buffer cells_buf;      // PERSISTENT per-cell resonator state
  gpu::Buffer net_buf;        // PERSISTENT network envelope
  gpu::Buffer mats_buf;
  gpu::Buffer reduce_uniform_buf;
  gpu::Buffer sim_uniform_buf;
  gpu::Buffer render_uniform_buf;
  bool initialized = false;

  // --- Schema-mirrored params ---
  float grain_rate      = 8.0f;
  float grain_size      = 0.6f;
  float edge_excite     = 2.0f;
  float input_color     = 1.0f;
  float tail            = 0.55f;   // 0..1 → 0.05 s .. 60 s, exponential; 1 = hold
  float spread          = 0.30f;
  float decay_shaping   = 0.0f;
  float hue_spread      = 0.0f;
  float hue_converge    = 0.0f;
  int   seed            = 0;
  int   pattern_count   = 4;
  float cycle_rate      = 12.0f;
  float ac_amount       = 1.0f;
  float ac_time         = 0.5f;
  float black_depth     = 1.5f;
  float contrast        = 0.0f;
  float contrast_floor  = 0.05f;
  float color_r = 1.0f, color_g = 0.92f, color_b = 0.78f;
  float band_hue = 0.0f, band_sat = 0.0f, band_val = 1.0f;
  float intensity       = 1.0f;
  float chroma_hold     = 0.0f;
  float input_opacity   = 1.0f;
  int   rows            = 10;

  // --- Runtime state ---
  fx::DiffusionNetwork4 net;     // CPU: builds + exports the cycling matrices
  double   pending_dt  = 0.0;    // accumulated tick dt, consumed by render()
  float    accum       = 0.0f;   // hop accumulator (seconds)
  int      hop_idx     = 0;
  uint32_t frame       = 0;      // grain hash clock
  bool     needs_reset = true;   // zero the GPU state on next dispatch
};

// Type-shared: compiled once in module_init().
static gpu::ComputePSO s_pso_reduce;
static gpu::ComputePSO s_pso_sim;
static gpu::ComputePSO s_pso_render;

static inline float clampf(float v, float lo, float hi) {
  return v < lo ? lo : (v > hi ? hi : v);
}
static inline int clampi(int v, int lo, int hi) {
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

// Tail knob (0..1) → the energy decay rate, per second. The knob is
// exponential in time — 0.05 s at 0, 60 s at 1 — so each step up the knob
// multiplies the tail rather than creeping toward a cliff at 1.0; the tail
// is the time to fall to 1% (ln 100 / T). The very top holds forever.
static float decay_rate(float knob) {
  knob = clampf(knob, 0.0f, 1.0f);
  if (knob >= 0.999f) return 0.0f;
  float seconds = 0.05f * std::pow(1200.0f, knob);
  return 4.60517f / seconds;
}

static uint32_t hash_u32(uint32_t a, uint32_t b, uint32_t c) {
  uint32_t h = a ^ (b * 0x9E3779B1u) ^ (c * 0x85EBCA77u);
  h ^= h >> 16; h *= 0x7FEB352Du; h ^= h >> 15; h *= 0x846CA68Bu; h ^= h >> 16;
  return h;
}

// Turn each exported pattern's shuffle P (the spread-0 derangement) into an
// ORTHOGONAL mixing matrix M = P · B, where B is a two-stage butterfly of
// Givens rotations — pairs (0,1),(2,3), then (0,2),(1,3). Each rotation's
// angle is spread · π/4 · a seeded ±(0.6..1): spread 0 is the bare shuffle,
// spread 1 is close to a signed 4×4 Hadamard (every bar feeds every bar, with
// mixed signs). Orthogonal means Σv² survives each hop: energy is spread
// around, never averaged into a flat level, and the signs make contributions
// interfere — the fix for a long, well-spread tail washing out to grey.
static void orthogonalize(float* mats, int pattern_count, float spread, uint32_t seed) {
  static const int kPairs[4][2] = {{0, 1}, {2, 3}, {0, 2}, {1, 3}};
  for (int k = 0; k < pattern_count; k++) {
    float B[4][4] = {{1, 0, 0, 0}, {0, 1, 0, 0}, {0, 0, 1, 0}, {0, 0, 0, 1}};
    for (int g = 0; g < 4; g++) {
      uint32_t h = hash_u32(seed, (uint32_t)k + 7u, (uint32_t)g + 131u);
      float mag = 0.6f + 0.4f * (float)(h >> 9) * (1.0f / 8388608.0f);
      float ang = spread * 0.78539816f * mag * ((h & 1u) ? 1.0f : -1.0f);
      float c = std::cos(ang), sn = std::sin(ang);
      int a = kPairs[g][0], b = kPairs[g][1];
      // B ← R · B, R the rotation in the (a, b) plane.
      for (int col = 0; col < 4; col++) {
        float xa = B[a][col], xb = B[b][col];
        B[a][col] = c * xa - sn * xb;
        B[b][col] = sn * xa + c * xb;
      }
    }
    float* M = mats + k * 48;   // M[i*4+j], row-major; P is one-hot
    float P[16];
    for (int i = 0; i < 16; i++) P[i] = M[i];
    for (int i = 0; i < 4; i++)
      for (int j = 0; j < 4; j++) {
        float acc = 0.0f;
        for (int m = 0; m < 4; m++) acc += P[i * 4 + m] * B[m][j];
        M[i * 4 + j] = acc;
      }
  }
}

void module_init() {
  state::init("filter.light.bounce_resonator", {1, 0, 0},
    state::Schema()
      .helpField("intro",
        "## Bounce Resonator FX\n"
        "A reverb for light. The picture is cut into a grid of cells — four "
        "bars by one cell per segment — and every cell is a resonator that "
        "the picture keeps plucking: bright cells fizz with random kicks, and "
        "anything that suddenly brightens rings out. The energy bounces "
        "between the bars through the same seeded exchange pattern on every "
        "row, so the picture's shape is carried along in the tail.\n\n"
        "The output is **A/C**: each cell shows how far it swings from its "
        "own recent average, so a steady glow cancels out and a busy "
        "resonator swings both ways — bright above the average, and choking "
        "the picture to **black** below it.\n\n"
        "**Try:** push **Cycle Rate** all the way up so the energy flips "
        "every frame for a solid strobe; raise **Grain Rate** over bright "
        "footage for sparkle; use **Edge Excite** so cuts and flashes ring.")
      // --- Excitation ---
      .group("excite", "Excite")
        .groupHelp(
          "How the picture plucks the resonator — continuously, no triggers. "
          "**Grain Rate** is the rate of random kicks per cell at full "
          "brightness (dimmer cells kick proportionally less often); **Grain "
          "Size** is how hard each kick is. **Edge Excite** kicks a cell by "
          "how much it just brightened, so cuts and flashes ring. **Input "
          "Color** takes the kick's colour from the picture (1) or from "
          "**Bar Color** (0).")
      .floatField("grain_rate",      8.0f, 0.0f, 60.0f,  state::PrimaryInput).label("Grain Rate", "Grain")
      .floatField("grain_size",      0.6f, 0.0f, 2.0f,   state::PrimaryInput).label("Grain Size", "GSize")
      .floatField("edge_excite",     2.0f, 0.0f, 8.0f,   state::PrimaryInput).label("Edge Excite", "Edge")
      .floatField("input_color",     1.0f, 0.0f, 1.0f,   state::PrimaryInput).label("Input Color", "InCol")
      // --- Diffusion network ---
      .group("network", "Diffusion Network")
        .groupHelp(
          "The bouncing engine, shared by every row. **Tail** is how long "
          "energy rings, in real time whatever the hop rate — the knob is "
          "exponential, and the very top holds forever. **Spread** is how "
          "widely each hop fans out: 0 shuffles each bar's energy whole onto "
          "another bar, 1 scrambles every bar into every other with mixed "
          "signs. The mix keeps its energy and signs rather than averaging, so "
          "even a long, wide tail stays punchy. **Cycle Rate** is "
          "hops per second, capped at one hop per frame — at or past your "
          "frame rate it hops every frame, and with a low Spread the energy "
          "flips bar to bar each frame. **Pattern Count** and **Seed** vary "
          "the routing; the **Hue** knobs let colour wander or settle back on "
          "the bar colour.")
      .floatField("tail",            0.55f, 0.0f, 1.0f,  state::PrimaryInput,
                  nullptr, 0.f, nullptr,
                  "How long energy rings: the time to fade to 1%, exponential "
                  "from 0.05 s (0) through ~0.35 s (0.25), ~2.5 s (0.55) and "
                  "~17 s (0.83) to 60 s; the very top holds forever.")
        .label("Tail", "Tail")
      .floatField("spread",          0.30f, 0.0f, 1.0f,  state::PrimaryInput).label("Spread", "Sprd")
      .floatField("decay_shaping",   0.0f, -1.0f, 1.0f,  state::PrimaryInput).label("Decay Shaping", "Decay")
      .floatField("hue_spread",      0.0f, 0.0f, 1.0f,   state::PrimaryInput).label("Hue Spread", "HueSp")
      .floatField("hue_converge",    0.0f, 0.0f, 1.0f,   state::PrimaryInput).label("Hue Converge", "HueCv")
      .intField  ("seed",            0, 0, 0x7FFFFFFF,   state::PrimaryInput).label("Seed", "Seed")
      .intField  ("pattern_count",   4, 1, 16,           state::PrimaryInput).label("Pattern Count", "Ptn")
      .floatField("cycle_rate",      12.0f, 0.0f, 120.0f, state::PrimaryInput).label("Cycle Rate", "Cycle")
      // --- A/C ---
      .group("ac", "A/C")
        .groupHelp(
          "Each cell subtracts its own running average, so only the swing "
          "shows. **A/C Amount** blends from plain additive light (0) to full "
          "swing (1); **A/C Time** is how long the average takes to catch up "
          "— short keeps only fast flicker, long lets a kick glow before it "
          "cancels. **Black Depth** is how hard a swing below the average "
          "chokes the picture toward black. **Contrast** rides the gain: it "
          "divides the swing by the grid's recent level, so a fading tail keeps "
          "swinging at full scale instead of dimming out — one gain for the "
          "whole grid, so the picture's shape survives. It lets go once the "
          "level falls under **Contrast Floor**; lower floors stretch deeper "
          "into the tail (and into the noise). A/C Time also sets how quickly "
          "Contrast re-adapts after a loud moment.")
      .floatField("ac_amount",       1.0f, 0.0f, 1.0f,   state::PrimaryInput).label("A/C Amount", "AC")
      .floatField("ac_time",         0.5f, 0.02f, 4.0f,  state::PrimaryInput).label("A/C Time", "ACTime")
      .floatField("black_depth",     1.5f, 0.0f, 4.0f,   state::PrimaryInput).label("Black Depth", "Black")
      .floatField("contrast",        0.0f, 0.0f, 1.0f,   state::PrimaryInput).label("Contrast", "Contr")
      .floatField("contrast_floor",  0.05f, 0.001f, 1.0f, state::PrimaryInput).label("Contrast Floor", "Floor")
      // --- Look ---
      .group("look", "Look")
        .groupHelp(
          "**Bar Color** tints the kicks (see *Input Color*); **Intensity** "
          "scales the swing both ways. **Chroma Hold** keeps hot cells "
          "saturated instead of blooming white. **Input Opacity** is the dry "
          "picture underneath — 0 leaves only the resonance on black.")
      .rgbField  ("band_color",      1.0f, 0.92f, 0.78f, state::PrimaryInput).label("Bar Color", "Color")
      .floatField("intensity",       1.0f, 0.0f, 10.0f,  state::PrimaryInput).label("Intensity", "Int")
      .floatField("chroma_hold",     0.0f, 0.0f, 1.0f,   state::PrimaryInput).label("Chroma Hold", "Chroma")
      .floatField("input_opacity",   1.0f, 0.0f, 1.0f,   state::PrimaryInput).label("Input Opacity", "Opac")
      // --- Grid ---
      .group("grid", "Grid")
        .groupHelp(
          "Cells per bar — one per LED segment. The bars are always four "
          "(the network's size).")
      .intField  ("rows",            10, 1, MAX_ROWS,    state::PrimaryInput).label("Rows", "Rows")
      .textureField("tex_in",  state::PrimaryInput)
      .textureField("tex_out", state::PrimaryOutput)
  );

  if (gpu::Device::backend() == gpu::Backend::None) return;

  state::registerShaderSPV("bounce_resonator_fx_reduce", REDUCE_SPV, REDUCE_SPV_SIZE);
  state::registerShaderSPV("bounce_resonator_fx_sim",    SIM_SPV,    SIM_SPV_SIZE);
  state::registerShaderSPV("bounce_resonator_fx_render", RENDER_SPV, RENDER_SPV_SIZE);
  auto cs_reduce = gpu::Device::createShaderModuleByName("bounce_resonator_fx_reduce");
  auto cs_sim    = gpu::Device::createShaderModuleByName("bounce_resonator_fx_sim");
  auto cs_render = gpu::Device::createShaderModuleByName("bounce_resonator_fx_render");
  if (!cs_reduce || !cs_sim || !cs_render) return;

  s_pso_reduce = gpu::Device::createComputePSO(cs_reduce, "main", gpu::Bindings()
      .tex2d(0)
      .storageRW(1)
      .uniform(2));
  s_pso_sim = gpu::Device::createComputePSO(cs_sim, "main", gpu::Bindings()
      .storage(0)
      .storageRW(1)
      .storage(2)
      .storageRW(3)
      .uniform(4));
  s_pso_render = gpu::Device::createComputePSO(cs_render, "main", gpu::Bindings()
      .tex2d(0)
      .storageTex2d(1)
      .uniform(2)
      .storage(3));

  state::log("bounce_resonator_fx: module initialized");
}

void* create() {
  auto* s = new State();
  s->stats_buf          = gpu::Device::createBuffer(sizeof(CellStat) * MAX_CELLS,  gpu::BufferUsage::Storage);
  s->cells_buf          = gpu::Device::createBuffer(sizeof(CellState) * MAX_CELLS, gpu::BufferUsage::Storage);
  s->net_buf            = gpu::Device::createBuffer(sizeof(NetState),              gpu::BufferUsage::Storage);
  s->mats_buf           = gpu::Device::createBuffer(sizeof(MatsBuf),               gpu::BufferUsage::Storage);
  s->reduce_uniform_buf = gpu::Device::createBuffer(sizeof(ReduceUniforms),        gpu::BufferUsage::Uniform);
  s->sim_uniform_buf    = gpu::Device::createBuffer(sizeof(SimUniforms),           gpu::BufferUsage::Uniform);
  s->render_uniform_buf = gpu::Device::createBuffer(sizeof(RenderUniforms),        gpu::BufferUsage::Uniform);
  return s;
}

void destroy(void* self) {
  auto* s = static_cast<State*>(self);
  if (!s) return;
  s->stats_buf.release();
  s->cells_buf.release();
  s->net_buf.release();
  s->mats_buf.release();
  s->reduce_uniform_buf.release();
  s->sim_uniform_buf.release();
  s->render_uniform_buf.release();
  delete s;
}

void init(void* self) {
  auto* s = static_cast<State*>(self);
  if (!s) return;
  s->initialized = false;
  s->pending_dt = 0.0;
  s->accum = 0.0f;
  s->hop_idx = 0;
  s->frame = 0;
  s->needs_reset = true;
  rgb_to_hsv(s->color_r, s->color_g, s->color_b, s->band_hue, s->band_sat, s->band_val);
  s->net.reset();

  if (!s_pso_reduce.valid() || !s_pso_sim.valid() || !s_pso_render.valid()) return;
  if (!s->stats_buf.valid() || !s->cells_buf.valid() || !s->net_buf.valid()
      || !s->mats_buf.valid() || !s->reduce_uniform_buf.valid()
      || !s->sim_uniform_buf.valid() || !s->render_uniform_buf.valid()) return;

  s->initialized = true;
  state::log("bounce_resonator_fx: initialized");
}

void tick(void* self, double dt) {
  auto* s = static_cast<State*>(self);
  if (!s || !s->initialized) return;
  // Accumulated here, consumed in render() — the runner may tick several
  // times per render, and the hop cap is per RENDERED frame. Clamp so a
  // stall can't dump a burst of grains at once.
  if (dt > 0.0) s->pending_dt += dt;
  if (s->pending_dt > 0.25) s->pending_dt = 0.25;
}

void on_state_patched(void* self, int n, const char* pb, const int* off,
                      const int* len, const int* ops) {
  auto* s = static_cast<State*>(self);
  if (!s) return;
  for (int i = 0; i < n; i++) {
    if (ops[i] != state::PatchReplace) continue;
    const char* path = pb + off[i];
    int plen = len[i];
    if      (state::pathIs(path, plen, "grain_rate"))      s->grain_rate      = state::patchFloat(i);
    else if (state::pathIs(path, plen, "grain_size"))      s->grain_size      = state::patchFloat(i);
    else if (state::pathIs(path, plen, "edge_excite"))     s->edge_excite     = state::patchFloat(i);
    else if (state::pathIs(path, plen, "input_color"))     s->input_color     = state::patchFloat(i);
    else if (state::pathIs(path, plen, "tail"))            s->tail            = state::patchFloat(i);
    else if (state::pathIs(path, plen, "spread"))          s->spread          = state::patchFloat(i);
    else if (state::pathIs(path, plen, "contrast"))        s->contrast        = state::patchFloat(i);
    else if (state::pathIs(path, plen, "contrast_floor"))  s->contrast_floor  = state::patchFloat(i);
    else if (state::pathIs(path, plen, "decay_shaping"))   s->decay_shaping   = state::patchFloat(i);
    else if (state::pathIs(path, plen, "hue_spread"))      s->hue_spread      = state::patchFloat(i);
    else if (state::pathIs(path, plen, "hue_converge"))    s->hue_converge    = state::patchFloat(i);
    else if (state::pathIs(path, plen, "seed"))            s->seed            = (int)state::patchFloat(i);
    else if (state::pathIs(path, plen, "pattern_count"))   s->pattern_count   = (int)state::patchFloat(i);
    else if (state::pathIs(path, plen, "cycle_rate"))      s->cycle_rate      = state::patchFloat(i);
    else if (state::pathIs(path, plen, "ac_amount"))       s->ac_amount       = state::patchFloat(i);
    else if (state::pathIs(path, plen, "ac_time"))         s->ac_time         = state::patchFloat(i);
    else if (state::pathIs(path, plen, "black_depth"))     s->black_depth     = state::patchFloat(i);
    else if (state::pathIs(path, plen, "band_color")) {
      auto v = state::patchVec3(i);
      s->color_r = v.x; s->color_g = v.y; s->color_b = v.z;
      rgb_to_hsv(s->color_r, s->color_g, s->color_b, s->band_hue, s->band_sat, s->band_val);
    }
    else if (state::pathIs(path, plen, "intensity"))       s->intensity       = state::patchFloat(i);
    else if (state::pathIs(path, plen, "chroma_hold"))     s->chroma_hold     = state::patchFloat(i);
    else if (state::pathIs(path, plen, "input_opacity"))   s->input_opacity   = state::patchFloat(i);
    else if (state::pathIs(path, plen, "rows"))            s->rows            = (int)state::patchFloat(i);
  }
}

void render(void* self, int vp_w, int vp_h) {
  auto* s = static_cast<State*>(self);
  if (!s || !s->initialized || vp_w <= 0 || vp_h <= 0) return;
  auto in  = gpu::Device::textureForField("tex_in");
  auto out = gpu::Device::textureForField("tex_out");
  if (!in.valid() || !out.valid()) return;

  int rows = clampi(s->rows, 1, MAX_ROWS);
  int nc   = clampi(s->pattern_count, 1, fx::DiffusionNetwork4::kMaxN);
  float dt = (float)s->pending_dt;

  // Build + upload the cycling matrices (rebuilt only when routing changes).
  fx::DiffusionNetwork4::Params p;
  p.feedback        = 1.0f;   // unused: the decay is per second, in the sim
  // Spread 0 makes the network export its bare seeded derangement (plus the
  // per-edge hue shifts); the spread itself is applied as rotations below.
  p.spread          = 0.0f;
  p.spread_contrast = 0.0f;
  p.decay_shaping   = clampf(s->decay_shaping, -1.0f, 1.0f);
  p.hue_spread      = clampf(s->hue_spread, 0.0f, 1.0f);
  p.hue_converge    = clampf(s->hue_converge, 0.0f, 1.0f);
  p.home_hue        = s->band_hue;
  p.seed            = (uint32_t)s->seed;
  p.pattern_count   = nc;
  p.rate            = clampf(s->cycle_rate, 0.0f, 120.0f);
  s->net.setParams(p);
  MatsBuf mats = {};
  s->net.exportMatrices(mats.data);
  orthogonalize(mats.data, nc, clampf(s->spread, 0.0f, 1.0f), (uint32_t)s->seed);
  s->mats_buf.writeOne(mats);

  // At most ONE hop per rendered frame: a rate at or past the frame rate
  // hops every frame, and nothing banks up behind the cap.
  bool do_hop = false;
  if (dt > 0.0f && p.rate > 0.0f) {
    float period = 1.0f / p.rate;
    s->accum += dt;
    if (s->accum >= period) {
      do_hop = true;
      s->accum -= period;
      if (s->accum >= period) s->accum = 0.0f;
    }
  }

  ReduceUniforms ru = { rows, vp_w, vp_h, 0 };
  s->reduce_uniform_buf.writeOne(ru);

  SimUniforms su = {};
  su.decay_k       = decay_rate(s->tail);
  su.decay_shaping = p.decay_shaping;
  su.hue_converge  = p.hue_converge;
  su.home_hue      = s->band_hue;
  su.pattern_count = nc;
  su.hop_idx       = s->hop_idx;
  su.do_hop        = do_hop ? 1 : 0;
  su.rows          = rows;
  su.grain_rate    = clampf(s->grain_rate, 0.0f, 60.0f);
  su.grain_size    = clampf(s->grain_size, 0.0f, 2.0f);
  su.edge_excite   = clampf(s->edge_excite, 0.0f, 8.0f);
  su.input_color   = clampf(s->input_color, 0.0f, 1.0f);
  su.band_hue      = s->band_hue;
  su.band_sat      = s->band_sat;
  su.ac_amount     = clampf(s->ac_amount, 0.0f, 1.0f);
  su.ac_time       = clampf(s->ac_time, 0.02f, 4.0f);
  su.dt            = dt;
  su.frame         = s->frame;
  su.seed          = (uint32_t)s->seed;
  su.do_reset      = s->needs_reset ? 1 : 0;
  su.contrast       = clampf(s->contrast, 0.0f, 1.0f);
  su.contrast_floor = clampf(s->contrast_floor, 0.001f, 1.0f);
  s->sim_uniform_buf.writeOne(su);

  RenderUniforms cu = {};
  cu.band_val      = s->band_val;
  cu.intensity     = clampf(s->intensity, 0.0f, 10.0f);
  cu.input_opacity = clampf(s->input_opacity, 0.0f, 1.0f);
  cu.chroma_hold   = clampf(s->chroma_hold, 0.0f, 1.0f);
  cu.black_depth   = clampf(s->black_depth, 0.0f, 4.0f);
  cu.rows          = rows;
  s->render_uniform_buf.writeOne(cu);

  // Pass 1 — reduce: one thread per cell → per-cell colour + luma.
  {
    auto cp = gpu::ComputePass::begin();
    cp.setPSO(s_pso_reduce);
    cp.setTexture(in, 0, 0);
    cp.setBuffer(s->stats_buf,          1);
    cp.setBuffer(s->reduce_uniform_buf, 2);
    cp.dispatch((BARS + 7) / 8, (rows + 7) / 8);
    cp.end();
  }

  // Pass 2 — sim (single-thread): hop, excite, A/C.
  {
    auto cp = gpu::ComputePass::begin();
    cp.setPSO(s_pso_sim);
    cp.setBuffer(s->stats_buf,       0);
    cp.setBuffer(s->cells_buf,       1);
    cp.setBuffer(s->mats_buf,        2);
    cp.setBuffer(s->net_buf,         3);
    cp.setBuffer(s->sim_uniform_buf, 4);
    cp.dispatch(1, 1, 1);
    cp.end();
  }

  // Pass 3 — render: composite the swing over the input.
  {
    auto cp = gpu::ComputePass::begin();
    cp.setPSO(s_pso_render);
    cp.setTexture(in,  0, 0);
    cp.setTexture(out, 1, 1);
    cp.setBuffer(s->render_uniform_buf, 2);
    cp.setBuffer(s->cells_buf,          3);
    cp.dispatch((vp_w + 7) / 8, (vp_h + 7) / 8);
    cp.end();
  }

  gpu::Device::submit();

  // Consume per-frame inputs so a re-render without a tick steps nothing
  // (the sim's dt == 0 path).
  if (do_hop) s->hop_idx = (s->hop_idx + 1) % nc;
  if (dt > 0.0f) s->frame++;
  s->pending_dt = 0.0;
  s->needs_reset = false;
}

} // namespace bounce_resonator_fx
