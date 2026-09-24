/*
 * source.pixel.ocean — pixel-art ocean generator.
 *
 * A rotatable coarse pixel grid painted a flat ocean blue, sprinkled with
 * sparse tiny wave sprites (dot-line / omega / unrolling wind-curl) that
 * animate and drift forward in discrete steps — Monster-Hunter-world-map
 * style. Fully procedural: the shader derives every wave from integer hashes
 * over a stratified spawn-cell lattice plus two global step clocks, so there
 * is no particle pool and the only cross-frame state is two accumulators.
 *
 * The two clocks (shape animation, forward drift) advance as accumulators
 * (style guide §2.1) and are passed to the shader as integer steps + fraction.
 * Per-wave jitter params stagger each clock per cell: 0 = the whole sea ticks
 * in lock step, 1 = fully staggered. See compute.hlsl for the lattice model.
 *
 * Per-instance ABI (§0): mutable state in State; the compute PSO + 1x1 black
 * input fallback are type-shared file statics built once in module_init.
 */

#include <gpu.h>
#include <host.h>
#include <effect_utils.h>
#include "pixel_ocean_shaders.h"

#include <cmath>
#include <cstdint>

namespace pixel_ocean {

enum Composite { CompOcean = 0, CompTransparent = 1, CompCustom = 2, CompInput = 3 };

static const int PO_CYCLE_LEN = 12;   // anim steps per life cycle (matches compute.hlsl)
static const int PO_MAX_SPARK = 24;   // hard cap on live sparkles (uniform array size)

// A sparkle is a small stateful twinkle that spawns near a live wave and plays
// its 5-frame bloom on its own clock. Its position is fixed in the (separate,
// viewport-aligned) sparkle grid at spawn — it does not track the wave after.
struct Sparkle {
  bool  active = false;
  int   type = 0;         // 0 bloom, 1 blink
  float age = 0.f;        // 0..1 across its life; frame = min(4, age*5)
  float rate = 1.f;       // life/sec for this sparkle (carries the timing jitter)
  int   gx = 0, gy = 0;   // sparkle-grid cell of the sprite centre
  int   cx = 0, cy = 0;   // the wave cell it claimed (so two sparkles don't share)
  uint32_t cdir = 0;
};

// Uniform layout — MUST match compute.hlsl's cbuffer byte-for-byte.
struct Uniforms {
  float aspect_x, aspect_y;   // u_aspect
  float cos_r, sin_r;         // u_cos, u_sin
  float ocean[4];             // u_ocean
  float wave[4];              // u_wave
  float bg[4];                // u_bg
  float cell_px;              // u_cell_px
  uint32_t spawn_size;        // u_spawn
  uint32_t composite;         // u_composite
  uint32_t seed;              // u_seed
  uint32_t cyc_index;         // u_cyc_index
  float cyc_frac;             // u_cyc_frac
  uint32_t drift_steps;       // u_drift_steps
  float drift_frac;           // u_drift_frac
  uint32_t forward_steps;     // u_forward_steps
  float forward_frac;         // u_forward_frac
  uint32_t debug_cells;       // u_debug
  float anim_jitter;          // u_anim_jitter
  float dens_cur, dens_prev;  // u_dens_cur / u_dens_prev
  float back_cur, back_prev;  // u_back_cur / u_back_prev
  float djit_cur, djit_prev;  // u_djit_cur / u_djit_prev
  float fjit_cur, fjit_prev;  // u_fjit_cur / u_fjit_prev
  float t0_cur, t0_prev;      // u_t0_cur / u_t0_prev  (type thresholds)
  float t1_cur, t1_prev;      // u_t1_cur / u_t1_prev
  // Sparkle layer.
  float spark_cell_px;        // u_spark_cell_px
  float spark_cos, spark_sin; // u_spark_cos / u_spark_sin
  uint32_t spark_pad0;
  float spark_color[4];       // u_spark_color
  int32_t sparkles[24][4];    // u_sparkles: (gridX, gridY, frame, active)
};
static_assert(sizeof(Uniforms) == 576, "cbuffer mirror drifted from compute.hlsl");

struct State {
  // Pixel grid.
  float pixel_size = 0.40f;
  float rotation = 0.06f;
  // Look.
  float ocean_color[3] = { 0.10f, 0.32f, 0.55f };
  float wave_color[3]  = { 0.0f, 0.0f, 0.0f };
  float density = 0.35f;
  int composite = CompOcean;
  float bg_color[4] = { 0.0f, 0.0f, 0.0f, 1.0f };
  // Motion.
  float anim_rate = 0.50f;
  float anim_jitter = 1.0f;
  float drift_rate = 0.40f;       // X-axis (sideways) drift
  float drift_jitter = 1.0f;
  float forward_rate = 0.30f;     // Y-axis (forward) drift
  float forward_jitter = 1.0f;
  float backwards = 0.10f;
  // Wave mix — relative spawn weights per shape type (normalised at spawn).
  float dot_weight = 0.45f;
  float omega_weight = 0.35f;
  float spiral_weight = 0.20f;
  // Sparkles (separate viewport-aligned grid).
  float sparkle_rate = 0.5f;      // spawn rate
  int sparkle_cap = 24;           // max live
  float sparkle_size = 0.40f;     // sparkle grid pixel size (own, unlinked)
  float sparkle_rotation = 0.0f;  // default 0 → aligned to the viewport
  float sparkle_speed = 0.5f;     // animation rate
  float sparkle_jitter = 0.5f;    // per-sparkle timing spread
  float sparkle_bloom = 0.5f;     // relative weight: bloom type
  float sparkle_blink = 0.5f;     // relative weight: blink type
  float sparkle_color[3] = { 1.0f, 1.0f, 1.0f };
  // Tuning.
  int spawn_size = 12;
  float seed = 0.0f;
  bool debug_cells = false;

  // Step-clock accumulators (§2.1 — never time*rate).
  double drift_acc = 0.0;     // X-axis (live)
  double forward_acc = 0.0;   // Y-axis (live)

  // Anim PHASE clock (capture-on-spawn). cyc_phase counts whole cycles; it
  // advances at the anim rate CAPTURED when the current cycle began, so a live
  // anim-rate change never re-speeds a wave already alive — it lands at the next
  // respawn. Existence/shape params are latched the same way: we keep the
  // snapshot for the current cycle and the previous one (every visible wave is
  // in one of those two), captured at each cycle boundary.
  double cyc_phase = 0.0;
  uint64_t cyc_index = 0;             // = floor(cyc_phase)
  double captured_anim_rate = 0.0;    // steps/sec latched at current cycle start
  bool clock_init = false;
  float dens_cur = 0.35f, dens_prev = 0.35f;
  float back_cur = 0.10f, back_prev = 0.10f;
  float djit_cur = 1.0f,  djit_prev = 1.0f;
  float fjit_cur = 1.0f,  fjit_prev = 1.0f;
  float t0_cur = 0.45f,   t0_prev = 0.45f;
  float t1_cur = 0.80f,   t1_prev = 0.80f;

  // Sparkle runtime state.
  Sparkle sparkles[PO_MAX_SPARK];
  double spark_spawn_acc = 0.0;
  uint32_t rng = 0;            // xorshift PRNG (deterministic given the tick stream)
  bool spark_init = false;
  int last_vp_w = 0, last_vp_h = 0;   // remembered from render() for tick's spawn math

  bool initialized = false;
  gpu::Buffer uniform_buf;
};

static gpu::ComputePSO s_pso;   // type-shared
static gpu::Texture s_black;    // type-shared 1x1 fallback for the "input"
                                // composite when nothing is wired upstream.

// Rate sliders → steps/sec. Exponential-with-zero-at-zero (§1.3 family):
// equal slider distance ≈ equal tempo change, and 0 freezes the clock.
// anim: 0 → 0, 0.5 → 2, 1 → 12 steps/s.  drift: 0.4 → ~0.53, 1 → 4 steps/s.
static inline double animStepsPerSec(float p) {
  if (p < 0.0f) p = 0.0f;
  if (p > 1.0f) p = 1.0f;
  return (std::pow(25.0, (double)p) - 1.0) * 0.5;
}
static inline double driftStepsPerSec(float p) {
  if (p < 0.0f) p = 0.0f;
  if (p > 1.0f) p = 1.0f;
  return (std::pow(17.0, (double)p) - 1.0) * 0.25;
}
// Forward (Y) drift shares the drift curve — same feel on both axes.
static inline double forwardStepsPerSec(float p) { return driftStepsPerSec(p); }

// The three shape weights → two cumulative thresholds partitioning [0,1):
// hash < t0 → dot, < t1 → omega, else spiral. Weights are clamped ≥ 0; if they
// all vanish, fall back to dot-only so the sea never silently empties.
static inline void typeThresholds(const State& s, float& t0, float& t1) {
  float w0 = s.dot_weight    > 0.f ? s.dot_weight    : 0.f;
  float w1 = s.omega_weight  > 0.f ? s.omega_weight  : 0.f;
  float w2 = s.spiral_weight > 0.f ? s.spiral_weight : 0.f;
  float sum = w0 + w1 + w2;
  if (sum <= 0.f) { t0 = 1.f; t1 = 1.f; return; }   // all-zero → dot only
  t0 = w0 / sum;
  t1 = (w0 + w1) / sum;
}

void module_init() {
  state::init("source.pixel.ocean", {1, 0, 0},
    state::Schema()
      .helpField("intro",
        "## Pixel Ocean\n"
        "A pixel-art sea, world-map style: a chunky rotatable pixel grid of "
        "flat blue, dotted with sparse little black waves — flecks, omega "
        "crests, and wind-curls that unroll — each animating and drifting in "
        "hard discrete steps.\n\n"
        "**Try:** drop *Anim Jitter* and *Drift Jitter* to 0 so the whole sea "
        "ticks in lock step like a game boot screen; raise *Density* for a "
        "busier crossing; nudge *Backwards* for cross-chop; set *Composite → "
        "Input* to scatter waves over the layer below.")
      // --- Pixel grid: resolution + orientation ---
      .group("grid", "Pixel Grid")
        .groupHelp(
          "*Pixel Size* sets how chunky the grid is (exponential: left = fine "
          "~256 columns, right = huge ~16). *Rotation* turns the whole ocean — "
          "grid, waves, and travel direction together (waves always swim along "
          "the grid's own axis, ±half a turn).")
      .floatField("pixel_size", 0.40f, 0.f, 1.f, state::PrimaryInput).label("Pixel Size", "Px")
      .floatField("rotation", 0.06f, -1.f, 1.f, state::PrimaryInput).label("Rotation", "Rot")
      // --- Look: colors + how many waves ---
      .group("look", "Ocean")
        .groupHelp(
          "*Ocean* and *Wave* are the two flat colours. *Density* is the chance "
          "each spawn slot hosts a wave. It's LATCHED at spawn: raising it never "
          "pops a wave in mid-animation and lowering it never culls one — the "
          "change simply decides which cells do or don't respawn as they reach "
          "their next cycle, so it rolls in cleanly over about one cycle. "
          "**Composite** picks the backdrop: Ocean / Transparent / Custom / the "
          "Input image, with waves drawn on top.")
      .rgbField("ocean_color", 0.10f, 0.32f, 0.55f, state::PrimaryInput).label("Ocean Colour", "Ocean")
      .rgbField("wave_color", 0.0f, 0.0f, 0.0f, state::PrimaryInput).label("Wave Colour", "Wave")
      .floatField("density", 0.35f, 0.f, 1.f, state::PrimaryInput).label("Density", "Dens")
      .selectField("composite", CompOcean, state::PrimaryInput,
                   {{"Ocean", CompOcean}, {"Transparent", CompTransparent},
                    {"Custom", CompCustom}, {"Input", CompInput}}).label("Composite", "Comp")
      .rgbaField("bg_color", 0.0f, 0.0f, 0.0f, 1.0f, state::SecondaryInput).label("Background", "BG")
      // --- Motion: the two step clocks + their stagger ---
      .group("motion", "Motion")
        .groupHelp(
          "Independent step clocks, each marching waves in whole-pixel steps "
          "along the grid's own axes: *Anim Rate* ticks each wave through its "
          "shape frames; *Drift* slides them sideways (the grid ±X axis); "
          "*Forward* carries them along the grid's Y axis. Give both a rate and "
          "waves swim diagonally. Waves ALWAYS spawn out of phase with each "
          "other — *Anim Jitter* only sets whether their step ticks land on the "
          "same quantized instants (0 = the whole sea ticks together, though "
          "each wave is still at its own point in its animation) or spread onto "
          "per-wave instants (1). *Drift/Forward Jitter* stagger each axis's "
          "sub-step. *Backwards* is the chance a wave runs against the current — "
          "it reverses BOTH drift and forward. **Anim Rate, the jitters and "
          "Backwards latch at spawn** (like Density): changing one never "
          "re-speeds or redirects a wave already alive — it takes hold at each "
          "wave's next respawn. Drift/Forward SPEED stays live, so all waves "
          "glide together and speed changes read smoothly rather than as a jolt.")
      .floatField("anim_rate", 0.50f, 0.f, 1.f, state::PrimaryInput).label("Anim Rate", "Anim")
      .floatField("anim_jitter", 1.0f, 0.f, 1.f, state::PrimaryInput).label("Anim Jitter", "AJit")
      .floatField("drift_rate", 0.40f, 0.f, 1.f, state::PrimaryInput).label("Drift Rate (X)", "Drift")
      .floatField("drift_jitter", 1.0f, 0.f, 1.f, state::PrimaryInput).label("Drift Jitter", "DJit")
      .floatField("forward_rate", 0.30f, 0.f, 1.f, state::PrimaryInput).label("Forward Rate (Y)", "Fwd")
      .floatField("forward_jitter", 1.0f, 0.f, 1.f, state::PrimaryInput).label("Forward Jitter", "FJit")
      .floatField("backwards", 0.10f, 0.f, 1.f, state::PrimaryInput).label("Backwards", "Back")
      // --- Wave mix: relative spawn weight per shape ---
      .group("mix", "Wave Mix")
        .groupHelp(
          "Relative odds each new wave is a *Dot* (fleck that blinks and splits), "
          "an *Omega* (two-hump crest), or a *Swirl* (unrolling wind-curl). The "
          "three are normalised, so only their ratio matters; set one to 0 to "
          "drop that shape. Like the rest of the wave settings these latch at "
          "spawn, so a change re-mixes future waves, not ones already swimming.")
      .floatField("dot_weight", 0.45f, 0.f, 1.f, state::PrimaryInput).label("Dot Weight", "Dot")
      .floatField("omega_weight", 0.35f, 0.f, 1.f, state::PrimaryInput).label("Omega Weight", "Omega")
      .floatField("spiral_weight", 0.20f, 0.f, 1.f, state::PrimaryInput).label("Swirl Weight", "Swirl")
      // --- Sparkles: a twinkle layer on its OWN viewport-aligned pixel grid ---
      .group("sparkle", "Sparkles")
        .groupHelp(
          "A separate twinkle layer. Each sparkle claims a live wave (one no "
          "other sparkle has), blooms its little star just up-left of it, then "
          "fades — on its own clock, staggered from the rest. *Rate* is how "
          "fast they appear (up to *Cap* live at once); *Speed* how fast they "
          "bloom, *Timing Jitter* how much their clocks scatter. Critically the "
          "sparkle grid is UNLINKED from the waves: *Sparkle Size* is its own "
          "pixel pitch and *Sparkle Angle* its own rotation (default 0 = square "
          "to the viewport, whatever the ocean's rotation).")
      .floatField("sparkle_rate", 0.50f, 0.f, 1.f, state::PrimaryInput).label("Sparkle Rate", "Rate")
      .intField("sparkle_cap", 24, 0, 24, state::PrimaryInput).label("Sparkle Cap", "Cap")
      .floatField("sparkle_speed", 0.50f, 0.f, 1.f, state::PrimaryInput).label("Sparkle Speed", "Spd")
      .floatField("sparkle_jitter", 0.50f, 0.f, 1.f, state::PrimaryInput).label("Timing Jitter", "SJit")
      .floatField("sparkle_bloom", 0.50f, 0.f, 1.f, state::PrimaryInput).label("Bloom Weight", "Bloom")
      .floatField("sparkle_blink", 0.50f, 0.f, 1.f, state::PrimaryInput).label("Blink Weight", "Blink")
      .rgbField("sparkle_color", 1.0f, 1.0f, 1.0f, state::PrimaryInput).label("Sparkle Colour", "SpkC")
      .floatField("sparkle_size", 0.40f, 0.f, 1.f, state::SecondaryInput).label("Sparkle Size", "SpkPx")
      .floatField("sparkle_rotation", 0.0f, -1.f, 1.f, state::SecondaryInput).label("Sparkle Angle", "SpkR")
      // --- Tuning + debug ---
      .group("tuning", "Tuning")
        .groupHelp(
          "*Spawn Cell* is the stratified lattice pitch in grid pixels — each "
          "cell hosts at most one wave, which is what keeps the spread so even. "
          "Smaller cells = a denser ceiling. *Seed* re-deals the whole sea. "
          "*Show Cells* overlays the lattice (borders + a tint on live cells).")
      .intField("spawn_size", 12, 8, 24, state::SecondaryInput).label("Spawn Cell", "Cell")
      .floatField("seed", 0.0f, 0.f, 1.f, state::SecondaryInput).label("Seed", "Seed")
      .boolField("debug_cells", false, state::SecondaryInput).label("Show Cells", "Cells")
      .textureField("tex_in", state::PrimaryInput)
      .textureField("tex_out", state::PrimaryOutput)
      .capability(state::Capability::Generator)
      .capability(state::Capability::SeekableApproximate)
  );

  if (gpu::Device::backend() == gpu::Backend::None) return;
  state::registerShaderSPV("pixel_ocean_compute", COMPUTE_SPV, COMPUTE_SPV_SIZE);
  auto cs = gpu::Device::createShaderModuleByName("pixel_ocean_compute");
  if (!cs) return;
  s_pso = gpu::Device::createComputePSO(cs, "main",
    gpu::Bindings().tex2d(0).storageTex2d(1).uniform(2));

  // 1x1 black bound at the input slot when this generator starts a chain
  // (no upstream tex_in) — "Input" composite then falls back to black.
  s_black = gpu::Device::createTexture(1, 1, gpu::TextureFormat::RGBA8);
  if (s_black.valid()) gpu::Device::clear(s_black, 0.f, 0.f, 0.f, 1.f);
}

// --- Sparkle helpers -------------------------------------------------------
// Integer hash (mirror of nano_hash.hlsl's nano_uhash) + po_hash01, so the CPU
// can ask "is this wave cell live?" exactly as the shader would.
static inline uint32_t nano_uhash(uint32_t x) {
  x ^= x >> 17; x *= 0xed5ad4bbu; x ^= x >> 11; x *= 0xac4c1b51u;
  x ^= x >> 15; x *= 0x31848babu; x ^= x >> 14; return x;
}
static inline float po_hash01(int cx, int cy, uint32_t cycle, uint32_t stream, uint32_t seed) {
  uint32_t h = nano_uhash(seed ^ (stream * 0x9E3779B9u));
  h = nano_uhash(h + (uint32_t)cx);
  h = nano_uhash(h + (uint32_t)cy);
  h = nano_uhash(h + cycle);
  return (float)h * (1.0f / 4294967296.0f);
}
enum { SS_GATE = 0, SS_TYPE = 1, SS_POSX = 2, SS_POSY = 3, SS_ANIM = 4 };  // *2+dir

static inline uint32_t xorshift(uint32_t& s) { s ^= s << 13; s ^= s >> 17; s ^= s << 5; return s; }
static inline float rnd01(uint32_t& s) { return (float)(xorshift(s) >> 8) * (1.0f / 16777216.0f); }
static inline int floorDiv(int a, int b) { int q = a / b; if ((a % b != 0) && ((a < 0) != (b < 0))) q--; return q; }
static inline float gridColsFor(float px_size) { return 256.0f * std::pow(16.0f / 256.0f, px_size); }
static inline int clampSpawn(int S) { return S < 8 ? 8 : (S > 24 ? 24 : S); }
static inline float clamp01(float v) { return v < 0.f ? 0.f : (v > 1.f ? 1.f : v); }

// Is the wave in spawn cell (cx,cy) of the `dir` lattice live now? If so, return
// its anchor in co-moving wave-grid pixels. Mirrors po_cell_covers' gate/type/
// active logic (ignoring the ±1 drift sub-step, which doesn't matter here).
static bool waveLive(const State& s, uint32_t seed, int cx, int cy, uint32_t dir,
                     float cycFrac, uint32_t cycIndex, float& ax, float& ay) {
  int S = clampSpawn(s.spawn_size);
  float phs = po_hash01(cx, cy, 0, SS_ANIM * 2u + dir, seed) * (float)PO_CYCLE_LEN;
  float phi = (std::floor(phs) + (phs - std::floor(phs)) * s.anim_jitter) / (float)PO_CYCLE_LEN;
  uint32_t cycle; float f;
  if (cycFrac >= phi) { cycle = cycIndex;      f = cycFrac - phi; }
  else                { cycle = cycIndex - 1u; f = cycFrac - phi + 1.0f; }
  uint32_t step = (uint32_t)(f * (float)PO_CYCLE_LEN);
  if (step >= (uint32_t)PO_CYCLE_LEN) step = PO_CYCLE_LEN - 1;
  bool cur = (cycle == cycIndex);
  float dens = cur ? s.dens_cur : s.dens_prev, back = cur ? s.back_cur : s.back_prev;
  float t0 = cur ? s.t0_cur : s.t0_prev, t1 = cur ? s.t1_cur : s.t1_prev;
  float p = (dir == 0u) ? dens * (1.0f - back) : dens * back;
  float th = po_hash01(cx, cy, cycle, SS_TYPE * 2u + dir, seed);
  uint32_t type = th < t0 ? 0u : (th < t1 ? 1u : 2u);
  uint32_t gateKey;
  if (type == 2u) { if (step >= 8u) return false; gateKey = cycle; }
  else            { gateKey = cycle * 3u + step / 4u; }
  if (po_hash01(cx, cy, gateKey, SS_GATE * 2u + dir, seed) >= p) return false;
  int jx = (int)(po_hash01(cx, cy, cycle, SS_POSX * 2u + dir, seed) * (float)S);
  int jy = (int)(po_hash01(cx, cy, cycle, SS_POSY * 2u + dir, seed) * (float)S);
  ax = (float)(cx * S + jx);
  ay = (float)(cy * S + jy);
  return true;
}

// Try once to spawn a sparkle: sample random on-screen points until one lands on
// a live wave no other sparkle has claimed, then place the sparkle just up-left
// of that wave in the (separate, viewport-aligned) sparkle grid.
static void trySpawnSparkle(State& s) {
  if (s.last_vp_w <= 0 || s.last_vp_h <= 0) return;
  auto [aspx, aspy] = fx::coverSquare(s.last_vp_w, s.last_vp_h);
  const float PI = 3.14159265f;
  float wcell = 2.0f / gridColsFor(s.pixel_size);
  float wcos = std::cos(s.rotation * PI), wsin = std::sin(s.rotation * PI);
  float scell = 2.0f / gridColsFor(s.sparkle_size);
  float scos = std::cos(s.sparkle_rotation * PI), ssin = std::sin(s.sparkle_rotation * PI);
  int S = clampSpawn(s.spawn_size);
  uint32_t seed = (uint32_t)(s.seed * 65535.0f);
  float cycFrac = (float)(s.cyc_phase - std::floor(s.cyc_phase));
  uint32_t cycIndex = (uint32_t)s.cyc_index;
  int driftSteps = (int)(uint32_t)(uint64_t)s.drift_acc;
  int fwdSteps   = (int)(uint32_t)(uint64_t)s.forward_acc;

  for (int attempt = 0; attempt < 48; attempt++) {
    uint32_t dir = (rnd01(s.rng) < s.backwards) ? 1u : 0u;
    int d = (dir == 0u) ? 1 : -1;
    // Random viewport point → cover square → wave grid → co-moving cell.
    float sqx = (rnd01(s.rng) - 0.5f) / aspx, sqy = (rnd01(s.rng) - 0.5f) / aspy;
    float gwx = ( wcos * sqx + wsin * sqy) / wcell;
    float gwy = (-wsin * sqx + wcos * sqy) / wcell;
    int px = (int)std::floor(gwx) - d * driftSteps;
    int py = (int)std::floor(gwy) + d * fwdSteps;
    int cx = floorDiv(px, S), cy = floorDiv(py, S);

    float ax, ay;
    if (!waveLive(s, seed, cx, cy, dir, cycFrac, cycIndex, ax, ay)) continue;
    bool taken = false;
    for (int i = 0; i < PO_MAX_SPARK; i++)
      if (s.sparkles[i].active && s.sparkles[i].cx == cx && s.sparkles[i].cy == cy
          && s.sparkles[i].cdir == dir) { taken = true; break; }
    if (taken) continue;

    // Wave anchor → its screen grid pos → cover square → sparkle grid cell.
    float gsx = ax + (float)(d * driftSteps);
    float gsy = ay - (float)(d * fwdSteps);
    float wsqx = wcell * (wcos * gsx - wsin * gsy);
    float wsqy = wcell * (wsin * gsx + wcos * gsy);
    float sgx = ( scos * wsqx + ssin * wsqy) / scell;
    float sgy = (-ssin * wsqx + scos * wsqy) / scell;
    int scx = (int)std::floor(sgx - (rnd01(s.rng) * 3.0f + 0.5f));   // biased up-left
    int scy = (int)std::floor(sgy - (rnd01(s.rng) * 3.0f + 0.5f));

    for (int i = 0; i < PO_MAX_SPARK; i++) {
      if (s.sparkles[i].active) continue;
      Sparkle& sp = s.sparkles[i];
      float j = 1.0f + (rnd01(s.rng) - 0.5f) * 1.2f * s.sparkle_jitter;
      sp.active = true; sp.age = 0.f;
      sp.rate = (0.4f + s.sparkle_speed * 3.0f) * (j < 0.2f ? 0.2f : j);
      float wsum = s.sparkle_bloom + s.sparkle_blink;
      float thr = wsum > 0.f ? s.sparkle_bloom / wsum : 0.5f;
      sp.type = (rnd01(s.rng) < thr) ? 0 : 1;   // weighted bloom / blink
      sp.gx = scx; sp.gy = scy; sp.cx = cx; sp.cy = cy; sp.cdir = dir;
      return;
    }
    return;   // no free slot
  }
}

void* create() {
  auto* s = new State();
  s->uniform_buf = gpu::Device::createBuffer(sizeof(Uniforms), gpu::BufferUsage::Uniform);
  return s;
}

void destroy(void* self) {
  auto* s = static_cast<State*>(self);
  if (!s) return;
  s->uniform_buf.release();
  delete s;
}

void init(void* self) {
  auto* s = static_cast<State*>(self);
  if (!s) return;
  gpu::Buffer buf = s->uniform_buf;   // preserve the allocated buffer across reset
  *s = State();
  s->uniform_buf = buf;
  s->initialized = buf.valid();
}

void tick(void* self, double dt) {
  auto* s = static_cast<State*>(self);
  if (!s) return;

  // Cycle 0 begins on the first tick: latch the current params + rate as its
  // captured snapshot so the opening cycle already reflects the live settings.
  if (!s->clock_init) {
    s->dens_cur = s->dens_prev = s->density;
    s->back_cur = s->back_prev = s->backwards;
    s->djit_cur = s->djit_prev = s->drift_jitter;
    s->fjit_cur = s->fjit_prev = s->forward_jitter;
    typeThresholds(*s, s->t0_cur, s->t1_cur);
    s->t0_prev = s->t0_cur; s->t1_prev = s->t1_cur;
    s->captured_anim_rate = animStepsPerSec(s->anim_rate);
    s->clock_init = true;
  }

  // Advance the phase clock at the rate captured for the current cycle. If that
  // cycle was captured frozen (rate 0), follow the live rate instead so raising
  // the slider from 0 unfreezes rather than latching us out forever.
  double rate = s->captured_anim_rate > 0.0 ? s->captured_anim_rate
                                            : animStepsPerSec(s->anim_rate);
  s->cyc_phase += dt * rate / (double)PO_CYCLE_LEN;

  // Each cycle boundary: shift current snapshot → previous and re-capture the
  // live params + rate as the new cycle's latched values. This is the only place
  // param changes take hold, which is what makes them "capture on spawn".
  while (std::floor(s->cyc_phase) > (double)s->cyc_index) {
    s->cyc_index++;
    s->dens_prev = s->dens_cur; s->dens_cur = s->density;
    s->back_prev = s->back_cur; s->back_cur = s->backwards;
    s->djit_prev = s->djit_cur; s->djit_cur = s->drift_jitter;
    s->fjit_prev = s->fjit_cur; s->fjit_cur = s->forward_jitter;
    s->t0_prev = s->t0_cur; s->t1_prev = s->t1_cur;
    typeThresholds(*s, s->t0_cur, s->t1_cur);
    s->captured_anim_rate = animStepsPerSec(s->anim_rate);
  }

  // Drift / forward speed stay LIVE (the rigid co-moving lattice needs every
  // wave in a lattice to share one translation).
  s->drift_acc   += dt * driftStepsPerSec(s->drift_rate);
  s->forward_acc += dt * forwardStepsPerSec(s->forward_rate);

  // Sparkles: advance each one's own clock, retire finished ones, and spawn new
  // ones (on live waves) up to the cap at the spawn rate.
  if (!s->spark_init) {
    s->rng = nano_uhash(0x53504b21u ^ (uint32_t)(s->seed * 65535.0f)) | 1u;
    s->spark_init = true;
  }
  int live = 0;
  for (int i = 0; i < PO_MAX_SPARK; i++) {
    Sparkle& sp = s->sparkles[i];
    if (!sp.active) continue;
    sp.age += (float)dt * sp.rate;
    if (sp.age >= 1.0f) sp.active = false;
    else live++;
  }
  int cap = s->sparkle_cap < 0 ? 0 : (s->sparkle_cap > PO_MAX_SPARK ? PO_MAX_SPARK : s->sparkle_cap);
  s->spark_spawn_acc += dt * (std::pow(30.0, (double)clamp01(s->sparkle_rate)) - 1.0);
  int guard = 0;
  while (s->spark_spawn_acc >= 1.0 && live < cap && guard++ < PO_MAX_SPARK) {
    s->spark_spawn_acc -= 1.0;
    trySpawnSparkle(*s);
    live = 0;
    for (int i = 0; i < PO_MAX_SPARK; i++) if (s->sparkles[i].active) live++;
  }
}

void on_state_patched(void* self, int n, const char* pb, const int* off,
                      const int* len, const int* ops) {
  auto* s = static_cast<State*>(self);
  if (!s) return;
  for (int i = 0; i < n; i++) {
    if (ops[i] != state::PatchReplace) continue;
    const char* p = pb + off[i];
    const int l = len[i];
    if      (state::pathIs(p, l, "pixel_size"))   s->pixel_size = state::patchFloat(i);
    else if (state::pathIs(p, l, "rotation"))     s->rotation = state::patchFloat(i);
    else if (state::pathIs(p, l, "ocean_color")) {
      auto v = state::patchVec3(i);
      s->ocean_color[0] = v.x; s->ocean_color[1] = v.y; s->ocean_color[2] = v.z;
    }
    else if (state::pathIs(p, l, "wave_color")) {
      auto v = state::patchVec3(i);
      s->wave_color[0] = v.x; s->wave_color[1] = v.y; s->wave_color[2] = v.z;
    }
    else if (state::pathIs(p, l, "density"))      s->density = state::patchFloat(i);
    else if (state::pathIs(p, l, "composite"))    s->composite = state::patchInt(i);
    else if (state::pathIs(p, l, "bg_color")) {
      auto v = state::patchVec4(i);
      s->bg_color[0] = v.x; s->bg_color[1] = v.y; s->bg_color[2] = v.z; s->bg_color[3] = v.w;
    }
    else if (state::pathIs(p, l, "anim_rate"))    s->anim_rate = state::patchFloat(i);
    else if (state::pathIs(p, l, "anim_jitter"))  s->anim_jitter = state::patchFloat(i);
    else if (state::pathIs(p, l, "drift_rate"))   s->drift_rate = state::patchFloat(i);
    else if (state::pathIs(p, l, "drift_jitter")) s->drift_jitter = state::patchFloat(i);
    else if (state::pathIs(p, l, "forward_rate"))   s->forward_rate = state::patchFloat(i);
    else if (state::pathIs(p, l, "forward_jitter")) s->forward_jitter = state::patchFloat(i);
    else if (state::pathIs(p, l, "backwards"))    s->backwards = state::patchFloat(i);
    else if (state::pathIs(p, l, "dot_weight"))    s->dot_weight = state::patchFloat(i);
    else if (state::pathIs(p, l, "omega_weight"))  s->omega_weight = state::patchFloat(i);
    else if (state::pathIs(p, l, "spiral_weight")) s->spiral_weight = state::patchFloat(i);
    else if (state::pathIs(p, l, "sparkle_rate"))     s->sparkle_rate = state::patchFloat(i);
    else if (state::pathIs(p, l, "sparkle_cap"))      s->sparkle_cap = state::patchInt(i);
    else if (state::pathIs(p, l, "sparkle_speed"))    s->sparkle_speed = state::patchFloat(i);
    else if (state::pathIs(p, l, "sparkle_jitter"))   s->sparkle_jitter = state::patchFloat(i);
    else if (state::pathIs(p, l, "sparkle_bloom"))    s->sparkle_bloom = state::patchFloat(i);
    else if (state::pathIs(p, l, "sparkle_blink"))    s->sparkle_blink = state::patchFloat(i);
    else if (state::pathIs(p, l, "sparkle_size"))     s->sparkle_size = state::patchFloat(i);
    else if (state::pathIs(p, l, "sparkle_rotation")) s->sparkle_rotation = state::patchFloat(i);
    else if (state::pathIs(p, l, "sparkle_color")) {
      auto v = state::patchVec3(i);
      s->sparkle_color[0] = v.x; s->sparkle_color[1] = v.y; s->sparkle_color[2] = v.z;
    }
    else if (state::pathIs(p, l, "spawn_size"))   s->spawn_size = state::patchInt(i);
    else if (state::pathIs(p, l, "seed"))         s->seed = state::patchFloat(i);
    else if (state::pathIs(p, l, "debug_cells"))  s->debug_cells = state::patchBool(i);
  }
}

void on_resolume_param(void*, long long, double) {}

static void fillUniforms(State* s, int vp_w, int vp_h, Uniforms& u) {
  auto [ax, ay] = fx::coverSquare(vp_w, vp_h);
  u.aspect_x = ax; u.aspect_y = ay;
  const float angle = s->rotation * 3.14159265f;
  u.cos_r = std::cos(angle);
  u.sin_r = std::sin(angle);
  for (int i = 0; i < 3; i++) {
    u.ocean[i] = s->ocean_color[i];
    u.wave[i] = s->wave_color[i];
  }
  u.ocean[3] = 1.0f; u.wave[3] = 1.0f;
  for (int i = 0; i < 4; i++) u.bg[i] = s->bg_color[i];

  // pixel_size → grid columns across the cover square, exponential 256 → 16.
  float cols = 256.0f * std::pow(16.0f / 256.0f, s->pixel_size);
  u.cell_px = 2.0f / cols;

  int S = s->spawn_size;
  if (S < 8) S = 8;      // < 8 would break the shader's candidate-cell bounds
  if (S > 24) S = 24;
  u.spawn_size = (uint32_t)S;
  u.composite = (uint32_t)s->composite;
  u.seed = (uint32_t)(s->seed * 65535.0f);

  // Anim PHASE clock: integer cycle index + fractional position, so the shader's
  // per-cell cycle math stays exact. Drift/forward stay as live step + fraction.
  u.cyc_index = (uint32_t)s->cyc_index;
  u.cyc_frac  = (float)(s->cyc_phase - (double)s->cyc_index);
  u.drift_steps = (uint32_t)(uint64_t)s->drift_acc;
  u.drift_frac  = (float)(s->drift_acc - std::floor(s->drift_acc));
  u.forward_steps = (uint32_t)(uint64_t)s->forward_acc;
  u.forward_frac  = (float)(s->forward_acc - std::floor(s->forward_acc));

  u.debug_cells = s->debug_cells ? 1u : 0u;
  u.anim_jitter = s->anim_jitter;   // live: the per-cell phase spread
  // Captured per-cycle snapshots (current + previous).
  u.dens_cur = s->dens_cur; u.dens_prev = s->dens_prev;
  u.back_cur = s->back_cur; u.back_prev = s->back_prev;
  u.djit_cur = s->djit_cur; u.djit_prev = s->djit_prev;
  u.fjit_cur = s->fjit_cur; u.fjit_prev = s->fjit_prev;
  u.t0_cur = s->t0_cur; u.t0_prev = s->t0_prev;
  u.t1_cur = s->t1_cur; u.t1_prev = s->t1_prev;

  // Sparkle layer: its own grid + the live sparkle list (grid cell + frame).
  u.spark_cell_px = 2.0f / gridColsFor(s->sparkle_size);
  u.spark_cos = std::cos(s->sparkle_rotation * 3.14159265f);
  u.spark_sin = std::sin(s->sparkle_rotation * 3.14159265f);
  u.spark_pad0 = 0;
  u.spark_color[0] = s->sparkle_color[0];
  u.spark_color[1] = s->sparkle_color[1];
  u.spark_color[2] = s->sparkle_color[2];
  u.spark_color[3] = 1.0f;
  for (int i = 0; i < PO_MAX_SPARK; i++) {
    const Sparkle& sp = s->sparkles[i];
    if (sp.active) {
      int frame = (int)(sp.age * 5.0f);
      frame = frame < 0 ? 0 : (frame > 4 ? 4 : frame);
      u.sparkles[i][0] = sp.gx; u.sparkles[i][1] = sp.gy;
      u.sparkles[i][2] = frame; u.sparkles[i][3] = sp.type + 1;   // w: 0 off, else type+1
    } else {
      u.sparkles[i][0] = u.sparkles[i][1] = u.sparkles[i][2] = u.sparkles[i][3] = 0;
    }
  }
}

void render(void* self, int vp_w, int vp_h) {
  auto* s = static_cast<State*>(self);
  if (!s || !s->initialized || vp_w <= 0 || vp_h <= 0) return;
  s->last_vp_w = vp_w; s->last_vp_h = vp_h;   // remembered for tick's sparkle spawn

  auto in  = gpu::Device::textureForField("tex_in");
  auto out = gpu::Device::textureForField("tex_out");
  if (!out.valid()) return;
  if (!in.valid()) in = s_black;   // chain-start: no upstream input
  if (!in.valid()) return;

  Uniforms u = {};
  fillUniforms(s, vp_w, vp_h, u);
  s->uniform_buf.writeOne(u);

  auto cp = gpu::ComputePass::begin();
  cp.setPSO(s_pso);
  cp.setTexture(in,  0, 0);
  cp.setTexture(out, 1, 1);
  cp.setBuffer(s->uniform_buf, 2);
  cp.dispatch((vp_w + 7) / 8, (vp_h + 7) / 8);
  cp.end();
  gpu::Device::submit();
}

} // namespace pixel_ocean
