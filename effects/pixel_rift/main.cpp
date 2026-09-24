/*
 * source.pixel.rift — coarse-grid ocean waves crossing a hidden mid-rift.
 *
 * The look of pixel_descent (a hard cols × rows cell grid, default 4 × 10)
 * carrying the wave life of pixel_ocean: tiny dot / omega sprites that drift
 * left → right and slightly up in whole-cell steps, spawning and dying on
 * their own staggered animation clocks.
 *
 * The signature feature is the RIFT: `rift_cols` extra virtual columns are
 * spliced between the left and right halves of the visible grid. Waves move
 * across the full virtual width but rift columns never render — a wave slides
 * in from the left, vanishes into the middle, and re-emerges on the right.
 * The virtual X axis is a torus (virtual width + a 4-column off-grid margin)
 * so waves also slide fully off the right edge before re-entering left.
 *
 * Unlike pixel_ocean there is no procedural lattice: on a grid this coarse a
 * handful of waves is plenty, so a small CPU pool (the ocean's sparkle
 * pattern) owns all state. Motion and animation run on three live global step
 * accumulators (§2.1 — rigid translation, all waves glide together). Each
 * wave carries raw stagger draws that the LIVE jitter params scale at use
 * (ocean semantics): at jitter 0 every offset is whole-step, so the whole sea
 * ticks on the same global instants — lock-step — while each wave is still at
 * its own point in its animation; at 1 ticks spread onto per-wave instants.
 * Shape type and lifespan are CAPTURED AT SPAWN, and density only gates
 * births (never culls) — the ocean's latching feel.
 *
 * Class-like instance model: module_init() sets up the type-shared compute
 * PSO + schema once; each chain entry gets its own State via create().
 */

#include <gpu.h>
#include <host.h>
#include "pixel_rift_shaders.h"

#include <cmath>
#include <cstdint>

namespace pixel_rift {

enum Composite { CompBlack = 0, CompTransparent = 1, CompCustom = 2, CompInput = 3 };

static const int MAX_WAVES  = 12;
static const int MAX_COLS   = 16;
static const int TORUS_PAD  = 4;    // off-grid margin (> sprite width) — matches render.hlsl
static const double LOOP_LEN = 4.0; // anim steps per sprite loop (ping-pong 0,1,2,1)

// Uniform layout — MUST match render.hlsl's cbuffer byte-for-byte.
struct Uniforms {
  int32_t cols, rows, rift, composite;
  float   color[4];
  float   bg[4];
  float   intensity;
  int32_t n_waves;
  float   pad0, pad1;
  int32_t waves[MAX_WAVES][4];   // (torus col, row, type*4+frame, active)
};
static_assert(sizeof(Uniforms) == 256, "Uniforms layout mismatch");

// One pooled wave. Shape/lifespan and the raw stagger draws are latched at
// spawn; position and frame are derived per frame from the global clocks
// (rigid translation). The staggers are stored RAW in [0,1) and scaled by the
// LIVE jitter params at use, so jitter changes take hold immediately: at 0
// every offset is whole-step and the entire sea ticks on the same global
// instants (lock-step); at 1 each wave ticks on its own staggered instants.
struct Wave {
  bool   active = false;
  int    type = 0;                 // 0 dot, 1 omega
  int    spawn_col = 0;            // torus X at spawn
  int    spawn_row = 0;
  float  stag_x = 0.0f;            // raw sub-step stagger draws [0,1)
  float  stag_y = 0.0f;
  float  stag_anim = 0.0f;
  int    anim_whole = 0;           // whole-step anim offset — the PHASE is always
                                   // staggered, jitter only spreads the instants
  double drift_at_spawn = 0.0;
  double rise_at_spawn = 0.0;
  double anim_at_spawn = 0.0;
  int    life_steps = 16;          // lifespan in anim steps (loops × LOOP_LEN)
};

struct State {
  // Schema-mirrored params
  int   columns = 4;
  int   rows = 10;
  int   rift_cols = 4;
  float drift_rate = 0.4f;
  float rise = 0.3f;
  float drift_jitter = 1.0f;
  float anim_rate = 0.5f;
  float anim_jitter = 1.0f;
  float density = 0.4f;
  float dot_weight = 0.55f;
  float omega_weight = 0.45f;
  float seed = 0.0f;
  float color_r = 1.0f, color_g = 1.0f, color_b = 1.0f;
  float intensity = 1.0f;
  int   composite = CompBlack;
  float bg_color[4] = { 0.0f, 0.0f, 0.0f, 1.0f };

  // Live global step clocks (§2.1 — never time×rate).
  double drift_acc = 0.0;
  double rise_acc = 0.0;
  double anim_acc = 0.0;   // sprite-frame clock, shared by every wave

  // Wave pool + its deterministic PRNG (seeded on first tick).
  Wave wave[MAX_WAVES];
  double spawn_acc = 0.0;
  uint32_t rng = 0;
  bool rng_init = false;

  bool initialized = false;
  gpu::Buffer uniform_buf;
};

// Type-shared: compiled once in module_init(), reused by every instance.
static gpu::ComputePSO s_pso;
static gpu::Texture s_black;   // 1x1 fallback when this generator starts a chain

static inline float clampf(float v, float lo, float hi) { return v < lo ? lo : (v > hi ? hi : v); }
static inline int   clampi(int v, int lo, int hi)       { return v < lo ? lo : (v > hi ? hi : v); }
static inline int   posmod(long a, int m) { int r = (int)(a % (long)m); return r < 0 ? r + m : r; }

static inline uint32_t nano_uhash(uint32_t x) {
  x ^= x >> 17; x *= 0xed5ad4bbu; x ^= x >> 11; x *= 0xac4c1b51u;
  x ^= x >> 15; x *= 0x31848babu; x ^= x >> 14; return x;
}
static inline uint32_t xorshift(uint32_t& s) { s ^= s << 13; s ^= s >> 17; s ^= s << 5; return s; }
static inline float rnd01(uint32_t& s) { return (float)(xorshift(s) >> 8) * (1.0f / 16777216.0f); }

// Rate sliders → steps/sec, pixel_ocean's exponential-with-zero-at-zero
// curves. drift: 0.4 → ~0.53, 1 → 4 steps/s; anim: 0.5 → 2, 1 → 12 steps/s.
// Rise shares the drift curve scaled to a quarter — "slightly up".
static inline double animStepsPerSec(float p) {
  p = clampf(p, 0.0f, 1.0f);
  return (std::pow(25.0, (double)p) - 1.0) * 0.5;
}
static inline double driftStepsPerSec(float p) {
  p = clampf(p, 0.0f, 1.0f);
  return (std::pow(17.0, (double)p) - 1.0) * 0.25;
}
static inline double riseStepsPerSec(float p) { return driftStepsPerSec(p) * 0.25; }

// Whole steps a global clock has ticked for a wave since its spawn, with the
// wave's offset `off` deciding WHERE in the accumulator the tick instants sit.
// A whole-number off (jitter 0) puts every wave's ticks on the same global
// integer crossings — lock-step; a fractional off staggers them.
static long stepsSince(double acc, double at_spawn, double off) {
  return (long)std::floor(acc + off) - (long)std::floor(at_spawn + off);
}

// A wave's current torus position from the global clocks. Rising is −row
// (up); both axes wrap (the shader draws sprites torus-wrapped too). The raw
// stagger draws are scaled by the LIVE drift_jitter.
static void wavePos(const State& s, const Wave& w, int torus, int rows, int& x, int& y) {
  double jit = (double)clampf(s.drift_jitter, 0.0f, 1.0f);
  long sx = stepsSince(s.drift_acc, w.drift_at_spawn, (double)w.stag_x * jit);
  long sy = stepsSince(s.rise_acc,  w.rise_at_spawn,  (double)w.stag_y * jit);
  x = posmod((long)w.spawn_col + sx, torus);
  y = posmod((long)w.spawn_row - sy, rows);
}

// Anim steps this wave has lived (drives both its frame and its death). The
// whole-step part of the offset always staggers the phase; anim_jitter only
// scales the sub-step part — whether ticks land on shared or per-wave instants.
static long waveAnimSteps(const State& s, const Wave& w) {
  double off = (double)w.anim_whole
             + (double)w.stag_anim * (double)clampf(s.anim_jitter, 0.0f, 1.0f);
  return stepsSince(s.anim_acc, w.anim_at_spawn, off);
}

// Sprite frame from the wave's lived steps: 4-step loop, ping-pong 0,1,2,1.
static int waveFrame(const State& s, const Wave& w) {
  int m = (int)(waveAnimSteps(s, w) % (long)LOOP_LEN);
  m = clampi(m, 0, 3);
  return m == 3 ? 1 : m;
}

// Try once to spawn a wave into a free slot: hashed position on the torus,
// rejected if its 2×3 box (sprites are drawn turned on their sides — see
// render.hlsl) would overlap a live wave (keeps the sea readable with so few
// cells). Type / timing / lifespan latch here.
static void trySpawnWave(State& s) {
  int slot = -1;
  for (int i = 0; i < MAX_WAVES; i++) if (!s.wave[i].active) { slot = i; break; }
  if (slot < 0) return;

  const int cols  = clampi(s.columns, 1, MAX_COLS);
  const int rows  = clampi(s.rows, 2, 32);
  const int rift  = clampi(s.rift_cols, 0, MAX_COLS);
  const int torus = cols + rift + TORUS_PAD;

  for (int attempt = 0; attempt < 8; attempt++) {
    int x = clampi((int)(rnd01(s.rng) * (float)torus), 0, torus - 1);
    int y = clampi((int)(rnd01(s.rng) * (float)rows), 0, rows - 1);
    bool overlap = false;
    for (int i = 0; i < MAX_WAVES && !overlap; i++) {
      if (!s.wave[i].active) continue;
      int ox, oy;
      wavePos(s, s.wave[i], torus, rows, ox, oy);
      int dx = posmod((long)(x - ox), torus); if (dx > torus / 2) dx -= torus;
      int dy = posmod((long)(y - oy), rows);  if (dy > rows / 2)  dy -= rows;
      overlap = dx > -2 && dx < 2 && dy > -3 && dy < 3;
    }
    if (overlap) continue;

    Wave& w = s.wave[slot];
    w.active = true;
    // Weighted dot/omega (both zero → dot, so the sea never silently empties).
    float wd = s.dot_weight > 0.f ? s.dot_weight : 0.f;
    float wo = s.omega_weight > 0.f ? s.omega_weight : 0.f;
    float thr = (wd + wo) > 0.f ? wd / (wd + wo) : 1.f;
    w.type = rnd01(s.rng) < thr ? 0 : 1;
    w.spawn_col = x;
    w.spawn_row = y;
    w.stag_x = rnd01(s.rng);        // raw draws; scaled by the live jitters at use
    w.stag_y = rnd01(s.rng);
    w.stag_anim = rnd01(s.rng);
    w.anim_whole = (int)(xorshift(s.rng) % 4u);
    w.drift_at_spawn = s.drift_acc;
    w.rise_at_spawn = s.rise_acc;
    w.anim_at_spawn = s.anim_acc;   // steps count from 0 — born at frame 0
    w.life_steps = (3 + (int)(xorshift(s.rng) % 4u)) * (int)LOOP_LEN;   // 3..6 loops
    return;
  }
}

static void apply_composite_visibility(int composite) {
  state::setFieldHidden("bg_color", composite != CompCustom);
}

void eval_visibility(int n, const char* pb, const int* off, const int* len, const int* ops) {
  int composite = CompBlack;
  for (int i = 0; i < n; i++) {
    if (ops[i] != state::PatchReplace) continue;
    if (state::pathIs(pb + off[i], len[i], "composite")) composite = (int)state::patchFloat(i);
  }
  apply_composite_visibility(composite);
}

static void on_state_ready(void* self) {
  auto* s = static_cast<State*>(self);
  if (!s) return;
  apply_composite_visibility(s->composite);
}

void module_init() {
  state::init("source.pixel.rift", {1, 0, 0},
    state::Schema()
      .helpField("intro",
        "## Pixel Rift\n"
        "Ocean waves on a chunky cell grid: tiny dot and omega sprites drift "
        "left to right (and slightly up) in whole-cell steps, each on its own "
        "staggered clock. Down the middle runs a hidden RIFT — extra virtual "
        "columns that are never drawn — so a wave slides in from the left, "
        "vanishes into the gap, and re-emerges on the right.\n\n"
        "**Try:** widen *Rift Columns* to stretch the disappearance; *Rise* 0 "
        "for a flat crossing; drop *Anim Jitter* and *Drift Jitter* to 0 so "
        "the whole sea ticks in lock step; set *Composite → Input* to run the "
        "waves over the layer below.")
      // --- Grid ---
      .group("grid", "Grid")
        .groupHelp(
          "The viewport splits evenly into *Columns* × *Rows* cells. *Rift "
          "Columns* splices that many INVISIBLE columns between the left and "
          "right halves — waves cross them at full speed but nothing renders "
          "there. 0 closes the rift.")
      .intField("columns", 4, 1, MAX_COLS, state::PrimaryInput).label("Columns", "Cols")
      .intField("rows", 10, 2, 32, state::PrimaryInput).label("Rows", "Rows")
      .intField("rift_cols", 4, 0, MAX_COLS, state::PrimaryInput).label("Rift Columns", "Rift")
      // --- Motion ---
      .group("motion", "Motion")
        .groupHelp(
          "*Drift Rate* slides every wave rightward in whole-cell steps; "
          "*Rise* carries them gently upward on its own (quarter-speed) clock; "
          "*Anim Rate* ticks each sprite through its frames — dots blink and "
          "split, omegas breathe. All three run LIVE on shared clocks — the "
          "sea moves as one. The jitters set WHEN each wave's ticks land: at "
          "0 every step in the sea snaps to the same instants (lock-step, "
          "game-boot-screen style); at 1 each wave ticks on its own staggered "
          "instants. *Drift Jitter* staggers the motion clocks, *Anim Jitter* "
          "the frame clock — waves are always at their own point in their "
          "animation either way.")
      .floatField("drift_rate", 0.4f, 0.0f, 1.0f, state::PrimaryInput).label("Drift Rate", "Drift")
      .floatField("rise", 0.3f, 0.0f, 1.0f, state::PrimaryInput).label("Rise", "Rise")
      .floatField("drift_jitter", 1.0f, 0.0f, 1.0f, state::PrimaryInput).label("Drift Jitter", "DJit")
      .floatField("anim_rate", 0.5f, 0.0f, 1.0f, state::PrimaryInput).label("Anim Rate", "Anim")
      .floatField("anim_jitter", 1.0f, 0.0f, 1.0f, state::PrimaryInput).label("Anim Jitter", "AJit")
      // --- Waves ---
      .group("waves", "Waves")
        .groupHelp(
          "*Density* sets how many waves the grid hosts at once (the pool "
          "tops out at 12 — plenty on a grid this coarse). Like the ocean it "
          "LATCHES: raising it births new waves staggered over a moment, "
          "lowering it never culls — the surplus drains as waves reach the end "
          "of their natural lives. *Dot* / *Omega Weight* set the odds each "
          "newborn is a blinking fleck or a breathing crest (ratio only). "
          "*Seed* re-deals the spawn stream.")
      .floatField("density", 0.4f, 0.0f, 1.0f, state::PrimaryInput).label("Density", "Dens")
      .floatField("dot_weight", 0.55f, 0.0f, 1.0f, state::PrimaryInput).label("Dot Weight", "Dot")
      .floatField("omega_weight", 0.45f, 0.0f, 1.0f, state::PrimaryInput).label("Omega Weight", "Omega")
      .floatField("seed", 0.0f, 0.0f, 1.0f, state::SecondaryInput).label("Seed", "Seed")
      // --- Look ---
      .group("look", "Look")
        .groupHelp(
          "*Wave Colour* and *Intensity* shape the lit cells. **Composite** "
          "picks what the rest of the grid is: Black / Transparent / a Custom "
          "colour / the Input image passed through.")
      .rgbField("pixel_color", 1.0f, 1.0f, 1.0f, state::PrimaryInput).label("Wave Colour", "Colour")
      .floatField("intensity", 1.0f, 0.0f, 2.0f, state::PrimaryInput).label("Intensity", "Int")
      .selectField("composite", CompBlack, state::PrimaryInput,
                   {{"Black", CompBlack}, {"Transparent", CompTransparent},
                    {"Custom", CompCustom}, {"Input", CompInput}}).label("Composite", "Comp")
      .rgbaField("bg_color", 0.0f, 0.0f, 0.0f, 1.0f, state::SecondaryInput).label("Background", "BG")
      .textureField("tex_in", state::PrimaryInput)
      .textureField("tex_out", state::PrimaryOutput)
      .capability(state::Capability::Generator)
      .capability(state::Capability::SeekableApproximate)
  );
  state::setOnStateReady(&on_state_ready);

  if (gpu::Device::backend() == gpu::Backend::None) return;

  state::registerShaderSPV("pixel_rift_render", RENDER_SPV, RENDER_SPV_SIZE);
  auto cs = gpu::Device::createShaderModuleByName("pixel_rift_render");
  if (!cs) return;
  s_pso = gpu::Device::createComputePSO(cs, "main", gpu::Bindings()
      .tex2d(0)
      .storageTex2d(1)
      .uniform(2));

  s_black = gpu::Device::createTexture(1, 1, gpu::TextureFormat::RGBA8);
  if (s_black.valid()) gpu::Device::clear(s_black, 0.f, 0.f, 0.f, 1.f);
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
  s->initialized = s_pso.valid() && buf.valid();
}

void tick(void* self, double dt) {
  auto* s = static_cast<State*>(self);
  if (!s) return;

  if (!s->rng_init) {
    s->rng = nano_uhash(0x52494654u ^ (uint32_t)(clampf(s->seed, 0.f, 1.f) * 65535.0f)) | 1u;
    s->rng_init = true;
  }

  // Live global step clocks: the whole sea shares one rigid translation and
  // one sprite-frame clock (per-wave offsets stagger the phases; the jitters
  // decide whether tick INSTANTS are shared or spread — see stepsSince).
  s->drift_acc += dt * driftStepsPerSec(s->drift_rate);
  s->rise_acc  += dt * riseStepsPerSec(s->rise);
  s->anim_acc  += dt * animStepsPerSec(s->anim_rate);

  int live = 0;
  for (int i = 0; i < MAX_WAVES; i++) {
    Wave& w = s->wave[i];
    if (!w.active) continue;
    if (waveAnimSteps(*s, w) >= (long)w.life_steps) w.active = false;
    else live++;
  }

  // Births only (density never culls): spawn toward the target population at
  // a bounded rate so raising density staggers waves in rather than popping
  // the whole pool at once.
  int target = clampi((int)std::lround((double)clampf(s->density, 0.f, 1.f) * MAX_WAVES),
                      0, MAX_WAVES);
  s->spawn_acc += dt * 2.0;
  if (s->spawn_acc > 4.0) s->spawn_acc = 4.0;   // don't bank a burst across idle spans
  while (s->spawn_acc >= 1.0) {
    s->spawn_acc -= 1.0;
    if (live < target) {
      trySpawnWave(*s);
      live = 0;
      for (int i = 0; i < MAX_WAVES; i++) if (s->wave[i].active) live++;
    }
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
    if      (state::pathIs(p, l, "columns"))      s->columns = (int)state::patchFloat(i);
    else if (state::pathIs(p, l, "rows"))         s->rows = (int)state::patchFloat(i);
    else if (state::pathIs(p, l, "rift_cols"))    s->rift_cols = (int)state::patchFloat(i);
    else if (state::pathIs(p, l, "drift_rate"))   s->drift_rate = state::patchFloat(i);
    else if (state::pathIs(p, l, "rise"))         s->rise = state::patchFloat(i);
    else if (state::pathIs(p, l, "drift_jitter")) s->drift_jitter = state::patchFloat(i);
    else if (state::pathIs(p, l, "anim_rate"))    s->anim_rate = state::patchFloat(i);
    else if (state::pathIs(p, l, "anim_jitter"))  s->anim_jitter = state::patchFloat(i);
    else if (state::pathIs(p, l, "density"))      s->density = state::patchFloat(i);
    else if (state::pathIs(p, l, "dot_weight"))   s->dot_weight = state::patchFloat(i);
    else if (state::pathIs(p, l, "omega_weight")) s->omega_weight = state::patchFloat(i);
    else if (state::pathIs(p, l, "seed"))         s->seed = state::patchFloat(i);
    else if (state::pathIs(p, l, "pixel_color")) {
      auto v = state::patchVec3(i);
      s->color_r = v.x; s->color_g = v.y; s->color_b = v.z;
    }
    else if (state::pathIs(p, l, "intensity"))    s->intensity = state::patchFloat(i);
    else if (state::pathIs(p, l, "composite")) {
      s->composite = (int)state::patchFloat(i);
      apply_composite_visibility(s->composite);
    }
    else if (state::pathIs(p, l, "bg_color")) {
      auto v = state::patchVec4(i);
      s->bg_color[0] = v.x; s->bg_color[1] = v.y; s->bg_color[2] = v.z; s->bg_color[3] = v.w;
    }
  }
}

void render(void* self, int vp_w, int vp_h) {
  auto* s = static_cast<State*>(self);
  if (!s || !s->initialized || vp_w <= 0 || vp_h <= 0) return;
  auto in  = gpu::Device::textureForField("tex_in");
  auto out = gpu::Device::textureForField("tex_out");
  if (!out.valid()) return;
  if (!in.valid()) in = s_black;   // chain-start: no upstream input
  if (!in.valid()) return;

  const int cols  = clampi(s->columns, 1, MAX_COLS);
  const int rows  = clampi(s->rows, 2, 32);
  const int rift  = clampi(s->rift_cols, 0, MAX_COLS);
  const int torus = cols + rift + TORUS_PAD;

  Uniforms u = {};
  u.cols = cols;
  u.rows = rows;
  u.rift = rift;
  u.composite = s->composite;
  u.color[0] = s->color_r; u.color[1] = s->color_g; u.color[2] = s->color_b; u.color[3] = 1.0f;
  for (int i = 0; i < 4; i++) u.bg[i] = s->bg_color[i];
  u.intensity = clampf(s->intensity, 0.0f, 4.0f);
  u.n_waves = MAX_WAVES;
  for (int i = 0; i < MAX_WAVES; i++) {
    const Wave& w = s->wave[i];
    if (!w.active) continue;
    int x, y;
    wavePos(*s, w, torus, rows, x, y);
    u.waves[i][0] = x;
    u.waves[i][1] = y;
    u.waves[i][2] = w.type * 4 + waveFrame(*s, w);
    u.waves[i][3] = 1;
  }
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

} // namespace pixel_rift
