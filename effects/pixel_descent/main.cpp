/*
 * source.pixel.descent — beat-locked stepping grid.
 *
 * The screen splits into a cols × rows grid (default 4 × 10) with exactly one
 * lit pixel per column. All columns start at the top and step linearly to the
 * bottom over a beat-clock loop (default 8 beats), so unjittered they read as
 * a solid line sweeping down. Two randomizers break the line up:
 *
 *  - Per-cycle jitter: each column draws a random "eagerness" each cycle —
 *    that value is the CHANCE any given step lands early, by a fixed fraction
 *    of a step set by the Jitter param.
 *  - Per-step skip: each step there's a Skip Chance that one random column
 *    takes its step VERY early (~0.9 of a step — a near-double-step stutter),
 *    then sits on that row extra-long.
 *
 * Step-time model: with R rows and an L-beat loop, the unwrapped step clock is
 * T = beats * R / L (in step units); global step n belongs to cycle n/R and
 * row n%R, and fires for column c at t = n - offset(c, n) with all offsets in
 * [0, 1). Offsets < 1 keep each column's step times monotonic, and the lit
 * step is floor(T) or floor(T)+1 — a two-candidate O(1) check per column.
 * The wrap into row 0 is a step like any other, so eager columns pop back to
 * the top slightly BEFORE the downbeat.
 *
 * The clock is bar-locked (mod_time style: bar-wrap counting + barPhase, 1 bar
 * = 4 beats), so cycle starts stay phase-locked to the host downbeat. Offsets
 * come from deterministic hashes of (column, step, cycle, seed) — no RNG
 * state, so the pattern is a pure function of the transport (seek-friendly).
 * If the host provides no beat info (barPhase pinned at 0), the grid sits on
 * its top row.
 *
 * Class-like instance model: module_init() sets up the type-shared compute
 * PSO + schema once; each chain entry gets its own State via create().
 */

#include <gpu.h>
#include <host.h>
#include "pixel_descent_shaders.h"

#include <cmath>
#include <cstdint>

namespace pixel_descent {

enum Composite { CompBlack = 0, CompTransparent = 1, CompCustom = 2, CompInput = 3 };

static const int MAX_COLS = 16;
static const double kBeatsPerBar = 4.0;

// Uniform layout — MUST match render.hlsl's cbuffer byte-for-byte. The 16
// lit-row ints travel as int4[4] (cbuffer arrays have 16-byte stride).
struct Uniforms {
  int32_t cols;
  int32_t rows;
  int32_t composite;
  float   intensity;
  float   color[4];
  float   bg[4];
  int32_t lit[MAX_COLS];
};
static_assert(sizeof(Uniforms) == 112, "Uniforms layout mismatch");

// Per-instance state. One per chain entry.
struct State {
  // Schema-mirrored params
  int   columns     = 4;
  int   rows        = 10;
  float beats       = 8.0f;
  float jitter      = 0.35f;
  float skip_chance = 0.15f;
  float seed        = 0.0f;
  float color_r = 1.0f, color_g = 1.0f, color_b = 1.0f;
  float intensity   = 1.0f;
  int   composite   = CompBlack;
  float bg_color[4] = { 0.0f, 0.0f, 0.0f, 1.0f };

  // Bar-locked beat clock (mod_time pattern): count bar wraps, reconstruct
  // beats as (bars + barPhase) * 4 so cycles stay locked to the host downbeat.
  double prev_bar_phase = -1.0;   // sentinel: -1 = unseeded
  long   bars = 0;

  bool initialized = false;
  gpu::Buffer uniform_buf;
};

// Type-shared: compiled once in module_init(), reused by every instance.
static gpu::ComputePSO s_pso;

static inline float clampf(float v, float lo, float hi) { return v < lo ? lo : (v > hi ? hi : v); }
static inline int   clampi(int v, int lo, int hi)       { return v < lo ? lo : (v > hi ? hi : v); }

// Integer hash (mirror of nano_hash.hlsl's nano_uhash) + a keyed hash01, same
// construction as pixel_ocean's CPU mirror.
static inline uint32_t nano_uhash(uint32_t x) {
  x ^= x >> 17; x *= 0xed5ad4bbu; x ^= x >> 11; x *= 0xac4c1b51u;
  x ^= x >> 15; x *= 0x31848babu; x ^= x >> 14; return x;
}
static inline float hash01(uint32_t a, uint32_t b, uint32_t stream, uint32_t seed) {
  uint32_t h = nano_uhash(seed ^ (stream * 0x9E3779B9u));
  h = nano_uhash(h + a);
  h = nano_uhash(h + b);
  return (float)h * (1.0f / 4294967296.0f);
}
enum { ST_EAGER = 0, ST_GATE = 1, ST_SKIP = 2, ST_SKIPCOL = 3 };

// How early (fraction of a step, in [0, 1)) column c takes global step n.
static float stepOffset(const State& s, uint32_t seed, int cols, int rows, long n, int c) {
  if (n < 0) return 0.0f;
  uint32_t un = (uint32_t)n;
  uint32_t cycle = (uint32_t)(n / rows);
  float off = 0.0f;
  // Per-cycle eagerness IS the chance this step lands early.
  float eager = hash01((uint32_t)c, cycle, ST_EAGER, seed);
  if (hash01((uint32_t)c, un, ST_GATE, seed) < eager)
    off = clampf(s.jitter, 0.0f, 0.95f);
  // Per-step skip: one hashed column near-double-steps (~0.9 of a step early).
  if (s.skip_chance > 0.0f && hash01(un, 0u, ST_SKIP, seed) < s.skip_chance) {
    int pick = (int)(nano_uhash(un ^ nano_uhash(seed + ST_SKIPCOL)) % (uint32_t)cols);
    if (pick == c && off < 0.9f) off = 0.9f;
  }
  return off;
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

// Type-level setup: schema + shared compute PSO. Runs once per type.
void module_init() {
  state::init("source.pixel.descent", {1, 0, 0},
    state::Schema()
      .helpField("intro",
        "## Pixel Descent\n"
        "A beat-locked grid: one lit pixel per column, all starting at the top "
        "on the downbeat and stepping to the bottom over the loop — a line "
        "sweeping down, broken up by per-column timing jitter. Each cycle every "
        "column draws a fresh personality: how often its steps rush ahead of "
        "the clock.\n\n"
        "**Try:** *Jitter* 0 for a laser-straight line; push it up so eager "
        "columns lean ahead (they pop back to the top slightly BEFORE the "
        "beat); add *Skip Chance* for the occasional stuttered double-step; "
        "set *Composite → Input* to run the pixels over the layer below.")
      // --- Grid ---
      .group("grid", "Grid")
        .groupHelp(
          "The full viewport splits evenly into *Columns* × *Rows* cells. One "
          "cell per column is lit at any moment; *Rows* also sets how many "
          "steps the descent takes per loop.")
      .intField("columns", 4, 1, MAX_COLS, state::PrimaryInput).label("Columns", "Cols")
      .intField("rows", 10, 2, 32, state::PrimaryInput).label("Rows", "Rows")
      // --- Clock ---
      .group("clock", "Clock")
        .groupHelp(
          "*Loop Beats* is the descent's length on the host beat clock (8 = "
          "two 4/4 bars). The loop is phase-locked to the transport: cycle "
          "starts land on the downbeat, wherever the effect was loaded.")
      .floatField("beats", 8.0f, 1.0f, 64.0f, state::PrimaryInput, nullptr, 1.0f, "beats")
        .label("Loop Beats", "Beats")
      // --- Motion ---
      .group("motion", "Motion")
        .groupHelp(
          "*Jitter* is how far ahead of the clock an early step lands, as a "
          "fraction of one step. WHICH steps rush is each column's per-cycle "
          "personality — a random eagerness redrawn every loop, so one pass a "
          "column leans ahead constantly, the next it holds the line. *Skip "
          "Chance* is a per-step chance that one random column takes its step "
          "almost a whole step early — a stutter that 'skips a beat' — then "
          "sits there extra-long. Both also stagger how columns reappear at "
          "the top when the cycle restarts. *Seed* re-deals all of it.")
      .floatField("jitter", 0.35f, 0.0f, 1.0f, state::PrimaryInput).label("Jitter", "Jit")
      .floatField("skip_chance", 0.15f, 0.0f, 1.0f, state::PrimaryInput).label("Skip Chance", "Skip")
      .floatField("seed", 0.0f, 0.0f, 1.0f, state::SecondaryInput).label("Seed", "Seed")
      // --- Look ---
      .group("look", "Look")
        .groupHelp(
          "*Pixel Colour* and *Intensity* shape the lit cells. **Composite** "
          "picks what the rest of the grid is: Black / Transparent / a Custom "
          "colour / the Input image passed through.")
      .rgbField("pixel_color", 1.0f, 1.0f, 1.0f, state::PrimaryInput).label("Pixel Colour", "Colour")
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

  state::registerShaderSPV("pixel_descent_render", RENDER_SPV, RENDER_SPV_SIZE);
  auto cs = gpu::Device::createShaderModuleByName("pixel_descent_render");
  if (!cs) return;

  s_pso = gpu::Device::createComputePSO(cs, "main", gpu::Bindings()
      .tex2d(0)
      .storageTex2d(1)
      .uniform(2));
}

// Per-instance construction: allocate State + its own uniform buffer.
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
  (void)dt;
  // Bar-locked clock: count wraps, treat backward jumps (host restart/scrub)
  // as wraps too so the clock never runs backwards past a cycle boundary.
  const double bp = host::barPhase();
  if (s->prev_bar_phase < 0.0) { s->prev_bar_phase = bp; return; }
  if (bp < s->prev_bar_phase - 0.5) s->bars++;
  s->prev_bar_phase = bp;
}

void on_state_patched(void* self, int n, const char* pb, const int* off,
                      const int* len, const int* ops) {
  auto* s = static_cast<State*>(self);
  if (!s) return;
  for (int i = 0; i < n; i++) {
    if (ops[i] != state::PatchReplace) continue;
    const char* path = pb + off[i];
    int plen = len[i];
    if      (state::pathIs(path, plen, "columns"))     s->columns = (int)state::patchFloat(i);
    else if (state::pathIs(path, plen, "rows"))        s->rows = (int)state::patchFloat(i);
    else if (state::pathIs(path, plen, "beats"))       s->beats = state::patchFloat(i);
    else if (state::pathIs(path, plen, "jitter"))      s->jitter = state::patchFloat(i);
    else if (state::pathIs(path, plen, "skip_chance")) s->skip_chance = state::patchFloat(i);
    else if (state::pathIs(path, plen, "seed"))        s->seed = state::patchFloat(i);
    else if (state::pathIs(path, plen, "pixel_color")) {
      auto v = state::patchVec3(i);
      s->color_r = v.x; s->color_g = v.y; s->color_b = v.z;
    }
    else if (state::pathIs(path, plen, "intensity"))   s->intensity = state::patchFloat(i);
    else if (state::pathIs(path, plen, "composite")) {
      s->composite = (int)state::patchFloat(i);
      apply_composite_visibility(s->composite);
    }
    else if (state::pathIs(path, plen, "bg_color")) {
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
  if (!in.valid() || !out.valid()) return;

  const int cols = clampi(s->columns, 1, MAX_COLS);
  const int rows = clampi(s->rows, 2, 32);
  const double loop_beats = (double)clampf(s->beats, 0.25f, 64.0f);
  const uint32_t seed = (uint32_t)(clampf(s->seed, 0.0f, 1.0f) * 65535.0f);

  // Unwrapped step clock (step units). prev_bar_phase < 0 = clock unseeded.
  const double bp = s->prev_bar_phase < 0.0 ? 0.0 : s->prev_bar_phase;
  const double beats = ((double)s->bars + bp) * kBeatsPerBar;
  const double T = beats * (double)rows / loop_beats;
  const long base = (long)T;

  Uniforms u = {};
  u.cols = cols;
  u.rows = rows;
  u.composite = s->composite;
  u.intensity = clampf(s->intensity, 0.0f, 4.0f);
  u.color[0] = s->color_r; u.color[1] = s->color_g; u.color[2] = s->color_b; u.color[3] = 1.0f;
  for (int i = 0; i < 4; i++) u.bg[i] = s->bg_color[i];
  for (int c = 0; c < MAX_COLS; c++) {
    long lit = base;
    if (c < cols) {
      // Offsets are < 1 step, so the only candidate beyond floor(T) is the
      // next step — take it if its (possibly early) fire time has passed.
      float off = stepOffset(*s, seed, cols, rows, base + 1, c);
      if (off > 0.0f && (double)(base + 1) - (double)off <= T) lit = base + 1;
    }
    u.lit[c] = (int32_t)(lit % (long)rows);
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

} // namespace pixel_descent
