/*
 * source.light.side_jet — JPL-style engine-test plume.
 *
 * The engine under test is FIXED at the left edge. When it lights and
 * throttles up, a jet expands rightward; its entire structure is a live
 * function of throttle / chamber pressure, so you feel the thrust in how
 * the plume reacts. Designed for the LED bars AND for direct screen use.
 *
 * Pipeline (all inside one render()):
 *   Stage 1  sim.hlsl   — 1D axial solver, single workgroup, substepped.
 *                          Two propagation speeds: pressure FAST (snappy
 *                          structure), luminous material SLOW (physical
 *                          transit). Writes the persistent cell buffer.
 *   Stage 2  color.hlsl — 2D synthesis: potential core, shock diamonds,
 *                          Mach disk, KH shear vortices, crackle, sparks.
 *   Stage 2b motion.hlsl — analytic u(x) motion field for downstream blur.
 *
 * CPU control integrator (tick): throttle → spool-lagged chamber pressure
 * → exit velocity / pressure ratio / ignition-front BC. Ignition rising
 * edge sprays the low-frequency spark pool and kicks a startup overshoot.
 */

#include <gpu.h>
#include <host.h>
#include <effect_utils.h>
#include "side_jet_shaders.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>

namespace side_jet {

static constexpr int NUM_CELLS  = 192;   // axial resolution (<= 256)
static constexpr int MAX_SPARKS = 64;

struct GpuCell {
  float u, p, b, m;
  float kappa, phi, lit, _pad;
};
static_assert(sizeof(GpuCell) == 32, "GpuCell layout mismatch");

struct GpuSpark {
  float x, y;        // screen position (jet-uv space)
  float vx, vy;      // screen velocity (for orientation / streaking)
  float life, size;  // life = brightness, size = base radius
  float _p0, _p1;
};
static_assert(sizeof(GpuSpark) == 32, "GpuSpark layout mismatch");

struct SimUniforms {
  float dt;
  uint32_t substeps;
  uint32_t W;
  float dx;

  float chamberP;
  float exitVel;
  float pressureRatio;
  float litTarget;

  float wavespeed;
  float maturityGrowth;
  float coreDecay;
  float flameSpeed;

  float diamondSpacing;
  float velRelax;
  float _pad0;
  float _pad1;
};
static_assert(sizeof(SimUniforms) == 64, "SimUniforms layout mismatch");

struct ColorUniforms {
  float intensity, centerline_y, nozzle_radius, spread;
  float radial_sharpness, diamond_amp, mach_disk_x, mach_disk_amp;
  float mach_disk_width, shimmer_phase, kh_amp, kh_scale;
  float kh_phase, crackle_amp, crackle_phase, mixture;
  float zoom, _padc1, _padc2, aspect;
  float _pade0, _pade1, _pade2, core_brightness;
  uint32_t cell_count, spark_count, debug_show_axis;
  float motion_scale;
};
static_assert(sizeof(ColorUniforms) == 112, "ColorUniforms layout mismatch");

// Sparks live in a 3D world to suggest the (un-rendered) physical engine
// cone: x = downstream, y = vertical, z = depth (+ toward viewer). They are
// spawned around the nozzle RIM (a circle in the y-z plane) and projected to
// screen with a slight 3/4 view, so the ring + perspective read as a real
// cone mouth. A fraction spawn "bounced" — ricocheting off the structure.
struct CpuSpark {
  float px, py, pz;     // world position
  float vx, vy, vz;     // world velocity
  float life, max_life;
  float size, bright;
  bool  active;
};

// "One knob" drama mode — per-param multipliers blended across three throttle
// phases. Applied on top of the user's values (1.0 = no change).
struct DramaMods {
  float sharp = 1, spacing = 1, bright = 1, shimmer = 1, kh = 1, crackle = 1,
        crackle_hz = 1, length = 1, spread = 1, diamamp = 1, shear = 1, machdisk = 1;
};

struct State {
  // --- GPU resources (per-instance) ---
  gpu::Buffer  cell_buf;
  gpu::Buffer  sim_uniform_buf;
  gpu::Buffer  color_uniform_buf;
  gpu::Buffer  spark_buf;
  gpu::Texture motion_tex;
  gpu::Texture zero_motion_tex;
  int          motion_w = 0;
  int          motion_h = 0;
  bool         initialized = false;

  // --- Schema-mirrored params ---
  bool  ignition          = true;
  float throttle          = 0.7f;
  float mixture           = 0.3f;
  float intensity         = 1.0f;
  float drama             = 0.0f;   // "one knob" mode amount (0 = off)

  float spool_time        = 0.06f;
  float startup_overshoot = 0.4f;
  float overshoot_time    = 0.18f;  // overshoot decay time constant (sec)

  float centerline_y      = 0.5f;
  float nozzle_radius     = 0.45f;
  float spread            = 0.15f;
  float length_scale      = 1.0f;

  float core_brightness   = 1.7f;
  float radial_sharpness  = 5.0f;
  float diamond_amp       = 0.6f;
  float diamond_spacing   = 0.06f;
  float mach_disk_amp     = 0.8f;
  float core_length       = 1.0f;   // white potential-core extent along X

  float shear_turbulence  = 0.5f;
  float shear_scale       = 18.0f;
  float crackle           = 0.3f;
  float shimmer_rate_hz   = 9.0f;
  float kh_rate_hz        = 6.0f;
  float crackle_rate_hz   = 22.0f;

  float zoom              = 1.0f;   // magnify, anchored at left-center

  float propagation       = 0.6f;
  int   substeps          = 128;
  float motion_scale      = 0.5f;
  float spark_amount      = 0.6f;   // ignition-burst size
  float spark_rate        = 12.0f;  // continuous sparks/sec while firing
  float spark_scale       = 1.0f;   // render-size multiplier
  float spark_speed       = 1.6f;   // initial-velocity multiplier (ballistic)
  int   seed              = 0x5A1E7;
  bool  debug_show_axis   = false;

  // --- CPU control integrator ---
  double chamberP   = 0.0;     // spool-lagged chamber pressure
  double overshoot  = 0.0;     // decaying startup transient
  bool   ign_prev   = false;
  float  mach_disk_x = 0.15f;

  // --- Phase accumulators (§2.1) ---
  double shimmer_phase = 0.0;
  double kh_phase      = 0.0;
  double crackle_phase = 0.0;

  // --- Spark pool ---
  CpuSpark sparks[MAX_SPARKS];
  double   spark_accum = 0.0;   // continuous-spawn fractional accumulator
  uint32_t spawn_rng = 0xB16B00B5u;

  // --- Drama-mode multipliers (recomputed each tick from throttle) ---
  DramaMods dmod;
};

static gpu::ComputePSO s_pso_sim;
static gpu::ComputePSO s_pso_color;
static gpu::ComputePSO s_pso_motion;

static inline float clampf(float v, float lo, float hi) {
  return v < lo ? lo : (v > hi ? hi : v);
}

// Soft-knee floor: returns ~x well above `floor`, then smoothly asymptotes
// toward `floor` below it (heavy squash). Exact passthrough for
// x >= floor + knee, so it only bites the region we want to protect.
static inline float softFloor(float x, float floor, float knee) {
  if (knee <= 1e-5f) return x > floor ? x : floor;
  float d = x - floor;
  if (d >= knee) return x;
  return floor + knee * std::exp((d - knee) / knee);
}
static inline uint32_t lcg_next(uint32_t& s) {
  s = s * 1664525u + 1013904223u; return s;
}
static inline float lcg_unit(uint32_t& s) {
  return (lcg_next(s) >> 8) * (1.0f / (float)(1u << 24));
}
static inline float lcg_signed(uint32_t& s) {
  return lcg_unit(s) * 2.0f - 1.0f;
}

// "One knob" drama: throttle sweeps through three phases, each a smooth bump
// (centered at 0.17 / 0.50 / 0.83). Each param blends three per-phase target
// multipliers (A,B,C) — chosen NON-monotonically so params go up→down→up,
// giving the sweep distinct "character". The result is scaled by `drama` and
// applied multiplicatively on top of the user's values.
//
//   Phase A (low)  — coiled & tense: sharp, tight/fast diamonds, fast shimmer.
//   Phase B (mid)  — a breath: softens, diamonds widen/slow, brightness swells.
//   Phase C (high) — full roar: sharpest, tightest fast diamonds, crackle bites.
static DramaMods computeDrama(float throttle, float drama) {
  DramaMods d;
  float t  = clampf(throttle, 0.0f, 1.0f);
  float dr = clampf(drama, 0.0f, 1.0f);
  float a = (t - 0.17f) / 0.20f, b = (t - 0.50f) / 0.20f, c = (t - 0.83f) / 0.20f;
  float wA = std::exp(-a * a), wB = std::exp(-b * b), wC = std::exp(-c * c);
  float ws = wA + wB + wC + 1e-5f; wA /= ws; wB /= ws; wC /= ws;
#define PH(A, B, C) (1.0f + dr * ((wA * (A) + wB * (B) + wC * (C)) - 1.0f))
  d.sharp      = PH(1.50f, 0.62f, 1.95f);  // expand+sharpen → intensity (down then up)
  d.spacing    = PH(0.55f, 1.70f, 0.42f);  // diamonds: fast → slow → fast
  d.bright     = PH(0.85f, 1.25f, 1.60f);  // swell then bloom
  d.shimmer    = PH(1.40f, 0.75f, 1.80f);  // shimmer rate
  d.kh         = PH(1.20f, 0.85f, 1.50f);  // shear churn rate
  d.crackle    = PH(0.40f, 0.80f, 1.70f);  // subtle, bites at the top
  d.crackle_hz = PH(0.85f, 1.00f, 1.50f);
  d.length     = PH(0.95f, 1.12f, 1.00f);  // gentle (per spec)
  d.spread     = PH(0.90f, 1.20f, 1.00f);  // gentle (per spec)
  d.diamamp    = PH(0.80f, 1.00f, 1.35f);
  d.shear      = PH(0.85f, 1.00f, 1.35f);
  d.machdisk   = PH(0.70f, 1.10f, 1.50f);
#undef PH
  return d;
}

static void spawn_sparks(State& s, int count) {
  const float TAU = 6.2831853f;
  float R = s.nozzle_radius * 1.05f;   // rim ~ visible nozzle mouth
  for (int n = 0; n < count; n++) {
    int slot = -1;
    for (int i = 0; i < MAX_SPARKS; i++) if (!s.sparks[i].active) { slot = i; break; }
    if (slot < 0) return;              // pool full
    CpuSpark& sp = s.sparks[slot];

    float th = lcg_unit(s.spawn_rng) * TAU;
    float ct = std::cos(th), st = std::sin(th);
    sp.px = 0.0f;                       // nozzle plane
    sp.py = R * ct;                     // around the rim...
    sp.pz = R * st;                     // ...as a circle in y-z
    float oy = ct, oz = st;            // outward radial direction

    bool bounced = lcg_unit(s.spawn_rng) < 0.30f;
    if (!bounced) {
      // Entrained by the jet — swept downstream, a little outward.
      sp.vx = 0.25f + lcg_unit(s.spawn_rng) * 0.70f;
      float out = 0.10f + lcg_unit(s.spawn_rng) * 0.25f;
      sp.vy = oy * out + lcg_signed(s.spawn_rng) * 0.08f;
      sp.vz = oz * out + lcg_signed(s.spawn_rng) * 0.08f;
      sp.bright = 0.7f + lcg_unit(s.spawn_rng) * 0.3f;
    } else {
      // Ricochet — little downstream, strong outward + upward kick, as if it
      // bounced off the (invisible) casing / test-stand structure.
      sp.vx = -0.05f + lcg_unit(s.spawn_rng) * 0.25f;
      float out = 0.35f + lcg_unit(s.spawn_rng) * 0.55f;
      sp.vy = oy * out - 0.30f;        // bias up (world -y → screen up)
      sp.vz = oz * out + lcg_signed(s.spawn_rng) * 0.20f;
      sp.bright = 0.9f + lcg_unit(s.spawn_rng) * 0.4f;
    }
    // Ballistic launch — scale the whole initial velocity by spark_speed.
    float spd = clampf(s.spark_speed, 0.05f, 8.0f);
    sp.vx *= spd; sp.vy *= spd; sp.vz *= spd;
    sp.life = sp.max_life = 0.5f + lcg_unit(s.spawn_rng) * 1.1f;
    sp.size = 0.004f + lcg_unit(s.spawn_rng) * 0.006f;
    sp.active = true;
  }
}

void module_init() {
  state::init("source.light.side_jet", {2, 0, 1},
    state::Schema()
      // Top-level manual: high-level "what is this / how to use / what to try".
      .helpField("intro",
        "## Side Jet\n"
        "A JPL-style rocket engine-test plume: the nozzle is fixed at the left "
        "edge and a supersonic jet blasts rightward — shock diamonds, Mach disk, "
        "shear vortices, crackle and rim sparks all react live to the throttle. "
        "Built for LED bars and for direct screen use.\n\n"
        "**How to drive it:** toggle **Ignition** to light the engine, then ride "
        "**Throttle** — the plume spools up with real inertia and overshoots on "
        "light-up. **Try:** turn **Drama** up and sweep the throttle for a "
        "one-knob performance that resculpts the whole plume through its power "
        "band; drop **Mixture** for a tighter blue core, or crank **Zoom** to fill "
        "the frame with the shock train.")
      // --- Drive (performable) ---
      .group("drive", "Drive")
        .groupHelp(
          "The performable core. **Ignition** lights the engine (a rising edge "
          "sprays sparks and kicks a startup overshoot); **Throttle** then drives "
          "everything downstream through spool inertia. **Mixture** shifts the "
          "chemistry — lower is a tighter, bluer core, higher over-expands and "
          "pushes the Mach disk out. **Drama** is a one-knob macro: turn it up and "
          "the throttle sweep alone resculpts sharpness, diamond spacing, crackle "
          "and brightness across the power band.")
      .boolField ("ignition",          true,                     state::PrimaryInput).label("Ignition", "Ign")
      .floatField("throttle",          0.7f,  0.0f, 1.0f,        state::PrimaryInput).label("Throttle", "Thrt")
      .floatField("mixture",           0.3f,  0.0f, 1.0f,        state::PrimaryInput).label("Mixture", "Mix")
      .floatField("intensity",         1.0f,  0.0f, 3.0f,        state::PrimaryInput).label("Intensity", "Int")
      .floatField("drama",             0.0f,  0.0f, 1.0f,        state::PrimaryInput).label("Drama", "Drama")
      // --- Engine dynamics ---
      .group("engine", "Engine Dynamics")
      .floatField("spool_time",        0.06f, 0.01f, 1.0f,       state::PrimaryInput).label("Spool Time", "Spool")
      .floatField("startup_overshoot", 0.4f,  0.0f, 1.0f,        state::PrimaryInput).label("Startup Overshoot", "OShoot")
      .floatField("overshoot_time",    0.18f, 0.02f, 2.0f,       state::PrimaryInput).label("Overshoot Time", "OvrT")
      // --- Geometry ---
      .group("geometry", "Geometry")
      .floatField("centerline_y",      0.5f,  0.0f, 1.0f,        state::PrimaryInput).label("Centerline Y", "Ctr Y")
      .floatField("nozzle_radius",     0.45f, 0.02f, 0.5f,       state::PrimaryInput).label("Nozzle Radius", "Nozzle")
      .floatField("spread",            0.15f, 0.0f, 1.0f,        state::PrimaryInput).label("Spread", "Sprd")
      .floatField("length_scale",      1.0f,  0.2f, 1.5f,        state::PrimaryInput).label("Length", "Len")
      // --- Plume structure ---
      .group("plume", "Plume Structure")
        .groupHelp(
          "The shock train inside the jet. **Core Brightness** and **Radial "
          "Sharpness** set how hot and tight the potential core reads; **Core "
          "Length** slides the white-to-blue handoff downstream. The **Diamond** "
          "controls size and space the standing shock cells, and the **Mach Disk** "
          "is the bright normal shock that appears as the flow over-expands.")
      .floatField("core_brightness",   1.7f,  0.0f, 3.0f,        state::PrimaryInput).label("Core Brightness", "Core B")
      .floatField("radial_sharpness",  5.0f,  1.0f, 16.0f,       state::PrimaryInput).label("Radial Sharpness", "Sharp")
      .floatField("diamond_amp",       0.6f,  0.0f, 1.0f,        state::PrimaryInput).label("Diamond Amount", "Diam")
      .floatField("diamond_spacing",   0.06f, 0.01f, 0.2f,       state::PrimaryInput).label("Diamond Spacing", "DiamSp")
      .floatField("mach_disk_amp",     0.8f,  0.0f, 2.0f,        state::PrimaryInput).label("Mach Disk", "Mach")
      .floatField("core_length",       1.0f,  0.2f, 3.0f,        state::PrimaryInput).label("Core Length", "CoreL")
      // --- Detail (screen) ---
      .group("detail", "Detail")
        .groupHelp(
          "Fine turbulent texture layered on the plume. **Shear Turbulence** and "
          "**Scale** grow Kelvin-Helmholtz vortices along the jet edge; **Crackle** "
          "adds the sharp popping speckle of a hot exhaust. The **Rate** knobs set "
          "how fast the shimmer, shear and crackle churn — pull them down for a "
          "slow-motion look, push them up for a violent, energetic burn.")
      .floatField("shear_turbulence",  0.5f,  0.0f, 1.0f,        state::PrimaryInput).label("Shear Turbulence", "Shear")
      .floatField("shear_scale",       18.0f, 4.0f, 40.0f,       state::PrimaryInput).label("Shear Scale", "ShrScl")
      .floatField("crackle",           0.3f,  0.0f, 1.0f,        state::PrimaryInput).label("Crackle", "Crkl")
      .floatField("shimmer_rate_hz",   9.0f,  0.0f, 30.0f,       state::PrimaryInput).label("Shimmer Rate", "Shim")
      .floatField("kh_rate_hz",        6.0f,  0.0f, 30.0f,       state::PrimaryInput).label("Shear Rate", "ShrRt")
      .floatField("crackle_rate_hz",   22.0f, 0.0f, 60.0f,       state::PrimaryInput).label("Crackle Rate", "CrklRt")
      // --- View ---
      .group("view", "View")
      .floatField("zoom",              1.0f,  1.0f, 12.0f,       state::PrimaryInput).label("Zoom", "Zoom")
      // --- Solver / detail amount ---
      .group("solver", "Solver & Sparks")
      .floatField("propagation",       0.6f,  0.0f, 1.0f,        state::PrimaryInput).label("Propagation", "Prop")
      .intField  ("substeps",          128, 8, 256,              state::PrimaryInput).label("Substeps", "Subst")
      .floatField("motion_scale",      0.5f,  0.0f, 1.0f,        state::PrimaryInput).label("Motion Scale", "Motion")
      .floatField("spark_amount",      0.6f,  0.0f, 1.0f,        state::PrimaryInput).label("Spark Amount", "SpkAmt")
      .floatField("spark_rate",        12.0f, 0.0f, 60.0f,       state::PrimaryInput).label("Spark Rate", "SpkRt")
      .floatField("spark_scale",       1.0f,  0.1f, 5.0f,        state::PrimaryInput).label("Spark Size", "SpkSz")
      .floatField("spark_speed",       1.6f,  0.1f, 5.0f,        state::PrimaryInput).label("Spark Speed", "SpkSpd")
      .intField  ("seed",              0x5A1E7, 0, 0x7FFFFFFF,   state::PrimaryInput).label("Seed", "Seed")
      // --- Debug ---
      .group("debug", "Debug")
      .boolField ("debug_show_axis",   false,                    state::PrimaryInput).label("Show Axis", "Axis")
      // --- I/O ---
      .textureField("tex_in",  state::PrimaryInput)
      .textureField("tex_out", state::PrimaryOutput)
      .renderOutputs(state::PrimaryOutput)
      .renderOutputs(state::PrimaryInput, "render_outputs_in")
        .capability(state::Capability::Generator)
    );

  if (gpu::Device::backend() == gpu::Backend::None) return;

  state::registerShaderSPV("side_jet_sim",    SIM_SPV,    SIM_SPV_SIZE);
  state::registerShaderSPV("side_jet_color",  COLOR_SPV,  COLOR_SPV_SIZE);
  state::registerShaderSPV("side_jet_motion", MOTION_SPV, MOTION_SPV_SIZE,
                           "rgba16float", "write");
  auto cs_sim    = gpu::Device::createShaderModuleByName("side_jet_sim");
  auto cs_color  = gpu::Device::createShaderModuleByName("side_jet_color");
  auto cs_motion = gpu::Device::createShaderModuleByName("side_jet_motion");
  if (!cs_sim || !cs_color || !cs_motion) return;

  s_pso_sim = gpu::Device::createComputePSO(cs_sim, "main", gpu::Bindings()
      .storageRW(0)
      .uniform(1));
  s_pso_color = gpu::Device::createComputePSO(cs_color, "main", gpu::Bindings()
      .tex2d(0)
      .storageTex2d(1)
      .uniform(2)
      .storage(3)
      .storage(4));
  s_pso_motion = gpu::Device::createComputePSO(cs_motion, "main", gpu::Bindings()
      .tex2d(0)
      .storageTex2d(1, gpu::TextureFormat::RGBA16F)
      .uniform(2)
      .storage(3));

  state::log("side_jet: module initialized");
}

void* create() {
  auto* s = new State();
  s->cell_buf         = gpu::Device::createBuffer(sizeof(GpuCell) * NUM_CELLS, gpu::BufferUsage::Storage);
  s->sim_uniform_buf  = gpu::Device::createBuffer(sizeof(SimUniforms), gpu::BufferUsage::Uniform);
  s->color_uniform_buf= gpu::Device::createBuffer(sizeof(ColorUniforms), gpu::BufferUsage::Uniform);
  s->spark_buf        = gpu::Device::createBuffer(sizeof(GpuSpark) * MAX_SPARKS, gpu::BufferUsage::Storage);
  return s;
}

void destroy(void* self) {
  auto* s = static_cast<State*>(self);
  if (!s) return;
  s->cell_buf.release();
  s->sim_uniform_buf.release();
  s->color_uniform_buf.release();
  s->spark_buf.release();
  s->motion_tex.release();
  s->zero_motion_tex.release();
  delete s;
}

void init(void* self) {
  auto* s = static_cast<State*>(self);
  if (!s) return;
  s->initialized = false;
  s->chamberP = 0.0;
  s->overshoot = 0.0;
  s->ign_prev = false;
  s->shimmer_phase = s->kh_phase = s->crackle_phase = 0.0;
  s->spawn_rng = (uint32_t)s->seed ^ 0xB16B00B5u;
  s->mach_disk_x = 0.15f;
  for (int i = 0; i < MAX_SPARKS; i++) s->sparks[i] = {};
  s->spark_accum = 0.0;
  s->motion_w = s->motion_h = 0;

  // Zero the persistent cell field (ambient pressure = 1, dark, cold).
  GpuCell zero[NUM_CELLS];
  for (int i = 0; i < NUM_CELLS; i++) { zero[i] = {}; zero[i].p = 1.0f; }
  if (s->cell_buf.valid()) s->cell_buf.writeBytes(zero, (int)sizeof(zero));

  if (!s_pso_sim.valid() || !s_pso_color.valid() || !s_pso_motion.valid()) return;
  s->initialized = true;
  state::log("side_jet: initialized");
}

void tick(void* self, double dt) {
  auto* s = static_cast<State*>(self);
  if (!s || !s->initialized) return;
  float fdt = (float)(dt > 0.1 ? 0.1 : dt);   // clamp pathological frames

  bool firing = s->ignition;

  // Spool-lagged chamber pressure (the engine's mechanical inertia → the
  // performer-felt throttle delay).
  double target = firing ? (double)clampf(s->throttle, 0.0f, 1.0f) : 0.0;
  double alpha  = 1.0 - std::exp(-(double)fdt / (double)clampf(s->spool_time, 0.01f, 2.0f));
  s->chamberP += (target - s->chamberP) * alpha;
  // Startup overshoot decays back down (real engines overshoot then settle).
  s->overshoot *= std::exp(-(double)fdt / (double)clampf(s->overshoot_time, 0.02f, 5.0f));
  double effP = s->chamberP + s->overshoot;
  if (effP < 0.0) effP = 0.0;

  // Drama "one knob" multipliers driven by the EFFECTIVE engine power (effP =
  // spool-lagged chamber pressure + startup overshoot), not the raw throttle
  // param — so the spool ramp and the ignition overshoot sweep the curves too
  // (the overshoot briefly pushes drama into the higher phase, then settles as
  // it decays). effP can exceed 1; computeDrama clamps internally.
  s->dmod = computeDrama((float)effP, s->drama);

  // Phase accumulators (rate params modulated by drama).
  s->shimmer_phase += (double)fdt * (double)s->shimmer_rate_hz * (double)s->dmod.shimmer;
  s->kh_phase      += (double)fdt * (double)s->kh_rate_hz      * (double)s->dmod.kh;
  s->crackle_phase += (double)fdt * (double)s->crackle_rate_hz * (double)s->dmod.crackle_hz;

  // Mach-disk position: pushes downstream with over-expansion.
  float pr = 1.0f + (float)effP * 2.0f * (0.5f + s->mixture);
  s->mach_disk_x = clampf(0.05f + 0.22f * std::sqrt(std::max(pr - 1.0f, 0.0f)), 0.02f, 0.85f);

  // Sporadic spray off the rim while firing — occasional sparks, not a steady
  // fountain. Poisson-ish: accumulate, release one at a time with a gate.
  if (firing && s->spark_rate > 0.0f) {
    s->spark_accum += (double)fdt * (double)s->spark_rate;
    while (s->spark_accum >= 1.0) {
      s->spark_accum -= 1.0;
      if (lcg_unit(s->spawn_rng) < 0.8f) spawn_sparks(*s, 1);  // jittered timing
    }
  }

  // Integrate sparks in 3D (ballistic, gravity, drag, decay). Note: world +y
  // projects to screen-DOWN, so gravity ADDS to vy to fall down on screen.
  for (int i = 0; i < MAX_SPARKS; i++) {
    CpuSpark& sp = s->sparks[i];
    if (!sp.active) continue;
    sp.life -= fdt;
    if (sp.life <= 0.0f) { sp.active = false; continue; }
    sp.px += sp.vx * fdt;
    sp.py += sp.vy * fdt;
    sp.pz += sp.vz * fdt;
    sp.vy += 0.7f * fdt;                // gravity (screen-down) → nice arc
    float drag = 1.0f - 0.25f * fdt;    // low drag → ballistic, not floaty
    sp.vx *= drag; sp.vy *= drag; sp.vz *= drag;
    if (sp.px > 2.5f || sp.py > 1.6f || sp.py < -1.6f) sp.active = false;
  }
}

void on_state_patched(void* self, int n, const char* pb, const int* off,
                      const int* len, const int* ops) {
  auto* s = static_cast<State*>(self);
  if (!s) return;
  for (int i = 0; i < n; i++) {
    const char* path = pb + off[i];
    int plen = len[i];
    if (ops[i] != state::PatchReplace) continue;

    if (state::pathIs(path, plen, "ignition")) {
      bool v = state::patchFloat(i) != 0.0f;
      if (v && !s->ign_prev) {                 // rising edge → light up
        s->overshoot += (double)clampf(s->startup_overshoot, 0.0f, 1.0f);
        spawn_sparks(*s, (int)(s->spark_amount * MAX_SPARKS * 0.6f));  // ignition burst
      }
      s->ignition = v;
      s->ign_prev = v;
    }
    else if (state::pathIs(path, plen, "throttle"))         s->throttle         = state::patchFloat(i);
    else if (state::pathIs(path, plen, "mixture"))          s->mixture          = state::patchFloat(i);
    else if (state::pathIs(path, plen, "intensity"))        s->intensity        = state::patchFloat(i);
    else if (state::pathIs(path, plen, "drama"))            s->drama            = state::patchFloat(i);
    else if (state::pathIs(path, plen, "spool_time"))       s->spool_time       = state::patchFloat(i);
    else if (state::pathIs(path, plen, "startup_overshoot"))s->startup_overshoot= state::patchFloat(i);
    else if (state::pathIs(path, plen, "overshoot_time"))   s->overshoot_time   = state::patchFloat(i);
    else if (state::pathIs(path, plen, "centerline_y"))     s->centerline_y     = state::patchFloat(i);
    else if (state::pathIs(path, plen, "nozzle_radius"))    s->nozzle_radius    = state::patchFloat(i);
    else if (state::pathIs(path, plen, "spread"))           s->spread           = state::patchFloat(i);
    else if (state::pathIs(path, plen, "length_scale"))     s->length_scale     = state::patchFloat(i);
    else if (state::pathIs(path, plen, "core_brightness"))  s->core_brightness  = state::patchFloat(i);
    else if (state::pathIs(path, plen, "radial_sharpness")) s->radial_sharpness = state::patchFloat(i);
    else if (state::pathIs(path, plen, "diamond_amp"))      s->diamond_amp      = state::patchFloat(i);
    else if (state::pathIs(path, plen, "diamond_spacing"))  s->diamond_spacing  = state::patchFloat(i);
    else if (state::pathIs(path, plen, "mach_disk_amp"))    s->mach_disk_amp    = state::patchFloat(i);
    else if (state::pathIs(path, plen, "core_length"))      s->core_length      = state::patchFloat(i);
    else if (state::pathIs(path, plen, "shear_turbulence")) s->shear_turbulence = state::patchFloat(i);
    else if (state::pathIs(path, plen, "shear_scale"))      s->shear_scale      = state::patchFloat(i);
    else if (state::pathIs(path, plen, "crackle"))          s->crackle          = state::patchFloat(i);
    else if (state::pathIs(path, plen, "shimmer_rate_hz"))  s->shimmer_rate_hz  = state::patchFloat(i);
    else if (state::pathIs(path, plen, "kh_rate_hz"))       s->kh_rate_hz       = state::patchFloat(i);
    else if (state::pathIs(path, plen, "crackle_rate_hz"))  s->crackle_rate_hz  = state::patchFloat(i);
    else if (state::pathIs(path, plen, "zoom"))             s->zoom             = state::patchFloat(i);
    else if (state::pathIs(path, plen, "propagation"))      s->propagation      = state::patchFloat(i);
    else if (state::pathIs(path, plen, "substeps"))         s->substeps         = (int)state::patchFloat(i);
    else if (state::pathIs(path, plen, "motion_scale"))     s->motion_scale     = state::patchFloat(i);
    else if (state::pathIs(path, plen, "spark_amount"))     s->spark_amount     = state::patchFloat(i);
    else if (state::pathIs(path, plen, "spark_rate"))       s->spark_rate       = state::patchFloat(i);
    else if (state::pathIs(path, plen, "spark_scale"))      s->spark_scale      = state::patchFloat(i);
    else if (state::pathIs(path, plen, "spark_speed"))      s->spark_speed      = state::patchFloat(i);
    else if (state::pathIs(path, plen, "seed")) {
      int v = (int)state::patchFloat(i);
      if (v != s->seed) { s->seed = v; s->spawn_rng = (uint32_t)v ^ 0xB16B00B5u; }
    }
    else if (state::pathIs(path, plen, "debug_show_axis"))  s->debug_show_axis  = state::patchFloat(i) != 0.0f;
  }
}

void render(void* self, int vp_w, int vp_h) {
  auto* s = static_cast<State*>(self);
  if (!s || !s->initialized || vp_w <= 0 || vp_h <= 0) return;
  auto in  = gpu::Device::textureForField("tex_in");
  auto out = gpu::Device::textureForField("tex_out");
  if (!in.valid() || !out.valid()) return;

  float fdt = (float)host::deltaTime();
  if (fdt <= 0.0f) fdt = 1.0f / 60.0f;
  if (fdt > 0.1f)  fdt = 0.1f;

  double effP = s->chamberP + s->overshoot;
  if (effP < 0.0) effP = 0.0;

  // --- Stage 1: 1D axial solver. ---
  {
    SimUniforms su = {};
    su.dt = fdt;
    su.substeps = (uint32_t)(s->substeps < 1 ? 1 : (s->substeps > 256 ? 256 : s->substeps));
    su.W = (uint32_t)NUM_CELLS;
    su.dx = 1.0f / (float)(NUM_CELLS - 1);
    su.chamberP = (float)effP;
    // Supersonic exhaust — fast flow so the plume establishes in ~2 frames
    // (it expands nearly instantly on light-up, not over a slow transit).
    su.exitVel = 8.0f + 26.0f * (float)effP;                 // canvas-uv/sec
    su.pressureRatio = 1.0f + (float)effP * 2.0f * (0.5f + s->mixture);
    su.litTarget = s->ignition ? 1.0f : 0.0f;
    // propagation maps to wavespeed (cells/frame). High → pressure crosses
    // the jet structure in <1 frame.
    su.wavespeed = (20.0f + 160.0f * clampf(s->propagation, 0.0f, 1.0f));
    // Scale growth with flow speed so the breakdown distance is speed-
    // independent (steady-state m(x) ≈ growth·x/u). core_length is the
    // performer-facing inverse: a longer white potential core ⇔ slower
    // maturity growth, so the white→blue handoff moves downstream.
    su.maturityGrowth = su.exitVel * 4.0f / clampf(s->core_length, 0.2f, 5.0f);
    // Length-relative decay: plume length ≈ length_scale regardless of how
    // fast the flow fills it (b ~ exp(-x/length_scale) since coreDecay = u/L).
    su.coreDecay = su.exitVel / clampf(s->length_scale * s->dmod.length, 0.1f, 3.0f);
    su.flameSpeed = 30.0f;
    su.diamondSpacing = clampf(s->diamond_spacing * s->dmod.spacing, 0.005f, 0.5f);
    su.velRelax = 0.12f;
    s->sim_uniform_buf.writeOne(su);

    auto cp = gpu::ComputePass::begin();
    cp.setPSO(s_pso_sim);
    cp.setBuffer(s->cell_buf, 0);
    cp.setBuffer(s->sim_uniform_buf, 1);
    cp.dispatch(1, 1, 1);            // single workgroup; substeps loop inside
    cp.end();
  }

  // --- Pack sparks: project 3D world → screen (jet-uv space, un-zoomed; the
  //     color shader applies zoom). A slight 3/4 view turns the rim circle
  //     into a thin tilted ellipse and gives near/far perspective. ---
  const float P_AX = 0.55f;   // downstream world → uv-x
  const float P_AY = 1.00f;   // vertical world → uv-y (rim ≈ nozzle mouth)
  const float P_BX = 0.10f;   // depth → uv-x (gives the rim its ellipse width)
  const float P_BY = -0.12f;  // depth → uv-y (tilt — viewed slightly off-axis)
  const float P_PERSP = 0.5f; // near sparks (pz>0) render larger/brighter
  float cy = clampf(s->centerline_y, 0.0f, 1.0f);
  GpuSpark gs[MAX_SPARKS] = {};
  for (int i = 0; i < MAX_SPARKS; i++) {
    const CpuSpark& sp = s->sparks[i];
    if (!sp.active || sp.life <= 0.0f) { gs[i].life = 0.0f; continue; }
    float persp = clampf(1.0f + sp.pz * P_PERSP, 0.25f, 2.5f);
    gs[i].x = sp.px * P_AX + sp.pz * P_BX;          // nozzle anchored at x=0
    gs[i].y = cy + sp.py * P_AY + sp.pz * P_BY;
    // Screen-space velocity (project the world velocity the same way) — the
    // shader streaks the spark along this, so an arcing trajectory visibly
    // rotates the spark as it flies.
    gs[i].vx = sp.vx * P_AX + sp.vz * P_BX;
    gs[i].vy = sp.vy * P_AY + sp.vz * P_BY;
    gs[i].size = sp.size * persp * clampf(s->spark_scale, 0.05f, 10.0f);
    gs[i].life = clampf(sp.life / sp.max_life, 0.0f, 1.0f) * sp.bright
               * (0.5f + 0.5f * persp);
  }
  s->spark_buf.writeBytes(gs, (int)sizeof(gs));

  // --- Shared color/motion uniforms. ---
  ColorUniforms u = {};
  u.intensity = clampf(s->intensity, 0.0f, 8.0f);
  u.centerline_y = clampf(s->centerline_y, 0.0f, 1.0f);
  u.nozzle_radius = clampf(s->nozzle_radius, 0.001f, 0.3f);
  u.spread = clampf(s->spread * s->dmod.spread, 0.0f, 2.0f);
  // radial_sharpness below ~3 grows a large skirt that lets the noise layers
  // flash and hijack the character. Soft-knee floor the DRAMA modulation at 3
  // (or the user's base, if they deliberately chose lower) so the one-knob
  // sweep can't drag it into skirt territory. The lift is drama-scaled, so
  // drama=0 is exact passthrough.
  {
    float base  = s->radial_sharpness;
    float raw   = base * s->dmod.sharp;
    float floor = base < 3.0f ? base : 3.0f;
    float lifted = softFloor(raw, floor, 1.2f);
    float eff   = raw + (lifted - raw) * clampf(s->drama, 0.0f, 1.0f);
    u.radial_sharpness = clampf(eff, 0.5f, 32.0f);
  }
  u.diamond_amp = clampf(s->diamond_amp * s->dmod.diamamp, 0.0f, 2.0f);
  u.mach_disk_x = s->mach_disk_x;
  u.mach_disk_amp = clampf(s->mach_disk_amp * s->dmod.machdisk, 0.0f, 4.0f);
  u.mach_disk_width = 0.03f;
  u.shimmer_phase = (float)((s->shimmer_phase - std::floor(s->shimmer_phase)) * 6.28318530718);
  u.kh_amp = clampf(s->shear_turbulence * s->dmod.shear, 0.0f, 2.0f);
  u.kh_scale = clampf(s->shear_scale, 1.0f, 64.0f);
  u.kh_phase = (float)s->kh_phase;
  u.crackle_amp = clampf(s->crackle * s->dmod.crackle, 0.0f, 1.0f);
  u.crackle_phase = (float)s->crackle_phase;
  u.mixture = clampf(s->mixture, 0.0f, 1.0f);
  u.zoom = clampf(s->zoom, 1.0f, 16.0f);
  u.aspect = (float)vp_w / (float)vp_h;
  u.core_brightness = clampf(s->core_brightness * s->dmod.bright, 0.0f, 8.0f);
  u.cell_count = (uint32_t)NUM_CELLS;
  u.spark_count = (uint32_t)MAX_SPARKS;
  u.debug_show_axis = s->debug_show_axis ? 1u : 0u;
  // [0,1] slider maps to an effective [0,0.025] — the flow is supersonic, so
  // even a tiny scale is a strong streak; this gives usable resolution.
  u.motion_scale = clampf(s->motion_scale, 0.0f, 1.0f) * 0.025f;
  s->color_uniform_buf.writeOne(u);

  // --- Stage 2: color synthesis. ---
  {
    auto cp = gpu::ComputePass::begin();
    cp.setPSO(s_pso_color);
    cp.setTexture(in,  0, 0);
    cp.setTexture(out, 1, 1);
    cp.setBuffer(s->color_uniform_buf, 2);
    cp.setBuffer(s->cell_buf, 3);
    cp.setBuffer(s->spark_buf, 4);
    cp.dispatch((vp_w + 7) / 8, (vp_h + 7) / 8);
    cp.end();
  }

  // --- Stage 2b: motion emission (skip with no downstream consumer). ---
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
        cp.setBuffer(s->color_uniform_buf, 2);
        cp.setBuffer(s->cell_buf, 3);
        cp.dispatch((vp_w + 7) / 8, (vp_h + 7) / 8);
        cp.end();
      }
    }
  }

  gpu::Device::submit();
}


} // namespace side_jet
