/*
 * source.sdf.helio_field — simulation-centric SDF provider: magnetic
 * field lines and solar storms on a sphere.
 *
 * Where plume_field SCULPTS its field, this effect SIMULATES one: a 2D
 * MHD-lite system on an octahedral map — a tangent velocity field
 * (semi-Lagrangian advection, vorticity-confined eddies, granulation
 * stirring, zonal differential rotation) coupled to a magnetic scalar
 * potential A whose contours are the field lines (frozen-in flux). The
 * Lorentz tension −∇²A·∇A combs the flow into filaments and current
 * sheets; the differential rotation slowly winds the lines up. The
 * surface height is DERIVED from the live state each frame: ridges ride
 * the field lines (constant spatial width, so the Lipschitz bound stays
 * analytic no matter how hard the sim bunches the contours), published
 * on the `sdf_field` rail for any SDF renderer downstream (e.g. Plume).
 *
 * Storms (the next milestone) will add an excitable layer that ignites
 * where lines kink too hard, propagates along them, and releases the
 * kink by locally diffusing A — self-organized criticality, stable →
 * edge-of-chaos → self-resonant on one knob. The dynamics pass already
 * carries the reconnection hook (recon × heat).
 *
 * Video is untouched: tex_in passes through to tex_out (clear when
 * unwired); the sim only runs while something consumes the rail.
 */

#include <gpu.h>
#include <host.h>
#include <effect_sdf_field.h>

#include <cmath>
#include <cstdint>

#include "../plume/field_gen.h"
#include "helio_field_shaders.h"

namespace helio_field {

constexpr int kSimRes = 512;      // sim maps + published shell (full)
constexpr int kCoarseRes = 256;   // bake source shell
constexpr int kDustPool = 131072; // dust particle slots (≤ rail kDustMax)

struct DynUniforms {
  float dt, reset, sim_res, seed;
  float rot_rate, rot_relax, stir_gain, stir_phase;
  float mag_gain, conf_gain, drag, vmax;
  float sim_eps, recon, emerge, emerge_phase;
  float resist, force_eps, resist_w, visc;
};
static_assert(sizeof(DynUniforms) == 80, "DynUniforms layout mismatch");

struct HShellUniforms {
  float res, amp, line_k, line_w;
  float base_floor, heat_gain, sim_eps, ga_cap;
  float storm_amp, dust_amp, dust_gain, _pad2;
};
static_assert(sizeof(HShellUniforms) == 48, "HShellUniforms layout mismatch");

struct DustUniforms {
  float dt, reset, sim_res, seed;
  float feed, kill, diff, gs;
  float eps, line_kill, nucleate, drift;
  float gate_eps, _p0, _p1, _p2;
};
static_assert(sizeof(DustUniforms) == 64, "DustUniforms layout mismatch");

struct DustSimUniforms {
  float count, seed, R, lift;
  float size, thresh, dt, reset;
  float tumble, life0, _p0, _p1;
};
static_assert(sizeof(DustSimUniforms) == 48, "DustSimUniforms layout mismatch");

struct DustAccumUniforms { float count, stride, gain, _p2; };
static_assert(sizeof(DustAccumUniforms) == 16, "DustAccumUniforms mismatch");

struct DustFoldUniforms { float norm, _p0, _p1, _p2; };
static_assert(sizeof(DustFoldUniforms) == 16, "DustFoldUniforms mismatch");

struct AccumClearUniforms { float count, ones, _p0, _p1; };
static_assert(sizeof(AccumClearUniforms) == 16, "AccumClearUniforms mismatch");

struct StormUniforms {
  float dt, thresh, prop, burn;
  float cool, charge, recover, kink_gain;
  float force_eps, sim_res, reset, _pad0;
};
static_assert(sizeof(StormUniforms) == 48, "StormUniforms layout mismatch");

static gpu::ComputePSO s_pso_prefill;
static gpu::ComputePSO s_pso_dynamics;
static gpu::ComputePSO s_pso_storm;
static gpu::ComputePSO s_pso_dust;
static gpu::ComputePSO s_pso_dust_sim;
static gpu::ComputePSO s_pso_dust_accum;
static gpu::ComputePSO s_pso_dust_fold;
static gpu::ComputePSO s_pso_accum_clear;
static gpu::ComputePSO s_pso_shell;

struct State {
  bool initialized = false;
  fx::sdf_field::Publisher rail_pub;

  gpu::Texture dyn[2];          // RGBA32F (vel.xyz, spare)
  gpu::Texture aux[2];          // RGBA32F (A, 0, 0, 0)
  gpu::Texture storm[2];        // RGBA16F (u, v, heat, 0)
  gpu::Texture dust[2];         // RGBA16F (a, b, 0, 0) — granule chemistry
  gpu::Texture shell_full;      // RGBA16F 512² (h, crest)
  gpu::Texture shell_coarse;    // RGBA16F 256²
  gpu::Texture sdf_vol;         // RGBA16F 128³ (baked, .a = 0)
  gpu::Texture sdf_vol_pub;     // RGBA16F 128³ (bake + dust density .a)
  gpu::Buffer dust_parts;       // kDustPool × 2 float4 (rail dust layout)
  gpu::Buffer dust_state;       // kDustPool × float4 (age, life, salt, hover)
  gpu::Buffer dust_accum;       // 128³ uints (fixed-point density counts)
  gpu::Buffer ub_dyn, ub_storm, ub_dust, ub_dust_sim, ub_dust_accum,
              ub_dust_fold, ub_accum_clear, ub_shell_full,
              ub_shell_coarse, ub_bake;
  gpu::Sampler samp;

  int ping = 0;                 // dyn/aux index written LAST frame
  int st_ping = 0;              // storm index written LAST frame
  int du_ping = 0;              // dust index written LAST frame
  int dust_hwm = 0;             // pool slots ever initialized (grow = reseed)
  bool reset_pending = true;    // write initial conditions on next step
  double stir_phase = 0.0;
  double emerge_phase = 0.0;
  float last_dt = 1.0f / 60.0f;

  // Param mirrors.
  float radius = 0.5f;
  float variation = 0.0f;
  float sim_rate = 0.5f;
  float rotation = 0.5f;
  float stir = 0.5f;
  float eddies = 0.5f;
  float magnet = 0.6f;
  float relief = 0.5f;
  float line_scale = 0.5f;
  float granules = 0.5f;
  float grain_size = 0.5f;
  float dust_amt = 0.35f;
  float dust_size = 0.5f;
  float excite = 0.5f;
  float storm_h = 0.6f;
  float glow = 0.7f;
  float calm = 0.5f;

  // Master speed: 0 freezes the sun, 1 runs it hot.
  float rateScale() const { return 3.0f * sim_rate * sim_rate; }
};

void module_init() {
  auto schema = state::Schema()
      .helpField("intro",
        "## Helio Field\n"
        "A simulated sun published on the `sdf_field` rail: a fluid on "
        "the sphere carries a magnetic field whose lines you SEE — the "
        "surface ridges ARE the field lines, combed and bunched by the "
        "flow, wound up by differential rotation, stirred into eddies. "
        "Wire it into an SDF renderer downstream (e.g. **Plume**) to "
        "light and render it.\n\n"
        "Nothing here is keyframed: the lines move because the fluid "
        "moves them, and the STORMS self-ignite where the rotation has "
        "wound a line past its breaking kink — burning along the line "
        "as a glowing extruded curtain until the release (reconnection) "
        "has erased the kink that lit them.\n\n"
        "The video input passes through untouched.")
      .group("body", "Body")
      .groupHelp(
          "*Radius* is the body. *Variation* reseeds the initial "
          "magnetic field — a different sun.")
      .floatField("radius", 0.5f, 0.f, 1.f, state::PrimaryInput)
          .label("Radius", "Rad")
      .floatField("variation", 0.0f, 0.f, 1.f, state::PrimaryInput)
          .label("Variation", "Var")
      .group("flow", "Flow")
      .groupHelp(
          "The fluid the lines are frozen into. *Sim Rate* is the "
          "master clock (0 freezes everything). *Rotation* is the "
          "differential spin — equator faster than the poles — that "
          "shears and winds the lines. *Stir* is small-scale "
          "granulation churn. *Eddies* sharpens vortices against the "
          "advection's smearing. *Magnetism* is the line tension: how "
          "hard kinked lines pull straight and comb the flow into "
          "filaments.")
      .floatField("sim_rate", 0.5f, 0.f, 1.f, state::PrimaryInput)
          .label("Sim Rate", "Rate")
      .floatField("rotation", 0.5f, 0.f, 1.f, state::PrimaryInput)
          .label("Rotation", "Rot")
      .floatField("stir", 0.5f, 0.f, 1.f, state::PrimaryInput)
          .label("Stir", "Stir")
      .floatField("eddies", 0.5f, 0.f, 1.f, state::PrimaryInput)
          .label("Eddies", "Eddy")
      .floatField("magnet", 0.6f, 0.f, 1.f, state::PrimaryInput)
          .label("Magnetism", "Mag")
      .group("lines", "Lines")
      .groupHelp(
          "How the field lines read as relief. *Relief* is the ridge "
          "height; *Line Scale* the line density (contour pitch of the "
          "magnetic potential).")
      .floatField("relief", 0.5f, 0.f, 1.f, state::PrimaryInput)
          .label("Relief", "Rel")
      .floatField("line_scale", 0.5f, 0.f, 1.f, state::PrimaryInput)
          .label("Line Scale", "Scale")
      .group("granules", "Granules")
      .groupHelp(
          "Detail for the quiet flatlands between the lines: a granule "
          "chemistry riding the same fluid — discrete spots that space "
          "themselves out, curl along the eddies, and starve where the "
          "field is strong (granulation lives only in the quiet sun). "
          "Purely a surface detail: it never feeds back into the lines "
          "or the storms. *Granules* is the bump height, *Grain Size* "
          "the spot scale.")
      .floatField("granules", 0.5f, 0.f, 1.f, state::PrimaryInput)
          .label("Granules", "Grain")
      .floatField("grain_size", 0.5f, 0.f, 1.f, state::PrimaryInput)
          .label("Grain Size", "Size")
      .group("dust", "Dust")
      .groupHelp(
          "Glinting motes hovering just off the surface — dust riding "
          "the granulation. Each mote is born on a live granule, rides "
          "the same fluid as everything else (watch them stream around "
          "the eddies), tumbles so its facet twinkles in the sun, and "
          "dies where its granule starves near a strong line. Published "
          "on the rail's dust channel: the downstream renderer (e.g. "
          "Plume) splats each mote SHARP with exact depth, while their "
          "aggregate density scatters fog and dims the sun softly. "
          "*Dust* is how many, *Dust Size* the mote size.")
      .floatField("dust", 0.35f, 0.f, 1.f, state::PrimaryInput)
          .label("Dust", "Dust")
      .floatField("dust_size", 0.5f, 0.f, 1.f, state::PrimaryInput)
          .label("Dust Size", "DSize")
      .group("storms", "Storms")
      .groupHelp(
          "Storms are not triggered — they SELF-IGNITE where the sim "
          "kinks a field line harder than it is strong, then burn ALONG "
          "the line as a tall glowing curtain until they've released "
          "(reconnected) the kink that lit them. *Excitability* is the "
          "criticality dial: low = quiet sun, the middle = storms "
          "firing stochastically every few seconds wherever the "
          "rotation has wound the field tight, high = self-resonant, "
          "the storms never stop. *Storm Height* is the curtain "
          "extrusion, *Glow* the afterglow left in the crest channel, "
          "*Calm* the dead time before a burned region can re-ignite.")
      .floatField("excite", 0.5f, 0.f, 1.f, state::PrimaryInput)
          .label("Excitability", "Excit")
      .floatField("storm_h", 0.6f, 0.f, 1.f, state::PrimaryInput)
          .label("Storm Height", "Storm")
      .floatField("glow", 0.7f, 0.f, 1.f, state::PrimaryInput)
          .label("Glow", "Glow")
      .floatField("calm", 0.5f, 0.f, 1.f, state::PrimaryInput)
          .label("Calm", "Calm")
      // --- I/O ---
      .textureField("tex_in", state::PrimaryInput)
      .textureField("tex_out", state::PrimaryOutput);
  fx::sdf_field::declare(schema, state::SecondaryOutput);
  schema.capability(state::Capability::Generator);
  state::init("source.sdf.helio_field", {1, 0, 0}, schema);

  if (gpu::Device::backend() == gpu::Backend::None) return;

  // Bake shader (shell → SDF volume) comes from the shared plume
  // sculptor; we reuse its PSO with our own shell/volume textures.
  plume_gen::moduleInit();
  state::registerShaderSPV("helio_field_prefill", HELIO_FIELD_PREFILL_SPV,
                           HELIO_FIELD_PREFILL_SPV_SIZE);
  state::registerShaderSPV("helio_field_dynamics", HELIO_FIELD_DYNAMICS_SPV,
                           HELIO_FIELD_DYNAMICS_SPV_SIZE,
                           "rgba32float", "write");
  state::registerShaderSPV("helio_field_storm", HELIO_FIELD_STORM_SPV,
                           HELIO_FIELD_STORM_SPV_SIZE,
                           "rgba16float", "write");
  state::registerShaderSPV("helio_field_dust", HELIO_FIELD_DUST_SPV,
                           HELIO_FIELD_DUST_SPV_SIZE,
                           "rgba16float", "write");
  state::registerShaderSPV("helio_field_dust_sim", HELIO_FIELD_DUST_SIM_SPV,
                           HELIO_FIELD_DUST_SIM_SPV_SIZE);
  state::registerShaderSPV("helio_field_dust_accum", HELIO_FIELD_DUST_ACCUM_SPV,
                           HELIO_FIELD_DUST_ACCUM_SPV_SIZE);
  state::registerShaderSPV("helio_field_dust_fold", HELIO_FIELD_DUST_FOLD_SPV,
                           HELIO_FIELD_DUST_FOLD_SPV_SIZE,
                           "rgba16float", "write");
  // The plume dust splat's clear pass, re-registered under this effect
  // (same SPV — a uint-buffer fill is a uint-buffer fill).
  state::registerShaderSPV("helio_field_accum_clear",
                           HELIO_FIELD_ACCUM_CLEAR_SPV,
                           HELIO_FIELD_ACCUM_CLEAR_SPV_SIZE);
  state::registerShaderSPV("helio_field_shell", HELIO_FIELD_SHELL_SPV,
                           HELIO_FIELD_SHELL_SPV_SIZE,
                           "rgba16float", "write");
  auto cs_prefill = gpu::Device::createShaderModuleByName("helio_field_prefill");
  auto cs_dyn = gpu::Device::createShaderModuleByName("helio_field_dynamics");
  auto cs_storm = gpu::Device::createShaderModuleByName("helio_field_storm");
  auto cs_dust = gpu::Device::createShaderModuleByName("helio_field_dust");
  auto cs_dsim = gpu::Device::createShaderModuleByName("helio_field_dust_sim");
  auto cs_dacc = gpu::Device::createShaderModuleByName("helio_field_dust_accum");
  auto cs_dfold = gpu::Device::createShaderModuleByName("helio_field_dust_fold");
  auto cs_aclr = gpu::Device::createShaderModuleByName("helio_field_accum_clear");
  auto cs_shell = gpu::Device::createShaderModuleByName("helio_field_shell");
  if (!cs_prefill || !cs_dyn || !cs_storm || !cs_dust || !cs_dsim ||
      !cs_dacc || !cs_dfold || !cs_aclr || !cs_shell) return;
  s_pso_prefill = gpu::Device::createComputePSO(cs_prefill, "main",
      gpu::Bindings()
          .tex2d(0)
          .storageTex2d(1));
  s_pso_dynamics = gpu::Device::createComputePSO(cs_dyn, "main",
      gpu::Bindings()
          .tex2d(0)          // dyn (previous)
          .tex2d(1)          // aux (previous)
          .sampler(2)
          .storageTex2d(3, gpu::TextureFormat::RGBA32F)  // dyn (next)
          .storageTex2d(4, gpu::TextureFormat::RGBA32F)  // aux (next)
          .uniform(5)
          .tex2d(6));        // storm (previous — reconnection gate)
  s_pso_storm = gpu::Device::createComputePSO(cs_storm, "main",
      gpu::Bindings()
          .tex2d(0)          // aux (current)
          .tex2d(1)          // storm (previous)
          .sampler(2)
          .storageTex2d(3, gpu::TextureFormat::RGBA16F)  // storm (next)
          .uniform(4));
  s_pso_dust = gpu::Device::createComputePSO(cs_dust, "main",
      gpu::Bindings()
          .tex2d(0)          // dyn (current — velocity)
          .tex2d(1)          // aux (current — gate)
          .tex2d(2)          // dust (previous)
          .sampler(3)
          .storageTex2d(4, gpu::TextureFormat::RGBA16F)  // dust (next)
          .uniform(5));
  s_pso_dust_sim = gpu::Device::createComputePSO(cs_dsim, "main",
      gpu::Bindings()
          .tex2d(0)          // dust chemistry (current)
          .tex2d(1)          // shell_full (hover height)
          .tex2d(2)          // velocity (current)
          .sampler(3)
          .storageRW(4)      // rail particle buffer
          .storageRW(5)      // private life-cycle state
          .uniform(6));
  s_pso_dust_accum = gpu::Device::createComputePSO(cs_dacc, "main",
      gpu::Bindings()
          .storage(0)        // particles
          .storageRW(1)      // count volume (atomics)
          .uniform(2));
  s_pso_dust_fold = gpu::Device::createComputePSO(cs_dfold, "main",
      gpu::Bindings()
          .tex3d(0)          // baked volume
          .storage(1)        // count volume
          .storageTex3d(2, gpu::TextureFormat::RGBA16F)
          .uniform(3));
  s_pso_accum_clear = gpu::Device::createComputePSO(cs_aclr, "main",
      gpu::Bindings()
          .storageRW(0)
          .uniform(1));
  s_pso_shell = gpu::Device::createComputePSO(cs_shell, "main",
      gpu::Bindings()
          .tex2d(0)          // aux (current)
          .sampler(1)
          .storageTex2d(2, gpu::TextureFormat::RGBA16F)  // shell target
          .uniform(3)
          .tex2d(4)          // storm (current)
          .tex2d(5));        // dust (current)

  state::log("helio_field: module initialized");
}

void* create() {
  auto* s = new State();
  s->ub_dyn = gpu::Device::createBuffer(sizeof(DynUniforms),
                                        gpu::BufferUsage::Uniform);
  s->ub_storm = gpu::Device::createBuffer(sizeof(StormUniforms),
                                          gpu::BufferUsage::Uniform);
  s->ub_dust = gpu::Device::createBuffer(sizeof(DustUniforms),
                                         gpu::BufferUsage::Uniform);
  s->ub_dust_sim = gpu::Device::createBuffer(sizeof(DustSimUniforms),
                                             gpu::BufferUsage::Uniform);
  s->ub_dust_accum = gpu::Device::createBuffer(sizeof(DustAccumUniforms),
                                               gpu::BufferUsage::Uniform);
  s->ub_dust_fold = gpu::Device::createBuffer(sizeof(DustFoldUniforms),
                                              gpu::BufferUsage::Uniform);
  s->ub_accum_clear = gpu::Device::createBuffer(sizeof(AccumClearUniforms),
                                                gpu::BufferUsage::Uniform);
  s->ub_shell_full = gpu::Device::createBuffer(sizeof(HShellUniforms),
                                               gpu::BufferUsage::Uniform);
  s->ub_shell_coarse = gpu::Device::createBuffer(sizeof(HShellUniforms),
                                                 gpu::BufferUsage::Uniform);
  s->ub_bake = gpu::Device::createBuffer(sizeof(plume_gen::BakeUniforms),
                                         gpu::BufferUsage::Uniform);
  s->samp = gpu::Device::createSampler(gpu::FilterMode::Linear,
                                       gpu::AddressMode::ClampToEdge);
  return s;
}

void destroy(void* self) {
  auto* s = static_cast<State*>(self);
  if (!s) return;
  s->dyn[0].release();
  s->dyn[1].release();
  s->aux[0].release();
  s->aux[1].release();
  s->storm[0].release();
  s->storm[1].release();
  s->dust[0].release();
  s->dust[1].release();
  s->shell_full.release();
  s->shell_coarse.release();
  s->sdf_vol.release();
  s->sdf_vol_pub.release();
  s->dust_parts.release();
  s->dust_state.release();
  s->dust_accum.release();
  s->ub_dyn.release();
  s->ub_storm.release();
  s->ub_dust.release();
  s->ub_dust_sim.release();
  s->ub_dust_accum.release();
  s->ub_dust_fold.release();
  s->ub_accum_clear.release();
  s->ub_shell_full.release();
  s->ub_shell_coarse.release();
  s->ub_bake.release();
  s->samp.release();
  delete s;
}

void init(void* self) {
  auto* s = static_cast<State*>(self);
  if (!s) return;
  s->reset_pending = true;
  s->stir_phase = 0.0;
  s->emerge_phase = 0.0;
  s->ping = 0;
  s->st_ping = 0;
  s->du_ping = 0;
  s->initialized = s->ub_dyn.valid() && s_pso_dynamics.valid() &&
                   s_pso_dust.valid() && s_pso_shell.valid() &&
                   plume_gen::g_pso_bake.valid();
}

void tick(void* self, double dt) {
  auto* s = static_cast<State*>(self);
  if (!s) return;
  if (!(dt > 0.0)) dt = 0.0;
  if (dt > 0.050) dt = 0.050;
  s->last_dt = (float)(dt > 0.0 ? dt : 1.0 / 60.0);
  // Both noise domains drift on the SIM clock so Sim Rate scales
  // everything; emergence drifts far slower than the stirring.
  s->stir_phase += dt * s->rateScale() * 0.35;
  s->emerge_phase += dt * s->rateScale() * 0.05;
}

void on_state_patched(void* self, int n, const char* pb, const int* off,
                      const int* len, const int* ops) {
  auto* s = static_cast<State*>(self);
  if (!s) return;
  for (int i = 0; i < n; i++) {
    if (ops[i] != state::PatchReplace) continue;
    const char* p = pb + off[i];
    const int l = len[i];
    if      (state::pathIs(p, l, "radius"))     s->radius = state::patchFloat(i);
    else if (state::pathIs(p, l, "variation")) {
      float v = state::patchFloat(i);
      if (v != s->variation) s->reset_pending = true;  // reseed = new sun
      s->variation = v;
    }
    else if (state::pathIs(p, l, "sim_rate"))   s->sim_rate = state::patchFloat(i);
    else if (state::pathIs(p, l, "rotation"))   s->rotation = state::patchFloat(i);
    else if (state::pathIs(p, l, "stir"))       s->stir = state::patchFloat(i);
    else if (state::pathIs(p, l, "eddies"))     s->eddies = state::patchFloat(i);
    else if (state::pathIs(p, l, "magnet"))     s->magnet = state::patchFloat(i);
    else if (state::pathIs(p, l, "relief"))     s->relief = state::patchFloat(i);
    else if (state::pathIs(p, l, "line_scale")) s->line_scale = state::patchFloat(i);
    else if (state::pathIs(p, l, "granules"))   s->granules = state::patchFloat(i);
    else if (state::pathIs(p, l, "grain_size")) s->grain_size = state::patchFloat(i);
    else if (state::pathIs(p, l, "dust"))       s->dust_amt = state::patchFloat(i);
    else if (state::pathIs(p, l, "dust_size"))  s->dust_size = state::patchFloat(i);
    else if (state::pathIs(p, l, "excite"))     s->excite = state::patchFloat(i);
    else if (state::pathIs(p, l, "storm_h"))    s->storm_h = state::patchFloat(i);
    else if (state::pathIs(p, l, "glow"))       s->glow = state::patchFloat(i);
    else if (state::pathIs(p, l, "calm"))       s->calm = state::patchFloat(i);
  }
}

// One sim + publish frame. Returns false when resources are unavailable.
static bool runField(State* s) {
  for (int i = 0; i < 2; i++) {
    if (!s->dyn[i].valid())
      s->dyn[i] = gpu::Device::createTexture(kSimRes, kSimRes,
                                             gpu::TextureFormat::RGBA32F);
    if (!s->aux[i].valid())
      s->aux[i] = gpu::Device::createTexture(kSimRes, kSimRes,
                                             gpu::TextureFormat::RGBA32F);
    if (!s->storm[i].valid())
      s->storm[i] = gpu::Device::createTexture(kSimRes, kSimRes,
                                               gpu::TextureFormat::RGBA16F);
    if (!s->dust[i].valid())
      s->dust[i] = gpu::Device::createTexture(kSimRes, kSimRes,
                                              gpu::TextureFormat::RGBA16F);
    if (!s->dyn[i].valid() || !s->aux[i].valid() || !s->storm[i].valid() ||
        !s->dust[i].valid())
      return false;
  }
  if (!s->shell_full.valid())
    s->shell_full = gpu::Device::createTexture(kSimRes, kSimRes,
                                               gpu::TextureFormat::RGBA16F);
  if (!s->shell_coarse.valid())
    s->shell_coarse = gpu::Device::createTexture(kCoarseRes, kCoarseRes,
                                                 gpu::TextureFormat::RGBA16F);
  if (!s->sdf_vol.valid())
    s->sdf_vol = gpu::Device::createTexture3D(plume_gen::kVolRes,
                                              plume_gen::kVolRes,
                                              plume_gen::kVolRes,
                                              gpu::TextureFormat::RGBA16F);
  if (!s->shell_full.valid() || !s->shell_coarse.valid() ||
      !s->sdf_vol.valid()) return false;

  const float dt_sim = s->last_dt * s->rateScale();
  const float R = 0.28f + 0.27f * s->radius;
  float amp = 0.35f * R * s->relief;
  if (R + amp > 0.82f) amp = 0.82f - R;
  // Storm curtains extrude ABOVE the line relief; the pair shares the
  // volume budget (crest sphere must stay inside the grid extent).
  float storm_amp = 0.45f * R * s->storm_h;
  if (R + amp + storm_amp > 0.82f)
    storm_amp = std::fmax(0.0f, 0.82f - R - amp);
  float dust_amp = 0.10f * R * s->granules;
  if (R + amp + storm_amp + dust_amp > 0.82f)
    dust_amp = std::fmax(0.0f, 0.82f - R - amp - storm_amp);
  const float line_k = 2.0f + 10.0f * s->line_scale;
  const float line_w = 0.07f - 0.04f * s->line_scale;  // radians
  // Granule spot radius follows the diffusion ring (about three rings).
  const float dust_eps = (1.5f + 3.0f * s->grain_size) / (float)kSimRes;

  // --- Pass 0: dynamics step (or initial conditions) ---
  DynUniforms du = {};
  du.dt = dt_sim;
  du.reset = s->reset_pending ? 1.0f : 0.0f;
  du.sim_res = (float)kSimRes;
  du.seed = s->variation * 10.0f;
  du.rot_rate = 1.5f * s->rotation;
  du.rot_relax = 0.6f * s->rotation;
  du.stir_gain = 4.0f * s->stir * s->stir;
  du.stir_phase = (float)s->stir_phase;
  // Floored: with ZERO tension the shear folds A into hard-cornered
  // sheets (blocky plateaus). Some tension is what makes it magnetic.
  du.mag_gain = 6.0f * (0.08f + 0.92f * s->magnet);
  du.conf_gain = 3.0f * s->eddies;
  du.drag = 0.3f;
  du.vmax = 1.2f;
  du.sim_eps = 5.0f / (float)kSimRes;
  du.recon = 0.0f;   // storms milestone
  du.emerge = 0.04f;
  du.emerge_phase = (float)s->emerge_phase;
  du.resist = std::fmin(0.25f, 1.2f * dt_sim);
  du.force_eps = 15.0f / (float)kSimRes;
  // Sized so the loading↔dissipation equilibrium kink level sits MID
  // threshold range: that's what gives Excitability true subcritical
  // states (low = never fires, not just "fires rarely") instead of the
  // pure-SOC regime where every threshold converges to the same rate.
  du.resist_w = std::fmin(0.3f, 1.5f * dt_sim);
  du.visc = std::fmin(0.85f, 30.0f * dt_sim);
  // Reconnection efficiency runs INVERSE to Excitability — this is what
  // makes the dial actually span the SOC regimes: the self-organized
  // steady state is set by energy in vs energy released per storm, not
  // by the ignition threshold alone. Low excite: storms fully consume
  // their kink (rare, complete, self-quenching). High excite: storms
  // barely release fuel, so the same kinks re-ignite — self-resonant.
  du.recon = std::fmin(0.2f, 2.0f * dt_sim) *
             (0.15f + 1.0f * (1.0f - s->excite));
  s->ub_dyn.writeOne(du);

  const bool reset = s->reset_pending;
  const int cur = s->ping;
  const int nxt = 1 - cur;
  const int scur = s->st_ping;
  const int snxt = 1 - scur;
  {
    auto cp = gpu::ComputePass::begin();
    cp.setPSO(s_pso_dynamics);
    cp.setTexture(s->dyn[cur], 0, 0);
    cp.setTexture(s->aux[cur], 1, 0);
    cp.setSampler(s->samp, 2);
    cp.setTexture(s->dyn[nxt], 3, 1);
    cp.setTexture(s->aux[nxt], 4, 1);
    cp.setBuffer(s->ub_dyn, 5);
    cp.setTexture(s->storm[scur], 6, 0);
    cp.dispatch(kSimRes / 8, kSimRes / 8);
    cp.end();
  }
  s->ping = nxt;
  s->reset_pending = false;

  // --- Pass 0.5: storm layer (excitable medium over THIS frame's A) ---
  StormUniforms su = {};
  su.dt = dt_sim;
  // Threshold mapping calibrated against the measured kink distribution
  // (background turbulence ~0.2, p90 ~0.5, genuine sheets 1+): excite
  // 0.5 sits at ~p99 (rare discrete storms), 0.9 near p90 (resonant).
  su.thresh = 0.25f + 3.0f * (1.0f - s->excite) * (1.0f - s->excite);
  su.prop = 1.4f;
  su.burn = 8.0f;
  su.cool = 2.5f;
  su.charge = 1.8f;
  su.recover = 0.08f + 1.2f * (1.0f - s->calm);
  su.kink_gain = 3.0f;
  su.force_eps = 15.0f / (float)kSimRes;
  su.sim_res = (float)kSimRes;
  su.reset = reset ? 1.0f : 0.0f;
  s->ub_storm.writeOne(su);
  {
    auto cp = gpu::ComputePass::begin();
    cp.setPSO(s_pso_storm);
    cp.setTexture(s->aux[nxt], 0, 0);
    cp.setTexture(s->storm[scur], 1, 0);
    cp.setSampler(s->samp, 2);
    cp.setTexture(s->storm[snxt], 3, 1);
    cp.setBuffer(s->ub_storm, 4);
    cp.dispatch(kSimRes / 8, kSimRes / 8);
    cp.end();
  }
  s->st_ping = snxt;

  // --- Pass 0.7: granule chemistry (passive — reads vel + A, feeds
  // back into nothing) ---
  DustUniforms uu = {};
  uu.dt = dt_sim;
  uu.reset = reset ? 1.0f : 0.0f;
  uu.sim_res = (float)kSimRes;
  uu.seed = s->variation * 10.0f;
  // Gray-Scott worm/labyrinth corner: self-spacing shapes that GROW and
  // tile whatever quiet space they're given (the spot-only soliton
  // corner stalled at ~8% coverage under the flow's shear — measured).
  uu.feed = 0.046f;
  uu.kill = 0.061f;
  uu.diff = 0.64f;
  // The reaction runs on its own accelerated clock — GS patterns in
  // ~unit steps, so ~70 of them per sim-second forms granules within a
  // few seconds. Clamped at the standard stable step.
  uu.gs = std::fmin(1.0f, 70.0f * dt_sim);
  uu.eps = dust_eps;
  uu.line_kill = 0.05f;
  uu.nucleate = 0.04f;
  uu.drift = (float)s->stir_phase;
  uu.gate_eps = 10.0f / (float)kSimRes;
  s->ub_dust.writeOne(uu);
  const int dcur = s->du_ping;
  const int dnxt = 1 - dcur;
  {
    auto cp = gpu::ComputePass::begin();
    cp.setPSO(s_pso_dust);
    cp.setTexture(s->dyn[nxt], 0, 0);
    cp.setTexture(s->aux[nxt], 1, 0);
    cp.setTexture(s->dust[dcur], 2, 0);
    cp.setSampler(s->samp, 3);
    cp.setTexture(s->dust[dnxt], 4, 1);
    cp.setBuffer(s->ub_dust, 5);
    cp.dispatch(kSimRes / 8, kSimRes / 8);
    cp.end();
  }
  s->du_ping = dnxt;

  // --- Pass 1: shell maps from the fresh state (full + coarse) ---
  HShellUniforms hu = {};
  hu.amp = amp;
  hu.line_k = line_k;
  hu.line_w = line_w;
  hu.base_floor = 0.12f;
  hu.heat_gain = s->glow;
  // Wider gradient step than the dynamics uses: |∇A| feeds the ridge
  // width, and at texel scale bilinear kinks make it noisy — the walls
  // grow serrated "fur". Two texels of smoothing reads clean.
  hu.sim_eps = 10.0f / (float)kSimRes;
  hu.ga_cap = 2.0f;
  hu.storm_amp = storm_amp;
  hu.dust_amp = dust_amp;
  // Soliton-interior b sits around 0.3 — normalize so a mature granule
  // reaches the full bump height.
  hu.dust_gain = 3.5f;
  hu.res = (float)kSimRes;
  s->ub_shell_full.writeOne(hu);
  hu.res = (float)kCoarseRes;
  s->ub_shell_coarse.writeOne(hu);
  {
    auto cp = gpu::ComputePass::begin();
    cp.setPSO(s_pso_shell);
    cp.setTexture(s->aux[nxt], 0, 0);
    cp.setSampler(s->samp, 1);
    cp.setTexture(s->shell_full, 2, 1);
    cp.setBuffer(s->ub_shell_full, 3);
    cp.setTexture(s->storm[snxt], 4, 0);
    cp.setTexture(s->dust[dnxt], 5, 0);
    cp.dispatch(kSimRes / 8, kSimRes / 8);
    cp.end();
  }
  {
    auto cp = gpu::ComputePass::begin();
    cp.setPSO(s_pso_shell);
    cp.setTexture(s->aux[nxt], 0, 0);
    cp.setSampler(s->samp, 1);
    cp.setTexture(s->shell_coarse, 2, 1);
    cp.setBuffer(s->ub_shell_coarse, 3);
    cp.setTexture(s->storm[snxt], 4, 0);
    cp.setTexture(s->dust[dnxt], 5, 0);
    cp.dispatch(kCoarseRes / 8, kCoarseRes / 8);
    cp.end();
  }

  // --- Pass 1.5: dust motes (rail particle channel) — a persistent
  // pool advected by the sim velocity; born on granules, dying where
  // they starve. Pool growth (knob up / fresh buffers) reseeds so no
  // slot ever runs on uninitialized state. ---
  int dust_count = 0;
  if (s->dust_amt > 0.001f && s_pso_dust_sim.valid() &&
      s->ub_dust_sim.valid()) {
    if (!s->dust_parts.valid()) {
      s->dust_parts = gpu::Device::createBuffer(
          (long long)kDustPool * 2 * 16, gpu::BufferUsage::Storage);
      s->dust_hwm = 0;
    }
    if (!s->dust_state.valid()) {
      s->dust_state = gpu::Device::createBuffer(
          (long long)kDustPool * 16, gpu::BufferUsage::Storage);
      s->dust_hwm = 0;
    }
    if (s->dust_parts.valid() && s->dust_state.valid()) {
      // Square taper: the low half of the knob stays sparse. The ceiling
      // sits well under the pool: full-knob dust should read as heavy
      // glitter, not a crust (measured: ~64k motes tile the granule
      // fields solid at 512²).
      dust_count = (int)(49152.0f * s->dust_amt * s->dust_amt);
      dust_count = dust_count / 64 * 64;
    }
  }
  if (dust_count > 0) {
    const bool dust_reset = reset || dust_count > s->dust_hwm;
    if (dust_count > s->dust_hwm) s->dust_hwm = dust_count;
    DustSimUniforms du2 = {};
    du2.count = (float)dust_count;
    du2.seed = s->variation * 10.0f;
    du2.R = R;
    du2.lift = 0.018f;
    // Bottom reaches sub-pixel at real viewport sizes (single-px glints
    // via the splat presence floor) — keep in step with dust_halo's.
    du2.size = 0.0001f + 0.0049f * s->dust_size;
    // Chemistry b tops out ~0.35 (seeded) / ~0.45 (mature): accept from
    // the blob shoulders up so dust tracks granules, not just cores.
    du2.thresh = 0.12f;
    du2.dt = dt_sim;
    du2.reset = dust_reset ? 1.0f : 0.0f;
    du2.tumble = 2.5f;
    du2.life0 = 7.0f;
    s->ub_dust_sim.writeOne(du2);
    auto cp = gpu::ComputePass::begin();
    cp.setPSO(s_pso_dust_sim);
    cp.setTexture(s->dust[dnxt], 0, 0);
    cp.setTexture(s->shell_full, 1, 0);
    cp.setTexture(s->dyn[nxt], 2, 0);
    cp.setSampler(s->samp, 3);
    cp.setBuffer(s->dust_parts, 4);
    cp.setBuffer(s->dust_state, 5);
    cp.setBuffer(s->ub_dust_sim, 6);
    cp.dispatch(dust_count / 64);
    cp.end();
  }

  // --- Pass 2: bake (shared plume sculptor shader) ---
  // Lipschitz: the shell pass renders fixed-spatial-width ridges, so the
  // max slope is (amp + storm curtain)·0.86/line_w regardless of how the
  // sim bunches the lines; 1.5 covers the profile constant + bilinear-A
  // wiggle. Storm u rides the same ridge profile, so it shares the bound.
  // Granule bumps are narrower than the line ridges (radius ~3 rings),
  // so they get their own slope term against their own width.
  const float amp_total = amp + storm_amp + dust_amp;
  const float Rf = std::fmax(R, 0.1f);
  const float lip_true =
      1.0f / (1.0f + 1.5f * (amp + storm_amp) / (line_w * Rf)
                   + 1.5f * dust_amp / (3.0f * dust_eps * Rf));
  const float lip = std::fmax(lip_true, 0.15f);
  plume_gen::BakeUniforms bu = {
      R, lip,
      3.0f * (2.0f * plume_gen::kExt0 / (float)plume_gen::kVolRes), 0.f };
  s->ub_bake.writeOne(bu);
  {
    auto cp = gpu::ComputePass::begin();
    cp.setPSO(plume_gen::g_pso_bake);
    cp.setTexture(s->shell_coarse, 0, 0);
    cp.setSampler(s->samp, 1);
    cp.setTexture(s->sdf_vol, 2, 1);
    cp.setBuffer(s->ub_bake, 3);
    cp.dispatch(plume_gen::kVolRes / 4, plume_gen::kVolRes / 4,
                plume_gen::kVolRes / 4);
    cp.end();
  }

  // --- Pass 2.5: dust density → published grid .a (the soft half of
  // dust: fog scattering + sun extinction read this as an aggregate
  // medium; the motes themselves stay sharp in the consumer's splat).
  // Runs on a COPY of the baked volume — when dust is off the baked
  // volume (with .a = 0) publishes directly and this whole stage is
  // skipped. ---
  bool dust_vol = dust_count > 0 && s_pso_dust_accum.valid() &&
                  s_pso_dust_fold.valid() && s_pso_accum_clear.valid() &&
                  s->ub_dust_accum.valid() && s->ub_dust_fold.valid() &&
                  s->ub_accum_clear.valid();
  if (dust_vol) {
    const int vres = plume_gen::kVolRes;
    const int vcount = vres * vres * vres;
    if (!s->dust_accum.valid())
      s->dust_accum = gpu::Device::createBuffer((long long)vcount * 4,
                                                gpu::BufferUsage::Storage);
    if (!s->sdf_vol_pub.valid())
      s->sdf_vol_pub = gpu::Device::createTexture3D(
          vres, vres, vres, gpu::TextureFormat::RGBA16F);
    dust_vol = s->dust_accum.valid() && s->sdf_vol_pub.valid();
  }
  if (dust_vol) {
    const int vres = plume_gen::kVolRes;
    const int vcount = vres * vres * vres;
    AccumClearUniforms au = { (float)vcount, 0.f, 0.f, 0.f };  // fill zeros
    s->ub_accum_clear.writeOne(au);
    {
      auto cp = gpu::ComputePass::begin();
      cp.setPSO(s_pso_accum_clear);
      cp.setBuffer(s->dust_accum, 0);
      cp.setBuffer(s->ub_accum_clear, 1);
      cp.dispatch((vcount + 63) / 64);
      cp.end();
    }
    // Decimated: the influence field only needs a SAMPLE of the cloud
    // (mean-preserving stride + weight — see dust_accum.hlsl). Dense
    // clumps otherwise serialize their trilinear atomics on a handful
    // of voxels.
    const int kAccumCap = 24576;
    const int a_stride = (dust_count + kAccumCap - 1) / kAccumCap;
    const int a_count = (dust_count + a_stride - 1) / a_stride;
    DustAccumUniforms cu = { (float)a_count, (float)a_stride,
                             (float)dust_count / (float)a_count, 0.f };
    s->ub_dust_accum.writeOne(cu);
    {
      auto cp = gpu::ComputePass::begin();
      cp.setPSO(s_pso_dust_accum);
      cp.setBuffer(s->dust_parts, 0);
      cp.setBuffer(s->dust_accum, 1);
      cp.setBuffer(s->ub_dust_accum, 2);
      cp.dispatch((a_count + 63) / 64);
      cp.end();
    }
    // Full density at ~10 motes/voxel (each deposits 256 fixed-point).
    // 20 motes saturate a voxel — halved density gain so clump influence
    // (fog scatter, sun dtau) grades softly instead of blobbing at 128³.
    DustFoldUniforms fu = { 1.0f / (256.0f * 20.0f), 0.f, 0.f, 0.f };
    s->ub_dust_fold.writeOne(fu);
    {
      auto cp = gpu::ComputePass::begin();
      cp.setPSO(s_pso_dust_fold);
      cp.setTexture(s->sdf_vol, 0, 0);
      cp.setBuffer(s->dust_accum, 1);
      cp.setTexture(s->sdf_vol_pub, 2, 1);
      cp.setBuffer(s->ub_dust_fold, 3);
      cp.dispatch(vres / 4, vres / 4, vres / 4);
      cp.end();
    }
  }

  // --- Publish ---
  fx::sdf_field::Desc d;
  d.field_class = fx::sdf_field::SphericalHeightmap;
  d.radius = R;
  d.lip = lip;
  d.lip_true = lip_true;
  d.crest_amp = amp_total;
  d.crest_gain = 1.0f;   // the lines ARE the crest
  d.grid_ext = fx::sdf_field::kGridExt;
  d.shell_res = (float)kSimRes;
  d.has_grid = d.has_shell = true;
  d.dust_count = dust_count;
  d.has_dust = s->dust_parts.valid();
  s->rail_pub.publish(d,
                      dust_vol ? s->sdf_vol_pub.id : s->sdf_vol.id,
                      s->shell_full.id,
                      s->dust_parts.valid() ? s->dust_parts.id : -1);
  return true;
}

void render(void* self, int vp_w, int vp_h) {
  auto* s = static_cast<State*>(self);
  if (!s || !s->initialized || vp_w <= 0 || vp_h <= 0) return;

  auto in = gpu::Device::textureForField("tex_in");
  auto out = gpu::Device::textureForField("tex_out");
  if (!out.valid()) return;

  // Simulate + publish only while something consumes the rail.
  if (state::isOutputConnected("sdf_field")) runField(s);

  // Video passthrough (clear when unwired — the field is the output).
  if (in.valid()) {
    auto cp = gpu::ComputePass::begin();
    cp.setPSO(s_pso_prefill);
    cp.setTexture(in, 0, 0);
    cp.setTexture(out, 1, 1);
    cp.dispatch((vp_w + 7) / 8, (vp_h + 7) / 8);
    cp.end();
  } else {
    gpu::Device::clear(out, 0.0f, 0.0f, 0.0f, 0.0f);
  }
  gpu::Device::submit();
}

void on_resolume_param(void* self, long long param_id, double value) {
  (void)self; (void)param_id; (void)value;
}

} // namespace helio_field
