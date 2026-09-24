/*
 * source.sdf.plume_field — the plume sculptor as a standalone SDF
 * provider.
 *
 * Contains the generator half of source.sdf.plume (shared single-source
 * via plume/field_gen.h): the Shape params author a ridged, morphing
 * displacement field on an octahedral shell map, baked into a 128³ SDF
 * volume — and instead of rendering it, this effect publishes it on the
 * `sdf_field` rail (effect_sdf_field.h) for any SDF renderer downstream.
 * First proof of the "SDFs should be able to be anything / providable by
 * other effects" direction: the geometry now travels separately from the
 * camera/light/atmosphere.
 *
 * On top of the shared sculpt, this effect (and only this effect) has a
 * Simulate mode: a tracer population sweeps over the manifold of the
 * displaced sphere in long arcs, curling around the sphere's axis. Each
 * tracer is a valley-cutter (steers into valleys, gently digs along its
 * path) or a ridge-builder (rides crests, builds them up); the zonal
 * flow carries it ALONG the feature it hugs, so existing relief is
 * emphasised and extended into streamlines, and the reshaped terrain
 * steers later tracers. Traffic paints trails into the crest channel.
 * Implementation: sim_step (tracers + fixed-point atomic deposits) →
 * sim_resolve (deposits → persistent overlay map, ping-pong) → the
 * sculptor's compose pass folds the overlay into the published field.
 * At sim 0 the sculpt path is exactly the static one.
 *
 * Video is untouched: tex_in passes through to tex_out (clear when
 * unwired), and the field is only sculpted while something actually
 * consumes the rail.
 */

#include <gpu.h>
#include <host.h>
#include <effect_sdf_field.h>

#include <cmath>
#include <cstdint>
#include <vector>

#include "../plume/field_gen.h"
#include "plume_field_shaders.h"

namespace plume_field {

// Tracer simulation (the mode plume itself doesn't have): a population of
// tracers runs over the sculpted manifold, digs/builds a persistent
// overlay map, and the sculptor composes it into the published field.
// Lockstep with sim_step.hlsl / sim_resolve.hlsl.
constexpr int kSimRes = 256;      // overlay + deposit resolution
constexpr int kTracers = 8192;

struct SimUniforms {
  float dt, accel, vmax, relax;
  float curl, carve_gain, ridge_frac, ov_amp;
  float sim_res, frame, life, _pad0;
};
static_assert(sizeof(SimUniforms) == 48, "SimUniforms layout mismatch");

struct ResolveUniforms {
  float res, rate_h, rate_f, blur;
  float decay_h, decay_f, _pad0, _pad1;
};
static_assert(sizeof(ResolveUniforms) == 32, "ResolveUniforms layout mismatch");

static gpu::ComputePSO s_pso_prefill;
static gpu::ComputePSO s_pso_sim_step;
static gpu::ComputePSO s_pso_sim_resolve;

struct State {
  bool initialized = false;
  plume_gen::Sculptor gen;
  fx::sdf_field::Publisher rail_pub;

  // --- Simulation state ---
  gpu::Buffer tracer_buf;      // kTracers × 2 float4
  gpu::Buffer deposit_buf;     // kSimRes² × 2 int (atomics)
  gpu::Buffer ub_sim, ub_resolve;
  gpu::Texture overlay[2];     // RGBA32F ping-pong (.r height, .g flow) —
                               // full float: it's an accumulator, and fp16
                               // absorbs tiny per-frame deposits near |h|~1
  int ov_ping = 0;             // index written LAST frame (the current map)
  bool sim_cleared = false;    // overlay textures cleared since (re)init
  uint32_t sim_frame = 0;
  float last_dt = 1.0f / 60.0f;

  // Sim param mirrors.
  float sim = 0.0f;
  float sim_rate = 0.5f;
  float sim_swirl = 0.6f;
  float sim_carve = 0.5f;
  float sim_fade = 0.1f;
  float sim_trail = 0.6f;
};

void module_init() {
  auto schema = state::Schema()
      .helpField("intro",
        "## Plume Field\n"
        "The plume sculptor on its own: the same ridged, wind-swept, "
        "morphing displaced sphere — but instead of rendering it, this "
        "effect publishes the geometry on the `sdf_field` rail. Wire it "
        "into an SDF renderer downstream (e.g. **Plume**, whose own "
        "Shape group goes inert) to split the shape from the look: one "
        "effect sculpts, another lights and renders.\n\n"
        "**Simulate** brings the surface alive: tracers stream over the "
        "manifold, carving valleys and building ridges that persist and "
        "steer the tracers that follow.\n\n"
        "The video input passes through untouched.");
  plume_gen::Sculptor::declareSchema(schema,
        "*Radius* is the body; *Ridge Depth/Scale* shape the displacement "
        "riding on it. *Sharpness* carves the field into terraced plates — "
        "at 0 the surface stays a smooth rolling heightfield. *Feathering* "
        "smears the pattern along the flow into wind-swept shingles. "
        "*Morph* drifts the field along a closed loop — it breathes "
        "forever without ever jumping. *Variation* picks a different "
        "pattern. The result leaves on `sdf_field`, not on the video "
        "output.")
      // --- Simulation (the standalone provider's own mode) ---
      .group("sim", "Simulate")
      .groupHelp(
          "A population of **tracers** streams over the sculpted surface in "
          "long arcs, curling around the sphere's axis. Each tracer is born "
          "a *valley-cutter* (hugs valleys, gently digs along its whole "
          "path) or a *ridge-builder* (rides crests, builds them up) — the "
          "swirl carries it ALONG the feature while the slope holds it ON "
          "the feature, so existing valleys and ridges get emphasised and "
          "extended into streamlines. *Simulate* is the master amount (0 = "
          "off, the field is exactly the static sculpt). *Swirl* is the "
          "strength of the axial circulation. *Carve* mixes the population "
          "(1 = all cutters, 0 = all builders). *Fade* lets the carvings "
          "heal (0 = permanent). *Streamlines* paints the tracer traffic "
          "into the crest channel, so trails light up in a downstream "
          "renderer's material.")
      .floatField("sim", 0.0f, 0.f, 1.f, state::PrimaryInput)
          .label("Simulate", "Sim")
      .floatField("sim_rate", 0.5f, 0.f, 1.f, state::PrimaryInput)
          .label("Sim Rate", "Rate")
      .floatField("sim_swirl", 0.6f, 0.f, 1.f, state::PrimaryInput)
          .label("Swirl", "Swirl")
      .floatField("sim_carve", 0.5f, 0.f, 1.f, state::PrimaryInput)
          .label("Carve", "Carve")
      .floatField("sim_fade", 0.1f, 0.f, 1.f, state::PrimaryInput)
          .label("Fade", "Fade")
      .floatField("sim_trail", 0.6f, 0.f, 1.f, state::PrimaryInput)
          .label("Streamlines", "Trail")
      // --- I/O ---
      .textureField("tex_in", state::PrimaryInput)
      .textureField("tex_out", state::PrimaryOutput);
  fx::sdf_field::declare(schema, state::SecondaryOutput);
  schema.capability(state::Capability::Generator);
  state::init("source.sdf.plume_field", {1, 0, 0}, schema);

  if (gpu::Device::backend() == gpu::Backend::None) return;

  // Generator shaders (shell/bake/compose) register with the shared sculptor.
  plume_gen::moduleInit();
  state::registerShaderSPV("plume_field_prefill", PLUME_FIELD_PREFILL_SPV,
                           PLUME_FIELD_PREFILL_SPV_SIZE);
  state::registerShaderSPV("plume_field_sim_step", PLUME_FIELD_SIM_STEP_SPV,
                           PLUME_FIELD_SIM_STEP_SPV_SIZE);
  state::registerShaderSPV("plume_field_sim_resolve",
                           PLUME_FIELD_SIM_RESOLVE_SPV,
                           PLUME_FIELD_SIM_RESOLVE_SPV_SIZE,
                           "rgba32float", "write");
  auto cs_prefill = gpu::Device::createShaderModuleByName("plume_field_prefill");
  auto cs_step = gpu::Device::createShaderModuleByName("plume_field_sim_step");
  auto cs_resolve =
      gpu::Device::createShaderModuleByName("plume_field_sim_resolve");
  if (!cs_prefill || !cs_step || !cs_resolve) return;
  s_pso_prefill = gpu::Device::createComputePSO(cs_prefill, "main",
      gpu::Bindings()
          .tex2d(0)
          .storageTex2d(1));
  s_pso_sim_step = gpu::Device::createComputePSO(cs_step, "main",
      gpu::Bindings()
          .tex2d(0)          // shellCoarse (base field)
          .tex2d(1)          // overlay (previous)
          .sampler(2)
          .storageRW(3)      // tracers
          .storageRW(4)      // deposit bins
          .uniform(5));
  s_pso_sim_resolve = gpu::Device::createComputePSO(cs_resolve, "main",
      gpu::Bindings()
          .storageRW(0)      // deposit bins (read + clear)
          .tex2d(1)          // overlay (previous)
          .storageTex2d(2, gpu::TextureFormat::RGBA32F)  // overlay (next)
          .uniform(3));

  state::log("plume_field: module initialized");
}

void* create() {
  auto* s = new State();
  s->gen.createBuffers();
  s->tracer_buf = gpu::Device::createBuffer(
      kTracers * 2 * 4 * sizeof(float), gpu::BufferUsage::Storage);
  s->deposit_buf = gpu::Device::createBuffer(
      kSimRes * kSimRes * 2 * sizeof(int32_t), gpu::BufferUsage::Storage);
  s->ub_sim = gpu::Device::createBuffer(sizeof(SimUniforms),
                                        gpu::BufferUsage::Uniform);
  s->ub_resolve = gpu::Device::createBuffer(sizeof(ResolveUniforms),
                                            gpu::BufferUsage::Uniform);
  return s;
}

void destroy(void* self) {
  auto* s = static_cast<State*>(self);
  if (!s) return;
  s->gen.release();
  s->tracer_buf.release();
  s->deposit_buf.release();
  s->ub_sim.release();
  s->ub_resolve.release();
  s->overlay[0].release();
  s->overlay[1].release();
  delete s;
}

void init(void* self) {
  auto* s = static_cast<State*>(self);
  if (!s) return;
  s->gen.resetPhase();
  // Restart the simulation from a blank slate: zeroed tracers respawn
  // themselves (sim_step's |dir| < 0.5 branch), zeroed bins deposit
  // nothing, and the overlay textures are re-cleared on next use.
  {
    static const std::vector<uint8_t> zeros(kTracers * 2 * 4 * sizeof(float));
    s->tracer_buf.writeBytes(zeros.data(), zeros.size());
    static const std::vector<uint8_t> zbins(kSimRes * kSimRes * 2 *
                                            sizeof(int32_t));
    s->deposit_buf.writeBytes(zbins.data(), zbins.size());
  }
  s->sim_cleared = false;
  s->sim_frame = 0;
  s->ov_ping = 0;
  s->initialized = s->gen.valid() && s_pso_prefill.valid();
}

void tick(void* self, double dt) {
  auto* s = static_cast<State*>(self);
  if (!s) return;
  if (!(dt > 0.0)) dt = 0.0;
  if (dt > 0.050) dt = 0.050;
  s->gen.tick(dt);
  s->last_dt = (float)(dt > 0.0 ? dt : 1.0 / 60.0);
}

void on_state_patched(void* self, int n, const char* pb, const int* off,
                      const int* len, const int* ops) {
  auto* s = static_cast<State*>(self);
  if (!s) return;
  for (int i = 0; i < n; i++) {
    if (ops[i] != state::PatchReplace) continue;
    const char* p = pb + off[i];
    const int l = len[i];
    if (s->gen.patch(p, l, i)) continue;
    if      (state::pathIs(p, l, "sim"))       s->sim = state::patchFloat(i);
    else if (state::pathIs(p, l, "sim_rate"))  s->sim_rate = state::patchFloat(i);
    else if (state::pathIs(p, l, "sim_swirl")) s->sim_swirl = state::patchFloat(i);
    else if (state::pathIs(p, l, "sim_carve")) s->sim_carve = state::patchFloat(i);
    else if (state::pathIs(p, l, "sim_fade"))  s->sim_fade = state::patchFloat(i);
    else if (state::pathIs(p, l, "sim_trail")) s->sim_trail = state::patchFloat(i);
  }
}

// One simulation frame: step the tracers over last frame's composed
// height (base shell + current overlay), then resolve their deposits
// into the next overlay map (ping-pong). Returns the Overlay to compose,
// or amp 0 when the sim can't run yet (first frame: no base shell).
static plume_gen::Overlay stepSim(State* s) {
  plume_gen::Overlay ov;
  if (!s_pso_sim_step.valid() || !s_pso_sim_resolve.valid()) return ov;
  if (!s->tracer_buf.valid() || !s->deposit_buf.valid() ||
      !s->ub_sim.valid() || !s->ub_resolve.valid()) return ov;
  // The tracers sample the BASE field; it exists after the sculptor's
  // first frame. Until then, skip (the caller composes nothing).
  if (!s->gen.shell_coarse.valid()) return ov;

  for (int i = 0; i < 2; i++) {
    if (!s->overlay[i].valid())
      s->overlay[i] = gpu::Device::createTexture(
          kSimRes, kSimRes, gpu::TextureFormat::RGBA32F);
    if (!s->overlay[i].valid()) return ov;
  }
  if (!s->sim_cleared) {
    gpu::Device::clear(s->overlay[0], 0.f, 0.f, 0.f, 0.f);
    gpu::Device::clear(s->overlay[1], 0.f, 0.f, 0.f, 0.f);
    s->sim_cleared = true;
  }

  const float dt = s->last_dt;
  const float R = s->gen.worldRadius();
  // Master amount sizes the overlay's world amplitude against the body —
  // meaningful even on a smooth sphere (ridge_depth 0): the sim can carve
  // terrain into a blank world.
  const float ov_amp = s->sim * 0.5f * R;

  SimUniforms su = {};
  su.dt = dt;
  su.accel = 0.8f + 10.0f * s->sim_rate * s->sim_rate;
  su.vmax = 0.6f + 1.8f * s->sim_rate;
  // Velocity relaxes toward the zonal (around-the-axis) flow: relax doubles
  // as drag, and the steady advection is what stretches every tracer's path
  // well past ~35 deg of arc. curl is a solid-body angular rate (rad/s at
  // the equator), so at default knobs a tracer sweeps ~30 deg/s.
  su.relax = 1.0f;
  su.curl = (0.05f + 0.95f * s->sim_swirl) * (0.25f + 1.25f * s->sim_rate);
  su.carve_gain = 2.5f;               // gentle, continuous along the path
  su.ridge_frac = 1.0f - s->sim_carve;  // Carve 1 = all valley-cutters
  su.ov_amp = ov_amp;
  su.sim_res = (float)kSimRes;
  su.frame = (float)(s->sim_frame++);
  su.life = 18.0f;
  s->ub_sim.writeOne(su);

  const int cur = s->ov_ping;          // overlay written last frame
  const int nxt = 1 - cur;
  {
    auto cp = gpu::ComputePass::begin();
    cp.setPSO(s_pso_sim_step);
    cp.setTexture(s->gen.shell_coarse, 0, 0);
    cp.setTexture(s->overlay[cur], 1, 0);
    cp.setSampler(s->gen.samp_clamp, 2);
    cp.setBuffer(s->tracer_buf, 3);
    cp.setBuffer(s->deposit_buf, 4);
    cp.setBuffer(s->ub_sim, 5);
    cp.dispatch(kTracers / 64, 1, 1);
    cp.end();
  }

  ResolveUniforms ru = {};
  ru.res = (float)kSimRes;
  // Deposits are already dt-scaled in the step; these are unit gains
  // tuned so an active population reshapes the field over seconds.
  ru.rate_h = 1.8f;
  ru.rate_f = 0.45f;
  // Diffusion must be dt-scaled (a per-frame constant would out-diffuse
  // the tracers' digging) — just enough to keep channel walls smooth.
  ru.blur = std::fmin(0.3f, 1.6f * dt);
  ru.decay_h = std::exp(-dt * 1.8f * s->sim_fade * s->sim_fade);
  // Flow decays fast regardless of Fade: trails read as RECENT traffic
  // (moving streamlines), not the whole run's history smeared to white.
  ru.decay_f = std::exp(-dt * (1.5f + 5.0f * s->sim_fade));
  s->ub_resolve.writeOne(ru);
  {
    auto cp = gpu::ComputePass::begin();
    cp.setPSO(s_pso_sim_resolve);
    cp.setBuffer(s->deposit_buf, 0);
    cp.setTexture(s->overlay[cur], 1, 0);
    cp.setTexture(s->overlay[nxt], 2, 1);
    cp.setBuffer(s->ub_resolve, 3);
    cp.dispatch(kSimRes / 8, kSimRes / 8);
    cp.end();
  }
  s->ov_ping = nxt;

  ov.tex = s->overlay[nxt];
  ov.amp = ov_amp;
  ov.trail = s->sim_trail;
  return ov;
}

void render(void* self, int vp_w, int vp_h) {
  auto* s = static_cast<State*>(self);
  if (!s || !s->initialized || vp_w <= 0 || vp_h <= 0) return;

  auto in = gpu::Device::textureForField("tex_in");
  auto out = gpu::Device::textureForField("tex_out");
  if (!out.valid()) return;

  // Sculpt + publish only while something consumes the rail. With the
  // simulation active, the tracers step first (over last frame's field)
  // and their overlay is composed into this frame's sculpt; at sim 0 the
  // path is exactly the static sculptor.
  if (state::isOutputConnected("sdf_field")) {
    fx::sdf_field::Desc field;
    gpu::Texture grid, shell;
    if (s->sim > 0.001f) {
      plume_gen::Overlay ov = stepSim(s);
      if (s->gen.run(field, grid, shell, ov.amp > 0.0f ? &ov : nullptr))
        s->rail_pub.publish(field, grid.id, shell.id);
    } else if (s->gen.run(field, grid, shell)) {
      s->rail_pub.publish(field, grid.id, shell.id);
    }
  }

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

} // namespace plume_field
