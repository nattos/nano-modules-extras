/*
 * warp.legacy.d_wave — "D Wave" radial-ripple distortion field.
 *
 * A focused v2 port of the only block of the shipped NanoGraph "Darkburst" the
 * team actually used live: its distortion field ("D wave"). The rest of
 * Darkburst (the bursting star system, glowing core, laser-line output) is
 * dropped — per the catalogue, everything except the D wave was rarely used.
 *
 * Two layers:
 *  • WAVE FIELD (stateful) — a persistent polar buffer (angle × radius). Fresh
 *    turbulent grain is injected at the centre each frame, marches OUTWARD, and
 *    decays — a propagating, stateful ripple field (field.hlsl, ping-pong). This
 *    is the signature look; its sense of history is the point.
 *  • DAMPENING FLASHES (transient) — a pool of fast wave-particles that live in
 *    a mid-radius band (never the centre, recycled before the rim), rendered as
 *    thin elongated blobs into a SEPARATE damp texture and SUBTRACTED from the
 *    wave field at warp time. They punch quick streaks of reduced distortion
 *    without polluting the wave field's advection.
 *
 * Each output pixel reads (wave − damp) in polar space and is radially displaced
 * by it. A trigger fires a turbulent shock ring in the wave field.
 *
 * Passes/frame: field update (compute) → particle update (compute) → flash splat
 * (instanced, additive, into the damp texture) → warp (compute).
 */

#include <gpu.h>
#include <host.h>
#include "d_wave_shaders.h"

#include <cmath>
#include <cstdint>

namespace d_wave {

// Polar field resolution. Independent of viewport: angle columns × radius rows.
static constexpr int   ANG = 512;
static constexpr int   RAD = 256;
static constexpr int   MAX_PARTICLES = 4096;

// Wave-field tuning constants.
static constexpr float SPEED_ROWS_PER_SEC = 600.0f;  // wave_speed=1 → rows/sec
static constexpr float DECAY_MAX          = 4.0f;    // wave_decay=1 → e-folds/sec
static constexpr float SHARP_MIN          = 3.0f;    // soften=1 → wide ring
static constexpr float SHARP_MAX          = 60.0f;   // soften=0 → tight ring
static constexpr float MAX_CELLS          = 47.0f;   // density=1 → 48 noise cells
static constexpr float BURST_TAU          = 0.18f;   // trigger-burst half-life (s)

// Dampening-flash tuning constants.
static constexpr float DAMP_SPEED_SCALE = 3.0f;      // damp_rate=1 → radius/sec
static constexpr float DAMP_SPREAD      = 0.6f;      // per-flash speed variation
static constexpr float DAMP_AW          = 0.020f;    // flash angular half-width
static constexpr float DAMP_LEN         = 0.060f;    // flash radial half-length

struct FieldUniforms {
  float y_shift, decay, rate, sharp;
  float ang_cells, noise_power, burst;
  uint32_t frame;
};
struct ParticleUniforms {
  uint32_t count, frame, seed, _u;
  float dt, speed, spread, _p;
};
struct BlobUniforms {
  uint32_t count;
  float ang_halfwidth, rad_halflen, grain;
};
struct WarpUniforms {
  float aspect, distortion, scale, squeeze;
  float render_alpha, debug_field, center_x, center_y;
  float damp_amount, _d0, _d1, _d2;
};
struct MotionUniforms {
  float aspect, distortion, scale, squeeze;
  float damp_amount, center_x, center_y, motion_scale;
};

struct State {
  gpu::Texture field[2];          // ping-pong stateful wave field (RGBA16F)
  gpu::Texture damp_tex[2];           // ping-pong transient flash layer (RGBA16F)
  int   cur = 0;                  // shared rd=cur / wr=cur^1 for field + damp
  bool  cleared = false;

  // Motion-vector output (render_outputs/motion) — only when a sink reads it.
  gpu::Texture motion_tex;
  gpu::Texture zero_motion_tex;   // 1×1 fallback for an unwired upstream motion
  int   motion_w = 0, motion_h = 0;

  gpu::Buffer  particle_buf;
  gpu::Buffer  field_uniform;
  gpu::Buffer  particle_uniform;
  gpu::Buffer  blob_uniform;
  gpu::Buffer  warp_uniform;
  gpu::Buffer  motion_uniform;
  gpu::Sampler samp_field;        // Linear + Repeat (angle wrap)
  gpu::Sampler samp_in;          // Linear + ClampToEdge
  bool initialized = false;
  bool seeded      = false;

  // Wave-field params.
  float distortion = 0.5f;
  float rate       = 0.5f;
  float wave_speed = 0.3f;
  float scale      = 1.0f;
  float density    = 0.3f;
  float wave_decay = 0.5f;
  float soften     = 0.5f;
  float grain      = 0.43f;
  // Dampening-flash params.
  float damp       = 0.5f;
  int   damp_count = 96;
  float damp_rate  = 0.5f;
  // Motion output.
  float motion_scale = 1.0f;
  // Warp params.
  float squeeze    = 0.0f;
  float render_alpha = 1.0f;
  float center_x   = 0.0f;
  float center_y   = 0.0f;
  float debug_field = 0.0f;

  // Trigger.
  bool  gate_prev = false;
  bool  trigger_prev = false;
  float burst_env = 0.0f;

  // Per-frame.
  float    dt = 1.0f / 60.0f;
  uint32_t frame = 0;
};

static gpu::ComputePSO s_pso_field;
static gpu::ComputePSO s_pso_particles;
static gpu::RenderPSO  s_pso_blob;
static gpu::ComputePSO s_pso_warp;
static gpu::ComputePSO s_pso_motion;

void module_init() {
  state::init("warp.legacy.d_wave", {1, 2, 3},
    state::Schema()
      // ---- Standard ---- (floatField: name,def,min,max,io,magnitude,step,units,description)
      .floatField("distortion",  0.5f,  0.0f, 1.0f, state::PrimaryInput, nullptr, 0.01f, nullptr,
                  "Radial warp magnitude (quadratic, fine at the low end).")
      .floatField("rate",        0.5f,  0.0f, 1.0f, state::PrimaryInput, nullptr, 0.01f, nullptr,
                  "Wave grain density (fraction of angles emitting).")
      .floatField("wave_speed",  0.3f,  0.0f, 1.0f, state::PrimaryInput, nullptr, 0.01f, nullptr,
                  "How fast the waves drift outward.")
      .floatField("scale",       1.0f,  0.1f, 4.0f, state::PrimaryInput, nullptr, 0.01f, nullptr,
                  "Spatial size of the wave field.")
      .floatField("density",     0.3f,  0.0f, 1.0f, state::PrimaryInput, nullptr, 0.01f, nullptr,
                  "Wave grain frequency around the circle.")
      .boolField ("gate",        false, state::PrimaryInput,
                  "Rising edge fires a turbulent shock ring.")
      .eventField("trigger",     state::PrimaryInput)
      // ---- Tuning: wave field ----
      .floatField("wave_decay",  0.5f,  0.0f, 1.0f, state::SecondaryInput, nullptr, 0.01f, nullptr,
                  "How fast the waves fade as they travel.")
      .floatField("soften",      0.5f,  0.0f, 1.0f, state::SecondaryInput, nullptr, 0.01f, nullptr,
                  "Radial thickness of each wave band.")
      .floatField("grain",       0.43f, 0.0f, 1.0f, state::SecondaryInput, nullptr, 0.01f, nullptr,
                  "Wave grain contrast: low = haze, high = sparkle.")
      // ---- Tuning: dampening flashes ----
      .floatField("damp",        0.5f,  0.0f, 1.0f, state::SecondaryInput, nullptr, 0.01f, nullptr,
                  "How strongly the flashes dampen the distortion.")
      .intField  ("damp_count",  96,    0,    MAX_PARTICLES, state::SecondaryInput)
      .floatField("damp_rate",   0.5f,  0.0f, 1.0f, state::SecondaryInput, nullptr, 0.01f, nullptr,
                  "How fast the dampening flashes evolve.")
      .floatField("motion_scale", 1.0f, 0.0f, 4.0f, state::SecondaryInput, nullptr, 0.01f, nullptr,
                  "Scales the emitted motion-vector magnitude.")
      // ---- Warp ----
      .floatField("squeeze",     0.0f, -1.0f, 1.0f, state::SecondaryInput, nullptr, 0.01f, nullptr,
                  "Radial offset of the wave pattern.")
      .floatField("render_alpha", 1.0f, 0.0f, 1.0f, state::SecondaryInput, nullptr, 0.01f, nullptr,
                  "Opacity of the distorted layer over the input.")
      .vec2Field ("center",      0.0f, 0.0f, state::SecondaryInput, -1.0f, 1.0f)
      // ---- Debug ----
      .floatField("debug_field",  0.0f, 0.0f, 1.0f, state::SecondaryInput, nullptr, 0.01f, nullptr,
                  "Overlay the (dampened) wave field.")
      // ---- I/O ----
      .textureField("tex_in",  state::PrimaryInput)
      .textureField("tex_out", state::PrimaryOutput)
      // Motion-vector rail: produced only when a downstream sink reads it.
      // render_outputs_in carries an optional upstream motion field to compose.
      .renderOutputs(state::PrimaryOutput)
      .renderOutputs(state::PrimaryInput, "render_outputs_in")
      .capability(state::Capability::SeekableApproximate)
  );

  if (gpu::Device::backend() == gpu::Backend::None) return;

  // The wave field pass writes an RGBA16F storage texture → override naga's
  // default (warp writes the sketch-default tex_out).
  state::registerShaderSPV("dw_field",      FIELD_SPV,     FIELD_SPV_SIZE, "rgba16float", "write");
  state::registerShaderSPV("dw_particles",  PARTICLES_SPV, PARTICLES_SPV_SIZE);
  state::registerShaderSPV("dw_blob_vs",    BLOB_VS_SPV,   BLOB_VS_SPV_SIZE);
  state::registerShaderSPV("dw_blob_fs",    BLOB_FS_SPV,   BLOB_FS_SPV_SIZE);
  state::registerShaderSPV("dw_warp",       WARP_SPV,      WARP_SPV_SIZE);
  state::registerShaderSPV("dw_motion",     MOTION_SPV,    MOTION_SPV_SIZE, "rgba16float", "write");

  auto cs_field     = gpu::Device::createShaderModuleByName("dw_field");
  auto cs_particles = gpu::Device::createShaderModuleByName("dw_particles");
  auto vs_blob      = gpu::Device::createShaderModuleByName("dw_blob_vs");
  auto fs_blob      = gpu::Device::createShaderModuleByName("dw_blob_fs");
  auto cs_warp      = gpu::Device::createShaderModuleByName("dw_warp");
  auto cs_motion    = gpu::Device::createShaderModuleByName("dw_motion");
  if (!cs_field || !cs_particles || !vs_blob || !fs_blob || !cs_warp || !cs_motion) return;

  s_pso_field = gpu::Device::createComputePSO(cs_field, "main", gpu::Bindings()
      .tex2d(0).sampler(1).storageTex2d(2, gpu::TextureFormat::RGBA16F).uniform(3));
  s_pso_particles = gpu::Device::createComputePSO(cs_particles, "main", gpu::Bindings()
      .storageRW(0).uniform(1));
  s_pso_blob = gpu::Device::createInstancedRenderPSO(
      vs_blob, "main", fs_blob, "main",
      gpu::TextureFormat::RGBA16F,
      gpu::Bindings().storage(0).uniform(1),
      gpu::Device::BlendMode::Additive);
  s_pso_warp = gpu::Device::createComputePSO(cs_warp, "main", gpu::Bindings()
      .tex2d(0).tex2d(1).tex2d(2).sampler(3).sampler(4).storageTex2d(5).uniform(6));
  s_pso_motion = gpu::Device::createComputePSO(cs_motion, "main", gpu::Bindings()
      .tex2d(0).tex2d(1).tex2d(2).tex2d(3).tex2d(4).sampler(5).sampler(6)
      .storageTex2d(7, gpu::TextureFormat::RGBA16F).uniform(8));

  state::log("d_wave: module initialized");
}

void* create() {
  auto* s = new State();
  s->field[0] = gpu::Device::createTexture(ANG, RAD, gpu::TextureFormat::RGBA16F);
  s->field[1] = gpu::Device::createTexture(ANG, RAD, gpu::TextureFormat::RGBA16F);
  s->damp_tex[0]  = gpu::Device::createTexture(ANG, RAD, gpu::TextureFormat::RGBA16F);
  s->damp_tex[1]  = gpu::Device::createTexture(ANG, RAD, gpu::TextureFormat::RGBA16F);
  s->particle_buf = gpu::Device::createBuffer(MAX_PARTICLES * 4 * sizeof(float), gpu::BufferUsage::Storage);
  s->field_uniform    = gpu::Device::createBuffer(sizeof(FieldUniforms), gpu::BufferUsage::Uniform);
  s->particle_uniform = gpu::Device::createBuffer(sizeof(ParticleUniforms), gpu::BufferUsage::Uniform);
  s->blob_uniform     = gpu::Device::createBuffer(sizeof(BlobUniforms), gpu::BufferUsage::Uniform);
  s->warp_uniform     = gpu::Device::createBuffer(sizeof(WarpUniforms), gpu::BufferUsage::Uniform);
  s->motion_uniform   = gpu::Device::createBuffer(sizeof(MotionUniforms), gpu::BufferUsage::Uniform);
  s->samp_field = gpu::Device::createSampler(gpu::FilterMode::Linear, gpu::AddressMode::Repeat);
  s->samp_in    = gpu::Device::createSampler(gpu::FilterMode::Linear, gpu::AddressMode::ClampToEdge);
  return s;
}

void destroy(void* self) {
  auto* s = static_cast<State*>(self);
  if (!s) return;
  s->field[0].release();
  s->field[1].release();
  s->damp_tex[0].release();
  s->damp_tex[1].release();
  s->motion_tex.release();
  s->zero_motion_tex.release();
  s->particle_buf.release();
  s->field_uniform.release();
  s->particle_uniform.release();
  s->blob_uniform.release();
  s->warp_uniform.release();
  s->motion_uniform.release();
  s->samp_field.release();
  s->samp_in.release();
  delete s;
}

void init(void* self) {
  auto* s = static_cast<State*>(self);
  if (!s) return;
  if (!s_pso_field.valid() || !s_pso_particles.valid() || !s_pso_blob.valid() || !s_pso_warp.valid()) return;
  s->cleared = false;
  s->seeded = false;
  s->cur = 0;
  s->frame = 0;
  s->burst_env = 0.0f;
  s->initialized = true;
}

void tick(void* self, double dt) {
  auto* s = static_cast<State*>(self);
  if (!s) return;
  s->dt = (float)dt;
  s->frame++;
  s->burst_env *= std::exp(-(float)dt / BURST_TAU);
  if (s->burst_env < 1e-3f) s->burst_env = 0.0f;
}

void on_resolume_param(void*, long long, double) {}

void on_state_patched(void* self, int n, const char* pb, const int* off,
                      const int* len, const int* ops) {
  auto* s = static_cast<State*>(self);
  if (!s) return;
  for (int i = 0; i < n; i++) {
    if (ops[i] != state::PatchReplace) continue;
    const char* p = pb + off[i];
    int l = len[i];
    if      (state::pathIs(p, l, "distortion"))    s->distortion = state::patchFloat(i);
    else if (state::pathIs(p, l, "rate"))          s->rate = state::patchFloat(i);
    else if (state::pathIs(p, l, "wave_speed"))    s->wave_speed = state::patchFloat(i);
    else if (state::pathIs(p, l, "scale"))         s->scale = state::patchFloat(i);
    else if (state::pathIs(p, l, "density"))       s->density = state::patchFloat(i);
    else if (state::pathIs(p, l, "wave_decay"))    s->wave_decay = state::patchFloat(i);
    else if (state::pathIs(p, l, "soften"))        s->soften = state::patchFloat(i);
    else if (state::pathIs(p, l, "grain"))         s->grain = state::patchFloat(i);
    else if (state::pathIs(p, l, "damp"))          s->damp = state::patchFloat(i);
    else if (state::pathIs(p, l, "damp_count"))    s->damp_count = state::patchInt(i);
    else if (state::pathIs(p, l, "damp_rate"))     s->damp_rate = state::patchFloat(i);
    else if (state::pathIs(p, l, "motion_scale"))  s->motion_scale = state::patchFloat(i);
    else if (state::pathIs(p, l, "squeeze"))       s->squeeze = state::patchFloat(i);
    else if (state::pathIs(p, l, "render_alpha"))  s->render_alpha = state::patchFloat(i);
    else if (state::pathIs(p, l, "debug_field"))   s->debug_field = state::patchFloat(i);
    else if (state::pathIs(p, l, "center"))        { auto v = state::patchVec2(i); s->center_x = v.x; s->center_y = v.y; }
    else if (state::pathIs(p, l, "gate")) {
      bool g = state::patchBool(i);
      if (g && !s->gate_prev) s->burst_env = 1.0f;   // rising edge → shock ring
      s->gate_prev = g;
    }
    else if (state::pathIs(p, l, "trigger")) {
      bool t = state::patchFloat(i) != 0.0f;
      if (t && !s->trigger_prev) s->burst_env = 1.0f; // rising edge → shock ring
      s->trigger_prev = t;
    }
  }
}

void render(void* self, int vp_w, int vp_h) {
  auto* s = static_cast<State*>(self);
  if (!s || !s->initialized || vp_w <= 0 || vp_h <= 0) return;

  auto in  = gpu::Device::textureForField("tex_in");
  auto out = gpu::Device::textureForField("tex_out");
  if (!in.valid() || !out.valid()) return;
  if (!s->field[0].valid() || !s->field[1].valid() || !s->damp_tex[0].valid() || !s->damp_tex[1].valid()) return;

  if (!s->cleared) {
    gpu::Device::clear(s->field[0], 0.f, 0.f, 0.f, 0.f);
    gpu::Device::clear(s->field[1], 0.f, 0.f, 0.f, 0.f);
    gpu::Device::clear(s->damp_tex[0], 0.f, 0.f, 0.f, 0.f);
    gpu::Device::clear(s->damp_tex[1], 0.f, 0.f, 0.f, 0.f);
    s->cleared = true;
  }

  float dt = s->dt;
  int rd = s->cur, wr = s->cur ^ 1;

  // --- Pass 1: stateful wave field (inject + advect + decay) ---
  FieldUniforms fu = {};
  fu.y_shift     = (s->wave_speed * SPEED_ROWS_PER_SEC * dt) / float(RAD);
  fu.decay       = std::exp(-s->wave_decay * DECAY_MAX * dt);
  fu.rate        = s->rate;
  fu.sharp       = SHARP_MAX + (SHARP_MIN - SHARP_MAX) * s->soften;
  fu.ang_cells   = std::round(1.0f + s->density * MAX_CELLS);
  fu.noise_power = 0.5f + s->grain * 3.5f;
  fu.burst       = s->burst_env;
  fu.frame       = s->frame;
  s->field_uniform.writeOne(fu);
  {
    auto cp = gpu::ComputePass::begin();
    cp.setPSO(s_pso_field);
    cp.setTexture(s->field[rd], 0, 0);
    cp.setSampler(s->samp_field, 1);
    cp.setTexture(s->field[wr], 2, 1);
    cp.setBuffer(s->field_uniform, 3);
    cp.dispatch((ANG + 7) / 8, (RAD + 7) / 8);
    cp.end();
  }

  // --- Pass 2: dampening-flash particle pool ---
  int dcount = s->damp_count;
  if (dcount < 0) dcount = 0;
  if (dcount > MAX_PARTICLES) dcount = MAX_PARTICLES;
  ParticleUniforms pu = {};
  pu.count  = (uint32_t)dcount;
  pu.frame  = s->frame;
  pu.seed   = s->seeded ? 0u : 1u;
  pu.dt     = dt;
  pu.speed  = s->damp_rate * DAMP_SPEED_SCALE;
  pu.spread = DAMP_SPREAD;
  s->particle_uniform.writeOne(pu);
  s->seeded = true;
  {
    auto cp = gpu::ComputePass::begin();
    cp.setPSO(s_pso_particles);
    cp.setBuffer(s->particle_buf, 0);
    cp.setBuffer(s->particle_uniform, 1);
    cp.dispatch((MAX_PARTICLES + 63) / 64);
    cp.end();
  }

  // --- Pass 3: splat the flashes into the (cleared) damp texture ---
  BlobUniforms bu = {};
  bu.count         = (uint32_t)dcount;
  bu.ang_halfwidth = DAMP_AW;
  bu.rad_halflen   = DAMP_LEN;
  bu.grain         = s->grain;
  s->blob_uniform.writeOne(bu);
  gpu::Device::clear(s->damp_tex[wr], 0.f, 0.f, 0.f, 0.f);
  if (dcount > 0 && s->damp > 0.0f) {
    auto rp = gpu::RenderPass::beginLoad(s->damp_tex[wr]);
    rp.setPSO(s_pso_blob);
    rp.setBuffer(s->particle_buf, 0);
    rp.setBuffer(s->blob_uniform, 1);
    rp.draw(6, dcount);
    rp.end();
  }

  // Shared warp geometry: quadratic distortion curve, 1/10 of the old range
  // (the previous linear [0,1] was far too strong across most of the slider),
  // and the cover-square anchor → uv offset.
  float maxDim = float(vp_w > vp_h ? vp_w : vp_h);
  float effDist = s->distortion * s->distortion * 0.1f;
  float cx_uv = s->center_x * (0.5f * maxDim / float(vp_w));
  float cy_uv = s->center_y * (0.5f * maxDim / float(vp_h));

  // --- Pass 4: warp + composite (wave − damp) ---
  WarpUniforms wu = {};
  wu.aspect       = float(vp_w) / float(vp_h);
  wu.distortion   = effDist;
  wu.scale        = s->scale;
  wu.squeeze      = s->squeeze;
  wu.render_alpha = s->render_alpha;
  wu.debug_field  = s->debug_field;
  wu.center_x     = cx_uv;
  wu.center_y     = cy_uv;
  wu.damp_amount  = s->damp;
  s->warp_uniform.writeOne(wu);
  {
    auto cp = gpu::ComputePass::begin();
    cp.setPSO(s_pso_warp);
    cp.setTexture(in, 0, 0);
    cp.setTexture(s->field[wr], 1, 0);
    cp.setTexture(s->damp_tex[wr], 2, 0);
    cp.setSampler(s->samp_in, 3);
    cp.setSampler(s->samp_field, 4);
    cp.setTexture(out, 5, 1);
    cp.setBuffer(s->warp_uniform, 6);
    cp.dispatch((vp_w + 7) / 8, (vp_h + 7) / 8);
    cp.end();
  }

  // --- Pass 5: motion vectors (render_outputs/motion) — only when a sink reads
  // the rail. The warp's per-frame velocity comes from the field strength delta
  // (now vs the ping-ponged prev), so both the wave and the fast flashes emit
  // motion; the optional upstream motion field rides through warped.
  if (state::isOutputConnected("render_outputs")) {
    if (!s->motion_tex.valid() || s->motion_w != vp_w || s->motion_h != vp_h) {
      s->motion_tex.release();
      s->motion_tex = gpu::Device::createTexture(vp_w, vp_h, gpu::TextureFormat::RGBA16F);
      s->motion_w = vp_w; s->motion_h = vp_h;
      if (s->motion_tex.valid()) state::setGpuTexture("render_outputs/motion", s->motion_tex.id);
    }
    if (s->motion_tex.valid()) {
      auto up = gpu::Device::textureForField("render_outputs_in/motion");
      if (!up.valid()) {
        if (!s->zero_motion_tex.valid())
          s->zero_motion_tex = gpu::Device::createTexture(1, 1, gpu::TextureFormat::RGBA16F);
        up = s->zero_motion_tex;
      }
      MotionUniforms mu = {};
      mu.aspect       = float(vp_w) / float(vp_h);
      mu.distortion   = effDist;
      mu.scale        = s->scale;
      mu.squeeze      = s->squeeze;
      mu.damp_amount  = s->damp;
      mu.center_x     = cx_uv;
      mu.center_y     = cy_uv;
      mu.motion_scale = s->motion_scale;
      s->motion_uniform.writeOne(mu);

      auto cp = gpu::ComputePass::begin();
      cp.setPSO(s_pso_motion);
      cp.setTexture(s->field[wr], 0, 0);   // wave now
      cp.setTexture(s->field[rd], 1, 0);   // wave prev
      cp.setTexture(s->damp_tex[wr], 2, 0);    // flash now
      cp.setTexture(s->damp_tex[rd], 3, 0);    // flash prev
      cp.setTexture(up, 4, 0);             // upstream motion (or 1×1 zero)
      cp.setSampler(s->samp_field, 5);
      cp.setSampler(s->samp_in, 6);
      cp.setTexture(s->motion_tex, 7, 1);
      cp.setBuffer(s->motion_uniform, 8);
      cp.dispatch((vp_w + 7) / 8, (vp_h + 7) / 8);
      cp.end();
    }
  }

  s->cur = wr;
  gpu::Device::submit();
}

} // namespace d_wave
