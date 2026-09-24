/*
 * filter.blur.smear — "Smear": a directional Pixulant.
 *
 * Pixulant (warp.legacy.pixulant) dissolves an image with an ISOTROPIC random
 * scatter. Smear keeps that spirit but drives the distribution off a user AXIS,
 * so it reads as a directional blur with an optional Pixulant-style scatter grain.
 *
 * The kernel footprint is shared by both modes:
 *   - a MAJOR axis (angle) with an asymmetric reach — `tail` shrinks the forward
 *     (head) reach so a bright point trails a STREAK behind it rather than a blob;
 *   - a MINOR axis (perpendicular) whose width is scaled by a GLOBAL perspective
 *     gradient — `tilt` ramps the minor-blur width across the whole frame along the
 *     major axis (narrow head side, wide rear side; flips sign). Tilt-shift, not a
 *     per-streak comet.
 *
 * Modes:
 *   Blur    — two separable compute passes (major → scratch, minor → out). Same
 *             PSO, run twice with different uniforms (the effect_blur.h pattern).
 *   Scatter — one Pixulant-style pass: salted random sampling from the SAME tilted
 *             footprint + the dive / abs-difference / exposure "strange colours"
 *             cascade, animated by `motion`.
 *
 * Animated (scatter salt drift) → SeekableApproximate, and NO is_identity (a
 * stateful stage that reports identity can be permanently sidelined).
 */

#include <gpu.h>
#include <host.h>
#include "smear_shaders.h"

#include <cmath>
#include <cstdint>

namespace smear {

static constexpr float PI          = 3.14159265358979323846f;
static constexpr float MAJOR_MAX   = 1.10f;  // length=1 → 110% of the short axis (long streak reach)
static constexpr float MINOR_MAX   = 0.15f;  // width=1  → 15% of the short axis (cross reach)
static constexpr float MOTION_RATE = 0.8f;   // motion=1 → 0.8 salt-cycles/sec (scatter churn)

enum Mode : int { MODE_BLUR = 0, MODE_SCATTER = 1 };

struct BlurUniforms {
  float axis_x, axis_y;         // aspect-scaled UV step per unit reach (this pass)
  float reach_fwd, reach_back;  // asymmetric reach, short-axis fractions
  float major_x, major_y;       // screen-unit major dir (perspective proj)
  float tilt;                   // perspective amount (0 on the major pass)
  float falloff_k;              // gaussian tail sharpness (from softness)
  float exposure;               // output gain (final pass only)
  int   samples;
  float _pad0, _pad1;
};
static_assert(sizeof(BlurUniforms) == 48, "BlurUniforms layout mismatch");

struct ScatterUniforms {
  float axis_maj_x, axis_maj_y;
  float axis_min_x, axis_min_y;
  float reach_fwd, reach_back;
  float width;
  float salt_base;
  float major_x, major_y;
  float tilt;
  float dive;
  float exposure_gain;
  float edge_artifacts;
  float exposure;
  float softness;
};
static_assert(sizeof(ScatterUniforms) == 64, "ScatterUniforms layout mismatch");

struct State {
  gpu::Buffer  uniform_major;   // blur pass 1
  gpu::Buffer  uniform_minor;   // blur pass 2
  gpu::Buffer  uniform_scatter; // scatter pass
  gpu::Sampler sampler;
  gpu::Texture scratch;         // blur pass-1 output (pass 2 reads it)
  int          scratch_w = 0, scratch_h = 0;
  bool         initialized = false;

  // Schema-mirrored params.
  int   mode     = MODE_BLUR;
  float angle    = 0.0f;   // [-1,1] → [-π,π]
  float strength = 1.0f;   // master scale on length + width
  float length   = 0.35f;
  float width    = 0.12f;
  float tail    = 0.5f;
  float tilt    = 0.0f;   // [-1,1]
  float softness = 0.6f;  // gaussian tail sharpness
  float exposure = 1.0f;  // global output gain
  int   samples = 12;
  float dive               = 0.7f;
  float motion             = 0.7f;
  float dive_contrast_bias = 1.0f;
  float dive_cap           = 1.0f;
  float edge_artifacts     = 0.0f;

  // Runtime: scatter salt animation (Saw + on-change Random), Pixulant-style.
  double    phase    = 0.0;
  long long last_cyc = -1;
  float     rand_val = 0.0f;
};

// Type-shared, compiled once.
static gpu::ComputePSO s_pso_blur;
static gpu::ComputePSO s_pso_scatter;

static inline float clamp01(float x) { return x < 0.f ? 0.f : (x > 1.f ? 1.f : x); }
static inline float ease(float x)    { x = clamp01(x); return x * x * (3.f - 2.f * x); }

static inline float hash01(uint32_t v) {
  v ^= v >> 16; v *= 0x7feb352du; v ^= v >> 15; v *= 0x846ca68bu; v ^= v >> 16;
  return (float)(v & 0x00FFFFFFu) / (float)0x01000000u;
}

// Show `samples` in Blur mode, the scatter (dive) group in Scatter mode. Touches
// the type-shared schema, so it takes the mode value (not per-instance state).
static void apply_visibility(int mode) {
  bool scatter = (mode == MODE_SCATTER);
  state::setFieldHidden("samples",            scatter);  // blur-only (tap count)
  state::setFieldHidden("dive",               !scatter);
  state::setFieldHidden("motion",             !scatter);
  state::setFieldHidden("dive_contrast_bias", !scatter);
  state::setFieldHidden("dive_cap",           !scatter);
  state::setFieldHidden("edge_artifacts",     !scatter);
}

// Static (self-less) visibility evaluator — pure over a candidate state.
void eval_visibility(int n, const char* pb, const int* off, const int* len, const int* ops) {
  int mode = MODE_BLUR;
  for (int i = 0; i < n; i++) {
    if (ops[i] != state::PatchReplace) continue;
    if (state::pathIs(pb + off[i], len[i], "mode")) mode = state::patchInt(i);
  }
  apply_visibility(mode);
}

static void on_state_ready(void* self) {
  auto* s = static_cast<State*>(self);
  if (s) apply_visibility(s->mode);
}

void module_init() {
  state::init("filter.blur.smear", {1, 0, 3},
    state::Schema()
      .helpField("intro",
        "## Smear\n"
        "A **directional Pixulant** — a separable blur along an axis you choose, "
        "tilted into a *tail* rather than a symmetric blob, with a photographic "
        "*perspective* pinch across the frame.\n\n"
        "**Try:** raise *Length* and set an *Angle* for a motion streak, dial *Tail* "
        "for a comet trail, and push *Perspective* for a tilt-shift ramp. Switch to "
        "**Scatter** and crank *Dive* for Pixulant's dissolving coloured grain, now "
        "smeared along the axis.")
      .group("axis", "Axis")
        .groupHelp("*Angle* sets the major smear axis (0..1 of a turn, signed). The "
                   "minor axis is perpendicular.")
      .selectField("mode", MODE_BLUR, state::PrimaryInput,
                   {{"Blur", MODE_BLUR}, {"Scatter", MODE_SCATTER}}).label("Mode", "Mode")
      .floatField("angle", 0.0f, -1.0f, 1.0f, state::PrimaryInput, nullptr, 0.01f,
                  nullptr, "Major-axis direction (−1..+1 = a full turn).").label("Angle", "Ang")
      .group("shape", "Shape")
        .groupHelp("*Strength* is the master amount — scales both Length and Width "
                   "(0 = off). *Length* is the streak reach along the axis; *Width* the "
                   "perpendicular spread. *Tail* biases the reach behind the head "
                   "(0 = symmetric blob, 1 = one-sided streak). *Perspective* ramps "
                   "the minor width across the frame (tilt-shift; flips sign).")
      .floatField("strength", 1.0f, 0.0f, 1.0f, state::PrimaryInput, nullptr, 0.01f,
                  nullptr, "Master amount — scales Length and Width together (0 = off).")
                  .label("Strength", "Str")
      .floatField("length", 0.35f, 0.0f, 1.0f, state::PrimaryInput, nullptr, 0.01f,
                  nullptr, "Streak reach along the major axis.").label("Length", "Len")
      .floatField("width", 0.12f, 0.0f, 1.0f, state::PrimaryInput, nullptr, 0.01f,
                  nullptr, "Reach along the minor (perpendicular) axis.").label("Width", "Wid")
      .floatField("tail", 0.5f, 0.0f, 1.0f, state::PrimaryInput, nullptr, 0.01f,
                  nullptr, "Kernel asymmetry: 0 = symmetric blob, 1 = one-sided tail.")
                  .label("Tail", "Tail")
      .floatField("tilt", 0.0f, -1.0f, 1.0f, state::PrimaryInput, nullptr, 0.01f,
                  nullptr, "Perspective: pinches the minor width on the head side, "
                  "expands it on the rear (flips sign).").label("Perspective", "Persp")
      .floatField("softness", 0.6f, 0.0f, 1.0f, state::PrimaryInput, nullptr, 0.01f,
                  nullptr, "Distribution falloff: 0 = boxy with hard edges, 1 = smooth "
                  "gaussian. Softens the Blur tail and the Scatter grain (removes the "
                  "hard edges at high Tail).").label("Softness", "Soft")
      .group("look", "Look")
        .groupHelp("*Exposure* scales the output up — a long/thin smear averages a "
                   "bright line down toward black, so lift it back here.")
      .floatField("exposure", 1.0f, 0.25f, 8.0f, state::PrimaryInput, nullptr, 0.05f,
                  nullptr, "Output gain — brightens the (often dark) smeared result.")
                  .label("Exposure", "Exp")
      .group("blur", "Blur")
        .groupHelp("*Samples* trades speed for smoothness along the streak.")
      .intField("samples", 12, 4, 32, state::PrimaryInput, 0, nullptr,
                "Taps per separable pass (quality).").label("Samples", "Smpl")
      .group("scatter", "Scatter")
        .groupHelp("Pixulant's dissolve. *Dive* mixes from the image into the "
                   "differenced grain; *Exposure* lifts its brightness; *Motion* is "
                   "the churn rate. Only active in Scatter mode.")
      .floatField("dive", 0.7f, 0.0f, 1.0f, state::PrimaryInput, nullptr, 0.01f,
                  nullptr, "Blend from the image (0) into the differenced grain (1).")
                  .label("Dive", "Dive")
      .floatField("motion", 0.7f, 0.0f, 1.0f, state::PrimaryInput, nullptr, 0.01f,
                  nullptr, "Churn rate — how fast the scatter animates (0 = frozen).")
                  .label("Motion", "Mot")
      .floatField("dive_contrast_bias", 1.0f, 0.0f, 5.0f, state::PrimaryInput, nullptr, 0.01f,
                  nullptr, "Contrast/brightness ceiling of the grain at full dive.")
                  .label("Grain Contrast", "Grain")
      .floatField("dive_cap", 1.0f, 0.0f, 1.0f, state::PrimaryInput, nullptr, 0.01f,
                  nullptr, "Upper clamp on Dive.").label("Dive Cap", "Cap")
      .floatField("edge_artifacts", 0.0f, 0.0f, 1.0f, state::PrimaryInput, nullptr, 0.01f,
                  nullptr, "Reproduce Pixulant's bright bottom-edge grain (0 = clean).")
                  .label("Edge Artifacts", "Edge")
      .capability(state::Capability::SeekableApproximate)
      .textureField("tex_in",  state::PrimaryInput)
      .textureField("tex_out", state::PrimaryOutput));

  if (gpu::Device::backend() == gpu::Backend::None) return;

  state::registerShaderSPV("smear_blur", BLUR_SPV, BLUR_SPV_SIZE, "rgba8unorm", "write");
  state::registerShaderSPV("smear_scatter", SCATTER_SPV, SCATTER_SPV_SIZE, "rgba8unorm", "write");
  auto cs_blur = gpu::Device::createShaderModuleByName("smear_blur");
  auto cs_scat = gpu::Device::createShaderModuleByName("smear_scatter");
  if (!cs_blur || !cs_scat) return;
  auto bindings = gpu::Bindings()
      .tex2d(0).sampler(1).storageTex2d(2, gpu::TextureFormat::RGBA8).uniform(3);
  s_pso_blur    = gpu::Device::createComputePSO(cs_blur, "main", bindings);
  s_pso_scatter = gpu::Device::createComputePSO(cs_scat, "main", bindings);

  state::log("smear: module initialized");
}

void* create() {
  auto* s = new State();
  s->uniform_major   = gpu::Device::createBuffer(sizeof(BlurUniforms), gpu::BufferUsage::Uniform);
  s->uniform_minor   = gpu::Device::createBuffer(sizeof(BlurUniforms), gpu::BufferUsage::Uniform);
  s->uniform_scatter = gpu::Device::createBuffer(sizeof(ScatterUniforms), gpu::BufferUsage::Uniform);
  s->sampler = gpu::Device::createSampler(gpu::FilterMode::Linear, gpu::AddressMode::ClampToEdge);
  return s;
}

void destroy(void* self) {
  auto* s = static_cast<State*>(self);
  if (!s) return;
  s->uniform_major.release();
  s->uniform_minor.release();
  s->uniform_scatter.release();
  s->sampler.release();
  if (s->scratch.valid()) s->scratch.release();
  delete s;
}

void init(void* self) {
  auto* s = static_cast<State*>(self);
  if (!s) return;
  s->phase = 0.0; s->last_cyc = -1; s->rand_val = 0.0f;
  state::setOnStateReady(&on_state_ready);
  if (!s_pso_blur.valid() || !s_pso_scatter.valid()) return;
  if (!s->uniform_major.valid() || !s->uniform_minor.valid() || !s->uniform_scatter.valid()) return;
  s->initialized = true;
}

void tick(void* self, double dt) {
  auto* s = static_cast<State*>(self);
  if (!s) return;
  if (dt < 0.0) dt = 0.0;
  // Saw phase advanced by Motion; Random re-rolls on each Saw cycle (Pixulant).
  s->phase += dt * (double)s->motion * (double)MOTION_RATE;
  if (s->phase > 1.0e7) s->phase = std::fmod(s->phase, 1024.0);
  long long cyc = (long long)std::floor(s->phase);
  if (cyc != s->last_cyc) {
    s->last_cyc = cyc;
    s->rand_val = hash01((uint32_t)(cyc * 2654435761LL + 1013904223LL));
  }
}

void on_state_patched(void* self, int n, const char* pb, const int* off,
                      const int* len, const int* ops) {
  auto* s = static_cast<State*>(self);
  if (!s) return;
  bool mode_changed = false;
  for (int i = 0; i < n; i++) {
    if (ops[i] != state::PatchReplace) continue;
    const char* p = pb + off[i]; int l = len[i];
    if      (state::pathIs(p, l, "mode")) {
      int m = state::patchInt(i);
      if (m != s->mode) { s->mode = m; mode_changed = true; }
    }
    else if (state::pathIs(p, l, "angle"))              s->angle              = state::patchFloat(i);
    else if (state::pathIs(p, l, "strength"))          s->strength           = state::patchFloat(i);
    else if (state::pathIs(p, l, "length"))            s->length             = state::patchFloat(i);
    else if (state::pathIs(p, l, "width"))             s->width              = state::patchFloat(i);
    else if (state::pathIs(p, l, "tail"))              s->tail               = state::patchFloat(i);
    else if (state::pathIs(p, l, "tilt"))              s->tilt               = state::patchFloat(i);
    else if (state::pathIs(p, l, "softness"))          s->softness           = state::patchFloat(i);
    else if (state::pathIs(p, l, "exposure"))          s->exposure           = state::patchFloat(i);
    else if (state::pathIs(p, l, "samples"))           s->samples            = state::patchInt(i);
    else if (state::pathIs(p, l, "dive"))              s->dive               = state::patchFloat(i);
    else if (state::pathIs(p, l, "motion"))            s->motion             = state::patchFloat(i);
    else if (state::pathIs(p, l, "dive_contrast_bias")) s->dive_contrast_bias = state::patchFloat(i);
    else if (state::pathIs(p, l, "dive_cap"))          s->dive_cap           = state::patchFloat(i);
    else if (state::pathIs(p, l, "edge_artifacts"))    s->edge_artifacts     = state::patchFloat(i);
  }
  if (s->samples < 4)  s->samples = 4;
  if (s->samples > 32) s->samples = 32;
  if (mode_changed) apply_visibility(s->mode);
}

void on_resolume_param(void* self, long long, double) { (void)self; }

// NO is_identity — scatter is tick-evolved; a stage reporting identity can be
// permanently sidelined.

static void ensure_scratch(State* s, int w, int h) {
  if (s->scratch.valid() && s->scratch_w == w && s->scratch_h == h) return;
  if (s->scratch.valid()) s->scratch.release();
  s->scratch = gpu::Device::createTexture(w, h);
  s->scratch_w = w;
  s->scratch_h = h;
}

void render(void* self, int vp_w, int vp_h) {
  auto* s = static_cast<State*>(self);
  if (!s || !s->initialized || vp_w <= 0 || vp_h <= 0) return;
  auto in  = gpu::Device::textureForField("tex_in");
  auto out = gpu::Device::textureForField("tex_out");
  if (!in.valid() || !out.valid()) return;

  // Axis basis. angle ∈ [-1,1] → θ ∈ [-π,π]. Aspect-correct the UV step so a unit
  // of reach is equal SCREEN distance in u and v.
  float th = s->angle * PI;
  float ct = std::cos(th), stt = std::sin(th);
  int   min_dim = vp_w < vp_h ? vp_w : vp_h;
  float sx = (float)min_dim / (float)vp_w;
  float sy = (float)min_dim / (float)vp_h;
  float maj_x =  ct * sx, maj_y =  stt * sy;   // major axis UV step
  float min_x = -stt * sx, min_y =  ct * sy;   // minor axis UV step

  float amt        = clamp01(s->strength);                   // master scale on both axes
  float reach_len  = s->length * amt * MAJOR_MAX;
  float reach_fwd  = reach_len * (1.0f - clamp01(s->tail));  // head shrinks with tail
  float reach_back = reach_len;
  float reach_wid  = s->width * amt * MINOR_MAX;
  float falloff_k  = 1.0f + clamp01(s->softness) * 7.0f;     // boxy → soft gaussian fade

  if (s->mode == MODE_SCATTER) {
    float dive_c = clamp01(s->dive);
    if (dive_c > s->dive_cap) dive_c = clamp01(s->dive_cap);
    float expo = 0.5f + (s->dive_contrast_bias - 0.5f) * ease(dive_c);

    float saw = (float)(s->phase - std::floor(s->phase)) * 2.0f - 1.0f; // bipolar Saw
    ScatterUniforms u = {};
    u.axis_maj_x = maj_x; u.axis_maj_y = maj_y;
    u.axis_min_x = min_x; u.axis_min_y = min_y;
    u.reach_fwd = reach_fwd; u.reach_back = reach_back; u.width = reach_wid;
    u.salt_base = saw + s->rand_val;
    u.major_x = ct; u.major_y = stt; u.tilt = s->tilt;
    u.dive = dive_c;
    u.exposure_gain = std::exp2(expo);
    u.edge_artifacts = s->edge_artifacts;   // raw [0,1]; shader bakes the 0.1 sensitivity
                                            // AND uses it to unmask off-frame taps
    u.exposure = s->exposure;
    u.softness = clamp01(s->softness);
    s->uniform_scatter.writeOne(u);

    auto cp = gpu::ComputePass::begin();
    cp.setPSO(s_pso_scatter);
    cp.setTexture(in, 0, 0);
    cp.setSampler(s->sampler, 1);
    cp.setTexture(out, 2, 1);
    cp.setBuffer(s->uniform_scatter, 3);
    cp.dispatch((vp_w + 7) / 8, (vp_h + 7) / 8);
    cp.end();
    gpu::Device::submit();
    return;
  }

  // Blur mode — two separable passes through the scratch texture. Both passes run
  // even at zero reach (a zero-reach pass is just a copy), so tex_out is always
  // produced without a copy-to-intermediate.
  ensure_scratch(s, vp_w, vp_h);
  if (!s->scratch.valid()) return;

  BlurUniforms um = {};                 // pass 1: major axis, asymmetric tail, no tilt
  um.axis_x = maj_x; um.axis_y = maj_y;
  um.reach_fwd = reach_fwd; um.reach_back = reach_back;
  um.major_x = ct; um.major_y = stt; um.tilt = 0.0f;
  um.falloff_k = falloff_k; um.exposure = 1.0f;  // exposure applied on the final pass only
  um.samples = s->samples;
  s->uniform_major.writeOne(um);

  BlurUniforms un = {};                 // pass 2: minor axis, symmetric, perspective
  un.axis_x = min_x; un.axis_y = min_y;
  un.reach_fwd = reach_wid; un.reach_back = reach_wid;
  un.major_x = ct; un.major_y = stt; un.tilt = s->tilt;
  un.falloff_k = falloff_k; un.exposure = s->exposure;
  un.samples = s->samples;
  s->uniform_minor.writeOne(un);

  {
    auto cp = gpu::ComputePass::begin();
    cp.setPSO(s_pso_blur);
    cp.setTexture(in, 0, 0);
    cp.setSampler(s->sampler, 1);
    cp.setTexture(s->scratch, 2, 1);
    cp.setBuffer(s->uniform_major, 3);
    cp.dispatch((vp_w + 7) / 8, (vp_h + 7) / 8);
    cp.end();
  }
  {
    auto cp = gpu::ComputePass::begin();
    cp.setPSO(s_pso_blur);
    cp.setTexture(s->scratch, 0, 0);
    cp.setSampler(s->sampler, 1);
    cp.setTexture(out, 2, 1);
    cp.setBuffer(s->uniform_minor, 3);
    cp.dispatch((vp_w + 7) / 8, (vp_h + 7) / 8);
    cp.end();
  }

  gpu::Device::submit();
}

} // namespace smear
