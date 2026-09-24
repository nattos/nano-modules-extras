/*
 * warp.legacy.pixulant — "Pixulant" (v2 of the Resolume Wire patch).
 *
 * A roiling, dissolving pixel-scatter "dive": the image is scattered three times
 * (light → mid → heavy) and the heavy copy is abs-differenced against the light
 * one, leaving coloured edge halos that bloom out of flat regions and churn over
 * time. Turning up Dive pushes the picture off the screen and INTO that grain;
 * Scatter widens the displacement; Motion sets the churn rate.
 *
 * Source patch (Wire/Patches/Pixulant, 44 nodes / 53 conns): two ISF shaders —
 * "Radial Stretch Sample" (salted per-pixel UV scatter, instanced 3× in a
 * cascade) and "Difference" (mix(rhs, abs(lhs-rhs), Alpha*2)) — plus an Exposure
 * node, fed by a Curve×10 / Multiply heavy modulation web. Exposed knobs: Dive,
 * Scatter, Scatter 2, Dive Contrast Bias, Motion, Scatter Modulate, Scatter 1
 * Modulate, Dive Cap.
 *
 * THE LOAD-BEARING QUIRK (per EFFECTS_CATALOG.md): subtracting the image from
 * itself does NOT go pure black — it leaves a halo. In the graph this is real,
 * not a Wire rounding accident: node 203 clamps the base scatter to a tiny FLOOR
 * (curve0 → 1e-4) so even at Scatter=0 the heavy pass (×0.617) displaces more
 * than the light pass (×0.12); abs(lhs-rhs) is then a thin sub-pixel edge halo
 * that the Exposure gain lifts into visible, oddly-coloured grain. We reproduce
 * the floor + the abs-difference + the exposure faithfully (SCATTER_FLOOR below).
 *
 * v2 RE-ARCHITECTURE (flagged per DNODE_MIGRATION_NOTES §3):
 *  - ONE compute pass. Each Radial Stretch pass is a pure resample, so the
 *    three-deep cascade COMPOSES: we chain the displacements and take a single
 *    input sample per side (1 for rhs, 3 for lhs) instead of three full-frame
 *    render targets. Shimmer-free; differs from the original only by the
 *    intermediate passes' bilinear filtering (sub-pixel).
 *  - The Curve×10 modulation web's preset shapes are approximated by smoothstep
 *    eases (cosmetic); the magnitudes that read live — the scatter strength
 *    ceilings (0.12 / 0.617), the floor, Dive→Difference-alpha, Dive→Exposure —
 *    are kept faithful, with all eight knobs exposed.
 *  - No actual frame feedback (the patch has no Video Mixer): the "dive" is the
 *    multi-pass scatter+difference animated by Motion, not a recursive buffer.
 *  - NEW knob `dive_rolloff` (not in the patch): below that scatter level the
 *    dive gently rolls off to zero, so one scatter automation brings the picture
 *    back — the coupling the team used to fake with a separate dive envelope.
 *    Defaults 0.15 (a gentle rolloff); 0 disables it for the faithful behaviour.
 *  - OPT-IN knob `edge_artifacts` (default 0/off): the ISF clamps the sample y
 *    to work around a Resolume bug where sampling past the bottom edge returned
 *    a constant edge value — which, surviving the abs-difference and boosted by
 *    the Exposure, made the bright bottom-edge grain the team actually liked.
 *    Turn it up to drop the clamp and reproduce that "bug" as flair. The
 *    original returned TRANSPARENT (which punched holes downstream — a recurring
 *    headache), so we use OPAQUE WHITE: same flair, no transparency. (HLSL row 0
 *    is the top, so the screen bottom is uv.y > 1 — that's where it fires.)
 *
 * Animated (Motion drift + Scatter-2 smoothing) → SeekableApproximate, and NO
 * is_identity: the state that would make it a passthrough is tick-evolved, and a
 * stage that ever reports identity can be permanently sidelined (it never runs
 * again) — so this cheap single pass always runs.
 */

#include <gpu.h>
#include <host.h>
#include "pixulant_shaders.h"

#include <cmath>
#include <cstdint>

namespace pixulant {

static constexpr float STR_RHS_MAX     = 0.12f;  // node 193 curve12 max — light pass
static constexpr float STR_LHS_MAX     = 0.617f; // node 188 curve10 max — heavy pass
static constexpr float SALT_OFF_MID    = 0.4f;   // node 192 (+0.4) — middle pass salt
static constexpr float SALT_OFF_LHS    = 0.7f;   // node 190 (+0.7) — heavy pass salt
static constexpr float SCATTER_FLOOR   = 0.003f; // node 202/203 floor (graph 1e-4, lifted
                                                 // a touch so the resting halo is visible)
static constexpr float SMOOTH_DUR      = 0.1f;   // node 234 Smooth duration (seconds)
static constexpr float MOTION_RATE     = 0.8f;   // motion=1 → 0.8 salt-cycles/sec (churn)

struct Uniforms {
  float str_rhs, str_mid, str_lhs, diff_t;
  float salt_rhs, salt_mid, salt_lhs, exposure_gain;
  float aspect, inv_h, edge_artifacts, _pad1;
};
static_assert(sizeof(Uniforms) == 48, "Uniforms layout mismatch");

struct State {
  gpu::Buffer  uniform_buf;
  gpu::Sampler sampler;
  bool initialized = false;

  // Schema-mirrored params.
  float dive               = 1.0f;
  float dive_cap           = 1.0f;
  float dive_rolloff       = 0.15f;
  float dive_contrast_bias = 1.0f;
  float scatter            = 0.0f;
  float scatter_2          = 0.0f;
  float scatter_modulate   = 1.0f;
  float scatter_1_modulate = 1.0f;
  float motion             = 0.7f;
  float edge_artifacts     = 0.0f;

  // Runtime: salt animation (Saw + on-change Random) + Scatter-2 smoothing.
  double    phase    = 0.0;
  long long last_cyc = -1;
  float     rand_val = 0.0f;
  float     smooth_s2 = 0.0f;
};

static gpu::ComputePSO s_pso;

static inline float clamp01(float x) { return x < 0.0f ? 0.0f : (x > 1.0f ? 1.0f : x); }
static inline float ease(float x)    { x = clamp01(x); return x * x * (3.0f - 2.0f * x); }

// A cheap, deterministic [0,1) hash for the on-change Random (seed=cycle index).
static inline float hash01(uint32_t v) {
  v ^= v >> 16; v *= 0x7feb352du; v ^= v >> 15; v *= 0x846ca68bu; v ^= v >> 16;
  return (float)(v & 0x00FFFFFFu) / (float)0x01000000u;
}

void module_init() {
  state::init("warp.legacy.pixulant", {1, 0, 0},
    state::Schema()
      .floatField("dive", 1.0f, 0.0f, 1.0f, state::PrimaryInput, nullptr, 0.01f,
                  nullptr, "Dive from the image (0) into the differenced grain (1).")
      .floatField("scatter", 0.0f, 0.0f, 1.0f, state::PrimaryInput, nullptr, 0.01f,
                  nullptr, "Scatter displacement amount — widens the churning grain.")
      .floatField("scatter_2", 0.0f, 0.0f, 1.0f, state::PrimaryInput, nullptr, 0.01f,
                  nullptr, "Secondary (smoothed) scatter amount, added to Scatter.")
      .floatField("motion", 0.7f, 0.0f, 1.0f, state::PrimaryInput, nullptr, 0.01f,
                  nullptr, "Churn rate — how fast the scatter field animates (0 = frozen).")
      .floatField("dive_contrast_bias", 1.0f, 0.0f, 5.0f, state::PrimaryInput, nullptr, 0.01f,
                  nullptr, "Exposure/contrast ceiling at full dive — brightness of the grain.")
      .floatField("dive_cap", 1.0f, 0.0f, 1.0f, state::PrimaryInput, nullptr, 0.01f,
                  nullptr, "Upper clamp on Dive (limits how far the dive can go).")
      .floatField("dive_rolloff", 0.15f, 0.0f, 1.0f, state::PrimaryInput, nullptr, 0.01f,
                  nullptr, "Below this scatter level, Dive (and its halo) rolls "
                  "off toward zero so the picture returns (0 = no rolloff).")
      .floatField("scatter_modulate", 1.0f, 0.0f, 1.0f, state::PrimaryInput, nullptr, 0.01f,
                  nullptr, "Overall multiplier on the combined scatter amount.")
      .floatField("scatter_1_modulate", 1.0f, 0.0f, 1.0f, state::PrimaryInput, nullptr, 0.01f,
                  nullptr, "Multiplier on the primary Scatter only.")
      .floatField("edge_artifacts", 0.0f, 0.0f, 1.0f, state::PrimaryInput, nullptr, 0.01f,
                  nullptr, "Reproduce the original's bottom-edge sampling bug as "
                  "bright white grain along the bottom (0 = clean/fixed).")
      .capability(state::Capability::SeekableApproximate)
      .textureField("tex_in",  state::PrimaryInput)
      .textureField("tex_out", state::PrimaryOutput));

  if (gpu::Device::backend() == gpu::Backend::None) return;

  state::registerShaderSPV("pixulant_pixulant", PIXULANT_SPV, PIXULANT_SPV_SIZE);
  auto cs = gpu::Device::createShaderModuleByName("pixulant_pixulant");
  if (!cs) return;
  s_pso = gpu::Device::createComputePSO(cs, "main", gpu::Bindings()
      .tex2d(0).sampler(1).storageTex2d(2).uniform(3));

  state::log("pixulant: module initialized");
}

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

void init(void* self) {
  auto* s = static_cast<State*>(self);
  if (!s) return;
  s->phase = 0.0; s->last_cyc = -1; s->rand_val = 0.0f; s->smooth_s2 = s->scatter_2;
  if (!s_pso.valid() || !s->uniform_buf.valid()) return;
  s->initialized = true;
}

void tick(void* self, double dt) {
  auto* s = static_cast<State*>(self);
  if (!s) return;
  if (dt < 0.0) dt = 0.0;

  // Smooth node (0.1s) on Scatter 2.
  float k = (SMOOTH_DUR <= 1e-5f) ? 1.0f : (1.0f - std::exp(-(float)dt / SMOOTH_DUR));
  s->smooth_s2 += (s->scatter_2 - s->smooth_s2) * k;

  // Saw phase, advanced by Motion; Random re-rolls on each Saw cycle (On Change).
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
  for (int i = 0; i < n; i++) {
    if (ops[i] != state::PatchReplace) continue;
    const char* p = pb + off[i];
    int l = len[i];
    if      (state::pathIs(p, l, "dive"))               s->dive               = state::patchFloat(i);
    else if (state::pathIs(p, l, "scatter"))            s->scatter            = state::patchFloat(i);
    else if (state::pathIs(p, l, "scatter_2"))          s->scatter_2          = state::patchFloat(i);
    else if (state::pathIs(p, l, "motion"))             s->motion             = state::patchFloat(i);
    else if (state::pathIs(p, l, "dive_contrast_bias")) s->dive_contrast_bias = state::patchFloat(i);
    else if (state::pathIs(p, l, "dive_cap"))           s->dive_cap           = state::patchFloat(i);
    else if (state::pathIs(p, l, "dive_rolloff"))       s->dive_rolloff       = state::patchFloat(i);
    else if (state::pathIs(p, l, "scatter_modulate"))   s->scatter_modulate   = state::patchFloat(i);
    else if (state::pathIs(p, l, "scatter_1_modulate")) s->scatter_1_modulate = state::patchFloat(i);
    else if (state::pathIs(p, l, "edge_artifacts"))     s->edge_artifacts     = state::patchFloat(i);
  }
}

void on_resolume_param(void* self, long long, double) { (void)self; }

// NO is_identity — see header. The passthrough condition would depend on
// tick-evolved state, and a stage that reports identity can be sidelined.

void render(void* self, int vp_w, int vp_h) {
  auto* s = static_cast<State*>(self);
  if (!s || !s->initialized || vp_w <= 0 || vp_h <= 0) return;
  auto in  = gpu::Device::textureForField("tex_in");
  auto out = gpu::Device::textureForField("tex_out");
  if (!in.valid() || !out.valid()) return;

  // Base scatter: (Scatter·Scatter1Mod + smooth(Scatter2)) · ScatterModulate
  // (nodes 228/230/232/234). This is also the rolloff's input below.
  float sc_raw  = clamp01(s->scatter * s->scatter_1_modulate + s->smooth_s2);
  float S       = clamp01(sc_raw * s->scatter_modulate);

  // Dive: clamp(Dive, 0, Dive Cap) (nodes 242/240), then the dive ROLLOFF — a v2
  // addition (not in the patch): below `dive_rolloff` scatter, the dive (and its
  // halo) gently smoothsteps to zero, so a single scatter automation brings the
  // picture back without a second envelope. dive_rolloff = the scatter level at
  // which Dive reaches full (0 = no rolloff = faithful). Uses the pre-floor S so
  // the dive can actually reach zero.
  float dive_c = clamp01(s->dive);
  if (dive_c > s->dive_cap) dive_c = clamp01(s->dive_cap);
  if (s->dive_rolloff > 1e-4f) dive_c *= ease(S / s->dive_rolloff);

  // Scatter floored to keep the resting edge halo alive while dived (203/202).
  // The floor scales with dive_c, so as the dive rolls off the halo fades too.
  float floor_s = SCATTER_FLOOR * dive_c;
  float Sc      = S > floor_s ? S : floor_s;

  float e = ease(Sc);
  // Middle-pass strength modulation (nodes 205/206/209/211/212): higher Dive +
  // higher scatter pulls the middle pass toward 0.3× (curve11 range [0.3,1]).
  float midMod  = 0.3f + 0.7f * ease(1.0f - ease(1.0f - Sc) * ease(dive_c));

  Uniforms u = {};
  u.str_rhs  = e * STR_RHS_MAX;
  u.str_lhs  = e * STR_LHS_MAX;
  u.str_mid  = e * midMod;
  u.diff_t   = dive_c;                                       // Alpha*2 = Dive
  // Exposure: curve7(Dive) remapped to [0.5, Dive Contrast Bias], as a gain.
  float expo = 0.5f + (s->dive_contrast_bias - 0.5f) * ease(dive_c); // node 199
  u.exposure_gain = std::exp2(expo);

  float saw = (float)(s->phase - std::floor(s->phase)) * 2.0f - 1.0f; // bipolar Saw
  float salt_base = saw + s->rand_val;                                // node 189
  u.salt_rhs = salt_base;
  u.salt_mid = salt_base + SALT_OFF_MID;
  u.salt_lhs = salt_base + SALT_OFF_LHS;

  u.aspect = (float)vp_w / (float)vp_h;
  u.inv_h  = 1.0f / (float)vp_h;
  u.edge_artifacts = s->edge_artifacts;
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

} // namespace pixulant
