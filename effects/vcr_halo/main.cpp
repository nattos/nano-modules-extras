/*
 * filter.glow.vcr_halo — the Layer^3 look, applied to an arbitrary image.
 *
 * source.mesh.three_planes gets its halo for free: it knows the exact signed
 * distance to every outline, so `exp(-d/r)` is a closed form and widening the
 * glow costs nothing. A post-process has no geometry, so the same profile has
 * to be CONVOLVED. That is the one real concession — this effect cannot be as
 * morphologically exact, and it is allowed to shimmer under motion where the
 * analytic version is rock solid.
 *
 * Everything AFTER the halo is shared, byte for byte: both effects accumulate
 * in linear HDR, split the channels horizontally, and hand the result to the
 * single nano_vcr_grade() in shaders_common/nano_vcr.hlsl. That header is the
 * seam — the look lives there, halo generation is per-effect.
 *
 * The pyramid (Jimenez, SIGGRAPH 2014 — 13-tap down, 9-tap tent up) is a
 * progressive upsample that folds each octave in on the way back down:
 *
 *     S[N-1] = w[N-1] * D[N-1]
 *     S[k]   = tent(S[k+1]) + w[k] * D[k]
 *     glow   = S[0]  ==  sum_k w[k] * blur_k(emitter)
 *
 * which is the same shape as the analytic
 * `sum_j a_j * exp(-d / (R * s_j))` — a weighted stack of kernels at
 * geometrically spaced radii. Here the radii are fixed at octaves and the
 * RADIUS knob slides weight between neighbouring levels instead, so it stays
 * continuously modulatable with no level popping in.
 *
 * Cost: the pyramid is geometric and starts at half res, so the whole chain
 * is ~0.67x of one fullscreen pass. Measured through
 * `benchmark_barrel --module filter.glow.vcr_halo` (marginal ms per added
 * instance, M-series):
 *
 *              1080p    4K
 *   vcr_halo   0.219   0.941
 *   blur.fast  0.118   0.469     <- ~2x a dual-filter blur
 *   gaussian   1.353  10.208     <- ~6-11x more expensive than we are
 *   3 planes   0.440   1.925     <- the ANALYTIC sibling costs twice this
 *
 * and BASICALLY FLAT IN RADIUS (0.237 at radius 0, 0.196 at radius 1 —
 * levels below a thousandth of the peak weight are skipped, but a geometric
 * pyramid's tail was nearly free anyway). Flat is the useful property here:
 * sweeping the halo from an envelope cannot spike a frame. The one real
 * saving is Halo Gain at 0, which skips the chain outright for ~45% off.
 */

#include <gpu.h>
#include <host.h>
#include "vcr_halo_shaders.h"

#include <cmath>
#include <cstdint>

namespace vcr_halo {

// 7 octaves reaches ~192px of blur radius, past the widest the radius knob
// maps to at 1080p. Anything more is memory for no reach.
static constexpr int MAX_LEVELS = 7;

// --- Uniform blocks (mirror the cbuffers row for row) ---------------------

struct PrefilterUniforms {
  float src_texel[2];
  float threshold;
  float knee;
  float saturation;
  float _pad0;
  float _pad1;
  float _pad2;
  float tint[4];
};
static_assert(sizeof(PrefilterUniforms) == 48, "prefilter.hlsl cbuffer mismatch");

// Shared by down.hlsl and up.hlsl (down ignores everything but src_texel).
struct PyramidUniforms {
  float src_texel[2];
  float src_scale;
  float add_weight;
  float outline;
  float _pad0;
  float _pad1;
  float _pad2;
};
static_assert(sizeof(PyramidUniforms) == 32, "down/up.hlsl cbuffer mismatch");

struct CompositeUniforms {
  float vp[2];
  float chroma_uv_off;
  float debug_mode;
  float input_gain;
  float halo_gain;
  float _pad0;
  float _pad1;
  float grade[16];
};
static_assert(sizeof(CompositeUniforms) == 96, "composite.hlsl cbuffer mismatch");

struct State {
  // --- Halo ---
  float halo_gain       = 3.00f;
  float halo_radius     = 0.30f;
  float halo_falloff    = 0.45f;
  float halo_compensate = 1.00f;
  float threshold       = 0.25f;
  float knee            = 0.50f;
  float outline         = 0.00f;
  float glow_saturation = 1.30f;
  float glow_tint[3]    = {1.0f, 1.0f, 1.0f};

  // --- Image ---
  float input_gain = 1.0f;

  // --- Grade (the nano_vcr.hlsl stack; defaults match three_planes) ---
  float exposure        = 1.0f;
  float warmth          = 0.35f;
  float drive           = 0.35f;
  float asymmetry       = 0.20f;
  float toe             = 0.25f;
  float shoulder        = 0.50f;
  float highlight_desat = 0.70f;
  float highlight_tint[3] = {1.00f, 0.22f, 0.62f};
  float highlight_tint_amount = 0.0f;
  float highlight_tint_pivot  = 1.0f;
  float chroma_bleed    = 0.25f;
  float scanline        = 0.12f;
  int   scanline_count  = 240;
  float grain           = 0.08f;

  // --- Debug ---
  bool debug_show_halo    = false;
  bool debug_show_emitter = false;

  // --- Resources ---
  gpu::Texture down[MAX_LEVELS];      // D[k], the octave pyramid
  gpu::Texture up[MAX_LEVELS];        // S[k], the progressive accumulation
  int level_w[MAX_LEVELS] = {};
  int level_h[MAX_LEVELS] = {};
  int level_count = 0;
  int tex_w = 0, tex_h = 0;

  gpu::Buffer prefilter_buf;
  gpu::Buffer pyramid_bufs[(MAX_LEVELS - 1) * 2];
  gpu::Buffer composite_buf;

  bool initialized = false;
};

static gpu::ComputePSO s_pso_prefilter;
static gpu::ComputePSO s_pso_down;
static gpu::ComputePSO s_pso_up;
static gpu::ComputePSO s_pso_composite;
static gpu::Sampler    s_sampler;

static inline float clamp01(float v) { return v < 0.f ? 0.f : (v > 1.f ? 1.f : v); }
static inline int half_up(int v) { int h = (v + 1) / 2; return h < 1 ? 1 : h; }

// --- The radius model -----------------------------------------------------

// Characteristic blur radius of pyramid level k, in FULL-RES pixels. Level 0
// is the half-res 13-tap downsample, which reaches about 3 full-res pixels
// once it has been tented back up; every octave after that doubles.
static inline float levelRadiusPx(int k) {
  return 3.0f * float(1 << k);
}

// Deliberately the SAME curve as three_planes' haloRadius(), converted from
// cover-square units to pixels, so the two effects' Halo Radius sliders agree
// on what "0.4" means. Resolution-relative, so the look holds at any size.
static inline float haloRadiusPx(float t, int vp_w, int vp_h) {
  const float cover_half = 0.5f * float(vp_w > vp_h ? vp_w : vp_h);
  return 0.006f * std::pow(40.0f, clamp01(t)) * cover_half;
}

// Where the compensation below is defined to be exactly 1.0.
static constexpr float kRadiusRef = 0.30f;

/**
 * Radius compensation — the one place the convolution has to be told what the
 * analytic version knows for free.
 *
 * three_planes evaluates `exp(-d/r)` against the distance to the outline, so
 * the halo is at FULL strength on the line no matter how wide it is: widening
 * `r` adds energy. A convolution spreads a fixed amount of light instead, and
 * for a thin line the peak of an isotropic blur falls as 1/r — so with plain
 * energy normalisation the glow fades out exactly as you ask for more of it,
 * which is backwards.
 *
 * Scaling the gain by r/r_ref restores it: constant glow brightness for a
 * LINE as the radius sweeps, which is what makes this knob feel like the
 * geometric one. It is not universal — the same compensation over-brightens a
 * large bright FIELD (whose blurred peak was already radius-independent), so
 * this is a knob rather than a constant. At 0 the halo is purely
 * energy-conserving.
 */
static inline float radiusCompensation(float radius_px, float ref_px, float amount) {
  if (ref_px <= 1e-6f) return 1.0f;
  return std::pow(radius_px / ref_px, clamp01(amount));
}

/**
 * Per-level weights: a log-domain lobe centred on the target radius.
 *
 * Working in log2 is the whole point. The pyramid's radii are geometric, so a
 * Gaussian in log-radius slides smoothly across levels as the knob sweeps —
 * the weight leaves one octave exactly as fast as it arrives at the next, and
 * nothing pops. `falloff` widens the lobe: narrow is one dominant octave
 * (tight and punchy), wide spreads across four or five (the stacked-kernel
 * profile that actually reads as glow).
 *
 * Returns the number of levels worth dispatching — trailing levels whose
 * weight is under a thousandth of the peak are dropped. That saves little
 * (a geometric pyramid's tail is tiny), but it keeps a tight halo from
 * running dispatches whose output is provably invisible.
 */
static int computeWeights(float radius_px, float falloff, int avail, float* w) {
  const float sigma = 0.60f + 1.20f * clamp01(falloff);
  const float lr = std::log2(radius_px > 1e-3f ? radius_px : 1e-3f);

  int used = 1;
  for (int k = 0; k < avail; k++) {
    const float d = std::log2(levelRadiusPx(k)) - lr;
    w[k] = std::exp(-(d * d) / (2.0f * sigma * sigma));
    if (w[k] > 1e-3f) used = k + 1;
  }

  float sum = 0.0f;
  for (int k = 0; k < used; k++) sum += w[k];
  if (sum <= 1e-6f) {
    for (int k = 0; k < avail; k++) w[k] = 0.0f;
    w[0] = 1.0f;
    return 1;
  }
  // Normalised, so sliding the radius redistributes energy instead of adding
  // it — the halo widens and softens at constant total brightness rather than
  // getting hotter as it grows.
  for (int k = 0; k < used; k++) w[k] /= sum;
  return used;
}

// --- Lifecycle ------------------------------------------------------------

void module_init() {
  state::init("filter.glow.vcr_halo", {1, 0, 0},
    state::Schema()
      .helpField("intro",
        "## VCR Halo\n"
        "The *Three Planes* look, unbolted from its geometry and pointed at "
        "whatever you feed it — a warm neon bloom plus the analogue "
        "dehancement tail (soft clip, toe/shoulder, chroma split, scanlines, "
        "grain).\n\n"
        "The halo is a multi-octave glow pyramid, so *Halo Radius* is "
        "continuous and a wide halo is barely more expensive than a narrow "
        "one. It is a convolution, though, which means it cannot know where "
        "an outline *is* the way the geometric version does: point it at a "
        "filled shape and you get a soft lump. **Outline** is the fix — it "
        "band-passes the emitter so only edges glow.\n\n"
        "**Try:** Threshold up until only the highlights survive, Outline "
        "around 0.6, Halo Radius wide, Chroma Bleed up. On typography or "
        "line art that is the sign-in-the-rain look in three knobs.")

      // ---------------- Halo ----------------
      .group("halo", "Halo")
        .groupHelp(
          "**Threshold** and **Knee** decide what is allowed to glow; turn on "
          "*Show Emitter* in Debug and set them there rather than guessing "
          "through the grade. Remember the input is usually already tone "
          "mapped, so its brightest pixel is 1.0 and a threshold near that "
          "leaves nothing. **Outline** then chooses between blooming those "
          "bright areas whole (0) and blooming only their edges (1).\n\n"
          "**Halo Radius** slides weight across the pyramid's octaves, so it "
          "modulates smoothly and never pops a level in. By default the glow "
          "holds its brightness as it widens (see *Radius Compensation*), the "
          "way the geometric version does; pull that to 0 and the halo "
          "conserves energy instead, spreading and dimming.\n\n"
          "The defaults are tuned for line art and neon, where the bright "
          "areas are thin. On thick, already-bright content that gain clips "
          "the bodies out — push *Threshold* up so only the real highlights "
          "glow, or *Outline* up so the bodies stop emitting at all and only "
          "their edges do.")
      .floatField("halo_gain", 3.00f, 0.f, 8.f, state::PrimaryInput)
        .label("Halo Gain", "Halo")
      .floatField("halo_radius", 0.30f, 0.f, 1.f, state::PrimaryInput)
        .label("Halo Radius", "Halo R")
      .floatField("threshold", 0.25f, 0.f, 2.f, state::PrimaryInput,
                  nullptr, 0.f, nullptr,
                  "How bright a pixel has to be before it glows. The range "
                  "runs past 1.0 for HDR chains, but an ordinary tone-mapped "
                  "image tops out AT 1.0 — set this high on one and almost "
                  "nothing survives to glow.")
        .label("Threshold", "Thresh")
      .floatField("outline", 0.00f, 0.f, 1.f, state::PrimaryInput,
                  nullptr, 0.f, nullptr,
                  "0 blooms bright areas whole; 1 blooms only their edges, "
                  "which is what makes a filled shape read as neon tubing. "
                  "The edge it finds widens with Halo Radius, so the two "
                  "knobs stay in step.")
        .label("Outline", "Outline")
      .floatField("knee", 0.50f, 0.01f, 1.f, state::SecondaryInput,
                  nullptr, 0.f, nullptr,
                  "Softness of the threshold shoulder. Low is a hard gate "
                  "that pops as things brighten; high fades the glow in.")
        .label("Knee", "Knee")
      .floatField("halo_compensate", 1.00f, 0.f, 1.f, state::SecondaryInput,
                  nullptr, 0.f, nullptr,
                  "How the glow behaves as Halo Radius widens. 1 holds a "
                  "LINE's glow at constant brightness (matches Three Planes); "
                  "0 conserves energy instead, so it spreads and dims — safer "
                  "on large bright areas.")
        .label("Radius Compensation", "Comp")
      .floatField("halo_falloff", 0.45f, 0.f, 1.f, state::SecondaryInput,
                  nullptr, 0.f, nullptr,
                  "0 = one tight octave, punchy. 1 = spread across many, the "
                  "soft stacked profile that reads as real glow.")
        .label("Halo Falloff", "Fall")
      .floatField("glow_saturation", 1.30f, 0.f, 2.f, state::SecondaryInput,
                  nullptr, 0.f, nullptr,
                  "Chroma of the halo only. Above 1 the glow is more "
                  "saturated than the light that made it — desaturated glow "
                  "reads as fog, not neon.")
        .label("Glow Saturation", "Glow Sat")
      .rgbField("glow_tint", 1.0f, 1.0f, 1.0f, state::SecondaryInput)
        .label("Glow Tint", "Tint")

      // ---------------- Image ----------------
      .group("image", "Image")
      .floatField("input_gain", 1.0f, 0.f, 2.f, state::PrimaryInput,
                  nullptr, 0.f, nullptr,
                  "Gain on the source before the grade. Push past 1 to drive "
                  "highlights into the bleach; drop to 0 for the halo alone.")
        .label("Input Gain", "In")

      // ---------------- Grade ----------------
      .group("grade", "Warmth & Dehancement")
        .groupHelp(
          "The same stack Three Planes wears, from the shared "
          "`nano_vcr.hlsl` — set both effects the same and they match.\n\n"
          "*Drive* and *Asymmetry* are where warmth actually lives: the "
          "asymmetric bias makes even and odd harmonics unequal instead of "
          "just rounding the peaks. *Highlight Desat* runs before the curve, "
          "in HDR, so hot cores bleach to white properly.")
      .floatField("chroma_bleed", 0.25f, 0.f, 1.f, state::PrimaryInput)
        .label("Chroma Bleed", "Chroma")
      .floatField("warmth", 0.35f, -1.f, 1.f, state::PrimaryInput, "signed")
        .label("Warmth", "Warm")
      .floatField("drive", 0.35f, 0.f, 1.f, state::PrimaryInput)
        .label("Drive", "Drive")
      .floatField("exposure", 1.0f, 0.f, 2.f, state::PrimaryInput)
        .label("Exposure", "Expo")
      .floatField("asymmetry", 0.20f, -1.f, 1.f, state::SecondaryInput, "signed")
        .label("Asymmetry", "Asym")
      .floatField("toe", 0.25f, 0.f, 1.f, state::SecondaryInput)
        .label("Toe", "Toe")
      .floatField("shoulder", 0.50f, 0.f, 1.f, state::SecondaryInput)
        .label("Shoulder", "Shldr")
      .floatField("highlight_desat", 0.70f, 0.f, 1.f, state::SecondaryInput)
        .label("Highlight Desat", "HiDesat")
      .rgbField("highlight_tint", 1.00f, 0.22f, 0.62f, state::SecondaryInput)
        .label("Highlight Tint", "Hi Tint")
      .floatField("highlight_tint_amount", 0.0f, 0.f, 1.f, state::PrimaryInput,
                  nullptr, 0.f, nullptr,
                  "Colours the blown-out cores that Highlight Desat just "
                  "bleached white. The swatch is what a fully clipped pixel "
                  "BECOMES, so what you pick is what you get — dim it for a "
                  "deeper, more saturated core, keep it hot for a tinted "
                  "white one.")
        .label("Highlight Tint Amount", "Tint Amt")
      .floatField("highlight_tint_pivot", 1.0f, 0.2f, 4.f, state::SecondaryInput,
                  nullptr, 0.f, nullptr,
                  "Where the tint starts biting, and how much of the image it "
                  "catches. 1.0 is exactly at clipping and the tint arrives "
                  "fully a stop above that; drop it to pull colour into "
                  "highlights that would have survived the tone map intact.")
        .label("Tint Pivot", "Pivot")
      .floatField("scanline", 0.12f, 0.f, 1.f, state::SecondaryInput)
        .label("Scanlines", "Scan")
      .intField("scanline_count", 240, 30, 720, state::SecondaryInput, 0, "lines")
        .label("Scanline Count", "Lines")
      .floatField("grain", 0.08f, 0.f, 1.f, state::SecondaryInput)
        .label("Grain", "Grain")

      // ---------------- Debug ----------------
      .group("debug", "Debug")
      .boolField("debug_show_halo", false, state::SecondaryInput,
                 "The halo alone on black, ungraded — the only honest way to "
                 "judge radius and falloff.")
        .label("Show Halo", "Halo")
      .boolField("debug_show_emitter", false, state::SecondaryInput,
                 "What was allowed to glow. Set Threshold / Knee / Outline "
                 "here.")
        .label("Show Emitter", "Emit")

      .textureField("tex_in",  state::PrimaryInput)
      .textureField("tex_out", state::PrimaryOutput)

      // The grain is derived from absolute host time rather than an
      // accumulator, and nothing else here carries frame-to-frame state, so a
      // scrub lands on exactly the right frame.
      .capability(state::Capability::TimeIndependent)
  );

  if (gpu::Device::backend() == gpu::Backend::None) return;

  // The pyramid is HDR (a bright core has to survive being spread over a
  // hundred pixels and come back), so every intermediate is pinned RGBA16F
  // and naga's default rgba32float storage format has to be overridden.
  // The composite writes tex_out, which follows the sketch's format.
  state::registerShaderSPV("vcr_halo_prefilter", PREFILTER_SPV, PREFILTER_SPV_SIZE, "rgba16float", "write");
  state::registerShaderSPV("vcr_halo_down",      DOWN_SPV,      DOWN_SPV_SIZE,      "rgba16float", "write");
  state::registerShaderSPV("vcr_halo_up",        UP_SPV,        UP_SPV_SIZE,        "rgba16float", "write");
  state::registerShaderSPV("vcr_halo_composite", COMPOSITE_SPV, COMPOSITE_SPV_SIZE);

  auto cs_pre  = gpu::Device::createShaderModuleByName("vcr_halo_prefilter");
  auto cs_down = gpu::Device::createShaderModuleByName("vcr_halo_down");
  auto cs_up   = gpu::Device::createShaderModuleByName("vcr_halo_up");
  auto cs_comp = gpu::Device::createShaderModuleByName("vcr_halo_composite");
  if (!cs_pre || !cs_down || !cs_up || !cs_comp) return;

  s_pso_prefilter = gpu::Device::createComputePSO(cs_pre, "main", gpu::Bindings()
      .tex2d(0)                                       // tex_in
      .storageTex2d(1, gpu::TextureFormat::RGBA16F)   // D[0]
      .sampler(2)
      .uniform(3));

  s_pso_down = gpu::Device::createComputePSO(cs_down, "main", gpu::Bindings()
      .tex2d(0)                                       // D[k-1]
      .storageTex2d(1, gpu::TextureFormat::RGBA16F)   // D[k]
      .sampler(2)
      .uniform(3));

  s_pso_up = gpu::Device::createComputePSO(cs_up, "main", gpu::Bindings()
      .tex2d(0)                                       // S[k+1] (or D[N-1])
      .tex2d(1)                                       // D[k]
      .tex2d(2)                                       // D[k+1] (band-pass)
      .storageTex2d(3, gpu::TextureFormat::RGBA16F)   // S[k]
      .sampler(4)
      .uniform(5));

  s_pso_composite = gpu::Device::createComputePSO(cs_comp, "main", gpu::Bindings()
      .tex2d(0)                                       // tex_in
      .tex2d(1)                                       // S[0]
      .tex2d(2)                                       // D[0] (debug)
      .storageTex2d(3)                                // tex_out
      .sampler(4)
      .uniform(5));

  // ClampToEdge matters: Repeat would wrap a bright edge's glow around to the
  // far side of the frame, and Mirror would double it back on itself.
  s_sampler = gpu::Device::createSampler(gpu::FilterMode::Linear,
                                         gpu::AddressMode::ClampToEdge);

  state::log("vcr_halo: module initialized");
}

void* create() {
  auto* s = new State();
  s->prefilter_buf = gpu::Device::createBuffer(sizeof(PrefilterUniforms), gpu::BufferUsage::Uniform);
  s->composite_buf = gpu::Device::createBuffer(sizeof(CompositeUniforms), gpu::BufferUsage::Uniform);
  // One buffer per dispatch — writing the same uniform buffer twice inside a
  // single submit would make the second write win for both passes.
  for (auto& b : s->pyramid_bufs) {
    b = gpu::Device::createBuffer(sizeof(PyramidUniforms), gpu::BufferUsage::Uniform);
  }
  return s;
}

void destroy(void* self) {
  auto* s = static_cast<State*>(self);
  if (!s) return;
  for (int k = 0; k < MAX_LEVELS; k++) { s->down[k].release(); s->up[k].release(); }
  s->prefilter_buf.release();
  s->composite_buf.release();
  for (auto& b : s->pyramid_bufs) b.release();
  delete s;
}

void init(void* self) {
  auto* s = static_cast<State*>(self);
  if (!s) return;
  if (!s_pso_composite.valid() || !s->composite_buf.valid()) return;
  s->initialized = true;
}

void tick(void* self, double dt) { (void)self; (void)dt; }

void on_state_patched(void* self, int n, const char* pb, const int* off,
                      const int* len, const int* ops) {
  auto* s = static_cast<State*>(self);
  if (!s) return;

  for (int i = 0; i < n; i++) {
    if (ops[i] != state::PatchReplace) continue;
    const char* p = pb + off[i];
    const int   l = len[i];

    if      (state::pathIs(p, l, "halo_gain"))       s->halo_gain = state::patchFloat(i);
    else if (state::pathIs(p, l, "halo_radius"))     s->halo_radius = state::patchFloat(i);
    else if (state::pathIs(p, l, "halo_falloff"))    s->halo_falloff = state::patchFloat(i);
    else if (state::pathIs(p, l, "halo_compensate")) s->halo_compensate = state::patchFloat(i);
    else if (state::pathIs(p, l, "threshold"))       s->threshold = state::patchFloat(i);
    else if (state::pathIs(p, l, "knee"))            s->knee = state::patchFloat(i);
    else if (state::pathIs(p, l, "outline"))         s->outline = state::patchFloat(i);
    else if (state::pathIs(p, l, "glow_saturation")) s->glow_saturation = state::patchFloat(i);
    else if (state::pathIs(p, l, "glow_tint")) {
      auto v = state::patchVec3(i);
      s->glow_tint[0] = v.x; s->glow_tint[1] = v.y; s->glow_tint[2] = v.z;
    }
    else if (state::pathIs(p, l, "input_gain"))      s->input_gain = state::patchFloat(i);

    else if (state::pathIs(p, l, "chroma_bleed"))    s->chroma_bleed = state::patchFloat(i);
    else if (state::pathIs(p, l, "warmth"))          s->warmth = state::patchFloat(i);
    else if (state::pathIs(p, l, "drive"))           s->drive = state::patchFloat(i);
    else if (state::pathIs(p, l, "exposure"))        s->exposure = state::patchFloat(i);
    else if (state::pathIs(p, l, "asymmetry"))       s->asymmetry = state::patchFloat(i);
    else if (state::pathIs(p, l, "toe"))             s->toe = state::patchFloat(i);
    else if (state::pathIs(p, l, "shoulder"))        s->shoulder = state::patchFloat(i);
    else if (state::pathIs(p, l, "highlight_desat")) s->highlight_desat = state::patchFloat(i);
    else if (state::pathIs(p, l, "highlight_tint")) {
      auto v = state::patchVec3(i);
      s->highlight_tint[0] = v.x; s->highlight_tint[1] = v.y; s->highlight_tint[2] = v.z;
    }
    else if (state::pathIs(p, l, "highlight_tint_amount")) s->highlight_tint_amount = state::patchFloat(i);
    else if (state::pathIs(p, l, "highlight_tint_pivot"))  s->highlight_tint_pivot = state::patchFloat(i);
    else if (state::pathIs(p, l, "scanline"))        s->scanline = state::patchFloat(i);
    else if (state::pathIs(p, l, "scanline_count"))  s->scanline_count = state::patchInt(i);
    else if (state::pathIs(p, l, "grain"))           s->grain = state::patchFloat(i);

    else if (state::pathIs(p, l, "debug_show_halo"))    s->debug_show_halo = state::patchBool(i);
    else if (state::pathIs(p, l, "debug_show_emitter")) s->debug_show_emitter = state::patchBool(i);
  }
}

// --- Resources ------------------------------------------------------------

static bool ensure_textures(State* s, int vp_w, int vp_h) {
  if (s->tex_w == vp_w && s->tex_h == vp_h && s->down[0].valid()) return true;

  for (int k = 0; k < MAX_LEVELS; k++) { s->down[k].release(); s->up[k].release(); }
  s->level_count = 0;

  int w = half_up(vp_w), h = half_up(vp_h);
  for (int k = 0; k < MAX_LEVELS; k++) {
    // Stop before the taps degenerate: a 13-tap kernel on a 3px-wide level is
    // just a very expensive average.
    if (k > 0 && (w < 4 || h < 4)) break;
    s->down[k] = gpu::Device::createTexture(w, h, gpu::TextureFormat::RGBA16F);
    if (!s->down[k].valid()) return false;
    // S[k] only exists for levels the up chain writes into.
    if (k < MAX_LEVELS - 1) {
      s->up[k] = gpu::Device::createTexture(w, h, gpu::TextureFormat::RGBA16F);
      if (!s->up[k].valid()) return false;
    }
    s->level_w[k] = w;
    s->level_h[k] = h;
    s->level_count = k + 1;
    w = half_up(w);
    h = half_up(h);
  }
  if (s->level_count < 1) return false;

  // The skip path (halo gain 0, no debug) binds D[0] and S[0] without having
  // written them this frame. Clear once so that is defined rather than
  // whatever the allocator handed us.
  gpu::Device::clear(s->down[0], 0.f, 0.f, 0.f, 1.f);
  gpu::Device::clear(s->up[0],   0.f, 0.f, 0.f, 1.f);

  s->tex_w = vp_w;
  s->tex_h = vp_h;
  return true;
}

void render(void* self, int vp_w, int vp_h) {
  auto* s = static_cast<State*>(self);
  if (!s || !s->initialized || vp_w <= 0 || vp_h <= 0) return;

  auto in  = gpu::Device::textureForField("tex_in");
  auto out = gpu::Device::textureForField("tex_out");
  if (!in.valid() || !out.valid()) return;
  if (!ensure_textures(s, vp_w, vp_h)) return;

  const int debug = s->debug_show_halo ? 1 : (s->debug_show_emitter ? 2 : 0);
  const bool need_glow = (s->halo_gain > 0.0f) || (debug != 0);

  const float radius_px = haloRadiusPx(s->halo_radius, vp_w, vp_h);

  int N = 1;
  if (need_glow) {
    float w[MAX_LEVELS] = {};
    N = computeWeights(radius_px, s->halo_falloff, s->level_count, w);

    // --- Prefilter: tex_in (full res) -> D[0] (half res) ---
    {
      PrefilterUniforms pu = {};
      pu.src_texel[0] = 1.0f / float(vp_w);
      pu.src_texel[1] = 1.0f / float(vp_h);
      pu.threshold    = s->threshold;
      pu.knee         = s->knee;
      pu.saturation   = s->glow_saturation;
      pu.tint[0] = s->glow_tint[0];
      pu.tint[1] = s->glow_tint[1];
      pu.tint[2] = s->glow_tint[2];
      pu.tint[3] = 1.0f;
      s->prefilter_buf.writeOne(pu);

      auto cp = gpu::ComputePass::begin();
      cp.setPSO(s_pso_prefilter);
      cp.setTexture(in, 0, 0);
      cp.setTexture(s->down[0], 1, 1);
      cp.setSampler(s_sampler, 2);
      cp.setBuffer(s->prefilter_buf, 3);
      cp.dispatch((s->level_w[0] + 7) / 8, (s->level_h[0] + 7) / 8);
      cp.end();
    }

    int ui = 0;

    // --- Down chain: D[k-1] -> D[k] ---
    for (int k = 1; k < N; k++) {
      PyramidUniforms pu = {};
      pu.src_texel[0] = 1.0f / float(s->level_w[k - 1]);
      pu.src_texel[1] = 1.0f / float(s->level_h[k - 1]);
      s->pyramid_bufs[ui].writeOne(pu);

      auto cp = gpu::ComputePass::begin();
      cp.setPSO(s_pso_down);
      cp.setTexture(s->down[k - 1], 0, 0);
      cp.setTexture(s->down[k], 1, 1);
      cp.setSampler(s_sampler, 2);
      cp.setBuffer(s->pyramid_bufs[ui], 3);
      cp.dispatch((s->level_w[k] + 7) / 8, (s->level_h[k] + 7) / 8);
      cp.end();
      ui++;
    }

    // --- Up chain: S[k] = tent(S[k+1]) + w[k] * D[k], seeded from D[N-1] ---
    for (int k = N - 2; k >= 0; k--) {
      const bool seed = (k == N - 2);
      const gpu::Texture& src = seed ? s->down[k + 1] : s->up[k + 1];

      PyramidUniforms pu = {};
      pu.src_texel[0] = 1.0f / float(s->level_w[k + 1]);
      pu.src_texel[1] = 1.0f / float(s->level_h[k + 1]);
      // The coarsest level's own weight rides in on the seed pass instead of
      // needing a separate scaling dispatch.
      // At full outline the coarsest level has no coarser neighbour to
      // subtract, so its low-pass residue is dropped instead — otherwise a
      // flat field would still leave a DC glow behind.
      pu.src_scale  = seed ? w[k + 1] * (1.0f - clamp01(s->outline)) : 1.0f;
      pu.add_weight = w[k];
      pu.outline    = s->outline;
      s->pyramid_bufs[ui].writeOne(pu);

      auto cp = gpu::ComputePass::begin();
      cp.setPSO(s_pso_up);
      cp.setTexture(src, 0, 0);
      cp.setTexture(s->down[k], 1, 0);
      cp.setTexture(s->down[k + 1], 2, 0);
      cp.setTexture(s->up[k], 3, 1);
      cp.setSampler(s_sampler, 4);
      cp.setBuffer(s->pyramid_bufs[ui], 5);
      cp.dispatch((s->level_w[k] + 7) / 8, (s->level_h[k] + 7) / 8);
      cp.end();
      ui++;
    }
  }

  // With a single level there is no up chain — D[0] already carries the whole
  // (normalised) glow.
  const gpu::Texture& glow = (need_glow && N == 1) ? s->down[0] : s->up[0];

  CompositeUniforms cu = {};
  cu.vp[0] = float(vp_w);
  cu.vp[1] = float(vp_h);
  // three_planes displaces by `bleed * 0.02` cover-square units; convert to uv
  // so the two effects split by the same number of pixels at any aspect.
  cu.chroma_uv_off = s->chroma_bleed * 0.02f
                   * (0.5f * float(vp_w > vp_h ? vp_w : vp_h)) / float(vp_w);
  cu.debug_mode = float(debug);
  cu.input_gain = s->input_gain;
  cu.halo_gain  = need_glow
      ? s->halo_gain * radiusCompensation(radius_px,
                                          haloRadiusPx(kRadiusRef, vp_w, vp_h),
                                          s->halo_compensate)
      : 0.0f;

  cu.grade[0]  = s->exposure;
  cu.grade[1]  = s->warmth;
  cu.grade[2]  = s->drive;
  cu.grade[3]  = s->asymmetry;
  cu.grade[4]  = s->toe;
  cu.grade[5]  = s->shoulder;
  cu.grade[6]  = s->highlight_desat;
  cu.grade[7]  = s->scanline;
  cu.grade[8]  = float(s->scanline_count);
  cu.grade[9]  = s->grain;
  // Absolute host time, not an accumulator — see TimeIndependent above.
  cu.grade[10] = float(std::fmod(host::time() * 997.0, 4096.0));
  cu.grade[11] = s->highlight_tint_pivot;
  cu.grade[12] = s->highlight_tint[0];
  cu.grade[13] = s->highlight_tint[1];
  cu.grade[14] = s->highlight_tint[2];
  cu.grade[15] = s->highlight_tint_amount;
  s->composite_buf.writeOne(cu);

  {
    auto cp = gpu::ComputePass::begin();
    cp.setPSO(s_pso_composite);
    cp.setTexture(in, 0, 0);
    cp.setTexture(glow, 1, 0);
    cp.setTexture(s->down[0], 2, 0);
    cp.setTexture(out, 3, 1);
    cp.setSampler(s_sampler, 4);
    cp.setBuffer(s->composite_buf, 5);
    cp.dispatch((vp_w + 7) / 8, (vp_h + 7) / 8);
    cp.end();
  }

  gpu::Device::submit();
}

} // namespace vcr_halo
