/*
 * filter.blur.lens — "Lens".
 *
 * A GPU port of the single-plane lens simulation research harness
 * (nano-fx-prototypes/lens-sim). Models a real photographic lens as a linear-HDR
 * pipeline: highlight energy boost → shaped-aperture bokeh gather (the DOF cost
 * centre) → coating colour grade → in-frame veiling glare → off-frame spectral
 * sun/stray-light (veil + directional glow + diffraction streaks + a 6-ghost
 * internal-reflection chain) → halation+bloom → distortion + transverse chromatic
 * aberration → filmic finish (vignette, highlight desat, tonemap, grain).
 *
 * All intermediates RGBA16F, linear-HDR; tonemap only in the final pass (writes
 * the RGBA8 tex_out). The bokeh gather runs at a downsampled proc resolution; the
 * flare/glow passes are low-frequency and run reduced. Every pass is a separate
 * host dispatch so a disabled component is simply not issued (§"Skip whole
 * stages"). See lenssim/{pipeline,optics,coatings}.py for the per-pass math this
 * ports 1:1; the out*\/ study montages are the eyeball golden.
 *
 * All nine passes are implemented (ping-ponged bufA/bufB, RGBA16F linear-HDR):
 * prepare (sRGB→linear + highlight boost) → bokeh gather (Vogel-disc shaped
 * aperture: cats-eye/swirl/anamorphic/field-curvature/LoCA) → coating colour
 * grade → veiling glare (hood) → off-frame spectral sun (veil/glow/streak/ghost
 * chain + thin-film coating) → halation+bloom → distortion + transverse CA →
 * finish (exposure/vignette/hl-desat/tonemap/grain). Wide blurs use the RGBA16F-
 * safe Blur16. Performance: the bokeh gather runs at a downsampled proc
 * resolution (a FIXED tap count covers the CoC densely, then bilinear upsample);
 * the flare/glow/sun stack runs at a reduced flare resolution and upsample-adds
 * (a wide blur at full 1080p would clamp to a 128-tap kernel and wall the frame).
 *
 * Per-instance instance ABI (class-like): module_init() compiles the shared PSOs
 * + publishes the schema once per type; each chain entry gets its own State.
 * Presets are UI-only — the `preset` field is inert here (pure serialization); a
 * custom web inspector applies the character-param overrides on change.
 */

#include <gpu.h>
#include <host.h>
#include "lens_shaders.h"
#include "blur16.h"

#include <cmath>
#include <cstdint>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

namespace lens {

// --- baked coating table (coatings.py:41-55) ----------------------------------
// tint, transmission, contrast, loca, rim, onion, flare, flare_tint, coat
// designs (≤3 wavelengths nm) + count, coat_r0. Indexed by the `coating` select.
struct Coating {
  float tint[3];
  float transmission, contrast;
  float loca, rim, onion, flare;
  float flare_tint[3];
  float designs[3]; int ndesigns;
  float r0;
};
static const Coating COATINGS[4] = {
  // SMC
  {{0.99f,1.00f,1.01f}, 1.00f,1.06f, 0.30f,0.20f,0.10f,0.18f, {0.55f,0.75f,1.00f}, {455.f,545.f,630.f},3, 0.11f},
  // Single
  {{1.05f,1.00f,0.92f}, 0.92f,0.96f, 0.70f,0.60f,0.40f,0.60f, {1.00f,0.80f,0.50f}, {535.f,0.f,0.f},1, 0.24f},
  // Uncoated
  {{1.10f,1.00f,0.84f}, 0.82f,0.86f, 1.00f,1.00f,0.70f,1.00f, {1.00f,0.86f,0.60f}, {0.f,0.f,0.f},0, 0.42f},
  // Custom
  {{1.00f,1.00f,1.00f}, 1.00f,1.00f, 1.00f,1.00f,1.00f,1.00f, {1.00f,1.00f,1.00f}, {500.f,0.f,0.f},1, 0.16f},
};
static inline const Coating& coatingOf(int i) { return COATINGS[(i < 0 || i > 3) ? 3 : i]; }

// --- shader-portable helpers (mirror optics.py) -------------------------------
static inline float ss(float e0, float e1, float x) {
  float t = (x - e0) / (e1 - e0);
  t = t < 0.f ? 0.f : (t > 1.f ? 1.f : t);
  return t * t * (3.f - 2.f * t);
}
static inline float ss_down(float edge, float x, float soft) {
  float w = edge * soft; if (w < 1e-4f) w = 1e-4f;
  float t = (x - (edge - w)) / (2.f * w);
  t = t < 0.f ? 0.f : (t > 1.f ? 1.f : t);
  return 1.f - t * t * (3.f - 2.f * t);
}
// optics.poly_boundary_radius (blades polygon, circumradius 1).
static inline float polyBoundaryR(float theta, int blades, float rot, float curv) {
  int n = blades < 3 ? 3 : blades;
  float sector = 2.f * (float)M_PI / n;
  float a = theta - rot;
  float rem = a - std::floor(a / sector) * sector;   // torch.remainder
  float phi = rem - sector * 0.5f;
  float apo = std::cos((float)M_PI / n);
  float cp = std::cos(phi); if (cp < 1e-4f) cp = 1e-4f;
  float poly_r = apo / cp;
  return poly_r + (1.f - poly_r) * curv;             // lerp(poly_r, 1, curv)
}

// --- per-pass uniform layouts (16-byte rows) ----------------------------------
struct PrepareU { float hl_threshold, hl_boost, in_brightness, in_contrast; };
static_assert(sizeof(PrepareU) == 16, "PrepareU layout");

struct BokehU {
  float half, dimw, dimh, coc_px;
  float field_curv, focus_cx, focus_cy, cats_eye;
  float swirl, anamorphic, loca_scale; uint32_t taps;
};
static_assert(sizeof(BokehU) == 48, "BokehU layout");

struct FinishU {
  float half, dimw, dimh, exposure;
  float vignette, hl_desat, tone, tone_black;
  float grain, _p0, _p1, _p2;
};
static_assert(sizeof(FinishU) == 48, "FinishU layout");

struct ColorU { float mr, mg, mb, contrast; };
static_assert(sizeof(ColorU) == 16, "ColorU layout");

struct DebugU {
  float mode, hl_threshold, half, dimw;
  float dimh, coc_px, field_curv, focus_cx;
  float focus_cy, _p0, _p1, _p2;
};
static_assert(sizeof(DebugU) == 48, "DebugU layout");

struct GeoU {
  float half, dimw, dimh, distortion;
  float wave, tca, _p0, _p1;
};
static_assert(sizeof(GeoU) == 32, "GeoU layout");

struct DownsampleU { uint32_t ds, _p0, _p1, _p2; };
static_assert(sizeof(DownsampleU) == 16, "DownsampleU layout");

struct ExtractU { float threshold, _p0, _p1, _p2; };
static_assert(sizeof(ExtractU) == 16, "ExtractU layout");

struct HoodU { float ftr, ftg, ftb, strength; };   // u_flare_tint (vec3) then u_strength
static_assert(sizeof(HoodU) == 16, "HoodU layout");

struct GlowU {
  float bloom, halation, _p0, _p1;
  float hr, hg, hb, _p2;                            // u_hal_color
};
static_assert(sizeof(GlowU) == 32, "GlowU layout");

struct SunU {
  float half, dimw, dimh, gate;
  float azimuth, obliqueness, sun_r, sun_g;
  float sun_b, w_glow, w_veil, w_streak;
  float w_ghost, coat_flare, elem_curv, dispersion;
  float aperture_rot, blade_curv, coat_r0, _p0;
  float design0, design1, design2; uint32_t ndesigns;
  uint32_t blades, nsp; float _p1, _p2;
};
static_assert(sizeof(SunU) == 112, "SunU layout");

// Normalized-slider → prototype-raw mappings for the geometry pass (the raw
// coefficients are small; presets sit around ±0.05 distortion / 0.01 TCA).
static constexpr float DIST_SCALE = 0.30f;   // distortion / mustache slider → raw
static constexpr float TCA_SCALE  = 0.03f;   // TCA slider → raw

static constexpr int MAX_TAPS = 192;

// --- per-instance state -------------------------------------------------------
struct State {
  int  tex_w = 0, tex_h = 0;
  bool initialized = false;

  // GPU resources (per-instance).
  gpu::Texture bufA, bufB;       // full-res linear-HDR ping/pong
  gpu::Texture small, bokeh_s;   // proc-res bokeh working buffers (downsampled)
  gpu::Texture flo, hi, hi_a, hi_b;  // flare-res: downsampled image + highlight + 2 blurs
  gpu::Texture dbg;              // full-res snapshot for the flare-only debug view (lazy)
  gpu::Sampler sampler;          // Linear/ClampToEdge (bokeh taps, flare upsample)
  gpu::Buffer  tap_buf;          // Vogel tap set (float4/tap: ox,oy,base_w,_)
  gpu::Buffer  prepare_buf, bokeh_buf, color_buf, geo_buf, finish_buf;
  gpu::Buffer  extract_buf, hood_buf, glow_buf, sun_buf, downsample_buf, debug_buf;
  int proc_w = 0, proc_h = 0;    // current bokeh working resolution
  int flare_w = 0, flare_h = 0;  // reduced flare/glow working resolution

  // Schema-mirrored params (normalized unless noted). Defaults mirror the schema.
  int   preset = 0;             // inert (UI-only); serialized

  // input levels (Resolume-style, pivot black) — small default lift so the filmic
  // pipeline doesn't crush blacks on already-processed footage.
  float in_brightness = 0.15f;
  float in_contrast = -0.13f;
  // focus
  float blur_amount = 0.16f;
  float field_curvature = 0.0f;
  float focus_cx = 0.0f, focus_cy = 0.0f;
  // bokeh
  int   blades = 7;
  float blade_curvature = 0.15f;
  float aperture_rotation = 0.0f;   // [0,1] → ×TAU
  float cats_eye = 0.8f;
  float swirl = 0.0f;
  float anamorphic = 0.0f;
  float loca = 0.25f;
  float rim = 0.12f;
  float onion_ring = 0.06f;
  float apodize = 0.55f;
  // highlight
  float hl_threshold = 1.0f;
  float hl_boost = 0.375f;          // [0,1] → ×8
  // coating
  int   coating = 0;                // 0 SMC,1 Single,2 Uncoated,3 Custom
  float warmth = 0.0f;
  float transmission = 1.0f;
  // flare
  float flare_strength = 0.5f;
  float hood_extension = 1.0f;
  float hood_shape = 0.3f;
  // sun
  float sun_intensity = 0.0f;
  float sun_azimuth = 0.0955f;      // [0,1] → ×TAU  (0.6 rad)
  float sun_obliqueness = 0.35f;
  float sun_r = 1.0f, sun_g = 0.85f, sun_b = 0.6f;
  float sun_veil = 0.6f;
  float sun_glow = 0.7f;
  float sun_streak = 0.18f;
  float sun_ghost = 0.15f;
  float element_curvature = 0.45f;
  float dispersion = 0.35f;
  // glow
  float halation = 0.22f;
  float hal_r = 1.0f, hal_g = 0.4f, hal_b = 0.22f;
  float bloom = 0.12f;
  // geometry
  float distortion = 0.0f;
  float distortion_wave = 0.0f;
  float tca = 0.0f;
  // finish
  float exposure = 0.0f;            // signed stops [-1,1] → 2^(3s)
  float mech_vignette = 0.25f;
  float hl_desat = 0.6f;
  float tone = 0.85f;
  float tone_black = 0.02f;
  float grain = 0.05f;
  // quality
  int   quality = 1;                // 0 Cheap,1 Standard,2 Max
  int   taps = 96;
  float work_radius = 11.0f;
  float fill = 0.7f;
  // debug
  int   debug_view = 0;
};

// --- type-shared PSOs ---------------------------------------------------------
// NOTE: fx::GaussianBlur is NOT usable on RGBA16F intermediates (its scratch is
// RGBA8, which clamps HDR — see line_reconstruct's Blur16). The `fill` anti-
// stipple micro-blur is deferred until an RGBA16F-safe blur is wired for the
// flare/glow passes.
static gpu::ComputePSO s_pso_prepare, s_pso_bokeh, s_pso_color, s_pso_geo, s_pso_finish;
static gpu::ComputePSO s_pso_extract, s_pso_hood, s_pso_glow, s_pso_sun;
static gpu::ComputePSO s_pso_downsample, s_pso_upsample, s_pso_debug;
static Blur16 s_blur16;   // RGBA16F wide blur (veiling glare, halation, bloom)

void module_init() {
  state::init("filter.blur.lens", {1, 1, 0},
    state::Schema()
      .helpField("intro",
        "## Lens\n"
        "A **full photographic-lens model** in one effect: shaped-aperture "
        "depth-of-field **bokeh**, lens **flare & glare**, chromatic aberration, "
        "distortion, and a filmic **finish** (tonemap + grain). Unlike a plain "
        "blur it energy-normalises highlights (so bright points bloom into real "
        "bokeh discs), shapes the blur with the aperture polygon, and composites "
        "everything in linear HDR before the film curve — the look real glass has "
        "and phones clip away.\n\n"
        "**Try:** raise *Blur Amount* for depth-of-field; open the *Bokeh Shape* "
        "group for character (cat's-eye, swirl, anamorphic ovals); switch *Coating* "
        "and dial *Warmth*; enable *Sun* for anamorphic streaks and ghosts; add "
        "*Halation* / *Bloom* for filmic glow. The **preset** picker (SMC prime, "
        "vintage swirl, anamorphic, dreamy, clinical) sets a whole look at once — "
        "then tweak.")

      // preset — inert in the effect; the custom web inspector applies it.
      .selectField("preset", 0, state::SecondaryInput,
                   {{"Custom", 0}, {"Modern Prime", 1}, {"Vintage", 2},
                    {"Anamorphic", 3}, {"Dreamy", 4}, {"Clinical", 5}},
                   false, "A named lens look. Applied from the UI (sets the "
                   "character params below); the effect itself only stores it.")
        .label("Preset", "Preset")

      .group("input", "Input Levels")
        .groupHelp(
          "Conditions the incoming image *before* the lens processes it. The filmic "
          "finish (toe + tonemap) tends to crush blacks, so a small default lift "
          "keeps already-graded footage from clipping. **Brightness** lifts, "
          "**Contrast** is Resolume-style — it scales to/from black (0.0), not gray. "
          "Zero both for a raw, punchier response.")
      .floatField("in_brightness", 0.15f, -1.f, 1.f, state::SecondaryInput, nullptr, 0.01f,
                  nullptr, "Additive lift on the input (display space). Raises the floor so blacks aren't crushed.").label("Input Brightness", "In Bri")
      .floatField("in_contrast", -0.13f, -1.f, 1.f, state::SecondaryInput, nullptr, 0.01f,
                  nullptr, "Resolume-style input contrast — scales to/from black (0.0), not gray. Negative softens.").label("Input Contrast", "In Con")

      .group("focus", "Depth of Field")
        .groupHelp(
          "The blur itself. *Blur Amount* is the circle-of-confusion radius (how "
          "much out-of-focus areas spread). *Field Curvature* keeps the centre "
          "sharp while softening the corners (Petzval — that vintage 3D pop). "
          "*Focus Centre* moves where the field is sharpest.")
      .floatField("blur_amount", 0.16f, 0.f, 1.f, state::PrimaryInput, nullptr, 0.01f,
                  nullptr, "Circle-of-confusion radius (fraction of the frame's short side).").label("Blur Amount", "Blur")
      .floatField("field_curvature", 0.0f, 0.f, 1.f, state::PrimaryInput, nullptr, 0.01f,
                  nullptr, "Petzval field curvature: sharp centre, softer corners.").label("Field Curvature", "Curve")
      .vec2Field("focus_center", 0.0f, 0.0f, state::PrimaryInput, -1.f, 1.f)
        .label("Focus Centre", "Focus")

      .group("bokeh", "Bokeh Shape")
        .groupHelp(
          "The character of the out-of-focus highlights. *Blades* + *Roundness* "
          "set the aperture polygon (few straight blades = hard hexagons; many "
          "round = creamy discs). *Cat's Eye* squashes bokeh toward the frame edges "
          "(optical vignetting — lemon shapes). *Swirl* + *Anamorphic* give Petzval "
          "swirl and oval bokeh. *Apodize* softens the disc edge (creamy), *Rim* / "
          "*Onion* add the nervous bright-rim and aspheric ring texture. *LoCA* is "
          "longitudinal chromatic fringing on the bokeh.")
      .intField("blades", 7, 3, 14, state::PrimaryInput, 1, nullptr,
                "Aperture blade count (polygon sides).").label("Blades", "Blades")
      .floatField("blade_curvature", 0.15f, 0.f, 1.f, state::PrimaryInput, nullptr, 0.01f,
                  nullptr, "0 = hard N-gon, 1 = perfect circle.").label("Roundness", "Round")
      .floatField("aperture_rotation", 0.0f, 0.f, 1.f, state::PrimaryInput, nullptr, 0.01f,
                  nullptr, "Iris rotation (0..1 → full turn).").label("Iris Rotation", "Iris")
      .floatField("cats_eye", 0.8f, 0.f, 1.f, state::PrimaryInput, nullptr, 0.01f,
                  nullptr, "Optical vignetting: bokeh squashed to lemons toward the edges.").label("Cat's Eye", "CatEye")
      .floatField("swirl", 0.0f, 0.f, 1.f, state::PrimaryInput, nullptr, 0.01f,
                  nullptr, "Petzval tangential swirl of the bokeh toward the edges.").label("Swirl", "Swirl")
      .floatField("anamorphic", 0.0f, 0.f, 1.f, state::PrimaryInput, nullptr, 0.01f,
                  nullptr, "Oval (anamorphic) bokeh: stretch horizontal, squeeze vertical.").label("Anamorphic", "Anam")
      .floatField("loca", 0.25f, 0.f, 1.f, state::PrimaryInput, nullptr, 0.01f,
                  nullptr, "Longitudinal chromatic aberration — colour fringe on out-of-focus edges.").label("LoCA", "LoCA")
      .floatField("rim", 0.12f, 0.f, 1.f, state::PrimaryInput, nullptr, 0.01f,
                  nullptr, "Bright-rim (nervous) bokeh.").label("Rim", "Rim")
      .floatField("onion_ring", 0.06f, 0.f, 1.f, state::PrimaryInput, nullptr, 0.01f,
                  nullptr, "Aspheric concentric-ring texture inside defocused bokeh discs. Low = subtle/physical, high = exaggerated vintage ring-stack (quadratic). Most visible on defocused highlights.").label("Onion Ring", "Onion")
      .floatField("apodize", 0.55f, 0.f, 1.f, state::PrimaryInput, nullptr, 0.01f,
                  nullptr, "Creamy centre-weighted disc falloff (raised-cosine apodization).").label("Apodize", "Apod")

      .group("highlight", "Highlights")
        .groupHelp(
          "Highlight energy for the bokeh. *Threshold* sets what counts as a "
          "highlight; *Boost* multiplies its energy so bright points survive the "
          "energy-normalised gather as vivid discs rather than dim smears.")
      .floatField("hl_threshold", 1.0f, 0.f, 2.f, state::SecondaryInput, nullptr, 0.01f,
                  nullptr, "Linear luma above which a pixel counts as a highlight.").label("HL Threshold", "HL Thr")
      .floatField("hl_boost", 0.375f, 0.f, 1.f, state::SecondaryInput, nullptr, 0.01f,
                  nullptr, "Extra energy given to highlights (0..1 → 0..8×).").label("HL Boost", "HL Boost")

      .group("coating", "Coating & Colour")
        .groupHelp(
          "The lens coating: its colour cast and how it flares. *SMC* is a clean "
          "modern multicoat; *Single* / *Uncoated* are warmer, flarier vintage "
          "glass. *Warmth* and *Transmission* trim the overall tone and brightness.")
      .selectField("coating", 0, state::PrimaryInput,
                   {{"SMC", 0}, {"Single", 1}, {"Uncoated", 2}, {"Custom", 3}},
                   false, "Coating stack: drives colour cast, flare colour and amount.").label("Coating", "Coat")
      .floatField("warmth", 0.0f, -1.f, 1.f, state::PrimaryInput, nullptr, 0.01f,
                  nullptr, "Bipolar colour warmth (− cool / + warm).").label("Warmth", "Warm")
      .floatField("transmission", 1.0f, 0.5f, 1.5f, state::SecondaryInput, nullptr, 0.01f,
                  nullptr, "Overall light throughput multiplier.").label("Transmission", "Trans")

      .group("flare", "Flare & Glare")
        .groupHelp(
          "In-frame veiling glare: bright sources within the frame scatter into a "
          "wide, coating-tinted bloom that lifts the blacks (that hazy backlit "
          "look). *Hood Extension* at 1 is a clean shaded lens; retracting it "
          "(toward 0) lets the glare in.")
      .floatField("flare_strength", 0.5f, 0.f, 1.f, state::PrimaryInput, nullptr, 0.01f,
                  nullptr, "In-frame veiling glare amount.").label("Veiling Glare", "Glare")
      .floatField("hood_extension", 1.0f, 0.f, 1.f, state::PrimaryInput, nullptr, 0.01f,
                  nullptr, "1 = shaded/clean, 0 = retracted (flary).").label("Hood Extension", "Hood")
      .floatField("hood_shape", 0.3f, 0.f, 1.f, state::SecondaryInput, nullptr, 0.01f,
                  nullptr, "0 = cylindrical hood, 1 = petal (admits more on diagonals).").label("Hood Shape", "Petal")

      .group("sun", "Sun / Stray Light")
        .groupHelp(
          "An off-frame bright source (the sun just outside the shot) refracting "
          "through the glass — the showpiece. *Intensity* is the master (0 = off). "
          "*Azimuth* / *Obliqueness* place the source. It manifests as a veil "
          "pedestal, a broad *Glow* toward the source, thin diffraction *Streaks* "
          "locked to the aperture blades, and a *Ghost* chain of aperture-shaped "
          "reflections along the source→centre axis. *Dispersion* spreads the "
          "spectrum for rainbow fringing.")
      .floatField("sun_intensity", 0.0f, 0.f, 1.f, state::PrimaryInput, nullptr, 0.01f,
                  nullptr, "Off-frame source brightness. 0 = the whole sun stack is skipped.").label("Sun Intensity", "Sun")
      .floatField("sun_azimuth", 0.0955f, 0.f, 1.f, state::PrimaryInput, nullptr, 0.01f,
                  nullptr, "Direction to the source (0..1 → full turn).").label("Sun Azimuth", "Azim")
      .floatField("sun_obliqueness", 0.35f, 0.f, 1.f, state::PrimaryInput, nullptr, 0.01f,
                  nullptr, "0 = grazing the frame edge, 1 = far off-axis.").label("Obliqueness", "Obliq")
      .rgbField("sun_color", 1.0f, 0.85f, 0.6f, state::SecondaryInput).label("Sun Colour", "SunCol")
      .floatField("sun_veil", 0.6f, 0.f, 1.f, state::SecondaryInput, nullptr, 0.01f,
                  nullptr, "Near-flat veiling pedestal.").label("Sun Veil", "Veil")
      .floatField("sun_glow", 0.7f, 0.f, 1.f, state::SecondaryInput, nullptr, 0.01f,
                  nullptr, "Broad directional scatter toward the source.").label("Sun Glow", "SunGlow")
      .floatField("sun_streak", 0.18f, 0.f, 1.f, state::SecondaryInput, nullptr, 0.01f,
                  nullptr, "Diffraction sunstar spikes on the aperture blades.").label("Sun Streak", "Streak")
      .floatField("sun_ghost", 0.15f, 0.f, 1.f, state::SecondaryInput, nullptr, 0.01f,
                  nullptr, "Aperture-shaped internal-reflection ghost chain.").label("Sun Ghost", "Ghost")
      .floatField("element_curvature", 0.45f, 0.f, 1.f, state::SecondaryInput, nullptr, 0.01f,
                  nullptr, "Ghost focus / edge definition.").label("Element Curve", "ElemC")
      .floatField("dispersion", 0.35f, 0.f, 1.f, state::SecondaryInput, nullptr, 0.01f,
                  nullptr, "Per-wavelength radial spread (spectral fringing).").label("Dispersion", "Disp")

      .group("glow", "Halation & Bloom")
        .groupHelp(
          "Filmic highlight glow. *Halation* is a warm wide ring (the red-ish "
          "bleed film gets around bright edges); *Bloom* is a gentle neutral "
          "spread. Both bleed from the highlights.")
      .floatField("halation", 0.22f, 0.f, 1.f, state::PrimaryInput, nullptr, 0.01f,
                  nullptr, "Warm highlight bleed (wide ring).").label("Halation", "Halo")
      .rgbField("halation_color", 1.0f, 0.4f, 0.22f, state::SecondaryInput).label("Halation Colour", "HaloCol")
      .floatField("bloom", 0.12f, 0.f, 1.f, state::PrimaryInput, nullptr, 0.01f,
                  nullptr, "Neutral highlight bloom (gentle spread).").label("Bloom", "Bloom")

      .group("geometry", "Distortion")
        .groupHelp(
          "Lens geometry. *Distortion* bends straight lines (− barrel / + "
          "pincushion); *Mustache* adds the higher-order wave. *Chromatic "
          "Aberration* magnifies red/blue oppositely so high-contrast edges fringe "
          "red/cyan toward the corners.")
      .floatField("distortion", 0.0f, -1.f, 1.f, state::PrimaryInput, nullptr, 0.01f,
                  nullptr, "Radial distortion (− barrel / + pincushion).").label("Distortion", "Dist")
      .floatField("distortion_wave", 0.0f, -1.f, 1.f, state::SecondaryInput, nullptr, 0.01f,
                  nullptr, "Higher-order mustache distortion term.").label("Mustache", "Wave")
      .floatField("tca", 0.0f, 0.f, 1.f, state::PrimaryInput, nullptr, 0.01f,
                  nullptr, "Transverse chromatic aberration (edge red/cyan fringing).").label("Chromatic Aberration", "TCA")

      .group("finish", "Finish")
        .groupHelp(
          "The film-back. *Exposure* is stops of gain. *Vignette* darkens the "
          "corners. *HL Desat* rolls clipped highlights toward white (no neon "
          "cores). *Tone* blends Reinhard→ACES filmic; *Black* crushes the toe. "
          "*Grain* adds fine film noise. This stage always runs — it produces the "
          "final image.")
      .floatField("exposure", 0.0f, -1.f, 1.f, state::PrimaryInput, nullptr, 0.01f,
                  nullptr, "Exposure in stops (−1..+1 → ÷8..×8).").label("Exposure", "Exp")
      .floatField("mech_vignette", 0.25f, 0.f, 1.f, state::PrimaryInput, nullptr, 0.01f,
                  nullptr, "Mechanical corner falloff (cos⁴-ish).").label("Vignette", "Vig")
      .floatField("hl_desat", 0.6f, 0.f, 1.f, state::SecondaryInput, nullptr, 0.01f,
                  nullptr, "Highlight desaturation toward white.").label("HL Desat", "Desat")
      .floatField("tone", 0.85f, 0.f, 1.f, state::PrimaryInput, nullptr, 0.01f,
                  nullptr, "Tonemap: 0 = Reinhard, 1 = full ACES filmic.").label("Tone", "Tone")
      .floatField("tone_black", 0.02f, 0.f, 1.f, state::SecondaryInput, nullptr, 0.01f,
                  nullptr, "Filmic toe crush (lift/clip the blacks).").label("Black Point", "Black")
      .floatField("grain", 0.05f, 0.f, 1.f, state::PrimaryInput, nullptr, 0.01f,
                  nullptr, "Fine film grain.").label("Grain", "Grain")

      .group("quality", "Quality")
        .groupHelp(
          "Cost/quality. *Quality* scales the gather: **Cheap** thins the tap "
          "count and downsamples more; **Max** adds taps and downsamples less "
          "(sharper discs, slower). It scales the advanced overrides below, which "
          "set the base: *Taps* (base tap count), *Work Radius* (working-res blur "
          "cap before downsampling), *Fill* (anti-stipple micro-blur).")
      .selectField("quality", 1, state::SecondaryInput,
                   {{"Cheap", 0}, {"Standard", 1}, {"Max", 2}},
                   false, "Quality tier — scales tap count and gather downsample (×⅓ / ×1 / ×5⁄3 taps).").label("Quality", "Qual")
      .intField("taps", 96, 16, 192, state::SecondaryInput, 1, nullptr,
                "Bokeh gather tap count (higher = smoother discs, slower).").label("Taps", "Taps")
      .floatField("work_radius", 11.0f, 4.f, 24.f, state::SecondaryInput, nullptr, 0.5f, "px",
                  "Max working-res CoC before the gather downsamples.").label("Work Radius", "WorkR")
      .floatField("fill", 0.7f, 0.f, 2.f, state::SecondaryInput, nullptr, 0.05f, "px",
                  "Working-res micro-blur that hides tap stipple.").label("Fill", "Fill")

      .group("debug", "Debug")
        .groupHelp(
          "Inspection aids. *Debug View* replaces the output with an internal "
          "stage — the highlight mask, the circle-of-confusion field, the isolated "
          "bokeh, or the isolated flare.")
      .selectField("debug_view", 0, state::SecondaryInput,
                   {{"Off", 0}, {"Highlight Mask", 1}, {"CoC Field", 2},
                    {"Bokeh Only", 3}, {"Flare Only", 4}},
                   true, "Visualize an internal stage instead of the composited output.").label("Debug View", "Debug")

      .endGroup()
      .capability(state::Capability::TimeIndependent)
      .textureField("tex_in",  state::PrimaryInput)
      .textureField("tex_out", state::PrimaryOutput));

  if (gpu::Device::backend() == gpu::Backend::None) return;

  const auto F16 = gpu::TextureFormat::RGBA16F;
  state::registerShaderSPV("lens_prepare", PREPARE_SPV, PREPARE_SPV_SIZE, "rgba16float", "write");
  state::registerShaderSPV("lens_bokeh",   BOKEH_SPV,   BOKEH_SPV_SIZE,   "rgba16float", "write");
  state::registerShaderSPV("lens_color",   COLOR_SPV,   COLOR_SPV_SIZE,   "rgba16float", "write");
  state::registerShaderSPV("lens_geo",     GEO_SPV,     GEO_SPV_SIZE,     "rgba16float", "write");
  state::registerShaderSPV("lens_downsample", DOWNSAMPLE_SPV, DOWNSAMPLE_SPV_SIZE, "rgba16float", "write");
  state::registerShaderSPV("lens_upsample",   UPSAMPLE_SPV,   UPSAMPLE_SPV_SIZE,   "rgba16float", "write");
  state::registerShaderSPV("lens_extract", EXTRACT_SPV, EXTRACT_SPV_SIZE, "rgba16float", "write");
  state::registerShaderSPV("lens_hood",    HOOD_SPV,    HOOD_SPV_SIZE,    "rgba16float", "write");
  state::registerShaderSPV("lens_glow",    GLOW_SPV,    GLOW_SPV_SIZE,    "rgba16float", "write");
  state::registerShaderSPV("lens_sun",     SUN_SPV,     SUN_SPV_SIZE,     "rgba16float", "write");
  state::registerShaderSPV("lens_finish",  FINISH_SPV,  FINISH_SPV_SIZE,  "rgba8unorm",  "write");
  state::registerShaderSPV("lens_debug",   DEBUG_SPV,   DEBUG_SPV_SIZE,   "rgba8unorm",  "write");
  auto cs_prepare = gpu::Device::createShaderModuleByName("lens_prepare");
  auto cs_bokeh   = gpu::Device::createShaderModuleByName("lens_bokeh");
  auto cs_color   = gpu::Device::createShaderModuleByName("lens_color");
  auto cs_geo     = gpu::Device::createShaderModuleByName("lens_geo");
  auto cs_downsample = gpu::Device::createShaderModuleByName("lens_downsample");
  auto cs_upsample   = gpu::Device::createShaderModuleByName("lens_upsample");
  auto cs_extract = gpu::Device::createShaderModuleByName("lens_extract");
  auto cs_hood    = gpu::Device::createShaderModuleByName("lens_hood");
  auto cs_glow    = gpu::Device::createShaderModuleByName("lens_glow");
  auto cs_sun     = gpu::Device::createShaderModuleByName("lens_sun");
  auto cs_finish  = gpu::Device::createShaderModuleByName("lens_finish");
  auto cs_debug   = gpu::Device::createShaderModuleByName("lens_debug");
  if (!cs_prepare || !cs_bokeh || !cs_color || !cs_geo || !cs_downsample ||
      !cs_upsample || !cs_extract || !cs_hood || !cs_glow || !cs_sun || !cs_finish ||
      !cs_debug) {
    state::log("lens: a shader failed to compile");
    return;
  }
  s_pso_prepare = gpu::Device::createComputePSO(cs_prepare, "main", gpu::Bindings()
      .tex2d(0).storageTex2d(1, F16).uniform(2));
  s_pso_bokeh = gpu::Device::createComputePSO(cs_bokeh, "main", gpu::Bindings()
      .tex2d(0).storage(1).sampler(2).storageTex2d(3, F16).uniform(4));
  s_pso_color = gpu::Device::createComputePSO(cs_color, "main", gpu::Bindings()
      .tex2d(0).storageTex2d(1, F16).uniform(2));
  s_pso_geo = gpu::Device::createComputePSO(cs_geo, "main", gpu::Bindings()
      .tex2d(0).sampler(1).storageTex2d(2, F16).uniform(3));
  s_pso_downsample = gpu::Device::createComputePSO(cs_downsample, "main", gpu::Bindings()
      .tex2d(0).storageTex2d(1, F16).uniform(2));
  s_pso_upsample = gpu::Device::createComputePSO(cs_upsample, "main", gpu::Bindings()
      .tex2d(0).sampler(1).storageTex2d(2, F16));
  s_pso_extract = gpu::Device::createComputePSO(cs_extract, "main", gpu::Bindings()
      .tex2d(0).storageTex2d(1, F16).uniform(2));
  s_pso_hood = gpu::Device::createComputePSO(cs_hood, "main", gpu::Bindings()
      .tex2d(0).tex2d(1).sampler(2).storageTex2d(3, F16).uniform(4));
  s_pso_glow = gpu::Device::createComputePSO(cs_glow, "main", gpu::Bindings()
      .tex2d(0).tex2d(1).tex2d(2).sampler(3).storageTex2d(4, F16).uniform(5));
  s_pso_sun = gpu::Device::createComputePSO(cs_sun, "main", gpu::Bindings()
      .storageTex2d(1, F16).uniform(2));   // no input — writes contribution only
  s_pso_finish = gpu::Device::createComputePSO(cs_finish, "main", gpu::Bindings()
      .tex2d(0).storageTex2d(1, gpu::TextureFormat::RGBA8).uniform(2));
  s_pso_debug = gpu::Device::createComputePSO(cs_debug, "main", gpu::Bindings()
      .tex2d(0).tex2d(1).storageTex2d(2, gpu::TextureFormat::RGBA8).uniform(3));
  s_blur16.init();

  gpu::ComputePSO* psos[] = { &s_pso_prepare, &s_pso_bokeh, &s_pso_color, &s_pso_geo,
    &s_pso_extract, &s_pso_hood, &s_pso_glow, &s_pso_sun, &s_pso_finish, &s_pso_debug };
  for (auto* p : psos) if (!p->valid()) state::log("lens: a PSO is INVALID");
  if (!s_blur16.valid()) state::log("lens: Blur16 INVALID");
  state::log("lens: module_init done");
}

void* create() {
  auto* s = new State();
  s->sampler = gpu::Device::createSampler(gpu::FilterMode::Linear, gpu::AddressMode::ClampToEdge);
  s->tap_buf = gpu::Device::createBuffer(sizeof(float) * 4 * MAX_TAPS, gpu::BufferUsage::Storage);
  s->prepare_buf = gpu::Device::createBuffer(sizeof(PrepareU), gpu::BufferUsage::Uniform);
  s->bokeh_buf   = gpu::Device::createBuffer(sizeof(BokehU),   gpu::BufferUsage::Uniform);
  s->color_buf   = gpu::Device::createBuffer(sizeof(ColorU),   gpu::BufferUsage::Uniform);
  s->geo_buf     = gpu::Device::createBuffer(sizeof(GeoU),     gpu::BufferUsage::Uniform);
  s->finish_buf  = gpu::Device::createBuffer(sizeof(FinishU),  gpu::BufferUsage::Uniform);
  s->extract_buf = gpu::Device::createBuffer(sizeof(ExtractU), gpu::BufferUsage::Uniform);
  s->hood_buf    = gpu::Device::createBuffer(sizeof(HoodU),    gpu::BufferUsage::Uniform);
  s->glow_buf    = gpu::Device::createBuffer(sizeof(GlowU),    gpu::BufferUsage::Uniform);
  s->sun_buf     = gpu::Device::createBuffer(sizeof(SunU),     gpu::BufferUsage::Uniform);
  s->downsample_buf = gpu::Device::createBuffer(sizeof(DownsampleU), gpu::BufferUsage::Uniform);
  s->debug_buf   = gpu::Device::createBuffer(sizeof(DebugU),    gpu::BufferUsage::Uniform);
  return s;
}

void destroy(void* self) {
  auto* s = static_cast<State*>(self);
  if (!s) return;
  if (s->bufA.valid()) s->bufA.release();
  if (s->bufB.valid()) s->bufB.release();
  if (s->small.valid())   s->small.release();
  if (s->bokeh_s.valid()) s->bokeh_s.release();
  if (s->flo.valid())  s->flo.release();
  if (s->hi.valid())   s->hi.release();
  if (s->hi_a.valid()) s->hi_a.release();
  if (s->hi_b.valid()) s->hi_b.release();
  s->sampler.release();
  s->tap_buf.release();
  s->prepare_buf.release();
  s->bokeh_buf.release();
  s->color_buf.release();
  s->geo_buf.release();
  s->finish_buf.release();
  s->extract_buf.release();
  s->hood_buf.release();
  s->glow_buf.release();
  s->sun_buf.release();
  s->downsample_buf.release();
  delete s;
}

void init(void* self) {
  auto* s = static_cast<State*>(self);
  if (!s) return;
  // Core pipeline must be valid; the flare/glow passes degrade gracefully (each
  // guarded on its own PSO at dispatch time).
  s->initialized = s_pso_prepare.valid() && s_pso_bokeh.valid() &&
                   s_pso_color.valid() && s_pso_geo.valid() && s_pso_finish.valid() &&
                   s->tap_buf.valid() && s->finish_buf.valid();
}

void tick(void* self, double dt) { (void)self; (void)dt; }

void on_state_patched(void* self, int n, const char* pb, const int* off,
                      const int* len, const int* ops) {
  auto* s = static_cast<State*>(self);
  if (!s) return;
  for (int i = 0; i < n; i++) {
    if (ops[i] != state::PatchReplace) continue;
    const char* p = pb + off[i];
    int l = len[i];
    if      (state::pathIs(p, l, "preset"))            s->preset = state::patchInt(i);
    else if (state::pathIs(p, l, "in_brightness"))     s->in_brightness = state::patchFloat(i);
    else if (state::pathIs(p, l, "in_contrast"))       s->in_contrast = state::patchFloat(i);
    else if (state::pathIs(p, l, "blur_amount"))       s->blur_amount = state::patchFloat(i);
    else if (state::pathIs(p, l, "field_curvature"))   s->field_curvature = state::patchFloat(i);
    else if (state::pathIs(p, l, "focus_center"))    { auto v = state::patchVec2(i); s->focus_cx = v.x; s->focus_cy = v.y; }
    else if (state::pathIs(p, l, "blades"))            s->blades = state::patchInt(i);
    else if (state::pathIs(p, l, "blade_curvature"))   s->blade_curvature = state::patchFloat(i);
    else if (state::pathIs(p, l, "aperture_rotation")) s->aperture_rotation = state::patchFloat(i);
    else if (state::pathIs(p, l, "cats_eye"))          s->cats_eye = state::patchFloat(i);
    else if (state::pathIs(p, l, "swirl"))             s->swirl = state::patchFloat(i);
    else if (state::pathIs(p, l, "anamorphic"))        s->anamorphic = state::patchFloat(i);
    else if (state::pathIs(p, l, "loca"))              s->loca = state::patchFloat(i);
    else if (state::pathIs(p, l, "rim"))               s->rim = state::patchFloat(i);
    else if (state::pathIs(p, l, "onion_ring"))        s->onion_ring = state::patchFloat(i);
    else if (state::pathIs(p, l, "apodize"))           s->apodize = state::patchFloat(i);
    else if (state::pathIs(p, l, "hl_threshold"))      s->hl_threshold = state::patchFloat(i);
    else if (state::pathIs(p, l, "hl_boost"))          s->hl_boost = state::patchFloat(i);
    else if (state::pathIs(p, l, "coating"))           s->coating = state::patchInt(i);
    else if (state::pathIs(p, l, "warmth"))            s->warmth = state::patchFloat(i);
    else if (state::pathIs(p, l, "transmission"))      s->transmission = state::patchFloat(i);
    else if (state::pathIs(p, l, "flare_strength"))    s->flare_strength = state::patchFloat(i);
    else if (state::pathIs(p, l, "hood_extension"))    s->hood_extension = state::patchFloat(i);
    else if (state::pathIs(p, l, "hood_shape"))        s->hood_shape = state::patchFloat(i);
    else if (state::pathIs(p, l, "sun_intensity"))     s->sun_intensity = state::patchFloat(i);
    else if (state::pathIs(p, l, "sun_azimuth"))       s->sun_azimuth = state::patchFloat(i);
    else if (state::pathIs(p, l, "sun_obliqueness"))   s->sun_obliqueness = state::patchFloat(i);
    else if (state::pathIs(p, l, "sun_color"))       { auto v = state::patchVec3(i); s->sun_r = v.x; s->sun_g = v.y; s->sun_b = v.z; }
    else if (state::pathIs(p, l, "sun_veil"))          s->sun_veil = state::patchFloat(i);
    else if (state::pathIs(p, l, "sun_glow"))          s->sun_glow = state::patchFloat(i);
    else if (state::pathIs(p, l, "sun_streak"))        s->sun_streak = state::patchFloat(i);
    else if (state::pathIs(p, l, "sun_ghost"))         s->sun_ghost = state::patchFloat(i);
    else if (state::pathIs(p, l, "element_curvature")) s->element_curvature = state::patchFloat(i);
    else if (state::pathIs(p, l, "dispersion"))        s->dispersion = state::patchFloat(i);
    else if (state::pathIs(p, l, "halation"))          s->halation = state::patchFloat(i);
    else if (state::pathIs(p, l, "halation_color"))  { auto v = state::patchVec3(i); s->hal_r = v.x; s->hal_g = v.y; s->hal_b = v.z; }
    else if (state::pathIs(p, l, "bloom"))             s->bloom = state::patchFloat(i);
    else if (state::pathIs(p, l, "distortion"))        s->distortion = state::patchFloat(i);
    else if (state::pathIs(p, l, "distortion_wave"))   s->distortion_wave = state::patchFloat(i);
    else if (state::pathIs(p, l, "tca"))               s->tca = state::patchFloat(i);
    else if (state::pathIs(p, l, "exposure"))          s->exposure = state::patchFloat(i);
    else if (state::pathIs(p, l, "mech_vignette"))     s->mech_vignette = state::patchFloat(i);
    else if (state::pathIs(p, l, "hl_desat"))          s->hl_desat = state::patchFloat(i);
    else if (state::pathIs(p, l, "tone"))              s->tone = state::patchFloat(i);
    else if (state::pathIs(p, l, "tone_black"))        s->tone_black = state::patchFloat(i);
    else if (state::pathIs(p, l, "grain"))             s->grain = state::patchFloat(i);
    else if (state::pathIs(p, l, "quality"))           s->quality = state::patchInt(i);
    else if (state::pathIs(p, l, "taps"))              s->taps = state::patchInt(i);
    else if (state::pathIs(p, l, "work_radius"))       s->work_radius = state::patchFloat(i);
    else if (state::pathIs(p, l, "fill"))              s->fill = state::patchFloat(i);
    else if (state::pathIs(p, l, "debug_view"))        s->debug_view = state::patchInt(i);
  }
}

void on_resolume_param(void* self, long long, double) { (void)self; }

// Conservative whole-effect passthrough. The finish stage normally transforms
// (tonemap), so identity requires an explicitly-neutral config. STAGE 1's render
// is a straight copy regardless; this predicate is refined when the real finish
// lands. Stateless (TimeIndependent) — safe to skip.
static inline bool near0(float x) { return x > -1e-4f && x < 1e-4f; }
int32_t is_identity(void* self) {
  auto* s = static_cast<State*>(self);
  if (!s) return 0;
  if (s->debug_view != 0) return 0;   // a debug view always transforms the frame
  bool optics_off = near0(s->blur_amount) && near0(s->hl_boost) &&
                    near0(s->sun_intensity) && near0(s->flare_strength) &&
                    near0(s->halation) && near0(s->bloom) &&
                    near0(s->distortion) && near0(s->distortion_wave) && near0(s->tca);
  bool color_off  = (s->coating == 3) && near0(s->warmth) &&
                    (s->transmission > 1.f - 1e-4f && s->transmission < 1.f + 1e-4f);
  bool finish_off = near0(s->exposure) && near0(s->mech_vignette) &&
                    near0(s->hl_desat) && near0(s->grain) && near0(s->tone);
  bool input_off  = near0(s->in_brightness) && near0(s->in_contrast);
  return (input_off && optics_off && color_off && finish_off) ? 1 : 0;
}

static bool ensureTextures(State* s, int w, int h) {
  if (s->tex_w == w && s->tex_h == h && s->bufA.valid() && s->bufB.valid()) return true;
  const auto F16 = gpu::TextureFormat::RGBA16F;
  gpu::Texture* texs[] = { &s->bufA, &s->bufB };
  bool ok = true;
  for (auto* t : texs) {
    if (t->valid()) t->release();
    *t = gpu::Device::createTexture(w, h, F16);
    if (!t->valid()) ok = false;
  }
  if (!ok) return false;
  s->tex_w = w; s->tex_h = h;
  return true;
}

// Full-res snapshot buffer for the flare-only debug view — allocated lazily (the
// common non-debug path never pays for it).
static bool ensureDbg(State* s, int w, int h) {
  if (s->dbg.valid() && s->tex_w == w && s->tex_h == h) return true;
  if (s->dbg.valid()) s->dbg.release();
  s->dbg = gpu::Device::createTexture(w, h, gpu::TextureFormat::RGBA16F);
  return s->dbg.valid();
}

// Reduced-resolution flare/glow working buffers (downsampled image + highlight +
// two blurred copies). The wide veil/halation/bloom blurs run here so their sigma
// (and Blur16 tap count) shrink by the flare-downsample factor.
static bool ensureFlareTextures(State* s, int fw, int fh) {
  if (s->flare_w == fw && s->flare_h == fh && s->flo.valid()) return true;
  const auto F16 = gpu::TextureFormat::RGBA16F;
  gpu::Texture* texs[] = { &s->flo, &s->hi, &s->hi_a, &s->hi_b };
  bool ok = true;
  for (auto* t : texs) {
    if (t->valid()) t->release();
    *t = gpu::Device::createTexture(fw, fh, F16);
    if (!t->valid()) ok = false;
  }
  if (!ok) return false;
  s->flare_w = fw; s->flare_h = fh;
  return true;
}

// Reduced-resolution bokeh working buffers (reallocated when the proc size
// changes — ds only jumps at a handful of blur-amount thresholds).
static bool ensureProcTextures(State* s, int pw, int ph) {
  if (s->proc_w == pw && s->proc_h == ph && s->small.valid() && s->bokeh_s.valid())
    return true;
  const auto F16 = gpu::TextureFormat::RGBA16F;
  if (s->small.valid())   s->small.release();
  if (s->bokeh_s.valid()) s->bokeh_s.release();
  s->small   = gpu::Device::createTexture(pw, ph, F16);
  s->bokeh_s = gpu::Device::createTexture(pw, ph, F16);
  if (!s->small.valid() || !s->bokeh_s.valid()) return false;
  s->proc_w = pw; s->proc_h = ph;
  return true;
}

// The Quality tier (Cheap/Standard/Max) scales the gather cost. It's the friendly
// knob; the advanced Taps / Work Radius sliders are the base values it scales, so
// both stay live. Cheap ×⅓ taps + smaller work-radius (more downsample); Max ×5⁄3
// taps + larger work-radius (less downsample, sharper discs).
static int effectiveTaps(const State* s) {
  int base = s->taps < 1 ? 1 : (s->taps > MAX_TAPS ? MAX_TAPS : s->taps);
  float mul = s->quality == 0 ? (1.f / 3.f) : (s->quality == 2 ? (5.f / 3.f) : 1.f);
  int k = (int)std::lround((float)base * mul);
  if (k < 1) k = 1; if (k > MAX_TAPS) k = MAX_TAPS;
  return k;
}
static float effectiveWorkRadius(const State* s) {
  float base = s->work_radius < 1.f ? 1.f : s->work_radius;
  float mul = s->quality == 0 ? 0.65f : (s->quality == 2 ? 1.6f : 1.f);
  return base * mul;
}

// Downsample factor for the bokeh gather (pipeline.pass_bokeh :168-175): size it
// off the LARGEST circle-of-confusion in frame so the working-res disc stays
// ≤ work_radius px — a FIXED tap count then covers it densely.
static int bokehDownsample(State* s, int vp_w, int vp_h, float coc_px0) {
  float half = (float)(vp_w > vp_h ? vp_w : vp_h) * 0.5f;
  float fx = s->focus_cx, fy = s->focus_cy;
  float r_max = 0.f;
  for (int sx = -1; sx <= 1; sx += 2)
    for (int sy = -1; sy <= 1; sy += 2) {
      float dx = (sx * vp_w * 0.5f) / half - fx;
      float dy = (sy * vp_h * 0.5f) / half - fy;
      float rr = std::sqrt(dx * dx + dy * dy);
      if (rr > r_max) r_max = rr;
    }
  float fc = s->field_curvature < 0.f ? 0.f : s->field_curvature;
  float coc_px_max = coc_px0 * (1.f + fc * r_max * r_max);
  float wr = effectiveWorkRadius(s);
  int ds = 1;
  while (coc_px_max / ds > wr && ds < 8) ds *= 2;
  return ds;
}

// Rebuild the Vogel tap set + its pixel-independent weight (aperture · rim ·
// apodization) into the tap storage buffer (pipeline.pass_bokeh :183-190;
// optics.py). Recomputed per frame — K≤192 cheap trig on the CPU.
static void writeTaps(State* s) {
  const Coating& coat = coatingOf(s->coating);
  int K = effectiveTaps(s);
  const float ga = (float)M_PI * (3.f - std::sqrt(5.f));   // golden angle
  const float rot = s->aperture_rotation * (float)(2.0 * M_PI);
  const float rimv   = s->rim * coat.rim;
  // Onion-ring magnitude. The prototype gates it hard by coating quality
  // (coat.onion=0.10 for SMC), which made it invisible on the default lens. Keep
  // the coating as a modifier but floor it (0.5 + 0.5·coat.onion). The slider maps
  // through a QUADRATIC curve (1.5·o + 3·o²): a gentle, near-physical low end with
  // fine control, extending to an exaggerated 'nostalgic' ring stack at the top.
  // Most visible in the bokeh discs of defocused highlights.
  const float o = s->onion_ring;
  const float onionv = (0.5f + 0.5f * coat.onion) * (1.5f * o + 3.0f * o * o);
  float buf[4 * MAX_TAPS];
  for (int k = 0; k < K; k++) {
    float r = std::sqrt((k + 0.5f) / K);
    float t = k * ga;
    float ox = r * std::cos(t), oy = r * std::sin(t);
    float theta = std::atan2(oy, ox);
    float bnd = polyBoundaryR(theta, s->blades, rot, s->blade_curvature);
    float aw = ss_down(bnd, r, 0.05f);
    float u = r < 0.f ? 0.f : (r > 1.f ? 1.f : r);
    float edge = ss(0.55f, 1.0f, u);
    float rw = 1.f + rimv * edge;
    if (onionv != 0.f)
      rw += onionv * 0.5f * (0.5f - 0.5f * std::cos(u * 5.f * (float)M_PI)) * u;
    float apod = (1.f - s->apodize) + s->apodize * (0.5f + 0.5f * std::cos((float)M_PI * u));
    buf[4 * k + 0] = ox;
    buf[4 * k + 1] = oy;
    buf[4 * k + 2] = aw * rw * apod;
    buf[4 * k + 3] = 0.f;
  }
  s->tap_buf.writeBytes(buf, sizeof(float) * 4 * K, 0);
}

void render(void* self, int vp_w, int vp_h) {
  auto* s = static_cast<State*>(self);
  if (!s || !s->initialized || vp_w <= 0 || vp_h <= 0) return;
  auto in  = gpu::Device::textureForField("tex_in");
  auto out = gpu::Device::textureForField("tex_out");
  if (!in.valid() || !out.valid()) return;
  if (!ensureTextures(s, vp_w, vp_h)) return;

  const int gx = (vp_w + 7) / 8, gy = (vp_h + 7) / 8;
  const float md0  = (float)(vp_w < vp_h ? vp_w : vp_h);
  const float coc_px0 = s->blur_amount * md0 * 0.25f;
  const float half = (float)(vp_w > vp_h ? vp_w : vp_h) * 0.5f;

  // Debug views (0 = off, normal output). Bokeh-only (3) short-circuits the mid
  // passes so `src` reaches finish as the isolated gather; flare-only (4) snapshots
  // the pre-flare image so the debug pass can subtract it. Falls back to the normal
  // path if the debug PSO failed to build.
  const int debug = s_pso_debug.valid() ? s->debug_view : 0;
  const bool dbg_bokeh = debug == 3;   // skip coating/flare/glow/geo
  const bool dbg_flare = debug == 4;   // snapshot pre-flare, skip geo

  // 1. prepare: sRGB→linear + highlight boost → bufA (linear HDR).
  { PrepareU u = { s->hl_threshold, s->hl_boost * 8.f, s->in_brightness, s->in_contrast };
    s->prepare_buf.writeOne(u);
    auto cp = gpu::ComputePass::begin();
    cp.setPSO(s_pso_prepare);
    cp.setTexture(in, 0, 0); cp.setTexture(s->bufA, 1, 1);
    cp.setBuffer(s->prepare_buf, 2);
    cp.dispatch(gx, gy); cp.end(); }

  // Ping-pong the linear-HDR image between bufA (holds prepare's output) and bufB.
  gpu::Texture src = s->bufA, dst = s->bufB;
  auto swap = [&]() { gpu::Texture t = src; src = dst; dst = t; };

  // 2. bokeh gather. Skip at ~zero blur. Downsampled-before-gather when the disc
  // is large so a FIXED tap count covers it (the GPU-DOF cost lever).
  if (coc_px0 > 0.5f) {
    writeTaps(s);
    int ds = bokehDownsample(s, vp_w, vp_h, coc_px0);
    uint32_t K = (uint32_t)effectiveTaps(s);
    float loca_scale = s->loca * coatingOf(s->coating).loca;

    if (ds <= 1) {
      // full-res gather: src → dst.
      BokehU u = {};
      u.half = half; u.dimw = (float)vp_w; u.dimh = (float)vp_h; u.coc_px = coc_px0;
      u.field_curv = s->field_curvature; u.focus_cx = s->focus_cx; u.focus_cy = s->focus_cy;
      u.cats_eye = s->cats_eye; u.swirl = s->swirl; u.anamorphic = s->anamorphic;
      u.loca_scale = loca_scale; u.taps = K;
      s->bokeh_buf.writeOne(u);
      auto cp = gpu::ComputePass::begin();
      cp.setPSO(s_pso_bokeh);
      cp.setTexture(src, 0, 0); cp.setBuffer(s->tap_buf, 1); cp.setSampler(s->sampler, 2);
      cp.setTexture(dst, 3, 1); cp.setBuffer(s->bokeh_buf, 4);
      cp.dispatch(gx, gy); cp.end();
      swap();
    } else {
      int pw = vp_w / ds < 1 ? 1 : vp_w / ds;
      int ph = vp_h / ds < 1 ? 1 : vp_h / ds;
      if (ensureProcTextures(s, pw, ph)) {
        const int pgx = (pw + 7) / 8, pgy = (ph + 7) / 8;
        // downsample src → small (proc-res box average).
        { DownsampleU du = { (uint32_t)ds, 0, 0, 0 }; s->downsample_buf.writeOne(du);
          auto cp = gpu::ComputePass::begin(); cp.setPSO(s_pso_downsample);
          cp.setTexture(src, 0, 0); cp.setTexture(s->small, 1, 1);
          cp.setBuffer(s->downsample_buf, 2); cp.dispatch(pgx, pgy); cp.end(); }
        // gather at proc-res: small → bokeh_s.
        { BokehU u = {};
          u.half = (float)(pw > ph ? pw : ph) * 0.5f; u.dimw = (float)pw; u.dimh = (float)ph;
          u.coc_px = coc_px0 / ds;
          u.field_curv = s->field_curvature; u.focus_cx = s->focus_cx; u.focus_cy = s->focus_cy;
          u.cats_eye = s->cats_eye; u.swirl = s->swirl; u.anamorphic = s->anamorphic;
          u.loca_scale = loca_scale; u.taps = K;
          s->bokeh_buf.writeOne(u);
          auto cp = gpu::ComputePass::begin(); cp.setPSO(s_pso_bokeh);
          cp.setTexture(s->small, 0, 0); cp.setBuffer(s->tap_buf, 1); cp.setSampler(s->sampler, 2);
          cp.setTexture(s->bokeh_s, 3, 1); cp.setBuffer(s->bokeh_buf, 4);
          cp.dispatch(pgx, pgy); cp.end(); }
        // bilinear upsample bokeh_s → dst (full res).
        { auto cp = gpu::ComputePass::begin(); cp.setPSO(s_pso_upsample);
          cp.setTexture(s->bokeh_s, 0, 0); cp.setSampler(s->sampler, 1);
          cp.setTexture(dst, 2, 1); cp.dispatch(gx, gy); cp.end(); }
        swap();
      }
    }
  }

  // 3. coating colour grade (linear tint × warmth × transmission + micro-contrast).
  { const Coating& coat = coatingOf(s->coating);
    float w = s->warmth < -1.f ? -1.f : (s->warmth > 1.f ? 1.f : s->warmth);
    float t = coat.transmission * s->transmission;
    ColorU cu;
    cu.mr = coat.tint[0] * (1.f + 0.18f * w) * t;
    cu.mg = coat.tint[1] * t;
    cu.mb = coat.tint[2] * (1.f - 0.18f * w) * t;
    cu.contrast = coat.contrast;
    bool neutral = std::fabs(cu.mr - 1.f) < 1e-3f && std::fabs(cu.mg - 1.f) < 1e-3f &&
                   std::fabs(cu.mb - 1.f) < 1e-3f && std::fabs(cu.contrast - 1.f) < 1e-3f;
    if (!neutral && !dbg_bokeh) {
      s->color_buf.writeOne(cu);
      auto cp = gpu::ComputePass::begin();
      cp.setPSO(s_pso_color);
      cp.setTexture(src, 0, 0); cp.setTexture(dst, 1, 1);
      cp.setBuffer(s->color_buf, 2);
      cp.dispatch(gx, gy); cp.end();
      swap();
    }
  }

  // Reduced-resolution flare/glow tier: the wide veil/halation/bloom blurs run
  // here so their sigma (and Blur16 tap count) shrink by `fds` — otherwise a wide
  // blur at full 1080p clamps to a 128-tap-per-side kernel and walls the frame.
  const int maxdim = vp_w > vp_h ? vp_w : vp_h;
  const int fds = maxdim < 360 ? 1 : (maxdim + 179) / 360;
  const int flare_w = vp_w / fds < 1 ? 1 : vp_w / fds;
  const int flare_h = vp_h / fds < 1 ? 1 : vp_h / fds;
  const float fmin = (float)(flare_w < flare_h ? flare_w : flare_h);
  const int fgx = (flare_w + 7) / 8, fgy = (flare_h + 7) / 8;
  const bool blur_ok = s_pso_extract.valid() && s_blur16.valid() && s_pso_downsample.valid();

  // Downsample the CURRENT image to `flo` (flare res) for the extract.
  auto toFlare = [&](gpu::Texture t) {
    DownsampleU du = { (uint32_t)fds, 0, 0, 0 }; s->downsample_buf.writeOne(du);
    auto cp = gpu::ComputePass::begin(); cp.setPSO(s_pso_downsample);
    cp.setTexture(t, 0, 0); cp.setTexture(s->flo, 1, 1);
    cp.setBuffer(s->downsample_buf, 2); cp.dispatch(fgx, fgy); cp.end();
  };
  auto extractFlare = [&](float threshold) {
    ExtractU eu = { threshold, 0.f, 0.f, 0.f }; s->extract_buf.writeOne(eu);
    auto cp = gpu::ComputePass::begin(); cp.setPSO(s_pso_extract);
    cp.setTexture(s->flo, 0, 0); cp.setTexture(s->hi, 1, 1);
    cp.setBuffer(s->extract_buf, 2); cp.dispatch(fgx, fgy); cp.end();
  };

  // Flare-only debug: snapshot the pre-flare image (identity upsample copy) so the
  // debug pass can subtract it from the post-flare result. Only the flare passes
  // (4/5/6) and NOT geo run after this, so src−snapshot is exactly the flare add.
  const bool run_flare = !dbg_bokeh;
  if (dbg_flare && ensureDbg(s, vp_w, vp_h)) {
    auto cp = gpu::ComputePass::begin(); cp.setPSO(s_pso_upsample);
    cp.setTexture(src, 0, 0); cp.setSampler(s->sampler, 1); cp.setTexture(s->dbg, 2, 1);
    cp.dispatch(gx, gy); cp.end();
  }

  // 4. in-frame veiling glare (hood). Skip when the hood fully shades the frame.
  if (run_flare) { const Coating& coat = coatingOf(s->coating);
    float strength = s->flare_strength * coat.flare * (1.f - s->hood_extension);
    if (strength > 1e-4f && blur_ok && s_pso_hood.valid() && ensureFlareTextures(s, flare_w, flare_h)) {
      toFlare(src);
      extractFlare(s->hl_threshold);
      s_blur16.apply(s->hi, s->hi_a, flare_w, flare_h, 0.06f * fmin);
      HoodU hu = { coat.flare_tint[0], coat.flare_tint[1], coat.flare_tint[2], strength };
      s->hood_buf.writeOne(hu);
      auto cp = gpu::ComputePass::begin(); cp.setPSO(s_pso_hood);
      cp.setTexture(src, 0, 0); cp.setTexture(s->hi_a, 1, 0); cp.setSampler(s->sampler, 2);
      cp.setTexture(dst, 3, 1); cp.setBuffer(s->hood_buf, 4);
      cp.dispatch(gx, gy); cp.end();
      swap();
    }
  }

  // 5. off-frame sun / stray light. Off by default (sun_intensity 0 → skipped).
  // Analytic + low-frequency → computed at flare res, then upsample-added.
  if (run_flare && s->sun_intensity > 1e-4f && s_pso_sun.valid() && s_pso_hood.valid() &&
      ensureFlareTextures(s, flare_w, flare_h)) {
    const Coating& coat = coatingOf(s->coating);
    float az = s->sun_azimuth * (float)(2.0 * M_PI);
    float petal = 1.f + s->hood_shape * 0.5f * std::cos(4.f * (az - (float)M_PI / 4.f));
    float mp = petal < 0.2f ? 0.2f : petal;
    float cutoff = (0.06f + (1.f - s->hood_extension) * 1.10f) * mp;
    float gate = (1.f - ss(0.65f * cutoff, cutoff, s->sun_obliqueness)) * s->sun_intensity;
    if (gate > 1e-4f) {
      int blades = s->blades < 3 ? 3 : s->blades;
      int nsp = (blades % 2 == 0) ? blades : 2 * blades;
      SunU u = {};
      u.half = (float)(flare_w > flare_h ? flare_w : flare_h) * 0.5f;
      u.dimw = (float)flare_w; u.dimh = (float)flare_h; u.gate = gate;
      u.azimuth = az; u.obliqueness = s->sun_obliqueness;
      u.sun_r = s->sun_r; u.sun_g = s->sun_g; u.sun_b = s->sun_b;
      u.w_glow = s->sun_glow; u.w_veil = s->sun_veil; u.w_streak = s->sun_streak;
      u.w_ghost = s->sun_ghost; u.coat_flare = coat.flare;
      u.elem_curv = s->element_curvature; u.dispersion = s->dispersion;
      u.aperture_rot = s->aperture_rotation * (float)(2.0 * M_PI);
      u.blade_curv = s->blade_curvature; u.coat_r0 = coat.r0;
      u.design0 = coat.designs[0]; u.design1 = coat.designs[1]; u.design2 = coat.designs[2];
      u.ndesigns = (uint32_t)coat.ndesigns;
      u.blades = (uint32_t)blades; u.nsp = (uint32_t)nsp;
      s->sun_buf.writeOne(u);
      // sun contribution → hi (flare res).
      { auto cp = gpu::ComputePass::begin(); cp.setPSO(s_pso_sun);
        cp.setTexture(s->hi, 1, 1); cp.setBuffer(s->sun_buf, 2);
        cp.dispatch(fgx, fgy); cp.end(); }
      // upsample-add via the hood composite (tint white, strength 1).
      HoodU hu = { 1.f, 1.f, 1.f, 1.f };
      s->hood_buf.writeOne(hu);
      auto cp = gpu::ComputePass::begin(); cp.setPSO(s_pso_hood);
      cp.setTexture(src, 0, 0); cp.setTexture(s->hi, 1, 0); cp.setSampler(s->sampler, 2);
      cp.setTexture(dst, 3, 1); cp.setBuffer(s->hood_buf, 4);
      cp.dispatch(gx, gy); cp.end();
      swap();
    }
  }

  // 6. halation + bloom. Skip when both off.
  if (run_flare && (s->halation > 1e-4f || s->bloom > 1e-4f) && blur_ok && s_pso_glow.valid() &&
      ensureFlareTextures(s, flare_w, flare_h)) {
    toFlare(src);
    extractFlare(0.75f);
    s_blur16.apply(s->hi, s->hi_a, flare_w, flare_h, 0.02f * fmin);   // bloom
    s_blur16.apply(s->hi, s->hi_b, flare_w, flare_h, 0.05f * fmin);   // halation
    GlowU gu = {};
    gu.bloom = s->bloom; gu.halation = s->halation;
    gu.hr = s->hal_r; gu.hg = s->hal_g; gu.hb = s->hal_b;
    s->glow_buf.writeOne(gu);
    auto cp = gpu::ComputePass::begin(); cp.setPSO(s_pso_glow);
    cp.setTexture(src, 0, 0); cp.setTexture(s->hi_a, 1, 0); cp.setTexture(s->hi_b, 2, 0);
    cp.setSampler(s->sampler, 3); cp.setTexture(dst, 4, 1); cp.setBuffer(s->glow_buf, 5);
    cp.dispatch(gx, gy); cp.end();
    swap();
  }

  // 7. geometry: distortion + transverse chromatic aberration. Skip if neutral.
  { float dist = s->distortion * DIST_SCALE;
    float wave = s->distortion_wave * DIST_SCALE;
    float tca  = s->tca * TCA_SCALE;
    bool active = std::fabs(dist) > 1e-5f || std::fabs(wave) > 1e-5f || std::fabs(tca) > 1e-5f;
    if (active && !dbg_bokeh && !dbg_flare) {
      GeoU u = { half, (float)vp_w, (float)vp_h, dist, wave, tca, 0.f, 0.f };
      s->geo_buf.writeOne(u);
      auto cp = gpu::ComputePass::begin();
      cp.setPSO(s_pso_geo);
      cp.setTexture(src, 0, 0); cp.setSampler(s->sampler, 1);
      cp.setTexture(dst, 2, 1); cp.setBuffer(s->geo_buf, 3);
      cp.dispatch(gx, gy); cp.end();
      swap();
    }
  }

  if (debug == 0) {
    // 8. finish: exposure → vignette → hl_desat → tonemap → sRGB → grain → tex_out.
    FinishU u = {};
    u.half = half; u.dimw = (float)vp_w; u.dimh = (float)vp_h;
    u.exposure = std::pow(2.f, 3.f * s->exposure);
    u.vignette = s->mech_vignette; u.hl_desat = s->hl_desat;
    u.tone = s->tone; u.tone_black = s->tone_black; u.grain = s->grain;
    s->finish_buf.writeOne(u);
    auto cp = gpu::ComputePass::begin();
    cp.setPSO(s_pso_finish);
    cp.setTexture(src, 0, 0); cp.setTexture(out, 1, 1);
    cp.setBuffer(s->finish_buf, 2);
    cp.dispatch(gx, gy); cp.end();
  } else {
    // debug view: render an internal stage instead. aux = tex_in for the highlight
    // mask, the pre-flare snapshot for flare-only, else an (ignored) valid bind.
    DebugU u = {};
    u.mode = (float)debug; u.hl_threshold = s->hl_threshold;
    u.half = half; u.dimw = (float)vp_w; u.dimh = (float)vp_h;
    u.coc_px = coc_px0; u.field_curv = s->field_curvature;
    u.focus_cx = s->focus_cx; u.focus_cy = s->focus_cy;
    s->debug_buf.writeOne(u);
    gpu::Texture aux = (debug == 4 && s->dbg.valid()) ? s->dbg : in;
    auto cp = gpu::ComputePass::begin();
    cp.setPSO(s_pso_debug);
    cp.setTexture(src, 0, 0); cp.setTexture(aux, 1, 0);
    cp.setTexture(out, 2, 1); cp.setBuffer(s->debug_buf, 3);
    cp.dispatch(gx, gy); cp.end();
  }

  gpu::Device::submit();
}

} // namespace lens
