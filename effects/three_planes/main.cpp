/*
 * source.mesh.three_planes — three isometric planes stacked like a 3D chess
 * board, shaded as VCR-era neon.
 *
 * Built for the "Layer^3" show (a three-floor venue). Everything rhythmic is
 * driven from OUTSIDE: the host wires per-plane emission / fill / colour, and
 * sweeps the orbit azimuth on an envelope. This effect just renders the stack
 * beautifully and publishes the screen-space rails the rest of the rig needs.
 *
 * Architecture: the CPU projects 12 corner points per frame (orthographic, so
 * the projection is a plain affine map — no perspective divide, no near
 * plane), and ONE fullscreen compute pass does all the shading from an exact
 * SDF. See render.hlsl for why this beats rasterising the quads.
 *
 * The camera math is deliberately viewport-INDEPENDENT: cover-square coords
 * are already aspect-normalised, so the projected geometry and both published
 * rails can be computed in tick() with no viewport and no GPU readback.
 */

#include <gpu.h>
#include <host.h>
#include <val.h>
#include <effect_utils.h>   // fx::coverSquare
#include <sketch/three_planes_glints.h>
#include <sketch/three_planes_strobe.h>
#include <sketch/led_bars.h>
#include "three_planes_shaders.h"

#include <cmath>
#include <cstdint>

namespace three_planes {

static constexpr int PLANES = 3;
// What a throw does with the stack. See the Release group help.
static constexpr int kModeGrow   = 0;
static constexpr int kModeStrobe = 1;
static constexpr float kPi = 3.14159265358979323846f;

// True isometric: the deck tilt where a unit cube's three visible faces
// project to equal areas. Our default elevation.
static const float kIsoElevationDeg = 35.264389682754654f;

// Mirrors `cbuffer Uniforms` in render.hlsl, row for row.
struct Uniforms {
  float corners[6][4];      // rows 0-5:  plane i -> rows 2i, 2i+1
  float plane_color[3][4];  // rows 6-8:  rgb = colour, w = emission drive
  float fills[4];           // row  9:    xyz = signed fill per plane
  float neon0[4];           // row 10:    line half-width, line gain, whiten, halo r
  float neon1[4];           // row 11:    halo gain, falloff, corner r, aa width
  float misc[4];            // row 12:    fill gain, chroma bleed, input opacity, debug
  float view[4];            // row 13:    vp_w, vp_h, aspect_x, aspect_y
  // The release: the stack thrown. Both modes draw through these same rows —
  // Grow expands them and opens the halo out, Strobe leaves them exactly on
  // the quads and gates them one floor at a time.
  float ghosts[6][4];       // rows 14-19: ghost i -> rows 2i, 2i+1, like `corners`
  float rel[4];             // row 20:    ghost halo radius, falloff, -, -
  float ring_gain[4];       // row 21:    per-ghost gain: the throw, damped and gated
  float glim0[4];           // row 22:    travel dir x, dir y, -, -
  // One row per glint IN FLIGHT: where it is on the travel axis (cover-square),
  // its half-width there, and the two look values it was born with. A dead slot
  // is zero gain and zero shade, so the shader needs no count and no branch.
  float glints[8][4];       // rows 23-30: axis, half-width, gain, shade
  float mask[4];            // row 31:    strength (0 = nothing wired), halo cut, -, -
  float grade[16];          // rows 32-35: VcrGrade
};
static_assert(sizeof(Uniforms) == 576, "Uniforms layout mismatch with render.hlsl");
static_assert(three_planes_glints::kMaxLive == 8, "glint rows must match kMaxLive");
static_assert(PLANES == led_bars::kLayers, "the LED map shows every floor");

// Mirrors `cbuffer LedUniforms` in shaders_common/nano_led_bars.hlsl.
struct LedUniforms {
  float cell[led_bars::kCells][4];   // bar-major, segment 0 at the bottom
  float ctl[4];                      // quantize, -, -, -
};
static_assert(sizeof(LedUniforms) == 16 * (led_bars::kCells + 1),
              "LedUniforms layout mismatch with nano_led_bars.hlsl");

// Mirrors `cbuffer WallUniforms` in wall.hlsl, row for row. Rings 0-2 are the
// floors and 3-5 the release ghosts — one bank, because by the time it gets
// here a ghost is just another ring of light in the room.
static constexpr int WALL_RINGS = PLANES * 2;
struct WallUniforms {
  float ring[WALL_RINGS][4];    // rows 0-5:   rgb = colour, w = level
  float ring_g[WALL_RINGS][4];  // rows 6-11:  centre height, centre depth, radius
  float ring_c[24][4];          // rows 12-35: a corner each, turned AND tilted
  float wall[4];                // row 36:     wash reach, wash weight, gain, gap
  float look[4];                // row 37:     warmth, side, -, wall x
  float glim[4];                // row 38:     travel heading across the floor
  float glints[8][4];           // rows 39-46
};
static_assert(sizeof(WallUniforms) == 752, "WallUniforms layout mismatch with wall.hlsl");

struct State {
  // --- Planes (the externally-driven rhythm surface) ---
  float emission[PLANES] = {0.85f, 0.85f, 0.85f};
  float fill[PLANES]     = {0.0f, 0.0f, 0.0f};
  float color[PLANES][3] = {{1.00f, 0.22f, 0.62f},   // magenta  (bottom)
                            {0.30f, 0.85f, 1.00f},   // cyan     (middle)
                            {0.72f, 0.35f, 1.00f}};  // violet   (top)

  // --- Camera ---
  float orbit_azimuth = 0.125f;              // [0,1] -> 0..360 deg; 0.125 = 45 deg
  float elevation_deg = kIsoElevationDeg;
  float zoom          = 0.55f;
  float plane_spacing = 0.42f;
  float plane_size    = 0.62f;
  float corner_radius = 0.012f;

  // --- Neon ---
  float line_width   = 0.18f;
  float line_gain    = 1.60f;
  float core_whiten  = 0.85f;
  float halo_radius  = 0.30f;
  float halo_gain    = 0.55f;
  float halo_falloff = 0.45f;
  float halo_smooth  = 0.35f;
  float fill_gain    = 0.22f;

  // --- Release ---
  // The throw, from the rig: 1 the instant a mute spends the latched charge,
  // falling to 0 across the ring-out. Everything here is a pure function of
  // it, which is what keeps the rings stateless.
  float release        = 0.0f;
  float release_expand = 1.80f;   // how far the rings fly, as a fraction of the stack
  float release_gain   = 2.20f;
  float release_blur   = 10.0f;   // how far the halo opens out as they go
  float release_contrast = 0.80f; // how hard a lit plane holds its own ring down

  // Which throw. Grow flings the stack outward; Strobe keeps it exactly where
  // it is and flams the floors one at a time on a roll that putters out. The
  // two draw through the SAME three ghost quads — see render() — so the mode
  // costs nothing but the numbers fed to them.
  int   release_mode  = kModeGrow;
  float strobe_rate   = 22.0f;    // steps per second
  float strobe_duty   = 0.55f;    // on-window per step, at full release
  float strobe_gain   = 1.10f;
  float strobe_glow   = 0.28f;    // halo radius, as a fraction of the live one
  float strobe_grace  = 0.35f;
  three_planes_strobe::Core strobe;

  // One frame of nothing between the throw and the rings appearing. Counted
  // in tick(), because render() can be called without one.
  float prev_release = 0.0f;
  int ring_delay = 0;

  // --- Glimmer ---
  // The rhythm still comes from outside — `glimmer_drive` is the rig's Sweep
  // knob — but the glints themselves are PARTICLES with lifetimes, so unlike
  // everything else in this effect they are an accumulator. See
  // <sketch/three_planes_glints.h>; it is why the effect is only
  // SeekableApproximate rather than TimeIndependent.
  three_planes_glints::Core glint_core;
  float glimmer_sweep   = 0.5f;    // the knob itself, wired from the rig
  float glimmer_band    = 0.45f;   // launch band, fraction of each half of the throw
  float glimmer_ratio   = 1.0f;    // crossings per knob range
  float glimmer_chaos   = 4.0f;    // small extra glints per second, when sweeping fast
  float glimmer_angle   = 45.0f;   // degrees, travel direction, CCW from +x
  float glimmer_width   = 0.07f;   // glint half-width, fraction of one crossing
  float glimmer_gain    = 1.6f;
  float glimmer_shadow  = 0.55f;

  // --- Grade ---
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
  float input_opacity   = 1.0f;
  float mask_strength   = 1.0f;
  float mask_halo       = 0.6f;

  // --- LED bars ---
  // The pixel map for the house rig, published on `led_out`. Effect-owned and
  // dispatched only when something is wired to it — see renderLed().
  // --- Walls ---
  // The impact light the stack throws into the room, published on `left_out`
  // and `right_out`. Effect-owned and dispatched only when wired.
  float wall_gain    = 1.0f;
  float wall_gap     = 0.55f;
  float wall_reach   = 0.45f;
  float wall_bounce  = 0.14f;
  float wall_warmth  = 0.55f;
  float wall_depth   = 1.0f;
  gpu::Texture wall_tex[2];
  gpu::Buffer  wall_buf[2];
  int wall_w[2] = {0, 0};
  int wall_h[2] = {0, 0};

  bool  led_quantize = true;
  // What the bars show when `led_solid` is dialled off the picture. Wired from
  // Three Planes Rig's six LED rails; the defaults mirror the plane fields, so
  // an unwired card that reaches for the knob finds a plausible tower rather
  // than a black one.
  float led_emission[PLANES] = {0.85f, 0.85f, 0.85f};
  float led_color[PLANES][3] = {{1.00f, 0.22f, 0.62f},
                                {0.30f, 0.85f, 1.00f},
                                {0.72f, 0.35f, 1.00f}};
  float led_solid = 1.0f;
  gpu::Texture led_tex;
  gpu::Buffer  led_buf;
  int   led_w = 0;
  int   led_h = 0;

  // --- Debug ---
  bool debug_show_sdf    = false;
  bool debug_show_planes = false;

  // --- Derived, recomputed each tick ---
  float corner_x[PLANES][4] = {};
  float corner_y[PLANES][4] = {};
  float plane_y[PLANES]     = {};
  float plane_half_h[PLANES]= {};

  bool initialized = false;
  gpu::Buffer uniform_buf;
};

static gpu::ComputePSO s_pso;
static gpu::ComputePSO s_led_pso;
static gpu::ComputePSO s_wall_pso;

// --- Perceptual mappings (style guide 1.3) --------------------------------
// Every one of these takes a normalised slider and returns the value the
// shader actually wants, so the UI stays in [0,1] and taps compose.

static inline float clamp01f(float v) { return v < 0.0f ? 0.0f : (v > 1.0f ? 1.0f : v); }

// Emission: a dimmer curve. Slightly steeper than linear so the bottom of the
// range stays dark and the top has real punch to blow the cores out.
static inline float emissionDrive(float e) {
  return std::pow(e < 0.0f ? 0.0f : e, 1.8f) * 3.2f;
}
// What one plane is worth to anything OUTSIDE the picture — the LED map, the
// walls. The SAME dimmer curve, normalised so a plane at full emission is a
// fully-lit segment and a fully-lit wall. Divided by the curve rather than
// restated as a number, so neither can drift from what the eye sees on screen
// — and so the headroom above 1.0 that the emission field carries (a bounce
// overshooting the base level) reads out there as saturation, which is what an
// overdriven LED and an overexposed wall both do.
static inline float litLevel(float e) {
  return clamp01f(emissionDrive(e) / emissionDrive(1.0f));
}
// Line half-width in cover-square units: 0.0012 .. 0.022, quadratic so the
// hairline end of the range gets most of the slider.
static inline float lineHalfWidth(float w) {
  float t = w < 0.0f ? 0.0f : (w > 1.0f ? 1.0f : w);
  return 0.0012f + 0.0208f * t * t;
}
// How far the wall stands off the stack, in COVER-SQUARE units — the picture's
// own, so the light keeps step with the tower as the zoom moves it. 0.008 ..
// 0.4, exponential, because the useful end is the tight one: below about half
// the floor spacing the three pools read as three and above it they merge.
static inline float wallGap(float t) {
  float u = t < 0.0f ? 0.0f : (t > 1.0f ? 1.0f : t);
  return 0.01f * std::pow(30.0f, u);
}
// How far the bounce carries: 0.1 .. 1.0 of the same units, linear — it is a
// wash, and there is nothing to resolve at either end of it.
static inline float wallReach(float t) {
  float u = t < 0.0f ? 0.0f : (t > 1.0f ? 1.0f : t);
  return 0.1f + 0.9f * u;
}
// Halo radius: exponential, 0.006 .. 0.24 cover-square units.
static inline float haloRadius(float r) {
  float t = r < 0.0f ? 0.0f : (r > 1.0f ? 1.0f : r);
  return 0.006f * std::pow(40.0f, t);
}

// Strobe's halo, as a fraction of the live one. Linear, and never quite zero:
// the profile peaks at 1 whatever its radius, so at 0 you would still get a
// line — just an aliased one-pixel line with nothing around it, which is a
// worse picture than the hairline this floors it at.
static inline float strobeGlow(float t) {
  float k = t < 0.0f ? 0.0f : (t > 1.0f ? 1.0f : t);
  return 0.02f + 0.98f * k;
}

// Strobe's per-floor duty scale: exactly Grow's local-contrast damping, spent
// on the LENGTH of a hit instead of on its brightness. A floor that is already
// lit underneath keeps its flam at full strength and gets less of it, and once
// its window falls under a frame the flam simply stops landing there.
static inline float strobeWeight(const State& s, int i) {
  const float lit = s.emission[i] < 0.0f ? 0.0f
                  : (s.emission[i] > 1.0f ? 1.0f : s.emission[i]);
  const float w = 1.0f - s.release_contrast * lit;
  return w < 0.0f ? 0.0f : w;
}

// How far a glint's drawn profile reaches on either side of its centre, in its
// own half-widths. Mirrors render.hlsl: the super-Gaussian core has died by
// about 1.5, and the Gaussian wake trailing it by about 5.5.
//
// Deliberately asymmetric, because the two margins are doing different jobs.
// The entry margin only has to hide the core's leading edge so the glint fades
// in rather than popping; the exit margin has to let the whole wake finish
// crossing before the glint is retired.
static constexpr float kGlintSkirt = 1.5f;
static constexpr float kGlintWakeReach = 5.5f;

// --- Camera ---------------------------------------------------------------
// Planes are squares in the model XZ plane at y = -spacing, 0, +spacing. The
// origin IS the middle plane's centre, so orbiting about it is free.
//
//   azimuth theta about Y, then elevation phi tilts the deck. Orthographic,
//   so this is one affine map — corners project exactly, no divide.
//
// Cover-square y grows DOWNWARD (uv.y is 0 at the top), hence the negations.
static void projectPlanes(State& s) {
  const float th = s.orbit_azimuth * 2.0f * kPi;
  const float ph = s.elevation_deg * (kPi / 180.0f);
  const float ct = std::cos(th), st = std::sin(th);
  const float cp = std::cos(ph), sp = std::sin(ph);
  const float half = s.plane_size;

  // Model-space corners of a square in XZ, wound consistently.
  const float cx[4] = {-half, +half, +half, -half};
  const float cz[4] = {-half, -half, +half, +half};

  for (int i = 0; i < PLANES; i++) {
    const float y = (float(i) - 1.0f) * s.plane_spacing;   // 0 = bottom floor
    for (int k = 0; k < 4; k++) {
      const float xr =  cx[k] * ct - cz[k] * st;
      const float zr =  cx[k] * st + cz[k] * ct;
      s.corner_x[i][k] = xr * s.zoom;
      s.corner_y[i][k] = (-y * cp - zr * sp) * s.zoom;
    }
    // The plane's CENTRE sits on the orbit axis (x = z = 0), so its screen Y
    // is a pure function of elevation and zoom — azimuth cannot move it.
    // That is what makes this rail stable enough to composite against.
    s.plane_y[i] = -y * cp * s.zoom;
    // Silhouette half-height DOES swing with azimuth: max |zr| over the four
    // corners is half * (|sin| + |cos|).
    s.plane_half_h[i] =
        sp * s.zoom * half * (std::fabs(st) + std::fabs(ct));
  }
}

// The two throw modes have disjoint controls and nothing to say about each
// other's, so a card only ever shows the one it is in.
static void applyModeVisibility(const State* s) {
  const bool strobe = s->release_mode == kModeStrobe;
  state::setFieldHidden("release_expand", strobe);
  state::setFieldHidden("release_gain",   strobe);
  state::setFieldHidden("release_blur",   strobe);
  state::setFieldHidden("strobe_rate",   !strobe);
  state::setFieldHidden("strobe_duty",   !strobe);
  state::setFieldHidden("strobe_gain",   !strobe);
  state::setFieldHidden("strobe_glow",   !strobe);
  state::setFieldHidden("strobe_grace",  !strobe);
  // `release_contrast` belongs to both: it is the rule about two bright things
  // in one place, and Strobe puts them there by construction.
}

// Fires once after init and the initial state replay, so a restored card never
// flashes the other mode's fields.
static void on_state_ready(void* self) {
  auto* s = static_cast<State*>(self);
  if (s) applyModeVisibility(s);
}

static void publish(const char* name, float value) {
  auto vh = val::number(value);
  state::setValPath(name, vh);
  val::release(vh);
}

// Both rails are pure functions of the camera params, so publishing them is
// cheap and idempotent. We do it from tick() AND render(): tick() so taps read
// THIS frame's value before it is consumed, render() so a host that renders
// without ticking (thumbnails, off-playhead previews) still gets live rails
// instead of zeros.
static void publishRails(const State& s) {
  publish("plane1_y", s.plane_y[0]);
  publish("plane2_y", s.plane_y[1]);
  publish("plane3_y", s.plane_y[2]);
  publish("plane1_half_h", s.plane_half_h[0]);
  publish("plane2_half_h", s.plane_half_h[1]);
  publish("plane3_half_h", s.plane_half_h[2]);
}

void module_init() {
  state::init("source.mesh.three_planes", {1, 0, 0},
    state::Schema()
      .helpField("intro",
        "## Three Planes\n"
        "Three isometric planes stacked like a 3D chess board, shaded as "
        "VCR-era neon. Built for a three-floor venue, but it reads as a "
        "level meter anywhere.\n\n"
        "Each plane has three states: **empty** (fill 0, just the glowing "
        "outline), **filled** (fill > 0, neon flood) and **masked** "
        "(fill < 0, a black body that eats the glow of everything beneath "
        "it while keeping its own outline). That last one is the whole "
        "trick — it lets the stack read as solid geometry instead of three "
        "transparent overlays.\n\n"
        "**Try:** drive the three *Emission* knobs from an envelope follower "
        "for a peak-holding VU tower — hold the peak plane on a different "
        "*Colour*. Sweep *Orbit Azimuth* slowly under it; the published "
        "`planeN_y` rails stay rock steady while it turns, so anything you "
        "composite on top stays glued to its floor.")

      // ---------------- Planes ----------------
      .group("planes", "Planes")
        .groupHelp(
          "The performance surface — wire all nine of these. **Emission** is "
          "the light coming up, on a dimmer curve. **Fill** is signed: push "
          "it positive to flood the plane with neon, negative to turn it "
          "into a black mask that occludes the planes below. **Colour** is "
          "what the halo carries; the line core always blows out toward "
          "white (see *Core Whiten*).\n\n"
          "Plane 1 is the ground floor, plane 3 the top.\n\n"
          "**Emission runs past 1.** Fully lit is 1; the rest of the range is "
          "overdrive, where the cores blow out and the halo goes with them. "
          "That headroom is there so a level can OVERSHOOT its own base — it "
          "is what Three Planes Rig's *Bounce* spends on the way back in — "
          "and it is yours to dial by hand too.")
      .floatField("plane1_emission", 0.85f, 0.f, 1.5f, state::PrimaryInput)
        .label("Plane 1 Emission", "P1 Emit")
      .floatField("plane1_fill", 0.0f, -1.f, 1.f, state::PrimaryInput, "signed")
        .label("Plane 1 Fill", "P1 Fill")
      .rgbField("plane1_color", 1.00f, 0.22f, 0.62f, state::PrimaryInput)
        .label("Plane 1 Colour", "P1 Col")
      .floatField("plane2_emission", 0.85f, 0.f, 1.5f, state::PrimaryInput)
        .label("Plane 2 Emission", "P2 Emit")
      .floatField("plane2_fill", 0.0f, -1.f, 1.f, state::PrimaryInput, "signed")
        .label("Plane 2 Fill", "P2 Fill")
      .rgbField("plane2_color", 0.30f, 0.85f, 1.00f, state::PrimaryInput)
        .label("Plane 2 Colour", "P2 Col")
      .floatField("plane3_emission", 0.85f, 0.f, 1.5f, state::PrimaryInput)
        .label("Plane 3 Emission", "P3 Emit")
      .floatField("plane3_fill", 0.0f, -1.f, 1.f, state::PrimaryInput, "signed")
        .label("Plane 3 Fill", "P3 Fill")
      .rgbField("plane3_color", 0.72f, 0.35f, 1.00f, state::PrimaryInput)
        .label("Plane 3 Colour", "P3 Col")

      // ---------------- Camera ----------------
      .group("camera", "Camera")
        .groupHelp(
          "Orthographic throughout — orbiting never introduces perspective, "
          "so the three planes stay exactly parallel and the stack keeps "
          "reading as a diagram rather than a photograph.\n\n"
          "*Orbit Azimuth* is normalised to [0,1] precisely because it is "
          "meant to be swept from an envelope. *Elevation* defaults to "
          "35.26 deg, the true isometric tilt; drop it toward 0 for a flat "
          "side-on stack, push it up for a top-down board.")
      .floatField("orbit_azimuth", 0.125f, 0.f, 1.f, state::PrimaryInput,
                  "unsigned", 0.f, nullptr,
                  "Turntable angle. 0..1 maps to a full 360 deg turn.")
        .label("Orbit Azimuth", "Orbit")
      .floatField("elevation", kIsoElevationDeg, 0.f, 89.f, state::PrimaryInput,
                  nullptr, 0.f, "deg",
                  "Deck tilt. 35.26 deg is true isometric.")
        .label("Elevation", "Elev")
      .floatField("zoom", 0.55f, 0.05f, 2.f, state::PrimaryInput)
        .label("Zoom", "Zoom")
      .floatField("plane_spacing", 0.42f, 0.f, 1.5f, state::PrimaryInput)
        .label("Plane Spacing", "Space")
      .floatField("plane_size", 0.62f, 0.05f, 1.5f, state::PrimaryInput)
        .label("Plane Size", "Size")
      .floatField("corner_radius", 0.012f, 0.f, 0.25f, state::SecondaryInput)
        .label("Corner Radius", "Corner")

      // ---------------- Neon ----------------
      .group("neon", "Neon & Halo")
        .groupHelp(
          "The halo is analytic — an exponential of the true distance to the "
          "outline — so *Halo Radius* costs nothing to widen and stays "
          "perfectly smooth around corners.\n\n"
          "**Core Whiten** is the knob that decides whether this reads as "
          "neon at all: real neon photographs have a white-hot filament with "
          "the colour surviving only out in the glow. At 0 the line stays "
          "fully tinted and looks like vector art; push it up and the tube "
          "lights.")
      .floatField("line_width", 0.18f, 0.f, 1.f, state::PrimaryInput)
        .label("Line Width", "Width")
      .floatField("line_gain", 1.60f, 0.f, 4.f, state::PrimaryInput)
        .label("Line Gain", "Line")
      .floatField("core_whiten", 0.85f, 0.f, 1.f, state::PrimaryInput)
        .label("Core Whiten", "Whiten")
      .floatField("halo_radius", 0.30f, 0.f, 1.f, state::PrimaryInput)
        .label("Halo Radius", "Halo R")
      .floatField("halo_gain", 0.55f, 0.f, 3.f, state::PrimaryInput)
        .label("Halo Gain", "Halo")
      .floatField("halo_falloff", 0.45f, 0.f, 1.f, state::SecondaryInput,
                  nullptr, 0.f, nullptr,
                  "0 = tight and punchy, 1 = wide and soft.")
        .label("Halo Falloff", "Fall")
      .floatField("halo_smooth", 0.35f, 0.f, 1.f, state::SecondaryInput,
                  nullptr, 0.f, nullptr,
                  "Rounds the corner creases just OUTSIDE the outline, as a "
                  "fraction of the halo radius. 0 shows the raw ridge. The "
                  "interior does not use this — it sums the four edges.")
        .label("Halo Smoothing", "Smooth")
      .floatField("fill_gain", 0.22f, 0.f, 2.f, state::SecondaryInput)
        .label("Fill Gain", "Fill G")

      // ---------------- Release ----------------
      .group("release", "Release")
        .groupHelp(
          "The throw. Wire *Release* from **Three Planes Rig** and reaching "
          "either end of the sweep stops being merely a blackout: the stack is "
          "flung outward as three expanding rings that ring down on their own "
          "clock, over the muted picture.\n\n"
          "They are the SAME three quads, thrown — the outline and nothing "
          "else, since a ring has no inside. As they fly they open out and go "
          "soft: bright and tight at the throw, wide and dull by the end. That "
          "is deliberately what a filter closing sounds like, and it is why "
          "the tail reads as a special state rather than as the picture merely "
          "being dimmer.\n\n"
          "The rig gives the tail a straight-line fall on purpose, which makes "
          "this a constant outward speed — a shockwave with an end, rather "
          "than something that leaps out and then creeps.\n\n"
          "Nothing about where the knob goes next can cancel a throw, so "
          "sweeping straight back relights the tower over a tail still "
          "running. That overlap is the move — and *Local Contrast* is what "
          "keeps it legible, holding a ring down while it is still sitting on "
          "the plane that threw it.\n\n"
          "The rings appear one frame AFTER the throw, on purpose. The frame "
          "that fires one is already black, so the picture lands on nothing "
          "before it lands on the release, and the hit reads harder for it.\n\n"
          "**Strobe** is the same throw pointed the other way. Nothing flies: "
          "the stack comes straight back exactly where it was, as bare "
          "wireframe with barely any glow, and then breaks up — the floors "
          "flam one at a time on a fast roll whose hits get SHORTER as the "
          "tail runs down, until they start missing frames outright and it "
          "sputters out.\n\n"
          "Nothing in Strobe ever dims. Every hit lands at full strength or "
          "does not land, and everything that would turn one down — the decay, "
          "the local contrast — takes its window away instead. A strobe that "
          "fades reads as a light going out; one that thins reads as a thing "
          "running down.\n\n"
          "Grow is energy leaving the frame; Strobe is energy rattling around "
          "inside it, and they cut against each other well enough to be worth "
          "switching between on the fly.")
      .floatField("release", 0.0f, 0.f, 1.f, state::PrimaryInput,
                  "unsigned", 0.f, nullptr,
                  "The throw, ringing out. At 0 there is nothing to see, so an "
                  "unwired card is exactly as it was.")
        .label("Release", "Rel")
      .selectField("release_mode", kModeGrow, state::PrimaryInput,
                   {{"Grow", kModeGrow}, {"Strobe", kModeStrobe}}, false,
                   "What a throw does with the stack. **Grow** flings it "
                   "outward as three rings that open up and go soft. "
                   "**Strobe** leaves it where it is and flams the floors one "
                   "at a time until the roll putters out. Each mode shows only "
                   "its own controls; *Local Contrast* belongs to both.")
        .label("Throw Mode", "Mode")
      .floatField("release_expand", 1.80f, 0.f, 4.f, state::PrimaryInput,
                  nullptr, 0.f, nullptr,
                  "How far the rings fly, as a fraction of the stack\'s own "
                  "size. They start exactly on the quads that threw them.")
        .label("Throw Distance", "Throw")
      .floatField("release_gain", 2.20f, 0.f, 6.f, state::PrimaryInput,
                  nullptr, 0.f, nullptr,
                  "How bright the rings are at the moment of the throw. They "
                  "dim as they open out, so this is the punch, not the tail.")
        .label("Throw Gain", "RelGain")
      .floatField("release_blur", 10.0f, 1.f, 30.f, state::SecondaryInput,
                  nullptr, 0.f, nullptr,
                  "How far the rings open out as they fly — the filter "
                  "closing. 1 keeps them as tight as the tubes they came off; "
                  "high values leave a wide, dull smear behind.")
        .label("Throw Blur", "RelBlur")

      // Strobe. Rate stays put through the whole tail on purpose — the decay
      // is the WINDOW closing, not the roll slowing down, because a roll that
      // slows reads as a machine winding down and a window that closes reads
      // as one being switched off.
      .floatField("strobe_rate", 22.0f, 4.f, 60.f, state::PrimaryInput,
                  nullptr, 0.f, "steps/s",
                  "How fast the roll goes. It does not change as the tail runs "
                  "down; only the hits get shorter.")
        .label("Roll Rate", "Rate")
      .floatField("strobe_duty", 0.55f, 0.f, 1.f, state::PrimaryInput,
                  nullptr, 0.f, nullptr,
                  "How much of each step a floor is lit for, at the moment of "
                  "the throw. This is the thing that putters out: the release "
                  "scales it, so by the end the window is shorter than a frame "
                  "and the hits start missing frames altogether. High values "
                  "run the floors into each other and it reads as a wave; low "
                  "ones are already sparse at the top of the tail.")
        .label("Duty", "Duty")
      .floatField("strobe_gain", 1.10f, 0.f, 4.f, state::PrimaryInput,
                  nullptr, 0.f, nullptr,
                  "How bright a hit is — and every hit is this bright, "
                  "including the last one. The decay is in the timing.")
        .label("Roll Gain", "RollG")
      .floatField("strobe_glow", 0.28f, 0.f, 1.f, state::PrimaryInput,
                  nullptr, 0.f, nullptr,
                  "How much halo the wireframe carries, as a fraction of the "
                  "live one. Near 0 it is a bare tube with a breath around it, "
                  "which is the whole point of the mode: the throw looks like "
                  "the drawing under the picture rather than like the picture "
                  "again.")
        .label("Wire Glow", "WireG")
      .floatField("strobe_grace", 0.35f, 0.f, 1.f, state::SecondaryInput,
                  nullptr, 0.f, nullptr,
                  "The flam. The next floor speaks quietly just before each "
                  "step changes, then lands — two strokes where a roll would "
                  "have one, as a fraction of a step. At 0 it is a plain "
                  "metronome up and down the tower; near 1 the two strokes run "
                  "together and the roll reads as a wave instead.")
        .label("Flam", "Flam")

      .floatField("release_contrast", 0.80f, 0.f, 1.f, state::PrimaryInput,
                  nullptr, 0.f, nullptr,
                  "How hard a lit plane holds its own ghost down while the "
                  "ghost is still on top of it. Sweeping straight back after a "
                  "throw relights the tower underneath a ring that has barely "
                  "left it, and two bright things in the same place read as "
                  "one; this keeps them apart. In **Grow** it fades out as the "
                  "ring flies clear, so a ring well away from the stack is at "
                  "full strength whatever the tower is doing.\n\n"
                  "**Strobe** spends it on the WINDOW instead of on the "
                  "brightness — a lit floor gets a shorter flam, not a dimmer "
                  "one, and once its window drops under a frame the flam stops "
                  "landing on that floor at all. Nothing in that mode ever "
                  "turns a hit down when it could drop it.")
        .label("Local Contrast", "Contrast")

      // ---------------- Glimmer ----------------
      .group("glimmer", "Glimmer")
        .groupHelp(
          "Slanted glints that travel across the stack — the glare off metal "
          "in an old cel-animated show.\n\n"
          "**The glint is the gesture.** Wire *Glint Sweep* from the rig\'s "
          "*Sweep Out* and the rule is: move the knob at a steady speed across "
          "its whole range, and ONE glint crosses the stack at exactly that "
          "rate. The launch is the moment the knob enters the middle band — "
          "which happens once per traverse, and can only happen while you are "
          "moving, so there is always a real speed for the new glint to "
          "take.\n\n"
          "Which WAY you move it is thrown away. Reverse mid-gesture and the "
          "glints already out there carry on exactly as they were: they are "
          "objects in flight, not a readout. Nothing about the knob reaches a "
          "live glint again except its speed — and that is shared by all of "
          "them, because at their own speeds two would eventually cross, and "
          "the moment they overlap they stop being two things.\n\n"
          "Sweep hard and *Chaos* starts adding small ones underneath, at "
          "random intervals. They are dimmer and narrower on purpose, so the "
          "gesture stays legible through them.\n\n"
          "Glints multiply the **emission** of everything the card draws — the "
          "planes and the release ghosts alike — rather than the finished "
          "picture. That is the whole trick: emission scales the line core, "
          "the halo and the fill together, so one crossing a tube brightens "
          "the glow around it too and reads as light IN the tube rather than a "
          "highlight pasted over it. They are also born just outside the stack "
          "and retired just past it, rather than crossing the whole frame, so "
          "one starts working the moment it is thrown.\n\n"
          "*Shadow* is what sells it — a dark wake trailing each glint, so the "
          "stack gains contrast rather than just getting brighter.\n\n"
          "A throw holds whatever is in the air at full strength while it "
          "rings out, in both modes — letting go of the knob to reach a mute "
          "is exactly the gesture that fires one, so a glint being thrown is "
          "never a glint running down. What differs is where they go: **Grow** "
          "slings them off with everything else, and **Strobe**, which keeps "
          "everything inside the frame, lets them hang about and drift over "
          "the flam instead.")
      .floatField("glimmer_sweep", 0.5f, 0.f, 1.f, state::PrimaryInput,
                  "unsigned", 0.f, nullptr,
                  "The sweep knob itself — wire it from Three Planes Rig's "
                  "*Sweep Out*. Its MOTION is the whole input: a glint "
                  "launches every time it enters the middle band, travelling "
                  "at the speed you moved.")
        .label("Glint Sweep", "Sweep")
      .floatField("glimmer_ratio", 1.0f, 0.f, 4.f, state::PrimaryInput,
                  nullptr, 0.f, nullptr,
                  "Crossings of the picture per full range of the knob. At 1 "
                  "a sweep across the whole knob is one glint across the whole "
                  "stack, in the same time — turn it up and the glint outruns "
                  "your hand.")
        .label("Glint Ratio", "Ratio")
      .floatField("glimmer_band", 0.45f, 0.02f, 1.f, state::PrimaryInput,
                  nullptr, 0.f, nullptr,
                  "How wide the middle band is, as a fraction of each half of "
                  "the throw. Entering it is what launches a glint, so this is "
                  "where in the sweep it happens. Matching the rig's Deadzone "
                  "puts the launch exactly where the tower is at full "
                  "brightness.")
        .label("Launch Band", "Band")
      .floatField("glimmer_chaos", 4.0f, 0.f, 16.f, state::PrimaryInput,
                  nullptr, 0.f, "/s",
                  "Small extra glints per second, once the sweep is brisk. "
                  "They are dimmer and narrower than the launched one on "
                  "purpose — the gesture stays legible and the chaos sits "
                  "under it. 0 leaves one clean glint per pass.")
        .label("Chaos", "Chaos")
      .floatField("glimmer_gain", 1.6f, 0.f, 4.f, state::PrimaryInput,
                  nullptr, 0.f, nullptr,
                  "How hard a glint lifts the emission it crosses. Each one "
                  "takes a random share of this at birth.")
        .label("Glint Gain", "GlGain")
      .floatField("glimmer_shadow", 0.55f, 0.f, 1.f, state::PrimaryInput,
                  nullptr, 0.f, nullptr,
                  "The dark wake trailing each glint, as a fraction of full "
                  "extinction.")
        .label("Glint Shadow", "Shadow")
      .floatField("glimmer_angle", 45.f, 0.f, 360.f, state::SecondaryInput,
                  nullptr, 0.f, "deg",
                  "Which way the glints travel. 45 deg runs bottom-left to "
                  "top-right; each glint sits square across that.")
        .label("Glint Angle", "GlAng")
      .floatField("glimmer_width", 0.07f, 0.01f, 0.4f, state::SecondaryInput,
                  nullptr, 0.f, nullptr,
                  "Glint width, as a fraction of the distance it travels — "
                  "so it scales with the stack rather than with the frame. "
                  "Small is a hard slash, large is a soft sheen.")
        .label("Glint Width", "GlWid")

      // ---------------- Grade ----------------
      .group("grade", "Warmth & Dehancement")
        .groupHelp(
          "The analogue tail. *Chroma Bleed* splits R/G/B horizontally the "
          "way a tape transport does — here it is exact, because the whole "
          "stack is re-evaluated at three offsets rather than blurred.\n\n"
          "*Drive* and *Asymmetry* are where warmth actually lives: the "
          "asymmetric bias makes even and odd harmonics unequal instead of "
          "just rounding the peaks. *Highlight Desat* runs before the curve, "
          "in HDR, so hot cores bleach to white properly.\n\n"
          "**Try:** Drive up + Toe up + Scanlines low is a tired VHS dub; "
          "everything near 0 is a clean vector look.")
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
      .floatField("input_opacity", 1.0f, 0.f, 1.f, state::SecondaryInput,
                  nullptr, 0.f, nullptr,
                  "How much of the incoming image survives under the stack.")
        .label("Input Opacity", "In Op")

      // ---------------- Debug ----------------
      .group("debug", "Debug")
      .boolField("debug_show_sdf", false, state::SecondaryInput,
                 "Banded distance field — check corner morphology.")
        .label("Show Distance Field", "SDF")
      .boolField("debug_show_planes", false, state::SecondaryInput,
                 "Flat per-plane keys, no glow or grade — check projection "
                 "and stacking order.")
        .label("Show Plane Keys", "Keys")

      // ---------------- Outputs ----------------
      // Declared min/max IS the modulation contract for these rails.
      .floatField("plane1_y", 0.f, -1.f, 1.f, state::PrimaryOutput, "signed",
                  0.f, nullptr, "Screen Y of plane 1's centre, cover-square.")
        .label("Plane 1 Y", "P1 Y")
      .floatField("plane2_y", 0.f, -1.f, 1.f, state::PrimaryOutput, "signed",
                  0.f, nullptr, "Screen Y of plane 2's centre, cover-square.")
        .label("Plane 2 Y", "P2 Y")
      .floatField("plane3_y", 0.f, -1.f, 1.f, state::PrimaryOutput, "signed",
                  0.f, nullptr, "Screen Y of plane 3's centre, cover-square.")
        .label("Plane 3 Y", "P3 Y")
      .floatField("plane1_half_h", 0.f, 0.f, 1.f, state::PrimaryOutput, "unsigned",
                  0.f, nullptr, "Half-height of plane 1's silhouette.")
        .label("Plane 1 Half Height", "P1 H")
      .floatField("plane2_half_h", 0.f, 0.f, 1.f, state::PrimaryOutput, "unsigned",
                  0.f, nullptr, "Half-height of plane 2's silhouette.")
        .label("Plane 2 Half Height", "P2 H")
      .floatField("plane3_half_h", 0.f, 0.f, 1.f, state::PrimaryOutput, "unsigned",
                  0.f, nullptr, "Half-height of plane 3's silhouette.")
        .label("Plane 3 Half Height", "P3 H")

      // tex_in FIRST, and then the mask. The editor takes a module's texture
      // input to be the FIRST texture input its schema declares
      // (schema-channels.ts, firstFieldOfType — it sorts on declaration
      // order), so a second texture input declared ahead of tex_in quietly
      // becomes THE input: the chain's image is what the card offers a
      // dropped wire, and the aux port is the one that disappears.
      .textureField("tex_in",  state::PrimaryInput)
      .textureField("tex_out", state::PrimaryOutput)

      // ---------------- Mask ----------------
      .group("mask", "Mask")
      .groupHelp(
        "A second image — text, a logo, any shape — wired into *Mask In* and "
        "laid over the stack in SCREEN space, so what you see in that input is "
        "where it lands.\n\n"
        "What it covers is CUT AWAY, exactly the way a plane at Fill -1 cuts: "
        "a hole through the picture rather than a sticker on it. That is why "
        "there is no colour here — a mask has a shape, not a look.\n\n"
        "Its weight is the mask's **alpha times its luma**, and it wants both. "
        "A shape that is present but black is not a mask, and neither is a "
        "bright shape that is not there — so an ordinary rendered logo works "
        "as it comes, with no separate matte to author and keep in step.\n\n"
        "*Halo Cut* is the one place the hole is not clean. A tube behind a "
        "letter still throws light around the letter's edges, so by default "
        "the mask takes the neon's body outright and only some of its glow. "
        "Turn it up for a hard stencil; turn it down and the shape sits deep "
        "in the light instead of on the glass.")
      .textureField("mask_in", state::SecondaryInput)
        .label("Mask In", "Mask")
      .floatField("mask_strength", 1.0f, 0.f, 1.f, state::PrimaryInput,
                  nullptr, 0.f, nullptr,
                  "How hard the mask cuts. 0 ignores it entirely.")
        .label("Mask Amount", "Amt")
      .floatField("mask_halo", 0.6f, 0.f, 1.f, state::PrimaryInput,
                  nullptr, 0.f, nullptr,
                  "How much of the HALO the mask takes with it. 1 cuts the "
                  "glow as hard as the tube; 0 lets all of it bleed over the "
                  "shape.")
        .label("Halo Cut", "Halo")


      // ---------------- LED bars ----------------
      // AFTER tex_out, and that matters as much as it did for mask_in: the
      // editor takes a module's texture output to be the FIRST one its schema
      // declares (schema-channels.ts, firstFieldOfType — it sorts on
      // declaration order), so an aux output declared ahead of tex_out would
      // quietly become THE output and the chain's picture would stop here.
      .group("led", "LED Bars")
      .groupHelp(
        "A pixel map for the house rig: four vertical bars, ten segments each, "
        "as a 4x10 grid of flat blocks. It is a CONTROL SIGNAL, not a picture "
        "— wire *LED Out* at whatever drives the bars and leave it alone.\n\n"
        "Every bar shows the same tower, so the rig reads as three floors "
        "lighting the way the stack does. Ten segments do not divide by three: "
        "the spare one goes to the TOP floor, so from the bottom it is "
        "**3 / 3 / 4**. The top of a meter is the part you read.\n\n"
        "**The bars do not have to show what the screen shows.** They are a "
        "fixture standing in the room, not a picture of the picture — so "
        "*Solid Mix* crossfades them between the planes above and the six *LED "
        "Source* rails, which Three Planes Rig drives with its meter whatever "
        "the screen is doing. Cut the screen to Solid and the rig in the room "
        "keeps reading the feed.\n\n"
        "Unwired, none of this is drawn.")
      .textureField("led_out", state::SecondaryOutput)
        .label("LED Out", "LED")
      .boolField("led_quantize", true, state::SecondaryInput,
                 "Each segment a flat block of colour, which is what you want: "
                 "anything sampling or averaging inside a block then comes away "
                 "with the exact colour that segment asked for. Turn it off to "
                 "interpolate between segment centres — a soft wash, and mud on "
                 "a physical rig.")
        .label("Quantize", "Quant")
      .floatField("led_solid", 1.0f, 0.f, 1.f, state::PrimaryInput,
                  nullptr, 0.f, nullptr,
                  "How much of the PICTURE the bars show, against the six *LED "
                  "Source* rails below. 1 — the default — is the bars mirroring "
                  "the screen, which is what they did before there was a "
                  "choice. 0 hands them over to the sources entirely; wire "
                  "those from Three Planes Rig and the rig in the room goes on "
                  "reading the meter while the screen sits Solid or chops. "
                  "Anywhere between is a mix, per floor, of both the brightness "
                  "and the colour.")
        .label("Solid Mix", "Solid")
      .floatField("led1_emission", 0.85f, 0.f, 1.5f, state::SecondaryInput)
        .label("LED 1 Source", "L1 Em")
      .rgbField("led1_color", 1.00f, 0.22f, 0.62f, state::SecondaryInput)
        .label("LED 1 Colour", "L1 Col")
      .floatField("led2_emission", 0.85f, 0.f, 1.5f, state::SecondaryInput)
        .label("LED 2 Source", "L2 Em")
      .rgbField("led2_color", 0.30f, 0.85f, 1.00f, state::SecondaryInput)
        .label("LED 2 Colour", "L2 Col")
      .floatField("led3_emission", 0.85f, 0.f, 1.5f, state::SecondaryInput)
        .label("LED 3 Source", "L3 Em")
      .rgbField("led3_color", 0.72f, 0.35f, 1.00f, state::SecondaryInput)
        .label("LED 3 Colour", "L3 Col")

      // ---------------- Walls ----------------
      // After tex_out, like every other aux output on this card — the editor
      // takes a module's texture output to be the first one its schema
      // declares (schema-channels.ts, firstFieldOfType).
      .group("walls", "Impact Light")
      .groupHelp(
        "The light the stack throws into the room it is standing in. *Left "
        "Out* and *Right Out* are the two side walls, flat on — what a muzzle "
        "flash does to a corridor, not a second picture of the tower.\n\n"
        "Nothing here draws a quad. Each floor is a ring of light a little way "
        "off the wall, so what lands is a BAR — flat across the ring, falling "
        "away past its ends — with a second, wider lobe under it from the far "
        "side of the same ring. A throw slams both walls; the glints sweep "
        "across them, and because they cross the room rather than the picture "
        "they reach the two walls at different moments.\n\n"
        "Each floor is FOUR TUBES, turned by the orbit, so what lands depends "
        "on how the ring is facing: square on the near edge does the work and "
        "lays a flat bar, turned off it one corner is nearest and the pool "
        "leans that way. That is also what tells the two walls apart. Pull "
        "*Depth* down when it gets too point-like — it draws the far side of "
        "each ring in toward the wall.\n\n"
        "*Distance* is the knob that matters, and it is not a softness — it is "
        "where the wall stands. A pool is half as bright exactly that far out, "
        "so close is tight and burnt and far is broad, and there is no way to "
        "set the two against each other into a shape that could not happen. "
        "That, and the flat crown across each ring, is the difference between "
        "light on a wall and a gradient.\n\n"
        "Unwired, none of this is drawn.")
      .textureField("left_out",  state::SecondaryOutput)
        .label("Left Out", "L")
      .textureField("right_out", state::SecondaryOutput)
        .label("Right Out", "R")
      .floatField("wall_gain", 1.4f, 0.f, 4.f, state::PrimaryInput,
                  nullptr, 0.f, nullptr,
                  "How hard the stack lights the room. Past the top of the "
                  "range the pools blow out warm, which is what an "
                  "overexposed wall does.")
        .label("Wall Gain", "Gain")
      .floatField("wall_gap", 0.45f, 0.f, 1.f, state::PrimaryInput,
                  nullptr, 0.f, nullptr,
                  "How far off the wall stands. This is the whole shape of the "
                  "light: a pool is half as bright exactly this far out, so "
                  "close is tight and burnt and far is broad. Below about half "
                  "the floor spacing the three read as three; above it they "
                  "merge into one.")
        .label("Distance", "Dist")
      .floatField("wall_reach", 0.45f, 0.f, 1.f, state::SecondaryInput,
                  nullptr, 0.f, nullptr,
                  "How far the bounce carries past the pools.")
        .label("Reach", "Reach")
      .floatField("wall_bounce", 0.14f, 0.f, 1.f, state::PrimaryInput,
                  nullptr, 0.f, nullptr,
                  "The room answering: a wide dim wash under the pools. At 0 "
                  "the wall is a black void with bars floating on it, which is "
                  "exactly how bad lighting reads.")
        .label("Bounce", "Bnce")
      .floatField("wall_warmth", 0.55f, 0.f, 1.f, state::PrimaryInput,
                  nullptr, 0.f, nullptr,
                  "How far a hot core blows out toward warm white. The "
                  "surround keeps its hue outright — this only takes the "
                  "centre, which is what real light does and what stops a "
                  "bright pool reading as a flat coloured shape.")
        .label("Warmth", "Warm")
      .floatField("wall_depth", 1.0f, 0.f, 1.f, state::PrimaryInput,
                  nullptr, 0.f, nullptr,
                  "How much of each ring's real depth counts. At 1 the far side "
                  "of a ring genuinely is further off, so one turned away from "
                  "the wall throws a pool that leans hard into its nearest "
                  "corner and can get quite point-like. Turn it down to draw "
                  "the far side in — at 0 the whole ring is the same distance "
                  "away and lays a flat bar, whatever the orbit is doing.")
        .label("Depth", "Depth")

      .capability(state::Capability::Generator)
      // Every envelope still lives outside this effect and the grain is
      // derived from absolute host time — but the GLINTS are particles with
      // lifetimes, and that is a real accumulator. A seek lands on a
      // different set of them in flight and is otherwise the same frame,
      // which is exactly what SeekableApproximate says. (With Drive at 0
      // there are none, and the effect is time-independent in practice.)
      .capability(state::Capability::SeekableApproximate)
      .capability(state::Capability::ModulationSource)
      .capability(state::Capability::ModulationSourceMulti)
  );

  if (gpu::Device::backend() == gpu::Backend::None) return;

  state::registerShaderSPV("three_planes_render", RENDER_SPV, RENDER_SPV_SIZE);
  auto cs = gpu::Device::createShaderModuleByName("three_planes_render");
  if (!cs) return;

  s_pso = gpu::Device::createComputePSO(cs, "main", gpu::Bindings()
      .tex2d(0)
      .storageTex2d(1)
      .uniform(2)
      .tex2d(3));

  state::registerShaderSPV("three_planes_led", LED_SPV, LED_SPV_SIZE);
  if (auto led = gpu::Device::createShaderModuleByName("three_planes_led")) {
    s_led_pso = gpu::Device::createComputePSO(led, "main", gpu::Bindings()
        .storageTex2d(0)
        .uniform(1));
  }

  state::registerShaderSPV("three_planes_wall", WALL_SPV, WALL_SPV_SIZE);
  if (auto wall = gpu::Device::createShaderModuleByName("three_planes_wall")) {
    s_wall_pso = gpu::Device::createComputePSO(wall, "main", gpu::Bindings()
        .storageTex2d(0)
        .uniform(1));
  }

  state::setOnStateReady(&on_state_ready);
  state::log("three_planes: module initialized");
}

void* create() {
  auto* s = new State();
  s->uniform_buf = gpu::Device::createBuffer(sizeof(Uniforms), gpu::BufferUsage::Uniform);
  s->led_buf = gpu::Device::createBuffer(sizeof(LedUniforms), gpu::BufferUsage::Uniform);
  for (int i = 0; i < 2; i++)
    s->wall_buf[i] = gpu::Device::createBuffer(sizeof(WallUniforms),
                                               gpu::BufferUsage::Uniform);
  return s;
}

void destroy(void* self) {
  auto* s = static_cast<State*>(self);
  if (!s) return;
  s->uniform_buf.release();
  s->led_buf.release();
  s->led_tex.release();
  for (int i = 0; i < 2; i++) { s->wall_buf[i].release(); s->wall_tex[i].release(); }
  delete s;
}

void init(void* self) {
  auto* s = static_cast<State*>(self);
  if (!s) return;
  projectPlanes(*s);
  if (!s_pso.valid() || !s->uniform_buf.valid()) return;
  s->initialized = true;
}

// The projection and both rails are viewport-free, so they belong here — no
// GPU readback, and downstream taps see this frame's values before render.
//
// The glints advance here too, and ONLY here: render() may be called without a
// tick (thumbnails, off-playhead previews) and stepping the particles from
// there would age them by a frame every time somebody looked at the card.
void tick(void* self, double dt) {
  auto* s = static_cast<State*>(self);
  if (!s) return;

  projectPlanes(*s);
  publishRails(*s);

  // THE BEAT BEFORE THE HIT. A throw fires on the frame the tower mutes, so
  // that frame is already black — and holding the rings back for exactly one
  // more makes the picture land on nothing before it lands on the release.
  // A hit reads harder for the silence in front of it.
  if (s->release > s->prev_release + 1e-4f) s->ring_delay = 1;
  else if (s->ring_delay > 0) --s->ring_delay;
  s->prev_release = s->release;

  three_planes_glints::Params gp;
  gp.sweep = s->glimmer_sweep;
  gp.band = s->glimmer_band;
  gp.ratio = s->glimmer_ratio;
  gp.chaos = s->glimmer_chaos;
  gp.fling = s->release;
  // A throw holds the glints up in both modes, but only Grow carries them off
  // with it. Strobe keeps everything inside the frame — slinging the glints
  // out of the picture is the one thing in it that would go the other way — so
  // there they are held and left to drift instead.
  gp.sling = s->release_mode == kModeStrobe ? 0.0f : 1.0f;
  // The margins, in CROSSINGS — the units `pos` is in. A glint is born `lead`
  // before the lit picture starts and retired `trail` after it ends, sized by
  // the widest glint the birth spread can draw so every one of them maps from
  // `pos` to the screen the same way (see render()). The projection cancels:
  // `glimmer_width` is already a fraction of one crossing, so a half-width is
  // half of it whatever the stack's size on screen turns out to be.
  const float hw_max = s->glimmer_width * three_planes_glints::kMaxWidthFactor;
  gp.lead = kGlintSkirt * hw_max * 0.5f;
  gp.trail = kGlintWakeReach * hw_max * 0.5f;
  s->glint_core.tick(gp, (float)dt);

  // THE ROLL, if this is a Strobe throw. Its clock does not start until the
  // picture does — the held frame above is silence on purpose, and letting the
  // roll run through it would eat most of the arrival it is there to set up.
  three_planes_strobe::Params sp;
  sp.release = s->release;
  sp.rate    = s->strobe_rate;
  sp.duty    = s->strobe_duty;
  sp.grace   = s->strobe_grace;
  // Local contrast, as a WINDOW and not as a dimmer. See render() for what
  // Grow does with the same number, and the header for why Strobe refuses to
  // turn a hit down when it could drop it instead.
  for (int i = 0; i < PLANES; i++) sp.weight[i] = strobeWeight(*s, i);
  s->strobe.tick(sp, s->ring_delay > 0 ? 0.0f : (float)dt);
}

void on_resolume_param(void* self, long long param_id, double value) {
  (void)self; (void)param_id; (void)value;
}

void on_state_patched(void* self, int n, const char* pb, const int* off,
                      const int* len, const int* ops) {
  auto* s = static_cast<State*>(self);
  if (!s) return;

  bool vis_dirty = false;
  for (int i = 0; i < n; i++) {
    if (ops[i] != state::PatchReplace) continue;
    const char* p = pb + off[i];
    const int   l = len[i];

    if      (state::pathIs(p, l, "plane1_emission")) s->emission[0] = state::patchFloat(i);
    else if (state::pathIs(p, l, "plane2_emission")) s->emission[1] = state::patchFloat(i);
    else if (state::pathIs(p, l, "plane3_emission")) s->emission[2] = state::patchFloat(i);
    else if (state::pathIs(p, l, "plane1_fill"))     s->fill[0] = state::patchFloat(i);
    else if (state::pathIs(p, l, "plane2_fill"))     s->fill[1] = state::patchFloat(i);
    else if (state::pathIs(p, l, "plane3_fill"))     s->fill[2] = state::patchFloat(i);
    else if (state::pathIs(p, l, "plane1_color")) {
      auto v = state::patchVec3(i);
      s->color[0][0] = v.x; s->color[0][1] = v.y; s->color[0][2] = v.z;
    } else if (state::pathIs(p, l, "plane2_color")) {
      auto v = state::patchVec3(i);
      s->color[1][0] = v.x; s->color[1][1] = v.y; s->color[1][2] = v.z;
    } else if (state::pathIs(p, l, "plane3_color")) {
      auto v = state::patchVec3(i);
      s->color[2][0] = v.x; s->color[2][1] = v.y; s->color[2][2] = v.z;
    }
    else if (state::pathIs(p, l, "orbit_azimuth"))   s->orbit_azimuth = state::patchFloat(i);
    else if (state::pathIs(p, l, "elevation"))       s->elevation_deg = state::patchFloat(i);
    else if (state::pathIs(p, l, "zoom"))            s->zoom = state::patchFloat(i);
    else if (state::pathIs(p, l, "plane_spacing"))   s->plane_spacing = state::patchFloat(i);
    else if (state::pathIs(p, l, "plane_size"))      s->plane_size = state::patchFloat(i);
    else if (state::pathIs(p, l, "corner_radius"))   s->corner_radius = state::patchFloat(i);

    else if (state::pathIs(p, l, "line_width"))      s->line_width = state::patchFloat(i);
    else if (state::pathIs(p, l, "line_gain"))       s->line_gain = state::patchFloat(i);
    else if (state::pathIs(p, l, "core_whiten"))     s->core_whiten = state::patchFloat(i);
    else if (state::pathIs(p, l, "halo_radius"))     s->halo_radius = state::patchFloat(i);
    else if (state::pathIs(p, l, "halo_gain"))       s->halo_gain = state::patchFloat(i);
    else if (state::pathIs(p, l, "halo_falloff"))    s->halo_falloff = state::patchFloat(i);
    else if (state::pathIs(p, l, "halo_smooth"))     s->halo_smooth = state::patchFloat(i);
    else if (state::pathIs(p, l, "fill_gain"))       s->fill_gain = state::patchFloat(i);

    else if (state::pathIs(p, l, "release"))         s->release = state::patchFloat(i);
    else if (state::pathIs(p, l, "release_expand"))  s->release_expand = state::patchFloat(i);
    else if (state::pathIs(p, l, "release_gain"))    s->release_gain = state::patchFloat(i);
    else if (state::pathIs(p, l, "release_blur"))    s->release_blur = state::patchFloat(i);
    else if (state::pathIs(p, l, "release_contrast")) s->release_contrast = state::patchFloat(i);
    else if (state::pathIs(p, l, "release_mode")) {
      s->release_mode = state::patchInt(i);
      vis_dirty = true;
    }
    else if (state::pathIs(p, l, "strobe_rate"))    s->strobe_rate = state::patchFloat(i);
    else if (state::pathIs(p, l, "strobe_duty"))    s->strobe_duty = state::patchFloat(i);
    else if (state::pathIs(p, l, "strobe_gain"))    s->strobe_gain = state::patchFloat(i);
    else if (state::pathIs(p, l, "strobe_glow"))    s->strobe_glow = state::patchFloat(i);
    else if (state::pathIs(p, l, "strobe_grace"))   s->strobe_grace = state::patchFloat(i);

    else if (state::pathIs(p, l, "glimmer_sweep"))   s->glimmer_sweep = state::patchFloat(i);
    else if (state::pathIs(p, l, "glimmer_band"))    s->glimmer_band = state::patchFloat(i);
    else if (state::pathIs(p, l, "glimmer_ratio"))   s->glimmer_ratio = state::patchFloat(i);
    else if (state::pathIs(p, l, "glimmer_chaos"))   s->glimmer_chaos = state::patchFloat(i);
    else if (state::pathIs(p, l, "glimmer_angle"))   s->glimmer_angle = state::patchFloat(i);
    else if (state::pathIs(p, l, "glimmer_width"))   s->glimmer_width = state::patchFloat(i);
    else if (state::pathIs(p, l, "glimmer_gain"))    s->glimmer_gain = state::patchFloat(i);
    else if (state::pathIs(p, l, "glimmer_shadow"))  s->glimmer_shadow = state::patchFloat(i);

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
    else if (state::pathIs(p, l, "input_opacity"))   s->input_opacity = state::patchFloat(i);
    else if (state::pathIs(p, l, "mask_strength"))  s->mask_strength = state::patchFloat(i);
    else if (state::pathIs(p, l, "mask_halo"))      s->mask_halo = state::patchFloat(i);

    else if (state::pathIs(p, l, "wall_gain"))     s->wall_gain = state::patchFloat(i);
    else if (state::pathIs(p, l, "wall_gap"))      s->wall_gap = state::patchFloat(i);
    else if (state::pathIs(p, l, "wall_reach"))    s->wall_reach = state::patchFloat(i);
    else if (state::pathIs(p, l, "wall_bounce"))   s->wall_bounce = state::patchFloat(i);
    else if (state::pathIs(p, l, "wall_warmth"))   s->wall_warmth = state::patchFloat(i);
    else if (state::pathIs(p, l, "wall_depth"))    s->wall_depth = state::patchFloat(i);

    else if (state::pathIs(p, l, "led_quantize"))   s->led_quantize = state::patchBool(i);
    else if (state::pathIs(p, l, "led_solid"))      s->led_solid = state::patchFloat(i);
    else if (state::pathIs(p, l, "led1_emission"))  s->led_emission[0] = state::patchFloat(i);
    else if (state::pathIs(p, l, "led2_emission"))  s->led_emission[1] = state::patchFloat(i);
    else if (state::pathIs(p, l, "led3_emission"))  s->led_emission[2] = state::patchFloat(i);
    else if (state::pathIs(p, l, "led1_color")) {
      auto v = state::patchVec3(i);
      s->led_color[0][0] = v.x; s->led_color[0][1] = v.y; s->led_color[0][2] = v.z;
    } else if (state::pathIs(p, l, "led2_color")) {
      auto v = state::patchVec3(i);
      s->led_color[1][0] = v.x; s->led_color[1][1] = v.y; s->led_color[1][2] = v.z;
    } else if (state::pathIs(p, l, "led3_color")) {
      auto v = state::patchVec3(i);
      s->led_color[2][0] = v.x; s->led_color[2][1] = v.y; s->led_color[2][2] = v.z;
    }

    else if (state::pathIs(p, l, "debug_show_sdf"))    s->debug_show_sdf = state::patchBool(i);
    else if (state::pathIs(p, l, "debug_show_planes")) s->debug_show_planes = state::patchBool(i);
  }
  if (vis_dirty) applyModeVisibility(s);
}

// --- The impact light -----------------------------------------------------
//
// Two more aux passes on the same pattern as the LED map: effect-owned, one
// dispatch each, and only when something downstream is wired.
//
// Everything the wall needs is in the STACK's own units, not the picture's —
// floor heights and ring sizes straight off the parameters rather than off the
// projected corners. That is what makes it orbit-independent, which is not a
// shortcut: a square turned about its own axis presents the same silhouette to
// both walls, so the light genuinely does not change.
static const float kGhostWallGain = 0.70f;   ///< a throw, as light on the wall
/// How much of the ghost's own opening-out reaches the wall as softness.
///
/// A thrown ring does two things at once and only one of them was here at
/// first. It flies OUTWARD, which brings it at the wall, and it OPENS OUT,
/// which is the halo spreading as the light dissipates. Modelling the flight
/// alone made the pool converge on a razor line — correct for an ideal line
/// source arriving at a plane, and it read as a WIRE, three more tubes
/// switching on rather than a throw. The opening is what makes it a flash: it
/// lands tight and hot exactly on the bars the tower was showing, then blooms
/// wide and goes out.
static const float kGhostBloom = 0.5f;

/// A tube's own radius, as a fraction of how far off the wall it is.
///
/// Not a detail. A ring turned to 45 presents a CORNER to the wall and nothing
/// else — that is true, and it is the effect asked for — but a mathematical
/// line through that corner throws a pinpoint, and the tower reads as three
/// sparks. Real tube has thickness and real glow around it, and at this
/// fraction the corner lands as a blob the size of the gap instead.
static const float kTubeRadius = 0.60f;



static void renderWalls(State* s, int vp_w, int vp_h, const Uniforms& u,
                        float grow, float opened) {
  if (!s_wall_pso.valid()) return;
  static const char* const kField[2] = {"left_out", "right_out"};

  for (int side = 0; side < 2; side++) {
    if (!state::isOutputConnected(kField[side])) continue;
    if (!s->wall_buf[side].valid()) continue;

    if (!s->wall_tex[side].valid() || s->wall_w[side] != vp_w ||
        s->wall_h[side] != vp_h) {
      s->wall_tex[side].release();
      s->wall_tex[side] = gpu::Device::createTexture(vp_w, vp_h);
      s->wall_w[side] = vp_w;
      s->wall_h[side] = vp_h;
      if (!s->wall_tex[side].valid()) continue;
      state::setGpuTexture(kField[side], s->wall_tex[side].id);
    }

    const float gap = wallGap(s->wall_gap);

    // The ring in the ROOM, turned by the orbit — the same rotation
    // projectPlanes uses, kept in (x, z) instead of being flattened to the
    // screen. Scaled by the zoom, so everything here is in the picture's own
    // cover-square units and the light keeps step with the tower.
    const float th = s->orbit_azimuth * 2.0f * kPi;
    const float ct = std::cos(th), st = std::sin(th);
    const float half = s->plane_size;
    const float cx[4] = {-half, +half, +half, -half};
    const float cz[4] = {-half, -half, +half, +half};

    // Where the wall stands: clear of the ring's silhouette by exactly the gap,
    // so the NEAREST tube is always that far off however the stack has turned.
    //
    // A stylisation, and a deliberate one — a real room's walls do not move
    // because the thing in it turned. Pinning them instead, at the circumradius
    // so nothing could poke through, made the distance breathe between the
    // ring's inradius and its corners: a 1.4x swing under a 1/d^3 falloff, so
    // the level moved by most of two stops as the stack orbited and no single
    // exposure was right at more than one angle. It also put a floor of 0.41
    // ring-widths under the distance, which is already wider than the floors
    // are apart — the three pools could never come out as three.
    //
    // Holding the nearest tube fixed keeps the exposure still and lets the
    // orbit do what was actually asked of it: change the SHAPE. Square on, the
    // whole near edge is at the gap and lays a flat bar; turned off it, one
    // corner is at the gap and the edge falls away behind it, so the pool leans;
    // at 45 there is only the corner. That is the geometry, and it is mirrored
    // between the two walls.
    const float support = half * (std::fabs(ct) + std::fabs(st)) * s->zoom;
    const float wall_x = support + gap;
    const float d_min = gap;

    // How much of the ring's real depth counts. Below 1 the far side of it is
    // drawn in toward the wall — the NEAR side does not move, so the gap and
    // everything normalised against it stay put, and what changes is only how
    // much further away the rest of the ring is than its closest point.
    //
    // At 1 the geometry is honest and a ring turned off square throws a pool
    // that leans hard into the one corner facing the wall. At 0 every part of
    // it is the same distance off, which is the flat bar this pass drew before
    // any of the four-tube work — including, at that end, the far edge landing
    // exactly on the near one and the two between them shortening to nothing.
    // The doubling that comes of it is not a glitch: it is what collapsing a
    // ring onto a plane means.
    const float depth = clamp01f(s->wall_depth);
    const float side_sign = (side == 0) ? -1.0f : 1.0f;

    // THE ELEVATION TILTS EACH RING, ABOUT ITS OWN CENTRE.
    //
    // An orthographic camera raised over a flat stack and a camera on the
    // horizon looking at a stack tilted by the same angle draw the same main
    // output, so the picture cannot tell you which room it is a picture of —
    // but a side wall can, and only the tilted one has anything for it to see.
    // So the rings tilt, and their near edges run uphill at tan(phi), and the
    // pools slant.
    //
    // Tilting the STACK as one body is the honest version of that and it is not
    // what happens here. Doing it leans the whole column back: floor by floor
    // the centres walk off along the wall by y * sin(phi), so the pools stagger
    // diagonally and the corner hotspots — which are the most legible thing on
    // the wall — string out into a slanted line instead of stacking. Past about
    // 60 degrees the column has laid over far enough to stop reading as a tower
    // at all.
    //
    // So the centres are pinned on the vertical: each ring tilts in place. The
    // room this describes cannot exist, and every other thing about it stays
    // true — a ring's centre is still at y * cos(phi), exactly where the picture
    // draws that floor, and its near edge still climbs at tan(phi). What it
    // buys is the isometric read: three slanted pools in a vertical column,
    // with their hotspots in two clean vertical files.
    const float ep = s->elevation_deg * (kPi / 180.0f);
    const float e_cos = std::cos(ep), e_sin = std::sin(ep);

    WallUniforms w = {};
    for (int i = 0; i < PLANES; i++) {
      const float y = (float(i) - 1.0f) * s->plane_spacing;   // 0 = bottom floor
      // Ring i is the floor; ring i + PLANES is the ghost it threw. Same rows,
      // same code — the ghost is flown and opened, and nothing downstream needs
      // to know which is which.
      const int ring_of[2] = {i, i + PLANES};
      const float scale_of[2] = {1.0f, grow};
      const float soft_of[2] = {
          d_min * kTubeRadius,
          d_min * kTubeRadius * (1.0f + (opened - 1.0f) * kGhostBloom)};
      // The throw's gain multiplies `opened` back, because the picture divides
      // it out to conserve a widening halo — and nothing is widening on a wall.
      const float level_of[2] = {litLevel(s->emission[i]),
                                 u.ring_gain[i] * opened * kGhostWallGain};

      for (int g = 0; g < 2; g++) {
        const int r = ring_of[g];
        const float sc = scale_of[g];
        for (int c = 0; c < 3; c++) w.ring[r][c] = s->color[i][c];
        w.ring[r][3] = level_of[g];
        // The ring's own wall-facing extreme, which is what the damping pulls
        // the rest of it toward. In x, which the tilt leaves alone — so the gap
        // and everything normalised against it are the same at any elevation.
        const float near_x = side_sign * support * sc;
        const float y0 = y * s->zoom * sc;
        for (int k = 0; k < 4; k++) {
          float xr = (cx[k] * ct - cz[k] * st) * s->zoom * sc;
          const float zr = (cx[k] * st + cz[k] * ct) * s->zoom * sc;
          xr = near_x + (xr - near_x) * depth;
          w.ring_c[r * 4 + k][0] = xr;
          // Tilted about x, the same rotation the picture is drawn through — so
          // a corner's height here IS its height on screen. A ring flat on the
          // horizon has every corner at one height and lays a hard line; tilted,
          // its near edge climbs at tan(phi) and the pool slants with it.
          //
          // Only the ring's OWN offset is tilted, not its height: the `zr * sp`
          // that lifts a corner is kept and the `y0 * sp` that would carry the
          // whole ring backward is dropped. See the note above — that dropped
          // term is the stack leaning over, and the wall reads better without
          // it.
          w.ring_c[r * 4 + k][1] = y0 * e_cos + zr * e_sin;
          w.ring_c[r * 4 + k][2] = zr * e_cos;
        }
        // The ring's centre — where the bounce comes from. On the vertical
        // axis at every elevation, which is the whole of the cheat.
        w.ring_g[r][0] = y0 * e_cos;
        w.ring_g[r][1] = 0.0f;
        w.ring_g[r][2] = soft_of[g];
      }
    }

    w.wall[0] = wallReach(s->wall_reach);
    w.wall[1] = s->wall_bounce;
    // Damping the depth brings MORE of the ring up against the wall — at 0 the
    // far edge lands on the near one and both are at the gap — so the light
    // roughly doubles on the way down. That is honest, and it would also blow
    // the pools out at one end of the knob's travel with the Gain having to be
    // ridden against it. Divided out, so what the knob changes is the SHAPE.
    //
    // How much it doubles by, rather than a straight line between 1 and 2: the
    // far edge sits at gap + 2 * support * depth and falls off as the square of
    // it, so it stays negligible until the depth is nearly gone and then rushes
    // in. A linear compensation reads that as gradual and dims the whole middle
    // of the knob, where nothing has happened yet.
    const float far_d = gap + 2.0f * support * depth;
    w.wall[2] = s->wall_gain / (1.0f + (gap * gap) / (far_d * far_d));
    w.wall[3] = d_min;

    w.look[0] = s->wall_warmth;
    w.look[1] = (side == 0) ? -1.0f : 1.0f;
    w.look[2] = 0.0f;
    w.look[3] = wall_x;

    w.glim[0] = u.glim0[0];
    w.glim[1] = u.glim0[1];
    for (int g = 0; g < 8; g++)
      for (int c = 0; c < 4; c++) w.glints[g][c] = u.glints[g][c];

    s->wall_buf[side].writeOne(w);

    auto cp = gpu::ComputePass::begin();
    cp.setPSO(s_wall_pso);
    cp.setTexture(s->wall_tex[side], 0, 1);
    cp.setBuffer(s->wall_buf[side], 1);
    cp.dispatch((vp_w + 7) / 8, (vp_h + 7) / 8);
    cp.end();
  }
}

// --- The LED map ----------------------------------------------------------
//
// A second, tiny pass, on chroma_wave's aux-output pattern: the effect owns the
// allocation, publishes the handle once per allocation, and does not dispatch
// at all unless something downstream is wired to it. The executor allocates and
// sizes ONLY tex_out.
//
// Drawn at the full viewport rather than as a 4x10 thumbnail on purpose. The
// map is forty flat blocks either way, but at viewport size each block is
// hundreds of pixels across, so a consumer that resamples — a pixel mapper
// picking a point, a scaler fitting it to a surface — stays well inside one
// block and comes away with the exact colour. A 4x10 texture stretched up
// would be bilinearly smeared into precisely the averaging we are avoiding.
static void renderLed(State* s, int vp_w, int vp_h) {
  if (!s_led_pso.valid() || !s->led_buf.valid()) return;
  if (!state::isOutputConnected("led_out")) return;

  if (!s->led_tex.valid() || s->led_w != vp_w || s->led_h != vp_h) {
    s->led_tex.release();
    s->led_tex = gpu::Device::createTexture(vp_w, vp_h);
    s->led_w = vp_w;
    s->led_h = vp_h;
    if (!s->led_tex.valid()) return;
    // Published on ALLOCATION only — the executor never clears an output
    // handle, so it persists across frames.
    state::setGpuTexture("led_out", s->led_tex.id);
  }

  // Two towers, crossfaded per floor: the PICTURE's planes and the six LED
  // Source rails. At Solid Mix 1 this is exactly the old single-source path.
  //
  // The mix happens AFTER the dimmer curve, in the units the eye is in, so
  // half way is half as bright rather than half as far along a gamma. Colour
  // crossfades alongside it — the bars have to be able to hold the meter's
  // moving cap while the screen holds a fixed colour, and a level without its
  // colour would put the cap on the wrong floor's tint.
  const float mix = clamp01f(s->led_solid);
  float level[PLANES];
  float rgb[PLANES][3];
  for (int i = 0; i < PLANES; i++) {
    const float a = litLevel(s->led_emission[i]);
    const float b = litLevel(s->emission[i]);
    level[i] = a + (b - a) * mix;
    for (int c = 0; c < 3; c++)
      rgb[i][c] = s->led_color[i][c] + (s->color[i][c] - s->led_color[i][c]) * mix;
  }
  // The fills are deliberately absent. A plane set to -1 cuts a hole in the
  // PICTURE, where there is something behind it to cut; a bar has nothing
  // behind it, and a floor that is lit is lit.
  const led_bars::Cells cells = led_bars::planeCells(rgb, level);

  LedUniforms u = {};
  for (int i = 0; i < led_bars::kCells; i++) {
    u.cell[i][0] = cells.rgb[i][0];
    u.cell[i][1] = cells.rgb[i][1];
    u.cell[i][2] = cells.rgb[i][2];
    u.cell[i][3] = 1.0f;
  }
  u.ctl[0] = s->led_quantize ? 1.0f : 0.0f;
  s->led_buf.writeOne(u);

  auto cp = gpu::ComputePass::begin();
  cp.setPSO(s_led_pso);
  cp.setTexture(s->led_tex, 0, 1);
  cp.setBuffer(s->led_buf, 1);
  cp.dispatch((vp_w + 7) / 8, (vp_h + 7) / 8);
  cp.end();
}

void render(void* self, int vp_w, int vp_h) {
  auto* s = static_cast<State*>(self);
  if (!s || !s->initialized || vp_w <= 0 || vp_h <= 0) return;

  auto in  = gpu::Device::textureForField("tex_in");
  auto out = gpu::Device::textureForField("tex_out");
  if (!in.valid() || !out.valid()) return;
  // Nothing wired: bind the input in its place and hand the shader a strength
  // of 0. The sample still happens and reads nothing into the picture, which
  // is cheaper than a branch and leaves the binding table the same shape every
  // frame.
  auto mask = gpu::Device::textureForField("mask_in");
  const bool has_mask = mask.valid();
  if (!has_mask) mask = in;

  projectPlanes(*s);
  publishRails(*s);

  Uniforms u = {};
  for (int i = 0; i < PLANES; i++) {
    u.corners[i * 2 + 0][0] = s->corner_x[i][0];
    u.corners[i * 2 + 0][1] = s->corner_y[i][0];
    u.corners[i * 2 + 0][2] = s->corner_x[i][1];
    u.corners[i * 2 + 0][3] = s->corner_y[i][1];
    u.corners[i * 2 + 1][0] = s->corner_x[i][2];
    u.corners[i * 2 + 1][1] = s->corner_y[i][2];
    u.corners[i * 2 + 1][2] = s->corner_x[i][3];
    u.corners[i * 2 + 1][3] = s->corner_y[i][3];

    u.plane_color[i][0] = s->color[i][0];
    u.plane_color[i][1] = s->color[i][1];
    u.plane_color[i][2] = s->color[i][2];
    u.plane_color[i][3] = emissionDrive(s->emission[i]);
    u.fills[i] = s->fill[i];
  }

  // One pixel measured in cover-square units is 2 / max(W, H); widen it a
  // touch so the edge lands soft rather than stair-stepped.
  u.fills[3] = s->halo_smooth;

  const float px = 2.0f / float(vp_w > vp_h ? vp_w : vp_h);
  const auto  cs = fx::coverSquare(vp_w, vp_h);

  // THE RELEASE RINGS. The same three quads, thrown outward from the stack's
  // centre — which is the origin, because the middle plane's centre is where
  // the camera orbits — so the floors spread apart as they fly rather than
  // just growing. Everything is a pure function of `release`: at 1 a ring sits
  // exactly on the quad that threw it, and by 0 it has flown its full distance
  // and gone out. That is what keeps them stateless despite reading as
  // objects with a life.
  const float rel = s->release < 0.0f ? 0.0f : (s->release > 1.0f ? 1.0f : s->release);
  const float flown = 1.0f - rel;                       // 0 at the throw, 1 spent
  const bool strobe = s->release_mode == kModeStrobe;
  // STROBE throws the stack nowhere. The ghosts sit exactly on the quads that
  // made them and stay there, which is what makes them read as the bars coming
  // BACK — bare wireframe over a muted picture — rather than as light leaving.
  // Everything after this point is the same code for both modes with different
  // numbers in it, which is the whole reason there is only one ghost path.
  const float grow = strobe ? 1.0f : 1.0f + s->release_expand * flown;
  for (int i = 0; i < PLANES; i++) {
    u.ghosts[i * 2 + 0][0] = s->corner_x[i][0] * grow;
    u.ghosts[i * 2 + 0][1] = s->corner_y[i][0] * grow;
    u.ghosts[i * 2 + 0][2] = s->corner_x[i][1] * grow;
    u.ghosts[i * 2 + 0][3] = s->corner_y[i][1] * grow;
    u.ghosts[i * 2 + 1][0] = s->corner_x[i][2] * grow;
    u.ghosts[i * 2 + 1][1] = s->corner_y[i][2] * grow;
    u.ghosts[i * 2 + 1][2] = s->corner_x[i][3] * grow;
    u.ghosts[i * 2 + 1][3] = s->corner_y[i][3] * grow;
  }
  // Bright and tight at the throw, wide and dull by the end: the core goes
  // first and only the glow is left, which is a low-pass closing drawn in
  // space rather than heard.
  //
  // The gain is divided by how far it has opened, and that is not a taste
  // decision — the halo profile peaks at 1 whatever its radius, so widening it
  // alone spreads the SAME peak over more picture and the ring gets BRIGHTER
  // as it dissipates. Dividing conserves roughly the light it was thrown with,
  // which is the difference between a ring going out and a ring blooming.
  //
  // Strobe opts out of all of that: it is a fixed look, not a flight. Its halo
  // is a fraction of the live one — barely there, so what is left is the tube
  // and a breath around it — and its gain is exactly what it says, since
  // nothing is dissipating.
  const float opened = strobe ? 1.0f : 1.0f + (s->release_blur - 1.0f) * flown;
  const float ring_scale = strobe ? strobeGlow(s->strobe_glow) : opened;
  u.rel[0] = haloRadius(s->halo_radius) * ring_scale;
  u.rel[1] = s->halo_falloff;

  // FAKED LOCAL CONTRAST. Sweep straight back after a throw and the tower
  // relights UNDERNEATH a ring that has barely left it — two bright things in
  // the same place, which reads as one bright thing and loses the ring. So a
  // lit plane holds its own ring down, and only while the ring is still on top
  // of it: the damping is the plane's brightness times how close the ring
  // still is, so by the time it has flown clear it is back to full strength
  // whatever the tower is doing. Per ring, because the floors light
  // separately — the cap can be blazing while the ground floor is dark.
  const float gate = s->ring_delay > 0 ? 0.0f : 1.0f;
  for (int i = 0; i < PLANES; i++) {
    if (strobe) {
      // Strobe never touches a brightness. Its decay is the window closing and
      // its local contrast is the window closing — both already spent on the
      // roll's timing back in tick(), where `strobeWeight` went in — so a hit
      // that happens at all happens at full strength. See the header: a
      // dimming strobe reads as a light going out, a thinning one reads as a
      // thing running down, and only the second is the move.
      u.ring_gain[i] = s->strobe_gain * s->strobe.gain[i] * gate;
      continue;
    }
    // Grow does fade, because brightness IS its decay: the ring opens out and
    // goes dull as it flies, and the damping rides on how close it still is.
    const float lit = s->emission[i] < 0.0f ? 0.0f : (s->emission[i] > 1.0f ? 1.0f : s->emission[i]);
    float damp = 1.0f - s->release_contrast * lit * rel;
    if (damp < 0.0f) damp = 0.0f;
    u.ring_gain[i] = s->release_gain * rel / opened * damp * gate;
  }

  u.neon0[0] = lineHalfWidth(s->line_width);
  u.neon0[1] = s->line_gain;
  u.neon0[2] = s->core_whiten;
  u.neon0[3] = haloRadius(s->halo_radius);
  u.neon1[0] = s->halo_gain;
  u.neon1[1] = s->halo_falloff;
  u.neon1[2] = s->corner_radius;
  u.neon1[3] = px * 1.2f;

  u.misc[0] = s->fill_gain;
  u.misc[1] = s->chroma_bleed;
  u.misc[2] = s->input_opacity;
  u.misc[3] = s->debug_show_sdf ? 1.0f : (s->debug_show_planes ? 2.0f : 0.0f);

  u.mask[0] = has_mask ? clamp01f(s->mask_strength) : 0.0f;
  u.mask[1] = clamp01f(s->mask_halo);
  u.mask[2] = 0.0f;
  u.mask[3] = 0.0f;

  u.view[0] = float(vp_w);
  u.view[1] = float(vp_h);
  u.view[2] = cs.ax;
  u.view[3] = cs.ay;

  // Travel direction, measured CCW from +x the way an angle normally is —
  // hence the negated sine, because cover-square y grows DOWNWARD. Each glint
  // sits square across this.
  const float ga = s->glimmer_angle * (kPi / 180.0f);
  const float dx = std::cos(ga), dy = -std::sin(ga);
  u.glim0[0] = dx;
  u.glim0[1] = dy;

  // How far the frame reaches along that axis: the corner that projects
  // furthest onto it. Deriving anything here from the VIEWPORT rather than
  // from a fixed number is what makes Speed and Width mean the same thing
  // whatever shape the output is and whichever way the glints are running.
  const float span = std::fabs(dx) * (0.5f / cs.ax) + std::fabs(dy) * (0.5f / cs.ay);

  // ...but the frame is NOT what a glint travels across. It multiplies
  // emission, so out where there is no geometry it is multiplying nothing, and
  // every unit of travel spent there is dead time between throwing a glint and
  // seeing it. On a default stack that dead run is most of the way in from
  // each corner — and at the idle speed it is seconds of it.
  //
  // So the trip is bounded by the LIT extent instead: the furthest any plane's
  // corner projects onto the travel axis, plus the halo and line it carries
  // out past that. A glint is then doing something almost from the moment it
  // is born.
  float reach = 0.0f;
  for (int i = 0; i < PLANES; i++) {
    for (int k = 0; k < 4; k++) {
      const float a = std::fabs(s->corner_x[i][k] * dx + s->corner_y[i][k] * dy);
      if (a > reach) reach = a;
    }
  }
  reach += haloRadius(s->halo_radius) + lineHalfWidth(s->line_width);
  // Never further than the picture: an enormous zoom would otherwise send
  // glints off on a tour of geometry nobody can see. And never zero, so a
  // collapsed camera cannot divide the travel down to nothing.
  if (reach > span) reach = span;
  if (reach < 0.05f) reach = 0.05f;

  // A crossing is the whole lit extent, so `pos` 0..1 spans -reach..+reach and
  // a glint's half-width is half of `glimmer_width` of that. Which is what
  // makes the header's invariant exact: `pos` advances at the knob's own speed
  // in crossings per second, and one crossing is one traverse of the picture.
  const float hw = s->glimmer_width * reach;

  for (int i = 0; i < three_planes_glints::kMaxLive; i++) {
    const auto& g = s->glint_core.glints[i];
    // A dead slot is a zero-gain, zero-shade glint of unit width: it costs the
    // shader two exponentials and contributes exactly nothing, which is
    // cheaper than a branch and keeps the loop fully unrolled.
    u.glints[i][0] = g.live ? -reach + g.pos * 2.0f * reach : 0.0f;
    // drawWidth/drawGain/drawShade, not the raw birth values: a glint that has
    // outlived its gesture is puttering out, and that is where it shows.
    u.glints[i][1] = g.live ? hw * g.drawWidth() : 1.0f;
    u.glints[i][2] = g.live ? s->glimmer_gain * g.drawGain() : 0.0f;
    u.glints[i][3] = g.live ? s->glimmer_shadow * g.drawShade() : 0.0f;
  }

  u.grade[0]  = s->exposure;
  u.grade[1]  = s->warmth;
  u.grade[2]  = s->drive;
  u.grade[3]  = s->asymmetry;
  u.grade[4]  = s->toe;
  u.grade[5]  = s->shoulder;
  u.grade[6]  = s->highlight_desat;
  u.grade[7]  = s->scanline;
  u.grade[8]  = float(s->scanline_count);
  u.grade[9]  = s->grain;
  // Derived from ABSOLUTE host time, not an accumulator, so the effect stays
  // TimeIndependent: a scrub lands on the right frame with the right grain.
  u.grade[10] = float(std::fmod(host::time() * 997.0, 4096.0));
  u.grade[11] = s->highlight_tint_pivot;
  u.grade[12] = s->highlight_tint[0];
  u.grade[13] = s->highlight_tint[1];
  u.grade[14] = s->highlight_tint[2];
  u.grade[15] = s->highlight_tint_amount;

  s->uniform_buf.writeOne(u);

  auto cp = gpu::ComputePass::begin();
  cp.setPSO(s_pso);
  cp.setTexture(in,  0, 0);
  cp.setTexture(out, 1, 1);
  cp.setBuffer(s->uniform_buf, 2);
  cp.setTexture(mask, 3, 0);
  cp.dispatch((vp_w + 7) / 8, (vp_h + 7) / 8);
  cp.end();

  renderLed(s, vp_w, vp_h);
  renderWalls(s, vp_w, vp_h, u, grow, opened);

  gpu::Device::submit();
}

} // namespace three_planes
