/*
 * source.mesh.three_walls — three neon quads rushing at you down a tunnel.
 *
 * The companion to source.mesh.three_planes, pointed the other way. three_planes
 * is VERTICAL — quads stacked with gravity, bottom heaviest, read as a level.
 * This one is FRONTAL: the same neon motif coming out of the screen, read as an
 * arrival. It borrows the whole quad field (shaders_common/nano_neon_quad.hlsl,
 * lifted out of three_planes for exactly this) and the VCR grade, and adds
 * perspective and a set of moves.
 *
 * THREE OUTPUTS, THREE WALLS. The room is a hollow box; the camera sits at its
 * mouth looking at the back wall. `tex_out` IS that back wall, and `left_out`
 * and `right_out` are the two side walls running away from it. The outputs
 * PARTITION the room rather than each re-rendering it: a frame's left edge is
 * either still on the back wall or already on the left wall, never in two
 * pictures at once.
 *
 * So a pulse starts as a small rectangle in the middle of the main output,
 * grows until it fills it, and then OVERFLOWS — leaving the back wall and
 * reappearing as bars sweeping down the two side walls toward you. The halo
 * carries around the corner on its own, because a frame just past the wall's
 * edge still bleeds light back into the picture.
 *
 * The whole geometry is one number: `a`, a frame's apparent half-size in wall
 * half-widths. Below 1 it is on the back wall, above 1 its edges are on the
 * side walls at depth 1/a. That reciprocal IS the perspective — see the room
 * section below.
 *
 * The aux outputs follow chroma_wave's `wave_out` pattern: the effect owns the
 * allocation, publishes the handle once, and skips the dispatch entirely unless
 * something downstream is wired to it. The executor allocates and sizes ONLY
 * `tex_out`.
 *
 * UNLIKE three_planes, this effect owns its rhythm. There is no companion rig
 * and no pattern input — four triggerable moves live in
 * <sketch/three_walls_show.h>, host-free, so the Catch2 goldens in
 * native/tests/test_three_walls_show.cpp drive them at an exact dt with no wasm
 * and no GPU. That also means it is NOT TimeIndependent: every move is an
 * accumulator and cannot be seeked.
 */

#include <gpu.h>
#include <host.h>
#include <val.h>
#include <effect_utils.h>   // fx::coverSquare
#include <sketch/three_walls_show.h>
#include <sketch/led_bars.h>
#include "three_walls_shaders.h"

#include <cmath>
#include <cstdio>
#include <cstring>

namespace three_walls {

namespace show = three_walls_show;

static constexpr int QUADS = show::kQuads;
static constexpr int VIEWS = 3;      // main, left, right
static constexpr float kPi = 3.14159265358979323846f;

// Mirrors `cbuffer Uniforms` in render.hlsl, row for row.
struct Uniforms {
  float corners[6][4];      // rows 0-5   projected, this view
  float quad_color[3][4];   // rows 6-8   rgb = colour, w = emission drive
  float fills[4];           // row  9
  float depth[4];           // row 10     xyz = per-quad neon scale
  float misc[4];            // row 11     bleed, input opacity, debug, -
  float view[4];            // row 12     vp_w, vp_h, aspect_x, aspect_y
  float style[12];          // rows 13-15 NeonStyle, verbatim
  float grade[16];          // rows 16-19 VcrGrade, verbatim
};
static_assert(sizeof(Uniforms) == 320, "Uniforms layout mismatch with render.hlsl");

// Mirrors `cbuffer LedUniforms` in shaders_common/nano_led_bars.hlsl.
struct LedUniforms {
  float cell[led_bars::kCells][4];   // bar-major, segment 0 at the bottom
  float ctl[4];                      // quantize, -, -, -
};
static_assert(sizeof(LedUniforms) == 16 * (led_bars::kCells + 1),
              "LedUniforms layout mismatch with nano_led_bars.hlsl");
static_assert(QUADS == led_bars::kFrames, "the LED map sees every frame");

struct State {
  show::Params show_p;
  show::Core core;

  /// Rising/falling-edge memory for the four move triggers. The executor
  /// replays every stored value as a PatchReplace EVERY frame, so an event that
  /// fired on patch arrival would re-arm forever (style guide 8.2). The gates
  /// need the falling edge too, which is why this is a level and not a pulse.
  bool move_prev[show::kMoveCount] = {};
  /// Patches only ARM; tick() acts. A trigger and its knobs arriving in the
  /// same transaction then resolve the same way whatever order they land in
  /// (mod_latch's discipline).
  int pending_press = show::MoveNone;
  int pending_release = show::MoveNone;

  // --- Scene ---
  /// A frame's apparent half-size on the back wall is quad_size / z, so at 1.0
  /// a frame crosses the wall's edge exactly at z = 1 — a third of the way
  /// through the default travel.
  float quad_size = 1.0f;
  /// 0 = each side output is that wall's own flat surface; 1 = the wall as the
  /// camera sees it. See wallU() for which you want when.
  float wall_keystone = 0.0f;
  /// Scales the run along a side wall, anchored at the seam. Above 1 the
  /// frames tear past — the wall reads as raked away rather than square.
  float wall_stretch = 1.0f;
  /// How hard the neon itself scales with depth. See neonScale().
  float depth_scale = 1.0f;

  // --- Quads ---
  float color[QUADS][3] = {{1.00f, 0.22f, 0.62f},   // magenta  (quad 1)
                           {0.72f, 0.35f, 1.00f},   // violet   (quad 2)
                           {0.30f, 0.85f, 1.00f}};  // cyan     (quad 3, highlight)
  float emission[QUADS] = {0.85f, 0.85f, 0.85f};
  float fill[QUADS] = {0.0f, 0.0f, 0.0f};

  // --- Neon ---
  float line_width = 0.18f;
  float line_gain = 1.60f;
  float core_whiten = 0.85f;
  float halo_radius = 0.30f;
  float halo_gain = 0.55f;
  float halo_falloff = 0.45f;
  float halo_smooth = 0.35f;
  float corner_radius = 0.012f;
  float fill_gain = 0.22f;

  // --- Grade ---
  float exposure = 1.0f;
  float warmth = 0.35f;
  float drive = 0.35f;
  float asymmetry = 0.20f;
  float toe = 0.25f;
  float shoulder = 0.50f;
  float highlight_desat = 0.70f;
  float highlight_tint[3] = {1.00f, 0.22f, 0.62f};
  float highlight_tint_amount = 0.0f;
  float highlight_tint_pivot = 1.0f;
  float chroma_bleed = 0.25f;
  float scanline = 0.12f;
  int scanline_count = 240;
  float grain = 0.08f;
  float input_opacity = 1.0f;
  bool debug_show_quads = false;

  // --- Derived, recomputed each frame ---
  show::Out frame;

  // --- GPU ---
  bool initialized = false;
  gpu::Buffer uniform_buf[VIEWS];
  /// The two auxiliary views are the effect's own textures — the executor
  /// allocates only tex_out. Sized lazily, and only while something reads them.
  // --- LED bars ---
  // The pixel map for the house rig, published on `led_out`. See the LED note
  // below the room section for where the bars stand.
  bool  led_quantize = true;
  float led_spacing = 1.5f;
  float led_width = 0.45f;
  float led_decay = 0.12f;
  /// The peak-held colour of each bar, advanced in tick() because that is where
  /// dt lives — render() is handed a viewport and nothing else.
  led_bars::Bars led_now;

  gpu::Texture led_tex;
  gpu::Buffer  led_buf;
  int led_w = 0;
  int led_h = 0;

  gpu::Texture aux_tex[2];
  int aux_w[2] = {0, 0};
  int aux_h[2] = {0, 0};
};

static gpu::ComputePSO s_pso;
static gpu::ComputePSO s_led_pso;

// --- Perceptual mappings (style guide 1.3) --------------------------------
// Shared with three_planes, deliberately: the two cards' neon knobs have to
// mean the same thing or a preset cannot move between them.

static inline float emissionDrive(float e) {
  return std::pow(e < 0.0f ? 0.0f : e, 1.8f) * 3.2f;
}
// What one frame is worth on the LED rig: the SAME dimmer curve, normalised so
// a frame at full emission is a fully-lit segment. Divided by the curve rather
// than restated as a number, so the map cannot drift from the picture.
static inline float ledLevel(float e) {
  const float v = emissionDrive(e) / emissionDrive(1.0f);
  return v < 0.0f ? 0.0f : (v > 1.0f ? 1.0f : v);
}
static inline float lineHalfWidth(float w) {
  float t = w < 0.0f ? 0.0f : (w > 1.0f ? 1.0f : w);
  return 0.0012f + 0.0208f * t * t;
}
static inline float haloRadius(float r) {
  float t = r < 0.0f ? 0.0f : (r > 1.0f ? 1.0f : r);
  return 0.006f * std::pow(40.0f, t);
}

// --- The room -------------------------------------------------------------
//
// A hollow box. The camera sits at its mouth looking at the BACK WALL; the two
// side walls run away from it to left and right. Each of the three outputs is
// ONE WALL, so a given piece of the picture belongs to exactly one of them —
// a frame's left edge is either still on the back wall or already on the left
// wall, never both. The outputs partition the room; they do not each re-render
// it.
//
// The whole geometry collapses to one number. Let `a` be a frame's apparent
// half-size on the back wall, measured in wall half-widths, so a = 1 is exactly
// the wall's edge:
//
//   a < 1   the frame is a rectangle on the BACK WALL, growing as it comes
//   a = 1   it fills the wall exactly, and its edges reach all four corners
//   a > 1   it has left the back wall; its left and right edges are now bars
//           on the side walls, at wall depth d = 1/a, running from the far
//           end (d = 1) toward the camera (d -> 0)
//
// `d = 1/a` is the entire perspective story: as the frame keeps growing at a
// steady apparent rate, the bar's slide along the wall stretches out toward the
// camera exactly the way a real corridor does. Nothing here is a fudge factor.
//
// NOTHING IS CULLED. A frame past the back wall's edge draws a rectangle whose
// outline is simply outside the picture, and a bar short of the wall's far end
// draws off its edge — in both cases the halo still bleeds in from beyond the
// frame, which is what carries the light around the corner and makes the
// overflow read as continuous rather than as a hand-off.

/// The back wall IS the picture, so its half-extents are the frame's own bounds
/// in cover-square coords (see nano_coords.hlsl: +/-1 on the long axis, less on
/// the short one).
struct WallExtent { float ex, ey; };

static WallExtent frameExtent(const fx::CoverSquare& cs) {
  WallExtent w;
  w.ex = 0.5f / cs.ax;
  w.ey = 0.5f / cs.ay;
  return w;
}

/// A frame's apparent half-size on the back wall. `quad_size` just moves where
/// the overflow happens: at 1.0 a frame crosses the wall's edge at z = 1.
static inline float detailClamp(float v, float lo, float hi) {
  return v < lo ? lo : (v > hi ? hi : v);
}

static inline float apparentSize(const State& s, float z) {
  const float zz = z > 1e-4f ? z : 1e-4f;
  return s.quad_size / zz;
}

/// Where a side wall's depth `d` lands across its texture, as 0 at the far end
/// (against the back wall) and 1 at the near end (against the camera).
///
/// The two ends of the `keystone` knob are two different pictures of the same
/// wall. At 0 the texture is the wall's own flat surface, linear in depth —
/// what you want when the output drives a physical screen standing where that
/// wall is, because the real geometry then supplies the perspective. At 1 it is
/// the wall as the CAMERA sees it, linear in apparent position instead, which
/// is what you want when the three outputs go to three flat panels side by side
/// and the perspective has to be baked in.
///
/// `stretch` then rakes the wall: it MAGNIFIES the run, anchored at the seam.
/// Above 1 the same depth covers more picture, so frames tear along the wall
/// and then run clean off the near edge — which is the point. A frame that
/// passes you should be gone, not parked at the edge of the frame.
///
/// Pinning both ends instead (warping u in place so the near end stays put) is
/// the obvious-looking alternative and it is wrong: nothing can then leave, so
/// frames pile up against the outer edge and sit there. The wall showing less
/// of the tunnel at high stretch is not content being lost, it is the wall
/// being raked — you are seeing a shorter run, larger.
static float wallU(const State& s, float d) {
  const float near_ = s.show_p.z_near > 1e-4f ? s.show_p.z_near : 1e-4f;
  const float dd = d < near_ ? near_ : d;
  // Flat: linear in depth, far end (d = 1) at 0.
  const float flat = (1.0f - dd) / (1.0f - near_ + 1e-6f);
  // Camera: linear in apparent position, which is 1/d.
  const float s_far = 1.0f, s_near = 1.0f / near_;
  const float persp = (1.0f / dd - s_far) / (s_near - s_far + 1e-6f);
  const float k = s.wall_keystone < 0.0f ? 0.0f : (s.wall_keystone > 1.0f ? 1.0f : s.wall_keystone);
  const float u = flat + (persp - flat) * k;

  // Multiplied, not warped — including where u is negative (the frame is still
  // on the back wall, bleeding its halo around the corner). Scaling that region
  // too keeps the corner's approach consistent with the wall it feeds.
  return u * detailClamp(s.wall_stretch, 0.2f, 5.0f);
}

/// How tall the wall is at depth `d`, as a multiple of the texture's height.
///
/// THE SEAM MUST MATCH. A frame straddling the threshold has its edge on the
/// back wall and on the side wall in the same instant, and the two have to be
/// the same height or the picture visibly tears at the corner. At the far end
/// (d = 1) the side wall IS the back wall's edge, so this returns exactly 1
/// whatever the keystone is — the two ends of the knob are different pictures
/// of the wall, not different rooms.
///
/// From there, keystoned, it grows as 1/d: the wall is nearer the camera as it
/// comes forward, so it looks BIGGER, and a frame sweeping down it swells past
/// the top and bottom of the picture the way the mouth of a tunnel does. Flat,
/// it stays 1 the whole way, because a real wall is a rectangle.
static float wallHeight(const State& s, float d) {
  const float near_ = s.show_p.z_near > 1e-4f ? s.show_p.z_near : 1e-4f;
  const float dd = d < near_ ? near_ : d;
  const float k = s.wall_keystone < 0.0f ? 0.0f : (s.wall_keystone > 1.0f ? 1.0f : s.wall_keystone);
  return 1.0f + (1.0f / dd - 1.0f) * k;
}

/// How big a frame's own NEON should be, relative to its authored size.
///
/// A real tube has a fixed thickness, so what reaches the eye scales with
/// everything else: near, a fat bar with a wide bloom; far, a hairline with
/// almost none. Holding the width constant is exactly what makes a receding
/// frame read as a flat shrinking rectangle rather than an object going away —
/// the glow carries as much of the depth as the size does.
///
/// Anchored on the frame's APPARENT SIZE and nothing else, so a frame that
/// looks a given size always wears the same tube. Anchoring it on the far end
/// of the travel instead — the obvious move — makes the width depend on where
/// the journey happened to start, so shortening the travel silently fattens
/// everything and the nearest frame's halo floods the picture white.
///
/// Floored near a third of the authored width rather than allowed to vanish: the line
/// is already about a pixel wide, so scaling far below that stops drawing the
/// frame rather than making it look distant. Capped at 4x for the same reason
/// from the other end — past a doubling or two the halo is glare, not depth.
///
/// The scale needs no special case at the corner: on the back wall it is how
/// big the rectangle is, and on a side wall the same number is 1/depth, so the
/// two agree at the seam for free.
static float neonScale(const State& s, float z) {
  const float a = apparentSize(s, z);
  const float k = detailClamp(s.depth_scale, 0.0f, 1.0f);
  return 1.0f + (detailClamp(a, 0.3f, 4.0f) - 1.0f) * k;
}

/// The four corners of quad `q` as they land on view `v`, in cover-square coords.
/// `v` 0 = back wall, 1 = left wall, 2 = right wall.
static void wallCorners(const State& s, const WallExtent& w, int v, float z,
                        float out[4][2]) {
  if (v == 0) {
    // A SQUARE on the back wall — isotropic, so it looks like a frame rather
    // than a stretched one, whatever the output's aspect is.
    //
    // Its half-size is measured against the wall's HORIZONTAL extent, because
    // that is the edge that matters: crossing it is what hands the frame over
    // to the side walls. On a wide output that means the square runs off the
    // top and bottom first — its horizontal edges leave through the ceiling and
    // floor, which this room does not have — and for a while you see just its
    // two verticals, still on the back wall, before they cross over too.
    const float a = apparentSize(s, z);
    const float hx = w.ex * a, hy = hx;
    out[0][0] = -hx; out[0][1] = -hy;
    out[1][0] = +hx; out[1][1] = -hy;
    out[2][0] = +hx; out[2][1] = +hy;
    out[3][0] = -hx; out[3][1] = +hy;
    return;
  }

  // A side wall. The frame's vertical edge crosses it at depth d = 1/a; the
  // texture runs far-to-near AWAY from the back wall, so the left wall has its
  // far end on the right and the right wall mirrors it. Laid out side by side,
  // the three outputs are then continuous across both seams.
  const float a = apparentSize(s, z);
  const float d = 1.0f / (a > 1e-4f ? a : 1e-4f);
  const float u = wallU(s, d);
  const float h = w.ey * wallHeight(s, d);
  // The far end abuts the BACK WALL, so it sits on the inner side of each side
  // output: right edge for the left wall, left edge for the right wall. Laid
  // out left / main / right, the three pictures are then continuous across both
  // seams and a frame crossing over does not jump.
  const float sign = (v == 1) ? -1.0f : 1.0f;
  const float x = sign * (2.0f * u - 1.0f) * w.ex;

  // Wound the same way as the back wall's rectangle, with zero width: the edge
  // IS a line, and the shader's own line width is what gives it body.
  out[0][0] = x; out[0][1] = -h;
  out[1][0] = x; out[1][1] = -h;
  out[2][0] = x; out[2][1] = +h;
  out[3][0] = x; out[3][1] = +h;
}

// --- Schema ---------------------------------------------------------------

void module_init() {
  state::init("source.mesh.three_walls", {1, 0, 0},
    state::Schema()
      .helpField("intro",
        "## Three Walls\n"
        "Three neon frames rushing at you down a tunnel. The companion to "
        "**Three Planes** — same look, pointed the other way. Where that one "
        "stacks upward and reads as a level, this one comes out of the screen "
        "and reads as a hit.\n\n"
        "It is quiet until you fire a **move**. *Pulse* throws the three "
        "frames at you one after another. The other three are held: they run "
        "for as long as you hold the trigger and speed up the whole time.\n\n"
        "It has **three outputs**, and they are three cameras on the same "
        "tunnel — the main one looks straight down it, and the two extras "
        "watch it from either side. A pulse that fills the main frame is the "
        "same instant a bar sweeping across the side views, which is what "
        "sells it as something you are passing through. Wire the extras only "
        "if you want them; unconnected, they cost nothing.\n\n"
        "**Try:** hold *Resonate* and watch it climb past the frame rate — "
        "past a point it stops reading as motion and starts strobing into "
        "standing patterns. That is the move doing its job.")

      // ---------------- Moves ----------------
      .group("moves", "Moves")
        .groupHelp(
          "Four ways to send the frames down the tunnel. Only one runs at a "
          "time: firing another drops whatever was going, where it stood.\n\n"
          "**Pulse** is a one-shot — it plays out and stops. It leads with "
          "*Frame 3*, the highlight, and the other two chase it in: the bright "
          "one is the hit and the rest are its tail. The other three moves are "
          "**held**: they run while the trigger is high and their rate ramps "
          "the whole time, so how long you hold is the performance.\n\n"
          "*Cycles* runs two frames against each other, one toward you and one "
          "away, on separate rate curves so they drift in and out of step. "
          "*Resonate* sends all three at once, evenly spaced, fast enough to "
          "beat against the frame rate. *Resonate Rev* is the same going away.")
      .eventField("pulse", state::PrimaryInput).label("Pulse", "Pulse")
      .eventField("cycles", state::PrimaryInput).label("Cycles", "Cycles")
      .eventField("resonate", state::PrimaryInput).label("Resonate", "Reso")
      .eventField("resonate_rev", state::PrimaryInput).label("Resonate Rev", "Reso R")
      .floatField("ease", 0.5f, 0.f, 1.f, state::PrimaryInput, nullptr, 0.f, nullptr,
                  "Travel shape: 0 straight-line, 0.5 eased, 1 heavily eased.")
        .label("Ease", "Ease")

      .floatField("pulse_time", 0.9f, 0.05f, 6.f, state::SecondaryInput,
                  nullptr, 0.f, "s", "How long one frame takes to cross.")
        .label("Pulse Time", "PlsT")
      .floatField("pulse_stagger", 0.12f, 0.f, 2.f, state::SecondaryInput,
                  nullptr, 0.f, "s",
                  "Gap between the three frames of a pulse. The train leads "
                  "with Frame 3, the highlight, and the others chase it in.")
        .label("Pulse Stagger", "Stgr")

      .floatField("cycles_f0", 0.4f, 0.f, 30.f, state::SecondaryInput,
                  nullptr, 0.f, "Hz", "Cycles rate at the press.")
        .label("Cycles From", "CycF0")
      .floatField("cycles_f1", 2.6f, 0.f, 30.f, state::SecondaryInput,
                  nullptr, 0.f, "Hz", "Cycles rate at the end of the ramp.")
        .label("Cycles To", "CycF1")
      .floatField("cycles_rev_f0", 0.3f, 0.f, 30.f, state::SecondaryInput,
                  nullptr, 0.f, "Hz",
                  "The reverse frame's own start rate. Keep it different from "
                  "the forward one — two identical opposed ramps beat at a "
                  "fixed rate, which is much less interesting than two that "
                  "drift apart.")
        .label("Cycles Rev From", "RevF0")
      .floatField("cycles_rev_f1", 2.0f, 0.f, 30.f, state::SecondaryInput,
                  nullptr, 0.f, "Hz", "The reverse frame's end rate.")
        .label("Cycles Rev To", "RevF1")
      .floatField("cycles_ramp", 4.0f, 0.1f, 30.f, state::SecondaryInput,
                  nullptr, 0.f, "s", "Seconds to get from the start rate to the end rate.")
        .label("Cycles Ramp", "CycRmp")
      .floatField("cycles_duty", 0.5f, 0.f, 1.f, state::SecondaryInput,
                  nullptr, 0.f, nullptr,
                  "How much of each round the third frame spends going toward you.")
        .label("Duty", "Duty")
      .floatField("cycles_duty_period", 1.5f, 0.05f, 10.f, state::SecondaryInput,
                  nullptr, 0.f, "s", "One there-and-back round of the duty cycle.")
        .label("Duty Period", "DutyT")

      .floatField("resonate_f0", 2.0f, 0.f, 60.f, state::SecondaryInput,
                  nullptr, 0.f, "Hz", "Resonate rate at the press.")
        .label("Resonate From", "ResF0")
      .floatField("resonate_f1", 14.0f, 0.f, 60.f, state::SecondaryInput,
                  nullptr, 0.f, "Hz",
                  "Resonate rate at the end of the ramp. Past roughly half the "
                  "frame rate this stops reading as motion and starts landing "
                  "on standing patterns — which is the point of the move.")
        .label("Resonate To", "ResF1")
      .floatField("resonate_ramp", 6.0f, 0.1f, 30.f, state::SecondaryInput,
                  nullptr, 0.f, "s", "Seconds to get from the start rate to the end rate.")
        .label("Resonate Ramp", "ResRmp")

      // ---------------- Tunnel ----------------
      .group("tunnel", "Room")
        .groupHelp(
          "The box the frames travel through. The camera sits at its mouth "
          "looking at the back wall, and the three outputs are its three "
          "walls — so a frame's left edge is either still on the back wall or "
          "already on the left one, never in two pictures at once.\n\n"
          "*Travel Start* and *Travel End* are where a frame enters and "
          "leaves. It crosses off the back wall and onto the sides partway "
          "through; *Frame Size* moves where that happens.\n\n"
          "*Keystone* decides what the two side outputs actually contain. At "
          "0 each is that wall's own flat surface, which is what you want when "
          "the output drives a screen standing where that wall is — the real "
          "geometry then does the perspective for you. At 1 it is the wall as "
          "the camera sees it, so a frame SWELLS as it comes at you and runs "
          "off the top and bottom of the picture. Either way the corner holds: "
          "a frame straddling the threshold is the same height in both "
          "outputs, so the room never tears at the seam.\n\n"
          "*Stretch* rakes the side walls — a lie about the shape of the "
          "space, and the strongest single thing here for selling depth. Above "
          "1 the frames tear along them and run clean off the near end, the "
          "way something passing you does. You are seeing a shorter length of "
          "tunnel, larger; below 1 you see more of it, crawling.")
      .floatField("quad_size", 1.0f, 0.1f, 4.f, state::PrimaryInput,
                  nullptr, 0.f, nullptr,
                  "Moves where a frame crosses off the back wall and onto the "
                  "sides. Bigger overflows sooner.")
        .label("Frame Size", "Size")
      .floatField("z_far", 8.0f, 1.f, 40.f, state::SecondaryInput,
                  nullptr, 0.f, nullptr,
                  "How far back a frame starts. Larger begins it smaller and "
                  "deeper in.")
        .label("Travel Start", "Start")
      .floatField("z_near", 0.15f, 0.02f, 1.f, state::SecondaryInput,
                  nullptr, 0.f, nullptr,
                  "How near a frame gets before it is gone — also how far "
                  "along the side walls the bars reach.")
        .label("Travel End", "End")
      .floatField("wall_keystone", 0.0f, 0.f, 1.f, state::PrimaryInput,
                  nullptr, 0.f, nullptr,
                  "0 = the side outputs are those walls' own flat surfaces; "
                  "1 = the walls as the camera sees them, growing toward you. "
                  "Either way a frame crossing the corner keeps its height.")
        .label("Keystone", "Keyst")
      .floatField("wall_stretch", 1.0f, 0.2f, 5.f, state::PrimaryInput,
                  nullptr, 0.f, nullptr,
                  "Rakes the side walls. Above 1 the frames tear along them and "
                  "run off the near end — you see a shorter length of tunnel, "
                  "larger. Below 1 you see more of it, crawling.")
        .label("Stretch", "Strch")
      .floatField("depth_scale", 1.0f, 0.f, 1.f, state::PrimaryInput,
                  nullptr, 0.f, nullptr,
                  "How hard the neon itself scales with depth — near frames "
                  "get a fat tube and a wide bloom, far ones a hairline. At 0 "
                  "the line is the same width everywhere, which reads flat.")
        .label("Depth Glow", "DGlow")

      // ---------------- Quads ----------------
      .group("quads", "Frames")
        .groupHelp(
          "The three frames. Their colours default to Three Planes' own, so "
          "the two cards look like they belong in the same piece.\n\n"
          "*Fill* is signed: positive floods the frame with its own colour, "
          "negative turns it into a black body that eats the glow of anything "
          "behind it while keeping its own outline.")
      .rgbField("quad1_color", 1.00f, 0.22f, 0.62f, state::SecondaryInput)
        .label("Frame 1 Colour", "C1")
      .floatField("quad1_emission", 0.85f, 0.f, 1.f, state::SecondaryInput)
        .label("Frame 1 Emission", "E1")
      .floatField("quad1_fill", 0.0f, -1.f, 1.f, state::SecondaryInput, "signed")
        .label("Frame 1 Fill", "F1")
      .rgbField("quad2_color", 0.72f, 0.35f, 1.00f, state::SecondaryInput)
        .label("Frame 2 Colour", "C2")
      .floatField("quad2_emission", 0.85f, 0.f, 1.f, state::SecondaryInput)
        .label("Frame 2 Emission", "E2")
      .floatField("quad2_fill", 0.0f, -1.f, 1.f, state::SecondaryInput, "signed")
        .label("Frame 2 Fill", "F2")
      .rgbField("quad3_color", 0.30f, 0.85f, 1.00f, state::SecondaryInput)
        .label("Frame 3 Colour", "C3")
      .floatField("quad3_emission", 0.85f, 0.f, 1.f, state::SecondaryInput)
        .label("Frame 3 Emission", "E3")
      .floatField("quad3_fill", 0.0f, -1.f, 1.f, state::SecondaryInput, "signed")
        .label("Frame 3 Fill", "F3")

      // ---------------- Neon ----------------
      .group("neon", "Neon")
        .groupHelp(
          "The tube itself, shared with Three Planes so a look moves between "
          "the two cards. *Core Whiten* is what makes it read as neon at all: "
          "real neon photographs blows its core to white and keeps the hue "
          "only out in the halo.")
      .floatField("line_width", 0.18f, 0.f, 1.f, state::PrimaryInput)
        .label("Line Width", "Width")
      .floatField("line_gain", 1.60f, 0.f, 4.f, state::PrimaryInput)
        .label("Line Gain", "Line G")
      .floatField("core_whiten", 0.85f, 0.f, 1.f, state::SecondaryInput)
        .label("Core Whiten", "Whiten")
      .floatField("halo_radius", 0.30f, 0.f, 1.f, state::PrimaryInput)
        .label("Halo Radius", "Halo R")
      .floatField("halo_gain", 0.55f, 0.f, 3.f, state::PrimaryInput)
        .label("Halo Gain", "Halo G")
      .floatField("halo_falloff", 0.45f, 0.f, 1.f, state::SecondaryInput,
                  nullptr, 0.f, nullptr, "0 = tight and punchy, 1 = wide and soft.")
        .label("Halo Falloff", "Fall")
      .floatField("halo_smooth", 0.35f, 0.f, 1.f, state::SecondaryInput,
                  nullptr, 0.f, nullptr,
                  "Rounds the corner creases just outside the outline, as a "
                  "fraction of the halo radius. 0 shows the raw ridge.")
        .label("Halo Smoothing", "Smooth")
      .floatField("corner_radius", 0.012f, 0.f, 0.25f, state::SecondaryInput)
        .label("Corner Radius", "Corner")
      .floatField("fill_gain", 0.22f, 0.f, 2.f, state::SecondaryInput)
        .label("Fill Gain", "Fill G")

      // ---------------- Grade ----------------
      .group("grade", "Warmth & Dehancement")
        .groupHelp(
          "The analogue tail, shared verbatim with Three Planes. *Chroma "
          "Bleed* splits R/G/B horizontally the way a tape transport does — "
          "exact here, because the whole stack is re-evaluated at three "
          "offsets rather than blurred.")
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
      .floatField("highlight_tint_amount", 0.0f, 0.f, 1.f, state::PrimaryInput)
        .label("Highlight Tint Amount", "Tint Amt")
      .floatField("highlight_tint_pivot", 1.0f, 0.2f, 4.f, state::SecondaryInput)
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
      .boolField("debug_show_quads", false, state::SecondaryInput,
                 "Flat per-frame keys, no glow or grade — check the projection "
                 "and the near-plane culling on their own.")
        .label("Show Frame Keys", "Keys")

      // ---------------- Outputs ----------------
      // Declared min/max IS the modulation contract for these rails.
      .floatField("depth1", 0.f, 0.f, 1.f, state::PrimaryOutput, "unsigned",
                  0.f, nullptr, "Frame 1's travel, 0 at the far end and 1 at the near.")
        .label("Frame 1 Depth", "D1")
      .floatField("depth2", 0.f, 0.f, 1.f, state::SecondaryOutput, "unsigned",
                  0.f, nullptr, "Frame 2's travel, 0 at the far end and 1 at the near.")
        .label("Frame 2 Depth", "D2")
      .floatField("depth3", 0.f, 0.f, 1.f, state::SecondaryOutput, "unsigned",
                  0.f, nullptr, "Frame 3's travel, 0 at the far end and 1 at the near.")
        .label("Frame 3 Depth", "D3")
      .floatField("move_rate", 0.f, 0.f, 60.f, state::SecondaryOutput, "unsigned",
                  0.f, "Hz", "The cycling rate right now; 0 when nothing is cycling.")
        .label("Move Rate", "Rate")

      .textureField("tex_in",  state::PrimaryInput)
      .textureField("tex_out", state::PrimaryOutput)
      // The two extra cameras. Effect-owned, and dispatched only when wired —
      // see the note at the top of the file.
      .textureField("left_out",  state::SecondaryOutput)
      .textureField("right_out", state::SecondaryOutput)

      // ---------------- LED bars ----------------
      // After tex_out, and it matters: the editor takes a module's texture
      // output to be the FIRST one its schema declares (schema-channels.ts,
      // firstFieldOfType — it sorts on declaration order).
      .group("led", "LED Bars")
        .groupHelp(
          "A pixel map for the house rig: four vertical bars, ten segments "
          "each, as a 4x10 grid of flat blocks. It is a CONTROL SIGNAL, not a "
          "picture — wire *LED Out* at whatever drives the bars.\n\n"
          "The bars stand in the room BEYOND the two side walls, two a side. A "
          "frame fills the back wall, breaks out onto the sides, tears down "
          "them, and then reaches these — so the rig fires LAST, as a pair of "
          "lights opening outward from the screen. The room is symmetric, so "
          "the outer two always agree and so do the inner two.\n\n"
          "Unwired, none of this is drawn.")
      .textureField("led_out", state::SecondaryOutput)
        .label("LED Out", "LED")
      .floatField("led_spacing", 1.5f, 0.25f, 6.f, state::SecondaryInput,
                  nullptr, 0.f, nullptr,
                  "How far out the bars stand, as the gap between the two "
                  "rings — measured in back-wall widths, so 1 is one whole "
                  "wall past its edge. Small brings them in tight behind the "
                  "screen; large pushes them out to the end of the throw.")
        .label("Bar Spacing", "Space")
      .floatField("led_width", 0.45f, 0.05f, 2.f, state::SecondaryInput,
                  nullptr, 0.f, nullptr,
                  "How long a bar stays lit as a frame goes past, as a "
                  "fraction of the spacing. Narrow is a hard blink; wide has "
                  "the two rings overlapping into a wash.")
        .label("Pass Width", "Width")
      .floatField("led_decay", 0.12f, 0.f, 1.f, state::SecondaryInput,
                  nullptr, 0.f, "s",
                  "How long a bar takes to fall back after it is hit. The "
                  "frames tear through this part of the room fastest, so "
                  "without some decay a quick move puts the whole pass between "
                  "two frames and the bar never lights at all.")
        .label("Bar Decay", "Decay")
      .boolField("led_quantize", true, state::SecondaryInput,
                 "Each segment a flat block of colour, which is what you want: "
                 "anything sampling or averaging inside a block then comes away "
                 "with the exact colour that segment asked for. Turn it off to "
                 "interpolate between segment centres.")
        .label("Quantize", "Quant")

      .capability(state::Capability::Generator)
      // NOT TimeIndependent, unlike three_planes: every move here is an
      // accumulator, so a frame is not a pure function of the current inputs
      // and this cannot be seeked.
      .capability(state::Capability::ModulationSource)
      .capability(state::Capability::ModulationSourceMulti)
  );

  if (gpu::Device::backend() == gpu::Backend::None) return;

  state::registerShaderSPV("three_walls_render", RENDER_SPV, RENDER_SPV_SIZE);
  auto cs = gpu::Device::createShaderModuleByName("three_walls_render");
  if (!cs) return;

  s_pso = gpu::Device::createComputePSO(cs, "main", gpu::Bindings()
      .tex2d(0)
      .storageTex2d(1)
      .uniform(2));

  state::registerShaderSPV("three_walls_led", LED_SPV, LED_SPV_SIZE);
  if (auto led = gpu::Device::createShaderModuleByName("three_walls_led")) {
    s_led_pso = gpu::Device::createComputePSO(led, "main", gpu::Bindings()
        .storageTex2d(0)
        .uniform(1));
  }

  state::log("three_walls: module initialized");
}

void* create() {
  auto* s = new State();
  for (int v = 0; v < VIEWS; v++)
    s->uniform_buf[v] = gpu::Device::createBuffer(sizeof(Uniforms),
                                                  gpu::BufferUsage::Uniform);
  s->led_buf = gpu::Device::createBuffer(sizeof(LedUniforms),
                                         gpu::BufferUsage::Uniform);
  return s;
}

void destroy(void* self) {
  auto* s = static_cast<State*>(self);
  if (!s) return;
  for (int v = 0; v < VIEWS; v++) s->uniform_buf[v].release();
  for (int i = 0; i < 2; i++) s->aux_tex[i].release();
  s->led_buf.release();
  s->led_tex.release();
  delete s;
}

void init(void* self) {
  auto* s = static_cast<State*>(self);
  if (!s) return;
  s->core.reset();
  if (!s_pso.valid() || !s->uniform_buf[0].valid()) return;
  s->initialized = true;
}

// --- Publishing -----------------------------------------------------------

static void publish(const char* name, float value) {
  auto vh = val::number(value);
  state::setValPath(name, vh);
  val::release(vh);
}

/// The travel rails, 0 at the far end and 1 at the near. Published from tick()
/// so taps read THIS frame's value before it is consumed.
static void publishRails(const State& s) {
  publish("depth1", s.frame.phase[0]);
  publish("depth2", s.frame.phase[1]);
  publish("depth3", s.frame.phase[2]);
  publish("move_rate", s.frame.rate);
}

// --- The LED bars ---------------------------------------------------------
//
// Four vertical bars standing in the room BEYOND the two side walls, two on
// each side. A frame comes at you, fills the back wall, breaks out onto the
// side walls, tears down them — and then reaches these, which are the last
// thing it passes on its way through you.
//
// They are placed in `a`, the frame's apparent half-size (the room note above),
// because `a = 1` is exactly the back wall's edge and everything past that is
// outside the box. Deliberately NOT in the side wall's own texture coordinate:
// keystone and stretch are how those PICTURES get drawn, and a light standing
// in the room does not move because you remapped a projector.
//
// The response and the ordering live in <sketch/led_bars.h>, host-free, so the
// Catch2 goldens can drive a frame past the rig at an exact dt. All this does
// is read the scene out of the show and hand it over.
static void updateLed(State& s, float dt) {
  led_bars::WallScene scene;
  for (int q = 0; q < QUADS; q++) {
    scene.a[q] = apparentSize(s, s.frame.z[q]);
    scene.level[q] = ledLevel(s.emission[q]);
    scene.live[q] = s.frame.live[q];
    for (int c = 0; c < 3; c++) scene.rgb[q][c] = s.color[q][c];
  }
  const led_bars::Bars inst =
      led_bars::wallBars(scene, s.led_spacing, s.led_width);
  for (int b = 0; b < led_bars::kBars; b++)
    for (int c = 0; c < 3; c++)
      s.led_now.rgb[b][c] = led_bars::peakHold(s.led_now.rgb[b][c],
                                               inst.rgb[b][c], dt, s.led_decay);
}

void tick(void* self, double dt) {
  auto* s = static_cast<State*>(self);
  if (!s) return;

  // A move armed by a patch starts HERE, so it always sees the knobs that
  // arrived with it. Release first, then press: a press in the same transaction
  // as somebody else's release must win.
  if (s->pending_release != show::MoveNone) {
    s->core.release(s->pending_release);
    s->pending_release = show::MoveNone;
  }
  if (s->pending_press != show::MoveNone) {
    s->core.trigger(s->pending_press);
    s->pending_press = show::MoveNone;
  }

  s->frame = s->core.tick(s->show_p, static_cast<float>(dt));
  publishRails(*s);
  updateLed(*s, static_cast<float>(dt));
}

// --- Render ---------------------------------------------------------------

/// Fill the uniform block for one camera. `quad_order` is near-to-far, so a
/// masking frame in front eats the glow of the ones behind it.
static void fillUniforms(State* s, int v, int vp_w, int vp_h, Uniforms& u) {
  const auto cs = fx::coverSquare(vp_w, vp_h);
  const WallExtent wall = frameExtent(cs);

  // Sort far-to-near, so a nearer frame's mask eats the glow of the ones behind
  // it. Three items, so a hand-rolled insertion sort beats pulling in
  // <algorithm>.
  int order[QUADS] = {0, 1, 2};
  for (int i = 1; i < QUADS; i++) {
    const int key = order[i];
    int j = i - 1;
    while (j >= 0 && s->frame.z[order[j]] < s->frame.z[key]) {
      order[j + 1] = order[j];
      j--;
    }
    order[j + 1] = key;
  }

  for (int slot = 0; slot < QUADS; slot++) {
    const int q = order[slot];
    float pts[4][2] = {};
    wallCorners(*s, wall, v, s->frame.z[q], pts);

    // Rows 2*slot and 2*slot+1: (c0.xy, c1.zw) then (c2.xy, c3.zw).
    u.corners[slot * 2 + 0][0] = pts[0][0];
    u.corners[slot * 2 + 0][1] = pts[0][1];
    u.corners[slot * 2 + 0][2] = pts[1][0];
    u.corners[slot * 2 + 0][3] = pts[1][1];
    u.corners[slot * 2 + 1][0] = pts[2][0];
    u.corners[slot * 2 + 1][1] = pts[2][1];
    u.corners[slot * 2 + 1][2] = pts[3][0];
    u.corners[slot * 2 + 1][3] = pts[3][1];

    u.quad_color[slot][0] = s->color[q][0];
    u.quad_color[slot][1] = s->color[q][1];
    u.quad_color[slot][2] = s->color[q][2];
    // A frame that is not live goes dark rather than being skipped: the shader
    // loop is unrolled over a fixed three, and zero emission costs the same as
    // a branch would. Frames that have left THIS wall need no such treatment —
    // they simply land outside the picture, and their halo bleeding in around
    // the corner is exactly what we want.
    u.quad_color[slot][3] = s->frame.live[q] ? emissionDrive(s->emission[q]) : 0.0f;
    u.fills[slot] = s->frame.live[q] ? s->fill[q] : 0.0f;
  }
  u.fills[3] = 0.0f;

  for (int slot = 0; slot < QUADS; slot++)
    u.depth[slot] = neonScale(*s, s->frame.z[order[slot]]);
  u.depth[3] = 0.0f;

  const float px = 2.0f / float(vp_w > vp_h ? vp_w : vp_h);

  u.misc[0] = s->chroma_bleed;
  u.misc[1] = s->input_opacity;
  u.misc[2] = s->debug_show_quads ? 1.0f : 0.0f;
  u.misc[3] = 0.0f;

  u.view[0] = float(vp_w);
  u.view[1] = float(vp_h);
  u.view[2] = cs.ax;
  u.view[3] = cs.ay;

  u.style[0]  = lineHalfWidth(s->line_width);
  u.style[1]  = s->line_gain;
  u.style[2]  = s->core_whiten;
  u.style[3]  = haloRadius(s->halo_radius);
  u.style[4]  = s->halo_gain;
  u.style[5]  = s->halo_falloff;
  // Corner rounding is meaningless on a side wall and actively wrong: a frame's
  // edge there is a zero-area LINE, and shrinking a line by corner_r pushes its
  // outline out to either side, so the bar renders as two cores with a notch
  // down the middle. Zero it and the line is a line.
  u.style[6]  = (v == 0) ? s->corner_radius : 0.0f;
  u.style[7]  = px * 1.2f;
  u.style[8]  = s->fill_gain;
  u.style[9]  = s->halo_smooth;
  u.style[10] = 0.0f;
  u.style[11] = 0.0f;

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
  u.grade[10] = float(std::fmod(host::time() * 997.0, 4096.0));
  u.grade[11] = s->highlight_tint_pivot;
  u.grade[12] = s->highlight_tint[0];
  u.grade[13] = s->highlight_tint[1];
  u.grade[14] = s->highlight_tint[2];
  u.grade[15] = s->highlight_tint_amount;
}

/// The LED map, on the same aux-output pattern as the two side cameras: the
/// effect owns the allocation, publishes the handle once, and does not dispatch
/// unless something downstream is wired to it.
///
/// Drawn at the full viewport rather than as a 4x10 thumbnail. The map is forty
/// flat blocks either way, but at viewport size each block is hundreds of
/// pixels across, so anything that resamples it stays well inside one and comes
/// away with the exact colour. A 4x10 texture stretched up would be bilinearly
/// smeared into precisely the averaging the quantising is there to avoid.
static void renderLed(State* s, int vp_w, int vp_h) {
  if (!s_led_pso.valid() || !s->led_buf.valid()) return;
  if (!state::isOutputConnected("led_out")) return;

  if (!s->led_tex.valid() || s->led_w != vp_w || s->led_h != vp_h) {
    s->led_tex.release();
    s->led_tex = gpu::Device::createTexture(vp_w, vp_h);
    s->led_w = vp_w;
    s->led_h = vp_h;
    if (!s->led_tex.valid()) return;
    state::setGpuTexture("led_out", s->led_tex.id);
  }

  const led_bars::Cells cells = led_bars::barsToCells(s->led_now);

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
  if (!s || !s->initialized) return;

  auto in  = gpu::Device::textureForField("tex_in");
  auto out = gpu::Device::textureForField("tex_out");
  if (!in.valid() || !out.valid()) return;

  static const char* const kAuxField[2] = {"left_out", "right_out"};

  for (int v = 0; v < VIEWS; v++) {
    gpu::Texture target = out;
    if (v > 0) {
      const int a = v - 1;
      // Nothing downstream, nothing drawn. The aux views are the only optional
      // work here and they are a full-viewport dispatch each.
      if (!state::isOutputConnected(kAuxField[a])) continue;
      if (!s->aux_tex[a].valid() || s->aux_w[a] != vp_w || s->aux_h[a] != vp_h) {
        s->aux_tex[a].release();
        s->aux_tex[a] = gpu::Device::createTexture(vp_w, vp_h);
        s->aux_w[a] = vp_w;
        s->aux_h[a] = vp_h;
        if (!s->aux_tex[a].valid()) continue;
        // Publish on ALLOCATION only — the executor never clears an output
        // handle, so it persists across frames.
        state::setGpuTexture(kAuxField[a], s->aux_tex[a].id);
      }
      target = s->aux_tex[a];
    }

    Uniforms u = {};
    fillUniforms(s, v, vp_w, vp_h, u);
    s->uniform_buf[v].writeOne(u);

    auto cp = gpu::ComputePass::begin();
    cp.setPSO(s_pso);
    cp.setTexture(in, 0, 0);
    cp.setTexture(target, 1, 1);
    cp.setBuffer(s->uniform_buf[v], 2);
    cp.dispatch((vp_w + 7) / 8, (vp_h + 7) / 8);
    cp.end();
  }

  renderLed(s, vp_w, vp_h);

  gpu::Device::submit();
}

// --- Patch decode ---------------------------------------------------------

void on_state_patched(void* self, int n, const char* pb, const int* off,
                      const int* len, const int* ops) {
  auto* s = static_cast<State*>(self);
  if (!s) return;

  static const char* const kMoveField[show::kMoveCount] = {
      "pulse", "cycles", "resonate", "resonate_rev"};

  for (int i = 0; i < n; i++) {
    if (ops[i] != state::PatchReplace) continue;
    const char* p = pb + off[i];
    const int l = len[i];

    // The four move triggers. EDGES ONLY — the executor replays every stored
    // value as a PatchReplace each frame, so acting on arrival would re-arm
    // forever (style guide 8.2). Rising presses; falling releases a gate.
    {
      bool matched = false;
      for (int m = 0; m < show::kMoveCount; m++) {
        if (!state::pathIs(p, l, kMoveField[m])) continue;
        const bool t = state::patchEvent(i);
        if (t && !s->move_prev[m]) s->pending_press = m;
        else if (!t && s->move_prev[m] && show::isGate(m)) s->pending_release = m;
        s->move_prev[m] = t;
        matched = true;
        break;
      }
      if (matched) continue;
    }

    // "quad<k>_color" / "_emission" / "_fill", matched by shape.
    if (l > 6 && std::memcmp(p, "quad", 4) == 0 && p[5] == '_') {
      const int k = p[4] - '1';
      if (k >= 0 && k < QUADS) {
        const char* tail = p + 6;
        const int tl = l - 6;
        if (tl == 5 && std::memcmp(tail, "color", 5) == 0) {
          auto v = state::patchVec3(i);
          s->color[k][0] = v.x; s->color[k][1] = v.y; s->color[k][2] = v.z;
          continue;
        }
        if (tl == 8 && std::memcmp(tail, "emission", 8) == 0) {
          s->emission[k] = state::patchFloat(i);
          continue;
        }
        if (tl == 4 && std::memcmp(tail, "fill", 4) == 0) {
          s->fill[k] = state::patchFloat(i);
          continue;
        }
      }
    }

    if      (state::pathIs(p, l, "ease"))          s->show_p.ease = state::patchFloat(i);
    else if (state::pathIs(p, l, "pulse_time"))    s->show_p.pulse_time = state::patchFloat(i);
    else if (state::pathIs(p, l, "pulse_stagger")) s->show_p.pulse_stagger = state::patchFloat(i);
    else if (state::pathIs(p, l, "cycles_f0"))     s->show_p.cycles_f0 = state::patchFloat(i);
    else if (state::pathIs(p, l, "cycles_f1"))     s->show_p.cycles_f1 = state::patchFloat(i);
    else if (state::pathIs(p, l, "cycles_rev_f0")) s->show_p.cycles_rev_f0 = state::patchFloat(i);
    else if (state::pathIs(p, l, "cycles_rev_f1")) s->show_p.cycles_rev_f1 = state::patchFloat(i);
    else if (state::pathIs(p, l, "cycles_ramp"))   s->show_p.cycles_ramp = state::patchFloat(i);
    else if (state::pathIs(p, l, "cycles_duty"))   s->show_p.cycles_duty = state::patchFloat(i);
    else if (state::pathIs(p, l, "cycles_duty_period")) s->show_p.cycles_duty_period = state::patchFloat(i);
    else if (state::pathIs(p, l, "resonate_f0"))   s->show_p.resonate_f0 = state::patchFloat(i);
    else if (state::pathIs(p, l, "resonate_f1"))   s->show_p.resonate_f1 = state::patchFloat(i);
    else if (state::pathIs(p, l, "resonate_ramp")) s->show_p.resonate_ramp = state::patchFloat(i);
    else if (state::pathIs(p, l, "z_far"))         s->show_p.z_far = state::patchFloat(i);
    else if (state::pathIs(p, l, "z_near"))        s->show_p.z_near = state::patchFloat(i);
    else if (state::pathIs(p, l, "quad_size"))     s->quad_size = state::patchFloat(i);
    else if (state::pathIs(p, l, "wall_keystone")) s->wall_keystone = state::patchFloat(i);
    else if (state::pathIs(p, l, "wall_stretch"))  s->wall_stretch = state::patchFloat(i);
    else if (state::pathIs(p, l, "depth_scale"))   s->depth_scale = state::patchFloat(i);
    else if (state::pathIs(p, l, "line_width"))    s->line_width = state::patchFloat(i);
    else if (state::pathIs(p, l, "line_gain"))     s->line_gain = state::patchFloat(i);
    else if (state::pathIs(p, l, "core_whiten"))   s->core_whiten = state::patchFloat(i);
    else if (state::pathIs(p, l, "halo_radius"))   s->halo_radius = state::patchFloat(i);
    else if (state::pathIs(p, l, "halo_gain"))     s->halo_gain = state::patchFloat(i);
    else if (state::pathIs(p, l, "halo_falloff"))  s->halo_falloff = state::patchFloat(i);
    else if (state::pathIs(p, l, "halo_smooth"))   s->halo_smooth = state::patchFloat(i);
    else if (state::pathIs(p, l, "corner_radius")) s->corner_radius = state::patchFloat(i);
    else if (state::pathIs(p, l, "fill_gain"))     s->fill_gain = state::patchFloat(i);
    else if (state::pathIs(p, l, "chroma_bleed"))  s->chroma_bleed = state::patchFloat(i);
    else if (state::pathIs(p, l, "warmth"))        s->warmth = state::patchFloat(i);
    else if (state::pathIs(p, l, "drive"))         s->drive = state::patchFloat(i);
    else if (state::pathIs(p, l, "exposure"))      s->exposure = state::patchFloat(i);
    else if (state::pathIs(p, l, "asymmetry"))     s->asymmetry = state::patchFloat(i);
    else if (state::pathIs(p, l, "toe"))           s->toe = state::patchFloat(i);
    else if (state::pathIs(p, l, "shoulder"))      s->shoulder = state::patchFloat(i);
    else if (state::pathIs(p, l, "highlight_desat")) s->highlight_desat = state::patchFloat(i);
    else if (state::pathIs(p, l, "highlight_tint")) {
      auto v = state::patchVec3(i);
      s->highlight_tint[0] = v.x; s->highlight_tint[1] = v.y; s->highlight_tint[2] = v.z;
    }
    else if (state::pathIs(p, l, "highlight_tint_amount")) s->highlight_tint_amount = state::patchFloat(i);
    else if (state::pathIs(p, l, "highlight_tint_pivot"))  s->highlight_tint_pivot = state::patchFloat(i);
    else if (state::pathIs(p, l, "scanline"))       s->scanline = state::patchFloat(i);
    else if (state::pathIs(p, l, "scanline_count")) s->scanline_count = state::patchInt(i);
    else if (state::pathIs(p, l, "grain"))          s->grain = state::patchFloat(i);
    else if (state::pathIs(p, l, "input_opacity"))  s->input_opacity = state::patchFloat(i);
    else if (state::pathIs(p, l, "led_spacing"))   s->led_spacing = state::patchFloat(i);
    else if (state::pathIs(p, l, "led_width"))     s->led_width = state::patchFloat(i);
    else if (state::pathIs(p, l, "led_decay"))     s->led_decay = state::patchFloat(i);
    else if (state::pathIs(p, l, "led_quantize"))  s->led_quantize = state::patchBool(i);
    else if (state::pathIs(p, l, "debug_show_quads")) s->debug_show_quads = state::patchBool(i);
  }
}

}  // namespace three_walls
