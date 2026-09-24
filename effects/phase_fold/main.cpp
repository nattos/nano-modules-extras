/*
 * source.phase_fold — emergent limit-cycle phase-portrait generator.
 *
 * Productionized from the phase-fold research testbed. A baked GxG atlas of
 * emergent level-set limit-cycle FIELDS (axes eccentricity x × lobedness y) is
 * uploaded to the GPU as a flat cell buffer. The XY pad picks 4 corner cells +
 * blend weights on the CPU each frame; everything else runs on the GPU:
 *
 *   backdrop (compute) — the blended scalar field H, banded as a muted
 *                        diverging colormap. Seeds tex_out.
 *   stream   (compute) — traces an NS×NS grid of streamlines through the blended
 *                        vector field v = level-set flow + WIND(z); a continuous
 *                        glow rides down each line (flow_phase), no quantized
 *                        arrowhead.
 *   solve    (compute) — STATEFUL limit-cycle discoverer. A persistent ring of
 *                        PF_PARTICLES particles spawns on the resting cycle and
 *                        is Newton-relaxed onto the (wind-corrected) cycle each
 *                        frame, moving only along ∇H so the ring keeps its
 *                        distribution. Re-seeds on a hard timer.
 *   cycle    (compute) — builds segments between consecutive particles, with
 *                        BREAK DETECTION (gap too large or ∇H flips → drop the
 *                        segment), so a cycle that fails to close (killed by
 *                        wind past the SNIC, or just isn't one) visibly opens.
 *   line raster (vs/fs) — draws the traced segments as soft anti-aliased quads,
 *                        blended over the backdrop. Streamlines and the limit-
 *                        cycle tracer are SEPARATE, independently toggleable
 *                        stages.
 *
 * The prototype did the tracing / arrow animation / cycle integration on the
 * CPU every frame; here it is all GPU compute. WIND (z) is the non-potential
 * z-axis force that distorts then kills the cycle (SNIC bifurcation). An
 * optional autopilot spirals the XY automatically and broadcasts its live
 * position (autopilot_x/y) without mutating the inputs. Pure generator (no
 * input). Multi-pass + raster → NO fusion. Internal clock → NOT is_identity.
 */

#include <gpu.h>
#include <host.h>
#include <val.h>
#include <effect_utils.h>
#include "phase_fold_shaders.h"
#include "phase_fold_atlas.h"

#include <cmath>
#include <cstdint>
#include <algorithm>

namespace phase_fold {

// Must match common.hlsl.
static constexpr int   PF_NS          = 15;    // streamline seed grid
static constexpr int   PF_SL_STEPS    = 16;    // segments stored per streamline
static constexpr int   PF_PARTICLES   = PF_NOUT;  // stateful cycle-solver particles
static constexpr float PF_EXTENT      = 1.35f; // phase window half-size
static constexpr float PF_STEP_MIN    = 0.001f; // step_size slider 0 → this
static constexpr float PF_STEP_MAX    = 0.5f;   // step_size slider 1 → this
static constexpr int   PF_TRACE_MAX   = 400;   // tracer history cap (Tracer mode)
static constexpr int   PF_TRACE_PER_FRAME = 4; // tracer steps advanced per frame
static constexpr float PF_TRACE_DT    = 0.02f; // tracer integration step

static constexpr int   kStreamSegs = PF_NS * PF_NS * PF_SL_STEPS;     // 3600
static constexpr int   kCycleSegs  = PF_PARTICLES;                    // one seg per pair
static constexpr int   kSegFloats  = 12;       // Segment = 3×float4
static constexpr int   kCellFloats  = PF_GRID * PF_GRID * PF_STRIDE;
static constexpr int   kCurveFloats = PF_GRID * PF_GRID * PF_NOUT * 2;
static constexpr int   kParticleFloats = PF_PARTICLES * 4;            // float4 each

static constexpr float kPi = 3.14159265358979323846f;

// Autopilot epicycle (port of app.js): two summed circular motions 90° out of
// phase, incommensurate rates → sweeps the pad without stalling at the centre.
static constexpr float kApA = 0.34f, kApB = 0.16f, kApW2 = 0.382f, kApPhi = kPi * 0.5f;

// Skip-empty jog: the atlas has invalid "holes" that render as a flat dark fill.
// When the effective XY nears one (its validity coverage drops), a C2 ramp
// (fx::SkipJog) accelerates the autopilot orbit to glide through the dead patch.
// There is no time lever here (the field is XY-only), so this needs autopilot on.
// Jog is a MULTIPLE of the current orbit speed (not an absolute rate), and the
// ramp is asymmetric: a gentle C2 ease-IN, a quick ease-OUT once a live field
// returns (leaving a flat fill, so a harsher slowdown reads fine).
static constexpr float kSkipOrbitMult  = 6.0f;  // jog up to Nx the current orbit speed
static constexpr float kSkipRampInSec  = 0.6f;
static constexpr float kSkipRampOutSec = 0.15f;
// Eager recovery: ease out as soon as a valid field returns (a small gap above
// the engage trigger — the most sensitive a stable hysteresis latch allows).
static constexpr float kSkipHyst       = 0.03f;

// Uniform block — mirrors `U` in common.hlsl (std140, 96 bytes).
struct Uniforms {
  float res_x, res_y, extent, bias;
  float wind, n_bands, contrast, flow_phase;
  float nearest_cell, respawn, stream_width, cycle_width;
  float backdrop_dim, stream_alpha, shading_mode, solve_steps;
  float break_dist, explore, spread, rand_seed;
  float step_size, momentum, morph_rate, respawn_arc;
  float good_init, break_turn_cos, stream_spread, _pad2;
  float corners[4];
  float weights[4];
};
static_assert(sizeof(Uniforms) == 144, "Uniforms layout");

struct State {
  gpu::Buffer uniform_buf;
  gpu::Buffer cell_buf;      // PF_CELLS, uploaded once
  gpu::Buffer curve_buf;      // PF_CURVE (resting cycles), uploaded once — respawn seeds
  gpu::Buffer particle_buf[2]; // STATEFUL cycle-solver ring, double-buffered (ping-pong)
  gpu::Buffer good_buf;       // last "good" cycle (respawn source), morphed each frame
  gpu::Buffer status_buf;     // cycle health (PF_ST_*), shared solve<->select
  gpu::Buffer stream_buf;     // streamline segments
  gpu::Buffer cycle_buf;      // limit-cycle segments
  gpu::Texture flow_tex;      // baked velocity field (flow_field output), viewport-sized
  gpu::Buffer  flow_uniform_buf;  // separate uniforms for the tick-time flow bake
  int  flow_w = 0, flow_h = 0;
  bool initialized = false;

  // Stateful solver bookkeeping.
  bool     particles_init = false;  // false until the first respawn seeds the ring
  float    respawn_accum  = 0.0f;   // seconds since the last respawn
  int      particle_cur   = 0;      // which particle_buf holds the latest ring
  uint32_t frame_counter  = 0;      // drives the per-frame random-walk seed

  // --- Tracer mode (CPU) state ---
  bool  tr_init = false;            // tracer seeded yet
  float tr_x = 0.0f, tr_y = 0.0f;   // tracer position
  float tr_hx[PF_TRACE_MAX];        // tracer trajectory history
  float tr_hy[PF_TRACE_MAX];
  int   tr_count = 0;
  int   tr_sink = 0;                // consecutive near-stationary steps (sink detect)
  bool  loop_valid = false;         // a closed loop is currently detected
  float loop_x[PF_PARTICLES];       // the detected loop, resampled to the ring count
  float loop_y[PF_PARTICLES];
  bool  ring_init = false;
  float ring_x[PF_PARTICLES], ring_y[PF_PARTICLES];   // CPU ring positions
  float ring_vx[PF_PARTICLES], ring_vy[PF_PARTICLES]; // CPU ring velocities (real momentum)

  // --- Schema-mirrored params ---
  float eccentricity = 0.2f;   // XY pad x
  float lobedness    = 0.2f;   // XY pad y
  float wind         = 0.0f;   // z (non-potential force)
  float wind_jitter  = 0.0f;   // chaotic, fling-ey wobble of the wind value
  float wind_jitter_speed = 0.5f;
  float bias         = 0.0f;   // shifts the cycle level
  float scale        = 1.0f;   // domain zoom (higher = zoom IN, like shape_fold)
  bool  interpolate  = true;   // blend H across 4 cells vs snap
  int   shading_mode = 0;      // 0 = Bands (height field), 1 = Gradient (flow)
  float bands        = 13.0f;  // backdrop contour bands
  float contrast     = 1.6f;   // backdrop band contrast
  float backdrop_dim = 0.42f;  // backdrop colour strength (muting)
  bool  show_streamlines = true;
  float stream_width = 0.012f;
  float stream_spread = 1.6f;  // seed-grid extent scale (>1 = start outside, flow in)
  float flow_speed   = 0.5f;   // arrow animation rate
  float line_opacity = 0.55f;
  bool  show_limit_cycle = true;
  int   cycle_mode   = 0;      // 0 = Relax (GPU solver), 1 = Tracer (CPU)
  float cycle_width  = 0.02f;
  // --- Tracer / Trace mode (CPU) params ---
  float arc_angle    = 0.0f;   // where on the resting cycle the tracer restarts
  float trace_pull   = 0.05f;  // ring attraction force toward the detected loop
  float trace_step   = 0.02f;  // tracer integration step size (lower = slower)
  float trace_steps  = 4.0f;   // tracer steps advanced per frame
  float trace_eps    = 0.06f;  // loop-close distance (how near to count as closed)
  float solve_steps  = 4.0f;   // Newton relaxation iterations per frame (X)
  float break_dist   = 0.2f;   // max gap between particles before it's a break
  float respawn_time = 2.0f;   // hard re-seed timer (seconds)
  float explore      = 0.3f;   // tangential random-walk amount (back and forth)
  float spread       = 0.5f;   // neighbour-spacing gain (spread the ring out)
  float step_size    = 0.75f;  // [0,1] exp-mapped to PF_STEP_MIN..MAX (force scale)
  float momentum     = 0.6f;   // velocity retention — higher = more wobble
  float morph_rate   = 0.1f;   // good-cycle lerp rate (toward live / resting)
  float respawn_arc  = 1.0f;   // respawn when broken & longest chain shorter than this
  float break_turn   = 0.5f;   // [0,1] doubling-back sensitivity (0=off, 0.5=90°)
  bool  autopilot    = false;
  float ap_speed     = 0.35f;
  float jitter       = 0.0f;    // chaotic, fling-ey orbit of the XY around its base
  float jitter_speed = 0.5f;
  // --- Skip empty: sense the atlas holes and jog the orbit through them ---
  bool  skip_empty   = false;   // master enable for detector + orbit jog
  float skip_thresh  = 0.12f;   // validity floor [0,1]: engage below this

  // --- Internal clocks (advanced in tick) ---
  float flow_phase = 0.0f;
  float orbit      = 0.0f;
  fx::SkipJog jog;                    // skip-empty C2 engagement ramp
  float content    = 1.0f;            // last frame's validity coverage [0,1]
  float eff_x = 0.2f, eff_y = 0.2f;   // effective XY used for rendering
  float eff_wind = 0.0f;              // effective wind (base + jitter) used for rendering

  // Jitter: a weighty/fling-ey chaotic orbit around the base XY. A 2D mass on a
  // spring (stays near base) + swirl (orbits) + chaotically-walking drive freqs.
  float jox = 0, joy = 0, jvx = 0, jvy = 0;   // offset + velocity (momentum)
  float jp1 = 0, jp2 = 0;                      // drive LFO phases
  float jr1 = 1, jr2 = 1, jr1t = 1, jr2t = 1;  // drive rates + walk targets
  float jwalk = 0;                            // rate random-walk accumulator
  uint32_t jrng = 0x9E3779B9u;                // per-instance PRNG for the rate walk

  // Wind jitter: the 1D scalar analogue of the XY jitter (mass on a spring back
  // to 0, no swirl), added to the wind value.
  float wox = 0, wvx = 0;                       // offset + velocity (momentum)
  float wp1 = 0, wp2 = 0;                        // drive LFO phases
  float wr1 = 1, wr2 = 1, wr1t = 1, wr2t = 1;   // drive rates + walk targets
  float wwalk = 0;                              // rate random-walk accumulator
};

static void apply_visibility(bool show_streamlines, bool autopilot,
                             bool show_limit_cycle, int cycle_mode,
                             bool skip_empty) {
  state::setFieldHidden("stream_width", !show_streamlines);
  state::setFieldHidden("stream_spread", !show_streamlines);
  state::setFieldHidden("flow_speed",   !show_streamlines);
  state::setFieldHidden("ap_speed",     !autopilot);
  state::setFieldHidden("skip_thresh",  !skip_empty);

  // Limit-cycle params depend on the selected algorithm.
  bool cyc   = show_limit_cycle;
  bool relax = cyc && cycle_mode == 0;                       // GPU spring solver
  bool ring  = cyc && cycle_mode == 1;                       // Tracer (drawn ring)
  bool trace = cyc && (cycle_mode == 1 || cycle_mode == 2);  // any flow tracer
  state::setFieldHidden("cycle_mode",   !cyc);
  state::setFieldHidden("cycle_width",  !cyc);
  state::setFieldHidden("momentum",     !(relax || ring));
  // Relax-only
  state::setFieldHidden("solve_steps",  !relax);
  state::setFieldHidden("step_size",    !relax);
  state::setFieldHidden("break_dist",   !relax);
  state::setFieldHidden("break_turn",   !relax);
  state::setFieldHidden("respawn_time", !relax);
  state::setFieldHidden("respawn_arc",  !relax);
  state::setFieldHidden("morph_rate",   !relax);
  state::setFieldHidden("explore",      !relax);
  state::setFieldHidden("spread",       !relax);
  // Tracer family
  state::setFieldHidden("arc_angle",    !trace);
  state::setFieldHidden("trace_step",   !trace);
  state::setFieldHidden("trace_steps",  !trace);
  state::setFieldHidden("trace_eps",    !trace);
  state::setFieldHidden("trace_pull",   !ring);   // only the Tracer ring uses pull
}

// Static (self-less) visibility evaluator — pure over state (see crop).
void eval_visibility(int n, const char* pb, const int* off, const int* len, const int* ops) {
  bool show_streamlines = true, autopilot = false, show_limit_cycle = true;
  int cycle_mode = 0;
  bool skip_empty = false;
  for (int i = 0; i < n; i++) {
    if (ops[i] != state::PatchReplace) continue;
    const char* p = pb + off[i]; int l = len[i];
    if      (state::pathIs(p, l, "show_streamlines")) show_streamlines = state::patchFloat(i) != 0.0f;
    else if (state::pathIs(p, l, "autopilot"))        autopilot = state::patchFloat(i) != 0.0f;
    else if (state::pathIs(p, l, "show_limit_cycle")) show_limit_cycle = state::patchFloat(i) != 0.0f;
    else if (state::pathIs(p, l, "cycle_mode"))       cycle_mode = (int)state::patchFloat(i);
    else if (state::pathIs(p, l, "skip_empty"))       skip_empty = state::patchFloat(i) != 0.0f;
  }
  apply_visibility(show_streamlines, autopilot, show_limit_cycle, cycle_mode, skip_empty);
}

static void on_state_ready(void* self);

// Type-shared, compiled once in module_init().
static gpu::ComputePSO s_pso_backdrop;
static gpu::ComputePSO s_pso_stream;
static gpu::ComputePSO s_pso_solve;
static gpu::ComputePSO s_pso_cycle;
static gpu::ComputePSO s_pso_select;
static gpu::ComputePSO s_pso_flow;     // bakes the flow_field velocity texture
static gpu::RenderPSO  s_pso_lines;
static gpu::RenderPSO  s_pso_contour;

void module_init() {
  state::init("source.phase_fold", {1, 0, 2},
    state::Schema()
      // Top-level manual: high-level "what is this / how to use / what to try".
      .helpField("intro",
        "## Phase Fold\n"
        "An emergent **phase-portrait** generator — it draws the flow and limit "
        "cycles of a dynamical system. Drag the XY pad to explore a baked atlas of "
        "fields: **X** sets eccentricity, **Y** sets lobedness. Three layers stack "
        "up: a banded **backdrop** height field, tracing **streamlines**, and a "
        "discovered **limit cycle**.\n\n"
        "**Try:** push **Wind** up until the orbit distorts and finally snaps open "
        "(a SNIC bifurcation — the cycle dies); add a little **Jitter** for organic "
        "wander; flip on **Autopilot** to let the whole portrait drift on its own.")
      // --- Shape axes (the custom XY pad drives eccentricity + lobedness) ---
      .group("shape", "Shape")
        .groupHelp(
          "The **atlas** is a montage of pre-baked dynamical fields. *Eccentricity* "
          "(X) and *Lobedness* (Y) interpolate between neighbouring cells to reshape "
          "the orbit. **Wind** is the non-potential force that warps the cycle and, "
          "pushed far enough, kills it outright (the orbit collapses to a point). "
          "Add **Wind Jitter** for a restless, fling-ey wobble on that force.")
      .floatField("eccentricity", 0.2f, 0.0f, 1.0f, state::PrimaryInput).label("Eccentricity", "Ecc")
      .floatField("lobedness", 0.2f, 0.0f, 1.0f, state::PrimaryInput).label("Lobedness", "Lobes")
      // Wind (z): the non-potential force that distorts the cycle and, past the
      // bifurcation, kills it (the orbit collapses to a fixed point).
      .floatField("wind", 0.0f, -1.0f, 1.0f, state::PrimaryInput).label("Wind", "Wind")
      // Wind jitter: a chaotic, weighty/fling-ey wobble of the wind value (the
      // 1D analogue of the XY jitter) with its own speed.
      .floatField("wind_jitter", 0.0f, 0.0f, 1.0f, state::PrimaryInput).label("Wind Jitter", "WJit")
      .floatField("wind_jitter_speed", 0.5f, 0.0f, 1.0f, state::PrimaryInput).label("Wind Jitter Speed", "WJSpd")
      // --- Domain ---
      .group("domain", "Domain")
      // Bias shifts the cycle level → slides the limit cycle across contours.
      .floatField("bias", 0.0f, -0.6f, 0.6f, state::PrimaryInput).label("Bias", "Bias")
      // Domain zoom. Higher = zoom IN (bigger features); lower zooms out.
      .floatField("scale", 1.0f, 0.1f, 8.0f, state::PrimaryInput).label("Scale", "Scl")
      .boolField("interpolate", true, state::PrimaryInput).label("Interpolate", "Interp")
      // --- Backdrop ---
      .group("backdrop", "Backdrop")
        .groupHelp(
          "The coloured field behind everything — a contoured map of the system's "
          "scalar height. *Shading* switches between the banded height field, the "
          "wind-aware flow gradient, and a set of matplotlib colormaps. *Bands* and "
          "*Contrast* set how the contours read; drop *Backdrop Dim* to mute it so "
          "the streamlines and cycle stand out.")
      // Shading: height-field Bands, the wind-aware flow Gradient, or a banded
      // matplotlib colormap of the height field.
      .selectField("shading_mode", 0, state::PrimaryInput,
                   {{"Bands", 0}, {"Gradient", 1}, {"Magma", 2}, {"Inferno", 3},
                    {"Viridis", 4}, {"Plasma", 5}, {"Turbo", 6}}, /*wrap=*/true).label("Shading", "Shade")
      .floatField("bands", 13.0f, 2.0f, 24.0f, state::PrimaryInput).label("Bands", "Bands")
      .floatField("contrast", 1.6f, 0.4f, 4.0f, state::PrimaryInput).label("Contrast", "Cntr")
      .floatField("backdrop_dim", 0.42f, 0.0f, 1.0f, state::PrimaryInput).label("Backdrop Dim", "Dim")
      // --- Streamlines (toggleable stage) ---
      .group("streamlines", "Streamlines")
        .groupHelp(
          "A grid of glowing lines traced through the flow field — the moving "
          "current of the portrait. *Seed Spread* above 1 starts lines outside the "
          "frame so they flow inward and keep it dense. *Flow Speed* animates the "
          "glow travelling down each line (0 freezes it); *Line Width* and *Opacity* "
          "set their weight.")
      .boolField("show_streamlines", true, state::PrimaryInput).label("Streamlines", "Stream")
      .floatField("stream_width", 0.012f, 0.002f, 0.05f, state::PrimaryInput).label("Line Width", "Width")
      // Seed-grid spread: >1 seeds streamlines outside the window so they flow in
      // and keep the frame dense (the flow converges toward the centre).
      .floatField("stream_spread", 1.6f, 0.5f, 4.0f, state::PrimaryInput).label("Seed Spread", "Sprd")
      .floatField("flow_speed", 0.5f, 0.0f, 1.0f, state::PrimaryInput).label("Flow Speed", "Flow")
      .floatField("line_opacity", 0.55f, 0.0f, 1.0f, state::PrimaryInput).label("Opacity", "Opac")
      // --- Limit-cycle tracer (toggleable stage) ---
      .group("limit_cycle", "Limit Cycle")
        .groupHelp(
          "Discovers and draws the system's **limit cycle** — the closed orbit the "
          "flow settles onto. *Algorithm* picks how it's found: **Relax** springs a "
          "particle ring onto the cycle, **Tracer** follows the flow until it closes "
          "a loop, **Trace** shows the raw trajectory, **Contour** draws the level "
          "set directly. Most knobs below only apply to the active algorithm — push "
          "**Wind** until the cycle breaks open and watch it fail to close.")
      .boolField("show_limit_cycle", true, state::PrimaryInput).label("Limit Cycle", "Cycle")
      // Algorithm: Relax = GPU spring-solver ring; Tracer = a CPU flow tracer that
      // detects a closed loop and pulls a momentum ring onto it; Trace = the same
      // tracer but draws its raw trajectory (debug viz); Contour = draw the height
      // field's zero level-set directly (no particles).
      .selectField("cycle_mode", 0, state::PrimaryInput,
                   {{"Relax", 0}, {"Tracer", 1}, {"Trace", 2}, {"Contour", 3}}).label("Algorithm", "Algo")
      .floatField("cycle_width", 0.02f, 0.004f, 0.06f, state::PrimaryInput).label("Cycle Width", "Width")
      // Tracer: where on the resting cycle the tracer restarts (arc fraction).
      .floatField("arc_angle", 0.0f, 0.0f, 1.0f, state::PrimaryInput).label("Restart Arc", "Arc")
      // Tracer ring: how hard the ring accelerates toward the detected loop.
      .floatField("trace_pull", 0.05f, 0.0f, 0.4f, state::PrimaryInput).label("Ring Pull", "Pull")
      // Tracer integration step size (how far the tracer moves per step).
      .floatField("trace_step", 0.02f, 0.002f, 0.06f, state::PrimaryInput).label("Trace Step", "TrStep")
      // Tracer steps advanced per frame (how fast it traces).
      .floatField("trace_steps", 4.0f, 1.0f, 16.0f, state::PrimaryInput).label("Trace Rate", "TrRate")
      // Loop-close distance — how near the tracer must return to count as closed.
      .floatField("trace_eps", 0.06f, 0.01f, 0.2f, state::PrimaryInput).label("Close Dist", "Close")
      // Newton relaxation steps per frame — how hard the ring solves onto the cycle.
      .floatField("solve_steps", 4.0f, 1.0f, 16.0f, state::PrimaryInput).label("Solve Steps", "Solve")
      // How far each relaxation step pushes (scales the per-step force).
      // Exponential: the [0,1] slider maps to PF_STEP_MIN..PF_STEP_MAX
      // (~0.001 .. 0.5) so the small end has fine control.
      .floatField("step_size", 0.75f, 0.0f, 1.0f, state::PrimaryInput).label("Step Size", "Step")
      // Velocity retention — particles carry momentum, so the ring wobbles
      // around the cycle (underdamped). 0 = no wobble, ~0.9 = very springy.
      .floatField("momentum", 0.6f, 0.0f, 0.95f, state::PrimaryInput).label("Momentum", "Mom")
      // Good-cycle morph rate — how fast the remembered cycle tracks the live
      // one (when closed) or decays back to the resting cycle (when broken).
      .floatField("morph_rate", 0.1f, 0.0f, 1.0f, state::PrimaryInput).label("Morph Rate", "Morph")
      // Respawn when the cycle is broken and its longest chain's arc length is
      // shorter than this (the discovery attempt has clearly failed).
      .floatField("respawn_arc", 1.0f, 0.0f, 4.0f, state::PrimaryInput).label("Respawn Arc", "RspArc")
      // Doubling-back sensitivity: break the cycle where the polyline reverses.
      // 0 = off; 0.5 = break turns sharper than 90°; 1 = very aggressive.
      .floatField("break_turn", 0.5f, 0.0f, 1.0f, state::PrimaryInput).label("Break Turn", "Turn")
      // Break sensitivity: the max gap between adjacent particles before the
      // cycle is considered broken there (and that segment is dropped).
      .floatField("break_dist", 0.2f, 0.05f, 0.6f, state::PrimaryInput).label("Break Gap", "Gap")
      // Hard re-seed timer — periodically respawn the ring on the resting cycle.
      .floatField("respawn_time", 2.0f, 0.1f, 10.0f, state::PrimaryInput).label("Respawn Time", "RspTim")
      // Exploration: how far particles random-walk back and forth along the
      // contour each frame (jitters a true cycle, explores otherwise).
      .floatField("explore", 0.3f, 0.0f, 1.0f, state::PrimaryInput).label("Explore", "Expl")
      // Spread: neighbour-spacing gain that keeps the ring evenly distributed.
      .floatField("spread", 0.5f, 0.0f, 1.0f, state::PrimaryInput).label("Spread", "Sprd")
      // --- Autopilot (non-destructive XY override + broadcast) ---
      .group("autopilot", "Autopilot")
        .groupHelp(
          "Spirals the shape's XY position on its own, **without touching** your "
          "Eccentricity/Lobedness inputs (it broadcasts the live position instead). "
          "*Orbit Speed* sets the drift rate. **Jitter** layers a weighty, fling-ey "
          "chaotic wobble over whatever XY is active (user or autopilot); push its "
          "*Speed* up and the portrait flickers across cells frame to frame.")
      .boolField("autopilot", false, state::PrimaryInput).label("Autopilot", "Auto")
      .floatField("ap_speed", 0.35f, 0.0f, 1.0f, state::PrimaryInput).label("Orbit Speed", "Spd")
      // Jitter: chaotically orbit the (user or autopilot) XY — weighty, fling-ey.
      .floatField("jitter", 0.0f, 0.0f, 1.0f, state::PrimaryInput).label("Jitter", "Jit")
      .floatField("jitter_speed", 0.5f, 0.0f, 1.0f, state::PrimaryInput).label("Jitter Speed", "JitSpd")
      // Broadcast: effective XY so the custom editor shows the live position.
      .floatField("autopilot_x", 0.2f, 0.0f, 1.0f, state::SecondaryOutput)
      .floatField("autopilot_y", 0.2f, 0.0f, 1.0f, state::SecondaryOutput)
      // --- Skip empty: glide the orbit through the atlas holes ---
      .group("skip", "Skip Empty")
        .groupHelp(
          "The atlas has invalid **holes** that render as a flat dark fill. Turn "
          "this on and — while **Autopilot** is running — the orbit **speeds up "
          "smoothly** (on a C2 curve) as it nears a hole, so it glides through the "
          "dead patch instead of parking in it, then eases back to normal once a "
          "live field returns. *Sensitivity* sets how close to a hole it reacts. "
          "(The field is position-only, so this needs Autopilot on to have "
          "anything to move.)")
      .boolField("skip_empty", false, state::PrimaryInput).label("Skip Empty", "Skip")
      .floatField("skip_thresh", 0.12f, 0.0f, 0.5f, state::PrimaryInput).label("Sensitivity", "Sens")
      // Broadcast: the live engagement [0,1] so an editor can show when it fires.
      .floatField("skip_active", 0.0f, 0.0f, 1.0f, state::SecondaryOutput)
      // --- I/O: pure generator (no input) ---
      .textureField("tex_out", state::PrimaryOutput)
      // Flow field: the induced vector field, baked to a velocity texture so a
      // downstream swarm / flow modifier can consume it (separates the field
      // from its rendering). Baked only when wired.
      .flowField(state::PrimaryOutput)
        .capability(state::Capability::Generator)
    );

  if (gpu::Device::backend() == gpu::Backend::None) return;

  state::registerShaderSPV("phase_fold_backdrop", BACKDROP_SPV, BACKDROP_SPV_SIZE);
  state::registerShaderSPV("phase_fold_stream",   STREAM_SPV,   STREAM_SPV_SIZE);
  state::registerShaderSPV("phase_fold_solve",    SOLVE_SPV,    SOLVE_SPV_SIZE);
  state::registerShaderSPV("phase_fold_cycle",    CYCLE_SPV,    CYCLE_SPV_SIZE);
  state::registerShaderSPV("phase_fold_select",   SELECT_SPV,   SELECT_SPV_SIZE);
  // The flow bake writes an rgba16float velocity texture. The shader's
  // RWTexture2D<float4> defaults to rgba8unorm in SPIR-V, so register with the
  // storage-format override (as flash_particles does for its motion prefill)
  // to match the RGBA16F PSO binding — otherwise PSO creation fails the layout
  // format check on Dawn.
  state::registerShaderSPV("phase_fold_flow",     FLOW_SPV,     FLOW_SPV_SIZE,
                           "rgba16float", "write");
  state::registerShaderSPV("phase_fold_line_vs",  LINE_VS_SPV,  LINE_VS_SPV_SIZE);
  state::registerShaderSPV("phase_fold_line_fs",  LINE_FS_SPV,  LINE_FS_SPV_SIZE);
  state::registerShaderSPV("phase_fold_contour_vs", CONTOUR_VS_SPV, CONTOUR_VS_SPV_SIZE);
  state::registerShaderSPV("phase_fold_contour_fs", CONTOUR_FS_SPV, CONTOUR_FS_SPV_SIZE);

  auto cs_backdrop = gpu::Device::createShaderModuleByName("phase_fold_backdrop");
  auto cs_stream   = gpu::Device::createShaderModuleByName("phase_fold_stream");
  auto cs_solve    = gpu::Device::createShaderModuleByName("phase_fold_solve");
  auto cs_cycle    = gpu::Device::createShaderModuleByName("phase_fold_cycle");
  auto cs_select   = gpu::Device::createShaderModuleByName("phase_fold_select");
  auto cs_flow     = gpu::Device::createShaderModuleByName("phase_fold_flow");
  auto vs_lines    = gpu::Device::createShaderModuleByName("phase_fold_line_vs");
  auto fs_lines    = gpu::Device::createShaderModuleByName("phase_fold_line_fs");
  auto vs_contour  = gpu::Device::createShaderModuleByName("phase_fold_contour_vs");
  auto fs_contour  = gpu::Device::createShaderModuleByName("phase_fold_contour_fs");
  if (!cs_backdrop || !cs_stream || !cs_solve || !cs_cycle || !cs_select ||
      !cs_flow || !vs_lines || !fs_lines || !vs_contour || !fs_contour) return;

  s_pso_backdrop = gpu::Device::createComputePSO(cs_backdrop, "main", gpu::Bindings()
      .uniform(0)
      .storage(1)                                    // cells (read)
      .storageTex2d(2));  // tex_out

  s_pso_stream = gpu::Device::createComputePSO(cs_stream, "main", gpu::Bindings()
      .uniform(0)
      .storage(1)        // cells (read)
      .storageRW(2));    // stream segments (write)

  // Stateful solver: relax the persistent particle ring onto the cycle.
  // Double-buffered — reads last frame's ring, writes the next (race-free
  // neighbour reads for the spacing term).
  s_pso_solve = gpu::Device::createComputePSO(cs_solve, "main", gpu::Bindings()
      .uniform(0)
      .storage(1)        // cells (read)
      .storageRW(2)      // particles_next (write)
      .storage(3)        // curve / resting-cycle respawn seeds (read)
      .storage(4)        // particles_prev (read, last frame)
      .storage(5)        // good ring (respawn source, read)
      .storage(6));      // status (read PF_ST_RESPAWN)

  // Build + break-detect: relaxed particles → cycle segments.
  s_pso_cycle = gpu::Device::createComputePSO(cs_cycle, "main", gpu::Bindings()
      .uniform(0)
      .storage(1)        // cells (read, for the gradient-flip break check)
      .storageRW(2)      // cycle segments (write)
      .storage(3));      // particles (read)

  // Longest-run cull + cycle health + good-cycle morph (single-thread).
  s_pso_select = gpu::Device::createComputePSO(cs_select, "main", gpu::Bindings()
      .uniform(0)
      .storageRW(2)      // cycle segments (cull in place)
      .storageRW(3)      // good ring (morphed)
      .storageRW(4)      // status (written)
      .storage(5)        // live ring (read)
      .storage(6));      // curve / resting cycle (read)

  // Flow bake: evaluate pf_velocity per texel → rgba16float velocity texture.
  s_pso_flow = gpu::Device::createComputePSO(cs_flow, "main", gpu::Bindings()
      .uniform(0)
      .storage(1)                                      // cells (read)
      .storageTex2d(2, gpu::TextureFormat::RGBA16F));  // flow velocity (write)

  s_pso_lines = gpu::Device::createInstancedRenderPSO(
      vs_lines, "main", fs_lines, "main", gpu::TextureFormat::Surface,
      gpu::Bindings()
          .uniform(0)    // shared uniforms (vertex)
          .storage(1),   // segment buffer (vertex)
      gpu::Device::BlendMode::AlphaOver);

  // Contour mode: full-screen pass that draws the height field's zero level-set.
  s_pso_contour = gpu::Device::createInstancedRenderPSO(
      vs_contour, "main", fs_contour, "main", gpu::TextureFormat::Surface,
      gpu::Bindings()
          .uniform(0)    // shared uniforms
          .storage(1),   // cells (fragment field eval)
      gpu::Device::BlendMode::AlphaOver);

  state::log("phase_fold: module initialized");
}

void* create() {
  auto* s = new State();
  s->uniform_buf = gpu::Device::createBuffer(sizeof(Uniforms), gpu::BufferUsage::Uniform);
  s->cell_buf    = gpu::Device::createBuffer(kCellFloats * sizeof(float), gpu::BufferUsage::Storage);
  s->curve_buf   = gpu::Device::createBuffer(kCurveFloats * sizeof(float), gpu::BufferUsage::Storage);
  s->particle_buf[0] = gpu::Device::createBuffer(kParticleFloats * sizeof(float), gpu::BufferUsage::Storage);
  s->particle_buf[1] = gpu::Device::createBuffer(kParticleFloats * sizeof(float), gpu::BufferUsage::Storage);
  s->good_buf    = gpu::Device::createBuffer(kParticleFloats * sizeof(float), gpu::BufferUsage::Storage);
  s->status_buf  = gpu::Device::createBuffer(4 * sizeof(float), gpu::BufferUsage::Storage);
  s->stream_buf  = gpu::Device::createBuffer(kStreamSegs * kSegFloats * sizeof(float), gpu::BufferUsage::Storage);
  s->cycle_buf   = gpu::Device::createBuffer(kCycleSegs * kSegFloats * sizeof(float), gpu::BufferUsage::Storage);
  s->flow_uniform_buf = gpu::Device::createBuffer(sizeof(Uniforms), gpu::BufferUsage::Uniform);
  return s;
}

void destroy(void* self) {
  auto* s = static_cast<State*>(self);
  if (!s) return;
  s->uniform_buf.release();
  s->cell_buf.release();
  s->curve_buf.release();
  s->particle_buf[0].release();
  s->particle_buf[1].release();
  s->good_buf.release();
  s->status_buf.release();
  s->stream_buf.release();
  s->cycle_buf.release();
  s->flow_tex.release();
  s->flow_uniform_buf.release();
  delete s;
}

void init(void* self) {
  auto* s = static_cast<State*>(self);
  if (!s) return;
  s->initialized = false;
  if (!s_pso_backdrop.valid() || !s_pso_stream.valid() || !s_pso_solve.valid() ||
      !s_pso_cycle.valid() || !s_pso_select.valid() || !s_pso_flow.valid() ||
      !s_pso_lines.valid() || !s_pso_contour.valid()) return;
  if (!s->uniform_buf.valid() || !s->cell_buf.valid() || !s->curve_buf.valid() ||
      !s->particle_buf[0].valid() || !s->particle_buf[1].valid() ||
      !s->good_buf.valid() || !s->status_buf.valid() ||
      !s->stream_buf.valid() || !s->cycle_buf.valid()) return;
  // Upload the (immutable) atlas cell + resting-cycle buffers once. The particle
  // and good rings are left uninitialized — the first frame seeds them. Clear
  // the status buffer so the first solve doesn't see a garbage respawn flag.
  s->cell_buf.write(PF_CELLS, kCellFloats);
  s->curve_buf.write(PF_CURVE, kCurveFloats);
  float zero_status[4] = {0, 0, 0, 0};
  s->status_buf.write(zero_status, 4);
  s->particles_init = false;
  s->respawn_accum = 0.0f;
  s->particle_cur = 0;
  s->initialized = true;
  state::setOnStateReady(&on_state_ready);
}

static void on_state_ready(void* self) {
  auto* s = static_cast<State*>(self);
  if (!s) return;
  apply_visibility(s->show_streamlines, s->autopilot, s->show_limit_cycle, s->cycle_mode, s->skip_empty);
}

static inline float clampf(float v, float lo, float hi) { return v < lo ? lo : (v > hi ? hi : v); }
static inline int   clampi(int v, int lo, int hi) { return v < lo ? lo : (v > hi ? hi : v); }

// Per-instance uniform random in [0,1) (LCG) — drives the jitter rate walk.
static inline float rng_unit(State* s) {
  s->jrng = s->jrng * 1664525u + 1013904223u;
  return (float)((s->jrng >> 8) & 0xFFFFFFu) / (float)0x1000000;
}

// Port of app.js corners(): bilinear (smoothstep) 4-cell blend or nearest snap,
// re-weighted by per-cell validity. Also returns the nearest cell (for the
// limit-cycle seed) and whether any valid cell was hit.
static void compute_corners(const State* s, float sx, float sy,
                            float corners[4], float weights[4], int& nearest,
                            float* validity = nullptr) {
  const int G = PF_GRID;
  float fx = clampf(sx, 0.0f, 1.0f) * (G - 1);
  float fy = clampf(sy, 0.0f, 1.0f) * (G - 1);
  int x0, x1, y0, y1; float tx, ty;
  if (s->interpolate) {
    x0 = clampi((int)std::floor(fx), 0, G - 1); x1 = std::min(x0 + 1, G - 1); tx = fx - x0;
    y0 = clampi((int)std::floor(fy), 0, G - 1); y1 = std::min(y0 + 1, G - 1); ty = fy - y0;
    tx = tx * tx * (3.0f - 2.0f * tx);   // smoothstep → C¹ across grid lines
    ty = ty * ty * (3.0f - 2.0f * ty);
  } else {
    x0 = x1 = clampi((int)std::lround(fx), 0, G - 1);
    y0 = y1 = clampi((int)std::lround(fy), 0, G - 1);
    tx = ty = 0.0f;
  }
  int idx[4] = { y0 * G + x0, y0 * G + x1, y1 * G + x0, y1 * G + x1 };
  float w[4] = { (1 - tx) * (1 - ty), tx * (1 - ty), (1 - tx) * ty, tx * ty };
  float sum = 0.0f;
  for (int i = 0; i < 4; i++) { w[i] *= PF_VALID[idx[i]]; sum += w[i]; }
  // `sum` (pre-normalization) is the fraction of the bilinear weight landing on
  // VALID cells — 1 fully inside the atlas, 0 in a hole. It's the skip-empty
  // content signal (a hole renders as a flat dark fill).
  if (validity) *validity = clampf(sum, 0.0f, 1.0f);
  if (sum > 1e-4f) for (int i = 0; i < 4; i++) w[i] /= sum;
  for (int i = 0; i < 4; i++) { corners[i] = (float)idx[i]; weights[i] = w[i]; }
  int c = clampi((int)std::lround(sx * (G - 1)), 0, G - 1);
  int r = clampi((int)std::lround(sy * (G - 1)), 0, G - 1);
  nearest = r * G + c;
}

static inline void orbit_xy(float orbit, float& ox, float& oy) {
  ox = clampf(0.5f + kApA * std::cos(orbit) + kApB * std::cos(orbit * kApW2 + kApPhi), 0.01f, 0.99f);
  oy = clampf(0.5f + kApA * std::sin(orbit) + kApB * std::sin(orbit * kApW2 + kApPhi), 0.01f, 0.99f);
}

// Bake the induced velocity field into flow_tex for downstream consumers. This
// lives in tick() (NOT render()) so it runs every frame even when the visual
// render is skipped — at opacity 0 the executor calls tick but not render, and a
// patch wiring phase_fold purely as a flow source still needs fresh data. Uses
// its own minimal uniform buffer (pf_velocity only reads res/extent/bias/wind/
// corners/weights) so it doesn't depend on render()'s full per-frame uniform.
static void bake_flow_field(State* s) {
  if (!state::isOutputConnected("flow_field") || !s_pso_flow.valid()) return;
  int vw = host::viewportWidth();
  int vh = host::viewportHeight();
  if (vw <= 0 || vh <= 0) return;

  if (!s->flow_tex.valid() || s->flow_w != vw || s->flow_h != vh) {
    s->flow_tex.release();   // free the old texture on viewport resize (no leak)
    s->flow_tex = gpu::Device::createTexture(vw, vh, gpu::TextureFormat::RGBA16F);
    s->flow_w = vw;
    s->flow_h = vh;
    if (!s->flow_tex.valid()) return;
    state::setGpuTexture("flow_field/velocity", s->flow_tex.id);  // on (re)alloc only
  }

  int nearest = 0;
  Uniforms u = {};
  u.res_x = (float)vw;
  u.res_y = (float)vh;
  u.extent = PF_EXTENT / (s->scale > 1e-2f ? s->scale : 1e-2f);
  u.bias = s->bias;
  u.wind = s->eff_wind;
  compute_corners(s, s->eff_x, s->eff_y, u.corners, u.weights, nearest);
  s->flow_uniform_buf.writeOne(u);

  auto cp = gpu::ComputePass::begin();
  cp.setPSO(s_pso_flow);
  cp.setBuffer(s->flow_uniform_buf, 0);
  cp.setBuffer(s->cell_buf, 1);
  cp.setTexture(s->flow_tex, 2, 1);  // access 1 = storage write
  cp.dispatch((vw + 7) / 8, (vh + 7) / 8);
  cp.end();
  gpu::Device::submit();
}

void tick(void* self, double dt) {
  auto* s = static_cast<State*>(self);
  if (!s) return;
  float fdt = (float)dt;

  // Arrows flow down the streamlines (frozen at flow_speed = 0). Quadratic map
  // → fine control at the low end, up to ~30 cycles/s at the top (fast flicker).
  s->flow_phase += fdt * s->flow_speed * s->flow_speed * 30.0f;
  s->flow_phase -= std::floor(s->flow_phase);

  // Hard re-seed timer for the stateful cycle solver.
  s->respawn_accum += fdt;

  // Skip-empty: engage the C2 ramp on last frame's validity coverage. The only
  // lever on the (XY-only) field is the autopilot orbit, so this speeds the orbit
  // up to glide through a hole; with autopilot off it's inert (we never mutate the
  // user's XY). `content` lags a frame — fine (the ramp is smooth + hysteretic).
  float skip_e = 0.0f;
  if (s->skip_empty) {
    float lo = s->skip_thresh;
    float hi = std::min(lo + kSkipHyst, 0.95f);
    skip_e = s->jog.update(s->content, lo, hi, kSkipRampInSec, kSkipRampOutSec, fdt);
  } else {
    s->jog = fx::SkipJog{};        // fully reset when disabled (no residual jog)
  }

  // Base XY: autopilot epicycle, or the user's pad position.
  float base_x, base_y;
  if (s->autopilot) {
    float ap_actual = 0.05f + s->ap_speed * s->ap_speed * (1.6f - 0.05f);
    float orbit_jog = skip_e * ap_actual * kSkipOrbitMult;  // accelerate through holes
    s->orbit -= fdt * (ap_actual + orbit_jog);  // clockwise drift (+ skip jog)
    orbit_xy(s->orbit, base_x, base_y);
  } else {
    base_x = s->eccentricity;
    base_y = s->lobedness;
  }

  // Jitter: a chaotic, weighty/fling-ey orbit of the effective XY around the
  // base. A 2D mass (jox/joy with momentum jvx/jvy) on a spring back to base
  // plus a perpendicular swirl, driven by two LFOs whose frequencies random-walk
  // (organic, non-repeating). Drive amplitude scales with `jitter`, so it settles
  // back to the base as jitter → 0. Light damping = it overshoots and flings.
  // Quadratic response: the slider's low end is gentle, the top still reaches
  // full throw (0.5 → 0.25 of the old linear amount).
  float jraw = clampf(s->jitter, 0.0f, 1.0f);
  float amt = jraw * jraw;
  float flx = 0.0f, fly = 0.0f;
  {
    float spd = clampf(s->jitter_speed, 0.0f, 1.0f);
    // The orbit's OWN frequency scales hard with speed: a slow weighty orbit at
    // the bottom, a fast tight wobble near the top (k clamped for explicit-
    // integration stability). The drive scales too so the chaos keeps up.
    float k = 15.0f + spd * spd * 1500.0f;          // ω ≈ 3.9 .. 39 rad/s
    float swirl = 8.0f + spd * 30.0f;
    float drvspd = 0.5f + spd * spd * 10.0f;
    float drive = 1.1f * (1.0f + spd * 5.0f);
    float damp = std::exp(-fdt * (0.8f + spd * 3.0f));
    // random-walk the two drive rates (style guide §4.2)
    s->jwalk += fdt * 0.6f * drvspd;
    if (s->jwalk >= 1.0f) {
      s->jwalk -= std::floor(s->jwalk);
      s->jr1t = 0.5f + rng_unit(s) * 1.8f;
      s->jr2t = 0.5f + rng_unit(s) * 1.8f;
    }
    float sm = 1.0f - std::exp(-fdt / 0.3f);
    s->jr1 += (s->jr1t - s->jr1) * sm;
    s->jr2 += (s->jr2t - s->jr2) * sm;
    s->jp1 += fdt * s->jr1 * drvspd;
    s->jp2 += fdt * s->jr2 * drvspd;
    float tau = 2.0f * kPi;
    float dfx = std::sin(tau * s->jp1) + 0.6f * std::sin(tau * s->jp2 * 1.7f + 1.0f);
    float dfy = std::cos(tau * s->jp1 * 1.3f) + 0.6f * std::sin(tau * s->jp2);
    float ax = -k * s->jox - swirl * s->joy + dfx * drive * amt;
    float ay = -k * s->joy + swirl * s->jox + dfy * drive * amt;
    s->jvx = (s->jvx + ax * fdt) * damp;
    s->jvy = (s->jvy + ay * fdt) * damp;
    s->jox += s->jvx * fdt;
    s->joy += s->jvy * fdt;
    // soft radius cap so the orbit stays on the pad
    float rmax = amt * 0.45f;
    float r = std::sqrt(s->jox * s->jox + s->joy * s->joy);
    if (r > rmax && r > 1e-6f) { float f = rmax / r; s->jox *= f; s->joy *= f; s->jvx *= f; s->jvy *= f; }
    // Framerate flicker: a per-frame random kick (NOT through the spring, so it
    // doesn't get low-passed) that ramps in at the top of the speed range — at
    // max speed the XY flickers across cells every frame.
    float ft = clampf((spd - 0.55f) / 0.45f, 0.0f, 1.0f);
    float flick = ft * ft * (3.0f - 2.0f * ft) * amt * 0.5f;   // smoothstep
    flx = (rng_unit(s) * 2.0f - 1.0f) * flick;
    fly = (rng_unit(s) * 2.0f - 1.0f) * flick;
  }

  s->eff_x = clampf(base_x + s->jox + flx, 0.0f, 1.0f);
  s->eff_y = clampf(base_y + s->joy + fly, 0.0f, 1.0f);

  // Wind jitter — the 1D scalar analogue of the XY orbit above (no swirl). Same
  // quadratic response, chaotic LFO drive, momentum + light damping (overshoot)
  // and a top-of-range framerate flicker. Added to the base wind.
  {
    float wjraw = clampf(s->wind_jitter, 0.0f, 1.0f);
    float wamt = wjraw * wjraw;
    float spd = clampf(s->wind_jitter_speed, 0.0f, 1.0f);
    float k = 15.0f + spd * spd * 1500.0f;
    float drvspd = 0.5f + spd * spd * 10.0f;
    float drive = 1.1f * (1.0f + spd * 5.0f);
    float damp = std::exp(-fdt * (0.8f + spd * 3.0f));
    s->wwalk += fdt * 0.6f * drvspd;
    if (s->wwalk >= 1.0f) {
      s->wwalk -= std::floor(s->wwalk);
      s->wr1t = 0.5f + rng_unit(s) * 1.8f;
      s->wr2t = 0.5f + rng_unit(s) * 1.8f;
    }
    float sm = 1.0f - std::exp(-fdt / 0.3f);
    s->wr1 += (s->wr1t - s->wr1) * sm;
    s->wr2 += (s->wr2t - s->wr2) * sm;
    s->wp1 += fdt * s->wr1 * drvspd;
    s->wp2 += fdt * s->wr2 * drvspd;
    float tau = 2.0f * kPi;
    float df = std::sin(tau * s->wp1) + 0.6f * std::sin(tau * s->wp2 * 1.7f + 1.0f);
    float ax = -k * s->wox + df * drive * wamt;
    s->wvx = (s->wvx + ax * fdt) * damp;
    s->wox += s->wvx * fdt;
    // soft cap: wind is bipolar [-1,1], so allow up to a near-full-swing wobble.
    float rmax = wamt * 0.9f;
    float r = std::fabs(s->wox);
    if (r > rmax && r > 1e-6f) { float f = rmax / r; s->wox *= f; s->wvx *= f; }
    float ft = clampf((spd - 0.55f) / 0.45f, 0.0f, 1.0f);
    float flick = ft * ft * (3.0f - 2.0f * ft) * wamt * 0.6f;   // smoothstep
    float fwj = (rng_unit(s) * 2.0f - 1.0f) * flick;
    s->eff_wind = clampf(s->wind + s->wox + fwj, -1.0f, 1.0f);
  }

  auto vx = val::number(s->eff_x);
  state::setValPath("autopilot_x", vx);
  val::release(vx);
  auto vy = val::number(s->eff_y);
  state::setValPath("autopilot_y", vy);
  val::release(vy);

  // Sample the validity coverage at the (new) effective XY for next frame's
  // skip-empty detector, and broadcast the live engagement. Computed here (not in
  // render) so it stays fresh even when render() is skipped at opacity 0.
  if (s->skip_empty) {
    float cn[4], cw[4]; int cnear = 0; float validity = 1.0f;
    compute_corners(s, s->eff_x, s->eff_y, cn, cw, cnear, &validity);
    s->content = validity;
  } else {
    s->content = 1.0f;
  }
  auto vsk = val::number(skip_e);
  state::setValPath("skip_active", vsk);
  val::release(vsk);

  // Produce the flow field every frame — even when render() is skipped (opacity
  // 0), so downstream consumers always see fresh data.
  bake_flow_field(s);
}


void on_state_patched(void* self, int n, const char* pb, const int* off,
                      const int* len, const int* ops) {
  auto* s = static_cast<State*>(self);
  if (!s) return;
  bool vis_changed = false;
  for (int i = 0; i < n; i++) {
    if (ops[i] != state::PatchReplace) continue;
    const char* path = pb + off[i];
    int plen = len[i];
    if      (state::pathIs(path, plen, "eccentricity"))   s->eccentricity = state::patchFloat(i);
    else if (state::pathIs(path, plen, "lobedness"))      s->lobedness = state::patchFloat(i);
    else if (state::pathIs(path, plen, "wind"))           s->wind = state::patchFloat(i);
    else if (state::pathIs(path, plen, "wind_jitter"))    s->wind_jitter = state::patchFloat(i);
    else if (state::pathIs(path, plen, "wind_jitter_speed")) s->wind_jitter_speed = state::patchFloat(i);
    else if (state::pathIs(path, plen, "bias"))           s->bias = state::patchFloat(i);
    else if (state::pathIs(path, plen, "scale"))          s->scale = state::patchFloat(i);
    else if (state::pathIs(path, plen, "interpolate"))    s->interpolate = state::patchFloat(i) != 0.0f;
    else if (state::pathIs(path, plen, "shading_mode"))   s->shading_mode = (int)state::patchFloat(i);
    else if (state::pathIs(path, plen, "bands"))          s->bands = state::patchFloat(i);
    else if (state::pathIs(path, plen, "contrast"))       s->contrast = state::patchFloat(i);
    else if (state::pathIs(path, plen, "backdrop_dim"))   s->backdrop_dim = state::patchFloat(i);
    else if (state::pathIs(path, plen, "show_streamlines")) { bool v = state::patchFloat(i) != 0.0f; if (v != s->show_streamlines) { s->show_streamlines = v; vis_changed = true; } }
    else if (state::pathIs(path, plen, "stream_width"))   s->stream_width = state::patchFloat(i);
    else if (state::pathIs(path, plen, "stream_spread"))  s->stream_spread = state::patchFloat(i);
    else if (state::pathIs(path, plen, "flow_speed"))     s->flow_speed = state::patchFloat(i);
    else if (state::pathIs(path, plen, "line_opacity"))   s->line_opacity = state::patchFloat(i);
    else if (state::pathIs(path, plen, "show_limit_cycle")) { bool v = state::patchFloat(i) != 0.0f; if (v != s->show_limit_cycle) { s->show_limit_cycle = v; vis_changed = true; } }
    else if (state::pathIs(path, plen, "cycle_mode"))     { int v = (int)state::patchFloat(i); if (v != s->cycle_mode) { s->cycle_mode = v; s->particles_init = false; s->tr_init = false; s->ring_init = false; s->loop_valid = false; vis_changed = true; } }
    else if (state::pathIs(path, plen, "cycle_width"))    s->cycle_width = state::patchFloat(i);
    else if (state::pathIs(path, plen, "arc_angle"))      s->arc_angle = state::patchFloat(i);
    else if (state::pathIs(path, plen, "trace_pull"))     s->trace_pull = state::patchFloat(i);
    else if (state::pathIs(path, plen, "trace_step"))     s->trace_step = state::patchFloat(i);
    else if (state::pathIs(path, plen, "trace_steps"))    s->trace_steps = state::patchFloat(i);
    else if (state::pathIs(path, plen, "trace_eps"))      s->trace_eps = state::patchFloat(i);
    else if (state::pathIs(path, plen, "solve_steps"))    s->solve_steps = state::patchFloat(i);
    else if (state::pathIs(path, plen, "break_dist"))     s->break_dist = state::patchFloat(i);
    else if (state::pathIs(path, plen, "respawn_time"))   s->respawn_time = state::patchFloat(i);
    else if (state::pathIs(path, plen, "explore"))        s->explore = state::patchFloat(i);
    else if (state::pathIs(path, plen, "spread"))         s->spread = state::patchFloat(i);
    else if (state::pathIs(path, plen, "step_size"))      s->step_size = state::patchFloat(i);
    else if (state::pathIs(path, plen, "momentum"))       s->momentum = state::patchFloat(i);
    else if (state::pathIs(path, plen, "morph_rate"))     s->morph_rate = state::patchFloat(i);
    else if (state::pathIs(path, plen, "respawn_arc"))    s->respawn_arc = state::patchFloat(i);
    else if (state::pathIs(path, plen, "break_turn"))     s->break_turn = state::patchFloat(i);
    else if (state::pathIs(path, plen, "autopilot"))      { bool v = state::patchFloat(i) != 0.0f; if (v != s->autopilot) { s->autopilot = v; vis_changed = true; } }
    else if (state::pathIs(path, plen, "ap_speed"))       s->ap_speed = state::patchFloat(i);
    else if (state::pathIs(path, plen, "jitter"))         s->jitter = state::patchFloat(i);
    else if (state::pathIs(path, plen, "jitter_speed"))   s->jitter_speed = state::patchFloat(i);
    else if (state::pathIs(path, plen, "skip_empty"))     { bool v = state::patchFloat(i) != 0.0f; if (v != s->skip_empty) { s->skip_empty = v; vis_changed = true; } }
    else if (state::pathIs(path, plen, "skip_thresh"))    s->skip_thresh = state::patchFloat(i);
  }
  if (vis_changed) apply_visibility(s->show_streamlines, s->autopilot, s->show_limit_cycle, s->cycle_mode, s->skip_empty);
}

// ---- Tracer mode (CPU) — a different limit-cycle algorithm ----------------
// A single tracer integrates the blended flow forward over many frames; when its
// recent trajectory closes on itself we resample that loop and pull a momentum
// ring onto it. If the tracer zooms off or stalls at a sink we restart it from a
// point on the resting cycle (arc_angle). All on the CPU — the sequential trace
// + loop detection is awkward on the GPU and cheap here.

// Blended flow velocity at p (CPU port of pf_velocity in field.hlsl).
static void cpu_velocity(const State* s, const float* corners, const float* weights,
                         float px, float py, float& vx, float& vy) {
  float H = 0, gx = 0, gy = 0, lev = 0, mu = 0, Wx = 0, Wy = 0;
  for (int c = 0; c < 4; c++) {
    float wi = weights[c];
    if (wi <= 0.0f) continue;
    int base = (int)corners[c] * PF_STRIDE;
    float well = PF_CELLS[base];
    float h = -0.5f * well * (px * px + py * py);
    float cgx = -well * px, cgy = -well * py;
    for (int k = 0; k < PF_KERNELS; k++) {
      int o = base + 4 + k * 4;
      float dx = px - PF_CELLS[o], dy = py - PF_CELLS[o + 1];
      float sg = PF_CELLS[o + 2]; float s2 = sg * sg; if (s2 < 1e-6f) s2 = 1e-6f;
      float e = PF_CELLS[o + 3] * std::exp(-(dx * dx + dy * dy) / (2 * s2));
      h += e; cgx -= e * dx / s2; cgy -= e * dy / s2;
    }
    float rho = std::sqrt(px * px + py * py); if (rho < 1e-6f) rho = 1e-6f;
    for (int j = 0; j < PF_RINGS; j++) {
      int o = base + 4 + PF_KERNELS * 4 + j * 3;
      float dr = rho - PF_CELLS[o]; float rs = PF_CELLS[o + 1]; float rs2 = rs * rs; if (rs2 < 1e-6f) rs2 = 1e-6f;
      float e = PF_CELLS[o + 2] * std::exp(-dr * dr / (2 * rs2));
      h += e; float dHdrho = e * (-dr / rs2);
      cgx += dHdrho * px / rho; cgy += dHdrho * py / rho;
    }
    int wb = base + 45;
    H += wi * h; gx += wi * cgx; gy += wi * cgy;
    lev += wi * PF_CELLS[base + 1]; mu += wi * PF_CELLS[base + 2];
    Wx += wi * s->eff_wind * PF_CELLS[wb + 2] * PF_CELLS[wb + 0];
    Wy += wi * s->eff_wind * PF_CELLS[wb + 2] * PF_CELLS[wb + 1];
  }
  float sgn = -mu * (H - lev - s->bias);
  vx = -gy + sgn * gx + Wx; vy = gx + sgn * gy + Wy;
}

// RK2 step; returns the displacement magnitude (for sink detection).
static float cpu_step(const State* s, const float* c, const float* w, float& x, float& y, float dt) {
  float v0x, v0y; cpu_velocity(s, c, w, x, y, v0x, v0y);
  float v1x, v1y; cpu_velocity(s, c, w, x + 0.5f * dt * v0x, y + 0.5f * dt * v0y, v1x, v1y);
  float sx = v1x * dt, sy = v1y * dt;
  float m = std::sqrt(sx * sx + sy * sy); const float cap = 0.06f;
  if (m > cap) { sx *= cap / m; sy *= cap / m; m = cap; }
  x += sx; y += sy;
  return m;
}

// Arc-length resample of the closed loop hx/hy[start .. start+n-1] to PF_PARTICLES.
static void resample_loop(const float* hx, const float* hy, int start, int n,
                          float* outx, float* outy) {
  const int N = PF_PARTICLES;
  if (n < 2) { for (int i = 0; i < N; i++) { outx[i] = hx[start]; outy[i] = hy[start]; } return; }
  float cum[PF_TRACE_MAX + 1];
  cum[0] = 0.0f;
  for (int i = 0; i < n; i++) {
    int a = start + i, b = start + ((i + 1) % n);
    float dx = hx[b] - hx[a], dy = hy[b] - hy[a];
    cum[i + 1] = cum[i] + std::sqrt(dx * dx + dy * dy);
  }
  float total = cum[n];
  if (total < 1e-6f) { for (int i = 0; i < N; i++) { outx[i] = hx[start]; outy[i] = hy[start]; } return; }
  int seg = 0;
  for (int i = 0; i < N; i++) {
    float t = (float)i / (float)N * total;
    while (seg < n - 1 && cum[seg + 1] < t) seg++;
    float segl = cum[seg + 1] - cum[seg];
    float f = (segl > 1e-6f) ? (t - cum[seg]) / segl : 0.0f;
    int a = start + seg, b = start + ((seg + 1) % n);
    outx[i] = hx[a] + (hx[b] - hx[a]) * f;
    outy[i] = hy[a] + (hy[b] - hy[a]) * f;
  }
}

static void cpu_cycle_tracer(State* s, const float* corners, const float* weights, int nearest) {
  const int N = PF_PARTICLES;
  float wsum = weights[0] + weights[1] + weights[2] + weights[3];

  auto restart = [&]() {
    int idx = clampi((int)(clampf(s->arc_angle, 0.0f, 1.0f) * (PF_NOUT - 1)), 0, PF_NOUT - 1);
    int co = (nearest * PF_NOUT + idx) * 2;
    s->tr_x = PF_CURVE[co]; s->tr_y = PF_CURVE[co + 1];
    s->tr_sink = 0;
    s->loop_valid = false;
    s->tr_hx[0] = s->tr_x; s->tr_hy[0] = s->tr_y; s->tr_count = 1;
  };

  if (!s->tr_init || wsum < 1e-4f) { restart(); s->tr_init = true; }
  if (!s->ring_init) {
    for (int i = 0; i < N; i++) {
      int co = (nearest * PF_NOUT + i) * 2;
      s->ring_x[i] = PF_CURVE[co]; s->ring_y[i] = PF_CURVE[co + 1];
      s->ring_vx[i] = 0; s->ring_vy[i] = 0;
    }
    s->ring_init = true;
  }
  if (wsum < 1e-4f) return;   // hole

  // --- advance the tracer, detect a closed loop, restart on failure ---
  int steps = clampi((int)(s->trace_steps + 0.5f), 1, 16);
  float dt = (s->trace_step > 1e-4f) ? s->trace_step : 1e-4f;
  float sink_disp = 0.05f * dt;   // "barely moving" scales with the step size
  for (int st = 0; st < steps; st++) {
    float disp = cpu_step(s, corners, weights, s->tr_x, s->tr_y, dt);
    if (s->tr_count < PF_TRACE_MAX) { s->tr_hx[s->tr_count] = s->tr_x; s->tr_hy[s->tr_count] = s->tr_y; s->tr_count++; }
    s->tr_sink = (disp < sink_disp) ? (s->tr_sink + 1) : 0;

    bool zoom = std::sqrt(s->tr_x * s->tr_x + s->tr_y * s->tr_y) > 3.0f;
    bool sink = s->tr_sink > 10;
    bool timeout = s->tr_count >= PF_TRACE_MAX;
    if (zoom || sink || timeout) { restart(); break; }

    // loop detection: did we return near an older history point?
    const float eps = clampf(s->trace_eps, 0.005f, 0.5f); const int mingap = 20;
    int jf = -1;
    for (int j = 0; j + mingap < s->tr_count; j++) {
      float dx = s->tr_x - s->tr_hx[j], dy = s->tr_y - s->tr_hy[j];
      if (dx * dx + dy * dy < eps * eps) { jf = j; break; }
    }
    if (jf >= 0) {
      int n = s->tr_count - jf;
      resample_loop(s->tr_hx, s->tr_hy, jf, n, s->loop_x, s->loop_y);
      s->loop_valid = true;
      // keep the closing lap as the new baseline so we re-detect the next one
      for (int q = 0; q < n; q++) { s->tr_hx[q] = s->tr_hx[jf + q]; s->tr_hy[q] = s->tr_hy[jf + q]; }
      s->tr_count = n;
    }
  }

  // --- ring: real momentum. Pull toward the detected loop, else drift. ---
  //
  // The loop is a CLOSED curve with an arbitrary (and drifting) phase origin and
  // winding, so pulling ring[i] → loop[i] by index is the classic broken shape
  // morph — particles chase the wrong phase and never settle. Instead find the
  // cyclic rotation k + winding dir that best aligns the loop to the current
  // ring (min sum of squared index-wise distance, O(N²) with early-out), and
  // pull along THAT correspondence so the ring actually converges onto the loop.
  int bestK = 0, bestDir = 1;
  if (s->loop_valid) {
    float bestCost = 1e30f;
    for (int dir = -1; dir <= 1; dir += 2) {
      for (int k = 0; k < N; k++) {
        float cost = 0.0f;
        for (int i = 0; i < N; i++) {
          int li = ((dir > 0 ? i : -i) + k) % N; if (li < 0) li += N;
          float dx = s->ring_x[i] - s->loop_x[li], dy = s->ring_y[i] - s->loop_y[li];
          cost += dx * dx + dy * dy;
          if (cost >= bestCost) break;   // can't beat the best — prune
        }
        if (cost < bestCost) { bestCost = cost; bestK = k; bestDir = dir; }
      }
    }
  }

  float damp = clampf(s->momentum, 0.0f, 0.98f);
  for (int i = 0; i < N; i++) {
    if (s->loop_valid) {
      int li = ((bestDir > 0 ? i : -i) + bestK) % N; if (li < 0) li += N;
      s->ring_vx[i] += (s->loop_x[li] - s->ring_x[i]) * s->trace_pull;
      s->ring_vy[i] += (s->loop_y[li] - s->ring_y[i]) * s->trace_pull;
    }
    s->ring_vx[i] *= damp; s->ring_vy[i] *= damp;
    float vm = std::sqrt(s->ring_vx[i] * s->ring_vx[i] + s->ring_vy[i] * s->ring_vy[i]);
    if (vm > 0.15f) { s->ring_vx[i] *= 0.15f / vm; s->ring_vy[i] *= 0.15f / vm; }
    s->ring_x[i] += s->ring_vx[i]; s->ring_y[i] += s->ring_vy[i];
  }
}

void render(void* self, int vp_w, int vp_h) {
  auto* s = static_cast<State*>(self);
  if (!s || !s->initialized || vp_w <= 0 || vp_h <= 0) return;
  auto out = gpu::Device::textureForField("tex_out");
  if (!out.valid()) return;

  int nearest = 0;
  Uniforms u = {};
  u.res_x = (float)vp_w;
  u.res_y = (float)vp_h;
  // Domain zoom: higher scale → smaller visible extent → bigger features.
  u.extent = PF_EXTENT / (s->scale > 1e-2f ? s->scale : 1e-2f);
  u.shading_mode = (float)s->shading_mode;
  u.bias = s->bias;
  u.wind = s->eff_wind;
  u.n_bands = s->bands;
  u.contrast = s->contrast;
  u.flow_phase = s->flow_phase;
  u.stream_width = s->stream_width;
  u.stream_spread = (s->stream_spread > 1e-2f) ? s->stream_spread : 1e-2f;
  u.cycle_width = s->cycle_width;
  u.backdrop_dim = s->backdrop_dim;
  u.stream_alpha = s->line_opacity;
  u.solve_steps = s->solve_steps;
  u.break_dist = s->break_dist;
  u.explore = s->explore;
  u.spread = s->spread;
  // Exponential step size: [0,1] slider → PF_STEP_MIN..PF_STEP_MAX.
  u.step_size = PF_STEP_MIN * std::pow(PF_STEP_MAX / PF_STEP_MIN, clampf(s->step_size, 0.0f, 1.0f));
  u.momentum = s->momentum;
  u.morph_rate = s->morph_rate;
  u.respawn_arc = s->respawn_arc;
  // Turn threshold: break_turn 0 → cos -1 (never), 0.5 → 0 (90°), 1 → ~1.
  u.break_turn_cos = std::cos(kPi * (1.0f - clampf(s->break_turn, 0.0f, 1.0f)));
  u.good_init = s->particles_init ? 1.0f : 0.0f;
  u.rand_seed = (float)(s->frame_counter++ & 0xFFFFu);
  compute_corners(s, s->eff_x, s->eff_y, u.corners, u.weights, nearest);
  // The stateful solver respawns onto the nearest cell's baked resting cycle.
  u.nearest_cell = (float)nearest;
  // Respawn the particle ring on the first frame or when the hard timer elapses.
  bool do_respawn = !s->particles_init || s->respawn_accum >= s->respawn_time;
  u.respawn = do_respawn ? 1.0f : 0.0f;
  s->uniform_buf.writeOne(u);

  // 1 — backdrop (seeds tex_out).
  {
    auto cp = gpu::ComputePass::begin();
    cp.setPSO(s_pso_backdrop);
    cp.setBuffer(s->uniform_buf, 0);
    cp.setBuffer(s->cell_buf, 1);
    cp.setTexture(out, 2, 1);
    cp.dispatch((vp_w + 7) / 8, (vp_h + 7) / 8);
    cp.end();
  }

  // 2 — streamlines: trace (compute) then raster over the backdrop.
  if (s->show_streamlines) {
    {
      auto cp = gpu::ComputePass::begin();
      cp.setPSO(s_pso_stream);
      cp.setBuffer(s->uniform_buf, 0);
      cp.setBuffer(s->cell_buf, 1);
      cp.setBuffer(s->stream_buf, 2);
      cp.dispatch((PF_NS * PF_NS + 63) / 64, 1, 1);
      cp.end();
    }
    {
      auto rp = gpu::RenderPass::beginLoad(out);
      rp.setPSO(s_pso_lines);
      rp.setBuffer(s->uniform_buf, 0);
      rp.setBuffer(s->stream_buf, 1);
      rp.draw(6, kStreamSegs);
      rp.end();
    }
  }

  // 3 — limit-cycle: Contour draws the height field's zero directly; the others
  //     fill cycle_buf (Relax GPU / Tracer CPU) and line-raster it.
  if (s->show_limit_cycle && s->cycle_mode == 3) {
    auto rp = gpu::RenderPass::beginLoad(out);
    rp.setPSO(s_pso_contour);
    rp.setBuffer(s->uniform_buf, 0);
    rp.setBuffer(s->cell_buf, 1);
    rp.draw(3, 1);   // full-screen triangle
    rp.end();
  } else if (s->show_limit_cycle) {
    if (s->cycle_mode == 0) {
      // --- Relax: stateful GPU solver → build/break → longest-run/morph ---
      int groups = (PF_PARTICLES + 63) / 64;
      int prev = s->particle_cur;
      int next = 1 - prev;
      {
        auto cp = gpu::ComputePass::begin();
        cp.setPSO(s_pso_solve);
        cp.setBuffer(s->uniform_buf, 0);
        cp.setBuffer(s->cell_buf, 1);
        cp.setBuffer(s->particle_buf[next], 2);
        cp.setBuffer(s->curve_buf, 3);
        cp.setBuffer(s->particle_buf[prev], 4);
        cp.setBuffer(s->good_buf, 5);
        cp.setBuffer(s->status_buf, 6);
        cp.dispatch(groups, 1, 1);
        cp.end();
      }
      s->particle_cur = next;
      {
        auto cp = gpu::ComputePass::begin();
        cp.setPSO(s_pso_cycle);
        cp.setBuffer(s->uniform_buf, 0);
        cp.setBuffer(s->cell_buf, 1);
        cp.setBuffer(s->cycle_buf, 2);
        cp.setBuffer(s->particle_buf[next], 3);
        cp.dispatch(groups, 1, 1);
        cp.end();
      }
      {
        auto cp = gpu::ComputePass::begin();
        cp.setPSO(s_pso_select);
        cp.setBuffer(s->uniform_buf, 0);
        cp.setBuffer(s->cycle_buf, 2);
        cp.setBuffer(s->good_buf, 3);
        cp.setBuffer(s->status_buf, 4);
        cp.setBuffer(s->particle_buf[next], 5);
        cp.setBuffer(s->curve_buf, 6);
        cp.dispatch(1, 1, 1);
        cp.end();
      }
      s->particles_init = true;
      if (do_respawn) s->respawn_accum = 0.0f;
    } else {
      // --- Tracer / Trace (CPU): run the flow tracer, then either draw the
      //     momentum RING (Tracer) or the tracer's raw TRAJECTORY (Trace). ---
      cpu_cycle_tracer(s, u.corners, u.weights, nearest);
      float seg[kCycleSegs * kSegFloats];
      auto put = [&](int i, float x0, float y0, float x1, float y1, float arc, bool dead) {
        int o = i * kSegFloats;
        seg[o + 0] = x0; seg[o + 1] = y0; seg[o + 2] = x1; seg[o + 3] = y1;
        seg[o + 4] = 0.90f; seg[o + 5] = 0.95f; seg[o + 6] = s->cycle_width; seg[o + 7] = dead ? 1.0f : 0.0f;
        seg[o + 8] = arc; seg[o + 9] = 0; seg[o + 10] = 0; seg[o + 11] = 0;
      };
      if (s->cycle_mode == 1) {
        // Closed ring of N particles pulled onto the detected loop.
        for (int i = 0; i < kCycleSegs; i++) {
          int n = (i + 1) % PF_PARTICLES;
          put(i, s->ring_x[i], s->ring_y[i], s->ring_x[n], s->ring_y[n],
              (float)i / (float)PF_PARTICLES, false);
        }
      } else {
        // Trace: the tracer's most recent raw trajectory (open polyline), so the
        // actual step spacing is visible — long segments = stepping too fast.
        int avail = s->tr_count - 1;
        int K = avail < kCycleSegs ? avail : kCycleSegs;
        if (K < 0) K = 0;
        int start = s->tr_count - 1 - K;
        for (int i = 0; i < kCycleSegs; i++) {
          if (i < K) {
            int a = start + i;
            put(i, s->tr_hx[a], s->tr_hy[a], s->tr_hx[a + 1], s->tr_hy[a + 1],
                (float)i / (float)(K > 0 ? K : 1), false);
          } else {
            put(i, 0, 0, 0, 0, 0, true);
          }
        }
      }
      s->cycle_buf.write(seg, kCycleSegs * kSegFloats);
    }

    // raster (both algorithms)
    {
      auto rp = gpu::RenderPass::beginLoad(out);
      rp.setPSO(s_pso_lines);
      rp.setBuffer(s->uniform_buf, 0);
      rp.setBuffer(s->cycle_buf, 1);
      rp.draw(6, kCycleSegs);
      rp.end();
    }
  }

  // (The flow_field bake lives in tick() now — see bake_flow_field — so it runs
  // every frame even when this render() is skipped at opacity 0.)

  gpu::Device::submit();
}

} // namespace phase_fold
