/*
 * source.shape_fold — evolving-shape generator.
 *
 * Productionized from the shape-fold research testbed. A baked 3D atlas of
 * resolved shape parameters — axes frequency (x) × simplicity (y) × temporal-
 * complexity (z) — is interpolated on the CPU each frame (sampleTerms) down to
 * a handful of "terms" + dc/bold_gain; those ride in the uniform buffer and the
 * GPU evaluates the scalar SDF field from them. The atlas itself never touches
 * the GPU (it's CPU-only constant data in shape_fold_atlas.h).
 *
 * The field is histogram auto-leveled (median → 0) every frame, driven by an
 * exposure, and shown as grayscale or one of several colormap grades (magma /
 * inferno / viridis / plasma / turbo) — the raw field, no line/contour/shading
 * modes (dropped on purpose; downstream effects style it). The square field
 * COVERS the viewport uniformly (no bars) with a domain `scale` zoom.
 *
 * Autopilot spirals the (x,y) automatically via an epicycle. It is
 * NON-destructive: it overrides the effective XY internally for rendering but
 * never mutates the frequency/simplicity inputs — instead it broadcasts the
 * current XY on the autopilot_x / autopilot_y output fields so the web UI's
 * custom XY-pad editor can show the live position.
 *
 * GPU passes (one submit): minmax → hist → buildlut → present (auto-levels),
 * sharing common.hlsl. Generator → only tex_out, no tex_in. Multi-pass → NO
 * fusion. Internal time/orbit clock → NOT is_identity.
 */

#include <gpu.h>
#include <host.h>
#include <val.h>
#include <effect_utils.h>
#include "shape_fold_shaders.h"
#include "shape_fold_atlas.h"

#include <cmath>
#include <cstdint>
#include <algorithm>
#include <vector>

namespace shape_fold {

// Must match the #defines in common.hlsl (the HLSL constants aren't visible to
// C++; the shaders header carries only SPV bytecode).
static constexpr int SF_MAX_TERMS = 8;     // resolved-term array size in the uniform block
static constexpr int SF_NB        = 256;   // histogram bins / LUT entries
static constexpr int SF_SN        = 160;   // auto-levels downsample grid

static constexpr float kPi = 3.14159265358979323846f;
static constexpr float kLoopSecs = 6.0f;   // time_speed=1 → ~6 s loop (testbed dt/6)

// Key-moment playback: play a short window anchored on the analyzed center peak
// (SF_KM_PEAK — a frame index where new structure has streamed in and frames the
// centre) instead of the whole loop. The played span is a FIXED fraction of the
// full loop: kKmPre before the peak through +0.08 after (matching the web testbed,
// app.js). A cell whose baked SF_KM_SCORE is 0 has no usable window and falls back
// to the full [0,1] loop. This atlas carries the km sidecar directly (no separate
// key-moment atlas, no sky/reachability channel — unlike brutal_fold).
static constexpr float kKmPre  = 0.32f;
static constexpr float kKmSpan = 0.40f;   // kKmPre + 0.08

// Autopilot epicycle constants (verbatim from app.js). Two summed circular
// motions, 90° out of phase, incommensurate rates → sweeps the annulus without
// stalling at the centre.
static constexpr float kApA = 0.29f, kApB = 0.16f, kApW2 = 0.382f, kApPhi = kPi * 0.5f;
// Golden angle — used to advance the orbit on a snap/trigger jump so each new
// point is well-spread and distinct even when the orbit is barely drifting.
static constexpr float kGoldenAngle = 2.39996323f;

// --- Skip static: detect a large, mostly-STILL construct and jog past it -----
// shape_fold is auto-leveled, so it's essentially never a flat solid colour (the
// case brutal_fold hunts). What reads as "dead" here is a richly-detailed field
// that just sits there barely moving. So the detector is MOTION-dominant, and the
// motion is reduced by the GLOBAL MEAN over the frame (not the per-tile MAX
// brutal_fold uses) — a small moving corner shouldn't veto jogging through an
// otherwise-static frame. When engaged, a C2 ramp (fx::SkipJog) advances the loop
// clock (and, under autopilot, the orbit) as a MULTIPLE of the current rate, so
// Speed 0 stays frozen and a faster Speed skips faster.
static constexpr float kSkipTimeMult  = 8.0f;   // jog up to Nx the current loop speed
static constexpr float kSkipOrbitMult = 6.0f;   // jog up to Nx the current orbit speed
static constexpr float kSkipRampInSec = 0.6f;   // gentle ease-IN into the skip
static constexpr float kMaxRecoverSec = 1.0f;   // recover=0 → this slow ease-OUT (1 → instant)
static constexpr float kSkipHyst      = 0.004f; // recover margin above the trigger (hysteresis)
// Sensitivity [0,1] maps to this much activity (mean squashed motion, blended with
// the variance/edge feature weights) at the trigger. The activity metric lives in a
// TINY numeric range — a nearly-static frame's mean squashed motion is ~1e-3 — so
// the span is small and calibrated by eye: the default Sensitivity (0.5) sits right
// at the useful operating point (trigger ≈ 0.0025) with ~2× headroom either way,
// instead of jammed against the bottom of the knob. Tune live vs the skip_motion
// broadcast; if the useful setting drifts back toward 0, shrink this span again.
static constexpr float kSkipTrigSpan  = 0.005f;

// GPU detector geometry — must match edge.hlsl / debug.hlsl.
static constexpr int   kTileGrid      = 16;
static constexpr int   kSampleGrid    = 256;
static constexpr int   kNumTiles      = kTileGrid * kTileGrid;
// per tile: [edge,v,v²,motion,count, nx²,ny²,nxny,nx·dv,ny·dv,dv²] — features are
// measured on the LEVELED FIELD (see edge.hlsl); the last 6 are the SIGNED Lucas-
// Kanade flow sums feeding the uniform-drift penalty.
static constexpr int   kSlots         = 11;
static constexpr int   kEdgeStatsInts = kNumTiles * kSlots;
static constexpr float kStatsScale    = 65536.0f;// must match edge.hlsl kStatsScale
static constexpr float kEdgeNormGain  = 2.0f;    // per-tile edge/var-max sensitivity (tuning feats)
static constexpr float kVarFloor      = 0.008f;  // deadzone on per-tile luma std (quantization noise)

// Uniform block — mirrors SF_UNIFORMS in common.hlsl (std140). 3 scalar rows
// then the resolved terms (per term: (theta,mtheta,curv,freq)(phase,h,k,amp)
// (mix,spc,_,_)).
struct Uniforms {
  float res_x, res_y, n_terms, dc;
  float bold_gain, birth_softness, domain_scale, level_ease;
  float output_mode, exposure, _pad1, _pad2;
  float terms[SF_MAX_TERMS * 3 * 4];
};
static_assert(sizeof(Uniforms) == (12 + SF_MAX_TERMS * 3 * 4) * 4, "Uniforms layout");

// stats buffer (ints): [0]=lo(asint), [1]=hi(asint), [2..2+NB-1]=hist.
static constexpr int kStatsInts = 2 + SF_NB;
// lut buffer (floats): [0..NB-1]=LUT, [NB]=lo, [NB+1]=hi, [NB+2]=blank.
static constexpr int kLutFloats = SF_NB + 4;

// Small uniform for the skip-static debug visualizer pass (debug.hlsl).
struct DebugUniforms {
  float res_x, res_y, mode, _pad0;      // mode: 1=var 2=edge 3=motion 4=combined
  float w_var, w_edge, w_motion, _pad1;
};

struct State {
  gpu::Buffer uniform_buf;
  gpu::Buffer stats_buf;            // auto-levels min/max + histogram
  gpu::Buffer lut_buf;
  gpu::Buffer edge_stats_buf;       // skip-static motion/variance reduce → readback
  gpu::Buffer prev_field_buf;        // persistent per-sample previous-frame LEVELED field (motion)
  gpu::Buffer debug_uniform_buf;    // skip-static debug visualizer params
  bool skip_gpu_ready = false;      // true once the first edge readback has arrived
  bool initialized = false;

  // --- Schema-mirrored params ---
  float frequency           = 0.25f;
  float simplicity          = 0.85f;
  float temporal_complexity = 0.66f;
  float scale               = 1.0f;    // domain zoom (>1 reveals beyond [-1,1])
  float time_speed          = 0.58f;   // [0,1] quadratic → ~1.0 actual (≈6 s loop)
  float ease                = 0.0f;
  float birth_softness      = 0.45f;
  bool  autopilot           = false;
  float ap_speed            = 0.43f;   // [0,1] quadratic → ~0.6 actual
  bool  ap_snap             = false;
  float ap_hold_period      = 2.0f;    // seconds; 0 = no auto-jump (trigger only)
  float ap_hold_jitter      = 0.0f;    // 0..1 → randomize each hold interval ±fraction
  float level_ease          = 0.25f;
  float exposure            = 1.0f;    // pre-grade value drive (boost / reduce)
  int   output_mode         = 1;       // 0 = Grayscale, 1 = Magma (default)

  // --- Key moment: play a curated inflow→peak window from the atlas sidecar ---
  bool  key_moment        = false;  // master enable
  int   km_time_mode      = 2;      // 0 = Trigger (one-shot) 1 = Time (manual) 2 = Loop
  float km_time           = 0.0f;   // Time-mode manual playhead [0,1] over the window
  float km_duration       = 2.0f;   // seconds to play the window (Trigger/Loop)
  float km_ease           = 1.5f;   // ease-out toward the settled peak (0 = linear)
  float km_trigger_prev   = 0.0f;   // rising-edge state for the km_trigger event

  // --- Skip static: detect a near-still construct and jog past it ---
  bool  skip_empty        = false;  // master enable for detector + jog
  float skip_thresh       = 0.5f;   // Sensitivity [0,1] → activity trigger (× kSkipTrigSpan)
  // Per-feature weights [0,1]. shape_fold is never flat and has no hard edges, so
  // variance/edge default OFF (they'd read "busy" every frame); MOTION drives it.
  float skip_w_var        = 0.0f;
  float skip_w_edge       = 0.0f;
  float skip_w_motion     = 1.0f;
  int   skip_debug        = 0;      // 0=off 1=variance 2=edge 3=motion 4=combined (viz)
  // Penalize motion that is a uniform global DRIFT (same-direction scroll): [0,1],
  // 1 = fully ignore pure translation (a scrolling shape reads as "still" → jog).
  float skip_drift_penalty = 1.0f;
  float skip_recover      = 0.25f;  // Recover [0,1]: how fast the jog STOPS on motion (1 = instant)
  float skip_rate         = 0.5f;   // jog strength (time + orbit advance)
  bool  skip_autopilot    = true;   // also accelerate/snap the orbit (autopilot only)

  // --- Internal clocks (advanced in tick) ---
  float clock_t   = 0.0f;              // loop phase 0..1
  float orbit     = 0.0f;              // autopilot epicycle phase
  // Key-moment playhead: km_u ∈ [0,1] sweeps the window (Loop/Trigger accumulate it
  // in tick; Time reads km_time). km_playing gates the Trigger one-shot.
  float km_u       = 0.0f;
  bool  km_playing = false;
  float km_phase   = 0.0f;             // last resolved playhead (broadcast for the editor)
  fx::SkipJog jog;                     // skip-static C2 engagement ramp
  float content   = 1.0f;              // last frame's activity metric [0,1] (1 = moving/rich)
  float coherence = 0.0f;              // last frame's uniform-drift fraction [0,1] (for tuning)
  float snap_accum = 0.0f;            // snap-mode hold timer
  float next_hold  = 0.0f;            // jittered target for the current interval
  uint32_t rng     = 0x2545F491u;     // per-instance PRNG state (seeded in create)
  bool  held_valid = false;
  float held_x = 0.5f, held_y = 0.5f;
  float eff_x = 0.25f, eff_y = 0.85f;  // effective XY used for rendering
  float ap_jump_prev = 0.0f;           // rising-edge state for the jump trigger
  bool  ap_jump_pending = false;       // a trigger fired since the last tick
};

static void apply_visibility(bool autopilot, bool ap_snap, bool skip_empty,
                             bool key_moment, int km_time_mode) {
  // Key moment: mode picker + its mode-specific control (a manual Time scrub for
  // Time mode, a Duration + Trigger for Trigger/Loop). Ease applies whenever it's on.
  state::setFieldHidden("km_time_mode",  !key_moment);
  state::setFieldHidden("km_time",       !(key_moment && km_time_mode == 1));   // Time
  state::setFieldHidden("km_duration",   !(key_moment && km_time_mode != 1));   // Trigger/Loop
  state::setFieldHidden("km_trigger",    !(key_moment && km_time_mode != 1));   // Trigger/Loop
  state::setFieldHidden("km_ease",       !key_moment);
  // Continuous-clock controls superseded by the key-moment playhead. Hidden in KM.
  state::setFieldHidden("time_speed",    key_moment);   // superseded by km_duration
  state::setFieldHidden("ease",          key_moment);   // superseded by km_ease
  // Skip-static jogs the continuous loop clock — inert (and hidden) in KM mode.
  state::setFieldHidden("skip_empty",    key_moment);
  state::setFieldHidden("ap_speed",       !autopilot);
  state::setFieldHidden("ap_snap",        !autopilot);
  state::setFieldHidden("ap_hold_period", !(autopilot && ap_snap));
  state::setFieldHidden("ap_hold_jitter", !(autopilot && ap_snap));
  state::setFieldHidden("ap_jump",        !(autopilot && ap_snap));
  bool skip = skip_empty && !key_moment;
  state::setFieldHidden("skip_thresh",     !skip);
  state::setFieldHidden("skip_w_var",      !skip);
  state::setFieldHidden("skip_w_edge",     !skip);
  state::setFieldHidden("skip_w_motion",   !skip);
  state::setFieldHidden("skip_debug",      !skip);
  state::setFieldHidden("skip_drift_penalty", !skip);
  state::setFieldHidden("skip_recover",    !skip);
  state::setFieldHidden("skip_rate",       !skip);
  // Jogging the orbit only means anything under autopilot.
  state::setFieldHidden("skip_autopilot",  !(skip && autopilot));
}

// Static (self-less) visibility evaluator — pure over state (see crop).
void eval_visibility(int n, const char* pb, const int* off, const int* len, const int* ops) {
  bool autopilot = false, ap_snap = false, skip_empty = false, key_moment = false;
  int km_time_mode = 2;
  for (int i = 0; i < n; i++) {
    if (ops[i] != state::PatchReplace) continue;
    const char* p = pb + off[i]; int l = len[i];
    if      (state::pathIs(p, l, "autopilot"))    autopilot    = state::patchFloat(i) != 0.0f;
    else if (state::pathIs(p, l, "ap_snap"))      ap_snap      = state::patchFloat(i) != 0.0f;
    else if (state::pathIs(p, l, "skip_empty"))   skip_empty   = state::patchFloat(i) != 0.0f;
    else if (state::pathIs(p, l, "key_moment"))   key_moment   = state::patchFloat(i) != 0.0f;
    else if (state::pathIs(p, l, "km_time_mode")) km_time_mode = state::patchInt(i);
  }
  apply_visibility(autopilot, ap_snap, skip_empty, key_moment, km_time_mode);
}

static void on_state_ready(void* self);

// Type-shared, compiled once in module_init().
static gpu::ComputePSO s_pso_minmax;
static gpu::ComputePSO s_pso_hist;
static gpu::ComputePSO s_pso_buildlut;
static gpu::ComputePSO s_pso_present;
static gpu::ComputePSO s_pso_edge;      // motion/variance reduce over tex_out (skip-static)
static gpu::ComputePSO s_pso_debug;     // per-tile feature heatmap (debug viz)

void module_init() {
  state::init("source.shape_fold", {1, 1, 0},
    state::Schema()
      // Top-level manual: high-level "what is this / how to use / what to try".
      .helpField("intro",
        "## Shape Fold\n"
        "An evolving-shape generator. Drag the XY pad to explore a baked atlas of "
        "resolved shapes: **X** raises *Frequency* (denser detail), **Y** raises "
        "*Simplicity* (cleaner forms). The field is auto-leveled and shown as "
        "grayscale or a colormap — raw material for downstream effects to style.\n\n"
        "**Try:** push *Temporal Complexity* up and *Speed* low for a slow, richly "
        "animating field; turn on *Autopilot* to let it wander the atlas on its own; "
        "swap the *Colormap* to recolour the whole look.")
      // --- Shape axes (the custom XY pad drives frequency + simplicity) ---
      .group("shape", "Shape")
        .groupHelp(
          "The core look. *Frequency* (X) sets how dense the detail is; *Simplicity* "
          "(Y) trades chaos for clean forms. *Temporal Complexity* picks how richly "
          "the chosen shape animates, and *Scale* zooms the field in or out.")
      .floatField("frequency", 0.25f, 0.0f, 1.0f, state::PrimaryInput).label("Frequency", "Freq")
      .floatField("simplicity", 0.85f, 0.0f, 1.0f, state::PrimaryInput).label("Simplicity", "Simp")
      // Temporal-complexity (z): 0 = hold still → 1 = animate as richly as the
      // shape allows (trilinear trajectory layer select).
      .floatField("temporal_complexity", 0.66f, 0.0f, 1.0f, state::PrimaryInput).label("Temporal Complexity", "Temp")
      // Zoom. Higher = zoom IN (bigger features); lower zooms out, revealing
      // more of the periodic field beyond the prototype's [-1,1] window.
      .floatField("scale", 1.0f, 0.1f, 8.0f, state::PrimaryInput).label("Scale", "Scale")
      // --- Animation ---
      .group("animation", "Animation")
      // Autoplay clock speed (0 = frozen). [0,1] with a quadratic bend onto the
      // real 0..3 range, so the low end has fine control.
      .floatField("time_speed", 0.58f, 0.0f, 1.0f, state::PrimaryInput).label("Speed", "Spd")
      // Time-warp (bipolar). τ(t) = t − (ease/2π)·sin(2π t). +1 = rest at the
      // loop point, surge through the middle; −1 = surge at the loop point, rest
      // in the middle; 0 = uniform. |ease|≤1 keeps it monotone.
      .floatField("ease", 0.0f, -1.0f, 1.0f, state::PrimaryInput).label("Ease", "Ease")
      // How gradually AND-edges fade in/out (the soft birth gate width).
      .floatField("birth_softness", 0.45f, 0.02f, 1.0f, state::PrimaryInput).label("Birth Softness", "Birth")
      // --- Key moment: play the analyzed inflow→peak window instead of the loop ---
      .group("keymoment", "Key Moment")
        .groupHelp(
          "Each atlas cell (and temporal layer) was scored offline for its **key "
          "moment** — the short window where new structure streams in and comes to "
          "frame the centre. Turn this on to play just that window (anchored on the "
          "settled peak) instead of the whole loop; it also snaps to the exact cell "
          "so what plays is precisely what was scored. **Time Mode** picks how it "
          "plays: *Trigger* fires it once per **Trigger** and holds on the peak; "
          "*Time* lets you scrub the window by hand; *Loop* replays it continuously "
          "(and **Trigger** restarts it). *Duration* sets the playback length for "
          "Trigger/Loop. Cells with no clean key moment fall back to the full loop.")
      .boolField("key_moment", false, state::PrimaryInput).label("Key Moment", "KM")
      .selectField("km_time_mode", 2, state::PrimaryInput,
                   {{"Trigger", 0}, {"Time", 1}, {"Loop", 2}}).label("Time Mode", "Mode")
      .floatField("km_duration", 2.0f, 0.1f, 10.0f, state::PrimaryInput,
                  nullptr, /*step=*/0.05f, /*units=*/"s",
                  "How long the window takes to play (Trigger/Loop), in seconds.")
                  .label("Duration", "Dur")
      .floatField("km_ease", 1.5f, 0.0f, 4.0f, state::PrimaryInput,
                  nullptr, /*step=*/0.05f, /*units=*/nullptr,
                  "Ease-out toward the settled peak: 0 = linear, higher = lingers longer "
                  "on the framed moment at the end of the window.").label("Ease", "Ease")
      .floatField("km_time", 0.0f, 0.0f, 1.0f, state::PrimaryInput,
                  nullptr, /*step=*/0.005f, /*units=*/nullptr,
                  "Manual playhead over the key-moment window (Time mode): 0 = window "
                  "start, 1 = settled on the centre peak.").label("Time", "Time")
      .eventField("km_trigger", state::PrimaryInput).label("Trigger", "Trig")
      // Broadcast: the live playhead [0,1] within the window, for the editor readout.
      .floatField("km_phase", 0.0f, 0.0f, 1.0f, state::SecondaryOutput)
      // --- Autopilot (non-destructive XY override + broadcast) ---
      .group("autopilot", "Autopilot")
        .groupHelp(
          "Spirals the shape's XY position on its own, **without touching** your "
          "Frequency/Simplicity inputs. *Orbit Speed* sets the rate. Turn on **Snap** "
          "to hold each shape and hop to a new one every *Hold* seconds (with optional "
          "*Jitter*), or fire **Jump** to leap manually.")
      .boolField("autopilot", false, state::PrimaryInput).label("Autopilot", "Auto")
      // Orbit speed. [0,1] with a quadratic bend onto the real 0.05..3 range.
      .floatField("ap_speed", 0.43f, 0.0f, 1.0f, state::PrimaryInput).label("Orbit Speed", "Spd")
      // Snap: hold the current shape, then jump to a new point.
      .boolField("ap_snap", false, state::PrimaryInput).label("Snap", "Snap")
      // Hold seconds between auto-jumps. 0 = never auto-jump (hold until the
      // jump trigger fires).
      .floatField("ap_hold_period", 2.0f, 0.0f, 8.0f, state::PrimaryInput).label("Hold Period", "Hold")
      // Randomize each hold interval by ± this fraction of the base period.
      .floatField("ap_hold_jitter", 0.0f, 0.0f, 1.0f, state::PrimaryInput).label("Hold Jitter", "Jit")
      // Jump now — switch to a fresh point immediately (and reset the hold timer).
      .eventField("ap_jump", state::PrimaryInput).label("Jump", "Jump")
      // --- Auto-levels (histogram normalization, median → 0) ---
      .group("levels", "Auto-levels")
      // Below this contrast, taper the auto-levels boost so the field eases
      // toward black instead of flashing as it collapses to solid.
      .floatField("level_ease", 0.25f, 0.0f, 0.5f, state::PrimaryInput).label("Level Ease", "LvlEas")
      // --- Output ---
      .group("output", "Output")
      // Pre-grade value drive: >1 boosts (pushes brights into the rolloff),
      // <1 reduces toward mid. 1 = unity.
      .floatField("exposure", 1.0f, 0.0f, 4.0f, state::PrimaryInput).label("Exposure", "Expo")
      .selectField("output_mode", 1, state::PrimaryInput,
                   {{"Grayscale", 0}, {"Magma", 1}, {"Inferno", 2},
                    {"Viridis", 3}, {"Plasma", 4}, {"Turbo", 5}}, /*wrap=*/true).label("Colormap", "Color")
      // Broadcast: the effective XY (epicycle when autopilot is on, else the
      // input XY) so the custom editor can show the live position.
      .floatField("autopilot_x", 0.25f, 0.0f, 1.0f, state::SecondaryOutput)
      .floatField("autopilot_y", 0.85f, 0.0f, 1.0f, state::SecondaryOutput)
      // --- Skip static: jog past a large near-still construct ---
      .group("skip", "Skip Static")
        .groupHelp(
          "The atlas sometimes settles into a big, richly-detailed construct that "
          "just **sits there, barely moving**. Turn this on and the effect gently "
          "**jogs the loop forward** (on a smooth C2 curve) to glide past those "
          "still stretches instead of dwelling on them. Because shape_fold is never "
          "a flat solid colour, detection is by **motion** (how much the frame is "
          "changing) — not by flatness. *Sensitivity* sets how still a frame has to "
          "get before it kicks in; *Jog Rate* how briskly it skips. With "
          "**Autopilot** on it also nudges the orbit onward — or, if **Snap** is on, "
          "hops straight to a fresh shape the moment it goes still.")
      .boolField("skip_empty", false, state::PrimaryInput).label("Skip Static", "Skip")
      .floatField("skip_thresh", 0.5f, 0.0f, 1.0f, state::PrimaryInput,
                  nullptr, /*step=*/0.01f, /*units=*/nullptr,
                  "How readily a frame counts as still/dead. Higher flags more "
                  "scenes (more residual motion tolerated); 1 catches almost "
                  "anything but a briskly-animating frame.").label("Sensitivity", "Sens")
      // Per-feature weights, combined by weighted MAX. Motion is the one that
      // matters here; variance/edge default off (shape_fold is never flat).
      .floatField("skip_w_var", 0.0f, 0.0f, 1.0f, state::PrimaryInput,
                  nullptr, /*step=*/0.01f, /*units=*/nullptr,
                  "Weight of local tonal VARIANCE in the stillness test. Keep at 0: "
                  "variance measures spatial DETAIL, which is high for a FROZEN "
                  "detailed shape just as much as a live one — so any weight blocks "
                  "the jog on exactly the detailed-but-still frames you want to skip. "
                  "Only motion tells frozen from moving.")
                  .label("Variance Wt", "Var")
      .floatField("skip_w_edge", 0.0f, 0.0f, 1.0f, state::PrimaryInput,
                  nullptr, /*step=*/0.01f, /*units=*/nullptr,
                  "Weight of spatial EDGES in the stillness test (off by default — "
                  "shape_fold's soft SDF forms have no hard edges).")
                  .label("Edge Wt", "Edge")
      .floatField("skip_w_motion", 1.0f, 0.0f, 1.0f, state::PrimaryInput,
                  nullptr, /*step=*/0.01f, /*units=*/nullptr,
                  "Weight of MOTION (frame-to-frame change, averaged over the whole "
                  "frame) — the jog runs only while the frame is nearly still.")
                  .label("Motion Wt", "Motn")
      .selectField("skip_debug", 0, state::PrimaryInput,
                   {{"Off", 0}, {"Variance", 1}, {"Edge", 2}, {"Motion", 3}, {"Combined", 4}})
                  .label("Debug View", "Dbg")
      .floatField("skip_drift_penalty", 1.0f, 0.0f, 1.0f, state::PrimaryInput,
                  nullptr, /*step=*/0.01f, /*units=*/nullptr,
                  "Penalize motion that is a uniform DRIFT — everything sliding the "
                  "same direction (a scrolling shape isn't really evolving). Higher "
                  "discounts that coherent drift so the jog treats it as still and "
                  "skips through; a genuine morph or rotation is unaffected. 0 = off.")
                  .label("Drift Penalty", "Drift")
      .floatField("skip_recover", 0.25f, 0.0f, 1.0f, state::PrimaryInput,
                  nullptr, /*step=*/0.01f, /*units=*/nullptr,
                  "How fast the jog STOPS once the frame starts moving again "
                  "(the still→moving transition). Higher = snappier so it doesn't "
                  "skip past the motion; 1 = instant hard stop. Lower eases out "
                  "gently. (Onset stays a gentle C2 glide.)").label("Recover", "Rec")
      .floatField("skip_rate", 0.5f, 0.0f, 1.0f, state::PrimaryInput).label("Jog Rate", "Rate")
      .boolField("skip_autopilot", true, state::PrimaryInput).label("Jog Autopilot", "JogAP")
      // Broadcast: the live engagement [0,1] so an editor can show when it fires.
      .floatField("skip_active", 0.0f, 0.0f, 1.0f, state::SecondaryOutput)
      // Broadcast: the live activity metric (mean motion / feature blend) for tuning.
      .floatField("skip_motion", 0.0f, 0.0f, 1.0f, state::SecondaryOutput)
      // Broadcast: the live uniform-drift coherence [0,1] (1 = pure scroll) for tuning.
      .floatField("skip_coherence", 0.0f, 0.0f, 1.0f, state::SecondaryOutput)
      // --- I/O: pure generator (no input) ---
      .textureField("tex_out", state::PrimaryOutput)
        .capability(state::Capability::Generator)
        .capability(state::Capability::SeekableApproximate)
    );

  if (gpu::Device::backend() == gpu::Backend::None) return;

  // minmax/hist/buildlut write storage BUFFERS (no format hint); present writes
  // a sketch-default storage texture.
  state::registerShaderSPV("shape_fold_minmax",   MINMAX_SPV,   MINMAX_SPV_SIZE);
  state::registerShaderSPV("shape_fold_hist",     HIST_SPV,     HIST_SPV_SIZE);
  state::registerShaderSPV("shape_fold_buildlut", BUILDLUT_SPV, BUILDLUT_SPV_SIZE);
  state::registerShaderSPV("shape_fold_present",  PRESENT_SPV,  PRESENT_SPV_SIZE);
  state::registerShaderSPV("shape_fold_edge",     EDGE_SPV,     EDGE_SPV_SIZE);
  state::registerShaderSPV("shape_fold_debug",    DEBUG_SPV,    DEBUG_SPV_SIZE);

  auto cs_minmax   = gpu::Device::createShaderModuleByName("shape_fold_minmax");
  auto cs_hist     = gpu::Device::createShaderModuleByName("shape_fold_hist");
  auto cs_buildlut = gpu::Device::createShaderModuleByName("shape_fold_buildlut");
  auto cs_present  = gpu::Device::createShaderModuleByName("shape_fold_present");
  if (!cs_minmax || !cs_hist || !cs_buildlut || !cs_present) return;

  s_pso_minmax = gpu::Device::createComputePSO(cs_minmax, "main", gpu::Bindings()
      .uniform(0)
      .storageRW(1));       // stats (atomic min/max)

  s_pso_hist = gpu::Device::createComputePSO(cs_hist, "main", gpu::Bindings()
      .uniform(0)
      .storageRW(1));       // stats (atomic add)

  s_pso_buildlut = gpu::Device::createComputePSO(cs_buildlut, "main", gpu::Bindings()
      .uniform(0)
      .storage(1)           // stats (read)
      .storageRW(2));       // lut (write)

  s_pso_present = gpu::Device::createComputePSO(cs_present, "main", gpu::Bindings()
      .uniform(0)
      .storage(1)                                   // lut (read)
      .storageTex2d(2)); // tex_out

  auto cs_edge = gpu::Device::createShaderModuleByName("shape_fold_edge");
  if (!cs_edge) return;
  s_pso_edge = gpu::Device::createComputePSO(cs_edge, "main", gpu::Bindings()
      .uniform(0)      // shared cbuffer U (terms, res, domain_scale) — evaluates the field
      .storage(1)      // lut (auto-levels lo/hi, read)
      .storageRW(2)    // edge stats (read-write atomics)
      .storageRW(3));  // prevField (per-sample leveled-field motion history)

  auto cs_debug = gpu::Device::createShaderModuleByName("shape_fold_debug");
  if (!cs_debug) return;
  s_pso_debug = gpu::Device::createComputePSO(cs_debug, "main", gpu::Bindings()
      .uniform(0)      // debug uniform
      .storage(1)      // edge stats (read)
      .storageTex2d(2)); // tex_out (write viz)

  state::log("shape_fold: module initialized");
}

// Distinct PRNG seed per instance (no wall-clock / RNG primitive needed).
static uint32_t s_seed_counter = 0x9E3779B9u;

void* create() {
  auto* s = new State();
  s_seed_counter = s_seed_counter * 1664525u + 1013904223u;
  s->rng = s_seed_counter ^ 0xC0FFEEu;
  s->uniform_buf = gpu::Device::createBuffer(sizeof(Uniforms), gpu::BufferUsage::Uniform);
  s->stats_buf   = gpu::Device::createBuffer(kStatsInts * sizeof(int32_t), gpu::BufferUsage::Storage);
  s->lut_buf     = gpu::Device::createBuffer(kLutFloats * sizeof(float), gpu::BufferUsage::Storage);
  s->edge_stats_buf = gpu::Device::createBuffer(kEdgeStatsInts * sizeof(int32_t), gpu::BufferUsage::Storage);
  s->prev_field_buf  = gpu::Device::createBuffer(kSampleGrid * kSampleGrid * sizeof(float), gpu::BufferUsage::Storage);
  s->debug_uniform_buf = gpu::Device::createBuffer(sizeof(DebugUniforms), gpu::BufferUsage::Uniform);
  return s;
}

void destroy(void* self) {
  auto* s = static_cast<State*>(self);
  if (!s) return;
  s->uniform_buf.release();
  s->stats_buf.release();
  s->lut_buf.release();
  s->edge_stats_buf.release();
  s->prev_field_buf.release();
  s->debug_uniform_buf.release();
  delete s;
}

void init(void* self) {
  auto* s = static_cast<State*>(self);
  if (!s) return;
  s->initialized = false;
  if (!s_pso_minmax.valid() || !s_pso_hist.valid() ||
      !s_pso_buildlut.valid() || !s_pso_present.valid() ||
      !s_pso_edge.valid() || !s_pso_debug.valid()) return;
  if (!s->uniform_buf.valid() || !s->stats_buf.valid() || !s->lut_buf.valid() ||
      !s->edge_stats_buf.valid() || !s->prev_field_buf.valid() ||
      !s->debug_uniform_buf.valid()) return;
  // Zero the edge stats so the first poll (before any edge pass) reads count=0
  // and the detector stays inert (content=1) until a real readback arrives.
  int32_t z[kEdgeStatsInts] = {};
  s->edge_stats_buf.write(z, kEdgeStatsInts);
  // Large-negative sentinel so the first frame's motion reads 0 (no spurious spike).
  // Must sit below any real clamped field value (the field can dip negative), which
  // a plain -1 would not — the edge pass checks pf < -1e29.
  std::vector<float> negs(kSampleGrid * kSampleGrid, -1e30f);
  s->prev_field_buf.write(negs.data(), (int)negs.size());
  s->skip_gpu_ready = false;
  s->initialized = true;
  state::setOnStateReady(&on_state_ready);
}

static void on_state_ready(void* self) {
  auto* s = static_cast<State*>(self);
  if (!s) return;
  apply_visibility(s->autopilot, s->ap_snap, s->skip_empty, s->key_moment, s->km_time_mode);
}

static inline float clampf(float v, float lo, float hi) { return v < lo ? lo : (v > hi ? hi : v); }
static inline float lerpf(float a, float b, float f) { return a + (b - a) * f; }
static inline int   clampi(int v, int lo, int hi) { return v < lo ? lo : (v > hi ? hi : v); }

// Per-instance uniform random in [0,1) (LCG).
static inline float rng_unit(State* s) {
  s->rng = s->rng * 1664525u + 1013904223u;
  return (float)((s->rng >> 8) & 0xFFFFFFu) / (float)0x1000000;
}

// Epicycle position at a given orbit phase. Two summed circular motions, 90°
// out of phase — sweeps the annulus without stalling at the centre. Clamped to
// stay inside the pad. Port of app.js's autopilot.
static inline void orbit_xy(float orbit, float& ox, float& oy) {
  float a1 = orbit;
  float a2 = orbit * kApW2 + kApPhi;
  ox = clampf(0.5f + kApA * std::cos(a1) + kApB * std::cos(a2), 0.03f, 0.97f);
  oy = clampf(0.5f + kApA * std::sin(a1) + kApB * std::sin(a2), 0.03f, 0.97f);
}

// Advance the internal clocks and compute the effective (broadcast) XY.
void tick(void* self, double dt) {
  auto* s = static_cast<State*>(self);
  if (!s) return;
  float fdt = (float)dt;

  // Quadratic speed response: the [0,1] params bend onto their real ranges
  // (time 0..3, ap 0.05..3) so the low end has fine control.
  float time_actual = s->time_speed * s->time_speed * 3.0f;
  float ap_actual   = 0.05f + s->ap_speed * s->ap_speed * (3.0f - 0.05f);

  // Skip static: engage the C2 ramp on last frame's activity. It always jogs the
  // loop clock forward; under autopilot it also nudges the orbit (or, in Snap
  // mode, fires a one-shot hop). `content` lags a frame — that's fine (the ramp
  // is smooth and hysteretic).
  float skip_e = 0.0f;
  if (s->skip_empty) {
    // Poll the GPU reduce (motion/variance over last frame's LEVELED FIELD — the
    // edge pass evaluates the field + stable linear levels, not the flashy tex_out).
    // MOTION is reduced by the GLOBAL MEAN over the frame (so a small moving region
    // doesn't veto jogging through an otherwise-still frame); variance and edge take
    // the per-tile MAX (tuning features, default off). Poll before the jog so this
    // frame's decision uses the freshest available metric.
    int32_t raw[kEdgeStatsInts];
    if (s->edge_stats_buf.pollReadback(raw, sizeof(raw)) == (int)sizeof(raw)) {
      float wv = clampf(s->skip_w_var, 0.0f, 1.0f);
      float we = clampf(s->skip_w_edge, 0.0f, 1.0f);
      float wm = clampf(s->skip_w_motion, 0.0f, 1.0f);
      float var_max = 0.0f, edge_max = 0.0f;
      double motion_sum = 0.0;   // Σ squashed per-sample motion, over all tiles
      long   count_sum  = 0;     // Σ samples, over all tiles
      // Global Lucas-Kanade flow accumulators (raw signed sums summed over tiles).
      double Sxx = 0, Syy = 0, Sxy = 0, Sxt = 0, Syt = 0, Stt = 0;
      for (int t = 0; t < kNumTiles; t++) {
        int base = t * kSlots;
        int cnt = raw[base + 4];
        if (cnt <= 0) continue;
        float ne = (float)cnt;
        float lu = ((float)raw[base + 1] / kStatsScale) / ne;    // tile mean luma
        float lq = ((float)raw[base + 2] / kStatsScale) / ne;    // tile mean luma²
        float var = lq - lu * lu; if (var < 0.0f) var = 0.0f;
        float sd = std::sqrt(var);
        sd = sd > kVarFloor ? sd - kVarFloor : 0.0f;             // local luma std (deadzoned)
        if (sd > var_max) var_max = sd;
        // Concentrated edge saturates regardless of area (/√tilePixels).
        float edge = clampf((float)raw[base + 0] / kStatsScale / std::sqrt(ne) * kEdgeNormGain, 0.0f, 1.0f);
        if (edge > edge_max) edge_max = edge;
        motion_sum += (double)raw[base + 3] / kStatsScale;       // squashed |dv| (leveled field)
        count_sum  += cnt;
        Sxx += (double)raw[base + 5] / kStatsScale;
        Syy += (double)raw[base + 6] / kStatsScale;
        Sxy += (double)raw[base + 7] / kStatsScale;
        Sxt += (double)raw[base + 8] / kStatsScale;
        Syt += (double)raw[base + 9] / kStatsScale;
        Stt += (double)raw[base + 10] / kStatsScale;
      }
      // Uniform-drift coherence: fit ONE global translation u to the brightness-
      // constancy model dt ≈ -(nx,ny)·u (least squares over all samples). The
      // fraction of temporal energy it explains, coh = Σ(model²)/Σdt² ∈ [0,1], is
      // ~1 for a shape merely scrolling in one direction and ~0 for a morph/rotation
      // (whose flow cancels globally). Regularize the 2×2 for the aperture case.
      float coh = 0.0f;
      if (Stt > 1e-9) {
        double eps = 1e-4 * (Sxx + Syy) + 1e-9;
        double a11 = Sxx + eps, a22 = Syy + eps, a12 = Sxy;
        double det = a11 * a22 - a12 * a12;
        if (det > 1e-12) {
          double bx = -Sxt, by = -Syt;                           // RHS of the normal eqs
          double ux = ( a22 * bx - a12 * by) / det;
          double uy = (-a12 * bx + a11 * by) / det;
          double expl = ux * bx + uy * by;                       // Σ(model²) = u·b
          coh = (float)clampf((float)(expl / Stt), 0.0f, 1.0f);
        }
      }
      s->coherence = coh;
      if (count_sum > 0) {
        float motion_mean = (float)(motion_sum / (double)count_sum);   // global mean motion [0,1]
        // Discount motion that is a uniform global drift: same-direction scrolling
        // isn't really evolving, so let the jog treat it as "still" and skip through.
        motion_mean *= (1.0f - clampf(s->skip_drift_penalty, 0.0f, 1.0f) * coh);
        float content = motion_mean * wm;
        float ev = edge_max * we; if (ev > content) content = ev;
        float vv = var_max * wv;  if (vv > content) content = vv;
        s->content = content;
        s->skip_gpu_ready = true;
      }
    }
    // Sensitivity [0,1] → mean-motion trigger. Higher = flags more scenes as still.
    float lo = clampf(s->skip_thresh, 0.0f, 1.0f) * kSkipTrigSpan;
    float hi = lo + kSkipHyst;
    // Recover controls the ease-OUT (motion→stop); 1 = instant (rampOut≈0 → snap).
    float rampOut = (1.0f - clampf(s->skip_recover, 0.0f, 1.0f)) * kMaxRecoverSec;
    skip_e = s->jog.update(s->content, lo, hi, kSkipRampInSec, rampOut, fdt);
    // Snap mode: the moment we go still, hop straight to a fresh held position.
    if (s->autopilot && s->skip_autopilot && s->ap_snap && s->jog.rising())
      s->ap_jump_pending = true;
  } else {
    s->jog = fx::SkipJog{};          // fully reset when disabled (no residual jog)
    s->content = 1.0f;
    s->skip_gpu_ready = false;
  }
  // Jog is a MULTIPLE of the current rate, so Speed 0 stays frozen (no auto-play)
  // and a faster Speed skips proportionally faster.
  float time_jog = skip_e * s->skip_rate * time_actual * kSkipTimeMult;
  // Extra orbit advance from the orbit jog — only in continuous (non-Snap) mode.
  float orbit_jog = (s->skip_autopilot && !s->ap_snap)
                        ? skip_e * s->skip_rate * ap_actual * kSkipOrbitMult : 0.0f;

  s->clock_t += fdt * (time_actual + time_jog) / kLoopSecs;
  s->clock_t -= std::floor(s->clock_t);

  if (s->autopilot) {
    s->orbit -= fdt * (ap_actual + orbit_jog);   // clockwise drift (+ skip jog)

    bool trig = s->ap_jump_pending;
    s->ap_jump_pending = false;

    if (s->ap_snap) {
      bool hold_on = s->ap_hold_period > 1e-4f;
      bool jump = trig || !s->held_valid;        // trigger or first frame
      if (hold_on) {                             // auto-jump only when Hold > 0
        s->snap_accum += fdt;
        if (s->snap_accum >= s->next_hold) jump = true;
      }
      if (jump) {
        // Hold ON: snap to where the drifting orbit currently is — "where the
        // continuous autopilot point would have been" (auto-jump AND trigger).
        // Hold OFF: the trigger advances to a fresh random orbit point.
        if (trig && !hold_on && s->held_valid) s->orbit -= kGoldenAngle;
        orbit_xy(s->orbit, s->held_x, s->held_y);
        s->held_valid = true;
        s->snap_accum = 0.0f;                     // reset the hold timer
        // Schedule the next interval, jittered ± a fraction of the base period.
        float jit = s->ap_hold_jitter * (2.0f * rng_unit(s) - 1.0f);
        s->next_hold = s->ap_hold_period * (1.0f + jit);
        if (s->next_hold < 0.02f) s->next_hold = 0.02f;
      }
      s->eff_x = s->held_x; s->eff_y = s->held_y;
    } else {
      if (trig) s->orbit -= kGoldenAngle;         // a manual switch in continuous too
      orbit_xy(s->orbit, s->eff_x, s->eff_y);
      s->held_valid = false;
    }
  } else {
    s->eff_x = s->frequency;
    s->eff_y = s->simplicity;
    s->held_valid = false;          // re-enabling snap jumps immediately
    s->snap_accum = 0.0f;
    s->ap_jump_pending = false;     // ignore triggers while autopilot is off
  }

  // --- Key-moment playhead. Loop/Trigger advance km_u 0→1 over km_duration
  //     seconds; Time reads the manual km_time. (The scene cell + window are
  //     resolved in render from the atlas sidecar.) ---
  if (s->key_moment) {
    float rate = 1.0f / std::max(s->km_duration, 0.05f);   // playhead units/sec
    if (s->km_time_mode == 1) {                       // Time — manual scrub
      s->km_u = clampf(s->km_time, 0.0f, 1.0f);
    } else if (s->km_time_mode == 2) {                // Loop — continuous replay
      s->km_u += fdt * rate;
      s->km_u -= std::floor(s->km_u);
    } else {                                          // Trigger — one-shot, hold on peak
      if (s->km_playing) {
        s->km_u += fdt * rate;
        if (s->km_u >= 1.0f) { s->km_u = 1.0f; s->km_playing = false; }
      }
    }
    s->km_phase = s->km_u;
  }
  auto vkm = val::number(s->km_phase);   // live key-moment playhead (for the editor)
  state::setValPath("km_phase", vkm);
  val::release(vkm);

  // Broadcast the effective XY so the editor can show the live position.
  auto vx = val::number(s->eff_x);
  state::setValPath("autopilot_x", vx);
  val::release(vx);
  auto vy = val::number(s->eff_y);
  state::setValPath("autopilot_y", vy);
  val::release(vy);
  auto vsk = val::number(skip_e);
  state::setValPath("skip_active", vsk);
  val::release(vsk);
  auto vsm = val::number(s->content);   // live activity metric (for calibration)
  state::setValPath("skip_motion", vsm);
  val::release(vsm);
  auto vco = val::number(s->coherence); // live uniform-drift fraction (for calibration)
  state::setValPath("skip_coherence", vco);
  val::release(vco);
}


void on_state_patched(void* self, int n, const char* pb, const int* off,
                      const int* len, const int* ops) {
  auto* s = static_cast<State*>(self);
  if (!s) return;
  bool mode_changed = false;
  for (int i = 0; i < n; i++) {
    if (ops[i] != state::PatchReplace) continue;
    const char* path = pb + off[i];
    int plen = len[i];
    if      (state::pathIs(path, plen, "frequency"))           s->frequency = state::patchFloat(i);
    else if (state::pathIs(path, plen, "simplicity"))          s->simplicity = state::patchFloat(i);
    else if (state::pathIs(path, plen, "temporal_complexity")) s->temporal_complexity = state::patchFloat(i);
    else if (state::pathIs(path, plen, "scale"))               s->scale = state::patchFloat(i);
    else if (state::pathIs(path, plen, "time_speed"))          s->time_speed = state::patchFloat(i);
    else if (state::pathIs(path, plen, "ease"))                s->ease = state::patchFloat(i);
    else if (state::pathIs(path, plen, "birth_softness"))      s->birth_softness = state::patchFloat(i);
    else if (state::pathIs(path, plen, "autopilot"))           { bool v = state::patchFloat(i) != 0.0f; if (v != s->autopilot) { s->autopilot = v; mode_changed = true; } }
    else if (state::pathIs(path, plen, "ap_speed"))            s->ap_speed = state::patchFloat(i);
    else if (state::pathIs(path, plen, "ap_snap"))             { bool v = state::patchFloat(i) != 0.0f; if (v != s->ap_snap) { s->ap_snap = v; mode_changed = true; } }
    else if (state::pathIs(path, plen, "ap_hold_period"))      s->ap_hold_period = state::patchFloat(i);
    else if (state::pathIs(path, plen, "ap_hold_jitter"))      s->ap_hold_jitter = state::patchFloat(i);
    else if (state::pathIs(path, plen, "ap_jump")) {
      float v = state::patchFloat(i);
      if (v != 0.0f && s->ap_jump_prev == 0.0f) s->ap_jump_pending = true;  // rising edge
      s->ap_jump_prev = v;
    }
    else if (state::pathIs(path, plen, "level_ease"))          s->level_ease = state::patchFloat(i);
    else if (state::pathIs(path, plen, "exposure"))            s->exposure = state::patchFloat(i);
    else if (state::pathIs(path, plen, "output_mode"))         s->output_mode = (int)state::patchFloat(i);
    else if (state::pathIs(path, plen, "key_moment"))          { bool v = state::patchFloat(i) != 0.0f; if (v != s->key_moment) { s->key_moment = v; mode_changed = true; } }
    else if (state::pathIs(path, plen, "km_time_mode"))        { int v = state::patchInt(i); if (v != s->km_time_mode) { s->km_time_mode = v; mode_changed = true; } }
    else if (state::pathIs(path, plen, "km_time"))             s->km_time = state::patchFloat(i);
    else if (state::pathIs(path, plen, "km_duration"))         s->km_duration = state::patchFloat(i);
    else if (state::pathIs(path, plen, "km_ease"))             s->km_ease = state::patchFloat(i);
    else if (state::pathIs(path, plen, "km_trigger")) {
      float v = state::patchFloat(i);
      if (v != 0.0f && s->km_trigger_prev == 0.0f) { s->km_u = 0.0f; s->km_playing = true; }  // rising edge → (re)start
      s->km_trigger_prev = v;
    }
    else if (state::pathIs(path, plen, "skip_empty"))          { bool v = state::patchFloat(i) != 0.0f; if (v != s->skip_empty) { s->skip_empty = v; mode_changed = true; } }
    else if (state::pathIs(path, plen, "skip_thresh"))         s->skip_thresh = state::patchFloat(i);
    else if (state::pathIs(path, plen, "skip_w_var"))          s->skip_w_var = state::patchFloat(i);
    else if (state::pathIs(path, plen, "skip_w_edge"))         s->skip_w_edge = state::patchFloat(i);
    else if (state::pathIs(path, plen, "skip_w_motion"))       s->skip_w_motion = state::patchFloat(i);
    else if (state::pathIs(path, plen, "skip_debug"))          s->skip_debug = (int)state::patchFloat(i);
    else if (state::pathIs(path, plen, "skip_drift_penalty"))  s->skip_drift_penalty = state::patchFloat(i);
    else if (state::pathIs(path, plen, "skip_recover"))        s->skip_recover = state::patchFloat(i);
    else if (state::pathIs(path, plen, "skip_rate"))           s->skip_rate = state::patchFloat(i);
    else if (state::pathIs(path, plen, "skip_autopilot"))      s->skip_autopilot = state::patchFloat(i) != 0.0f;
  }
  if (mode_changed) apply_visibility(s->autopilot, s->ap_snap, s->skip_empty, s->key_moment, s->km_time_mode);
}

// easeOut on the window playhead: slow toward the settled peak. Mirrors app.js —
// e>0 eases out, e<=0 stays linear (reuses brutal_fold's shared km_ease formula).
static inline float km_ease_out(float u, float e) {
  u = clampf(u, 0.0f, 1.0f);
  return e <= 0.0f ? u : 1.0f - std::pow(1.0f - u, 1.0f + e);
}

// Resolve the flat key-moment sidecar index for the current effective XY + layer:
// row = simplicity (y), col = frequency (x), z = temporal_complexity — snapped to
// the nearest grid cell (mirrors app.js kmWindow). idx = (row*G + col)*Z + z, the
// same layout as SF_KM_* and the script fields.
static inline int km_cell_index(const State* s) {
  const int G = SF_GRID, Z = SF_NZ;
  int col = clampi((int)std::lround(clampf(s->eff_x, 0.0f, 1.0f) * (G - 1)), 0, G - 1);
  int row = clampi((int)std::lround(clampf(s->eff_y, 0.0f, 1.0f) * (G - 1)), 0, G - 1);
  int z   = clampi((int)std::lround(clampf(s->temporal_complexity, 0.0f, 1.0f) * (Z - 1)), 0, Z - 1);
  return (row * G + col) * Z + z;
}

// CPU atlas resolve: bilinear over (x,y) for base/cell fields, trilinear over
// (x,y,z) for the temporal trajectory, then the periodic time model. Port of
// sampleTerms in app.js. Fills u.terms / u.n_terms / u.dc / u.bold_gain. When
// `snap`, picks the exact nearest cell + temporal layer (no blend) — key-moment
// mode plays precisely the trajectory that was scored.
static void sample_terms(const State* s, float sx, float sy, float t, Uniforms& u, bool snap) {
  const int G = SF_GRID, B = SF_NTERMS, Z = SF_NZ;
  float fx = clampf(sx, 0.0f, 1.0f) * (G - 1);
  float fy = clampf(sy, 0.0f, 1.0f) * (G - 1);
  int x0, y0, x1, y1, z0, z1;
  float tx, ty, tz;
  if (snap) {
    x0 = x1 = clampi((int)std::lround(fx), 0, G - 1);
    y0 = y1 = clampi((int)std::lround(fy), 0, G - 1);
    z0 = z1 = clampi((int)std::lround(clampf(s->temporal_complexity, 0.0f, 1.0f) * (Z - 1)), 0, Z - 1);
    tx = ty = tz = 0.0f;
  } else {
    x0 = clampi((int)std::floor(fx), 0, G - 1); y0 = clampi((int)std::floor(fy), 0, G - 1);
    x1 = std::min(x0 + 1, G - 1); y1 = std::min(y0 + 1, G - 1);
    tx = fx - x0; ty = fy - y0;
    float fz = clampf(s->temporal_complexity, 0.0f, 1.0f) * (Z - 1);
    z0 = clampi((int)std::floor(fz), 0, Z - 1); z1 = std::min(z0 + 1, Z - 1);
    tz = fz - z0;
  }
  int c00 = y0 * G + x0, c01 = y0 * G + x1, c10 = y1 * G + x0, c11 = y1 * G + x1;

  auto baseBi = [&](const float* arr, int i) {
    return lerpf(lerpf(arr[c00 * B + i], arr[c01 * B + i], tx),
                 lerpf(arr[c10 * B + i], arr[c11 * B + i], tx), ty);
  };
  auto layerBi = [&](const float* arr, int i, int z) {
    int o = z * B;
    return lerpf(lerpf(arr[c00 * Z * B + o + i], arr[c01 * Z * B + o + i], tx),
                 lerpf(arr[c10 * Z * B + o + i], arr[c11 * Z * B + o + i], tx), ty);
  };
  auto scriptTri = [&](const float* arr, int i) {
    return lerpf(layerBi(arr, i, z0), layerBi(arr, i, z1), tz);
  };
  auto cellBi = [&](const float* arr) {
    return lerpf(lerpf(arr[c00], arr[c01], tx), lerpf(arr[c10], arr[c11], tx), ty);
  };

  // Eased time: rests at the loop point, surges through the middle. Seamless
  // (τ(0)=0, τ(1)=1) so integer omega/drift keep the loop closed.
  float tau = t - (s->ease / (2.0f * kPi)) * std::sin(2.0f * kPi * t);

  int nt = std::min(B, SF_MAX_TERMS);
  for (int i = 0; i < nt; i++) {
    float h0 = baseBi(SF_H, i), phase0 = baseBi(SF_PHASE, i);
    float ampH = scriptTri(SF_AMP_H, i);
    float omega = std::round(scriptTri(SF_OMEGA_H, i));   // integer periods → loop closes
    float psi = scriptTri(SF_PSI_H, i);
    float drift = std::round(scriptTri(SF_DRIFT, i));
    float h = h0 + ampH * std::sin(2.0f * kPi * (omega * tau + psi));   // periodic height/birth
    float phase = phase0 + drift * tau;                                // periodic normal sweep
    int o = i * 3 * 4;
    u.terms[o + 0] = baseBi(SF_THETA, i);
    u.terms[o + 1] = baseBi(SF_MTHETA, i);
    u.terms[o + 2] = baseBi(SF_CURV, i);
    u.terms[o + 3] = baseBi(SF_FREQ, i);
    u.terms[o + 4] = phase;
    u.terms[o + 5] = h;
    u.terms[o + 6] = baseBi(SF_K, i);
    u.terms[o + 7] = baseBi(SF_AMP, i);
    u.terms[o + 8] = baseBi(SF_MIX, i);
    u.terms[o + 9] = baseBi(SF_SPC, i);
    u.terms[o + 10] = 0.0f;
    u.terms[o + 11] = 0.0f;
  }
  for (int i = nt; i < SF_MAX_TERMS; i++) {
    int o = i * 3 * 4;
    for (int k = 0; k < 12; k++) u.terms[o + k] = 0.0f;
  }
  u.n_terms = (float)nt;
  u.dc = cellBi(SF_DC);
  u.bold_gain = cellBi(SF_BOLD_GAIN);
}

// Buffer-only dispatch: uniform at slot 0, plus up to two storage buffers.
static inline void disp_buf(const gpu::ComputePSO& pso, int x, int y,
                            const gpu::Buffer& ub,
                            const gpu::Buffer* b1, int b1slot,
                            const gpu::Buffer* b2, int b2slot) {
  auto cp = gpu::ComputePass::begin();
  cp.setPSO(pso);
  cp.setBuffer(ub, 0);
  if (b1) cp.setBuffer(*b1, b1slot);
  if (b2) cp.setBuffer(*b2, b2slot);
  cp.dispatch(x, y);
  cp.end();
}

void render(void* self, int vp_w, int vp_h) {
  auto* s = static_cast<State*>(self);
  if (!s || !s->initialized || vp_w <= 0 || vp_h <= 0) return;
  auto out = gpu::Device::textureForField("tex_out");
  if (!out.valid()) return;

  Uniforms u = {};
  u.res_x = (float)vp_w;
  u.res_y = (float)vp_h;
  u.birth_softness = s->birth_softness;
  // `scale` is user-facing zoom (higher = zoom IN = bigger features), so the
  // sampled domain shrinks: p = sq / scale.
  u.domain_scale = (s->scale > 1e-4f) ? 1.0f / s->scale : 1.0f;
  u.level_ease = s->level_ease;
  u.exposure = s->exposure;
  u.output_mode = (float)s->output_mode;
  // Key moment: if the snapped cell has a scored window, play [peak-kKmPre,
  // peak+0.08] of the loop with the eased playhead (snapping to the exact scored
  // cell); else — and in continuous mode — run the normal continuous loop clock.
  if (s->key_moment) {
    int idx = km_cell_index(s);
    if (SF_KM_NFRAMES > 0 && SF_KM_SCORE[idx] > 0.0f) {
      float peak = SF_KM_PEAK[idx] / (float)SF_KM_NFRAMES;   // normalized loop phase
      float t = peak - kKmPre + km_ease_out(s->km_u, s->km_ease) * kKmSpan;
      t -= std::floor(t);                                    // wrap into [0,1)
      sample_terms(s, s->eff_x, s->eff_y, t, u, /*snap=*/true);
    } else {
      sample_terms(s, s->eff_x, s->eff_y, s->clock_t, u, /*snap=*/false);
    }
  } else {
    sample_terms(s, s->eff_x, s->eff_y, s->clock_t, u, /*snap=*/false);
  }
  s->uniform_buf.writeOne(u);

  // Reset stats: lo = +INF-ish (max i32), hi = 0 (F ≥ 0), hist = 0.
  int32_t reset[kStatsInts];
  reset[0] = 0x7fffffff;
  reset[1] = 0;
  for (int i = 2; i < kStatsInts; i++) reset[i] = 0;
  s->stats_buf.write(reset, kStatsInts);

  int sg = (SF_SN + 7) / 8;

  // 1 — field min/max over the SN×SN grid.
  disp_buf(s_pso_minmax, sg, sg, s->uniform_buf, &s->stats_buf, 1, nullptr, 0);
  // 2 — histogram into the same stats buffer.
  disp_buf(s_pso_hist, sg, sg, s->uniform_buf, &s->stats_buf, 1, nullptr, 0);
  // 3 — invert the histogram into the remap LUT (single invocation).
  disp_buf(s_pso_buildlut, 1, 1, s->uniform_buf, &s->stats_buf, 1, &s->lut_buf, 2);

  // 4 — present: auto-leveled field → grayscale / magma, square-fit.
  {
    auto cp = gpu::ComputePass::begin();
    cp.setPSO(s_pso_present);
    cp.setBuffer(s->uniform_buf, 0);
    cp.setBuffer(s->lut_buf, 1);
    cp.setTexture(out, 2, 1);
    cp.dispatch((vp_w + 7) / 8, (vp_h + 7) / 8);
    cp.end();
  }

  // 5 — skip-static detector: motion/variance reduce over the LEVELED FIELD (the
  // edge pass evaluates the field itself + the stable linear lo/hi from the lut,
  // NOT the flashy equalized tex_out), then request an async readback. Only when
  // the detector is on. Reset runs BEFORE the pass; the tick() poll reads last
  // frame's stats before this reset each frame, so the CPU zero-write is race-free.
  if (s->skip_empty) {
    int32_t zeros[kEdgeStatsInts] = {};
    s->edge_stats_buf.write(zeros, kEdgeStatsInts);
    auto ep = gpu::ComputePass::begin();
    ep.setPSO(s_pso_edge);
    ep.setBuffer(s->uniform_buf, 0);
    ep.setBuffer(s->lut_buf, 1);         // auto-levels lo/hi (evaluate the leveled field)
    ep.setBuffer(s->edge_stats_buf, 2);
    ep.setBuffer(s->prev_field_buf, 3);
    ep.dispatch((kSampleGrid + 7) / 8, (kSampleGrid + 7) / 8);   // fixed grid, not per-pixel
    ep.end();
    s->edge_stats_buf.requestReadback(kEdgeStatsInts * sizeof(int32_t));

    // Debug: overwrite tex_out with a per-tile heatmap of the selected feature.
    // Reads the stats the edge pass just wrote (same submit) → no tex_out hazard.
    if (s->skip_debug != 0) {
      DebugUniforms du = {};
      du.res_x = (float)vp_w; du.res_y = (float)vp_h; du.mode = (float)s->skip_debug;
      du.w_var = s->skip_w_var; du.w_edge = s->skip_w_edge; du.w_motion = s->skip_w_motion;
      s->debug_uniform_buf.writeOne(du);
      auto dp = gpu::ComputePass::begin();
      dp.setPSO(s_pso_debug);
      dp.setBuffer(s->debug_uniform_buf, 0);
      dp.setBuffer(s->edge_stats_buf, 1);
      dp.setTexture(out, 2, 1);          // access 1 = Write
      dp.dispatch((vp_w + 7) / 8, (vp_h + 7) / 8);
      dp.end();
    }
  }

  gpu::Device::submit();
}

} // namespace shape_fold
