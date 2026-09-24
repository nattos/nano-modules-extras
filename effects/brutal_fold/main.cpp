/*
 * source.brutal_fold — brutalist axonometric-prism generator.
 *
 * Productionized from the brutal-fold research testbed. A baked control surface
 * — axes complexity (x) × order (y) × liveliness (z), with a co-folded second
 * structure — is interpolated on the CPU each frame (build_params) down to two
 * structures' worth of "terms" + scene scalars; those ride in a uniform buffer
 * and the GPU composites the receding prism layers from them. The atlas itself
 * never touches the GPU (it's CPU-only constant data in brutal_fold_atlas.h).
 *
 * The field is an algebraic occupancy field rendered as solid axonometric
 * (oblique-depth) prisms — "3D without a vanishing point" — in grayscale, with
 * fog fading distant receding layers toward the light sky tone. The solid
 * threshold (`level`) is resolved on the CPU (a coarse occupancy min/max scan)
 * so the GPU work is a SINGLE present compute pass (no auto-levels) — a port of
 * the prototype's field.ts build + shader.wgsl composite.
 *
 * Animation is a seamless bounded loop: integer temporal frequencies return to
 * their t=0 value at t=1, so the loop closes at any speed. Speeds map [0,1]
 * through a QUADRATIC bend onto a deliberately low actual max — the effect reads
 * best very slow. An optional autopilot spirals the (x,y) automatically and
 * broadcasts its live position (autopilot_x/y) without mutating the inputs.
 *
 * GPU passes (one submit): present only. Generator → only tex_out, no tex_in.
 * Internal time/orbit clock → NOT is_identity.
 */

#include <gpu.h>
#include <host.h>
#include <val.h>
#include <effect_utils.h>
#include "brutal_fold_shaders.h"
#include "brutal_fold_atlas.h"

#include <cmath>
#include <cstdint>
#include <algorithm>
#include <vector>

namespace brutal_fold {

static constexpr float kPi = 3.14159265358979323846f;
static constexpr float kTau = 6.28318530717958647692f;

// Parameter-buffer layout (mirrors field.ts): per structure,
// [ terms (NT*8) | scene (SCN) | level ]. Two structures sit back to back.
static constexpr int SB1 = BF_NTERMS * 8;   // scene base within a structure (48)
static constexpr int SCN = 19;              // scene scalar count (matches SCENE_FIELDS)
static constexpr int STR = SB1 + SCN + 1;   // per-structure stride incl. level (68)
static_assert(2 * STR <= 256, "P buffer (both structures) must fit in 256 floats");

// Scene scalar positions WITHIN a structure's scene block — order MUST match
// atlas.ts SCENE and the present shader's offsets.
enum Scene {
  S_SEV = 0, S_GX, S_GY, S_FORM_SCALE, S_THRESH, S_BACK_LEN, S_BACK_ANG, S_EXTRUDE,
  S_LAYERS, S_SEP, S_FRONT_DETAIL, S_WIN_DARK, S_FOG, S_FACE, S_SKY_VAL, S_DC,
  S_BOLD_GAIN, S_ROT, S_SLEW,
};

// Atlas accessor tables (the baked flat arrays, ordered to match the loops).
static const float* const TERM_A[8] = {
  BF_THETA, BF_MTHETA, BF_SHEAR, BF_FREQ, BF_PHASE, BF_H, BF_AMP, BF_MIX };
static const float* const B_TERM_A[8] = {
  BF_B_THETA, BF_B_MTHETA, BF_B_SHEAR, BF_B_FREQ, BF_B_PHASE, BF_B_H, BF_B_AMP, BF_B_MIX };
static const float* const SCENE_A[SCN] = {
  BF_SEV, BF_GX, BF_GY, BF_FORM_SCALE, BF_THRESH, BF_BACK_LEN, BF_BACK_ANG, BF_EXTRUDE,
  BF_LAYERS, BF_SEP, BF_FRONT_DETAIL, BF_WIN_DARK, BF_FOG, BF_FACE, BF_SKY_VAL, BF_DC,
  BF_BOLD_GAIN, BF_ROT, BF_SLEW };
static const float* const B_SCENE_A[SCN] = {
  BF_B_SEV, BF_B_GX, BF_B_GY, BF_B_FORM_SCALE, BF_B_THRESH, BF_B_BACK_LEN, BF_B_BACK_ANG, BF_B_EXTRUDE,
  BF_B_LAYERS, BF_B_SEP, BF_B_FRONT_DETAIL, BF_B_WIN_DARK, BF_B_FOG, BF_B_FACE, BF_B_SKY_VAL, BF_B_DC,
  BF_B_BOLD_GAIN, BF_B_ROT, BF_B_SLEW };
// Animation script channels (structure 1 only): h_amp, h_om, h_psi, phase_drift.
static const float* const SCRIPT_A[4] = { BF_H_AMP, BF_H_OM, BF_H_PSI, BF_PHASE_DRIFT };

// Key-moment atlas (BFKM_*) — its own scenes, XY only (NZ=1), curated windows + sky.
static const float* const KM_TERM_A[8] = {
  BFKM_THETA, BFKM_MTHETA, BFKM_SHEAR, BFKM_FREQ, BFKM_PHASE, BFKM_H, BFKM_AMP, BFKM_MIX };
static const float* const KM_B_TERM_A[8] = {
  BFKM_B_THETA, BFKM_B_MTHETA, BFKM_B_SHEAR, BFKM_B_FREQ, BFKM_B_PHASE, BFKM_B_H, BFKM_B_AMP, BFKM_B_MIX };
static const float* const KM_SCENE_A[SCN] = {
  BFKM_SEV, BFKM_GX, BFKM_GY, BFKM_FORM_SCALE, BFKM_THRESH, BFKM_BACK_LEN, BFKM_BACK_ANG, BFKM_EXTRUDE,
  BFKM_LAYERS, BFKM_SEP, BFKM_FRONT_DETAIL, BFKM_WIN_DARK, BFKM_FOG, BFKM_FACE, BFKM_SKY_VAL, BFKM_DC,
  BFKM_BOLD_GAIN, BFKM_ROT, BFKM_SLEW };
static const float* const KM_B_SCENE_A[SCN] = {
  BFKM_B_SEV, BFKM_B_GX, BFKM_B_GY, BFKM_B_FORM_SCALE, BFKM_B_THRESH, BFKM_B_BACK_LEN, BFKM_B_BACK_ANG, BFKM_B_EXTRUDE,
  BFKM_B_LAYERS, BFKM_B_SEP, BFKM_B_FRONT_DETAIL, BFKM_B_WIN_DARK, BFKM_B_FOG, BFKM_B_FACE, BFKM_B_SKY_VAL, BFKM_B_DC,
  BFKM_B_BOLD_GAIN, BFKM_B_ROT, BFKM_B_SLEW };
static const float* const KM_SCRIPT_A[4] = { BFKM_H_AMP, BFKM_H_OM, BFKM_H_PSI, BFKM_PHASE_DRIFT };

// One resolvable control surface. build_params/build_struct read through this so
// the same resolve code drives either the explore atlas (continuous loop) or the
// key-moment atlas (curated inflow→peak windows + sky reachability). km_* is null
// on the explore atlas (no windows).
struct AtlasRef {
  int grid, nz, n_terms, co_fold;
  float h_act;
  const float* const* term;
  const float* const* b_term;
  const float* const* scene;
  const float* const* b_scene;
  const float* const* script;
  const float* b_tilt;
  const float* km_t1;
  const float* km_score;
  const float* km_covmax;
  const float* km_sky;
};

static const AtlasRef kAtlasExplore = {
  BF_GRID, BF_NZ, BF_NTERMS, BF_CO_FOLD, BF_H_ACT,
  TERM_A, B_TERM_A, SCENE_A, B_SCENE_A, SCRIPT_A, BF_B_TILT,
  nullptr, nullptr, nullptr, nullptr };
static const AtlasRef kAtlasKm = {
  BFKM_GRID, BFKM_NZ, BFKM_NTERMS, BFKM_CO_FOLD, BFKM_H_ACT,
  KM_TERM_A, KM_B_TERM_A, KM_SCENE_A, KM_B_SCENE_A, KM_SCRIPT_A, BFKM_B_TILT,
  BFKM_T1, BFKM_SCORE, BFKM_COVMAX, BFKM_SKY };

// Speed mapping: [0,1] squared onto a low actual max (the effect reads best slow).
static constexpr float kTimeMax = 1.0f;    // loops/sec at time_speed=1 (~1 s min loop)
static constexpr float kApMin = 0.05f;     // orbit rad/sec floor
static constexpr float kApMax = 0.6f;      // orbit rad/sec at ap_speed=1
static constexpr float kScrollCells = 1.0f; // structure-2 integer cells scrolled per loop

// Drift: a bounded second-order random walk (the velocity random-walks; a spring
// + damping keep it mean-reverting and bounded; a hard reflect at ±range is the
// final guard). drift_speed scales the whole evolution rate. Smooth (C1) — no
// twitch-style per-frame discontinuity. Tuned by eye.
static constexpr float kDriftSigma  = 3.5f;   // velocity kick magnitude
static constexpr float kDriftSpring = 1.6f;   // mean-reversion toward 0 (weak → wanders wide)
static constexpr float kDriftDamp   = 0.93f;  // per-step velocity damping
static constexpr float kDriftRate   = 1.6f;   // base loops of evolution / sec
static constexpr float kDriftMaxSpeed = 2.5f; // drift_speed=1 → this rate multiplier
// Depth is a sensitive axis (a thin Z window shifts visibly), so its drift gets an
// extra low-pass (one-pole EMA) on top of the walk to keep it gliding, not jumping.
static constexpr float kDepthDriftSmoothRate = 3.0f;  // ~0.33 s time constant

// The blob's radius/softness/depth sliders are normalized 0..1 (default 0.5);
// these map them onto the shader's actual ranges.
static constexpr float kVolRadiusMax = 2.0f;   // radius      slider 1 → 2.0
static constexpr float kVolSoftXYMax = 2.0f;   // softness_xy slider 1 → 2.0
static constexpr float kVolSoftZMax  = 1.0f;   // softness_z  slider 1 → 1.0
static constexpr float kVolDepthMax  = 0.6f;   // depth       slider 1 → 0.6 (Z half-extent)

// Skip-empty jog: when the front reads as flat solid colour (all sky or all
// panel), engage a C2 ramp (fx::SkipJog) that gently advances the loop clock
// (and, under autopilot, the orbit) to move past the dead stretch. The jog is a
// MULTIPLE of the current rate (not an absolute speed) — so at Speed 0 the clock
// stays frozen (no surprise auto-play), and a faster Speed skips faster.
static constexpr float kSkipTimeMult  = 8.0f;  // jog up to Nx the current loop speed
static constexpr float kSkipOrbitMult = 6.0f;  // jog up to Nx the current orbit speed
// Asymmetric C2 ramp: gentle ease-IN (glide into the skip), fast ease-OUT — once a
// live frame returns we stop promptly so we don't jog past the content. The ease-
// OUT duration is user-controlled (skip_recover); this is its slow end (recover=0).
// recover=1 → ~0s = instant hard stop (SkipJog snaps the phase in one frame).
static constexpr float kSkipRampInSec  = 0.6f;
static constexpr float kMaxRecoverSec  = 1.0f;
// Recovery is deliberately eager: the moment content climbs back above the empty
// threshold we ease out. A tiny gap above the trigger is the most sensitive a
// hysteresis latch can be without chattering (recover must stay ≥ engage). Want
// it even more eager? Lower Sensitivity — that drops both the engage AND recover
// point together.
static constexpr float kSkipHyst       = 0.004f; // recover variance above the trigger
static constexpr int   kMaxLayers      = 6;     // BF_MAXL — receding layers/structure
// Sensitivity is a normalized [0,1] knob; this is the luminance std-dev it maps to
// at FULL sensitivity. Kept small so the knob resolves the low-variance regime
// where flat/near-flat frames actually live — only a genuinely busy frame climbs
// past even the top of the range.
static constexpr float kSkipStdSpan    = 0.12f;
// GPU flatness detector: a Sobel/variance reduce over the rendered tex_out
// (edge.hlsl) writes 4 int slots read back to the CPU. kStatsScale mirrors the
// shader's fixed-point scale. kEdgeGpuGain calibrates the 1px-Sobel edge term to
// the luminance-std anchor (a 1px gradient is a large multiple of the old CPU
// 120px-grid one) — tune live against the skip_variance broadcast.
// Flatness is measured PER TILE (kTileGrid², 4 int slots each) and reduced by MAX
// over tiles, so any single structured region — an edge or luma variance anywhere
// — reads as non-flat. edge_count per tile is normalized by the tile's LINEAR
// dimension (√pixels) so a concentrated edge saturates regardless of its area.
static constexpr int   kTileGrid     = 16;      // must match edge.hlsl kTileGrid
static constexpr int   kSampleGrid   = 256;     // must match edge.hlsl; fixed sampling grid
static constexpr int   kNumTiles     = kTileGrid * kTileGrid;
static constexpr int   kSlots        = 5;       // per tile: [edge,luma,luma²,motion,count]
static constexpr int   kStatsInts    = kNumTiles * kSlots;
static constexpr float kStatsScale   = 65536.0f; // must match edge.hlsl kStatsScale
static constexpr float kEdgeNormGain = 2.0f;    // higher = more edge-sensitive per tile
// Deadzone on local luma std to absorb residual fixed-point quantization noise
// (~0.005 at kStatsScale 65536) so a truly uniform frame reads as flat variance.
static constexpr float kVarFloor     = 0.008f;

// Key-moment playback: play a short window anchored on the analyzed center peak
// (a "framed" moment where structure has streamed in), instead of the whole
// loop. The span is a FIXED fraction of the full loop — kKmPre before the peak
// (t1) through kKmPost after — matching the web testbed (main.ts). A cell whose
// baked key-moment score is 0, or whose played span is too poppy (covmax above
// kKmMaxCov), has no usable window and falls back to the full [0,1] loop.
static constexpr float kKmPre    = 0.32f;
static constexpr float kKmPost   = 0.08f;
static constexpr float kKmSpan   = kKmPre + kKmPost;   // 0.40 of the full loop
static constexpr float kKmMaxCov = 0.12f;              // skip windows poppier than this

// Autopilot epicycle constants (verbatim from the shape_fold autopilot). Two
// summed circular motions, 90° out of phase, incommensurate rates → sweeps the
// annulus without stalling at the centre.
static constexpr float kApA = 0.29f, kApB = 0.16f, kApW2 = 0.382f, kApPhi = kPi * 0.5f;
static constexpr float kGoldenAngle = 2.39996323f;

// Uniform block — mirrors cbuffer U in common.hlsl. 4 std140 rows of scalars,
// then the packed parameter array P (both structures).
struct Uniforms {
  float res_x, res_y, n_terms, sb;
  float str, tilt, enable2, h_act;
  float diff_hue_lo, diff_hue_mid, diff_hue_hi, diff_sat;
  float diff_sat_lo, diff_sat_mid, diff_sat_hi, _dpad0;
  float diff_bri_lo, diff_bri_mid, diff_bri_hi, _dpad1;
  float fog_hue_lo, fog_hue_mid, fog_hue_hi, fog_sat;
  float fog_sat_lo, fog_sat_mid, fog_sat_hi, _fpad0;
  float sky_hue, sky_sat, sky_bri, _spad0;
  float noise_blob, noise_fog, noise_seed, noise_blob_tilt;
  // Volumetric fog (DRIFTED/effective values written each frame).
  float vol_amount, vol_anchor_x, vol_anchor_y, vol_z;
  float vol_shape, vol_angle, vol_radius, vol_softness_xy;
  float vol_depth, vol_softness_z, _vpad1, _vpad2;
  float P[256];
};
static_assert(sizeof(Uniforms) == (48 + 256) * 4, "Uniforms layout");

// Small uniform for the debug visualizer pass (debug.hlsl).
struct DebugUniforms {
  float res_x, res_y, mode, _pad0;      // mode: 1=var 2=edge 3=motion 4=combined
  float w_var, w_edge, w_motion, _pad1;
};

struct State {
  gpu::Buffer uniform_buf;
  gpu::Buffer stats_buf;            // GPU flatness reduce (edge.hlsl) → readback
  gpu::Buffer prev_luma_buf;        // persistent per-sample previous-frame luma (motion)
  gpu::Buffer debug_uniform_buf;    // debug visualizer params
  bool skip_gpu_ready = false;      // true once the first GPU readback has arrived
  bool initialized = false;

  // --- Schema-mirrored params ---
  float complexity        = 0.6f;   // atlas x
  float order             = 0.6f;   // atlas y
  float liveliness        = 1.0f;   // atlas z (temporal layer): 0 still → 1 lively
  float scale             = 1.0f;   // form zoom; higher = zoom IN (inverted in build)
  float balance           = 0.0f;   // signed: zoom S1 in as S2 out
  float extrude           = 1.0f;   // recession depth multiplier
  bool  second_structure  = true;   // enable the co-folded S2 layer
  bool  interp_cells      = true;   // bilinear-blend atlas cells
  float time_speed        = 0.5f;   // [0,1] quadratic → loops/sec (default ≈ 4 s loop)
  float ease              = 0.0f;   // time-warp: rest at loop point, surge through middle
  float anim_amount       = 1.0f;   // in/out (birth) oscillation amplitude
  float fog               = 1.0f;   // fog strength multiplier
  // --- Colour grade: 3-control-point tone→hue twist (diffuse over panel tone,
  //     fog over depth). sat=0 → grayscale passthrough. Defaults are a teal-shadow
  //     / warm-highlight split for diffuse and a cool→deep-blue fog. ---
  float diff_hue_lo       = 0.58f;
  float diff_hue_mid      = 0.08f;
  float diff_hue_hi       = 0.11f;
  float diff_sat          = 0.0f;   // overall strength
  float diff_sat_lo       = 1.0f;   // per-knot saturation (shadows/mids/highs)
  float diff_sat_mid      = 1.0f;
  float diff_sat_hi       = 1.0f;
  float diff_bri_lo       = 1.0f;   // per-knot brightness (shadows/mids/highs)
  float diff_bri_mid      = 1.0f;
  float diff_bri_hi       = 1.0f;
  float fog_hue_lo        = 0.55f;
  float fog_hue_mid       = 0.60f;
  float fog_hue_hi        = 0.66f;
  float fog_sat           = 0.0f;   // overall tint
  float fog_sat_lo        = 1.0f;   // per-knot saturation (near/mid/far)
  float fog_sat_mid       = 1.0f;
  float fog_sat_hi        = 1.0f;
  float sky_hue           = 0.0f;   // sky twist (relative to far): hue offset
  float sky_sat           = 0.0f;   // added saturation
  float sky_bri           = 1.0f;   // brightness scale
  float noise_blob        = 0.0f;   // TV static on the blob density
  float noise_blob_tilt   = 0.0f;   // +1 → static on the transparent halo, -1 → dense centre
  float noise_fog         = 0.0f;   // TV static on the distance fog
  float noise_speed       = 0.5f;   // reroll rate (global): 0 = frozen, 1 ≈ 30 Hz
  float noise_phase       = 0.0f;   // accumulator → floor() is the reroll frame id
  // --- Volumetric fog: a 3D "shape" blob (twitch-style) modulating the fog.
  //     vol_amount=0 → uniform depth fog (backward compatible). ---
  float vol_amount        = 0.0f;   // blend uniform depth fog → blob density
  float vol_anchor_x      = 0.0f;   // blob centre X (cover-square)
  float vol_anchor_y      = 0.0f;   // blob centre Y
  float vol_z             = 0.5f;   // blob Z ANCHOR [0,1]
  float vol_shape         = 1.0f;   // bipolar: |1| sphere → |0.5| slab → |0| solid
  float vol_angle         = 0.0f;   // turns; slab normal sweeps X ↔ depth-plane
  float vol_radius        = 0.5f;   // normalized 0..1 (× kVolRadiusMax)
  float vol_softness_xy    = 0.5f;  // normalized 0..1 (× kVolSoftXYMax) — screen edge
  float vol_softness_z     = 0.5f;  // normalized 0..1 (× kVolSoftZMax)  — depth edge
  float vol_depth         = 0.5f;   // normalized 0..1 (× kVolDepthMax); small = selective slice
  // Drift amounts (bounded 2nd-order random walk on each), + overall speed.
  float drift_xy          = 0.0f;
  float drift_z           = 0.0f;
  float drift_shape       = 0.0f;
  float drift_angle       = 0.0f;
  float drift_speed       = 0.3f;
  bool  autopilot         = false;
  float ap_speed          = 0.43f;  // [0,1] cubic → orbit rad/sec
  bool  ap_snap           = false;
  float ap_hold_period    = 2.0f;   // seconds; 0 = no auto-jump (trigger only)
  float ap_hold_jitter    = 0.0f;   // 0..1 → randomize each hold interval ±fraction
  // --- Skip empty: detect dead (flat solid-colour) frames and jog past them ---
  bool  skip_empty        = false;  // master enable for detector + jog
  float skip_thresh       = 0.7f;   // Sensitivity [0,1] → variance trigger (× kSkipStdSpan)
  // Per-feature weights [0,1] combined by weighted MAX (variance / edge / motion).
  // Default tuning: motion-dominant, a touch of edge, variance off.
  float skip_w_var        = 0.0f;
  float skip_w_edge       = 0.07f;
  float skip_w_motion     = 1.0f;
  int   skip_debug        = 0;      // 0=off 1=variance 2=edge 3=motion 4=combined (viz)
  float skip_recover      = 1.0f;   // Recover [0,1]: how fast the jog STOPS on content (1 = instant)
  float skip_rate         = 0.5f;   // jog strength (time + orbit advance)
  bool  skip_autopilot    = true;   // also accelerate/snap the orbit (autopilot only)

  // --- Key moment: play a curated inflow→peak window from the KM atlas ---
  bool  key_moment        = false;  // master enable (swaps to the KM atlas)
  int   km_time_mode      = 2;      // 0 = Trigger (one-shot) 1 = Time (manual) 2 = Loop
  float km_time           = 0.0f;   // Time-mode manual playhead [0,1] over the window
  float km_duration       = 2.0f;   // seconds to play the window (Trigger/Loop)
  float km_ease           = 1.5f;   // ease-out toward the settled peak (0 = linear)
  float sky_threshold     = 0.0f;   // reachability filter: drop cells emptier than this
  float km_trigger_prev   = 0.0f;   // rising-edge state for the km_trigger event

  // --- Internal clocks (advanced in tick) ---
  float clock_t    = 0.0f;          // loop phase 0..1
  float orbit      = 0.0f;          // autopilot epicycle phase
  // Key-moment playhead: km_u ∈ [0,1] sweeps the window (Loop/Trigger accumulate
  // it in tick; Time reads km_time). km_playing gates the Trigger one-shot.
  float km_u       = 0.0f;
  bool  km_playing = false;
  float km_phase   = 0.0f;          // last resolved playhead (broadcast for the editor)
  fx::SkipJog jog;                  // skip-empty C2 engagement ramp
  float content    = 1.0f;          // last frame's front content [0,1] (1 = rich)
  float snap_accum = 0.0f;          // snap-mode hold timer
  float next_hold  = 0.0f;          // jittered target for the current interval
  uint32_t rng     = 0x2545F491u;   // per-instance PRNG state (seeded in create)
  bool  held_valid = false;
  float held_x = 0.6f, held_y = 0.6f;
  float eff_x = 0.6f, eff_y = 0.6f; // effective XY used for rendering / broadcast
  float ap_jump_prev = 0.0f;        // rising-edge state for the jump trigger
  bool  ap_jump_pending = false;

  // Drift walks: offset (o) + velocity (v) per drifting quantity.
  float dx_o = 0.0f, dx_v = 0.0f;   // anchor x
  float dy_o = 0.0f, dy_v = 0.0f;   // anchor y
  float dz_o = 0.0f, dz_v = 0.0f;   // z
  float dz_smooth = 0.0f;           // extra low-pass on the depth drift
  float dsh_o = 0.0f, dsh_v = 0.0f; // shape
  float dan_o = 0.0f, dan_v = 0.0f; // angle
  // Effective (drifted) blob values — fed to the uniform + broadcast for preview.
  float eff_vol_x = 0.0f, eff_vol_y = 0.0f, eff_vol_z = 0.5f;
  float eff_vol_shape = 1.0f, eff_vol_angle = 0.0f;
};

static void apply_visibility(bool autopilot, bool ap_snap, bool skip_empty,
                             bool key_moment, int km_time_mode) {
  // Key moment: mode picker + its mode-specific control (a manual Time scrub for
  // Time mode, a Duration + Trigger for Trigger/Loop). Ease + Sky Threshold apply
  // whenever it's on. All KM controls hide in continuous mode.
  state::setFieldHidden("km_time_mode",  !key_moment);
  state::setFieldHidden("km_time",       !(key_moment && km_time_mode == 1));   // Time
  state::setFieldHidden("km_duration",   !(key_moment && km_time_mode != 1));   // Trigger/Loop
  state::setFieldHidden("km_trigger",    !(key_moment && km_time_mode != 1));   // Trigger/Loop
  state::setFieldHidden("km_ease",       !key_moment);
  state::setFieldHidden("sky_threshold", !key_moment);
  // Continuous-only controls that don't apply to the KM atlas (its own scenes,
  // XY-only, curated window, single cell). Hidden while key-moment mode is on.
  state::setFieldHidden("liveliness",   key_moment);   // KM atlas has no z axis
  state::setFieldHidden("time_speed",   key_moment);   // superseded by km_duration
  state::setFieldHidden("ease",         key_moment);   // superseded by km_ease
  state::setFieldHidden("interp_cells", key_moment);   // KM mode snaps to one cell
  // Skip-empty jogs the continuous loop clock — inert (and hidden) in KM mode.
  state::setFieldHidden("skip_empty",   key_moment);
  bool skip = skip_empty && !key_moment;
  state::setFieldHidden("skip_thresh",     !skip);
  state::setFieldHidden("skip_w_var",      !skip);
  state::setFieldHidden("skip_w_edge",     !skip);
  state::setFieldHidden("skip_w_motion",   !skip);
  state::setFieldHidden("skip_debug",      !skip);
  state::setFieldHidden("skip_recover",    !skip);
  state::setFieldHidden("skip_rate",       !skip);
  state::setFieldHidden("ap_speed",       !autopilot);
  state::setFieldHidden("ap_snap",        !autopilot);
  state::setFieldHidden("ap_hold_period", !(autopilot && ap_snap));
  state::setFieldHidden("ap_hold_jitter", !(autopilot && ap_snap));
  state::setFieldHidden("ap_jump",        !(autopilot && ap_snap));
  // Jogging the orbit only means anything under autopilot.
  state::setFieldHidden("skip_autopilot",  !(skip && autopilot));
}

// Static (self-less) visibility evaluator — pure over a candidate state.
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
static gpu::ComputePSO s_pso_present;
static gpu::ComputePSO s_pso_edge;      // Sobel/variance/motion reduce over tex_out
static gpu::ComputePSO s_pso_debug;     // per-tile feature heatmap (debug viz)

void module_init() {
  state::init("source.brutal_fold", {1, 1, 2},
    state::Schema()
      // Top-level manual: high-level "what is this / how to use / what to try".
      .helpField("intro",
        "## Brutal Fold\n"
        "A brutalist axonometric-prism generator — solid 3D forms **without a "
        "vanishing point**. Drag the XY pad to explore the baked atlas: **X** adds "
        "structural complexity, **Y** adds order.\n\n"
        "**Try:** keep *Speed* low (it reads best slow); add a little *Volumetrics* "
        "for a drifting fog blob; flip on *Autopilot* to let it wander on its own.")
      // --- Shape (the custom XY pad drives complexity + order) ---
      .group("shape", "Form")
        .groupHelp(
          "The **atlas** is a montage of pre-baked structures. *Complexity* (X) and "
          "*Order* (Y) interpolate between neighbouring cells; *Liveliness* picks how "
          "richly the cell animates. *Scale* zooms the form; *Balance* trades zoom "
          "between the two co-folded structures.")
      .floatField("complexity", 0.6f, 0.0f, 1.0f, state::PrimaryInput).label("Complexity", "Cplx")
      .floatField("order", 0.6f, 0.0f, 1.0f, state::PrimaryInput).label("Order", "Ord")
      // Liveliness (z): 0 = hold still → 1 = animate as richly as the cell allows.
      .floatField("liveliness", 1.0f, 0.0f, 1.0f, state::PrimaryInput).label("Liveliness", "Live")
      // --- Form ---
      .floatField("scale", 1.0f, 0.3f, 10.0f, state::PrimaryInput).label("Scale", "Scl")
      // Signed: zoom structure 1 IN as structure 2 zooms OUT (parallax balance).
      .floatField("balance", 0.0f, -1.5f, 1.5f, state::PrimaryInput).label("Balance", "Bal")
      // Recession depth — how far the prisms extrude back.
      .floatField("extrude", 1.0f, 0.0f, 6.0f, state::PrimaryInput).label("Extrude", "Extr")
      .boolField("second_structure", true, state::PrimaryInput).label("2nd Structure", "2nd")
      .boolField("interp_cells", true, state::PrimaryInput).label("Interpolate Cells", "Interp")
      // --- Animation ---
      .group("animation", "Animation")
      // Autoplay clock speed (0 = frozen). [0,1] with a QUADRATIC bend onto a low
      // actual max, so the bottom of the slider is mostly very-slow.
      .floatField("time_speed", 0.5f, 0.0f, 1.0f, state::PrimaryInput).label("Speed", "Spd")
      // Time-warp (bipolar). τ(t) = t − (ease/2π)·sin(2π t). +1 = rest at the
      // loop point, surge through the middle; 0 = uniform. |ease|≤1 keeps it monotone.
      .floatField("ease", 0.0f, -1.0f, 1.0f, state::PrimaryInput).label("Ease", "Ease")
      // In/out (birth) oscillation amplitude — how strongly prisms pop in and out.
      .floatField("anim_amount", 1.0f, 0.0f, 2.5f, state::PrimaryInput).label("Anim Amount", "Anim")
      // --- Key moment: play the analyzed inflow→peak window instead of the loop ---
      .group("keymoment", "Key Moment")
        .groupHelp(
          "Each atlas cell was scored offline for its **key moment** — the short "
          "window where new structure streams in and comes to frame the centre. "
          "Turn this on to play just that window (anchored on the settled peak) "
          "instead of the whole loop. **Time Mode** picks how it plays: *Trigger* "
          "fires it once per **Trigger** and holds on the peak; *Time* lets you "
          "scrub the window by hand; *Loop* replays it continuously (and **Trigger** "
          "restarts it). *Speed* sets the playback rate for Trigger/Loop. Cells with "
          "no clean key moment fall back to the full loop.")
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
      .eventField("km_trigger", state::PrimaryInput)
      .floatField("sky_threshold", 0.0f, 0.0f, 1.0f, state::PrimaryInput,
                  nullptr, /*step=*/0.01f, /*units=*/nullptr,
                  "Reachability filter: drop cells whose settled frame shows less than this "
                  "fraction of empty sky (less 'framed'), snapping to the nearest that passes. "
                  "0 keeps every cell.").label("Sky Threshold", "Sky")
      // Broadcast: the live playhead [0,1] within the window, for the editor readout.
      .floatField("km_phase", 0.0f, 0.0f, 1.0f, state::SecondaryOutput)
      // --- Atmosphere / colour grade ---
      .group("color", "Colour & Atmosphere")
      .floatField("fog", 1.0f, 0.0f, 5.0f, state::PrimaryInput).label("Fog", "Fog")
      // --- Colour grade (3-control-point tone→hue twist; sat 0 = grayscale) ---
      // Diffuse: hue at panel shadows / mids / highlights, + tint strength.
      .floatField("diff_hue_lo", 0.58f, 0.0f, 1.0f, state::PrimaryInput)
      .floatField("diff_hue_mid", 0.08f, 0.0f, 1.0f, state::PrimaryInput)
      .floatField("diff_hue_hi", 0.11f, 0.0f, 1.0f, state::PrimaryInput)
      .floatField("diff_sat", 0.0f, 0.0f, 1.0f, state::PrimaryInput)
      // Per-knot saturation + brightness (shadows / mids / highs).
      .floatField("diff_sat_lo", 1.0f, 0.0f, 1.0f, state::PrimaryInput)
      .floatField("diff_sat_mid", 1.0f, 0.0f, 1.0f, state::PrimaryInput)
      .floatField("diff_sat_hi", 1.0f, 0.0f, 1.0f, state::PrimaryInput)
      .floatField("diff_bri_lo", 1.0f, 0.0f, 2.0f, state::PrimaryInput)
      .floatField("diff_bri_mid", 1.0f, 0.0f, 2.0f, state::PrimaryInput)
      .floatField("diff_bri_hi", 1.0f, 0.0f, 2.0f, state::PrimaryInput)
      // Fog: hue at near / mid / far depth, + overall tint + per-knot saturation.
      .floatField("fog_hue_lo", 0.55f, 0.0f, 1.0f, state::PrimaryInput)
      .floatField("fog_hue_mid", 0.60f, 0.0f, 1.0f, state::PrimaryInput)
      .floatField("fog_hue_hi", 0.66f, 0.0f, 1.0f, state::PrimaryInput)
      .floatField("fog_sat", 0.0f, 0.0f, 1.0f, state::PrimaryInput)
      .floatField("fog_sat_lo", 1.0f, 0.0f, 1.0f, state::PrimaryInput)
      .floatField("fog_sat_mid", 1.0f, 0.0f, 1.0f, state::PrimaryInput)
      .floatField("fog_sat_hi", 1.0f, 0.0f, 1.0f, state::PrimaryInput)
      // Sky: additional twist on the infinite-distance pixels, relative to far.
      .floatField("sky_hue", 0.0f, -1.0f, 1.0f, state::PrimaryInput)
      .floatField("sky_sat", 0.0f, 0.0f, 1.0f, state::PrimaryInput)
      .floatField("sky_bri", 1.0f, 0.0f, 2.0f, state::PrimaryInput)
      // Stochastic "TV static" grain — separately on the blob and the distance fog.
      .floatField("noise_blob", 0.0f, 0.0f, 1.0f, state::PrimaryInput)
      .floatField("noise_blob_tilt", 0.0f, -1.0f, 1.0f, state::PrimaryInput)
      .floatField("noise_fog", 0.0f, 0.0f, 1.0f, state::PrimaryInput)
      // Reroll speed for BOTH statics: 0 = frozen, 1 ≈ 30 Hz.
      .floatField("noise_speed", 0.5f, 0.0f, 1.0f, state::PrimaryInput)
      // --- Volumetric fog: a 3D shape blob modulating the fog (0 = uniform) ---
      .group("volumetrics", "Volumetrics")
        .groupHelp(
          "A 3D **fog blob** that concentrates the depth fog into a shape instead of "
          "a uniform haze. *Amount* blends it in (0 = the old uniform fog). *Shape* "
          "morphs sphere → slab → solid; *Radius*, *Depth* and the two *Softness* "
          "knobs size and feather it. Pair with **Drift** to make it breathe.")
      .floatField("vol_amount", 0.0f, 0.0f, 1.0f, state::PrimaryInput).label("Amount", "Amt")
      .floatField("vol_anchor_x", 0.0f, -1.5f, 1.5f, state::PrimaryInput).label("Anchor X", "X")
      .floatField("vol_anchor_y", 0.0f, -1.5f, 1.5f, state::PrimaryInput).label("Anchor Y", "Y")
      .floatField("vol_z", 0.5f, 0.0f, 1.0f, state::PrimaryInput).label("Depth Anchor", "Z")
      // Bipolar: |1| sphere → |0.5| planar slab → |0| solid; sign flips polarity.
      .floatField("vol_shape", 1.0f, -1.0f, 1.0f, state::PrimaryInput).label("Shape", "Shp")
      // Turns; rotates the linear band in screen-space (depth is set by vol_depth).
      .floatField("vol_angle", 0.0f, 0.0f, 1.0f, state::PrimaryInput).label("Angle", "Ang")
      .floatField("vol_radius", 0.5f, 0.0f, 1.0f, state::PrimaryInput).label("Radius", "Rad")
      // Edge softness, split: screen-space (xy) and depth (z).
      .floatField("vol_softness_xy", 0.5f, 0.0f, 1.0f, state::PrimaryInput).label("Screen Softness", "Soft XY")
      .floatField("vol_softness_z", 0.5f, 0.0f, 1.0f, state::PrimaryInput).label("Depth Softness", "Soft Z")
      // Blob Z extent: small = very selective (a thin depth slice), large = spans
      // many depths. (vol_z is the Z anchor.) Normalized; maps to 0..kVolDepthMax.
      .floatField("vol_depth", 0.5f, 0.0f, 1.0f, state::PrimaryInput).label("Depth Extent", "Ext")
      // --- Drift: bounded 2nd-order random walk per quantity, + overall speed ---
      .group("drift", "Drift")
      .floatField("drift_xy", 0.0f, 0.0f, 1.0f, state::PrimaryInput).label("Drift XY", "XY")
      .floatField("drift_z", 0.0f, 0.0f, 1.0f, state::PrimaryInput).label("Drift Depth", "Z")
      .floatField("drift_shape", 0.0f, 0.0f, 1.0f, state::PrimaryInput).label("Drift Shape", "Shp")
      .floatField("drift_angle", 0.0f, 0.0f, 1.0f, state::PrimaryInput).label("Drift Angle", "Ang")
      .floatField("drift_speed", 0.3f, 0.0f, 1.0f, state::PrimaryInput).label("Drift Speed", "Spd")
      // Broadcast: the live (drifted) blob values, for the fog preview widget.
      .floatField("vol_x_live", 0.0f, -1.5f, 1.5f, state::SecondaryOutput)
      .floatField("vol_y_live", 0.0f, -1.5f, 1.5f, state::SecondaryOutput)
      .floatField("vol_z_live", 0.5f, 0.0f, 1.0f, state::SecondaryOutput)
      .floatField("vol_shape_live", 1.0f, -1.0f, 1.0f, state::SecondaryOutput)
      .floatField("vol_angle_live", 0.0f, 0.0f, 1.0f, state::SecondaryOutput)
      // --- Autopilot (non-destructive XY override + broadcast) ---
      .group("autopilot", "Autopilot")
        .groupHelp(
          "Spirals the shape's XY position on its own, **without touching** your "
          "Complexity/Order inputs (it broadcasts the live position instead). *Speed* "
          "sets the orbit rate. Turn on **Snap** to hop between held positions every "
          "*Hold* seconds (with optional *Jitter*), or fire **Jump** to leap manually.")
      .boolField("autopilot", false, state::PrimaryInput).label("Autopilot", "Auto")
      .floatField("ap_speed", 0.43f, 0.0f, 1.0f, state::PrimaryInput).label("Orbit Speed", "Spd")
      .boolField("ap_snap", false, state::PrimaryInput).label("Snap", "Snap")
      .floatField("ap_hold_period", 2.0f, 0.0f, 8.0f, state::PrimaryInput).label("Hold Period", "Hold")
      .floatField("ap_hold_jitter", 0.0f, 0.0f, 1.0f, state::PrimaryInput).label("Hold Jitter", "Jit")
      .eventField("ap_jump", state::PrimaryInput)
      // Broadcast: the effective XY so the custom editor can show the live position.
      .floatField("autopilot_x", 0.6f, 0.0f, 1.0f, state::SecondaryOutput)
      .floatField("autopilot_y", 0.6f, 0.0f, 1.0f, state::SecondaryOutput)
      // --- Skip empty: jog past dead (flat solid-colour) stretches ---
      .group("skip", "Skip Empty")
        .groupHelp(
          "The atlas and the in/out animation sometimes settle into a **flat, "
          "near-solid-colour** frame — all sky, or one solid panel. Turn this on "
          "and the effect gently **jogs the loop forward** (on a smooth C2 curve) to "
          "glide past those dead stretches instead of dwelling on them. Detection is "
          "by low visual **variance**, so it catches a screen full of one flat shape "
          "as readily as an empty sky. *Sensitivity* sets how flat a frame has to get "
          "before it kicks in; *Jog Rate* how "
          "briskly it skips. With **Autopilot** on it also nudges the orbit onward — "
          "or, if **Snap** is on, hops straight to a fresh position the moment it "
          "goes empty.")
      .boolField("skip_empty", false, state::PrimaryInput).label("Skip Empty", "Skip")
      .floatField("skip_thresh", 0.7f, 0.0f, 1.0f, state::PrimaryInput,
                  nullptr, /*step=*/0.01f, /*units=*/nullptr,
                  "How readily a frame counts as flat/empty. Higher flags more scenes "
                  "(lower visual variance tolerated); 1 catches almost anything but the "
                  "busiest frames.").label("Sensitivity", "Sens")
      // Per-feature weights, combined by weighted MAX. Any weighted feature clearing
      // the trigger keeps the frame from reading as flat. 0 disables that feature.
      .floatField("skip_w_var", 0.0f, 0.0f, 1.0f, state::PrimaryInput,
                  nullptr, /*step=*/0.01f, /*units=*/nullptr,
                  "Weight of local tonal VARIANCE in the flatness test.")
                  .label("Variance Wt", "Var")
      .floatField("skip_w_edge", 0.07f, 0.0f, 1.0f, state::PrimaryInput,
                  nullptr, /*step=*/0.01f, /*units=*/nullptr,
                  "Weight of spatial EDGES (hard face/sky boundaries) in the flatness test.")
                  .label("Edge Wt", "Edge")
      .floatField("skip_w_motion", 1.0f, 0.0f, 1.0f, state::PrimaryInput,
                  nullptr, /*step=*/0.01f, /*units=*/nullptr,
                  "Weight of MOTION (frame-to-frame change) in the flatness test — keeps "
                  "the jog running only while the frame is both flat AND still.")
                  .label("Motion Wt", "Motn")
      .selectField("skip_debug", 0, state::PrimaryInput,
                   {{"Off", 0}, {"Variance", 1}, {"Edge", 2}, {"Motion", 3}, {"Combined", 4}})
                  .label("Debug View", "Dbg")
      .floatField("skip_recover", 1.0f, 0.0f, 1.0f, state::PrimaryInput,
                  nullptr, /*step=*/0.01f, /*units=*/nullptr,
                  "How fast the jog STOPS once content reappears (the empty→happening "
                  "transition). Higher = snappier so it doesn't skip past the content; "
                  "1 = instant hard stop. Lower eases out gently. (Onset stays a gentle "
                  "C2 glide.)").label("Recover", "Rec")
      .floatField("skip_rate", 0.5f, 0.0f, 1.0f, state::PrimaryInput).label("Jog Rate", "Rate")
      .boolField("skip_autopilot", true, state::PrimaryInput).label("Jog Autopilot", "JogAP")
      // Broadcast: the live engagement [0,1] so an editor can show when it fires.
      .floatField("skip_active", 0.0f, 0.0f, 1.0f, state::SecondaryOutput)
      // Broadcast: the live flatness metric (luminance std/edge blend) for tuning.
      .floatField("skip_variance", 0.0f, 0.0f, 1.0f, state::SecondaryOutput)
      // --- I/O: pure generator (no input) ---
      .textureField("tex_out", state::PrimaryOutput)
        .capability(state::Capability::Generator)
        .capability(state::Capability::SeekableApproximate)
    );

  if (gpu::Device::backend() == gpu::Backend::None) return;

  state::registerShaderSPV("brutal_fold_present", PRESENT_SPV, PRESENT_SPV_SIZE);
  state::registerShaderSPV("brutal_fold_edge", EDGE_SPV, EDGE_SPV_SIZE);
  state::registerShaderSPV("brutal_fold_debug", DEBUG_SPV, DEBUG_SPV_SIZE);

  auto cs_present = gpu::Device::createShaderModuleByName("brutal_fold_present");
  if (!cs_present) return;

  s_pso_present = gpu::Device::createComputePSO(cs_present, "main", gpu::Bindings()
      .uniform(0)
      .storageTex2d(1));                            // tex_out

  auto cs_edge = gpu::Device::createShaderModuleByName("brutal_fold_edge");
  if (!cs_edge) return;
  s_pso_edge = gpu::Device::createComputePSO(cs_edge, "main", gpu::Bindings()
      .uniform(0)      // shared cbuffer U (res_x/res_y)
      .tex2d(1)        // tex_out (read)
      .storageRW(2)    // stats (read-write atomics)
      .storageRW(3));  // prevLuma (per-sample motion history)

  auto cs_debug = gpu::Device::createShaderModuleByName("brutal_fold_debug");
  if (!cs_debug) return;
  s_pso_debug = gpu::Device::createComputePSO(cs_debug, "main", gpu::Bindings()
      .uniform(0)      // debug uniform
      .storage(1)      // stats (read)
      .storageTex2d(2));                            // tex_out (write viz)

  state::log("brutal_fold: module initialized");
}

// Distinct PRNG seed per instance (no wall-clock / RNG primitive needed).
static uint32_t s_seed_counter = 0x9E3779B9u;

void* create() {
  auto* s = new State();
  s_seed_counter = s_seed_counter * 1664525u + 1013904223u;
  s->rng = s_seed_counter ^ 0xC0FFEEu;
  s->uniform_buf = gpu::Device::createBuffer(sizeof(Uniforms), gpu::BufferUsage::Uniform);
  s->stats_buf = gpu::Device::createBuffer(kStatsInts * sizeof(int32_t), gpu::BufferUsage::Storage);
  s->prev_luma_buf = gpu::Device::createBuffer(kSampleGrid * kSampleGrid * sizeof(float), gpu::BufferUsage::Storage);
  s->debug_uniform_buf = gpu::Device::createBuffer(sizeof(DebugUniforms), gpu::BufferUsage::Uniform);
  return s;
}

void destroy(void* self) {
  auto* s = static_cast<State*>(self);
  if (!s) return;
  s->uniform_buf.release();
  s->stats_buf.release();
  s->prev_luma_buf.release();
  s->debug_uniform_buf.release();
  delete s;
}

void init(void* self) {
  auto* s = static_cast<State*>(self);
  if (!s) return;
  s->initialized = false;
  if (!s_pso_present.valid() || !s_pso_edge.valid() || !s_pso_debug.valid()) return;
  if (!s->uniform_buf.valid() || !s->stats_buf.valid() ||
      !s->prev_luma_buf.valid() || !s->debug_uniform_buf.valid()) return;
  // Zero the stats so the first poll (before any edge pass has run) reads count=0
  // and the detector stays on the CPU fallback until a real readback arrives.
  int32_t z[kStatsInts] = {};
  s->stats_buf.write(z, kStatsInts);
  // Sentinel <0 so the first frame's motion reads 0 (no spurious spike).
  std::vector<float> negs(kSampleGrid * kSampleGrid, -1.0f);
  s->prev_luma_buf.write(negs.data(), (int)negs.size());
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
static inline float smoothstepf(float e0, float e1, float x) {
  float t = clampf((x - e0) / (e1 - e0 + 1e-9f), 0.0f, 1.0f);
  return t * t * (3.0f - 2.0f * t);
}

// Per-instance uniform random in [0,1) (LCG).
static inline float rng_unit(State* s) {
  s->rng = s->rng * 1664525u + 1013904223u;
  return (float)((s->rng >> 8) & 0xFFFFFFu) / (float)0x1000000;
}

// One step of a bounded second-order random walk: the velocity gets a random
// kick and mean-reverts (spring) with damping; the offset integrates velocity
// and hard-reflects at ±(amount). `speed` scales the evolution rate. Returns the
// new offset (also writes back o, v). amount<=0 parks it at 0.
static inline float drift_step(State* s, float& o, float& v, float amount,
                               float speed, float fdt) {
  if (amount <= 1e-6f) { o = 0.0f; v = 0.0f; return 0.0f; }
  float rate = speed * kDriftMaxSpeed * kDriftRate;
  float kick = (2.0f * rng_unit(s) - 1.0f) * kDriftSigma;
  v += (kick - kDriftSpring * o) * rate * fdt;
  v *= kDriftDamp;
  o += v * rate * fdt;
  if (o > amount)  { o = amount;  v = -v * 0.4f; }
  if (o < -amount) { o = -amount; v = -v * 0.4f; }
  return o;
}

// Epicycle position at a given orbit phase. Clamped to stay inside the pad.
static inline void orbit_xy(float orbit, float& ox, float& oy) {
  float a1 = orbit;
  float a2 = orbit * kApW2 + kApPhi;
  ox = clampf(0.5f + kApA * std::cos(a1) + kApB * std::cos(a2), 0.03f, 0.97f);
  oy = clampf(0.5f + kApA * std::sin(a1) + kApB * std::sin(a2), 0.03f, 0.97f);
}

// Resolve the KM atlas cell for the current XY: snap to the nearest grid cell,
// then (if a sky threshold is set) hop to the nearest cell whose settled frame is
// "framed" enough — sky fraction >= threshold. Mirrors field.nearestCell +
// nearestAboveSky in the web testbed. Writes the snapped (gi, gj).
static void km_resolve(const AtlasRef& A, const State* s, int* gi_out, int* gj_out) {
  const int G = A.grid;
  int gj = clampi((int)std::lround(clampf(s->eff_x, 0.0f, 1.0f) * (G - 1)), 0, G - 1);   // col = complexity
  int gi = clampi((int)std::lround(clampf(s->eff_y, 0.0f, 1.0f) * (G - 1)), 0, G - 1);   // row = order
  float thr = s->sky_threshold;
  if (thr > 0.0f && A.km_sky && A.km_sky[gi * G + gj] < thr) {
    int best = gi * G + gj; float bestD = 1e30f;
    for (int i = 0; i < G; i++) for (int j = 0; j < G; j++) {
      int c = i * G + j;
      if (A.km_sky[c] >= thr) {
        float d = (float)((i - gi) * (i - gi) + (j - gj) * (j - gj));
        if (d < bestD) { bestD = d; best = c; }
      }
    }
    gi = best / G; gj = best % G;
  }
  *gi_out = gi; *gj_out = gj;
}

// easeOut on the window playhead: slow toward the settled peak. Mirrors main.ts —
// e>0 eases out, e<=0 stays linear (reuses the effect's existing time lever).
static inline float km_ease_out(float u, float e) {
  u = clampf(u, 0.0f, 1.0f);
  return e <= 0.0f ? u : 1.0f - std::pow(1.0f - u, 1.0f + e);
}

// Advance the internal clocks and compute the effective (broadcast) XY.
void tick(void* self, double dt) {
  auto* s = static_cast<State*>(self);
  if (!s) return;
  float fdt = (float)dt;
  // Advance the TV-static reroll phase: speed=0 freezes it, speed=1 ≈ 30 Hz
  // (matches source.noise). floor(phase) is the per-frame seed in the shader.
  s->noise_phase += fdt * (s->noise_speed * 30.0f);
  if (s->noise_phase > 1.0e6f) s->noise_phase -= 1.0e6f;

  // Quadratic speed response onto a low actual max — fine control at the slow end.
  float ts = s->time_speed;
  float time_actual = ts * ts * kTimeMax;                      // loops/sec
  float aps = s->ap_speed;
  float ap_actual = kApMin + aps * aps * (kApMax - kApMin);

  // Skip-empty: engage the C2 ramp on last frame's front content. It always jogs
  // the loop clock forward; under autopilot it also nudges the orbit (or, in Snap
  // mode, fires a one-shot hop). `content` lags a frame — that's fine (the ramp
  // is smooth and hysteretic, and it reuses render()'s field scan for free).
  float skip_e = 0.0f;
  if (s->skip_empty && !s->key_moment) {   // KM windows are curated — no jog
    // Prefer the GPU flatness metric (Sobel/variance over the REAL rendered
    // frame, computed by last frame's edge pass) once a readback has arrived.
    // Until then s->content holds the CPU proxy from bf_build_params. Poll before
    // the jog so this frame's decision uses the freshest available metric.
    int32_t raw[kStatsInts];
    if (s->stats_buf.pollReadback(raw, sizeof(raw)) == (int)sizeof(raw)) {
      // Reduce PER TILE and take the weighted MAX over three features: any single
      // region with local luma variance OR a concentrated edge OR motion makes the
      // whole frame read as non-flat.
      float wv = clampf(s->skip_w_var, 0.0f, 1.0f);
      float we = clampf(s->skip_w_edge, 0.0f, 1.0f);
      float wm = clampf(s->skip_w_motion, 0.0f, 1.0f);
      float content = 0.0f;
      bool any = false;
      for (int t = 0; t < kNumTiles; t++) {
        int base = t * kSlots;
        int cnt = raw[base + 4];
        if (cnt <= 0) continue;
        any = true;
        float ne = (float)cnt;
        float lu = ((float)raw[base + 1] / kStatsScale) / ne;    // tile mean luma
        float lq = ((float)raw[base + 2] / kStatsScale) / ne;    // tile mean luma²
        float var = lq - lu * lu; if (var < 0.0f) var = 0.0f;
        float sd = std::sqrt(var);
        sd = sd > kVarFloor ? sd - kVarFloor : 0.0f;             // local luma std (deadzoned)
        // Soft edge/motion sums (contrast-squashed) / √tilePixels: a concentrated
        // feature saturates regardless of the area it covers.
        float edge   = clampf((float)raw[base + 0] / kStatsScale / std::sqrt(ne) * kEdgeNormGain, 0.0f, 1.0f);
        float motion = clampf((float)raw[base + 3] / kStatsScale / std::sqrt(ne) * kEdgeNormGain, 0.0f, 1.0f);
        float local = sd * wv;
        float ew = edge * we;   if (ew > local) local = ew;
        float mw = motion * wm; if (mw > local) local = mw;
        if (local > content) content = local;                    // max over tiles
      }
      if (any) { s->content = content; s->skip_gpu_ready = true; }
    }
    // Sensitivity [0,1] → luminance-std trigger. Higher = flags more scenes as flat.
    float lo = clampf(s->skip_thresh, 0.0f, 1.0f) * kSkipStdSpan;
    float hi = lo + kSkipHyst;
    // Recover controls the ease-OUT (content→stop); 1 = instant (rampOut≈0 → snap).
    float rampOut = (1.0f - clampf(s->skip_recover, 0.0f, 1.0f)) * kMaxRecoverSec;
    skip_e = s->jog.update(s->content, lo, hi, kSkipRampInSec, rampOut, fdt);
    // Snap mode: the moment we go empty, hop straight to a fresh held position.
    if (s->autopilot && s->skip_autopilot && s->ap_snap && s->jog.rising())
      s->ap_jump_pending = true;
  } else {
    s->jog = fx::SkipJog{};          // fully reset when disabled (no residual jog)
  }
  // Jog is a MULTIPLE of the current rate, so Speed 0 stays frozen (no auto-play)
  // and a faster Speed skips proportionally faster.
  float time_jog = skip_e * s->skip_rate * time_actual * kSkipTimeMult;
  // Extra orbit advance from the orbit jog — only in continuous (non-Snap) mode.
  float orbit_jog = (s->skip_autopilot && !s->ap_snap)
                        ? skip_e * s->skip_rate * ap_actual * kSkipOrbitMult : 0.0f;

  s->clock_t += fdt * (time_actual + time_jog);
  s->clock_t -= std::floor(s->clock_t);

  if (s->autopilot) {
    s->orbit -= fdt * (ap_actual + orbit_jog);   // clockwise drift (+ skip jog)

    bool trig = s->ap_jump_pending;
    s->ap_jump_pending = false;

    if (s->ap_snap) {
      bool hold_on = s->ap_hold_period > 1e-4f;
      bool jump = trig || !s->held_valid;        // trigger or first frame
      if (hold_on) {
        s->snap_accum += fdt;
        if (s->snap_accum >= s->next_hold) jump = true;
      }
      if (jump) {
        if (trig && !hold_on && s->held_valid) s->orbit -= kGoldenAngle;
        orbit_xy(s->orbit, s->held_x, s->held_y);
        s->held_valid = true;
        s->snap_accum = 0.0f;
        float jit = s->ap_hold_jitter * (2.0f * rng_unit(s) - 1.0f);
        s->next_hold = s->ap_hold_period * (1.0f + jit);
        if (s->next_hold < 0.02f) s->next_hold = 0.02f;
      }
      s->eff_x = s->held_x; s->eff_y = s->held_y;
    } else {
      if (trig) s->orbit -= kGoldenAngle;
      orbit_xy(s->orbit, s->eff_x, s->eff_y);
      s->held_valid = false;
    }
  } else {
    s->eff_x = s->complexity;
    s->eff_y = s->order;
    s->held_valid = false;
    s->snap_accum = 0.0f;
    s->ap_jump_pending = false;
  }

  // --- Key-moment playhead. Loop/Trigger advance km_u 0→1 over km_duration
  //     seconds; Time reads the manual km_time. (The scene cell + window are
  //     resolved in render from the KM atlas.) ---
  if (s->key_moment) {
    float rate = 1.0f / std::max(s->km_duration, 0.05f);   // playhead units/sec
    if (s->km_time_mode == 1) {                       // Time — manual scrub
      s->km_u = clampf(s->km_time, 0.0f, 1.0f);
    } else if (s->km_time_mode == 2) {                // Loop — continuous replay
      s->km_u += fdt * rate;
      s->km_u -= std::floor(s->km_u);
    } else {                                          // Trigger — one-shot, hold on the peak
      if (s->km_playing) {
        s->km_u += fdt * rate;
        if (s->km_u >= 1.0f) { s->km_u = 1.0f; s->km_playing = false; }
      }
    }
    s->km_phase = s->km_u;
  }

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
  auto vsv = val::number(s->content);   // live flatness metric (for calibration)
  state::setValPath("skip_variance", vsv);
  val::release(vsv);
  auto vkm = val::number(s->km_phase);  // live key-moment playhead (for the editor)
  state::setValPath("km_phase", vkm);
  val::release(vkm);

  // --- Volumetric blob drift (bounded 2nd-order random walks) ---
  float sp = s->drift_speed;
  float ox = drift_step(s, s->dx_o, s->dx_v, s->drift_xy, sp, fdt);
  float oy = drift_step(s, s->dy_o, s->dy_v, s->drift_xy, sp, fdt);
  float oz = drift_step(s, s->dz_o, s->dz_v, s->drift_z, sp, fdt);
  float osh = drift_step(s, s->dsh_o, s->dsh_v, s->drift_shape, sp, fdt);
  float oan = drift_step(s, s->dan_o, s->dan_v, s->drift_angle, sp, fdt);
  // Depth drift is sensitive → low-pass it so it glides instead of jumping.
  float zk = clampf(fdt * kDepthDriftSmoothRate, 0.0f, 1.0f);
  s->dz_smooth += (oz - s->dz_smooth) * zk;
  s->eff_vol_x = clampf(s->vol_anchor_x + ox, -1.5f, 1.5f);
  s->eff_vol_y = clampf(s->vol_anchor_y + oy, -1.5f, 1.5f);
  s->eff_vol_z = clampf(s->vol_z + s->dz_smooth, 0.0f, 1.0f);
  s->eff_vol_shape = clampf(s->vol_shape + osh, -1.0f, 1.0f);
  s->eff_vol_angle = s->vol_angle + oan;   // periodic — shader wraps via cos/sin

  // Broadcast the live (drifted) blob values for the fog preview widget.
  auto blx = val::number(s->eff_vol_x);     state::setValPath("vol_x_live", blx);     val::release(blx);
  auto bly = val::number(s->eff_vol_y);     state::setValPath("vol_y_live", bly);     val::release(bly);
  auto blz = val::number(s->eff_vol_z);     state::setValPath("vol_z_live", blz);     val::release(blz);
  auto bls = val::number(s->eff_vol_shape); state::setValPath("vol_shape_live", bls); val::release(bls);
  auto bla = val::number(s->eff_vol_angle - std::floor(s->eff_vol_angle));
  state::setValPath("vol_angle_live", bla); val::release(bla);
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
    if      (state::pathIs(path, plen, "complexity"))        s->complexity = state::patchFloat(i);
    else if (state::pathIs(path, plen, "order"))             s->order = state::patchFloat(i);
    else if (state::pathIs(path, plen, "liveliness"))        s->liveliness = state::patchFloat(i);
    else if (state::pathIs(path, plen, "scale"))             s->scale = state::patchFloat(i);
    else if (state::pathIs(path, plen, "balance"))           s->balance = state::patchFloat(i);
    else if (state::pathIs(path, plen, "extrude"))           s->extrude = state::patchFloat(i);
    else if (state::pathIs(path, plen, "second_structure"))  s->second_structure = state::patchFloat(i) != 0.0f;
    else if (state::pathIs(path, plen, "interp_cells"))      s->interp_cells = state::patchFloat(i) != 0.0f;
    else if (state::pathIs(path, plen, "time_speed"))        s->time_speed = state::patchFloat(i);
    else if (state::pathIs(path, plen, "ease"))              s->ease = state::patchFloat(i);
    else if (state::pathIs(path, plen, "anim_amount"))       s->anim_amount = state::patchFloat(i);
    else if (state::pathIs(path, plen, "fog"))               s->fog = state::patchFloat(i);
    else if (state::pathIs(path, plen, "diff_hue_lo"))       s->diff_hue_lo = state::patchFloat(i);
    else if (state::pathIs(path, plen, "diff_hue_mid"))      s->diff_hue_mid = state::patchFloat(i);
    else if (state::pathIs(path, plen, "diff_hue_hi"))       s->diff_hue_hi = state::patchFloat(i);
    else if (state::pathIs(path, plen, "diff_sat"))          s->diff_sat = state::patchFloat(i);
    else if (state::pathIs(path, plen, "diff_sat_lo"))       s->diff_sat_lo = state::patchFloat(i);
    else if (state::pathIs(path, plen, "diff_sat_mid"))      s->diff_sat_mid = state::patchFloat(i);
    else if (state::pathIs(path, plen, "diff_sat_hi"))       s->diff_sat_hi = state::patchFloat(i);
    else if (state::pathIs(path, plen, "diff_bri_lo"))       s->diff_bri_lo = state::patchFloat(i);
    else if (state::pathIs(path, plen, "diff_bri_mid"))      s->diff_bri_mid = state::patchFloat(i);
    else if (state::pathIs(path, plen, "diff_bri_hi"))       s->diff_bri_hi = state::patchFloat(i);
    else if (state::pathIs(path, plen, "fog_hue_lo"))        s->fog_hue_lo = state::patchFloat(i);
    else if (state::pathIs(path, plen, "fog_hue_mid"))       s->fog_hue_mid = state::patchFloat(i);
    else if (state::pathIs(path, plen, "fog_hue_hi"))        s->fog_hue_hi = state::patchFloat(i);
    else if (state::pathIs(path, plen, "fog_sat"))           s->fog_sat = state::patchFloat(i);
    else if (state::pathIs(path, plen, "fog_sat_lo"))        s->fog_sat_lo = state::patchFloat(i);
    else if (state::pathIs(path, plen, "fog_sat_mid"))       s->fog_sat_mid = state::patchFloat(i);
    else if (state::pathIs(path, plen, "fog_sat_hi"))        s->fog_sat_hi = state::patchFloat(i);
    else if (state::pathIs(path, plen, "sky_hue"))           s->sky_hue = state::patchFloat(i);
    else if (state::pathIs(path, plen, "sky_sat"))           s->sky_sat = state::patchFloat(i);
    else if (state::pathIs(path, plen, "sky_bri"))           s->sky_bri = state::patchFloat(i);
    else if (state::pathIs(path, plen, "noise_blob"))        s->noise_blob = state::patchFloat(i);
    else if (state::pathIs(path, plen, "noise_blob_tilt"))   s->noise_blob_tilt = state::patchFloat(i);
    else if (state::pathIs(path, plen, "noise_fog"))         s->noise_fog = state::patchFloat(i);
    else if (state::pathIs(path, plen, "noise_speed"))       s->noise_speed = state::patchFloat(i);
    else if (state::pathIs(path, plen, "vol_amount"))        s->vol_amount = state::patchFloat(i);
    else if (state::pathIs(path, plen, "vol_anchor_x"))      s->vol_anchor_x = state::patchFloat(i);
    else if (state::pathIs(path, plen, "vol_anchor_y"))      s->vol_anchor_y = state::patchFloat(i);
    else if (state::pathIs(path, plen, "vol_z"))             s->vol_z = state::patchFloat(i);
    else if (state::pathIs(path, plen, "vol_shape"))         s->vol_shape = state::patchFloat(i);
    else if (state::pathIs(path, plen, "vol_angle"))         s->vol_angle = state::patchFloat(i);
    else if (state::pathIs(path, plen, "vol_radius"))        s->vol_radius = state::patchFloat(i);
    else if (state::pathIs(path, plen, "vol_softness_xy"))   s->vol_softness_xy = state::patchFloat(i);
    else if (state::pathIs(path, plen, "vol_softness_z"))    s->vol_softness_z = state::patchFloat(i);
    else if (state::pathIs(path, plen, "vol_depth"))         s->vol_depth = state::patchFloat(i);
    else if (state::pathIs(path, plen, "drift_xy"))          s->drift_xy = state::patchFloat(i);
    else if (state::pathIs(path, plen, "drift_z"))           s->drift_z = state::patchFloat(i);
    else if (state::pathIs(path, plen, "drift_shape"))       s->drift_shape = state::patchFloat(i);
    else if (state::pathIs(path, plen, "drift_angle"))       s->drift_angle = state::patchFloat(i);
    else if (state::pathIs(path, plen, "drift_speed"))       s->drift_speed = state::patchFloat(i);
    else if (state::pathIs(path, plen, "autopilot"))         { bool v = state::patchFloat(i) != 0.0f; if (v != s->autopilot) { s->autopilot = v; mode_changed = true; } }
    else if (state::pathIs(path, plen, "ap_speed"))          s->ap_speed = state::patchFloat(i);
    else if (state::pathIs(path, plen, "ap_snap"))           { bool v = state::patchFloat(i) != 0.0f; if (v != s->ap_snap) { s->ap_snap = v; mode_changed = true; } }
    else if (state::pathIs(path, plen, "ap_hold_period"))    s->ap_hold_period = state::patchFloat(i);
    else if (state::pathIs(path, plen, "ap_hold_jitter"))    s->ap_hold_jitter = state::patchFloat(i);
    else if (state::pathIs(path, plen, "ap_jump")) {
      float v = state::patchFloat(i);
      if (v != 0.0f && s->ap_jump_prev == 0.0f) s->ap_jump_pending = true;  // rising edge
      s->ap_jump_prev = v;
    }
    else if (state::pathIs(path, plen, "skip_empty"))    { bool v = state::patchFloat(i) != 0.0f; if (v != s->skip_empty) { s->skip_empty = v; mode_changed = true; } }
    else if (state::pathIs(path, plen, "skip_thresh"))   s->skip_thresh = state::patchFloat(i);
    else if (state::pathIs(path, plen, "skip_w_var"))    s->skip_w_var = state::patchFloat(i);
    else if (state::pathIs(path, plen, "skip_w_edge"))   s->skip_w_edge = state::patchFloat(i);
    else if (state::pathIs(path, plen, "skip_w_motion")) s->skip_w_motion = state::patchFloat(i);
    else if (state::pathIs(path, plen, "skip_debug"))    s->skip_debug = state::patchInt(i);
    else if (state::pathIs(path, plen, "skip_recover"))  s->skip_recover = state::patchFloat(i);
    else if (state::pathIs(path, plen, "skip_rate"))     s->skip_rate = state::patchFloat(i);
    else if (state::pathIs(path, plen, "skip_autopilot")) s->skip_autopilot = state::patchFloat(i) != 0.0f;
    else if (state::pathIs(path, plen, "key_moment")) { bool v = state::patchFloat(i) != 0.0f; if (v != s->key_moment) { s->key_moment = v; mode_changed = true; } }
    else if (state::pathIs(path, plen, "km_time_mode")) { int v = state::patchInt(i); if (v != s->km_time_mode) { s->km_time_mode = v; mode_changed = true; } }
    else if (state::pathIs(path, plen, "km_time")) s->km_time = state::patchFloat(i);
    else if (state::pathIs(path, plen, "km_duration")) s->km_duration = state::patchFloat(i);
    else if (state::pathIs(path, plen, "km_ease")) s->km_ease = state::patchFloat(i);
    else if (state::pathIs(path, plen, "sky_threshold")) s->sky_threshold = state::patchFloat(i);
    else if (state::pathIs(path, plen, "km_trigger")) {
      float v = state::patchFloat(i);
      if (v != 0.0f && s->km_trigger_prev == 0.0f) { s->km_u = 0.0f; s->km_playing = true; }  // rising edge → (re)start
      s->km_trigger_prev = v;
    }
  }
  if (mode_changed) apply_visibility(s->autopilot, s->ap_snap, s->skip_empty, s->key_moment, s->km_time_mode);
}

// ---- CPU atlas resolve (port of field.ts: corners / occ / levelFor / build) ----

struct Corners { int idx[4]; float w[4]; };

static Corners bf_corners(const AtlasRef& A, float sx, float sy, bool interp) {
  const int G = A.grid;
  sx = clampf(sx, 0.0f, 1.0f); sy = clampf(sy, 0.0f, 1.0f);
  float fx = sx * (G - 1), fy = sy * (G - 1);
  int x0, x1, y0, y1; float tx, ty;
  if (interp) {
    x0 = clampi((int)std::floor(fx), 0, G - 1); x1 = std::min(x0 + 1, G - 1); tx = fx - x0;
    y0 = clampi((int)std::floor(fy), 0, G - 1); y1 = std::min(y0 + 1, G - 1); ty = fy - y0;
    tx = tx * tx * (3 - 2 * tx); ty = ty * ty * (3 - 2 * ty); // smoothstep
  } else {
    x0 = x1 = clampi((int)std::lround(fx), 0, G - 1);
    y0 = y1 = clampi((int)std::lround(fy), 0, G - 1);
    tx = ty = 0;
  }
  Corners c;
  c.idx[0] = y0 * G + x0; c.idx[1] = y0 * G + x1; c.idx[2] = y1 * G + x0; c.idx[3] = y1 * G + x1;
  c.w[0] = (1 - tx) * (1 - ty); c.w[1] = tx * (1 - ty); c.w[2] = (1 - tx) * ty; c.w[3] = tx * ty;
  return c;
}

// Occupancy O at field coords (wx,wy) for the structure based at P[base..].
static float bf_occ(const float* P, int base, float wx, float wy) {
  float uni = 0, inter = 1;
  float dc = P[base + SB1 + S_DC], bold = P[base + SB1 + S_BOLD_GAIN];
  for (int i = 0; i < BF_NTERMS; i++) {
    int o = base + i * 8;
    float th = P[o], mth = P[o + 1], sh = P[o + 2], fr = P[o + 3], ph = P[o + 4],
          h = P[o + 5], amp = P[o + 6], mix = P[o + 7];
    float d = std::cos(th) * wx + std::sin(th) * wy + sh * (std::cos(mth) * wx + std::sin(mth) * wy);
    float q = d * fr + ph; q = q - std::floor(q) - 0.5f;
    float box = (h - std::fabs(q) > 0.0f) ? 1.0f : 0.0f;
    float a = clampf(h / BF_H_ACT, 0.0f, 1.0f);
    if (mix >= 0.5f) inter *= (1 - a) + a * box; else uni += amp * box;
  }
  return dc + uni + bold * inter;
}

// Solid threshold lo + thresh*(hi-lo) over the form's value range (coarse grid).
static float bf_levelFor(const float* P, int base, float thresh) {
  float fs = P[base + SB1 + S_FORM_SCALE];
  float mn = 1e9f, mx = -1e9f;
  const int NG = 24;
  for (int gi = 0; gi < NG; gi++) {
    for (int gj = 0; gj < NG; gj++) {
      float o = bf_occ(P, base, ((gi + 0.5f) / NG - 0.5f) * fs, ((gj + 0.5f) / NG - 0.5f) * fs);
      if (o < mn) mn = o;
      if (o > mx) mx = o;
    }
  }
  return mn + thresh * (mx - mn);
}

// Rendered-luminance proxy at screen point (px,py) — in present-shader p0 units,
// [-1,1] over the visible window. Mirrors bf_fieldVal: walk the receding layers of
// BOTH structures near→far, take the frontmost solid one, compute its face tone
// (front / recession-revealed top vs side, exactly as bf_drawLayer's gradient
// split), fade it toward the sky by depth fog. Sky where nothing is solid. Tilt
// and window-detail micro-texture are dropped (immaterial to a flatness measure);
// the volumetric blob is ignored (an optional feature, off by default).
static float bf_sample_lum(const float* P, bool on2, float px, float py) {
  const int sb1 = SB1;
  float sky1 = P[sb1 + S_SKY_VAL];
  float fog  = P[sb1 + S_FOG];
  float bl = P[sb1 + S_BACK_LEN], ba = P[sb1 + S_BACK_ANG], sep = P[sb1 + S_SEP];
  float exR = bl * std::cos(ba), eyR = -bl * std::sin(ba);   // recession vector
  float cba = std::cos(ba), sba = std::sin(ba);
  int n1 = clampi((int)P[sb1 + S_LAYERS], 1, kMaxLayers);
  int n2 = on2 ? clampi((int)P[STR + sb1 + S_LAYERS], 1, kMaxLayers) : 0;
  float nm1 = std::max((float)n1 - 1.0f, 1.0f);
  float nm2 = std::max((float)n2 - 1.0f, 1.0f);
  for (int s = 0; s < 2 * kMaxLayers; s++) {   // near → far, interleaved S1 / S2
    int base; float d, depthT;
    if ((s & 1) == 0) {
      int dd = s / 2; if (dd >= n1) continue;
      base = 0; d = (float)dd; depthT = d / nm1;
    } else {
      if (!on2) continue;
      int lidx = (s - 1) / 2; if (lidx >= n2) continue;
      base = STR; d = (float)s * 0.5f; depthT = (float)lidx / nm2;
    }
    int b = base + sb1;
    float fs = P[b + S_FORM_SCALE];
    float level = P[b + SCN];
    float face = P[b + S_FACE], sky = P[b + S_SKY_VAL];
    float frontT = sky - face;
    float topT = sky - face * 0.42f;
    float sideT = std::max(sky - face * 1.75f, 0.0f);
    float extr = P[b + S_EXTRUDE];
    float Ex = cba * extr, Ey = -sba * extr;             // extrude (back-face) vector
    float offx = exR * (d * sep), offy = eyR * (d * sep);
    float psx = px - offx, psy = py - offy;
    bool front = bf_occ(P, base, psx * fs, psy * fs) > level;
    float qx = psx - Ex, qy = psy - Ey;
    float br = bf_occ(P, base, qx * fs, qy * fs);
    bool back = br > level;
    if (!(front || back)) continue;                       // this layer not drawn here
    float t;
    if (!front) {
      float e = 0.02f;                                    // top vs side via gradient
      float g1 = bf_occ(P, base, (qx + e) * fs, qy * fs) - br;
      float g2 = bf_occ(P, base, qx * fs, (qy + e) * fs) - br;
      float sm = smoothstepf(0.45f, 0.7f,
                             std::fabs(g2) / (std::fabs(g1) + std::fabs(g2) + 1e-6f));
      t = lerpf(sideT, topT, sm);
    } else {
      t = frontT;
    }
    float fogv = clampf(fog * depthT, 0.0f, 1.0f);
    return lerpf(t, sky1, fogv);                          // fade to sky by depth
  }
  return sky1;                                            // nothing solid → sky
}

// "Not empty" estimate from the rendered luminance over the visible window,
// blending two structure signals per `edge_bias` (0 = variance, 1 = edges):
//   • VARIANCE  — the luminance std-dev. Low for any flat screen (all-sky, or one
//     solid tone), but a scene split into a few big flat blocks still scores high.
//   • EDGES     — RMS of the local (neighbour) gradient. This is what tells a busy
//     brutalist field (many hard face/sky boundaries) from a couple of big blocks:
//     large flat regions contribute no gradient, only their borders do.
// Both are in luminance units, so they blend and compare against the Sensitivity
// threshold on the same scale. 16×16 grid; paid only when the detector is on.
static float bf_content(const float* P, bool on2, float edge_bias) {
  const int NG = 16;
  float lum[NG * NG];
  for (int gi = 0; gi < NG; gi++) {
    for (int gj = 0; gj < NG; gj++) {
      float px = ((gi + 0.5f) / NG) * 2.0f - 1.0f;
      float py = ((gj + 0.5f) / NG) * 2.0f - 1.0f;
      lum[gi * NG + gj] = bf_sample_lum(P, on2, px, py);
    }
  }
  // Variance (std-dev).
  float sum = 0.0f, sum2 = 0.0f;
  for (int k = 0; k < NG * NG; k++) { sum += lum[k]; sum2 += lum[k] * lum[k]; }
  float n = (float)(NG * NG);
  float mean = sum / n;
  float sd = std::sqrt(std::max(sum2 / n - mean * mean, 0.0f));
  // Edge energy (RMS of forward differences along both axes).
  float ge = 0.0f; int gc = 0;
  for (int gi = 0; gi < NG; gi++) {
    for (int gj = 0; gj < NG; gj++) {
      float c = lum[gi * NG + gj];
      if (gi + 1 < NG) { float d = lum[(gi + 1) * NG + gj] - c; ge += d * d; gc++; }
      if (gj + 1 < NG) { float d = lum[gi * NG + (gj + 1)] - c; ge += d * d; gc++; }
    }
  }
  float edge = std::sqrt(ge / (float)(gc > 0 ? gc : 1));
  return lerpf(sd, edge, clampf(edge_bias, 0.0f, 1.0f));
}

static void bf_build_struct(const AtlasRef& A, State* s, float* P, int base, bool useB,
                            bool animate, float z, float scaleMul, float extrudeMul, float tau,
                            const Corners& c) {
  const int B = A.n_terms, Z = A.nz;
  const float* const* TA = useB ? A.b_term : A.term;
  const float* const* SA = useB ? A.b_scene : A.scene;

  // interpolate term params
  for (int i = 0; i < B; i++) {
    for (int f = 0; f < 8; f++) {
      const float* src = TA[f];
      float v = 0;
      for (int cc = 0; cc < 4; cc++) v += c.w[cc] * src[c.idx[cc] * B + i];
      P[base + i * 8 + f] = v;
    }
  }
  // interpolate scene scalars
  for (int f = 0; f < SCN; f++) {
    const float* src = SA[f];
    float v = 0;
    for (int cc = 0; cc < 4; cc++) v += c.w[cc] * src[c.idx[cc]];
    P[base + SB1 + f] = v;
  }

  // finalize form_scale / extrude / fog (taste multipliers), THEN fix the level
  // from the BASE field — computing it before animation touches h keeps the
  // threshold stable (no popping). No upper clamp on form_scale (the prototype's
  // 0.16 cap was removed) so `scale` can push features large.
  P[base + SB1 + S_SEV] = 0; // orthographic
  P[base + SB1 + S_FORM_SCALE] *= scaleMul;
  P[base + SB1 + S_EXTRUDE] *= extrudeMul;
  P[base + SB1 + S_FOG] = std::min(P[base + SB1 + S_FOG] * s->fog, 0.95f);
  P[base + SB1 + SCN] = bf_levelFor(P, base, P[base + SB1 + S_THRESH]);

  if (animate) {
    float fz = z * (Z - 1);
    int z0 = clampi((int)std::floor(fz), 0, Z - 1);
    int z1 = std::min(z0 + 1, Z - 1);
    float tz = fz - z0;
    float am = s->anim_amount;
    for (int i = 0; i < B; i++) {
      // sc: 0=h_amp, 1=h_om, 2=h_psi, 3=phase_drift (trilinear over x,y,z)
      float sc[4];
      for (int ch = 0; ch < 4; ch++) {
        const float* src = A.script[ch];
        float v0 = 0, v1 = 0;
        for (int cc = 0; cc < 4; cc++) {
          v0 += c.w[cc] * src[(c.idx[cc] * Z + z0) * B + i];
          v1 += c.w[cc] * src[(c.idx[cc] * Z + z1) * B + i];
        }
        sc[ch] = v0 * (1 - tz) + v1 * tz;
      }
      // in/out: integer frequency → sin returns to its t=0 value at t=1 (seamless).
      P[base + i * 8 + 5] += am * sc[0] * std::sin(kTau * (std::round(sc[1]) * tau + sc[2]));
      // scroll: advance a whole number of cells per loop (round the blended drift).
      P[base + i * 8 + 4] += std::round(sc[3]) * tau;
    }
  } else {
    // structure 2 has no folded trajectory → a constant integer-cell scroll.
    float scroll = kScrollCells * tau;
    for (int i = 0; i < B; i++) P[base + i * 8 + 4] += scroll;
  }
}

// Fill the uniform buffer's P + header for the current frame.
static void bf_build_params(State* s, float sx, float sy, float t, Uniforms& u) {
  const AtlasRef& A = s->key_moment ? kAtlasKm : kAtlasExplore;
  float* P = u.P;
  bool interp = s->interp_cells;
  float z = s->liveliness;
  float tau;
  if (s->key_moment) {
    // Key-moment mode: resolve the KM atlas cell (snap + sky-threshold reachability),
    // render that single cell (interp off) at z=0 (the KM atlas is XY-only), and map
    // the eased playhead into the window's fixed span anchored on the centre peak.
    // Cells with no usable window fall back to the full loop (linear → seamless).
    interp = false; z = 0.0f;
    int gi, gj; km_resolve(A, s, &gi, &gj);
    sx = (float)gj / (A.grid - 1); sy = (float)gi / (A.grid - 1);
    int k = (gi * A.grid + gj) * A.nz;   // nz = 1 → z index 0
    bool kmv = A.km_score && A.km_score[k] > 0.0f && A.km_covmax[k] <= kKmMaxCov;
    if (kmv) {
      float tp = (A.km_t1[k] - kKmPre) + km_ease_out(s->km_u, s->km_ease) * kKmSpan;
      tau = tp - std::floor(tp);
    } else {
      tau = s->km_u;
    }
  } else {
    // Eased time: rests at the loop point, surges through the middle. Seamless
    // (τ(0)=0, τ(1)=1) so integer omega/drift keep the loop closed.
    tau = t - (s->ease / kTau) * std::sin(kTau * t);
  }
  Corners c = bf_corners(A, sx, sy, interp);

  // Inverted: higher `scale` → smaller form_scale → fewer cells per screen →
  // bigger features → "zoom IN". (scale=1 is unchanged; balance is the relative
  // S1↔S2 zoom on top.)
  float inv = 1.0f / std::max(s->scale, 1e-3f);
  float scaleA = inv * std::pow(2.0f, s->balance);
  float scaleB = inv * std::pow(2.0f, -s->balance);
  bf_build_struct(A, s, P, 0, /*useB=*/false, /*animate=*/true, z, scaleA, s->extrude, tau, c);

  float tilt = 0;
  bool on2 = s->second_structure && A.co_fold;
  if (on2) {
    bf_build_struct(A, s, P, STR, /*useB=*/true, /*animate=*/false, 0.0f, scaleB, s->extrude, tau, c);
    const float* bt = A.b_tilt;
    for (int cc = 0; cc < 4; cc++) tilt += c.w[cc] * bt[c.idx[cc]];
  }

  u.n_terms = (float)A.n_terms;
  u.sb = (float)SB1;
  u.str = (float)STR;
  u.tilt = tilt;
  u.enable2 = on2 ? 1.0f : 0.0f;
  u.h_act = A.h_act;

  u.diff_hue_lo = s->diff_hue_lo; u.diff_hue_mid = s->diff_hue_mid;
  u.diff_hue_hi = s->diff_hue_hi; u.diff_sat = s->diff_sat;
  u.diff_sat_lo = s->diff_sat_lo; u.diff_sat_mid = s->diff_sat_mid; u.diff_sat_hi = s->diff_sat_hi;
  u.diff_bri_lo = s->diff_bri_lo; u.diff_bri_mid = s->diff_bri_mid; u.diff_bri_hi = s->diff_bri_hi;
  u.fog_hue_lo = s->fog_hue_lo; u.fog_hue_mid = s->fog_hue_mid;
  u.fog_hue_hi = s->fog_hue_hi; u.fog_sat = s->fog_sat;
  u.fog_sat_lo = s->fog_sat_lo; u.fog_sat_mid = s->fog_sat_mid; u.fog_sat_hi = s->fog_sat_hi;
  u.sky_hue = s->sky_hue; u.sky_sat = s->sky_sat; u.sky_bri = s->sky_bri;
  u.noise_blob = s->noise_blob; u.noise_fog = s->noise_fog;
  u.noise_blob_tilt = s->noise_blob_tilt;
  u.noise_seed = s->noise_phase;   // shader floors via int() → discrete reroll id

  // Volumetric blob — feed the DRIFTED (effective) values.
  u.vol_amount = s->vol_amount;
  u.vol_anchor_x = s->eff_vol_x;
  u.vol_anchor_y = s->eff_vol_y;
  u.vol_z = s->eff_vol_z;
  u.vol_shape = s->eff_vol_shape;
  u.vol_angle = s->eff_vol_angle;
  u.vol_radius = s->vol_radius * kVolRadiusMax;
  u.vol_softness_xy = s->vol_softness_xy * kVolSoftXYMax;
  u.vol_softness_z = s->vol_softness_z * kVolSoftZMax;
  u.vol_depth = s->vol_depth * kVolDepthMax;

  // CPU flatness proxy — used only as the WARMUP fallback until the GPU edge
  // readback arrives (then tick() drives s->content from the real rendered
  // frame). Off → a high "clearly not flat" so the detector never engages.
  if (!s->skip_empty || s->key_moment) {
    s->content = 1.0f;
  } else if (!s->skip_gpu_ready) {
    s->content = bf_content(P, on2, s->skip_w_edge);
  }
  // else: GPU-driven (s->content set in tick from the readback) — leave it.
}

void render(void* self, int vp_w, int vp_h) {
  auto* s = static_cast<State*>(self);
  if (!s || !s->initialized || vp_w <= 0 || vp_h <= 0) return;
  auto out = gpu::Device::textureForField("tex_out");
  if (!out.valid()) return;

  Uniforms u = {};
  u.res_x = (float)vp_w;
  u.res_y = (float)vp_h;
  bf_build_params(s, s->eff_x, s->eff_y, s->clock_t, u);
  s->uniform_buf.writeOne(u);

  auto cp = gpu::ComputePass::begin();
  cp.setPSO(s_pso_present);
  cp.setBuffer(s->uniform_buf, 0);
  cp.setTexture(out, 1, 1);
  cp.dispatch((vp_w + 7) / 8, (vp_h + 7) / 8);
  cp.end();

  // GPU flatness detector: Sobel/variance reduce over the composited tex_out into
  // stats_buf, then request an async readback. Only when the detector is on.
  // Reset runs BEFORE the pass; the tick() poll reads last frame's stats before
  // this reset each frame, so the CPU zero-write is race-free (plan Risk #2).
  if (s->skip_empty && !s->key_moment) {
    int32_t zeros[kStatsInts] = {};
    s->stats_buf.write(zeros, kStatsInts);
    auto ep = gpu::ComputePass::begin();
    ep.setPSO(s_pso_edge);
    ep.setBuffer(s->uniform_buf, 0);
    ep.setTexture(out, 1, 0);            // access 0 = Read
    ep.setBuffer(s->stats_buf, 2);
    ep.setBuffer(s->prev_luma_buf, 3);
    ep.dispatch((kSampleGrid + 7) / 8, (kSampleGrid + 7) / 8);   // fixed grid, not per-pixel
    ep.end();
    s->stats_buf.requestReadback(kStatsInts * sizeof(int32_t));

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
      dp.setBuffer(s->stats_buf, 1);
      dp.setTexture(out, 2, 1);          // access 1 = Write
      dp.dispatch((vp_w + 7) / 8, (vp_h + 7) / 8);
      dp.end();
    }
  }

  gpu::Device::submit();
}

} // namespace brutal_fold
