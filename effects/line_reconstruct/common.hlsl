// filter.reconstruct.line — "Line Reconstruct" — shared shader helpers + constants.
//
// A GPU port of the re-edger research harness (nano-fx-prototypes/re-edger). We
// CLASSIFY each pixel (line / point / step-edge / junction / smooth gradient —
// subpixel center, width, orientation) then RE-RENDER lines & points as crisp,
// uniform-width, box-AA strokes ("4K-downsample" look) and de-band gradients.
//
// This header carries: the effect's shared uniform layout, the baked tuning
// constants (calibrated in study 1 / study 5 of the prototype — NOT exposed as
// params), Rec.709 luma (the prototype is calibrated on 709; the shared
// nano_luminance is Rec.601), and small math helpers reused across passes.

#ifndef LINE_RECONSTRUCT_COMMON_HLSL
#define LINE_RECONSTRUCT_COMMON_HLSL

static const float LSB = 1.0 / 255.0;

// Scale-space levels (px sigma). Detection uses 0..2; level 3 = wide context.
static const float LR_SIGMA0 = 0.7;
static const float LR_SIGMA1 = 1.4;
static const float LR_SIGMA2 = 2.8;
static const float LR_SIGMA3 = 5.6;

// --- baked tuning (Params defaults + study-1 CAL; see reconstruct plan) --------
static const float BETA_SOFTMAX = 6.0;     // scale-softmax sharpness
// w_est = AW2*s^2 + ALPHA_W*s + BETA_W  (study 1 calibration)
static const float AW2      = -1.040;
static const float ALPHA_W  =  6.382;
static const float BETA_W   = -4.069;
static const float ALPHA_B  =  1.35;       // R_est = ALPHA_B*sigma_b + BETA_B
static const float BETA_B   = -0.30;
static const float R0 = 0.06,  R1 = 0.20;  // ridge-response gate (post-normalize)
static const float B0 = 0.06,  B1 = 0.18;  // blob-response gate
static const float ABS0 = 0.008, ABS1 = 0.020;  // absolute (un-normalized) floor
static const float RHO_POW = 0.75;         // ridge-purity exponent (soft)
static const float J0 = 0.20,  J1 = 0.55;  // junction gate on lambda2/lambda1
static const float G0 = 2.0,   G1 = 8.0;   // deband gate: fine contrast in LSB
static const float FLANK0 = 0.25, FLANK1 = 0.65;  // flank-disparity gate

// --- shared uniform layout (16-byte rows; MUST match C++ Uniforms) ------------
struct LRUniforms {
  // row 0 — widths already mapped to PX host-side
  float strength;      // master mix; 0 handled by is_identity (never dispatched)
  float target_width;  // px, uniform line-width goal
  float retarget;      // 0 = clean AA at own width, 1 = fully uniform
  float point_radius;  // px, uniform point-radius goal
  // row 1
  float solidify;      // 0..1 dash/along-line colour rescue
  float deband;        // 0..1 -> clamp budget 0..4 LSB + dither
  float c_floor;       // CAS contrast floor = lerp(0.12, 0.01, sensitivity)
  float recover;       // contrast extrapolation past local evidence
  // row 2
  float max_width;     // px cap on what counts as a line
  float aspect;        // vp_w / vp_h
  uint  debug_view;    // 0 Off,1 Class,2 Width,3 Orientation,4 Centerline,5 PolCoh
  float inv_w;         // 1 / vp_w
  // row 3
  float inv_h;         // 1 / vp_h
  float _p0, _p1, _p2;
};

float lr_luma709(float3 rgb) {
  return dot(rgb, float3(0.2126, 0.7152, 0.0722));
}

float lr_smoothstep(float e0, float e1, float x) {
  float t = saturate((x - e0) / (e1 - e0));
  return t * t * (3.0 - 2.0 * t);
}

// Exact 1-px box-filter coverage of a band of width w at signed distance d --
// the profile a 4K box-downsample of an ideal line produces.
float lr_band_coverage(float d, float w) {
  float lo = max(abs(d) - 0.5, -w * 0.5);
  float hi = min(abs(d) + 0.5,  w * 0.5);
  return saturate(hi - lo);
}

// Interleaved gradient noise, triangular-remapped to [-1,1] (screen-anchored --
// temporally stable for video, by design).
float lr_ign(float2 p) {
  float g = frac(52.9829189 * frac(0.06711056 * p.x + 0.00583715 * p.y));
  float t = 2.0 * g - 1.0;
  return sign(t) * (1.0 - sqrt(saturate(1.0 - abs(t))));
}

#endif // LINE_RECONSTRUCT_COMMON_HLSL
