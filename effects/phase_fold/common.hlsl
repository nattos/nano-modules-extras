// source.phase_fold — shared math for all passes.
//
// A phase-portrait GENERATOR. The XY pad picks a cell in a GxG atlas of
// emergent level-set limit-cycle FIELDS (x = eccentricity, y = lobedness). The
// baked atlas (PF_CELLS in phase_fold_atlas.h) is uploaded to the GPU as a flat
// storage buffer so every pass can evaluate the blended scalar field H and its
// induced vector field v = level-set flow + WIND(z) directly on-device. This is
// the research testbed's split (evalV/cellH in app.js) ported to native — but
// here the streamline tracing, arrow animation and limit-cycle integration that
// the prototype did on the CPU each frame are done on the GPU as compute passes.
//
// The shared uniform block is bound at register(b0) in EVERY pass (backdrop,
// the two tracers, and the line raster), so this file declares it once and the
// field helpers read it as globals. The atlas cell buffer is bound at
// register(t1) wherever the field is evaluated.

#ifndef PHASE_FOLD_COMMON_HLSL
#define PHASE_FOLD_COMMON_HLSL

// Must match phase_fold_atlas.h AND the constants in main.cpp.
#define PF_K       8u     // blob kernels per cell
#define PF_R       3u     // ring-ridge kernels per cell
#define PF_STRIDE  50u    // floats per cell (4 + K*4 + R*3 + 5)
#define PF_WIND_OFF 45u   // offset of (wdx, wdy, wmax) within a cell

#define PF_NS        15      // streamline seed grid (NS x NS)
#define PF_SL_STEPS  16      // integration steps per streamline = segments stored
#define PF_SL_DT     0.05    // streamline step size
#define PF_NOUT      96      // resting-cycle points per cell (limit-cycle seeds)
#define PF_PARTICLES PF_NOUT // stateful limit-cycle solver particles (one per seed)
#define PF_BREAK_SAMPLES 4   // gradient samples between consecutive particles
#define PF_RELAX_CAP 0.10    // max per-iteration Newton step (solver stability)
#define PF_VEL_CAP   0.15    // max particle speed (momentum stability)
#define PF_RESPAWN_COOLDOWN 20.0  // min frames between broken-cycle respawns

// status buffer (float[4]): cycle health shared solve <-> select across frames.
#define PF_ST_CLOSED  0   // 1 if the longest run spans the whole ring (no breaks)
#define PF_ST_ARC     1   // arc length of the longest contiguous run
#define PF_ST_RESPAWN 2   // 1 → next frame's solve should respawn (broken + short)
#define PF_ST_COOLDOWN 3  // frames remaining before another broken-cycle respawn
#define PF_TDT       0.02    // streamline-style step size
#define PF_STEP_CAP  0.06    // per-step displacement clamp (keeps integration smooth)

// Colour code packed into Segment.b.x — the fragment shader maps it to a palette
// (streamline speed gradient below 0.58, gold limit cycle above 0.8).
#define PF_CODE_CYCLE  0.90   // gold limit-cycle

// Shared uniform block — identical layout in every pass (always register b0).
cbuffer U : register(b0) {
  float res_x;   float res_y;   float extent;       float bias;
  float wind;    float n_bands; float contrast;     float flow_phase;
  float nearest_cell; float respawn; float stream_width; float cycle_width;
  float backdrop_dim; float stream_alpha; float shading_mode; float solve_steps;
  float break_dist;   float explore;  float spread;   float rand_seed;
  float step_size;    float momentum; float morph_rate; float respawn_arc;
  float good_init;    float break_turn_cos; float stream_spread; float _pf_pad2;
  float4 corners;   // 4 corner cell indices (as float)
  float4 weights;   // 4 convex blend weights (sum 1, or 0 over a hole)
};

#define PF_TWO_PI 6.28318530718

// One line segment:
//   a = (p0.x, p0.y, p1.x, p1.y)
//   b = (code, alpha, width, dead)
//   c = (arc, stagger, _, _)   — arc ∈ [0,1) position along the line; stagger is
//       a per-streamline phase offset. The fragment shader rides a continuous
//       glow down the line via frac(arc - flow_phase + stagger) — no quantized
//       arrowhead. (The atlas cell buffer + field helpers live in field.hlsl so
//       the line raster can include this header without the t1 cells binding.)
struct Segment { float4 a; float4 b; float4 c; };

float pf_weight_sum() {
  return weights[0] + weights[1] + weights[2] + weights[3];
}

// --- Coordinate mapping (shared by backdrop write and line raster) ---------
//
// The square phase space [-extent,extent]² COVERS the viewport uniformly:
// scaled by the LONG axis, the short axis cropped (no bars). Phase-space y is
// up; screen y is down.

float2 pf_pixel_to_p(float2 px, float2 vp) {
  float mx = max(vp.x, vp.y);
  float2 sq = (px + 0.5 - 0.5 * vp) / (0.5 * mx);
  return float2(sq.x, -sq.y) * extent;
}

// phase space → world uv (y-down, 0 = top row); caller does clip = uv*2-1 and
// lets naga insert the Vulkan→WebGPU y-flip (see flash_particles/vs.hlsl note).
float2 pf_p_to_uv(float2 p, float2 vp) {
  float mx = max(vp.x, vp.y);
  float2 sq = float2(p.x, -p.y) / extent;
  float2 px = sq * (0.5 * mx) + 0.5 * vp;
  return px / vp;
}

// --- Backdrop colormap (diverging RdBu, muted) — port of app.js diverging() --

float3 pf_diverging(float t) {
  float x = saturate(t);
  float3 blue = float3(0.13, 0.30, 0.55), lblue = float3(0.62, 0.74, 0.86);
  float3 white = float3(0.97, 0.97, 0.98);
  float3 lred = float3(0.93, 0.62, 0.50), red = float3(0.62, 0.10, 0.12);
  if (x < 0.25) return lerp(blue, lblue, x / 0.25);
  if (x < 0.5)  return lerp(lblue, white, (x - 0.25) / 0.25);
  if (x < 0.75) return lerp(white, lred, (x - 0.5) / 0.25);
  return lerp(lred, red, (x - 0.75) / 0.25);
}

// --- Matplotlib colormaps — 6th-order poly fits (after Matt Zucker, shadertoy
//     WlfXRN), same as shape_fold. Cheap, branchless, close visual match. ------

float3 pf_poly6(float t, float3 c0, float3 c1, float3 c2, float3 c3, float3 c4, float3 c5, float3 c6) {
  float x = saturate(t);
  return c0 + x * (c1 + x * (c2 + x * (c3 + x * (c4 + x * (c5 + x * c6)))));
}
float3 pf_magma(float t) {
  return pf_poly6(t,
    float3(-0.002136, -0.000750, -0.005386), float3(0.251661, 0.677523, 2.494027),
    float3(8.353717, -3.577720, 0.314468),   float3(-27.668733, 14.264731, -13.649213),
    float3(52.176140, -27.943606, 12.944169), float3(-50.768525, 29.046583, 4.234153),
    float3(18.655705, -11.489774, -5.601962));
}
float3 pf_inferno(float t) {
  return pf_poly6(t,
    float3(0.000219, 0.001651, -0.019481),  float3(0.106513, 0.563956, 3.932712),
    float3(11.602493, -3.972854, -15.942394), float3(-41.703996, 17.436399, 44.354145),
    float3(77.162936, -33.402359, -81.807309), float3(-71.319428, 32.626064, 73.209520),
    float3(25.131126, -12.242669, -23.070325));
}
float3 pf_viridis(float t) {
  return pf_poly6(t,
    float3(0.277727, 0.005407, 0.334100),  float3(0.105093, 1.404614, 1.384590),
    float3(-0.330862, 0.214848, 0.095095), float3(-4.634230, -5.799101, -19.332441),
    float3(6.228270, 14.179933, 56.690553), float3(4.776385, -13.745145, -65.353033),
    float3(-5.435456, 4.645853, 26.312435));
}
float3 pf_plasma(float t) {
  return pf_poly6(t,
    float3(0.058732, 0.023337, 0.543340),  float3(2.176515, 0.238383, 0.753960),
    float3(-2.689460, -7.455851, 3.110800), float3(6.130348, 42.346188, -28.518855),
    float3(-11.107436, -82.666311, 60.139848), float3(10.023066, 71.413618, -54.072187),
    float3(-3.658714, -22.931535, 18.191908));
}
float3 pf_turbo(float t) {
  return pf_poly6(t,
    float3(0.114089, 0.062883, 0.224834),    float3(6.716419, 3.182287, 7.571582),
    float3(-66.094024, -4.927983, -10.094394), float3(228.766079, 25.049867, -91.541053),
    float3(-334.835157, -69.317497, 288.585885), float3(218.763722, 67.521506, -305.204577),
    float3(-52.889035, -21.545274, 110.517465));
}

// Banded colour for shading mode: 0 = diverging, 2..6 = matplotlib.
float3 pf_grade(float band, float mode) {
  if (mode < 0.5)      return pf_diverging(band);
  else if (mode < 2.5) return pf_magma(band);
  else if (mode < 3.5) return pf_inferno(band);
  else if (mode < 4.5) return pf_viridis(band);
  else if (mode < 5.5) return pf_plasma(band);
  return pf_turbo(band);
}

#endif // PHASE_FOLD_COMMON_HLSL
