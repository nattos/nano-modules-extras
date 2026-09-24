// source.shape_fold — shared math for all passes.
//
// An evolving-shape GENERATOR. A baked 3D atlas (frequency × simplicity ×
// temporal-complexity) is interpolated on the CPU each frame down to a handful
// of "terms"; those resolved terms ride in the uniform buffer and every GPU
// pass evaluates the same scalar SDF field `sf_field_at` from them — the 1.1 MB
// atlas never touches the GPU. This is the research testbed's split
// (sampleTerms in JS → fieldAt in WGSL), ported to native.
//
// Output is the raw field, histogram auto-leveled (median→0) and shown as
// grayscale or magma — no line/contour/shading modes (those were dropped on
// purpose). Auto-levels runs every frame on the GPU: minmax → hist → buildlut
// (CLAHE + percentile remap LUT) → present. The histogram needs atomic scatter,
// so stats + LUT ride in storage buffers. The shared uniform block is bound at
// register(b0) in every pass so this file can declare it once and the field
// helpers read it as globals.

#ifndef SHAPE_FOLD_COMMON_HLSL
#define SHAPE_FOLD_COMMON_HLSL

#define SF_MAX_TERMS 8
#define SF_NB        256       // histogram bins / LUT entries
#define SF_SN        160       // auto-levels downsample grid (SN×SN samples)

// Shared uniform block — identical layout in every pass (always register b0).
// 3 std140 rows of scalars, then the resolved terms (per term:
// (theta,mtheta,curv,freq)(phase,h,k,amp)(mix,spc,_,_)).
cbuffer U : register(b0) {
  float res_x;      float res_y;          float n_terms;       float dc;
  float bold_gain;  float birth_softness; float domain_scale;  float level_ease;
  float output_mode; float exposure;      float _sf_pad1;      float _sf_pad2;
  float4 terms[SF_MAX_TERMS * 3];
};

// One term's raw value at p (bowed, parabolically-warped, repeating cell →
// inverted parabola). Mirrors termRaw in app.js.
float sf_term_raw(uint i, float2 p) {
  float4 a = terms[i * 3u + 0u];          // theta, mtheta, curv, freq
  float4 b = terms[i * 3u + 1u];          // phase, h, k, amp
  float spc = terms[i * 3u + 2u].y;       // parabolic spacing warp
  float2 n = float2(cos(a.x), sin(a.x));
  float2 m = float2(cos(a.y), sin(a.y));
  float dm = dot(m, p);
  float d  = dot(n, p) + a.z * dm * dm;   // bowed coordinate → arcing lines
  float dw = d + spc * d * d;             // light parabola → local spacing varies
  float q  = frac(dw * a.w + b.x) - 0.5;  // repeating cell coord
  return b.y - b.z * q * q;               // inverted parabola
}

// The field: multiplicative AND terms carve shapes (with a soft birth gate),
// additive terms union stripes. Mirrors fieldAt in app.js. F ≥ 0 always
// (dc ≥ 0, additive ≥ 0, the carved product ∈ [0,1]).
float sf_field_at(float2 p) {
  float uni = 0.0;
  float inter = 1.0;
  uint nt = (uint)n_terms;
  for (uint i = 0u; i < nt; i++) {
    float4 b = terms[i * 3u + 1u];        // phase, h, k, amp
    float4 c = terms[i * 3u + 2u];        // mix, spc
    float raw = sf_term_raw(i, p);
    if (c.x >= 0.5) {                     // multiplicative AND (carves shapes)
      // Soft birth gate: active terms sit at h≫softness so statics stay fully
      // carved; births/deaths fade over a wide h-range instead of popping.
      float act = smoothstep(0.0, max(birth_softness, 1e-4), b.y);
      inter = inter * ((1.0 - act) + act * clamp(raw / max(b.y, 1e-3), 0.0, 1.0));
    } else {                              // additive union (stripes)
      uni = uni + b.w * max(0.0, raw);
    }
  }
  return dc + uni + bold_gain * inter;
}

// Auto-levels downsample sample point: canonical [-1,1] square × domain zoom.
// minmax + hist both use this so the leveling matches what present displays.
float2 sf_levels_p(uint2 gid) {
  return ((float2(gid) + 0.5) / float(SF_SN) * 2.0 - 1.0) * domain_scale;
}

// Polynomial colormaps — 6th-order fits to the matplotlib/Google maps (after
// Matt Zucker's "Simple analytic approximations", shadertoy WlfXRN). Cheap,
// branchless, and a close visual match.
float3 sf_poly6(float t, float3 c0, float3 c1, float3 c2, float3 c3, float3 c4, float3 c5, float3 c6) {
  float x = saturate(t);
  return c0 + x * (c1 + x * (c2 + x * (c3 + x * (c4 + x * (c5 + x * c6)))));
}

float3 sf_magma(float t) {
  return sf_poly6(t,
    float3(-0.002136, -0.000750, -0.005386), float3(0.251661, 0.677523, 2.494027),
    float3(8.353717, -3.577720, 0.314468),   float3(-27.668733, 14.264731, -13.649213),
    float3(52.176140, -27.943606, 12.944169), float3(-50.768525, 29.046583, 4.234153),
    float3(18.655705, -11.489774, -5.601962));
}

float3 sf_inferno(float t) {
  return sf_poly6(t,
    float3(0.000219, 0.001651, -0.019481),  float3(0.106513, 0.563956, 3.932712),
    float3(11.602493, -3.972854, -15.942394), float3(-41.703996, 17.436399, 44.354145),
    float3(77.162936, -33.402359, -81.807309), float3(-71.319428, 32.626064, 73.209520),
    float3(25.131126, -12.242669, -23.070325));
}

float3 sf_viridis(float t) {
  return sf_poly6(t,
    float3(0.277727, 0.005407, 0.334100),  float3(0.105093, 1.404614, 1.384590),
    float3(-0.330862, 0.214848, 0.095095), float3(-4.634230, -5.799101, -19.332441),
    float3(6.228270, 14.179933, 56.690553), float3(4.776385, -13.745145, -65.353033),
    float3(-5.435456, 4.645853, 26.312435));
}

float3 sf_plasma(float t) {
  return sf_poly6(t,
    float3(0.058732, 0.023337, 0.543340),  float3(2.176515, 0.238383, 0.753960),
    float3(-2.689460, -7.455851, 3.110800), float3(6.130348, 42.346188, -28.518855),
    float3(-11.107436, -82.666311, 60.139848), float3(10.023066, 71.413618, -54.072187),
    float3(-3.658714, -22.931535, 18.191908));
}

float3 sf_turbo(float t) {
  return sf_poly6(t,
    float3(0.114089, 0.062883, 0.224834),    float3(6.716419, 3.182287, 7.571582),
    float3(-66.094024, -4.927983, -10.094394), float3(228.766079, 25.049867, -91.541053),
    float3(-334.835157, -69.317497, 288.585885), float3(218.763722, 67.521506, -305.204577),
    float3(-52.889035, -21.545274, 110.517465));
}

#endif // SHAPE_FOLD_COMMON_HLSL
