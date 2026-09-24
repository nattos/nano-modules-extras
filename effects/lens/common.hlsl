// filter.blur.lens — "Lens" — shared shader helpers + baked constants.
//
// A GPU port of the single-plane lens simulation research harness
// (nano-fx-prototypes/lens-sim). Linear-HDR throughout; tonemap only in the
// final pass. This header carries small math helpers reused across the passes.
// The prototype is calibrated on Rec.709 luma; use lens_luma.

#ifndef LENS_COMMON_HLSL
#define LENS_COMMON_HLSL

static const float LENS_PI  = 3.14159265358979;
static const float LENS_TAU = 6.28318530717959;

float lens_luma(float3 rgb) {
  return dot(rgb, float3(0.2126, 0.7152, 0.0722));
}

// smoothstep, matching optics.smoothstep (clamped Hermite).
float lens_smoothstep(float e0, float e1, float x) {
  float t = saturate((x - e0) / (e1 - e0));
  return t * t * (3.0 - 2.0 * t);
}

// smoothstep_down (optics.py:152): 1 where x<edge, 0 where x>edge, soft ramp of
// half-width max(edge*softness,1e-4) centred on `edge`.
float lens_smoothstep_down(float edge, float x, float softness) {
  float w = max(edge * softness, 1e-4);
  float t = saturate((x - (edge - w)) / (2.0 * w));
  return 1.0 - t * t * (3.0 - 2.0 * t);
}

// sRGB <-> linear (pipeline._linear_to_srgb :484; inverse for ingest per REPORT §1).
float3 lens_srgb_to_linear(float3 c) {
  float3 lo = c / 12.92;
  float3 hi = pow(max((c + 0.055) / 1.055, 0.0), 2.4);
  return float3(c.r <= 0.04045 ? lo.r : hi.r,
                c.g <= 0.04045 ? lo.g : hi.g,
                c.b <= 0.04045 ? lo.b : hi.b);
}
float3 lens_linear_to_srgb(float3 c) {
  c = saturate(c);
  float3 lo = c * 12.92;
  float3 hi = 1.055 * pow(c, 1.0 / 2.4) - 0.055;
  return float3(c.r <= 0.0031308 ? lo.r : hi.r,
                c.g <= 0.0031308 ? lo.g : hi.g,
                c.b <= 0.0031308 ? lo.b : hi.b);
}

// Narkowicz ACES filmic fit (pipeline._aces :489).
float3 lens_aces(float3 x) {
  const float a = 2.51, b = 0.03, c = 2.43, d = 0.59, e = 0.14;
  return saturate((x * (a * x + b)) / (x * (c * x + d) + e));
}

// --- spectral / coating helpers (sun flare pass; coatings.py) ----------------
// Coarse spectral basis (coatings.py:91-102): wavelength nm + column-normalized
// linear-RGB response (a flat spectrum reconstructs neutral white).
static const float LENS_BAND_NM[7] = { 440.0, 470.0, 500.0, 530.0, 565.0, 600.0, 640.0 };
static const float3 LENS_BAND_RGB[7] = {
  float3(0.074074, 0.007273, 0.300000),
  float3(0.033670, 0.072727, 0.418182),
  float3(0.000000, 0.200000, 0.236364),
  float3(0.060606, 0.298182, 0.045455),
  float3(0.269360, 0.290909, 0.000000),
  float3(0.319865, 0.116364, 0.000000),
  float3(0.242424, 0.014545, 0.000000),
};
// Ghost chain (pipeline.py:398-405): (signed pos along u, base radius, brightness, incidence).
static const float4 LENS_GHOST[6] = {
  float4(-1.22, 0.20, 0.85, 0.98),
  float4(-0.80, 0.44, 0.50, 0.84),
  float4(-0.44, 0.28, 0.72, 0.68),
  float4(-0.10, 0.70, 0.30, 0.58),
  float4( 0.34, 0.34, 0.50, 0.90),
  float4( 0.74, 0.22, 0.62, 0.74),
};

float lens_hash1(float t) {                       // pipeline._hash1
  float x = sin(t * 127.1 + 11.7) * 43758.5453;
  return x - floor(x);
}

float lens_source_weight(float3 src, float3 band) {   // coatings.source_weight
  return dot(src, band) / max(band.r + band.g + band.b, 1e-6);
}

// coatings.coat_reflectance — residual AR-stack reflectance at wavelength nm.
float lens_coat_reflectance(float nm, float ct, float opd, float3 designs,
                            uint ndesigns, float r0) {
  if (ndesigns == 0) return r0;
  float acc = 0.0;
  [unroll] for (uint k = 0; k < 3; k++) {
    if (k < ndesigns) {
      float phase = 0.5 * LENS_PI * (designs[k] / nm) * ct + LENS_PI * opd / nm;
      float c = cos(phase);
      acc += c * c;
    }
  }
  return r0 * acc / (float)ndesigns;
}

// coatings.coat_color — coating residual colour at a single incidence.
float3 lens_coat_color(float3 src, float ct, float3 designs, uint ndesigns, float r0) {
  float3 acc = 0.0.xxx;
  [unroll] for (uint i = 0; i < 7; i++) {
    float refl = lens_coat_reflectance(LENS_BAND_NM[i], ct, 0.0, designs, ndesigns, r0);
    acc += refl * lens_source_weight(src, LENS_BAND_RGB[i]) * LENS_BAND_RGB[i];
  }
  return acc;
}

// optics.poly_boundary_radius / aperture_weight (tap space, circumradius 1).
float lens_poly_boundary_radius(float theta, uint blades, float rot, float curv) {
  uint n = blades < 3 ? 3 : blades;
  float sector = LENS_TAU / (float)n;
  float a = theta - rot;
  float rem = a - floor(a / sector) * sector;
  float phi = rem - sector * 0.5;
  float apo = cos(LENS_PI / (float)n);
  float poly_r = apo / max(cos(phi), 1e-4);
  return lerp(poly_r, 1.0, curv);
}
float lens_aperture_weight(float2 o, uint blades, float rot, float curv, float soft) {
  float bnd = lens_poly_boundary_radius(atan2(o.y, o.x), blades, rot, curv);
  return lens_smoothstep_down(bnd, length(o), soft);
}

#endif // LENS_COMMON_HLSL
