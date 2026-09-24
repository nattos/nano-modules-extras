// filter.blur.lens — pass 5 (off-frame sun / stray light). A bright out-of-frame
// source refracting through the glass, geometrically gated by the hood, rendered
// spectrally (7 bands → RGB) as four diffuse additive terms (pipeline.pass_sun
// :283): a broad directional GLOW, a coating-tinted VEIL pedestal, thin
// diffraction STREAKS locked to the aperture blades, and a GHOST chain of
// aperture-shaped internal reflections along the source→centre axis.
//
// All analytic ALU (no gathers), low-frequency — off by default (sun_intensity=0,
// whole pass skipped host-side). Runs at the reduced flare resolution and writes
// the additive CONTRIBUTION only (gate·add); the host upsample-adds it onto the
// full-res image. The hood_admit·sun_intensity gate is precomputed.

#include "common.hlsl"

RWTexture2D<float4> outputTex : register(u1);
cbuffer Uniforms : register(b2) {
  float u_half, u_dimw, u_dimh, u_gate;
  float u_azimuth, u_obliqueness, u_sun_r, u_sun_g;
  float u_sun_b, u_w_glow, u_w_veil, u_w_streak;
  float u_w_ghost, u_coat_flare, u_elem_curv, u_dispersion;
  float u_aperture_rot, u_blade_curv, u_coat_r0, _p0;
  float3 u_designs; uint u_ndesigns;
  uint u_blades; uint u_nsp; float _p1, _p2;
};

float3 sunColor() { return float3(u_sun_r, u_sun_g, u_sun_b); }

// (1) broad directional glow, spectrally dispersed (pipeline._spectral_glow).
float3 sun_glow(float2 xy, float2 S, float sg) {
  float3 sc = sunColor();
  float3 acc = 0.0.xxx;
  [unroll] for (uint i = 0; i < 7; i++) {
    float shift = u_dispersion * 0.22 * ((LENS_BAND_NM[i] - 540.0) / 100.0);
    float2 db = xy - S * (1.0 + shift);
    float g = exp(-dot(db, db) / (2.0 * sg * sg)) * lens_source_weight(sc, LENS_BAND_RGB[i]);
    acc += g * LENS_BAND_RGB[i];
  }
  return acc;
}

// (2) veiling pedestal — coating-tinted spectral residual (coatings.spectral_tint).
float3 sun_veil(float ct) {
  float3 sc = sunColor();
  float3 acc = 0.0.xxx;
  [unroll] for (uint i = 0; i < 7; i++) {
    float refl = lens_coat_reflectance(LENS_BAND_NM[i], ct, 0.0, u_designs, u_ndesigns, u_coat_r0);
    acc += refl * lens_source_weight(sc, LENS_BAND_RGB[i]) * LENS_BAND_RGB[i];
  }
  return acc;
}

// (3) aperture diffraction rays (pipeline._spectral_streak).
float3 sun_streak(float dist, float theta) {
  float step  = LENS_TAU / (float)u_nsp;
  float base  = u_aperture_rot + LENS_PI / max((float)u_nsp, 1.0);
  float prox  = exp(-2.2 * u_obliqueness);
  if (prox < 1e-3) return 0.0.xxx;
  float idx   = round((theta - base) / step);
  float jit_b = 0.35 + 1.2 * lens_hash1(idx);
  float jit_w = 0.020 * (0.6 + 0.9 * lens_hash1(idx + 3.0));
  float jit_l = 0.7 + 0.6 * lens_hash1(idx + 9.0);
  float hub   = lens_smoothstep(0.04, 0.26, dist);
  float3 sc   = sunColor();
  float3 acc  = 0.0.xxx;
  [unroll] for (uint i = 0; i < 7; i++) {
    float lam     = (LENS_BAND_NM[i] - 540.0) / 100.0;
    float len     = 0.8 * (1.0 + u_dispersion * 0.7 * lam) * jit_l;
    float ang_off = u_dispersion * 0.04 * lam;
    float nearest = round((theta - base - ang_off) / step) * step + base + ang_off;
    float x       = theta - nearest + LENS_PI;
    float adiff   = x - floor(x / LENS_TAU) * LENS_TAU - LENS_PI;   // wrap to [-π,π]
    float ang     = exp(-adiff * adiff / (2.0 * jit_w * jit_w));
    float cb      = cos(dist * (8.5 / len));
    float beads   = 0.35 + 0.65 * cb * cb;
    float radial  = hub * exp(-dist / len) * beads;
    float s       = ang * radial * jit_b * lens_source_weight(sc, LENS_BAND_RGB[i]);
    acc += s * LENS_BAND_RGB[i];
  }
  return acc * prox;
}

// (4) ghost chain — aperture-shaped, coating-tinted (pipeline._ghost_chain).
float3 sun_ghost(float2 xy, float2 u) {
  float cur = u_elem_curv;
  float stretch = 1.0 + 0.6 * u_obliqueness;
  float3 sc = sunColor();
  float3 acc = 0.0.xxx;
  [unroll] for (uint i = 0; i < 6; i++) {
    float4 g = LENS_GHOST[i];              // pos, rad, brightness, incidence
    float2 Gc = u * (g.x * stretch);
    float rr = g.y * (1.0 - 0.35 * cur);
    float2 local = (xy - Gc) / rr;
    float shape = lens_aperture_weight(local, u_blades, u_aperture_rot, u_blade_curv, 0.30);
    float lr = length(local);
    float fall = exp(-lr * lr / 2.4);
    float rim = 1.0 + (0.8 * cur) * lens_smoothstep(0.55, 1.0, clamp(lr, 0.0, 1.3));
    float env = shape * fall * rim;
    float3 col = lens_coat_color(sc, g.w, u_designs, u_ndesigns, u_coat_r0);
    acc += (6.0 * g.z) * env * col;
  }
  return acc;
}

[numthreads(8, 8, 1)]
void main(uint3 gid : SV_DispatchThreadID) {
  uint w, h;
  outputTex.GetDimensions(w, h);
  if (gid.x >= w || gid.y >= h) return;
  float2 pf = float2(gid.xy);

  float gx = (pf.x + 0.5 - u_dimw * 0.5) / u_half;
  float gy = (pf.y + 0.5 - u_dimh * 0.5) / u_half;
  float2 xy = float2(gx, gy);
  float r = max(length(xy), 1e-6);
  float2 rhat = xy / r;

  float2 u = float2(cos(u_azimuth), sin(u_azimuth));
  float2 S = u * (1.0 + u_obliqueness * 1.6);
  float2 d = xy - S;
  float dist = max(length(d), 1e-6);
  float sg = 0.7 + 0.5 * u_obliqueness;
  float ct = clamp(1.0 - 0.35 * u_obliqueness - 0.30 * r, 0.25, 1.0);

  float3 glow_rgb = sun_glow(xy, S, sg);
  float grad = 0.30 + 0.70 * saturate(0.5 + 0.5 * dot(rhat, u));
  float3 veil_rgb = sun_veil(ct) * grad;
  float theta = atan2(d.y, d.x);
  float3 streak_rgb = sun_streak(dist, theta) * (1.0 - u_blade_curv);
  float3 ghost_rgb = sun_ghost(xy, u);

  float3 add = u_w_glow * glow_rgb
             + u_w_veil * u_coat_flare * 2.3 * veil_rgb
             + u_w_streak * streak_rgb
             + u_w_ghost * u_coat_flare * 1.6 * ghost_rgb;

  outputTex[gid.xy] = float4(u_gate * add, 1.0);   // additive contribution only
}
