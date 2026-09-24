// source.brutal_fold — shared math for the present pass.
//
// A brutalist axonometric-prism GENERATOR. A baked control surface
// (complexity × order × liveliness, plus a co-folded second structure) is
// resolved on the CPU each frame down to two structures' worth of "terms" +
// scene scalars; those ride in a UNIFORM buffer as a flat float array P and the
// GPU composites the receding prism layers from them. The atlas never touches
// the GPU (it's CPU-only constant data in brutal_fold_atlas.h).
//
// The parameter buffer P holds BOTH structures back to back:
//   [ S1 terms (NT*8) | S1 scene (19) | S1 level ]  then the same block for S2.
// `sb` = scene base = NT*8; `str` = per-structure stride. Scene scalars are
// indexed by POSITION (form_scale=+3, back_len=+5, back_ang=+6, extrude=+7,
// layers=+8, sep=+9, front_detail=+10, win_dark=+11, fog=+12, face=+13,
// sky_val=+14, dc=+15, bold_gain=+16, level=+19) — must match build_params and
// the prototype's field.ts / atlas.ts layout.
//
// Ported verbatim from the research testbed's shader.wgsl (occ / samp /
// drawLayer / fieldVal). The field is orthographic (the builder forces sev=0).

#ifndef BRUTAL_FOLD_COMMON_HLSL
#define BRUTAL_FOLD_COMMON_HLSL

#include "nano_color.hlsl"   // nano_hsv_to_rgb
#include "nano_hash.hlsl"    // nano_hash31i (integer-domain white noise)

#define BF_MAXL 6        // up to 6 receding depth layers per structure (12 slots interleaved)

// Shared uniform block (always register b0). 4 std140 rows of scalars, then the
// packed parameter array P (up to 256 floats = both structures).
//
// Colour grade: a 3-control-point tone→colour map (knots at 0 / 0.5 / 1) twists
// the grayscale tone into hue. The diffuse grade runs over the panel tone
// (shadows→mids→highs); the fog grade runs over depth (near→mid→far). Each is a
// hue triple + a saturation; sat=0 → grayscale passthrough (backward compatible).
cbuffer U : register(b0) {
  float res_x;        float res_y;        float nTerms;       float sb;
  float str;          float tilt;         float enable2;      float hAct;
  // Per-control-point grade: hue + saturation (+ brightness for diffuse) at each
  // of the 3 tonal/depth knots; diff_sat / fog_sat are the overall masters.
  float diff_hue_lo;  float diff_hue_mid; float diff_hue_hi;  float diff_sat;
  float diff_sat_lo;  float diff_sat_mid; float diff_sat_hi;  float _dpad0;
  float diff_bri_lo;  float diff_bri_mid; float diff_bri_hi;  float _dpad1;
  float fog_hue_lo;   float fog_hue_mid;  float fog_hue_hi;   float fog_sat;
  float fog_sat_lo;   float fog_sat_mid;  float fog_sat_hi;   float _fpad0;
  // Sky: an ADDITIONAL twist applied to ONLY the infinite-distance (background)
  // pixels, on top of the far-depth fog grade. Defaults (0,0,1) = no change.
  float sky_hue;      float sky_sat;      float sky_bri;      float _spad0;
  // Stochastic "TV static": per-pixel grain on the blob density and (separately)
  // the distance fog. noise_seed reshuffles every frame. 0 = clean.
  // noise_blob_tilt re-weights the blob static by density (see bf_fogAmount).
  float noise_blob;   float noise_fog;    float noise_seed;   float noise_blob_tilt;
  // Volumetric fog: a 3D "shape" blob (twitch-style morph) whose density
  // modulates the fog. These hold the DRIFTED (effective) values from the CPU.
  float vol_amount;   float vol_anchor_x; float vol_anchor_y; float vol_z;
  float vol_shape;    float vol_angle;    float vol_radius;   float vol_softness_xy;
  // vol_depth = blob Z half-extent (small → selective slice). vol_z = Z anchor.
  // Softness is split: vol_softness_xy (screen edge) + vol_softness_z (depth edge).
  float vol_depth;    float vol_softness_z; float _vpad1;     float _vpad2;
  float4 P[64];
};

// 3-control-point quadratic over t∈[0,1] (Lagrange knots at 0, 0.5, 1).
float bf_quad3(float v0, float v1, float v2, float t) {
  t = saturate(t);
  float w0 = (2.0 * t - 1.0) * (t - 1.0);
  float w1 = -4.0 * t * (t - 1.0);
  float w2 = t * (2.0 * t - 1.0);
  return v0 * w0 + v1 * w1 + v2 * w2;
}

// Colorize a grayscale value by a hue twisted across the control axis `t`.
// `value` stays the luminance carrier (value × hue colour); sat=0 → gray.
float3 bf_grade(float value, float hue, float sat) {
  return value * nano_hsv_to_rgb(float3(frac(hue), saturate(sat), 1.0));
}
// The render only produces three clustered diffuse tone LEVELS (from the atlas
// sky/face): shadow/side ≈ 0, front ≈ 0.33, top/highlight ≈ 0.62. Remap the tone
// so those land on the grade's 0 / 0.5 / 1 knots — i.e. Shadows/Mids/Highs hues
// actually grade the side / front / top faces. (Brightness is unaffected: only
// the HUE lookup uses this; bf_grade still multiplies by the real tone.)
#define BF_TONE_FRONT 0.33
#define BF_TONE_TOP   0.62
float bf_toneNorm(float t) {
  if (t < BF_TONE_FRONT) return 0.5 * t / BF_TONE_FRONT;
  return saturate(0.5 + 0.5 * (t - BF_TONE_FRONT) / (BF_TONE_TOP - BF_TONE_FRONT));
}
float3 bf_gradeDiffuse(float tone) {
  float tn = bf_toneNorm(tone);
  float hue = bf_quad3(diff_hue_lo, diff_hue_mid, diff_hue_hi, tn);
  float sat = diff_sat * bf_quad3(diff_sat_lo, diff_sat_mid, diff_sat_hi, tn);   // overall × per-knot
  float bri = bf_quad3(diff_bri_lo, diff_bri_mid, diff_bri_hi, tn);              // per-knot brightness
  return bf_grade(tone * bri, hue, sat);
}
float3 bf_gradeFog(float skyVal, float depthT) {
  float hue = bf_quad3(fog_hue_lo, fog_hue_mid, fog_hue_hi, depthT);
  float sat = fog_sat * bf_quad3(fog_sat_lo, fog_sat_mid, fog_sat_hi, depthT);
  return bf_grade(skyVal, hue, sat);
}
// The infinite-distance sky: the far-depth fog grade, twisted RELATIVE to it —
// sky_hue offsets the far hue, sky_sat adds to the far saturation, sky_bri scales
// the brightness. Defaults (0,0,1) reproduce the far fog colour exactly.
float3 bf_gradeSky(float skyVal) {
  float farHue = bf_quad3(fog_hue_lo, fog_hue_mid, fog_hue_hi, 1.0);
  float farSat = fog_sat * bf_quad3(fog_sat_lo, fog_sat_mid, fog_sat_hi, 1.0);
  float hue = frac(farHue + sky_hue);
  float sat = saturate(farSat + sky_sat);
  float val = saturate(skyVal * sky_bri);
  return nano_hsv_to_rgb(float3(hue, sat, val));
}

// 3D volumetric "shape" blob density at screen pos p0 and normalized depth dz.
// Separable blob: an XY profile (radius vol_radius, edge vol_softness_xy) × a Z slab
// (half-extent vol_depth, edge vol_softness_z). The product is the 3D radial blob
// (a sphere/ellipsoid). vol_shape morphs it (solid → planar slab → radial); the
// sign flips polarity. Splitting XY/Z lets a blob be crisp in screen-space yet
// blend across depth slices (or vice-versa). Returns density in [0,1].
float bf_blob3(float2 p0, float dz) {
  float rxy = length(float2(p0.x - vol_anchor_x, p0.y - vol_anchor_y));
  float rz  = abs(dz - vol_z);
  float softXY = max(vol_softness_xy, 1e-4);
  float softZ  = max(vol_softness_z, 1e-4);

  // XY profile: solid (1) → an oriented band (vol_angle rotates it in screen
  // space) → a radial disk, morphed by |vol_shape|.
  float ang = vol_angle * 6.28318530717958647692;
  float u = (p0.x - vol_anchor_x) * cos(ang) + (p0.y - vol_anchor_y) * sin(ang);
  float xy_band = 1.0 - smoothstep(vol_radius, vol_radius + softXY, abs(u));
  float xy_disk = 1.0 - smoothstep(vol_radius, vol_radius + softXY, rxy);
  float a = abs(vol_shape);
  float xy = lerp(1.0, xy_band, saturate(a / 0.5));          // solid → band
  xy = lerp(xy, xy_disk, saturate((a - 0.5) * 2.0));          // band → disk

  // Z window applies to EVERY shape (incl. solid/band): a slab at vol_z with
  // half-extent vol_depth and a soft edge vol_softness_z. So depth/softness_z
  // control the depth extent of linear shapes too (a thin window = focal plane).
  float dZ = 1.0 - smoothstep(vol_depth, vol_depth + softZ, rz);

  float dens = xy * dZ;
  return (vol_shape >= 0.0) ? dens : (1.0 - dens);
}

// Fog amount (attenuation) at screen pos p0 and normalized depth depthLin.
// Physical model: fog is density accumulated along the view ray. The distance
// term `fogStrength·depthLin` is the uniform base. The blob ADDS local density
// (one-sided, UNGATED by depth — so a blob at the camera fogs the FRONT planes)
// and CUTS the base proportionally where it's sparse (clears a hole). vol_amount
// scales both; at 0 → pure distance fog. The preview mirrors this EXACTLY.
float bf_fogAmount(float2 p0, float depthLin, float fogStrength) {
  // Integer-domain white noise per screen PIXEL (float hashes on large coords band
  // into a few levels — that's why the old static looked ordered). `frame`
  // reshuffles it each frame; the blob/fog use decorrelated seed planes.
  int2 ipx = int2((p0 * 0.5 + 0.5) * float2(res_x, res_y));
  int frame = int(noise_seed);
  float dens = bf_blob3(p0, depthLin);
  // TV static on the blob — ADDITIVE grain (visible even where the blob is faint,
  // unlike multiplicative which vanishes at low density). The active band is a
  // bump centred on a target density: tilt -1 → dense CENTRE, +1 → transparent
  // HALO, 0 → the whole blob. The band narrows toward the endpoints (so more of
  // the blob is left untouched), and a presence floor keeps the empty void clean.
  if (noise_blob > 0.0) {
    float n = nano_hash31i(int3(ipx, frame)) * 2.0 - 1.0;
    float at = abs(noise_blob_tilt);
    float dT = 0.5 - 0.4 * noise_blob_tilt;          // -1→0.9 (centre), +1→0.1 (halo), 0→0.5
    // Narrow the band toward the endpoints so MORE of the blob (the centre at +1,
    // the halo at -1) is left clean. A moderate magnitude keeps the narrow band
    // gentle rather than a harsh snow ring.
    float width = lerp(1.5, 0.2, at);
    float g = exp(-(dens - dT) * (dens - dT) / (2.0 * width * width));
    g *= smoothstep(0.0, 0.03, dens);                 // never fog the empty void
    dens = saturate(dens + noise_blob * 0.5 * g * n);
  }
  float baseRaw = fogStrength * depthLin;
  // TV static on the distance fog (decorrelated via a different seed plane).
  if (noise_fog > 0.0) {
    float n = nano_hash31i(int3(ipx, frame + 40503)) * 2.0 - 1.0;
    baseRaw = saturate(baseRaw * (1.0 + noise_fog * n));
  }
  float s = 2.0 * dens - 1.0;                 // [-1,1]: + adds, - cuts
  float add = vol_amount * max(s, 0.0);       // local density added (any depth)
  float cut = vol_amount * max(-s, 0.0);      // [0,1] fraction of base cleared
  return saturate(baseRaw * (1.0 - cut) + add);
}

// Unpack float i from the vec4-packed parameter array (mirrors the WGSL P(i)).
float Pf(uint i) { return P[i >> 2u][i & 3u]; }

// Occupancy O at field coord w, for the structure whose terms start at `tb` and
// scene at `sb`. Multiplicative AND terms (mix>=0.5) carve, additive terms union.
float bf_occ(uint tb, uint sbase, float2 w) {
  uint nt = (uint)nTerms;
  float uni = 0.0;
  float inter = 1.0;
  for (uint i = 0u; i < nt; i++) {
    uint o = tb + i * 8u;
    float th = Pf(o);
    float mth = Pf(o + 1u);
    float d = (cos(th) * w.x + sin(th) * w.y)
            + Pf(o + 2u) * (cos(mth) * w.x + sin(mth) * w.y);
    float q = frac(d * Pf(o + 3u) + Pf(o + 4u)) - 0.5;
    float h = Pf(o + 5u);
    float box = (h - abs(q)) > 0.0 ? 1.0 : 0.0;
    float a = clamp(h / hAct, 0.0, 1.0);
    if (Pf(o + 7u) >= 0.5) {
      inter = inter * ((1.0 - a) + a * box);
    } else {
      uni = uni + Pf(o + 6u) * box;
    }
  }
  return Pf(sbase + 15u) + uni + Pf(sbase + 16u) * inter;
}

float bf_samp(uint tb, uint sbase, float px, float py, float fs) {
  return bf_occ(tb, sbase, float2(px * fs, py * fs)); // orthographic (sev forced 0 by the builder)
}

// One receding layer -> float2(tone, coverage). `tilt` shears screen-x (faces at
// an angle). Solid mask = step(occ - level); depth is faked by sampling the mask
// again offset by the recession vector (ex,ey) to find the shadowed side band.
float2 bf_drawLayer(uint tb, uint sbase, float level, float tilt_, float ex, float ey,
                    float fs, float offx, float offy, float2 p0) {
  float face = Pf(sbase + 13u);
  float sky = Pf(sbase + 14u);
  float frontT = sky - face;
  float topT = sky - face * 0.42;
  float sideT = max(sky - face * 1.75, 0.0);
  float winDark = Pf(sbase + 11u);
  float frontDetail = Pf(sbase + 10u);
  float psx = (p0.x + tilt_ * p0.y) - offx;
  float psy = p0.y - offy;
  float front = bf_samp(tb, sbase, psx, psy, fs) > level ? 1.0 : 0.0;
  float qx = psx - ex;
  float qy = psy - ey;
  float br = bf_samp(tb, sbase, qx, qy, fs);
  float back = br > level ? 1.0 : 0.0;
  if (max(front, back - front) <= 0.5) {
    return float2(0.0, 0.0);
  }
  float t = frontT;
  if (front < 0.5) {
    float e = 0.02;
    float g1 = bf_samp(tb, sbase, qx + e, qy, fs) - br;
    float g2 = bf_samp(tb, sbase, qx, qy + e, fs) - br;
    t = lerp(sideT, topT, smoothstep(0.45, 0.7, abs(g2) / (abs(g1) + abs(g2) + 1e-6)));
  } else if (frontDetail > 0.0) {
    float det = bf_samp(tb, sbase, psx, psy, fs * 3.0) > level ? 1.0 : 0.0;
    t = frontT * (1.0 - winDark * frontDetail * det);
  }
  return float2(t, 1.0);
}

// Both structures' receding layers, sorted + interleaved by depth, composited
// opaque near-over-far with per-layer depth fog toward the sky tone.
float3 bf_fieldVal(float2 p0) {
  uint sbA = (uint)sb;
  uint strv = (uint)str;
  float sky = Pf(sbA + 14u);
  float bl = Pf(sbA + 5u);
  float ba = Pf(sbA + 6u);
  float sep = Pf(sbA + 9u);
  float fog = Pf(sbA + 12u);
  float ex = bl * cos(ba);   // shared recession direction
  float ey = -bl * sin(ba);
  float n1 = Pf(sbA + 8u);
  float n2 = Pf(strv + sbA + 8u);
  float nm1 = max(n1 - 1.0, 1.0);
  float nm2 = max(n2 - 1.0, 1.0);
  float Ex1 = cos(ba) * Pf(sbA + 7u);
  float Ey1 = -sin(ba) * Pf(sbA + 7u);
  float Ex2 = cos(ba) * Pf(strv + sbA + 7u);
  float Ey2 = -sin(ba) * Pf(strv + sbA + 7u);
  float fs1 = Pf(sbA + 3u);
  float fs2 = Pf(strv + sbA + 3u);
  float lvl1 = Pf(sbA + 19u);
  float lvl2 = Pf(strv + sbA + 19u);
  bool on2 = enable2 > 0.5;

  // Background = the infinite-distance sky (far fog grade + the sky twist).
  float3 col = bf_gradeSky(sky);
  for (int s = 2 * BF_MAXL - 1; s >= 0; s = s - 1) { // far -> near
    float d = (float)s * 0.5;
    float lidx = floor(d + 0.001);
    float2 r = float2(0.0, 0.0);
    float fogv = 0.0;
    float depthT = 0.0;
    bool drawn = false;
    if ((s & 1) == 0) { // structure 1 (integer depths)
      if (d < n1) {
        r = bf_drawLayer(0u, sbA, lvl1, 0.0, Ex1, Ey1, fs1, ex * (d * sep), ey * (d * sep), p0);
        depthT = d / nm1;
        fogv = bf_fogAmount(p0, depthT, fog);
        drawn = true;
      }
    } else if (on2 && lidx < n2) { // structure 2 (half-step depths)
      r = bf_drawLayer(strv, strv + sbA, lvl2, tilt, Ex2, Ey2, fs2,
                       ex * (d * sep), ey * (d * sep), p0);
      depthT = lidx / nm2;
      fogv = bf_fogAmount(p0, depthT, fog);
      drawn = true;
    }
    if (drawn && r.y > 0.5) {
      // Graded diffuse tone, faded toward the fog colour. The fog HUE grades by
      // DEPTH (not the blob-modulated amount) so the blob changes opacity, not hue.
      col = lerp(bf_gradeDiffuse(r.x), bf_gradeFog(sky, depthT), fogv);
    }
  }
  return col;
}

#endif // BRUTAL_FOLD_COMMON_HLSL
