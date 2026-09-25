// nano_oklab.hlsl — Oklab / OkLCh conversions and a hue-preserving gamut clip.
//
// Oklab (Björn Ottosson, 2020) is the space to reach for when a colour move
// has to LOOK even: complements, hue arcs and chroma pulls land where the eye
// expects. See EFFECTS_STYLE_GUIDE.md §3.4.
//
// Conventions:
//   * The Oklab functions take and return LINEAR sRGB. Sampled 8-bit video is
//     sRGB-encoded — decode with nano_srgb_to_linear (nano_color.hlsl) first.
//   * OkLCh hue is in TURNS, [0, 1), so hue arithmetic wraps with frac().

#ifndef NANO_OKLAB_HLSL
#define NANO_OKLAB_HLSL

static const float NANO_TAU = 6.28318530717958648;

float _nano_cbrt(float x) { return sign(x) * pow(abs(x), 1.0 / 3.0); }

float3 nano_linear_srgb_to_oklab(float3 c) {
  float l = 0.4122214708 * c.r + 0.5363325363 * c.g + 0.0514459929 * c.b;
  float m = 0.2119034982 * c.r + 0.6806995451 * c.g + 0.1073969566 * c.b;
  float s = 0.0883024619 * c.r + 0.2817188376 * c.g + 0.6299787005 * c.b;
  l = _nano_cbrt(l); m = _nano_cbrt(m); s = _nano_cbrt(s);
  return float3(0.2104542553 * l + 0.7936177850 * m - 0.0040720468 * s,
                1.9779984951 * l - 2.4285922050 * m + 0.4505937099 * s,
                0.0259040371 * l + 0.7827717662 * m - 0.8086757660 * s);
}

float3 nano_oklab_to_linear_srgb(float3 lab) {
  float l = lab.x + 0.3963377774 * lab.y + 0.2158037573 * lab.z;
  float m = lab.x - 0.1055613458 * lab.y - 0.0638541728 * lab.z;
  float s = lab.x - 0.0894841775 * lab.y - 1.2914855480 * lab.z;
  l = l * l * l; m = m * m * m; s = s * s * s;
  return float3( 4.0767416621 * l - 3.3077115913 * m + 0.2309699292 * s,
                -1.2684380046 * l + 2.6097574011 * m - 0.3413193965 * s,
                -0.0041960863 * l - 0.7034186147 * m + 1.7076147010 * s);
}

// (L, a, b) -> (L, C, h turns) and back.
float3 nano_oklab_to_oklch(float3 lab) {
  float C = length(lab.yz);
  float h = (C > 1e-6) ? frac(atan2(lab.z, lab.y) / NANO_TAU + 1.0) : 0.0;
  return float3(lab.x, C, h);
}

float3 nano_oklch_to_oklab(float3 lch) {
  float a = NANO_TAU * lch.z;
  return float3(lch.x, lch.y * cos(a), lch.y * sin(a));
}

bool _nano_in_gamut(float3 rgb) {
  return all(rgb >= -1e-4) && all(rgb <= 1.0 + 1e-4);
}

// The most chroma, up to the requested C, that (L, h) can carry inside the
// [0,1] linear-sRGB cube. Hue and lightness are kept exactly — only chroma
// gives — so a colour pulled into gamut stays as punchy as the display allows
// instead of shifting hue the way a per-channel clamp does. Returns linear
// sRGB. L outside [0,1] is clamped first (there is no chroma at either end).
float3 nano_oklch_gamut_clip(float L, float C, float h) {
  L = saturate(L);
  float3 rgb = nano_oklab_to_linear_srgb(nano_oklch_to_oklab(float3(L, C, h)));
  if (_nano_in_gamut(rgb)) return rgb;
  float lo = 0.0, hi = max(C, 0.0);
  [unroll] for (int i = 0; i < 10; i++) {
    float mid = 0.5 * (lo + hi);
    float3 t = nano_oklab_to_linear_srgb(nano_oklch_to_oklab(float3(L, mid, h)));
    if (_nano_in_gamut(t)) lo = mid; else hi = mid;
  }
  return saturate(nano_oklab_to_linear_srgb(nano_oklch_to_oklab(float3(L, lo, h))));
}

#endif // NANO_OKLAB_HLSL
