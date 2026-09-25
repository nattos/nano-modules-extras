// filter.color.ember_tone — shared bits.
//
// The dominant-hue analysis measures hue in YIQ, exactly as
// color.legacy.bicolor_grad does, on the sRGB-encoded samples. The render pass
// measures each pixel in the same space so its offset from the anchor is a
// like-for-like difference.

#ifndef EMBER_TONE_COMMON_HLSL
#define EMBER_TONE_COMMON_HLSL

static const float3 kRGBToI = float3(0.596, -0.275, -0.321);
static const float3 kRGBToQ = float3(0.212, -0.523,  0.311);
static const float  ET_TAU  = 6.28318530717958648;
static const int    ET_BUCKETS = 64;

// (hue in turns [0,1), chroma).
float2 et_yiq_hue(float3 rgb) {
  float I = dot(rgb, kRGBToI);
  float Q = dot(rgb, kRGBToQ);
  float chroma = sqrt(I * I + Q * Q);
  float hue = (chroma <= 1e-5) ? 0.0 : frac(atan2(Q, I) / ET_TAU + 1.0);
  return float2(hue, chroma);
}

// analysis buffer layout (floats):
//   [0,1] anchor hue as a unit vector (YIQ)   [2] initialized flag
//   [3]   strong hue, OkLCh turns             [4] weak hue, OkLCh turns
//   [5]   the arc from strong to weak, signed turns (|arc| <= 0.5)
static const int ET_ANALYSIS_FLOATS = 8;

#endif
