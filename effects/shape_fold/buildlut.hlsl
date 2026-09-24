// source.shape_fold — auto-levels pass 3: invert the histogram into a remap LUT.
//
// Single-invocation pass (port of buildLUT in app.js). Median → 0, contrast
// leveled by blending a linear two-sided percentile stretch with the
// histogram-equalized CDF (fixed 0.7 mix), with a CLAHE-style bin cap (fixed)
// so a dominant flat region can't pin the median onto itself. Writes
// LUT[0..NB-1] plus lo/hi and a blank flag for present.

#include "common.hlsl"
#include "nano_histogram.hlsl"

static const float SF_LUT_GAUSSIAN = 0.7;   // linear-stretch ↔ hist-equalize mix
static const float SF_LUT_CLIP     = 5.0;   // CLAHE flat-region bin cap (× mean)

StructuredBuffer<int>     stats : register(t1);   // [0]=lo, [1]=hi, [2..]=hist[NB]
RWStructuredBuffer<float> lut   : register(u2);   // [0..NB-1]=LUT, [NB]=lo, [NB+1]=hi, [NB+2]=blank

[numthreads(1, 1, 1)]
void main(uint3 gid : SV_DispatchThreadID) {
  float lo = asfloat(stats[0]);
  float hi = asfloat(stats[1]);
  lut[SF_NB + 0] = lo;
  lut[SF_NB + 1] = hi;

  // Constant/empty field: no contrast → neutral 0 LUT, flagged blank.
  if (hi - lo <= 1e-5) {
    lut[SF_NB + 2] = 1.0;
    for (uint i = 0u; i < SF_NB; i++) lut[i] = 0.0;
    return;
  }
  lut[SF_NB + 2] = 0.0;

  // CLAHE-clipped equalization CDF (shared) + two-sided percentiles.
  float cdf[NANO_HIST_NB];
  nano_hist_cdf(stats, 2u, SF_LUT_CLIP, cdf);

  // Two-sided percentiles: median → 0, [2%,98%] → ±1.
  float range  = hi - lo;
  float median = nano_hist_percentile(cdf, lo, hi, 0.5);
  float p02    = nano_hist_percentile(cdf, lo, hi, 0.02);
  float p98    = nano_hist_percentile(cdf, lo, hi, 0.98);
  float loSpread = max(median - p02, 1e-4);
  float hiSpread = max(p98 - median, 1e-4);

  for (uint i = 0u; i < SF_NB; i++) {
    float v = lo + float(i) / float(SF_NB - 1u) * range;
    int bin = (int)clamp(floor((v - lo) / (range + 1e-9) * float(SF_NB)), 0.0, float(SF_NB - 1u));
    float lin = (v >= median) ? (v - median) / hiSpread : (v - median) / loSpread;
    float linPos = 0.5 + 0.5 * clamp(lin, -1.0, 1.0);
    float pos = linPos * (1.0 - SF_LUT_GAUSSIAN) + cdf[bin] * SF_LUT_GAUSSIAN;
    lut[i] = 2.0 * pos - 1.0;            // median → 0, both sides span [-1, 1]
  }
}
