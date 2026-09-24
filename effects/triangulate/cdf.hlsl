// triangulate — turn the ridge/corner histograms into normalization divisors:
// the value at a high percentile of each map. Dividing by it (in remap) auto-
// levels each map to ~[0,1] per frame, robust to input scale and to the
// difference between ridge magnitude and the much smaller corner determinant.
// Single-thread serial reduction (bins is tiny), then zero the bins for the
// next frame.
RWStructuredBuffer<uint>  hist : register(u0);
RWStructuredBuffer<float> pct  : register(u1);   // [0]=ridge divisor, [1]=corner divisor

cbuffer CdfUniforms : register(b2) {
  uint  u_bins;
  float u_percentile;   // e.g. 0.97 → normalize the top few % of feature pixels to 1
  float u_pad0, u_pad1;
};

float percentile_of(uint base) {
  uint total = 0u;
  for (uint k = 0u; k < u_bins; ++k) total += hist[base + k];
  if (total == 0u) return 1.0;
  uint target = (uint)(u_percentile * (float)total);
  uint cum = 0u;
  for (uint k = 0u; k < u_bins; ++k) {
    cum += hist[base + k];
    if (cum >= target) return ((float)k + 1.0) / (float)u_bins;
  }
  return 1.0;
}

[numthreads(1, 1, 1)]
void main(uint3 gid : SV_DispatchThreadID) {
  if (gid.x != 0u || gid.y != 0u || gid.z != 0u) return;
  pct[0] = max(percentile_of(0u),      1e-3);
  pct[1] = max(percentile_of(u_bins),  1e-3);
  for (uint k = 0u; k < u_bins * 2u; ++k) hist[k] = 0u;   // reset for next frame
}
