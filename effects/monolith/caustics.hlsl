// source.mesh.monolith — shared fake water caustic field.
//
// Two counter-drifting value-noise layers; bright filaments appear where
// they agree (the interference trick), sharpened to thin webs. Mean level
// ~0.2 — callers rebalance around 1.0 with their own gain/dim so cranking
// the amount doesn't change overall brightness. `t` is the free-running
// water clock (the ocean does not obey the transport).

#include "nano_hash.hlsl"

// Chroma-only transfer for Sun Source = From Input: given the env color
// averaged around the sun point (callers do a stabilized 5-tap cross),
// normalize by its own luminance so the env contributes HUE while the
// Sun knob keeps owning intensity. Fades to neutral when the sampled
// spot is near-black (no chroma noise from dark video).
float3 nano_sun_chroma(float3 e) {
  float l = dot(e, float3(0.299, 0.587, 0.114));
  float3 chroma = e / max(l, 0.12);
  return lerp(float3(1.0, 1.0, 1.0), chroma, smoothstep(0.02, 0.08, l));
}

float nano_caustic2(float2 p, float t) {
  // Broad interference layer: the big swaying bands.
  float n1 = nano_value_noise2(p + float2(t * 0.70, t * 0.40));
  float n2 = nano_value_noise2(p * 1.31 + float2(17.3 - t * 0.55, 9.1 + t * 0.62));
  float d = saturate(1.0 - abs(n1 - n2) * 2.2);
  float d2 = d * d;
  float base = d2 * d2 * d2;   // ^6 filament sharpening
  // Finer counter-drifting layer breaks the bands into the classic web.
  float m1 = nano_value_noise2(p * 2.70 + float2(31.7 + t * 0.90, 5.9 - t * 0.75));
  float m2 = nano_value_noise2(p * 3.43 + float2(t * 0.80 - 12.1, 27.7 + t * 1.05));
  float e = saturate(1.0 - abs(m1 - m2) * 2.2);
  float e2 = e * e;
  float fine = e2 * e2;        // ^4 — a touch softer than the base layer
  return saturate(base * (0.35 + 0.9 * fine) + 0.25 * fine);
}
