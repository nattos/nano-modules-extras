// source.mesh.monolith — god rays (light shafts from the surface above).
//
// For each pixel, march toward the sun's screen position accumulating a
// SYNTHETIC radiance — a sun-centered glow standing in for light entering
// through the water surface — gated by (1 - occlusion) from the composite
// alpha, with exponential decay per tap. The composite's COLOR is never
// gathered: scattering the 2D input radially reads as a camera-facing
// billboard being smeared, not a volume. Taps that leave the frame count
// as unoccluded (open water), so shafts don't truncate at the edges. The
// black slab carves dark shafts; the top-anchored caustic field (below)
// does the surface modulation. Output is an additive RGBA16F layer.

#include "caustics.hlsl"

Texture2D<float4>   compTex : register(t0);
RWTexture2D<float4> raysTex : register(u1);
SamplerState        clampS  : register(s2);

cbuffer Uniforms : register(b3) {
  float4 sun_screen;   // sun px, py, water_t, gain (rays * fade * sun)
  float4 march;        // taps, decay, max_step_px, caustics amount
  float4 glow;         // inv glow radius (px^-1), sun_color rgb
  float4 sun_env;      // sun sample uv.xy, mode (1 = hue from env), 0
};

Texture2D<float4> envTex : register(t4);   // sun-color sample source

[numthreads(8, 8, 1)]
void main(uint3 gid : SV_DispatchThreadID) {
  uint W, H;
  raysTex.GetDimensions(W, H);
  if (gid.x >= W || gid.y >= H) return;

  float2 p = float2(gid.xy) + 0.5;
  float2 inv_dims = float2(1.0 / float(W), 1.0 / float(H));
  float2 delta = sun_screen.xy - p;
  int taps = (int)march.x;
  float2 step_px = delta / march.x;
  float sl = length(step_px);
  if (sl > march.z) step_px *= march.z / sl;

  // Per-pixel march jitter kills the stair-step banding a fixed-phase
  // radial march leaves around the silhouette (stable per pixel — no
  // temporal shimmer).
  float jitter = nano_hash21(float2(gid.xy));

  float accum = 0.0;
  float wsum = 0.0;
  float w = 1.0;
  for (int i = 1; i <= taps; i++) {
    float2 sp_px = p + step_px * (float(i) - 1.0 + jitter);
    float2 sp = sp_px * inv_dims;
    float occ_t = 0.0;
    if (sp.x >= 0.0 && sp.x <= 1.0 && sp.y >= 0.0 && sp.y <= 1.0) {
      occ_t = compTex.SampleLevel(clampS, sp, 0).a;
    }
    // Synthetic surface light: the WHOLE surface emits (floor), brightest
    // toward the sun point — an off-screen sun must not extinguish the
    // shafts, only bias them.
    float d = length(sp_px - sun_screen.xy) * glow.x;
    float g = 0.30 + 0.70 / (1.0 + d * d);
    accum += (1.0 - saturate(occ_t)) * g * w;
    wsum += w;
    w *= march.y;
  }
  float rs = (wsum > 1e-4 ? accum / wsum : 0.0) * sun_screen.w;
  float3 tint = glow.yzw;   // sun color
  if (sun_env.z > 0.5) {
    float2 su = sun_env.xy;
    float3 e = envTex.SampleLevel(clampS, su, 0).rgb
             + envTex.SampleLevel(clampS, su + float2(0.02, 0.0), 0).rgb
             + envTex.SampleLevel(clampS, su - float2(0.02, 0.0), 0).rgb
             + envTex.SampleLevel(clampS, su + float2(0.0, 0.02), 0).rgb
             + envTex.SampleLevel(clampS, su - float2(0.0, 0.02), 0).rgb;
    tint *= nano_sun_chroma(e * 0.2);
  }

  // Water caustics: COLUMNS of differing density, not a pasted overlay.
  // The modulation is (nearly) constant ALONG each shaft and varies only
  // ACROSS the fan: it samples the caustic field on the shaft's unit
  // direction from the sun point (seamless around the full circle). With
  // the usual overhead/off-screen sun the in-frame fan is near-parallel,
  // so this reads as vertical light columns; with a centered eclipse sun
  // it becomes radial spokes. A faint along-shaft term keeps the columns
  // from being laser-uniform, and the water clock swings the whole fan.
  float ca = march.w;
  if (ca > 0.0) {
    float2 d_sun = p - sun_screen.xy;
    float r_px = length(d_sun);
    if (r_px > 1.0) {
      float2 dirs = d_sun / r_px;
      float along = r_px * glow.x;
      float c = nano_caustic2(dirs * 3.4 + float2(0.0, along * 0.15),
                              sun_screen.z);
      rs *= max(1.0 + ca * (c * 3.2 - 0.75), 0.0);
      // Mild 2D breakup (+-22%) so the columns never go laser-uniform,
      // especially near the sun point where the fan converges — and a
      // subtle per-column warm/cool dispersion around the sun color.
      float b = nano_value_noise2(p * (2.2 / float(W)) +
                                  float2(sun_screen.z * 0.15,
                                         -sun_screen.z * 0.10));
      rs *= 1.0 + ca * (0.44 * b - 0.22);
      tint *= 1.0 + ca * (b - 0.5) * float3(0.4, 0.0, -0.4);
    }
  }
  raysTex[gid.xy] = float4(max(rs * tint, 0.0), 0.0);
}
