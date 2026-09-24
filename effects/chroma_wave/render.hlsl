// source.light.chroma_wave — render pass (polyphonic).
//
// Loops over up to MAX_VOICES active blobs — each an independent CPU-managed
// charge/burst prismatic wave. Instead of grading each voice to RGB and adding
// (which washes overlaps toward white), the voices INTERACT in the transfer
// domain: we accumulate a presence-weighted combined band phase T = Σ tᵢ, so
// where voices overlap the hue rotates FURTHER around the wheel and the bands
// compound (more, finer rings). One grade → one soft per-channel rolloff. A
// lone voice (presence ≈ 1 in its body) grades exactly as before.
//   hue_interact blends avg-phase (0, hues just average) → summed-phase
//   (→2, full accumulation / extra rotation).
//
// Per-voice blob geometry + grade params live in a storage buffer (4 float4
// per voice, packed CPU-side); the constant tuning is in the cbuffer.

#include "nano_color.hlsl"

StructuredBuffer<float4> voices : register(t3);   // 4 float4 per voice

Texture2D<float4>   inputTex  : register(t0);
RWTexture2D<float4> outputTex : register(u1);
// Isolated wave layer (rgb on opaque black) — only written when write_wave is
// set (a sink is wired to `wave_out`); a 1x1 dummy is bound otherwise.
RWTexture2D<float4> waveTex   : register(u4);

cbuffer Uniforms : register(b2) {
  // row 0
  float cres_off;
  float band_tilt;
  float hue_span;
  float saturation;
  // row 1
  float alpha_gamma;
  float intensity;
  float base_hue;
  float debug_field;
  // row 2
  float color_r;
  float color_g;
  float color_b;
  uint  voice_count;
  // row 3
  float hue_interact;   // 0 = average overlapping hues, →2 = accumulate/rotate
  float hue_warp_a;     // hue twist: shift(h) = a + b·cos(2πh) + c·sin(2πh)
  float hue_warp_b;
  float hue_warp_c;
  // row 4
  float write_wave;     // 1 = also write the isolated wave layer to waveTex
  float _pw0;
  float _pw1;
  float _pw2;
};

static const float TAU = 6.28318530717958647692;

// Density field for one voice: super-gaussian (plateau) minus an upward-shifted
// carve disk (crescent), anisotropic + aspect-corrected. Returns the signed
// downward coord `qy` for the band tilt.
float voice_field(uint vi, float2 uv, float asp, out float qy_out) {
  float4 a = voices[vi * 4u + 0u];   // cx, cy, radius, elong
  float4 b = voices[vi * 4u + 1u];   // ycomp, sharp, plateau_p, cres
  float cx = a.x, cy = a.y, radius = a.z, elong = a.w;
  float ycomp = b.x, sharp = b.y, plateau_p = b.z, cres = b.w;

  float2 rel = uv - float2(cx, cy);
  float qx = (rel.x * asp) / max(radius * elong, 1e-5);
  float qy = (rel.y)       / max(radius * ycomp, 1e-5);
  qy_out = qy;

  float r2  = qx * qx + qy * qy;
  float qyu = qy + cres_off;
  float ru2 = qx * qx + qyu * qyu;

  float g_main  = exp(-pow(r2, plateau_p) * sharp);
  float g_carve = exp(-ru2 * sharp * 1.6);
  return saturate(g_main - cres * g_carve);
}

[numthreads(8, 8, 1)]
void main(uint3 gid : SV_DispatchThreadID) {
  uint W, H;
  outputTex.GetDimensions(W, H);
  if (gid.x >= W || gid.y >= H) return;

  float asp = float(W) / float(H);
  float2 uv = (float2(gid.xy) + 0.5) / float2(W, H);
  float4 base = inputTex[gid.xy];

  // Accumulate in the transfer domain. `pres` zeroes out voices that aren't
  // here so absent voices don't shift the hue; `bright` is the additive
  // coverage (no band — the band comes from the COMBINED phase).
  float sum_pres = 0.0, sum_phase = 0.0, sum_hoff = 0.0, sum_bc = 0.0, sum_bright = 0.0;
  float dbg = 0.0;
  for (uint vi = 0u; vi < voice_count; vi++) {
    float qy;
    float g = voice_field(vi, uv, asp, qy);
    dbg = max(dbg, g);

    float4 c = voices[vi * 4u + 2u];   // grade_freq, grade_phase, band_contrast, overlay_alpha
    float4 d = voices[vi * 4u + 3u];   // hue_offset, _, _, _
    float t = g * c.x + c.y + band_tilt * qy;
    float pres = smoothstep(0.0, 0.05, g);

    sum_pres   += pres;
    sum_phase  += pres * t;
    sum_hoff   += pres * d.x;
    sum_bc     += pres * c.z;
    sum_bright += c.w * pow(g, alpha_gamma);
  }

  if (debug_field != 0.0) {
    if (write_wave != 0.0) waveTex[gid.xy] = float4(dbg.xxx, 1.0);
    outputTex[gid.xy] = float4(base.rgb * 0.15 + dbg.xxx, base.a);
    return;
  }

  float inv = 1.0 / max(sum_pres, 1e-4);
  // Average phase (overlap → blended hue) vs summed phase (overlap → rotated
  // further). hue_interact tweens between them; a lone voice has avg == sum.
  float T    = lerp(sum_phase * inv, sum_phase, hue_interact);
  float hoff = lerp(sum_hoff  * inv, sum_hoff,  hue_interact);
  float bc   = sum_bc * inv;                     // presence-weighted band contrast

  float hue = base_hue + hoff + T * hue_span;
  // Twist the hue wheel: shift by the R/G/B amounts where the hue lands on
  // each primary, smoothly interpolated (first harmonic through hue 0,1/3,2/3).
  float hw = frac(hue);
  hue += hue_warp_a + hue_warp_b * cos(TAU * hw) + hue_warp_c * sin(TAU * hw);

  float band = 0.5 + 0.5 * cos(TAU * T);
  float band_w = lerp(1.0 - bc, 1.0, band);

  float3 col = nano_hsv_to_rgb(float3(hue, saturation, 1.0))
             * float3(color_r, color_g, color_b);
  float3 bloom = 1.0 - exp(-col * (sum_bright * band_w) * intensity);
  // The isolated layer is exactly the additive contribution on opaque black —
  // composite it over the input downstream and you reconstruct tex_out.
  if (write_wave != 0.0) waveTex[gid.xy] = float4(saturate(bloom), 1.0);
  outputTex[gid.xy] = float4(saturate(base.rgb + bloom), base.a);
}
