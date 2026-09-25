// filter.color.ember_tone — Pass 3: the grade, per pixel.
//
// Everything is decided in OkLCh from the pixel's LIGHTNESS; the input's own
// colour only steers the optional hue offset and source saturation.
//
//   1. CLIMB. x = L_in * drive rises through an exponential shoulder,
//      (1 - e^(-3 x^e)) / (1 - e^-3), to the Body lightness. Chroma climbs
//      with it, so black stays black and the midtones land on a full, warm
//      body rather than on a clipped channel.
//   2. SIZZLE. Past Sizzle Point the pixel races to white: lightness to 1,
//      chroma out on (1 - heat^2), so the hottest pixels pass through a bright
//      tint of the strong hue before they blow out.
//   3. TWO TONES. A soft threshold on L_in picks weak (below) or strong
//      (above). The crossover interpolates the HUE along the arc resolved in
//      analyze.hlsl with chroma held — never through the a/b origin — so the
//      midpoint of two near-opposite colours is a third, equally saturated
//      colour, not grey.
//   4. HUE OFFSET. The pixel's YIQ hue minus the anchor, d in +-0.5 turns,
//      bent by the signed power curve and scaled by Amount, turns the output
//      hue. Gated by the pixel's own chroma: grey has no hue to measure.
//   5. SOURCE SAT. Chroma scaled toward the input's HSV saturation.
//   6. The most of that chroma sRGB can carry at this L and hue
//      (nano_oklch_gamut_clip), then back to sRGB, mixed with the dry input.

#include "nano_color.hlsl"
#include "nano_oklab.hlsl"
#include "common.hlsl"

Texture2D<float4>       inputTex : register(t0);
SamplerState           samp     : register(s1);
StructuredBuffer<float> aBuf     : register(t2);
RWTexture2D<float4>     outTex   : register(u3);

cbuffer U : register(b4) {
  float drive; float body; float chroma; float climb_exp;
  float sizzle; float sizzle_point; float threshold; float threshold_soft;
  float hue_amount; float hue_curve_exp; float source_sat; float mix_amt;
};

static const float kChromaMax = 0.32;   // OkLCh chroma at Chroma 1 (before the gamut)
static const float kSizzleSpan = 0.2;   // x from Sizzle Point to white

[numthreads(8, 8, 1)]
void main(uint3 gid : SV_DispatchThreadID) {
  uint W, H;
  outTex.GetDimensions(W, H);
  if (gid.x >= W || gid.y >= H) return;

  float2 uv = (float2(gid.xy) + 0.5) / float2(W, H);
  float4 src = inputTex.SampleLevel(samp, uv, 0);
  float3 enc = saturate(src.rgb);
  float L_in = nano_linear_srgb_to_oklab(nano_srgb_to_linear(enc)).x;

  // 1. Climb.
  float x = max(L_in * drive, 0.0);
  float climb = (1.0 - exp(-3.0 * pow(x, climb_exp))) / (1.0 - exp(-3.0));
  float L = body * climb;
  float C = chroma * kChromaMax * climb;

  // 2. Sizzle.
  float heat = saturate(sizzle) * smoothstep(sizzle_point, sizzle_point + kSizzleSpan, x);
  L = lerp(L, 1.0, heat);
  C *= 1.0 - heat * heat;

  // 3. Two tones.
  float soft = max(threshold_soft, 1e-4);
  float t = smoothstep(threshold - soft, threshold + soft, L_in);
  float hue = aBuf[3] + (1.0 - t) * aBuf[5];

  // 4. Hue offset.
  if (hue_amount != 0.0) {
    float2 hc = et_yiq_hue(enc);
    float anchor = frac(atan2(aBuf[1], aBuf[0]) / ET_TAU + 1.0);
    float d = frac(hc.x - anchor + 1.5) - 0.5;
    float bent = pow(saturate(2.0 * abs(d)), hue_curve_exp) * 0.5;
    hue += hue_amount * sign(d) * bent * smoothstep(0.02, 0.08, hc.y);
  }

  // 5. Source saturation.
  float in_sat = nano_rgb_to_hsv(enc).y;
  C *= lerp(1.0, in_sat, saturate(source_sat));

  // 6. Into gamut, out to sRGB.
  float3 graded = nano_linear_to_srgb(nano_oklch_gamut_clip(L, C, frac(hue + 1.0)));
  outTex[gid.xy] = float4(lerp(src.rgb, graded, saturate(mix_amt)), src.a);
}
