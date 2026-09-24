// nano_vcr.hlsl — the shared "warm analogue dehancement" grade.
//
// This is the LOOK, factored out so more than one effect can wear it and stay
// byte-identical. It is deliberately NOT a halo generator: how the glow gets
// made is per-effect (source.mesh.three_planes derives it analytically from an
// exact SDF; a post-process sibling would convolve a blur pyramid), but what
// happens to the composite afterwards lives here.
//
// The stack, in order, on a LINEAR HDR accumulator:
//   exposure -> highlight bleach -> highlight tint -> warmth
//            -> asymmetric soft clip -> filmic toe/shoulder -> scanline + grain
//
// Ordering notes that matter:
//   * The highlight bleach runs in HDR, BEFORE any tone map. That is what
//     makes a neon core read as a white-hot filament with the colour surviving
//     only out in the halo. Bleach after tone mapping and you just get a
//     washed-out image.
//   * The highlight TINT runs immediately after the bleach, on the same
//     selector, so it is literally re-colouring what the bleach just took the
//     colour out of. Both are still in HDR, so the tinted highlight goes
//     through the same soft clip as everything else instead of being painted
//     on afterwards.
//   * The soft clip IS the tone map. tanh() saturates, so there is no separate
//     Reinhard/ACES step — stacking two non-linearities only muddies the knee.
//
// See EFFECTS_STYLE_GUIDE.md 3.1 (non-linear curves for warmth) and 3.4.

#ifndef NANO_VCR_HLSL
#define NANO_VCR_HLSL

#include "nano_hash.hlsl"

// Shared parameter block. Both the analytic and the convolution-based effect
// embed this VERBATIM in their cbuffer so the grade is literally the same
// code over the same bytes. 16 floats = 64 bytes = 4 std140 rows.
struct VcrGrade {
  float exposure;         // linear gain before the curve
  float warmth;           // -1 cool (dying CRT) .. +1 warm (tungsten/oxide)
  float drive;            // 0..1, soft-clip knee hardness
  float asymmetry;        // -1..1, pre-clip bias -> uneven even/odd harmonics

  float toe;              // 0..1 dark crush
  float shoulder;         // 0..1 highlight roll-off
  float highlight_desat;  // 0..1 how hard hot pixels bleach to white
  float scanline;         // 0..1 depth of the horizontal line structure

  float scanline_freq;    // scanlines across the full frame height
  float grain;            // 0..1 luma noise
  float grain_seed;       // advance per frame to animate the grain
  float tint_pivot;       // HDR level where the highlight tint starts biting

  float tint_r;           // colour the blown highlights are pulled toward
  float tint_g;
  float tint_b;
  float tint_amount;      // 0..1, 0 = leave the highlights alone
};

// Hot pixels lose saturation and race to white. HDR-only: energy at or below
// 1.0 keeps its hue exactly, so this never touches the body of the image.
float3 nano_vcr_highlight_desat(float3 c, float amount) {
  if (amount <= 0.0) return c;
  float peak = max(c.r, max(c.g, c.b));
  float t = saturate((peak - 1.0) / max(peak, 1e-4));
  return lerp(c, float3(peak, peak, peak), saturate(t * amount));
}

// The inverse of the bleach: pull blown highlights TOWARD a colour instead of
// away from one. Same region of the image, so the two knobs compose
// predictably — bleach first to flatten whatever hue the source had, then tint
// to put a chosen one back.
//
// `tint` is an ABSOLUTE linear colour, not a multiplier: it is literally what a
// fully-clipped pixel becomes, so the swatch reads as the result. That matters
// because of what comes next — nano_vcr_softclip with any real drive saturates
// by about 1.1, so anything still above that lands on white no matter what hue
// it nominally had. A multiplicative tint (or one scaled by the pixel's own
// peak) keeps every channel hot and therefore keeps clipping to white; the
// only way to make a blown core read as a COLOUR is to bring its weaker
// channels down below the knee. Hence: replace, don't scale.
//
// `pivot` is where it starts biting, in HDR units — 1.0 is exactly at clipping,
// lower reaches down into highlights that would have survived the tone map
// intact. The ramp is a smoothstep from `pivot` to `1.5*pivot`: deliberately
// tight, because the soft clip has already flattened everything past ~1.1 to
// white, so a ramp that took a full stop would spend most of its range tinting
// pixels that are indistinguishable from each other and leave a white fringe
// between the tinted core and the halo. Smoothstep rather than a linear ramp
// so that transition still has no crease in it.
float3 nano_vcr_highlight_tint(float3 c, float3 tint, float amount, float pivot) {
  if (amount <= 0.0) return c;
  float p = max(pivot, 1e-3);
  float peak = max(c.r, max(c.g, c.b));
  return lerp(c, tint, smoothstep(p, 1.5 * p, peak) * saturate(amount));
}

// Tungsten/oxide warmth: lift the red end, pull the blue. Signed, so the same
// knob cools the image toward the blue end of a tired CRT.
float3 nano_vcr_warmth(float3 c, float warmth) {
  float3 gain = float3(1.0 + 0.30 * warmth,
                       1.0 + 0.04 * warmth,
                       1.0 - 0.26 * warmth);
  return c * max(gain, 0.0);
}

// Asymmetric per-channel soft clip, normalised so 1.0 maps to 1.0.
//
// `drive` sets the knee hardness; `asymmetry` biases the input BEFORE the
// non-linearity so even and odd harmonics come out unequal — that inequality
// is where "warmth" actually lives. The per-channel spread on `k` mimics film
// dye layers / tape saturating at slightly different levels.
float3 nano_vcr_softclip(float3 c, float drive, float asymmetry) {
  float3 k = (1.0 + drive * 4.0) * float3(1.06, 1.0, 0.94);
  float3 bias = asymmetry * 0.12;
  float3 zero = tanh(bias * k);
  float3 num = tanh((c + bias) * k) - zero;
  float3 den = max(tanh((1.0 + bias) * k) - zero, 1e-4);
  return num / den;
}

// Filmic toe (dark crush) + shoulder (highlight roll-off).
float3 nano_vcr_toe_shoulder(float3 c, float toe, float shoulder) {
  float3 t = pow(saturate(c), 1.0 + toe * 1.5);
  float3 s = t * t * (3.0 - 2.0 * t);          // flat-topped smoothstep
  return lerp(t, s, saturate(shoulder));
}

// Horizontal line structure + luma grain. Applied last, on graded [0,1] data,
// so it reads as a transport artifact rather than as part of the image.
float3 nano_vcr_artifacts(float3 c, float2 uv, float scanline, float freq,
                          float grain, float seed) {
  if (scanline > 0.0) {
    // cos() lobes: bright band between lines, soft dark trough on them.
    float lines = cos(uv.y * freq * 6.2831853);
    c *= 1.0 - scanline * 0.5 * (1.0 - lines * lines);
  }
  if (grain > 0.0) {
    float n = nano_hash21(uv * 1024.0 + seed) - 0.5;
    c += n * grain * 0.12;
  }
  return c;
}

// The whole stack in one call. `uv` is viewport uv [0,1]; only the artifact
// tail reads it.
float3 nano_vcr_grade(float3 c, float2 uv, VcrGrade g) {
  c = max(c, 0.0) * max(g.exposure, 0.0);
  c = nano_vcr_highlight_desat(c, g.highlight_desat);
  c = nano_vcr_highlight_tint(c, float3(g.tint_r, g.tint_g, g.tint_b),
                              g.tint_amount, g.tint_pivot);
  c = nano_vcr_warmth(c, g.warmth);
  c = nano_vcr_softclip(c, g.drive, g.asymmetry);
  c = nano_vcr_toe_shoulder(c, g.toe, g.shoulder);
  c = nano_vcr_artifacts(c, uv, g.scanline, g.scanline_freq, g.grain, g.grain_seed);
  return saturate(c);
}

#endif // NANO_VCR_HLSL
