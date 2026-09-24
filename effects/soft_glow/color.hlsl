// source.light.soft_glow — render pass (v1 scaffold).
//
// Per pixel: sum gaussian contributions from N blobs, run the
// accumulated intensity through a hue-shifting ramp (low → hue_low,
// high → hue_high), crush toward white as the accumulated value
// approaches white_point. Additive over tex_in.
//
// Deferred for later iteration:
// - Random-walk LFO drift (currently simple per-blob velocity).
// - Divergence (per-bar hue offsets) — uniforms reserved but unused.

#include "nano_color.hlsl"
#include "nano_bars.hlsl"

static const uint MAX_BLOBS = 32u;

struct Blob {
  // .xy = uv-space center; .z = radius (cover-square units);
  // .w = per-blob amplitude (breathing/drift), CPU-driven.
  float4 pos_size;
  // .x = hue_offset; .yz = velocity (motion shader only).
  float4 jitters;
};

StructuredBuffer<Blob> blobs : register(t0);
Texture2D<float4>      inputTex  : register(t1);
RWTexture2D<float4>    outputTex : register(u2);

cbuffer Uniforms : register(b3) {
  uint  blob_count;
  float intensity;
  float ramp_curve;
  float white_point;

  float hue;             // hue at peak amplitude
  float hue_shift;       // added to hue as amp drops; |shift|>1 → banding
  float saturation;
  float aspect_x;        // (min(W,H) / W) — for cover-square radius

  float aspect_y;
  float intensity_skew;  // 0=isotropic blob, 1=wavefront-only contribution
  float hue_curve;       // power on (1-ramp_t) — shapes hue-transition contrast
  float overflow_band;   // 0=soft-clip hue at peak; >0 keeps rotating → banding

  float color_strength;  // final multiplier on glow rgb (color only — no motion impact)
  float _pad_a;
  float _pad_b;
  float _pad_c;
};

// Plummer softening scale: at skew=0 there's no singularity to soften,
// at skew=1 we want enough softening to kill the sub-pixel direction
// flip but leave the wavefront crescent intact. Linear in skew keeps
// the dial single-knob "playable".
static const float SKEW_SOFTENING_SCALE = 0.06;

[numthreads(8, 8, 1)]
void main(uint3 gid : SV_DispatchThreadID) {
  uint W, H;
  outputTex.GetDimensions(W, H);
  if (gid.x >= W || gid.y >= H) return;

  float2 uv = (float2(gid.xy) + 0.5) / float2(W, H);

  // Accumulate gaussian contributions from all blobs.
  float accum = 0.0;
  for (uint i = 0u; i < blob_count; i++) {
    Blob b = blobs[i];
    if (b.pos_size.z <= 0.0) continue;
    // Aspect-corrected distance (cover-square): make the gaussian round.
    float2 d = uv - b.pos_size.xy;
    d.x /= max(aspect_x, 1e-4);
    d.y /= max(aspect_y, 1e-4);
    float r2 = dot(d, d) / (b.pos_size.z * b.pos_size.z);

    // Wavefront skew — mirror of motion shader's skew. Positive when
    // the pixel lies ahead of the blob in the direction of its
    // velocity; zero behind. At intensity_skew=0 the mask is 1 (full
    // isotropic gaussian), at 1 the contribution concentrates on the
    // leading edge.
    //
    // Plummer softening replaces d/|d| with d/sqrt(|d|²+s²): the
    // direction vector smoothly collapses to 0 at the blob center
    // instead of flipping wildly within a sub-pixel of the singularity.
    // Outside the softening radius behavior matches an unsoftened
    // d_hat exactly.
    float2 v = b.jitters.yz;
    float v_len = length(v);
    float skew = saturate(intensity_skew);
    float front_factor = 1.0;
    if (v_len > 1e-5) {
      // Softening tied directly to skew — one knob.
      float s = skew * SKEW_SOFTENING_SCALE;
      float soft_inv_r = rsqrt(dot(d, d) + s * s);
      float2 d_soft = d * soft_inv_r;
      // smoothstep instead of saturate: a hard saturate would clip the
      // negative half of dot() to zero, leaving a C1-discontinuous seam
      // along the perpendicular plane through the blob center ("two
      // lobes glued together"). smoothstep has zero slope at both
      // endpoints, so the back boundary fades in C1-continuously.
      front_factor = smoothstep(0.0, 1.0, dot(d_soft, v / v_len));
    }
    float skew_mask = lerp(1.0, front_factor, skew);

    accum += exp(-r2 * 2.0) * b.pos_size.w * skew_mask;   // amp × skew
  }
  accum *= intensity;

  // Ramp lookup. ramp_curve shapes the inner response (the part of
  // the gradient up to the peak); the overflow beyond white_point is
  // tracked separately and feeds banding via overflow_band.
  float norm    = max(accum / max(white_point, 1e-4), 0.0);
  float ramp_t  = pow(saturate(norm), max(ramp_curve, 1e-3));
  // hue at brightest; shifts toward (hue + hue_shift) as amplitude
  // drops. |hue_shift|>1 wraps the wheel multiple times across the
  // ramp → concentric banding around each blob.
  //
  // hue_curve > 1 → hue clings to peak until late in the ramp, then
  // races toward (peak+shift) near zero amp (sharp edge contrast).
  // hue_curve < 1 → hue shifts quickly even at high amp and plateaus
  // (contrast near the peak, big plateau out at the rim).
  float hue_t = pow(saturate(1.0 - ramp_t), max(hue_curve, 1e-3));
  // Overflow band: when blobs overlap to push norm above 1, the
  // standard hue_t saturates at 0 (everything pins to hue — the
  // "soft clip" mode). overflow_band > 0 keeps the hue rotating past
  // the peak in the same direction the inner gradient was already
  // going (hue → hue + hue_shift × hue_t), so frac() at the HSV step
  // wraps the result into concentric bands of the wheel.
  float overflow = max(norm - 1.0, 0.0);
  float h_out = hue + hue_shift * (hue_t - overflow * overflow_band);

  // Brightness crushes toward white as accumulated approaches white_point.
  float sat = saturation * (1.0 - ramp_t * 0.6);
  float val = saturate(accum);

  float3 glow = nano_hsv_to_rgb(float3(frac(h_out), sat, val));

  float4 base = inputTex[gid.xy];
  // color_strength scales the emitted glow (NOT the motion field — the
  // motion shader has its own scaling). 0 = passthrough; 1 = nominal;
  // >1 brightens, clamped by saturate.
  //
  // Output alpha: the glow is rendered additively over the input. To
  // stay visible against a transparent background we have to push our
  // own alpha contribution — `val` is the HSV-pre-conversion brightness
  // (= saturate(accum)), which is exactly the "how much glow is here"
  // signal. max() with base.a keeps the input's opacity intact and
  // lets the glow only INCREASE opacity, never decrease it.
  float glow_alpha = saturate(val * color_strength);
  outputTex[gid.xy] = float4(
      saturate(base.rgb + glow * color_strength),
      max(base.a, glow_alpha));
}
