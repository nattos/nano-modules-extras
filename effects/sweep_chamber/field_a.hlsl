// source.particles.sweep_chamber — field pass A: sweep + downsample.
// One thread per FIELD_RES² texel; 16 bilinear taps on the full-res input in
// a fixed 4×4 cell-relative grid (resolution-independent).
//
// Applies the luma band-pass SWEEP window per tap, then reduces the cell to:
//   .r  — mean swept luma L' (the smooth scalar field_b takes the gradient of;
//         the 4×4 box average is the base smoothing, replacing the old
//         full-res Gaussian pre-blur)
//   .gb — intra-cell offset to the luma peak (texel units, |off| ≤ 0.5).
//         A SHARPENED WEIGHTED CENTROID (weight L'⁴), not an argmax: argmax
//         is discontinuous frame-to-frame (scintillating lines), while the
//         peak-biased centroid is continuous in the input and hugs the peak.
//         Exactly 0 in black cells, so free-space consumers never read junk.
//   .a  — SOFT-MAX swept luma over the cell (quartic power mean): the RIDGE
//         DETECTOR. A ridge thinner than a cell still reads strong here — and
//         unlike |∇L'|, it stays high ON the crest (where the gradient
//         vanishes), which is what line trapping and grip must key on. The
//         power mean (not a hard max) matters: a per-cell hard max is
//         piecewise-quantized, and bilinear interpolation of it imprints the
//         FIELD_RES lattice on everything gated by it (visible as grid
//         patterns in the particle flow). The quartic mean tracks the max
//         closely but varies smoothly across cells.

#include "common.hlsl"

RWTexture2D<float4> fieldA    : register(u0);
Texture2D<float4>   inputTex  : register(t1);
SamplerState        lin       : register(s2);
Texture2D<float4>   fieldPrev : register(t4);   // last frame's field_a (EMA)
RWTexture2D<float4> fieldOr   : register(u5);   // .r = band-side sign σ (curl
                                                // orientation), .g = plain luma

cbuffer Uniforms : register(b3) {
  uint  field_res;
  float sweep_center;
  float sweep_width;
  float sweep_soft;

  float blend_k;     // temporal EMA: 1 = replace outright, →0 = long memory.
                     // Every consumer (particles, tracers, stats) reads the
                     // blended field, so discrete input changes — video
                     // frames, 0.01-quantized sweep drags — become smooth
                     // glides instead of per-frame flow jumps.
  float _p0, _p1, _p2;
}

[numthreads(8, 8, 1)]
void main(uint3 gid : SV_DispatchThreadID) {
  if (gid.x >= field_res || gid.y >= field_res) return;
  float inv_res = 1.0 / float(field_res);

  float Lsum = 0.0, wsum = 0.0, side = 0.0, lum = 0.0;
  float2 cw = float2(0.0, 0.0);
  [unroll] for (int j = 0; j < 4; j++) {
    [unroll] for (int k = 0; k < 4; k++) {
      float2 o  = (float2(j, k) + 0.5) * 0.25;               // cell-relative [0,1)
      float2 uv = (float2(gid.xy) + o) * inv_res;
      float3 c  = inputTex.SampleLevel(lin, uv, 0).rgb;
      float  lm = swc_lum(c);
      float  Lp = swc_sweep(lm, sweep_center, sweep_width, sweep_soft);
      Lsum += Lp;
      lum  += lm;
      side += swc_sweep_side(lm, sweep_center, sweep_width, sweep_soft);
      float w = Lp * Lp; w *= w;                             // L'^4 — peak-biased
      cw   += (o - 0.5) * w;
      wsum += w;
    }
  }
  float2 off = (wsum > 1e-5) ? cw / wsum : float2(0.0, 0.0); // texel units, |off| ≤ 0.5
  float Lsoftmax = pow(wsum * (1.0 / 16.0), 0.25);           // quartic power mean ≈ smooth max
  float4 cur  = float4(Lsum * (1.0 / 16.0), off.x, off.y, Lsoftmax);
  float4 prev = fieldPrev.Load(int3(gid.xy, 0));
  fieldA[gid.xy] = lerp(prev, cur, saturate(blend_k));
  // No EMA here (rgba16f storage is write-only on web): σ is a wide smooth
  // ramp of the luma, steps in it are small and it only ORIENTS the curl.
  fieldOr[gid.xy] = float4(side * (1.0 / 16.0), lum * (1.0 / 16.0), 0.0, 0.0);
}
