// filter.legacy.glisten — anchor finder (single thread).
//
// Faithful port of NanoGraph GlistenFindAnchor.txt. Runs on the BLURRED 64×64
// search grid (not the full-res input): an 8×8 coarse argmax at cell centres,
// then an 8×8 fine argmax within the winning cell, then 4 linear taps for the
// luma gradient (stretch direction) and per-channel colour gradients (tint).
//
// Deliberately preserved original quirks — all load-bearing for the look:
//  - anchorUV gains an extra +coarseTexel/2, sitting the fan half a cell
//    down-right of the peak, so the gradient taps read down-slope colours.
//  - the fine texture carries a ×2-per-pass gain (BlurF contrast=1) but is
//    unorm8 like every "Global" texture in the original, so it SATURATES at
//    1 rather than amplifying — bright regions plateau and the fine argmax
//    tie-breaks to the first (top-left) saturated texel.
//  - the colour-gradient channels reassemble in R,B,G order (a shipped swap).
//  - gradient taps use clamp-to-zero addressing (the anchor offset can push
//    them past the border, reading black — an inward edge gradient).
//  - Position = anchorUV + grad: the fan rides the gradient off the peak.
//
// anchor buffer layout (floats):
//   [0,1]    Position.xy (uv)    [2,3]   Direction.xy (unit grad normal)
//   [4,5]    Grad.xy             [6..8]  Color.rgb
//   [9..11]  ColorGradX.rgb      [12..14] ColorGradY.rgb
//   [15]     valueAvg

#include "nano_color.hlsl"   // nano_srgb_to_linear

Texture2D<float4>         coarseTex : register(t0);
Texture2D<float4>         fineTex   : register(t1);
SamplerState              samp      : register(s2);
RWStructuredBuffer<float> anchor    : register(u3);

cbuffer U : register(b4) {
  float color_grad_soft;
  float color_grad_squash;
  float color_grad_adjust;
  float _p0;
};

static const float PI = 3.14159265358979323846;
static const float SAMPLING_WIDTH = 0.005;

float lum(float3 c) { return max(c.r, max(c.g, c.b)); }

// clamp_to_zero addressing (the original's sampler mode): out-of-range reads
// return transparent black instead of the edge texel. The search textures
// hold sRGB codes (the original's storage format); decode to linear — the
// gradient/colour math ran in linear space. The argmax loops skip the decode
// (it's monotonic, the winner is the same texel).
float4 tapZero(float2 uv) {
  if (any(uv != saturate(uv))) return float4(0, 0, 0, 0);
  float4 t = fineTex.SampleLevel(samp, uv, 0);
  t.rgb = nano_srgb_to_linear(t.rgb);
  return t;
}

[numthreads(1, 1, 1)]
void main(uint3 id : SV_DispatchThreadID) {
  const int C = 8;                      // coarse cells per axis
  const int F = 8;                      // fine texels per cell per axis
  float2 cstep = 1.0 / float(C);
  float2 fstep = 1.0 / float(C * F);

  // ---- coarse pass: one tap per cell centre (the OneEighth resample) ----
  float2 bestC = float2(0.0, 0.0);
  float  bestCV = -1.0;
  for (int cy = 0; cy < C; ++cy) {
    for (int cx = 0; cx < C; ++cx) {
      float2 uv = (float2(cx, cy) + 0.5) * cstep;
      float v = lum(coarseTex.SampleLevel(samp, uv, 0).rgb);
      if (v > bestCV) { bestCV = v; bestC = float2(cx, cy) * cstep; }
    }
  }

  // ---- fine pass within the winning cell ----
  float2 bestF = float2(0.0, 0.0);
  float  bestFV = -1.0;
  for (int fy = 0; fy < F; ++fy) {
    for (int fx = 0; fx < F; ++fx) {
      float2 uv = bestC + (float2(fx, fy) + 0.5) * fstep;
      float v = lum(fineTex.SampleLevel(samp, uv, 0).rgb);
      if (v > bestFV) { bestFV = v; bestF = float2(fx, fy); }
    }
  }
  // Original quirk: the extra +cstep/2 offsets the anchor half a coarse cell.
  float2 anchorUV = bestC + bestF * fstep + fstep * 0.5 + cstep * 0.5;

  // ---- local gradient (4 linear-space taps, clamp-to-zero addressed) ----
  float sw = SAMPLING_WIDTH;
  float4 s01 = tapZero(anchorUV + float2(-sw, 0));
  float4 s21 = tapZero(anchorUV + float2( sw, 0));
  float4 s10 = tapZero(anchorUV + float2(0, -sw));
  float4 s12 = tapZero(anchorUV + float2(0,  sw));
  float v01 = lum(s01.rgb), v21 = lum(s21.rgb), v10 = lum(s10.rgb), v12 = lum(s12.rgb);
  float valueAvg = (v01 + v21 + v10 + v12) * 0.25;

  float2 grad = float2(v21 - v01, v12 - v10);
  float gl = length(grad);
  // Degenerate-input fallback (uniform frame): keep a unit direction so the
  // fan doesn't collapse to a point. The original emitted (0,0) here.
  float2 gradNorm = (gl > 1e-6) ? (grad / gl) : float2(1.0, 0.0);

  float3 color = (s01.rgb + s21.rgb + s10.rgb + s12.rgb) * 0.25;
  float3 dCdx = s21.rgb - s01.rgb;
  float3 dCdy = s12.rgb - s10.rgb;

  // Per-channel atan compression, then the original's R,B,G reassembly.
  float2 gR = float2(dCdx.r, dCdy.r);
  float2 gG = float2(dCdx.g, dCdy.g);
  float2 gB = float2(dCdx.b, dCdy.b);
  gR /= max(atan((length(gR) + color_grad_soft) * color_grad_squash) * (PI * 2.0), 1e-6);
  gG /= max(atan((length(gG) + color_grad_soft) * color_grad_squash) * (PI * 2.0), 1e-6);
  gB /= max(atan((length(gB) + color_grad_soft) * color_grad_squash) * (PI * 2.0), 1e-6);
  float3 colorGradX = float3(gR.x, gB.x, gG.x) * valueAvg;
  float3 colorGradY = float3(gR.y, gB.y, gG.y) * valueAvg;

  // valueAvg ≤ 1 (unorm8 fine texture); bright anchors take the full adjust.
  float adj = 1.0 + (color_grad_adjust - 1.0) * valueAvg;
  color -= (gradNorm.x * colorGradX + gradNorm.y * colorGradY) * adj;

  float2 pos = anchorUV + grad;

  anchor[0] = pos.x;      anchor[1] = pos.y;
  anchor[2] = gradNorm.x; anchor[3] = gradNorm.y;
  anchor[4] = grad.x;     anchor[5] = grad.y;
  anchor[6] = color.r;    anchor[7] = color.g;    anchor[8] = color.b;
  anchor[9]  = colorGradX.r; anchor[10] = colorGradX.g; anchor[11] = colorGradX.b;
  anchor[12] = colorGradY.r; anchor[13] = colorGradY.g; anchor[14] = colorGradY.b;
  anchor[15] = valueAvg;
}
