// warp.legacy.pixulant — scatter-cascade + Difference "dive" (Wire "Pixulant").
//
// Two ISF shaders fused into one pass:
//   - "Radial Stretch Sample" (rss_*): a per-pixel salted-random scatter of the
//     SAMPLING uv, aspect-corrected on y, with Resolume's load-bearing bottom-row
//     clamp (outputUV.y = max(.,1/H)).
//   - "Difference": mix(rhs, abs(lhs-rhs), Alpha*2).
//
// The patch cascades three Radial Stretch passes (light→mid→heavy) then takes
// the Difference of the heavy copy (lhs) against the light copy (rhs). Because
// each pass is a pure resample (outUV = uv + dir·strength; sample), the cascade
// composes: we chain the three displacements and take ONE sample of the input at
// the composed coordinate per side (rhs needs one pass; lhs needs three). This
// is the v2 re-architecture — shimmer-free, no intermediate render targets —
// versus the original's three full-frame passes (whose intermediate bilinear
// filtering differs only sub-pixel from this).
//
// The "strange colours out of nowhere": abs(lhs-rhs) of two DIFFERENTLY-scattered
// copies is a per-channel edge halo (zero in flat regions, coloured at edges);
// even at Scatter=0 a tiny scatter floor makes the heavy copy displace more than
// the light one, so edges still differ — and the Exposure gain lifts that halo
// into vivid grain. Faithful to the patch, not a Wire rounding accident.

Texture2D<float4>   inputTex      : register(t0);
SamplerState        linearSampler : register(s1);
RWTexture2D<float4> outputTex     : register(u2);

cbuffer Uniforms : register(b3) {
  float str_rhs;        // light-pass scatter strength (RSS191)
  float str_mid;        // middle-pass scatter strength (RSS180)
  float str_lhs;        // heavy-pass scatter strength (RSS185)
  float diff_t;         // Difference mix factor = Alpha*2 = Dive

  float salt_rhs;       // RSS191 salt (= Saw+Random)
  float salt_mid;       // RSS180 salt (+0.4)
  float salt_lhs;       // RSS185 salt (+0.7)
  float exposure_gain;  // photographic gain on the differenced image

  float aspect;         // W/H, for scatterDir.y *= aspect (round scatter on screen)
  float inv_h;          // 1/H, Resolume's bottom-row clamp
  float edge_artifacts; // 0 = clamp (fixed); >0 = reproduce the bottom-edge bug
  float _pad1;
};

// ISF "Radial Stretch Sample" rand: a salted value-hash of the destination uv.
float rss_rand(float2 uv, float salt) {
  return frac(sin(dot(uv, float2(12.9898, 78.233 + salt * 1.56783)))
              * (43758.5453 + salt * 86183.526));
}

// One scatter pass: random per-pixel displacement of the SAMPLING uv.
float2 rss_scatter(float2 uv, float salt, float strength) {
  float2 dir = (float2(rss_rand(uv, 0.0 + salt), rss_rand(uv, 0.1 + salt)) - 0.5) * 2.0;
  dir.y *= aspect;
  float2 outUV = uv + dir * strength;
  // The ISF's workaround for Resolume's bottom-edge bug. With edge_artifacts on
  // we DON'T clamp — we let the sample run past the bottom so the bug can show.
  if (edge_artifacts <= 0.0) outUV.y = max(outUV.y, inv_h);
  return outUV;
}

// Soft in-frame weight for a SAMPLING uv: 1 when the tap lands comfortably inside
// [0,1]^2, ramping to 0 as it overshoots an edge by FRAME_MARGIN. Off-frame taps
// return a CLAMPED edge/corner texel (ClampToEdge), and a corner texel is the
// attractor for an entire off-screen quadrant — so a lone bright corner pixel gets
// outsized weight and its abs-difference blooms into flashing grain. We fade the
// dive by this weight so off-frame taps contribute no manufactured difference.
static const float FRAME_MARGIN = 0.03;
float in_frame(float2 uv) {
  float2 d = min(uv, 1.0 - uv);                 // signed dist to nearest edge (>0 inside)
  float2 m = saturate(d / FRAME_MARGIN + 1.0);  // 1 at/inside the edge, 0 by MARGIN outside
  return m.x * m.y;
}

// Sample the input, optionally reproducing Resolume's bottom-edge bug: sampling
// past the bottom (here uv.y > 1, since HLSL row 0 is the TOP — the screen
// bottom is large uv.y) returned a constant edge value that survives the
// abs-difference (doesn't cancel), so the Exposure boosts it → the bright grainy
// band along the bottom edge. The original returned TRANSPARENT there, which
// punched holes downstream — so we use OPAQUE WHITE instead: same bright flair,
// no transparency.
float4 sample_in(float2 uv) {
  float4 c = inputTex.SampleLevel(linearSampler, uv, 0.0);
  if (edge_artifacts > 0.0 && uv.y > 1.0)
    c = lerp(c, float4(1.0, 1.0, 1.0, 1.0), edge_artifacts);
  return c;
}

[numthreads(8, 8, 1)]
void main(uint3 gid : SV_DispatchThreadID) {
  uint W, H;
  outputTex.GetDimensions(W, H);
  if (gid.x >= W || gid.y >= H) return;
  float2 uv = (float2(gid.xy) + 0.5) / float2(W, H);

  // rhs = one light scatter of the input (RSS191 at this uv).
  float2 uvR = rss_scatter(uv, salt_rhs, str_rhs);
  float4 rhs = sample_in(uvR);

  // lhs = the three-deep cascade (RSS185 ∘ RSS180 ∘ RSS191). Each pass
  // re-randomizes from its own destination uv, so chain the displacements
  // and sample the input once at the composed coordinate.
  float2 uvB = rss_scatter(uv,  salt_lhs, str_lhs); // outer (3rd) pass
  float2 uvA = rss_scatter(uvB, salt_mid, str_mid); // middle (2nd) pass
  float2 uvT = rss_scatter(uvA, salt_rhs, str_rhs); // inner (1st) pass = RSS191
  float4 lhs = sample_in(uvT);

  // Off-frame taps clamp to a replicated edge/corner texel, whose abs-difference
  // is manufactured (and, near corners, hugely over-weighted) grain. The dive is
  // trustworthy only where BOTH taps stay on-frame — fade it out otherwise so the
  // pixel falls back to the light copy (rhs) instead of a boosted corner halo.
  // edge_artifacts (opt-in) restores full off-frame weight = the reproduced bug.
  float valid = in_frame(uvT) * in_frame(uvR);
  valid = lerp(valid, 1.0, saturate(edge_artifacts));

  // ISF "Difference": mix(rhs, abs(lhs-rhs), Alpha*2). diff_t = Alpha*2 = Dive.
  float3 diff = abs(lhs.rgb - rhs.rgb);
  float3 rgb  = lerp(rhs.rgb, diff, diff_t * valid);

  // Exposure node: photographic gain that lifts the (mostly dark) difference —
  // this is what makes the edge halo bloom into "strange colours".
  rgb *= exposure_gain;

  // Difference's alpha: mix(lhs.a, rhs.a, Alpha) where Alpha = diff_t*0.5.
  float a = lerp(lhs.a, rhs.a, diff_t * 0.5);
  outputTex[gid.xy] = float4(saturate(rgb), a);
}
