// filter.blur.smear (Scatter mode) — Pixulant's dissolve, steered directionally.
//
// Like warp.legacy.pixulant this scatters the SAMPLING uv per pixel with a salted
// random offset, samples a light copy (rhs) and a heavier copy (lhs), and blooms
// the abs-difference (the "dive": mix(rhs, abs(lhs-rhs), dive) then an exposure
// gain lifts the edge halo into coloured grain). The one change vs Pixulant: the
// random offset is drawn from the SAME tilted directional footprint the Blur mode
// uses — major reach with the tail bias, minor reach with the perspective tilt —
// instead of an isotropic disc. So the grain streaks along the axis.
//
// Salt animates via `salt_base` (a host tick accumulator + on-change hash). A tiny
// floor keeps the heavy copy displaced more than the light one even at reach 0, so
// the resting halo survives while dived (Pixulant's load-bearing quirk).

Texture2D<float4>   inputTex      : register(t0);
SamplerState        linearSampler : register(s1);
RWTexture2D<float4> outputTex     : register(u2);

cbuffer Uniforms : register(b3) {
  float axis_maj_x, axis_maj_y;   // aspect-scaled UV step per unit major reach
  float axis_min_x, axis_min_y;   // aspect-scaled UV step per unit minor reach
  float reach_fwd, reach_back;    // asymmetric major reach (short-axis fractions)
  float width;                    // minor reach
  float salt_base;                // animated salt (Saw + on-change Random)
  float major_x, major_y;         // screen-unit major dir (perspective proj)
  float tilt;                     // perspective gradient amount
  float dive;                     // mix(rhs, abs(lhs-rhs), dive)
  float exposure_gain;            // grain-contrast lift on the differenced image
  float edge_artifacts;           // bottom-edge grain flair (0 = clean)
  float exposure;                 // global output gain (shared with Blur mode)
  float softness;                 // uniform (boxy) → gaussian sampling blend
};

static const float LIGHT_STR = 0.35;  // rhs (light) copy displacement fraction
static const float HEAVY_STR = 1.0;   // lhs (heavy) copy displacement fraction
static const float FLOOR     = 0.004;  // resting-halo floor (reach units) × dive

// Pixulant's salted value-hash of the destination uv.
float s_rand(float2 uv, float salt) {
  return frac(sin(dot(uv, float2(12.9898, 78.233 + salt * 1.56783)))
              * (43758.5453 + salt * 86183.526));
}

// One scatter: random displacement of the sampling uv within the directional
// footprint at `strength`.
float2 foot_scatter(float2 uv, float salt, float strength) {
  float a = s_rand(uv, 0.0 + salt);
  float b = s_rand(uv, 0.1 + salt);

  float proj = clamp(dot(uv - 0.5, float2(major_x, major_y)) * 2.0, -1.0, 1.0);
  float persp = clamp(1.0 - tilt * proj, 0.05, 3.0);

  float floor_r = FLOOR * dive;
  float back = max(reach_back, floor_r);
  float fwd  = max(reach_fwd,  floor_r);
  float wid  = max(width,      floor_r) * persp;

  // Signed unit samples: a uniform draw (boxy — hard boundaries at both ends of
  // the span, which read as the two edges at high tail) blended toward a
  // Box–Muller gaussian (peaked, no hard boundary) by `softness`. The major side
  // is scaled asymmetrically (fwd vs back) so the tail keeps its one-sided shape.
  float uM = 2.0 * a - 1.0;
  float uN = 2.0 * b - 1.0;
  float rad = sqrt(-2.0 * log(max(a, 1e-6)));
  float ang = 6.28318530718 * b;
  float gM = clamp(rad * cos(ang) * 0.5, -1.3, 1.3);
  float gN = clamp(rad * sin(ang) * 0.5, -1.3, 1.3);
  float rM = lerp(uM, gM, softness);
  float rN = lerp(uN, gN, softness);

  float m = (rM >= 0.0 ? rM * fwd : rM * back) * strength;  // asymmetric major
  float n = rN * wid * strength;                            // symmetric minor
  float2 duv = m * float2(axis_maj_x, axis_maj_y) + n * float2(axis_min_x, axis_min_y);
  return uv + duv;
}

// Soft in-frame weight for a SAMPLING uv: 1 when the tap lands comfortably inside
// [0,1]^2, ramping to 0 as it overshoots an edge by FRAME_MARGIN. Off-frame taps
// return a CLAMPED edge/corner texel (ClampToEdge), and a corner is the attractor
// for a whole off-screen quadrant — so a lone bright corner pixel gets outsized
// weight and its abs-difference blooms into flashing grain. Fade the dive by this
// so off-frame taps contribute no manufactured difference (Pixulant's fix).
static const float FRAME_MARGIN = 0.03;
float in_frame(float2 uv) {
  float2 d = min(uv, 1.0 - uv);                 // signed dist to nearest edge (>0 inside)
  float2 m = saturate(d / FRAME_MARGIN + 1.0);  // 1 at/inside the edge, 0 by MARGIN outside
  return m.x * m.y;
}

float4 sample_in(float2 uv) {
  float4 c = inputTex.SampleLevel(linearSampler, uv, 0.0);
  // Reproduce Pixulant's bottom-edge bug as opaque-white flair when asked (row 0
  // is the top, so the screen bottom is uv.y > 1). edge_artifacts is the raw [0,1]
  // knob; 0.1 is the tuned sensitivity (the flair was overpowering at 1:1).
  if (edge_artifacts > 0.0 && uv.y > 1.0)
    c = lerp(c, float4(1.0, 1.0, 1.0, 1.0), edge_artifacts * 0.1);
  return c;
}

[numthreads(8, 8, 1)]
void main(uint3 gid : SV_DispatchThreadID) {
  uint W, H;
  outputTex.GetDimensions(W, H);
  if (gid.x >= W || gid.y >= H) return;
  float2 uv = (float2(gid.xy) + 0.5) / float2(W, H);

  float2 uvR = foot_scatter(uv, salt_base,       LIGHT_STR);
  float2 uvL = foot_scatter(uv, salt_base + 0.7, HEAVY_STR);
  float4 rhs = sample_in(uvR);
  float4 lhs = sample_in(uvL);

  // The dive is trustworthy only where BOTH taps stay on-frame; off-frame the pixel
  // falls back to the light copy (rhs) instead of a boosted edge/corner halo.
  // edge_artifacts (opt-in) restores full off-frame weight = the faithful bug.
  float valid = in_frame(uvL) * in_frame(uvR);
  valid = lerp(valid, 1.0, saturate(edge_artifacts));

  float3 diff = abs(lhs.rgb - rhs.rgb);
  float3 rgb  = lerp(rhs.rgb, diff, dive * valid) * exposure_gain * exposure;
  float  a    = lerp(lhs.a, rhs.a, dive * 0.5);

  outputTex[gid.xy] = float4(saturate(rgb), a);
}
