// source.particles.sweep_chamber — shared helpers + GPU-resident layouts.
//
// Successor to source.legacy.double_chamber, modeled on flow_swarm's
// structure. The sim couples three layers through ONE coarse field texture
// pair (built per frame from the swept input): a particle pool, streamline
// tracers ("lines"), and a global calm↔intense response derived from the
// swept image itself. Every consumer of the field takes bilinear taps at
// FIELD_RES — no full-res convolutions anywhere in the sim.
//
// Spaces (flow_swarm conventions):
//   pos      — screen uv. vel — uv/s.
//   iso      — screen-isotropic vector space (1 unit = min(W,H) px). Directions
//              and gradients live here so circles stay round on any aspect;
//              convert to uv displacement by multiplying by aspect=(min/W,min/H).
//   s-space  — centred iso position: s = (uv - 0.5) / aspect.

#ifndef SWEEP_CHAMBER_COMMON_HLSL
#define SWEEP_CHAMBER_COMMON_HLSL

// ---- PCG bit-mix integer hash ----
uint swc_hash(uint x) {
  x = x * 747796405u + 2891336453u;
  uint word = ((x >> ((x >> 28u) + 4u)) ^ x) * 277803737u;
  return (word >> 22u) ^ word;
}
uint swc_hash2(uint a, uint b) { return swc_hash(a + swc_hash(b)); }
uint swc_hash3(uint a, uint b, uint c) { return swc_hash(a + swc_hash(b + swc_hash(c))); }
float swc_unit(uint h)   { return float(h) * (1.0 / 4294967296.0); }   // [0,1)
float swc_signed(uint h) { return swc_unit(h) * 2.0 - 1.0; }            // [-1,1)

// ---- captured color + per-particle "z phase" packed into one float slot ----
// (z drives the per-particle image-curl factor; only ever bit-reinterpreted.)
uint swc_pack_rgbz(float3 c, float z) {
  uint r  = (uint)(saturate(c.r) * 255.0 + 0.5);
  uint g  = (uint)(saturate(c.g) * 255.0 + 0.5);
  uint b  = (uint)(saturate(c.b) * 255.0 + 0.5);
  uint zz = (uint)(saturate(z)   * 255.0 + 0.5);
  return r | (g << 8u) | (b << 16u) | (zz << 24u);
}
float3 swc_unpack_rgb(uint p) {
  return float3(float(p & 0xFFu), float((p >> 8u) & 0xFFu),
                float((p >> 16u) & 0xFFu)) * (1.0 / 255.0);
}
float swc_unpack_z(uint p) { return float((p >> 24u) & 0xFFu) * (1.0 / 255.0); }

// ---- luma metric (double_chamber parity) + 90° perp ----
float swc_lum(float3 c) { return max(c.r, max(c.g, c.b)); }
float2 swc_perp(float2 v) { return float2(v.y, -v.x); }

// ---- along-ridge undertow (shared by p_update / trace / field_debug) ----
// The undertow (to_image_curl term) is a soft-NORMALIZED level-curve tangent
// gated by RIDGE PRESENCE (field_a's L'max), not by |∇L'|: the gradient is
// exactly zero on a ridge crest — which is where the captured population
// sits — so a |∇L'|-scaled undertow (dc parity) dies there and the whole
// swarm freezes into a static stipple. Constant-speed transport keeps the
// captured particles/seeds streaming along the lines.
static const float SWC_UNDERTOW_VEL = 1.5;   // iso uv/s at to_image_curl=1, cf=1
// DIRECTION comes from the PLAIN luma gradient (g_luma, field_c.rg), and
// MAGNITUDE from swept presence (g_swept = grad L' + the ridge detector) —
// split on purpose. grad L' = W'(luma)·grad luma collapses to zero at the
// sweep window's flat top even after fixing W's sign flip, which put a
// travelling stop-start null through every captured region. The plain-luma
// tangent doesn't know where the band is, so as the sweep passes a region
// its motion is ONE smooth hump: up to max, back down — never through zero.
float2 swc_undertow(float2 g_swept, float2 g_luma, float ridge) {
  // Soft normalization: weak/noise-floor luma gradients get proportionally
  // weak undertow (a near-unit normalization turns texel wiggle in flat
  // regions into full-speed pseudo-random directions — grid-quantized flow).
  float2 tang = swc_perp(g_luma) / (length(g_luma) + 0.2);
  // Two sweep-gates, take the stronger: ridge presence covers the band
  // itself (including the window plateau where the swept gradient is zero);
  // |grad L'| covers the wide multi-scale basin AROUND the band, so the
  // contour-following swirl is felt at a distance. Wide smoothsteps — hard
  // thresholds imprint the field lattice.
  float gate = max(smoothstep(0.05, 0.5, ridge),
                   smoothstep(0.15, 0.6, length(g_swept)));
  return tang * (SWC_UNDERTOW_VEL * gate);
}

// ---- C1-smooth cubic B-spline texture sample (4 bilinear taps) ----
// Bilinear field sampling is only C0 — its direction kinks at every texel
// boundary, and slow particles trace those kinks out as visibly quantized
// little stop/turn steps. The B-spline approximates rather than
// interpolates (slight blur) — fine for a force field.
float4 swc_sample_bspline(Texture2D<float4> tex, SamplerState sst,
                          float2 uv, float res) {
  float2 x  = uv * res - 0.5;
  float2 ip = floor(x);
  float2 f  = x - ip;
  float2 f2 = f * f;
  float2 f3 = f2 * f;
  float2 w0 = (1.0 - 3.0 * f + 3.0 * f2 - f3) * (1.0 / 6.0);
  float2 w1 = (4.0 - 6.0 * f2 + 3.0 * f3)     * (1.0 / 6.0);
  float2 w2 = (1.0 + 3.0 * (f + f2 - f3))     * (1.0 / 6.0);
  float2 w3 = f3 * (1.0 / 6.0);
  float2 g0 = w0 + w1;
  float2 g1 = w2 + w3;
  float2 uv0 = (ip - 0.5 + w1 / g0) / res;
  float2 uv1 = (ip + 1.5 + w3 / g1) / res;
  return (tex.SampleLevel(sst, float2(uv0.x, uv0.y), 0) * g0.x
        + tex.SampleLevel(sst, float2(uv1.x, uv0.y), 0) * g1.x) * g0.y
       + (tex.SampleLevel(sst, float2(uv0.x, uv1.y), 0) * g0.x
        + tex.SampleLevel(sst, float2(uv1.x, uv1.y), 0) * g1.x) * g1.y;
}

// ---- HSV → RGB (flow-direction tint) ----
float3 swc_hsv_to_rgb(float3 hsv) {
  float h = frac(hsv.x);
  float s = saturate(hsv.y);
  float v = saturate(hsv.z);
  float h6 = h * 6.0;
  float c = v * s;
  float x = c * (1.0 - abs(fmod(h6, 2.0) - 1.0));
  float m = v - c;
  float3 rgb;
  if      (h6 < 1.0) rgb = float3(c, x, 0);
  else if (h6 < 2.0) rgb = float3(x, c, 0);
  else if (h6 < 3.0) rgb = float3(0, c, x);
  else if (h6 < 4.0) rgb = float3(0, x, c);
  else if (h6 < 5.0) rgb = float3(x, 0, c);
  else               rgb = float3(c, 0, x);
  return rgb + m;
}

// ---- point shape mask (particle-local corner ∈ [-1,1]²) ----
//   0 point · 1 gaussian · 2 circle/squircle · 3 solid
float swc_mask(float2 n, uint kind, float param) {
  if (kind == 1u) {
    float sigma = lerp(0.25, 0.85, saturate(param));
    float r2 = dot(n, n);
    float g = exp(-r2 / (sigma * sigma));
    float window = smoothstep(1.0, 0.85, sqrt(r2));
    return g * window;
  }
  if (kind == 2u) {
    float k = lerp(2.0, 8.0, saturate(param));
    float v = pow(abs(n.x), k) + pow(abs(n.y), k);
    float r = pow(max(v, 1e-8), 1.0 / k);
    return smoothstep(1.0, 0.92, r);
  }
  return 1.0;
}

// ===========================================================
// Sweep window — the built-in luma band-pass. Smooth trapezoid, C1
// everywhere. The center slider ∈ [0,1] is remapped so 0 and 1 always mean
// "window fully off either end of the luma range" regardless of width and
// softness: at both extremes the swept luma L' ≡ 0 everywhere → the whole
// image reads as black and the sim free-flows on the noise field alone.
// (This is the fix for the old effect's pure-white/pure-black endpoint
// pathologies — no external pre-processing sweep required.)
// ===========================================================
float swc_sweep(float luma, float center01, float width, float softness) {
  float hw   = width * 0.5;
  float soft = max(lerp(0.01, 0.5, saturate(softness)), 1e-3);
  float lo   = -(hw + soft) - 0.001;   // center here → window ≡ 0 below black
  float hi   = 1.0 + (hw + soft) + 0.001;   // → window ≡ 0 above white
  float c    = lerp(lo, hi, saturate(center01));
  return smoothstep(c - hw - soft, c - hw, luma)
       * (1.0 - smoothstep(c + hw, c + hw + soft, luma));
}

// Signed "side of the band": +1 below the sweep window's centre, -1 above,
// smooth ramp across the band (same centre remap as swc_sweep). The swept
// gradient is grad L' = W'(luma)·grad luma and W' changes sign between the
// window's rising and falling edges — so the curl direction REVERSED from
// one side of the band to the other, with a zero travelling along the band
// centre ("stops, then switches direction" as the sweep moves). Multiplying
// the gradient by this side sign cancels W's sign flip for the TANGENT
// (curl follows the plain luma contours in one consistent orientation);
// the attraction keeps the raw swept gradient — its reversal IS the
// trapping well.
float swc_sweep_side(float luma, float center01, float width, float softness) {
  float hw   = width * 0.5;
  float soft = max(lerp(0.01, 0.5, saturate(softness)), 1e-3);
  float lo   = -(hw + soft) - 0.001;
  float hi   = 1.0 + (hw + soft) + 0.001;
  float c    = lerp(lo, hi, saturate(center01));
  return 1.0 - 2.0 * smoothstep(c - hw - soft, c + hw + soft, luma);
}

// ===========================================================
// 2D gradient (Perlin) noise, quintic fade, ANALYTIC derivative — C2 in
// space so the velocity derived from it is C1: no seams, no kinks (the old
// dc_field's sign()/abs()/pow() discontinuities are exactly what this
// replaces). Corner gradients rotate over time at per-corner rates
// (Perlin–Neyret flow noise) so eddies churn smoothly.
// Returns ∇ψ of the scalar noise ψ at p.
// ===========================================================
float2 swc_gnoise_grad(float2 p, float t, uint oseed) {
  float2 fl = floor(p);
  int2   ii = int2(fl);
  float2 f  = p - fl;
  float2 g[4];
  [unroll] for (int k = 0; k < 4; k++) {
    int2 c = int2(k & 1, k >> 1);
    uint hh = swc_hash3(uint(ii.x + c.x), uint(ii.y + c.y), oseed);
    float a0   = swc_unit(hh) * 6.2831853;
    float spin = 0.5 + swc_unit(swc_hash(hh ^ 0x9E3779B1u));   // per-eddy churn rate
    float a    = a0 + t * spin;                                 // t pre-scaled by caller
    g[k] = float2(cos(a), sin(a));
  }
  float n00 = dot(g[0], f);
  float n10 = dot(g[1], f - float2(1, 0));
  float n01 = dot(g[2], f - float2(0, 1));
  float n11 = dot(g[3], f - float2(1, 1));
  float2 u  = f * f * f * (f * (f * 6.0 - 15.0) + 10.0);        // quintic (C2)
  float2 du = 30.0 * f * f * (f * (f - 2.0) + 1.0);
  float k1 = n10 - n00, k2 = n01 - n00, k4 = n00 - n10 - n01 + n11;
  float2 gi = g[0] + u.x * (g[1] - g[0]) + u.y * (g[2] - g[0])
            + u.x * u.y * (g[0] - g[1] - g[2] + g[3]);
  return gi + du * float2(k1 + k4 * u.y, k2 + k4 * u.x);        // analytic ∇ψ
}

// ===========================================================
// Field texture channel contract (both FIELD_RES², RGBA16F):
//   field_a: .r = mean swept luma L' over the cell   (smooth scalar, gradient src)
//            .gb = intra-cell offset to the luma peak (texel units, |off| ≤ 0.5;
//                  a sharpened L'^4-weighted centroid — continuous, unlike argmax)
//            .a = MAX swept luma over the cell        (ridge detector: stays high
//                  on a crest where the gradient vanishes)
//   field_b: .rg = curl-noise background velocity    (uv/s)
//            .ba = MULTI-SCALE image gradient G = ∇L'·GAIN·edgeFade, ISO
//                  space, 3 stencil scales (wide attraction basin)
//                  (to_image / to_image_curl are composed per-consumer so the
//                  per-particle z-phase curl factor survives a single tap)
// A consumer composes, given its curl factor cf and ridge = field_a.a there:
//   vel_uv = fb.rg + (fb.ba·to_image
//                     + swc_undertow(fb.ba, ridge)·to_image_curl·cf) · aspect
// ===========================================================

// ---- particle layout — 2 vec4 = 32 bytes ----
//   a.xy = pos (screen uv), a.z = life_remain (s), a.w = life_total (s)
//   b.xy = velocity (uv/s), b.z = size (isotropic uv), b.w = asfloat(packed rgbz)
struct Particle { float4 a; float4 b; };

// ---- tracers ("lines"): field streamline tracers with grip + ballistics ----
//   a.xy = seed pos (uv), a.z = life/time, a.w = seed angle
//   b.xy = seed velocity (uv/s — ballistic momentum, inherits the fling)
//   b.z  = grip ∈ [0,1] (EMA of trace-mean L'max: how hard the image holds it)
//   b.w  = curvature κ (signed, rad per iso-unit of path; rolled at reseed —
//          the free-space "ballistic arc" bend)
struct TracerState { float4 a; float4 b; };
//   Seg.a = (p0.xy, p1.xy) in uv;  Seg.b = (rgb, alpha·weight)
struct Seg { float4 a; float4 b; };

// ---- VS → FS varyings ----
struct VsOut {
  float4 pos     : SV_Position;
  float2 corner  : TEXCOORD0;                       // quad-local [-1,1]²
  nointerpolation float4 col_life : TEXCOORD1;      // rgb = captured color, w = life_norm
  nointerpolation float4 vel      : TEXCOORD2;      // xy = velocity, z = speed, w = unused
};

struct LineVsOut {
  float4 pos    : SV_Position;
  float2 local  : TEXCOORD0;   // .x along [0,1], .y across [-1,1]
  nointerpolation float4 col : TEXCOORD1;
};

// ---- motion-vector output (render_outputs/motion) varyings ----
// Per-pixel screen-space velocity (uv/frame) for downstream motion blur.
// Points carry their integrated velocity; lines carry their tangent.
struct MotionVsOut {
  float4 pos    : SV_Position;
  float2 corner : TEXCOORD0;                  // point-local corner ∈ [-1,1]²
  nointerpolation float2 motion : TEXCOORD1;  // uv/frame
};
struct LineMotionVsOut {
  float4 pos   : SV_Position;
  float2 local : TEXCOORD0;                   // .x along [0,1], .y across [-1,1]
  nointerpolation float2 motion : TEXCOORD1;  // uv/frame, along the segment
};

#endif // SWEEP_CHAMBER_COMMON_HLSL
