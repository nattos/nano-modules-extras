// source.legacy.double_chamber — shared helpers + particle layout.
//
// A v2 of the shipped NanoGraph "DoubleChamber" (the used subset: the P
// field-particles, the "Big" attractors, curl steering, image coupling — the
// charged-collision "K accelerator" block and the PONK laser output are
// dropped). Both update passes and the vs/fs raster pair include this so the
// GPU-resident particle layout stays consistent.

#ifndef DOUBLE_CHAMBER_COMMON_HLSL
#define DOUBLE_CHAMBER_COMMON_HLSL

// ---- PCG bit-mix integer hash ----
uint dc_hash(uint x) {
  x = x * 747796405u + 2891336453u;
  uint word = ((x >> ((x >> 28u) + 4u)) ^ x) * 277803737u;
  return (word >> 22u) ^ word;
}
uint dc_hash2(uint a, uint b) { return dc_hash(a + dc_hash(b)); }
uint dc_hash3(uint a, uint b, uint c) { return dc_hash(a + dc_hash(b + dc_hash(c))); }
float dc_unit(uint h)   { return float(h) * (1.0 / 4294967296.0); }   // [0,1)
float dc_signed(uint h) { return dc_unit(h) * 2.0 - 1.0; }            // [-1,1)

// ---- captured color + per-particle "z phase" packed into one float slot ----
uint dc_pack_rgbz(float3 c, float z) {
  uint r  = (uint)(saturate(c.r) * 255.0 + 0.5);
  uint g  = (uint)(saturate(c.g) * 255.0 + 0.5);
  uint b  = (uint)(saturate(c.b) * 255.0 + 0.5);
  uint zz = (uint)(saturate(z)   * 255.0 + 0.5);
  return r | (g << 8u) | (b << 16u) | (zz << 24u);
}
float3 dc_unpack_rgb(uint p) {
  return float3(float(p & 0xFFu), float((p >> 8u) & 0xFFu),
                float((p >> 16u) & 0xFFu)) * (1.0 / 255.0);
}
float dc_unpack_z(uint p) { return float((p >> 24u) & 0xFFu) * (1.0 / 255.0); }

// ---- HSV → RGB (render hue) ----
float3 dc_hsv_to_rgb(float3 hsv) {
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

float3 dc_rgb_to_hsv(float3 rgb) {
  float cmax = max(max(rgb.r, rgb.g), rgb.b);
  float cmin = min(min(rgb.r, rgb.g), rgb.b);
  float d = cmax - cmin;
  float h = 0.0;
  if (d > 1e-5) {
    if      (cmax == rgb.r) h = fmod((rgb.g - rgb.b) / d, 6.0);
    else if (cmax == rgb.g) h = (rgb.b - rgb.r) / d + 2.0;
    else                    h = (rgb.r - rgb.g) / d + 4.0;
    h *= 1.0 / 6.0;
    if (h < 0.0) h += 1.0;
  }
  float s = (cmax > 1e-5) ? (d / cmax) : 0.0;
  return float3(h, s, cmax);
}

// ---- PetriDish polynomial vector field (the P "Field Expr", verbatim) ----
// Input x is the particle's centred position × field_scale.
float2 dc_field(float2 x, float skew, float squash) {
  float fx = x.y * x.y - 1.0 - (x.x * skew * -0.6 + skew * 0.1);
  float fy = (x.x + skew * 0.7) * sign(x.y)
             * pow(abs(x.y) + 0.7 + squash * 0.8, 3.0) * -0.1 * squash;
  return float2(fx, fy);
}
float2 dc_perp(float2 v) { return float2(v.y, -v.x); }

// ---- point shape mask (particle-local corner ∈ [-1,1]²) ----
//   0 point/solid · 1 gaussian · 2 circle
float dc_mask(float2 n, uint kind, float param) {
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

// ---- particle layout — 2 vec4 = 32 bytes (shared by P and Big) ----
//   a.xy = pos (screen uv), a.z = life_remain, a.w = life_total
//   b.xy = velocity (uv/s),  b.z = size (isotropic uv), b.w = asfloat(packed rgbz)
struct Particle { float4 a; float4 b; };

// VS → FS varyings (points).
struct VsOut {
  float4 pos    : SV_Position;
  float2 corner : TEXCOORD0;
  nointerpolation float4 col_life : TEXCOORD1;  // rgb = color, w = life_norm
  nointerpolation float4 extra    : TEXCOORD2;  // x = speed
};

// ---- tracers (L block): gradient/field-following line tracers ----
//   TracerState.a = (seedPos.xy in s-space, time, seedAngle)
//   Seg.a = (p0.xy, p1.xy) in uv;  Seg.b = (rgb, alpha)
struct TracerState { float4 a; };
struct Seg { float4 a; float4 b; };

// Line VS → FS varyings.
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
  float2 corner : TEXCOORD0;            // point-local corner ∈ [-1,1]²
  nointerpolation float2 motion : TEXCOORD1;  // uv/frame
};
struct LineMotionVsOut {
  float4 pos   : SV_Position;
  float2 local : TEXCOORD0;             // .x along [0,1], .y across [-1,1]
  nointerpolation float2 motion : TEXCOORD1;  // uv/frame, along the segment
};

// ---- bridgers: stochastic chords between two P particles ----
//   a.xy = endpoint A (uv), a.zw = endpoint B (uv)
//   b.x  = asfloat(target particle index A), b.y = asfloat(target index B)
//   b.z  = freshA flag (0 = uninitialised → snap, 1 = glide)
//   b.w  = freshB flag
// Each frame an endpoint re-targets to a random particle with prob bridger_rate,
// then glides toward its live target's position; rendered as a Seg via line_vs/fs.
struct BridgerState { float4 a; float4 b; };

#endif // DOUBLE_CHAMBER_COMMON_HLSL
