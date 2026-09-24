// source.particles.flow_swarm — shared helpers (hashes, color, mask shapes, particle
// layout, the VS→FS varyings). Both update.hlsl and the vs/fs raster pair
// include this so the GPU-resident particle layout stays consistent.

#ifndef FLOW_SWARM_COMMON_HLSL
#define FLOW_SWARM_COMMON_HLSL

// ===========================================================
// PCG bit-mix integer hash (same construction as flash_particles).
// ===========================================================
uint fsw_hash(uint x) {
  x = x * 747796405u + 2891336453u;
  uint word = ((x >> ((x >> 28u) + 4u)) ^ x) * 277803737u;
  return (word >> 22u) ^ word;
}
uint fsw_hash2(uint a, uint b) { return fsw_hash(a + fsw_hash(b)); }
uint fsw_hash3(uint a, uint b, uint c) {
  return fsw_hash(a + fsw_hash(b + fsw_hash(c)));
}
float fsw_unit(uint h)   { return float(h) * (1.0 / 4294967296.0); }   // [0,1)
float fsw_signed(uint h) { return fsw_unit(h) * 2.0 - 1.0; }            // [-1,1)

// ===========================================================
// Captured color + depth packing: RGB in the low 3 bytes, the per-particle
// "depth" (∈[0,1], drives undertow) in the top byte → one float slot
// (asfloat of the packed uint). We only ever reinterpret the bits (never do
// float math on the slot), so a NaN/inf bit pattern is harmless.
// ===========================================================
uint fsw_pack_rgbd(float3 c, float d) {
  uint r  = (uint)(saturate(c.r) * 255.0 + 0.5);
  uint g  = (uint)(saturate(c.g) * 255.0 + 0.5);
  uint b  = (uint)(saturate(c.b) * 255.0 + 0.5);
  uint dd = (uint)(saturate(d)   * 255.0 + 0.5);
  return r | (g << 8u) | (b << 16u) | (dd << 24u);
}
float3 fsw_unpack_rgb(uint p) {
  return float3(float(p & 0xFFu), float((p >> 8u) & 0xFFu),
                float((p >> 16u) & 0xFFu)) * (1.0 / 255.0);
}
float fsw_unpack_depth(uint p) { return float((p >> 24u) & 0xFFu) * (1.0 / 255.0); }

// ===========================================================
// Undertow membership ∈ [0,1] for a particle of the given depth at the given
// split. Soft cutoff: low-depth particles join first. Endpoints are exact —
// split=0 → 0 for every depth (no undertow), split=1 → 1 for every depth (all
// particles undertow). Shared by the update (motion) and vs (tint) passes so
// they agree per particle.
// ===========================================================
static const float FSW_UNDERTOW_SOFT = 0.15;
float fsw_undertow(float depth, float split) {
  float t = split * (1.0 + 2.0 * FSW_UNDERTOW_SOFT) - FSW_UNDERTOW_SOFT;
  return smoothstep(depth - FSW_UNDERTOW_SOFT, depth + FSW_UNDERTOW_SOFT, t);
}

// ===========================================================
// HSV → RGB (for the optional flow-direction tint).
// ===========================================================
float3 fsw_hsv_to_rgb(float3 hsv) {
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

// ===========================================================
// Mask shapes (particle-local corner ∈ [-1,1]²).
//   0 point   — solid (the quad is sized to ~1px in the VS).
//   1 gaussian
//   2 circle/squircle
//   3 solid
// ===========================================================
float fsw_mask(float2 n, uint kind, float param) {
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
  return 1.0;   // point (0) and solid (3) fill the quad
}

// ===========================================================
// Particle layout — 2 vec4 = 32 bytes. 1M particles = 32 MB.
//   a.xy = pos (screen uv), a.z = life_remain (s), a.w = life_total (s)
//   b.xy = velocity (uv/s, carries momentum), b.z = size (isotropic uv),
//   b.w  = asfloat(packed rgba8 captured color)
// ===========================================================
struct Particle { float4 a; float4 b; };

// VS → FS varyings (shared so the two stages agree exactly).
struct VsOut {
  float4 pos     : SV_Position;
  float2 corner  : TEXCOORD0;                       // quad-local [-1,1]²
  nointerpolation float4 col_life : TEXCOORD1;      // rgb = captured color, w = life_norm
  nointerpolation float4 vel      : TEXCOORD2;      // xy = velocity, z = speed, w = undertow membership
};

#endif // FLOW_SWARM_COMMON_HLSL
