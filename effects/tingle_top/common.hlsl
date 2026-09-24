// source.light.tingle_top — shared helpers + GPU-resident particle layout. Both
// update.hlsl and render.hlsl include this so the pool struct stays in sync.

#ifndef TINGLE_TOP_COMMON_HLSL
#define TINGLE_TOP_COMMON_HLSL

uint tt_pcg(uint x) {
  x = x * 747796405u + 2891336453u;
  uint word = ((x >> ((x >> 28u) + 4u)) ^ x) * 277803737u;
  return (word >> 22u) ^ word;
}
uint tt_pcg2(uint a, uint b)         { return tt_pcg(a + tt_pcg(b)); }
uint tt_pcg3(uint a, uint b, uint c) { return tt_pcg(a + tt_pcg(b + tt_pcg(c))); }
float tt_unit(uint h)   { return float(h) * (1.0 / 4294967296.0); }   // [0,1)
float tt_signed(uint h) { return tt_unit(h) * 2.0 - 1.0; }            // [-1,1)

float3 tt_hsv_to_rgb(float3 hsv) {
  float h = frac(hsv.x), s = saturate(hsv.y), v = saturate(hsv.z);
  float h6 = h * 6.0, c = v * s, x = c * (1.0 - abs(fmod(h6, 2.0) - 1.0)), m = v - c;
  float3 rgb;
  if      (h6 < 1.0) rgb = float3(c, x, 0);
  else if (h6 < 2.0) rgb = float3(x, c, 0);
  else if (h6 < 3.0) rgb = float3(0, c, x);
  else if (h6 < 4.0) rgb = float3(0, x, c);
  else if (h6 < 5.0) rgb = float3(x, 0, c);
  else               rgb = float3(c, 0, x);
  return rgb + m;
}

// Sparkle mask in particle-local coords n ∈ [-1,1]² (caller culls |n|>1).
//   kind 0 solid, 1 circle/squircle, 2 gaussian. param tunes shape.
float tt_mask(float2 n, uint kind, float param) {
  if (kind == 0u) return 1.0;
  if (kind == 1u) {
    float e = lerp(2.0, 8.0, saturate(param));
    float r = pow(max(pow(abs(n.x), e) + pow(abs(n.y), e), 1e-8), 1.0 / e);
    return smoothstep(1.0, 0.92, r);
  }
  float sigma = lerp(0.25, 0.85, saturate(param));
  float r2 = dot(n, n);
  return exp(-r2 / (sigma * sigma)) * smoothstep(1.0, 0.85, sqrt(r2));
}

// Particle. 3 vec4 = 48 bytes; both passes share the stride.
struct Particle {
  float4 a;   // x=px, y=py (uv), z=size (uv half-extent), w=life_remain (sec)
  float4 b;   // x=life_total, y=respawn_remain, z=hue_offset, w=vx (uv/sec)
  float4 c;   // x=vy (uv/sec), y=bar (target bar index, as float), z/w reserved
};

#endif // TINGLE_TOP_COMMON_HLSL
