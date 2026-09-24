// source.particles.flash_particles — shared helpers (hashes, color, mask shapes,
// particle struct layout). Both update.hlsl and render.hlsl include
// this so the GPU-resident particle layout stays consistent across
// passes.

#ifndef FLASH_PARTICLES_COMMON_HLSL
#define FLASH_PARTICLES_COMMON_HLSL

// ===========================================================
// PCG bit-mix integer hash. Same construction as motion_field
// — bit-exact across coordinate ranges, no float artifacts.
// ===========================================================
uint pf_pcg_hash(uint x) {
  x = x * 747796405u + 2891336453u;
  uint word = ((x >> ((x >> 28u) + 4u)) ^ x) * 277803737u;
  return (word >> 22u) ^ word;
}
uint pf_pcg_hash2(uint a, uint b) { return pf_pcg_hash(a + pf_pcg_hash(b)); }
uint pf_pcg_hash3(uint a, uint b, uint c) {
  return pf_pcg_hash(a + pf_pcg_hash(b + pf_pcg_hash(c)));
}

// 32-bit hash → uniform [0, 1).
float pf_unit(uint h) { return float(h) * (1.0 / 4294967296.0); }
// 32-bit hash → uniform [-1, +1].
float pf_signed(uint h) { return pf_unit(h) * 2.0 - 1.0; }

// ===========================================================
// HSV / RGB
// ===========================================================
float3 pf_rgb_to_hsv(float3 rgb) {
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
  float v = cmax;
  return float3(h, s, v);
}

float3 pf_hsv_to_rgb(float3 hsv) {
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
// Mask shapes. Input `n` is the particle-local coord normalised
// to [-1, +1] across each half-axis of the quad. Output is a 0..1
// alpha mask. Callers cull with any(abs(n) > 1) before calling.
//
//   kind 0 : solid       — fills the entire quad.
//   kind 1 : circle/squ. — exponent morphs from 2 (round) toward
//                          higher (squircle/box). `param` ∈ [0, 1].
//   kind 2 : gaussian    — windowed; `param` ∈ [0, 1] tunes sigma
//                          (sharper at 0, softer at 1).
// ===========================================================
float pf_mask_solid() { return 1.0; }

float pf_mask_circle(float2 n, float param) {
  float exponent = lerp(2.0, 8.0, saturate(param));
  // |n.x|^k + |n.y|^k <= 1 defines a squircle of exponent k. Compute
  // the value at this point and smooth-step it across the boundary.
  float v = pow(abs(n.x), exponent) + pow(abs(n.y), exponent);
  // r = v^(1/exponent) — the distance metric for the L^k unit ball.
  float r = pow(max(v, 1e-8), 1.0 / exponent);
  return smoothstep(1.0, 0.92, r);
}

float pf_mask_gaussian(float2 n, float param) {
  // sigma controls falloff width. At param=0 sigma=0.25 (sharp),
  // at param=1 sigma=0.85 (soft).
  float sigma = lerp(0.25, 0.85, saturate(param));
  float r2 = dot(n, n);
  float g = exp(-r2 / (sigma * sigma));
  // Gentle window so the soft tail fades to zero before the rect bound,
  // avoiding a faint hard-edge ring at the quad boundary.
  float window = smoothstep(1.0, 0.85, sqrt(r2));
  return g * window;
}

float pf_mask(float2 n, uint kind, float param) {
  if (kind == 0u) return pf_mask_solid();
  if (kind == 1u) return pf_mask_circle(n, param);
  return pf_mask_gaussian(n, param);
}

// ===========================================================
// Particle layout. 5 vec4s = 80 bytes. Both update + render see
// the same stride. Field meanings are documented at the field
// definitions.
// ===========================================================
struct Particle {
  // .xy = uv-space center, .zw = uv-space full size (width, height).
  float4 pos_size;
  // RGBA captured from input tex at spawn (linear, no premultiplication).
  float4 captured;
  // x = rotation in radians (captured at spawn).
  // y = life_remain (sec). >0 means visible/alive.
  // z = life_total (captured at spawn).
  // w = respawn_remain (sec). Decays only after life hits 0; <=0 means
  //     time to spawn fresh on the next update.
  float4 state;
  // Per-particle color/alpha jitters, captured at spawn. Stored as
  // scaled offsets so the render shader applies them with no extra
  // params:
  //   x = hue offset      ∈ [-1, +1]   (added to HSV hue, fract'd)
  //   y = brightness mult ∈ [0,    2]  (multiplies HSV value)
  //   z = saturation mult ∈ [0,    2]  (multiplies HSV saturation)
  //   w = alpha mult      ∈ [0,    2]  (multiplies final alpha)
  float4 jitters;
  // x = respawn_total (captured at spawn).
  // y = per-particle frame-jitter seed (uint reinterpreted).
  // z, w = padding (reserved for future per-particle params).
  float4 meta;
};

#endif // FLASH_PARTICLES_COMMON_HLSL
