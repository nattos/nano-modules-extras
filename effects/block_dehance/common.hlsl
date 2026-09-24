// filter.glitch.block_dehance — shared helpers + the GPU-resident rect layout. Both
// update.hlsl and render.hlsl include this so the rect-pool struct stays
// consistent across passes.

#ifndef BLOCK_DEHANCE_COMMON_HLSL
#define BLOCK_DEHANCE_COMMON_HLSL

// PCG bit-mix integer hash (same construction used across the bundle).
uint bd_pcg(uint x) {
  x = x * 747796405u + 2891336453u;
  uint word = ((x >> ((x >> 28u) + 4u)) ^ x) * 277803737u;
  return (word >> 22u) ^ word;
}
uint bd_pcg2(uint a, uint b)         { return bd_pcg(a + bd_pcg(b)); }
uint bd_pcg3(uint a, uint b, uint c) { return bd_pcg(a + bd_pcg(b + bd_pcg(c))); }
float bd_unit(uint h)   { return float(h) * (1.0 / 4294967296.0); }   // [0,1)
float bd_signed(uint h) { return bd_unit(h) * 2.0 - 1.0; }            // [-1,1)

float bd_luma(float3 c) { return dot(c, float3(0.299, 0.587, 0.114)); }

// Dehance modes.
static const uint MODE_BLACK  = 0u;
static const uint MODE_MOSAIC = 1u;
static const uint MODE_NOISE  = 2u;

// Rect pool entry. 3 vec4s = 48 bytes; both passes see the same stride.
struct Rect {
  // .xy = uv-space center, .zw = uv-space full size (width, height).
  float4 pos_size;
  // x = life_remain (sec; >0 → visible), y = life_total,
  // z = respawn_remain (sec; decays after life hits 0), w = mode (as float).
  float4 state;
  // x = mosaic cell size (captured, uv), y = mode_seed (asfloat uint),
  // z = flicker_seed (asfloat uint), w = reserved.
  float4 params;
};

#endif // BLOCK_DEHANCE_COMMON_HLSL
