// triangulate — shared struct + helpers, #included by every stage.
#ifndef TRIANGULATE_COMMON_HLSL
#define TRIANGULATE_COMMON_HLSL

// One persistent seed point. pos in [0,1] uv; score caches the incumbent's
// last match score; flags reserved (age / active).
struct Seed {
  float2 pos;
  float  score;
  float  flags;
};

// --- hashing (pcg-style) ---------------------------------------------------
uint tri_hash_u(uint x) {
  x ^= x >> 16; x *= 0x7feb352du;
  x ^= x >> 15; x *= 0x846ca68bu;
  x ^= x >> 16; return x;
}
float tri_hash_f(uint x) { return (tri_hash_u(x) & 0x00FFFFFFu) / 16777216.0; }
float2 tri_hash2(uint a, uint b) {
  return float2(tri_hash_f(a * 747796405u + b),
                tri_hash_f(b * 2891336453u + a + 0x9E3779B9u));
}

// --- candidate packing -----------------------------------------------------
// Pack (quantized importance, pixel x, pixel y) into one uint so an
// InterlockedMax over a Voronoi cell yields the argmax-importance pixel.
// Requires proc dims < 1024 on each axis (enforced host-side).
uint tri_pack_cand(float w01, uint px, uint py) {
  uint wq = (uint)(saturate(w01) * 4095.0);   // 12 bits
  return (wq << 20) | ((py & 1023u) << 10) | (px & 1023u);
}
float tri_cand_w(uint p)  { return (float)(p >> 20) / 4095.0; }
uint  tri_cand_px(uint p) { return p & 1023u; }
uint  tri_cand_py(uint p) { return (p >> 10) & 1023u; }

// Seed-pool ceiling (must match MAX_SEEDS in main.cpp) — used as the edge
// dedup key stride (key = lo*TRI_MAX_SEEDS + hi, one bit per unordered pair).
static const uint TRI_MAX_SEEDS = 4096u;

// Fixed-point scale for the atomic mass / weighted-centroid sums.
static const float TRI_FX = 1024.0;

// Quantize/dequantize an importance weight for InterlockedMax adjacency.
uint  tri_qw(float w)  { return (uint)(saturate(w) * 65535.0); }
float tri_dqw(uint q)  { return (float)q / 65535.0; }

// Accumulator layout: 4 uints per seed.
//   [0] massFx   = Σ W·TRI_FX         (importance-weighted mass)
//   [1] wxFx     = Σ W·x·TRI_FX       (weighted-centroid numerator)
//   [2] wyFx     = Σ W·y·TRI_FX
//   [3] candPack = InterlockedMax of tri_pack_cand(...)  (argmax-importance pixel)
static const uint TRI_ACCUM_STRIDE = 4u;

#endif
