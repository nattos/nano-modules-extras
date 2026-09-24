// warp.dispersion — block-quantized UV-jitter sampler.
//
// Per pixel: figure out which block we're in (using the CPU-quantized
// block_size and start_offset that don't sliding-sweep), hash a stable
// random offset for that block, sample inputTex at (block_center +
// offset). Every pixel in the same block reads the same color.

// PCG bit-mix integer hash — same construction we use everywhere else
// in the bundle for stable per-cell randomness.
uint dsp_pcg(uint x) {
  x = x * 747796405u + 2891336453u;
  uint word = ((x >> ((x >> 28u) + 4u)) ^ x) * 277803737u;
  return (word >> 22u) ^ word;
}
uint dsp_hash3(uint a, uint b, uint c) {
  return dsp_pcg(a + dsp_pcg(b + dsp_pcg(c)));
}

Texture2D<float4>   inputTex  : register(t0);
SamplerState        linearSampler : register(s1);
RWTexture2D<float4> outputTex : register(u2);

cbuffer Uniforms : register(b3) {
  int   block_w;
  int   block_h;
  int   start_x;
  int   start_y;

  int   tick_index;
  float offset_max;
  float intensity;
  int   seed;
};

static const float TAU = 6.28318530717958647692;

[numthreads(8, 8, 1)]
void main(uint3 gid : SV_DispatchThreadID) {
  uint W, H;
  outputTex.GetDimensions(W, H);
  if (gid.x >= W || gid.y >= H) return;

  int px = int(gid.x);
  int py = int(gid.y);

  // Block index (handle negative-after-start_offset with explicit floor).
  int bw = max(block_w, 1);
  int bh = max(block_h, 1);
  int rel_x = px - start_x;
  int rel_y = py - start_y;
  // Floor division (rounds toward -inf), then block_ix is signed.
  int bix = (rel_x < 0) ? ((rel_x - bw + 1) / bw) : (rel_x / bw);
  int biy = (rel_y < 0) ? ((rel_y - bh + 1) / bh) : (rel_y / bh);

  // Block center in pixel coords (then convert to uv).
  int cx_px = bix * bw + start_x + bw / 2;
  int cy_px = biy * bh + start_y + bh / 2;
  float2 center_uv = float2(float(cx_px) / float(W),
                            float(cy_px) / float(H));

  // Stable random offset per (block, tick).
  uint h = dsp_hash3(uint(bix + 100000),
                     uint(biy + 100000),
                     uint(tick_index ^ seed));
  float angle = float(h & 0xFFFFu) * (1.0 / 65536.0) * TAU;
  float mag = float((h >> 16u) & 0xFFFFu) * (1.0 / 65536.0) * offset_max;
  float2 offset = float2(cos(angle), sin(angle)) * mag;

  float2 sample_uv = center_uv + offset;
  // Mirror wrap — implemented in shader (sampler clamp address mode
  // would also work; we mirror for nicer edge behavior).
  sample_uv = 1.0 - abs(1.0 - abs(fmod(sample_uv + 4.0, 2.0)));

  float4 dispersed = inputTex.SampleLevel(linearSampler, sample_uv, 0);
  float4 base = inputTex[gid.xy];
  outputTex[gid.xy] = lerp(base, dispersed, saturate(intensity));
}
