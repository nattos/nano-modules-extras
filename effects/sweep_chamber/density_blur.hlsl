// source.particles.sweep_chamber — density halo convolution (one axis).
//
// Turns the unit-mass scatter written by density_vs/fs into the crowding
// field the update pass reads, by convolving it with the SAME gaussian the
// old per-particle splat drew:
//
//   halo(d_iso) = exp(-(d_iso / (0.5·interaction_radius))²),  d_iso ≤ radius
//
// A gaussian is separable, so this runs twice (X then Y) with the per-axis σ
// in texels — σ_x = 0.5·radius·aspect_x·RES, σ_y likewise — which reproduces
// the round-in-screen-pixels halo on the square, stretched density buffer.
// Weights are deliberately UNNORMALISED: Σ_j halo(d_j) must stay a neighbour
// COUNT (so density_threshold / stream_density keep their meaning) rather
// than becoming a normalised average.
//
// Truncation matches the old quad's `discard` at d_iso > radius, i.e. 2σ.
// Out-of-bounds taps contribute nothing (no edge clamping): the old splat
// likewise lost the part of a halo that fell outside the buffer.

Texture2D<float4>   srcTex : register(t0);
RWTexture2D<float4> dstTex : register(u1);

cbuffer BlurUniforms : register(b2) {
  float res;        // density buffer resolution (texels per axis)
  float dir_x;      // 1,0 = horizontal pass · 0,1 = vertical pass
  float dir_y;
  float inv_sigma;  // 1 / σ, in texels along this axis

  float taps;       // kernel half-width in texels (≈ 2σ, capped)
  float _p0, _p1, _p2;
};

[numthreads(8, 8, 1)]
void main(uint3 gid : SV_DispatchThreadID) {
  int R = int(res);
  if (gid.x >= (uint)R || gid.y >= (uint)R) return;

  int2 d = int2(int(dir_x), int(dir_y));
  int2 base = int2(gid.xy);
  int  n = int(taps);

  float4 acc = float4(0.0, 0.0, 0.0, 0.0);
  [loop] for (int i = -n; i <= n; ++i) {
    int2 p = base + d * i;
    if (p.x < 0 || p.y < 0 || p.x >= R || p.y >= R) continue;
    float t = float(i) * inv_sigma;
    acc += srcTex.Load(int3(p, 0)) * exp(-t * t);
  }
  dstTex[gid.xy] = acc;
}
