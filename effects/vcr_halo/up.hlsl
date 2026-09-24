// vcr_halo up — one octave of the progressive upsample, with that octave's
// own contribution folded in.
//
//   S[k] = tent(S[k+1]) * src_scale + band(k) * add_weight
//
// Unrolled down the chain that is exactly
//   S[0] = sum_k w_k * kernel_k(emitter),
// the pyramid's answer to three_planes' analytic
//   halo_profile(d) = sum_j a_j * exp(-d / (R * s_j)).
// Both are a weighted stack of kernels at geometrically spaced radii. The
// analytic one gets the radii exactly; this one has them fixed at octaves and
// slides weight between neighbours instead, which is what keeps Halo Radius
// continuously modulatable with no level popping in.
//
// `outline` chooses what kernel_k is:
//
//   0 — GAUSSIAN pyramid. band(k) = D[k], a low pass. A filled shape blooms
//       whole, which is ordinary bloom.
//   1 — LAPLACIAN pyramid. band(k) = max(D[k] - stretch(D[k+1]), 0), the
//       octave's own band. Flat areas cancel to nothing and only edges
//       survive, so a filled shape glows at its boundary like neon tubing.
//
// Doing the band-pass HERE rather than in the prefilter is the whole point:
// each octave subtracts its own neighbour, so the width of the detected edge
// scales with the octave — and therefore with Halo Radius, since that is what
// picks which octaves carry weight. A fixed-offset high-pass in the prefilter
// would measure a ~2px band at every resolution and every radius, which is
// invisible as soon as the halo is wide.
//
// The coarser level is read with a plain bilinear stretch, not a tent. It is
// only the subtrahend, and everything downstream blurs it again.
//
// 9-tap tent for the upsample (Jimenez, SIGGRAPH 2014): a straight bilinear
// stretch is boxy, the 1/2/4 tent is not.

Texture2D<float4>   srcTex   : register(t0);   // coarser accumulation, S[k+1]
Texture2D<float4>   addTex   : register(t1);   // this octave's downsample, D[k]
Texture2D<float4>   coarseTex: register(t2);   // next octave down, D[k+1]
RWTexture2D<float4> dstTex   : register(u3);   // S[k]
SamplerState        samp     : register(s4);

cbuffer Uniforms : register(b5) {
  float src_texel_x;   // 1 / coarse level width
  float src_texel_y;   // 1 / coarse level height
  float src_scale;     // weight of the coarsest level (1.0 after the seed)
  float add_weight;    // w_k for this level

  float outline;       // 0 gaussian (low pass) .. 1 laplacian (band pass)
  float _pad0;
  float _pad1;
  float _pad2;
};

[numthreads(8, 8, 1)]
void main(uint3 gid : SV_DispatchThreadID) {
  uint dw, dh;
  dstTex.GetDimensions(dw, dh);
  if (gid.x >= dw || gid.y >= dh) return;

  float2 uv = (float2(gid.xy) + 0.5) / float2(dw, dh);
  float2 t  = float2(src_texel_x, src_texel_y);

  float4 t0 = srcTex.SampleLevel(samp, uv + t * float2(-1,  1), 0);
  float4 t1 = srcTex.SampleLevel(samp, uv + t * float2( 0,  1), 0);
  float4 t2 = srcTex.SampleLevel(samp, uv + t * float2( 1,  1), 0);
  float4 t3 = srcTex.SampleLevel(samp, uv + t * float2(-1,  0), 0);
  float4 t4 = srcTex.SampleLevel(samp, uv,                      0);
  float4 t5 = srcTex.SampleLevel(samp, uv + t * float2( 1,  0), 0);
  float4 t6 = srcTex.SampleLevel(samp, uv + t * float2(-1, -1), 0);
  float4 t7 = srcTex.SampleLevel(samp, uv + t * float2( 0, -1), 0);
  float4 t8 = srcTex.SampleLevel(samp, uv + t * float2( 1, -1), 0);

  float4 up = (t0 + t2 + t6 + t8) * (1.0 / 16.0)
            + (t1 + t3 + t5 + t7) * (2.0 / 16.0)
            + t4 * (4.0 / 16.0);

  float4 band = addTex.SampleLevel(samp, uv, 0);
  if (outline > 0.0) {
    float4 lo = coarseTex.SampleLevel(samp, uv, 0);
    // Clamped at zero so only the BRIGHT side of an edge emits — the dark
    // side would otherwise ring with a negative halo.
    band = lerp(band, max(band - lo, 0.0), saturate(outline));
  }

  dstTex[gid.xy] = up * src_scale + band * add_weight;
}
