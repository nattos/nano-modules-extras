// filter.sim.simulant — separable RGB spread kernel (Pixel Blur, nodes 30 + 67).
//
// Used for two jobs, each an H then V pass:
//   (1) WAVE spread: the big feedback pass. Repeated every frame, this outward
//       spread IS the propagation of the original — there is no wave equation.
//       Two techniques (mode):
//         DIFFUSE — separable Gaussian blur (the faithful soft rings).
//         DILATE  — separable PARABOLIC grayscale morphology (max of the window
//                   minus a quadratic distance penalty). Parabolic morphology is
//                   exactly Euclidean and separable, so fronts stay ROUND (a flat
//                   max window would give square/diamond fronts). This spreads
//                   bright regions as sharp EXPANDING FRONTS instead of soft
//                   diffusion; `post_mult` (< 1) is the per-frame decay so the
//                   fronts fade rather than fill the frame.
//       The V pass also folds in the WaveSpeed-tied contrast darkening (node 31).
//   (2) SMOOTHING: a small pre-edge Gaussian (node 67) so the Sobel traces clean
//       contours. Always DIFFUSE.
//
// sigma/step are per-axis uv (isotropic in pixels); taps scale with reach.

Texture2D<float4>   srcTex : register(t0);
SamplerState        samp   : register(s1);   // Linear + ClampToEdge
RWTexture2D<float4> dstTex : register(u2);

cbuffer Uniforms : register(b3) {
  float2 dir;        // (1,0) horizontal or (0,1) vertical
  float  step_uv;    // tap spacing in uv (= reach / taps)
  float  sigma_uv;   // Gaussian sigma in uv (<=0 → identity)
  float  contrast;   // Bright.Contrast about 0.5 (node 31 wave decay; 0 = none)
  float  mode;       // 0 = diffuse (Gaussian), 1 = dilate (parabolic morphology)
  float  parab;      // dilation parabola steepness (penalty per tap², in value)
  float  post_mult;  // final multiply (dilate per-frame decay; 1 = none)
  float  taps;       // samples each side (Quality). reach = taps * step_uv.
  float  _q0, _q1, _q2;
};

// Bright.Contrast (node 31): pivots about mid-grey.
float3 apply_contrast(float3 c) {
  return saturate((c - 0.5) * (1.0 + contrast) + 0.5);
}

[numthreads(8, 8, 1)]
void main(uint3 gid : SV_DispatchThreadID) {
  uint W, H;
  dstTex.GetDimensions(W, H);
  if (gid.x >= W || gid.y >= H) return;
  float2 uv = (float2(gid.xy) + 0.5) / float2(W, H);

  if (sigma_uv <= 1e-6) {
    float3 c = srcTex.SampleLevel(samp, uv, 0).rgb;
    dstTex[gid.xy] = float4(saturate(apply_contrast(c) * post_mult), 1.0);
    return;
  }

  int N = clamp((int)taps, 1, 64);   // samples each side — the Quality control

  float3 outc;
  if (mode > 0.5) {
    // DILATE — parabolic max: out = max_k( sample_k - parab*k² ). The k² penalty
    // adds across the H+V passes → total penalty parab*(kx²+ky²) = Euclidean.
    float3 acc = float3(-1e9, -1e9, -1e9);
    [loop] for (int k = -N; k <= N; k++) {
      float pen = parab * float(k * k);
      float3 s = srcTex.SampleLevel(samp, uv + dir * (float(k) * step_uv), 0).rgb;
      acc = max(acc, s - pen);
    }
    outc = saturate(apply_contrast(saturate(acc)) * post_mult);
  } else {
    // DIFFUSE — separable Gaussian.
    float3 acc = float3(0.0, 0.0, 0.0);
    float  wsum = 0.0;
    float  inv2s2 = 1.0 / (2.0 * sigma_uv * sigma_uv);
    [loop] for (int k = -N; k <= N; k++) {
      float d = float(k) * step_uv;
      float w = exp(-d * d * inv2s2);
      acc  += srcTex.SampleLevel(samp, uv + dir * d, 0).rgb * w;
      wsum += w;
    }
    outc = saturate(apply_contrast(acc / max(wsum, 1e-5)) * post_mult);
  }
  dstTex[gid.xy] = float4(outc, 1.0);
}
