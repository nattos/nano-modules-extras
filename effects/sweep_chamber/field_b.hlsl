// source.particles.sweep_chamber — field pass B: velocity field construction.
// One thread per FIELD_RES² texel.
//
// Combines the two motion sources into the single texture every sim consumer
// (particles, tracers) samples with ONE bilinear tap per step:
//   .rg — curl-noise background velocity (uv/s). 3 octaves of analytic-
//         derivative gradient noise with rotating corner gradients (flow
//         noise): smooth (C1), seam-free, and time-churning eddies. The
//         velocity is the noise gradient rotated by noise_curl·90°: at ±1 it
//         is exactly curl(ψ) → divergence-free eddies; at 0 it's ∇ψ.
//   .ba — swept-image MULTI-SCALE gradient G = ∇L'·GAIN·edgeFade in iso
//         space (3 stencil scales — wide attraction basin), taken from
//         field_a's coarse L' (the 4×4 box downsample already smoothed it;
//         image_smoothing widens the sample step on top, replacing the old
//         full-res Gaussian pre-blur). Stored RAW — to_image/to_image_curl
//         are composed per-consumer so per-particle curl variation survives.
//
// Octave amplitude note: using the raw gradient per octave (not fo·gr) gives
// each octave a ψ-amplitude ∝ 1/fo, i.e. an equal VELOCITY contribution —
// otherwise the highest octave dominates 4× and eddy_detail does nothing.

#include "common.hlsl"

RWTexture2D<float4> fieldB     : register(u0);
Texture2D<float4>   fieldA     : register(t1);
SamplerState        lin        : register(s2);
Texture2D<float4>   fieldOr    : register(t4);   // .r σ, .g plain luma (fresh)
RWTexture2D<float4> fieldC     : register(u5);   // out: .rg ∇luma (multi-scale,
                                                 // EMA'd), .b σ — curl direction
Texture2D<float4>   fieldCPrev : register(t6);   // last frame's field_c (EMA)

cbuffer Uniforms : register(b3) {
  uint  field_res;
  float aspect_x;        // min(W,H)/W
  float aspect_y;        // min(W,H)/H
  float noise_speed;     // background velocity amplitude (uv/s)

  float noise_curl;      // 0 = gradient flow, ±1 = pure div-free eddies CW/CCW
  float eddy_scale;      // base frequency: 2..16 eddies across the min dimension
  float eddy_detail;     // octave gain (3 octaves × 0.7·detail)
  float spin_phase;      // ∫ eddy_evolve dt   (CPU-accumulated: param-change smooth)

  float drift_phase;     // ∫ eddy_drift dt, wrapped (CPU-accumulated)
  float drift_dir;       // eddy advection direction (turns)
  float image_smoothing; // gradient step 1→3 field texels
  float blend_k;         // field_c temporal EMA factor (matches field_a's)
}

static const float SWC_IMG_GAIN = 6.0;   // image gradients are small; amplify (dc parity)

// Background velocity in uv/s at s-space position `s`.
float2 swc_noise_vel(float2 s) {
  float freq = exp2(lerp(1.0, 4.0, saturate(eddy_scale)));
  float da   = drift_dir * 6.2831853;
  float2 ddir = float2(cos(da), sin(da));
  float rot = noise_curl * 1.5707963;
  float cr = cos(rot), sr = sin(rot);
  float2 v = float2(0.0, 0.0);
  float amp = 1.0, tot = 0.0;
  [unroll] for (uint o = 0u; o < 3u; o++) {
    float fo = freq * exp2(float(o));
    // Higher octaves drift + churn faster → turbulence-like eddy advection.
    float2 pp = (s - ddir * drift_phase * (1.0 + 0.7 * float(o))) * fo;
    float2 gr = swc_gnoise_grad(pp, spin_phase * (1.0 + 0.35 * float(o)),
                                0xA53u + o * 0x9E1u);
    v += amp * float2(gr.x * cr - gr.y * sr, gr.x * sr + gr.y * cr);
    tot += amp;
    amp *= saturate(eddy_detail) * 0.7;
  }
  return (v * noise_speed / max(tot, 1e-4)) * float2(aspect_x, aspect_y);
}

[numthreads(8, 8, 1)]
void main(uint3 gid : SV_DispatchThreadID) {
  if (gid.x >= field_res || gid.y >= field_res) return;
  float2 aspect = float2(aspect_x, aspect_y);
  float inv_res = 1.0 / float(field_res);
  float2 uv = (float2(gid.xy) + 0.5) * inv_res;

  // Swept-luma gradient from field_a's coarse L' (.r), sampled over equal
  // SCREEN distances (aspect-corrected steps) → an iso-space gradient.
  // MULTI-SCALE (e, 4e, 16e): a single fine stencil only exerts force within
  // a couple of field texels of a feature — particles a short distance away
  // felt nothing and the image's reach read as "super short". The coarse
  // stencils extend the attraction basin to ~15% of the screen while the
  // fine one still localizes ridges up close.
  //
  // Each scale is an 8-direction RING stencil, not a 4-tap axis cross: at
  // the coarse radii a cross imprints axis-aligned SQUARES around every
  // large feature (a bright centre region projects a visible rectangle into
  // the particle flow). Σ dir·L over 8 unit dirs at radius e ≈ 4e·∇L, so
  // ×0.5 matches the 2e·∇L scale of a plain central difference. 24 taps,
  // but this is the once-per-frame 256² field-gen pass — per-particle cost
  // is unchanged.
  // Two gradients in one sweep of the ring stencils:
  //   G  — swept L' (attraction: pulls particles toward the band)
  //   Gl — PLAIN luma (curl direction: sweep-independent, so passing the
  //        band never nulls/reverses the along-contour flow)
  float e0 = lerp(1.0, 3.0, saturate(image_smoothing)) * inv_res;
  float2 G  = float2(0.0, 0.0);
  float2 Gl = float2(0.0, 0.0);
  {
    static const float2 RING[8] = {
      float2( 1.0, 0.0), float2( 0.70710678,  0.70710678),
      float2( 0.0, 1.0), float2(-0.70710678,  0.70710678),
      float2(-1.0, 0.0), float2(-0.70710678, -0.70710678),
      float2( 0.0,-1.0), float2( 0.70710678, -0.70710678),
    };
    float e = e0, w = 1.0;
    for (uint sc = 0u; sc < 3u; sc++) {
      float2 accA = float2(0.0, 0.0);
      float2 accL = float2(0.0, 0.0);
      [unroll] for (uint d = 0u; d < 8u; d++) {
        float2 p = saturate(uv + RING[d] * (e * aspect));
        accA += RING[d] * fieldA.SampleLevel(lin, p, 0).r;
        accL += RING[d] * fieldOr.SampleLevel(lin, p, 0).g;
      }
      G  += accA * (0.5 * w);
      Gl += accL * (0.5 * w);
      e *= 4.0;
      w *= 0.55;
    }
  }
  // Taper the image force to zero at the frame edge (ClampToEdge sampling
  // makes the gradient vanish past the border — without the fade particles
  // pile up right at the viewport edge). dc parity.
  float2 ed = min(uv, 1.0 - uv);
  float edgeFade = smoothstep(0.0, 0.05, min(ed.x, ed.y));
  G  *= SWC_IMG_GAIN * edgeFade;
  Gl *= SWC_IMG_GAIN * edgeFade;

  float2 s = (uv - 0.5) / max(aspect, 1e-4);
  float2 vn = swc_noise_vel(s);

  fieldB[gid.xy] = float4(vn, G);
  // field_c gets its own EMA here (field_or is fresh — rgba16f storage is
  // write-only on web, so its pass can't blend in place).
  float  sigma = fieldOr.Load(int3(gid.xy, 0)).r;
  float4 prevC = fieldCPrev.Load(int3(gid.xy, 0));
  fieldC[gid.xy] = lerp(prevC, float4(Gl, sigma, 0.0), saturate(blend_k));
}
