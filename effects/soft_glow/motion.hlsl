// source.light.soft_glow — motion-vector pass.
//
// Per pixel, accumulate per-blob velocity contributions weighted by
// the same gaussian footprint the color shader uses, optionally
// skewed toward the blob's direction of travel (wavefront) so a
// downstream motion blur only smears in front of each blob.
//
// Output: rgba16f motion texture. .xy = velocity (canvas-uv / sec);
// .w  = coverage mask used to lerp against upstream motion via the
// alpha-over render-pass blend convention shared across the motion-
// vector family.

#include "nano_bars.hlsl"

struct Blob {
  // .xy = uv center, .z = radius (cover-square units),
  // .w = per-blob amplitude — same coupling as color so dimmed blobs
  // also drive proportionally less motion.
  float4 pos_size;
  // .x = hue_offset (used by color shader), .yz = velocity (vx, vy),
  // .w = pad.
  float4 jitters;
};

StructuredBuffer<Blob> blobs       : register(t0);
Texture2D<float4>      upstreamMot : register(t1);
RWTexture2D<float4>    motionTex   : register(u2);

cbuffer Uniforms : register(b3) {
  uint  blob_count;
  float motion_strength;   // overall multiplier on emitted velocity
  float motion_skew;       // 0 = isotropic, 1 = full wavefront bias
  float aspect_x;

  float aspect_y;
  float motion_curl;       // signed [-1,+1] → rotate by curl·π (±π fully reverses)
  float intensity;         // scales coverage mask AND velocity magnitude
  float motion_extent;     // 1 = full blob footprint; <1 shrinks toward centers
};

// Plummer softening scale — same convention as the color shader so
// motion_skew and intensity_skew feel symmetric when ramped together.
static const float SKEW_SOFTENING_SCALE = 0.06;

[numthreads(8, 8, 1)]
void main(uint3 gid : SV_DispatchThreadID) {
  uint W, H;
  motionTex.GetDimensions(W, H);
  if (gid.x >= W || gid.y >= H) return;

  float2 uv = (float2(gid.xy) + 0.5) / float2(W, H);

  // Weighted-average accumulation — same pattern as motion_blobs /
  // motion_rect / motion_field. total_grad accumulates the analytical
  // gradient of the gaussian footprint (for motion_curl): the curl
  // tangent direction at this pixel is perpendicular to total_grad.
  float2 total_v    = float2(0.0, 0.0);
  float  total_w    = 0.0;
  float2 total_grad = float2(0.0, 0.0);

  for (uint i = 0u; i < blob_count; i++) {
    Blob b = blobs[i];
    if (b.pos_size.z <= 0.0) continue;

    // Aspect-corrected pixel-to-blob offset (cover-square space) so
    // the gaussian is isotropic regardless of viewport aspect.
    float2 r = uv - b.pos_size.xy;
    r.x /= max(aspect_x, 1e-4);
    r.y /= max(aspect_y, 1e-4);

    // Shrink the motion footprint toward the blob center by motion_extent:
    // the gaussian (and its gradient, so curl stays consistent) uses a
    // reduced effective radius. At 0.5 the vectors reach ~50% of the blob.
    float sigma = b.pos_size.z * max(motion_extent, 1e-4);
    float sigma2 = sigma * sigma;
    float r2 = dot(r, r) / sigma2;
    float gauss = exp(-r2 * 2.0) * b.pos_size.w;

    // Analytical gradient of gauss in r-space: ∂(amp·exp(-2|r|²/σ²))/∂r
    // = (-4/σ²) · r · gauss. Points TOWARD the blob center (because
    // intensity is higher there). Sums across blobs to give the field
    // gradient — perpendicular to which is the curl tangent.
    total_grad += (-4.0 / sigma2) * r * gauss;

    // Wavefront skew: project pixel-from-blob direction onto blob
    // velocity. Positive when the pixel is ahead of the blob (on
    // the leading edge); negative behind. Linear lerp between
    // isotropic (skew=0, mask=1) and front-biased (skew=1).
    //
    // Plummer softening: replace r/|r| with r/sqrt(|r|²+s²). The
    // direction vector smoothly collapses to 0 at the blob center
    // instead of flipping arbitrary direction across one pixel.
    // Asymptotically equal to the unsoftened r_hat past the softening
    // radius, so far-field behavior is unchanged.
    float2 v = b.jitters.yz;
    float v_len = length(v);
    float skew = saturate(motion_skew);
    float front_factor = 1.0;
    if (v_len > 1e-5) {
      // Softening tied directly to skew — one knob.
      float s = skew * SKEW_SOFTENING_SCALE;
      float soft_inv_r = rsqrt(dot(r, r) + s * s);
      float2 r_soft = r * soft_inv_r;
      // smoothstep, not saturate — eliminates the C1 seam along the
      // perpendicular plane through the blob center. Same convention
      // as the color shader's intensity_skew so behavior matches.
      front_factor = smoothstep(0.0, 1.0, dot(r_soft, v / v_len));
    }
    float skew_mask = lerp(1.0, front_factor, skew);

    float w = gauss * skew_mask;
    total_w += w;
    total_v += v * motion_strength * w;
  }

  float2 upstream = upstreamMot[gid.xy].xy;
  // Intensity scales BOTH coverage and velocity magnitude:
  //  * coverage → at intensity=0 the mask falls to 0 so upstream passes
  //    through unchanged (otherwise an invisible blob would still block
  //    the upstream field).
  //  * velocity → smear distance scales with how visible the blob is, so
  //    "visible blob = visible smear" stays a single-knob relationship.
  float local_mask = saturate(total_w * intensity);
  float2 local_motion = (total_w > 1e-5) ? (total_v / total_w) : float2(0.0, 0.0);
  local_motion *= intensity;

  // Gradient-driven curl: push the local motion contribution along
  // the *tangent* of the intensity field (perpendicular to ∇I) so the
  // result swirls AROUND blob centers like fluid flowing past
  // obstacles. Sign of motion_curl picks CCW (+) vs CW (−) around
  // each blob.
  //
  // Fade by smoothstep(|∇I|) so the swirl gently dies where the
  // intensity is flat — both far from any blob AND at the exact blob
  // center (where the gaussian's peak has zero gradient). Without the
  // fade the curl would also flicker direction on a single pixel at
  // each blob's center as the floating-point gradient wanders.
  float g_mag = length(total_grad);
  float2 perp_hat = (g_mag > 1e-5)
      ? float2(-total_grad.y, total_grad.x) / g_mag
      : float2(0.0, 0.0);
  float curl_fade = smoothstep(0.0, 1.0, g_mag);
  // Push proportional to existing motion magnitude so the swirl
  // strength tracks the in-frame motion field naturally (a slow blob
  // produces a gentle swirl; a fast one produces a wide one).
  float vel_mag = length(local_motion);
  local_motion += perp_hat * (vel_mag * motion_curl * curl_fade);

  float2 out_motion = lerp(upstream, local_motion, local_mask);
  motionTex[gid.xy] = float4(out_motion, 0.0, local_mask);
}
