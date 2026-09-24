// filter.sim.simulant — Pass D: the line extractor (display branch).
//
// Faithful port of the original's output chain reading accumRaw (node 20 out):
//   resize → Transform(Zoom) → Pixel Blur(Smoothing, done upstream) →
//   Bright.Contrast("Levels" bias/contrast, node 144) → Posterize(node 64) →
//   Edge Detection(Sobel, strength + sample-offset=Line Width, node 65) →
//   Crop(1px right/bottom, node 149) → Replace Alpha 1 (node 155) → out.
//
// The output is the extracted LINE FIELD (opaque), NOT composited over the
// original. Runs at viewport res, reconstructing the sim-res field with a
// B-spline BICUBIC filter: a plain bilinear tap is only C0, so its gradient
// JUMPS at every sim-texel boundary — and since the contour is an iso-line of
// that field, those jumps show up as a faceted "grid" kinking the lines. The
// smoothing cubic is C2, so the reconstructed field (and its Sobel gradient) is
// smooth → clean, grid-free contours even from a low-res field.

#include "nano_color.hlsl"

Texture2D<float4>   fieldTex : register(t0);   // smoothed accumRaw (sim-res)
SamplerState        samp     : register(s1);   // Linear + ClampToEdge
RWTexture2D<float4> outTex   : register(u2);   // rgba8

cbuffer Uniforms : register(b3) {
  float zoom, level_bias, level_contrast, posterize_levels;
  float edge_strength, line_width_px, crop_right, crop_bottom;
  float line_r, line_g, line_b, _p1;
};

// B-spline bicubic via 4 bilinear taps (Sigg & Hadwiger, GPU Gems 2). B-spline
// weights are all ≥ 0, so the s0/s1 denominators stay positive (Catmull-Rom's
// negative lobes would divide by ~0 → NaN). It is the *smoothing* cubic, exactly
// what de-grids a magnified low-res field.
float3 field_bicubic(float2 uv, float2 texSize) {
  float2 invTex = 1.0 / texSize;
  float2 tc = uv * texSize - 0.5;
  float2 c  = floor(tc) + 0.5;
  float2 f  = tc + 0.5 - c;

  float2 f2 = f * f, f3 = f2 * f;
  float2 w0 = (1.0 / 6.0) * (1.0 - 3.0 * f + 3.0 * f2 - f3);
  float2 w1 = (1.0 / 6.0) * (4.0 - 6.0 * f2 + 3.0 * f3);
  float2 w2 = (1.0 / 6.0) * (1.0 + 3.0 * f + 3.0 * f2 - 3.0 * f3);
  float2 w3 = (1.0 / 6.0) * f3;

  float2 s0 = w0 + w1;
  float2 s1 = w2 + w3;
  float2 o0 = w1 / s0;
  float2 o1 = w3 / s1;

  float2 t0 = (c - 1.0 + o0) * invTex;
  float2 t1 = (c + 1.0 + o1) * invTex;

  float3 a = fieldTex.SampleLevel(samp, float2(t0.x, t0.y), 0.0).rgb;
  float3 b = fieldTex.SampleLevel(samp, float2(t1.x, t0.y), 0.0).rgb;
  float3 cc= fieldTex.SampleLevel(samp, float2(t0.x, t1.y), 0.0).rgb;
  float3 d = fieldTex.SampleLevel(samp, float2(t1.x, t1.y), 0.0).rgb;
  return (a * s0.x + b * s1.x) * s0.y + (cc * s0.x + d * s1.x) * s1.y;
}

// Levels (node 144) then Posterize (node 64) over the bicubic-reconstructed luma.
float sample_luma(float2 uv, float2 fsize) {
  float g = nano_luminance(field_bicubic(uv, fsize));
  g = saturate((g - 0.5) * (1.0 + level_contrast) + 0.5 + level_bias);
  float L = max(posterize_levels, 2.0);
  return floor(g * L + 0.5) / L;
}

[numthreads(8, 8, 1)]
void main(uint3 gid : SV_DispatchThreadID) {
  uint W, H;
  outTex.GetDimensions(W, H);
  if (gid.x >= W || gid.y >= H) return;
  float2 uv = (float2(gid.xy) + 0.5) / float2(W, H);

  // Crop (node 149): trim the right / bottom edge (a faithful artifact guard).
  if (uv.x > 1.0 - crop_right || uv.y > 1.0 - crop_bottom) {
    outTex[gid.xy] = float4(0.0, 0.0, 0.0, 1.0);
    return;
  }

  uint fw, fh;
  fieldTex.GetDimensions(fw, fh);
  float2 fsize = float2(fw, fh);

  // Zoom about centre (Transform, node 116).
  float2 zuv = (uv - 0.5) / max(zoom, 1e-4) + 0.5;

  // Sobel over the levelled/posterized luma. sample-offset = Line Width (px).
  float2 o = line_width_px / float2(W, H);
  float tl = sample_luma(zuv + float2(-o.x, -o.y), fsize);
  float tc = sample_luma(zuv + float2( 0.0, -o.y), fsize);
  float tr = sample_luma(zuv + float2( o.x, -o.y), fsize);
  float ml = sample_luma(zuv + float2(-o.x,  0.0), fsize);
  float mr = sample_luma(zuv + float2( o.x,  0.0), fsize);
  float bl = sample_luma(zuv + float2(-o.x,  o.y), fsize);
  float bc = sample_luma(zuv + float2( 0.0,  o.y), fsize);
  float br = sample_luma(zuv + float2( o.x,  o.y), fsize);
  float gx = (tr + 2.0 * mr + br) - (tl + 2.0 * ml + bl);
  float gy = (bl + 2.0 * bc + br) - (tl + 2.0 * tc + tr);
  float edge = saturate(sqrt(gx * gx + gy * gy) * edge_strength);

  float3 col = edge * float3(line_r, line_g, line_b);
  outTex[gid.xy] = float4(col, 1.0);   // Replace Alpha 1
}
