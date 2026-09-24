// source.light.motion_blobs — color (darkening) pass.
//
// Per pixel: sum gaussian coverage from all alive blobs (distance
// computed in cover-square units so the footprint is aspect-correct).
// shadow_strength = saturate(sum_coverage) * shadow_darkness.
// out = lerp(tex_in, shadow_tint, shadow_strength).
//
// shadow_darkness == 0 → pass-through tex_in cleanly.

Texture2D<float4>   inputTex  : register(t0);
RWTexture2D<float4> outputTex : register(u1);

cbuffer Uniforms : register(b2) {
  float motion_strength;
  float shadow_darkness;
  float softness_curve;
  float motion_extent;       // motion pass only; unused here

  float shadow_r;
  float shadow_g;
  float shadow_b;
  float _pad1;

  float aspect_x;
  float aspect_y;
  float _pad2;
  float _pad3;

  uint  active_count;
  uint  debug_show_blobs;
  uint  _pad4;
  uint  _pad5;
};

struct GpuBlob {
  float x;
  float y;
  float vx;
  float vy;
  float radius;
  float _pp0;
  float _pp1;
  float _pp2;
};
StructuredBuffer<GpuBlob> blobs : register(t3);

[numthreads(8, 8, 1)]
void main(uint3 gid : SV_DispatchThreadID) {
  uint w, h;
  outputTex.GetDimensions(w, h);
  if (gid.x >= w || gid.y >= h) return;

  float2 uv = (float2(gid.xy) + 0.5) / float2(w, h);
  float4 base = inputTex[gid.xy];

  float2 cs_pixel = (uv - 0.5) / float2(aspect_x, aspect_y);

  float coverage = 0.0;
  bool any_center = false;

  uint N = active_count;
  if (N > 32u) N = 32u;
  for (uint i = 0u; i < N; i++) {
    GpuBlob b = blobs[i];
    float2 cs_blob = (float2(b.x, b.y) - 0.5) / float2(aspect_x, aspect_y);
    float r = max(b.radius, 1e-5);
    float d = length(cs_pixel - cs_blob) / r;
    float g = exp(-d * d * max(softness_curve, 0.01));
    coverage += g;
    if (debug_show_blobs != 0u && d < 0.03) any_center = true;
  }

  float shadow_strength = saturate(coverage) * saturate(shadow_darkness);
  float3 mixed = lerp(base.rgb, float3(shadow_r, shadow_g, shadow_b), shadow_strength);
  if (any_center) mixed = float3(0.0, 1.0, 0.4);
  outputTex[gid.xy] = float4(mixed, base.a);
}
