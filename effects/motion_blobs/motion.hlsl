// source.light.motion_blobs — motion-vector pass.
//
// Per pixel: gaussian-weighted sum of blob velocities * motion_strength.
// local_mask = saturate(total_w); blend upstream motion toward local
// by that mask. motion_strength == 0 → local contribution zeros out
// → mask still positive but local is zero, so upstream survives unless
// blobs are near. To get full passthrough on motion_strength == 0
// we just zero the local contribution AND the mask.

Texture2D<float4>   upstreamTex : register(t0);
RWTexture2D<float4> motionTex   : register(u1);

cbuffer Uniforms : register(b2) {
  float motion_strength;
  float shadow_darkness;
  float softness_curve;
  float motion_extent;       // 1 = full blob footprint; <1 shrinks toward centers

  float shadow_r; float shadow_g; float shadow_b; float _pad1;
  float aspect_x; float aspect_y; float _pad2; float _pad3;

  uint  active_count;
  uint  debug_show_blobs;
  uint  _pad4;
  uint  _pad5;
};

struct GpuBlob {
  float x;  float y;
  float vx; float vy;
  float radius;
  float _pp0; float _pp1; float _pp2;
};
StructuredBuffer<GpuBlob> blobs : register(t3);

[numthreads(8, 8, 1)]
void main(uint3 gid : SV_DispatchThreadID) {
  uint w, h;
  motionTex.GetDimensions(w, h);
  if (gid.x >= w || gid.y >= h) return;

  float2 uv = (float2(gid.xy) + 0.5) / float2(w, h);
  float4 upstream = upstreamTex[gid.xy];

  if (motion_strength <= 1e-6) {
    motionTex[gid.xy] = float4(upstream.xy, 0.0, 0.0);
    return;
  }

  float2 cs_pixel = (uv - 0.5) / float2(aspect_x, aspect_y);

  float2 v_accum = float2(0.0, 0.0);
  float  w_accum = 0.0;

  uint N = active_count;
  if (N > 32u) N = 32u;
  for (uint i = 0u; i < N; i++) {
    GpuBlob b = blobs[i];
    float2 cs_blob = (float2(b.x, b.y) - 0.5) / float2(aspect_x, aspect_y);
    // Shrink the motion footprint toward the blob center by motion_extent:
    // the gaussian uses a reduced effective radius, so at 0.5 the vectors
    // only reach ~50% of the visual blob's extent.
    float r = max(b.radius * motion_extent, 1e-5);
    float d = length(cs_pixel - cs_blob) / r;
    float g = exp(-d * d * max(softness_curve, 0.01));
    v_accum += float2(b.vx, b.vy) * (motion_strength * g);
    w_accum += g;
  }

  float mask = saturate(w_accum);
  float2 local = (w_accum > 1e-5) ? (v_accum / w_accum) : float2(0.0, 0.0);
  float2 mixed = lerp(upstream.xy, local, mask);
  motionTex[gid.xy] = float4(mixed, 0.0, 0.0);
}
