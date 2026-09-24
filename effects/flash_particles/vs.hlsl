// source.particles.flash_particles — instanced quad vertex shader.
//
// Six vertices per instance form one quad. Vertex/instance ids drive
// corner selection + particle lookup; the storage buffer of particles
// is bound at register(t0). The same VS is shared by the color and
// motion fragment passes — both consume the same particle data and
// only the FS differs.
//
// Aspect handling: particle width/height are stored in "isotropic uv"
// — one unit corresponds to min(W, H) pixels. So size.x == size.y
// always renders as a real pixel square, regardless of viewport
// aspect. We rotate the corner offset in this isotropic frame, then
// convert to true uv by multiplying the longer axis by min/dim.

#include "common.hlsl"

StructuredBuffer<Particle> particles : register(t0);

cbuffer Uniforms : register(b1) {
  float aspect_x;   // (min(W,H) / W) — uv-x conversion factor
  float aspect_y;   // (min(W,H) / H)
  float _pad0;
  float _pad1;
};

struct VsOut {
  float4 pos       : SV_Position;
  // Particle-local corner ∈ [-1, +1]² (interpolated → mask coord).
  float2 corner    : TEXCOORD0;
  // Per-particle data forwarded as flat values (FS reads the same value
  // at every fragment). nointerpolation tells DXC/naga to emit `flat`
  // qualifiers so the GPU doesn't bother computing barycentric
  // interpolation on data that's constant across the quad.
  nointerpolation float4 captured  : TEXCOORD1;
  // x = rotation, y = life_norm, z = alpha_jitter_mult, w = frame_seed_as_float
  nointerpolation float4 state     : TEXCOORD2;
  // Hue/brightness/saturation/alpha jitters (captured at spawn).
  nointerpolation float4 jitters   : TEXCOORD3;
};

[shader("vertex")]
VsOut main(uint vid : SV_VertexID, uint iid : SV_InstanceID) {
  // 6-vertex triangle list covering the [-1, +1]² quad.
  static const float2 corners[6] = {
    float2(-1.0, -1.0), float2( 1.0, -1.0), float2(-1.0,  1.0),
    float2( 1.0, -1.0), float2( 1.0,  1.0), float2(-1.0,  1.0),
  };
  float2 c = corners[vid % 6u];

  Particle p = particles[iid];
  VsOut o;

  // Dead particles collapse to a degenerate triangle outside clip space
  // so the GPU rasterises nothing for them. Cheaper than a separate
  // alive-list compaction pass.
  if (p.state.y <= 0.0 || p.pos_size.z <= 0.0 || p.pos_size.w <= 0.0) {
    o.pos     = float4(2.0, 2.0, 2.0, 1.0);
    o.corner  = float2(0.0, 0.0);
    o.captured = float4(0.0, 0.0, 0.0, 0.0);
    o.state   = float4(0.0, 0.0, 0.0, 0.0);
    o.jitters = float4(0.0, 0.0, 0.0, 0.0);
    return o;
  }

  // Local quad in isotropic-uv space (square when size.x == size.y).
  float2 half_iso = p.pos_size.zw * 0.5;
  float2 local = c * half_iso;

  // Rotate within isotropic-uv first, then map back to true uv. This
  // keeps rotated quads visually undistorted on non-square viewports.
  float rot = p.state.x;
  float cs = cos(rot), sn = sin(rot);
  float2 rotated = float2(cs * local.x - sn * local.y,
                          sn * local.x + cs * local.y);
  float2 offset_uv = float2(rotated.x * aspect_x, rotated.y * aspect_y);
  float2 world_uv  = p.pos_size.xy + offset_uv;

  // uv → clip. DXC's `-spirv` target emits Vulkan NDC (y-down) and
  // naga inserts an automatic y-negation when translating to WebGPU
  // NDC (y-up), so we leave the y axis alone here. uv.y = 0 (top of
  // texture) maps to clip.y = -1 (bottom of Vulkan NDC); naga flips
  // that to +1 (top of WebGPU NDC), which the rasterizer then sends
  // to framebuffer row 0 (top of texture). Adding our own flip would
  // double-flip and visibly invert the output.
  float2 clip = world_uv * 2.0 - 1.0;

  float life_total = max(p.state.z, 1e-5);
  float life_norm  = saturate(p.state.y / life_total);

  o.pos     = float4(clip, 0.0, 1.0);
  o.corner  = c;
  o.captured = p.captured;
  o.state   = float4(rot, life_norm, p.jitters.w, p.meta.y);
  o.jitters = p.jitters;
  return o;
}
