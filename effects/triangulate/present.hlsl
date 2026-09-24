// triangulate — present/debug pass. Bridges the internal proc-resolution maps to
// the viewport output and chooses what lands in tex_out:
//   debug_view: 0 off, 1 density, 2 ridge, 3 corner, 4 importance,
//               5 voronoi (cell id colour), 6 points (seed dots over input).
// The mesh is composited in a later phase; for now this is the terminal pass.
#include "common.hlsl"

Texture2D<float4>       featTex : register(t0);   // proc-res
Texture2D<float4>       inTex   : register(t1);   // viewport-res
[[vk::image_format("r32f")]] RWTexture2D<float> idTex : register(u2);  // proc-res
StructuredBuffer<Seed>  seeds   : register(t3);
RWTexture2D<float4>     outTex  : register(u4);    // viewport-res

cbuffer PresentUniforms : register(b5) {
  uint  u_debug_view;
  uint  u_bg_mode;
  uint  u_proc_w;
  uint  u_proc_h;
  float u_aspect;      // proc_w / proc_h
  float u_point_r;     // seed dot radius in uv
  uint  u_count;
  uint  u_pad0;
};

[numthreads(8, 8, 1)]
void main(uint3 gid : SV_DispatchThreadID) {
  uint w, h;
  outTex.GetDimensions(w, h);
  if (gid.x >= w || gid.y >= h) return;

  float2 uv = (float2(gid.xy) + 0.5) / float2(w, h);
  int2 pp = int2(clamp(uv, float2(0.0, 0.0), float2(0.99999, 0.99999)) * float2(u_proc_w, u_proc_h));
  float4 f = featTex.Load(int3(pp, 0));
  float3 inc = inTex[gid.xy].rgb;

  float3 c;
  if      (u_debug_view == 1u) c = f.rrr;                    // density
  else if (u_debug_view == 2u) c = float3(0.0, f.g, f.g);   // ridge (cyan)
  else if (u_debug_view == 3u) c = float3(f.b, 0.0, f.b);   // corner (magenta)
  else if (u_debug_view == 4u) c = f.aaa;                   // importance
  else if (u_debug_view == 5u) {                            // voronoi cells
    float cid = idTex[pp];
    if (cid < 0.0) c = float3(0.0, 0.0, 0.0);
    else {
      uint id = (uint)cid;
      c = float3(tri_hash_f(id * 3u + 1u), tri_hash_f(id * 3u + 2u), tri_hash_f(id * 3u + 3u));
    }
  } else if (u_debug_view == 6u) {
    c = inc;                                                 // points: over input
  } else {                                                   // 0 off: mesh backdrop
    if      (u_bg_mode == 1u) c = float3(0.0, 0.0, 0.0);     // dark
    else if (u_bg_mode == 2u) c = f.aaa * 0.5;              // feature (dim importance)
    else                       c = inc;                      // input
  }

  // Seed dots overlay (debug 6, and also on the voronoi view).
  if (u_debug_view == 6u || u_debug_view == 5u) {
    float cid = idTex[pp];
    if (cid >= 0.0) {
      float2 sp = seeds[(uint)cid].pos;
      float2 d = (uv - sp) * float2(u_aspect, 1.0);
      if (dot(d, d) < u_point_r * u_point_r) c = float3(1.0, 0.9, 0.2);
    }
  }

  outTex[gid.xy] = float4(c, 1.0);
}
