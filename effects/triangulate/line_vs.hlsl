// triangulate — instanced line-quad vertex shader. One instance per Delaunay
// edge (6 verts/quad). Reads the packed edge (two seed indices), looks up both
// endpoints in the seed pool, and builds a screen-space quad with a half-width
// perpendicular offset. Empty/degenerate slots collapse off-clip.
#include "common.hlsl"

StructuredBuffer<uint> edges : register(t1);
StructuredBuffer<Seed> seeds : register(t2);

cbuffer LineUniforms : register(b0) {
  float u_vp_x;
  float u_vp_y;
  float u_half_w;     // half line width in viewport pixels
  float u_threshold;  // binary edge-colour threshold on feature weight
  float u_cr, u_cg, u_cb, u_pad1;      // base colour (below threshold)
  float u_fcr, u_fcg, u_fcb, u_pad2;   // feature colour (above threshold)
};

struct VsOut {
  float4 pos   : SV_Position;
  float2 local : TEXCOORD0;   // y = across ∈ [-1,1] for the soft rim
  nointerpolation float3 color : TEXCOORD1;
};

VsOut main(uint vid : SV_VertexID, uint iid : SV_InstanceID) {
  static const float2 quad[6] = {
    float2(0.0, -1.0), float2(1.0, -1.0), float2(0.0, 1.0),
    float2(1.0, -1.0), float2(1.0,  1.0), float2(0.0, 1.0),
  };
  VsOut o;
  o.color = float3(u_cr, u_cg, u_cb);

  uint e = edges[iid];
  if (e == 0xFFFFFFFFu) { o.pos = float4(2, 2, 2, 1); o.local = float2(0, 0); return o; }
  uint a = e >> 16, b = e & 0xFFFFu;

  // Binary edge colour by feature weight: an edge is a "feature edge" when BOTH
  // endpoints sit on a feature (min of the two importance weights > threshold).
  float ew = min(seeds[a].score, seeds[b].score);
  o.color = (ew > u_threshold) ? float3(u_fcr, u_fcg, u_fcb) : float3(u_cr, u_cg, u_cb);

  float2 vp = float2(u_vp_x, u_vp_y);
  float2 p0 = seeds[a].pos * vp;
  float2 p1 = seeds[b].pos * vp;
  float2 d = p1 - p0;
  float len = length(d);
  if (len < 1e-4) { o.pos = float4(2, 2, 2, 1); o.local = float2(0, 0); return o; }

  float2 dir = d / len;
  float2 perp = float2(-dir.y, dir.x);
  float2 c = quad[vid % 6u];
  float2 wpx = lerp(p0, p1, c.x) + perp * (c.y * u_half_w);
  float2 uv = wpx / vp;
  o.pos = float4(uv * 2.0 - 1.0, 0.0, 1.0);   // y-down uv; naga flips to NDC
  o.local = c;
  return o;
}
