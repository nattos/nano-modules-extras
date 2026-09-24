// warp.envelope — 1D coordinate-map raster: one instanced quad per envelope
// segment.
//
// Each instance is one Seg record: a destination span [d0, d1] along the map's
// single-row axis and a source span [s0, s1] with an inverse-ease exponent.
// The quad covers x = d0..d1 of the (N x 1) map target and the full 1-pixel
// row vertically; the fragment computes the exact analytic source coordinate.
// Painter's order (instance order) resolves fold-over: later segments
// overwrite earlier ones (the map PSO uses Replace blend — required anyway,
// since the rgba32float target is not blendable).

struct Seg {
  float d0; float d1;   // destination span, in map-domain units
  float s0; float s1;   // source span
  float inv_e;          // inverse ease exponent: 2^(3*ease)
  float _pad0; float _pad1; float _pad2;
};

StructuredBuffer<Seg> segs : register(t0);

cbuffer Uniforms : register(b1) {
  float dom;            // map domain length (1 = axis maps, >1 = radial)
  float _p0; float _p1; float _p2;
};

struct VsOut {
  float4 pos : SV_Position;
  // Destination fraction across the segment, 0 at d0 -> 1 at d1. Linear in
  // dest space, so the fragment's inverse ease is exact.
  float  td  : TEXCOORD0;
  nointerpolation float3 seg : TEXCOORD1;   // s0, s1, inv_e
};

[shader("vertex")]
VsOut main(uint vid : SV_VertexID, uint iid : SV_InstanceID) {
  // 6-vertex triangle list; x is the segment fraction (0..1), y spans the row.
  static const float2 corners[6] = {
    float2(0.0, -1.0), float2(1.0, -1.0), float2(0.0, 1.0),
    float2(1.0, -1.0), float2(1.0, 1.0), float2(0.0, 1.0),
  };
  float2 c = corners[vid % 6u];
  Seg sg = segs[iid];

  VsOut o;
  // Degenerate span: collapse outside clip space so nothing rasterizes.
  if (sg.d0 == sg.d1) {
    o.pos = float4(2.0, 2.0, 2.0, 1.0);
    o.td  = 0.0;
    o.seg = float3(0.0, 0.0, 1.0);
    return o;
  }

  float d = lerp(sg.d0, sg.d1, c.x);
  o.pos = float4((d / dom) * 2.0 - 1.0, c.y, 0.0, 1.0);
  o.td  = c.x;
  o.seg = float3(sg.s0, sg.s1, sg.inv_e);
  return o;
}
