// overlay — instanced solid-quad vertex shader. Six vertices per instance form
// one rectangle; the instance index selects the OverlayRect. Pixel bounds are
// mapped to y-down uv then to clip space (naga inserts the WebGPU y-flip when
// translating SPV → WGSL, so we leave the y axis alone — same convention as
// flash_particles/vs.hlsl).

#include "common.hlsl"

StructuredBuffer<OverlayRect> rects : register(t1);

struct VsOut {
  float4 pos : SV_Position;
  nointerpolation float4 color : TEXCOORD0;
};

[shader("vertex")]
VsOut main(uint vid : SV_VertexID, uint iid : SV_InstanceID) {
  // Unit quad as a 6-vertex triangle list, corners in [0,1]².
  static const float2 quad[6] = {
    float2(0.0, 0.0), float2(1.0, 0.0), float2(0.0, 1.0),
    float2(1.0, 0.0), float2(1.0, 1.0), float2(0.0, 1.0),
  };

  OverlayRect r = rects[iid];
  float2 c = quad[vid % 6u];

  float2 px = r.rect.xy + c * r.rect.zw;          // output-pixel position
  float2 uv = px / float2(max(vp_w, 1.0), max(vp_h, 1.0));   // y-down uv

  VsOut o;
  o.pos = float4(uv * 2.0 - 1.0, 0.0, 1.0);
  o.color = r.color;
  return o;
}
