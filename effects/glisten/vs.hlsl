// filter.legacy.glisten — sparkle fan vertex shader.
//
// Faithful port of NanoGraph Glisten.txt geometry. Each level is an inscribed
// `blades`-gon disc triangulated as a FAN FROM A RIM VERTEX: the shared apex
// sits ON the circle at the gradient direction, and (blades−2) triangles
// sweep the rest of the polygon. Coverage is a filled disc — the "blade" look
// comes from the huge per-vertex colour gradients interpolating across the
// fan (seams radiate from the rim apex), not from spike geometry.
//
// Levels stack concentric discs: scale shrinks with level index while alpha
// grows, so the additive sum forms the radial falloff (no fragment falloff).
// Vertex colours are intentionally UNCLAMPED — negative colours subtract
// under additive blending (the "digs deep" character).
//
// Aspect quirk kept from the original: clip x is compressed by min(h/w, 1),
// which both makes the fan circular in pixels and pulls the anchor position
// toward the horizontal centre on wide outputs.

StructuredBuffer<float> anchor : register(t0);

cbuffer U : register(b1) {
  float aspect_x;            // min(h/w, 1) — original used h/w
  float blades, levels, size;

  float shape;               // gradation shape, ExpCurve-mapped, >= 0.01
  float stretch_grad, stretch_squash, stretch_x;

  float stretch_y, cg_power; // colour-gradient power (slider × 100)
  float intensity, _p0;
};

static const float PI = 3.14159265358979323846;

struct VsOut {
  float4 pos   : SV_Position;
  float4 color : TEXCOORD0;   // rgb (unclamped, may be negative) + level alpha
};

[shader("vertex")]
VsOut main(uint vid : SV_VertexID, uint iid : SV_InstanceID) {
  VsOut o;

  float2 pos   = float2(anchor[0], anchor[1]);
  float2 dir   = float2(anchor[2], anchor[3]);
  float2 grad  = float2(anchor[4], anchor[5]);
  float3 col   = float3(anchor[6], anchor[7], anchor[8]);
  float3 cgx   = float3(anchor[9], anchor[10], anchor[11]) * cg_power;
  float3 cgy   = float3(anchor[12], anchor[13], anchor[14]) * cg_power;

  int nBlades = max((int)blades, 3);
  int nLevels = max((int)levels, 1);
  int stepCount = nBlades - 2;          // fan from a rim vertex fills the polygon
  uint level = iid / (uint)stepCount;
  uint step  = iid % (uint)stepCount;

  float sh = max(shape, 0.01);
  float fL = (float)nLevels;
  float scale = (fL * sh) / ((float)level + fL * sh) * size;
  float alpha = (1.0 + (float)level) / (fL * (fL + 1.0)) * intensity;

  // Vertex k of the fan: apex (k=0) at dir, others swept k steps around.
  uint k = (vid == 0u) ? 0u : (step + vid);
  float ang = (float)k * (2.0 * PI / (float)nBlades);
  float c = cos(ang), s = sin(ang);
  float2 v = float2(dir.x * c - dir.y * s, dir.x * s + dir.y * c);

  // Stretch along the gradient direction, then per-axis squash.
  float stretchPow = atan(length(grad) * stretch_squash);
  v += dot(v, dir) * (stretch_grad - 1.0) * stretchPow;
  v *= float2(stretch_x, stretch_y);

  // Colour from the post-stretch UNIT-disc coordinate (not the scaled one).
  float3 shaded = col + v.x * cgx + v.y * cgy;

  float2 p = pos + v * scale;           // uv units on both axes
  o.pos = float4((p.x * 2.0 - 1.0) * aspect_x, p.y * 2.0 - 1.0, 0.0, 1.0);
  o.color = float4(shaded, alpha);
  return o;
}
