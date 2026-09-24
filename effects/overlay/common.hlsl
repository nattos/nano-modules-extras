// overlay — shared solid-quad shader for the in-effect debug-overlay toolbox
// (wasm_modules/include/overlay.h). One instance per rectangle; the vertex
// shader expands a unit quad to the rect's pixel bounds and forwards a flat
// straight-alpha colour that the AlphaOver render PSO blends onto the target.
//
// This is the "colored rectangles / borders" heavy-lifter the overlay toolbox
// calls through the normal GPU ABI — text is handled separately by the host
// text engine (text::render), so there is no glyph/atlas logic here.

// One filled rectangle. rect = (x, y, w, h) in OUTPUT PIXELS (top-left origin);
// color = straight (non-premultiplied) rgba. 32 bytes → matches the C++
// `overlay::GpuRect` POD and the storage buffer stride.
struct OverlayRect {
  float4 rect;
  float4 color;
};

// Viewport in pixels (so the VS can map pixel rects → NDC). b0 → binding 0.
cbuffer OverlayUniforms : register(b0) {
  float vp_w;
  float vp_h;
  float _pad0;
  float _pad1;
};
