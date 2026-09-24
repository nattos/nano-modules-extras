// filter.legacy.glisten — sparkle fragment shader (additive, half-res 16F).
//
// Flat passthrough: the radial falloff comes from the stacked levels and the
// post blur, not from the fragment. Colour is intentionally unclamped —
// negative rgb subtracts under additive blending.

struct VsOut {
  float4 pos   : SV_Position;
  float4 color : TEXCOORD0;
};

[shader("pixel")]
float4 main(VsOut i) : SV_Target0 {
  return i.color;
}
