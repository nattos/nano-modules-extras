// source.light.side_racks — the six-bar rig's pixel map, drawn straight onto
// tex_out: the compact grid of <sketch/rack_map.h>, 120 columns by 3 rows.
//
// The left half of the frame is the left rack and the right half the right
// rack; row 0 (the top of the frame) is the top bar. The host has already laid
// the cells out in grid order (rack_map::gridIndex), so all that is left here
// is painting them. Like nano_led_bars.hlsl this is a control signal, not a
// picture: quantized (the default) paints each segment as a flat block so a
// fixture footprint anywhere in it lands on the exact colour; off, it
// interpolates along a bar between segment centres — never across the gap
// between the racks, and never between bars.

#include "nano_coords.hlsl"

static const int kRackCols = 120;
static const int kRackRows = 3;
static const int kRackHalf = 60;

RWTexture2D<float4> outputTex : register(u0);

cbuffer RackUniforms : register(b1) {
  // Grid order: cell `row * 120 + column`, rgb in xyz, w unused.
  float4 rack_cells[kRackCols * kRackRows];
  // x = quantize (0 or 1), yzw unused.
  float4 rack_ctl;
};

float3 rack_fetch(int col, int row) {
  return rack_cells[row * kRackCols + col].rgb;
}

[numthreads(8, 8, 1)]
void main(uint3 gid : SV_DispatchThreadID) {
  uint W, H;
  outputTex.GetDimensions(W, H);
  if (gid.x >= W || gid.y >= H) return;

  float2 uv = nano_pixel_to_uv(float2(gid.xy), float2(W, H));
  int row = clamp(int(floor(uv.y * float(kRackRows))), 0, kRackRows - 1);

  // Column coordinate with segment CENTRES on integers, as in nano_led_bars.
  float cx = uv.x * float(kRackCols) - 0.5;
  int c0 = int(floor(cx));
  float t = frac(cx);
  if (rack_ctl.x > 0.5) {
    c0 = int(round(cx));
    t = 0.0;
  }

  // Keep to this pixel's own rack.
  int lo = 0;
  if (uv.x >= 0.5) lo = kRackHalf;
  int a = clamp(c0, lo, lo + kRackHalf - 1);
  int b = clamp(c0 + 1, lo, lo + kRackHalf - 1);
  float3 rgb = lerp(rack_fetch(a, row), rack_fetch(b, row), t);

  // Opaque, always: a control map with holes in its alpha leaks.
  outputTex[gid.xy] = float4(saturate(rgb), 1.0);
}
