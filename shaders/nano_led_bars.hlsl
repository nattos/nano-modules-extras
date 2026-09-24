// nano_led_bars.hlsl — the LED-bar pixel map, whole.
//
// UNUSUALLY FOR shaders_common, THIS IS A COMPLETE PASS and not a set of
// helpers: cbuffer, entry point and all. Every effect that publishes an LED
// map publishes the SAME map — four bars, ten segments, one colour per cell —
// and the only thing that differs between them is which colours they put in
// the buffer. So each effect's `led.hlsl` is a one-line include of this file,
// compiled and registered under its own name, and there is exactly one
// implementation of the painting.
//
// The host has already done all of the thinking (see <sketch/led_bars.h>);
// what is left here is laying 40 flat blocks over the output.
//
// WHY FLAT BLOCKS. This texture is a control signal, not a picture. Whatever
// consumes it — a pixel mapper sampling a point inside a region, a scaler
// averaging across one — has to come away with the exact colour the segment
// wants, and a gradient averages into mud. Hence `quantize`, which is on by
// default; turning it off interpolates between cell CENTRES, which is
// occasionally what you want for a soft wash and never what you want for a
// physical rig.

#include "nano_coords.hlsl"

static const int kNanoLedBars = 4;
static const int kNanoLedSegments = 10;
static const int kNanoLedCells = kNanoLedBars * kNanoLedSegments;

RWTexture2D<float4> outputTex : register(u0);

cbuffer LedUniforms : register(b1) {
  // Bar-major: cell `bar * 10 + segment`, rgb in xyz, w unused. Segment 0 is
  // the BOTTOM of the bar, matching <sketch/led_bars.h>.
  float4 led_cells[kNanoLedCells];
  // x = quantize (0 or 1), y = viewport width, z = viewport height, w unused.
  float4 led_ctl;
};

// uv (y running DOWN the image) to a continuous cell coordinate: bars on x,
// segments on y counting UP from the bottom, with cell CENTRES landing on
// integers. That last part is what makes the interpolated mode symmetric —
// interpolating between corners instead leaves a half-cell of flat colour at
// each edge and a doubled slope everywhere else.
float2 nano_led_cell_coord(float2 uv) {
  return float2(uv.x * float(kNanoLedBars) - 0.5,
                (1.0 - uv.y) * float(kNanoLedSegments) - 0.5);
}

float3 nano_led_fetch(int bar, int seg) {
  int b = clamp(bar, 0, kNanoLedBars - 1);
  int s = clamp(seg, 0, kNanoLedSegments - 1);
  return led_cells[b * kNanoLedSegments + s].rgb;
}

[numthreads(8, 8, 1)]
void main(uint3 gid : SV_DispatchThreadID) {
  uint W, H;
  outputTex.GetDimensions(W, H);
  if (gid.x >= W || gid.y >= H) return;

  float2 uv = nano_pixel_to_uv(float2(gid.xy), float2(W, H));
  float2 c = nano_led_cell_coord(uv);

  // Quantised: round to the nearest centre and take that cell whole. Not
  // floor() of the raw grid position — `c` is already centre-relative, so
  // rounding IS the nearest cell, and it keeps the two modes agreeing exactly
  // at every cell centre.
  float2 t = frac(c);
  int2 i0 = int2(floor(c));
  if (led_ctl.x > 0.5) {
    i0 = int2(round(c));
    t = float2(0.0, 0.0);
  }

  float3 c00 = nano_led_fetch(i0.x,     i0.y);
  float3 c10 = nano_led_fetch(i0.x + 1, i0.y);
  float3 c01 = nano_led_fetch(i0.x,     i0.y + 1);
  float3 c11 = nano_led_fetch(i0.x + 1, i0.y + 1);
  float3 rgb = lerp(lerp(c00, c10, t.x), lerp(c01, c11, t.x), t.y);

  // Opaque, always. A control map with holes in its alpha is a map that
  // whatever is under it leaks through.
  outputTex[gid.xy] = float4(saturate(rgb), 1.0);
}
