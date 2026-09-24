// nano_bars.hlsl — shared bar-layout helpers for the lights bundle.
//
// The show targets 4 vertical LED bars mapped from 4 vertical canvas
// slices. Effects that need to be bar-aware lift the bar index and
// within-bar local U coord via these helpers so the convention stays
// consistent across the bundle.

#ifndef NANO_BARS_HLSL
#define NANO_BARS_HLSL

static const uint NANO_BAR_COUNT = 4u;

// uv.x in [0, 1) → bar index in [0, NANO_BAR_COUNT).
uint nano_bar_index(float u) {
  return uint(clamp(floor(u * float(NANO_BAR_COUNT)),
                    0.0,
                    float(NANO_BAR_COUNT - 1u)));
}

// uv.x → within-bar horizontal coord in [0, 1).
float nano_bar_local_u(float u) {
  return frac(u * float(NANO_BAR_COUNT));
}

#endif // NANO_BARS_HLSL
