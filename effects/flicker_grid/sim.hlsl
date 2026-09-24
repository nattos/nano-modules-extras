// filter.light.flicker_grid — Pass 2: the per-column flicker machine.
//
// Single-threaded (gid 0 only) serial loop over columns — the native Metal
// backend hardcodes 8×8 threadgroups, which breaks single-workgroup parallel
// shaders, so per-column state stepping runs serially on one thread (the
// work is tiny: ≤ 16 columns × ≤ 32 rows of pre-reduced stats).
//
// Per column: reduce the rows to one luma L (peak or average), then run the
// pulse accumulator that lives in the persistent colstate buffer:
//   acc += min(L_norm * rate_max * dt, 0.5)   — capped at 0.5 cycles/frame,
//   pulse (gate=1) when acc wraps 1.           i.e. on/off every other frame.
// Since the increment never exceeds 0.5, two consecutive on-frames are
// impossible: pulses are exactly 1 frame and the gaps grow as L falls (the
// duty cycle shrinks). Demand beyond the cap can optionally "fill" the off
// frames: fill = (inc_raw - 0.5) / 0.5, reaching solid-on at 2× the cap.
// Below min_thr the column is dark (accumulator parked); at/above max_thr
// it's solid on.

#include "nano_sanitize.hlsl"
#include "common.hlsl"

StructuredBuffer<CellStat> stats : register(t0);
RWStructuredBuffer<ColState> colstate : register(u1);

cbuffer SimU : register(b2) {
  int cols; int rows; int mode; int do_reset;      // mode: 0 = peak, 1 = average
  float dt; float rate_max; float min_thr; float max_thr;
  int fill_enable; float neutral_pull; float upad0; float upad1;
};

[numthreads(64, 1, 1)]
void main(uint3 gid : SV_DispatchThreadID) {
  if (gid.x != 0u || gid.y != 0u || gid.z != 0u) return;

  [loop] for (int c = 0; c < cols; c++) {
    ColState prev = colstate[c];
    // Persistent state MUST sanitize on load (nano_sanitize.hlsl) — a stuck
    // NaN would otherwise freeze the column forever.
    float acc = nano_sanitize(prev.acc, 0.0, 0.0, 1.0);
    if (do_reset != 0) acc = 0.0;

    // Column reduce + leveling target in one row loop. The target is the
    // column's max POST-pull HSL lightness so it matches what render.hlsl
    // computes per cell.
    float l_peak = 0.0, l_sum = 0.0, target = 0.0;
    [loop] for (int r = 0; r < rows; r++) {
      CellStat st = stats[c * FG_MAX_ROWS + r];
      l_peak = max(l_peak, st.max_luma);
      l_sum += st.avg_luma;
      float3 rgb = saturate(float3(st.avg_r, st.avg_g, st.avg_b));
      float lite = (max(max(rgb.r, rgb.g), rgb.b) + min(min(rgb.r, rgb.g), rgb.b)) * 0.5;
      target = max(target, lerp(lite, 0.5, saturate(neutral_pull)));
    }
    float L = (mode == 0) ? l_peak : l_sum / float(max(rows, 1));

    float gate, fill;
    if (do_reset == 0 && dt <= 0.0) {
      // Re-render without a tick: hold last frame's outputs, don't advance.
      gate = prev.gate; fill = prev.fill;
    } else if (L < min_thr) {
      gate = 0.0; fill = 0.0; acc = 0.0;   // below the floor: dark, cycle parked
    } else if (L >= max_thr) {
      gate = 1.0; fill = 1.0; acc = 0.0;   // at the ceiling: solid on
    } else {
      float t = (L - min_thr) / max(max_thr - min_thr, 1e-4);
      float inc_raw = t * rate_max * dt;   // demanded cycles this frame
      float inc = min(inc_raw, 0.5);       // cap: strict on/off alternation
      acc += inc;
      gate = 0.0;
      if (acc >= 1.0) { acc -= 1.0; gate = 1.0; }
      fill = (fill_enable != 0) ? saturate((inc_raw - 0.5) / 0.5) : 0.0;
    }

    ColState next;
    next.acc = acc; next.gate = gate; next.fill = fill; next.level_target = target;
    colstate[c] = next;
  }
}
