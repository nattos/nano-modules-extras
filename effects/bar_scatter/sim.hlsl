// filter.light.bar_scatter — Pass 2: the decisions (GPU-resident state).
//
// Single-threaded (gid 0 only), like bounce_resonator_fx's sim: the native
// Metal backend hardcodes 8×8 threadgroups, and the work is tiny.
//
// Continuous, not latched: the input excites every band all the time, to
// whatever degree, and nothing is ever held as a decision. An event stores
// only its dice; every frame each bar's fate is re-derived from those dice
// against the LIVE params, and every effect is a pure function of the time
// since its event — so a knob move lands on the next frame, a fresh flash
// replaces whatever the last one did, and there is no state to get stuck in.
//
// Per frame:
//   1. CELL ONSET. Each cell's `ref` is a floor follower of its luma — it
//      drops to the luma at once and climbs back over `rise_time` — so the
//      rise above it is the size of a sudden jump and ~0 for a slow fade-in.
//      Reset zeroes it, so an image arriving is itself an onset.
//   2. EXCITATION. A band's onset is the MEAN rise over its cells (a whole
//      band flashing reads fully, one cell of ten a tenth), eased through a
//      soft knee: x = smoothstep(sensitivity/2, sensitivity, onset).
//   3. EVENTS. A band starts a new event whenever x beats its own recent
//      peak (`det`, released over DET_TIME) by a margin — each flash of a
//      fast strobe is its own event, while one flash's tail never is. The
//      event rolls fresh dice and records `co`, how strongly the OTHER bars
//      in its zone row are flashing with it (their x held over `window`,
//      summed, less `min_bars` − 2).
//   4. FATE, live. p = strength·(1 − tilt·c)·co, c = +1 for the centre pair
//      and −1 for the outer pair; the bar is hit when its dice fall under p,
//      and a hit is a delay when its second die falls under `delay_share`,
//      else a choke. If every bar whose event began within `window` of this
//      one is hit, the one closest to missing passes: a flash always lands
//      somewhere. A bar flashing alone has co 0 and passes.
//   5. CHOKE removes only the NEW light — live minus its pre-flash colour —
//      scaled by the event's excitation, held for `choke_length`, then
//      released over `choke_release`.
//   6. DELAY reads the band from history `lag` seconds back. The delayed bars
//      of one event are staggered 1, 2, 3 steps (shuffled, jittered). After
//      `lag + delay_hold` the lag decays over `return_time`: the read head
//      fast-forwards to live.
// dt = 0 (a re-render with no tick) steps nothing and re-derives `proc`.

#include "common.hlsl"

StructuredBuffer<CellStat>     stats     : register(t0);
StructuredBuffer<float4>       hist      : register(t1);   // BS_HIST × BS_MAX_CELLS
StructuredBuffer<float>        slot_time : register(t2);   // BS_HIST; < -1e8 = never written
RWStructuredBuffer<CellState>  cells     : register(u3);
RWStructuredBuffer<GroupState> groups    : register(u4);
RWStructuredBuffer<float4>     proc      : register(u5);

cbuffer SimU : register(b6) {
  float now; float dt; int rows; int zones;
  int head; int min_bars; int do_reset; uint seed;
  uint frame; float strength; float tilt; float delay_share;
  float sensitivity; float window; float choke_depth; float choke_length;
  float choke_release; float delay_step; float delay_jitter; float delay_hold;
  float return_time; float rise_time; float spad0; float spad1;
};

// How long a band's own excitation peak is held against its next flash. It
// must outlast the onset's own fall (rise_time) so a flash's tail never reads
// as a new flash, and stay short enough that a 16th-note strobe still does.
static const float DET_TIME = 0.2;
static const float DET_MARGIN = 0.1;
static const float MIN_EXCITE = 0.15;   // weaker than this never starts an event
static const float NEVER = -1e9;

static float hash01(uint a, uint b, uint c) {
  uint h = a ^ (b * 0x9E3779B1u) ^ (c * 0x85EBCA77u);
  h ^= h >> 16; h *= 0x7FEB352Du; h ^= h >> 15; h *= 0x846CA68Bu; h ^= h >> 16;
  return float(h >> 8) * (1.0 / 16777216.0);
}

// The centre pair (bars 1, 2) is +1, the outer pair −1.
static float centreness(int bar) { return (bar == 1 || bar == 2) ? 1.0 : -1.0; }

// A cell's colour at sim time `t`. Slots are walked back from `head`, newest
// first; a slot never written means there was no picture yet (black). A time
// before the whole ring clamps to its oldest slot.
float3 readHistory(int cell, float t) {
  float t_next = slot_time[head];
  int s_next = head;
  if (t >= t_next) return hist[head * BS_MAX_CELLS + cell].rgb;
  [loop] for (int i = 1; i < BS_HIST; i++) {
    int s = (head - i + BS_HIST) % BS_HIST;
    float ts = slot_time[s];
    if (ts < -1e8) return float3(0.0, 0.0, 0.0);
    if (ts <= t) {
      float f = (t_next > ts) ? saturate((t - ts) / (t_next - ts)) : 0.0;
      return lerp(hist[s * BS_MAX_CELLS + cell].rgb, hist[s_next * BS_MAX_CELLS + cell].rgb, f);
    }
    t_next = ts; s_next = s;
  }
  return hist[s_next * BS_MAX_CELLS + cell].rgb;
}

[numthreads(64, 1, 1)]
void main(uint3 gid : SV_DispatchThreadID) {
  if (gid.x != 0u || gid.y != 0u || gid.z != 0u) return;

  int nrows = clamp(rows, 1, BS_MAX_ROWS);
  int nzones = clamp(zones, 1, nrows);
  bool step = dt > 0.0;

  if (do_reset != 0) {
    [loop] for (int i = 0; i < BS_MAX_CELLS; i++) {
      CellState z;
      z.ref = 0.0;   // an image arriving is itself an onset
      z.prev_r = 0.0; z.prev_g = 0.0; z.prev_b = 0.0;
      z.snap_r = 0.0; z.snap_g = 0.0; z.snap_b = 0.0;
      z.pad = 0.0;
      cells[i] = z;
    }
    [loop] for (int j = 0; j < BS_MAX_GROUPS; j++) {
      GroupState g;
      g.det = 0.0; g.hot = 0.0; g.t_event = NEVER; g.co = 0.0; g.amp = 0.0;
      g.u_hit = 1.0; g.u_mode = 1.0; g.u_order = 0.0; g.u_jit = 0.5;
      g.pad0 = 0.0; g.pad1 = 0.0; g.pad2 = 0.0;
      groups[j] = g;
    }
  }

  // --- 1. Cell onset ---
  float rise[BS_MAX_CELLS];
  float follow = step ? 1.0 - exp(-dt / max(rise_time, 1e-3)) : 0.0;
  [loop] for (int c = 0; c < nrows * BS_BARS; c++) {
    CellState cs = cells[c];
    float l = stats[c].luma;
    rise[c] = max(l - cs.ref, 0.0);
    if (step) {
      cs.ref = (l < cs.ref) ? l : cs.ref + (l - cs.ref) * follow;
      cells[c] = cs;
    }
  }

  float sens = max(sensitivity, 1e-3);
  float det_keep = step ? exp(-dt / DET_TIME) : 1.0;
  float hot_keep = step ? ((window > 1e-3) ? exp(-dt / window) : 0.0) : 1.0;
  float base_p[BS_BARS];
  [unroll] for (int bp = 0; bp < BS_BARS; bp++)
    base_p[bp] = saturate(strength * (1.0 - tilt * centreness(bp)));

  [loop] for (int z = 0; z < nzones; z++) {
    int r0 = z * nrows / nzones, r1 = (z + 1) * nrows / nzones;
    int gbase = z * BS_BARS;
    GroupState G[BS_BARS];
    float x[BS_BARS];
    bool fresh[BS_BARS];

    // --- 2. Excitation, and the peak-holds ---
    [unroll] for (int b = 0; b < BS_BARS; b++) {
      G[b] = groups[gbase + b];
      float onset = 0.0;
      for (int r = r0; r < r1; r++) onset += rise[r * BS_BARS + b];
      onset /= float(max(r1 - r0, 1));
      x[b] = smoothstep(0.5 * sens, sens, onset);
      float det_prev = G[b].det * det_keep;
      fresh[b] = step && x[b] > MIN_EXCITE && x[b] > det_prev + DET_MARGIN;
      if (step) {
        G[b].det = max(det_prev, x[b]);
        G[b].hot = max(G[b].hot * hot_keep, x[b]);
      }
    }

    // --- 3. Events: fresh dice, and how many flashed along ---
    uint ev = frame * 131u + uint(z) * 7919u;
    [unroll] for (int b2 = 0; b2 < BS_BARS; b2++) {
      if (!fresh[b2]) continue;
      float others = 0.0;
      [unroll] for (int o = 0; o < BS_BARS; o++) if (o != b2) others += G[o].hot;
      G[b2].co = saturate(others - float(min_bars - 2));
      G[b2].amp = x[b2];
      G[b2].t_event = now;
      G[b2].u_hit   = hash01(seed, ev, uint(b2));
      G[b2].u_mode  = hash01(seed, ev, uint(b2) + 16u);
      G[b2].u_order = hash01(seed, ev, uint(b2) + 32u);
      G[b2].u_jit   = hash01(seed, ev, uint(b2) + 48u);
      // The pre-flash picture, for the choke to hold the band at.
      for (int r = r0; r < r1; r++) {
        int ci = r * BS_BARS + b2;
        CellState cs = cells[ci];
        cs.snap_r = cs.prev_r; cs.snap_g = cs.prev_g; cs.snap_b = cs.prev_b;
        cells[ci] = cs;
      }
    }

    // --- 4. Fate, from the dice and the live params ---
    float p[BS_BARS];
    bool hit[BS_BARS], delayed[BS_BARS];
    [unroll] for (int b3 = 0; b3 < BS_BARS; b3++) {
      p[b3] = base_p[b3] * G[b3].co;
      hit[b3] = G[b3].u_hit < p[b3];
      delayed[b3] = G[b3].u_mode < delay_share;
    }
    bool pass_anyway[BS_BARS];
    [unroll] for (int b4 = 0; b4 < BS_BARS; b4++) {
      // The survivor: among the bars whose events began within `window` of
      // this one's, if every one is hit, the one closest to missing passes.
      bool all_hit = true;
      int best = b4;
      float best_margin = G[b4].u_hit - p[b4];
      [unroll] for (int o2 = 0; o2 < BS_BARS; o2++) {
        if (abs(G[o2].t_event - G[b4].t_event) > window + 1e-4) continue;
        if (!hit[o2]) all_hit = false;
        float m = G[o2].u_hit - p[o2];
        if (m > best_margin || (m == best_margin && o2 < best)) { best_margin = m; best = o2; }
      }
      pass_anyway[b4] = all_hit && best == b4;
    }

    // --- 5–6. Each bar's band ---
    [unroll] for (int b5 = 0; b5 < BS_BARS; b5++) {
      GroupState g = G[b5];
      float el = now - g.t_event;
      bool is_hit = hit[b5] && !pass_anyway[b5];
      float gain = 0.0, lag = 0.0;
      if (is_hit && !delayed[b5]) {
        float rel = max(el - choke_length, 0.0);
        gain = saturate(choke_depth) * g.amp * exp(-rel / max(choke_release, 1e-3));
      } else if (is_hit) {
        // Staggered one step apart, in the event's shuffled order.
        int k = 1;
        [unroll] for (int o3 = 0; o3 < BS_BARS; o3++) {
          if (o3 == b5 || abs(G[o3].t_event - g.t_event) > window + 1e-4) continue;
          if (hit[o3] && !pass_anyway[o3] && delayed[o3] && G[o3].u_order < g.u_order) k++;
        }
        float full = float(k) * delay_step * max(1.0 + delay_jitter * (g.u_jit - 0.5), 0.05);
        float back = max(el - full - delay_hold, 0.0);
        lag = full * exp(-back / max(return_time, 1e-3));
        if (lag < 0.004) lag = 0.0;
      }

      for (int r = r0; r < r1; r++) {
        int ci = r * BS_BARS + b5;
        float3 lv = float3(stats[ci].r, stats[ci].g, stats[ci].b);
        float3 o = lv;
        if (gain > 1e-3) {
          CellState cs = cells[ci];
          float3 snap = float3(cs.snap_r, cs.snap_g, cs.snap_b);
          o = lv - gain * max(lv - snap, 0.0);
        } else if (lag > 0.0) {
          o = readHistory(ci, now - lag);
        }
        proc[ci] = float4(o, 1.0);
      }
      groups[gbase + b5] = g;
    }
  }

  // Last frame's live colour, for the next event's snapshot.
  if (step) {
    [loop] for (int c2 = 0; c2 < nrows * BS_BARS; c2++) {
      CellState cs = cells[c2];
      cs.prev_r = stats[c2].r; cs.prev_g = stats[c2].g; cs.prev_b = stats[c2].b;
      cells[c2] = cs;
    }
  }
}
