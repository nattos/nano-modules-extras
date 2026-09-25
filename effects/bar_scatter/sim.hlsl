// filter.light.bar_scatter — Pass 2: the decisions (GPU-resident state).
//
// Single-threaded (gid 0 only), like bounce_resonator_fx's sim: the native
// Metal backend hardcodes 8×8 threadgroups, and the work is tiny.
//
// Per frame:
//   1. CELL ONSET. Each cell's `ref` is a floor follower of its luma — it
//      drops to the luma at once and climbs back over RISE_TIME — so the rise
//      above it is the size of a sudden jump and ~0 for a slow fade-in. Reset
//      zeroes it, so an image arriving is itself an onset.
//   2. GROUP ONSET. Each bar is cut into `zones` bands; a band's onset is the
//      MEAN rise over its cells, so a whole band flashing reads fully and one
//      cell of ten reads a tenth. It fires above `sensitivity` and re-arms
//      below half of it.
//   3. CO-TRIGGER. A bar firing counts the bars in its zone row that fired
//      within `window` seconds, itself included. At `min_bars` or more, the
//      bars firing THIS frame roll; alone, it passes.
//   4. THE ROLL. Each is hit with p = strength·(1 − tilt·c), c = +1 for the
//      centre pair and −1 for the outer pair; a hit is a delay with
//      probability `delay_share`, else a choke. If every bar in the event
//      would end up hit, the one least likely to be hit passes instead: the
//      flash always lands somewhere. A bar already mid-effect keeps its fate
//      (a choke restarts its hold).
//   5. CHOKE removes only the NEW light — live minus its pre-flash colour —
//      held for `choke_length`, then released over `choke_release`.
//   6. DELAY reads the band from history `lag` seconds back. The delayed bars
//      of one event are staggered 1, 2, 3 steps (shuffled, jittered). Once
//      the delayed flash has played, it snaps back when the delayed and live
//      pictures agree; failing that, after `delay_hold` the lag decays over
//      `return_time` — the read head fast-forwards to live.
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

static float hash01(uint a, uint b, uint c) {
  uint h = a ^ (b * 0x9E3779B1u) ^ (c * 0x85EBCA77u);
  h ^= h >> 16; h *= 0x7FEB352Du; h ^= h >> 15; h *= 0x846CA68Bu; h ^= h >> 16;
  return float(h >> 8) * (1.0 / 16777216.0);
}

// The centre pair (bars 1, 2) is +1, the outer pair −1.
static float centreness(int bar) { return (bar == 1 || bar == 2) ? 1.0 : -1.0; }

// A cell's colour `lag` seconds ago. Slots are walked back from `head`, newest
// first; a slot never written means there was no picture yet (black). A lag
// past the whole ring clamps to its oldest slot.
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
      g.mode = BS_PASS; g.armed = 1;
      g.t_fire = -1e9; g.t_event = -1e9;
      g.lag = 0.0; g.gain = 0.0; g.pad0 = 0.0; g.pad1 = 0.0;
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
  [loop] for (int z = 0; z < nzones; z++) {
    int r0 = z * nrows / nzones, r1 = (z + 1) * nrows / nzones;
    int gbase = z * BS_BARS;
    GroupState G[BS_BARS];
    bool fired[BS_BARS];

    // --- 2. Group onset ---
    [unroll] for (int b = 0; b < BS_BARS; b++) {
      G[b] = groups[gbase + b];
      fired[b] = false;
      float onset = 0.0;
      for (int r = r0; r < r1; r++) onset += rise[r * BS_BARS + b];
      onset /= float(max(r1 - r0, 1));
      if (step) {
        if (G[b].armed != 0 && onset > sens) {
          fired[b] = true; G[b].armed = 0; G[b].t_fire = now;
        } else if (G[b].armed == 0 && onset < 0.5 * sens) {
          G[b].armed = 1;
        }
      }
    }

    // --- 3. Co-trigger ---
    bool any_fired = false;
    int in_window = 0;
    [unroll] for (int b2 = 0; b2 < BS_BARS; b2++) {
      any_fired = any_fired || fired[b2];
      if (now - G[b2].t_fire <= window + 1e-4) in_window++;
    }

    // --- 4. The roll ---
    if (any_fired && in_window >= min_bars) {
      uint ev = frame * 131u + uint(z);
      int fate[BS_BARS];
      bool fresh[BS_BARS];   // fired this frame, and free to take a new fate
      float p[BS_BARS];
      bool all_hit = true;
      [unroll] for (int b3 = 0; b3 < BS_BARS; b3++) {
        p[b3] = saturate(strength * (1.0 - tilt * centreness(b3)));
        fate[b3] = G[b3].mode;
        fresh[b3] = fired[b3] && G[b3].mode == BS_PASS;
        bool in_ev = now - G[b3].t_fire <= window + 1e-4;
        if (fresh[b3]) {
          if (hash01(seed, ev, uint(b3)) < p[b3])
            fate[b3] = (hash01(seed, ev, uint(b3) + 16u) < delay_share) ? BS_DELAY : BS_CHOKE;
        }
        if (in_ev && fate[b3] == BS_PASS) all_hit = false;
      }
      // The survivor: the least likely to be hit, a random tie-break.
      if (all_hit) {
        int best = -1; float best_key = 1e9;
        [unroll] for (int b4 = 0; b4 < BS_BARS; b4++) {
          if (!fresh[b4]) continue;
          float key = p[b4] + 1e-3 * hash01(seed, ev, uint(b4) + 32u);
          if (key < best_key) { best_key = key; best = b4; }
        }
        if (best >= 0) fate[best] = BS_PASS;
      }
      // Staggered lags for the delays new this frame, in shuffled order.
      float order[BS_BARS];
      [unroll] for (int b5 = 0; b5 < BS_BARS; b5++) order[b5] = hash01(seed, ev, uint(b5) + 48u);
      [unroll] for (int b6 = 0; b6 < BS_BARS; b6++) {
        if (!fired[b6]) continue;
        if (fate[b6] == BS_CHOKE) {
          if (fresh[b6]) {
            for (int r = r0; r < r1; r++) {
              int ci = r * BS_BARS + b6;
              CellState cs = cells[ci];
              cs.snap_r = cs.prev_r; cs.snap_g = cs.prev_g; cs.snap_b = cs.prev_b;
              cells[ci] = cs;
            }
          }
          G[b6].t_event = now;   // a choke restarts its hold
        } else if (fate[b6] == BS_DELAY && fresh[b6]) {
          int k = 1;
          [unroll] for (int o = 0; o < BS_BARS; o++)
            if (o != b6 && fresh[o] && fate[o] == BS_DELAY && order[o] < order[b6]) k++;
          float u = hash01(seed, ev, uint(b6) + 64u) - 0.5;
          G[b6].lag = float(k) * delay_step * max(1.0 + delay_jitter * u, 0.05);
          G[b6].t_event = now;
        }
        G[b6].mode = fate[b6];
      }
    }

    // --- 5–7. Advance each bar's fate and write the band ---
    [unroll] for (int b7 = 0; b7 < BS_BARS; b7++) {
      GroupState g = G[b7];
      float el = now - g.t_event;
      if (g.mode == BS_CHOKE) {
        float rel = max(el - choke_length, 0.0);
        g.gain = exp(-rel / max(choke_release, 1e-3));
        if (el > choke_length && g.gain < 0.01) { g.mode = BS_PASS; g.gain = 0.0; }
      } else {
        g.gain = 0.0;
      }

      if (g.mode == BS_DELAY && step && el > g.lag) {
        // Played: snap back once the two pictures agree.
        float diff = 0.0;
        for (int r = r0; r < r1; r++) {
          int ci = r * BS_BARS + b7;
          float3 dl = readHistory(ci, now - g.lag);
          float3 lv = float3(stats[ci].r, stats[ci].g, stats[ci].b);
          float3 ad = abs(dl - lv);
          diff = max(diff, max(ad.r, max(ad.g, ad.b)));
        }
        if (diff < 0.02) {
          g.mode = BS_PASS; g.lag = 0.0;
        } else if (el > g.lag + delay_hold) {
          g.lag *= exp(-dt / max(return_time, 1e-3));
          if (g.lag < 0.004) { g.mode = BS_PASS; g.lag = 0.0; }
        }
      }

      for (int r = r0; r < r1; r++) {
        int ci = r * BS_BARS + b7;
        float3 lv = float3(stats[ci].r, stats[ci].g, stats[ci].b);
        float3 o = lv;
        if (g.mode == BS_CHOKE) {
          CellState cs = cells[ci];
          float3 snap = float3(cs.snap_r, cs.snap_g, cs.snap_b);
          o = lv - saturate(choke_depth) * g.gain * max(lv - snap, 0.0);
        } else if (g.mode == BS_DELAY) {
          o = readHistory(ci, now - g.lag);
        }
        proc[ci] = float4(o, 1.0);
      }
      groups[gbase + b7] = g;
    }
  }

  // Last frame's live colour, for the next choke's snapshot.
  if (step) {
    [loop] for (int c2 = 0; c2 < nrows * BS_BARS; c2++) {
      CellState cs = cells[c2];
      cs.prev_r = stats[c2].r; cs.prev_g = stats[c2].g; cs.prev_b = stats[c2].b;
      cells[c2] = cs;
    }
  }
}
