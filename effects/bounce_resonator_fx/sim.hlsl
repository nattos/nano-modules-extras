// filter.light.bounce_resonator — Pass 2: the resonator (GPU-resident state).
//
// Single-threaded (gid 0 only): the native Metal backend hardcodes 8×8
// threadgroups, which breaks single-workgroup parallel shaders, and the
// work is tiny (≤ 64 cells, ≤ 1 hop per frame).
//
// Per frame:
//   0. DECAY, in real time and independent of the hop rate: every cell's
//      energy falls by exp(-decay_k·dt) (decay_k = ln(100)/tail seconds, so
//      the tail is the time to fall to 1%; 0 holds forever). Decay Shaping
//      bends it by where the grid is in its decay, as the source effect's
//      per-hop feedback does.
//   1. HOP (at most one — the rate caps at the frame rate). Pure
//      redistribution: the matrix columns sum to 1. Every row is
//      its own copy of the 4-bar network, and every row steps through the
//      SAME cycling matrix at the same hop, so the impulse-response pattern
//      travels across the whole grid at once while each cell keeps its own
//      energy — the differences between cells (the image) ring on in the
//      tail. Hue rides along exactly as in source.light.bounce_resonator.
//   2. EXCITE, continuously, from the input:
//      grains — each cell fires a random kick of grain_size with a Poisson
//               rate grain_rate·luma, so brightness becomes sparkle density;
//      edges  — a cell's luma RISE since last frame kicks edge_excite·rise.
//      The kick carries the cell's colour, blended toward the band colour.
//   3. A/C — each cell's running mean of v is subtracted (ac_amount), so
//      steady energy cancels and what shows is the swing: above the mean
//      lights, below it chokes the image toward black.
// dt = 0 (a re-render with no tick) steps nothing and re-derives `swing`.

#include "nano_color.hlsl"
#include "common.hlsl"

static const float TAU = 6.28318530717958648;

StructuredBuffer<CellStat>     stats : register(t0);
RWStructuredBuffer<CellState>  cells : register(u1);
StructuredBuffer<float>        mats  : register(t2);   // pattern_count × 48 floats
RWStructuredBuffer<NetState>   net   : register(u3);

cbuffer SimU : register(b4) {
  float decay_k; float decay_shaping; float hue_converge; float home_hue;
  int   pattern_count; int hop_idx; int do_hop; int rows;
  float grain_rate; float grain_size; float edge_excite; float input_color;
  float band_hue; float band_sat; float ac_amount; float ac_time;
  float dt; uint frame; uint seed; int do_reset;
};

static float wrapT(float h) { return frac(h + 1.0); }   // → [0,1)

static float hash01(uint a, uint b, uint c) {
  uint h = a ^ (b * 0x9E3779B1u) ^ (c * 0x85EBCA77u);
  h ^= h >> 16; h *= 0x7FEB352Du; h ^= h >> 15; h *= 0x846CA68Bu; h ^= h >> 16;
  return float(h >> 8) * (1.0 / 16777216.0);
}

// Colour mixing treats a cell as a hue vector of length value·saturation:
// agreeing hues keep saturation, opposing hues cancel toward grey.
void injectCell(inout CellState c, float amt, float hue, float sat) {
  if (amt <= 0.0) return;
  float a0 = TAU * c.h, a1 = TAU * hue;
  float w0 = c.v * c.s, w1 = amt * sat;
  float cx = w0 * cos(a0) + w1 * cos(a1);
  float cy = w0 * sin(a0) + w1 * sin(a1);
  c.v += amt;
  float rl = sqrt(cx * cx + cy * cy);
  if (rl > 1e-6) c.h = wrapT(atan2(cy, cx) / TAU);
  c.s = (c.v > 1e-6) ? saturate(rl / c.v) : sat;
}

[numthreads(64, 1, 1)]
void main(uint3 gid : SV_DispatchThreadID) {
  if (gid.x != 0u || gid.y != 0u || gid.z != 0u) return;

  int nrows = clamp(rows, 1, BR_MAX_ROWS);
  int ncells = nrows * BR_BARS;
  NetState ns = net[0];

  if (do_reset != 0) {
    ns.env = 0.0;
    [loop] for (int i = 0; i < BR_MAX_CELLS; i++) {
      CellState z;
      z.v = 0.0; z.h = band_hue; z.s = band_sat; z.mean = 0.0;
      z.luma_prev = 0.0;   // an image arriving is itself an edge
      z.swing = 0.0; z.pad0 = 0.0; z.pad1 = 0.0;
      cells[i] = z;
    }
  }

  // --- 0. Decay phase + shaped decay ---
  // Where the grid is in its decay (1 = just peaked, 0 = faded), from a
  // peak-follower on its total energy (effect_diffusion_network.h's proxy,
  // its per-hop 0.98 release restated per second at 60 Hz).
  float total = 0.0;
  [loop] for (int i = 0; i < ncells; i++) total += cells[i].v;
  if (dt > 0.0) ns.env = (total > ns.env) ? total : ns.env * exp(-1.2 * dt);
  float phase = (ns.env > 1e-6) ? min(total / ns.env, 1.0) : 0.0;
  // +shaping: slow while bright, fast at the tail (sustain then snap);
  // -shaping: fast while bright, slow at the tail (pluck then linger).
  float k = decay_k * max(1.0 + decay_shaping * 0.80 * (1.0 - 2.0 * phase), 0.0);
  float keep = (dt > 0.0) ? exp(-k * dt) : 1.0;

  // --- 1. Hop ---
  if (do_hop != 0) {
    int base = (max(hop_idx, 0) % max(pattern_count, 1)) * 48;
    float conv = 0.0;
    if (hue_converge > 0.0) { float t = 1.0 - phase; conv = min(hue_converge * t * t, 1.0); }

    [loop] for (int r = 0; r < nrows; r++) {
      CellState c[4];
      float cj[4], sj[4];
      [unroll] for (int j = 0; j < 4; j++) {
        c[j] = cells[r * BR_BARS + j];
        float a = TAU * c[j].h;
        cj[j] = c[j].s * cos(a); sj[j] = c[j].s * sin(a);
      }
      [unroll] for (int i2 = 0; i2 < 4; i2++) {
        float a = 0.0, hx = 0.0, hy = 0.0;
        [unroll] for (int jj = 0; jj < 4; jj++) {
          float w  = mats[base + i2 * 4 + jj] * c[jj].v;
          a += w;
          float cd = mats[base + 16 + i2 * 4 + jj];
          float sd = mats[base + 32 + i2 * 4 + jj];
          hx += w * (cj[jj] * cd - sj[jj] * sd);
          hy += w * (sj[jj] * cd + cj[jj] * sd);
        }
        CellState o = cells[r * BR_BARS + i2];
        o.v = clamp(a, 0.0, 1e6);
        float rl = sqrt(hx * hx + hy * hy);
        if (rl > 1e-6) o.h = wrapT(atan2(hy, hx) / TAU);
        if (a > 1e-6) o.s = saturate(rl / a);
        if (conv > 0.0) {
          float delta = frac(home_hue - o.h + 1.5) - 0.5;
          o.h = wrapT(o.h + delta * conv);
        }
        cells[r * BR_BARS + i2] = o;
      }
    }
  }

  // --- 2. Excite, 3. A/C ---
  float cstr = saturate(input_color);
  float follow = (dt > 0.0) ? 1.0 - exp(-dt / max(ac_time, 1e-3)) : 0.0;
  [loop] for (int k = 0; k < ncells; k++) {
    CellState c = cells[k];
    CellStat st = stats[k];

    if (dt > 0.0) {
      c.v *= keep;

      // The injected colour: the cell's own, pulled toward the band colour.
      float3 hsv = nano_rgb_to_hsv(float3(st.r, st.g, st.b));
      float dh = frac(hsv.x - band_hue + 1.5) - 0.5;   // shortest-path hue lerp
      float hue = wrapT(band_hue + dh * cstr);
      float sat = lerp(band_sat, hsv.y, cstr);

      float p = 1.0 - exp(-max(grain_rate, 0.0) * max(st.luma, 0.0) * dt);
      if (hash01(seed, frame, uint(k)) < p) injectCell(c, grain_size, hue, sat);

      float rise = max(st.luma - c.luma_prev, 0.0);
      injectCell(c, edge_excite * rise, hue, sat);
      c.luma_prev = st.luma;
      c.v = min(c.v, 1e6);

      c.mean += (c.v - c.mean) * follow;
    }
    c.swing = c.v - saturate(ac_amount) * c.mean;
    cells[k] = c;
  }

  net[0] = ns;
}
