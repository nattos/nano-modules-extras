// source.die_fold — the whole renderer, as one compute pass.
//
// A port of the die-fold prototype's `web/src/gpu/die.wgsl`, which is itself
// the Python (`diefold/`) at measured pixel parity. Stage for stage, in the
// prototype's order: hashes -> world -> descend -> leafField -> wires ->
// straps -> slab -> frame. Fixed loops, no data-dependent branching, no
// lookups, no textures, and no gradient field: every boundary is a true
// Euclidean signed distance in track units, so the conversion to pixels is one
// per-frame scalar `kpx` and anti-aliasing is one `hard(d * kpx)`.
//
// THE RULES, each already paid for once in the prototype:
//
//   1. Compose distances with `min`/`max` ONLY. `boxSd` is POSITIVE INSIDE, so
//      union is `max`, intersection is `min`, complement is unary `-`. A smooth
//      union breaks |grad d| == 1 and therefore anti-aliasing everywhere.
//   2. Right shifts are on uint.
//   3. `%` and `fmod` are TRUNCATED in HLSL; torch.remainder is FLOORED. Use
//      `fmodf`. There is no bare `%` or `fmod` in this file.
//   4. `clamp` with crossed bounds is undefined; `tclamp` is torch's.
//   5. `round` is half-to-even in torch and WGSL; `rne` spells that out rather
//      than trusting HLSL `round` through SPIR-V to MSL and WGSL.
//   6. NO `?:`, `&&` or `||` on the per-pixel path. HLSL 2021 SHORT-CIRCUITS
//      them on scalars, so DXC emits a branch for each: written that way this
//      file compiled to ~200 divergent branches and ran at half the speed of
//      the prototype's WGSL `select`s. Use `select`, `and`, `or`.
//
// RESOLUTION. `kpx` (real pixels per track) does the anti-aliasing. Every
// CONTENT decision -- which levels of the hierarchy resolve, how many tracks a
// corridor shows, how far erosion reaches -- reads `kq`, the pixels per track
// at the composition's REFERENCE height (host::pxScale), so a capped preview
// draws the same die as a full-resolution export. With no reference they are
// equal and this is the prototype exactly.

#include "layout.hlsl"

// The packed parameter block (layout.hlsl), a UNIFORM buffer: every index is a
// compile-time P_* constant (KW is read only in unrolled loops), so the backend
// can keep it in constant memory instead of re-reading device memory per use.
cbuffer Params : register(b0) { float4 prm[PRM_VEC4]; };
RWTexture2D<float4> outTex : register(u1);

float P(uint i) { return prm[i >> 2u][i & 3u]; }

// =========================================================================
// hashes.py
// =========================================================================

static const uint M1 = 0x7FEB352Du;   // lowbias32
static const uint M2 = 0x846CA68Bu;
static const uint MK = 0x9E3779B1u;
static const uint H0 = 0x2545F491u;

uint mix1(uint h) {
  h = h ^ (h >> 16u);
  h = h * M1;
  h = h ^ (h >> 15u);
  h = h * M2;
  h = h ^ (h >> 16u);
  return h;
}

// `hash_u32` is variadic in Python; unrolled by arity. One `mix1` round per
// key, in order -- the order is part of the value.
uint h1(uint a) { return mix1(H0 ^ (a * MK)); }
uint h2(uint a, uint b) { return mix1(h1(a) ^ (b * MK)); }
uint h3(uint a, uint b, uint c) { return mix1(h2(a, b) ^ (c * MK)); }
uint h4(uint a, uint b, uint c, uint d) { return mix1(h3(a, b, c) ^ (d * MK)); }
uint h5(uint a, uint b, uint c, uint d, uint e) { return mix1(h4(a, b, c, d) ^ (e * MK)); }
uint h6(uint a, uint b, uint c, uint d, uint e, uint f) {
  return mix1(h5(a, b, c, d, e) ^ (f * MK));
}

// The TOP 24 bits.
float hf(uint h) { return float(h >> 8u) * (1.0 / 16777216.0); }

float hf1(uint a) { return hf(h1(a)); }
float hf2(uint a, uint b) { return hf(h2(a, b)); }
float hf3(uint a, uint b, uint c) { return hf(h3(a, b, c)); }
float hf4(uint a, uint b, uint c, uint d) { return hf(h4(a, b, c, d)); }
float hf5(uint a, uint b, uint c, uint d, uint e) { return hf(h5(a, b, c, d, e)); }

/** `hash_int`: uniform in [0, n). */
int hi_of(float u, int n) { return clamp(int(u * float(n)), 0, n - 1); }

/** int -> the uint the hash sees. Two's complement, so a reinterpret. */
uint k(int x) { return asuint(x); }

// =========================================================================
// the semantic traps
// =========================================================================

/** `torch.remainder` -- FLOORED modulo, the sign of the DIVISOR. */
float fmodf(float a, float b) { return a - b * floor(a / b); }

/** `torch.clamp` -- `min(max(x, lo), hi)`, defined when lo > hi. */
float tclamp(float x, float lo, float hi) { return min(max(x, lo), hi); }

/** Round half to even (torch.round, WGSL round). */
float rne(float x) {
  float f = floor(x);
  float d = x - f;
  float up = f + 1.0;
  float even = select(fmodf(f, 2.0) < 0.5, f, up);
  return select(d > 0.5, up, select(d < 0.5, f, even));
}

// =========================================================================
// coordinates -- hashes.cover_square, world_from_pixel; floorplan.view_tracks
// =========================================================================

/** `floorplan.view_tracks`: half-width of the frame in tracks. */
float viewTracks() { return P(P_SPAN_CELLS) * N_TRACK_F / exp2(P(P_ZOOM)); }

/** `floorplan.pixels_per_track`, off the render target's height. */
float pixelsPerTrack() { return P(P_SURF_H) / (2.0 * viewTracks()); }

/** Screen coordinates on the cover square (pixel centers). */
float2 coverSquare(float2 pos) {
  return float2((2.0 * pos.x - P(P_SURF_W)) / P(P_SURF_H),
                (2.0 * pos.y - P(P_SURF_H)) / P(P_SURF_H));
}

/** `floorplan.pixel_world`: screen -> TRACKS, relative to `cam_cell`. */
float2 pixelWorld(float2 s) {
  float span = viewTracks() / N_TRACK_F;              // in super-cells
  return (float2(P(P_CAM_FRAC_X), P(P_CAM_FRAC_Y)) + s * span) * N_TRACK_F;
}

/** `cam_cell * N_TRACK`, in int. Allowed to wrap: its only consumer is a hash. */
int2 camOrigin() {
  return int2(int(P(P_CAM_CELL_X)) * N_TRACK, int(P(P_CAM_CELL_Y)) * N_TRACK);
}

uint dieSeed() { return k(int(P(P_SEED))); }

// =========================================================================
// floorplan.py -- the hash-seeded guillotine hierarchy, per pixel
// =========================================================================

struct Rect {
  float px; float py;
  float x0; float y0; float x1; float y1;
  float w_xlo; float w_ylo; float w_xhi; float w_yhi;
  uint key;
  uint v_key; float v_pos; float v_w; float v_a0; float v_a1;
  uint h_key; float h_pos; float h_w; float h_a0; float h_a1;
};

/** One corridor, in its own (perp, along) frame. */
struct Chan {
  uint key; float perp; float along; float len; float w; bool vertical;
};

float rectSd(Rect R) {
  return min(min(R.px - R.x0, R.x1 - R.px), min(R.py - R.y0, R.y1 - R.py));
}

/** ...to the block body, each side pulled in by ITS OWN channel's half-width. */
float rectSdInset(Rect R) {
  return min(min(R.px - (R.x0 + R.w_xlo), (R.x1 - R.w_xhi) - R.px),
             min(R.py - (R.y0 + R.w_ylo), (R.y1 - R.w_yhi) - R.py));
}

float2 rectSize(Rect R) { return float2(R.x1 - R.x0, R.y1 - R.y0); }

Chan rectChannel(Rect R, bool vertical) {
  Chan c;
  c.vertical = vertical;
  c.key  = select(vertical, R.v_key, R.h_key);
  c.perp = select(vertical, R.px - R.v_pos, R.py - R.h_pos);
  c.w    = select(vertical, R.v_w, R.h_w);
  float a0 = select(vertical, R.v_a0, R.h_a0);
  float a1 = select(vertical, R.v_a1, R.h_a1);
  c.along = select(vertical, R.py, R.px) - a0;
  c.len   = a1 - a0;
  return c;
}

/** Distance to a channel's slab. Positive OUTSIDE (Chebyshev, nearest only). */
float segDist(float perp, float along, float a0, float a1) {
  return max(abs(perp), max(a0 - along, along - a1));
}

/** `node_key` -- the ONE place a node is identified, from its geometry alone. */
uint nodeKey(int2 o, float x0, float y0, float x1, float y1, uint sd) {
  return h5(k(o.x + int(x0)), k(o.y + int(y0)), k(o.x + int(x1)), k(o.y + int(y1)), sd);
}

Rect descend(float px, float py) {
  float nt = N_TRACK_F;
  // Rounded to whole tracks: a clamp bound on the cut offset.
  float leaf_min = rne(P(P_LEAF_MIN));
  float cut_bias = P(P_CUT_BIAS);
  float long_bias = P(P_LONG_BIAS);
  float stop_p = P(P_STOP_P);
  int2 o = camOrigin();
  uint sd = dieSeed();

  Rect R = (Rect)0;
  R.px = px;
  R.py = py;
  R.x0 = floor(px / nt) * nt;
  R.y0 = floor(py / nt) * nt;
  R.x1 = R.x0 + nt;
  R.y1 = R.y0 + nt;

  float cw0 = P(P_CHAN_FRAC) * nt;
  // A cut narrower than this has no corridor at all -- at the REFERENCE framing.
  float cw_min = (FADE_FULL_PX * viewTracks()) / RES_REF_F;
  R.w_xlo = cw0; R.w_ylo = cw0; R.w_xhi = cw0; R.w_yhi = cw0;

  // The super-cell boundary IS a channel, the widest one.
  R.v_pos = select(abs(px - R.x0) <= abs(R.x1 - px), R.x0, R.x1);
  R.h_pos = select(abs(py - R.y0) <= abs(R.y1 - py), R.y0, R.y1);
  R.v_a0 = R.y0; R.v_a1 = R.y1;
  R.h_a0 = R.x0; R.h_a1 = R.x1;
  R.v_w = cw0; R.h_w = cw0;
  // Keyed by the EDGE, not by the cell.
  R.v_key = h5(K_CUT, 101u, k(o.x + int(R.v_pos)), k(o.y + int(R.y0)), sd);
  R.h_key = h5(K_CUT, 202u, k(o.y + int(R.h_pos)), k(o.x + int(R.x0)), sd);
  float v_dist = segDist(px - R.v_pos, py, R.v_a0, R.v_a1);
  float h_dist = segDist(py - R.h_pos, px, R.h_a0, R.h_a1);

  // `alive` is a MASK, not a break: a fixed nine levels of straight selects.
  bool alive = true;
  [loop]
  for (int d = 0; d < D_MAX; d++) {
    uint key = nodeKey(o, R.x0, R.y0, R.x1, R.y1, sd);
    float wt = R.x1 - R.x0;
    float ht = R.y1 - R.y0;

    bool long_x = wt >= ht;
    bool take_long = hf2(key, K_AXIS) < long_bias;
    bool axis_x = select(take_long, long_x, !long_x);

    float n = select(axis_x, wt, ht);
    bool can = n >= 2.0 * leaf_min;
    bool early = hf2(key, K_SPLIT) < stop_p;
    bool do_cut = and(and(alive, can), !early);

    float f = 0.5 + (hf2(key, K_POS) - 0.5) * (2.0 * cut_bias);
    float off = tclamp(rne(f * n), leaf_min, n - leaf_min);
    float cut = select(axis_x, R.x0, R.y0) + off;

    bool hi = select(axis_x, px, py) > cut;
    float cw = cw0 * pow(P(P_CHAN_FALL), float(d + 1));
    bool has_chan = cw >= cw_min;

    uint cut_key = h3(key, K_CUT, k(d));
    float ca0 = select(axis_x, R.y0, R.x0);
    float ca1 = select(axis_x, R.y1, R.x1);
    float cd = segDist(select(axis_x, px - cut, py - cut), select(axis_x, py, px), ca0, ca1);

    bool tv = and(and(do_cut, has_chan), and(axis_x, cd < v_dist));
    v_dist  = select(tv, cd, v_dist);
    R.v_key = select(tv, cut_key, R.v_key);
    R.v_pos = select(tv, cut, R.v_pos);
    R.v_w   = select(tv, cw, R.v_w);
    R.v_a0  = select(tv, ca0, R.v_a0);
    R.v_a1  = select(tv, ca1, R.v_a1);
    bool th = and(and(do_cut, has_chan), and(!axis_x, cd < h_dist));
    h_dist  = select(th, cd, h_dist);
    R.h_key = select(th, cut_key, R.h_key);
    R.h_pos = select(th, cut, R.h_pos);
    R.h_w   = select(th, cw, R.h_w);
    R.h_a0  = select(th, ca0, R.h_a0);
    R.h_a1  = select(th, ca1, R.h_a1);

    bool dx = and(do_cut, axis_x);
    bool dy = and(do_cut, !axis_x);
    float cwz = select(has_chan, cw, 0.0);
    R.x1 = select(and(dx, !hi), cut, R.x1);
    R.x0 = select(and(dx, hi), cut, R.x0);
    R.y1 = select(and(dy, !hi), cut, R.y1);
    R.y0 = select(and(dy, hi), cut, R.y0);
    R.w_xhi = select(and(dx, !hi), cwz, R.w_xhi);
    R.w_xlo = select(and(dx, hi), cwz, R.w_xlo);
    R.w_yhi = select(and(dy, !hi), cwz, R.w_yhi);
    R.w_ylo = select(and(dy, hi), cwz, R.w_ylo);

    alive = and(alive, do_cut);
  }
  R.key = nodeKey(o, R.x0, R.y0, R.x1, R.y1, sd);
  return R;
}

// =========================================================================
// render.py -- the anti-aliasing primitives
// =========================================================================

/** `step(0, d)` with a shoulder `blur` pixels wide. `dp` is in FINAL PIXELS. */
float hard(float dp, float blur) {
  float t = clamp(dp / (AA_PX * blur) + 0.5, 0.0, 1.0);
  return t * t * (3.0 - 2.0 * t);
}

/** A SOLID stroke of half-width `w` final pixels about a zero set. */
float strokeOf(float dp, float w, float blur) {
  float t = clamp((abs(dp) - w) / (AA_PX * blur) + 0.5, 0.0, 1.0);
  return 1.0 - t * t * (3.0 - 2.0 * t);
}

// =========================================================================
// stack.py -- the layers, and where the camera is among them
// =========================================================================

/** The camera's depth, in layers. Advances by exactly N_LAYERS over the loop. */
float zCam(float t) { return P(P_Z0) + fmodf(t * N_LAYERS_F, N_LAYERS_F); }

float slabLi(float t, int s) { return floor(zCam(t)) + float(s - 1); }
float slabDz(float t, int s) { return slabLi(t, s) - zCam(t); }

/** Where this block's own layer boundaries sit, as a signed fraction of a layer. */
float blockPhase(uint key) { return hf3(key, K_LAYER, 5u) - 0.5; }

/** Value noise along z, so proportions DEFORM down the stack. */
float zLerp(uint key, uint salt, float zf) {
  float l0 = floor(zf);
  float f = zf - l0;
  float a = hf3(key, salt, k(int(l0)));
  float b = hf3(key, salt, k(int(l0 + 1.0)));
  return a + (b - a) * (f * f * (3.0 - 2.0 * f));
}

/** The content seed for a layer. PERIODIC in N_LAYERS, which closes the loop. */
uint contentKey(float li) {
  int per = int(fmodf(li, N_LAYERS_F));
  bool use_layer = hf2(h3(k(per), K_LAYER, dieSeed()), 1u) < P(P_VARIETY);
  return h3(k(select(use_layer, per + 1, 0)), K_LAYER, 2u);
}

/** Metal alternates H and V down the stack. */
bool routesVertical(float li) { return fmodf(floor(li), 2.0) < 0.5; }

// =========================================================================
// leaf.py -- block interiors
// =========================================================================

/** Signed distance to an axis-aligned box, positive INSIDE. Exact. */
float boxSd(float qx, float qy, float hx, float hy) {
  float ax = abs(qx) - hx;
  float ay = abs(qy) - hy;
  float o = sqrt(max(ax, 0.0) * max(ax, 0.0) + max(ay, 0.0) * max(ay, 0.0));
  return -(o + min(max(ax, ay), 0.0));
}

/** `ox + v.to(int32)` -- an absolute track coordinate, as a hash key. INTEGER. */
uint akey(int o, float v) { return k(o + int(v)); }

float motifRing(float cu, float cv, float hx, float hy, float t) {
  return min(boxSd(cu, cv, hx, hy), -boxSd(cu, cv, hx - t, hy - t));
}

/** The motif vocabulary, in a footprint normalized to [-0.5, 0.5]^2. */
float motifSd(float mid, float cu, float cv) {
  switch (int(mid)) {
    case 1:   // a ring
      return motifRing(cu, cv, 0.38, 0.38, 0.11);
    case 2:   // an L
      return max(boxSd(cu + 0.16, cv, 0.11, 0.36), boxSd(cu, cv - 0.25, 0.36, 0.11));
    case 3:   // a T
      return max(boxSd(cu, cv - 0.22, 0.38, 0.10), boxSd(cu, cv + 0.11, 0.10, 0.29));
    case 4:   // a quad of contacts
      return max(max(boxSd(cu - 0.18, cv - 0.18, 0.11, 0.11),
                     boxSd(cu + 0.18, cv - 0.18, 0.11, 0.11)),
                 max(boxSd(cu - 0.18, cv + 0.18, 0.11, 0.11),
                     boxSd(cu + 0.18, cv + 0.18, 0.11, 0.11)));
    case 5:   // a bar with a bite out of one end
      return min(boxSd(cu, cv, 0.42, 0.15), -boxSd(cu - 0.30, cv, 0.11, 0.075));
    case 6:   // a cross
      return max(boxSd(cu, cv, 0.42, 0.09), boxSd(cu, cv, 0.09, 0.42));
    case 7:   // a square inside a ring
      return max(motifRing(cu, cv, 0.42, 0.42, 0.10), boxSd(cu, cv, 0.15, 0.15));
    default:  // 0: a pair of legs under a tie
      return max(max(boxSd(cu + 0.21, cv - 0.06, 0.085, 0.30),
                     boxSd(cu - 0.21, cv - 0.06, 0.085, 0.30)),
                 boxSd(cu, cv + 0.30, 0.33, 0.085));
  }
}

/** Per-leaf cell pitch in tracks. Powers of two. */
float2 leafPitches(uint key, float kq, float bh) {
  float jit = float(hi_of(hf2(key, K_PITCH), 3)) - 1.0;          // -1, 0, +1
  float ex = rne(P(P_CELL_LOG2) + jit);
  float pw = exp2(clamp(ex, PITCH_LO_F, PITCH_HI_F));
  // CELLS STAY RESOLVABLE.
  float floor_tracks = (FADE_FULL_PX * 0.62) / kq;
  pw = max(pw, exp2(max(ceil(log2(floor_tracks)), 1.0)));
  float grow = clamp(floor(P(P_ASPECT) * 3.0 + hf3(key, K_PITCH, 7u) * 0.9), 0.0, 3.0);
  float ph = min(pw * exp2(grow), exp2(floor(log2(max(bh, 2.0) * 0.34))));
  return float2(pw, max(ph, pw));
}

/** Pick this block's fabric from the die's mix. */
float leafKind(uint key) {
  float tot = 0.0;
  [unroll]
  for (int q = 0; q < N_KIND; q++) { tot = tot + max(P(KW[q]), 1e-4); }
  float u = hf2(key, K_TYPE) * tot;
  float kd = 0.0;
  float acc = 0.0;
  [unroll]
  for (int q2 = 0; q2 < N_KIND - 1; q2++) {
    acc = acc + max(P(KW[q2]), 1e-4);
    kd = kd + select(u > acc, 1.0, 0.0);
  }
  return kd;
}

/** The run-start bit hashes the cell's ABSOLUTE track coordinate. */
bool runStart(float c, uint row_key, float p_start, int oxi, float pw) {
  return hf3(row_key, K_FILL, akey(oxi, c * pw)) < p_start;
}

/** Merge cells along a row into runs, by a fixed backward+forward scan. */
float2 runSpan(float i, uint row_key, float p_start, int oxi, float pw) {
  float start = i;
  bool found = runStart(i, row_key, p_start, oxi, pw);
  [unroll]
  for (int q = 1; q < RUN_MAX; q++) {
    float cand = i - float(q);
    bool hit = and(!found, runStart(cand, row_key, p_start, oxi, pw));
    start = select(hit, cand, start);
    found = or(found, hit);
  }
  start = select(found, start, i - float(RUN_MAX - 1));
  float e = start + 1.0;
  bool done = false;
  [unroll]
  for (int q2 = 1; q2 < RUN_MAX; q2++) {
    float cand = start + float(q2);
    bool hit = and(!done, runStart(cand, row_key, p_start, oxi, pw));
    e = select(hit, cand, e);
    done = or(done, hit);
  }
  return float2(start, select(done, e, start + float(RUN_MAX)));
}

struct Leaf {
  float d_cell;
  float d_hatch;
  uint cell_key;
  float d_fine;
  float fine_px;
  float d_dev;
  float dev_px;
  float dev_mean;
  uint dev_key;
  float fu;          // where in its own device tile a pixel sits, in [-1,1]
  float fv;
  float cell_mean;   // analytic duty cycles
  float fine_mean;
  float hue_off;
  float pitch_px;
};

Leaf leafField(Rect R, float kq, uint layer_key, int oxi, int oyi, float zf) {
  float px = R.px;
  float py = R.py;
  Leaf L = (Leaf)0;

  // THE FABRIC DRIFTS DOWN THE STACK, a function of the LAYER, not of time.
  float mfill = zLerp(R.key, 91u, zf);
  float mdev = zLerp(R.key, 92u, zf);

  float2 sz = rectSize(R);
  float bw = sz.x;
  float bh = sz.y;
  float2 pp2 = leafPitches(R.key, kq, bh);
  float pw = pp2.x;
  float ph = pp2.y;
  float body = rectSdInset(R);
  float kind = leafKind(R.key);

  float i = floor(px / pw);
  float j = floor(py / ph);
  L.cell_key = h5(R.key, K_DEFECT, akey(oxi, i * pw), akey(oyi, j * ph), layer_key);

  float cx = (i + 0.5) * pw;
  float cy = (j + 0.5) * ph;
  float fill_z = clamp(P(P_FILL) * (0.78 + 0.44 * mfill), 0.20, 0.92);
  float hx = 0.5 * pw * fill_z;
  float hy = 0.5 * ph * fill_z;

  // --- ARRAY: a regular lattice, broken periodically by straps -------------
  float nb = max(rne(P(P_STRAP_EVERY)), 2.0);
  bool brk = or(fmodf(i, nb) < 0.5, fmodf(j, nb) < 0.5);
  float d_array = boxSd(px - cx, py - cy, hx, hy);
  d_array = select(brk, boxSd(px - cx, py - cy, hx * 0.35, hy * 0.35), d_array);

  // --- ROWS: run-length-merged cells ---------------------------------------
  uint row_key = h5(R.key, K_FILL, 3u, akey(oyi, j * ph), layer_key);
  float2 run = runSpan(i, row_key, clamp(1.0 / P(P_RUNLEN), 0.05, 1.0), oxi, pw);
  float d_rows = boxSd(px - (run.x + run.y) * 0.5 * pw, py - cy,
                       (run.y - run.x) * 0.5 * pw * fill_z, hy);

  // --- LOGIC: fine irregular speckle, at a third of the block pitch ---------
  float pl = max(pw / 3.0, exp2(max(ceil(log2(1.15 / kq)), 0.0)));
  float il = floor(px / pl);
  float jl = floor(py / pl);
  uint lgk = h6(R.key, K_FILL, 11u, akey(oxi, il * pl), akey(oyi, jl * pl), layer_key);
  bool on = hf2(lgk, 1u) < P(P_LOGIC_FILL);
  float lhx = pl * (0.14 + 0.34 * hf2(lgk, 2u));
  float lhy = pl * (0.14 + 0.34 * hf2(lgk, 3u));
  float d_logic = boxSd(px - (il + 0.5) * pl, py - (jl + 0.5) * pl, lhx, lhy);
  d_logic = select(on, d_logic, NEG);

  // --- DPATH: long directional stripes, segmented --------------------------
  bool along_x = bw >= bh;
  float perp = select(along_x, py, px);
  float alng = select(along_x, px, py);
  float pp = select(along_x, ph, pw);
  float kk = floor(perp / pp);
  uint sk = h5(R.key, K_FILL, 17u, akey(select(along_x, oyi, oxi), kk * pp), layer_key);
  float shw = pp * (0.16 + 0.30 * hf2(sk, 1u));
  float seg = max(pw * P(P_SEG_LEN), pw * 2.0);
  float si = floor(alng / seg);
  bool seg_live = hf2(h2(sk, k(int(si))), 2u) < 0.86;
  float d_perp = shw - abs(perp - (kk + 0.5) * pp);
  float d_alng = (0.5 - abs(alng / seg - si - 0.5)) * seg - shw * 0.6;
  float d_dpath = min(d_perp, d_alng);
  d_dpath = select(seg_live, d_dpath, NEG);

  // --- COMB: interdigitated fingers ----------------------------------------
  float blen = select(along_x, bw, bh);
  float cb0 = select(along_x, R.x0, R.y0);
  float cu_comb = alng - cb0;
  bool from_lo = fmodf(kk, 2.0) < 0.5;
  float clen = blen * (0.42 + 0.44 * hf2(sk, 5u));
  float d_span = select(from_lo, clen - cu_comb, cu_comb - (blen - clen));
  float d_comb = min(shw - abs(perp - (kk + 0.5) * pp), d_span);

  // --- MACRO: a banked body with a ring, a hatch, and INTERNAL DETAIL ------
  float mx = (R.x0 + R.x1) * 0.5;
  float my = (R.y0 + R.y1) * 0.5;
  float inset = min(bw, bh) * 0.16;
  float mring = boxSd(px - mx, py - my, bw * 0.5 - inset, bh * 0.5 - inset);
  float nbank = 2.0 + float(hi_of(hf3(R.key, K_TYPE, 21u), 4));
  float bkw = max(bw / nbank, pw * 4.0);
  float bkh = max(bh / nbank, ph * 4.0);
  float seam = min((0.5 - abs(px / bkw - floor(px / bkw) - 0.5)) * bkw,
                   (0.5 - abs(py / bkh - floor(py / bkh) - 0.5)) * bkh) - pw * 0.5;
  float inner = boxSd(px - cx, py - cy, hx * 0.92, hy * 0.92);
  float d_macro = min(mring, max(inner, seam));
  bool horiz = hf3(R.key, K_TYPE, 5u) < 0.5;
  float tq = select(horiz, py, px) / P(P_HATCH);
  float d_hatch = (P(P_HATCH) * 0.25) - abs(tq - floor(tq) - 0.5) * P(P_HATCH);

  // --- MOTIF: a small vocabulary, clustered by district --------------------
  float msz = max(pw * rne(P(P_MOTIF_SIZE)), pw * 2.0);
  msz = max(min(msz, min(bw, bh) / 3.0), pw);
  float mi = floor(px / msz);
  float mj = floor(py / msz);
  float dsz = msz * rne(P(P_MOTIF_DISTRICT));
  float di = floor(px / dsz);
  float dj = floor(py / dsz);
  // Keyed on ABSOLUTE position, not on the block.
  uint site_k = h5(K_TYPE, 31u, akey(oxi, mi * msz), akey(oyi, mj * msz), layer_key);
  uint dist_k = h5(K_TYPE, 37u, akey(oxi, di * dsz), akey(oyi, dj * dsz), layer_key);
  float mid = select(hf2(site_k, 5u) < P(P_MOTIF_CLUSTER), float(hi_of(hf2(dist_k, 1u), N_MOTIF)), float(hi_of(hf2(site_k, 2u), N_MOTIF)));
  float cu_m = px / msz - mi - 0.5;
  float cv_m = py / msz - mj - 0.5;
  float d_motif = motifSd(mid, cu_m, cv_m) * msz;

  // --- banks: a coarse internal division, on every fabric ------------------
  float bk = max(pw * rne(P(P_BANK_EVERY)), pw * 3.0);
  float bkx = (0.5 - abs(px / bk - floor(px / bk) - 0.5)) * bk;
  float bky = (0.5 - abs(py / bk - floor(py / bk) - 0.5)) * bk;
  float d_bank = min(bkx, bky) - pw * 0.42;
  d_bank = select(hf3(R.key, K_PITCH, 61u) < P(P_BANK_P), d_bank, 1e4);

  // --- select ---------------------------------------------------------------
  float d_cell = d_array;
  d_cell = select(kind == KIND_ROWS_F, d_rows, d_cell);
  d_cell = select(kind == KIND_LOGIC_F, d_logic, d_cell);
  d_cell = select(kind == KIND_DPATH_F, d_dpath, d_cell);
  d_cell = select(kind == KIND_COMB_F, d_comb, d_cell);
  d_cell = select(kind == KIND_MACRO_F, d_macro, d_cell);
  d_cell = select(kind == KIND_MOTIF_F, d_motif, d_cell);
  bool is_macro = kind == KIND_MACRO_F;
  d_cell = select(is_macro, d_cell, min(d_cell, d_bank));
  d_hatch = select(is_macro, d_hatch, NEG);

  // --- defects: the breaks that make a repeat worth looking at -------------
  bool lattice = kind < KIND_DPATH_F;
  bool broken = and(hf3(L.cell_key, K_DEFECT, 1u) < P(P_DEFECT), lattice);
  bool notched = and(broken, hf3(L.cell_key, K_DEFECT, 2u) < P(P_NOTCH_P));
  bool dropped = and(broken, !notched);
  float nq = float(hi_of(hf3(L.cell_key, K_DEFECT, 3u), 4));
  float nsx = select(fmodf(nq, 2.0) > 0.5, 1.0, -1.0);
  float nsy = select(nq > 1.5, 1.0, -1.0);
  float ns = min(hx, hy) * 0.85;
  float notch = boxSd(px - (cx + nsx * hx), py - (cy + nsy * hy), ns, ns);
  d_cell = select(notched, min(d_cell, -notch), d_cell);
  d_cell = select(dropped, NEG, d_cell);
  L.d_cell = d_cell;
  L.d_hatch = d_hatch;

  // Per-block hue: the fabric sets the family, a jitter separates neighbors.
  L.hue_off = P(P_HUE_BLOCK) * (kind / N_KIND_F + 0.35 * (hf3(R.key, K_TYPE, 9u) - 0.5));

  // --- the fringe: pin ticks just inside the block edge --------------------
  float dxl = px - (R.x0 + R.w_xlo);
  float dxh = (R.x1 - R.w_xhi) - px;
  float dyl = py - (R.y0 + R.w_ylo);
  float dyh = (R.y1 - R.w_yhi) - py;
  bool near_h = min(dyl, dyh) < min(dxl, dxh);
  float tpin = select(near_h, px, py);
  float comb = pw * 0.22 - abs(tpin / pw - floor(tpin / pw) - 0.5) * pw;
  float pdep = P(P_PIN_DEPTH) * pw;
  float d_pin = min(min(comb, body), pdep - body);
  d_pin = select(hf3(R.key, K_PITCH, 51u) < P(P_PIN_P), d_pin, NEG);

  // --- the level below the cell: contacts ----------------------------------
  float pf = max(pw / 4.0, 0.5);
  float cfu = px / pf - floor(px / pf) - 0.5;
  float cfv = py / pf - floor(py / pf) - 0.5;
  float d_fine = boxSd(cfu * pf, cfv * pf, pf * 0.17, pf * 0.17);
  d_fine = min(d_fine, d_cell);
  d_fine = select(hf3(L.cell_key, K_FILL, 41u) < P(P_FINE_P), d_fine, NEG);
  // Clipped to the block body here, not in the renderer.
  L.d_fine = min(max(d_fine, d_pin), body);
  L.fine_px = pf * kq;

  // --- each fabric's duty cycle --------------------------------------------
  float cell_mean = clamp(fill_z * fill_z, 0.0, 1.0);             // ARRAY / ROWS
  cell_mean = select(kind == KIND_LOGIC_F, clamp(P(P_LOGIC_FILL) * 0.38, 0.0, 1.0), cell_mean);
  cell_mean = select(kind == KIND_DPATH_F, 0.53, cell_mean);
  cell_mean = select(kind == KIND_COMB_F, 0.40, cell_mean);
  cell_mean = select(kind == KIND_MACRO_F, 0.62, cell_mean);
  cell_mean = select(kind == KIND_MOTIF_F, 0.26, cell_mean);
  cell_mean = cell_mean * (1.0 - 0.5 * P(P_DEFECT));
  L.cell_mean = cell_mean;
  L.fine_mean = clamp(0.34 * 0.34 * P(P_FINE_P) * cell_mean, 0.0, 1.0);

  // --- devices: the level that exists only as shading ----------------------
  float pdx = max(pw / 7.0, 2.0 / kq);
  pdx = min(pdx, pw / 3.0);
  bool dvert = hf3(R.key, K_FILL, 81u) < 0.5;
  uint rk = h3(L.cell_key, K_FILL, 82u);
  float grp = 2.0 + float(hi_of(hf2(rk, 2u), 4));
  float flen = 4.0 + float(hi_of(hf2(rk, 4u), 6));
  float wf = (0.26 + 0.20 * hf2(rk, 3u)) * (0.72 + 0.56 * mdev);
  float across = select(dvert, px, py);
  float dalong = select(dvert, py, px);
  float a0 = select(dvert, i * pw, j * ph);
  float kf = floor((across - a0) / pdx);
  bool fing_live = fmodf(kf, grp) < (grp - 1.0);
  float d_across = pdx * wf - abs(across - a0 - (kf + 0.5) * pdx);
  float pu = pdx * flen;
  float db0 = select(dvert, j * ph, i * pw);
  float ju = floor((dalong - db0) / pu);
  float cu_d = (dalong - db0) / pu - ju - 0.5;
  bool alt = fmodf(kf, 2.0) < 0.5;
  float plo = select(alt, 0.03, 0.17);
  float phi = select(alt, 0.17, 0.03);
  float d_along = min(0.5 - plo - max(cu_d, 0.0), 0.5 - phi - max(-cu_d, 0.0)) * pu;
  float d_dev = min(d_across, d_along);
  d_dev = select(fing_live, d_dev, NEG);
  L.d_dev = min(d_dev, body);
  L.dev_mean = clamp(2.0 * wf * (grp - 1.0) / grp * 0.80, 0.0, 1.0);
  L.dev_key = h2(rk, k(int(kf)));
  L.dev_px = pdx * kq;
  float fu_raw = ((across - a0) / pdx - kf - 0.5) * 2.0;
  float fv_raw = ((dalong - db0) / pu - ju - 0.5) * 2.0;
  L.fu = select(dvert, fu_raw, fv_raw);
  L.fv = select(dvert, fv_raw, fu_raw);

  float fine = select(kind == KIND_LOGIC_F, pl, min(pw, ph));
  fine = select(kind == KIND_MOTIF_F, msz, fine);
  L.pitch_px = fine * kq;
  return L;
}

// =========================================================================
// route.py -- channels, wires, vias, and the power grid
// =========================================================================

struct Wires {
  float d_wire;     // positive inside; the union of every wire near this pixel
  float d_via;      // vias, at ENDS AND CORNERS ONLY -- never at crossings
  float pitch_px;   // the track pitch in (reference) pixels
};

/** A square-capped wire along `along` from a to b, centered on perp = c. */
float wireSeg(float perp, float along, float a, float b, float c, float hw) {
  return boxSd(perp - c, along - (a + b) * 0.5, hw, max(b - a, 0.0) * 0.5);
}

/** Every wire that could cover this pixel: THREE candidate home tracks. */
Wires wires(Chan ch, float kq, uint layer_key) {
  // THE TRACK COUNT IS CLAMPED BY THE PIXEL BUDGET.
  float budget = max(floor(2.0 * ch.w * kq / FADE_FULL_PX), 1.0);
  float ntr = max(min(rne(P(P_N_TRACKS)), budget), 1.0);
  float tp = (2.0 * ch.w) / ntr;
  float perp = ch.perp;
  float along = ch.along;
  float len = ch.len;
  float k0 = rne(perp / tp);
  float over = ch.w;
  // A MINIMUM WEIGHT IN PIXELS.
  float hw = max(P(P_WIRE_W) * tp, 0.55 / kq);
  float vr = P(P_VIA_R) * hw;

  float d_wire = NEG;
  float d_via = NEG;
  [unroll]
  for (int q = 0; q < 3; q++) {
    float kv = k0 + float(q - 1);
    uint tk = h4(ch.key, K_TRACK, k(int(kv)), layer_key);
    bool live = hf3(tk, K_TRACK, 1u) < P(P_OCCUPANCY);

    float span = len * clamp(P(P_REACH) * (0.5 + 1.0 * hf3(tk, K_ENDS, 2u)), 0.35, 1.0);
    float a = hf3(tk, K_ENDS, 1u) * max(len - span, 0.0);
    float b = min(a + span, len);
    // A run to the channel's end OVERHANGS the junction.
    a = select(hf3(tk, K_ENDS, 3u) < P(P_CONT_P), -over, a);
    b = select(hf3(tk, K_ENDS, 4u) < P(P_CONT_P), len + over, b);

    // The jog: one step to a neighboring track, partway along the run.
    bool jog = hf3(tk, K_JOG, 1u) < P(P_JOG_P);
    float jt = a + (b - a) * (0.25 + 0.5 * hf3(tk, K_JOG, 2u));
    float stp = select(hf3(tk, K_JOG, 3u) < 0.5, -1.0, 1.0);
    float c0 = kv * tp;
    float c1 = select(jog, (kv + stp) * tp, c0);

    float d1 = wireSeg(perp, along, a, select(jog, jt, b), c0, hw);
    float d2 = select(jog, wireSeg(perp, along, jt, b, c1, hw), NEG);
    float clo = min(c0, c1);
    float chi = max(c0, c1);
    float dc = select(jog, boxSd(perp - (clo + chi) * 0.5, along - jt, (chi - clo) * 0.5 + hw, hw), NEG);
    d_wire = max(d_wire, select(live, max(max(d1, d2), dc), NEG));

    // Vias at both ends and at the corner. NOT at crossings.
    d_via = max(d_via, select(live, boxSd(perp - c0, along - a, vr, vr), NEG));
    d_via = max(d_via, select(live, boxSd(perp - c1, along - b, vr, vr), NEG));
    d_via = max(d_via, select(and(live, jog), boxSd(perp - c0, along - jt, vr, vr), NEG));
    d_via = max(d_via, select(and(live, jog), boxSd(perp - c1, along - jt, vr, vr), NEG));
  }

  // BOTH fields are gated ACROSS the corridor as well as along it.
  float acrossd = ch.w * 1.05 - abs(perp);
  float inside = min(along + over, len + over - along);
  Wires W;
  W.d_wire = min(d_wire, min(inside, acrossd));
  W.d_via = min(d_via, min(inside, acrossd));
  W.pitch_px = tp * kq;
  return W;
}

/** One tier of straps, at two widths (metal, stitch) in a single pass. */
float2 strapFamily(float t, float pitch, float hw, uint key, float dens, float jit) {
  float k0 = rne(t / pitch);
  float2 d = float2(NEG, NEG);
  [unroll]
  for (int q = 0; q < 3; q++) {
    float kv = k0 + float(q - 1);
    uint kk = h2(key, k(int(kv)));
    float pos = (kv + jit * (hf2(kk, 1u) - 0.5)) * pitch;
    float w = hw * (0.40 + 1.20 * hf2(kk, 2u));
    bool live = hf2(kk, 3u) < dens;
    float r = abs(t - pos);
    d = float2(max(d.x, select(live, w - r, NEG)), max(d.y, select(live, w * 0.8 - r, NEG)));
  }
  return d;
}

struct Straps { float d; float stitch; float pitch_px; };

/** The global power lattice, in two tiers. IGNORES THE FLOORPLAN. */
Straps straps(Rect R, float kq, uint layer_key, bool vertical) {
  float lm = N_TRACK_F / 16.0;
  uint sd = dieSeed();
  bool on = hf2(h3(K_STRAP, layer_key, sd), 1u) < P(P_STRAP_ON);
  float tm = select(vertical, R.px, R.py);
  float ts = select(vertical, R.py, R.px);
  float p_mesh = max(rne(P(P_STRAP_MULT) * lm), 4.0);
  float p_spine = max(rne(P(P_SPINE_MULT) * lm), 10.0);
  uint km = h4(K_STRAP, 1u, layer_key, sd);
  uint ks = h4(K_STRAP, 2u, layer_key, sd);
  float dens = P(P_STRAP_DENS);
  float jit = P(P_STRAP_JIT);
  float2 fm = strapFamily(tm, p_mesh, max(P(P_STRAP_W) * lm, 0.35), km, dens, jit);
  float2 fs = strapFamily(ts, p_spine, max(P(P_SPINE_W) * lm, 0.6), ks, dens, jit);
  Straps S;
  S.d = select(on, max(fm.x, fs.x), NEG);
  // The stitch is a column THROUGH the layers: NOT gated by `on`.
  S.stitch = min(fm.y, fs.y);
  S.pitch_px = p_mesh * kq;
  return S;
}

// =========================================================================
// render.py -- the pitch fade, the palette, and one shaded layer
// =========================================================================

/** A repeating family under FADE_MIN_PX is replaced by its own mean coverage. */
float fade(float pitch_px) {
  float t = clamp((pitch_px - FADE_MIN_PX) / (FADE_FULL_PX - FADE_MIN_PX), 0.0, 1.0);
  return t * t * (3.0 - 2.0 * t);
}

/** A family, resolved where the pixels allow and TONED where they do not. */
float resolve(float dpx, float mean, float pitch_px, uint grain_key, float blur) {
  float f = fade(pitch_px);
  float amp = 1.15 * sqrt(max(mean * (1.0 - mean), 0.0)) * clamp(pitch_px, 0.0, 1.0);
  float tone = mean + amp * (hf2(grain_key, 77u) - 0.5);
  return f * hard(dpx, blur) + (1.0 - f) * clamp(tone, 0.0, 1.0);
}

/** HSV -> RGB. */
float3 hsv(float h, float s, float v) {
  float hh = (h - floor(h)) * 6.0;
  float i = fmodf(floor(hh), 6.0);
  float f = hh - floor(hh);
  float p = v * (1.0 - s);
  float q = v * (1.0 - s * f);
  float tt = v * (1.0 - s * (1.0 - f));
  float r = select(i < 0.5, v,  select(i < 1.5, q, select(i < 2.5, p,  select(i < 3.5, p, select(i < 4.5, tt, v)))));
  float g = select(i < 0.5, tt, select(i < 1.5, v, select(i < 2.5, v,  select(i < 3.5, q, select(i < 4.5, p,  p)))));
  float b = select(i < 0.5, p,  select(i < 1.5, p, select(i < 2.5, tt, select(i < 3.5, v, select(i < 4.5, v,  q)))));
  return float3(r, g, b);
}

/** One feature class's color on one layer. */
float3 layerColor(float li, float cls, float off) {
  float stp = fmodf(li, N_LAYERS_F) / N_LAYERS_F;
  return hsv(P(P_HUE) + P(P_HUE_SPREAD) * (cls + 0.35 * stp) + off, P(P_SAT), P(P_VAL_HI));
}

/** How much deeper the focus plane sits at this pixel, in layers. */
float tilt(float px, float py) {
  float a = P(P_SCAN_ANG) * 6.2831853;
  float v = viewTracks();
  float cx = P(P_CAM_FRAC_X) * N_TRACK_F;
  float cy = P(P_CAM_FRAC_Y) * N_TRACK_F;
  return P(P_SCAN_TILT) * (cos(a) * (px - cx) + sin(a) * (py - cy)) / v;
}

struct Slab { float3 col; float cov; float vis; };

/** One layer, shaded. Coverage is what this layer OCCLUDES with. */
Slab slab(float px, float py, float li, float dz, float kpx, float kq) {
  float dzf = dz + tilt(px, py);
  float win = min(P(P_WINDOW), 2.0);
  // VISIBILITY IS ASYMMETRIC IN DEPTH.
  float farv = clamp((win - dzf) / 0.35, 0.0, 1.0);
  float vis = farv * farv * (3.0 - 2.0 * farv);
  float off = P(P_PARALLAX) * dzf;
  Rect R = descend(px + off, py + off * 0.6);
  // ...and STAGGERED PER BLOCK.
  float bph = blockPhase(R.key);
  float dzs = dzf + P(P_STAGGER) * bph;

  // A LAYER PASSING THE CAMERA IS ERODED AWAY, each family on its own schedule.
  float erk = P(P_ERODE) / kq;
  float er_fine  = erk * max(1.0 - dzs - Z_EXTENT * (0.00 - 0.42), 0.0);
  float er_hatch = erk * max(1.0 - dzs - Z_EXTENT * (0.10 - 0.42), 0.0);
  float er_cell  = erk * max(1.0 - dzs - Z_EXTENT * (0.25 - 0.42), 0.0);
  float er_wire  = erk * max(1.0 - dzs - Z_EXTENT * (0.50 - 0.42), 0.0);
  float er_strap = erk * max(1.0 - dzs - Z_EXTENT * (0.62 - 0.42), 0.0);
  float er       = erk * max(1.0 - dzs - Z_EXTENT * (0.75 - 0.42), 0.0);
  float er_via = 0.0;                  // a via is a column; it never clears

  // THE CONTENT STAYS ATTACHED TO THE PHYSICAL LAYER.
  uint lk = contentKey(li);
  float zf = li + 0.5 + P(P_STAGGER) * bph;
  int2 o = camOrigin();
  Leaf L = leafField(R, kq, lk, o.x, o.y, zf);
  bool vert = routesVertical(li);
  Wires W = wires(rectChannel(R, vert), kq, lk);
  Straps S = straps(R, kq, lk, vert);

  float coarse = 1.0 + 1.1 * dzs * dzs;
  // THE DEFOCUS: out-of-focus material does not go faint, it goes SOFT. Its
  // shoulder is a fraction of the reference frame, never under a real pixel.
  float blur = max(1.0, (1.0 + P(P_DEFOCUS) * dzs * dzs) * P(P_PX_SCALE));
  float f_cell = fade(L.pitch_px / coarse);
  float f_fine = fade(L.fine_px / coarse);
  float f_wire = fade(W.pitch_px / coarse);
  float f_strap = fade(S.pitch_px / coarse);

  float sdi = rectSdInset(R);
  float body = hard((sdi - er) * kpx, blur);
  float cell = resolve((L.d_cell - er_cell) * kpx, L.cell_mean, L.pitch_px / coarse,
                       L.cell_key, blur) * body;
  float fine = resolve((L.d_fine - er_fine) * kpx, L.fine_mean, L.fine_px / coarse,
                       L.cell_key + 1u, blur) * body;
  float hatch = hard((L.d_hatch - er_hatch) * kpx, blur)
              * fade(P(P_HATCH) * kq / coarse) * body;
  float dev = resolve((L.d_dev - er_fine) * kpx, L.dev_mean, L.dev_px / coarse,
                      L.dev_key, blur);
  float wire = hard((W.d_wire - er_wire) * kpx, blur) * f_wire;
  float via = max(hard((W.d_via - er_via) * kpx, blur) * f_wire,
                  hard((S.stitch - er_via) * kpx, blur) * f_strap);
  float strap = hard((S.d - er_strap) * kpx, blur) * f_strap;

  // Height, for the SEM register.
  float h = H_SUB * body;
  h = max(h, H_CELL * cell);
  h = max(h, H_FINE * fine);
  h = h * (0.78 + 0.40 * dev);
  h = max(h, H_HATCH * hatch);
  h = max(h, H_WIRE * wire);
  h = max(h, H_VIA * via);
  h = max(h, H_STRAP * strap);

  float lw = P(P_LINE_W) * P(P_PX_SCALE);
  // THE BLOCK OUTLINE FADES BY THE BLOCK'S OWN SIZE.
  float2 sz = rectSize(R);
  float f_blk = fade(min(sz.x, sz.y) * kq);
  float sdo = rectSd(R);
  float ink = strokeOf((sdo - er) * kpx, lw, blur) * f_blk;
  ink = max(ink, strokeOf((sdi - er) * kpx, lw * 0.7, blur) * 0.7 * f_blk);
  ink = max(ink, strokeOf((L.d_cell - er_cell) * kpx, lw * 0.6, blur) * f_cell * body);
  ink = max(ink, strokeOf((L.d_fine - er_fine) * kpx, lw * 0.45, blur) * f_fine * body * 0.8);
  ink = max(ink, strokeOf((W.d_wire - er_wire) * kpx, lw * 0.6, blur) * f_wire);

  // --- the layout register -------------------------------------------------
  float3 lay = hsv(P(P_HUE) + P(P_HUE_SPREAD) * 0.2, P(P_SAT) * 0.5, P(P_VAL_LO));
  lay = lay + (layerColor(li, 0.0, L.hue_off * 0.6) - lay) * (body * 0.55);
  lay = lay + (layerColor(li, 1.0, L.hue_off) - lay) * cell;
  lay = lay + (layerColor(li, 2.0, L.hue_off) - lay) * (hatch * 0.8);
  lay = lay + (layerColor(li, 3.0, 0.0) - lay) * strap;
  lay = lay + (layerColor(li, 4.0, 0.0) - lay) * wire;
  lay = lay + (layerColor(li, 5.0, 0.0) - lay) * via;
  // Contacts are a DARKENING of the cell they sit in.
  lay = lay * (1.0 - 0.45 * fine);
  // APPARENT RELIEF.
  float fa = P(P_FACET_ANG) * 6.2831853 + 0.9 * dzs;
  float facet = clamp(0.5 + 0.5 * (L.fu * cos(fa) + L.fv * sin(fa)), 0.0, 1.0);
  lay = lay * (1.0 + P(P_FACET) * (facet - 0.5));
  lay = lay * (0.62 + 0.66 * dev);
  float3 keyc = hsv(P(P_HUE), P(P_SAT) * 0.3, P(P_VAL_LO) * 0.35);
  lay = lay + (keyc - lay) * (ink * 0.85);

  // --- the SEM register ----------------------------------------------------
  float halo = 0.0;
  halo = max(halo, strokeOf(sdo * kpx, lw * (0.5 + 0.5 * 1.2), blur) * (1.2 / H_STRAP));
  halo = max(halo, strokeOf(L.d_cell * kpx, lw * (0.5 + 0.5 * H_CELL), blur) * (H_CELL / H_STRAP));
  halo = max(halo, strokeOf(L.d_fine * kpx, lw * (0.5 + 0.5 * H_FINE), blur) * (H_FINE / H_STRAP));
  halo = max(halo, strokeOf(W.d_wire * kpx, lw * (0.5 + 0.5 * H_WIRE), blur) * (H_WIRE / H_STRAP));
  halo = max(halo, strokeOf(W.d_via * kpx, lw * (0.5 + 0.5 * H_VIA), blur) * (H_VIA / H_STRAP));
  halo = max(halo, strokeOf(S.d * kpx, lw * (0.5 + 0.5 * H_STRAP), blur) * (H_STRAP / H_STRAP));
  float base = P(P_VAL_LO) + (P(P_VAL_HI) - P(P_VAL_LO)) * clamp(h / H_STRAP, 0.0, 1.0) * 0.75;
  float ao = 0.16 * (1.0 - body);
  // A HASH, not a random: fixed to the layer, so it travels with it.
  float grain = P(P_SCAN_AMP) * (hf4(k(int(rne(px * 4.0))), k(int(rne(py * 4.0))), lk, K_LAYER) - 0.5);
  float sem = clamp(base + P(P_EDGE_GAIN) * 0.55 * halo - ao + grain, 0.0, 1.0);

  Slab res;
  res.col = lay + (float3(sem, sem, sem) - lay) * P(P_LOOK);
  res.vis = vis;
  // The bulk: a layer is mostly solid even where nothing is drawn on it.
  float cov = max(max(body, max(cell, fine)), max(max(wire, via), strap));
  cov = max(cov, P(P_BULK) * hard((0.6 - er) * kpx, blur));
  res.cov = clamp(cov, 0.0, 1.0);
  return res;
}

/** The frame: four slabs, NEAREST FIRST, carrying transmittance. */
float3 frame(float px, float py, float kpx, float kq, float t) {
  float cue = P(P_DEPTH_CUE);
  float hz = P(P_HAZE);
  float3 med = hsv(P(P_HUE) + P(P_HUE_SPREAD) * 0.6, P(P_SAT) * 0.30,
                   P(P_VAL_LO) + 0.35 * (P(P_VAL_HI) - P(P_VAL_LO)));
  float3 back = hsv(P(P_HUE) + P(P_HUE_SPREAD) * 0.6, P(P_SAT) * 0.30,
                    P(P_VAL_LO) + 0.15 * (P(P_VAL_HI) - P(P_VAL_LO)));
  float tr = 1.0;                   // transmittance still unspent
  float tau = 0.0;                  // material above, in layers
  float3 acc = float3(0.0, 0.0, 0.0);
  [loop]
  for (int s = 0; s < N_SLABS; s++) {
    Slab S = slab(px, py, slabLi(t, s), slabDz(t, s), kpx, kq);
    float q = tau + PASSIVATION;
    float3 col = S.col * pow(cue, q);
    col = col + (med - col) * clamp(hz * q, 0.0, 0.92);
    float a = clamp(S.cov * S.vis, 0.0, 1.0);
    acc = acc + tr * a * col;
    tr = tr * (1.0 - a);
    tau = tau + a;
  }
  float q2 = tau + PASSIVATION;
  back = back * pow(cue, q2);
  back = back + (med - back) * clamp(hz * q2, 0.0, 0.92);
  return acc + tr * back;
}

// =========================================================================
// the frame
// =========================================================================

float3 shade(float2 pos) {
  float2 s = coverSquare(pos);
  float2 w = pixelWorld(s);
  float kpx = pixelsPerTrack();
  float kq = kpx / max(P(P_PX_SCALE), 1e-3);

#ifdef DIE_FOLD_PROBES
  // The prototype's per-stage probes (parity_ref.py --probe N). Each inlines
  // another copy of the pipeline, so they are compiled out unless a parity hunt
  // asks for them: dxc ... -D DIE_FOLD_PROBES.
  int probe = int(P(P_PROBE));
  if (probe == 1) {
    // The hash stream, per super-cell.
    int2 c = int2(int(floor(w.x / N_TRACK_F)), int(floor(w.y / N_TRACK_F)));
    float v = hf3(k(int(P(P_CAM_CELL_X)) + c.x), k(int(P(P_CAM_CELL_Y)) + c.y), dieSeed());
    return float3(v, v, v);
  }
  if (probe == 6) {
    // The nearest slab: coverage, visibility, luma.
    Slab S = slab(w.x, w.y, slabLi(P(P_T), 1), slabDz(P(P_T), 1), kpx, kq);
    float lum = dot(S.col, float3(0.2126, 0.7152, 0.0722));
    return float3(S.cov, S.vis, clamp(lum, 0.0, 1.0));
  }
  if (probe == 5) {
    // Routing: wires, vias, the power grid.
    Rect R = descend(w.x, w.y);
    float li = slabLi(P(P_T), 1);
    uint lk = contentKey(li);
    bool vert = routesVertical(li);
    Wires W = wires(rectChannel(R, vert), kq, lk);
    Straps S = straps(R, kq, lk, vert);
    return float3(hard(W.d_wire * kpx, 1.0), hard(W.d_via * kpx, 1.0), hard(S.d * kpx, 1.0));
  }
  if (probe == 4) {
    // Block interiors: cells, the contact level, the analytic duty cycle.
    Rect R = descend(w.x, w.y);
    float li = slabLi(P(P_T), 1);
    float zf = li + 0.5 + P(P_STAGGER) * blockPhase(R.key);
    int2 o = camOrigin();
    Leaf L = leafField(R, kq, contentKey(li), o.x, o.y, zf);
    return float3(hard(L.d_cell * kpx, 1.0), hard(L.d_fine * kpx, 1.0), clamp(L.cell_mean, 0.0, 1.0));
  }
  if (probe == 3) {
    // The floorplan: the inset leaf body, and the two nearest corridors.
    Rect R = descend(w.x, w.y);
    return float3(hard(rectSdInset(R) * kpx, 1.0),
                  hard((R.v_w - abs(R.px - R.v_pos)) * kpx, 1.0),
                  hard((R.h_w - abs(R.py - R.h_pos)) * kpx, 1.0));
  }
  if (probe == 2) {
    // The fractional position within the super-cell.
    float fx = w.x / N_TRACK_F;
    float fy = w.y / N_TRACK_F;
    return float3(fx - floor(fx), fy - floor(fy), 0.0);
  }
#endif
  return clamp(frame(w.x, w.y, kpx, kq, P(P_T)), 0.0, 1.0);
}

[numthreads(8, 8, 1)]
void main(uint3 gid : SV_DispatchThreadID) {
  if (gid.x >= (uint)P(P_SURF_W) || gid.y >= (uint)P(P_SURF_H)) return;
  outTex[gid.xy] = float4(shade(float2(gid.xy) + 0.5), 1.0);
}
