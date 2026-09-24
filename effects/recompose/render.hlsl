// warp.recompose — render / cell-translation pass.
//
// Reads the nine latched correction vectors and inverse-maps each output pixel:
// cell k draws Rect_k at Rect_k + D_k, so pixel q is covered by cell k exactly
// when rc_cell_index(q − D_k) == k.
//
// Two phases, so the common case costs ONE texture read. Phase 1 is pure
// arithmetic over the nine cells — two buffer loads and a rect test each, no
// sampling — building a coverage bitmask and the heaviest coverer. Phase 2 then
// samples only what actually won: a single tap for the overwhelming majority of
// pixels (one coverer, or Heaviest On Top), and a loop only across genuinely
// overlapped pixels under Blend / Additive.
//
// ---- Rift vs edge fill ----
//
// The nine rects tile the frame exactly, so a COVERED pixel's source is always
// in-frame — plane_shear's "covered but pre-image outside" test would make
// edge_fill dead code here. The distinction instead comes from the HOME cell of
// an UNCOVERED pixel: cell h contains q, so h needed content from q − D_h.
//   q − D_h outside the frame → h reached past the viewport border → edge_fill
//   q − D_h inside            → an interior seam between separated cells → rift_fill
// Geometrically that puts edge_fill along the border where the outer ring slid
// inward, and rift_fill on interior gaps — matching plane_shear's user-facing
// meaning ("the gap between halves pulled apart" vs "the viewport border a slid
// half reveals"). The centre cell can only ever produce a rift.
//
// Known limitation: cell boundaries are hard geometric seams, as plane_shear's
// dividing line is. There is no feathering.

#include "common.hlsl"

Texture2D<float4>       inputTex : register(t0);
SamplerState            samp     : register(s1);   // Linear / ClampToEdge
StructuredBuffer<float> solve    : register(t2);
RWTexture2D<float4>     outTex   : register(u3);

cbuffer U : register(b4) {
  float aspect_x, aspect_y;
  float rift_fill;      // 0 transparent / 1 original / 2 edge-stretch / 3 mirror / 4 black
  float edge_fill;      // same five
  float overlap_mode;   // 0 heaviest on top / 1 blend / 2 additive
  float debug_show;     // 0 off / 1 grid + centroid / 2 cell mass
  float center_bias;    // for the cell-mass overlay's ideal reference
  float _p0;
};

float4 rc_sample(float2 x, float2 aspect) {
  return inputTex.SampleLevel(samp, nano_cover_square_to_uv(x, aspect), 0);
}

// Reflect p into [lo, hi] — the same triangle wave plane_shear uses for its
// mirror fill. t = 1.2 → 0.8, t = -0.2 → 0.2, t = 0.5 → 0.5.
float2 rc_mirror_into(float2 p, float2 lo, float2 hi) {
  float2 span = max(hi - lo, 1e-5);
  float2 t    = (p - lo) / span;
  float2 tm   = 1.0 - abs(frac(t * 0.5) * 2.0 - 1.0);
  return lo + tm * span;
}

// The five fill modes, shared by rift and edge — only the [lo,hi] bounds differ
// (the home cell's rect for a rift, the whole frame for an edge reveal).
float4 rc_fill(int mode, float2 q, float2 src, float2 lo, float2 hi, float2 aspect) {
  if (mode == 0) return float4(0.0, 0.0, 0.0, 0.0);                      // transparent
  if (mode == 1) return rc_sample(q, aspect);                            // original at this pixel
  if (mode == 3) return rc_sample(rc_mirror_into(src, lo, hi), aspect);  // mirror
  if (mode == 4) return float4(0.0, 0.0, 0.0, 1.0);                      // black
  return rc_sample(clamp(src, lo, hi), aspect);                          // 2 = edge stretch
}

[numthreads(8, 8, 1)]
void main(uint3 gid : SV_DispatchThreadID) {
  uint W, H;
  outTex.GetDimensions(W, H);
  if (gid.x >= W || gid.y >= H) return;

  float2 aspect = float2(aspect_x, aspect_y);
  float2 E      = rc_extent(aspect);
  float2 uv     = (float2(gid.xy) + 0.5) / float2(W, H);
  float2 q      = nano_uv_to_cover_square(uv, aspect);

  int om = (int)(overlap_mode + 0.5);

  // ---- phase 1: coverage mask + heaviest coverer (no texture reads) ----
  uint mask  = 0u;
  int  best  = -1;
  int  nCover = 0;
  float bestM = -1.0;
  [unroll] for (int k = 0; k < 9; ++k) {
    float2 D = float2(solve[RC_S_D + 2 * k], solve[RC_S_D + 2 * k + 1]);
    if (rc_cell_index(q - D, E) == k) {
      mask |= (1u << (uint)k);
      nCover++;
      float mk = solve[RC_S_M + k];
      if (mk > bestM) { bestM = mk; best = k; }
    }
  }

  float4 col;
  if (mask == 0u) {
    // ---- phase 2a: uncovered → rift or edge fill (see the header note) ----
    int h = rc_cell_index(q, E);
    if (h < 0) h = 4;                       // defensive; q is in-frame by construction
    float2 Dh   = float2(solve[RC_S_D + 2 * h], solve[RC_S_D + 2 * h + 1]);
    float2 srcH = q - Dh;
    bool outOfFrame = (rc_cell_index(srcH, E) < 0);

    float2 lo, hi;
    if (outOfFrame) { lo = -E; hi = E; }
    else            { rc_cell_rect(h, E, lo, hi); }
    int fm = (int)((outOfFrame ? edge_fill : rift_fill) + 0.5);
    col = rc_fill(fm, q, srcH, lo, hi, aspect);

  } else if (om == RC_OM_HEAVIEST || nCover == 1) {
    // ---- phase 2b: single sample ----
    float2 D = float2(solve[RC_S_D + 2 * best], solve[RC_S_D + 2 * best + 1]);
    col = rc_sample(q - D, aspect);

  } else {
    // ---- phase 2c: blend / additive across the covering set ----
    float4 acc  = float4(0.0, 0.0, 0.0, 0.0);
    float3 add  = float3(0.0, 0.0, 0.0);
    float  n    = 0.0;
    float  amax = 0.0;
    [unroll] for (int k2 = 0; k2 < 9; ++k2) {
      if ((mask & (1u << (uint)k2)) == 0u) continue;
      float2 D = float2(solve[RC_S_D + 2 * k2], solve[RC_S_D + 2 * k2 + 1]);
      float4 c = rc_sample(q - D, aspect);
      acc += c; n += 1.0;
      add += c.rgb; amax = max(amax, c.a);
    }
    col = (om == RC_OM_ADDITIVE) ? float4(saturate(add), amax) : (acc / max(n, 1.0));
  }

  // ---- Debug overlays ----
  int ds = (int)(debug_show + 0.5);
  if (ds == 1) {
    // Thirds grid + the measured centre of mass and its target.
    // One pixel is exactly 2/max(W,H) cover-square units on BOTH axes (since
    // ax = max(W,H)/(2W) and ay = max(W,H)/(2H)), so this is a true 1.5px line
    // at any resolution or aspect. Deriving it any other way goes sub-pixel on
    // small viewports and the grid dashes out.
    float eps = 3.0 / float(max(W, H));
    float lines = max(1.0 - smoothstep(0.0, eps, abs(abs(q.x) - E.x / 3.0)),
                      1.0 - smoothstep(0.0, eps, abs(abs(q.y) - E.y / 3.0)));
    col.rgb = lerp(col.rgb, float3(1.0, 0.15, 0.15), lines * 0.85);

    float2 c   = float2(solve[RC_S_CX], solve[RC_S_CY]);
    float2 P   = float2(solve[RC_S_PX], solve[RC_S_PY]);
    float  rad = 6.0 * eps;
    float  dq  = length(q - c);
    float  dp  = length(q - P);
    float  cdot = 1.0 - smoothstep(rad * 0.6, rad, dq);
    float  ring = (1.0 - smoothstep(rad * 0.8, rad, dp)) *
                  smoothstep(rad * 0.45, rad * 0.65, dp);
    col.rgb = lerp(col.rgb, float3(0.2, 1.0, 0.3), cdot);   // green: where the weight is
    col.rgb = lerp(col.rgb, float3(0.3, 0.6, 1.0), ring);   // blue:  where it should be
    col.a   = max(col.a, max(lines, max(cdot, ring)));
  } else if (ds == 2) {
    // Per-cell surplus/deficit heat map: red = too much mass, blue = too little.
    int h = rc_cell_index(q, E);
    if (h >= 0) {
      float t = clamp(9.0 * (solve[RC_S_M + h] - rc_ideal(h, center_bias)), -1.0, 1.0);
      float3 grey = float3(0.5, 0.5, 0.5);
      float3 heat = (t >= 0.0) ? lerp(grey, float3(1.0, 0.2, 0.15), t)
                               : lerp(grey, float3(0.15, 0.4, 1.0), -t);
      col.rgb = lerp(col.rgb, heat, 0.55);
      col.a   = max(col.a, 0.55);
    }
  }

  outTex[gid.xy] = col;
}
