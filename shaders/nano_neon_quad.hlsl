// nano_neon_quad.hlsl — the neon-quad field, factored out so more than one
// effect can wear it and stay byte-identical.
//
// Give it four corner points in 2D and it hands back the two fields a neon
// outline needs: an EXACT signed distance for the crisp core, and a separate
// glow field built to have no artefacts of its own. Nothing here knows how many
// quads there are, where they came from, or what cbuffer the caller uses —
// every input is an argument. `source.mesh.three_planes` projects three
// isometric floors into it; `source.mesh.three_walls` projects the same three
// quads three times, once per camera.
//
// This is the same arrangement as nano_vcr.hlsl: a header of pure functions
// plus a parameter STRUCT the caller embeds verbatim in its own cbuffer, so two
// effects run literally the same code over the same bytes without being forced
// onto one cbuffer layout.
//
// --- The two fields, and why they are different ---------------------------
//
//   sd   — the exact signed distance to the outline (iq's polygon SDF,
//          unrolled to four explicit edges so the SPIR-V -> WGSL/MSL
//          translation stays trivial). Drives the crisp line core and the
//          antialiased interior coverage. Exact for any simple quadrilateral,
//          not just a parallelogram — a perspective-projected quad is fine.
//
//   glow — never a plain function of `sd`, for the reasons below.
//
// Why the halo does NOT reuse `sd` directly: min() is only C0 where two edges
// are equidistant, so exp(-|sd|/r) inherits the distance field's medial axis —
// the "roof ridge" inside the quad where the nearest edge switches. The
// gradient flips across it, which Mach-bands into a visible crease running out
// of every corner (and a matching curvature kink just outside each vertex).
//
// The halo therefore folds the four edge distances with a LOG-SUM-EXP soft
// minimum instead. Two properties make it the right operator here:
//
//   * It is C-infinity — no derivative of any order jumps. A polynomial smin
//     is only C1 and still has a C2 break at its band edges, which Mach-bands
//     faintly on its own; this has nothing to band on.
//   * Its bias is SELF-LIMITING. Where one edge dominates, the sum collapses
//     to that term and the result is the true minimum, exactly. It only pulls
//     below min where several edges are comparable — which is precisely the
//     crease, and precisely where we want rounding. So the field away from the
//     medial axis, and hence the tuned brightness and falloff, is untouched.
//
// Note a plain sum of per-edge kernels, sum(exp(-d_i/r)), is this same softmin
// with T pinned to the halo radius: smooth, but it under-reads distance by up
// to r*ln(4) and washes the whole figure out. Decoupling T from r is the fix.
//
// ...OUTSIDE. Inside the quad the softmin cannot win, because BOTH of its terms
// carry the skeleton: `m` ridges along the medial axis (those points really are
// furthest from the outline) and the bias `t*ln(n)` brightens wherever n edges
// tie — which is the same locus. Deep inside a rhombus the medial axis is both
// diagonals meeting at the centre, so the interior reads as a dark four-pointed
// star when the ridge wins, and as a bright centre blob ringed by a dark
// contour when the bias does. Widening the band just trades one for the other.
//
// So the interior does not use a nearest-edge construction at all. It SUMS the
// four edges' halo kernels, which is what light from four tubes actually does:
// smooth, no skeleton, and monotone inward. That sum is exactly the "washes the
// figure out" case above — true where it would double-count the corners on the
// OUTSIDE, harmless inside, where there is no tuned look to protect and the
// alternative is a visible skeleton.
//
// The two are cross-faded over the first `kInteriorBlend` halo radii of depth,
// so the outline and everything outside it stay bit-for-bit what they were.
// Near the outline the far edges contribute nothing anyway, so the fade has
// almost nothing to do; by the time it matters, it is fully inside.

#ifndef NANO_NEON_QUAD_HLSL
#define NANO_NEON_QUAD_HLSL

// Shared style block. Embed VERBATIM in the caller's cbuffer, mirrored CPU-side
// as a plain float array. 12 floats = 48 bytes = 3 std140 rows.
struct NeonStyle {
  float line_hw;       // line half-width, in the caller's 2D units
  float line_gain;     // core intensity
  float core_whiten;   // how hard the core blows to white
  float halo_r;        // halo radius, same units as line_hw

  float halo_gain;     // halo intensity
  float falloff;       // 0 tight and punchy, 1 wide and soft
  float corner_r;      // corner rounding — shrinks the shape
  float aa;            // antialias width (one pixel, usually)

  float fill_gain;     // interior flood, for fill > 0
  float halo_smooth;   // softmin band as a fraction of halo_r; 0 = raw ridge
  float pad0;
  float pad1;
};

float nano_neon_seg_dist(float2 p, float2 a, float2 b) {
  float2 e = b - a;
  float2 w = p - a;
  float2 q = w - e * saturate(dot(w, e) / max(dot(e, e), 1e-12));
  return sqrt(dot(q, q));
}

// Even-odd crossing test for one edge — supplies the SIGN of `sd`.
bool nano_neon_seg_crosses(float2 p, float2 a, float2 b) {
  float2 e = b - a;
  float2 w = p - a;
  bool3 c = bool3(p.y >= a.y, p.y < b.y, e.x * w.y > e.y * w.x);
  return all(c) || all(!c);
}

// Stacked exponentials at 0.25x / 1x / 4x the radius. A single Gaussian reads
// flat and synthetic; three octaves is what makes it read as glow. Weights sum
// to 1 at distance 0, so `halo_gain` stays the only intensity control.
// `falloff` 0 = tight and punchy, 1 = wide and soft.
float nano_neon_halo_profile(float ad, float r, float falloff) {
  float r0 = max(r, 1e-4);
  float w = saturate(falloff);
  float3 e = float3(exp(-ad / (r0 * 0.25)),
                    exp(-ad / (r0 * 1.00)),
                    exp(-ad / (r0 * 4.00)));
  float3 wts = lerp(float3(0.60, 0.30, 0.10), float3(0.15, 0.30, 0.55), w);
  return dot(e, wts);
}

// Log-sum-exp soft minimum of four distances, shifted by the true min so the
// exponentials stay in [0,1] and the log argument stays in [1,4] — exact and
// overflow-free regardless of how far away the point is.
float nano_neon_softmin4(float d0, float d1, float d2, float d3, float t) {
  float m = min(min(d0, d1), min(d2, d3));
  float e = exp(-(d0 - m) / t) + exp(-(d1 - m) / t)
          + exp(-(d2 - m) / t) + exp(-(d3 - m) / t);
  return m - t * log(e);
}

/// How deep, in halo radii, the interior light-sum takes over from the softmin.
static const float kNanoNeonInteriorBlend = 2.0;

struct NanoNeonField { float sd; float glow; };

NanoNeonField nano_neon_quad(float2 p, float2 a, float2 b, float2 c, float2 d,
                             float corner_r, float halo_r, float falloff,
                             float smooth_frac) {
  float d0 = nano_neon_seg_dist(p, a, b);
  float d1 = nano_neon_seg_dist(p, b, c);
  float d2 = nano_neon_seg_dist(p, c, d);
  float d3 = nano_neon_seg_dist(p, d, a);

  float s = 1.0;
  if (nano_neon_seg_crosses(p, a, b)) s = -s;
  if (nano_neon_seg_crosses(p, b, c)) s = -s;
  if (nano_neon_seg_crosses(p, c, d)) s = -s;
  if (nano_neon_seg_crosses(p, d, a)) s = -s;

  NanoNeonField f;
  f.sd = s * min(min(d0, d1), min(d2, d3)) - corner_r;

  // The rounding band scales with the halo, because that is the only scale at
  // which the crease is bright enough to see: further in, the glow has already
  // fallen to nothing and the ridge is invisible whatever we do here.
  float t = max(halo_r * smooth_frac, 1e-5);
  float g = nano_neon_softmin4(d0, d1, d2, d3, t);
  // Re-sign BEFORE the rounding offset: corner_r shrinks the shape, so it has
  // to move the signed field. Subtracting it from an unsigned distance would
  // read interior points as 2*corner_r closer than they are and flood the
  // inside with glow.
  float glow_near = nano_neon_halo_profile(abs(s * g - corner_r), halo_r, falloff);

  // The interior light-sum (see the note above). Each edge is offset by
  // corner_r for the same reason the softmin branch is.
  float glow_sum = nano_neon_halo_profile(d0 + corner_r, halo_r, falloff)
                 + nano_neon_halo_profile(d1 + corner_r, halo_r, falloff)
                 + nano_neon_halo_profile(d2 + corner_r, halo_r, falloff)
                 + nano_neon_halo_profile(d3 + corner_r, halo_r, falloff);

  // Depth measured from the outline in halo radii. smoothstep rather than a
  // clamp so the handover has no gradient step of its own to band on.
  float depth = smoothstep(0.0, max(halo_r * kNanoNeonInteriorBlend, 1e-5), -f.sd);
  f.glow = lerp(glow_near, glow_sum, depth);
  return f;
}

// One quad's contribution to the accumulator.
//
// `fill` is signed and does two different jobs: positive floods the interior
// with the quad's own colour, negative turns the quad into a MASK that eats
// whatever is already accumulated (returned in `occlusion`, for the caller's
// `acc = acc * (1 - occlusion) + emit`). Fixed-function blend cannot express
// both in one draw, which is why the callers resolve in a loop.
// The emission, split into the two parts a MASK has to treat differently.
//
// `halo` is the wide soft field the tube throws; the RETURN is everything else
// — the line core and the interior flood, which are the plane as an object.
// Anything that cuts a shape out of the picture has to cut those two by
// different amounts, because a tube behind a letter still throws light around
// the letter's edges, and once the halo has been summed into the body that is
// no longer expressible. Callers with nothing to mask want the sum and should
// use nano_neon_quad_emit below, which IS the sum.
float3 nano_neon_quad_emit_split(NanoNeonField f, float3 col, float emis,
                                 float fill, NeonStyle st,
                                 out float occlusion, out float3 halo) {
  float aa     = max(st.aa, 1e-6);
  float inside = saturate(0.5 - f.sd / aa);
  float core   = 1.0 - smoothstep(st.line_hw - aa, st.line_hw + aa, abs(f.sd));

  // Real neon photographs blow their core to white and keep the hue only out
  // in the halo. This is the knob that makes it read as neon at all.
  float3 tint = lerp(col, float3(1.0, 1.0, 1.0), saturate(st.core_whiten * core));

  occlusion = -min(fill, 0.0) * inside;
  halo = emis * tint * (f.glow * st.halo_gain);
  return emis * tint * (core * st.line_gain
                      + max(fill, 0.0) * inside * st.fill_gain);
}

float3 nano_neon_quad_emit(NanoNeonField f, float3 col, float emis, float fill,
                           NeonStyle st, out float occlusion) {
  float3 halo;
  return nano_neon_quad_emit_split(f, col, emis, fill, st, occlusion, halo) + halo;
}

#endif  // NANO_NEON_QUAD_HLSL
