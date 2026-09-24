// source.mesh.three_planes — the whole effect, in one fullscreen compute pass.
//
// The CPU has already projected three isometric quads into cover-square
// screen coords and handed us 12 corner points. Per pixel we:
//
//   1. take the EXACT signed distance to each quad's outline and its glow
//      field (nano_neon_quad.hlsl — shared with source.mesh.three_walls, and
//      where the reasoning behind the two fields lives);
//   2. turn that into a line core, a stacked-exponential halo, and an
//      antialiased interior coverage;
//   2b. scale that emission by whichever glint particles are passing, so a
//      glint lights the halo as well as the core (see `glimmer_at`);
//   3. resolve all three planes bottom-to-top in ONE expression, so a
//      masking plane can occlude the halos beneath it while still emitting
//      its own (see `resolve` below — this is the whole reason the effect
//      is a fullscreen pass rather than three additive draws);
//   3b. add the release ghosts — the same quads again, either thrown outward
//      and opening out as they fly or sitting exactly where they were as bare
//      wireframe, depending on the throw mode. The host has already resolved
//      which; from here it is one path (see `ring_at`), and it is lit by the
//      same glints as everything else;
//   3c. cut the screen-space MASK out of the result — a stencil that removes
//      the neon where it covers, letting only some of the halo spill past its
//      edge (see `mask_at` and the split accumulators in `resolve`);
//   4. grade the composite through the shared VCR stack.
//
// Nothing here needs an intermediate texture: the accumulator lives in
// registers at float precision and is tone-mapped exactly once, on write.

#include "nano_coords.hlsl"
#include "nano_glint.hlsl"
#include "nano_neon_quad.hlsl"
#include "nano_vcr.hlsl"

Texture2D<float4>   inputTex  : register(t0);
RWTexture2D<float4> outputTex : register(u1);
// The stencil. Bound to `mask_in` when something is wired there, and to the
// input texture otherwise — `mask.x` is 0 in that case, so nothing is read
// into the picture and there is no branch to take.
Texture2D<float4>   maskTex   : register(t3);

cbuffer Uniforms : register(b2) {
  // Projected corners, cover-square coords. Plane i occupies rows 2i and
  // 2i+1: (c0.xy, c1.zw) then (c2.xy, c3.zw), wound consistently.
  float4 corners[6];

  // rgb = plane colour, w = emission drive (already curved by the host).
  float4 plane_color[3];

  float4 fills;       // xyz = signed fill per plane (+neon / -mask), w = halo smooth
  float4 neon0;       // line half-width, line gain, core whiten, halo radius
  float4 neon1;       // halo gain, halo falloff, corner radius, aa width
  float4 misc;        // fill gain, chroma bleed, has_input, debug mode
  float4 view;        // vp_w, vp_h, aspect_x, aspect_y
  // The release rings: the same three quads, thrown outward. Ring i occupies
  // rows 2i and 2i+1, wound the same way `corners` is.
  float4 ghosts[6];
  float4 rel;         // ring halo radius, falloff, -, -
  // Per-ring gain, carrying the release, the opening-out, the local-contrast
  // damping and the one-frame hold — all of it worked out on the host.
  float4 ring_gain;
  float4 glim0;       // glint travel dir x, dir y, -, -
  // One row per glint IN FLIGHT: where it sits on the travel axis, its
  // half-width there, and the brightness and wake depth it was born with. A
  // dead slot is zero gain and zero shade, so there is no count and no branch.
  float4 glints[8];
  // x = mask strength (0 when nothing is wired), y = how much of the HALO the
  // mask takes — 1 cuts it as hard as the body, 0 lets all of it bleed.
  float4 mask;

  VcrGrade grade;
};

// This effect predates the shared NeonStyle block and keeps its own historical
// float4 packing, so the style is assembled here rather than embedded in the
// cbuffer. Same bytes in, same bytes out — three_walls, which is new, embeds
// NeonStyle directly instead.
NeonStyle neon_style() {
  NeonStyle st;
  st.line_hw     = neon0.x;
  st.line_gain   = neon0.y;
  st.core_whiten = neon0.z;
  st.halo_r      = neon0.w;
  st.halo_gain   = neon1.x;
  st.falloff     = neon1.y;
  st.corner_r    = neon1.z;
  st.aa          = neon1.w;
  st.fill_gain   = misc.x;
  st.halo_smooth = fills.w;
  st.pad0        = 0.0;
  st.pad1        = 0.0;
  return st;
}

// --- The release ghosts ---------------------------------------------------
// The throw: the same three quads again. In Grow they are flung outward off
// the stack and ring down; in Strobe they do not move at all and flam one at a
// time as bare wireframe. The host has already worked out where they are, how
// open they have gone and how bright each one is this frame — everything here
// is a pure function of that, so a ghost reads as an object with a life while
// costing no state at all.
//
// Deliberately NOT nano_neon_quad. A ring has no inside: no fill to flood, no
// mask to occlude with, and no core — by the time you can see one it is
// already past being a tube. So the exact SIGNED field, the corner rounding
// and the interior light-sum are all work with nothing to show for it, and
// what is left is the cheap half: unsigned distance to four edges, through one
// halo. That is the whole shape.
float ring_at(float2 p, int i) {   // ghost i's outline, unshaded
  float4 r0 = ghosts[i * 2 + 0];
  float4 r1 = ghosts[i * 2 + 1];
  float2 a = r0.xy, b = r0.zw, c = r1.xy, d = r1.zw;
  float dist = min(min(nano_neon_seg_dist(p, a, b), nano_neon_seg_dist(p, b, c)),
                   min(nano_neon_seg_dist(p, c, d), nano_neon_seg_dist(p, d, a)));
  return nano_neon_halo_profile(dist, rel.x, rel.y);
}

// --- Glimmer --------------------------------------------------------------
// Travelling glints, multiplied into each plane's EMISSION rather than
// composited over the finished picture. That is the point of doing it here at
// all: emission scales the line core, the halo and the fill together, so a
// glint crossing a tube brightens the glow around it as well and reads as
// light in the tube instead of a highlight pasted on top.
//
// These are PARTICLES. The host owns their lives (see
// <sketch/three_planes_glints.h>) and hands us however many are in flight,
// already projected onto the travel axis; all that is left here is to add up
// what they look like. Each is a bright band with a darker, wider wake behind
// it — a glint alone gets brighter, a glint with a wake sweeps CONTRAST past,
// which is what the eye reads as a moving highlight on a surface.
//
// The per-glint kernel — and the reasoning behind its shape — lives in
// shaders_common/nano_glint.hlsl, because the WALL pass lights itself with the
// same glints and the two must not drift apart.
float glimmer_at(float2 p) {
  float axis = dot(p, glim0.xy);

  // BRANCHLESS AND FULLY UNROLLED. An early `return` for the idle case looks
  // free and is not: DXC compiles one inside a function into a local naga
  // rejects ("has a type that can't be stored in a local variable"), the
  // SPIR-V -> WGSL translation fails, and the whole effect silently renders
  // nothing on WebGPU. A dead slot carries zero gain instead.
  float m = 0.0;
  [unroll]
  for (int i = 0; i < 8; i++) m += nano_glint_at(axis, glints[i]);

  // Clamped at 0 so a deep wake extinguishes a plane rather than inverting it
  // — emission is a multiplier on light, and there is no negative light.
  return max(0.0, 1.0 + m);
}

// --- The mask -------------------------------------------------------------
// A screen-space stencil laid over the stack — text, a logo, anything with a
// shape. Its weight is ALPHA times LUMA, and it wants both: a shape that is
// present but black is not a mask, and neither is a bright shape that is not
// there. Taking the product is what lets an ordinary rendered logo work as-is,
// with no separate matte to author and keep in step.
//
// Sampled in SCREEN space, not in the stack's, so what you see in the mask
// input is where it lands. Nearest, and clamped: no sampler, and a mask of a
// different size than the output simply stretches over it.
//
// Branchless, like glimmer_at, and for the same reason — an early return
// inside a function is what DXC turns into a local that naga refuses, and the
// whole effect then renders nothing on WebGPU.
float mask_at(float2 uv) {
  uint mw, mh;
  maskTex.GetDimensions(mw, mh);
  float2 sz = float2(max(mw, 1u), max(mh, 1u));
  int2 mp = int2(clamp(uv * sz, float2(0.0, 0.0), sz - 1.0));
  float4 t = maskTex.Load(int3(mp, 0));
  float luma = dot(t.rgb, float3(0.299, 0.587, 0.114));
  return saturate(t.a * luma * mask.x);
}

float2 corner_of(int i, int k) {
  float4 row = corners[i * 2 + (k >> 1)];
  return (k & 1) ? row.zw : row.xy;
}

NanoNeonField plane_field(float2 p, int i) {
  return nano_neon_quad(p, corner_of(i, 0), corner_of(i, 1),
                           corner_of(i, 2), corner_of(i, 3),
                        neon1.z, neon0.w, neon1.y, fills.w);
}

// --- The resolve ----------------------------------------------------------
// Bottom-to-top, explicitly ordered. `acc *= (1 - A)` is what lets a black
// plane eat the glow of everything beneath it; `acc += E` immediately after is
// what keeps its OWN outline and halo alive over that black. Fixed-function
// blend cannot express both in one draw — this loop is the effect.
float3 resolve(float2 p, float3 base) {
  NeonStyle st = neon_style();
  // TWO accumulators, occluded identically and summed at the end — so with no
  // mask this is exactly the single-accumulator resolve it replaces. The split
  // exists because the mask below has to take the BODY of the neon and leave
  // some of the halo, and a sum cannot be un-summed.
  float3 body = base;
  float3 halo = float3(0.0, 0.0, 0.0);
  // One sample for all three planes: a glint is a property of the SCREEN, a
  // light sweeping across the whole installation, not of any one plane.
  float glint = glimmer_at(p);

  [unroll]
  for (int i = 0; i < 3; i++) {
    NanoNeonField f = plane_field(p, i);
    float A;
    float3 h;
    float3 E = nano_neon_quad_emit_split(f, plane_color[i].rgb,
                                         plane_color[i].w * glint,
                                         fills[i], st, A, h);
    body = body * (1.0 - A) + E;
    halo = halo * (1.0 - A) + h;
  }

  // The ghosts go on TOP of the resolve, additively and without occluding
  // anything. They are light already thrown — nothing left behind can mask
  // them, and they have no body to be masked.
  //
  // They take the SAME glint the planes do. A glint is a property of the
  // screen — a light sweeping across the whole installation — and a throw is
  // still the installation: without this, a mode that mutes the tower and
  // shows only ghosts is a mode the glints cannot touch at all, and they hang
  // there over the picture doing nothing to it.
  //
  // And they land on the HALO side of the split, because that is what they
  // are: ring_at is the halo profile and nothing else. So a stencil lets them
  // spill over its edge exactly as it lets a tube's glow spill.
  [unroll]
  for (int k = 0; k < 3; k++)
    halo += plane_color[k].rgb * (ring_at(p, k) * ring_gain[k] * glint);

  // The stencil, applied the way a fill of -1 applies: what it covers is taken
  // AWAY rather than drawn over, base included, so the shape reads as a hole
  // cut through the picture rather than as a sticker on it. The halo is only
  // partly taken — a tube behind a letter still throws light around the
  // letter's edges, and that spill is the whole difference between a mask that
  // sits in the scene and one that sits on the glass.
  float m = mask_at(nano_cover_square_to_uv(p, view.zw));
  return body * (1.0 - m) + halo * (1.0 - m * mask.y);
}

[numthreads(8, 8, 1)]
void main(uint3 gid : SV_DispatchThreadID) {
  uint W, H;
  outputTex.GetDimensions(W, H);
  if (gid.x >= W || gid.y >= H) return;

  float2 vp     = view.xy;
  float2 aspect = view.zw;
  float2 uv     = nano_pixel_to_uv(float2(gid.xy), vp);
  float2 p      = nano_uv_to_cover_square(uv, aspect);

  float4 src   = inputTex.Load(int3(int(gid.x), int(gid.y), 0));
  float  in_op = misc.z;
  float3 base  = src.rgb * in_op;

  int debug_mode = int(misc.w + 0.5);
  if (debug_mode == 1) {
    // Raw distance field of the nearest plane, banded so the morphology of
    // the corners is legible.
    float d = 1e9;
    [unroll]
    for (int i = 0; i < 3; i++) d = min(d, abs(plane_field(p, i).sd));
    float bands = frac(d * 20.0);
    outputTex[gid.xy] = float4(bands.xxx * saturate(1.0 - d * 2.0), 1.0);
    return;
  }
  if (debug_mode == 2) {
    // Flat per-plane fill, no glow and no grade — checks the projection and
    // the stacking order on their own.
    float3 flat_c = base * 0.15;
    [unroll]
    for (int i = 0; i < 3; i++) {
      float ins = saturate(0.5 - plane_field(p, i).sd / max(neon1.w, 1e-6));
      float3 key = float3(i == 0 ? 1.0 : 0.0, i == 1 ? 1.0 : 0.0, i == 2 ? 1.0 : 0.0);
      flat_c = lerp(flat_c, key, ins * 0.75);
    }
    outputTex[gid.xy] = float4(flat_c, 1.0);
    return;
  }

  float3 c;
  float bleed = misc.y;
  if (bleed > 0.0) {
    // Analytic VCR chroma split: re-evaluate the WHOLE resolve at three
    // horizontally-offset positions and keep one channel from each. Exact
    // separation of outline, halo AND masking — no blur kernel involved.
    float o = bleed * 0.02;
    c.r = resolve(p + float2(-o, 0.0), base).r;
    c.g = resolve(p,                   base).g;
    c.b = resolve(p + float2( o, 0.0), base).b;
  } else {
    c = resolve(p, base);
  }

  float3 graded = nano_vcr_grade(c, uv, grade);
  // Stay layerable: alpha carries whatever the input had plus whatever we
  // emitted, so the stack composites correctly when used as an overlay.
  float  a = saturate(max(src.a * in_op,
                          max(graded.r, max(graded.g, graded.b))));
  outputTex[gid.xy] = float4(graded, a);
}
