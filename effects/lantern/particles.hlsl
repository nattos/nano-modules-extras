// source.mesh.lantern — the mica flakes: shared layout + field helpers.
//
// ~1000 flakes of mica, grit and silt (the poster's dust, part_dust.py) flying
// around the structure, driven by the sweep knob (main.cpp). Every flake is
// simulated (p_update), splatted into a coarse PRESENCE grid the fog reads as
// dust (p_resolve), shaded per flake (p_shade) and drawn as an EMBER — a
// hard-edged capsule from where it is back along where it came from, tapering
// and dimming to its tail, a sub-pixel flake keeping its size as coverage —
// into an HDR layer the composite lays over the scene (spark_vs / spark_fs).
//
// THE MOTION is regimes blended by the knob:
//   ring     — choked: ballistic orbits in a loose horizontal ring, under a
//              soft nonlinear field that only bites well away from it.
//   jet      — sustaining: free flakes ride a smoke-ring flow about the jet
//              axis — drawn in, blown through the core, looped back round.
//   stream   — sustaining: blocks are magnets (stronger the further they
//              strain out, and in a glint's band). A flake caught near one
//              is BOUND to that block: steered (not pushed) at high speed
//              round its own rounded shell, about an axis picked per block
//              (one it does not jut along, its own sense) — across the outer
//              face and through the seams of a flush block, the whole loop
//              round a jutting one — in its own lane above it (lanes
//              spread like the smoke lines over a wing; the deeper the knob
//              is in its sweep, the wider and slower they run),
//              drawn sideways into the ember grooves, pushed (not steered)
//              so it keeps its momentum. Its lane keeps an AIR GAP: never
//              nearer the block than its altitude, lifted toward each edge so
//              it turns the edge in a gentle arc (a neighbor in its way it
//              slips through, out of sight a moment). The field leads each
//              turn, supplying most of the push that keeps it level as the
//              lane bends. Its ALTITUDE is held by a spring for the rest —
//              damped, hardening as it is pushed off, soft deep in the sweep
//              and nearly pinching in the middle — so it rides up over an
//              edge under the load and springs back with a wobble; only a big
//              excursion tears it loose. In flight, a pull that ramps up hard
//              as it closes (the magneto envelope's eps/(d^2+eps)) swings it
//              back in; striking its block it ricochets. The strained blocks hold hardest. Over a face
//              looking out of the structure and square to its block's spin
//              axis (where a stream would only circle), an updraft lifts its
//              lane: the flakes vortex up off it until they fly off, and a
//              soft gravity outside the structure arcs them back in.
//   shell    — the thrown tail: released flakes are pushed onto a sphere
//              around the structure (p_update), where p_shade shakes them at
//              the tail's falling frequency (a render-only offset whose speed
//              becomes the ember's length and the motion vectors).

#ifndef LANTERN_PARTICLES_HLSL
#define LANTERN_PARTICLES_HLSL

#include "common.hlsl"
#include "nano_hash.hlsl"

static const int LTN_P_ROWS = 5;     // state rows per flake (p_update)
static const int LTN_R_ROWS = 4;     // render rows per flake (p_shade -> spark)
static const int LTN_GRID = 32;      // presence grid, per axis (LTN_GRID in main.cpp)

// State rows: s0 (pos, captor box or -1), s1 (vel, bound 0/1),
//             s2 (flake normal, tumble rate), s3 (tumble axis, susceptibility),
//             s4 (disc radius m, kind, ring jitter r, ring jitter z).

#include "part_ub.hlsl"

// Hash -> [0,1) per flake and stream.
float ltn_prand(uint id, uint k) {
  return float(nano_uhash(id * 0x9E3779B9u + k * 0x85EBCA6Bu + 0x68E31DA4u) & 0xFFFFFFu) *
         (1.0 / 16777216.0);
}

float3 ltn_prand_dir(uint id, uint k) {
  float z = ltn_prand(id, k) * 2.0 - 1.0;
  float ph = ltn_prand(id, k + 1u) * 6.2831853;
  float r = sqrt(max(1.0 - z * z, 0.0));
  return float3(r * cos(ph), r * sin(ph), z);
}

// ~N(0,1) from four uniforms (Irwin-Hall).
float ltn_pgauss(uint id, uint k) {
  return (ltn_prand(id, k) + ltn_prand(id, k + 1u) + ltn_prand(id, k + 2u) +
          ltn_prand(id, k + 3u) - 2.0) * 1.7320508;
}

// Signed distance from p to box i, and the outward unit normal of the
// nearest feature (world).
float ltn_box_sdf(int i, float3 p, out float3 n) {
  float4 r0 = boxes[i * LTN_BOX_ROWS + 0];
  float4 r1 = boxes[i * LTN_BOX_ROWS + 1];
  float4 r2 = boxes[i * LTN_BOX_ROWS + 2];
  float4 r3 = boxes[i * LTN_BOX_ROWS + 3];
  float3 o = p - r0.xyz;
  float3 lq = float3(dot(o, r1.xyz), dot(o, r2.xyz), dot(o, r3.xyz));
  float3 dq = abs(lq) - float3(r1.w, r2.w, r3.w);
  float3 s = select(lq < 0.0, -1.0, 1.0);
  float3 outside = max(dq, 0.0);
  float lo = length(outside);
  float3 g;
  float d;
  if (lo > 1e-6) {
    g = s * outside / lo;
    d = lo;
  } else {
    d = max(dq.x, max(dq.y, dq.z));
    g = (dq.x >= dq.y && dq.x >= dq.z) ? float3(s.x, 0.0, 0.0)
      : (dq.y >= dq.z ? float3(0.0, s.y, 0.0) : float3(0.0, 0.0, s.z));
  }
  n = g.x * r1.xyz + g.y * r2.xyz + g.z * r3.xyz;
  return d;
}

// World -> the camera's pixel grid (continuous, pixel centers at +0.5) for a
// camera given as three basis rows whose .w is its position. False behind it.
bool ltn_project(float4 cr, float4 cu, float4 cf, float3 p, out float2 px, out float w) {
  float3 rel = p - float3(cr.w, cu.w, cf.w);
  w = dot(rel, cf.xyz);
  px = float2(0.0, 0.0);
  if (w < 0.05) return false;
  float2 ndc = float2(dot(rel, cr.xyz), dot(rel, cu.xyz)) / w *
               (2.0 * cam_p.x) * float2(cam_p.y, cam_p.z);
  px = float2(0.5 * (ndc.x + 1.0), 0.5 * (1.0 - ndc.y)) * vp.xy;
  return true;
}

// Presence grid: world -> normalized grid coords.
float3 ltn_grid_uvw(float3 p) {
  return (p + grid.x) / (2.0 * grid.x);
}

#endif  // LANTERN_PARTICLES_HLSL
