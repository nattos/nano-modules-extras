// three_planes' IMPACT LIGHT — what the stack throws onto the walls beside it.
//
// The stack sits inside a hollow room and this is one of its side walls, flat
// on, from outside: horizontal is depth through the room, vertical is height.
// Two dispatches, one per wall.
//
// It is a LIGHT MODEL, not a picture of the stack. Nothing here draws a quad.
// What it draws is what four lengths of glowing tube do to a wall a little way
// off, the way a muzzle flash lights a corridor.
//
// EACH RING IS FOUR TUBES, in the room, turned by the orbit — not one bar-shaped
// emitter. That is the whole of it, and everything the light does comes out of
// it rather than being drawn on afterwards:
//
//   * Distance varies ALONG the ring. Turned off square, one corner is nearer
//     the wall than the other and the pool leans that way, brighter and tighter
//     at the near end. A single emitter at one distance can only ever lay down
//     a flat bar.
//   * A tube throws across itself and nothing along its length, so a length
//     pointing AT the wall barely lights it. Square on, the near edge does all
//     the work and the two running away contribute almost nothing; turned to
//     45 the four share it and the pool breaks into the diamond the ring
//     actually is.
//   * That also breaks the two walls apart. A square turned about its own axis
//     presents the same silhouette to both — which is why the orbit was
//     ignored here at first — but the near-corner PATTERN is mirrored between
//     them, so at any angle but square-on or 45 the two are different pictures.
//   * ELEVATION TILTS THE RINGS, because that is what the picture is a picture
//     of. An orthographic camera raised by phi over a flat deck and a camera on
//     the horizon looking at a deck tilted by phi draw the SAME main output —
//     but they are different rooms, and only the second one has anything for a
//     side wall to see. Flat on the horizon a ring is edge-on, every part of it
//     at one height, and it lays a thin hard line; tilt it and its near edge
//     runs uphill at exactly tan(phi), so the pool slants. Squashing the wall's
//     vertical axis instead — which is what this did first — gets the floors
//     to the right heights and nothing else: no slant at any angle, and flat
//     on the horizon it came out at its most DIFFUSE, which is backwards.
//
//     Each ring tilts about ITS OWN CENTRE, and that part is a cheat: the
//     centres stay on the vertical however far the deck goes over. See the
//     note in main.cpp — tilting the stack as one body is the honest version
//     and it strings the pools out diagonally, which loses the isometric read
//     the whole card is built on.
//
// The falloff is inverse square times the GRAZING cosine — what a surface
// catches is not how much light reaches it but how squarely — and the two
// together go as 1/d^3. That extra power is most of why this reads as light
// landing on something rather than a glow drawn over it. Under all of it, one
// wide dim wash per ring: the light that has bounced rather than arrived. A
// room without one is a black void with lamps floating in it.
//
// The tubes' own stacked-exponential halo was the obvious thing to reach for
// and it is wrong for a wall: its widest octave barely falls off across a
// room, so every floor laid down the same flat pedestal and the three pools
// melted into one bright rectangle. Inverse square has a long tail and no
// pedestal, which is the difference.

#include "nano_coords.hlsl"
#include "nano_glint.hlsl"

RWTexture2D<float4> outputTex : register(u0);

/// Rings 0-2 are the floors; 3-5 are the release ghosts, which the host has
/// already flown, opened and gated — from here it is one path.
static const int kNanoWallRings = 6;

cbuffer WallUniforms : register(b1) {
  float4 ring[6];      // rgb = colour, w = level
  float4 ring_g[6];    // x, y = the ring's centre (height, depth); z = source radius
  // The four corners of each ring, in the ROOM: turned by the orbit and tilted
  // by the elevation, so each carries its own height. Ring r takes rows 4r..4r+3.
  float4 ring_c[24];
  float4 wall;   // wash reach, wash weight, gain, the nearest tube's distance
  float4 look;   // warmth, which wall (-1 left, +1 right), -, wall x
  float4 glim;   // the glints' heading across the floor
  float4 glints[8];
};

float3 ring_corner(int r, int k) { return ring_c[r * 4 + k].xyz; }

/// What one length of tube throws at one point on the wall.
///
/// `soft2` is the source's own radius squared — a tube has thickness, and a
/// thrown ring has opened out besides, so nothing here ever converges on a
/// line. `g2` normalises against the NEAREST tube in the room this frame, so
/// the geometry shapes the pool without also setting the exposure.
float tube_light(float3 P, float3 A, float3 B, float soft2, float g2) {
  float3 e = B - A;
  float ee = max(dot(e, e), 1e-8);
  float t = saturate(dot(P - A, e) / ee);
  float3 d = P - (A + e * t);
  float q = dot(d, d) + soft2;
  float inv = rsqrt(q);
  float3 dh = d * inv;

  // How squarely the wall faces it. The wall is x-facing, so the cosine is
  // just the direction's own x and there is no dot product to write out.
  float face = abs(dh.x);
  // How much the tube throws this way. A line source radiates in the plane
  // across itself and nothing at all along its length.
  float3 eh = e * rsqrt(ee);
  float ax = dot(eh, dh);
  float along = sqrt(saturate(1.0 - ax * ax));

  // How much TUBE there is. Everything above is the nearest point of the edge
  // and says nothing about its length, which is fine while an edge is long
  // compared to how far off it is and wrong as soon as it is not: a tube
  // shortening toward nothing goes on throwing a full point's worth of light.
  // That shows up the moment Depth collapses a ring — the two edges running
  // away from the wall shorten to nothing and were left standing as two bright
  // dots on the bar. A short tube throws less; at zero length it throws
  // nothing, and a long one is unchanged.
  float len = sqrt(ee);
  float extent = len / (len + sqrt(q));

  return g2 * face * along * extent / q;
}

[numthreads(8, 8, 1)]
void main(uint3 gid : SV_DispatchThreadID) {
  uint W, H;
  outputTex.GetDimensions(W, H);
  if (gid.x >= W || gid.y >= H) return;

  float reach  = max(wall.x, 1e-3);
  float bounce = wall.y;
  float gain   = wall.z;
  float g2     = wall.w * wall.w;

  float2 vp = float2(float(W), float(H));
  float2 uv = nano_pixel_to_uv(float2(gid.xy), vp);
  float m = max(vp.x, vp.y);
  float2 aspect = float2(m / (2.0 * vp.x), m / (2.0 * vp.y));
  float2 sq = nano_uv_to_cover_square(uv, aspect);

  // The wall point, in the room. Cover-square straight through: the room is
  // measured in the picture's own units, so a floor's pool lands at the height
  // that floor is drawn at with nothing to convert.
  //
  // Depth runs OUTWARD FROM THE BACK WALL on each side, which is why it carries
  // the side's sign: the far end of a side wall abuts the back one, so it wants
  // to be on the inner edge of its own picture — the right of the left wall and
  // the left of the right wall. Laid out left / main / right, the room is then
  // continuous across both seams. (Same convention as source.mesh.three_walls;
  // running both the same way put the two far ends on the same side and the
  // room folded in half at one of the corners.)
  float3 P = float3(look.y * look.w, -sq.y, -look.y * sq.x);

  // The glints, sweeping THROUGH the room rather than across a picture. Their
  // travel angle is read as a heading on the floor, so a glint crossing the
  // stack reaches this wall and the far one at different moments.
  float axis = look.y * look.w * glim.x + sq.x * glim.y;
  float mg = 0.0;
  [unroll]
  for (int gi = 0; gi < 8; gi++) mg += nano_glint_at(axis, glints[gi]);
  float glint = max(0.0, 1.0 + mg);

  float3 c = float3(0.0, 0.0, 0.0);
  float r2 = reach * reach;
  [unroll]
  for (int r = 0; r < kNanoWallRings; r++) {
    float lvl = ring[r].w;
    // A dark floor and a spent ghost cost nothing: the level is a uniform, so
    // every thread takes the same side of this and the branch is free.
    if (lvl > 1e-4) {
      float soft2 = ring_g[r].z * ring_g[r].z;

      float lit = 0.0;
      [unroll]
      for (int k = 0; k < 4; k++)
        lit += tube_light(P, ring_corner(r, k), ring_corner(r, (k + 1) & 3),
                          soft2, g2);

      // The bounce: the room answering, from the ring as a whole rather than
      // from any one tube of it. No grazing term — it arrives from everywhere,
      // so there is no angle for a surface to be square to.
      float3 dc = P - float3(0.0, ring_g[r].x, ring_g[r].y);
      lit += bounce * (r2 / (r2 + dot(dc, dc)));

      c += ring[r].rgb * (lvl * glint * lit);
    }
  }
  c *= gain;

  // --- Tone -----------------------------------------------------------------
  // A soft knee on the BRIGHTEST CHANNEL, with every channel scaled by the same
  // factor. Per-channel compression is what mutes a coloured light — the strong
  // channel is squashed hardest and the hue crawls toward white on its own — and
  // this instrument's whole palette is saturated. Scaling together keeps the
  // colour exactly and only takes the level.
  float e = max(c.r, max(c.g, c.b));
  if (e > 1.0) {
    float over = e - 1.0;
    c *= (1.0 + over / (1.0 + over)) / e;
  }

  // ...and then the core goes WARM, which is the part that makes it read as
  // light rather than as a coloured shape. Real light blows its centre out
  // toward white and takes a little warmth with it as it does; the surround
  // keeps its hue outright, so nothing is washed out to get there.
  float hot = saturate((e - 0.7) * 1.4);
  float3 warm = float3(1.0, 0.92, 0.82) * max(c.r, max(c.g, c.b));
  c = lerp(c, warm, hot * saturate(look.x));

  // A dither well under one code value. The pools are wide and smooth, and
  // this is going on a wall several metres across where 8-bit banding would be
  // the most artificial thing in the picture. Integer hash, so both backends
  // land on the same noise.
  uint h = gid.x * 1973u + gid.y * 9277u + 26699u;
  h = (h ^ (h >> 13u)) * 1274126177u;
  c += (float(h & 0xFFFFu) / 65535.0 - 0.5) * (1.5 / 255.0);

  // Opaque: black here is a projector showing nothing, not a hole.
  outputTex[gid.xy] = float4(saturate(c), 1.0);
}
