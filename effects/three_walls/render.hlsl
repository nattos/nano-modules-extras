// source.mesh.three_walls — one room, rendered once per wall.
//
// The CPU has already done the geometry: it works out where each of the three
// frames lands on THIS dispatch's wall and hands us four corner points in
// cover-square coords, exactly the way three_planes does. So this shader
// neither knows nor cares which wall it is drawing — the back wall showing a
// growing rectangle and a side wall showing a bar sweeping past are the same
// code over different corners.
//
// That is the whole reason the geometry lives on the CPU. `nano_neon_quad`
// takes four arbitrary 2D points, so a rectangle and a zero-area line need no
// new shader code between them.
//
// A frame that has left this wall is not culled: its corners simply fall
// outside the picture, and the halo bleeding in from beyond the edge is what
// carries the light around the corner.

#include "nano_coords.hlsl"
#include "nano_neon_quad.hlsl"
#include "nano_vcr.hlsl"

Texture2D<float4>   inputTex  : register(t0);
RWTexture2D<float4> outputTex : register(u1);

cbuffer Uniforms : register(b2) {
  // Projected corners for THIS view, cover-square coords. Quad i occupies rows
  // 2i and 2i+1: (c0.xy, c1.zw) then (c2.xy, c3.zw), wound consistently.
  float4 corners[6];

  // rgb = quad colour, w = emission drive (already curved by the host, and
  // already zeroed for a quad that is not live in this view).
  float4 quad_color[3];

  float4 fills;   // xyz = signed fill per quad (+neon / -mask), w unused
  float4 depth;   // xyz = per-quad neon scale (see below), w unused
  float4 misc;    // chroma bleed, input opacity, debug mode, unused
  float4 view;    // vp_w, vp_h, aspect_x, aspect_y

  NeonStyle style;   // embedded VERBATIM — see nano_neon_quad.hlsl
  VcrGrade  grade;   // embedded VERBATIM — see nano_vcr.hlsl
};

// A real neon tube has a fixed thickness, so what reaches the eye scales with
// everything else in the picture: near, it is a fat bar with a wide bloom;
// far, a hairline with barely any. Holding the line width constant is what
// makes a receding frame read as a flat shrinking rectangle instead of an
// object going away from you — the glow carries as much of the depth cue as
// the size does.
//
// The scale is the frame's own apparent size, so it needs no special case at
// the corner: on the back wall that is how big the rectangle is, on a side wall
// it is 1/depth, and the two agree at the seam by construction.
NeonStyle quad_style(int i) {
  NeonStyle st = style;
  const float k = depth[i];
  st.line_hw  *= k;
  st.halo_r   *= k;
  st.corner_r *= k;
  // NOT the antialias width: that is one pixel, and a pixel is a pixel however
  // far away the thing it is smoothing happens to be.
  return st;
}

float2 corner_of(int i, int k) {
  float4 row = corners[i * 2 + (k >> 1)];
  return (k & 1) ? row.zw : row.xy;
}

NanoNeonField quad_field(float2 p, int i) {
  NeonStyle st = quad_style(i);
  return nano_neon_quad(p, corner_of(i, 0), corner_of(i, 1),
                           corner_of(i, 2), corner_of(i, 3),
                        st.corner_r, st.halo_r, st.falloff, st.halo_smooth);
}

// Near to far, so a nearer quad's mask can eat the glow of the ones behind it.
// The CPU sorts the rows into that order; here it is just the loop direction.
float3 resolve(float2 p, float3 base) {
  float3 acc = base;
  [unroll]
  for (int i = 0; i < 3; i++) {
    NanoNeonField f = quad_field(p, i);
    float A;
    float3 E = nano_neon_quad_emit(f, quad_color[i].rgb, quad_color[i].w,
                                   fills[i], quad_style(i), A);
    acc = acc * (1.0 - A) + E;
  }
  return acc;
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
  float  in_op = misc.y;
  float3 base  = src.rgb * in_op;

  int debug_mode = int(misc.z + 0.5);
  if (debug_mode == 1) {
    // Flat per-quad keys, no glow and no grade — isolates the projection and
    // the near-plane culling from the shading.
    float3 flat_c = base * 0.15;
    [unroll]
    for (int i = 0; i < 3; i++) {
      float ins = saturate(0.5 - quad_field(p, i).sd / max(style.aa, 1e-6));  // aa is pixel-space
      float3 key = float3(i == 0 ? 1.0 : 0.0, i == 1 ? 1.0 : 0.0, i == 2 ? 1.0 : 0.0);
      flat_c = lerp(flat_c, key, ins * 0.75 * saturate(quad_color[i].w * 1e6));
    }
    outputTex[gid.xy] = float4(flat_c, 1.0);
    return;
  }

  float3 c;
  float bleed = misc.x;
  if (bleed > 0.0) {
    // Analytic VCR chroma split, same as three_planes: re-evaluate the WHOLE
    // resolve at three horizontally-offset positions and keep one channel from
    // each. No blur kernel, so the outline and the halo separate exactly.
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
