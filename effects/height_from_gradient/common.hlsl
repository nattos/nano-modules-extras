// filter.height_from_gradient — shared math for all passes.
//
// Gradient-domain height reconstruction. We synthesize a 2D gradient field
// from the input, take its divergence, and solve the Poisson equation
// laplacian(h) = div(g) for the least-squares height whose gradient best
// matches g. The solve is a coarse-to-fine (FMG-lite) multigrid cascade.
//
// Multigrid spacing trick: a geometric Poisson coarsening scales the RHS by
// the squared grid spacing. Rather than feed a per-level spacing uniform into
// the (level-reused) Jacobi PSO — which WebGPU can't vary between dispatches
// in one submit — we PRE-SCALE the divergence pyramid. We store
//   F_k = (2^k)^2 * div_k
// so the Jacobi stencil is spacing-agnostic at every level:
//   h' = (hL + hR + hD + hU - F) / 4.
// Restriction then becomes a plain SUM of the 2x2 children (F_{k+1} = 4*mean
// = sum). See restrict.hlsl. Scalars live in the R channel of RGBA16F (R32F
// can't be sampled as Float on WebGPU).

#ifndef HFG_COMMON_HLSL
#define HFG_COMMON_HLSL

#include "nano_coords.hlsl"   // nano_pixel_to_cover_square (radial center, §1.5)
#include "nano_color.hlsl"    // nano_luminance

// Normalize a 2D vector, returning zero at (near-)zero length (the radial
// field is exactly zero at the center — keep it zero rather than NaN).
float2 hfg_normalize2(float2 v) {
  float l = length(v);
  return (l < 1e-7) ? float2(0.0, 0.0) : v / l;
}

// Clamped luma fetch at an integer texel (edge-extend boundary).
float hfg_luma_at(Texture2D<float4> tex, int2 p, int2 hi) {
  return nano_luminance(tex[uint2(clamp(p, int2(0, 0), hi))].rgb);
}

// Decode a 2D vector from a texel for the Motion / Normal Map / Gradient Field
// sources, which may be stored "strangely". channel_mode picks the layout
// (0 = RG, 1 = RG with Y flipped — the GL↔DX normal-map gotcha, 2 = AG, the
// BC5/DXT5nm swizzle). vector_sign: 0 = signed (0 is zero), 1 = unsigned
// (0.5 is zero → remap [0,1]→[-1,1]).
float2 hfg_decode_vec(float4 t, float channel_mode, float vector_sign) {
  float2 v = (channel_mode > 1.5) ? float2(t.a, t.g)   // AG
                                  : float2(t.r, t.g);  // RG (and RG flip-Y)
  if (vector_sign > 0.5) v = v * 2.0 - 1.0;            // unsigned → signed
  if (channel_mode > 0.5 && channel_mode < 1.5) v.y = -v.y;  // flip Y
  return v;
}

// Manual bilinear sample of the R channel in texel-index space (integer
// coordinate = texel center). Used by the prolongation upsample.
float hfg_bil_r(Texture2D<float4> tex, float2 p, int2 dims) {
  int2 i0 = (int2)floor(p);
  float2 f = p - floor(p);
  int2 hi = dims - 1;
  int2 a = clamp(i0,              int2(0, 0), hi);
  int2 b = clamp(i0 + int2(1, 0), int2(0, 0), hi);
  int2 c = clamp(i0 + int2(0, 1), int2(0, 0), hi);
  int2 d = clamp(i0 + int2(1, 1), int2(0, 0), hi);
  float c00 = tex[uint2(a)].x, c10 = tex[uint2(b)].x;
  float c01 = tex[uint2(c)].x, c11 = tex[uint2(d)].x;
  return lerp(lerp(c00, c10, f.x), lerp(c01, c11, f.x), f.y);
}

// Shared uniform layout (32 floats = 8 std140 rows). Only the gradient and
// present passes read it; the solver passes (divergence/restrict/jacobi/
// prolong) take no uniforms. Declared identically wherever it's bound so the
// two passes can't drift on field order.
#define HFG_UNIFORMS \
  float grad_gain;   float source;        float center_x;      float center_y;        \
  float aspect_x;    float aspect_y;      float present_mode;  float relief_scale;     \
  float light_x;     float light_y;       float light_z;       float light_gain;       \
  float ambient;     float mix_amount;    float height_scale;  float height_offset;    \
  float tint_r;      float tint_g;        float tint_b;        float debug_show_gradient; \
  float core_radius; float core_softness; float bias_mode;     float bias_x;           \
  float bias_y;      float edge_mode;     float edge_threshold; float edge_gain;       \
  float contour_density; float line_width; float channel_mode; float vector_sign;

#endif
