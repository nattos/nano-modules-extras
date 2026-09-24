// filter.height_from_gradient — gradient generation pass.
//
// Synthesizes the source gradient field g(p) from the input. The `source`
// switch is the seam where gradient generators plug in; the solver and
// presenter never change.
//
//   Radial       — g points outward from an adjustable center, magnitude =
//                  luma. A bowl/cone field; core_radius tames the 1/r
//                  divergence singularity at the anchor.
//   Level Curves — treat the input as a contour map. A contour map IS a
//                  sampled height field: g = ∇h is PERPENDICULAR to the level
//                  curves, magnitude ∝ curve density. We recover the across-
//                  curve normal from a structure tensor (sign-agnostic, so it
//                  survives the gradient dipole at a thin line), resolve the
//                  uphill/downhill SIGN with a global bias (radial or a linear
//                  sweep), and set the magnitude per contour. Integrated by
//                  the Poisson solve, each crossed contour becomes a height
//                  step — a staircase the least-squares smoothing rounds off.
//   Motion / Normal Map / Gradient Field — the source already encodes a vector
//                  field. The host binds the right texture (the incoming
//                  render_outputs/motion rail for Motion, the input image
//                  otherwise) and channel_mode/vector_sign decode it. Motion &
//                  Gradient Field use the vector AS the gradient; Normal Map
//                  integrates a surface normal: g = -n.xy / n.z.
//
// Output: RG = g (gx, gy) at full res. Generally non-conservative, so the
// Poisson solve is a least-squares best-fit.

#include "common.hlsl"

Texture2D<float4>   inputTex : register(t0);   // full-res input
RWTexture2D<float4> gradOut  : register(u1);   // full-res gradient (RG)

cbuffer Uniforms : register(b2) { HFG_UNIFORMS };

[numthreads(8, 8, 1)]
void main(uint3 gid : SV_DispatchThreadID) {
  uint w, h;
  gradOut.GetDimensions(w, h);
  if (gid.x >= w || gid.y >= h) return;

  int2 hi = int2(int(w) - 1, int(h) - 1);
  int2 p  = int2(gid.xy);
  float2 sq = nano_pixel_to_cover_square(float2(gid.xy), float2(w, h),
                                         float2(aspect_x, aspect_y));

  float2 g = float2(0.0, 0.0);

  if (source > 1.5) {
    // ---- Vector sources: Motion (2) / Normal Map (3) / Gradient Field (4) ----
    // inputTex is whatever the host bound (the motion rail for Motion, the
    // input image otherwise). Decode the stored vector, then interpret it.
    float2 v = hfg_decode_vec(inputTex[gid.xy], channel_mode, vector_sign);
    if (abs(source - 3.0) < 0.5) {
      // Normal Map → gradient of the height the normal encodes. Reconstruct z
      // if it wasn't stored (a unit normal): n.z = sqrt(1 - |n.xy|^2).
      float nz = sqrt(max(1.0 - dot(v, v), 1e-4));
      g = -v / nz;
    } else {
      // Motion / Gradient Field: the vector IS the gradient.
      g = v;
    }
    g *= grad_gain;
  } else if (source < 0.5) {
    // ---- Radial ----
    float luma = nano_luminance(inputTex[gid.xy].rgb);
    float2 d = sq - float2(center_x, center_y);
    float r = length(d);
    float2 dir = hfg_normalize2(d);

    // Core smoothing — ramp the magnitude up from zero across core_radius so
    // the field grows smoothly out of the anchor (finite divergence) instead
    // of snapping to a unit vector. At core_radius≈0 this is a no-op.
    float radius = max(core_radius, 1e-5);
    float ramp = pow(smoothstep(0.0, radius, r), 1.0 + core_softness * 4.0);

    g = dir * luma * grad_gain * ramp;
  } else {
    // ---- Level Curves ----
    // Structure tensor over a 3x3 window of central-difference luma gradients.
    // J = sum( g g^T ). Its dominant eigenvector is the (undirected) across-
    // curve normal; squaring the gradient makes it robust to the sign dipole a
    // thin line produces.
    float Jxx = 0.0, Jxy = 0.0, Jyy = 0.0;
    [unroll] for (int dy = -1; dy <= 1; dy++)
    [unroll] for (int dx = -1; dx <= 1; dx++) {
      int2 q = p + int2(dx, dy);
      float lx = 0.5 * (hfg_luma_at(inputTex, q + int2(1, 0), hi) -
                        hfg_luma_at(inputTex, q + int2(-1, 0), hi));
      float ly = 0.5 * (hfg_luma_at(inputTex, q + int2(0, 1), hi) -
                        hfg_luma_at(inputTex, q + int2(0, -1), hi));
      Jxx += lx * lx;
      Jxy += lx * ly;
      Jyy += ly * ly;
    }
    // Dominant eigenvalue/eigenvector of the symmetric 2x2 tensor.
    float tr   = Jxx + Jyy;
    float disc = sqrt(max(tr * tr * 0.25 - (Jxx * Jyy - Jxy * Jxy), 0.0));
    float lam  = tr * 0.5 + disc;
    float2 n = float2(Jxy, lam - Jxx);
    if (dot(n, n) < 1e-12) n = float2(lam - Jyy, Jxy);   // degenerate fallback
    n = hfg_normalize2(n);

    // Resolve the uphill/downhill sign with a global bias.
    float2 bias = (bias_mode < 0.5)
        ? hfg_normalize2(sq - float2(center_x, center_y))   // Radial
        : float2(bias_x, bias_y);                           // Linear sweep
    if (dot(n, bias) < 0.0) n = -n;

    // Per-contour magnitude: equal step (thresholded) or proportional to edge
    // energy. sqrt(lam) ≈ local gradient strength.
    float energy = sqrt(max(lam, 0.0));
    float mag = (edge_mode < 0.5)
        ? ((energy > edge_threshold) ? 1.0 : 0.0)   // Uniform per contour
        : (energy * edge_gain * 8.0);               // Proportional

    g = n * mag * grad_gain;
  }

  gradOut[gid.xy] = float4(g, 0.0, 1.0);
}
