// warp.recompose — solve pass.
//
// One thread. Reduces the latched stats into the nine per-cell correction
// vectors, and publishes the imbalance scalars.
//
// Runs EVERY frame, unlike accumulate/weigh (which are host-gated to the update
// rate). That split is deliberate: only the CONTENT-derived quantities
// (centroid, cell masses) are temporally smoothed, and everything downstream —
// the power point, G, R', the clamp scale, D_k — is re-derived from the
// smoothed values each frame. So `correct` / `spread` / `axis` / `distance`
// respond to modulation instantly while the analysis itself still eases. If D_k
// were smoothed directly, a fast `correct` modulation would lag and smear.
//
// ---- The correction math ----
//
// If every cell translates by the same D, the whole content translates by D, so
// the saliency centroid moves by exactly D. Once cells move differently the
// centroid only moves by the MASS-WEIGHTED average:
//
//     c' = (1/W) Σ_k Σ_{p∈k} w_p (p + D_k) = c + Σ_k m_k D_k
//
// So the per-cell redistribution term is mean-removed, R'_k = R_k − Σ_j m_j R_j,
// giving Σ_k m_k R'_k = 0 identically and therefore
//
//     Σ_k m_k D_k = s · gain · (µ ⊙ G)
//
// independent of `spread`. `spread` redistributes cells AROUND a guaranteed
// mean instead of fighting it, and the axis mask µ (componentwise linear)
// commutes with the sum, so masking one axis preserves the guarantee exactly on
// the other. The travel cap is ONE global scale s rather than a per-cell clamp,
// because clamping each D_k independently would perturb the weighted mean and
// break the identity.
//
// Since |c' − P*| = |1 − s·gain|·|G| on the unmasked axes, the imbalance
// strictly shrinks for 0 < s·gain < 2 and is merely reflected at exactly 2.
// With correct ∈ [-1,1], overshoot ∈ [1,2] and s ≤ 1 the product cannot exceed
// 2, so the parameter space CANNOT express a divergent setting.
//
// Exact in the model; approximate in the render (content translated off-frame
// is lost, overlaps double-count, and "Original" fills re-inject saliency). All
// degrade gracefully — the guarantee never inverts.

#include "common.hlsl"

StructuredBuffer<int>     stats : register(t0);
RWStructuredBuffer<float> solve : register(u1);

cbuffer U : register(b2) {
  float aspect_x, aspect_y;
  float correct;       // [-1,1] signed; 0 = passthrough
  float overshoot;     // [1,2] multiplier
  float spread;        // [0,1] uniform shift ↔ per-cell redistribution
  float distance;      // [0,1] travel cap, fraction of the short half-axis
  float axis;          // 0 both / 1 X only / 2 Y only
  float center_bias;   // [0,1] how empty the ideal centre cell is
  float alpha;         // temporal EMA coefficient this frame (1 = snap)
  float _p0, _p1, _p2;
};

[numthreads(1, 1, 1)]
void main() {
  float2 aspect = float2(aspect_x, aspect_y);
  float2 E      = rc_extent(aspect);

  // ---- Raw measurement ----
  float  Wsum   = float(stats[RC_B_W]) / RC_SCALE_W;
  float2 numer  = float2(float(stats[RC_B_WX]), float(stats[RC_B_WY])) / RC_SCALE_W;
  float2 c_raw  = (Wsum > 1e-5) ? (numer / Wsum) : float2(0.0, 0.0);

  float m_raw[9];
  float msum = 0.0;
  int k;
  [unroll] for (k = 0; k < 9; ++k) {
    m_raw[k] = max(float(stats[RC_B_M + k]) / RC_SCALE_W, 0.0);
    msum += m_raw[k];
  }
  float minv = 1.0 / max(msum, 1e-5);
  [unroll] for (k = 0; k < 9; ++k) m_raw[k] *= minv;

  bool haveStats = (stats[RC_A_N] > 0) && (msum > 1e-5);

  // ---- Temporal smoothing (the only stateful part) ----
  float a = (solve[RC_S_INIT] < 0.5) ? 1.0 : saturate(alpha);
  if (!haveStats) a = 0.0;               // nothing measured yet — hold

  float2 c_prev = float2(solve[RC_S_CX], solve[RC_S_CY]);
  float2 c      = lerp(c_prev, c_raw, a);

  float m[9];
  [unroll] for (k = 0; k < 9; ++k) {
    float prev = solve[RC_S_M + k];
    if (solve[RC_S_INIT] < 0.5) prev = 1.0 / 9.0;   // neutral seed
    m[k] = lerp(prev, m_raw[k], a);
  }

  // ---- Target and global correction ----
  float2 P = rc_nearest_power(c, E);
  float2 G = P - c;

  // ---- 3x3 distribution error, |·| halved into [0,1] ----
  float err = 0.0;
  [unroll] for (k = 0; k < 9; ++k) err += abs(m[k] - rc_ideal(k, center_bias));
  err = saturate(err * 0.5);

  // ---- Per-cell redistribution ----
  //
  // For an outer cell the vector to its nearest power point is exactly ±E/3 on
  // each axis, so the direction collapses to a SIGN per axis. The centre
  // row/column is equidistant from all four power points, so it needs a
  // tie-break: it donates toward whichever outer band is in deficit. (Without
  // this the centre cell — the one the rule of thirds most wants to empty —
  // would have a degenerate direction and never move.)
  float colM[3], rowM[3];
  [unroll] for (k = 0; k < 3; ++k) { colM[k] = 0.0; rowM[k] = 0.0; }
  [unroll] for (k = 0; k < 9; ++k) { colM[k % 3] += m[k]; rowM[k / 3] += m[k]; }

  float ideal0 = (1.0 + center_bias * 0.5) / 3.0;   // outer band ideal (Σ over the band)
  float ex0 = colM[0] - ideal0, ex2 = colM[2] - ideal0;
  float ey0 = rowM[0] - ideal0, ey2 = rowM[2] - ideal0;
  float sigx = (ex0 >= ex2) ? 1.0 : -1.0;   // left surplus → centre donates rightward
  float sigy = (ey0 >= ey2) ? 1.0 : -1.0;

  float2 R[9];
  float2 Rbar = float2(0.0, 0.0);
  [unroll] for (k = 0; k < 9; ++k) {
    int i = k % 3, j = k / 3;
    float rx = (i == 0) ? 1.0 : ((i == 2) ? -1.0 : sigx);
    float ry = (j == 0) ? 1.0 : ((j == 2) ? -1.0 : sigy);
    // Bounded gain: 9x so a typical surplus reads O(1), clamped so |R| ≤ E/3
    // by construction and no per-cell clamp is ever needed downstream.
    float g = clamp(9.0 * (m[k] - rc_ideal(k, center_bias)), -1.0, 1.0);
    R[k] = float2(rx * E.x / 3.0, ry * E.y / 3.0) * g;
    Rbar += m[k] * R[k];
  }

  // ---- Assemble, mask, then cap ----
  int ax = (int)(axis + 0.5);
  float2 mu = float2((ax == 0 || ax == 1) ? 1.0 : 0.0,
                     (ax == 0 || ax == 2) ? 1.0 : 0.0);

  float gain = correct * overshoot;

  float2 D[9];
  float  maxlen = 0.0;
  [unroll] for (k = 0; k < 9; ++k) {
    D[k] = mu * (gain * (G + spread * (R[k] - Rbar)));
    maxlen = max(maxlen, length(D[k]));
  }
  // The cap is computed from the MASKED vectors, so a suppressed axis never
  // eats the travel budget.
  float dmax = distance * min(E.x, E.y);
  float s = (maxlen > 1e-6) ? min(1.0, dmax / maxlen) : 1.0;

  // ---- Latch ----
  solve[RC_S_BAL_X]   = clamp(G.x / max(E.x, 1e-5), -1.0, 1.0);
  solve[RC_S_BAL_Y]   = clamp(G.y / max(E.y, 1e-5), -1.0, 1.0);
  solve[RC_S_CELLERR] = err;
  solve[RC_S_VALID]   = haveStats ? 1.0 : 0.0;
  solve[RC_S_CX]      = c.x;
  solve[RC_S_CY]      = c.y;
  solve[RC_S_PX]      = P.x;
  solve[RC_S_PY]      = P.y;
  solve[RC_S_GX]      = G.x;
  solve[RC_S_GY]      = G.y;
  solve[RC_S_SCALE]   = s;
  solve[RC_S_INIT]    = haveStats ? 1.0 : solve[RC_S_INIT];
  [unroll] for (k = 0; k < 9; ++k) solve[RC_S_M + k] = m[k];
  [unroll] for (k = 0; k < 9; ++k) {
    solve[RC_S_D + 2 * k]     = s * D[k].x;
    solve[RC_S_D + 2 * k + 1] = s * D[k].y;
  }
}
