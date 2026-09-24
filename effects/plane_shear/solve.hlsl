// warp.plane_shear — solve pass (single thread).
//
// Reduces the accumulated stats into a dividing line (center + unit normal) for
// the selected algorithm, applies an optional fixed-angle override, then — only
// when `latch` is set — writes it into the persistent plane buffer as a HARD
// overwrite (stiff snap, never a lerp). Between updates the host skips this pass
// entirely and the plane is simply held.
//
// plane buffer (floats): [0,1]=center.xy  [2,3]=normal.xy  [4]=confidence  [5]=initialized

#include "common.hlsl"

StructuredBuffer<int>     stats  : register(t0);   // read-only
RWStructuredBuffer<float> plane  : register(u1);

cbuffer U : register(b2) {
  float algorithm;      // PS_ALG_*
  float lock_angle;     // 0/1 — fix the normal to `angle_rad`, solve position only
  float angle_rad;      // fixed normal angle (radians)
  float off_max;        // grid offset half-range
  float latch;          // 0/1 — commit to the plane buffer this frame
  float center_weight;  // 0..1 — bias the plane toward passing through the center
  float _p1, _p2;
};

float momf(int i) { return (float)stats[PS_MOM_BASE + i]; }
float gridf(int a, int o) { return (float)stats[PS_GRID_BASE + a * PS_NO + o]; }
float offOf(int oi) { return (((float)oi + 0.5) / PS_NO * 2.0 - 1.0) * off_max; }

// Centrality falloff for an offset bin: 1 at center → (1-center_weight) at the
// edge. Down-weights off-center candidates in the search algorithms.
float centralWeight(int oi) {
  return 1.0 - center_weight * saturate(abs(offOf(oi)) / off_max);
}
// Seam offset band half-width: NO/4 at center_weight=0 → 1 at center_weight=1
// (the band collapses toward the central offset).
int seamHalf() { return (int)lerp((float)(PS_NO / 4), 1.0, center_weight); }

// Unit eigenvector for the LARGER eigenvalue of the symmetric 2x2 [[a,b],[b,c]].
float2 dominantEigvec(float a, float b, float c) {
  float tr   = a + c;
  float disc = sqrt(max(tr * tr * 0.25 - (a * c - b * b), 0.0));
  float lam  = tr * 0.5 + disc;
  float2 v = float2(b, lam - a);
  if (dot(v, v) < 1e-12) v = float2(lam - c, b);
  if (dot(v, v) < 1e-12) v = float2(1.0, 0.0);
  return normalize(v);
}

[numthreads(1, 1, 1)]
void main(uint3 id : SV_DispatchThreadID) {
  int alg = (int)(algorithm + 0.5);
  bool lock = lock_angle > 0.5;

  float2 center = float2(0.0, 0.0);
  float2 normal = float2(1.0, 0.0);
  float  conf   = 0.0;

  if (alg == PS_ALG_DOMINANT) {
    float gw = max(momf(PS_M_GW), 1e-4);
    center = float2(momf(PS_M_GWX) / gw, momf(PS_M_GWY) / gw);
    // Plane normal = strongest-gradient direction ⇒ the line runs ALONG edges.
    normal = dominantEigvec(momf(PS_M_JXX), momf(PS_M_JXY), momf(PS_M_JYY));
    float tr = momf(PS_M_JXX) + momf(PS_M_JYY);
    float disc = sqrt(max(tr * tr * 0.25 -
                 (momf(PS_M_JXX) * momf(PS_M_JYY) - momf(PS_M_JXY) * momf(PS_M_JXY)), 0.0));
    conf = (tr > 1e-4) ? saturate(2.0 * disc / tr) : 0.0;
  } else if (alg == PS_ALG_PCA) {
    float w = max(momf(PS_M_W), 1e-4);
    float mx = momf(PS_M_WX) / w, my = momf(PS_M_WY) / w;
    center = float2(mx, my);
    float cxx = momf(PS_M_WXX) / w - mx * mx;
    float cxy = momf(PS_M_WXY) / w - mx * my;
    float cyy = momf(PS_M_WYY) / w - my * my;
    // Principal axis = max-variance eigenvector; plane normal ⟂ it.
    float2 axis = dominantEigvec(cxx, cxy, cyy);
    normal = float2(-axis.y, axis.x);
    float tr = cxx + cyy;
    float disc = sqrt(max(tr * tr * 0.25 - (cxx * cyy - cxy * cxy), 0.0));
    conf = (tr > 1e-6) ? saturate(2.0 * disc / tr) : 0.0;
  } else if (alg == PS_ALG_HOUGH) {
    // Strongest straight edge = argmax over the (theta, rho) vote grid, biased
    // toward central lines by center_weight.
    int bestA = 0, bestO = PS_NO / 2; float best = -1.0;
    [loop] for (int a = 0; a < PS_NA; ++a)
      [loop] for (int o = 0; o < PS_NO; ++o) {
        float v = gridf(a, o) * centralWeight(o);
        if (v > best) { best = v; bestA = a; bestO = o; }
      }
    float theta = (bestA + 0.5) / PS_NA * PS_PI;
    float rho   = offOf(bestO);
    normal = float2(cos(theta), sin(theta));
    center = rho * normal;
    conf = 1.0;
  } else { // PS_ALG_SEAM — lowest-energy straight cut (argmin, central band).
    int lo = PS_NO / 2 - seamHalf(), hi = PS_NO / 2 + seamHalf();
    int bestA = 0, bestO = PS_NO / 2; float best = 1e30;
    [loop] for (int a = 0; a < PS_NA; ++a)
      [loop] for (int o = lo; o < hi; ++o) {
        float v = gridf(a, o);
        if (v < best) { best = v; bestA = a; bestO = o; }
      }
    float psi = (bestA + 0.5) / PS_NA * PS_PI;
    float rho = offOf(bestO);
    normal = float2(cos(psi), sin(psi));
    center = rho * normal;
    conf = 0.5;
  }

  // ---- Fixed-angle override: keep the analyzed POSITION, fix the ANGLE. ----
  if (lock) {
    float theta = angle_rad;
    // Fold into [0, PI) so the fixed-angle row lookups line up with accumulate.
    if (theta < 0.0)      theta += PS_PI;
    if (theta >= PS_PI)   theta -= PS_PI;
    float2 n = float2(cos(theta), sin(theta));
    if (alg == PS_ALG_HOUGH || alg == PS_ALG_SEAM) {
      int a = clamp((int)(theta / PS_PI * PS_NA), 0, PS_NA - 1);
      int lo = (alg == PS_ALG_SEAM) ? (PS_NO / 2 - seamHalf()) : 0;
      int hi = (alg == PS_ALG_SEAM) ? (PS_NO / 2 + seamHalf()) : PS_NO;
      int bestO = PS_NO / 2; float best = (alg == PS_ALG_SEAM) ? 1e30 : -1.0;
      [loop] for (int o = lo; o < hi; ++o) {
        float v = gridf(a, o);
        if (alg == PS_ALG_HOUGH) v *= centralWeight(o);
        bool better = (alg == PS_ALG_SEAM) ? (v < best) : (v > best);
        if (better) { best = v; bestO = o; }
      }
      center = offOf(bestO) * n;
    }
    // Dominant / PCA keep their analyzed centroid; only the normal is replaced.
    normal = n;
  }

  if (dot(normal, normal) < 1e-8) normal = float2(1.0, 0.0);
  normal = normalize(normal);

  // Pull the plane's offset toward 0 so it passes through more of the center.
  // Keeps the orientation; only reduces the perpendicular distance from center.
  // Only engages in the upper half of the range (>0.5): the lower half is pure
  // selection bias (Hough/seam), the upper half additionally pulls to center.
  float pull = saturate((center_weight - 0.5) * 2.0);
  center = center - normal * dot(center, normal) * pull;

  if (latch > 0.5) {
    plane[0] = center.x; plane[1] = center.y;
    plane[2] = normal.x; plane[3] = normal.y;
    plane[4] = conf;
    plane[5] = 1.0;
  }
}
