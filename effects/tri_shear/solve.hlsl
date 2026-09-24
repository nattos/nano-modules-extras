// warp.tri_shear — 3-line discovery (single thread).
//
// Reads the (angle,offset) energy grid and picks THREE lines forming a triangle.
// `size` sets a small TARGET incircle radius r_target. Each edge searches within
// its own sector for the line that best trades off feature strength against
// staying near r_target; `obliqueness` sets how wide that angular search is —
// 0 locks the edges at 120° (equilateral), 1 lets each tilt across its full 120°
// sector (freely OBLIQUE / scalene). Result is written (latch-gated, stiff).
//
// tri_buf (floats): per edge k → [k*4+0,1]=center.xy  [k*4+2,3]=normal.xy ; [12]=initialized

#include "../plane_shear/common.hlsl"

StructuredBuffer<int>     stats : register(t0);   // (angle,offset) grid
RWStructuredBuffer<float> tri   : register(u1);

cbuffer U : register(b2) {
  float algorithm;    // 1 = Strongest Edges (maximize energy) / 2 = Low-energy Seams (minimize)
  float size;         // 0..1 — triangle scale (maps to a small target incircle radius)
  float off_max;      // grid offset half-range
  float latch;        // 0/1 — commit this frame
  float obliqueness;  // 0 = equilateral (locked 120°) → 1 = free within the 120° sector
  float _p1, _p2, _p3;
};

static const float R_MIN    = 0.02;   // search bounds on outward distance
static const float R_MAX    = 0.65;
static const float R_LO     = 0.04;   // target incircle radius at size=0 (tiny)
static const float R_HI     = 0.45;   // target incircle radius at size=1 (moderate)
static const float W_TARGET = 3.0;    // how firmly size controls the radius
static const float TWO_PI_3 = 2.0943951023931953;   // 120° sector width

float gridf(int a, int o) { return (float)stats[PS_GRID_BASE + a * PS_NO + o]; }
float offOf(int oi) { return (((float)oi + 0.5) / PS_NO * 2.0 - 1.0) * off_max; }

[numthreads(1, 1, 1)]
void main(uint3 id : SV_DispatchThreadID) {
  bool wantMax = algorithm < 1.5;     // strongest edges maximize; seams minimize

  // Normalize energies so the target penalty is on a comparable [0,1] scale.
  float gmax = 1e-4;
  [loop] for (int a0 = 0; a0 < PS_NA; ++a0)
    [loop] for (int o0 = 0; o0 < PS_NO; ++o0)
      gmax = max(gmax, abs(gridf(a0, o0)));

  float r_target = lerp(R_LO, R_HI, saturate(size));

  float3 winPhi = float3(0.0, TWO_PI_3, 2.0 * TWO_PI_3);
  float3 winR   = float3(r_target, r_target, r_target);

  const int NANG = 24;
  float window = saturate(obliqueness) * TWO_PI_3;   // 0 → locked; 1 → full sector
  [unroll] for (int k = 0; k < 3; ++k) {
    float sectorC  = k * TWO_PI_3;               // sector centers: 0°, 120°, 240°
    float bestEdge = -1e30;
    float bestPhi  = sectorC;
    float bestR    = r_target;

    // Sweep the (obliqueness-scaled) window around the sector center — the edge
    // angle tilts to the strongest feature; wider window ⇒ more oblique.
    [loop] for (int j = 0; j < NANG; ++j) {
      float phi  = sectorC + ((j + 0.5) / NANG - 0.5) * window;
      float phiP = phi - PS_PI * floor(phi / PS_PI);              // fold to [0, PI)
      int   a    = clamp((int)(phiP / PS_PI * PS_NA), 0, PS_NA - 1);
      float phi2 = phi - 2.0 * PS_PI * floor(phi / (2.0 * PS_PI)); // [0, 2PI)
      bool  neg  = phi2 >= PS_PI;   // outward is the negative-grid-offset half

      [loop] for (int o = 0; o < PS_NO; ++o) {
        float rho = offOf(o);
        float r = neg ? -rho : rho;              // outward distance for this bin
        if (r < R_MIN || r > R_MAX) continue;
        float e = gridf(a, o) / gmax;            // [0,1]
        float sc = (wantMax ? e : (1.0 - e)) - W_TARGET * abs(r - r_target);
        if (sc > bestEdge) { bestEdge = sc; bestPhi = phi; bestR = r; }
      }
    }
    winPhi[k] = bestPhi;
    winR[k]   = bestR;
  }

  if (latch > 0.5) {
    [unroll] for (int k = 0; k < 3; ++k) {
      float2 nrm = float2(cos(winPhi[k]), sin(winPhi[k]));
      float2 ctr = winR[k] * nrm;
      tri[k * 4 + 0] = ctr.x; tri[k * 4 + 1] = ctr.y;
      tri[k * 4 + 2] = nrm.x; tri[k * 4 + 3] = nrm.y;
    }
    tri[12] = 1.0;
  }
}
