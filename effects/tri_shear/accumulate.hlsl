// warp.tri_shear — accumulate pass.
//
// Identical to plane_shear's: build the (angle,offset) edge-energy grid. tri_shear
// drives it with algorithm = Hough (1, "Strongest Edges") or Seam (2, "Low-energy
// Seams"); the moment sums for Dominant/PCA are simply unused. Reused verbatim so
// the two effects can never drift.
#include "../plane_shear/accumulate.hlsl"
