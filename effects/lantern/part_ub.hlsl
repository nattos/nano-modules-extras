// source.mesh.lantern — the flakes' per-frame uniforms (PartUniforms in
// main.cpp), shared by the particle passes and the spark raster.

#ifndef LANTERN_PART_UB_HLSL
#define LANTERN_PART_UB_HLSL

cbuffer Part : register(b4) {
  float4 pa;        // count, first index to (re)initialize, substep dt, substeps
  float4 pb;        // sustain S, release R, fling impulse (this frame), time
  float4 ring;      // radius, looseness, orbit speed (m/s), return stiffness
  float4 jet;       // axis (unit, world), strength (already x S)
  float4 jet2;      // bore speed (m/s), 0, 0, 0
  float4 cap;       // capture accel, capture range (m), hold accel, lane strength
  float4 orbit;     // standoff now (m), orbit speed now (m/s), curl (rad/s), capture
  float4 shell;     // radius, stiffness, fling speed, 0
  float4 vibe;      // amplitude now (m), phase (rad), spread, angular freq (rad/s)
  float4 flake;     // size scale, mica share, tumble, sparkle gain
  float4 grid;      // half extent (m), res, EMA blend, splat scale
  float4 pframe;    // frame dt, frame index, motion on, 0
  float4 pview;     // W, H, 1/W, 1/H
  float4 prev_r;    // the previous frame's camera, as cam_r / cam_u / cam_f
  float4 prev_u;
  float4 prev_f;
};

#endif  // LANTERN_PART_UB_HLSL
