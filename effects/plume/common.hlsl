// source.sdf.plume — shared constants + mappings.
//
// World space: x right, y up, z into the screen (monolith convention);
// object centered at origin. The tier-0 volume is the cube
// [-PLM_EXT0, PLM_EXT0]³; distances in the SDF volume are WORLD units
// (already Lipschitz-compressed at bake so trilinear sphere-tracing with
// a modest safety factor cannot overshoot).
//
// 3D texture sizes are compile-time constants (querying a 3D storage
// texture's dimensions is non-portable — see lut_collection).

#ifndef PLUME_COMMON_HLSL
#define PLUME_COMMON_HLSL

#include "nano_octahedral.hlsl"

static const int   PLM_VOL_RES  = 128;   // tier-0 sdf volume, per axis
static const int   PLM_GI_RES   = 64;    // tier-0 radiance volume, per axis
static const float PLM_EXT0     = 0.85;  // tier-0 half-extent, world units
static const int   PLM_SHELL_RES  = 1024; // shell_full
static const int   PLM_COARSE_RES = 256;  // shell_coarse

// World position -> tier-0 volume uvw in [0,1]³.
float3 plm_world_to_uvw(float3 p) {
  return p * (0.5 / PLM_EXT0) + 0.5;
}

// Tier-0 voxel index -> world position of the voxel center.
float3 plm_voxel_to_world(int3 v) {
  return ((float3(v) + 0.5) * (1.0 / float(PLM_VOL_RES)) - 0.5)
         * (2.0 * PLM_EXT0);
}

// Linear -> sRGB OETF (exact piecewise curve). The chain's rgba8 frames
// are display-referred; plume grades in linear light and by default hands
// that straight through — this is the optional bridge (the Render group's
// sRGB Output knob), applied to plume's own color only, never the bg.
float3 plm_srgb_encode(float3 c) {
  c = max(c, 0.0);
  return lerp(12.92 * c,
              1.055 * pow(c, 1.0 / 2.4) - 0.055,
              step(0.0031308, c));
}

#endif // PLUME_COMMON_HLSL
