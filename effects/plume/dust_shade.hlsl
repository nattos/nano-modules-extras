// source.sdf.plume — dust splat pass 2: the winning particle shades.
//
// Same per-particle iteration as pass 1; a thread writes a pixel only
// where the depth buffer holds ITS depth bits — exactly one writer per
// pixel (two particles at bit-identical depth both "win", and then write
// the same depth and near-identical color — benign). Output is the
// dedicated DUST LAYER (color only — depth stays in the resolve buffer,
// alpha comes from pass 1's coverage sum): the scene buffer keeps the
// pure surface depth, so the half-res fog march never sees point
// occluders, and composite blends the layer depth-aware. Shading is
// computed ONCE per particle (the footprint is a few pixels of one tiny
// mote — per-pixel variation would be invisible): a two-sided diffuse
// presence, a sun GLINT off the particle's normal (the rail's normal is
// the particle's orientation — providers tumble it for twinkle), the
// grid soft shadow, and the GI field. Exposure + shoulder match the
// march so dust grades identically through composite.
//
// Dust has its OWN material, decoupled from the body's porcelain:
// albedo.rgb is the dust color and misc.y is METALLIC — one slider from
// chalk to glitter. As metal rises the diffuse presence fades (a metal
// has no diffuse) while the facet glint grows, tightens, and takes the
// dust color (a metal's reflection is tinted; a dielectric's is white).
// GI stays tinted-full at any metal: for a speck it reads as the flake
// reflecting its environment, and it keeps metallic dust from becoming
// holes in shadow.

#define DUST_UB_REG b6
#include "dust_common.hlsl"
#include "common.hlsl"

Texture3D<float4>      sdfVol   : register(t1);
Texture3D<float4>      radVol   : register(t2);
SamplerState           linSamp  : register(s3);
StructuredBuffer<uint> depthBuf : register(t4);
RWTexture2D<float4>    outDust  : register(u5);

[numthreads(64, 1, 1)]
void main(uint3 gid : SV_DispatchThreadID) {
  if (gid.x >= (uint)cam_p.w) return;
  float3 pos, nrm;
  float seed, t, rp;
  float2 ctr;
  if (!du_project(gid.x, pos, nrm, seed, ctr, t, rp)) return;

  // --- Shade (once per particle) ---
  float3 ro = float3(cam_row0.w, cam_row1.w, cam_row2.w);
  float3 rd = (pos - ro) / max(t, 1e-5);
  float3 N = normalize(nrm);

  // Soft sun shadow: abbreviated march through the (compressed) grid —
  // 6 steps reads the body silhouette; out-of-box taps count as clear.
  float sh = 1.0;
  if (shade_p.x > 0.001) {
    const float voxel = 2.0 * PLM_EXT0 / float(PLM_VOL_RES);
    float st = 2.0 * voxel;
    float dtau = 0.0;   // dust self-shadow: motes dim inside a clump
    [loop] for (int m = 0; m < 6; m++) {
      float3 sp = pos + sun_p.xyz * st;
      if (abs(sp.x) > PLM_EXT0 || abs(sp.y) > PLM_EXT0 ||
          abs(sp.z) > PLM_EXT0) break;
      float4 gs = sdfVol.SampleLevel(linSamp, plm_world_to_uvw(sp), 0);
      float d = gs.r * shade_p.w;
      sh = min(sh, 5.0 * d / st);
      if (sh < 0.02) break;
      float step = clamp(d, 0.8 * voxel, 5.0 * voxel);
      dtau += gs.a * step;
      st += step;
    }
    sh = lerp(1.0, saturate(sh) * exp2(-8.0 * dtau), shade_p.x);
  }

  // Two-sided diffuse (a mote has no meaningful "back"); the normal
  // still matters through the glint below. The 0.55 base dims motes
  // BELOW the surface they hover over — matte-white dust on a
  // matte-white body is invisible; a mote must read as a dark speck
  // that FLASHES when its facet catches the sun.
  float lam = 0.35 + 0.65 * abs(dot(N, sun_p.xyz));
  float key = sun_p.w * lam * sh;
  float fill = shade_p.y * 0.4 * (0.55 + 0.45 * saturate(N.y * 0.8 + 0.5));
  float3 gi = float3(0.0, 0.0, 0.0);
  if (shade_p.z > 0.001)
    gi = radVol.SampleLevel(linSamp, plm_world_to_uvw(pos), 0).rgb
       * shade_p.z;

  // Diffuse presence fades with metallic but keeps a floor — a metal
  // flake between flashes is a dark colored speck, not a hole.
  float metal = misc.y;
  float dw = lerp(0.68, 0.15, metal);
  float3 c = albedo.rgb * (dw * (key + fill) + gi);

  // Glint: a Blinn lobe off the oriented normal — wide enough that a
  // useful fraction of random facets catch it, tight enough to flash.
  // Metallic drives it whole: gain AND tightness rise together (soft
  // satin sheen low, hard glitter high), and the lobe color slides from
  // dielectric white to the tinted dust color.
  if (metal > 0.001) {
    float3 Hv = normalize(sun_p.xyz - rd);
    float spec_pow = exp2(4.0 + 2.5 * metal);
    float spec = metal * (1.5 + 2.5 * metal)
               * pow(saturate(abs(dot(N, Hv))), spec_pow);
    c += lerp(float3(1.0, 1.0, 1.0), albedo.rgb, metal)
       * (spec * sun_p.w * (0.15 + 0.85 * sh));
  }

  // Exposure + gentle shoulder, matching march.hlsl's scene-mode output.
  c *= albedo.w;
  c = c / (1.0 + 0.18 * c);

  // --- Write the pixels this particle won ---
  int W = (int)vp.x, H = (int)vp.y;
  float reach = rp + DUST_FEATHER;
  int x0 = max((int)floor(ctr.x - reach), 0);
  int x1 = min((int)floor(ctr.x + reach), W - 1);
  int y0 = max((int)floor(ctr.y - reach), 0);
  int y1 = min((int)floor(ctr.y + reach), H - 1);
  uint du = asuint(t);
  for (int py = y0; py <= y1; py++) {
    for (int px = x0; px <= x1; px++) {
      float2 d = float2(px, py) + 0.5 - ctr;
      if (du_cov(length(d), rp) <= 0.0) continue;
      if (depthBuf[py * W + px] != du) continue;
      outDust[int2(px, py)] = float4(c, 1.0);
    }
  }
}
