// source.sdf.plume — primary march (milestone 2: two-tier + lit).
//
// Full-res compute: camera ray → tier-0 volume cube → COARSE sphere-trace
// on the trilinear SDF grid → inside a narrow band, FINE stepping against
// the exact shell-map surface (radial distance from shell_full — the same
// data the grid was baked from, so the tiers can't disagree) → shade with
// sun key (soft-shadow marched through the grid), grid AO, hemispheric
// fill, fresnel rim → gentle shoulder tonemap → composite over the input.
//
// The fine distance is a TRUE radial distance, not a Euclidean bound, so
// the fine phase never takes d-sized steps — it creeps in fixed fractions
// of a voxel and finishes with a short bisection. Fine evals cost one 2D
// texture sample, cheaper than a grid tap chain.

#include "common.hlsl"
#include "nano_hash.hlsl"

Texture3D<float4>   sdfVol     : register(t0);
Texture2D<float4>   bgTex      : register(t1);
Texture2D<float4>   shellFull  : register(t2);
SamplerState        linearSamp : register(s3);
RWTexture2D<float4> outTex     : register(u4);
Texture3D<float4>   radVol     : register(t6);   // GI radiance (or 1³ zeros)

cbuffer MarchUniforms : register(b5) {
  float4 cam_row0;    // view right (world), w = cam_pos.x
  float4 cam_row1;    // view up (world),    w = cam_pos.y
  float4 cam_row2;    // view fwd (world),   w = cam_pos.z
  float4 cam_p;       // focal, cover_ax, cover_ay, has_bg
  float4 sun_p;       // sun dir (world, toward light), w = intensity
  float4 albedo;      // rgb, w = opacity
  float4 vp;          // w, h, 1/w, 1/h
  float4 shade_p;     // shadow, ao, ambient, rim
  float4 fine_p;      // R (base radius), px_world (per unit t), inv_lip, bounce
  float4 misc;        // scene_mode (fog pipeline), band widen, crest gain,
                      // wrap-light amount
  float4 mat;         // reflect, roughness, transmission, thickness
  float4 misc2;       // wrap lit-gate, exposure gain, black level,
                      // shell texel width (world units)
  float4 grade;       // srgb encode, 0, 0, 0
};

bool plm_box(float3 ro, float3 rd, out float t0, out float t1) {
  float3 inv = 1.0 / rd;
  float3 ta = (-PLM_EXT0 - ro) * inv;
  float3 tb = ( PLM_EXT0 - ro) * inv;
  float3 tmin = min(ta, tb), tmax = max(ta, tb);
  t0 = max(max(tmin.x, tmin.y), tmin.z);
  t1 = min(min(tmax.x, tmax.y), tmax.z);
  return t1 > max(t0, 0.0);
}

float plm_sdf(float3 p) {
  return sdfVol.SampleLevel(linearSamp, plm_world_to_uvw(p), 0).r;
}

// Exact radial distance to the full-band shell surface (star-shaped).
float plm_fine(float3 p) {
  float r = length(p);
  float3 dir = p / max(r, 1e-5);
  float h = shellFull.SampleLevel(linearSamp, nano_oct_encode(dir), 0).r;
  return r - fine_p.x - h;
}

[numthreads(8, 8, 1)]
void main(uint3 gid : SV_DispatchThreadID) {
  uint W = (uint)vp.x, H = (uint)vp.y;
  if (gid.x >= W || gid.y >= H) return;

  float4 bg = cam_p.w > 0.5 ? bgTex.Load(int3(int(gid.x), int(gid.y), 0))
                            : float4(0.0, 0.0, 0.0, 0.0);

  float2 uv = (float2(gid.xy) + 0.5) * vp.zw;
  float2 ndc = float2(uv.x * 2.0 - 1.0, 1.0 - uv.y * 2.0);
  float3 V = normalize(float3(ndc.x / (2.0 * cam_p.y * cam_p.x),
                              ndc.y / (2.0 * cam_p.z * cam_p.x), 1.0));
  float3 rd = normalize(cam_row0.xyz * V.x + cam_row1.xyz * V.y +
                        cam_row2.xyz * V.z);
  float3 ro = float3(cam_row0.w, cam_row1.w, cam_row2.w);

  // Fog-pipeline mode: emit (shaded color, hit distance) for the fog +
  // composite passes instead of compositing over the input here.
  bool scene_mode = misc.x > 0.5;

  float t0, t1;
  if (!plm_box(ro, rd, t0, t1)) {
    outTex[gid.xy] = scene_mode ? float4(0.0, 0.0, 0.0, 6.0e4) : bg;
    return;
  }

  const float voxel = 2.0 * PLM_EXT0 / float(PLM_VOL_RES);
  // Handoff band widens (misc.y) when the bake's Lipschitz floor is
  // engaged: floored grids store distances LONGER than the true bound, so
  // the coarse trace overshoots — enter the (overshoot-immune, exact
  // radial) fine tier correspondingly earlier.
  const float band = 1.6 * misc.y * voxel;
  float t = max(t0, 0.0) + 0.5 * voxel;
  bool hit = false;
  float t_prev = t;
  [loop] for (int i = 0; i < 220; i++) {
    if (t > t1) break;
    float3 p = ro + rd * t;
    float d = plm_sdf(p);
    if (d < band) {
      // Fine tier: exact shell surface. Hit on sign change, refined below.
      float df = plm_fine(p);
      float eps = max(0.20 * voxel, 0.9 * fine_p.y * t);
      if (df < eps) { hit = true; break; }
      t_prev = t;
      t += clamp(df * 0.55, 0.18 * voxel, band);
    } else {
      t_prev = t;
      t += d * 0.9;
    }
  }

  if (!hit) {
    outTex[gid.xy] = scene_mode ? float4(0.0, 0.0, 0.0, 6.0e4) : bg;
    return;
  }

  // Bisection polish between the last sample and the hit — ALWAYS when
  // the last sample was still outside, even if it was a coarse step (an
  // overshooting coarse step can land the first fine sample deep inside
  // the body; without the polish those hits shade as pixel garbage).
  if (plm_fine(ro + rd * t_prev) > 0.0) {
    float ta = t_prev, tb = t;
    [unroll] for (int j = 0; j < 5; j++) {
      float tm = 0.5 * (ta + tb);
      if (plm_fine(ro + rd * tm) < 0.0) tb = tm; else ta = tm;
    }
    t = tb;
  }
  float3 hp = ro + rd * t;

  // Snap the hit RADIALLY onto the exact shell surface. Fine stepping
  // accepts a hit anywhere under eps (~0.2 voxel) ABOVE the surface, and
  // that residual is quantized by the step phase; grazing shadow rays at
  // the terminator amplify it into coherent arc banding. The radial
  // excess is exactly plm_fine(hp) (h is a function of direction only),
  // so remove it exactly — one 2D tap.
  float3 rdir = hp / max(length(hp), 1e-5);
  hp -= rdir * plm_fine(hp);

  // Normal: tetrahedron taps on the fine surface, screen-adaptive epsilon
  // (small up close for crisp flakes, wider far away to kill shimmer).
  // Floor at ~2 shell-map texels: below that the taps read the map's
  // bilinear facets and the normal speckles. The texel width comes from
  // the host (the field's DECLARED shell_res — a provider's map need not
  // match plume's own PLM_SHELL_RES).
  float texw = misc2.w;
  float e = clamp(0.8 * fine_p.y * t, max(0.0012, 2.0 * texw), 0.02);
  float2 k = float2(1.0, -1.0);
  float3 N = normalize(
      k.xyy * plm_fine(hp + k.xyy * e) + k.yyx * plm_fine(hp + k.yyx * e) +
      k.yxy * plm_fine(hp + k.yxy * e) + k.xxx * plm_fine(hp + k.xxx * e));

  // The grid stores Lipschitz-COMPRESSED distances (safe stepping);
  // penumbra and AO interpret distance as free space, so decompress first.
  float inv_lip = fine_p.z;

  // Soft shadow: march the coarse grid toward the sun.
  // (An improved between-sample penumbra estimator was tried and backed
  // out twice: it visibly deepens the plate crevice shadows away from the
  // approved look, and the faint dark-gradient rings it was suspected of
  // causing turned out to be plain 8-bit output quantization.)
  float sh = 1.0;
  if (shade_p.x > 0.001) {
    float st = 2.5 * voxel;
    // Dust optical depth rides the same taps: the grid's .a channel is
    // the provider's aggregate dust density (0 for dust-free fields —
    // the term is then exactly 1 and the look is bit-identical).
    float dtau = 0.0;
    [loop] for (int m = 0; m < 20; m++) {
      float3 sp = hp + sun_p.xyz * st;
      if (abs(sp.x) > PLM_EXT0 || abs(sp.y) > PLM_EXT0 ||
          abs(sp.z) > PLM_EXT0) break;
      float4 gs = sdfVol.SampleLevel(linearSamp, plm_world_to_uvw(sp), 0);
      float d = gs.r * inv_lip;
      sh = min(sh, 5.0 * d / st);
      if (sh < 0.02) break;
      float step = clamp(d, 0.6 * voxel, 4.0 * voxel);
      dtau += gs.a * step;
      st += step;
    }
    sh = lerp(1.0, saturate(sh) * exp2(-8.0 * dtau), shade_p.x);
  }

  // AO: how much the SDF falls short of free space along the normal.
  float ao = 1.0;
  if (shade_p.y > 0.001) {
    float occ = 0.0;
    float w = 0.5;
    [unroll] for (int a = 1; a <= 4; a++) {
      float hstep = float(a) * 1.1 * voxel;
      float d = plm_sdf(hp + N * hstep) * inv_lip;
      occ += w * (hstep - clamp(d, 0.0, hstep)) / hstep;
      w *= 0.65;
    }
    ao = saturate(1.0 - shade_p.y * 1.15 * occ);
  }

  // Crest emphasis, gated by ridge depth (misc.z): the crest channel
  // carries the raw field even at depth 0, and ungated it paints a ±10%
  // pattern onto a geometrically smooth sphere.
  float crest = sdfVol.SampleLevel(linearSamp, plm_world_to_uvw(hp), 0).b
              * misc.z;
  float lam = saturate(dot(N, sun_p.xyz));
  float ndv = saturate(dot(N, -rd));
  float rim = pow(1.0 - ndv, 3.0);

  // Studio-matte combine: the porcelain look is FORM-shaded — white tops
  // rolling to gray sides (straight lambert key), a restrained AO'd fill,
  // and near-black gaps between plates. The key floor and the rim term
  // below are deliberate NON-physical studio wrap; misc.w (Wrap Light)
  // scales both — at 0 the dark side is strictly sun-only. misc2.x
  // (Wrap Gate) confines the wrap to regions ACTUALLY receiving sun:
  // lam alone is not enough (a spike flank can face the sun yet sit in
  // the body's full shadow — the rim term is not otherwise sh-gated),
  // and the soft saturate(2*lam) ramp keeps the terminator from
  // knife-edging.
  float wgate = lerp(1.0, saturate(2.0 * lam) * sh, misc2.x);
  float wrap_floor = 0.06 * misc.w * wgate;
  float key = sun_p.w * (wrap_floor + (1.0 - wrap_floor) * lam) * sh;
  float sky = 0.55 + 0.45 * saturate(N.y * 0.8 + 0.5);
  float fill = shade_p.z * 0.4 * sky * (0.15 + 0.85 * ao);
  float3 c = albedo.rgb * (key + fill) * (0.90 + 0.10 * crest);
  c += shade_p.w * rim * ao * 0.25 * sun_p.w * wgate;

  // Porcelain sheen: a tight sun specular, widened by roughness and dimmed
  // in shadow — the plates catch a glossy edge without needing an env map.
  if (mat.x > 0.001) {
    float3 Hv = normalize(sun_p.xyz - rd);
    float spec_pow = exp2(3.0 + 7.0 * (1.0 - mat.y));
    float fres = 0.25 + 0.75 * pow(1.0 - ndv, 2.0);
    float spec = mat.x * fres * pow(saturate(dot(N, Hv)), spec_pow);
    c += spec * sun_p.w * (0.15 + 0.85 * sh) * ao;
  }

  // Translucency: integrate body density toward the sun from just inside
  // the hit — thin plates pass light and glow when backlit. `thickness`
  // sets the attenuation length; the forward-scatter lobe keeps it a
  // backlit effect rather than a flat gain.
  if (mat.z > 0.001) {
    float tlen = 0.03 + 0.14 * mat.w;
    float tau = 0.0;
    [unroll] for (int q = 1; q <= 5; q++) {
      float3 tp = hp + sun_p.xyz * (float(q) * 0.4 * tlen);
      tau += sdfVol.SampleLevel(linearSamp, plm_world_to_uvw(tp), 0).g;
    }
    tau *= 0.4;   // per-tap spacing / tlen, folded
    float fwd = 0.35 + 0.65 * pow(saturate(dot(rd, sun_p.xyz)), 2.0);
    float glow = mat.z * exp2(-tau * 4.3) * fwd;
    c += albedo.rgb * sun_p.w * glow;
  }

  // Bounce: the wave-GI radiance field, sampled a little off the surface
  // along the normal (the light lives in the air next to the shell).
  if (fine_p.w > 0.001) {
    float goff = 2.5 * (2.0 * PLM_EXT0 / float(PLM_GI_RES));
    float3 gi = radVol.SampleLevel(linearSamp,
                                   plm_world_to_uvw(hp + N * goff), 0).rgb;
    c += albedo.rgb * gi * fine_p.w * (0.35 + 0.65 * ao);
  }

  // Exposure is camera gain AHEAD of the shoulder — highlights roll off
  // instead of clipping chalky when pushed.
  c *= misc2.y;
  // Gentle shoulder: keeps the key from clipping chalky.
  c = c / (1.0 + 0.18 * c);

  if (scene_mode) {
    outTex[gid.xy] = float4(c, t);
    return;
  }
  // Black level: >0 lifts the floor toward a filmic pedestal, <0 crushes
  // the darks to true black. Graded before the opacity fade (opacity 0
  // stays pure passthrough) and BEFORE the dither, so the ±half-LSB
  // amplitude stays exact at the output quantizer no matter how the grade
  // reshapes the gradients. (The fog pipeline grades in composite.hlsl.)
  float b = misc2.z;
  c = b >= 0.0 ? b + (1.0 - b) * c : max((c + b) / (1.0 + b), 0.0);
  // Optional sRGB encode — plume's color only, after the grade (the grade
  // knobs keep their tuned linear-space feel), before the opacity fade
  // (the bg is already display-referred; the crossfade happens in the
  // chain's own space) and the dither (which must sit at the quantizer).
  if (grade.x > 0.5) c = plm_srgb_encode(c);
  float w_op = albedo.w;
  // ±half-LSB output dither on the direct (fog-off) path — the fog
  // pipeline's composite.hlsl dithers its own final write; see there.
  float3 outc = lerp(bg.rgb, c, w_op)
              + (nano_ign(float2(gid.xy)) - 0.5) * (1.0 / 255.0);
  outTex[gid.xy] = float4(outc, max(bg.a, w_op));
}
