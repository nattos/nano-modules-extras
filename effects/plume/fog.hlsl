// source.sdf.plume — volumetric atmosphere (half-res ray march).
//
// Marches the fog domain (a sphere of world air around the object, out to
// PLM_FOG_EXT) front-to-back, stopping at the surface depth the march
// pass recorded. Two density terms:
//   shell haze — exp(−d/σ) hugging the displaced surface. Inside the
//     tier-0 box d comes from the SDF grid (detailed); outside, the
//     displacement is irrelevant at haze scales, so d = |p| − R
//     analytically — the "tier-1" density needs no baked volume.
//   room floor — a thin constant medium for depth.
// Lighting per step: ambient + the wave-GI radiance field (the fog is
// GI-lit for free — light bleeds from the shape into the haze, and at
// high resonance the haze RINGS) + direct sun with a Henyey-Greenstein
// phase and a 2-tap occlusion sampled where the sun ray crosses the
// object (full-length silhouette shafts, cone-widening penumbra).
// Output: (in-scattered light, transmittance) — composited full-res by
// composite.hlsl with a bilinear (fog is soft) upsample.

#include "common.hlsl"
#include "nano_hash.hlsl"

static const float PLM_FOG_EXT = 3.2;   // fog domain radius, world units

Texture3D<float4>   sdfVol     : register(t0);
Texture3D<float4>   radVol     : register(t1);
Texture2D<float4>   sceneTex   : register(t2);   // .a = hit distance
SamplerState        linearSamp : register(s3);
RWTexture2D<float4> fogTex     : register(u4);

cbuffer FogUniforms : register(b5) {
  float4 cam_row0;    // view right (world), w = cam_pos.x
  float4 cam_row1;    // view up (world),    w = cam_pos.y
  float4 cam_row2;    // view fwd (world),   w = cam_pos.z
  float4 cam_p;       // focal, cover_ax, cover_ay, R (base radius)
  float4 sun_p;       // sun dir (world, toward light), w = intensity
  float4 fog_p;       // shell gain, inv_soft, room gain, phase g
  float4 misc;        // inv_lip, ambient, bounce, ridge amp (world)
  float4 vp;          // half W, half H, 1/(half W), 1/(half H)
  float4 misc2;       // iso-lobe blend, 0, 0, 0
};

bool plm_sphere(float3 ro, float3 rd, float rad, out float t0, out float t1) {
  t0 = 0.0;
  t1 = 0.0;
  float b = dot(ro, rd);
  float cc = dot(ro, ro) - rad * rad;
  float disc = b * b - cc;
  if (disc < 0.0) return false;
  float sq = sqrt(disc);
  t0 = -b - sq;
  t1 = -b + sq;
  return t1 > 0.0;
}

[numthreads(8, 8, 1)]
void main(uint3 gid : SV_DispatchThreadID) {
  uint W = (uint)vp.x, H = (uint)vp.y;
  if (gid.x >= W || gid.y >= H) return;

  float2 uv = (float2(gid.xy) + 0.5) * vp.zw;
  float2 ndc = float2(uv.x * 2.0 - 1.0, 1.0 - uv.y * 2.0);
  float3 V = normalize(float3(ndc.x / (2.0 * cam_p.y * cam_p.x),
                              ndc.y / (2.0 * cam_p.z * cam_p.x), 1.0));
  float3 rd = normalize(cam_row0.xyz * V.x + cam_row1.xyz * V.y +
                        cam_row2.xyz * V.z);
  float3 ro = float3(cam_row0.w, cam_row1.w, cam_row2.w);

  // Surface depth from the (full-res) scene buffer.
  float depth = sceneTex.Load(int3(int(gid.x) * 2, int(gid.y) * 2, 0)).a;

  float t0, t1;
  if (!plm_sphere(ro, rd, PLM_FOG_EXT, t0, t1)) {
    fogTex[gid.xy] = float4(0.0, 0.0, 0.0, 1.0);
    return;
  }
  t0 = max(t0, 0.0);
  t1 = min(t1, depth);
  if (t1 <= t0) {
    fogTex[gid.xy] = float4(0.0, 0.0, 0.0, 1.0);
    return;
  }

  const int STEPS = 36;
  float dt = (t1 - t0) / float(STEPS);
  float jitter = nano_ign(float2(gid.xy));
  float t = t0 + dt * jitter;

  // Henyey-Greenstein phase for the direct term.
  float g = fog_p.w;
  float cosv = dot(rd, sun_p.xyz);
  float hg = (1.0 - g * g) /
             max(pow(1.0 + g * g - 2.0 * g * cosv, 1.5), 1e-3) * 0.25;
  // Dual lobe (gated by the knob): blend a small isotropic lobe (0.25 is
  // HG at g=0 in this normalization) as a multi-scatter stand-in. A pure
  // forward lobe is single-scatter physics — its backward tail is ~1/80th
  // of its peak at high Phase, so frontlit fog goes black; real fog's
  // multiple bounces lose direction and put a floor under that.
  hg = lerp(hg, 0.25, misc2.x);

  float3 acc = float3(0.0, 0.0, 0.0);
  float trans = 1.0;
  [loop] for (int i = 0; i < STEPS; i++) {
    float3 p = ro + rd * t;

    // Shell-hug distance: grid inside the tier-0 box, analytic outside.
    float d;
    float dust_d = 0.0;   // provider dust density (grid .a; 0 when unused)
    bool in0 = abs(p.x) < PLM_EXT0 && abs(p.y) < PLM_EXT0 && abs(p.z) < PLM_EXT0;
    if (in0) {
      float4 gs = sdfVol.SampleLevel(linearSamp, plm_world_to_uvw(p), 0);
      dust_d = gs.a;
      d = gs.r * misc.x;
      // The baked distance is RADIAL (r - R - h(dir)), not a true SDF: the
      // angular spike pattern of h persists undiminished at every altitude,
      // which paints density spokes out to the box edge — on screen a
      // sunburst of rays radiating from the object center (worst with tight
      // fog_soft + spiky ridges, where exp2(-d*inv_soft) amplifies gap
      // contrast by 2^(inv_soft*h)). Haze should read as a smooth ball from
      // afar, so fade the displaced detail toward the analytic sphere gap.
      // The fade key MUST be the smooth altitude da, not the grid gap: the
      // gap is itself spike-modulated, so keying on it lets every spike
      // keep its spoke for an extra h of radius before the ramp engages.
      // Ramp: full detail below the crest sphere (R + amp — the skin coats
      // spike tips and crevices alike), then out over ~2 density e-folds.
      float da = length(p) - cam_p.w;
      float sig = 1.4427 / fog_p.y;      // density e-fold length, wu
      d = lerp(d, da, smoothstep(misc.w, misc.w + 2.0 * sig, da));
    }
    else     d = length(p) - cam_p.w;

    // Dust clumps scatter like local pockets of shell haze — scaled by
    // the same Fog knob so dust-in-fog obeys the atmosphere controls.
    // The gain is deliberately gentle: this term is the CLOUD's aggregate
    // influence, and at 128³ it must read as soft haze, not voxel blobs.
    float sigma = fog_p.x * exp2(-max(d, 0.0) * fog_p.y)
                + fog_p.z * 0.22
                + fog_p.x * 0.45 * dust_d;
    if (sigma > 1e-4) {
      // Direct sun occlusion: sample the SDF around the point where THIS
      // sun ray crosses the object — its closest approach to the origin —
      // not at fixed near offsets. Fixed 0.22/0.55 taps truncate every
      // shadow ~0.5 wu behind the object (columns collapse into a
      // surface-hugging blob) and out-of-box taps used to count as clear.
      // Sampling the crossing keeps silhouette-detailed shafts through the
      // whole fog domain for the same 2 taps; outside the box the base
      // sphere stands in analytically. Penumbra widens with distance to
      // the crossing (sun-disk cone), and the tap pair rides the pixel
      // jitter so coarse-grid blotch dithers into noise the composite's
      // tent upsample absorbs.
      // Only rays whose crossing lies AHEAD can be shadowed: on the
      // sun side of the terminator plane the forward sun ray moves
      // radially outward and nothing can block it — but the clamped
      // near taps would still read "spike radially below this point"
      // and manufacture a shadow skin keyed to dir(p), which projects
      // as a radial sunburst around the whole ball. Fade the shadow in
      // across the terminator instead.
      float occ = 1.0;
      float scs = -dot(p, sun_p.xyz);
      if (sun_p.w > 1e-3 && scs > 0.0) {
        float sa  = max(scs, 0.30) + (jitter - 0.5) * 0.24;
        float wid = 14.0 / (1.0 + 1.1 * scs);
        [unroll] for (int k = 0; k < 2; k++) {
          float3 ps = p + sun_p.xyz * max(sa + 0.36 * float(k) - 0.18, 0.12);
          float ds;
          if (abs(ps.x) < PLM_EXT0 && abs(ps.y) < PLM_EXT0 &&
              abs(ps.z) < PLM_EXT0) {
            float4 g2 = sdfVol.SampleLevel(linearSamp, plm_world_to_uvw(ps), 0);
            ds = g2.r * misc.x;
          }
          else
            ds = length(ps) - cam_p.w;
          occ *= saturate(ds * wid + 0.5);
        }
        occ = lerp(1.0, occ, smoothstep(0.0, 0.15, scs));
      }

      // Dust shadow: the provider dust is a DISTRIBUTED medium, so its
      // sun occlusion must be an optical-depth INTEGRAL. Point taps
      // pinned at the body crossing (the old 2-tap product) print
      // displaced hard copies of the cloud's cross-section — a dust
      // ring cast nested crescent sheets into the fog. Instead: clip
      // the sun ray to the density grid box and take 2 stratified taps
      // whose in-segment phase is re-jittered EVERY STEP (golden-ratio
      // stride), so across the 36-step integral the torus is sampled
      // densely — one smooth curtain, dithered into noise the tent
      // upsample absorbs.
      if (sun_p.w > 1e-3) {
        float3 sd = sun_p.xyz;
        float3 sds = float3(abs(sd.x) < 1e-5 ? 1e-5 : sd.x,
                            abs(sd.y) < 1e-5 ? 1e-5 : sd.y,
                            abs(sd.z) < 1e-5 ? 1e-5 : sd.z);
        float3 ext = float3(PLM_EXT0, PLM_EXT0, PLM_EXT0);
        float3 ta = (-ext - p) / sds;
        float3 tb = ( ext - p) / sds;
        float3 tmn = min(ta, tb), tmx = max(ta, tb);
        float sin0 = max(max(max(tmn.x, tmn.y), tmn.z), 0.05);
        float sout = min(min(tmx.x, tmx.y), tmx.z);
        if (sout > sin0) {
          float jf = frac(jitter + 0.6180339887 * float(i));
          float dtau_d = 0.0;
          [unroll] for (int m = 0; m < 2; m++) {
            float sf = (float(m) + jf) * 0.5;
            float3 pd = p + sd * lerp(sin0, sout, sf);
            dtau_d += sdfVol.SampleLevel(linearSamp,
                                         plm_world_to_uvw(pd), 0).a;
          }
          occ *= exp2(-4.0 * 0.5 * dtau_d * (sout - sin0));
        }
      }

      float3 gi = float3(0.0, 0.0, 0.0);
      if (misc.z > 0.001 && in0)
        gi = radVol.SampleLevel(linearSamp, plm_world_to_uvw(p), 0).rgb * misc.z;

      float3 light = misc.y.xxx * 0.35
                   + gi
                   + (sun_p.w * hg * occ).xxx;
      float a = 1.0 - exp2(-sigma * dt * 1.4427);   // 1 - e^(-sigma dt)
      acc += trans * a * light;
      trans *= 1.0 - a;
      if (trans < 0.01) break;
    }
    t += dt;
  }

  fogTex[gid.xy] = float4(acc, trans);
}
