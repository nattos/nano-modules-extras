// source.mesh.lantern — shared uniforms + scene queries.
//
// World space is the poster scene's (Blender axes: Z up), in metres, with the
// structure's center at the origin. The scene is a list of oriented boxes in a
// storage buffer (SIX float4 rows per box, see LTN_BOX_ROWS); every visibility
// query is an exact ray-box test against all of them — no distance field, so
// the ~6 cm seams between blocks are resolved exactly.
//
// The core light's shadows come from an ATLAS of octahedral depth maps, one
// tile per point sampled on the light's sphere (shadow.hlsl writes it): a
// point is lit by sample k when it is no farther from that sample than the
// first box along the same direction. Averaging the samples gives the soft
// shadow of a large light on the surfaces. The fog reads a smooth version
// seen from the core's center instead (escape.hlsl, vis.hlsl).
//
// Light units follow Cycles (calibrated against it — main.cpp): every light
// arrives here as IRRADIANCE at 1 m (point/area) or at the surface (sun), so a
// white diffuse surface's radiance is E / pi. Cycles lights volumes less than
// that suggests (a point light by 1/pi), so media scale by vol_k.

#ifndef LANTERN_COMMON_HLSL
#define LANTERN_COMMON_HLSL

#include "nano_octahedral.hlsl"

static const int   LTN_BOX_ROWS = 6;
static const int   LTN_MAX_SAMPLES = 8;
static const float LTN_PI = 3.14159265;
static const float LTN_MISS = 6.0e4;   // largest sentinel an RGBA16F holds

// What an aux output carries (main.cpp's Aux select).
static const int LTN_AUX_DEPTH = 1;
static const int LTN_AUX_SURFACES = 2;
static const int LTN_AUX_INSCATTER = 3;
static const int LTN_AUX_TRANSMIT = 4;
static const int LTN_AUX_FLAKES = 5;
static const int LTN_AUX_GLINT = 6;
static const int LTN_AUX_LINEAR = 7;

cbuffer Frame : register(b0) {
  float4 cam_r;     // camera right (world), w = camera position x
  float4 cam_u;     // camera up,            w = camera position y
  float4 cam_f;     // camera forward,       w = camera position z
  float4 cam_p;     // focal, cover ax, cover ay, box count
  float4 vp;        // W, H, 1/W, 1/H (full res)
  float4 core_p;    // core position (world), w = light radius
  float4 core_i;    // core irradiance at 1 m (rgb), w = light sample count
  float4 amb_d;     // ambient sun: unit direction TOWARD it
  float4 amb_e;     // ambient sun irradiance (rgb)
  float4 rim_d;     // rim sun: unit direction toward it
  float4 rim_e;     // rim sun irradiance (rgb), surfaces only
  float4 kick_p;    // kicker disk position, w = disk radius
  float4 kick_d;    // kicker emission axis (unit, toward the scene)
  float4 kick_i;    // kicker irradiance at 1 m head-on (rgb), surfaces only
  float4 water_s;   // water scattering (1/m, rgb), w = phase g
  float4 water_t;   // water extinction (1/m, rgb), w = march sphere radius
  float4 halo_c;    // halo center (world), w = peak density (1/m)
  float4 halo_k;    // 1 / halo scale (rgb), w = wisp amount
  float4 halo_p;    // phase g, wisp frequency, drift, 0
  float4 shade;     // bevel width (m), reflection gain, ember gain, debug view
  float4 grade;     // exposure gain, opacity, has_bg, contrast
  float4 misc;      // frame, atlas tile res, atlas cols, march steps
  float4 egg_r0;    // world -> structure frame rotation rows (material coords)
  float4 egg_r1;
  float4 egg_r2;
  float4 vol_k;     // volume light scale vs surfaces: core, ambient sun; core clear (m), 0
  float4 heart;     // the heart's emission (rgb radiance), 0
  float4 air;       // the air wall: view depth its edge starts, ends; amount, 0
  float4 surf;      // shell glow, chip amount, albedo scale, 0
  float4 dust;      // flake dust: extinction per (flake/m^3), scatter albedo,
                    //   presence grid half extent (m), spark layer on (0/1)
  float4 aux;       // what Aux 1 / 2 carry (LTN_AUX_*, 0: not wired);
                    //   render_outputs depth wired (0/1); the structure's view depth (m)
  float4 lsamp[LTN_MAX_SAMPLES];  // core light sample positions (world)
};

StructuredBuffer<float4> boxes : register(t1);

// Does an aux output carry `what` this frame?
bool ltn_aux_wants(int what) { return (int)aux.x == what || (int)aux.y == what; }

int ltn_box_count() { return (int)cam_p.w; }

// Camera ray through a pixel center of a W x H target.
void ltn_camera_ray(float2 px, float2 size, out float3 ro, out float3 rd) {
  float2 uv = (px + 0.5) / size;
  float2 ndc = float2(uv.x * 2.0 - 1.0, 1.0 - uv.y * 2.0);
  float3 v = float3(ndc.x / (2.0 * cam_p.y * cam_p.x),
                    ndc.y / (2.0 * cam_p.z * cam_p.x), 1.0);
  rd = normalize(cam_r.xyz * v.x + cam_u.xyz * v.y + cam_f.xyz * v.z);
  ro = float3(cam_r.w, cam_u.w, cam_f.w);
}

// Ray vs box i (slab test in the box's frame). tn/tf: entry/exit distances;
// axis: the slab the ray enters through (its face normal is axis * -sign(ld)).
bool ltn_box_ray(int i, float3 ro, float3 rd, out float tn, out float tf,
                 out int axis, out float3 ld) {
  float4 r0 = boxes[i * LTN_BOX_ROWS + 0];
  float4 r1 = boxes[i * LTN_BOX_ROWS + 1];
  float4 r2 = boxes[i * LTN_BOX_ROWS + 2];
  float4 r3 = boxes[i * LTN_BOX_ROWS + 3];
  float3 o = ro - r0.xyz;
  float3 lo = float3(dot(o, r1.xyz), dot(o, r2.xyz), dot(o, r3.xyz));
  ld = float3(dot(rd, r1.xyz), dot(rd, r2.xyz), dot(rd, r3.xyz));
  float3 h = float3(r1.w, r2.w, r3.w);
  float3 safe = select(abs(ld) < 1e-8, 1e-8, ld);
  float3 inv = 1.0 / safe;
  float3 ta = (-h - lo) * inv;
  float3 tb = ( h - lo) * inv;
  float3 tmn = min(ta, tb), tmx = max(ta, tb);
  tn = max(max(tmn.x, tmn.y), tmn.z);
  tf = min(min(tmx.x, tmx.y), tmx.z);
  axis = (tmn.x >= tmn.y && tmn.x >= tmn.z) ? 0 : (tmn.y >= tmn.z ? 1 : 2);
  return tf >= max(tn, 0.0);
}

// Nearest box hit in front of ro (tn > tmin). Returns the box index, -1 on miss.
int ltn_trace(float3 ro, float3 rd, float tmin, float tmax, out float t,
              out int axis, out float3 ld) {
  int best = -1;
  t = tmax;
  axis = 0;
  ld = float3(0.0, 0.0, 1.0);
  int n = ltn_box_count();
  [loop] for (int i = 0; i < n; i++) {
    float tn, tf;
    int ax;
    float3 l;
    if (ltn_box_ray(i, ro, rd, tn, tf, ax, l) && tn > tmin && tn < t) {
      t = tn;
      best = i;
      axis = ax;
      ld = l;
    }
  }
  return best;
}

// Any box between ro and ro + rd * tmax? (1 = clear, 0 = blocked)
float ltn_visible(float3 ro, float3 rd, float tmax) {
  int n = ltn_box_count();
  [loop] for (int i = 0; i < n; i++) {
    float tn, tf;
    int ax;
    float3 l;
    if (ltn_box_ray(i, ro, rd, tn, tf, ax, l) && tn > 1e-4 && tn < tmax)
      return 0.0;
  }
  return 1.0;
}

// --- Core light shadow atlas ------------------------------------------------

// Texel of direction `dir` in light sample k's tile.
int2 ltn_atlas_texel(int k, float3 dir) {
  int res = (int)misc.y, cols = (int)misc.z;
  int2 tile = int2(k % cols, k / cols);
  int2 t = int2(nano_oct_encode(dir) * float(res));
  return tile * res + clamp(t, int2(0, 0), int2(res - 1, res - 1));
}

// Is p lit by light sample k? (Point-compare against the first-hit depth.)
float ltn_core_vis(Texture2D<float4> atlas, int k, float3 p) {
  float3 d = p - lsamp[k].xyz;
  float dist = length(d);
  float depth = atlas.Load(int3(ltn_atlas_texel(k, d / max(dist, 1e-5)), 0)).r;
  return dist <= depth + 0.03 + 0.006 * dist ? 1.0 : 0.0;
}

// --- The fog's view of the core (escape.hlsl, vis.hlsl) ---------------------

static const int   LTN_ESC_RES = 256;   // escape map + prefiltered levels
static const int   LTN_ESC_LOW = 64;    // the escape map box-averaged
static const float LTN_SHELL = 2.5;     // where the blockers are taken to sit
static const float LTN_PENUMBRA = 0.9;

// The angular radius (from the core's center) a point r out sees the light's
// shadow blurred by: 0 at the shell, rising to ltn_fog_rho_max() far away.
float ltn_fog_rho_max() { return LTN_PENUMBRA * core_p.w / LTN_SHELL; }
float ltn_fog_rho(float r) {
  return ltn_fog_rho_max() * max(r - LTN_SHELL, 0.0) / max(r, 1e-3);
}

// --- The air wall -------------------------------------------------------------
//
// Not physical: a slab of clear air between the camera and a plane facing
// it, so dense water can surround the structure without burying it. Water
// and halo densities are scaled by ltn_air_f: 1 - amount far side of the
// plane's soft edge (a smoothstep in view depth), 1 beyond it. `c` is the
// ray's cosine to the view axis (view depth = t * c).

float ltn_air_f(float t, float c) {
  return 1.0 - air.z * (1.0 - smoothstep(air.x, air.y, t * c));
}

// Antiderivative of smoothstep(air.x, air.y, z) in z (continuous everywhere).
float ltn_air_smooth_int(float z) {
  float w = max(air.y - air.x, 1e-3);
  float x = saturate((z - air.x) / w);
  return w * (x * x * x - 0.5 * x * x * x * x) + max(z - air.y, 0.0);
}

// The length of water [ta, tb] along a ray amounts to: its integrated density.
float ltn_air_len(float ta, float tb, float c) {
  float len = max(tb - ta, 0.0);
  if (air.z <= 0.0 || len <= 0.0) return len;
  float cc = max(c, 1e-3);
  float wet = (ltn_air_smooth_int(tb * cc) - ltn_air_smooth_int(ta * cc)) / cc;
  return len - air.z * (len - wet);
}

// Henyey-Greenstein, normalized over the sphere. cosv = cos(angle between the
// light's travel direction and the direction toward the eye).
float ltn_hg(float g, float cosv) {
  float gg = g * g;
  return (1.0 - gg) / (4.0 * LTN_PI * pow(max(1.0 + gg - 2.0 * g * cosv, 1e-4), 1.5));
}

// The core light's irradiance at distance^2 r2: inverse square outside the
// light (a sphere light's exact falloff there), held inside half its radius.
float3 ltn_core_irr(float r2) {
  return core_i.rgb / max(r2, 0.25 * core_p.w * core_p.w);
}

// ... and the radiant intensity / r^2 a medium sees.
float3 ltn_core_irr_vol(float r2) {
  return ltn_core_irr(r2) * vol_k.x;
}

// Halo density at p (1/m): an ellipsoidal ball around the structure whose
// density falls to nothing well inside its bound, broken into wisps.
float ltn_halo_density(float3 p, float wisp_noise) {
  float3 q = (p - halo_c.xyz) * halo_k.xyz;
  float r = length(q);
  float base = halo_c.w * (1.0 - smoothstep(0.1, 0.85, r));
  // Blender's Map Range 0.35..0.7 -> 0.25..1.7 of a noise centered on 0.5.
  float w = lerp(0.25, 1.7, saturate((wisp_noise - 0.35) / 0.35));
  return base * lerp(1.0, w, halo_k.w);
}

#endif  // LANTERN_COMMON_HLSL
