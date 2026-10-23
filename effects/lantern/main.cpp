/*
 * source.mesh.lantern — "Lantern".
 *
 * A hollow cluster of stone and shell blocks hanging in deep water with a
 * light at its heart: the light escapes only through the seams and the gaps
 * where blocks have jutted out, carving beams through a halo of murky water.
 * After the poster scene in ~/Code/blender (part_structure.py); its exact
 * block layout is Layout 0 (layout_ref.h), other layouts are generated the
 * same way from a seed.
 *
 * The scene is ~70 oriented boxes (a storage buffer, rebuilt on the CPU every
 * frame — animation is free), traced EXACTLY rather than through a distance
 * field: a baked grid fine enough to resolve the 6 cm seams would be ~4x
 * finer than plume's.
 *
 * Pass chain (all compute):
 *   shadow    — octahedral depth maps of the boxes from 8 points on the core
 *               light's sphere (one atlas, 512² a tile): the core's soft
 *               shadows on surfaces.
 *   escape    — what escapes the boxes, seen from the core's center (256²,
 *   + reduce    supersampled; and box-averaged to 64²).
 *   vis       — that map prefiltered at four penumbra widths: the beams'
 *               smooth visibility in the water (no random lookups — the
 *               march's beams don't crawl in motion).
 *   scene     — full res: nearest box, then shading (bevelled normals, shell
 *               and stone, the core + two suns + the kicker, reflections,
 *               ember) -> (radiance, hit distance).
 *   fog       — half res: water + halo, equiangular march about the core.
 *   blur (x2) — depth-aware separable blur of the march's dithered in-scatter.
 *   composite — depth-aware upsample, AgX, dither, over the input.
 *   aux       — only while wired: the Aux 1 / 2 pictures and render_outputs'
 *               depth.
 *
 * And the FLAKES (particles.hlsl): ~1000 specks of mica, grit and silt,
 * simulated (p_update), splatted into a presence grid the fog reads as dust
 * (p_resolve), shaded (p_shade) and drawn as EWA splats into a spark layer
 * (spark_vs/fs) the composite lays over the scene. They are played by the
 * SWEEP — one knob (sweep_knob.h, as three_planes' rig and Side Racks ride
 * it): its level SUSTAINS (blocks capture flakes into tight fast orbits, a
 * jet blows the free ones through the core), its bounce OVERSHOOTS, a glint
 * (three_planes_glints.h) sweeps a slanted band of extra magnetism across the
 * screen, the choke lets them go ballistic in a loose ring, and a throw FLINGS
 * them onto a sphere where the tail shakes them as it rings out.
 *
 * The sweep plays the blocks and the core too, through a matrix of SENDS:
 * its envelope, its overshoot, each block's glint charge and the thrown tail
 * (the reverb) each drive Jut, Seam, Jitter (knocks rung out on the glide's
 * spring), the core's Power and the blocks' Magnetism by their own amounts;
 * and a change of jut twangs through that spring as well.
 *
 * With render_outputs read downstream, the blocks' and the flakes' motion
 * vectors go out on its motion leaf (motion.hlsl + the spark raster) for a
 * motion blur to streak.
 *
 * Units are the scene's: metres, Blender axes (Z up), the structure's center
 * at the origin, and Cycles light units (calibrated below), so the scene's
 * light powers and densities carry over unchanged.
 */

#include <gpu.h>
#include <host.h>
#include <effect_utils.h>

#include <cmath>
#include <cstdint>

#include <sketch/glide.h>
#include <sketch/sweep_knob.h>
#include <sketch/three_planes_glints.h>

#include "layout_ref.h"
#include "lantern_shaders.h"

namespace lantern {

constexpr float kPi = 3.14159265358979323846f;
constexpr float kTau = 2.0f * kPi;
constexpr float kDeg = kPi / 180.0f;

// The poster structure (part_structure.py).
constexpr float kCell = 0.95f;
constexpr float kJutMax = 1.6f;
constexpr float kRadii[3] = {2.8f, 2.5f, 2.4f};

// Cycles light units -> irradiance, measured on a white diffuse plane
// (sun strength 1 -> E 1.154; a point / spot of P watts at 1 m -> 0.2359 P;
// a small area light head-on at 1 m -> 0.4732 P) and in a thin isotropic
// medium (a sun lights it with E = strength; a point light with intensity
// 0.0756 P — 1/pi of what the surface number implies).
constexpr float kSunE = 1.154f;
constexpr float kPointE = 0.2359f;
constexpr float kAreaE = 0.4732f;
constexpr float kSunVolK = 1.0f / kSunE;
constexpr float kPointVolK = 0.0756f / kPointE;

constexpr int kMaxBoxes = 192;
constexpr int kBoxRows = 6;
constexpr int kSamples = 8;      // core light samples (atlas tiles)
constexpr int kTileRes = 512;
constexpr int kAtlasCols = 4;
constexpr int kAtlasRows = kSamples / kAtlasCols;
constexpr int kFogSteps = 64;
constexpr int kEscRes = 256;     // LTN_ESC_RES
constexpr int kEscLow = 64;      // LTN_ESC_LOW

// The flakes (particles.hlsl).
constexpr int kMaxFlakes = 4096;
constexpr int kPRows = 7;        // LTN_P_ROWS
constexpr int kRRows = 4;        // LTN_R_ROWS
constexpr int kGrid = 32;        // LTN_GRID
constexpr float kGridExtent = 8.0f;   // m, half the presence grid's side
constexpr float kSplat = 256.0f;      // fixed-point scale of a splat
constexpr float kSubstep = 1.0f / 360.0f;

struct FrameUniforms {
  float cam_r[4], cam_u[4], cam_f[4], cam_p[4], vp[4];
  float core_p[4], core_i[4], amb_d[4], amb_e[4], rim_d[4], rim_e[4];
  float kick_p[4], kick_d[4], kick_i[4];
  float water_s[4], water_t[4], halo_c[4], halo_k[4], halo_p[4];
  float shade[4], grade[4], misc[4];
  float egg_r0[4], egg_r1[4], egg_r2[4];
  float vol_k[4];
  float heart[4];
  float air[4];
  float surf[4];
  float dust[4];
  float aux[4];
  float lsamp[kSamples][4];
};
static_assert(sizeof(FrameUniforms) == 39 * 16, "FrameUniforms layout mismatch");

// Mirrors `cbuffer Part` (part_ub.hlsl), row for row.
struct PartUniforms {
  float pa[4], pb[4], ring[4], jet[4], jet2[4], cap[4], orbit[4], shell[4];
  float vibe[4], flake[4], grid[4], pframe[4], pview[4];
  float prev_r[4], prev_u[4], prev_f[4];
  float bite[4];
  float mvs[4];
};
static_assert(sizeof(PartUniforms) == 18 * 16, "PartUniforms layout mismatch");

struct Row { float v[4]; };

// ---------------------------------------------------------------------------
// Small math.

struct V3 { float x, y, z; };
inline V3 v3(float x, float y, float z) { return {x, y, z}; }
inline V3 operator+(V3 a, V3 b) { return {a.x + b.x, a.y + b.y, a.z + b.z}; }
inline V3 operator-(V3 a, V3 b) { return {a.x - b.x, a.y - b.y, a.z - b.z}; }
inline V3 operator*(V3 a, float s) { return {a.x * s, a.y * s, a.z * s}; }
inline float dot(V3 a, V3 b) { return a.x * b.x + a.y * b.y + a.z * b.z; }
inline V3 cross(V3 a, V3 b) {
  return {a.y * b.z - a.z * b.y, a.z * b.x - a.x * b.z, a.x * b.y - a.y * b.x};
}
inline V3 normalize(V3 a) {
  float l = std::sqrt(dot(a, a));
  return l > 1e-12f ? a * (1.0f / l) : v3(0, 0, 1);
}

struct M3 { float m[3][3]; };   // row-major; columns are the frame's axes
inline M3 mul(const M3& a, const M3& b) {
  M3 r;
  for (int i = 0; i < 3; i++)
    for (int j = 0; j < 3; j++)
      r.m[i][j] = a.m[i][0] * b.m[0][j] + a.m[i][1] * b.m[1][j] + a.m[i][2] * b.m[2][j];
  return r;
}
inline V3 mul(const M3& a, V3 v) {
  return {a.m[0][0] * v.x + a.m[0][1] * v.y + a.m[0][2] * v.z,
          a.m[1][0] * v.x + a.m[1][1] * v.y + a.m[1][2] * v.z,
          a.m[2][0] * v.x + a.m[2][1] * v.y + a.m[2][2] * v.z};
}
inline V3 col(const M3& a, int j) { return {a.m[0][j], a.m[1][j], a.m[2][j]}; }
inline M3 transpose(const M3& a) {
  M3 r;
  for (int i = 0; i < 3; i++)
    for (int j = 0; j < 3; j++) r.m[i][j] = a.m[j][i];
  return r;
}
// A rotation as its rotation vector (axis x angle), and back (Rodrigues).
inline V3 rotLog(const M3& r) {
  const float tr = r.m[0][0] + r.m[1][1] + r.m[2][2];
  const float a = std::acos(std::fmax(-1.0f, std::fmin(1.0f, 0.5f * (tr - 1.0f))));
  const V3 w = v3(r.m[2][1] - r.m[1][2], r.m[0][2] - r.m[2][0], r.m[1][0] - r.m[0][1]);
  if (a < 1e-4f) return w * 0.5f;
  if (a > kPi - 1e-3f) {
    // Half a turn: the axis from R + I = 2 n n^T, its largest column.
    int k = 0;
    for (int i = 1; i < 3; i++)
      if (r.m[i][i] > r.m[k][k]) k = i;
    const V3 c = v3(r.m[0][k], r.m[1][k], r.m[2][k]) + v3(k == 0, k == 1, k == 2);
    return normalize(c) * a;
  }
  return w * (a / (2.0f * std::sin(a)));
}
inline M3 rotExp(V3 w) {
  const float a = std::sqrt(dot(w, w));
  if (a < 1e-6f) return {{{1, -w.z, w.y}, {w.z, 1, -w.x}, {-w.y, w.x, 1}}};
  const V3 n = w * (1.0f / a);
  const float c = std::cos(a), sn = std::sin(a), t = 1.0f - c;
  return {{{t * n.x * n.x + c, t * n.x * n.y - sn * n.z, t * n.x * n.z + sn * n.y},
           {t * n.x * n.y + sn * n.z, t * n.y * n.y + c, t * n.y * n.z - sn * n.x},
           {t * n.x * n.z - sn * n.y, t * n.y * n.z + sn * n.x, t * n.z * n.z + c}}};
}
// Blender XYZ euler: R = Rz * Ry * Rx.
inline M3 euler(float x, float y, float z) {
  float cx = std::cos(x), sx = std::sin(x), cy = std::cos(y), sy = std::sin(y);
  float cz = std::cos(z), sz = std::sin(z);
  M3 rx = {{{1, 0, 0}, {0, cx, -sx}, {0, sx, cx}}};
  M3 ry = {{{cy, 0, sy}, {0, 1, 0}, {-sy, 0, cy}}};
  M3 rz = {{{cz, -sz, 0}, {sz, cz, 0}, {0, 0, 1}}};
  return mul(rz, mul(ry, rx));
}

// Deterministic generator (splitmix64) + value noise for layouts.
struct Rng {
  uint64_t s;
  explicit Rng(uint64_t seed) : s(seed * 0x9E3779B97F4A7C15ull + 0x1234567ull) {}
  uint64_t next() {
    uint64_t z = (s += 0x9E3779B97F4A7C15ull);
    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ull;
    z = (z ^ (z >> 27)) * 0x94D049BB133111EBull;
    return z ^ (z >> 31);
  }
  float uniform() { return (float)((next() >> 40) * (1.0 / 16777216.0)); }
  float uniform(float a, float b) { return a + (b - a) * uniform(); }
  float gauss(float mu, float sigma) {
    float u1 = std::fmax(uniform(), 1e-7f), u2 = uniform();
    return mu + sigma * std::sqrt(-2.0f * std::log(u1)) * std::cos(kTau * u2);
  }
};

inline float hash3(int x, int y, int z, uint32_t seed) {
  uint32_t h = (uint32_t)x * 0x8da6b343u ^ (uint32_t)y * 0xd8163841u ^
               (uint32_t)z * 0xcb1ab31fu ^ seed * 0x9e3779b9u;
  h ^= h >> 15; h *= 0x2c1b3c6du; h ^= h >> 12; h *= 0x297a2d39u; h ^= h >> 15;
  return (float)(h & 0xffffff) * (1.0f / 16777216.0f);
}

// Smooth value noise in [-1, 1].
inline float vnoise(V3 p, uint32_t seed) {
  float fx = std::floor(p.x), fy = std::floor(p.y), fz = std::floor(p.z);
  int ix = (int)fx, iy = (int)fy, iz = (int)fz;
  float tx = p.x - fx, ty = p.y - fy, tz = p.z - fz;
  tx = tx * tx * (3 - 2 * tx); ty = ty * ty * (3 - 2 * ty); tz = tz * tz * (3 - 2 * tz);
  float c[2][2][2];
  for (int a = 0; a < 2; a++)
    for (int b = 0; b < 2; b++)
      for (int d = 0; d < 2; d++) c[a][b][d] = hash3(ix + a, iy + b, iz + d, seed);
  auto lx = [&](int b, int d) { return c[0][b][d] + (c[1][b][d] - c[0][b][d]) * tx; };
  float y0 = lx(0, 0) + (lx(1, 0) - lx(0, 0)) * ty;
  float y1 = lx(0, 1) + (lx(1, 1) - lx(0, 1)) * ty;
  return (y0 + (y1 - y0) * tz) * 2.0f - 1.0f;
}

// ---------------------------------------------------------------------------
// Layout: blocks in the structure's untilted frame.

struct Block {
  bool fragment;
  V3 rest;        // block: rest cell center; fragment: position
  V3 axis;        // block: jut axis; fragment: drift direction
  float jut;      // block: jut distance (m)
  V3 tip;         // block: tip direction in [-1,1]³; fragment: rotation (rad)
  V3 spin;        // fragment: tumble rate
  V3 size;        // fragment size; blocks use span
  int span[3];
  bool stone;
  float seed;     // [0,1): shell tint, edge wear, breathing phase
};

inline float jutStrain(float jut) { return jut / (kCell * kJutMax); }

int buildReference(Block* out) {
  int n = 0;
  for (int i = 0; i < kRefCount && n < kMaxBoxes; i++) {
    const RefBlock& r = kRefBlocks[i];
    Block& b = out[n++];
    b.fragment = r.fragment != 0;
    b.rest = v3(r.rest[0], r.rest[1], r.rest[2]);
    b.axis = v3(r.axis[0], r.axis[1], r.axis[2]);
    b.jut = r.jut;
    b.stone = r.stone != 0;
    b.seed = hash3(i, 7, 3, 21u);
    b.size = v3(r.size[0], r.size[1], r.size[2]);
    b.spin = v3(r.spin[0], r.spin[1], r.spin[2]);
    for (int a = 0; a < 3; a++) b.span[a] = r.span[a];
    if (b.fragment) {
      b.axis = normalize(b.axis);
      b.tip = v3(r.rot[0], r.rot[1], r.rot[2]);
    } else {
      // Recover the tip direction from the scene's rotation (its tip angles
      // are uniform in ±0.18 strain^1.5), so tipping follows the jut.
      float tip = std::pow(jutStrain(r.jut), 1.5f) * 0.18f;
      auto dirOf = [&](float a, int k) {
        return tip > 1e-4f ? std::fmax(-1.0f, std::fmin(1.0f, a / tip))
                           : hash3(i, k, 11, 5u) * 2.0f - 1.0f;
      };
      b.tip = v3(dirOf(r.rot[0], 0), dirOf(r.rot[1], 1), dirOf(r.rot[2], 2));
    }
  }
  return n;
}

// The scene's generator (part_structure.py), on our own RNG and noise: a
// hollow, slightly irregular superellipsoid shell of cells, some fused 2x2x2
// into slabs, each jutting along its nearest grid axis; plus six fragments.
int buildProcedural(Block* out, int seed) {
  Rng rng((uint64_t)seed);
  const uint32_t nseed = (uint32_t)seed * 977u + 13u;
  constexpr int kReach = 5;
  constexpr int kDim = 2 * kReach + 1;
  auto idx = [](int i, int j, int k) {
    return ((i + kReach) * kDim + (j + kReach)) * kDim + (k + kReach);
  };
  bool inside[kDim * kDim * kDim] = {};
  for (int i = -kReach; i <= kReach; i++)
    for (int j = -kReach; j <= kReach; j++)
      for (int k = -kReach; k <= kReach; k++) {
        V3 p = v3(i * kCell, j * kCell, k * kCell);
        float e = 3.2f;
        float r = std::pow(std::pow(std::fabs(p.x / kRadii[0]), e) +
                           std::pow(std::fabs(p.y / kRadii[1]), e) +
                           std::pow(std::fabs(p.z / kRadii[2]), e), 1.0f / e);
        r += 0.06f * vnoise(p * 0.4f + v3(3.1f, 7.7f, 1.3f), nseed);
        inside[idx(i, j, k)] = r <= 1.0f;
      }
  auto isIn = [&](int i, int j, int k) {
    if (i < -kReach || i > kReach || j < -kReach || j > kReach || k < -kReach || k > kReach)
      return false;
    return inside[idx(i, j, k)];
  };
  auto shell = [&](int i, int j, int k) {
    return isIn(i, j, k) &&
           (!isIn(i + 1, j, k) || !isIn(i - 1, j, k) || !isIn(i, j + 1, k) ||
            !isIn(i, j - 1, k) || !isIn(i, j, k + 1) || !isIn(i, j, k - 1));
  };
  auto fdiv = [](int a) { return a >= 0 ? a / 2 : -((-a + 1) / 2); };
  auto normalAt = [](V3 o) {
    return normalize(v3(o.x / (kRadii[0] * kRadii[0]), o.y / (kRadii[1] * kRadii[1]),
                        o.z / (kRadii[2] * kRadii[2])));
  };
  auto axisOf = [](V3 n) {
    float ax = std::fabs(n.x), ay = std::fabs(n.y), az = std::fabs(n.z);
    if (ax >= ay && ax >= az) return v3(n.x >= 0 ? 1.f : -1.f, 0, 0);
    if (ay >= az) return v3(0, n.y >= 0 ? 1.f : -1.f, 0);
    return v3(0, 0, n.z >= 0 ? 1.f : -1.f);
  };
  auto jutFor = [&](V3 n) {
    return kCell * kJutMax * std::pow(rng.uniform(), 1.7f) * (1.0f + 0.6f * std::fmax(n.z, 0.0f));
  };
  auto stoneFor = [&](V3 rest, float glow) {
    float s = vnoise(rest * 0.45f + v3(5.3f, -2.9f, 8.1f), nseed + 7u) + 0.3f - 1.1f * glow;
    return s > 0.0f;
  };

  int n = 0;
  const int kmin = fdiv(-kReach), kmax = fdiv(kReach);
  for (int gi = kmin; gi <= kmax; gi++)
    for (int gj = kmin; gj <= kmax; gj++)
      for (int gk = kmin; gk <= kmax; gk++) {
        int members[8][3];
        int m = 0;
        for (int i = -kReach; i <= kReach; i++)
          for (int j = -kReach; j <= kReach; j++)
            for (int k = -kReach; k <= kReach; k++)
              if (fdiv(i) == gi && fdiv(j) == gj && fdiv(k) == gk && shell(i, j, k)) {
                members[m][0] = i; members[m][1] = j; members[m][2] = k;
                m++;
              }
        if (m == 0) continue;
        auto place = [&](V3 center, const int* span, float jut_mul) {
          if (n >= kMaxBoxes) return;
          V3 nn = normalAt(center);
          Block& b = out[n++];
          b.fragment = false;
          b.rest = center;
          b.axis = axisOf(nn);
          b.jut = jutFor(nn) * jut_mul;
          b.tip = v3(rng.uniform(-1, 1), rng.uniform(-1, 1), rng.uniform(-1, 1));
          b.spin = v3(0, 0, 0);
          for (int a = 0; a < 3; a++) b.span[a] = span[a];
          b.size = v3(span[0] * kCell, span[1] * kCell, span[2] * kCell);
          b.seed = rng.uniform();
          b.stone = stoneFor(center, jutStrain(b.jut));
        };
        if (m >= 4 && rng.uniform() < 0.45f) {
          int lo[3] = {99, 99, 99}, hi[3] = {-99, -99, -99};
          for (int q = 0; q < m; q++)
            for (int a = 0; a < 3; a++) {
              lo[a] = members[q][a] < lo[a] ? members[q][a] : lo[a];
              hi[a] = members[q][a] > hi[a] ? members[q][a] : hi[a];
            }
          int span[3] = {hi[0] - lo[0] + 1, hi[1] - lo[1] + 1, hi[2] - lo[2] + 1};
          place(v3((lo[0] + hi[0]) * 0.5f * kCell, (lo[1] + hi[1]) * 0.5f * kCell,
                   (lo[2] + hi[2]) * 0.5f * kCell), span, 0.6f);
        } else {
          for (int q = 0; q < m; q++) {
            int span[3] = {1, 1, 1};
            place(v3(members[q][0] * kCell, members[q][1] * kCell, members[q][2] * kCell),
                  span, 1.0f);
          }
        }
      }
  for (int f = 0; f < 6 && n < kMaxBoxes; f++) {
    V3 d = normalize(v3(rng.gauss(0, 1), rng.gauss(0, 1), std::fabs(rng.gauss(0.4f, 1))));
    float reach = rng.uniform(1.35f, 2.0f);
    Block& b = out[n++];
    b.fragment = true;
    b.rest = v3(d.x * kRadii[0], d.y * kRadii[1], d.z * kRadii[2]) * reach;
    float s = kCell * 0.94f * rng.uniform(0.25f, 0.45f);
    b.size = v3(s, s, s);
    b.tip = v3(rng.uniform(-kPi, kPi), rng.uniform(-kPi, kPi), rng.uniform(-kPi, kPi));
    b.spin = v3(rng.uniform(-0.12f, 0.12f), rng.uniform(-0.12f, 0.12f), rng.uniform(-0.12f, 0.12f));
    b.axis = d;
    b.jut = 0.0f;
    for (int a = 0; a < 3; a++) b.span[a] = 1;
    b.stone = true;
    b.seed = rng.uniform();
  }
  return n;
}

// ---------------------------------------------------------------------------
// Gliding between layouts.
//
// What is drawn is a list of SLOTS, not the layout's blocks: each slot is a
// block on screen, heading for one of the layout's blocks (or, when the new
// layout has fewer, shrinking into its nearest one on its way out) on a stiff
// spring (sketch/glide.h) that overshoots and rings out. A new layout pairs
// what is on screen with its blocks nearest first; blocks left over grow out
// of their nearest slot. Slots keep their index while they glide, so the
// flakes keep their blocks; a slot that has gone is compacted away and the
// flakes are told where everything moved (p_update's remap).

// A block's pose in the structure's frame (before the egg's turn).
struct Pose {
  V3 c = {0, 0, 0}, half = {0, 0, 0};
  M3 rot = {{{1, 0, 0}, {0, 1, 0}, {0, 0, 1}}};
  float glow = 0.0f;
};

// The glide's channels: the center (3), the half size (3), the turn (a
// rotation vector, 3) — offsets from the goal.
constexpr int kGlideCh = 9;

struct Slot {
  int block = -1;          // the layout's block it is (becoming); -1: leaving
  int into = -1;           // leaving: the block it shrinks into
  bool fragment = false;
  glide::Element<kGlideCh> g;
  Pose shown;              // as last drawn
  float jut = 1.0f;        // its Jut and Seam this frame, sends and all
  float seam = 0.057f;
  bool stone = false;      // what it was made of before the change...
  float seed = 0.0f;
  float flip = 0.0f;       // ...for this long yet (s), then the goal's
  float circ = -1.0f;      // the axis flakes circulate it about (row 4.w)
  float charge = 0.0f;     // its glint charge (row 5.x)
  int prev = -1;           // its index last frame (-1: new)
  bool dead = false;       // dropped to make room: gone this frame
};

// ---------------------------------------------------------------------------
// The sends: what the sweep does to the blocks and the core, as a matrix.
// Each target is its own knob plus, from each source, its send x that
// source:
//
//   Envelope   the sustain as the flakes feel it: Sustain at home, 0 at the
//              choke, past it on an overshoot (0..1.5);
//   Overshoot  the bounce alone, again (0 at rest, ~0.7 at a hard return):
//              to accentuate an overshoot beyond what the envelope carries;
//   Glint      each block's own charge from the glint band (the core: the
//              most charged block's);
//   Reverb     the thrown tail, 1 at a full throw ringing down to 0 over Tail.

enum Source { kEnv, kOver, kGlint, kReverb, kSources };
enum Target { kJut, kSeam, kJitter, kPower, kMagnet, kTargets };

// Field keys, labels and ranges, [source][target] and [target].
const char* const kSendKey[kSources][kTargets] = {
    {"env_jut", "env_seam", "env_jitter", "env_power", "env_magnet"},
    {"over_jut", "over_seam", "over_jitter", "over_power", "over_magnet"},
    {"glint_jut", "glint_seam", "glint_jitter", "glint_power", "glint_magnet"},
    {"reverb_jut", "reverb_seam", "reverb_jitter", "reverb_power", "reverb_magnet"}};
const char* const kSendLabel[kSources][kTargets] = {
    {"Env Jut", "Env Seam", "Env Jitter", "Env Power", "Env Magnet"},
    {"Over Jut", "Over Seam", "Over Jitter", "Over Power", "Over Magnet"},
    {"Glint Jut", "Glint Seam", "Glint Jitter", "Glint Power", "Glint Magnet"},
    {"Reverb Jut", "Reverb Seam", "Reverb Jitter", "Reverb Power", "Reverb Magnet"}};
const char* const kSendShort[kSources][kTargets] = {
    {"EJut", "ESeam", "EJit", "EPwr", "EMag"},
    {"OJut", "OSeam", "OJit", "OPwr", "OMag"},
    {"GJut", "GSeam", "GJit", "GPwr", "GMag"},
    {"RJut", "RSeam", "RJit", "RPwr", "RMag"}};
// A send's reach either way, in its target's units.
constexpr float kSendRange[kTargets] = {2.0f, 0.3f, 0.5f, 1.0f, 4.0f};
// Where the result is held.
constexpr float kTargetLo[kTargets] = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
constexpr float kTargetHi[kTargets] = {3.0f, 0.45f, 1.0f, 1.25f, 8.0f};
// Jitter's turn per metre of knock (rad).
constexpr float kJitterTurn = 0.5f;

// ---------------------------------------------------------------------------

static gpu::ComputePSO s_pso_shadow, s_pso_escape, s_pso_reduce, s_pso_vis, s_pso_scene,
    s_pso_fog, s_pso_blur, s_pso_comp, s_pso_prefill, s_pso_aux;
static gpu::ComputePSO s_pso_p_update, s_pso_p_resolve, s_pso_p_shade, s_pso_motion;
static gpu::RenderPSO s_pso_spark, s_pso_spark_mv;

struct BlurUniforms { float dir_x, dir_y, spacing, _pad; };

struct State {
  bool initialized = false;
  gpu::Buffer ub_frame, box_buf, ub_blur_h, ub_blur_v;
  gpu::Texture atlas, scene_tex, fog_i, fog_t, fog_tmp, fog_blur, zero_tex;
  gpu::Texture esc, esc_low, vis_f, vis_d;   // the fog's view of the core
  gpu::Sampler samp;
  int scratch_w = 0, scratch_h = 0;

  Block blocks[kMaxBoxes];
  int block_count = 0;
  int built_layout = -1;
  // What is drawn: slots gliding to the layout's blocks; their rows, then
  // (on a frame slots were compacted) the flakes' remap, a row per last
  // frame's slot.
  Slot slots[kMaxBoxes];
  int slot_count = 0;
  int last_slot_count = 0;
  uint32_t slot_serial = 0;
  bool pose_jump = false;      // the blocks jumped this frame: no motion to show
  Row rows[kMaxBoxes * (kBoxRows + 1)];

  // The flakes on the GPU: state (kPRows a flake), the presence grid's
  // counts + running average, render records (kRRows a flake).
  gpu::Buffer ub_part, p_state, p_grid, p_accum, p_recs, prev_box_buf;
  gpu::Texture presence, presence_zero, spark, motion_tex;
  int spark_w = 0, spark_h = 0, motion_w = 0, motion_h = 0;
  // The optional outputs (made only once wired): Aux 1 / 2, render_outputs'
  // depth, the scene's glint charge per pixel; and 1x1 stand-ins bound in
  // their place, never written.
  gpu::Texture aux_tex[2], depth_tex, glint_tex, stand_in[3];
  int aux_w[2] = {}, aux_h[2] = {}, depth_w = 0, depth_h = 0, glint_w = 0, glint_h = 0;
  int p_live = 0;            // flakes the state buffer holds
  // Last frame's poses, for motion vectors.
  Row prev_rows[kMaxBoxes * kBoxRows];
  int prev_count = -1;
  float prev_cam[3][4] = {};
  bool prev_cam_valid = false;

  // Params.
  int layout = 0;
  float glide_time = 0.4f;     // s a change of layout rings out over (0: a cut)
  float stiffness = 10.0f;     // Hz it rings at
  float glitch = 0.5f;
  float glide_smooth = 0.0f;   // 0..1: how much calmer the glide is
  float jut = 1.0f;
  float twang = 1.0f;          // how much of a change of Jut goes through the spring
  float jitter = 0.0f;         // m, how hard the blocks are knocked about
  float jitter_rate = 4.0f;    // Hz, knocks a block takes a second
  float breathe = 0.0f;
  float breathe_rate = 0.3f;
  float seam = 0.057f;
  float spin = 0.0f;
  int spin_axis = 0;   // 0 = vertical (world up), 1 = the structure's own axis
  float drift = 0.3f;
  float core_power = 0.5f;
  float core_r = 1.0f, core_g = 0.56f, core_b = 0.26f;
  float core_radius = 0.25f;
  float core_clear = 0.4f;     // blocks nearer the center than this pass the beams
  float wander = 0.0f;
  float wander_rate = 0.3f;
  float flicker = 0.0f;
  float halo = 0.08f;
  float water = 0.013f;
  float air = 0.0f;          // the air wall's clarity (0 = off)
  float air_depth = 6.0f;    // m of water left in front of the structure's center
  float air_soft = 4.0f;     // m, the wall's soft edge
  float wisps = 1.0f;
  float halo_phase = 0.2f;
  float ambient = 2.0f;
  float rim = 8.0f;
  float kicker = 6000.0f;
  float yaw = -10.0f;
  float pitch = -15.0f;
  float distance = 38.0f;
  float lens = 135.0f;
  float orbit = 0.0f;
  float exposure = 0.0f;
  float contrast = 0.2f;
  float bevel = 0.1f;
  float reflect = 1.0f;
  float ember = 1.0f;
  float shell_glow = 1.0f;
  float chips = 1.0f;
  float albedo = 1.0f;
  float opacity = 1.0f;
  int debug_view = 0;
  int aux_what[2] = {1, 3};    // what Aux 1 / 2 carry (LTN_AUX_*): Depth, In-scatter

  // The sweep.
  float sweep = sweep_knob::kSweepCenter;
  float deadzone = 0.45f;
  float bounce = 0.55f;
  float sense = 2.0f;
  float tail = 2.4f;           // s the thrown tail rings out over
  float sustain = 1.0f;        // the level the knob's full position holds
  float overshoot = 1.0f;
  float glint_angle = 35.0f;
  float glint_width = 0.12f;
  float glint_time = 1.2f;     // s a glint takes to cross once the knob rests
  float glint_fade = 0.15f;    // s (1/e) a block keeps its charge after the band
  float glint_choke = 0.75f;   // crossings: a glint short of this at the choke fades out
  float glint_pull = 1.0f;
  float magnetism = 1.0f;      // the blocks' pull on the flakes, x their strain
  // The sends, [source][target] (see Source / Target); all 0 but a glint's
  // pull on the magnets.
  float sends[kSources][kTargets] = {{0, 0, 0, 0, 0}, {0, 0, 0, 0, 0},
                                     {0, 0, 0, 0, 1.5f}, {0, 0, 0, 0, 0}};
  // The flakes.
  int flakes = 1200;
  float flake_size = 1.0f;
  float mica = 0.3f;
  float sparkle = 1.5f;
  float tumble = 1.0f;
  float ring_radius = 4.2f;
  float ring_loose = 0.6f;
  float ring_speed = 1.0f;
  float jet = 0.6f;
  float jet_angle = 20.0f;
  float capture = 0.6f;
  float bite = 0.5f;
  float reach = 0.5f;
  float standoff = 0.08f;
  float standoff_deep = 0.7f;
  float speed_deep = 3.0f;
  float local = 0.85f;
  float orbit_speed = 10.0f;
  float streak = 1.0f;
  float motion_flakes = 1.0f;  // scales the flakes' motion vectors
  float motion_scene = 1.0f;   // ...and everything else's (the blocks)
  float heat = 1.0f;
  float hold = 1.0f;
  float stick = 0.8f;
  float ricochet = 0.6f;
  float vortex = 1.0f;
  float lanes = 0.7f;
  float fling = 1.0f;
  float shell_radius = 4.0f;
  float vibe = 0.06f;
  float vibe_hz = 16.0f;
  float vibe_spread = 0.25f;
  float dust = 0.5f;

  sweep_knob::Knob knob;
  three_planes_glints::Core glint_core;
  // The most any block is charged by a glint (each slot's own: Slot.charge).
  float glint_cmax = 0.0f;
  double sim_dt = 0.0;         // time the flakes haven't been stepped through
  float frame_dt = 1.0f / 60.0f;
  float pending_fling = 0.0f;  // a throw this frame (its charge)
  float pending_bite = 0.0f;   // how far the sustain has risen since the flakes last stepped
  float bite_prev_S = -1.0f;
  double vibe_phase = 0.0;

  // Accumulators.
  double time = 0.0;
  double breathe_phase = 0.0;
  double spin_phase = 0.0;
  double wander_phase = 0.0;
  double orbit_phase = 0.0;
  double drift_phase = 0.0;
  uint32_t frame = 0;
};

glide::Params glideParams(const State* s) {
  glide::Params p;
  p.time = std::fmax(s->glide_time, 0.0f);
  p.stiffness = std::fmax(s->stiffness, 0.1f);
  // In step with the ringing (0.4 of its period at 1), so Smooth reads the
  // same at any Stiffness: the first overshoot ~60% -> ~40% at 0.5 -> ~13%
  // at 1.
  p.smooth = 0.4f * std::fmin(std::fmax(s->glide_smooth, 0.0f), 1.0f) / p.stiffness;
  return p;
}

// The sends' sources this frame (Glint's is the core's: the most charged
// block's; each block reads its own).
struct Drive { float src[kSources]; };

Drive driveOf(const State* s) {
  Drive d;
  const float over = 2.5f * s->knob.bounce;
  d.src[kEnv] = std::fmin(1.5f, std::fmax(0.0f, s->sustain * s->knob.level + s->overshoot * over));
  d.src[kOver] = over;
  d.src[kGlint] = s->glint_cmax;
  d.src[kReverb] = s->knob.release;
  return d;
}

// Target `t`: its own knob `base` plus every send x its source, `glint` the
// charge of the block it is for.
float sent(const State* s, const Drive& d, int t, float base, float glint) {
  float v = base;
  for (int k = 0; k < kSources; k++) v += s->sends[k][t] * (k == kGlint ? glint : d.src[k]);
  return std::fmin(kTargetHi[t], std::fmax(kTargetLo[t], v));
}

inline double rateHz(float k, double mid) {
  if (std::fabs(k) <= 0.001f) return 0.0;
  double s = k < 0 ? -1.0 : 1.0;
  return s * mid * std::pow(2.0, (std::fabs((double)k) - 0.5) * 5.0);
}

void module_init() {
  auto schema = state::Schema()
      .helpField("intro",
        "## Lantern\n"
        "A hollow cluster of stone and shell blocks hanging in deep water, a "
        "light burning at its heart. The light only escapes through the seams "
        "and through the gaps where blocks have jutted out, cutting beams "
        "through a halo of murky water around it.\n\n"
        "**Try:** *Breathe* up for blocks that drift open and shut; *Wander* "
        "to swing the beams around; *Seam* wider to let the light out.")
      .group("structure", "Structure")
        .groupHelp(
          "*Layout* 0 is the poster scene's exact arrangement; other numbers "
          "grow new ones the same way. A new layout doesn't cut in: the blocks "
          "spring there, overshooting and ringing out over *Glide* (0 cuts), "
          "ringing *Stiffness* times a second; *Glitch* breaks it up — blocks "
          "letting go one after another, moving an axis at a time, freezing "
          "for a few frames and jumping on; *Smooth* calms all of it, "
          "rounding off the overshoot, the ringing and the jumps. *Jut* scales how far the blocks stand "
          "out from the core, and *Twang* is how much of a change of it goes "
          "through that spring too (1: a jut driven by the sweep twangs; 0: "
          "the blocks follow it at once). *Jitter* knocks every block about "
          "on the same spring, *Jitter Rate* knocks a second each, so they "
          "ring on their own. *Breathe* rolls slow waves through them on top "
          "of that, some sealing shut while others push out, at *Breathe Rate*. *Seam* is the gap "
          "between neighboring blocks. *Spin* turns the whole structure; "
          "*Drift* sets the loose fragments bobbing.")
      .intField("layout", 0, 0, 999, state::SecondaryInput)
          .label("Layout", "Lay")
      .floatField("glide", 0.4f, 0.f, 2.f, state::PrimaryInput, nullptr, 0.f, "s")
          .label("Glide", "Glide")
      .floatField("stiffness", 10.0f, 2.f, 40.f, state::PrimaryInput, nullptr, 0.f, "Hz")
          .logScale()
          .label("Stiffness", "Stiff")
      .floatField("glitch", 0.5f, 0.f, 1.f, state::PrimaryInput)
          .label("Glitch", "Glitch")
      .floatField("glide_smooth", 0.0f, 0.f, 1.f, state::PrimaryInput)
          .label("Smooth", "Smth")
      .floatField("jut", 1.0f, 0.f, 2.f, state::PrimaryInput)
          .label("Jut", "Jut")
      .floatField("twang", 1.0f, 0.f, 1.f, state::PrimaryInput)
          .label("Twang", "Twng")
      .floatField("jitter", 0.0f, 0.f, 0.5f, state::PrimaryInput, nullptr, 0.f, "m")
          .label("Jitter", "Jit")
      .floatField("jitter_rate", 4.0f, 0.25f, 30.f, state::PrimaryInput, nullptr, 0.f, "Hz")
          .logScale()
          .label("Jitter Rate", "JRate")
      .floatField("breathe", 0.0f, 0.f, 1.f, state::PrimaryInput)
          .label("Breathe", "Brth")
      .floatField("breathe_rate", 0.3f, 0.f, 1.f, state::PrimaryInput)
          .label("Breathe Rate", "BRate")
      .floatField("seam", 0.057f, 0.f, 0.3f, state::PrimaryInput, nullptr, 0.f, "m")
          .label("Seam", "Seam")
      .floatField("spin", 0.0f, -1.f, 1.f, state::PrimaryInput)
          .label("Spin", "Spin")
      .selectField("spin_axis", 0, state::SecondaryInput, {{"Vertical", 0}, {"Object", 1}})
          .label("Spin Axis", "Axis")
      .floatField("drift", 0.3f, 0.f, 1.f, state::PrimaryInput)
          .label("Drift", "Drift")
      .group("core", "Core")
        .groupHelp(
          "The light inside. *Power* is its brightness (the scene's 8 kW at "
          "the middle), *Radius* how large it is — larger softens the beams. "
          "*Wander* moves it around inside the hollow, which swings every beam "
          "at once; *Flicker* makes it gutter. *Clear*: a block that strays "
          "nearer the middle than this lets the beams through, fading out the "
          "nearer it comes, so one brushing the heart doesn't mute the core.")
      .floatField("core_power", 0.5f, 0.f, 1.f, state::PrimaryInput)
          .label("Power", "Pwr")
      .rgbField("core_color", 1.0f, 0.56f, 0.26f, state::PrimaryInput)
          .label("Color", "Col")
      .floatField("core_radius", 0.25f, 0.05f, 2.f, state::PrimaryInput, nullptr, 0.f, "m")
          .label("Radius", "Rad")
      .floatField("core_clear", 0.4f, 0.f, 1.5f, state::PrimaryInput, nullptr, 0.f, "m")
          .label("Clear", "Clr")
      .floatField("wander", 0.0f, 0.f, 1.f, state::PrimaryInput)
          .label("Wander", "Wndr")
      .floatField("wander_rate", 0.3f, 0.f, 1.f, state::PrimaryInput)
          .label("Wander Rate", "WRate")
      .floatField("flicker", 0.0f, 0.f, 1.f, state::PrimaryInput)
          .label("Flicker", "Flkr")
      .group("water", "Water")
        .groupHelp(
          "*Halo* is the thick water hanging around the structure that the "
          "beams light up; *Wisps* breaks it into clumps; *Halo Phase* pushes "
          "its glow toward looking into the light. *Water* is the thin "
          "blue depth everywhere else. *Air* clears the water between the "
          "camera and a wall *Air Depth* in front of the structure (its edge "
          "*Air Soft* wide) -- not physical, but it lets thick water surround "
          "the structure without burying it.")
      .floatField("halo", 0.08f, 0.f, 0.3f, state::PrimaryInput)
          .label("Halo", "Halo")
      .floatField("wisps", 1.0f, 0.f, 1.f, state::PrimaryInput)
          .label("Wisps", "Wisp")
      .floatField("halo_phase", 0.2f, 0.f, 0.9f, state::PrimaryInput)
          .label("Halo Phase", "HPh")
      .floatField("water", 0.013f, 0.001f, 0.25f, state::PrimaryInput)
          .logScale()
          .label("Water", "Water")
      .floatField("air", 0.0f, 0.f, 1.f, state::PrimaryInput)
          .label("Air", "Air")
      .floatField("air_depth", 6.0f, 0.f, 40.f, state::PrimaryInput, nullptr, 0.f, "m")
          .label("Air Depth", "ADep")
      .floatField("air_soft", 4.0f, 0.f, 20.f, state::PrimaryInput, nullptr, 0.f, "m")
          .label("Air Soft", "ASoft")
      .group("lights", "Lights")
        .groupHelp(
          "The cool lights outside: *Ambient* is the faint blue from far "
          "above (it also lights the water), *Rim* a grazing light from "
          "behind that traces edges, *Kicker* a broad cool light behind and "
          "to the left that catches the glossy bevels.")
      .floatField("ambient", 2.0f, 0.f, 8.f, state::PrimaryInput)
          .label("Ambient", "Amb")
      .floatField("rim", 8.0f, 0.f, 32.f, state::PrimaryInput)
          .label("Rim", "Rim")
      .floatField("kicker", 6000.0f, 0.f, 24000.f, state::PrimaryInput, nullptr, 0.f, "W")
          .label("Kicker", "Kick")
      .group("camera", "Camera")
      .floatField("yaw", -10.0f, -180.f, 180.f, state::PrimaryInput, nullptr, 0.f, "deg")
          .label("Yaw", "Yaw")
      .floatField("pitch", -15.0f, -85.f, 85.f, state::PrimaryInput, nullptr, 0.f, "deg")
          .label("Pitch", "Ptch")
      .floatField("distance", 38.0f, 6.f, 150.f, state::PrimaryInput, nullptr, 0.f, "m")
          .label("Distance", "Dist")
      .floatField("lens", 135.0f, 18.f, 300.f, state::PrimaryInput, nullptr, 0.f, "mm")
          .label("Lens", "Lens")
      .floatField("orbit", 0.0f, -1.f, 1.f, state::PrimaryInput)
          .label("Orbit", "Orbit")
      .group("render", "Render")
        .groupHelp(
          "*Exposure* in stops ahead of the AgX tonemap; *Contrast* steepens "
          "it (the scene used Medium High Contrast, about 0.2). *Albedo* scales "
          "everything the blocks reflect (toward 0 they become dark "
          "silhouettes against the lit water; what glows still glows). *Bevel* is "
          "the widest rounded edge wear, *Chips* how often edges are chipped "
          "and corners sheared off, *Reflect* scales the glossy reflections, "
          "*Ember* the glow in the grooves of straining stone, *Shell Glow* "
          "the warmth of straining shell blocks (the whole block glows, so "
          "a little goes a long way).")
      .floatField("exposure", 0.0f, -4.f, 4.f, state::PrimaryInput, nullptr, 0.f, "EV")
          .label("Exposure", "Expo")
      .floatField("contrast", 0.2f, -0.5f, 1.f, state::PrimaryInput)
          .label("Contrast", "Ctr")
      .floatField("albedo", 1.0f, 0.f, 2.f, state::PrimaryInput)
          .label("Albedo", "Albd")
      .floatField("bevel", 0.1f, 0.f, 0.3f, state::PrimaryInput, nullptr, 0.f, "m")
          .label("Bevel", "Bev")
      .floatField("chips", 1.0f, 0.f, 1.f, state::PrimaryInput)
          .label("Chips", "Chip")
      .floatField("reflect", 1.0f, 0.f, 2.f, state::PrimaryInput)
          .label("Reflect", "Refl")
      .floatField("ember", 1.0f, 0.f, 4.f, state::PrimaryInput)
          .label("Ember", "Embr")
      .floatField("shell_glow", 1.0f, 0.f, 4.f, state::PrimaryInput)
          .label("Shell Glow", "SGlw")
      .floatField("opacity", 1.0f, 0.f, 1.f, state::PrimaryInput)
          .label("Opacity", "Opac")
      .group("sweep", "Sweep")
        .groupHelp(
          "One knob plays the flakes, like a strummed string. **Sweep** is "
          "the knob: 0.5 is home. Across the middle (*Deadzone*) the flakes "
          "SUSTAIN at *Sustain*: the blocks become magnets that catch them "
          "into fast, tight orbits, and a jet blows the free ones through the "
          "core. Toward either end the sustain fades, and at the end (the "
          "choke) they let go into loose ballistic orbits around a ring. "
          "Sweep back in fast and it *Overshoots* -- tighter and faster for "
          "a moment. Each pass through the middle sends a GLINT across the "
          "screen, a slanted band (*Glint Angle*, *Glint Width*; as fast as "
          "the sweep, or *Glint Time* to cross once the knob rests) that "
          "makes the blocks under it stronger magnets (the Glint sends' "
          "*Magnet*), for a moment after it passes (*Glint Fade*; one that has barely "
          "started when the sweep chokes fades out, *Glint Choke*), and lures "
          "the flakes off the blocks around them onto those (*Glint Pull*): "
          "the streams are handed along with the band. And a hard "
          "pass through the middle that goes on to the choke THROWS: the "
          "flakes are flung onto a sphere around the structure and shaken "
          "there while the tail rings out over *Tail* seconds -- stop short "
          "of the choke and nothing is thrown (a muted string).")
      .floatField("sweep", 0.5f, 0.f, 1.f, state::PrimaryInput, nullptr, 0.f, nullptr,
                  "The knob. 0.5 is home: full sustain, no motion.").wireCombine("replace")
          .label("Sweep", "Sweep")
      .floatField("sustain", 1.0f, 0.f, 1.f, state::PrimaryInput, nullptr, 0.f, nullptr,
                  "How hard the blocks hold the flakes with the knob in the middle.")
          .label("Sustain", "Sus")
      .floatField("deadzone", 0.45f, 0.f, 0.95f, state::SecondaryInput, nullptr, 0.f, nullptr,
                  "Fraction of each half of the knob that stays at full sustain.")
          .label("Deadzone", "DZ")
      .floatField("bounce", 0.55f, 0.f, 1.f, state::SecondaryInput, nullptr, 0.f, nullptr,
                  "How far ahead of the knob a fast return runs.")
          .label("Bounce", "Bnc")
      .floatField("overshoot", 1.0f, 0.f, 4.f, state::PrimaryInput, nullptr, 0.f, nullptr,
                  "How much of that lead pushes the sustain past its level.")
          .label("Overshoot", "Over")
      .floatField("sense", 2.0f, 0.1f, 8.f, state::SecondaryInput, nullptr, 0.f, nullptr,
                  "Knob ranges per second that count as a full-speed sweep.")
          .label("Sensitivity", "Sens")
      .floatField("tail", 2.4f, 0.2f, 8.f, state::PrimaryInput, nullptr, 0.f, "s",
                  "How long a throw's tail rings out.")
          .label("Tail", "Tail")
      .floatField("glint_angle", 35.0f, -90.f, 90.f, state::SecondaryInput, nullptr, 0.f, "deg",
                  "Which way the glints cross the screen; 0 left to right.")
          .label("Glint Angle", "GAng")
      .floatField("glint_width", 0.12f, 0.02f, 0.5f, state::SecondaryInput, nullptr, 0.f, nullptr,
                  "A glint's half-width, as a fraction of the structure it crosses.")
          .label("Glint Width", "GWid")
      .floatField("glint_time", 1.2f, 0.2f, 6.f, state::SecondaryInput, nullptr, 0.f, "s",
                  "How long a glint takes to cross the screen once the knob rests.")
          .label("Glint Time", "GTime")
      .floatField("glint_fade", 0.15f, 0.f, 2.f, state::SecondaryInput, nullptr, 0.f, "s",
                  "How long a block keeps its glint charge once the band has passed "
                  "(to about a third); 0 lets go the moment it has crossed.")
          .label("Glint Fade", "GFade")
      .floatField("glint_choke", 0.75f, 0.f, 1.f, state::SecondaryInput, nullptr, 0.f, nullptr,
                  "How far across a glint can be when the sweep reaches the choke and "
                  "still fade out there, rather than finish crossing a picture that "
                  "has let go (a full throw reaches the choke with its glint about "
                  "halfway); 0 lets every glint finish.")
          .label("Glint Choke", "GChk")
      .floatField("glint_pull", 1.0f, 0.f, 3.f, state::PrimaryInput, nullptr, 0.f, nullptr,
                  "How hard a glint lures flakes off the blocks around it onto the "
                  "blocks under it.")
          .label("Glint Pull", "GPull");
  // The sends: a group per source, a send per target.
  {
    static const char* const kGroup[kSources][2] = {
        {"sends_env", "Envelope Sends"}, {"sends_over", "Overshoot Sends"},
        {"sends_glint", "Glint Sends"}, {"sends_reverb", "Reverb Sends"}};
    static const char* const kHelp[kSources] = {
        "What the sweep's ENVELOPE does to the blocks and the core: each send "
        "adds that much of its target at full sustain (the knob at home: "
        "*Sustain*), none at the choke, and more on an overshoot (up to 1.5x). "
        "*Jut* in Jut's units, *Seam* in metres, *Jitter* in metres, *Power* "
        "in Power's, *Magnet* in Magnetism's; negative sends take away. So a "
        "block's jut is Jut + Env Jut x envelope + Over Jut x overshoot + "
        "Glint Jut x its glint + Reverb Jut x reverb.",
        "The OVERSHOOT again, on its own -- 0 at rest, about 0.7 coming back in "
        "hard -- for targets that should leap on a return more than the "
        "envelope already makes them.",
        "Each block's GLINT charge (as the band crosses it, fading over "
        "*Glint Fade* after); the core reads the most charged block's. *Magnet* is how "
        "much stronger a glint makes the magnets under it.",
        "The REVERB: the thrown tail, 1 at a full throw and ringing down to 0 "
        "over *Tail*."};
    for (int k = 0; k < kSources; k++) {
      schema.group(kGroup[k][0], kGroup[k][1]).groupHelp(kHelp[k]);
      for (int t = 0; t < kTargets; t++)
        schema.floatField(kSendKey[k][t], k == kGlint && t == kMagnet ? 1.5f : 0.0f,
                          -kSendRange[t], kSendRange[t], state::PrimaryInput)
            .label(kSendLabel[k][t], kSendShort[k][t]);
    }
  }
  schema
      .group("flakes", "Flakes")
        .groupHelp(
          "The specks of mica, grit and silt flying around the structure. "
          "*Flakes* is how many (0 turns them off); *Size* scales the "
          "poster's 4-13 mm flakes; *Mica* is the share that are polished "
          "and glint, the rest are dark grit and dull silt that show against "
          "the glow; *Sparkle* scales how bright they shine and *Tumble* how "
          "fast they turn; *Streak* stretches each into an ember along its "
          "flight (in frames of travel, 0 a dot) and *Heat* is how hot a fast "
          "one glows. Choked, they orbit a ring *Ring Radius* out, "
          "*Ring Loose*, at *Ring Speed*. Sustained, *Jet* blows the free "
          "ones through the core along *Jet Angle* (0 is the screen's left "
          "to right), and the blocks catch them: *Capture* is the magnets' "
          "pull, *Magnetism* how strong each block is as a magnet (x its "
          "strain; the sends drive it), *Reach* how far it carries; a caught flake streams round the "
          "block that caught it (*Local* 1; toward 0, round the whole structure) "
          "in a lane above its surface, keeping its momentum, its altitude "
          "held by a spring: soft and flexible deep in the sweep, nearly "
          "pinching in the middle (*Hold* scales it), so it rides up over an "
          "edge under the load and springs back, and only a big excursion "
          "tears it loose. In flight the pull (*Stick*) swings it back in; "
          "striking a block it ricochets (*Ricochet*). Over a face square to "
          "a block's spin axis the flakes VORTEX up off it, circling, until "
          "they fly off (*Vortex*). Strained blocks hold "
          "hardest. *Lanes* draws it into the ember grooves. How deep the knob is in its "
          "sweep sets the rest: in the middle the lanes hug at *Standoff* and "
          "run at *Orbit Speed*; toward either end they widen to *Standoff "
          "Deep* and slow to *Speed Deep* while the magnets weaken, until the "
          "choke lets them go. *Dust* is how much where they crowd darkens the "
          "light in the water.")
      .intField("flakes", 1200, 0, kMaxFlakes, state::SecondaryInput)
          .label("Flakes", "Flks")
      .floatField("flake_size", 1.0f, 0.25f, 4.f, state::PrimaryInput)
          .label("Size", "Size")
      .floatField("mica", 0.3f, 0.f, 1.f, state::PrimaryInput)
          .label("Mica", "Mica")
      .floatField("sparkle", 1.5f, 0.f, 4.f, state::PrimaryInput)
          .label("Sparkle", "Sprk")
      .floatField("tumble", 1.0f, 0.f, 4.f, state::PrimaryInput)
          .label("Tumble", "Tmbl")
      .floatField("streak", 1.0f, 0.f, 4.f, state::PrimaryInput)
          .label("Streak", "Strk")
      .floatField("heat", 1.0f, 0.f, 4.f, state::PrimaryInput)
          .label("Heat", "Heat")
      .floatField("ring_radius", 4.2f, 1.f, 8.f, state::PrimaryInput, nullptr, 0.f, "m")
          .label("Ring Radius", "RRad")
      .floatField("ring_loose", 0.6f, 0.f, 1.f, state::PrimaryInput)
          .label("Ring Loose", "RLse")
      .floatField("ring_speed", 1.0f, 0.f, 4.f, state::PrimaryInput, nullptr, 0.f, "m/s")
          .label("Ring Speed", "RSpd")
      .floatField("jet", 0.6f, 0.f, 1.f, state::PrimaryInput)
          .label("Jet", "Jet")
      .floatField("jet_angle", 20.0f, -180.f, 180.f, state::SecondaryInput, nullptr, 0.f, "deg")
          .label("Jet Angle", "JAng")
      .floatField("capture", 0.6f, 0.f, 1.f, state::PrimaryInput)
          .label("Capture", "Capt")
      .floatField("magnetism", 1.0f, 0.f, 4.f, state::PrimaryInput)
          .label("Magnetism", "Mag")
      .floatField("bite", 0.5f, 0.f, 1.f, state::PrimaryInput)
          .label("Bite", "Bite")
      .floatField("reach", 0.5f, 0.05f, 2.f, state::PrimaryInput, nullptr, 0.f, "m")
          .label("Reach", "Rch")
      .floatField("standoff", 0.08f, 0.01f, 0.5f, state::PrimaryInput, nullptr, 0.f, "m")
          .label("Standoff", "Stnd")
      .floatField("orbit_speed", 10.0f, 0.f, 40.f, state::PrimaryInput, nullptr, 0.f, "m/s")
          .label("Orbit Speed", "OSpd")
      .floatField("standoff_deep", 0.7f, 0.01f, 3.f, state::PrimaryInput, nullptr, 0.f, "m")
          .label("Standoff Deep", "StDp")
      .floatField("speed_deep", 3.0f, 0.f, 40.f, state::PrimaryInput, nullptr, 0.f, "m/s")
          .label("Speed Deep", "SpDp")
      .floatField("local", 0.85f, 0.f, 1.f, state::PrimaryInput)
          .label("Local", "Locl")
      .floatField("hold", 1.0f, 0.f, 4.f, state::PrimaryInput)
          .label("Hold", "Hold")
      .floatField("stick", 0.8f, 0.f, 4.f, state::PrimaryInput)
          .label("Stick", "Stck")
      .floatField("ricochet", 0.6f, 0.f, 1.f, state::PrimaryInput)
          .label("Ricochet", "Rico")
      .floatField("vortex", 1.0f, 0.f, 4.f, state::PrimaryInput)
          .label("Vortex", "Vrtx")
      .floatField("lanes", 0.7f, 0.f, 1.f, state::PrimaryInput)
          .label("Lanes", "Lane")
      .floatField("dust", 0.5f, 0.f, 2.f, state::PrimaryInput)
          .label("Dust", "Dust")
      .group("tail", "Throw")
        .groupHelp(
          "What a throw does. *Fling* is how hard the flakes are blown out "
          "onto a sphere *Shell Radius* around the structure; while the tail "
          "rings out they shake there -- *Shake* far, starting at *Shake Hz* "
          "and slowing as the tail dies, all in step unless *Shake Spread* "
          "detunes them. Wire a motion blur after this effect to streak it.")
      .floatField("fling", 1.0f, 0.f, 4.f, state::PrimaryInput)
          .label("Fling", "Flng")
      .floatField("shell_radius", 4.0f, 2.f, 10.f, state::PrimaryInput, nullptr, 0.f, "m")
          .label("Shell Radius", "SRad")
      .floatField("vibe", 0.06f, 0.f, 0.5f, state::PrimaryInput, nullptr, 0.f, "m")
          .label("Shake", "Shk")
      .floatField("vibe_hz", 16.0f, 0.5f, 60.f, state::PrimaryInput, nullptr, 0.f, "Hz")
          .label("Shake Hz", "ShHz")
      .floatField("vibe_spread", 0.25f, 0.f, 1.f, state::PrimaryInput)
          .label("Shake Spread", "ShSp")
      .group("motion", "Motion Vectors")
        .groupHelp(
          "What a motion blur wired after this effect reads: how far each "
          "pixel moved this frame. *Flake Motion* scales the flakes' share "
          "(0: they stay sharp), *Scene Motion* everything else's -- the "
          "blocks spinning, breathing and gliding. Past 1 they smear longer "
          "than they really moved.")
      .floatField("motion_flakes", 1.0f, 0.f, 4.f, state::PrimaryInput)
          .label("Flake Motion", "FMot")
      .floatField("motion_scene", 1.0f, 0.f, 4.f, state::PrimaryInput)
          .label("Scene Motion", "SMot")
      .group("debug", "Debug")
      .selectField("debug_view", 0, state::SecondaryInput,
                   {{"Off", 0}, {"Linear", 1}, {"Surfaces", 2}, {"In-scatter", 3},
                    {"Transmittance", 4}, {"Flakes", 5}, {"Anchors", 6}, {"X-Ray", 7},
                    {"Glint", 8}, {"Depth", 9}})
          .label("Debug View", "Dbg")
      .group("aux", "Aux Outputs")
        .groupHelp(
          "Two extra pictures to wire on to other effects, each showing what "
          "you pick: *Depth* (a ramp across the structure, white nearest, "
          "black in open water), the *Surfaces* alone, the "
          "lit water's *In-scatter* alone, its *Transmittance*, the *Flakes* "
          "alone, the blocks' *Glint* charge (white = charged), or the "
          "*Linear* picture before the tonemap. Each is only rendered while "
          "something is wired to it. (Effects reading render_outputs also get "
          "the blocks' true depth, in metres, beside the motion vectors.)")
      // (What an aux output can carry: common.hlsl's LTN_AUX_*.)
      .selectField("aux1", 1, state::SecondaryInput,
                   {{"Depth", 1}, {"Surfaces", 2}, {"In-scatter", 3}, {"Transmittance", 4},
                    {"Flakes", 5}, {"Glint", 6}, {"Linear", 7}})
          .label("Aux 1", "Aux1")
      .selectField("aux2", 3, state::SecondaryInput,
                   {{"Depth", 1}, {"Surfaces", 2}, {"In-scatter", 3}, {"Transmittance", 4},
                    {"Flakes", 5}, {"Glint", 6}, {"Linear", 7}})
          .label("Aux 2", "Aux2")
      .textureField("tex_in", state::PrimaryInput)
      .textureField("tex_out", state::PrimaryOutput)
      .textureField("aux1_out", state::SecondaryOutput)
      .textureField("aux2_out", state::SecondaryOutput)
      // Motion vectors (blocks + flakes) on the render_outputs motion leaf,
      // made only when something downstream reads it.
      .renderOutputs(state::PrimaryOutput);
  schema.capability(state::Capability::Generator);
  state::init("source.mesh.lantern", {1, 0, 0}, schema);

  if (gpu::Device::backend() == gpu::Backend::None) return;

  state::registerShaderSPV("lantern_shadow", LANTERN_SHADOW_SPV, LANTERN_SHADOW_SPV_SIZE,
                           "rgba16float", "write");
  state::registerShaderSPV("lantern_escape", LANTERN_ESCAPE_SPV, LANTERN_ESCAPE_SPV_SIZE,
                           "rgba16float", "write");
  state::registerShaderSPV("lantern_escape_reduce", LANTERN_ESCAPE_REDUCE_SPV,
                           LANTERN_ESCAPE_REDUCE_SPV_SIZE, "rgba16float", "write");
  state::registerShaderSPV("lantern_vis", LANTERN_VIS_SPV, LANTERN_VIS_SPV_SIZE,
                           "rgba16float", "write");
  state::registerShaderSPV("lantern_scene", LANTERN_SCENE_SPV, LANTERN_SCENE_SPV_SIZE,
                           "rgba16float", "write");
  state::registerShaderSPV("lantern_fog", LANTERN_FOG_SPV, LANTERN_FOG_SPV_SIZE,
                           "rgba16float", "write");
  state::registerShaderSPV("lantern_blur", LANTERN_BLUR_SPV, LANTERN_BLUR_SPV_SIZE,
                           "rgba16float", "write");
  state::registerShaderSPV("lantern_composite", LANTERN_COMPOSITE_SPV,
                           LANTERN_COMPOSITE_SPV_SIZE);
  state::registerShaderSPV("lantern_prefill", LANTERN_PREFILL_SPV, LANTERN_PREFILL_SPV_SIZE);
  state::registerShaderSPV("lantern_aux", LANTERN_AUX_SPV, LANTERN_AUX_SPV_SIZE, "rgba16float",
                           "write");
  state::registerShaderSPV("lantern_p_update", LANTERN_P_UPDATE_SPV, LANTERN_P_UPDATE_SPV_SIZE);
  state::registerShaderSPV("lantern_p_resolve", LANTERN_P_RESOLVE_SPV, LANTERN_P_RESOLVE_SPV_SIZE,
                           "rgba16float", "write");
  state::registerShaderSPV("lantern_p_shade", LANTERN_P_SHADE_SPV, LANTERN_P_SHADE_SPV_SIZE);
  state::registerShaderSPV("lantern_motion", LANTERN_MOTION_SPV, LANTERN_MOTION_SPV_SIZE,
                           "rgba16float", "write");
  state::registerShaderSPV("lantern_spark_vs", LANTERN_SPARK_VS_SPV, LANTERN_SPARK_VS_SPV_SIZE);
  state::registerShaderSPV("lantern_spark_fs", LANTERN_SPARK_FS_SPV, LANTERN_SPARK_FS_SPV_SIZE);
  state::registerShaderSPV("lantern_spark_mv_fs", LANTERN_SPARK_MV_FS_SPV,
                           LANTERN_SPARK_MV_FS_SPV_SIZE);
  auto cs_shadow = gpu::Device::createShaderModuleByName("lantern_shadow");
  auto cs_escape = gpu::Device::createShaderModuleByName("lantern_escape");
  auto cs_reduce = gpu::Device::createShaderModuleByName("lantern_escape_reduce");
  auto cs_vis = gpu::Device::createShaderModuleByName("lantern_vis");
  auto cs_scene = gpu::Device::createShaderModuleByName("lantern_scene");
  auto cs_fog = gpu::Device::createShaderModuleByName("lantern_fog");
  auto cs_blur = gpu::Device::createShaderModuleByName("lantern_blur");
  auto cs_comp = gpu::Device::createShaderModuleByName("lantern_composite");
  auto cs_prefill = gpu::Device::createShaderModuleByName("lantern_prefill");
  auto cs_aux = gpu::Device::createShaderModuleByName("lantern_aux");
  auto cs_p_update = gpu::Device::createShaderModuleByName("lantern_p_update");
  auto cs_p_resolve = gpu::Device::createShaderModuleByName("lantern_p_resolve");
  auto cs_p_shade = gpu::Device::createShaderModuleByName("lantern_p_shade");
  auto cs_motion = gpu::Device::createShaderModuleByName("lantern_motion");
  auto vs_spark = gpu::Device::createShaderModuleByName("lantern_spark_vs");
  auto fs_spark = gpu::Device::createShaderModuleByName("lantern_spark_fs");
  auto fs_spark_mv = gpu::Device::createShaderModuleByName("lantern_spark_mv_fs");
  if (!cs_shadow || !cs_escape || !cs_reduce || !cs_vis || !cs_scene || !cs_fog || !cs_blur || !cs_comp || !cs_prefill || !cs_aux) return;
  if (!cs_p_update || !cs_p_resolve || !cs_p_shade || !cs_motion || !vs_spark || !fs_spark ||
      !fs_spark_mv) return;

  s_pso_shadow = gpu::Device::createComputePSO(cs_shadow, "main", gpu::Bindings()
      .uniform(0)
      .storage(1)
      .storageTex2d(2, gpu::TextureFormat::RGBA16F));
  s_pso_escape = gpu::Device::createComputePSO(cs_escape, "main", gpu::Bindings()
      .uniform(0)
      .storage(1)
      .storageTex2d(2, gpu::TextureFormat::RGBA16F));
  s_pso_reduce = gpu::Device::createComputePSO(cs_reduce, "main", gpu::Bindings()
      .uniform(0)
      .storage(1)
      .tex2d(2)
      .storageTex2d(3, gpu::TextureFormat::RGBA16F));
  s_pso_vis = gpu::Device::createComputePSO(cs_vis, "main", gpu::Bindings()
      .uniform(0)
      .storage(1)
      .tex2d(2)
      .tex2d(3)
      .sampler(4)
      .storageTex2d(5, gpu::TextureFormat::RGBA16F)
      .storageTex2d(6, gpu::TextureFormat::RGBA16F));
  s_pso_scene = gpu::Device::createComputePSO(cs_scene, "main", gpu::Bindings()
      .uniform(0)
      .storage(1)
      .tex2d(2)
      .storageTex2d(3, gpu::TextureFormat::RGBA16F)
      .storageTex2d(4, gpu::TextureFormat::RGBA16F));
  s_pso_fog = gpu::Device::createComputePSO(cs_fog, "main", gpu::Bindings()
      .uniform(0)
      .storage(1)
      .tex2d(2)
      .tex2d(3)
      .storageTex2d(4, gpu::TextureFormat::RGBA16F)
      .storageTex2d(5, gpu::TextureFormat::RGBA16F)
      .tex2d(6)
      .tex2d(7)
      .sampler(8)
      .tex3d(9));
  s_pso_blur = gpu::Device::createComputePSO(cs_blur, "main", gpu::Bindings()
      .uniform(0)
      .storage(1)
      .tex2d(2)
      .storageTex2d(3, gpu::TextureFormat::RGBA16F)
      .uniform(4));
  s_pso_comp = gpu::Device::createComputePSO(cs_comp, "main", gpu::Bindings()
      .uniform(0)
      .storage(1)
      .tex2d(2)
      .tex2d(3)
      .tex2d(4)
      .tex2d(5)
      .storageTex2d(6)
      .tex2d(7));
  s_pso_aux = gpu::Device::createComputePSO(cs_aux, "main", gpu::Bindings()
      .uniform(0)
      .storage(1)
      .tex2d(2)
      .tex2d(3)
      .tex2d(4)
      .tex2d(5)
      .tex2d(6)
      .storageTex2d(7, gpu::TextureFormat::RGBA16F)
      .storageTex2d(8, gpu::TextureFormat::RGBA16F)
      .storageTex2d(9, gpu::TextureFormat::RGBA16F));
  s_pso_prefill = gpu::Device::createComputePSO(cs_prefill, "main", gpu::Bindings()
      .tex2d(0)
      .storageTex2d(1));
  s_pso_p_update = gpu::Device::createComputePSO(cs_p_update, "main", gpu::Bindings()
      .uniform(0)
      .storage(1)
      .storageRW(2)
      .storageRW(3)
      .uniform(4));
  s_pso_p_resolve = gpu::Device::createComputePSO(cs_p_resolve, "main", gpu::Bindings()
      .storageRW(3)
      .uniform(4)
      .storageRW(5)
      .storageTex3d(6, gpu::TextureFormat::RGBA16F));
  s_pso_p_shade = gpu::Device::createComputePSO(cs_p_shade, "main", gpu::Bindings()
      .uniform(0)
      .storage(1)
      .storage(2)
      .uniform(4)
      .tex2d(5)
      .tex2d(6)
      .storageRW(7));
  s_pso_motion = gpu::Device::createComputePSO(cs_motion, "main", gpu::Bindings()
      .uniform(0)
      .storage(1)
      .storage(2)
      .tex2d(3)
      .uniform(4)
      .storageTex2d(5, gpu::TextureFormat::RGBA16F));
  s_pso_spark = gpu::Device::createInstancedRenderPSO(vs_spark, "main", fs_spark, "main",
      gpu::TextureFormat::RGBA16F, gpu::Bindings().uniform(4).storage(7),
      gpu::Device::BlendMode::Additive);
  s_pso_spark_mv = gpu::Device::createInstancedRenderPSOMRT(vs_spark, "main", fs_spark_mv, "main",
      {gpu::TextureFormat::RGBA16F, gpu::TextureFormat::RGBA16F},
      gpu::Bindings().uniform(4).storage(7),
      {gpu::Device::BlendMode::Additive, gpu::Device::BlendMode::AlphaOver});
  state::log("lantern: module initialized");
}

void* create() {
  auto* s = new State();
  s->ub_frame = gpu::Device::createBuffer(sizeof(FrameUniforms), gpu::BufferUsage::Uniform);
  s->box_buf = gpu::Device::createBuffer((long long)sizeof(Row) * kMaxBoxes * (kBoxRows + 1),
                                         gpu::BufferUsage::Storage);
  s->atlas = gpu::Device::createTexture(kTileRes * kAtlasCols, kTileRes * kAtlasRows,
                                        gpu::TextureFormat::RGBA16F);
  s->ub_blur_h = gpu::Device::createBuffer(sizeof(BlurUniforms), gpu::BufferUsage::Uniform);
  s->ub_blur_v = gpu::Device::createBuffer(sizeof(BlurUniforms), gpu::BufferUsage::Uniform);
  s->zero_tex = gpu::Device::createTexture(1, 1, gpu::TextureFormat::RGBA16F);
  if (s->zero_tex.valid()) gpu::Device::clear(s->zero_tex, 0.0f, 0.0f, 0.0f, 0.0f);
  s->esc = gpu::Device::createTexture(kEscRes, kEscRes, gpu::TextureFormat::RGBA16F);
  s->esc_low = gpu::Device::createTexture(kEscLow, kEscLow, gpu::TextureFormat::RGBA16F);
  s->vis_f = gpu::Device::createTexture(kEscRes, kEscRes, gpu::TextureFormat::RGBA16F);
  s->vis_d = gpu::Device::createTexture(kEscRes, kEscRes, gpu::TextureFormat::RGBA16F);
  s->samp = gpu::Device::createSampler(gpu::FilterMode::Linear, gpu::AddressMode::ClampToEdge);
  s->ub_part = gpu::Device::createBuffer(sizeof(PartUniforms), gpu::BufferUsage::Uniform);
  s->p_state = gpu::Device::createBuffer((long long)16 * kPRows * kMaxFlakes,
                                         gpu::BufferUsage::Storage);
  s->p_recs = gpu::Device::createBuffer((long long)16 * kRRows * kMaxFlakes,
                                        gpu::BufferUsage::Storage);
  s->p_grid = gpu::Device::createBuffer((long long)4 * kGrid * kGrid * kGrid,
                                        gpu::BufferUsage::Storage);
  s->p_accum = gpu::Device::createBuffer((long long)4 * kGrid * kGrid * kGrid,
                                         gpu::BufferUsage::Storage);
  s->prev_box_buf = gpu::Device::createBuffer((long long)16 * 4 * kMaxBoxes,
                                              gpu::BufferUsage::Storage);
  s->presence = gpu::Device::createTexture3D(kGrid, kGrid, kGrid, gpu::TextureFormat::RGBA16F);
  s->presence_zero = gpu::Device::createTexture3D(1, 1, 1, gpu::TextureFormat::RGBA16F);
  for (auto& t : s->stand_in) t = gpu::Device::createTexture(1, 1, gpu::TextureFormat::RGBA16F);
  {
    // Storage buffers start undefined: the grid's counts and average start at 0.
    static uint32_t zeros[kGrid * kGrid * kGrid] = {};
    if (s->p_grid.valid()) s->p_grid.write(zeros, kGrid * kGrid * kGrid);
    if (s->p_accum.valid()) s->p_accum.write(zeros, kGrid * kGrid * kGrid);
  }
  return s;
}

void destroy(void* self) {
  auto* s = static_cast<State*>(self);
  if (!s) return;
  s->ub_frame.release();
  s->box_buf.release();
  s->atlas.release();
  s->scene_tex.release();
  s->fog_i.release();
  s->fog_t.release();
  s->fog_tmp.release();
  s->fog_blur.release();
  s->ub_blur_h.release();
  s->ub_blur_v.release();
  s->zero_tex.release();
  s->esc.release();
  s->esc_low.release();
  s->vis_f.release();
  s->vis_d.release();
  s->ub_part.release();
  s->p_state.release();
  s->p_recs.release();
  s->p_grid.release();
  s->p_accum.release();
  s->prev_box_buf.release();
  s->presence.release();
  s->presence_zero.release();
  s->spark.release();
  s->motion_tex.release();
  for (auto& t : s->aux_tex) t.release();
  s->depth_tex.release();
  s->glint_tex.release();
  for (auto& t : s->stand_in) t.release();
  delete s;
}

void init(void* self) {
  auto* s = static_cast<State*>(self);
  if (!s) return;
  s->initialized = s->ub_frame.valid() && s->box_buf.valid() && s->atlas.valid() &&
                   s->esc.valid() && s->esc_low.valid() && s->vis_f.valid() &&
                   s->vis_d.valid() && s_pso_escape.valid() && s_pso_reduce.valid() &&
                   s_pso_vis.valid() && s_pso_shadow.valid() && s_pso_scene.valid() && s_pso_fog.valid() &&
                   s_pso_blur.valid() && s_pso_comp.valid() && s_pso_prefill.valid() &&
                   s_pso_aux.valid() &&
                   s->ub_blur_h.valid() && s->ub_blur_v.valid() &&
                   s->ub_part.valid() && s->p_state.valid() && s->p_recs.valid() &&
                   s->p_grid.valid() && s->p_accum.valid() && s->prev_box_buf.valid() &&
                   s->presence.valid() && s->presence_zero.valid() &&
                   s->stand_in[0].valid() && s->stand_in[1].valid() && s->stand_in[2].valid() &&
                   s_pso_p_update.valid() && s_pso_p_resolve.valid() && s_pso_p_shade.valid() &&
                   s_pso_motion.valid() && s_pso_spark.valid() && s_pso_spark_mv.valid();
}

void tick(void* self, double dt) {
  auto* s = static_cast<State*>(self);
  if (!s) return;
  if (!(dt > 0.0)) dt = 0.0;
  if (dt > 0.05) dt = 0.05;
  s->time += dt;
  s->breathe_phase += dt * rateHz(s->breathe_rate, 0.08);
  s->spin_phase = std::fmod(s->spin_phase + dt * rateHz(s->spin, 0.02), 1.0);
  s->wander_phase += dt * rateHz(s->wander_rate, 0.12);
  s->orbit_phase = std::fmod(s->orbit_phase + dt * rateHz(s->orbit, 0.02), 1.0);
  s->drift_phase += dt * 0.07;
  // Keep the noise-domain accumulators small (float precision on the GPU).
  if (s->breathe_phase > 4096.0) s->breathe_phase -= 4096.0;
  if (s->wander_phase > 4096.0 * kTau) s->wander_phase -= 4096.0 * kTau;
  if (s->drift_phase > 4096.0) s->drift_phase -= 4096.0;
  if (s->time > 4096.0) s->time -= 4096.0;
  // The sweep: level, bounce, throw (sweep_knob.h); the glints it launches.
  sweep_knob::Params kp;
  kp.sweep = s->sweep;
  kp.sense = s->sense;
  kp.deadzone = s->deadzone;
  kp.bounce = s->bounce;
  kp.ring_time = s->tail;
  s->knob.tick(kp, (float)dt);
  if (s->knob.fresh_throw) s->pending_fling = std::fmax(s->pending_fling, s->knob.release);
  three_planes_glints::Params gp;
  gp.sweep = s->sweep;
  gp.band = s->deadzone;
  gp.fling = s->knob.release;
  gp.sling = 0.0f;   // a throw holds them in the frame, it doesn't carry them off
  gp.lead = s->glint_width * 1.5f;
  gp.trail = s->glint_width * 4.0f;
  // Here a glint is a pull, not a glare: it never runs down where it is, and
  // once the knob rests it still crosses, in Glint Time.
  gp.drift = 1.0f / std::fmax(s->glint_time, 0.05f);
  gp.hold = 1.0f;
  // ...unless the sweep chokes while it has barely started: then it is a
  // glint arriving after the gesture is over, and it fades out.
  gp.choked = s->knob.level <= sweep_knob::kMuteGain;
  gp.quench_before = s->glint_choke;
  s->glint_core.tick(gp, (float)dt);
  // The blocks' glide (to a new layout, after their jut), and their jitter:
  // knocks at random, Jitter Rate a second, each ringing out on the glide's
  // spring.
  {
    const glide::Params sp = glideParams(s);
    const glide::Glitch gl = glide::Glitch::from(s->glitch, sp);
    const Drive d = driveOf(s);
    const float knock = 1.0f - std::exp(-std::fmax(s->jitter_rate, 0.0f) * (float)dt);
    const float w = kTau * sp.stiffness;
    for (int i = 0; i < s->slot_count; i++) {
      Slot& sl = s->slots[i];
      sl.flip -= (float)dt;
      auto& g = sl.g;
      if (sl.block >= 0 && sp.time > 1e-4f && knock > 0.0f) {
        const float A = sent(s, d, kJitter, s->jitter, sl.charge);
        if (A > 1e-4f && g.rnd() < knock) {
          float dv[kGlideCh] = {};
          for (int k = 0; k < 2; k++) {   // a random way to move, and to turn
            const float z = 2.0f * g.rnd() - 1.0f, ph = kTau * g.rnd();
            const float r = std::sqrt(std::fmax(0.0f, 1.0f - z * z));
            const float a = A * w * (k == 0 ? 1.0f : kJitterTurn);
            const int c0 = k == 0 ? 0 : 6;
            dv[c0] = a * r * std::cos(ph);
            dv[c0 + 1] = a * r * std::sin(ph);
            dv[c0 + 2] = a * z;
          }
          g.kick(dv);
        }
      }
      if (g.active(1e-5f)) g.step((float)dt, sp, gl);
      else if (g.age < 1e8f) g.snap();
    }
  }
  // The tail's shake: one phase for every flake, its frequency falling as
  // the tail rings out.
  const double hz = s->vibe_hz * (0.12 + 0.88 * s->knob.release);
  s->vibe_phase = std::fmod(s->vibe_phase + kTau * hz * dt, kTau * 4096.0);
  s->sim_dt += dt;
  if (s->sim_dt > 0.1) s->sim_dt = 0.1;
  if (dt > 0.0) s->frame_dt = (float)dt;
}

void on_state_patched(void* self, int n, const char* pb, const int* off,
                      const int* len, const int* ops) {
  auto* s = static_cast<State*>(self);
  if (!s) return;
  for (int i = 0; i < n; i++) {
    if (ops[i] != state::PatchReplace) continue;
    const char* p = pb + off[i];
    int l = len[i];
    if      (state::pathIs(p, l, "layout"))       s->layout = state::patchInt(i);
    else if (state::pathIs(p, l, "glide"))        s->glide_time = state::patchFloat(i);
    else if (state::pathIs(p, l, "stiffness"))    s->stiffness = state::patchFloat(i);
    else if (state::pathIs(p, l, "glitch"))       s->glitch = state::patchFloat(i);
    else if (state::pathIs(p, l, "glide_smooth")) s->glide_smooth = state::patchFloat(i);
    else if (state::pathIs(p, l, "jut"))          s->jut = state::patchFloat(i);
    else if (state::pathIs(p, l, "breathe"))      s->breathe = state::patchFloat(i);
    else if (state::pathIs(p, l, "breathe_rate")) s->breathe_rate = state::patchFloat(i);
    else if (state::pathIs(p, l, "seam"))         s->seam = state::patchFloat(i);
    else if (state::pathIs(p, l, "spin"))         s->spin = state::patchFloat(i);
    else if (state::pathIs(p, l, "spin_axis"))    s->spin_axis = state::patchInt(i);
    else if (state::pathIs(p, l, "drift"))        s->drift = state::patchFloat(i);
    else if (state::pathIs(p, l, "core_power"))   s->core_power = state::patchFloat(i);
    else if (state::pathIs(p, l, "core_color")) {
      auto v = state::patchVec3(i);
      s->core_r = v.x; s->core_g = v.y; s->core_b = v.z;
    }
    else if (state::pathIs(p, l, "core_radius"))  s->core_radius = state::patchFloat(i);
    else if (state::pathIs(p, l, "core_clear"))   s->core_clear = state::patchFloat(i);
    else if (state::pathIs(p, l, "wander"))       s->wander = state::patchFloat(i);
    else if (state::pathIs(p, l, "wander_rate"))  s->wander_rate = state::patchFloat(i);
    else if (state::pathIs(p, l, "flicker"))      s->flicker = state::patchFloat(i);
    else if (state::pathIs(p, l, "halo"))         s->halo = state::patchFloat(i);
    else if (state::pathIs(p, l, "wisps"))        s->wisps = state::patchFloat(i);
    else if (state::pathIs(p, l, "halo_phase"))   s->halo_phase = state::patchFloat(i);
    else if (state::pathIs(p, l, "water"))        s->water = state::patchFloat(i);
    else if (state::pathIs(p, l, "air"))          s->air = state::patchFloat(i);
    else if (state::pathIs(p, l, "air_depth"))    s->air_depth = state::patchFloat(i);
    else if (state::pathIs(p, l, "air_soft"))     s->air_soft = state::patchFloat(i);
    else if (state::pathIs(p, l, "ambient"))      s->ambient = state::patchFloat(i);
    else if (state::pathIs(p, l, "rim"))          s->rim = state::patchFloat(i);
    else if (state::pathIs(p, l, "kicker"))       s->kicker = state::patchFloat(i);
    else if (state::pathIs(p, l, "yaw"))          s->yaw = state::patchFloat(i);
    else if (state::pathIs(p, l, "pitch"))        s->pitch = state::patchFloat(i);
    else if (state::pathIs(p, l, "distance"))     s->distance = state::patchFloat(i);
    else if (state::pathIs(p, l, "lens"))         s->lens = state::patchFloat(i);
    else if (state::pathIs(p, l, "orbit"))        s->orbit = state::patchFloat(i);
    else if (state::pathIs(p, l, "exposure"))     s->exposure = state::patchFloat(i);
    else if (state::pathIs(p, l, "contrast"))     s->contrast = state::patchFloat(i);
    else if (state::pathIs(p, l, "bevel"))        s->bevel = state::patchFloat(i);
    else if (state::pathIs(p, l, "reflect"))      s->reflect = state::patchFloat(i);
    else if (state::pathIs(p, l, "ember"))        s->ember = state::patchFloat(i);
    else if (state::pathIs(p, l, "shell_glow"))   s->shell_glow = state::patchFloat(i);
    else if (state::pathIs(p, l, "chips"))        s->chips = state::patchFloat(i);
    else if (state::pathIs(p, l, "albedo"))       s->albedo = state::patchFloat(i);
    else if (state::pathIs(p, l, "opacity"))      s->opacity = state::patchFloat(i);
    else if (state::pathIs(p, l, "debug_view"))   s->debug_view = state::patchInt(i);
    else if (state::pathIs(p, l, "aux1"))         s->aux_what[0] = state::patchInt(i);
    else if (state::pathIs(p, l, "aux2"))         s->aux_what[1] = state::patchInt(i);
    else if (state::pathIs(p, l, "sweep"))        s->sweep = state::patchFloat(i);
    else if (state::pathIs(p, l, "sustain"))      s->sustain = state::patchFloat(i);
    else if (state::pathIs(p, l, "deadzone"))     s->deadzone = state::patchFloat(i);
    else if (state::pathIs(p, l, "ricochet"))     s->ricochet = state::patchFloat(i);
    else if (state::pathIs(p, l, "vortex"))       s->vortex = state::patchFloat(i);
    else if (state::pathIs(p, l, "overshoot"))    s->overshoot = state::patchFloat(i);
    else if (state::pathIs(p, l, "sense"))        s->sense = state::patchFloat(i);
    else if (state::pathIs(p, l, "tail"))         s->tail = state::patchFloat(i);
    else if (state::pathIs(p, l, "glint_angle"))  s->glint_angle = state::patchFloat(i);
    else if (state::pathIs(p, l, "magnetism"))    s->magnetism = state::patchFloat(i);
    else if (state::pathIs(p, l, "twang"))        s->twang = state::patchFloat(i);
    else if (state::pathIs(p, l, "jitter"))       s->jitter = state::patchFloat(i);
    else if (state::pathIs(p, l, "jitter_rate"))  s->jitter_rate = state::patchFloat(i);
    else if (state::pathIs(p, l, "glint_width"))  s->glint_width = state::patchFloat(i);
    else if (state::pathIs(p, l, "glint_time"))   s->glint_time = state::patchFloat(i);
    else if (state::pathIs(p, l, "glint_fade"))   s->glint_fade = state::patchFloat(i);
    else if (state::pathIs(p, l, "glint_choke"))  s->glint_choke = state::patchFloat(i);
    else if (state::pathIs(p, l, "glint_pull"))   s->glint_pull = state::patchFloat(i);
    else if (state::pathIs(p, l, "flakes"))       s->flakes = state::patchInt(i);
    else if (state::pathIs(p, l, "flake_size"))   s->flake_size = state::patchFloat(i);
    else if (state::pathIs(p, l, "mica"))         s->mica = state::patchFloat(i);
    else if (state::pathIs(p, l, "sparkle"))      s->sparkle = state::patchFloat(i);
    else if (state::pathIs(p, l, "tumble"))       s->tumble = state::patchFloat(i);
    else if (state::pathIs(p, l, "streak"))       s->streak = state::patchFloat(i);
    else if (state::pathIs(p, l, "motion_flakes")) s->motion_flakes = state::patchFloat(i);
    else if (state::pathIs(p, l, "motion_scene")) s->motion_scene = state::patchFloat(i);
    else if (state::pathIs(p, l, "heat"))         s->heat = state::patchFloat(i);
    else if (state::pathIs(p, l, "ring_radius"))  s->ring_radius = state::patchFloat(i);
    else if (state::pathIs(p, l, "ring_loose"))   s->ring_loose = state::patchFloat(i);
    else if (state::pathIs(p, l, "ring_speed"))   s->ring_speed = state::patchFloat(i);
    else if (state::pathIs(p, l, "jet"))          s->jet = state::patchFloat(i);
    else if (state::pathIs(p, l, "jet_angle"))    s->jet_angle = state::patchFloat(i);
    else if (state::pathIs(p, l, "capture"))      s->capture = state::patchFloat(i);
    else if (state::pathIs(p, l, "bite"))         s->bite = state::patchFloat(i);
    else if (state::pathIs(p, l, "reach"))        s->reach = state::patchFloat(i);
    else if (state::pathIs(p, l, "standoff"))     s->standoff = state::patchFloat(i);
    else if (state::pathIs(p, l, "orbit_speed"))  s->orbit_speed = state::patchFloat(i);
    else if (state::pathIs(p, l, "standoff_deep")) s->standoff_deep = state::patchFloat(i);
    else if (state::pathIs(p, l, "speed_deep"))   s->speed_deep = state::patchFloat(i);
    else if (state::pathIs(p, l, "local"))        s->local = state::patchFloat(i);
    else if (state::pathIs(p, l, "hold"))         s->hold = state::patchFloat(i);
    else if (state::pathIs(p, l, "stick"))        s->stick = state::patchFloat(i);
    else if (state::pathIs(p, l, "bounce"))       s->bounce = state::patchFloat(i);
    else if (state::pathIs(p, l, "lanes"))        s->lanes = state::patchFloat(i);
    else if (state::pathIs(p, l, "dust"))         s->dust = state::patchFloat(i);
    else if (state::pathIs(p, l, "fling"))        s->fling = state::patchFloat(i);
    else if (state::pathIs(p, l, "shell_radius")) s->shell_radius = state::patchFloat(i);
    else if (state::pathIs(p, l, "vibe"))         s->vibe = state::patchFloat(i);
    else if (state::pathIs(p, l, "vibe_hz"))      s->vibe_hz = state::patchFloat(i);
    else if (state::pathIs(p, l, "vibe_spread"))  s->vibe_spread = state::patchFloat(i);
    else
      for (int k = 0; k < kSources; k++)
        for (int t = 0; t < kTargets; t++)
          if (state::pathIs(p, l, kSendKey[k][t])) s->sends[k][t] = state::patchFloat(i);
  }
}

// A block's goal this frame, in the structure's frame: where its layout puts
// it, jutted, breathing, bobbing.
Pose goalPose(const State* s, const Block& b, float jutk, float seam) {
  const float bp = (float)s->breathe_phase;
  const float dp = (float)s->drift_phase;
  const float jmax = kCell * kJutMax;
  Pose p;
  if (b.fragment) {
    float ph = kTau * (dp + b.seed);
    p.c = b.rest + b.axis * (0.7f * s->drift * std::sin(ph));
    float tt = (float)s->drift_phase * 14.0f * s->drift;
    p.rot = euler(b.tip.x + b.spin.x * tt, b.tip.y + b.spin.y * tt, b.tip.z + b.spin.z * tt);
    p.half = b.size * 0.5f;
    p.glow = 1.0f;
  } else {
    // Slow waves of opening and closing: value noise over the rest
    // position, moving through the noise domain with the breathing phase.
    // It swings each block both ways about its rest jut — some seal shut
    // while others push out — so the light comes and goes in patches
    // rather than the whole shell gaping open.
    float open = 0.5f + 0.5f * vnoise(b.rest * 0.45f + v3(0.0f, 0.0f, bp), 101u);
    open = open * open * (3.0f - 2.0f * open);
    float jut = std::fmax(b.jut * jutk + s->breathe * jmax * 0.9f * (open - 0.5f), 0.0f);
    float strain = jutStrain(jut);
    float tip = std::pow(std::fmin(strain, 1.5f), 1.5f) * 0.18f;
    p.c = b.rest + b.axis * jut;
    p.rot = euler(b.tip.x * tip, b.tip.y * tip, b.tip.z * tip);
    V3 half = v3(b.span[0] * kCell - seam, b.span[1] * kCell - seam,
                 b.span[2] * kCell - seam) * 0.5f;
    p.half = v3(std::fmax(half.x, 0.02f), std::fmax(half.y, 0.02f), std::fmax(half.z, 0.02f));
    p.glow = std::fmin(strain, 1.0f);
  }
  return p;
}

// The axis flakes circulate a block about — one of the two it does not jut
// along, picked per block (-1: a loose fragment).
float circOf(const Block& b) {
  if (b.fragment) return -1.0f;
  const float ax = std::fabs(b.axis.x), ay = std::fabs(b.axis.y), az = std::fabs(b.axis.z);
  const int jut = (ax >= ay && ax >= az) ? 0 : (ay >= az ? 1 : 2);
  const float pick = b.seed * 7.31f - std::floor(b.seed * 7.31f);
  return (float)((jut + (pick < 0.5f ? 1 : 2)) % 3);
}

// A slot's goal: its block's pose, or leaving, the block it shrinks into's,
// at no size.
Pose goalOf(const State* s, const Slot& sl) {
  if (sl.block >= 0) return goalPose(s, s->blocks[sl.block], sl.jut, sl.seam);
  Pose p = goalPose(s, s->blocks[sl.into], sl.jut, sl.seam);
  p.half = v3(0.0f, 0.0f, 0.0f);
  p.glow = sl.shown.glow;
  return p;
}

// The glide's channels for showing `shown` while heading for `goal`.
void offsetOf(const Pose& shown, const Pose& goal, float* o) {
  const V3 dc = shown.c - goal.c, dh = shown.half - goal.half;
  const V3 w = rotLog(mul(shown.rot, transpose(goal.rot)));
  o[0] = dc.x; o[1] = dc.y; o[2] = dc.z;
  o[3] = dh.x; o[4] = dh.y; o[5] = dh.z;
  o[6] = w.x;  o[7] = w.y;  o[8] = w.z;
}

// Drop the slots that have finished leaving (or were dropped for room).
void compact(State* s) {
  int n = 0;
  for (int i = 0; i < s->slot_count; i++) {
    const Slot& sl = s->slots[i];
    if (sl.block < 0 && (sl.dead || !sl.g.active(1e-3f))) continue;
    if (n != i) s->slots[n] = sl;
    n++;
  }
  s->slot_count = n;
}

// What a slot shows now: its goal and the glide's offset over it.
Pose showOf(const State* s, const Slot& sl) {
  const Pose goal = goalOf(s, sl);
  const float* o = sl.g.out;
  Pose p;
  p.c = goal.c + v3(o[0], o[1], o[2]);
  const V3 h = goal.half + v3(o[3], o[4], o[5]);
  // (Overshooting through no size, a block blinks out a moment.)
  p.half = v3(std::fmax(h.x, 0.002f), std::fmax(h.y, 0.002f), std::fmax(h.z, 0.002f));
  p.rot = mul(rotExp(v3(o[6], o[7], o[8])), goal.rot);
  p.glow = goal.glow;
  return p;
}

// A new layout: build it, and send what is on screen there — nearest first,
// fragments only to fragments; what is left over shrinks into its nearest
// new block, and new blocks left over grow out of their nearest slot.
void relayout(State* s) {
  // Where everything is NOW (the glide has moved on since it was drawn):
  // measured from last frame's pose, a layout changing every frame would
  // pin every block where it was.
  for (int i = 0; i < s->slot_count; i++) s->slots[i].shown = showOf(s, s->slots[i]);
  s->block_count = s->layout == 0 ? buildReference(s->blocks)
                                  : buildProcedural(s->blocks, s->layout);
  s->built_layout = s->layout;
  const glide::Params gp = glideParams(s);
  const int m = s->block_count;
  // A new slot's Jut and Seam: the sends with no glint of its own yet.
  const Drive d = driveOf(s);
  const float jut0 = sent(s, d, kJut, s->jut, 0.0f);
  const float seam0 = sent(s, d, kSeam, s->seam, 0.0f);
  if (s->slot_count == 0 || gp.time <= 1e-3f) {   // a cut
    s->slot_count = m;
    for (int j = 0; j < m; j++) {
      Slot& sl = s->slots[j];
      sl = Slot();
      sl.block = j;
      sl.fragment = s->blocks[j].fragment;
      sl.stone = s->blocks[j].stone;
      sl.seed = s->blocks[j].seed;
      sl.g.seed = ++s->slot_serial;
      sl.jut = jut0;
      sl.seam = seam0;
      sl.shown = goalOf(s, sl);
    }
    s->pose_jump = true;
    return;
  }
  const glide::Glitch gl = glide::Glitch::from(s->glitch, gp);
  static Pose goals[kMaxBoxes];
  for (int j = 0; j < m; j++) goals[j] = goalPose(s, s->blocks[j], jut0, seam0);
  const int n = s->slot_count;
  int to[kMaxBoxes], from[kMaxBoxes], best[kMaxBoxes];
  float bestc[kMaxBoxes];
  glide::matchNearest(
      n, m,
      [&](int i, int j) {
        if (s->slots[i].fragment != s->blocks[j].fragment) return (float)INFINITY;
        const V3 d = s->slots[i].shown.c - goals[j].c;
        return dot(d, d);
      },
      to, from, best, bestc);
  auto nearestBlock = [&](V3 p) {
    int k = 0;
    float bd = 1e30f;
    for (int j = 0; j < m; j++) {
      const V3 d = p - goals[j].c;
      if (dot(d, d) < bd) { bd = dot(d, d); k = j; }
    }
    return k;
  };
  float o[kGlideCh];
  for (int i = 0; i < n; i++) {
    Slot& sl = s->slots[i];
    sl.dead = false;
    if (to[i] >= 0) {
      sl.block = to[i];
      sl.into = -1;
      offsetOf(sl.shown, goalOf(s, sl), o);
      sl.g.retargetTo(o, gl);
      // Its stuff changes (shell <-> stone) at a moment of its own (one
      // already coming stays put: changes every frame still get there).
      const float flip = gp.time * (0.05f + 0.45f * sl.g.rnd());
      if (!(sl.flip > 0.0f)) sl.flip = flip;
    } else {
      sl.block = -1;
      sl.into = nearestBlock(sl.shown.c);
      offsetOf(sl.shown, goalOf(s, sl), o);
      sl.g.retargetTo(o, gl);
    }
  }
  // New blocks left over, each grown out of the slot nearest it.
  static Pose start[kMaxBoxes];
  int spawn = 0;
  for (int j = 0; j < m; j++) {
    if (from[j] >= 0) continue;
    int k = 0;
    float bd = 1e30f;
    for (int i = 0; i < n; i++) {
      const V3 d = s->slots[i].shown.c - goals[j].c;
      if (dot(d, d) < bd) { bd = dot(d, d); k = i; }
    }
    start[j] = s->slots[k].shown;
    start[j].half = v3(0.0f, 0.0f, 0.0f);
    spawn++;
  }
  // No room for them all: the slots already on their way out go at once.
  if (n + spawn > kMaxBoxes) {
    int total = n + spawn;
    for (int i = 0; i < n && total > kMaxBoxes; i++)
      if (s->slots[i].block < 0) { s->slots[i].dead = true; total--; }
    compact(s);
  }
  for (int j = 0; j < m && s->slot_count < kMaxBoxes; j++) {
    if (from[j] >= 0) continue;
    Slot& sl = s->slots[s->slot_count++];
    sl = Slot();
    sl.block = j;
    sl.fragment = s->blocks[j].fragment;
    sl.stone = s->blocks[j].stone;
    sl.seed = s->blocks[j].seed;
    sl.shown = start[j];
    sl.g.seed = ++s->slot_serial;
    sl.jut = jut0;
    sl.seam = seam0;
    offsetOf(start[j], goals[j], o);
    sl.g.retargetTo(o, gl);
  }
}

// Pose every slot for this frame and write the box rows. Returns the
// structure's rotation (world <- structure frame).
M3 poseBlocks(State* s, int& count) {
  s->pose_jump = false;
  for (int i = 0; i < s->slot_count; i++) s->slots[i].prev = i;
  if (s->built_layout != s->layout) relayout(s);
  compact(s);
  // Each slot's Jut and Seam this frame, the sends and its glint charge in.
  // A change of jut goes (Twang of it) through the glide's spring: what it
  // showed holds back and twangs after its new goal. Seam goes straight in.
  {
    const Drive d = driveOf(s);
    const float k = std::fmin(std::fmax(s->twang, 0.0f), 1.0f);
    const bool spring = k > 0.0f && glideParams(s).time > 1e-4f;
    float o[kGlideCh], jump[kGlideCh];
    for (int i = 0; i < s->slot_count; i++) {
      Slot& sl = s->slots[i];
      const float jut = sent(s, d, kJut, s->jut, sl.charge);
      if (spring && std::fabs(jut - sl.jut) > 1e-6f) {
        const Pose was = showOf(s, sl);
        sl.jut = jut;
        offsetOf(was, goalOf(s, sl), o);
        for (int c = 0; c < kGlideCh; c++) jump[c] = sl.g.out[c] - o[c];
        jump[3] = jump[4] = jump[5] = 0.0f;   // (jut never resizes it)
        sl.g.carry(jump, k);
      }
      sl.jut = jut;
      sl.seam = sent(s, d, kSeam, s->seam, sl.charge);
    }
  }
  const M3 tilt = euler(kRefTiltDeg[0] * kDeg, kRefTiltDeg[1] * kDeg, kRefTiltDeg[2] * kDeg);
  // Spin turns the tilted structure about the world's vertical, or about its
  // own (tilted) axis.
  const M3 turn = euler(0.0f, 0.0f, (float)(s->spin_phase * kTau));
  const M3 egg = s->spin_axis == 0 ? mul(turn, tilt) : mul(tilt, turn);

  count = s->slot_count;
  for (int i = 0; i < count; i++) {
    Slot& sl = s->slots[i];
    const Pose p = showOf(s, sl);
    sl.shown = p;
    if (sl.block >= 0) {
      const Block& b = s->blocks[sl.block];
      sl.circ = circOf(b);
      if (sl.flip <= 0.0f) { sl.stone = b.stone; sl.seed = b.seed; }
    }
    const M3 w = mul(egg, p.rot);
    const V3 c = mul(egg, p.c);
    const V3 ax = col(w, 0), ay = col(w, 1), az = col(w, 2);
    Row* r = &s->rows[i * kBoxRows];
    r[0] = {{c.x, c.y, c.z, p.glow}};
    r[1] = {{ax.x, ax.y, ax.z, p.half.x}};
    r[2] = {{ay.x, ay.y, ay.z, p.half.y}};
    r[3] = {{az.x, az.y, az.z, p.half.z}};
    // .w: the axis flakes circulate this block about (-1: a fragment).
    r[4] = {{sl.stone ? 1.0f : 0.0f, sl.seed, 0.0f, sl.circ}};
    r[5] = {{0.0f, 0.0f, 0.0f, 0.0f}};   // .x: its glint charge (the flakes' pass)
  }
  return egg;
}

void render(void* self, int vp_w, int vp_h) {
  auto* s = static_cast<State*>(self);
  if (!s || !s->initialized || vp_w <= 0 || vp_h <= 0) return;
  auto in = gpu::Device::textureForField("tex_in");
  auto out = gpu::Device::textureForField("tex_out");
  if (!out.valid()) return;

  if (s->opacity < 1.0f / 255.0f) {
    if (in.valid()) {
      auto cp = gpu::ComputePass::begin();
      cp.setPSO(s_pso_prefill);
      cp.setTexture(in, 0, 0);
      cp.setTexture(out, 1, 1);
      cp.dispatch((vp_w + 7) / 8, (vp_h + 7) / 8);
      cp.end();
    } else {
      gpu::Device::clear(out, 0.0f, 0.0f, 0.0f, 0.0f);
    }
    gpu::Device::submit();
    return;
  }

  const int half_w = (vp_w + 1) / 2, half_h = (vp_h + 1) / 2;
  if (!s->scene_tex.valid() || s->scratch_w != vp_w || s->scratch_h != vp_h) {
    s->scene_tex.release();
    s->fog_i.release();
    s->fog_t.release();
    s->fog_tmp.release();
    s->fog_blur.release();
    s->scene_tex = gpu::Device::createTexture(vp_w, vp_h, gpu::TextureFormat::RGBA16F);
    s->fog_i = gpu::Device::createTexture(half_w, half_h, gpu::TextureFormat::RGBA16F);
    s->fog_t = gpu::Device::createTexture(half_w, half_h, gpu::TextureFormat::RGBA16F);
    s->fog_tmp = gpu::Device::createTexture(half_w, half_h, gpu::TextureFormat::RGBA16F);
    s->fog_blur = gpu::Device::createTexture(half_w, half_h, gpu::TextureFormat::RGBA16F);
    s->scratch_w = vp_w;
    s->scratch_h = vp_h;
    // Tap spacing holds the blur's reach at a fixed fraction of the frame.
    const float spacing = std::fmax(1.0f, std::round((float)half_h / 450.0f));
    s->ub_blur_h.writeOne(BlurUniforms{1.0f, 0.0f, spacing, 0.0f});
    s->ub_blur_v.writeOne(BlurUniforms{0.0f, 1.0f, spacing, 0.0f});
  }
  if (!s->scene_tex.valid() || !s->fog_i.valid() || !s->fog_t.valid() ||
      !s->fog_tmp.valid() || !s->fog_blur.valid()) return;

  // --- Scene: blocks, the core and its light samples ---
  int count = 0;
  const M3 egg = poseBlocks(s, count);

  const float wp = (float)s->wander_phase;
  const V3 wander_local = v3(std::sin(wp), std::sin(1.31f * wp + 1.0f),
                             0.6f * std::sin(0.73f * wp + 2.0f)) * (0.9f * s->wander);
  const V3 core = mul(egg, wander_local);
  float flick = 1.0f;
  if (s->flicker > 0.0f) {
    float t = (float)s->time;
    float n = 0.6f * vnoise(v3(t * 7.0f, 0.0f, 0.0f), 31u) +
              0.4f * vnoise(v3(t * 19.0f, 3.0f, 0.0f), 37u);
    flick = std::fmax(0.0f, 1.0f - s->flicker * (0.5f + 0.5f * n));
  }
  // (Last frame's glint charges: this frame's are measured below, on screen.)
  const Drive drive = driveOf(s);
  const float power = sent(s, drive, kPower, s->core_power, drive.src[kGlint]);
  const float watts = 8000.0f * std::pow(2.0f, (power - 0.5f) * 8.0f) * flick;
  const float core_e = kPointE * watts;

  FrameUniforms u = {};
  // Camera: aims at a point a little left of and above the core (the poster's
  // framing), from yaw / pitch / distance; the lens's 36 mm sensor spans the
  // frame's longer side (Blender's Auto fit).
  {
    const V3 target = v3(-1.0f, 0.0f, 0.1f);
    const float yaw = (s->yaw + 360.0f * (float)s->orbit_phase) * kDeg;
    const float pitch = s->pitch * kDeg;
    const V3 fwd = v3(-std::sin(yaw) * std::cos(pitch), std::cos(yaw) * std::cos(pitch),
                      std::sin(pitch));
    const V3 pos = target - fwd * s->distance;
    const V3 right = normalize(cross(fwd, v3(0, 0, 1)));
    const V3 up = cross(right, fwd);
    const auto cs = fx::coverSquare(vp_w, vp_h);
    u.cam_r[0] = right.x; u.cam_r[1] = right.y; u.cam_r[2] = right.z; u.cam_r[3] = pos.x;
    u.cam_u[0] = up.x;    u.cam_u[1] = up.y;    u.cam_u[2] = up.z;    u.cam_u[3] = pos.y;
    u.cam_f[0] = fwd.x;   u.cam_f[1] = fwd.y;   u.cam_f[2] = fwd.z;   u.cam_f[3] = pos.z;
    u.cam_p[0] = s->lens / 18.0f;
    u.cam_p[1] = cs.ax;
    u.cam_p[2] = cs.ay;
    u.cam_p[3] = (float)count;
    // The air wall: a plane facing the camera, air_depth in front of the
    // structure's center (view depth of the center = -pos . fwd).
    const float zc = -dot(pos, fwd);
    u.aux[3] = zc;
    const float zw = zc - s->air_depth;
    const float half = 0.5f * s->air_soft;
    u.air[0] = zw - half;
    u.air[1] = zw + half;
    u.air[2] = s->air;
  }
  u.vp[0] = (float)vp_w; u.vp[1] = (float)vp_h;
  u.vp[2] = 1.0f / vp_w; u.vp[3] = 1.0f / vp_h;
  u.core_p[0] = core.x; u.core_p[1] = core.y; u.core_p[2] = core.z;
  u.core_p[3] = s->core_radius;
  u.core_i[0] = core_e * s->core_r; u.core_i[1] = core_e * s->core_g;
  u.core_i[2] = core_e * s->core_b; u.core_i[3] = (float)kSamples;
  // Suns: directions TOWARD the light (the scene's lights point the other way).
  {
    const V3 amb = normalize(v3(0.06907761f, 0.13917311f, 0.98785585f));
    const V3 rim = normalize(v3(0.06504976f, 0.92718387f, 0.36891553f));
    u.amb_d[0] = amb.x; u.amb_d[1] = amb.y; u.amb_d[2] = amb.z;
    u.amb_e[0] = kSunE * s->ambient * 0.3f;
    u.amb_e[1] = kSunE * s->ambient * 0.45f;
    u.amb_e[2] = kSunE * s->ambient * 1.0f;
    u.rim_d[0] = rim.x; u.rim_d[1] = rim.y; u.rim_d[2] = rim.z;
    u.rim_e[0] = kSunE * s->rim * 0.45f;
    u.rim_e[1] = kSunE * s->rim * 0.6f;
    u.rim_e[2] = kSunE * s->rim * 1.0f;
  }
  {
    const V3 kp = v3(-12.0f, 14.0f, 9.0f);
    const V3 kd = normalize(kp * -1.0f);
    u.kick_p[0] = kp.x; u.kick_p[1] = kp.y; u.kick_p[2] = kp.z; u.kick_p[3] = 3.0f;
    u.kick_d[0] = kd.x; u.kick_d[1] = kd.y; u.kick_d[2] = kd.z;
    const float ke = kAreaE * s->kicker;
    u.kick_i[0] = ke * 0.55f; u.kick_i[1] = ke * 0.85f; u.kick_i[2] = ke * 1.0f;
  }
  // Water: Volume Scatter color (0.4, 0.55, 1.0) g 0.45 + Volume Absorption
  // color (0.12, 0.25, 0.7), both at the same density (Cycles: sigma_s =
  // density * color, sigma_a = density * (1 - color)); its slow clumps
  // average ~0.95.
  {
    const float d = s->water * 0.95f;
    const float sc[3] = {0.4f, 0.55f, 1.0f}, ab[3] = {0.12f, 0.25f, 0.7f};
    for (int c = 0; c < 3; c++) {
      u.water_s[c] = d * sc[c];
      u.water_t[c] = d * sc[c] + d * (1.0f - ab[c]);
    }
    u.water_s[3] = 0.45f;
    u.water_t[3] = 60.0f;   // march sphere radius about the core
  }
  // Halo: an ellipsoid (12, 12, 14) a little above the center, fixed in the
  // world (it doesn't turn with the structure).
  u.halo_c[0] = 0.0f; u.halo_c[1] = 0.0f; u.halo_c[2] = 2.0f; u.halo_c[3] = s->halo;
  u.halo_k[0] = 1.0f / 12.0f; u.halo_k[1] = 1.0f / 12.0f; u.halo_k[2] = 1.0f / 14.0f;
  u.halo_k[3] = s->wisps;
  u.halo_p[0] = s->halo_phase;
  u.halo_p[1] = 3.0f;
  u.halo_p[2] = (float)s->drift_phase * 0.5f;
  u.shade[0] = s->bevel;
  u.shade[1] = s->reflect;
  u.shade[2] = s->ember;
  u.surf[0] = s->shell_glow;
  u.surf[1] = s->chips;
  u.surf[2] = s->albedo;
  u.shade[3] = (float)s->debug_view;
  u.grade[0] = std::pow(2.0f, s->exposure);
  u.grade[1] = s->opacity;
  u.grade[2] = in.valid() ? 1.0f : 0.0f;
  u.grade[3] = s->contrast;
  u.misc[0] = (float)(s->frame % 64u);
  u.misc[1] = (float)kTileRes;
  u.misc[2] = (float)kAtlasCols;
  u.misc[3] = (float)kFogSteps;
  // World -> structure frame (the transpose), and the core's plain color in .w.
  u.egg_r0[0] = egg.m[0][0]; u.egg_r0[1] = egg.m[1][0]; u.egg_r0[2] = egg.m[2][0];
  u.egg_r1[0] = egg.m[0][1]; u.egg_r1[1] = egg.m[1][1]; u.egg_r1[2] = egg.m[2][1];
  u.egg_r2[0] = egg.m[0][2]; u.egg_r2[1] = egg.m[1][2]; u.egg_r2[2] = egg.m[2][2];
  // The heart: Emission (1, 0.75, 0.5) at strength 30 in the scene, scaled
  // with the core's power and tinted toward its color.
  {
    const float hs = 30.0f * watts / 8000.0f;
    u.heart[0] = hs * (0.5f + 0.5f * s->core_r);
    u.heart[1] = hs * (0.75f * (0.5f + 0.5f * s->core_g / 0.56f));
    u.heart[2] = hs * (0.5f * (0.5f + 0.5f * s->core_b / 0.26f));
  }
  u.vol_k[0] = kPointVolK;
  u.vol_k[1] = kSunVolK;
  u.vol_k[2] = s->core_clear;
  u.egg_r0[3] = s->core_r * flick; u.egg_r1[3] = s->core_g * flick; u.egg_r2[3] = s->core_b * flick;
  // Light samples: a Fibonacci sphere inside the light (a sphere seen from
  // afar is a disk; 0.7 R spreads the samples over most of it).
  for (int k = 0; k < kSamples; k++) {
    const float zz = 1.0f - (2.0f * k + 1.0f) / kSamples;
    const float rr = std::sqrt(std::fmax(0.0f, 1.0f - zz * zz));
    const float ph = k * 2.39996323f;
    const V3 o = v3(rr * std::cos(ph), rr * std::sin(ph), zz) * (0.7f * s->core_radius);
    u.lsamp[k][0] = core.x + o.x; u.lsamp[k][1] = core.y + o.y; u.lsamp[k][2] = core.z + o.z;
  }
  const int nflakes = s->flakes < 0 ? 0 : (s->flakes > kMaxFlakes ? kMaxFlakes : s->flakes);
  const bool want_motion = state::isOutputConnected("render_outputs");
  // The optional outputs: each only while something is wired to it.
  {
    auto fit = [&](gpu::Texture& t, int& tw, int& th) {
      if (t.valid() && tw == vp_w && th == vp_h) return;
      t.release();
      t = gpu::Device::createTexture(vp_w, vp_h, gpu::TextureFormat::RGBA16F);
      tw = vp_w;
      th = vp_h;
    };
    static const char* const kAuxField[2] = {"aux1_out", "aux2_out"};
    bool glint = false;
    for (int k = 0; k < 2; k++) {
      const int what = s->aux_what[k] >= 1 && s->aux_what[k] <= 7 ? s->aux_what[k] : 1;
      u.aux[k] = 0.0f;
      if (!state::isOutputConnected(kAuxField[k])) continue;
      fit(s->aux_tex[k], s->aux_w[k], s->aux_h[k]);
      if (!s->aux_tex[k].valid()) continue;
      state::setGpuTexture(kAuxField[k], s->aux_tex[k].id);
      u.aux[k] = (float)what;
      glint = glint || what == 6;
    }
    if (glint) fit(s->glint_tex, s->glint_w, s->glint_h);
    if (glint && !s->glint_tex.valid()) {   // (no charge to show: show none)
      for (int k = 0; k < 2; k++)
        if (u.aux[k] == 6.0f) u.aux[k] = 0.0f;
    }
    u.aux[2] = 0.0f;
    if (want_motion) {
      fit(s->depth_tex, s->depth_w, s->depth_h);
      if (s->depth_tex.valid()) {
        state::setGpuTexture("render_outputs/depth", s->depth_tex.id);
        u.aux[2] = 1.0f;
      }
    }
  }
  u.dust[0] = nflakes > 0 ? 0.012f * s->dust : 0.0f;
  u.dust[1] = 0.6f;
  u.dust[2] = kGridExtent;
  u.dust[3] = nflakes > 0 ? 1.0f : 0.0f;

  // --- The sweep, as the flakes feel it.
  const float release = s->knob.release;
  const float S = drive.src[kEnv];
  // The BITE: as the sustain rises (the knob coming in off the choke, its
  // bounce), flakes are snapped straight onto the blocks' lanes, already
  // streaming — the streamlines show the moment it grabs, not once the
  // flakes have drifted near. Bite is the share snapped over a full rise.
  if (s->bite_prev_S >= 0.0f) s->pending_bite += std::fmax(S - s->bite_prev_S, 0.0f);
  s->bite_prev_S = S;
  // Each block's magnetism: the further it strains out the stronger (the
  // ember blocks), x Magnetism and its sends (a glint's band, by default). A glint crosses the
  // STRUCTURE (0 to 1 from its first block on screen along the glint's way
  // to its last), not the whole screen: framed large or small, a band is the
  // same share of the blocks.
  {
    const float aspect = (float)vp_w / (float)vp_h;
    const float ga = s->glint_angle * kDeg;
    const float gc = std::cos(ga), gs = std::sin(ga);
    const float reach = std::fabs(aspect * gc) + std::fabs(gs);
    const V3 cpos = v3(u.cam_r[3], u.cam_u[3], u.cam_f[3]);
    const V3 cr = v3(u.cam_r[0], u.cam_r[1], u.cam_r[2]);
    const V3 cu = v3(u.cam_u[0], u.cam_u[1], u.cam_u[2]);
    const V3 cf = v3(u.cam_f[0], u.cam_f[1], u.cam_f[2]);
    const float glint_fade =
        s->glint_fade > 1e-3f ? std::exp(-s->frame_dt / s->glint_fade) : 0.0f;
    s->glint_cmax = 0.0f;
    float sn_of[kMaxBoxes];
    float smin = 1e9f, smax = -1e9f;
    for (int i = 0; i < count; i++) {
      const Row* r = &s->rows[i * kBoxRows];
      sn_of[i] = -1e9f;
      const V3 rel = v3(r[0].v[0], r[0].v[1], r[0].v[2]) - cpos;
      const float w = dot(rel, cf);
      if (w <= 0.05f) continue;
      const float nx = dot(rel, cr) / w * 2.0f * u.cam_p[0] * u.cam_p[1];
      const float ny = dot(rel, cu) / w * 2.0f * u.cam_p[0] * u.cam_p[2];
      sn_of[i] = (nx * aspect * gc + ny * gs) / reach;
      if (r[4].v[3] >= 0.0f) {
        smin = std::fmin(smin, sn_of[i]);
        smax = std::fmax(smax, sn_of[i]);
      }
    }
    const float span = std::fmax(smax - smin, 1e-3f);
    for (int i = 0; i < count; i++) {
      Row* r = &s->rows[i * kBoxRows];
      float e = 0.0f;
      if (sn_of[i] > -1e8f) {
        const float sn = (sn_of[i] - smin) / span;
        for (int g = 0; g < three_planes_glints::kMaxLive; g++) {
          const auto& gl = s->glint_core.glints[g];
          if (!gl.live) continue;
          const float x = (sn - gl.pos) / std::fmax(1e-3f, s->glint_width * gl.drawWidth());
          const float wk = (x + 1.5f) / 1.8f;
          e += gl.drawGain() * (std::exp(-x * x) - 0.5f * gl.drawShade() * std::exp(-wk * wk));
        }
      }
      // Its charge: the band's pull, held a moment after it has passed.
      float& c = s->slots[i].charge;
      c = std::fmax(std::fmax(e, 0.0f), c * glint_fade);
      e = c;
      s->glint_cmax = std::fmax(s->glint_cmax, c);
      r[5].v[0] = c;
      // (The loose fragments are no magnets: flakes held round so small a
      // box only spin in place.)
      r[4].v[2] = r[4].v[3] < 0.0f
                ? 0.0f
                : (0.35f + r[0].v[3]) * sent(s, drive, kMagnet, s->magnetism, e);
    }
  }
  // Slots compacted away this frame: where each of last frame's went
  // (-1: gone), a row each after the boxes, for the flakes holding onto them.
  bool remap = false;
  int remap_rows = 0;
  if (!s->pose_jump) {
    const int n0 = s->last_slot_count;
    Row* rm = &s->rows[count * kBoxRows];
    for (int o = 0; o < n0; o++) rm[o] = {{-1.0f, 0.0f, 0.0f, 0.0f}};
    for (int i = 0; i < count; i++) {
      const int pv = s->slots[i].prev;
      if (pv >= 0 && pv < n0) rm[pv].v[0] = (float)i;
    }
    for (int o = 0; o < n0 && o < count; o++) remap = remap || rm[o].v[0] != (float)o;
    if (remap) remap_rows = n0;
  }
  s->box_buf.write(s->rows, count * kBoxRows + remap_rows);

  PartUniforms pu = {};
  int p_init_from = s->p_live;
  if (nflakes < s->p_live) p_init_from = nflakes;
  int steps = (int)std::ceil(s->sim_dt / kSubstep - 1e-6);
  steps = steps < 0 ? 0 : (steps > 36 ? 36 : steps);
  if (nflakes > 0) {
    pu.pa[0] = (float)nflakes;
    pu.pa[1] = (float)p_init_from;
    pu.pa[2] = steps > 0 ? (float)(s->sim_dt / steps) : 0.0f;
    pu.pa[3] = (float)steps;
    pu.pb[0] = S;
    pu.pb[1] = release;
    pu.pb[2] = s->pending_fling;
    pu.pb[3] = (float)s->time;
    pu.ring[0] = s->ring_radius;
    pu.ring[1] = s->ring_loose;
    pu.ring[2] = s->ring_speed;
    pu.ring[3] = 0.6f;
    {
      // The jet's axis: the camera's right turned about the vertical.
      const float ja = s->jet_angle * kDeg;
      const V3 cr = normalize(v3(u.cam_r[0], u.cam_r[1], 0.0f));
      const V3 cb = v3(-cr.y, cr.x, 0.0f);
      const V3 J = normalize(cr * std::cos(ja) + cb * std::sin(ja));
      pu.jet[0] = J.x; pu.jet[1] = J.y; pu.jet[2] = J.z;
      pu.jet[3] = s->jet * std::fmin(S / 0.4f, 1.0f);
    }
    pu.jet2[0] = 6.0f + 10.0f * S;
    pu.jet2[1] = s->local;
    pu.jet2[2] = s->ricochet;
    pu.jet2[3] = s->vortex;
    pu.cap[0] = s->stick;
    pu.cap[1] = s->reach;
    pu.cap[2] = s->hold;
    pu.cap[3] = s->lanes;
    {
      const float x = std::fmin(S, 1.0f);
      const float over = std::fmax(S - 1.0f, 0.0f);
      // The depth of the sweep: in the middle the lanes hug at Standoff and
      // run at Orbit Speed; toward either end they widen to Standoff Deep and
      // slow to Speed Deep (and the pull and the hold weaken with S), until
      // the choke lets them go. An overshoot goes past the middle's.
      pu.orbit[0] = (s->standoff_deep + (s->standoff - s->standoff_deep) * x) * (1.0f - 0.5f * over);
      // (The pace eases in: the liminal stretch near either end runs slow and
      // loose, on the brink, and only the middle runs at full speed.)
      pu.orbit[1] = (s->speed_deep + (s->orbit_speed - s->speed_deep) * x * x) * (1.0f + 0.6f * over);
      pu.orbit[2] = 6.0f;
      pu.orbit[3] = 1.6f * s->capture;
      pu.bite[0] = std::fmin(0.6f, s->bite * s->pending_bite);
      pu.bite[1] = s->glint_cmax > 0.02f ? s->glint_pull : 0.0f;
      pu.bite[2] = remap ? 1.0f : 0.0f;
    }
    pu.shell[0] = s->shell_radius;
    pu.shell[1] = 9.0f;
    pu.shell[2] = 7.0f * s->fling;
    pu.shell[3] = s->heat;
    pu.vibe[0] = s->vibe * release;
    pu.vibe[1] = (float)s->vibe_phase;
    pu.vibe[2] = s->vibe_spread;
    pu.vibe[3] = (float)(kTau * s->vibe_hz * (0.12 + 0.88 * release));
    pu.flake[0] = s->flake_size;
    pu.flake[1] = s->mica;
    pu.flake[2] = s->tumble;
    pu.flake[3] = s->sparkle;
    pu.grid[0] = kGridExtent;
    pu.grid[1] = (float)kGrid;
    pu.grid[2] = 1.0f - std::exp(-(float)s->sim_dt / 0.4f);
    pu.grid[3] = kSplat;
  }
  pu.pframe[0] = s->frame_dt;
  pu.pframe[1] = (float)s->frame;
  pu.pframe[2] = (want_motion && s->prev_cam_valid) ? 1.0f : 0.0f;
  pu.pframe[3] = s->streak;
  pu.mvs[0] = s->motion_flakes;
  pu.mvs[1] = s->motion_scene;
  pu.pview[0] = (float)vp_w; pu.pview[1] = (float)vp_h;
  pu.pview[2] = 1.0f / vp_w; pu.pview[3] = 1.0f / vp_h;
  {
    const float* cur[3] = {u.cam_r, u.cam_u, u.cam_f};
    float* prv[3] = {pu.prev_r, pu.prev_u, pu.prev_f};
    for (int k = 0; k < 3; k++)
      for (int c = 0; c < 4; c++) prv[k][c] = s->prev_cam_valid ? s->prev_cam[k][c] : cur[k][c];
  }
  s->ub_part.writeOne(pu);

  s->ub_frame.writeOne(u);
  s->frame++;

  {
    auto cp = gpu::ComputePass::begin();
    cp.setPSO(s_pso_shadow);
    cp.setBuffer(s->ub_frame, 0);
    cp.setBuffer(s->box_buf, 1);
    cp.setTexture(s->atlas, 2, 1);
    cp.dispatch(kTileRes * kAtlasCols / 8, kTileRes * kAtlasRows / 8);
    cp.end();
  }
  {
    auto cp = gpu::ComputePass::begin();
    cp.setPSO(s_pso_escape);
    cp.setBuffer(s->ub_frame, 0);
    cp.setBuffer(s->box_buf, 1);
    cp.setTexture(s->esc, 2, 1);
    cp.dispatch(kEscRes / 8, kEscRes / 8);
    cp.end();
  }
  {
    auto cp = gpu::ComputePass::begin();
    cp.setPSO(s_pso_reduce);
    cp.setBuffer(s->ub_frame, 0);
    cp.setBuffer(s->box_buf, 1);
    cp.setTexture(s->esc, 2, 0);
    cp.setTexture(s->esc_low, 3, 1);
    cp.dispatch(kEscLow / 8, kEscLow / 8);
    cp.end();
  }
  {
    auto cp = gpu::ComputePass::begin();
    cp.setPSO(s_pso_vis);
    cp.setBuffer(s->ub_frame, 0);
    cp.setBuffer(s->box_buf, 1);
    cp.setTexture(s->esc, 2, 0);
    cp.setTexture(s->esc_low, 3, 0);
    cp.setSampler(s->samp, 4);
    cp.setTexture(s->vis_f, 5, 1);
    cp.setTexture(s->vis_d, 6, 1);
    cp.dispatch(kEscRes / 8, kEscRes / 8);
    cp.end();
  }
  {
    auto cp = gpu::ComputePass::begin();
    cp.setPSO(s_pso_scene);
    cp.setBuffer(s->ub_frame, 0);
    cp.setBuffer(s->box_buf, 1);
    cp.setTexture(s->atlas, 2, 0);
    cp.setTexture(s->scene_tex, 3, 1);
    cp.setTexture(s->glint_tex.valid() ? s->glint_tex : s->stand_in[0], 4, 1);
    cp.dispatch((vp_w + 7) / 8, (vp_h + 7) / 8);
    cp.end();
  }
  // The flakes move, and leave their presence for the fog.
  if (nflakes > 0) {
    {
      auto cp = gpu::ComputePass::begin();
      cp.setPSO(s_pso_p_update);
      cp.setBuffer(s->ub_frame, 0);
      cp.setBuffer(s->box_buf, 1);
      cp.setBuffer(s->p_state, 2);
      cp.setBuffer(s->p_grid, 3);
      cp.setBuffer(s->ub_part, 4);
      cp.dispatch((nflakes + 63) / 64, 1);
      cp.end();
    }
    {
      auto cp = gpu::ComputePass::begin();
      cp.setPSO(s_pso_p_resolve);
      cp.setBuffer(s->p_grid, 3);
      cp.setBuffer(s->ub_part, 4);
      cp.setBuffer(s->p_accum, 5);
      cp.setTexture(s->presence, 6, 1);
      cp.dispatch(kGrid / 4, kGrid / 4, kGrid / 4);
      cp.end();
    }
    s->p_live = nflakes;
    s->sim_dt = 0.0;
    s->pending_fling = 0.0f;
    s->pending_bite = 0.0f;
  } else {
    s->p_live = 0;
  }
  {
    auto cp = gpu::ComputePass::begin();
    cp.setPSO(s_pso_fog);
    cp.setBuffer(s->ub_frame, 0);
    cp.setBuffer(s->box_buf, 1);
    cp.setTexture(s->esc, 2, 0);
    cp.setTexture(s->scene_tex, 3, 0);
    cp.setTexture(s->fog_i, 4, 1);
    cp.setTexture(s->fog_t, 5, 1);
    cp.setTexture(s->vis_f, 6, 0);
    cp.setTexture(s->vis_d, 7, 0);
    cp.setSampler(s->samp, 8);
    cp.setTexture(nflakes > 0 ? s->presence : s->presence_zero, 9, 0);
    cp.dispatch((half_w + 7) / 8, (half_h + 7) / 8);
    cp.end();
  }
  for (int pass = 0; pass < 2; pass++) {
    auto cp = gpu::ComputePass::begin();
    cp.setPSO(s_pso_blur);
    cp.setBuffer(s->ub_frame, 0);
    cp.setBuffer(s->box_buf, 1);
    cp.setTexture(pass == 0 ? s->fog_i : s->fog_tmp, 2, 0);
    cp.setTexture(pass == 0 ? s->fog_tmp : s->fog_blur, 3, 1);
    cp.setBuffer(pass == 0 ? s->ub_blur_h : s->ub_blur_v, 4);
    cp.dispatch((half_w + 7) / 8, (half_h + 7) / 8);
    cp.end();
  }
  // The blocks' motion vectors, when a motion blur downstream reads them.
  if (want_motion) {
    if (!s->motion_tex.valid() || s->motion_w != vp_w || s->motion_h != vp_h) {
      s->motion_tex.release();
      s->motion_tex = gpu::Device::createTexture(vp_w, vp_h, gpu::TextureFormat::RGBA16F);
      s->motion_w = vp_w;
      s->motion_h = vp_h;
      if (s->motion_tex.valid()) state::setGpuTexture("render_outputs/motion", s->motion_tex.id);
    }
    // Last frame's poses (this frame's when the blocks were rebuilt).
    static Row prev4[kMaxBoxes * 4];
    for (int i = 0; i < count; i++) {
      const int pv = s->slots[i].prev;
      const bool had = !s->pose_jump && pv >= 0 && pv < s->prev_count;
      const Row* src = had ? &s->prev_rows[pv * kBoxRows] : &s->rows[i * kBoxRows];
      for (int k = 0; k < 4; k++) prev4[i * 4 + k] = src[k];
    }
    s->prev_box_buf.write(prev4, count * 4);
    if (s->motion_tex.valid()) {
      auto cp = gpu::ComputePass::begin();
      cp.setPSO(s_pso_motion);
      cp.setBuffer(s->ub_frame, 0);
      cp.setBuffer(s->box_buf, 1);
      cp.setBuffer(s->prev_box_buf, 2);
      cp.setTexture(s->scene_tex, 3, 0);
      cp.setBuffer(s->ub_part, 4);
      cp.setTexture(s->motion_tex, 5, 1);
      cp.dispatch((vp_w + 7) / 8, (vp_h + 7) / 8);
      cp.end();
    }
  }
  // The flakes: shaded, then splatted into the spark layer (and over the
  // blocks' motion).
  if (nflakes > 0) {
    if (!s->spark.valid() || s->spark_w != vp_w || s->spark_h != vp_h) {
      s->spark.release();
      s->spark = gpu::Device::createTexture(vp_w, vp_h, gpu::TextureFormat::RGBA16F);
      s->spark_w = vp_w;
      s->spark_h = vp_h;
    }
    {
      auto cp = gpu::ComputePass::begin();
      cp.setPSO(s_pso_p_shade);
      cp.setBuffer(s->ub_frame, 0);
      cp.setBuffer(s->box_buf, 1);
      cp.setBuffer(s->p_state, 2);
      cp.setBuffer(s->ub_part, 4);
      cp.setTexture(s->atlas, 5, 0);
      cp.setTexture(s->scene_tex, 6, 0);
      cp.setBuffer(s->p_recs, 7);
      cp.dispatch((nflakes + 63) / 64, 1);
      cp.end();
    }
    if (s->spark.valid()) {
      const bool mv = want_motion && s->motion_tex.valid();
      auto rp = mv ? gpu::RenderPass::beginMRT({{s->spark, 0.0f, 0.0f, 0.0f, 0.0f, false},
                                                {s->motion_tex, 0.0f, 0.0f, 0.0f, 0.0f, true}})
                   : gpu::RenderPass::begin(s->spark, 0.0f, 0.0f, 0.0f, 0.0f);
      rp.setPSO(mv ? s_pso_spark_mv : s_pso_spark);
      rp.setBuffer(s->ub_part, 4);
      rp.setBuffer(s->p_recs, 7);
      rp.draw(6, nflakes);
      rp.end();
    }
  }
  {
    auto cp = gpu::ComputePass::begin();
    cp.setPSO(s_pso_comp);
    cp.setBuffer(s->ub_frame, 0);
    cp.setBuffer(s->box_buf, 1);
    cp.setTexture(s->scene_tex, 2, 0);
    cp.setTexture(s->fog_blur, 3, 0);
    cp.setTexture(s->fog_t, 4, 0);
    cp.setTexture(in.valid() ? in : s->zero_tex, 5, 0);
    cp.setTexture(out, 6, 1);
    cp.setTexture(nflakes > 0 && s->spark.valid() ? s->spark : s->zero_tex, 7, 0);
    cp.dispatch((vp_w + 7) / 8, (vp_h + 7) / 8);
    cp.end();
  }
  // The optional outputs, while any is wired.
  if (u.aux[0] > 0.5f || u.aux[1] > 0.5f || u.aux[2] > 0.5f) {
    auto cp = gpu::ComputePass::begin();
    cp.setPSO(s_pso_aux);
    cp.setBuffer(s->ub_frame, 0);
    cp.setBuffer(s->box_buf, 1);
    cp.setTexture(s->scene_tex, 2, 0);
    cp.setTexture(s->fog_blur, 3, 0);
    cp.setTexture(s->fog_t, 4, 0);
    cp.setTexture(nflakes > 0 && s->spark.valid() ? s->spark : s->zero_tex, 5, 0);
    cp.setTexture(s->glint_tex.valid() ? s->glint_tex : s->zero_tex, 6, 0);
    cp.setTexture(u.aux[0] > 0.5f ? s->aux_tex[0] : s->stand_in[0], 7, 1);
    cp.setTexture(u.aux[1] > 0.5f ? s->aux_tex[1] : s->stand_in[1], 8, 1);
    cp.setTexture(u.aux[2] > 0.5f ? s->depth_tex : s->stand_in[2], 9, 1);
    cp.dispatch((vp_w + 7) / 8, (vp_h + 7) / 8);
    cp.end();
  }
  gpu::Device::submit();

  // This frame's poses, for next frame's motion.
  for (int i = 0; i < count * kBoxRows; i++) s->prev_rows[i] = s->rows[i];
  s->prev_count = count;
  s->last_slot_count = count;
  for (int c = 0; c < 4; c++) {
    s->prev_cam[0][c] = u.cam_r[c];
    s->prev_cam[1][c] = u.cam_u[c];
    s->prev_cam[2][c] = u.cam_f[c];
  }
  s->prev_cam_valid = true;
}

}  // namespace lantern
