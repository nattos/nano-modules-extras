/*
 * source.light.motion_blobs — pool of traveling soft blobs that emit motion
 * vectors and/or color darkening configurably. Replaces the original
 * placeholders fx.directional_blur, fx.zoom_blur, fx.shadow_flyover —
 * same blob field drives both outputs independently via
 * `motion_strength` and `shadow_darkness`.
 *
 * Common combinations:
 *   motion_strength=1.0, shadow_darkness=0.0  → motion rain (invisible blobs)
 *   motion_strength=0.0, shadow_darkness=0.7  → shadow flyover
 *   motion_strength=0.5, shadow_darkness=0.5  → cinematic shadow with motion blur
 *
 * Blob lifecycle:
 *   - Spawn on configurable edge (top/bottom/left/right), or random
 *     edge per spawn when spawn_edge_random is on.
 *   - Velocity has traverse (perpendicular INTO canvas) + drift
 *     (parallel to edge, jittered).
 *   - Position spawns just outside the canvas at spawn_offset (negative).
 *   - Die when blob exits the OPPOSITE side; respawn only while
 *     alive_count < density * blob_count_max.
 *
 * Class-like instance model: module_init() compiles the two shared
 * compute PSOs + publishes the schema once per type; each chain entry
 * gets its own State (params, blob pool, per-instance buffers/textures)
 * via create(). All instance callbacks take `self`.
 */

#include <gpu.h>
#include <host.h>
#include <effect_utils.h>
#include "motion_blobs_shaders.h"

#include <cmath>
#include <cstdint>
#include <cstring>

namespace motion_blobs {

enum SpawnEdge : int { EDGE_TOP = 0, EDGE_BOTTOM = 1, EDGE_LEFT = 2, EDGE_RIGHT = 3 };
static constexpr int MAX_BLOBS = 32;
// Safety recycle: a hard inward pinch can trap a blob orbiting the center;
// recycle any blob older than this so the field keeps flowing.
static constexpr float BLOB_MAX_AGE = 10.0f;

struct CpuBlob {
  bool   alive;
  float  x, y;
  float  vx0, vy0;        // free-stream velocity (constant, set at spawn)
  float  vx, vy;          // current velocity = flow field sampled at (x,y)
  float  radius;          // cover-square units
  float  age;             // seconds since spawn (safety recycle under hard pinch)
};

struct GpuBlob {
  float x; float y;
  float vx; float vy;
  float radius;
  float _pp0; float _pp1; float _pp2;
};
static_assert(sizeof(GpuBlob) == 32, "GpuBlob layout mismatch");

struct Uniforms {
  float motion_strength;
  float shadow_darkness;
  float softness_curve;
  float motion_extent;

  float shadow_r; float shadow_g; float shadow_b; float _pad1;
  float aspect_x; float aspect_y; float _pad2; float _pad3;

  uint32_t active_count;
  uint32_t debug_show_blobs;
  uint32_t _pad4;
  uint32_t _pad5;
};
static_assert(sizeof(Uniforms) == 64, "Uniforms layout mismatch");

// Per-instance state. One per chain entry.
struct State {
  gpu::Buffer  uniform_buf;
  gpu::Buffer  blob_buf;
  gpu::Texture motion_tex;
  gpu::Texture zero_motion_tex;   // 1x1 rgba16f fallback (no upstream)
  int          motion_w = 0;
  int          motion_h = 0;
  bool         initialized = false;

  // --- Schema-mirrored params (standard) ---
  float density               = 0.4f;
  float traverse_speed        = 0.7f;
  float traverse_speed_jitter = 0.5f;
  float drift                 = 0.0f;
  // Signed [-1,+1]. Biases the spawn distribution so trajectories pass nearer
  // the viewport center (-1, bell concentrated on the center-streamline) or
  // out toward the periphery (+1). 0 = uniform.
  float center_bias           = 0.0f;
  // Signed [-1,+1]. Curves trajectories radially about the center: +1 bulges
  // them OUTWARD (fish-eye / onion), -1 pinches them INWARD. 0 = straight.
  // Magnitude is the strength (how strongly paths follow the onion).
  float arc_bias              = 0.0f;
  // Onion size — the deflection cylinder's radius in cover-square units.
  // Larger = a bigger onion (the bow extends further from center).
  float arc_scale             = 0.5f;
  float motion_strength       = 1.0f;
  // Size of the emitted motion footprint relative to the blob, 1→0. 1 = full
  // blob extent (default); 0.5 = vectors reach only ~50% of the blob, focused
  // on its center.
  float motion_extent         = 1.0f;
  float shadow_darkness       = 0.0f;
  float shadow_tint_r         = 0.0f;
  float shadow_tint_g         = 0.0f;
  float shadow_tint_b         = 0.0f;
  float blob_size             = 0.12f;
  int   spawn_edge            = EDGE_TOP;
  // --- Tuning ---
  int   blob_count_max        = 8;
  float blob_size_jitter      = 0.3f;
  float drift_jitter          = 0.1f;
  float softness_curve        = 4.0f;
  bool  spawn_edge_random     = false;
  int   seed                  = 0x82C00;
  // --- Debug ---
  bool  debug_show_blobs      = false;

  // --- Runtime ---
  CpuBlob  blobs[MAX_BLOBS];
  uint32_t spawn_rng      = 0x82C00LU;
  // Cover-square aspect half-extents, cached from render() so spawn (which
  // runs in tick, before render) can place blobs fully outside the viewport
  // including their soft halo. Overwritten on the first render; spawning is
  // held off until then.
  float aspect_x     = 0.5f;
  float aspect_y     = 0.5f;
  bool  aspect_ready = false;
};

// Type-shared: compiled once in module_init().
static gpu::ComputePSO s_pso_color;
static gpu::ComputePSO s_pso_motion;

static inline float clampf(float v, float lo, float hi) {
  return v < lo ? lo : (v > hi ? hi : v);
}
static inline int clampi(int v, int lo, int hi) {
  return v < lo ? lo : (v > hi ? hi : v);
}
static inline uint32_t lcg_next(uint32_t& s) {
  s = s * 1664525u + 1013904223u; return s;
}
static inline float lcg_unit(uint32_t& s) {
  return (lcg_next(s) >> 8) * (1.0f / (float)(1u << 24));
}
static inline float lcg_signed(uint32_t& s) { return lcg_unit(s) * 2.0f - 1.0f; }

// Distance (uv, along the given perpendicular axis) a blob center must sit
// beyond the viewport edge for its gaussian halo to fall below ~2% alpha at
// the edge. The halo reaches sqrt(ln(1/ε)/softness) ≈ sqrt(4/softness)
// radii; +0.5 radius of slack. Converted from the cover-square radius to uv
// via the aspect half-extent of that axis.
static inline float halo_margin(float radius, float aspect_perp, float softness_curve) {
  float softness = softness_curve < 0.1f ? 0.1f : softness_curve;
  float clearance_radii = std::sqrt(4.0f / softness) + 0.5f;
  return clearance_radii * radius * aspect_perp;
}

// Warp a uniform edge-crossing u ∈ [0,1] toward (cb<0) or away from (cb>0)
// `pivot` — the point where the center-streamline crosses this edge. cb=0 is
// identity. A power curve on each side of the pivot gives a bell-ish
// concentration at the pivot (trajectories through center) or at the edge
// ends (trajectories to the periphery).
static inline float center_bias_warp(float u, float pivot, float cb) {
  if (cb > -1e-4f && cb < 1e-4f) return u;
  float p = pivot < 0.0f ? 0.0f : (pivot > 1.0f ? 1.0f : pivot);
  float gamma = std::pow(2.0f, -cb * 2.0f);   // cb<0 → >1 (toward pivot); cb>0 → <1 (toward ends)
  if (u <= p) {
    if (p < 1e-5f) return u;
    float t = u / p;                          // 1 at pivot
    return p * std::pow(t, 1.0f / gamma);
  }
  if (p > 1.0f - 1e-5f) return u;
  float t = (u - p) / (1.0f - p);             // 0 at pivot
  return p + (1.0f - p) * std::pow(t, gamma);
}

// Potential-flow-around-a-cylinder velocity field, centered on the viewport
// and aligned with the blob's free-stream (vx0, vy0). Far from center it IS
// the free-stream (so spawn / cull / coverage are unaffected); near center
// it deflects the path AROUND the center and the streamlines RECONVERGE
// downstream — the onion. `a2` is the signed doublet strength (cylinder
// radius², cover-square units) and sets the onion SIZE: >0 bulges outward,
// <0 pinches inward. `strength` ∈ [0,1] blends the (constant-speed) field
// direction against the free-stream, so it controls how strongly paths
// follow the onion independently of its size. Computed in cover-square space
// so the bow is circular on screen. Output velocity is in uv/sec.
static void field_velocity(float x, float y, float vx0, float vy0,
                           float a2, float strength,
                           float aspect_x, float aspect_y,
                           float* ovx, float* ovy) {
  if (strength < 1e-4f || (a2 > -1e-6f && a2 < 1e-6f)) { *ovx = vx0; *ovy = vy0; return; }
  float ax = aspect_x, ay = aspect_y;
  float qx = (x - 0.5f) / ax;                 // position rel. center (cover-square)
  float qy = (y - 0.5f) / ay;
  float ucx = vx0 / ax, ucy = vy0 / ay;       // free-stream (cover-square)
  float umag = std::sqrt(ucx * ucx + ucy * ucy);
  if (umag < 1e-6f) { *ovx = vx0; *ovy = vy0; return; }
  float dhx = ucx / umag, dhy = ucy / umag;   // flow direction
  float nhx = -dhy, nhy = dhx;                 // perpendicular
  float along = qx * dhx + qy * dhy;           // flow-frame coords
  float perp  = qx * nhx + qy * nhy;
  float r2 = along * along + perp * perp;
  float a2abs = a2 < 0.0f ? -a2 : a2;
  float floor_r2 = a2abs > 1e-3f ? a2abs : 1e-3f;   // stay outside the cylinder (bound the doublet)
  if (r2 < floor_r2) r2 = floor_r2;
  float r4 = r2 * r2;
  float u_local = umag * (1.0f - a2 * (along * along - perp * perp) / r4);
  float v_local = umag * (-2.0f * a2 * along * perp / r4);
  float vcx = u_local * dhx + v_local * nhx;   // back to cover-square world
  float vcy = u_local * dhy + v_local * nhy;
  // Use the field for DIRECTION only and renormalize to the free-stream
  // speed (umag). We want the curved streamline SHAPE (which reconverges —
  // the onion), but not its speed profile: the raw field slows to zero at
  // the cylinder's stagnation points (blobs would stall there — the
  // "attractor" at the onion's tail) and races at the flanks. Constant
  // screen speed reads far more naturally and never stalls. At a true
  // stagnation point the direction is undefined → fall back to free-stream.
  float fmag = std::sqrt(vcx * vcx + vcy * vcy);
  float fvx, fvy;                              // field direction at free-stream speed
  if (fmag > 1e-5f) {
    float s = umag / fmag;
    fvx = vcx * s; fvy = vcy * s;
  } else {
    fvx = ucx; fvy = ucy;
  }
  // Blend free-stream → field direction by `strength`, then renormalize to
  // free-stream speed (both endpoints have magnitude umag, so this is a
  // clean directional interpolation that keeps the steady speed).
  float bx = ucx + strength * (fvx - ucx);
  float by = ucy + strength * (fvy - ucy);
  float bmag = std::sqrt(bx * bx + by * by);
  if (bmag > 1e-5f) {
    float s = umag / bmag;
    bx *= s; by *= s;
  } else {
    bx = ucx; by = ucy;
  }
  *ovx = bx * ax;                              // back to uv
  *ovy = by * ay;
}

static void spawn_one(State& st, CpuBlob& b, bool scattered) {
  // The primary edge sets the flow ORIENTATION: traverse runs perpendicular
  // to it (into the canvas), drift runs parallel.
  int edge = st.spawn_edge;
  if (st.spawn_edge_random) {
    edge = (int)(lcg_unit(st.spawn_rng) * 4.0f);
    if (edge > 3) edge = 3;
  }
  float speed = st.traverse_speed * (1.0f + clampf(st.traverse_speed_jitter, 0.0f, 1.0f) * lcg_signed(st.spawn_rng));
  float drift = st.drift + clampf(st.drift_jitter, 0.0f, 1.0f) * lcg_signed(st.spawn_rng);
  float size  = st.blob_size * (1.0f + clampf(st.blob_size_jitter, 0.0f, 1.0f) * lcg_signed(st.spawn_rng));
  if (size < 1e-4f) size = 1e-4f;

  // Resolve the velocity (uv/sec) for this blob.
  float vx, vy;
  switch (edge) {
    case EDGE_TOP:    vx = drift;   vy = +speed;  break;
    case EDGE_BOTTOM: vx = drift;   vy = -speed;  break;
    case EDGE_LEFT:   vx = +speed;  vy = drift;   break;
    default:          vx = -speed;  vy = drift;   break;  // RIGHT
  }

  // Inject across EVERY inflow edge (those the flow enters through),
  // choosing one with probability proportional to its normal flux
  // (|v·n| · edge_length; edge_length = 1 in uv). For a constant velocity
  // field this is exactly the condition that fills the interior uniformly
  // — without it, an angled (drifting) flow leaves a growing wedge of the
  // viewport on the upstream side permanently uncovered.
  float fx = vx < 0.0f ? -vx : vx;        // flux through left/right (vertical) edges
  float fy = vy < 0.0f ? -vy : vy;        // flux through top/bottom (horizontal) edges
  float ftot = fx + fy;
  float mx = halo_margin(size, st.aspect_x, st.softness_curve);
  float my = halo_margin(size, st.aspect_y, st.softness_curve);

  // `e` is where the blob should CROSS the edge (uniform along it). The
  // spawn point sits one halo-margin outside (perpendicular), and we
  // back-project along the parallel axis by the drift accrued over the
  // approach time (margin / perpendicular speed). Without this, a blob with
  // a large margin (big/soft blob) and sideways drift would slide far along
  // the parallel axis during its long approach and cross the edge way off
  // the intended spot — or miss the viewport entirely — which is what left
  // the upstream side uncovered. (NOTE: this can place the spawn well past
  // the OPPOSITE side's margin, e.g. far to the right for a leftward flow —
  // the downstream-only cull in tick() is what keeps such still-approaching
  // blobs alive instead of killing them on frame one.)
  float e_uniform = lcg_unit(st.spawn_rng);

  float sx, sy;
  bool enter_horizontal_edge = (ftot <= 1e-6f) ? true
                             : (lcg_unit(st.spawn_rng) * ftot < fy);
  if (enter_horizontal_edge) {
    // Cross the top (moving down) or bottom (moving up) edge at x = e.
    // pivot = x where the center-streamline crosses this edge.
    float y_edge = (vy >= 0.0f) ? 0.0f : 1.0f;
    float pivot = (vy > 1e-5f || vy < -1e-5f) ? 0.5f + vx * (y_edge - 0.5f) / vy : 0.5f;
    float e = center_bias_warp(e_uniform, pivot, st.center_bias);
    float vperp = (vy >= 0.0f) ? vy : -vy;            // |vy|
    float t0 = (vperp > 1e-6f) ? my / vperp : 0.0f;   // spawn → crossing time
    sx = e - vx * t0;
    sy = (vy >= 0.0f) ? -my : 1.0f + my;
  } else {
    // Cross the left (moving right) or right (moving left) edge at y = e.
    // pivot = y where the center-streamline crosses this edge.
    float x_edge = (vx >= 0.0f) ? 0.0f : 1.0f;
    float pivot = (vx > 1e-5f || vx < -1e-5f) ? 0.5f + vy * (x_edge - 0.5f) / vx : 0.5f;
    float e = center_bias_warp(e_uniform, pivot, st.center_bias);
    float vperp = (vx >= 0.0f) ? vx : -vx;            // |vx|
    float t0 = (vperp > 1e-6f) ? mx / vperp : 0.0f;
    sy = e - vy * t0;
    sx = (vx >= 0.0f) ? -mx : 1.0f + mx;
  }

  // Initial seeding: advance the blob by a random fraction of its full
  // transit time so the pool starts distributed along the ENTIRE flow path
  // — upstream approach zone, viewport, and exit alike. This both fills the
  // viewport instantly (no opening wave) AND pre-loads the off-screen
  // approach pipeline, so there's no lull before the first edge-respawns
  // float in. Uniform-along-trajectory == the steady-state distribution.
  if (scattered) {
    float tx = 1e9f, ty = 1e9f;          // time to reach the downstream edge
    if (vx < -1e-6f)      tx = (-mx - sx) / vx;
    else if (vx > 1e-6f)  tx = (1.0f + mx - sx) / vx;
    if (vy < -1e-6f)      ty = (-my - sy) / vy;
    else if (vy > 1e-6f)  ty = (1.0f + my - sy) / vy;
    float t_exit = (tx < ty) ? tx : ty;
    if (t_exit < 0.0f) t_exit = 0.0f;
    float age = lcg_unit(st.spawn_rng) * t_exit;
    sx += vx * age;
    sy += vy * age;
  }

  b.x = sx;
  b.y = sy;
  b.vx0 = vx;  b.vy0 = vy;   // free-stream (constant)
  b.vx = vx;   b.vy = vy;    // current velocity (== free-stream until the field bends it)
  b.radius = size;
  b.age = 0.0f;
  b.alive = true;
}

// Type-level setup: schema + the two shared compute PSOs.
void module_init() {
  state::init("source.light.motion_blobs", {1, 0, 1},
    state::Schema()
      .helpField("intro",
        "## Motion Blobs\n"
        "A pool of soft blobs that drift across the frame and drive **two outputs at "
        "once**: a motion-vector field (for motion blur) and a colour darkening (for "
        "shadow flyovers). The same blob field powers both — mix them with *Motion "
        "Strength* and *Shadow Darkness*.\n\n"
        "**Try:** *Motion Strength* 1 / *Shadow Darkness* 0 for invisible motion rain; "
        "flip it to 0 / 0.7 for gliding shadows; or run both at ~0.5 for cinematic shadow "
        "with motion blur. Steer travel with *Traverse Speed* and the *Arc*/*Center* bias.")
      // --- Blobs & motion ---
      .group("field", "Blobs & Motion")
        .groupHelp(
          "*Density* sets how many blobs live at once. They enter from the spawn edge and "
          "cross the frame at *Traverse Speed* (jittered per blob); *Drift* slides them "
          "along the edge. *Center Bias* pulls travel toward the middle, and the *Arc* "
          "controls curve their paths — dial these to get lazy sweeps or tight fly-bys.")
      .floatField("density",               0.4f,  0.0f, 1.0f,    state::PrimaryInput).label("Density", "Dens")
      .floatField("traverse_speed",        0.7f,  0.0f, 3.0f,    state::PrimaryInput).label("Traverse Speed", "Trav")
      .floatField("traverse_speed_jitter", 0.5f,  0.0f, 1.0f,    state::PrimaryInput).label("Traverse Jitter", "TrvJit")
      .floatField("drift",                 0.0f, -1.0f, 1.0f,    state::PrimaryInput).label("Drift", "Drift")
      .floatField("center_bias",           0.0f, -1.0f, 1.0f,    state::PrimaryInput).label("Center Bias", "CtrBia")
      .floatField("arc_bias",              0.0f, -1.0f, 1.0f,    state::PrimaryInput).label("Arc Bias", "ArcBia")
      .floatField("arc_scale",             0.5f,  0.0f, 1.5f,    state::PrimaryInput).label("Arc Scale", "ArcScl")
      // --- Output (motion field + shadow) ---
      .group("output", "Output")
        .groupHelp(
          "The one blob field feeds both outputs independently. *Motion Strength* scales "
          "the emitted motion vectors (for downstream motion blur) and *Motion Extent* "
          "limits their reach; *Shadow Darkness* + *Shadow Tint* paint the blobs as a "
          "travelling darkening. Set either to 0 to use just the other.")
      .floatField("motion_strength",       1.0f,  0.0f, 2.0f,    state::PrimaryInput).label("Motion Strength", "Motion")
      .floatField("motion_extent",         1.0f,  0.0f, 1.0f,    state::PrimaryInput).label("Motion Extent", "MotExt")
      .floatField("shadow_darkness",       0.0f,  0.0f, 1.0f,    state::PrimaryInput).label("Shadow Darkness", "Shadow")
      .rgbField  ("shadow_tint",           0.0f,  0.0f, 0.0f,    state::PrimaryInput).label("Shadow Tint", "Tint")
      // --- Shape & spawn ---
      .group("shape", "Shape & Spawn")
      .floatField("blob_size",             0.12f, 0.0f, 1.0f,    state::PrimaryInput).label("Blob Size", "Size")
      .selectField("spawn_edge",           EDGE_TOP, state::PrimaryInput,
                   {{"Top", 0}, {"Bottom", 1}, {"Left", 2}, {"Right", 3}}).label("Spawn Edge", "Edge")
      // --- Tuning ---
      .group("tuning", "Tuning")
      .intField  ("blob_count_max",        8, 1, MAX_BLOBS,      state::PrimaryInput).label("Max Blobs", "MaxN")
      .floatField("blob_size_jitter",      0.3f, 0.0f, 1.0f,     state::PrimaryInput).label("Size Jitter", "SzJit")
      .floatField("drift_jitter",          0.1f, 0.0f, 0.5f,     state::PrimaryInput).label("Drift Jitter", "DrfJit")
      .floatField("softness_curve",        4.0f, 1.0f, 16.0f,    state::PrimaryInput).label("Softness Curve", "SoftCv")
      .boolField ("spawn_edge_random",     false,                state::PrimaryInput).label("Random Edge", "RndEdg")
      .intField  ("seed",                  0x82C00, 0, 0x7FFFFFFF, state::PrimaryInput).label("Seed", "Seed")
      // --- Debug ---
      .group("debug", "Debug")
      .boolField ("debug_show_blobs",      false,                state::PrimaryInput).label("Show Blobs", "Blobs")
      // --- I/O ---
      .textureField("tex_in",  state::PrimaryInput)
      .textureField("tex_out", state::PrimaryOutput)
      .renderOutputs(state::PrimaryOutput)
      .renderOutputs(state::PrimaryInput, "render_outputs_in")
        .capability(state::Capability::Generator)
    );

  if (gpu::Device::backend() == gpu::Backend::None) return;

  state::registerShaderSPV("motion_blobs_color", COLOR_SPV, COLOR_SPV_SIZE);
  state::registerShaderSPV("motion_blobs_motion", MOTION_SPV, MOTION_SPV_SIZE,
                           "rgba16float", "write");
  auto cs_color  = gpu::Device::createShaderModuleByName("motion_blobs_color");
  auto cs_motion = gpu::Device::createShaderModuleByName("motion_blobs_motion");
  if (!cs_color || !cs_motion) return;

  s_pso_color = gpu::Device::createComputePSO(cs_color, "main", gpu::Bindings()
      .tex2d(0)
      .storageTex2d(1)
      .uniform(2)
      .storage(3));
  s_pso_motion = gpu::Device::createComputePSO(cs_motion, "main", gpu::Bindings()
      .tex2d(0)
      .storageTex2d(1, gpu::TextureFormat::RGBA16F)
      .uniform(2)
      .storage(3));

  state::log("motion_blobs: module initialized");
}

// Per-instance construction: allocate State + its own GPU buffers.
void* create() {
  auto* st = new State();
  st->blob_buf    = gpu::Device::createBuffer(sizeof(GpuBlob) * MAX_BLOBS, gpu::BufferUsage::Storage);
  st->uniform_buf = gpu::Device::createBuffer(sizeof(Uniforms), gpu::BufferUsage::Uniform);
  return st;
}

void destroy(void* self) {
  auto* st = static_cast<State*>(self);
  if (!st) return;
  st->blob_buf.release();
  st->uniform_buf.release();
  st->motion_tex.release();
  st->zero_motion_tex.release();
  delete st;
}

// Per-instance init tail: reset params/pool + mark ready.
void init(void* self) {
  auto* st = static_cast<State*>(self);
  if (!st) return;
  if (!s_pso_color.valid() || !s_pso_motion.valid()) return;
  if (!st->blob_buf.valid() || !st->uniform_buf.valid()) return;

  st->initialized = false;
  st->motion_w = 0; st->motion_h = 0;
  st->aspect_ready = false;
  for (int i = 0; i < MAX_BLOBS; i++) {
    st->blobs[i].alive = false;
    st->blobs[i].x = st->blobs[i].y = 0.5f;
    st->blobs[i].vx0 = st->blobs[i].vy0 = 0.0f;
    st->blobs[i].vx = st->blobs[i].vy = 0.0f;
    st->blobs[i].radius = 0.0f;
    st->blobs[i].age = 0.0f;
  }
  st->spawn_rng = (uint32_t)st->seed ^ 0xBADBA110u;

  st->initialized = true;
}

void tick(void* self, double dt) {
  auto* s = static_cast<State*>(self);
  if (!s) return;
  if (!s->initialized) return;

  int cap = clampi(s->blob_count_max, 1, MAX_BLOBS);
  int target_alive = (int)std::round(clampf(s->density, 0.0f, 1.0f) * (float)cap);
  if (target_alive < 0) target_alive = 0;

  // 1. Integrate live blobs; cull when they've crossed the opposite side.
  int alive_count = 0;
  float fdt = (float)dt;
  // Onion size from arc_scale (cylinder radius² in cover-square), signed by
  // arc_bias (+ bulge / − pinch); strength = |arc_bias|.
  float arc_str  = s->arc_bias < 0.0f ? -s->arc_bias : s->arc_bias;
  float arc_sign = s->arc_bias < 0.0f ? -1.0f : 1.0f;
  float arc_a2   = arc_sign * s->arc_scale * s->arc_scale;
  for (int i = 0; i < cap; i++) {
    CpuBlob& b = s->blobs[i];
    if (!b.alive) continue;

    // Current velocity = potential-flow field sampled at the blob, curving
    // the path around the center (== free-stream when strength/a2 == 0).
    // Storing it back into vx/vy means the cull's downstream test and the
    // uploaded motion vectors both follow the curved path.
    field_velocity(b.x, b.y, b.vx0, b.vy0, arc_a2, arc_str,
                   s->aspect_x, s->aspect_y, &b.vx, &b.vy);
    b.x += b.vx * fdt;
    b.y += b.vy * fdt;
    b.age += fdt;

    // Cull only when the blob has EXITED past a DOWNSTREAM edge (the side
    // its velocity points toward), one halo-margin clear. Crucially we do
    // NOT cull on the upstream side: the back-projected spawn places a blob
    // far past the opposite margin while it's still approaching (e.g. a
    // leftward flow spawns blobs well to the right of 1+mx), and an
    // all-sides cull would kill them on frame one — which is exactly what
    // left coverage stuck in the downstream corner. BLOB_MAX_AGE is the
    // safety net for a hard inward pinch that would otherwise orbit forever.
    float mx = halo_margin(b.radius, s->aspect_x, s->softness_curve);
    float my = halo_margin(b.radius, s->aspect_y, s->softness_curve);
    bool exited = (b.vx > 0.0f && b.x > 1.0f + mx)
               || (b.vx < 0.0f && b.x < -mx)
               || (b.vy > 0.0f && b.y > 1.0f + my)
               || (b.vy < 0.0f && b.y < -my)
               || (b.age > BLOB_MAX_AGE);
    if (exited) {
      b.alive = false;
      continue;
    }
    alive_count++;
  }
  for (int i = cap; i < MAX_BLOBS; i++) s->blobs[i].alive = false;

  // 2. Respawn dead blobs up to target_alive. Hold off until render() has
  // reported the viewport aspect — spawn placement depends on it, so
  // spawning before then could pop a blob in at the wrong distance.
  if (!s->aspect_ready) return;
  // Cold start (filling 2+ from an empty pool — i.e. init, or density
  // ramped up from 0): seed those scattered across the viewport so the
  // field doesn't march in as one synchronized wave. Steady-state single
  // replacements float in from the edge as usual.
  bool scattered = (alive_count == 0 && target_alive > 1);
  for (int i = 0; i < cap && alive_count < target_alive; i++) {
    CpuBlob& b = s->blobs[i];
    if (b.alive) continue;
    spawn_one(*s, b, scattered);
    alive_count++;
  }
}


void on_state_patched(void* self, int n, const char* pb, const int* off,
                      const int* len, const int* ops) {
  auto* s = static_cast<State*>(self);
  if (!s) return;
  for (int i = 0; i < n; i++) {
    if (ops[i] != state::PatchReplace) continue;
    const char* path = pb + off[i];
    int plen = len[i];

    if      (state::pathIs(path, plen, "density"))               s->density               = state::patchFloat(i);
    else if (state::pathIs(path, plen, "traverse_speed"))        s->traverse_speed        = state::patchFloat(i);
    else if (state::pathIs(path, plen, "traverse_speed_jitter")) s->traverse_speed_jitter = state::patchFloat(i);
    else if (state::pathIs(path, plen, "drift"))                 s->drift                 = state::patchFloat(i);
    else if (state::pathIs(path, plen, "center_bias"))           s->center_bias           = state::patchFloat(i);
    else if (state::pathIs(path, plen, "arc_bias"))              s->arc_bias              = state::patchFloat(i);
    else if (state::pathIs(path, plen, "arc_scale"))             s->arc_scale             = state::patchFloat(i);
    else if (state::pathIs(path, plen, "motion_strength"))       s->motion_strength       = state::patchFloat(i);
    else if (state::pathIs(path, plen, "motion_extent"))         s->motion_extent         = state::patchFloat(i);
    else if (state::pathIs(path, plen, "shadow_darkness"))       s->shadow_darkness       = state::patchFloat(i);
    else if (state::pathIs(path, plen, "shadow_tint")) {
      auto v = state::patchVec3(i);
      s->shadow_tint_r = v.x; s->shadow_tint_g = v.y; s->shadow_tint_b = v.z;
    }
    else if (state::pathIs(path, plen, "blob_size"))             s->blob_size             = state::patchFloat(i);
    else if (state::pathIs(path, plen, "spawn_edge"))            s->spawn_edge            = (int)state::patchFloat(i);
    else if (state::pathIs(path, plen, "blob_count_max"))        s->blob_count_max        = (int)state::patchFloat(i);
    else if (state::pathIs(path, plen, "blob_size_jitter"))      s->blob_size_jitter      = state::patchFloat(i);
    else if (state::pathIs(path, plen, "drift_jitter"))          s->drift_jitter          = state::patchFloat(i);
    else if (state::pathIs(path, plen, "softness_curve"))        s->softness_curve        = state::patchFloat(i);
    else if (state::pathIs(path, plen, "spawn_edge_random"))     s->spawn_edge_random     = state::patchFloat(i) != 0.0f;
    else if (state::pathIs(path, plen, "seed")) {
      int v = (int)state::patchFloat(i);
      if (v != s->seed) {
        s->seed = v;
        s->spawn_rng = (uint32_t)v ^ 0xBADBA110u;
      }
    }
    else if (state::pathIs(path, plen, "debug_show_blobs"))      s->debug_show_blobs      = state::patchFloat(i) != 0.0f;
  }
}

void render(void* self, int vp_w, int vp_h) {
  auto* s = static_cast<State*>(self);
  if (!s) return;
  if (!s->initialized || vp_w <= 0 || vp_h <= 0) return;

  // Cache cover-square aspect for tick()'s spawn placement (runs before
  // textures are resolved so it survives a dangling-input frame).
  auto cs = fx::coverSquare(vp_w, vp_h);
  s->aspect_x = cs.ax;
  s->aspect_y = cs.ay;
  s->aspect_ready = true;

  auto in  = gpu::Device::textureForField("tex_in");
  auto out = gpu::Device::textureForField("tex_out");
  if (!in.valid() || !out.valid()) return;

  // Pack blobs.
  GpuBlob gpu_blobs[MAX_BLOBS] = {};
  int active_count = 0;
  int cap = clampi(s->blob_count_max, 1, MAX_BLOBS);
  for (int i = 0; i < cap; i++) {
    const CpuBlob& b = s->blobs[i];
    if (!b.alive) continue;
    GpuBlob& g = gpu_blobs[active_count++];
    g.x = b.x; g.y = b.y;
    g.vx = b.vx; g.vy = b.vy;
    g.radius = b.radius;
  }
  s->blob_buf.writeBytes(gpu_blobs, (int)sizeof(GpuBlob) * MAX_BLOBS);

  // Uniforms (aspect-aware blob math). cs computed at the top of render().
  Uniforms u = {};
  u.motion_strength = clampf(s->motion_strength, 0.0f, 4.0f);
  u.shadow_darkness = clampf(s->shadow_darkness, 0.0f, 1.0f);
  u.softness_curve  = clampf(s->softness_curve,  0.1f, 64.0f);
  u.motion_extent   = clampf(s->motion_extent,   0.0f, 1.0f);
  u.shadow_r = s->shadow_tint_r;
  u.shadow_g = s->shadow_tint_g;
  u.shadow_b = s->shadow_tint_b;
  u.aspect_x = cs.ax;
  u.aspect_y = cs.ay;
  u.active_count = (uint32_t)active_count;
  u.debug_show_blobs = s->debug_show_blobs ? 1u : 0u;
  s->uniform_buf.writeOne(u);

  // Color pass — always run; with shadow_darkness == 0 it produces
  // tex_in unchanged.
  {
    auto cp = gpu::ComputePass::begin();
    cp.setPSO(s_pso_color);
    cp.setTexture(in,  0, 0);
    cp.setTexture(out, 1, 1);
    cp.setBuffer(s->uniform_buf, 2);
    cp.setBuffer(s->blob_buf,    3);
    cp.dispatch((vp_w + 7) / 8, (vp_h + 7) / 8);
    cp.end();
  }

  // Motion pass — skip when nothing downstream is listening.
  if (state::isOutputConnected("render_outputs")) {
    if (!s->motion_tex.valid() || s->motion_w != vp_w || s->motion_h != vp_h) {
      s->motion_tex = gpu::Device::createTexture(vp_w, vp_h, gpu::TextureFormat::RGBA16F);
      s->motion_w = vp_w;
      s->motion_h = vp_h;
      if (s->motion_tex.valid()) {
        state::setGpuTexture("render_outputs/motion", s->motion_tex.id);
      }
    }
    if (s->motion_tex.valid()) {
      auto upstream = gpu::Device::textureForField("render_outputs_in/motion");
      if (!upstream.valid()) {
        if (!s->zero_motion_tex.valid()) {
          s->zero_motion_tex = gpu::Device::createTexture(1, 1, gpu::TextureFormat::RGBA16F);
        }
        upstream = s->zero_motion_tex;
      }
      if (upstream.valid()) {
        auto cp = gpu::ComputePass::begin();
        cp.setPSO(s_pso_motion);
        cp.setTexture(upstream,      0, 0);
        cp.setTexture(s->motion_tex, 1, 1);
        cp.setBuffer(s->uniform_buf, 2);
        cp.setBuffer(s->blob_buf,    3);
        cp.dispatch((vp_w + 7) / 8, (vp_h + 7) / 8);
        cp.end();
      }
    }
  }

  gpu::Device::submit();
}

} // namespace motion_blobs
