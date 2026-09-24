/*
 * source.mesh.monolith — Deferred, env-lit 3D primitive generator.
 *
 * Renders a simple convex solid — the 1:4:9 "monolith" slab or a regular
 * triangular pyramid (tetrahedron) — over the input, with up to three
 * concentric echo shells, environment reflections, fresnel, screen-space
 * refraction "glass", height/depth fog and god rays. Built for the
 * massive-towering-space-structure look: the default material is a near
 * void-black slab that reads entirely through its specular response.
 *
 * Architecture (the platform has NO z-buffer, so draw order has to be solved
 * closed-form; texture-dependent shading is done in compute here for that
 * reason, NOT because raster can't sample — effect render PSOs declare their
 * bindings vertex+fragment visible on both backends, so a fragment shader may
 * sample textures. If you hit a failure that looks like it can't, check for a
 * compute-authored shader reused in a fragment stage: a compute entry point
 * must use textureSampleLevel, since it has no implicit derivatives):
 *
 *   CPU (per frame): rotate -> camera (vantage/loom) -> project; emit
 *   FRONT faces only into per-copy vertex buffers (true homogeneous clip
 *   coords so world_y/view_z interpolate perspective-correct).
 *   Per copy round (core -> outermost): raster a tiny MRT G-buffer
 *   (normal+coverage, world_y+view_z; beginMRT clears, MRT PSO writes
 *   Replace) -> compute resolve shades covered pixels (fresnel env
 *   reflection from `env_in` [equirect] or tex_in [screen-space
 *   fallback], diffuse, refraction of the background, fog) and
 *   composites into an RGBA16F ping-pong. Then god rays (radial scatter
 *   occluded by the accumulated silhouette) and a final shoulder-tonemap
 *   combine into tex_out.
 *
 * Draw-order correctness is CLOSED FORM, not sorted: front faces of one
 * convex solid never overlap on screen, and rounds composite inner ->
 * outer, which matches the eye-ray hit order for nested convex scaled
 * copies from any exterior viewpoint.
 *
 * Passthrough purity: uncovered pixels are copied verbatim through every
 * pass; at opacity 0 the whole pipeline is skipped for a prefill copy.
 * Motion is accumulator-driven (style guide §2.1); Sync = Free (speed
 * knob) or Bars (host::barPhase() deltas — frozen without a transport).
 */

#include <gpu.h>
#include <host.h>
#include <effect_utils.h>
#include <effect_fast_blur.h>

#include <cmath>

#include "monolith_shaders.h"

namespace monolith {

constexpr float kPi = 3.14159265358979323846f;
constexpr float kTau = 2.0f * kPi;
constexpr float kDeg = kPi / 180.0f;

// --- Selects ---
constexpr int SHAPE_MONOLITH = 0;
constexpr int SHAPE_PYRAMID = 1;
constexpr int MOTION_ARC = 0;
constexpr int MOTION_TUMBLE = 1;
constexpr int MOTION_ARCING = 2;
constexpr int SYNC_FREE = 0;
constexpr int SYNC_BARS = 1;

// ---------------------------------------------------------------------------
// Tiny 3D helpers. View space: x right, y up, z INTO the screen; camera
// looks +z. "World" space = rotated/scaled object space, origin at the
// object center (the camera transform is applied on top of it).
// ---------------------------------------------------------------------------

struct V3 { float x, y, z; };

static inline V3 v3Sub(V3 a, V3 b) { return {a.x - b.x, a.y - b.y, a.z - b.z}; }
static inline float v3Dot(V3 a, V3 b) { return a.x * b.x + a.y * b.y + a.z * b.z; }
static inline V3 v3Cross(V3 a, V3 b) {
  return {a.y * b.z - a.z * b.y, a.z * b.x - a.x * b.z, a.x * b.y - a.y * b.x};
}
static inline V3 v3Norm(V3 a) {
  float len = std::sqrt(v3Dot(a, a));
  if (len < 1e-8f) return {0.0f, 0.0f, 1.0f};
  return {a.x / len, a.y / len, a.z / len};
}

struct M3 { float m[9]; };  // row-major

static inline M3 m3RotX(float a) {
  float c = std::cos(a), s = std::sin(a);
  return {{1, 0, 0,  0, c, -s,  0, s, c}};
}
static inline M3 m3RotY(float a) {
  float c = std::cos(a), s = std::sin(a);
  return {{c, 0, s,  0, 1, 0,  -s, 0, c}};
}
static inline M3 m3Mul(const M3& a, const M3& b) {
  M3 r;
  for (int i = 0; i < 3; i++)
    for (int j = 0; j < 3; j++)
      r.m[i * 3 + j] = a.m[i * 3 + 0] * b.m[0 * 3 + j] +
                       a.m[i * 3 + 1] * b.m[1 * 3 + j] +
                       a.m[i * 3 + 2] * b.m[2 * 3 + j];
  return r;
}
static inline V3 m3Apply(const M3& r, V3 v) {
  return {r.m[0] * v.x + r.m[1] * v.y + r.m[2] * v.z,
          r.m[3] * v.x + r.m[4] * v.y + r.m[5] * v.z,
          r.m[6] * v.x + r.m[7] * v.y + r.m[8] * v.z};
}

// ---------------------------------------------------------------------------
// Geometry tables. Winding normalized programmatically in module_init()
// (right-handed edge cross pointed INWARD) so sign(projected area) ==
// front-facing exactly under perspective.
// ---------------------------------------------------------------------------

// Regular triangular pyramid (tetrahedron), centroid at origin, apex up,
// one base vertex leading toward the camera. Circumradius 0.70.
static const V3 kPyrVerts[4] = {
    {0.000000f, 0.700000f, 0.000000f},    // apex
    {0.571577f, -0.233333f, 0.330000f},   // b0 (30 deg)
    {-0.571577f, -0.233333f, 0.330000f},  // b1 (150 deg)
    {0.000000f, -0.233333f, -0.660000f},  // b2 (leading vertex)
};
static uint8_t kPyrTris[4][3] = {
    {1, 2, 3},              // base
    {0, 1, 2},              // side b0-b1
    {0, 2, 3},              // side b1-b2
    {0, 3, 1},              // side b2-b0
};

// The 1:4:9 monolith (depth : width : height), height-normalized.
constexpr float kMonoHx = 0.575f * 4.0f / 9.0f;  // half-width
constexpr float kMonoHy = 0.575f;                // half-height
constexpr float kMonoHz = 0.575f / 9.0f;         // half-depth (slab faces camera)
static const V3 kMonoVerts[8] = {
    {-kMonoHx, -kMonoHy, -kMonoHz}, {kMonoHx, -kMonoHy, -kMonoHz},
    {kMonoHx, kMonoHy, -kMonoHz},   {-kMonoHx, kMonoHy, -kMonoHz},
    {-kMonoHx, -kMonoHy, kMonoHz},  {kMonoHx, -kMonoHy, kMonoHz},
    {kMonoHx, kMonoHy, kMonoHz},    {-kMonoHx, kMonoHy, kMonoHz},
};
static uint8_t kMonoTris[12][3] = {
    {0, 1, 2}, {0, 2, 3},   // near (toward camera at rest)
    {5, 4, 7}, {5, 7, 6},   // far
    {1, 5, 6}, {1, 6, 2},   // right
    {4, 0, 3}, {4, 3, 7},   // left
    {3, 2, 6}, {3, 6, 7},   // top
    {4, 5, 1}, {4, 1, 0},   // bottom
};

struct ShapeDef {
  const V3* verts;
  int vert_count;
  uint8_t (*tris)[3];
  int tri_count;
};
static const ShapeDef kShapes[2] = {
    {kMonoVerts, 8, kMonoTris, 12},  // SHAPE_MONOLITH
    {kPyrVerts, 4, kPyrTris, 4},     // SHAPE_PYRAMID
};

constexpr int kMaxCopies = 3;
constexpr int kMaxTrisPerCopy = 12;
constexpr int kMaxVertsPerCopy = kMaxTrisPerCopy * 3;   // 36

static void normalizeWinding(const ShapeDef& shape) {
  for (int t = 0; t < shape.tri_count; t++) {
    V3 a = shape.verts[shape.tris[t][0]];
    V3 b = shape.verts[shape.tris[t][1]];
    V3 c = shape.verts[shape.tris[t][2]];
    V3 centroid = {(a.x + b.x + c.x) / 3.0f, (a.y + b.y + c.y) / 3.0f,
                   (a.z + b.z + c.z) / 3.0f};
    V3 m = v3Cross(v3Sub(b, a), v3Sub(c, a));
    if (v3Dot(m, centroid) > 0.0f) {
      uint8_t tmp = shape.tris[t][1];
      shape.tris[t][1] = shape.tris[t][2];
      shape.tris[t][2] = tmp;
    }
  }
}

// ---------------------------------------------------------------------------
// GPU-side structs (16-byte rows).
// ---------------------------------------------------------------------------

struct GbufVertex {         // mirrors Vtx in gbuf_vs.hlsl — 48 B
  float pos[4];             // homogeneous clip: (clip.x*z, clip.y*z, 0.5*z, z)
  float nrm[4];             // view-space outward face normal, w = 1
  float misc[4];            // world_y, view_z, 0, 0
};

struct ResolveUniforms {
  float sun_view[4];        // xyz toward light (view, unit), w = intensity
  float sun_color[4];       // light tint rgb, 0
  float cam[4];             // focal, cover_ax, cover_ay, phi
  float material[4];        // reflect, roughness, refract, opacity
  float color_shade[4];     // rgb, shading
  float fog_p[4];           // fog, fog_y0, inv_fog_h, fog_depth_k
  float round_p[4];         // copy_weight, is_seed, env_mode, fog_z0
  float vp[4];              // w, h, 1/w, 1/h
  float caustic[4];         // amount, world scale, water_t, 0
  float sun_env[4];         // sun sample uv.xy, mode (1 = from env), 0
  float fog_color[4];       // medium tint rgb, w = inverse spatial zoom
};

struct RaysUniforms {
  float sun_screen[4];      // sun px, py, water_t, gain (rays*1.5*fade)
  float march[4];           // taps, decay, max_step_px, caustics amount
  float glow[4];            // inv glow radius (px^-1), sun_color rgb
  float sun_env[4];         // sun sample uv.xy, mode (1 = from env), 0
};

struct FinalUniforms { float p[4]; };     // has_rays, 0, 0, 0

// Type-shared PSOs.
static gpu::ComputePSO s_pso_prefill;
static gpu::RenderPSO s_pso_gbuf;
static gpu::ComputePSO s_pso_resolve;
static gpu::ComputePSO s_pso_rays;
static gpu::ComputePSO s_pso_final;

// Per-instance state.
struct State {
  bool initialized = false;

  // GPU resources.
  gpu::Buffer vtx_bufs[kMaxCopies];
  gpu::Buffer ub_resolve[kMaxCopies];
  gpu::Buffer ub_rays, ub_final;
  gpu::Texture gbufA, gbufB, compA, compB, rays_tex;   // RGBA16F, vp-sized
  gpu::Texture env_blur;                               // SketchDefault, vp-sized
  gpu::Texture zero_tex;                               // 1x1 RGBA16F zeros
  gpu::Sampler samp_clamp, samp_wrap;
  fx::FastBlur blur;
  int scratch_w = 0, scratch_h = 0;

  // Param mirrors.
  int shape = SHAPE_MONOLITH;
  int motion = MOTION_ARC;
  int sync = SYNC_FREE;
  int copies = 1;
  int bars = 4;
  float size = 0.5f;
  float opacity = 1.0f;
  float speed = 0.5f;
  float spread = 0.35f;
  float falloff = 0.6f;
  float color_r = 0.04f, color_g = 0.04f, color_b = 0.05f;
  float reflect_k = 0.5f;
  float roughness = 0.15f;
  float refract_k = 0.35f;
  float azimuth = 160.0f;
  float elevation = 25.0f;
  float sun = 0.6f;
  float sun_r = 1.0f, sun_g = 1.0f, sun_b = 1.0f;
  int sun_source = 0;       // 0 = Color, 1 = From Input (env chroma)
  float fog = 0.2f;
  float fog_r = 1.0f, fog_g = 1.0f, fog_b = 1.0f;
  float fog_scale = 0.0f;   // 0 = 1:1 with the scene, 1 = 4x zoomed sample
  float rays = 0.3f;
  float caustics = 0.35f;
  float caustic_scale = 0.5f;
  float caustic_speed = 0.5f;   // 0 = frozen water; 0.5 = classic rate
  float arc = 0.5f;
  float tilt = 0.0f;        // signed: + looks down at the top face
  float shading = 0.7f;
  float vantage = 0.3f;     // signed: + worm's-eye (camera low, looking up)
  float loom = 0.2f;        // dolly-in + fov widen

  // Accumulators (§2.1) — cycles in [0,1).
  double phase_a = 0.0;
  double phase_b = 0.0;
  double env_phase = 0.0;
  double last_bar_phase = -1.0;
  double water_t = 0.0;     // free-running caustic clock (ignores Sync)
};

static void apply_visibility(State* s) {
  state::setFieldHidden("speed", s->sync != SYNC_FREE);
  state::setFieldHidden("bars", s->sync != SYNC_BARS);
  state::setFieldHidden("arc", s->motion != MOTION_ARC);
}

static void on_state_ready(void* self) {
  auto* s = static_cast<State*>(self);
  if (!s) return;
  apply_visibility(s);
}

void module_init() {
  // 1.2.0: deferred rework — `alpha` renamed to `opacity`, `back_dim`
  // removed (back faces are gone; glass is refraction now), env_in added.
  // 1.2.1: water caustics (rays banding + surface dapple), additive fields.
  // 1.2.2: dapple = projected irradiance (cosine from above, grazing
  //        spill on walls, zero on undersides); rays shimmer top-anchored.
  // 1.2.3: fog is a real medium (blurred/desat/milk-lifted color, smooth
  //        full-body ramp); rays radiance is synthetic surface light, not
  //        gathered composite color.
  // 1.2.4: rays recalibrated - surface glow gets a floor (whole surface
  //        emits; off-screen sun only biases) and the gain restored.
  // 1.2.5: ray caustics are per-shaft density columns (constant along
  //        each shaft, varying across the fan), not a screen overlay.
  // 1.2.6: sun_color tint (light + rays w/ per-column dispersion);
  //        spread goes exponential above 0.5 (shells up to 9x, camera
  //        can end up inside them - per-tri near-plane guard).
  // 1.2.7: sun_source select - light hue sampled from the env/input at
  //        the sun's position (chroma-only, single point, no readback).
  // 1.2.8: sun_source From Tint Tex + tint_in texture input (bound only
  //        when active; the rays pass reuses its sun-sample slot).
  // 1.2.9: fog_color - tints the medium, white = previous look.
  // 1.2.10: fog_scale - spatial zoom on the fog's blurred-scene sample,
  //         decoupling the haze from the backdrop behind each pixel.
  // 1.3.0: bloom REMOVED (field deleted - use downstream post instead).
  // 1.3.1: caustic_speed - water clock rate, exp around the classic 0.35
  //        units/s at 0.5; exactly 0 freezes the surface.
  state::init("source.mesh.monolith", {1, 3, 1},
    state::Schema()
      .helpField("intro",
        "## Monolith\n"
        "A massive 3D structure — the 1:4:9 slab from *2001* or a regular "
        "triangular pyramid — lit by its environment. The default material "
        "is void-black: the shape reads through fresnel reflections of the "
        "input (or a wired *Env* texture), a sun glint sliding along its "
        "edges, haze swallowing its top, and god rays bleeding around its "
        "silhouette.\n\n"
        "**Try:** *Vantage* up with *Loom* for a worm's-eye tower; wire a "
        "starfield into *Env*; *Opacity* low with *Refract* high for dark "
        "glass; *Azimuth* near ±180 puts the sun behind the shape — full "
        "eclipse mode with the rays carving around it.")
      // --- Shape ---
      .group("shape", "Shape")
      .selectField("shape", SHAPE_MONOLITH, state::PrimaryInput,
                   {{"Monolith", SHAPE_MONOLITH}, {"Pyramid", SHAPE_PYRAMID}})
          .label("Shape", "Shape")
      .floatField("size", 0.5f, 0.f, 1.f, state::PrimaryInput)
          .label("Size", "Size")
      // --- Material ---
      .group("material", "Material")
        .groupHelp(
          "*Color* is the diffuse body — black is the intended default; "
          "the shape still reads through its reflections. *Opacity* runs "
          "solid (1) to clear glass (0): glass shows the scene refracted "
          "through the surface (*Refract* sets how strongly it bends). "
          "*Reflect* scales the fresnel environment reflection and "
          "*Roughness* blurs it — polished obsidian at 0, brushed metal "
          "up high.")
      .rgbField("color", 0.04f, 0.04f, 0.05f, state::PrimaryInput)
          .label("Color", "Color")
      .floatField("opacity", 1.0f, 0.f, 1.f, state::PrimaryInput)
          .label("Opacity", "Opac")
      .floatField("reflect", 0.5f, 0.f, 1.f, state::PrimaryInput)
          .label("Reflect", "Refl")
      .floatField("roughness", 0.15f, 0.f, 1.f, state::PrimaryInput)
          .label("Roughness", "Rough")
      .floatField("refract", 0.35f, 0.f, 1.f, state::PrimaryInput)
          .label("Refract", "Refr")
      // --- Motion ---
      .group("motion", "Motion")
        .groupHelp(
          "*Arc* sweeps the shape one way through a yaw arc — easing in "
          "and out of the sweep — then snaps back to the start instantly. "
          "*Tumble* is constant angular momentum about two incommensurate "
          "axes — the corners trace curves that never repeat. *Arcing "
          "Tumble* is the same tumble but its speed swells and relaxes, so "
          "it seems to arc through its curves. *Free* runs on the Speed "
          "knob; *Bars* locks one motion cycle to N bars of the host "
          "transport (and freezes when no transport is running) — with "
          "Arc, the snap-back lands exactly on the bar line.")
      .selectField("motion", MOTION_ARC, state::PrimaryInput,
                   {{"Arc", MOTION_ARC},
                    {"Tumble", MOTION_TUMBLE},
                    {"Arcing Tumble", MOTION_ARCING}})
          .label("Motion", "Motion")
      .selectField("sync", SYNC_FREE, state::PrimaryInput,
                   {{"Free", SYNC_FREE}, {"Bars", SYNC_BARS}})
          .label("Sync", "Sync")
      .floatField("speed", 0.5f, 0.f, 1.f, state::PrimaryInput)
          .label("Speed", "Speed")
      .selectField("bars", 4, state::PrimaryInput,
                   {{"1", 1}, {"2", 2}, {"4", 4}, {"8", 8}, {"16", 16}})
          .label("Bars / Cycle", "Bars")
      // --- Concentric copies ---
      .group("copies", "Concentric Copies")
        .groupHelp(
          "Echo shells of the same solid at growing scale and fading "
          "presence, all sharing the core's rotation. Outer shells "
          "refract and reflect over the ones inside them. *Spread* is "
          "exponential in its top half — cranked, the outer shells grow "
          "colossal and can swallow the camera entirely (they fade out "
          "gracefully as you end up inside them).")
      .intField("copies", 1, 1, kMaxCopies, state::PrimaryInput)
          .label("Copies", "Copies")
      .floatField("spread", 0.35f, 0.f, 1.f, state::PrimaryInput)
          .label("Spread", "Spread")
      .floatField("falloff", 0.6f, 0.f, 1.f, state::PrimaryInput)
          .label("Alpha Falloff", "Falloff")
      // --- Light ---
      .group("light", "Light")
        .groupHelp(
          "One sun drives everything coherently: diffuse shading, the "
          "specular glint, the god rays and their tint. *Azimuth* 0 "
          "lights from the camera; ±180 puts the sun BEHIND the shape — "
          "that's where the rays live. *Sun* is overall light intensity; "
          "*Sun Color* tints all of it (the ray columns also pick up a "
          "subtle per-column warm/cool dispersion around it). *Sun "
          "Source* → From Input steals the light's HUE from the env (or "
          "the input) sampled around the sun's position — intensity stays "
          "on the Sun knob, and Sun Color still filters on top. From Tint "
          "Tex reads the hue from the dedicated *Tint* texture input "
          "instead (falls back to From Input when nothing is wired) — "
          "feed it a palette strip or a slow color wash.")
      .floatField("azimuth", 160.0f, -180.f, 180.f, state::PrimaryInput,
                  nullptr, 0.f, "deg").label("Azimuth", "Azim")
      .floatField("elevation", 25.0f, -10.f, 80.f, state::PrimaryInput,
                  nullptr, 0.f, "deg").label("Elevation", "Elev")
      .floatField("sun", 0.6f, 0.f, 1.f, state::PrimaryInput)
          .label("Sun", "Sun")
      .rgbField("sun_color", 1.0f, 1.0f, 1.0f, state::PrimaryInput)
          .label("Sun Color", "SunCol")
      .selectField("sun_source", 0, state::PrimaryInput,
                   {{"Color", 0}, {"From Input", 1}, {"From Tint Tex", 2}})
          .label("Sun Source", "SunSrc")
      // --- Atmosphere ---
      .group("atmosphere", "Atmosphere")
        .groupHelp(
          "*Fog* melts the structure's top into haze and thickens with "
          "distance — the scale cue; *Fog Color* tints the medium (the "
          "haze still inherits its brightness from the scene). *Rays* scatter the bright environment "
          "radially from the sun, carved by the silhouette (needs the sun "
          "in front of the camera: azimuth toward ±180). *Caustics* folds "
          "the light through a moving water surface: the shafts band and "
          "flicker by angle, and the structure's lit faces catch drifting "
          "dapple webs — the underwater look. The water clock is free-"
          "running (it ignores Sync).")
      .floatField("fog", 0.2f, 0.f, 1.f, state::PrimaryInput)
          .label("Fog", "Fog")
      .rgbField("fog_color", 1.0f, 1.0f, 1.0f, state::PrimaryInput)
          .label("Fog Color", "FogCol")
      .floatField("rays", 0.3f, 0.f, 1.f, state::PrimaryInput)
          .label("God Rays", "Rays")
      .floatField("caustics", 0.35f, 0.f, 1.f, state::PrimaryInput)
          .label("Caustics", "Caust")
      // --- Tuning ---
      .group("tuning", "Tuning")
      .floatField("arc", 0.5f, 0.f, 1.f, state::SecondaryInput)
          .label("Arc Width", "Arc")
      .floatField("tilt", 0.0f, -1.f, 1.f, state::SecondaryInput)
          .label("Tilt", "Tilt")
      .floatField("shading", 0.7f, 0.f, 1.f, state::SecondaryInput)
          .label("Shading", "Shade")
      .floatField("vantage", 0.3f, -1.f, 1.f, state::SecondaryInput)
          .label("Vantage", "Vant")
      .floatField("loom", 0.2f, 0.f, 1.f, state::SecondaryInput)
          .label("Loom", "Loom")
      .floatField("caustic_scale", 0.5f, 0.f, 1.f, state::SecondaryInput)
          .label("Caustic Scale", "CScl")
      .floatField("caustic_speed", 0.5f, 0.f, 1.f, state::SecondaryInput)
          .label("Caustic Speed", "CSpd")
      .floatField("fog_scale", 0.0f, 0.f, 1.f, state::SecondaryInput)
          .label("Fog Scale", "FogScl")
      // --- I/O ---
      .textureField("tex_in", state::PrimaryInput)
      .textureField("env_in", state::SecondaryInput)
      .textureField("tint_in", state::SecondaryInput)
      .textureField("tex_out", state::PrimaryOutput)
      .capability(state::Capability::Generator)
      .capability(state::Capability::SeekableApproximate)
  );

  normalizeWinding(kShapes[0]);
  normalizeWinding(kShapes[1]);

  if (gpu::Device::backend() == gpu::Backend::None) return;

  state::registerShaderSPV("monolith_prefill", PREFILL_SPV, PREFILL_SPV_SIZE);
  state::registerShaderSPV("monolith_gbuf_vs", GBUF_VS_SPV, GBUF_VS_SPV_SIZE);
  state::registerShaderSPV("monolith_gbuf_fs", GBUF_FS_SPV, GBUF_FS_SPV_SIZE);
  state::registerShaderSPV("monolith_resolve", RESOLVE_SPV, RESOLVE_SPV_SIZE,
                           "rgba16float", "write");
  state::registerShaderSPV("monolith_rays", RAYS_SPV, RAYS_SPV_SIZE,
                           "rgba16float", "write");
  state::registerShaderSPV("monolith_final", FINAL_SPV, FINAL_SPV_SIZE);

  auto cs_prefill = gpu::Device::createShaderModuleByName("monolith_prefill");
  auto vs_gbuf = gpu::Device::createShaderModuleByName("monolith_gbuf_vs");
  auto fs_gbuf = gpu::Device::createShaderModuleByName("monolith_gbuf_fs");
  auto cs_resolve = gpu::Device::createShaderModuleByName("monolith_resolve");
  auto cs_rays = gpu::Device::createShaderModuleByName("monolith_rays");
  auto cs_final = gpu::Device::createShaderModuleByName("monolith_final");
  if (!cs_prefill || !vs_gbuf || !fs_gbuf || !cs_resolve || !cs_rays ||
      !cs_final) return;

  s_pso_prefill = gpu::Device::createComputePSO(cs_prefill, "main", gpu::Bindings()
      .tex2d(0)
      .storageTex2d(1));
  s_pso_gbuf = gpu::Device::createInstancedRenderPSOMRT(
      vs_gbuf, "main", fs_gbuf, "main",
      {gpu::TextureFormat::RGBA16F, gpu::TextureFormat::RGBA16F},
      gpu::Bindings().storage(0));
  s_pso_resolve = gpu::Device::createComputePSO(cs_resolve, "main", gpu::Bindings()
      .tex2d(0)   // gbufA
      .tex2d(1)   // gbufB
      .tex2d(2)   // bg (tex_in or comp[prev])
      .tex2d(3)   // env sharp
      .tex2d(4)   // env blurred (or sharp again)
      .storageTex2d(5, gpu::TextureFormat::RGBA16F)
      .sampler(6)
      .sampler(7)
      .uniform(8)
      .tex2d(9));   // tint_in (sun-color sample; 1x1 zero when inactive)
  s_pso_rays = gpu::Device::createComputePSO(cs_rays, "main", gpu::Bindings()
      .tex2d(0)
      .storageTex2d(1, gpu::TextureFormat::RGBA16F)
      .sampler(2)
      .uniform(3)
      .tex2d(4));   // env source (sun-color sample)
  s_pso_final = gpu::Device::createComputePSO(cs_final, "main", gpu::Bindings()
      .tex2d(0)
      .tex2d(1)
      .tex2d(2)
      .storageTex2d(3)
      .uniform(4));

  state::log("monolith: module initialized (deferred)");
}

void* create() {
  auto* s = new State();
  for (int i = 0; i < kMaxCopies; i++) {
    s->vtx_bufs[i] = gpu::Device::createBuffer(
        sizeof(GbufVertex) * kMaxVertsPerCopy, gpu::BufferUsage::Storage);
    s->ub_resolve[i] = gpu::Device::createBuffer(
        sizeof(ResolveUniforms), gpu::BufferUsage::Uniform);
  }
  s->ub_rays = gpu::Device::createBuffer(sizeof(RaysUniforms), gpu::BufferUsage::Uniform);
  s->ub_final = gpu::Device::createBuffer(sizeof(FinalUniforms), gpu::BufferUsage::Uniform);
  s->zero_tex = gpu::Device::createTexture(1, 1, gpu::TextureFormat::RGBA16F);
  s->samp_clamp = gpu::Device::createSampler(gpu::FilterMode::Linear,
                                             gpu::AddressMode::ClampToEdge);
  s->samp_wrap = gpu::Device::createSampler(gpu::FilterMode::Linear,
                                            gpu::AddressMode::Repeat);
  return s;
}

void destroy(void* self) {
  auto* s = static_cast<State*>(self);
  if (!s) return;
  for (int i = 0; i < kMaxCopies; i++) {
    s->vtx_bufs[i].release();
    s->ub_resolve[i].release();
  }
  s->ub_rays.release();
  s->ub_final.release();
  s->gbufA.release();
  s->gbufB.release();
  s->compA.release();
  s->compB.release();
  s->rays_tex.release();
  s->env_blur.release();
  s->zero_tex.release();
  s->samp_clamp.release();
  s->samp_wrap.release();
  delete s;
}

void init(void* self) {
  auto* s = static_cast<State*>(self);
  if (!s) return;
  s->phase_a = 0.0;
  s->phase_b = 0.0;
  s->env_phase = 0.0;
  s->last_bar_phase = -1.0;
  s->water_t = 0.0;
  state::setOnStateReady(&on_state_ready);
  s->blur.init();
  bool bufs_ok = true;
  for (int i = 0; i < kMaxCopies; i++) {
    bufs_ok = bufs_ok && s->vtx_bufs[i].valid() && s->ub_resolve[i].valid();
  }
  s->initialized = bufs_ok && s_pso_prefill.valid() && s_pso_gbuf.valid() &&
                   s_pso_resolve.valid() && s_pso_final.valid();
}

static inline double wrap01(double v) { return v - std::floor(v); }

// Speed knob -> cycles/sec, exponential: ~0.014 at 0, 0.08 centered, ~0.45 at 1.
static inline double rateHz(float speed) {
  return 0.08 * std::pow(2.0, ((double)speed - 0.5) * 5.0);
}

void tick(void* self, double dt) {
  auto* s = static_cast<State*>(self);
  if (!s) return;
  if (!(dt > 0.0)) dt = 0.0;
  if (dt > 0.050) dt = 0.050;   // stall guard: never jump the pose

  // The water runs on its own clock (even in Bars sync with a halted
  // transport). Accumulator: rate changes never jump the pattern (§2.1).
  // Exponential around the classic 0.35 units/s; exactly 0 freezes it.
  if (s->caustic_speed > 0.001f) {
    s->water_t += dt * 0.35 * std::pow(2.0, ((double)s->caustic_speed - 0.5) * 4.0);
    if (s->water_t > 512.0) s->water_t -= 512.0;
  }

  double d;   // cycles this frame
  if (s->sync == SYNC_FREE) {
    d = dt * rateHz(s->speed);
  } else {
    // Bars: consume barPhase deltas (§2.2). Frozen without a transport.
    double bp = host::barPhase();
    if (s->last_bar_phase < 0.0) { s->last_bar_phase = bp; return; }
    double dphase = bp - s->last_bar_phase;
    if (dphase < -0.5) dphase += 1.0;   // bar wrap
    if (dphase < 0.0) dphase = 0.0;     // scrub-back: hold rather than reverse
    s->last_bar_phase = bp;
    d = dphase / (double)(s->bars > 0 ? s->bars : 1);
  }

  double mult = 1.0;
  if (s->motion == MOTION_ARCING) {
    s->env_phase = wrap01(s->env_phase + d / 3.0);
    mult = 0.25 + 1.5 * (0.5 - 0.5 * std::cos(kTau * s->env_phase));
  }
  // Both phases always advance so switching motion modes never pops.
  s->phase_a = wrap01(s->phase_a + d * mult);
  s->phase_b = wrap01(s->phase_b + d * mult * 0.6180339887);
}

void on_state_patched(void* self, int n, const char* pb, const int* off,
                      const int* len, const int* ops) {
  auto* s = static_cast<State*>(self);
  if (!s) return;
  bool vis_dirty = false;
  for (int i = 0; i < n; i++) {
    if (ops[i] != state::PatchReplace) continue;
    const char* p = pb + off[i];
    int l = len[i];
    if      (state::pathIs(p, l, "shape"))     s->shape = state::patchInt(i);
    else if (state::pathIs(p, l, "size"))      s->size = state::patchFloat(i);
    else if (state::pathIs(p, l, "color")) {
      auto v = state::patchVec3(i);
      s->color_r = v.x; s->color_g = v.y; s->color_b = v.z;
    }
    else if (state::pathIs(p, l, "opacity"))   s->opacity = state::patchFloat(i);
    else if (state::pathIs(p, l, "reflect"))   s->reflect_k = state::patchFloat(i);
    else if (state::pathIs(p, l, "roughness")) s->roughness = state::patchFloat(i);
    else if (state::pathIs(p, l, "refract"))   s->refract_k = state::patchFloat(i);
    else if (state::pathIs(p, l, "motion")) {
      int m = state::patchInt(i);
      if (m != s->motion) { s->motion = m; vis_dirty = true; }
    }
    else if (state::pathIs(p, l, "sync")) {
      int v = state::patchInt(i);
      if (v != s->sync) {
        s->sync = v;
        s->last_bar_phase = -1.0;   // re-seed the bar tracker on re-entry
        vis_dirty = true;
      }
    }
    else if (state::pathIs(p, l, "speed"))     s->speed = state::patchFloat(i);
    else if (state::pathIs(p, l, "bars"))      s->bars = state::patchInt(i);
    else if (state::pathIs(p, l, "copies"))    s->copies = state::patchInt(i);
    else if (state::pathIs(p, l, "spread"))    s->spread = state::patchFloat(i);
    else if (state::pathIs(p, l, "falloff"))   s->falloff = state::patchFloat(i);
    else if (state::pathIs(p, l, "azimuth"))   s->azimuth = state::patchFloat(i);
    else if (state::pathIs(p, l, "elevation")) s->elevation = state::patchFloat(i);
    else if (state::pathIs(p, l, "sun"))       s->sun = state::patchFloat(i);
    else if (state::pathIs(p, l, "sun_color")) {
      auto v = state::patchVec3(i);
      s->sun_r = v.x; s->sun_g = v.y; s->sun_b = v.z;
    }
    else if (state::pathIs(p, l, "sun_source")) s->sun_source = state::patchInt(i);
    else if (state::pathIs(p, l, "fog"))       s->fog = state::patchFloat(i);
    else if (state::pathIs(p, l, "fog_color")) {
      auto v = state::patchVec3(i);
      s->fog_r = v.x; s->fog_g = v.y; s->fog_b = v.z;
    }
    else if (state::pathIs(p, l, "fog_scale")) s->fog_scale = state::patchFloat(i);
    else if (state::pathIs(p, l, "rays"))      s->rays = state::patchFloat(i);
    else if (state::pathIs(p, l, "caustics"))  s->caustics = state::patchFloat(i);
    else if (state::pathIs(p, l, "caustic_scale")) s->caustic_scale = state::patchFloat(i);
    else if (state::pathIs(p, l, "caustic_speed")) s->caustic_speed = state::patchFloat(i);
    else if (state::pathIs(p, l, "arc"))       s->arc = state::patchFloat(i);
    else if (state::pathIs(p, l, "tilt"))      s->tilt = state::patchFloat(i);
    else if (state::pathIs(p, l, "shading"))   s->shading = state::patchFloat(i);
    else if (state::pathIs(p, l, "vantage"))   s->vantage = state::patchFloat(i);
    else if (state::pathIs(p, l, "loom"))      s->loom = state::patchFloat(i);
  }
  if (vis_dirty) apply_visibility(s);
}

static gpu::Texture ensureTex(gpu::Texture& t, int w, int h,
                              gpu::TextureFormat fmt, bool size_changed) {
  if (!t.valid() || size_changed) {
    t.release();
    t = gpu::Device::createTexture(w, h, fmt);
  }
  return t;
}

void render(void* self, int vp_w, int vp_h) {
  auto* s = static_cast<State*>(self);
  if (!s || !s->initialized || vp_w <= 0 || vp_h <= 0) return;

  auto in = gpu::Device::textureForField("tex_in");
  auto out = gpu::Device::textureForField("tex_out");
  if (!out.valid()) return;
  auto env_field = gpu::Device::textureForField("env_in");
  const bool env_wired = env_field.valid();
  gpu::Texture env_src = env_wired ? env_field : in;

  const float vis_fade = std::fmin(1.0f, std::fmax(0.0f, s->opacity / 0.1f));

  // --- Idle: pure passthrough, bit-exact, regardless of atmosphere ---
  if (vis_fade < 1.0f / 255.0f || !in.valid()) {
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

  // --- Pose ---
  const float pitch = (-s->tilt * 40.0f) * kDeg;
  M3 rot;
  if (s->motion == MOTION_ARC) {
    // One-directional eased sweep, instant snap back on phase wrap.
    const float arc_half = (20.0f + 140.0f * s->arc) * kDeg;
    const float yaw = arc_half * -std::cos(kPi * (float)s->phase_a);
    rot = m3Mul(m3RotY(yaw), m3RotX(pitch));
  } else {
    rot = m3Mul(m3RotY(kTau * (float)s->phase_a),
                m3RotX(kTau * (float)s->phase_b + pitch));
  }

  // --- Camera: loom (fov/dolly) + vantage (low camera, pitched up) ---
  const ShapeDef& shape = kShapes[s->shape == SHAPE_PYRAMID ? 1 : 0];
  const auto cs = fx::coverSquare(vp_w, vp_h);
  const float size_world = 0.55f * std::pow(2.0f, (s->size - 0.5f) * 2.0f);
  // Spread: linear below the midpoint (matches the classic look), then
  // exponential — cranked, shells reach ~4x steps (9x outer scale) and
  // can grow right past the camera.
  const float spread_step = (0.15f + 0.85f * s->spread) *
      std::pow(2.0f, std::fmax(0.0f, s->spread - 0.5f) * 4.0f);
  const float falloff_mult = 1.0f - 0.8f * s->falloff;
  const int copies = s->copies < 1 ? 1 : (s->copies > kMaxCopies ? kMaxCopies : s->copies);
  const float outer_scale = size_world * (1.0f + (float)(copies - 1) * spread_step);

  const float fov = (30.0f + 45.0f * s->loom) * kDeg;
  const float focal = 1.0f / std::tan(fov * 0.5f);
  // Dolly holds framing; the CORE stays framed even when giant shells
  // engulf the camera (a shell you are inside shows only back faces,
  // which are culled — it fades out via the per-tri near guard below).
  float cam_d = std::fmax(3.0f * focal / 3.7320508f,
                          size_world * 0.72f + 0.4f);
  const float cam_y = -s->vantage * 1.5f * std::fmax(0.8f, size_world);
  const float phi = std::atan2(-cam_y, cam_d);          // re-center the object
  const M3 cam_rot = m3RotX(phi);

  // --- Sun (world = rotated-object space; azimuth 0 lights from camera) ---
  const float az = s->azimuth * kDeg, el = s->elevation * kDeg;
  const V3 sun_world = {std::sin(az) * std::cos(el), std::sin(el),
                        -std::cos(az) * std::cos(el)};
  const V3 sun_view = v3Norm(m3Apply(cam_rot, sun_world));
  const float sun_i = std::pow(2.0f, (s->sun - 0.5f) * 3.0f);

  // Sun screen position (shared by the rays march and the env sun-color
  // sample; may be off-frame) + the single sample point for Sun Source =
  // From Input: the sun's equirect coordinate on env_in, or its clamped
  // screen position on tex_in.
  const float sun_pz = std::fmax(sun_view.z, 0.05f);
  const float sun_px = ((focal * sun_view.x / sun_pz) * 2.0f * cs.ax + 1.0f)
                       * 0.5f * vp_w;
  const float sun_py = (1.0f - ((focal * sun_view.y / sun_pz) * 2.0f * cs.ay
                                + 1.0f) * 0.5f) * vp_h;
  // Effective mode: From Tint Tex degrades to From Input when nothing is
  // wired. The tint texture is sampled at the sun's SCREEN position;
  // From Input samples the env (equirect when wired, screen otherwise).
  auto tint = gpu::Device::textureForField("tint_in");
  int sun_mode = s->sun_source;
  if (sun_mode == 2 && !tint.valid()) sun_mode = 1;
  auto clampf = [](float v, float lo, float hi) {
    return v < lo ? lo : (v > hi ? hi : v);
  };
  float sun_uv_x, sun_uv_y;
  if (sun_mode == 1 && env_wired) {
    sun_uv_x = std::atan2(sun_world.x, sun_world.z) * 0.15915494f + 0.5f;
    float sy = sun_world.y < -1.f ? -1.f : (sun_world.y > 1.f ? 1.f : sun_world.y);
    sun_uv_y = std::acos(sy) * 0.31830989f;
    sun_uv_y = sun_uv_y < 0.004f ? 0.004f : (sun_uv_y > 0.996f ? 0.996f : sun_uv_y);
  } else {
    sun_uv_x = clampf(sun_px / (float)vp_w, 0.03f, 0.97f);
    sun_uv_y = clampf(sun_py / (float)vp_h, 0.03f, 0.97f);
  }
  const float sun_env_mode = (float)sun_mode;

  // --- Scratch textures (lazy, vp-sized) ---
  const bool resized = (s->scratch_w != vp_w || s->scratch_h != vp_h);
  ensureTex(s->gbufA, vp_w, vp_h, gpu::TextureFormat::RGBA16F, resized);
  ensureTex(s->gbufB, vp_w, vp_h, gpu::TextureFormat::RGBA16F, resized);
  ensureTex(s->compA, vp_w, vp_h, gpu::TextureFormat::RGBA16F, resized);
  ensureTex(s->compB, vp_w, vp_h, gpu::TextureFormat::RGBA16F, resized);
  s->scratch_w = vp_w;
  s->scratch_h = vp_h;
  if (!s->gbufA.valid() || !s->gbufB.valid() || !s->compA.valid() || !s->compB.valid())
    return;

  // --- Env pre-blur: roughness reflections AND the fog medium color.
  // Fog must never sample a sharp backdrop (that reads as transparency),
  // so fog alone forces a heavy blur even at roughness 0.
  gpu::Texture env_blurred = env_src;
  const bool want_blur = s->roughness > 0.01f || s->fog > 0.01f;
  if (want_blur && s->blur.valid()) {
    ensureTex(s->env_blur, vp_w, vp_h, gpu::Device::defaultTextureFormat(), resized);
    if (s->env_blur.valid()) {
      int iters = s->roughness > 0.01f ? 1 + (int)(s->roughness * 4.0f) : 0;
      if (s->fog > 0.01f && iters < 4) iters = 4;
      s->blur.apply(env_src, s->env_blur, vp_w, vp_h, iters);
      env_blurred = s->env_blur;
    }
  }

  // --- Per-copy rounds: raster G-buffer + resolve, inner -> outer ---
  // Height-fog ramp: smoothstep spanning past both ends of the body
  // (bottom ~1/4 strength, top ~0.9) — a gradient, not a plateau+knee.
  const float fog_h = std::fmax(0.62f * outer_scale, 1e-3f);   // ~apex height
  const float fog_y0 = -0.45f * fog_h;
  const float fog_inv_h = 1.0f / (1.6f * fog_h);
  const float fog_z0 = cam_d - 0.6f * outer_scale;             // haze starts before the body
  gpu::Texture comp_prev;   // invalid on round 0
  gpu::Texture comp_next = s->compA;
  int rounds_done = 0;

  for (int ci = 0; ci < copies; ci++) {
    const float scale = size_world * (1.0f + (float)ci * spread_step);
    const float w_i = vis_fade * std::pow(falloff_mult, (float)ci);
    if (w_i < 1.0f / 255.0f) continue;

    // Transform + project this copy's vertex ring. Verts at/behind the
    // near plane are flagged rather than failing the whole copy — a
    // giant shell straddling the camera keeps its valid triangles.
    V3 world[8], view[8];
    float sq[8][2];
    bool valid[8];
    for (int vi = 0; vi < shape.vert_count; vi++) {
      V3 p = shape.verts[vi];
      p = {p.x * scale, p.y * scale, p.z * scale};
      p = m3Apply(rot, p);
      world[vi] = p;
      V3 pv = m3Apply(cam_rot, V3{p.x, p.y - cam_y, p.z + cam_d});
      view[vi] = pv;
      valid[vi] = pv.z >= 0.1f;
      if (valid[vi]) {
        sq[vi][0] = focal * pv.x / pv.z;
        sq[vi][1] = focal * pv.y / pv.z;
      } else {
        sq[vi][0] = 0.0f;
        sq[vi][1] = 0.0f;
      }
    }

    GbufVertex verts[kMaxVertsPerCopy];
    int tri_count = 0;
    for (int t = 0; t < shape.tri_count; t++) {
      const uint8_t* idx = shape.tris[t];
      if (!valid[idx[0]] || !valid[idx[1]] || !valid[idx[2]]) continue;
      const float ax0 = sq[idx[0]][0], ay0 = sq[idx[0]][1];
      const float bx = sq[idx[1]][0], by = sq[idx[1]][1];
      const float cx = sq[idx[2]][0], cy = sq[idx[2]][1];
      const float area2 = (bx - ax0) * (cy - ay0) - (by - ay0) * (cx - ax0);
      if (area2 <= 1e-7f) continue;   // back face or sliver: dropped

      const V3 va = view[idx[0]], vb = view[idx[1]], vc = view[idx[2]];
      // Inward-cross winding => outward normal is the NEGATED cross.
      V3 nrm = v3Norm(v3Cross(v3Sub(vb, va), v3Sub(vc, va)));
      nrm = {-nrm.x, -nrm.y, -nrm.z};

      for (int k = 0; k < 3; k++) {
        GbufVertex& v = verts[tri_count * 3 + k];
        const float z = view[idx[k]].z;
        v.pos[0] = sq[idx[k]][0] * 2.0f * cs.ax * z;
        v.pos[1] = -sq[idx[k]][1] * 2.0f * cs.ay * z;   // THE single y-flip
        v.pos[2] = 0.5f * z;
        v.pos[3] = z;
        v.nrm[0] = nrm.x; v.nrm[1] = nrm.y; v.nrm[2] = nrm.z; v.nrm[3] = 1.0f;
        v.misc[0] = world[idx[k]].y;
        v.misc[1] = z;
        v.misc[2] = world[idx[k]].x;   // caustic dapple plane
        v.misc[3] = world[idx[k]].z;
      }
      tri_count++;
    }
    if (tri_count == 0) continue;
    s->vtx_bufs[ci].write<GbufVertex>(verts, tri_count * 3);

    // Raster the round's G-buffer (cleared; Replace writes).
    {
      auto rp = gpu::RenderPass::beginMRT({
          {s->gbufA, 0.0f, 0.0f, 0.0f, 0.0f},
          {s->gbufB, 0.0f, 0.0f, 0.0f, 0.0f},
      });
      rp.setPSO(s_pso_gbuf);
      rp.setBuffer(s->vtx_bufs[ci], 0);
      rp.draw(tri_count * 3, 1);
      rp.end();
    }

    // Resolve: shade + composite into the ping-pong.
    ResolveUniforms u = {};
    u.sun_view[0] = sun_view.x; u.sun_view[1] = sun_view.y;
    u.sun_view[2] = sun_view.z; u.sun_view[3] = sun_i;
    u.sun_color[0] = s->sun_r; u.sun_color[1] = s->sun_g;
    u.sun_color[2] = s->sun_b;
    u.cam[0] = focal; u.cam[1] = cs.ax; u.cam[2] = cs.ay; u.cam[3] = phi;
    u.material[0] = s->reflect_k; u.material[1] = s->roughness;
    u.material[2] = s->refract_k; u.material[3] = s->opacity;
    u.color_shade[0] = s->color_r; u.color_shade[1] = s->color_g;
    u.color_shade[2] = s->color_b; u.color_shade[3] = s->shading;
    u.fog_p[0] = s->fog; u.fog_p[1] = fog_y0;
    u.fog_p[2] = fog_inv_h; u.fog_p[3] = 0.45f / size_world;
    u.round_p[0] = w_i; u.round_p[1] = rounds_done == 0 ? 1.0f : 0.0f;
    u.round_p[2] = env_wired ? 1.0f : 0.0f; u.round_p[3] = fog_z0;
    u.vp[0] = (float)vp_w; u.vp[1] = (float)vp_h;
    u.vp[2] = 1.0f / vp_w; u.vp[3] = 1.0f / vp_h;
    u.caustic[0] = s->caustics;
    u.caustic[1] = 9.0f * std::pow(2.0f, (s->caustic_scale - 0.5f) * 2.0f);
    u.caustic[2] = (float)s->water_t;
    u.sun_env[0] = sun_uv_x; u.sun_env[1] = sun_uv_y;
    u.sun_env[2] = sun_env_mode;
    u.fog_color[0] = s->fog_r; u.fog_color[1] = s->fog_g;
    u.fog_color[2] = s->fog_b;
    u.fog_color[3] = 1.0f / (1.0f + 3.0f * s->fog_scale);   // inverse zoom
    s->ub_resolve[ci].writeOne(u);

    {
      auto cp = gpu::ComputePass::begin();
      cp.setPSO(s_pso_resolve);
      cp.setTexture(s->gbufA, 0, 0);
      cp.setTexture(s->gbufB, 1, 0);
      cp.setTexture(rounds_done == 0 ? in : comp_prev, 2, 0);
      cp.setTexture(env_src, 3, 0);
      cp.setTexture(env_blurred, 4, 0);
      cp.setTexture(comp_next, 5, 1);
      cp.setSampler(s->samp_clamp, 6);
      cp.setSampler(s->samp_wrap, 7);
      cp.setBuffer(s->ub_resolve[ci], 8);
      cp.setTexture(sun_mode == 2 ? tint : s->zero_tex, 9, 0);
      cp.dispatch((vp_w + 7) / 8, (vp_h + 7) / 8);
      cp.end();
    }

    comp_prev = comp_next;
    comp_next = (comp_prev.id == s->compA.id) ? s->compB : s->compA;
    rounds_done++;
  }

  if (rounds_done == 0) {
    // Everything skipped (degenerate camera): passthrough.
    auto cp = gpu::ComputePass::begin();
    cp.setPSO(s_pso_prefill);
    cp.setTexture(in, 0, 0);
    cp.setTexture(out, 1, 1);
    cp.dispatch((vp_w + 7) / 8, (vp_h + 7) / 8);
    cp.end();
    gpu::Device::submit();
    return;
  }
  gpu::Texture comp_final = comp_prev;

  // --- God rays: radial scatter from the sun, carved by the silhouette ---
  // Rays need the sun IN FRONT of the camera (sun_view.z > 0 — i.e. behind
  // the object). Fold the fade into the gain and skip the pass at zero.
  const float sun_fade = std::fmin(1.0f, std::fmax(0.0f, (sun_view.z - 0.02f) / 0.13f));
  const float rays_gain = s->rays * 0.9f * sun_fade * sun_i;
  bool has_rays = false;
  if (rays_gain > 1e-4f && s_pso_rays.valid()) {
    ensureTex(s->rays_tex, vp_w, vp_h, gpu::TextureFormat::RGBA16F, resized);
    if (s->rays_tex.valid()) {
      RaysUniforms ru = {};
      ru.sun_screen[0] = sun_px;
      ru.sun_screen[1] = sun_py;
      ru.sun_screen[2] = (float)s->water_t;
      ru.sun_screen[3] = rays_gain;
      ru.march[0] = 32.0f; ru.march[1] = 0.93f;
      ru.march[2] = (float)vp_w / 48.0f;
      ru.march[3] = s->caustics;
      ru.glow[0] = 1.0f / (0.5f * (float)vp_w);
      ru.glow[1] = s->sun_r; ru.glow[2] = s->sun_g; ru.glow[3] = s->sun_b;
      ru.sun_env[0] = sun_uv_x; ru.sun_env[1] = sun_uv_y;
      ru.sun_env[2] = sun_env_mode;
      s->ub_rays.writeOne(ru);

      auto cp = gpu::ComputePass::begin();
      cp.setPSO(s_pso_rays);
      cp.setTexture(comp_final, 0, 0);
      cp.setTexture(s->rays_tex, 1, 1);
      cp.setSampler(s->samp_clamp, 2);
      cp.setBuffer(s->ub_rays, 3);
      // One slot serves the sun-color sample: tint tex, env, or the 1x1
      // zero when the mode is plain Color (nothing real bound).
      cp.setTexture(sun_mode == 2 ? tint
                    : (sun_mode == 1 ? env_blurred : s->zero_tex), 4, 0);
      cp.dispatch((vp_w + 7) / 8, (vp_h + 7) / 8);
      cp.end();
      has_rays = true;
    }
  }

  // --- Final combine + tonemap -> tex_out ---
  FinalUniforms fu = {};
  fu.p[0] = has_rays ? 1.0f : 0.0f;
  s->ub_final.writeOne(fu);
  {
    auto cp = gpu::ComputePass::begin();
    cp.setPSO(s_pso_final);
    cp.setTexture(comp_final, 0, 0);
    cp.setTexture(has_rays ? s->rays_tex : s->zero_tex, 1, 0);
    cp.setTexture(in, 2, 0);
    cp.setTexture(out, 3, 1);
    cp.setBuffer(s->ub_final, 4);
    cp.dispatch((vp_w + 7) / 8, (vp_h + 7) / 8);
    cp.end();
  }

  gpu::Device::submit();
}

void on_resolume_param(void* self, long long param_id, double value) {
  (void)self; (void)param_id; (void)value;
}

} // namespace monolith
