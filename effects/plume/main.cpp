/*
 * source.sdf.plume — SDF volume renderer, flagship effect (milestone 1).
 *
 * A displaced-sphere "plume" authored on a spherical SHELL MAP (octahedral
 * S² parameterization: displacement + material channels), baked each frame
 * into a cartesian SDF volume, and sphere-traced in a full-res compute
 * pass. This is the skeleton of the reusable SDF renderer: shell → bake →
 * march. Later milestones split the march into G-buffer + deferred shade,
 * add the detail tier (shell residual), resonant wave GI, the tier-1
 * atmosphere cascade, and materials.
 *
 * Pass chain (all compute):
 *   shell (×2)  — author h(dir) on shell_full (1024², all octaves) and
 *                 shell_coarse (256², band-limited for the 128³ grid).
 *   bake        — shell_coarse → sdf volume (distance, density, crest).
 *   march       — per-pixel ray vs volume: trilinear sphere-trace, central-
 *                 difference normal, sun lambert + rim, composite over
 *                 tex_in.
 *   slice_debug — optional volume/shell inspector replacing tex_out.
 *
 * World space: x right, y up, z into the screen; object at origin; the
 * camera orbits (yaw accumulator + tilt) at a distance holding framing
 * (monolith conventions). Motion is accumulator-driven (§2.1); the morph
 * drift walks a CLOSED CIRCLE in the noise domain so it never accumulates
 * float error and never jumps.
 */

#include <gpu.h>
#include <host.h>
#include <effect_utils.h>
#include <effect_sdf_field.h>

#include <cmath>

#include "field_gen.h"
#include "plume_shaders.h"

namespace plume {

constexpr float kPi = 3.14159265358979323846f;
constexpr float kTau = 2.0f * kPi;
constexpr float kDeg = kPi / 180.0f;

// Grid/shell conventions live with the shared sculptor (field_gen.h);
// the renderer's shaders bake the same constants (common.hlsl), and the
// sculptor's static_asserts pin them to the sdf_field rail's.
using plume_gen::kExt0;
constexpr int kGiRes = 64;

constexpr int DBG_OFF = 0;
constexpr int DBG_SDF = 1;
constexpr int DBG_SHELL = 2;
constexpr int DBG_RESIDUAL = 3;
constexpr int DBG_RADIANCE = 4;

// ---------------------------------------------------------------------------
// GPU uniform structs (16-byte rows, lockstep with the .hlsl cbuffers).
// ---------------------------------------------------------------------------

struct MarchUniforms {
  float cam_row0[4];   // view right (world), w = cam_pos.x
  float cam_row1[4];   // view up (world),    w = cam_pos.y
  float cam_row2[4];   // view fwd (world),   w = cam_pos.z
  float cam_p[4];      // focal, cover_ax, cover_ay, has_bg
  float sun_p[4];      // sun dir (world, toward light), w = intensity
  float albedo[4];     // rgb, w = opacity
  float vp[4];         // w, h, 1/w, 1/h
  float shade_p[4];    // shadow, ao, ambient, rim
  float fine_p[4];     // R (base radius), px_world (per unit t), inv_lip, bounce
  float misc[4];       // scene_mode, band widen, crest gain, wrap amount
  float mat[4];        // reflect, roughness, transmission, thickness
  float misc2[4];      // wrap lit-gate, exposure gain, black level,
                       // shell texel width (world units)
  float grade[4];      // srgb encode, 0, 0, 0
};
static_assert(sizeof(MarchUniforms) == 208, "MarchUniforms layout mismatch");

struct FogUniforms {
  float cam_row0[4];
  float cam_row1[4];
  float cam_row2[4];
  float cam_p[4];      // focal, cover_ax, cover_ay, R
  float sun_p[4];      // sun dir toward light, w = intensity
  float fog_p[4];      // shell gain, inv_soft, room gain, phase g
  float misc[4];       // inv_lip, ambient, bounce, 0
  float vp[4];         // half w, half h, 1/half w, 1/half h
  float misc2[4];      // iso-lobe blend, 0, 0, 0
};
static_assert(sizeof(FogUniforms) == 144, "FogUniforms layout mismatch");

struct CompUniforms {
  float opacity, has_bg, exposure, black;
  float dust_on, t_far, srgb, _p1;
};
static_assert(sizeof(CompUniforms) == 32, "CompUniforms layout mismatch");

// Shared by both dust splat passes (dust_common.hlsl).
struct DustSplatUniforms {
  float cam_row0[4];
  float cam_row1[4];
  float cam_row2[4];
  float cam_p[4];      // focal, cover_ax, cover_ay, dust_count
  float sun_p[4];      // sun dir toward light, w = intensity
  float albedo[4];     // rgb, w = exposure gain
  float vp[4];         // w, h, 1/w, 1/h
  float shade_p[4];    // shadow, ambient, bounce, inv_lip
  float misc[4];       // px_world, dust metallic, 0, 0
};
static_assert(sizeof(DustSplatUniforms) == 144,
              "DustSplatUniforms layout mismatch");

struct DustClearUniforms { float count, ones, _p0, _p1; };
static_assert(sizeof(DustClearUniforms) == 16,
              "DustClearUniforms layout mismatch");

struct DebugUniforms { float mode, slice, scale, _pad0; };
static_assert(sizeof(DebugUniforms) == 16, "DebugUniforms layout mismatch");

struct InjectUniforms {
  float sun_p[4];    // sun dir toward light, w = intensity
  float albedo[4];   // rgb bounce color, w = inv_lip
};
static_assert(sizeof(InjectUniforms) == 32, "InjectUniforms layout mismatch");

struct PropUniforms { float c2, damp, decay_mul, inject_gain; };
static_assert(sizeof(PropUniforms) == 16, "PropUniforms layout mismatch");

// Type-shared PSOs (shell/bake live with the sculptor in field_gen.h).
static gpu::ComputePSO s_pso_march;
static gpu::ComputePSO s_pso_march_hdr;
static gpu::ComputePSO s_pso_prefill;
static gpu::ComputePSO s_pso_debug;
static gpu::ComputePSO s_pso_inject;
static gpu::ComputePSO s_pso_prop;
static gpu::ComputePSO s_pso_fog;
static gpu::ComputePSO s_pso_comp;
static gpu::ComputePSO s_pso_dust_clear;
static gpu::ComputePSO s_pso_dust_depth;
static gpu::ComputePSO s_pso_dust_shade;

// ---------------------------------------------------------------------------

struct State {
  bool initialized = false;

  // The sculpted-field generator (shape params, morph accumulator,
  // shell/coarse/sdf textures + passes) — shared with plume_field.
  plume_gen::Sculptor gen;

  gpu::Buffer ub_march, ub_debug;
  gpu::Buffer ub_inject, ub_prop, ub_fog, ub_comp;
  gpu::Texture rad_vol[3];                 // GI wave field: prev/cur/next ring
  gpu::Texture inject_vol;                 // GI source term (lazy with rad)
  gpu::Texture zero_vol;                   // 1³ zeros (GI-off stand-in)
  gpu::Texture zero_tex;                   // 1×1 zeros (unwired-input stand-in)
  gpu::Texture scene_tex;                  // vp RGBA16F: color + hit distance
  gpu::Texture fog_tex;                    // vp/2 RGBA16F: in-scatter + trans
  gpu::Buffer dust_depth_buf;              // vp uints: splat depth resolve
  gpu::Buffer dust_cov_buf;                // vp uints: 8.8 coverage sums
  gpu::Texture dust_tex;                   // vp RGBA16F: winner color layer
  gpu::Buffer dust_stub_buf;               // 16B storage: unbound-slot filler
  gpu::Buffer ub_dust, ub_dust_clear;
  int dust_buf_px = 0;
  gpu::Sampler samp_clamp;
  int rad_rot = 0;                         // ring index of "cur"
  int scratch_w = 0, scratch_h = 0;

  // Param mirrors (shape params live in `gen`).
  float orbit = 0.35f;
  float tilt = 0.1f;
  float zoom = 0.25f;
  float azimuth = -35.0f;
  float elevation = 35.0f;
  float sun = 0.6f;
  float shadow = 0.7f;
  float ao_amt = 0.7f;
  float ambient = 0.5f;
  float wrap = 1.0f;
  float wrap_gate = 0.0f;
  float bounce = 0.5f;
  float resonance = 0.35f;
  float gi_speed = 0.5f;
  float gi_decay = 0.5f;
  float fog = 0.25f;
  float fog_soft = 0.5f;
  float phase = 0.4f;
  bool dual_lobe = false;
  float room_fog = 0.0f;
  float albedo_r = 0.85f, albedo_g = 0.85f, albedo_b = 0.87f;
  float reflect_k = 0.25f;
  float roughness = 0.4f;
  float dust_r = 0.85f, dust_g = 0.85f, dust_b = 0.87f;
  float dust_metal = 0.3f;
  float transmission = 0.3f;
  float thickness = 0.5f;
  float opacity = 1.0f;
  float exposure = 0.5f;
  float black_lvl = 0.0f;
  bool srgb = false;
  int debug_view = DBG_OFF;
  float debug_slice = 0.5f;

  // Accumulator (§2.1), cycles in [0,1); morph's lives in `gen`.
  double orbit_phase = 0.0;

  // sdf_field rail. rail_in mirrors the wired provider's scalar leaves
  // (texture validity is resolved per frame); rail_reject_reason is the
  // last validate() failure (static strings — pointer compare) so a
  // mis-declared provider logs once per transition, not per frame.
  fx::sdf_field::Desc rail_in;
  const char* rail_reject_reason = nullptr;
  fx::sdf_field::Publisher rail_pub;
};

void module_init() {
  auto schema = state::Schema()
      .helpField("intro",
        "## Plume\n"
        "A raymarched volumetric shape — a sphere sheathed in ridged, "
        "wind-swept flakes, authored as a live displacement field on the "
        "sphere's surface and rendered through a signed-distance volume. "
        "This is the first effect on the SDF volume renderer; bounce "
        "lighting, atmosphere and materials are landing milestone by "
        "milestone.\n\n"
        "**Try:** *Ridge Scale* high with *Sharpness* up for fine feathering; "
        "*Morph* slow for a breathing, living surface.");
  // --- Shape (the shared sculptor's params, field_gen.h) ---
  plume_gen::Sculptor::declareSchema(schema,
          "*Radius* is the body; *Ridge Depth/Scale* shape the displacement "
          "riding on it. *Sharpness* carves the field into terraced plates — "
          "at 0 the surface stays a smooth rolling heightfield. *Feathering* "
          "smears the pattern along the flow into wind-swept shingles. "
          "*Morph* drifts the field along a closed loop — it breathes "
          "forever without ever jumping. *Variation* picks a different "
          "pattern. When an upstream SDF field is wired into `sdf_field_in` "
          "this whole group goes inert — the provider's geometry renders "
          "in its place.")
      // --- Motion / camera ---
      .group("motion", "Camera")
      .floatField("orbit", 0.35f, 0.f, 1.f, state::PrimaryInput)
          .label("Orbit", "Orbit")
      .floatField("tilt", 0.1f, -1.f, 1.f, state::PrimaryInput)
          .label("Tilt", "Tilt")
      .floatField("zoom", 0.25f, 0.f, 1.f, state::PrimaryInput)
          .label("Zoom", "Zoom")
      // --- Light ---
      .group("light", "Light")
      .floatField("azimuth", -35.0f, -180.f, 180.f, state::PrimaryInput,
                  nullptr, 0.f, "deg").label("Azimuth", "Azim")
      .floatField("elevation", 35.0f, -80.f, 80.f, state::PrimaryInput,
                  nullptr, 0.f, "deg").label("Elevation", "Elev")
      .floatField("sun", 0.6f, 0.f, 1.f, state::PrimaryInput)
          .label("Sun", "Sun")
      .floatField("shadow", 0.7f, 0.f, 1.f, state::PrimaryInput)
          .label("Shadow", "Shdw")
      .floatField("ao", 0.7f, 0.f, 1.f, state::PrimaryInput)
          .label("Occlusion", "AO")
      .floatField("ambient", 0.5f, 0.f, 1.f, state::PrimaryInput)
          .label("Ambient", "Amb")
      .floatField("wrap", 1.0f, 0.f, 1.f, state::PrimaryInput)
          .label("Wrap Light", "Wrap")
      .floatField("wrap_gate", 0.0f, 0.f, 1.f, state::PrimaryInput)
          .label("Wrap Gate", "WGate")
      // --- Bounce (resonant wave GI) ---
      .group("gi", "Bounce Light")
        .groupHelp(
          "Bounce light lives in a volumetric field around the shape and "
          "PROPAGATES as a damped wave. *Bounce* is how much of it the "
          "surface picks up. *Resonance* is the character: low is calm "
          "diffusion (classic soft GI), high lets the light slosh and ring "
          "— reverb for light; move the sun and watch the field chase it. "
          "*Speed* sets how fast light travels, *Decay* how long it "
          "lingers.")
      .floatField("bounce", 0.5f, 0.f, 1.f, state::PrimaryInput)
          .label("Bounce", "Bnce")
      .floatField("resonance", 0.35f, 0.f, 1.f, state::PrimaryInput)
          .label("Resonance", "Reso")
      .floatField("gi_speed", 0.5f, 0.f, 1.f, state::PrimaryInput)
          .label("Light Speed", "Spd")
      .floatField("gi_decay", 0.5f, 0.f, 1.f, state::PrimaryInput)
          .label("Decay", "Dcy")
      // --- Atmosphere ---
      .group("atmosphere", "Atmosphere")
        .groupHelp(
          "*Fog* is a haze that hugs the displaced surface (its light comes "
          "from the same wave field as the bounce — the shape glows into "
          "its own atmosphere). *Softness* sets how far the haze reaches "
          "off the shell; *Phase* pushes the sun's in-scatter forward "
          "(silvery backlit shafts near ±180 azimuth); *Dual Lobe* mixes a "
          "small isotropic lobe into the phase — a multi-scatter stand-in "
          "that keeps frontlit fog from dropping to black at high Phase; "
          "*Room Fog* adds a thin medium everywhere for depth.")
      .floatField("fog", 0.25f, 0.f, 1.f, state::PrimaryInput)
          .label("Fog", "Fog")
      .floatField("fog_soft", 0.5f, 0.f, 1.f, state::PrimaryInput)
          .label("Softness", "Soft")
      .floatField("phase", 0.4f, 0.f, 1.f, state::PrimaryInput)
          .label("Phase", "Phase")
      .boolField("dual_lobe", false, state::SecondaryInput)
          .label("Dual Lobe", "Dual")
      .floatField("room_fog", 0.0f, 0.f, 1.f, state::PrimaryInput)
          .label("Room Fog", "Room")
      // --- Material ---
      .group("material", "Material")
        .groupHelp(
          "*Reflect*/*Roughness* are the porcelain sheen — a glossy sun "
          "glint on the plates, tight when smooth and broad when rough. "
          "*Translucency* lets light pass through thin plates so their "
          "edges glow when backlit (sun azimuth toward ±180); *Thickness* "
          "sets how deep light penetrates before dying out. Rail dust has "
          "its OWN material: *Dust Color*, and *Dust Metallic* running "
          "chalk to glitter — matte specks at 0, dark mirror flakes at 1 "
          "that flash a color-tinted glint as they tumble.")
      .rgbField("albedo", 0.85f, 0.85f, 0.87f, state::PrimaryInput)
          .label("Albedo", "Alb")
      .floatField("reflect", 0.25f, 0.f, 1.f, state::PrimaryInput)
          .label("Reflect", "Refl")
      .floatField("roughness", 0.4f, 0.f, 1.f, state::PrimaryInput)
          .label("Roughness", "Rough")
      .floatField("transmission", 0.3f, 0.f, 1.f, state::PrimaryInput)
          .label("Translucency", "Trans")
      .floatField("thickness", 0.5f, 0.f, 1.f, state::PrimaryInput)
          .label("Thickness", "Thick")
      .rgbField("dust_color", 0.85f, 0.85f, 0.87f, state::PrimaryInput)
          .label("Dust Color", "DustC")
      .floatField("dust_metallic", 0.3f, 0.f, 1.f, state::PrimaryInput)
          .label("Dust Metallic", "DustM")
      .floatField("opacity", 1.0f, 0.f, 1.f, state::PrimaryInput)
          .label("Opacity", "Opac")
      // --- Render ---
      .group("render", "Render")
        .groupHelp(
          "Camera-style output grade. *Exposure* is gain ahead of the "
          "tonemapper (±2 stops) — highlights roll off instead of "
          "clipping. *Black Level* lifts the floor toward a hazy film "
          "gray (right) or crushes the darks to true black (left). "
          "Plume shades in LINEAR light; by default the tonemapped "
          "result is written raw into the display-referred chain — the "
          "shipped look, tuned against that path. *sRGB Output* encodes "
          "with the sRGB curve instead, for colour-managed comps: mids "
          "lift noticeably, so expect to re-grade Exposure.")
      .floatField("exposure", 0.5f, 0.f, 1.f, state::PrimaryInput)
          .label("Exposure", "Expo")
      .floatField("black", 0.0f, -1.f, 1.f, state::PrimaryInput)
          .label("Black Level", "Black")
      .boolField("srgb", false, state::SecondaryInput)
          .label("sRGB Output", "sRGB")
      // --- Debug ---
      .group("debug", "Debug")
      .selectField("debug_view", DBG_OFF, state::SecondaryInput,
                   {{"Off", DBG_OFF},
                    {"SDF Slice", DBG_SDF},
                    {"Shell Map", DBG_SHELL},
                    {"Shell Residual", DBG_RESIDUAL},
                    {"Radiance", DBG_RADIANCE}})
          .label("Debug View", "Dbg")
      .floatField("debug_slice", 0.5f, 0.f, 1.f, state::SecondaryInput)
          .label("Debug Slice", "Slice")
      // --- I/O ---
      .textureField("tex_in", state::PrimaryInput)
      .textureField("tex_out", state::PrimaryOutput);
  // The sdf_field provider seam (effect_sdf_field.h), both directions:
  // `sdf_field` publishes the field this instance renders; a wired
  // `sdf_field_in` (auto-binds by shape to an upstream `sdf_field`)
  // REPLACES the sculpted shape — the Shape group goes inert and the
  // foreign geometry renders with this instance's camera/light/
  // atmosphere.
  fx::sdf_field::declare(schema, state::SecondaryOutput);
  fx::sdf_field::declare(schema, state::SecondaryInput, "sdf_field_in");
  schema.capability(state::Capability::Generator);
  state::init("source.sdf.plume", {1, 0, 0}, schema);

  if (gpu::Device::backend() == gpu::Backend::None) return;

  // Generator shaders (shell/bake) register with the shared sculptor.
  plume_gen::moduleInit();
  // march is registered twice: the direct path writes tex_out (sketch
  // default format), the fog path writes the RGBA16F scene buffer — each
  // name gets the right naga storage-format substitution (flash_particles
  // prefill precedent).
  state::registerShaderSPV("plume_march", MARCH_SPV, MARCH_SPV_SIZE);
  state::registerShaderSPV("plume_march_hdr", MARCH_SPV, MARCH_SPV_SIZE,
                           "rgba16float", "write");
  state::registerShaderSPV("plume_prefill", PREFILL_SPV, PREFILL_SPV_SIZE);
  state::registerShaderSPV("plume_slice_debug", SLICE_DEBUG_SPV,
                           SLICE_DEBUG_SPV_SIZE);
  state::registerShaderSPV("plume_gi_inject", GI_INJECT_SPV, GI_INJECT_SPV_SIZE,
                           "rgba16float", "write");
  state::registerShaderSPV("plume_gi_prop", GI_PROP_SPV, GI_PROP_SPV_SIZE,
                           "rgba16float", "write");
  state::registerShaderSPV("plume_fog", FOG_SPV, FOG_SPV_SIZE,
                           "rgba16float", "write");
  state::registerShaderSPV("plume_composite", COMPOSITE_SPV, COMPOSITE_SPV_SIZE);
  state::registerShaderSPV("plume_dust_clear", DUST_CLEAR_SPV,
                           DUST_CLEAR_SPV_SIZE);
  state::registerShaderSPV("plume_dust_depth", DUST_DEPTH_SPV,
                           DUST_DEPTH_SPV_SIZE);
  state::registerShaderSPV("plume_dust_shade", DUST_SHADE_SPV,
                           DUST_SHADE_SPV_SIZE, "rgba16float", "write");

  auto cs_march = gpu::Device::createShaderModuleByName("plume_march");
  auto cs_prefill = gpu::Device::createShaderModuleByName("plume_prefill");
  auto cs_debug = gpu::Device::createShaderModuleByName("plume_slice_debug");
  auto cs_march_hdr = gpu::Device::createShaderModuleByName("plume_march_hdr");
  auto cs_inject = gpu::Device::createShaderModuleByName("plume_gi_inject");
  auto cs_prop = gpu::Device::createShaderModuleByName("plume_gi_prop");
  auto cs_fog = gpu::Device::createShaderModuleByName("plume_fog");
  auto cs_comp = gpu::Device::createShaderModuleByName("plume_composite");
  auto cs_dclear = gpu::Device::createShaderModuleByName("plume_dust_clear");
  auto cs_ddepth = gpu::Device::createShaderModuleByName("plume_dust_depth");
  auto cs_dshade = gpu::Device::createShaderModuleByName("plume_dust_shade");
  if (!cs_march || !cs_march_hdr || !cs_prefill ||
      !cs_debug || !cs_inject || !cs_prop || !cs_fog || !cs_comp ||
      !cs_dclear || !cs_ddepth || !cs_dshade) return;

  s_pso_march = gpu::Device::createComputePSO(cs_march, "main", gpu::Bindings()
      .tex3d(0)
      .tex2d(1)
      .tex2d(2)
      .sampler(3)
      .storageTex2d(4)
      .uniform(5)
      .tex3d(6));
  s_pso_march_hdr = gpu::Device::createComputePSO(cs_march_hdr, "main", gpu::Bindings()
      .tex3d(0)
      .tex2d(1)
      .tex2d(2)
      .sampler(3)
      .storageTex2d(4, gpu::TextureFormat::RGBA16F)
      .uniform(5)
      .tex3d(6));
  s_pso_prefill = gpu::Device::createComputePSO(cs_prefill, "main", gpu::Bindings()
      .tex2d(0)
      .storageTex2d(1));
  s_pso_debug = gpu::Device::createComputePSO(cs_debug, "main", gpu::Bindings()
      .tex3d(0)
      .tex2d(1)
      .tex2d(2)
      .sampler(3)
      .storageTex2d(4)
      .uniform(5)
      .tex3d(6));
  s_pso_inject = gpu::Device::createComputePSO(cs_inject, "main", gpu::Bindings()
      .tex3d(0)
      .sampler(1)
      .storageTex3d(2, gpu::TextureFormat::RGBA16F)
      .uniform(3));
  s_pso_prop = gpu::Device::createComputePSO(cs_prop, "main", gpu::Bindings()
      .tex3d(0)
      .tex3d(1)
      .tex3d(2)
      .tex3d(3)
      .sampler(4)
      .storageTex3d(5, gpu::TextureFormat::RGBA16F)
      .uniform(6));
  s_pso_fog = gpu::Device::createComputePSO(cs_fog, "main", gpu::Bindings()
      .tex3d(0)
      .tex3d(1)
      .tex2d(2)
      .sampler(3)
      .storageTex2d(4, gpu::TextureFormat::RGBA16F)
      .uniform(5));
  s_pso_comp = gpu::Device::createComputePSO(cs_comp, "main", gpu::Bindings()
      .tex2d(0)
      .tex2d(1)
      .tex2d(2)
      .sampler(3)
      .storageTex2d(4)
      .uniform(5)
      .tex2d(6)          // dust layer (zero_tex when dust is off)
      .storage(7)        // dust depth (stub when off)
      .storage(8));      // dust coverage (stub when off)
  s_pso_dust_clear = gpu::Device::createComputePSO(cs_dclear, "main",
      gpu::Bindings()
      .storageRW(0)
      .uniform(1));
  s_pso_dust_depth = gpu::Device::createComputePSO(cs_ddepth, "main",
      gpu::Bindings()
      .storage(0)        // dust particles (rail buffer)
      .tex2d(1)          // scene (.a = surface depth)
      .storageRW(2)      // depth resolve (atomics)
      .uniform(3)
      .storageRW(4));    // coverage accumulator (atomics)
  s_pso_dust_shade = gpu::Device::createComputePSO(cs_dshade, "main",
      gpu::Bindings()
      .storage(0)        // dust particles
      .tex3d(1)          // sdf grid (shadow march)
      .tex3d(2)          // GI radiance
      .sampler(3)
      .storage(4)        // depth resolve (read)
      .storageTex2d(5, gpu::TextureFormat::RGBA16F)
      .uniform(6));

  state::log("plume: module initialized");
}

void* create() {
  auto* s = new State();
  s->gen.createBuffers();
  s->ub_march = gpu::Device::createBuffer(sizeof(MarchUniforms),
                                          gpu::BufferUsage::Uniform);
  s->ub_debug = gpu::Device::createBuffer(sizeof(DebugUniforms),
                                          gpu::BufferUsage::Uniform);
  s->ub_inject = gpu::Device::createBuffer(sizeof(InjectUniforms),
                                           gpu::BufferUsage::Uniform);
  s->ub_prop = gpu::Device::createBuffer(sizeof(PropUniforms),
                                         gpu::BufferUsage::Uniform);
  s->ub_fog = gpu::Device::createBuffer(sizeof(FogUniforms),
                                        gpu::BufferUsage::Uniform);
  s->ub_comp = gpu::Device::createBuffer(sizeof(CompUniforms),
                                         gpu::BufferUsage::Uniform);
  s->ub_dust = gpu::Device::createBuffer(sizeof(DustSplatUniforms),
                                         gpu::BufferUsage::Uniform);
  s->ub_dust_clear = gpu::Device::createBuffer(sizeof(DustClearUniforms),
                                               gpu::BufferUsage::Uniform);
  s->dust_stub_buf = gpu::Device::createBuffer(16, gpu::BufferUsage::Storage);
  s->zero_vol = gpu::Device::createTexture3D(1, 1, 1, gpu::TextureFormat::RGBA16F);
  s->zero_tex = gpu::Device::createTexture(1, 1, gpu::TextureFormat::RGBA16F);
  if (s->zero_tex.valid())
    gpu::Device::clear(s->zero_tex, 0.0f, 0.0f, 0.0f, 0.0f);
  s->samp_clamp = gpu::Device::createSampler(gpu::FilterMode::Linear,
                                             gpu::AddressMode::ClampToEdge);
  return s;
}

void destroy(void* self) {
  auto* s = static_cast<State*>(self);
  if (!s) return;
  s->gen.release();
  s->ub_march.release();
  s->ub_debug.release();
  s->ub_inject.release();
  s->ub_prop.release();
  s->ub_fog.release();
  s->ub_comp.release();
  s->scene_tex.release();
  s->fog_tex.release();
  s->dust_depth_buf.release();
  s->dust_cov_buf.release();
  s->dust_tex.release();
  s->dust_stub_buf.release();
  s->ub_dust.release();
  s->ub_dust_clear.release();
  for (int i = 0; i < 3; i++) s->rad_vol[i].release();
  s->inject_vol.release();
  s->zero_vol.release();
  s->zero_tex.release();
  s->samp_clamp.release();
  delete s;
}

void init(void* self) {
  auto* s = static_cast<State*>(self);
  if (!s) return;
  s->gen.resetPhase();
  s->orbit_phase = 0.0;
  s->initialized = s->gen.valid() && s->ub_march.valid() &&
                   s->ub_debug.valid() && s->ub_inject.valid() &&
                   s->ub_prop.valid() && s_pso_march.valid() &&
                   s_pso_prefill.valid() && s_pso_debug.valid() &&
                   s_pso_inject.valid() && s_pso_prop.valid() &&
                   s_pso_march_hdr.valid() && s_pso_fog.valid() &&
                   s_pso_comp.valid() && s->ub_fog.valid() &&
                   s->ub_comp.valid();
}

void tick(void* self, double dt) {
  auto* s = static_cast<State*>(self);
  if (!s) return;
  if (!(dt > 0.0)) dt = 0.0;
  if (dt > 0.050) dt = 0.050;
  s->gen.tick(dt);
  s->orbit_phase = plume_gen::wrap01(
      s->orbit_phase + dt * plume_gen::rateHz(s->orbit, 0.02));
}

void on_state_patched(void* self, int n, const char* pb, const int* off,
                      const int* len, const int* ops) {
  auto* s = static_cast<State*>(self);
  if (!s) return;
  for (int i = 0; i < n; i++) {
    if (ops[i] != state::PatchReplace) continue;
    const char* p = pb + off[i];
    int l = len[i];
    if (s->gen.patch(p, l, i)) continue;   // Shape params (field_gen.h)
    if      (state::pathIs(p, l, "orbit"))       s->orbit = state::patchFloat(i);
    else if (state::pathIs(p, l, "tilt"))        s->tilt = state::patchFloat(i);
    else if (state::pathIs(p, l, "zoom"))        s->zoom = state::patchFloat(i);
    else if (state::pathIs(p, l, "azimuth"))     s->azimuth = state::patchFloat(i);
    else if (state::pathIs(p, l, "elevation"))   s->elevation = state::patchFloat(i);
    else if (state::pathIs(p, l, "sun"))         s->sun = state::patchFloat(i);
    else if (state::pathIs(p, l, "shadow"))      s->shadow = state::patchFloat(i);
    else if (state::pathIs(p, l, "ao"))          s->ao_amt = state::patchFloat(i);
    else if (state::pathIs(p, l, "ambient"))     s->ambient = state::patchFloat(i);
    else if (state::pathIs(p, l, "wrap"))        s->wrap = state::patchFloat(i);
    else if (state::pathIs(p, l, "wrap_gate"))   s->wrap_gate = state::patchFloat(i);
    else if (state::pathIs(p, l, "bounce"))      s->bounce = state::patchFloat(i);
    else if (state::pathIs(p, l, "resonance"))   s->resonance = state::patchFloat(i);
    else if (state::pathIs(p, l, "gi_speed"))    s->gi_speed = state::patchFloat(i);
    else if (state::pathIs(p, l, "gi_decay"))    s->gi_decay = state::patchFloat(i);
    else if (state::pathIs(p, l, "fog"))         s->fog = state::patchFloat(i);
    else if (state::pathIs(p, l, "fog_soft"))    s->fog_soft = state::patchFloat(i);
    else if (state::pathIs(p, l, "phase"))       s->phase = state::patchFloat(i);
    else if (state::pathIs(p, l, "dual_lobe"))   s->dual_lobe = state::patchBool(i);
    else if (state::pathIs(p, l, "room_fog"))    s->room_fog = state::patchFloat(i);
    else if (state::pathIs(p, l, "albedo")) {
      auto v = state::patchVec3(i);
      s->albedo_r = v.x; s->albedo_g = v.y; s->albedo_b = v.z;
    }
    else if (state::pathIs(p, l, "reflect"))     s->reflect_k = state::patchFloat(i);
    else if (state::pathIs(p, l, "roughness"))   s->roughness = state::patchFloat(i);
    else if (state::pathIs(p, l, "dust_color")) {
      auto v = state::patchVec3(i);
      s->dust_r = v.x; s->dust_g = v.y; s->dust_b = v.z;
    }
    else if (state::pathIs(p, l, "dust_metallic")) s->dust_metal = state::patchFloat(i);
    else if (state::pathIs(p, l, "transmission")) s->transmission = state::patchFloat(i);
    else if (state::pathIs(p, l, "thickness"))   s->thickness = state::patchFloat(i);
    else if (state::pathIs(p, l, "opacity"))     s->opacity = state::patchFloat(i);
    else if (state::pathIs(p, l, "exposure"))    s->exposure = state::patchFloat(i);
    else if (state::pathIs(p, l, "black"))       s->black_lvl = state::patchFloat(i);
    else if (state::pathIs(p, l, "srgb"))        s->srgb = state::patchBool(i);
    else if (state::pathIs(p, l, "debug_view"))  s->debug_view = state::patchInt(i);
    else if (state::pathIs(p, l, "debug_slice")) s->debug_slice = state::patchFloat(i);
    // sdf_field_in scalar leaves (the wired provider's declaration).
    else if (state::pathIs(p, l, "sdf_field_in/field_class"))
      s->rail_in.field_class = state::patchInt(i);
    else if (state::pathIs(p, l, "sdf_field_in/radius"))
      s->rail_in.radius = state::patchFloat(i);
    else if (state::pathIs(p, l, "sdf_field_in/lip"))
      s->rail_in.lip = state::patchFloat(i);
    else if (state::pathIs(p, l, "sdf_field_in/lip_true"))
      s->rail_in.lip_true = state::patchFloat(i);
    else if (state::pathIs(p, l, "sdf_field_in/crest_amp"))
      s->rail_in.crest_amp = state::patchFloat(i);
    else if (state::pathIs(p, l, "sdf_field_in/crest_gain"))
      s->rail_in.crest_gain = state::patchFloat(i);
    else if (state::pathIs(p, l, "sdf_field_in/grid_ext"))
      s->rail_in.grid_ext = state::patchFloat(i);
    else if (state::pathIs(p, l, "sdf_field_in/shell_res"))
      s->rail_in.shell_res = state::patchFloat(i);
    else if (state::pathIs(p, l, "sdf_field_in/dust_count"))
      s->rail_in.dust_count = state::patchInt(i);
  }
}

void render(void* self, int vp_w, int vp_h) {
  auto* s = static_cast<State*>(self);
  if (!s || !s->initialized || vp_w <= 0 || vp_h <= 0) return;

  auto in = gpu::Device::textureForField("tex_in");
  auto out = gpu::Device::textureForField("tex_out");
  if (!out.valid()) return;

  // --- Idle: pure passthrough / clear ---
  if (s->opacity < 1.0f / 255.0f && s->debug_view == DBG_OFF) {
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

  // --- Field source: wired provider, else self ---
  // The renderer half below (march/fog/GI/debug uniform fills + binds)
  // reads the field ONLY through this Desc + texture pair — the
  // sdf_field provider contract (effect_sdf_field.h) — so a foreign
  // field renders through code paths identical to the sculpted one.
  fx::sdf_field::Desc field;
  gpu::Texture field_grid, field_shell;
  gpu::Buffer field_dust;
  bool foreign = false;
  if (state::isInputConnected("sdf_field_in")) {
    fx::sdf_field::Desc fd = s->rail_in;
    gpu::Texture fg = gpu::Device::textureForField("sdf_field_in/grid");
    gpu::Texture fsh = gpu::Device::textureForField("sdf_field_in/shell");
    gpu::Buffer fdu = gpu::Device::bufferForField("sdf_field_in/dust");
    fd.has_grid = fg.valid();
    fd.has_shell = fsh.valid();
    fd.has_dust = fdu.valid();
    fx::sdf_field::Class cls;
    const char* reason = fx::sdf_field::validate(fd, &cls);
    if (!reason) {
      field = fd;
      field_grid = fg;
      field_shell = fsh;
      field_dust = fdu;
      foreign = true;
    }
    // A mis-declared provider falls back to the sculpted shape — log the
    // validate() reason once per transition so it's diagnosable (static
    // strings, so pointer identity tracks the transition).
    if (reason != s->rail_reject_reason) {
      s->rail_reject_reason = reason;
      if (reason) state::log(state::LogLevel::Warn, reason);
    }
  } else {
    s->rail_reject_reason = nullptr;
  }

  // Sculpt our own field (shell ×2 + bake, field_gen.h) unless a valid
  // foreign one replaced it.
  if (!foreign && !s->gen.run(field, field_grid, field_shell)) return;

  // --- Provider publish: the ACTIVE field (relays foreign when wired
  // through, so a chain of consumers sees one consistent declaration) ---
  if (state::isOutputConnected("sdf_field"))
    s->rail_pub.publish(field, field_grid.id, field_shell.id,
                        field_dust.valid() ? field_dust.id : -1);

  // --- GI: inject + wave propagation (whole stage skipped at bounce 0) ---
  bool gi_on = s->bounce > 0.001f || s->debug_view == DBG_RADIANCE;
  gpu::Texture rad_cur = s->zero_vol;
  if (gi_on) {
    const bool fresh = !s->rad_vol[0].valid();
    for (int i = 0; i < 3; i++) {
      if (!s->rad_vol[i].valid())
        s->rad_vol[i] = gpu::Device::createTexture3D(kGiRes, kGiRes, kGiRes,
                                                     gpu::TextureFormat::RGBA16F);
      gi_on = gi_on && s->rad_vol[i].valid();
    }
    if (!s->inject_vol.valid())
      s->inject_vol = gpu::Device::createTexture3D(kGiRes, kGiRes, kGiRes,
                                                   gpu::TextureFormat::RGBA16F);
    gi_on = gi_on && s->inject_vol.valid();

    if (gi_on) {
      const float saz = s->azimuth * kDeg, sel2 = s->elevation * kDeg;
      const float gi_sun = std::pow(2.0f, (s->sun - 0.5f) * 3.0f);
      InjectUniforms iu = {};
      iu.sun_p[0] = std::sin(saz) * std::cos(sel2);
      iu.sun_p[1] = std::sin(sel2);
      iu.sun_p[2] = -std::cos(saz) * std::cos(sel2);
      iu.sun_p[3] = gi_sun;
      iu.albedo[0] = s->albedo_r; iu.albedo[1] = s->albedo_g;
      iu.albedo[2] = s->albedo_b; iu.albedo[3] = 1.0f / field.lip;
      s->ub_inject.writeOne(iu);
      {
        auto cp = gpu::ComputePass::begin();
        cp.setPSO(s_pso_inject);
        cp.setTexture(field_grid, 0, 0);
        cp.setSampler(s->samp_clamp, 1);
        cp.setTexture(s->inject_vol, 2, 1);
        cp.setBuffer(s->ub_inject, 3);
        cp.dispatch(kGiRes / 4, kGiRes / 4, kGiRes / 4);
        cp.end();
      }

      // Telegrapher step params. Damping sweeps diffusion <-> ringing;
      // injection is scaled against survival so overall brightness stays
      // in the same ballpark across the decay range.
      const float c = 0.10f + 0.45f * s->gi_speed;   // voxels/step (CFL < 0.577)
      const float damp = 0.9f * std::pow(0.02f / 0.9f, s->resonance);
      const float decay_mul = 0.86f + 0.135f * s->gi_decay;
      // Steady state ~ gain / (1 - decay_mul): normalize so the field's
      // equilibrium brightness stays ~O(1) across the decay range.
      const float inject_gain = 1.1f * (1.005f - decay_mul);
      // Freshly created volumes hold garbage on some backends: a survival-0
      // step writes exact zeros, so three rotations scrub the whole ring.
      const int steps = fresh ? 3 : 2;
      for (int k = 0; k < steps; k++) {
        PropUniforms pu = fresh
            ? PropUniforms{0.0f, 1.0f, 0.0f, 0.0f}
            : PropUniforms{c * c, damp, decay_mul, inject_gain};
        s->ub_prop.writeOne(pu);
        gpu::Texture& prev = s->rad_vol[(s->rad_rot + 2) % 3];
        gpu::Texture& cur = s->rad_vol[s->rad_rot];
        gpu::Texture& next = s->rad_vol[(s->rad_rot + 1) % 3];
        auto cp = gpu::ComputePass::begin();
        cp.setPSO(s_pso_prop);
        cp.setTexture(cur, 0, 0);
        cp.setTexture(prev, 1, 0);
        cp.setTexture(s->inject_vol, 2, 0);
        cp.setTexture(field_grid, 3, 0);
        cp.setSampler(s->samp_clamp, 4);
        cp.setTexture(next, 5, 1);
        cp.setBuffer(s->ub_prop, 6);
        cp.dispatch(kGiRes / 4, kGiRes / 4, kGiRes / 4);
        cp.end();
        s->rad_rot = (s->rad_rot + 1) % 3;
      }
      rad_cur = s->rad_vol[s->rad_rot];
    }
  }

  // --- Debug views replace the output entirely ---
  if (s->debug_view != DBG_OFF) {
    DebugUniforms du = { (float)(s->debug_view - 1), s->debug_slice, 1.0f, 0.f };
    s->ub_debug.writeOne(du);
    auto cp = gpu::ComputePass::begin();
    cp.setPSO(s_pso_debug);
    // Inspect the field actually being rendered (foreign when wired);
    // the coarse map is plume-authoring-specific and stays self (it may
    // never have been sculpted when a foreign field is wired).
    cp.setTexture(field_grid, 0, 0);
    cp.setTexture(field_shell, 1, 0);
    cp.setTexture(s->gen.shell_coarse.valid() ? s->gen.shell_coarse
                                              : s->zero_tex, 2, 0);
    cp.setSampler(s->samp_clamp, 3);
    cp.setTexture(out, 4, 1);
    cp.setBuffer(s->ub_debug, 5);
    cp.setTexture(rad_cur, 6, 0);
    cp.dispatch((vp_w + 7) / 8, (vp_h + 7) / 8);
    cp.end();
    gpu::Device::submit();
    return;
  }

  // --- Scene-buffer path scratch (lazy, vp-sized). The scene buffer is
  // needed by fog AND by dust (the splat passes need per-pixel surface
  // depth and a rewritable color target) — dust forces the full
  // march → scene → composite route even at fog 0, with the fog texture
  // simply cleared to "no fog" so composite runs unchanged. ---
  bool fog_on = s->fog > 0.001f || s->room_fog > 0.001f;
  bool dust_on = field.dust_count > 0 && field_dust.valid() &&
                 s_pso_dust_clear.valid() && s_pso_dust_depth.valid() &&
                 s_pso_dust_shade.valid() && s->ub_dust.valid() &&
                 s->ub_dust_clear.valid();
  const bool resized = (s->scratch_w != vp_w || s->scratch_h != vp_h);
  const int half_w = (vp_w + 1) / 2, half_h = (vp_h + 1) / 2;
  bool scene_path = fog_on || dust_on;
  if (scene_path) {
    if (!s->scene_tex.valid() || resized) {
      s->scene_tex.release();
      s->scene_tex = gpu::Device::createTexture(vp_w, vp_h,
                                                gpu::TextureFormat::RGBA16F);
    }
    if (!s->fog_tex.valid() || resized) {
      s->fog_tex.release();
      s->fog_tex = gpu::Device::createTexture(half_w, half_h,
                                              gpu::TextureFormat::RGBA16F);
    }
    scene_path = s->scene_tex.valid() && s->fog_tex.valid();
    fog_on = fog_on && scene_path;
    dust_on = dust_on && scene_path;
    // Dims track the LIVE scratch only — if the path is off across a
    // resize, `resized` stays stale-true and the textures rebuild on
    // re-entry.
    s->scratch_w = vp_w;
    s->scratch_h = vp_h;
  }
  if (dust_on) {
    const int px = vp_w * vp_h;
    if (!s->dust_depth_buf.valid() || s->dust_buf_px != px) {
      s->dust_depth_buf.release();
      s->dust_depth_buf = gpu::Device::createBuffer(
          (long long)px * 4, gpu::BufferUsage::Storage);
      s->dust_cov_buf.release();
      s->dust_cov_buf = gpu::Device::createBuffer(
          (long long)px * 4, gpu::BufferUsage::Storage);
      s->dust_tex.release();
      s->dust_tex = gpu::Device::createTexture(vp_w, vp_h,
                                               gpu::TextureFormat::RGBA16F);
      s->dust_buf_px = px;
    }
    dust_on = s->dust_depth_buf.valid() && s->dust_cov_buf.valid() &&
              s->dust_tex.valid();
  }

  // --- Camera: orbit yaw accumulator + tilt, framing-hold dolly ---
  const auto cs = fx::coverSquare(vp_w, vp_h);
  const float yaw = kTau * (float)s->orbit_phase;
  const float el = s->tilt * 40.0f * kDeg;
  const float fov = (30.0f + 45.0f * s->zoom) * kDeg;
  const float focal = 1.0f / std::tan(fov * 0.5f);
  float cam_d = 3.0f * focal / 3.7320508f;
  cam_d = std::fmax(cam_d, kExt0 + 0.35f);

  const float cy = std::cos(yaw), sy = std::sin(yaw);
  const float ce = std::cos(el), se = std::sin(el);
  const float px = cam_d * (-sy * ce);
  const float py = cam_d * se;
  const float pz = cam_d * (-cy * ce);
  // forward = -pos/|pos|; right = norm(cross(up, fwd)); up = cross(fwd, right).
  const float fx_ = -px / cam_d, fy_ = -py / cam_d, fz_ = -pz / cam_d;
  float rx = fz_, rz = -fx_;   // cross((0,1,0), f), y component is 0
  const float rlen = std::sqrt(rx * rx + rz * rz);
  rx = rlen > 1e-5f ? rx / rlen : 1.0f;
  rz = rlen > 1e-5f ? rz / rlen : 0.0f;
  const float ux = fy_ * rz;                 // cross(f, r) with r.y = 0
  const float uy = fz_ * rx - fx_ * rz;
  const float uz = -fy_ * rx;

  // --- Sun (world; azimuth 0 lights from the resting camera direction) ---
  const float az = s->azimuth * kDeg, sel = s->elevation * kDeg;
  const float sun_i = std::pow(2.0f, (s->sun - 0.5f) * 3.0f);

  // Output grade: exposure = ±2 stops around unity; black level knob maps
  // to a ±0.5 lift/crush point (full ±1 would be degenerate).
  const float expo = std::pow(2.0f, (s->exposure - 0.5f) * 4.0f);
  const float black = 0.5f * s->black_lvl;

  MarchUniforms mu = {};
  mu.cam_row0[0] = rx;  mu.cam_row0[1] = 0.0f; mu.cam_row0[2] = rz;  mu.cam_row0[3] = px;
  mu.cam_row1[0] = ux;  mu.cam_row1[1] = uy;   mu.cam_row1[2] = uz;  mu.cam_row1[3] = py;
  mu.cam_row2[0] = fx_; mu.cam_row2[1] = fy_;  mu.cam_row2[2] = fz_; mu.cam_row2[3] = pz;
  mu.cam_p[0] = focal; mu.cam_p[1] = cs.ax; mu.cam_p[2] = cs.ay;
  mu.cam_p[3] = in.valid() ? 1.0f : 0.0f;
  mu.sun_p[0] = std::sin(az) * std::cos(sel);
  mu.sun_p[1] = std::sin(sel);
  mu.sun_p[2] = -std::cos(az) * std::cos(sel);
  mu.sun_p[3] = sun_i;
  mu.albedo[0] = s->albedo_r; mu.albedo[1] = s->albedo_g;
  mu.albedo[2] = s->albedo_b; mu.albedo[3] = s->opacity;
  mu.vp[0] = (float)vp_w; mu.vp[1] = (float)vp_h;
  mu.vp[2] = 1.0f / vp_w; mu.vp[3] = 1.0f / vp_h;
  mu.shade_p[0] = s->shadow;
  mu.shade_p[1] = s->ao_amt;
  mu.shade_p[2] = s->ambient;
  mu.shade_p[3] = 0.6f * s->wrap;   // rim rides the wrap-light control
  mu.fine_p[0] = field.radius;
  // World size of one pixel at unit distance (screen-adaptive normal eps).
  mu.fine_p[1] = 1.0f / ((float)vp_h * cs.ay * focal);
  // Decompression for penumbra/AO reads of the compressed grid distances.
  mu.fine_p[2] = 1.0f / field.lip;
  mu.fine_p[3] = gi_on ? 1.2f * s->bounce : 0.0f;
  mu.misc[0] = scene_path ? 1.0f : 0.0f;
  mu.misc[1] = std::fmin(3.0f, field.lip / field.lip_true);
  mu.misc[2] = field.crest_gain;
  mu.misc[3] = s->wrap;
  mu.misc2[0] = s->wrap_gate;
  mu.misc2[1] = expo;
  mu.misc2[2] = black;
  // Shell texel width (world units) for the normal-eps floor — computed
  // from the field's DECLARED shell_res (a provider's map may differ
  // from plume's own PLM_SHELL_RES).
  mu.misc2[3] = field.radius * 6.2832f / (2.0f * field.shell_res);
  mu.grade[0] = s->srgb ? 1.0f : 0.0f;
  mu.mat[0] = s->reflect_k;
  mu.mat[1] = s->roughness;
  mu.mat[2] = s->transmission;
  mu.mat[3] = s->thickness;
  s->ub_march.writeOne(mu);

  {
    auto cp = gpu::ComputePass::begin();
    cp.setPSO(scene_path ? s_pso_march_hdr : s_pso_march);
    cp.setTexture(field_grid, 0, 0);
    cp.setTexture(in.valid() ? in : s->zero_tex, 1, 0);
    cp.setTexture(field_shell, 2, 0);
    cp.setSampler(s->samp_clamp, 3);
    cp.setTexture(scene_path ? s->scene_tex : out, 4, 1);
    cp.setBuffer(s->ub_march, 5);
    cp.setTexture(rad_cur, 6, 0);
    cp.dispatch((vp_w + 7) / 8, (vp_h + 7) / 8);
    cp.end();
  }

  // --- Dust splats: nearest-particle depth + coverage resolve, then the
  // winner shades into the dedicated dust LAYER. The scene buffer keeps
  // the pure surface depth — the half-res fog march never sees point
  // occluders (one speck must not truncate a whole fog texel), and
  // composite seats each speck in the fog by its own depth instead. ---
  if (dust_on) {
    DustClearUniforms dcu = { (float)(vp_w * vp_h), 1.f, 0.f, 0.f };
    s->ub_dust_clear.writeOne(dcu);
    DustSplatUniforms du = {};
    for (int i = 0; i < 4; i++) {
      du.cam_row0[i] = mu.cam_row0[i];
      du.cam_row1[i] = mu.cam_row1[i];
      du.cam_row2[i] = mu.cam_row2[i];
      du.sun_p[i] = mu.sun_p[i];
      du.vp[i] = mu.vp[i];
    }
    du.cam_p[0] = focal; du.cam_p[1] = cs.ax; du.cam_p[2] = cs.ay;
    du.cam_p[3] = (float)field.dust_count;
    du.albedo[0] = s->dust_r; du.albedo[1] = s->dust_g;
    du.albedo[2] = s->dust_b; du.albedo[3] = expo;
    du.shade_p[0] = s->shadow;
    du.shade_p[1] = s->ambient;
    du.shade_p[2] = gi_on ? 1.2f * s->bounce : 0.0f;
    du.shade_p[3] = 1.0f / field.lip;
    du.misc[0] = mu.fine_p[1];   // px_world
    du.misc[1] = s->dust_metal;
    s->ub_dust.writeOne(du);
    const int pgroups = (field.dust_count + 63) / 64;
    {
      auto cp = gpu::ComputePass::begin();
      cp.setPSO(s_pso_dust_clear);
      cp.setBuffer(s->dust_depth_buf, 0);
      cp.setBuffer(s->ub_dust_clear, 1);
      cp.dispatch((vp_w * vp_h + 63) / 64);
      cp.end();
    }
    // Coverage clears to zero — the uniform rewrite is safe mid-frame
    // (the backend versions the backing buffer per dispatch).
    DustClearUniforms dcz = { (float)(vp_w * vp_h), 0.f, 0.f, 0.f };
    s->ub_dust_clear.writeOne(dcz);
    {
      auto cp = gpu::ComputePass::begin();
      cp.setPSO(s_pso_dust_clear);
      cp.setBuffer(s->dust_cov_buf, 0);
      cp.setBuffer(s->ub_dust_clear, 1);
      cp.dispatch((vp_w * vp_h + 63) / 64);
      cp.end();
    }
    {
      auto cp = gpu::ComputePass::begin();
      cp.setPSO(s_pso_dust_depth);
      cp.setBuffer(field_dust, 0);
      cp.setTexture(s->scene_tex, 1, 0);
      cp.setBuffer(s->dust_depth_buf, 2);
      cp.setBuffer(s->ub_dust, 3);
      cp.setBuffer(s->dust_cov_buf, 4);
      cp.dispatch(pgroups);
      cp.end();
    }
    {
      auto cp = gpu::ComputePass::begin();
      cp.setPSO(s_pso_dust_shade);
      cp.setBuffer(field_dust, 0);
      cp.setTexture(field_grid, 1, 0);
      cp.setTexture(rad_cur, 2, 0);
      cp.setSampler(s->samp_clamp, 3);
      cp.setBuffer(s->dust_depth_buf, 4);
      cp.setTexture(s->dust_tex, 5, 1);
      cp.setBuffer(s->ub_dust, 6);
      cp.dispatch(pgroups);
      cp.end();
    }
  }

  // Dust without fog still needs composite: park the fog texture at
  // "no in-scatter, full transmittance".
  if (scene_path && !fog_on)
    gpu::Device::clear(s->fog_tex, 0.0f, 0.0f, 0.0f, 1.0f);

  if (fog_on) {
    FogUniforms fu = {};
    for (int i = 0; i < 4; i++) {
      fu.cam_row0[i] = mu.cam_row0[i];
      fu.cam_row1[i] = mu.cam_row1[i];
      fu.cam_row2[i] = mu.cam_row2[i];
      fu.sun_p[i] = mu.sun_p[i];
    }
    fu.cam_p[0] = focal; fu.cam_p[1] = cs.ax; fu.cam_p[2] = cs.ay;
    fu.cam_p[3] = field.radius;
    fu.fog_p[0] = s->fog * 5.0f;
    fu.fog_p[1] = 1.4427f / (0.04f + 0.30f * s->fog_soft);
    fu.fog_p[2] = s->room_fog * 1.2f;
    fu.fog_p[3] = 0.75f * s->phase;
    fu.misc[0] = 1.0f / field.lip;
    fu.misc[1] = s->ambient;
    fu.misc[2] = gi_on ? 1.2f * s->bounce : 0.0f;
    fu.misc[3] = field.crest_amp;
    fu.vp[0] = (float)half_w; fu.vp[1] = (float)half_h;
    fu.vp[2] = 1.0f / half_w; fu.vp[3] = 1.0f / half_h;
    // Dual lobe blends 20% isotropic into the HG phase — enough of a
    // multi-scatter floor that frontlit fog stays visible, small enough
    // that the backlit forward glare keeps its character.
    fu.misc2[0] = s->dual_lobe ? 0.2f : 0.0f;
    s->ub_fog.writeOne(fu);
    {
      auto cp = gpu::ComputePass::begin();
      cp.setPSO(s_pso_fog);
      cp.setTexture(field_grid, 0, 0);
      cp.setTexture(rad_cur, 1, 0);
      cp.setTexture(s->scene_tex, 2, 0);
      cp.setSampler(s->samp_clamp, 3);
      cp.setTexture(s->fog_tex, 4, 1);
      cp.setBuffer(s->ub_fog, 5);
      cp.dispatch((half_w + 7) / 8, (half_h + 7) / 8);
      cp.end();
    }
  }

  if (scene_path) {
    // t_far: where miss rays effectively stop integrating fog (haze
    // hugs the body, so ~the camera-to-origin distance plus a couple of
    // world units covers the dense span). Only steers the front/back
    // apportioning of fog onto dust — coarse is fine.
    CompUniforms cu = { s->opacity, in.valid() ? 1.0f : 0.0f, expo, black,
                        dust_on ? 1.0f : 0.0f, cam_d + 2.0f,
                        s->srgb ? 1.0f : 0.0f, 0.f };
    s->ub_comp.writeOne(cu);
    {
      auto cp = gpu::ComputePass::begin();
      cp.setPSO(s_pso_comp);
      cp.setTexture(s->scene_tex, 0, 0);
      cp.setTexture(s->fog_tex, 1, 0);
      cp.setTexture(in.valid() ? in : s->zero_tex, 2, 0);
      cp.setSampler(s->samp_clamp, 3);
      cp.setTexture(out, 4, 1);
      cp.setBuffer(s->ub_comp, 5);
      cp.setTexture(dust_on ? s->dust_tex : s->zero_tex, 6, 0);
      cp.setBuffer(dust_on ? s->dust_depth_buf : s->dust_stub_buf, 7);
      cp.setBuffer(dust_on ? s->dust_cov_buf : s->dust_stub_buf, 8);
      cp.dispatch((vp_w + 7) / 8, (vp_h + 7) / 8);
      cp.end();
    }
  }

  gpu::Device::submit();
}

void on_resolume_param(void* self, long long param_id, double value) {
  (void)self; (void)param_id; (void)value;
}

} // namespace plume
