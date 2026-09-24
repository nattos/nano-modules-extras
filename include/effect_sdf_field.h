#pragma once
/*
 * effect_sdf_field.h — the pluggable SDF provider contract (`sdf_field`
 * struct rail).
 *
 * An SDF-volume renderer (today: plume's march/fog/GI pass chain) can
 * render any field that arrives in this shape — the rail is the seam
 * that lets OTHER effects provide the geometry. It mirrors the
 * render_outputs idiom: a canonical struct of leaves, every leaf
 * individually optional at the schema level, with the supported leaf
 * COMBINATIONS documented executably by validate() below.
 *
 * Rail leaves:
 *   field_class : int   — Class enum; what the provider promises about
 *                         the field (star-shaped heightmap, grid-only, …)
 *   radius      : float — base sphere R, world units
 *   lip         : float — compression applied to grid .r, (0, 1]; a
 *                         consumer multiplies grid distances by 1/lip
 *                         wherever it needs free-space (AO, penumbra)
 *   lip_true    : float — the provider's true uncompensated Lipschitz
 *                         bound; lip may be floored above it for march
 *                         speed, and lip/lip_true tells the consumer how
 *                         much the grid overstates distances (it widens
 *                         its fine-tier handoff band by that ratio)
 *   crest_amp   : float — max displacement above R, world units (the
 *                         crest sphere R+crest_amp bounds the surface)
 *   crest_gain  : float — strength of the grid's .b crest channel;
 *                         0 = channel inert (consumer must not shade it)
 *   grid_ext    : float — grid half-extent, world units (v1: must equal
 *                         kGridExt — consumer shaders bake it)
 *   shell_res   : float — shell map resolution per axis (consumers floor
 *                         normal-estimation epsilons at ~2 shell texels)
 *   dust_count  : int   — number of live dust particles (see `dust`);
 *                         0 = no dust, consumer skips every dust path
 *   grid        : texture — kGridRes³ RGBA16F volume over [-ext, ext]³:
 *                         .r signed distance × lip (world units),
 *                         .g soft density band (1 inside → 0 outside
 *                            across a few voxels — GI injection, fog,
 *                            translucency read this),
 *                         .b crest emphasis,
 *                         .a dust/extinction density ≥ 0 (OPTIONAL — 0
 *                            when unused; a soft aggregate medium the
 *                            consumer may add to fog scattering and
 *                            attenuate sun by; NOT part of the distance
 *                            field)
 *   shell     : texture — octahedral S² heightmap (nano_octahedral
 *                         mapping), RGBA16F: .r radial displacement
 *                         h(dir) ≥ 0 (world units), .g crest.
 *                         The exact surface is r = R + h(dir); consumers
 *                         use it for the fine tier (exact radial
 *                         distance, normals) — it MUST be the same field
 *                         the grid was baked from (band-limit differences
 *                         within the fine-tier handoff band are fine,
 *                         whole-feature divergence is not)
 *   dust      : gpu array — dust particles: small discrete glinting
 *                         motes, SHARP in primary visibility (the
 *                         consumer splats them with exact depth), soft
 *                         everywhere else (their aggregate presence is
 *                         the grid's .a channel). float4-element buffer,
 *                         TWO float4 rows per particle:
 *                           row 0: pos.xyz (world), .w = radius (world)
 *                           row 1: normal.xyz (unit; drives glint),
 *                                  .w = seed in [0, 1)
 *                         Rows [0, 2·dust_count) are live; capacity may
 *                         exceed the live count. (A future emissive
 *                         extension adds rows per particle behind a new
 *                         scalar leaf — validate() gates it then.)
 *
 * Producer side: fill the scalars each frame via setValPath (elide when
 * unchanged), setGpuTexture("sdf_field/grid"|"shell", id) on
 * (re)allocation, setGpuBuffer("sdf_field/dust", id) likewise when
 * publishing dust, markGpuDirty("sdf_field") per frame. Consumer side:
 * mirror scalar patches on "<name>/…", resolve textures with
 * gpu::Device::textureForField (buffers with bufferForField), then gate
 * rendering on validate().
 */

namespace fx {
namespace sdf_field {

// What a provider can promise about its field. Only SphericalHeightmap
// is renderable today; the others are declared so providers/tests can
// name them and support becomes additive (validate() rejects them with
// stable reason strings until a renderer path exists).
enum Class {
  None = 0,
  // Star-shaped about the origin: exact surface r = radius + h(dir),
  // h single-valued on S² (the shell map). Grid + shell + all scalars.
  SphericalHeightmap = 1,
  // Grid is a true (compressed) Euclidean SDF, no shell — consumer
  // caps out at voxel resolution, normals from grid gradient. FUTURE.
  GridOnly = 2,
  // .g is a genuinely volumetric density (no surface band) — needs
  // interior light injection + a density-integrating march. FUTURE.
  Volumetric = 3,
};

// v1 grid conventions. Consumer shaders bake these as compile-time
// constants (querying 3D texture dimensions is non-portable), so a
// provider must match them exactly; they ride the rail (grid_ext) and
// validate() checks, which keeps a future relaxation additive.
constexpr int   kGridRes = 128;
constexpr float kGridExt = 0.85f;

// Dust pool ceiling. Consumers size per-frame scratch against the LIVE
// count, not this, so the cap only bounds a mis-declared provider.
constexpr int   kDustMax = 262144;

// Scalar half of the rail, plus consumer-resolved texture validity.
struct Desc {
  int   field_class = 0;
  float radius      = 0.f;
  float lip         = 0.f;
  float lip_true    = 0.f;
  float crest_amp   = 0.f;
  float crest_gain  = 0.f;
  float grid_ext    = 0.f;
  float shell_res   = 0.f;
  int   dust_count  = 0;
  bool  has_grid    = false;
  bool  has_shell   = false;
  bool  has_dust    = false;
};

// THE executable documentation of the supported combinations — the
// renderer gates on it, and test_sdf_field.cpp pins every branch.
// Returns nullptr on success (renderable class in *out); otherwise a
// static reason string (and *out = None).
inline const char* validate(const Desc& d, Class* out) {
  *out = None;
  switch (d.field_class) {
    case SphericalHeightmap: break;
    case GridOnly:   return "sdf_field: class GridOnly declared but not yet supported";
    case Volumetric: return "sdf_field: class Volumetric declared but not yet supported";
    case None:       return "sdf_field: no field class declared";
    default:         return "sdf_field: unknown field class";
  }
  if (!d.has_grid)  return "sdf_field: SphericalHeightmap requires the grid texture";
  if (!d.has_shell) return "sdf_field: SphericalHeightmap requires the shell texture";
  if (!(d.radius > 0.f)) return "sdf_field: radius must be > 0";
  if (!(d.lip > 0.f) || d.lip > 1.f) return "sdf_field: lip must be in (0, 1]";
  if (!(d.lip_true > 0.f) || d.lip_true > d.lip)
    return "sdf_field: lip_true must be in (0, lip]";
  if (d.crest_amp < 0.f) return "sdf_field: crest_amp must be >= 0";
  if (d.grid_ext != kGridExt) return "sdf_field: grid_ext must match kGridExt (v1)";
  if (d.radius + d.crest_amp > d.grid_ext)
    return "sdf_field: crest sphere exceeds the grid extent";
  if (d.shell_res < 64.f) return "sdf_field: shell_res must be >= 64";
  // Dust is optional: a live count requires the buffer, but a resolvable
  // buffer with count 0 is fine (a provider parks its pool at 0).
  if (d.dust_count < 0) return "sdf_field: dust_count must be >= 0";
  if (d.dust_count > kDustMax)
    return "sdf_field: dust_count exceeds kDustMax";
  if (d.dust_count > 0 && !d.has_dust)
    return "sdf_field: dust_count > 0 requires the dust buffer";
  *out = SphericalHeightmap;
  return nullptr;
}

}  // namespace sdf_field
}  // namespace fx

// Schema declaration helper — only meaningful inside an effect bundle
// (host.h pulls in the WASM host imports), so it's gated off for native
// test builds that include this header for Desc/validate alone.
#if defined(__wasm__)
#include "host.h"

namespace fx {
namespace sdf_field {

// Canonical rail shape (mirrors Schema::renderOutputs). Auto-binding
// matches by schema shape, not field name, so an effect declaring both
// directions uses the default name for one and e.g. "sdf_field_in" for
// the other — both still rail-couple to canonical peers.
inline state::Schema& declare(state::Schema& s, int io,
                              const char* name = "sdf_field") {
  return s.beginObject(name, io)
      .intField("field_class", 0, 0, 8, state::None)
      .floatField("radius",     0.f, 0.f, 4.f,  state::None)
      .floatField("lip",        1.f, 0.f, 1.f,  state::None)
      .floatField("lip_true",   1.f, 0.f, 1.f,  state::None)
      .floatField("crest_amp",  0.f, 0.f, 4.f,  state::None)
      .floatField("crest_gain", 0.f, 0.f, 1.f,  state::None)
      .floatField("grid_ext",   kGridExt, 0.f, 4.f, state::None)
      .floatField("shell_res",  1024.f, 0.f, 8192.f, state::None)
      .intField("dust_count", 0, 0, kDustMax, state::None)
      .textureField("grid",  state::None)
      .textureField("shell", state::None)
      .gpuArrayField("dust", "float4")
      .endObject();
}

// Provider-side publish helper: pushes the ACTIVE field onto the
// canonical `sdf_field` output each frame, deduping texture handles and
// eliding unchanged scalar leaves (patch-churn hygiene). Call only when
// state::isOutputConnected("sdf_field"); pass gpu texture `.id`s.
struct Publisher {
  Desc last{};
  bool valid = false;
  int last_grid = -1, last_shell = -1, last_dust = -1;

  // dust_id < 0 = no dust buffer (dust_count must then be 0 to validate
  // consumer-side; passing a buffer with count 0 parks the pool).
  void publish(const Desc& d, int grid_id, int shell_id, int dust_id = -1) {
    if (grid_id != last_grid) {
      last_grid = grid_id;
      state::setGpuTexture("sdf_field/grid", grid_id);
    }
    if (shell_id != last_shell) {
      last_shell = shell_id;
      state::setGpuTexture("sdf_field/shell", shell_id);
    }
    if (dust_id >= 0 && dust_id != last_dust) {
      last_dust = dust_id;
      state::setGpuBuffer("sdf_field/dust", dust_id);
    }
    auto pub = [&](const char* path, float v, float prev) {
      if (valid && v == prev) return;
      int vh = val::number(v);
      state::setValPath(path, vh);
      val::release(vh);
    };
    pub("sdf_field/field_class", (float)d.field_class, (float)last.field_class);
    pub("sdf_field/radius", d.radius, last.radius);
    pub("sdf_field/lip", d.lip, last.lip);
    pub("sdf_field/lip_true", d.lip_true, last.lip_true);
    pub("sdf_field/crest_amp", d.crest_amp, last.crest_amp);
    pub("sdf_field/crest_gain", d.crest_gain, last.crest_gain);
    pub("sdf_field/grid_ext", d.grid_ext, last.grid_ext);
    pub("sdf_field/shell_res", d.shell_res, last.shell_res);
    pub("sdf_field/dust_count", (float)d.dust_count, (float)last.dust_count);
    last = d;
    valid = true;
    state::markGpuDirty("sdf_field");
  }
};

}  // namespace sdf_field
}  // namespace fx
#endif  // __wasm__
