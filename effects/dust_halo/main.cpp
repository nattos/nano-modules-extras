/*
 * source.sdf.dust_halo — an INTERMEDIATE stage on the `sdf_field` rail:
 * consumes an upstream provider's field and republishes it with shaped
 * dust added — a soft latitude band on a tiltable axis plus a radial
 * standoff profile. Band center at the axis pole = a hovering cap (the
 * beret); width up = a full spherical shell; center at the equator with
 * a thin band and a fat radial thickness = a planetary ring. Motes
 * orbit the axis with Keplerian shear and tumble for glint; both are
 * absolute phases off two accumulated clocks, so rate 0 freezes the
 * cloud exactly and generation is fully STATELESS (no pool).
 *
 * Upstream dust is MERGED (relayed into the outgoing buffer's tail),
 * upstream grid/shell/scalars pass through untouched — except the
 * grid when the halo has motes: their aggregate density folds into a
 * copy's .a channel (the rail's soft-influence contract) so fog
 * scatters and the sun dims inside the cloud. The fold pass, density
 * accumulate, and clear reuse helio_field's / plume's compiled SPVs —
 * the only new shader is the generator.
 *
 * Video passes through untouched (clear when unwired).
 */

#include <gpu.h>
#include <host.h>
#include <effect_sdf_field.h>

#include <cmath>
#include <cstdint>

#include "dust_halo_shaders.h"

namespace dust_halo {

// The halo's own mote budget; the merged total is clamped to the rail's
// kDustMax (upstream motes keep priority — they're alive, ours are cheap).
constexpr int kHaloBudget = 98304;

struct HaloGenUniforms {
  float axis_a[4];    // halo axis (unit), w = gen count
  float axis_b1[4];   // band basis 1,    w = total count
  float axis_b2[4];   // band basis 2,    w = provider radius R
  float band[4];      // theta_c, theta_w, soft, ecc
  float radial[4];    // gap, thick, gap_soft, mote size
  float motion[4];    // T_drift, T_tumble, 0, 0
};
static_assert(sizeof(HaloGenUniforms) == 96, "HaloGenUniforms mismatch");

struct DustAccumUniforms { float count, stride, gain, _p2; };
static_assert(sizeof(DustAccumUniforms) == 16, "DustAccumUniforms mismatch");

struct DustFoldUniforms { float norm, _p0, _p1, _p2; };
static_assert(sizeof(DustFoldUniforms) == 16, "DustFoldUniforms mismatch");

struct AccumClearUniforms { float count, ones, _p0, _p1; };
static_assert(sizeof(AccumClearUniforms) == 16, "AccumClearUniforms mismatch");

static gpu::ComputePSO s_pso_prefill;
static gpu::ComputePSO s_pso_gen;
static gpu::ComputePSO s_pso_accum;
static gpu::ComputePSO s_pso_fold;
static gpu::ComputePSO s_pso_clear;

struct State {
  bool initialized = false;
  fx::sdf_field::Publisher rail_pub;

  gpu::Buffer dust_buf;    // kDustMax × 2 float4 (merged rail layout)
  gpu::Buffer dummy_buf;   // bound at t0 when upstream has no dust
  gpu::Buffer accum_buf;   // 128³ uints (fixed-point density counts)
  gpu::Texture vol_pub;    // RGBA16F 128³ (upstream grid + halo .a)
  gpu::Buffer ub_gen, ub_accum, ub_fold, ub_clear;

  // Absolute motion phases (radians); knob rates fold into the
  // accumulation so the clocks freeze exactly at rate 0 and knob moves
  // change speed, never phase.
  double t_drift = 0.0;
  double t_tumble = 0.0;

  // Upstream rail mirror (patched leaves; textures resolve per frame).
  fx::sdf_field::Desc field{};

  // Param mirrors.
  float amount = 0.5f;
  float size = 0.5f;
  float tilt = 0.15f;
  float yaw = 0.0f;
  float arc = 0.0f;
  float width = 0.3f;
  float ecc = 0.0f;
  float soft = 0.5f;
  float gap = 0.15f;
  float gap_soft = 0.3f;
  float thick = 0.25f;
  float drift = 0.25f;
  float tumble = 0.5f;
};

void module_init() {
  auto schema = state::Schema()
      .helpField("intro",
        "## Dust Halo\n"
        "Sits BETWEEN an SDF provider and its renderer and adds dust to "
        "the field passing through: a shaped cloud of sharp glinting "
        "motes hovering off the body. One shape control set spans a "
        "tilted hovering cap (the beret), a full spherical shell, and a "
        "planetary ring — *Arc* slides the band from the axis pole to "
        "the equator, *Width* opens it toward full coverage, *Gap* and "
        "*Thickness* set the radial standoff profile.\n\n"
        "Motes orbit the halo axis with Keplerian shear (inner motes "
        "lap outer ones — rings visibly wind up) and tumble so their "
        "facets twinkle. The cloud also scatters the downstream fog and "
        "dims the sun softly, like any rail dust. Upstream dust (e.g. "
        "Helio Field's) is merged, not replaced.\n\n"
        "The video input passes through untouched.")
      .group("halo", "Halo")
      .groupHelp(
          "*Amount* is how many motes (squared taper), *Mote Size* how "
          "big each one splats.")
      .floatField("amount", 0.5f, 0.f, 1.f, state::PrimaryInput)
          .label("Amount", "Amt")
      .floatField("size", 0.5f, 0.f, 1.f, state::PrimaryInput)
          .label("Mote Size", "Size")
      .group("shape", "Shape")
      .groupHelp(
          "The band on the halo axis. *Tilt*/*Yaw* aim the axis (the "
          "beret's jaunty angle). *Arc* slides the band center from the "
          "axis pole (a cap) to the equator (a ring's plane). *Width* "
          "is the band's angular size — full width wraps the whole "
          "body in a shell. *Eccentricity* squashes the band into an "
          "ellipse; the envelope stays put while motes stream through "
          "it. *Softness* feathers the band edges.")
      .floatField("tilt", 0.15f, 0.f, 1.f, state::PrimaryInput)
          .label("Tilt", "Tilt")
      .floatField("yaw", 0.0f, 0.f, 1.f, state::PrimaryInput)
          .label("Yaw", "Yaw")
      .floatField("arc", 0.0f, 0.f, 1.f, state::PrimaryInput)
          .label("Arc", "Arc")
      .floatField("width", 0.3f, 0.f, 1.f, state::PrimaryInput)
          .label("Width", "Wid")
      .floatField("ecc", 0.0f, 0.f, 1.f, state::PrimaryInput)
          .label("Eccentricity", "Ecc")
      .floatField("soft", 0.5f, 0.f, 1.f, state::PrimaryInput)
          .label("Softness", "Soft")
      .group("radial", "Standoff")
      .groupHelp(
          "The gap between the body and the cloud. *Gap* is the "
          "standoff altitude, *Thickness* the cloud's radial depth, "
          "*Gap Shape* the edge profile — 0 is a crisp slab (a hard "
          "inner shelf holding the gap open), 1 feathers both faces.")
      .floatField("gap", 0.15f, 0.f, 1.f, state::PrimaryInput)
          .label("Gap", "Gap")
      .floatField("thick", 0.25f, 0.f, 1.f, state::PrimaryInput)
          .label("Thickness", "Thick")
      .floatField("gap_soft", 0.3f, 0.f, 1.f, state::PrimaryInput)
          .label("Gap Shape", "GShp")
      .group("motion", "Motion")
      .groupHelp(
          "*Drift* orbits the motes around the halo axis — signed, with "
          "Keplerian shear (closer = faster, so rings wind up "
          "differentially). *Tumble* spins each mote's facet for "
          "twinkle. Both freeze exactly at 0.")
      .floatField("drift", 0.25f, -1.f, 1.f, state::PrimaryInput)
          .label("Drift", "Drift")
      .floatField("tumble", 0.5f, 0.f, 1.f, state::PrimaryInput)
          .label("Tumble", "Tmbl")
      // --- I/O ---
      .textureField("tex_in", state::PrimaryInput)
      .textureField("tex_out", state::PrimaryOutput);
  fx::sdf_field::declare(schema, state::SecondaryInput, "sdf_field_in");
  fx::sdf_field::declare(schema, state::SecondaryOutput);
  schema.capability(state::Capability::Generator);
  state::init("source.sdf.dust_halo", {1, 0, 0}, schema);

  if (gpu::Device::backend() == gpu::Backend::None) return;

  // Only the generator is new; passthrough/accumulate/fold/clear reuse
  // plume's and helio_field's compiled SPVs under this effect's names.
  state::registerShaderSPV("dust_halo_prefill", DUST_HALO_PREFILL_SPV,
                           DUST_HALO_PREFILL_SPV_SIZE);
  state::registerShaderSPV("dust_halo_gen", DUST_HALO_GEN_SPV,
                           DUST_HALO_GEN_SPV_SIZE);
  state::registerShaderSPV("dust_halo_accum", DUST_HALO_ACCUM_SPV,
                           DUST_HALO_ACCUM_SPV_SIZE);
  state::registerShaderSPV("dust_halo_fold", DUST_HALO_FOLD_SPV,
                           DUST_HALO_FOLD_SPV_SIZE,
                           "rgba16float", "write");
  state::registerShaderSPV("dust_halo_clear", DUST_HALO_CLEAR_SPV,
                           DUST_HALO_CLEAR_SPV_SIZE);
  auto cs_prefill = gpu::Device::createShaderModuleByName("dust_halo_prefill");
  auto cs_gen = gpu::Device::createShaderModuleByName("dust_halo_gen");
  auto cs_accum = gpu::Device::createShaderModuleByName("dust_halo_accum");
  auto cs_fold = gpu::Device::createShaderModuleByName("dust_halo_fold");
  auto cs_clear = gpu::Device::createShaderModuleByName("dust_halo_clear");
  if (!cs_prefill || !cs_gen || !cs_accum || !cs_fold || !cs_clear) return;
  s_pso_prefill = gpu::Device::createComputePSO(cs_prefill, "main",
      gpu::Bindings()
          .tex2d(0)
          .storageTex2d(1));
  s_pso_gen = gpu::Device::createComputePSO(cs_gen, "main",
      gpu::Bindings()
          .storage(0)        // upstream motes (dummy when absent)
          .storageRW(1)      // outgoing merged motes
          .uniform(2));
  s_pso_accum = gpu::Device::createComputePSO(cs_accum, "main",
      gpu::Bindings()
          .storage(0)        // motes (halo's live at the head)
          .storageRW(1)      // count volume (atomics)
          .uniform(2));
  s_pso_fold = gpu::Device::createComputePSO(cs_fold, "main",
      gpu::Bindings()
          .tex3d(0)          // upstream grid volume
          .storage(1)        // count volume
          .storageTex3d(2, gpu::TextureFormat::RGBA16F)
          .uniform(3));
  s_pso_clear = gpu::Device::createComputePSO(cs_clear, "main",
      gpu::Bindings()
          .storageRW(0)
          .uniform(1));

  state::log("dust_halo: module initialized");
}

void* create() {
  auto* s = new State();
  s->ub_gen = gpu::Device::createBuffer(sizeof(HaloGenUniforms),
                                        gpu::BufferUsage::Uniform);
  s->ub_accum = gpu::Device::createBuffer(sizeof(DustAccumUniforms),
                                          gpu::BufferUsage::Uniform);
  s->ub_fold = gpu::Device::createBuffer(sizeof(DustFoldUniforms),
                                         gpu::BufferUsage::Uniform);
  s->ub_clear = gpu::Device::createBuffer(sizeof(AccumClearUniforms),
                                          gpu::BufferUsage::Uniform);
  s->dummy_buf = gpu::Device::createBuffer(32, gpu::BufferUsage::Storage);
  return s;
}

void destroy(void* self) {
  auto* s = static_cast<State*>(self);
  if (!s) return;
  // Upstream textures/buffers resolved via ForField are NOT ours to
  // release — only resources this effect created.
  s->dust_buf.release();
  s->dummy_buf.release();
  s->accum_buf.release();
  s->vol_pub.release();
  s->ub_gen.release();
  s->ub_accum.release();
  s->ub_fold.release();
  s->ub_clear.release();
  delete s;
}

void init(void* self) {
  auto* s = static_cast<State*>(self);
  if (!s) return;
  s->t_drift = 0.0;
  s->t_tumble = 0.0;
  s->initialized = s->ub_gen.valid() && s_pso_gen.valid() &&
                   s_pso_accum.valid() && s_pso_fold.valid() &&
                   s_pso_clear.valid();
}

void tick(void* self, double dt) {
  auto* s = static_cast<State*>(self);
  if (!s) return;
  if (!(dt > 0.0)) dt = 0.0;
  if (dt > 0.050) dt = 0.050;
  // Knob rates fold into the accumulation: rate 0 freezes exactly, and
  // moving a knob changes speed without snapping accumulated phase.
  s->t_drift += dt * 0.6 * s->drift;
  s->t_tumble += dt * 2.5 * s->tumble;
}

void on_state_patched(void* self, int n, const char* pb, const int* off,
                      const int* len, const int* ops) {
  auto* s = static_cast<State*>(self);
  if (!s) return;
  for (int i = 0; i < n; i++) {
    if (ops[i] != state::PatchReplace) continue;
    const char* p = pb + off[i];
    const int l = len[i];
    if      (state::pathIs(p, l, "amount"))    s->amount = state::patchFloat(i);
    else if (state::pathIs(p, l, "size"))      s->size = state::patchFloat(i);
    else if (state::pathIs(p, l, "tilt"))      s->tilt = state::patchFloat(i);
    else if (state::pathIs(p, l, "yaw"))       s->yaw = state::patchFloat(i);
    else if (state::pathIs(p, l, "arc"))       s->arc = state::patchFloat(i);
    else if (state::pathIs(p, l, "width"))     s->width = state::patchFloat(i);
    else if (state::pathIs(p, l, "ecc"))       s->ecc = state::patchFloat(i);
    else if (state::pathIs(p, l, "soft"))      s->soft = state::patchFloat(i);
    else if (state::pathIs(p, l, "gap"))       s->gap = state::patchFloat(i);
    else if (state::pathIs(p, l, "gap_soft"))  s->gap_soft = state::patchFloat(i);
    else if (state::pathIs(p, l, "thick"))     s->thick = state::patchFloat(i);
    else if (state::pathIs(p, l, "drift"))     s->drift = state::patchFloat(i);
    else if (state::pathIs(p, l, "tumble"))    s->tumble = state::patchFloat(i);
    // Upstream rail scalar leaves (the wired provider's declaration).
    else if (state::pathIs(p, l, "sdf_field_in/field_class"))
      s->field.field_class = (int)state::patchFloat(i);
    else if (state::pathIs(p, l, "sdf_field_in/radius"))
      s->field.radius = state::patchFloat(i);
    else if (state::pathIs(p, l, "sdf_field_in/lip"))
      s->field.lip = state::patchFloat(i);
    else if (state::pathIs(p, l, "sdf_field_in/lip_true"))
      s->field.lip_true = state::patchFloat(i);
    else if (state::pathIs(p, l, "sdf_field_in/crest_amp"))
      s->field.crest_amp = state::patchFloat(i);
    else if (state::pathIs(p, l, "sdf_field_in/crest_gain"))
      s->field.crest_gain = state::patchFloat(i);
    else if (state::pathIs(p, l, "sdf_field_in/grid_ext"))
      s->field.grid_ext = state::patchFloat(i);
    else if (state::pathIs(p, l, "sdf_field_in/shell_res"))
      s->field.shell_res = state::patchFloat(i);
    else if (state::pathIs(p, l, "sdf_field_in/dust_count"))
      s->field.dust_count = (int)state::patchFloat(i);
  }
}

// Generate + merge + fold + republish one frame. Returns false when the
// upstream field is absent/invalid (nothing published).
static bool runField(State* s) {
  gpu::Texture fg = gpu::Device::textureForField("sdf_field_in/grid");
  gpu::Texture fsh = gpu::Device::textureForField("sdf_field_in/shell");
  gpu::Buffer fdu = gpu::Device::bufferForField("sdf_field_in/dust");

  fx::sdf_field::Desc d = s->field;
  d.has_grid = fg.valid();
  d.has_shell = fsh.valid();
  d.has_dust = fdu.valid();
  fx::sdf_field::Class cls;
  if (fx::sdf_field::validate(d, &cls) != nullptr)
    return false;

  const float R = d.radius;
  const int up_count = fdu.valid() ? d.dust_count : 0;
  int halo_count = (int)(kHaloBudget * s->amount * s->amount);
  halo_count = halo_count / 64 * 64;
  int gen = halo_count;
  if (gen > fx::sdf_field::kDustMax - up_count)
    gen = fx::sdf_field::kDustMax - up_count;
  if (gen < 0) gen = 0;
  const int total = gen + up_count;

  // Nothing to add and nothing to relay: republish upstream verbatim
  // (the rail-level identity).
  if (total == 0 || !s->initialized) {
    s->rail_pub.publish(d, fg.id, fsh.id, fdu.valid() ? fdu.id : -1);
    return true;
  }

  if (!s->dust_buf.valid())
    s->dust_buf = gpu::Device::createBuffer(
        (long long)fx::sdf_field::kDustMax * 2 * 16,
        gpu::BufferUsage::Storage);
  if (!s->dust_buf.valid()) return false;

  // --- Generate halo motes (head) + relay upstream motes (tail) ---
  {
    const float tilt_r = s->tilt * 3.14159265f;
    const float yaw_r = s->yaw * 6.2831853f;
    const float st = std::sin(tilt_r), ct = std::cos(tilt_r);
    float ax = st * std::cos(yaw_r), ay = ct, az_ = st * std::sin(yaw_r);
    // Orthonormal band basis around the axis.
    float rx = 0.f, ry = 1.f, rz = 0.f;
    if (std::fabs(ay) > 0.99f) { rx = 1.f; ry = 0.f; }
    float b1x = ry * az_ - rz * ay, b1y = rz * ax - rx * az_,
          b1z = rx * ay - ry * ax;
    float bl = std::sqrt(b1x * b1x + b1y * b1y + b1z * b1z);
    b1x /= bl; b1y /= bl; b1z /= bl;
    float b2x = ay * b1z - az_ * b1y, b2y = az_ * b1x - ax * b1z,
          b2z = ax * b1y - ay * b1x;

    HaloGenUniforms gu = {};
    gu.axis_a[0] = ax; gu.axis_a[1] = ay; gu.axis_a[2] = az_;
    gu.axis_a[3] = (float)gen;
    gu.axis_b1[0] = b1x; gu.axis_b1[1] = b1y; gu.axis_b1[2] = b1z;
    gu.axis_b1[3] = (float)total;
    gu.axis_b2[0] = b2x; gu.axis_b2[1] = b2y; gu.axis_b2[2] = b2z;
    gu.axis_b2[3] = R;
    gu.band[0] = s->arc * 1.5707963f;
    gu.band[1] = (0.05f + 0.95f * s->width) * 1.5707963f;
    gu.band[2] = s->soft;
    gu.band[3] = s->ecc;
    gu.radial[0] = R * (0.03f + 0.85f * s->gap);
    gu.radial[1] = R * (0.02f + 0.6f * s->thick);
    gu.radial[2] = s->gap_soft;
    // Bottom must reach sub-pixel at real viewport sizes (a 1080p pixel
    // is ~0.001 world units at typical camera distance); below the splat
    // presence floor every size is the same 1-px glint, so no resolution
    // is wasted down there.
    gu.radial[3] = 0.0001f + 0.0049f * s->size;
    gu.motion[0] = (float)s->t_drift;
    gu.motion[1] = (float)s->t_tumble;
    s->ub_gen.writeOne(gu);

    auto cp = gpu::ComputePass::begin();
    cp.setPSO(s_pso_gen);
    cp.setBuffer(fdu.valid() ? fdu : s->dummy_buf, 0);
    cp.setBuffer(s->dust_buf, 1);
    cp.setBuffer(s->ub_gen, 2);
    cp.dispatch((total + 63) / 64);
    cp.end();
  }

  // --- Soft influence: halo motes → grid .a (upstream motes already
  // carry theirs in the upstream grid). Fold runs on a copy; when the
  // halo adds nothing the upstream grid id passes through untouched. ---
  const int vres = fx::sdf_field::kGridRes;
  const int vcount = vres * vres * vres;
  bool halo_vol = gen > 0;
  if (halo_vol) {
    if (!s->accum_buf.valid())
      s->accum_buf = gpu::Device::createBuffer((long long)vcount * 4,
                                               gpu::BufferUsage::Storage);
    if (!s->vol_pub.valid())
      s->vol_pub = gpu::Device::createTexture3D(
          vres, vres, vres, gpu::TextureFormat::RGBA16F);
    halo_vol = s->accum_buf.valid() && s->vol_pub.valid();
  }
  if (halo_vol) {
    AccumClearUniforms au = { (float)vcount, 0.f, 0.f, 0.f };  // fill zeros
    s->ub_clear.writeOne(au);
    {
      auto cp = gpu::ComputePass::begin();
      cp.setPSO(s_pso_clear);
      cp.setBuffer(s->accum_buf, 0);
      cp.setBuffer(s->ub_clear, 1);
      cp.dispatch((vcount + 63) / 64);
      cp.end();
    }
    // Head only (upstream motes carry their density in the upstream
    // grid), decimated: mean-preserving stride + weight, in step with
    // helio's accumulate (see dust_accum.hlsl).
    const int kAccumCap = 24576;
    const int a_stride = (gen + kAccumCap - 1) / kAccumCap;
    const int a_count = (gen + a_stride - 1) / a_stride;
    DustAccumUniforms cu = { (float)a_count, (float)a_stride,
                             (float)gen / (float)a_count, 0.f };
    s->ub_accum.writeOne(cu);
    {
      auto cp = gpu::ComputePass::begin();
      cp.setPSO(s_pso_accum);
      cp.setBuffer(s->dust_buf, 0);
      cp.setBuffer(s->accum_buf, 1);
      cp.setBuffer(s->ub_accum, 2);
      cp.dispatch((a_count + 63) / 64);
      cp.end();
    }
    // 20 motes saturate a voxel — kept in step with helio's fold gain.
    DustFoldUniforms fu = { 1.0f / (256.0f * 20.0f), 0.f, 0.f, 0.f };
    s->ub_fold.writeOne(fu);
    {
      auto cp = gpu::ComputePass::begin();
      cp.setPSO(s_pso_fold);
      cp.setTexture(fg, 0, 0);
      cp.setBuffer(s->accum_buf, 1);
      cp.setTexture(s->vol_pub, 2, 1);
      cp.setBuffer(s->ub_fold, 3);
      cp.dispatch(vres / 4, vres / 4, vres / 4);
      cp.end();
    }
  }

  // --- Republish: upstream field, dust merged, grid .a enriched ---
  d.dust_count = total;
  d.has_dust = true;
  s->rail_pub.publish(d,
                      halo_vol ? s->vol_pub.id : fg.id,
                      fsh.id,
                      s->dust_buf.id);
  return true;
}

void render(void* self, int vp_w, int vp_h) {
  auto* s = static_cast<State*>(self);
  if (!s || vp_w <= 0 || vp_h <= 0) return;

  auto in = gpu::Device::textureForField("tex_in");
  auto out = gpu::Device::textureForField("tex_out");
  if (!out.valid()) return;

  if (state::isOutputConnected("sdf_field") &&
      state::isInputConnected("sdf_field_in"))
    runField(s);

  // Video passthrough (clear when unwired — the field is the output).
  if (in.valid() && s_pso_prefill.valid()) {
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
}

void on_resolume_param(void* self, long long param_id, double value) {
  (void)self; (void)param_id; (void)value;
}

} // namespace dust_halo
