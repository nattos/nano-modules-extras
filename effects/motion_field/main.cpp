/*
 * motion.field — Image-driven motion vector generator.
 *
 * Activation: per-pixel luma soft-thresholded against `threshold` ±
 * `softness`. Pixels below threshold emit zero motion; above, full
 * motion; in between, a smoothstep ramp.
 *
 * Magnitude: `magnitude` × (1 ± mag_jitter * value-noise). The noise
 * is locally smooth (Perlin-style, controlled by mag_noise_scale)
 * so neighbouring pixels share similar magnitude — patches breathe
 * coherently rather than flickering.
 *
 * Direction: weighted sum of three components, each independently
 * weighted, sum then normalised:
 *   - Static rotation       — a fixed angle (`rotation`).
 *   - Radial outward        — from `radial_anchor` (uv coord).
 *   - Luma gradient + bias  — central-difference gradient of the
 *                             input luma, rotated by `gradient_bias`
 *                             (0 = uphill, ±90° = along iso-luma
 *                             contours, useful for edge-flow looks).
 * Plus a per-pixel angular jitter on top, again locally smooth so
 * the field swirls coherently instead of becoming salt-and-pepper.
 *
 * Output: tex_in passes through to tex_out unchanged. The motion
 * texture (`render_outputs/motion`) carries the computed velocity
 * field. Set `vis_opacity > 0` to overlay an HSV-polar visualization
 * of the motion vectors on top of the input — useful while tuning,
 * leave at 0 in production.
 *
 * Class-like instance model: module_init() compiles the two shared
 * compute PSOs + publishes the schema once per type; each chain entry
 * gets its own State (params, time accumulator, per-instance uniform
 * buffer + motion textures) via create(). All instance callbacks take
 * `self`.
 */

#include <gpu.h>
#include <host.h>
#include "motion_field_shaders.h"

#include <cmath>
#include <cstdint>

namespace motion_field {

// 5 float4-rows = 80 bytes total. std140-style alignment so each
// row sits at a 16-byte boundary; the WGSL `cbuffer` layout matches.
struct Uniforms {
  // row 0: activation
  float threshold;
  float softness;
  float magnitude;
  float mag_jitter;

  // row 1: noise scales + static rotation
  float mag_noise_scale;
  float rotation_rad;
  float rotation_weight;
  float radial_weight;

  // row 2: radial anchor + gradient
  float radial_anchor_x;
  float radial_anchor_y;
  float gradient_weight;
  float gradient_bias_rad;

  // row 3: angle jitter + viz
  float angle_jitter;
  float angle_noise_scale;
  float vis_opacity;
  float vis_scale;

  // row 4: temporal evolution
  float    noise_time;   // accumulated_time × evolution_rate
  float    _pad_t0;
  float    _pad_t1;
  float    _pad_t2;
};
static_assert(sizeof(Uniforms) == 80, "Uniforms layout mismatch");

static constexpr float DEG2RAD = 3.14159265358979323846f / 180.0f;

// Per-instance state. One per chain entry.
struct State {
  gpu::Buffer  uniform_buf;
  gpu::Texture motion_tex;
  gpu::Texture zero_motion_tex;   // 1x1 rgba16float fallback (no upstream)
  int          motion_w = 0;
  int          motion_h = 0;
  bool         initialized = false;

  // --- Schema-mirrored state ---
  float threshold       = 0.5f;
  float softness        = 0.05f;
  float magnitude       = 0.005f;
  float mag_jitter      = 0.5f;
  float mag_noise_scale = 16.0f;

  float rotation_deg     = 0.0f;
  float rotation_weight  = 1.0f;
  float radial_weight    = 0.0f;
  float radial_anchor_x  = 0.5f;
  float radial_anchor_y  = 0.5f;
  float gradient_weight  = 0.0f;
  float gradient_bias_deg = 90.0f;

  float angle_jitter     = 0.0f;
  float angle_noise_scale = 16.0f;
  // Hz at which the noise field mutates. 0 = static (single snapshot
  // repeats forever). 1 = one full snapshot transition per second.
  // Higher values give faster, more chaotic flow.
  float evolution_rate   = 0.0f;
  // Accumulated "noise time" — units of "noise snapshots". Advances
  // by `dt * evolution_rate` every tick. The integer part picks the
  // snapshot, the fractional part interpolates to the next.
  float noise_time       = 0.0f;
  float vis_opacity      = 0.0f;
  float vis_scale        = 100.0f;
};

// Type-shared: compiled once in module_init().
static gpu::ComputePSO s_pso_color;
static gpu::ComputePSO s_pso_motion;

// Type-level setup: schema + the two shared compute PSOs.
void module_init() {
  state::init("motion.field", {1, 0, 1},
    state::Schema()
      // Top-level manual: high-level "what is this / how to use / what to try".
      .helpField("intro",
        "## Motion Field\n"
        "Turns an input **image into a motion-vector field** on the `motion` rail — "
        "downstream effects (swarms, warps, blurs) read it as velocity. Bright pixels "
        "(above *Threshold*) emit motion; the *Direction* is a weighted blend of a "
        "fixed rotation, a radial push, and the image's own luma gradient.\n\n"
        "**Try:** raise *Vis Opacity* to see the field as a colour wheel while you "
        "tune, then drop it back to 0 for production. Push *Gradient Weight* with a "
        "±90° *Gradient Bias* to get flow that hugs the edges of your image, and add a "
        "little *Evolution Rate* so the field breathes over time.")
      // Activation
      .group("activation", "Activation")
        .groupHelp(
          "Which pixels move. A pixel's luma is soft-thresholded: below *Threshold* "
          "it emits **zero** motion, above it emits full motion, with *Softness* "
          "controlling the ramp width between. Lower the threshold to animate more of "
          "the frame; widen softness to feather the moving region's edges.")
      .floatField("threshold",         0.5f,  0.0f,  1.0f,  state::PrimaryInput).label("Threshold", "Thresh")
      .floatField("softness",          0.05f, 0.0f,  0.5f,  state::PrimaryInput).label("Softness", "Soft")
      // Magnitude
      .group("magnitude", "Magnitude")
        .groupHelp(
          "How fast active pixels move. *Magnitude* is the base speed; *Jitter* "
          "breaks it up with locally-smooth noise (patches breathe together instead "
          "of flickering), and *Noise Scale* sets how large those patches are.")
      .floatField("magnitude",         0.005f, 0.0f, 0.05f, state::PrimaryInput).label("Magnitude", "Mag")
      .floatField("mag_jitter",        0.5f,  0.0f,  1.0f,  state::PrimaryInput).label("Magnitude Jitter", "Jit")
      .floatField("mag_noise_scale",   16.0f, 1.0f,  100.0f, state::PrimaryInput).label("Magnitude Noise Scale", "Scale")
      // Direction — static / radial / luma gradient
      .group("direction", "Direction")
        .groupHelp(
          "The heading of each vector is a **weighted sum** of three sources, then "
          "normalised. *Rotation* is a fixed angle; *Radial* pushes outward from the "
          "*Anchor*; *Gradient* follows the image's luma slope, and *Gradient Bias* "
          "rotates it — 0° climbs uphill, ±90° runs **along** iso-luma contours for "
          "an edge-flow look. Balance the three weights to taste.")
      .floatField("rotation",          0.0f,  -360.f, 360.f, state::PrimaryInput).label("Rotation", "Rot")
      .floatField("rotation_weight",   1.0f,  0.0f,  1.0f,  state::PrimaryInput).label("Rotation Weight", "RotWt")
      // Direction — radial
      .floatField("radial_weight",     0.0f,  0.0f,  1.0f,  state::PrimaryInput).label("Radial Weight", "RadWt")
      .vec2Field ("radial_anchor",     0.5f,  0.5f,         state::PrimaryInput,
                  0.0f, 1.0f).label("Radial Anchor", "Anchor")
      // Direction — luma gradient
      .floatField("gradient_weight",   0.0f,  0.0f,  1.0f,  state::PrimaryInput).label("Gradient Weight", "GrdWt")
      .floatField("gradient_bias",     90.0f, -180.f, 180.f, state::PrimaryInput).label("Gradient Bias", "Bias")
      // Per-pixel angular jitter
      .group("jitter", "Angular Jitter")
        .groupHelp(
          "A locally-smooth swirl added on top of the heading so the field turns "
          "coherently instead of going salt-and-pepper. *Jitter* is the amount; "
          "*Noise Scale* sizes the swirls.")
      .floatField("angle_jitter",      0.0f,  0.0f,  1.0f,  state::PrimaryInput).label("Angle Jitter", "Jit")
      .floatField("angle_noise_scale", 16.0f, 1.0f,  100.0f, state::PrimaryInput).label("Angle Noise Scale", "Scale")
      // Evolution
      .group("evolution", "Time Evolution")
        .groupHelp(
          "How fast the magnitude and angle noise fields mutate. **0 freezes** the "
          "field (deterministic per pixel); higher values interpolate through fresh "
          "noise snapshots faster for an ever-shifting, chaotic flow.")
      // Hz at which the noise pattern mutates over time. 0 freezes
      // the field (deterministic per pixel); higher values stagger
      // per-cell ticks faster. At 60 Hz cells average one tick per
      // animation frame at typical refresh — fully chaotic look.
      // Replaces the old `seed` integer — schemas with a saved seed
      // will quietly drop it (no field by that name in the new
      // schema) and pick up the default evolution_rate of 0,
      // matching the old static behaviour.
      .floatField("evolution_rate",    0.0f,  0.0f,  60.0f, state::PrimaryInput).label("Evolution Rate", "Evolve")
      // Visualization (off by default — production effects are
      // expected to pass through tex_in unchanged).
      .group("viz", "Visualization")
        .groupHelp(
          "A tuning aid: overlays the motion vectors on the image as an HSV colour "
          "wheel (hue = direction, brightness = speed). *Opacity* fades it in; "
          "*Scale* exaggerates short vectors so faint motion is visible. Leave "
          "*Opacity* at 0 in production — the image otherwise passes through untouched.")
      .floatField("vis_opacity",       0.0f,  0.0f,  1.0f,  state::PrimaryInput).label("Overlay Opacity", "Opac")
      .floatField("vis_scale",         100.0f, 1.0f, 500.0f, state::PrimaryInput).label("Overlay Scale", "Scale")
      .textureField("tex_in",  state::PrimaryInput)
      .textureField("tex_out", state::PrimaryOutput)
      .renderOutputs(state::PrimaryOutput)
      // Upstream render-outputs (auxiliary input). When connected, the
      // motion shader blends our luma-driven velocity field on top of
      // the incoming one — pixels below the activation threshold
      // inherit upstream, active pixels override with this stage's
      // local velocity.
      .renderOutputs(state::PrimaryInput, "render_outputs_in")
      // Only cross-frame state is the free-running `noise_time` phase
      // accumulator (dt*evolution_rate) — a jump re-converges to a
      // noise-level-equivalent field.
      .capability(state::Capability::SeekableApproximate)
  );

  if (gpu::Device::backend() == gpu::Backend::None) return;

  state::registerShaderSPV("motion_field_color",  COLOR_SPV,  COLOR_SPV_SIZE);
  state::registerShaderSPV("motion_field_motion", MOTION_SPV, MOTION_SPV_SIZE,
                           "rgba16float", "write");
  auto cs_color  = gpu::Device::createShaderModuleByName("motion_field_color");
  auto cs_motion = gpu::Device::createShaderModuleByName("motion_field_motion");
  if (!cs_color || !cs_motion) return;

  s_pso_color = gpu::Device::createComputePSO(cs_color, "main", gpu::Bindings()
      .tex2d(0)
      .storageTex2d(1)
      .uniform(2));

  s_pso_motion = gpu::Device::createComputePSO(cs_motion, "main", gpu::Bindings()
      .tex2d(0)                                       // input texture (sampled
                                                      // for both luma + gradient)
      .storageTex2d(1, gpu::TextureFormat::RGBA16F)
      .uniform(2)
      .tex2d(3));                                     // upstream motion (zero
                                                      //  fallback when unwired)

  state::log("motion_field: module initialized");
}

// Per-instance construction: allocate State + its own GPU uniform buffer.
void* create() {
  auto* st = new State();
  st->uniform_buf = gpu::Device::createBuffer(sizeof(Uniforms), gpu::BufferUsage::Uniform);
  return st;
}

void destroy(void* self) {
  auto* st = static_cast<State*>(self);
  if (!st) return;
  st->uniform_buf.release();
  st->motion_tex.release();
  st->zero_motion_tex.release();
  delete st;
}

// Per-instance init tail: mark ready once the shared PSOs + this
// instance's uniform buffer are valid.
void init(void* self) {
  auto* st = static_cast<State*>(self);
  if (!st) return;
  st->initialized = false;
  if (!s_pso_color.valid() || !s_pso_motion.valid()) return;
  if (!st->uniform_buf.valid()) return;
  st->motion_w = 0;
  st->motion_h = 0;
  st->initialized = true;
}

void tick(void* self, double dt) {
  auto* st = static_cast<State*>(self);
  if (!st) return;
  // Advance the noise-field time accumulator. With evolution_rate=0
  // this is a no-op and the field stays put. With rate=1 we walk
  // through one integer-seed snapshot per second, interpolating
  // smoothly across the boundary via mf_value_noise_evolving.
  if (st->evolution_rate > 0.0f) {
    st->noise_time += float(dt) * st->evolution_rate;
  }
}


void on_state_patched(void* self, int n, const char* pb, const int* off,
                      const int* len, const int* ops) {
  auto* st = static_cast<State*>(self);
  if (!st) return;
  for (int i = 0; i < n; i++) {
    if (ops[i] != state::PatchReplace) continue;
    const char* path = pb + off[i];
    int plen = len[i];
    if      (state::pathIs(path, plen, "threshold"))         st->threshold = state::patchFloat(i);
    else if (state::pathIs(path, plen, "softness"))          st->softness = state::patchFloat(i);
    else if (state::pathIs(path, plen, "magnitude"))         st->magnitude = state::patchFloat(i);
    else if (state::pathIs(path, plen, "mag_jitter"))        st->mag_jitter = state::patchFloat(i);
    else if (state::pathIs(path, plen, "mag_noise_scale"))   st->mag_noise_scale = state::patchFloat(i);
    else if (state::pathIs(path, plen, "rotation"))          st->rotation_deg = state::patchFloat(i);
    else if (state::pathIs(path, plen, "rotation_weight"))   st->rotation_weight = state::patchFloat(i);
    else if (state::pathIs(path, plen, "radial_weight"))     st->radial_weight = state::patchFloat(i);
    else if (state::pathIs(path, plen, "radial_anchor")) {
      auto v = state::patchVec2(i);
      st->radial_anchor_x = v.x;
      st->radial_anchor_y = v.y;
    }
    else if (state::pathIs(path, plen, "gradient_weight"))   st->gradient_weight = state::patchFloat(i);
    else if (state::pathIs(path, plen, "gradient_bias"))     st->gradient_bias_deg = state::patchFloat(i);
    else if (state::pathIs(path, plen, "angle_jitter"))      st->angle_jitter = state::patchFloat(i);
    else if (state::pathIs(path, plen, "angle_noise_scale")) st->angle_noise_scale = state::patchFloat(i);
    else if (state::pathIs(path, plen, "evolution_rate"))    st->evolution_rate = state::patchFloat(i);
    else if (state::pathIs(path, plen, "vis_opacity"))       st->vis_opacity = state::patchFloat(i);
    else if (state::pathIs(path, plen, "vis_scale"))         st->vis_scale = state::patchFloat(i);
  }
}

void render(void* self, int vp_w, int vp_h) {
  auto* st = static_cast<State*>(self);
  if (!st || !st->initialized || vp_w <= 0 || vp_h <= 0) return;
  auto in  = gpu::Device::textureForField("tex_in");
  auto out = gpu::Device::textureForField("tex_out");
  if (!in.valid() || !out.valid()) return;

  Uniforms u = {
    st->threshold,
    st->softness,
    st->magnitude,
    st->mag_jitter,

    st->mag_noise_scale,
    st->rotation_deg * DEG2RAD,
    st->rotation_weight,
    st->radial_weight,

    st->radial_anchor_x,
    st->radial_anchor_y,
    st->gradient_weight,
    st->gradient_bias_deg * DEG2RAD,

    st->angle_jitter,
    st->angle_noise_scale,
    st->vis_opacity,
    st->vis_scale,

    // Continuous time index into the noise field. The shader's
    // mf_value_noise_evolving interpolates between adjacent integer
    // snapshots based on this; per-stream phase offsets bake-in to
    // decorrelate the magnitude and angle noise.
    st->noise_time,
    0.f, 0.f, 0.f,
  };
  st->uniform_buf.writeOne(u);

  // Pass 1 — color (identity copy + optional viz overlay).
  {
    auto cp = gpu::ComputePass::begin();
    cp.setPSO(s_pso_color);
    cp.setTexture(in,  0, 0);
    cp.setTexture(out, 1, 1);
    cp.setBuffer(st->uniform_buf, 2);
    cp.dispatch((vp_w + 7) / 8, (vp_h + 7) / 8);
    cp.end();
  }

  // Motion pass — only when something downstream actually reads the rail.
  if (!state::isOutputConnected("render_outputs")) {
    gpu::Device::submit();
    return;
  }

  // (Re)allocate the motion texture for the current viewport. Publish
  // the handle once per allocation; matches the convention in
  // motion_rect / motion_swarm / motion_static.
  if (!st->motion_tex.valid() || st->motion_w != vp_w || st->motion_h != vp_h) {
    st->motion_tex = gpu::Device::createTexture(vp_w, vp_h, gpu::TextureFormat::RGBA16F);
    st->motion_w = vp_w;
    st->motion_h = vp_h;
    if (st->motion_tex.valid()) {
      state::setGpuTexture("render_outputs/motion", st->motion_tex.id);
    }
  }
  if (!st->motion_tex.valid()) return;

  // Resolve upstream motion (or 1x1 zero fallback when unwired).
  auto upstream = gpu::Device::textureForField("render_outputs_in/motion");
  if (!upstream.valid()) {
    if (!st->zero_motion_tex.valid()) {
      st->zero_motion_tex = gpu::Device::createTexture(1, 1, gpu::TextureFormat::RGBA16F);
    }
    upstream = st->zero_motion_tex;
  }

  // Pass 2 — motion vectors.
  {
    auto cp = gpu::ComputePass::begin();
    cp.setPSO(s_pso_motion);
    cp.setTexture(in,            0, 0);
    cp.setTexture(st->motion_tex, 1, 1);
    cp.setBuffer(st->uniform_buf, 2);
    cp.setTexture(upstream,      3, 0);
    cp.dispatch((vp_w + 7) / 8, (vp_h + 7) / 8);
    cp.end();
  }

  gpu::Device::submit();
}

} // namespace motion_field
