/*
 * warp.plane_shear — analysis-driven shear / rift.
 *
 * Analyzes the input image to pick a "natural" dividing line (a plane in 2D),
 * then shears the two halves on either side of it. Four plane-finding
 * algorithms (Dominant Edge / Strongest Edge / Low-energy Seam / Content
 * Centroid), any of which can run at a FIXED angle (algorithm picks only the
 * position). The plane is STIFF: held between updates and hard-snapped (never
 * lerped) when it retargets at the configured rate. Only the shear translation
 * animates, CPU-timed (one-shot hold / ping-pong / loop, with a retrigger).
 *
 * Three GPU passes (bicolor_grad-style, all on-GPU — the host ABI has no buffer
 * readback): a coarse-grid `accumulate` scatter → a single-thread `solve` that
 * latches the line into a persistent buffer → a per-pixel `render` shear warp.
 * accumulate + solve run only on update frames; render runs every frame.
 *
 * Per-instance ABI: mutable state in `State`; the three PSOs are file-static,
 * compiled once in module_init().
 */

#include <gpu.h>
#include <host.h>
#include <effect_utils.h>
#include "plane_shear_shaders.h"

#include <cstdint>
#include <cmath>

namespace plane_shear {

// Keep in sync with common.hlsl.
static constexpr int GRID_SN    = 128;
static constexpr int NA         = 64;
static constexpr int NO         = 64;
static constexpr int GRID_BASE  = 16;
static constexpr int STATS_INTS = GRID_BASE + NA * NO;   // 4112
static constexpr int PLANE_FLOATS = 8;
static constexpr float OFF_MAX  = 2.0f;
static constexpr float TRANS_SCALE = 1.0f;               // trans=1 → one cover-half shift

enum Alg  : int { ALG_DOMINANT = 0, ALG_HOUGH = 1, ALG_SEAM = 2, ALG_PCA = 3 };
enum Anim : int { ANIM_ONESHOT = 0, ANIM_PINGPONG = 1, ANIM_LOOP = 2 };

struct AccumU  { float res_x, res_y, algorithm, off_max, aspect_x, aspect_y, _p0, _p1; };
struct SolveU  { float algorithm, lock_angle, angle_rad, off_max, latch, center_weight, _p1, _p2; };
struct RenderU { float aspect_x, aspect_y, dir, mA, mB, rift_fill, overlap_mode, debug_show,
                       edge_fill, tint, tintA_r, tintA_g, tintA_b, tintB_r, tintB_g, tintB_b,
                       tint_mode, _p0, _p1, _p2; };
static_assert(sizeof(AccumU)  == 8 * 4, "AccumU layout");
static_assert(sizeof(SolveU)  == 8 * 4, "SolveU layout");
static_assert(sizeof(RenderU) == 20 * 4, "RenderU layout");

struct State {
  // Per-instance GPU resources.
  gpu::Buffer  accum_uniform;
  gpu::Buffer  solve_uniform;
  gpu::Buffer  render_uniform;
  gpu::Buffer  stats_buf;     // int[STATS_INTS]
  gpu::Buffer  plane_buf;     // float[PLANE_FLOATS], persists → latched plane
  gpu::Sampler sampler;
  bool initialized = false;

  // --- Schema-mirrored params ---
  int   algorithm   = ALG_DOMINANT;
  float update_rate = 0.4f;
  float direction   = 0.0f;    // default: slip along the plane
  float duration    = 0.35f;
  int   anim_mode   = ANIM_ONESHOT;
  bool  retrigger   = true;
  bool  trigger_prev = false;  // rising-edge state for the "trigger now" button
  float distance    = 0.3f;    // overall shear translation (shared by both halves)
  float mult_a      = 1.0f;
  float mult_b      = 1.0f;
  bool  lock_angle  = false;
  float angle       = 0.0f;    // [-1,1] → [-90°, 90°]
  float center_weight = 0.0f;  // bias plane toward passing through the center
  int   rift_fill   = 4;       // black
  int   edge_fill   = 4;       // border reveal: black
  int   overlap_mode = 0;      // A on top
  float tint        = 0.0f;    // per-side colour tint amount
  int   tint_mode   = 0;       // 0 = multiply / 1 = add
  float tintA[3]    = {1.0f, 0.45f, 0.30f};   // side A (warm)
  float tintB[3]    = {0.30f, 0.55f, 1.0f};   // side B (cool)
  float ease_curve  = 0.0f;
  bool  debug_show_plane = false;

  // --- Runtime timing (CPU accumulators, style guide §2.1) ---
  double plane_timer = 0.0;
  double shear_phase = 0.0;
  float  shear_amt   = 0.0f;
  bool   do_update   = true;   // force analysis on the first frame
};

static gpu::ComputePSO s_pso_accum;
static gpu::ComputePSO s_pso_solve;
static gpu::ComputePSO s_pso_render;

static void apply_visibility(bool lock_angle) {
  state::setFieldHidden("angle", !lock_angle);
}

static void on_state_ready(void* self) {
  auto* s = static_cast<State*>(self);
  if (s) apply_visibility(s->lock_angle);
}

// Static (self-less) visibility evaluator — pure over a candidate state.
void eval_visibility(int n, const char* pb, const int* off, const int* len, const int* ops) {
  bool lock = false;
  for (int i = 0; i < n; i++) {
    if (ops[i] != state::PatchReplace) continue;
    if (state::pathIs(pb + off[i], len[i], "lock_angle")) lock = state::patchBool(i);
  }
  apply_visibility(lock);
}

void module_init() {
  state::init("warp.plane_shear", {1, 1, 0},
    state::Schema()
      .helpField("intro",
        "## Plane Shear\n"
        "Finds a **natural dividing line** in the input — a strong edge, a low-energy seam, "
        "the dominant grain, or the content's principal axis — and **shears the two halves** "
        "across it. The plane is *stiff*: it snaps to a fresh choice at the update rate (or "
        "only when you hit **Trigger**), never drifting. Only the translation animates.\n\n"
        "**Try:** *Direction* -1 rifts the halves apart, +1 overlaps them, 0 slips them along "
        "the plane; push *Distance* for a bigger throw. Set *Update Rate* to 0 and drive "
        "**Trigger** on the beat. Raise *Tint* to colour each side; turn on *Show Plane* to see it.")

      .group("plane", "Plane")
        .groupHelp(
          "How the dividing line is chosen. *Algorithm* picks the feature it follows; "
          "*Centering* biases the line toward the image centre; *Lock Angle* fixes the "
          "orientation so the algorithm only picks the position.")
      .selectField("algorithm", ALG_DOMINANT, state::PrimaryInput, {
        {"Dominant Edge", ALG_DOMINANT}, {"Strongest Edge", ALG_HOUGH},
        {"Low-energy Seam", ALG_SEAM}, {"Content Centroid", ALG_PCA},
      }, false, "Which natural feature the dividing line follows.").label("Algorithm", "Algo")
      .floatField("center_weight", 0.0f, 0.0f, 1.0f, state::PrimaryInput, nullptr, 0.01f, nullptr,
                  "Bias the plane toward the image centre. 0 = wherever the algorithm lands; 1 = always through the centre.").label("Centering", "Center")
      .boolField("lock_angle", false, state::PrimaryInput,
                 "Fix the plane angle; the algorithm then only picks the position.").label("Lock Angle", "Lock")
      .floatField("angle", 0.0f, -1.0f, 1.0f, state::PrimaryInput, nullptr, 0.01f, nullptr,
                  "Fixed plane angle when locked. -1..1 → -90°..90°.").label("Angle", "Angle")

      .group("motion", "Motion")
        .groupHelp(
          "How the two halves move. *Direction* morphs between rift (-1, apart), overlap "
          "(+1, together) and slip (0, sliding along the plane). *Distance* is the throw, "
          "shared by both halves; *Mult A/B* scale (and can flip) each half.")
      .floatField("direction", 0.0f, -1.0f, 1.0f, state::PrimaryInput, nullptr, 0.01f, nullptr,
                  "-1 = halves apart (rift), +1 = together (overlap), 0 = slip along the plane.").label("Direction", "Dir")
      .floatField("distance", 0.3f, 0.0f, 1.0f, state::PrimaryInput, nullptr, 0.01f, nullptr,
                  "Overall shear translation, shared by both halves (scaled per half by Mult A/B).").label("Distance", "Dist")
      .floatField("mult_a", 1.0f, -1.0f, 1.0f, state::PrimaryInput, nullptr, 0.01f, nullptr,
                  "Signed multiplier for side A's translation (negative flips it).").label("Mult A", "MulA")
      .floatField("mult_b", 1.0f, -1.0f, 1.0f, state::PrimaryInput, nullptr, 0.01f, nullptr,
                  "Signed multiplier for side B's translation (negative flips it).").label("Mult B", "MulB")

      .group("timing", "Timing")
        .groupHelp(
          "When the plane re-chooses and how the shear animates. *Update Rate* 0 = never "
          "auto-update (manual **Trigger** only). *Duration* is the ease time (0 = instant); "
          "*Anim Mode* is one-shot / ping-pong / loop; *Retrigger* restarts on retarget; "
          "*Trigger* re-chooses the plane and restarts the animation now.")
      .floatField("update_rate", 0.4f, 0.0f, 1.0f, state::PrimaryInput, nullptr, 0.01f, nullptr,
                  "How often the plane is re-chosen (0 = never, manual trigger only; higher = faster).").label("Update Rate", "Rate")
      .floatField("duration", 0.35f, 0.0f, 1.0f, state::PrimaryInput, nullptr, 0.01f, nullptr,
                  "Shear ease time. 0 = snap to max instantly.").label("Duration", "Dur")
      .selectField("anim_mode", ANIM_ONESHOT, state::PrimaryInput, {
        {"One-shot Hold", ANIM_ONESHOT}, {"Ping-pong", ANIM_PINGPONG}, {"Loop", ANIM_LOOP},
      }, false, "How the shear animates over each cycle.").label("Anim Mode", "Anim")
      .boolField("retrigger", true, state::PrimaryInput,
                 "Restart the shear animation from 0 when the plane retargets.").label("Retrigger", "Retrig")
      .eventField("trigger", state::PrimaryInput).label("Trigger", "Trig")

      .group("fills", "Fills")
        .groupHelp(
          "What fills the exposed regions. *Rift Fill* is the gap between halves pulled apart; "
          "*Edge Fill* is the viewport border a slid half reveals; *Overlap* is how halves "
          "combine where they cover each other. Rift/Edge default to solid black.")
      .selectField("rift_fill", 4, state::SecondaryInput, {
        {"Transparent", 0}, {"Original", 1}, {"Edge Stretch", 2}, {"Mirror", 3}, {"Black", 4},
      }, false, "Fills the rift gap between halves pulled apart.").label("Rift Fill", "Rift")
      .selectField("edge_fill", 4, state::SecondaryInput, {
        {"Transparent", 0}, {"Original", 1}, {"Edge Stretch", 2}, {"Mirror", 3}, {"Black", 4},
      }, false, "Fills the viewport border a slid half reveals.").label("Edge Fill", "Edge")
      .selectField("overlap_mode", 0, state::SecondaryInput, {
        {"A On Top", 0}, {"Blend", 1}, {"Additive", 2},
      }, false, "How the halves combine where they overlap.").label("Overlap", "Over")

      .group("tint", "Colour Tint")
        .groupHelp(
          "Tint each side of the plane with its own colour. *Tint* is the strength; *Tint "
          "Mode* is multiply / add / blend; *Side A/B Colour* are the two tints.")
      .floatField("tint", 0.0f, 0.0f, 1.0f, state::PrimaryInput, nullptr, 0.01f, nullptr,
                  "Colour tint amount. Each side of the plane is tinted with its own colour.").label("Tint", "Tint")
      .selectField("tint_mode", 0, state::SecondaryInput, {
        {"Multiply", 0}, {"Add", 1}, {"Blend", 2},
      }, false, "How the tint colour is combined with the image.").label("Tint Mode", "Mode")
      .rgbField("tint_a", 1.0f, 0.45f, 0.30f, state::SecondaryInput).label("Side A Colour", "A")
      .rgbField("tint_b", 0.30f, 0.55f, 1.0f, state::SecondaryInput).label("Side B Colour", "B")

      .group("tuning", "Tuning")
        .groupHelp("Fine-tuning. *Ease Curve* shapes the shear ramp.")
      .floatField("ease_curve", 0.0f, -1.0f, 1.0f, state::SecondaryInput, nullptr, 0.01f, nullptr,
                  "Shear ease shape. -1 slow-in, +1 slow-out.").label("Ease Curve", "Ease")

      .group("debug", "Debug")
        .groupHelp("Inspection aids. *Show Plane* overlays the dividing line.")
      .boolField("debug_show_plane", false, state::SecondaryInput,
                 "Overlay the dividing line.").label("Show Plane", "Plane")
      .endGroup()
      // ---- I/O ----
      .textureField("tex_in",  state::PrimaryInput)
      .textureField("tex_out", state::PrimaryOutput)
      .capability(state::Capability::SeekableApproximate)
  );

  if (gpu::Device::backend() == gpu::Backend::None) return;

  state::registerShaderSPV("plane_shear_accumulate", ACCUMULATE_SPV, ACCUMULATE_SPV_SIZE);
  state::registerShaderSPV("plane_shear_solve",      SOLVE_SPV,      SOLVE_SPV_SIZE);
  state::registerShaderSPV("plane_shear_render",     RENDER_SPV,     RENDER_SPV_SIZE);

  auto cs_accum = gpu::Device::createShaderModuleByName("plane_shear_accumulate");
  auto cs_solve = gpu::Device::createShaderModuleByName("plane_shear_solve");
  auto cs_render = gpu::Device::createShaderModuleByName("plane_shear_render");
  if (!cs_accum || !cs_solve || !cs_render) return;

  s_pso_accum = gpu::Device::createComputePSO(cs_accum, "main", gpu::Bindings()
      .tex2d(0).sampler(1).storageRW(2).uniform(3));
  s_pso_solve = gpu::Device::createComputePSO(cs_solve, "main", gpu::Bindings()
      .storage(0).storageRW(1).uniform(2));
  s_pso_render = gpu::Device::createComputePSO(cs_render, "main", gpu::Bindings()
      .tex2d(0).sampler(1).storage(2).storageTex2d(3).uniform(4));

  state::log("plane_shear: module initialized");
}

void* create() {
  auto* s = new State();
  s->accum_uniform  = gpu::Device::createBuffer(sizeof(AccumU),  gpu::BufferUsage::Uniform);
  s->solve_uniform  = gpu::Device::createBuffer(sizeof(SolveU),  gpu::BufferUsage::Uniform);
  s->render_uniform = gpu::Device::createBuffer(sizeof(RenderU), gpu::BufferUsage::Uniform);
  s->stats_buf      = gpu::Device::createBuffer(STATS_INTS * sizeof(int32_t),  gpu::BufferUsage::Storage);
  s->plane_buf      = gpu::Device::createBuffer(PLANE_FLOATS * sizeof(float),  gpu::BufferUsage::Storage);
  s->sampler        = gpu::Device::createSampler(gpu::FilterMode::Linear, gpu::AddressMode::ClampToEdge);
  return s;
}

void destroy(void* self) {
  auto* s = static_cast<State*>(self);
  if (!s) return;
  s->accum_uniform.release();
  s->solve_uniform.release();
  s->render_uniform.release();
  s->stats_buf.release();
  s->plane_buf.release();
  s->sampler.release();
  delete s;
}

void init(void* self) {
  auto* s = static_cast<State*>(self);
  if (!s) return;
  if (!s_pso_accum.valid() || !s_pso_solve.valid() || !s_pso_render.valid()) return;
  if (!s->plane_buf.valid()) return;

  // Seed a default centered vertical plane, flag not-yet-analyzed. The first
  // frame forces an update so a real plane lands immediately.
  float seed[PLANE_FLOATS] = {0};
  seed[0] = 0.0f; seed[1] = 0.0f;   // center
  seed[2] = 1.0f; seed[3] = 0.0f;   // normal (+x → a vertical dividing line)
  seed[4] = 0.0f;                   // confidence
  seed[5] = 0.0f;                   // initialized
  s->plane_buf.write(seed, PLANE_FLOATS);

  s->plane_timer = 0.0;
  s->shear_phase = 0.0;
  s->shear_amt   = 0.0f;
  s->do_update   = true;
  s->trigger_prev = false;
  s->initialized = true;

  state::setOnStateReady(&on_state_ready);
}

static float updateInterval(float slider) {
  // Slider 0 → 8 s (slow), 1 → 0.05 s (fast); exp-mapped (style guide §1.3).
  return 8.0f * std::pow(0.05f / 8.0f, slider);
}
static float durationSeconds(float slider) {
  // 0 → instant; quadratic feel up to ~4 s.
  return slider * slider * 4.0f;
}

void tick(void* self, double dt) {
  auto* s = static_cast<State*>(self);
  if (!s || !s->initialized) return;
  if (dt < 0.0) dt = 0.0;
  if (dt > 0.25) dt = 0.25;   // clamp starved frames

  // --- Plane update metronome (stiff: snap, never lerp). update_rate 0 = never
  //     auto-update (manual trigger only). ---
  s->plane_timer += dt;
  float interval = updateInterval(s->update_rate);
  if (!s->do_update && s->update_rate > 1e-4f && s->plane_timer >= interval) {
    s->do_update = true;
    s->plane_timer = 0.0;
    if (s->retrigger) s->shear_phase = 0.0;
  }

  // --- Shear animation phase (CPU accumulator) ---
  s->shear_phase += dt;
  float dur = durationSeconds(s->duration);
  float amt;
  if (dur <= 1e-4f) {
    amt = 1.0f;                                    // instant
  } else if (s->anim_mode == ANIM_ONESHOT) {
    amt = (float)std::fmin(s->shear_phase / dur, 1.0);   // ramp then hold
  } else if (s->anim_mode == ANIM_LOOP) {
    amt = (float)(std::fmod(s->shear_phase, (double)dur) / dur);  // saw 0→1
  } else {                                         // ping-pong
    double period = 2.0 * dur;
    double tp = std::fmod(s->shear_phase, period);
    amt = (float)((tp < dur) ? tp / dur : (period - tp) / dur);
  }
  // Ease shape (amt is already in [0,1]).
  s->shear_amt = std::pow(std::fmin(std::fmax(amt, 0.0f), 1.0f),
                          fx::signedSliderToExp(s->ease_curve));
}

void on_resolume_param(void*, long long, double) {}

void on_state_patched(void* self, int n, const char* pb, const int* off,
                      const int* len, const int* ops) {
  auto* s = static_cast<State*>(self);
  if (!s) return;
  bool vis_dirty = false;
  for (int i = 0; i < n; i++) {
    if (ops[i] != state::PatchReplace) continue;
    const char* p = pb + off[i]; int l = len[i];
    if      (state::pathIs(p, l, "algorithm"))    { s->algorithm = state::patchInt(i); s->do_update = true; }
    else if (state::pathIs(p, l, "update_rate"))  s->update_rate = state::patchFloat(i);
    else if (state::pathIs(p, l, "direction"))    s->direction = state::patchFloat(i);
    else if (state::pathIs(p, l, "duration"))     s->duration = state::patchFloat(i);
    else if (state::pathIs(p, l, "anim_mode"))    s->anim_mode = state::patchInt(i);
    else if (state::pathIs(p, l, "retrigger"))    s->retrigger = state::patchBool(i);
    else if (state::pathIs(p, l, "trigger")) {
      // "Trigger now" button — rising-edge (0→1) re-chooses the plane AND restarts
      // the shear animation. Replay-safe: constant replays never re-fire (§8.2).
      bool tval = state::patchFloat(i) != 0.0f;
      if (tval && !s->trigger_prev) { s->shear_phase = 0.0; s->do_update = true; }
      s->trigger_prev = tval;
    }
    else if (state::pathIs(p, l, "distance"))     s->distance = state::patchFloat(i);
    else if (state::pathIs(p, l, "mult_a"))       s->mult_a = state::patchFloat(i);
    else if (state::pathIs(p, l, "mult_b"))       s->mult_b = state::patchFloat(i);
    else if (state::pathIs(p, l, "lock_angle"))   { s->lock_angle = state::patchBool(i); s->do_update = true; vis_dirty = true; }
    else if (state::pathIs(p, l, "angle"))        { s->angle = state::patchFloat(i); if (s->lock_angle) s->do_update = true; }
    else if (state::pathIs(p, l, "center_weight")) { s->center_weight = state::patchFloat(i); s->do_update = true; }
    else if (state::pathIs(p, l, "rift_fill"))    s->rift_fill = state::patchInt(i);
    else if (state::pathIs(p, l, "edge_fill"))    s->edge_fill = state::patchInt(i);
    else if (state::pathIs(p, l, "overlap_mode")) s->overlap_mode = state::patchInt(i);
    else if (state::pathIs(p, l, "tint"))         s->tint = state::patchFloat(i);
    else if (state::pathIs(p, l, "tint_mode"))    s->tint_mode = state::patchInt(i);
    else if (state::pathIs(p, l, "tint_a"))       { auto v = state::patchVec3(i); s->tintA[0]=v.x; s->tintA[1]=v.y; s->tintA[2]=v.z; }
    else if (state::pathIs(p, l, "tint_b"))       { auto v = state::patchVec3(i); s->tintB[0]=v.x; s->tintB[1]=v.y; s->tintB[2]=v.z; }
    else if (state::pathIs(p, l, "ease_curve"))   s->ease_curve = state::patchFloat(i);
    else if (state::pathIs(p, l, "debug_show_plane")) s->debug_show_plane = state::patchBool(i);
  }
  if (vis_dirty) apply_visibility(s->lock_angle);
}

void render(void* self, int vp_w, int vp_h) {
  auto* s = static_cast<State*>(self);
  if (!s || !s->initialized || vp_w <= 0 || vp_h <= 0) return;

  auto in  = gpu::Device::textureForField("tex_in");
  auto out = gpu::Device::textureForField("tex_out");
  if (!in.valid() || !out.valid()) return;

  auto cover = fx::coverSquare(vp_w, vp_h);

  // --- Analysis (only on update frames) — snap the plane, stiff. ---
  if (s->do_update) {
    int32_t zeros[STATS_INTS] = {0};
    s->stats_buf.write(zeros, STATS_INTS);

    AccumU au = {};
    au.res_x = (float)vp_w; au.res_y = (float)vp_h;
    au.algorithm = (float)s->algorithm; au.off_max = OFF_MAX;
    au.aspect_x = cover.ax; au.aspect_y = cover.ay;
    s->accum_uniform.writeOne(au);

    SolveU su = {};
    su.algorithm = (float)s->algorithm;
    su.lock_angle = s->lock_angle ? 1.0f : 0.0f;
    su.angle_rad = s->angle * 1.57079632679f;   // [-1,1] → [-π/2, π/2]
    su.off_max = OFF_MAX;
    su.latch = 1.0f;
    su.center_weight = s->center_weight;
    s->solve_uniform.writeOne(su);

    int sg = (GRID_SN + 7) / 8;
    {
      auto cp = gpu::ComputePass::begin();
      cp.setPSO(s_pso_accum);
      cp.setTexture(in, 0, 0);
      cp.setSampler(s->sampler, 1);
      cp.setBuffer(s->stats_buf, 2);
      cp.setBuffer(s->accum_uniform, 3);
      cp.dispatch(sg, sg);
      cp.end();
    }
    {
      auto cp = gpu::ComputePass::begin();
      cp.setPSO(s_pso_solve);
      cp.setBuffer(s->stats_buf, 0);
      cp.setBuffer(s->plane_buf, 1);
      cp.setBuffer(s->solve_uniform, 2);
      cp.dispatch(1, 1, 1);
      cp.end();
    }
    s->do_update = false;
  }

  // --- Shear warp (every frame). ---
  float mA = s->distance * s->mult_a * s->shear_amt * TRANS_SCALE;
  float mB = s->distance * s->mult_b * s->shear_amt * TRANS_SCALE;

  RenderU ru = {};
  ru.aspect_x = cover.ax; ru.aspect_y = cover.ay;
  ru.dir = s->direction;
  ru.mA = mA; ru.mB = mB;
  ru.rift_fill = (float)s->rift_fill;
  ru.overlap_mode = (float)s->overlap_mode;
  ru.debug_show = s->debug_show_plane ? 1.0f : 0.0f;
  ru.edge_fill = (float)s->edge_fill;
  ru.tint = s->tint;
  ru.tint_mode = (float)s->tint_mode;
  ru.tintA_r = s->tintA[0]; ru.tintA_g = s->tintA[1]; ru.tintA_b = s->tintA[2];
  ru.tintB_r = s->tintB[0]; ru.tintB_g = s->tintB[1]; ru.tintB_b = s->tintB[2];
  s->render_uniform.writeOne(ru);

  {
    auto cp = gpu::ComputePass::begin();
    cp.setPSO(s_pso_render);
    cp.setTexture(in, 0, 0);
    cp.setSampler(s->sampler, 1);
    cp.setBuffer(s->plane_buf, 2);
    cp.setTexture(out, 3, 1);
    cp.setBuffer(s->render_uniform, 4);
    cp.dispatch((vp_w + 7) / 8, (vp_h + 7) / 8);
    cp.end();
  }

  gpu::Device::submit();
}

} // namespace plane_shear
