/*
 * warp.tri_shear — three-plane triangle shear.
 *
 * Discovers THREE natural lines biased to form a large triangle (a `size` param
 * weights the large-area reward), then shears the image by CHAINING the single-
 * plane shear three times — once per triangle edge (in → tmpA → tmpB → out). The
 * triangle is STIFF: held between updates and hard-snapped when it retargets.
 *
 * Reuses plane_shear's accumulate (the (angle,offset) energy grid) and the exact
 * shear/fill render math; adds a 3-line `solve` and the chained ping-pong render.
 * All on-GPU (no buffer readback). Per-instance ABI; PSOs file-static.
 */

#include <gpu.h>
#include <host.h>
#include <effect_utils.h>
#include "tri_shear_shaders.h"

#include <cstdint>
#include <cmath>

namespace tri_shear {

// Keep in sync with plane_shear/common.hlsl (shared accumulate grid).
static constexpr int GRID_SN    = 128;
static constexpr int NA         = 64;
static constexpr int NO         = 64;
static constexpr int GRID_BASE  = 16;
static constexpr int STATS_INTS = GRID_BASE + NA * NO;   // 4112
static constexpr int TRI_FLOATS = 16;                    // 3 lines × 4 + init + pad
static constexpr float OFF_MAX  = 2.0f;
static constexpr float TRANS_SCALE = 1.0f;
static constexpr float TWO_PI_3 = 2.0943951023931953f;

// Match the accumulate/solve grid semantics: Hough(1) = strongest, Seam(2) = low-energy.
enum Alg  : int { ALG_STRONGEST = 1, ALG_SEAM = 2 };
enum Anim : int { ANIM_ONESHOT = 0, ANIM_PINGPONG = 1, ANIM_LOOP = 2 };

struct AccumU  { float res_x, res_y, algorithm, off_max, aspect_x, aspect_y, _p0, _p1; };
struct SolveU  { float algorithm, size, off_max, latch, obliqueness, _p1, _p2, _p3; };
struct RenderU { float aspect_x, aspect_y, dir, mA, mB, rift_fill, overlap_mode, debug_show,
                       edge_fill, line_index, tint,
                       tint0_r, tint0_g, tint0_b, tint1_r, tint1_g, tint1_b,
                       tint2_r, tint2_g, tint2_b, tintC_r, tintC_g, tintC_b, tint_mode; };
static_assert(sizeof(AccumU)  == 8 * 4, "AccumU layout");
static_assert(sizeof(SolveU)  == 8 * 4, "SolveU layout");
static_assert(sizeof(RenderU) == 24 * 4, "RenderU layout");

struct State {
  gpu::Buffer  accum_uniform;
  gpu::Buffer  solve_uniform;
  gpu::Buffer  render_uniform[3];   // one per chained pass (line_index differs)
  gpu::Buffer  stats_buf;
  gpu::Buffer  tri_buf;             // 3 lines, persists → latched triangle
  gpu::Sampler sampler;
  gpu::Texture tmpA, tmpB;          // ping-pong scratch
  int   tmp_w = 0, tmp_h = 0;
  bool  initialized = false;

  // --- Schema-mirrored params ---
  int   algorithm   = ALG_STRONGEST;
  float size        = 0.3f;      // triangle scale (small target incircle radius)
  float obliqueness = 1.0f;      // 0 = equilateral → 1 = freely oblique
  float obliqueness_jitter = 0.0f; // per-update random spread around `obliqueness`
  uint32_t rng = 0x9E3779B9u;    // LCG for the jitter roll
  float update_rate = 0.4f;
  float direction   = 0.0f;
  float duration    = 0.35f;
  int   anim_mode   = ANIM_ONESHOT;
  bool  retrigger   = true;
  bool  trigger_prev = false;
  float distance    = 0.3f;
  float mult_a      = 1.0f;
  float mult_b      = 1.0f;
  int   rift_fill   = 4;         // black
  int   edge_fill   = 4;         // black
  int   overlap_mode = 0;
  float tint        = 0.0f;      // per-region colour tint amount
  int   tint_mode   = 0;         // 0 = multiply / 1 = add
  float tint0[3]    = {1.0f, 0.30f, 0.30f};   // wedge 0 (red)
  float tint1[3]    = {0.30f, 1.0f, 0.40f};   // wedge 1 (green)
  float tint2[3]    = {0.40f, 0.50f, 1.0f};   // wedge 2 (blue)
  float tintC[3]    = {1.0f, 1.0f, 1.0f};     // center (neutral)
  float ease_curve  = 0.0f;
  bool  debug_show_plane = false;

  // --- Runtime timing ---
  double plane_timer = 0.0;
  double shear_phase = 0.0;
  float  shear_amt   = 0.0f;
  bool   do_update   = true;
};

static gpu::ComputePSO s_pso_accum;
static gpu::ComputePSO s_pso_solve;
static gpu::ComputePSO s_pso_render;

void module_init() {
  state::init("warp.tri_shear", {1, 0, 0},
    state::Schema()
      .helpField("intro",
        "## Triangle Shear\n"
        "Discovers **three natural lines forming a triangle** and shears the image by chaining "
        "the single-plane shear once per edge. The triangle follows the input's strongest edges "
        "(or lowest-energy seams), sized and shaped by you. It is *stiff*: it snaps to a fresh "
        "triangle at the update rate (or on **Trigger**), never drifting.\n\n"
        "**Try:** raise *Obliqueness* for scalene triangles and *Obliqueness Jitter* to reshuffle "
        "each update; *Direction* -1 rifts, +1 overlaps, 0 slips. Set *Update Rate* to 0 and drive "
        "**Trigger** on the beat. Raise *Tint* to colour the three wedges + centre.")

      .group("triangle", "Triangle")
        .groupHelp(
          "How the three lines are found. *Algorithm* picks the feature they follow; *Size* is "
          "the target scale (small = a tight central triangle); *Obliqueness* frees each edge to "
          "tilt to features (0 = equilateral); *Obliqueness Jitter* rolls a fresh obliqueness "
          "around that value on every update.")
      .selectField("algorithm", ALG_STRONGEST, state::PrimaryInput, {
        {"Strongest Edges", ALG_STRONGEST}, {"Low-energy Seams", ALG_SEAM},
      }, false, "Which feature the three edges follow.").label("Algorithm", "Algo")
      .floatField("size", 0.3f, 0.0f, 1.0f, state::PrimaryInput, nullptr, 0.01f, nullptr,
                  "Triangle scale (target incircle radius). Small values → tight central triangle.").label("Size", "Size")
      .floatField("obliqueness", 1.0f, 0.0f, 1.0f, state::PrimaryInput, nullptr, 0.01f, nullptr,
                  "0 = equilateral (edges locked 120° apart); 1 = each edge tilts freely to features (oblique / scalene).").label("Obliqueness", "Obliq")
      .floatField("obliqueness_jitter", 0.0f, 0.0f, 1.0f, state::PrimaryInput, nullptr, 0.01f, nullptr,
                  "Each update, roll a fresh effective obliqueness around the centre value by ± this much.").label("Obliqueness Jitter", "Jit")

      .group("motion", "Motion")
        .groupHelp(
          "How the halves of each edge move. *Direction* morphs between rift (-1, apart), overlap "
          "(+1, together) and slip (0, along the edge). *Distance* is the throw shared by all three "
          "edges; *Mult A/B* scale (and can flip) each side.")
      .floatField("direction", 0.0f, -1.0f, 1.0f, state::PrimaryInput, nullptr, 0.01f, nullptr,
                  "-1 = halves apart (rift), +1 = together (overlap), 0 = slip along each edge.").label("Direction", "Dir")
      .floatField("distance", 0.3f, 0.0f, 1.0f, state::PrimaryInput, nullptr, 0.01f, nullptr,
                  "Overall shear translation, shared by all three edges.").label("Distance", "Dist")
      .floatField("mult_a", 1.0f, -1.0f, 1.0f, state::PrimaryInput, nullptr, 0.01f, nullptr,
                  "Signed multiplier for side A's translation (negative flips it).").label("Mult A", "MulA")
      .floatField("mult_b", 1.0f, -1.0f, 1.0f, state::PrimaryInput, nullptr, 0.01f, nullptr,
                  "Signed multiplier for side B's translation (negative flips it).").label("Mult B", "MulB")

      .group("timing", "Timing")
        .groupHelp(
          "When the triangle re-chooses and how the shear animates. *Update Rate* 0 = never "
          "auto-update (manual **Trigger** only). *Duration* is the ease time (0 = instant); "
          "*Anim Mode* is one-shot / ping-pong / loop; *Retrigger* restarts on retarget; "
          "*Trigger* re-chooses the triangle and restarts the animation now.")
      .floatField("update_rate", 0.4f, 0.0f, 1.0f, state::PrimaryInput, nullptr, 0.01f, nullptr,
                  "How often the triangle is re-chosen (0 = never, manual trigger only; higher = faster).").label("Update Rate", "Rate")
      .floatField("duration", 0.35f, 0.0f, 1.0f, state::PrimaryInput, nullptr, 0.01f, nullptr,
                  "Shear ease time. 0 = snap to max instantly.").label("Duration", "Dur")
      .selectField("anim_mode", ANIM_ONESHOT, state::PrimaryInput, {
        {"One-shot Hold", ANIM_ONESHOT}, {"Ping-pong", ANIM_PINGPONG}, {"Loop", ANIM_LOOP},
      }, false, "How the shear animates over each cycle.").label("Anim Mode", "Anim")
      .boolField("retrigger", true, state::PrimaryInput,
                 "Restart the shear animation from 0 when the triangle retargets.").label("Retrigger", "Retrig")
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
          "Tint each triangle region with its own colour. *Tint* is the strength; *Tint Mode* is "
          "multiply / add / blend; *Wedge 0/1/2* are the three outer regions and *Centre* is "
          "inside the triangle.")
      .floatField("tint", 0.0f, 0.0f, 1.0f, state::PrimaryInput, nullptr, 0.01f, nullptr,
                  "Colour tint amount. Each of the 3 outer wedges and the centre is tinted with its own colour.").label("Tint", "Tint")
      .selectField("tint_mode", 0, state::SecondaryInput, {
        {"Multiply", 0}, {"Add", 1}, {"Blend", 2},
      }, false, "How the tint colour is combined with the image.").label("Tint Mode", "Mode")
      .rgbField("tint_0",      1.0f, 0.30f, 0.30f, state::SecondaryInput).label("Wedge 0 Colour", "W0")
      .rgbField("tint_1",      0.30f, 1.0f, 0.40f, state::SecondaryInput).label("Wedge 1 Colour", "W1")
      .rgbField("tint_2",      0.40f, 0.50f, 1.0f, state::SecondaryInput).label("Wedge 2 Colour", "W2")
      .rgbField("tint_center", 1.0f, 1.0f, 1.0f, state::SecondaryInput).label("Centre Colour", "Ctr")

      .group("tuning", "Tuning")
        .groupHelp("Fine-tuning. *Ease Curve* shapes the shear ramp.")
      .floatField("ease_curve", 0.0f, -1.0f, 1.0f, state::SecondaryInput, nullptr, 0.01f, nullptr,
                  "Shear ease shape. -1 slow-in, +1 slow-out.").label("Ease Curve", "Ease")

      .group("debug", "Debug")
        .groupHelp("Inspection aids. *Show Triangle* overlays the three discovered edges.")
      .boolField("debug_show_plane", false, state::SecondaryInput,
                 "Overlay the three discovered edge lines.").label("Show Triangle", "Tri")
      .endGroup()
      .capability(state::Capability::SeekableApproximate)
      .textureField("tex_in",  state::PrimaryInput)
      .textureField("tex_out", state::PrimaryOutput)
  );

  if (gpu::Device::backend() == gpu::Backend::None) return;

  state::registerShaderSPV("tri_shear_accumulate", ACCUMULATE_SPV, ACCUMULATE_SPV_SIZE);
  state::registerShaderSPV("tri_shear_solve",      SOLVE_SPV,      SOLVE_SPV_SIZE);
  state::registerShaderSPV("tri_shear_render",     RENDER_SPV,     RENDER_SPV_SIZE);

  auto cs_accum  = gpu::Device::createShaderModuleByName("tri_shear_accumulate");
  auto cs_solve  = gpu::Device::createShaderModuleByName("tri_shear_solve");
  auto cs_render = gpu::Device::createShaderModuleByName("tri_shear_render");
  if (!cs_accum || !cs_solve || !cs_render) return;

  s_pso_accum = gpu::Device::createComputePSO(cs_accum, "main", gpu::Bindings()
      .tex2d(0).sampler(1).storageRW(2).uniform(3));
  s_pso_solve = gpu::Device::createComputePSO(cs_solve, "main", gpu::Bindings()
      .storage(0).storageRW(1).uniform(2));
  s_pso_render = gpu::Device::createComputePSO(cs_render, "main", gpu::Bindings()
      .tex2d(0).sampler(1).storage(2).storageTex2d(3).uniform(4));

  state::log("tri_shear: module initialized");
}

void* create() {
  auto* s = new State();
  s->accum_uniform  = gpu::Device::createBuffer(sizeof(AccumU),  gpu::BufferUsage::Uniform);
  s->solve_uniform  = gpu::Device::createBuffer(sizeof(SolveU),  gpu::BufferUsage::Uniform);
  for (int k = 0; k < 3; k++)
    s->render_uniform[k] = gpu::Device::createBuffer(sizeof(RenderU), gpu::BufferUsage::Uniform);
  s->stats_buf = gpu::Device::createBuffer(STATS_INTS * sizeof(int32_t), gpu::BufferUsage::Storage);
  s->tri_buf   = gpu::Device::createBuffer(TRI_FLOATS * sizeof(float),   gpu::BufferUsage::Storage);
  s->sampler   = gpu::Device::createSampler(gpu::FilterMode::Linear, gpu::AddressMode::ClampToEdge);
  return s;
}

void destroy(void* self) {
  auto* s = static_cast<State*>(self);
  if (!s) return;
  s->accum_uniform.release();
  s->solve_uniform.release();
  for (int k = 0; k < 3; k++) s->render_uniform[k].release();
  s->stats_buf.release();
  s->tri_buf.release();
  s->sampler.release();
  if (s->tmpA.valid()) s->tmpA.release();
  if (s->tmpB.valid()) s->tmpB.release();
  delete s;
}

void init(void* self) {
  auto* s = static_cast<State*>(self);
  if (!s) return;
  if (!s_pso_accum.valid() || !s_pso_solve.valid() || !s_pso_render.valid()) return;
  if (!s->tri_buf.valid()) return;

  // Seed a default centered equilateral triangle so the pre-analysis frame is
  // sane; the first frame forces a real update anyway.
  float seed[TRI_FLOATS] = {0};
  for (int k = 0; k < 3; k++) {
    float phi = 1.5707963f + k * TWO_PI_3;   // 90°, 210°, 330°
    float nx = std::cos(phi), ny = std::sin(phi);
    seed[k * 4 + 0] = 0.5f * nx; seed[k * 4 + 1] = 0.5f * ny;
    seed[k * 4 + 2] = nx;        seed[k * 4 + 3] = ny;
  }
  seed[12] = 0.0f;
  s->tri_buf.write(seed, TRI_FLOATS);

  s->plane_timer = 0.0;
  s->shear_phase = 0.0;
  s->shear_amt   = 0.0f;
  s->do_update   = true;
  s->trigger_prev = false;
  s->initialized = true;
}

static float rand_unit(State* s) {   // LCG → [0, 1)
  s->rng = s->rng * 1664525u + 1013904223u;
  return (float)((s->rng >> 8) & 0xFFFFFFu) * (1.0f / 16777216.0f);
}

static float updateInterval(float slider) {
  return 8.0f * std::pow(0.05f / 8.0f, slider);
}
static float durationSeconds(float slider) {
  return slider * slider * 4.0f;
}

void tick(void* self, double dt) {
  auto* s = static_cast<State*>(self);
  if (!s || !s->initialized) return;
  if (dt < 0.0) dt = 0.0;
  if (dt > 0.25) dt = 0.25;

  s->plane_timer += dt;
  float interval = updateInterval(s->update_rate);
  if (!s->do_update && s->update_rate > 1e-4f && s->plane_timer >= interval) {
    s->do_update = true;
    s->plane_timer = 0.0;
    if (s->retrigger) s->shear_phase = 0.0;
  }

  s->shear_phase += dt;
  float dur = durationSeconds(s->duration);
  float amt;
  if (dur <= 1e-4f) {
    amt = 1.0f;
  } else if (s->anim_mode == ANIM_ONESHOT) {
    amt = (float)std::fmin(s->shear_phase / dur, 1.0);
  } else if (s->anim_mode == ANIM_LOOP) {
    amt = (float)(std::fmod(s->shear_phase, (double)dur) / dur);
  } else {
    double period = 2.0 * dur;
    double tp = std::fmod(s->shear_phase, period);
    amt = (float)((tp < dur) ? tp / dur : (period - tp) / dur);
  }
  s->shear_amt = std::pow(std::fmin(std::fmax(amt, 0.0f), 1.0f),
                          fx::signedSliderToExp(s->ease_curve));
}

void on_resolume_param(void*, long long, double) {}

void on_state_patched(void* self, int n, const char* pb, const int* off,
                      const int* len, const int* ops) {
  auto* s = static_cast<State*>(self);
  if (!s) return;
  for (int i = 0; i < n; i++) {
    if (ops[i] != state::PatchReplace) continue;
    const char* p = pb + off[i]; int l = len[i];
    if      (state::pathIs(p, l, "algorithm"))    { s->algorithm = state::patchInt(i); s->do_update = true; }
    else if (state::pathIs(p, l, "size"))         { s->size = state::patchFloat(i); s->do_update = true; }
    else if (state::pathIs(p, l, "obliqueness"))  { s->obliqueness = state::patchFloat(i); s->do_update = true; }
    else if (state::pathIs(p, l, "obliqueness_jitter")) { s->obliqueness_jitter = state::patchFloat(i); s->do_update = true; }
    else if (state::pathIs(p, l, "update_rate"))  s->update_rate = state::patchFloat(i);
    else if (state::pathIs(p, l, "direction"))    s->direction = state::patchFloat(i);
    else if (state::pathIs(p, l, "duration"))     s->duration = state::patchFloat(i);
    else if (state::pathIs(p, l, "anim_mode"))    s->anim_mode = state::patchInt(i);
    else if (state::pathIs(p, l, "retrigger"))    s->retrigger = state::patchBool(i);
    else if (state::pathIs(p, l, "trigger")) {
      // Rising-edge (0→1) re-chooses the triangle AND restarts the shear animation.
      bool tval = state::patchFloat(i) != 0.0f;
      if (tval && !s->trigger_prev) { s->shear_phase = 0.0; s->do_update = true; }
      s->trigger_prev = tval;
    }
    else if (state::pathIs(p, l, "distance"))     s->distance = state::patchFloat(i);
    else if (state::pathIs(p, l, "mult_a"))       s->mult_a = state::patchFloat(i);
    else if (state::pathIs(p, l, "mult_b"))       s->mult_b = state::patchFloat(i);
    else if (state::pathIs(p, l, "rift_fill"))    s->rift_fill = state::patchInt(i);
    else if (state::pathIs(p, l, "edge_fill"))    s->edge_fill = state::patchInt(i);
    else if (state::pathIs(p, l, "overlap_mode")) s->overlap_mode = state::patchInt(i);
    else if (state::pathIs(p, l, "tint"))         s->tint = state::patchFloat(i);
    else if (state::pathIs(p, l, "tint_mode"))    s->tint_mode = state::patchInt(i);
    else if (state::pathIs(p, l, "tint_0"))       { auto v = state::patchVec3(i); s->tint0[0]=v.x; s->tint0[1]=v.y; s->tint0[2]=v.z; }
    else if (state::pathIs(p, l, "tint_1"))       { auto v = state::patchVec3(i); s->tint1[0]=v.x; s->tint1[1]=v.y; s->tint1[2]=v.z; }
    else if (state::pathIs(p, l, "tint_2"))       { auto v = state::patchVec3(i); s->tint2[0]=v.x; s->tint2[1]=v.y; s->tint2[2]=v.z; }
    else if (state::pathIs(p, l, "tint_center"))  { auto v = state::patchVec3(i); s->tintC[0]=v.x; s->tintC[1]=v.y; s->tintC[2]=v.z; }
    else if (state::pathIs(p, l, "ease_curve"))   s->ease_curve = state::patchFloat(i);
    else if (state::pathIs(p, l, "debug_show_plane")) s->debug_show_plane = state::patchBool(i);
  }
}

void render(void* self, int vp_w, int vp_h) {
  auto* s = static_cast<State*>(self);
  if (!s || !s->initialized || vp_w <= 0 || vp_h <= 0) return;

  auto in  = gpu::Device::textureForField("tex_in");
  auto out = gpu::Device::textureForField("tex_out");
  if (!in.valid() || !out.valid()) return;

  auto cover = fx::coverSquare(vp_w, vp_h);

  // (Re)allocate ping-pong scratch to the viewport size.
  if (s->tmp_w != vp_w || s->tmp_h != vp_h) {
    if (s->tmpA.valid()) s->tmpA.release();
    if (s->tmpB.valid()) s->tmpB.release();
    s->tmpA = gpu::Device::createTexture(vp_w, vp_h);
    s->tmpB = gpu::Device::createTexture(vp_w, vp_h);
    s->tmp_w = vp_w; s->tmp_h = vp_h;
  }
  if (!s->tmpA.valid() || !s->tmpB.valid()) return;

  // --- Analysis (only on update frames) — snap the triangle, stiff. ---
  if (s->do_update) {
    int32_t zeros[STATS_INTS] = {0};
    s->stats_buf.write(zeros, STATS_INTS);

    AccumU au = {};
    au.res_x = (float)vp_w; au.res_y = (float)vp_h;
    au.algorithm = (float)s->algorithm; au.off_max = OFF_MAX;
    au.aspect_x = cover.ax; au.aspect_y = cover.ay;
    s->accum_uniform.writeOne(au);

    // Roll a fresh effective obliqueness around the center for this update.
    float u = rand_unit(s);
    float eff_ob = s->obliqueness + (u * 2.0f - 1.0f) * s->obliqueness_jitter;
    eff_ob = eff_ob < 0.0f ? 0.0f : (eff_ob > 1.0f ? 1.0f : eff_ob);

    SolveU su = {};
    su.algorithm = (float)s->algorithm;
    su.size = s->size;
    su.off_max = OFF_MAX;
    su.latch = 1.0f;
    su.obliqueness = eff_ob;
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
      cp.setBuffer(s->tri_buf, 1);
      cp.setBuffer(s->solve_uniform, 2);
      cp.dispatch(1, 1, 1);
      cp.end();
    }
    s->do_update = false;
  }

  // --- Three chained shears: in → tmpA → tmpB → out (one edge each). ---
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
  ru.tint0_r = s->tint0[0]; ru.tint0_g = s->tint0[1]; ru.tint0_b = s->tint0[2];
  ru.tint1_r = s->tint1[0]; ru.tint1_g = s->tint1[1]; ru.tint1_b = s->tint1[2];
  ru.tint2_r = s->tint2[0]; ru.tint2_g = s->tint2[1]; ru.tint2_b = s->tint2[2];
  ru.tintC_r = s->tintC[0]; ru.tintC_g = s->tintC[1]; ru.tintC_b = s->tintC[2];
  for (int k = 0; k < 3; k++) {
    ru.line_index = (float)k;                 // distinct buffer per pass — one submit
    s->render_uniform[k].writeOne(ru);
  }

  gpu::Texture srcs[3] = { in, s->tmpA, s->tmpB };
  gpu::Texture dsts[3] = { s->tmpA, s->tmpB, out };
  for (int k = 0; k < 3; k++) {
    auto cp = gpu::ComputePass::begin();
    cp.setPSO(s_pso_render);
    cp.setTexture(srcs[k], 0, 0);
    cp.setSampler(s->sampler, 1);
    cp.setBuffer(s->tri_buf, 2);
    cp.setTexture(dsts[k], 3, 1);
    cp.setBuffer(s->render_uniform[k], 4);
    cp.dispatch((vp_w + 7) / 8, (vp_h + 7) / 8);
    cp.end();
  }

  gpu::Device::submit();
}

} // namespace tri_shear
