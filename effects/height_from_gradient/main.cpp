/*
 * filter.height_from_gradient — GPU gradient-domain height reconstruction.
 *
 * Synthesizes a 2D gradient field from the input, takes its divergence, and
 * solves the Poisson equation laplacian(h) = div(g) for the least-squares
 * height whose gradient best matches g — then visualizes the reconstructed
 * surface (hillshade / grayscale / normals / its own contour lines).
 * Gradient sources: Radial (outward from an adjustable center, magnitude =
 * luma), Level Curves (input as a contour map — the across-curve normal, its
 * uphill sign resolved by a global bias), and three that integrate an existing
 * vector field — Motion Vectors (the incoming render_outputs/motion rail),
 * Normal Map (g = -n.xy / n.z), and Gradient Field (channels are the gradient).
 * The field is generally non-conservative (curl != 0), so there's usually no
 * exact height; the Poisson solve is the "best try."
 *
 * Solver: coarse-to-fine multigrid cascade (FMG-lite). Build a pre-scaled
 * divergence pyramid, Jacobi-solve the coarsest level from zero, prolong
 * (upsample) the result as the initial guess for the next finer level, and
 * repeat to full res. The pre-scaling (restrict = SUM of 2x2, see
 * common.hlsl) keeps the Jacobi stencil spacing-agnostic so one PSO serves
 * every level — and avoids per-level uniform changes, which WebGPU can't
 * honor between dispatches in a single submit.
 *
 * Pass pipeline (shared common.hlsl):
 *   gradient   — source → RG gradient field g (radial / level-curve / decoded
 *                vector field, per `source`).
 *   divergence — g → F_0 = div(g) (central differences, level-0 spacing).
 *   restrict   — F_k → F_{k+1} (2x2 sum; builds the pre-scaled pyramid).
 *   jacobi     — one relaxation sweep h' = (hL+hR+hD+hU - F)/4 (reused/level).
 *   prolong    — coarse h → fine initial guess (bilinear upsample).
 *   mm_seed/mm_reduce — fold the height to its 1x1 global (min,max) so the
 *                presenter can normalize the arbitrary Poisson scale.
 *   present    — height → hillshade / grayscale / normals / contours (rgba8).
 *
 * Deterministic per-frame (no cross-frame state) → NO is_identity.
 * Freeform (multi-pass, pyramid, neighbor reads) → NO fusion.
 */

#include <gpu.h>
#include <host.h>
#include <effect_utils.h>
#include "height_from_gradient_shaders.h"

#include <cmath>
#include <utility>

namespace height_from_gradient {

// 32 floats = 8 std140 rows = 128 bytes. Mirrors HFG_UNIFORMS in common.hlsl;
// only the gradient + present passes bind it.
struct Uniforms {
  float grad_gain;    float source;        float center_x;      float center_y;
  float aspect_x;     float aspect_y;      float present_mode;  float relief_scale;
  float light_x;      float light_y;       float light_z;       float light_gain;
  float ambient;      float mix_amount;    float height_scale;  float height_offset;
  float tint_r;       float tint_g;        float tint_b;        float debug_show_gradient;
  float core_radius;  float core_softness; float bias_mode;     float bias_x;
  float bias_y;       float edge_mode;     float edge_threshold; float edge_gain;
  float contour_density; float line_width; float channel_mode;  float vector_sign;
};
static_assert(sizeof(Uniforms) == 128, "Uniforms layout mismatch");

// full / half / quarter / eighth. More levels = the low-frequency shape
// propagates globally in fewer fine-level iterations.
static constexpr int NUM_LEVELS = 4;

// Max levels in the height min/max reduction chain (full res → 1x1). 20 levels
// covers up to ~1M px on a side — far beyond any real viewport.
static constexpr int MM_LEVELS = 20;

static constexpr float kPi = 3.14159265358979323846f;

struct State {
  gpu::Buffer  uniform_buf;
  gpu::Texture grad_tex;               // full-res gradient (RG)
  gpu::Texture div[NUM_LEVELS];        // pre-scaled divergence pyramid (R)
  gpu::Texture h_a[NUM_LEVELS];        // height ping-pong A (R)
  gpu::Texture h_b[NUM_LEVELS];        // height ping-pong B (R)
  gpu::Texture mm[MM_LEVELS];          // height min/max reduction chain (RG)
  gpu::Texture zero_tex;               // 1x1 zero fallback (unwired motion rail)
  int  mm_count = 0;                   // active levels in the mm chain
  int  tex_w = 0, tex_h = 0;
  bool initialized = false;

  // --- Schema-mirrored params ---
  // 0 = Radial, 1 = Level Curves, 2 = Motion Vectors, 3 = Normal Map,
  // 4 = Gradient Field.
  int   source          = 0;
  float center_x        = 0.0f;    // cover-square anchor (radial field / bias)
  float center_y        = 0.0f;
  float grad_gain       = 1.0f;
  float core_radius     = 0.12f;   // (radial) smoothed-core size around anchor
  float core_softness   = 0.5f;    // (radial) transition feather of the core
  // Level-curves params.
  int   bias_mode       = 0;       // 0 = Radial bias, 1 = Linear sweep
  float sweep_angle     = 0.0f;    // (linear bias) 0..1 → direction
  int   edge_mode       = 0;       // 0 = Uniform per contour, 1 = Proportional
  float edge_threshold  = 0.1f;    // (uniform) edge-energy cutoff
  float edge_gain       = 0.5f;    // (proportional) edge-strength scale
  // Vector-source decode (Motion / Normal Map / Gradient Field).
  int   channel_mode    = 0;       // 0 = RG, 1 = RG flip-Y, 2 = AG
  int   vector_sign     = 0;       // 0 = signed (0=zero), 1 = unsigned (0.5=zero)
  int   present_mode    = 0;        // 0 Hillshade, 1 Grayscale, 2 Normals, 3 Contours
  float light_angle     = 0.375f;   // azimuth 0..1 → 0..2π (default ≈135° NW)
  float light_elevation = 0.5f;     // 0..1 → 0..π/2
  float relief_scale    = 0.4f;
  float mix_amount      = 0.0f;
  int   iterations      = 16;       // Jacobi sweeps per level
  float light_gain      = 1.0f;
  float ambient         = 0.15f;
  float tint_r          = 1.0f, tint_g = 1.0f, tint_b = 1.0f;
  float height_scale    = 1.0f;     // grayscale mode (height is normalized 0..1)
  float height_offset   = 0.0f;
  // Contours mode.
  float contour_density = 0.2f;     // 0..1 → iso-level count
  float line_width      = 0.5f;     // 0..1 → line thickness (exponential)
  bool  debug_show_gradient = false;
};

// Show only the fields the active source + present mode actually use. Called
// from on_state_ready (once after init + state replay) and from
// on_state_patched whenever a mode field changes. Touches the type-shared
// schema, so it takes the mode values rather than per-instance state.
static void apply_visibility(int source, int present_mode, int bias_mode, int edge_mode) {
  bool radial = (source == 0);
  bool curves = (source == 1);
  bool vector = (source >= 2);   // Motion / Normal Map / Gradient Field
  // Gradient-source params.
  state::setFieldHidden("center",         vector);          // unused by vector sources
  state::setFieldHidden("core_radius",    !radial);
  state::setFieldHidden("core_softness",  !radial);
  state::setFieldHidden("bias_mode",      !curves);
  state::setFieldHidden("edge_mode",      !curves);
  state::setFieldHidden("sweep_angle",    !(curves && bias_mode == 1));  // Linear bias
  state::setFieldHidden("edge_threshold", !(curves && edge_mode == 0));  // Uniform
  state::setFieldHidden("edge_gain",      !(curves && edge_mode == 1));  // Proportional
  state::setFieldHidden("channel_mode",   !vector);
  state::setFieldHidden("vector_sign",    !vector);

  // Present-mode params (0 Hillshade, 1 Grayscale, 2 Normals, 3 Contours).
  bool hill = (present_mode == 0);
  bool gray = (present_mode == 1);
  bool norm = (present_mode == 2);
  bool cont = (present_mode == 3);
  state::setFieldHidden("light_angle",     !hill);
  state::setFieldHidden("light_elevation", !hill);
  state::setFieldHidden("light_gain",      !hill);
  state::setFieldHidden("ambient",         !hill);
  state::setFieldHidden("relief_scale",    !(hill || norm));   // slope → normal
  state::setFieldHidden("height_scale",    !gray);
  state::setFieldHidden("height_offset",   !gray);
  state::setFieldHidden("contour_density", !cont);
  state::setFieldHidden("line_width",      !cont);
  state::setFieldHidden("tint",            !(hill || gray || cont));  // not normals
  // Always visible: source, center, grad_gain, present_mode, mix, iterations,
  // debug_show_gradient.
}

// Static (self-less) visibility evaluator — pure over state. Decodes the mode
// fields from a candidate state and reuses apply_visibility (see crop).
void eval_visibility(int n, const char* pb, const int* off, const int* len, const int* ops) {
  int source = 0, present_mode = 0, bias_mode = 0, edge_mode = 0;
  for (int i = 0; i < n; i++) {
    if (ops[i] != state::PatchReplace) continue;
    const char* p = pb + off[i]; int l = len[i];
    if      (state::pathIs(p, l, "source"))       source = (int)state::patchFloat(i);
    else if (state::pathIs(p, l, "present_mode")) present_mode = (int)state::patchFloat(i);
    else if (state::pathIs(p, l, "bias_mode"))    bias_mode = (int)state::patchFloat(i);
    else if (state::pathIs(p, l, "edge_mode"))    edge_mode = (int)state::patchFloat(i);
  }
  apply_visibility(source, present_mode, bias_mode, edge_mode);
}

static void on_state_ready(void* self);

// Type-shared, compiled once in module_init().
static gpu::ComputePSO s_pso_gradient;
static gpu::ComputePSO s_pso_divergence;
static gpu::ComputePSO s_pso_restrict;
static gpu::ComputePSO s_pso_jacobi;
static gpu::ComputePSO s_pso_prolong;
static gpu::ComputePSO s_pso_mm_seed;
static gpu::ComputePSO s_pso_mm_reduce;
static gpu::ComputePSO s_pso_present;

void module_init() {
  state::init("filter.height_from_gradient", {1, 0, 1},
    state::Schema()
      // Top-level manual: high-level "what is this / how to use / what to try".
      .helpField("intro",
        "## Height From Gradient\n"
        "Reconstructs a 3D **height field** from the image: it synthesizes a "
        "gradient, then solves the Poisson equation for the surface whose slope "
        "best matches it, and renders that surface as **hillshade, grayscale, "
        "normals, or contour lines**.\n\n"
        "**Try:** start with *Gradient Source* = Radial and *Display* = "
        "Hillshade, then sweep the *Light Angle* to relight the relief. Switch "
        "Source to *Level Curves* to read the image as a topo map, or feed a "
        "Motion / Normal / Gradient rail to integrate an existing vector field.")
      // --- Standard (live) ---
      .group("source", "Gradient Source")
        .groupHelp(
          "Where the height's gradient comes from. **Radial** pushes outward "
          "from the *Center* (magnitude = luma); **Level Curves** treats the "
          "image as a contour map; **Motion / Normal Map / Gradient Field** "
          "integrate an existing vector rail. *Gradient Gain* is the master "
          "strength; the **Core** knobs tame the radial singularity, the "
          "**Bias / Edge** knobs steer level-curve sign and step, and the "
          "**Channel / Vector** knobs decode a packed vector source.")
      // Gradient source. Radial — outward from `center`, magnitude = luma.
      // Level Curves — treat the input as a contour map. Motion Vectors — use
      // the incoming render_outputs/motion field. Normal Map — input is a
      // surface-normal map (integrated). Gradient Field — input channels are
      // the gradient directly.
      .selectField("source", 0, state::PrimaryInput,
                   {{"Radial", 0}, {"Level Curves", 1}, {"Motion Vectors", 2},
                    {"Normal Map", 3}, {"Gradient Field", 4}}, /*wrap=*/true).label("Gradient Source", "Source")
      // Anchor (cover-square, aspect-correct; (0,0)=viewport center, §1.5).
      // Radial source: the field center. Level Curves + Radial bias: the
      // up/downhill reference point.
      .vec2Field("center", 0.0f, 0.0f, state::PrimaryInput, -1.0f, 1.0f).label("Center", "Center")
      // Gradient magnitude scale (master).
      .floatField("grad_gain", 1.0f, 0.0f, 1.0f, state::PrimaryInput).label("Gradient Gain", "Gain")
      // (Radial) Core smoothing — tames the 1/r divergence singularity at the
      // anchor. `core_radius` sets the flattened-core size (cover-square
      // units); the magnitude ramps from zero across it, turning the spike
      // into a smooth dome. 0 = off (raw spiky field).
      .floatField("core_radius", 0.12f, 0.0f, 1.0f, state::PrimaryInput).label("Core Radius", "Core R")
      // (Radial) How gradually the core's suppression feathers out.
      .floatField("core_softness", 0.5f, 0.0f, 1.0f, state::PrimaryInput).label("Core Softness", "Core S")
      // (Level Curves) How the uphill/downhill sign is resolved: Radial =
      // height rises outward from `center` (great for nested closed contours);
      // Linear = height rises along `sweep_angle`.
      .selectField("bias_mode", 0, state::PrimaryInput, {{"Radial", 0}, {"Linear", 1}}).label("Bias Mode", "Bias")
      // (Level Curves, Linear bias) Sweep direction, 0..1 → full circle.
      .floatField("sweep_angle", 0.0f, 0.0f, 1.0f, state::PrimaryInput).label("Sweep Angle", "Sweep")
      // (Level Curves) Per-contour step: Uniform = every contour is one equal
      // step (faithful topo map); Proportional = step ∝ edge strength.
      .selectField("edge_mode", 0, state::PrimaryInput, {{"Uniform", 0}, {"Proportional", 1}}).label("Edge Mode", "Edge")
      // (Level Curves, Uniform) Edge-energy cutoff that counts as a contour.
      .floatField("edge_threshold", 0.1f, 0.0f, 1.0f, state::PrimaryInput).label("Edge Threshold", "Thresh")
      // (Level Curves, Proportional) Edge-strength → step-height scale.
      .floatField("edge_gain", 0.5f, 0.0f, 1.0f, state::PrimaryInput).label("Edge Gain", "E Gain")
      // (Motion / Normal Map / Gradient Field) How the 2D vector is packed:
      // RG, RG with the Y channel flipped (the GL↔DX normal gotcha), or AG
      // (BC5/DXT5nm swizzle).
      .selectField("channel_mode", 0, state::PrimaryInput,
                   {{"RG", 0}, {"RG Flip-Y", 1}, {"AG", 2}}).label("Channel Packing", "Chan")
      // (Motion / Normal Map / Gradient Field) Zero convention: Signed = 0.0 is
      // zero ([-1,1]); Unsigned = 0.5 is zero ([0,1], remapped).
      .selectField("vector_sign", 0, state::PrimaryInput,
                   {{"Signed", 0}, {"Unsigned", 1}}).label("Vector Sign", "Sign")
      // --- Solver ---
      .group("solver", "Solver")
        .groupHelp(
          "How hard the Poisson solve works. More *Iterations* = a smoother, "
          "more globally consistent surface (the low-frequency shape propagates "
          "farther) at a linear cost. Drop it for a rougher, faster result.")
      // Jacobi relaxation sweeps per pyramid level (solver tuning, all modes).
      // More = closer to the true least-squares solution (smoother, more
      // global), at linear cost.
      .intField("iterations", 16, 1, 200, state::PrimaryInput).label("Iterations", "Iters")
      // --- Display ---
      .group("display", "Display")
        .groupHelp(
          "How the reconstructed height is shown. **Hillshade** relights it "
          "(sweep *Light Angle* / *Elevation*); **Grayscale** maps height to "
          "brightness; **Normals** shows the surface normal; **Contours** draws "
          "iso-lines of the result. *Mix* cross-fades back toward the input, "
          "and *Tint* colours the shaded / contour output.")
      // How to visualize the reconstructed height. Contours draws iso-lines of
      // OUR reconstructed height — a contour map of the result.
      .selectField("present_mode", 0, state::PrimaryInput,
                   {{"Hillshade", 0}, {"Grayscale", 1}, {"Normals", 2}, {"Contours", 3}}, /*wrap=*/true).label("Display Mode", "Mode")
      // Cross-fade the visualization back toward the input image (all modes).
      .floatField("mix", 0.0f, 0.0f, 1.0f, state::PrimaryInput).label("Mix", "Mix")
      // Hillshade light azimuth (0..1 → full circle) and elevation
      // (0..1 → horizon→overhead).
      .floatField("light_angle", 0.375f, 0.0f, 1.0f, state::PrimaryInput).label("Light Angle", "L Ang")
      .floatField("light_elevation", 0.5f, 0.0f, 1.0f, state::PrimaryInput).label("Light Elevation", "L Elev")
      // Relief steepness — scales the surface slope used to build the normal.
      .floatField("relief_scale", 0.4f, 0.0f, 1.0f, state::PrimaryInput).label("Relief Scale", "Relief")
      .floatField("light_gain", 1.0f, 0.0f, 1.0f, state::PrimaryInput).label("Light Gain", "L Gain")
      .floatField("ambient", 0.15f, 0.0f, 1.0f, state::PrimaryInput).label("Ambient", "Amb")
      .rgbField("tint", 1.0f, 1.0f, 1.0f, state::PrimaryInput).label("Tint", "Tint")
      // Grayscale-mode brightness mapping (height is defined up to a constant,
      // so its DC is user-dialed here).
      .floatField("height_scale", 1.0f, 0.0f, 8.0f, state::PrimaryInput).label("Height Scale", "H Scl")
      .floatField("height_offset", 0.0f, -1.0f, 1.0f, state::PrimaryInput).label("Height Offset", "H Off")
      // (Contours mode) Iso-level count (how finely the height is sliced) and
      // line thickness (exponential — low values are razor-thin). tint colors
      // the lines. At density 0 the stage is skipped and the output is black.
      .floatField("contour_density", 0.2f, 0.0f, 1.0f, state::PrimaryInput).label("Contour Density", "Dens")
      .floatField("line_width", 0.5f, 0.0f, 1.0f, state::PrimaryInput).label("Line Width", "Width")
      // --- Debug (last) ---
      .group("debug", "Debug")
      .boolField("debug_show_gradient", false, state::PrimaryInput).label("Show Gradient", "Grad")
      .capability(state::Capability::TimeIndependent)
      // --- I/O ---
      .textureField("tex_in", state::PrimaryInput)
      .textureField("tex_out", state::PrimaryOutput)
      // Incoming render_outputs struct — the Motion Vectors source reads
      // render_outputs_in/motion from it.
      .renderOutputs(state::PrimaryInput, "render_outputs_in")
  );

  if (gpu::Device::backend() == gpu::Backend::None) return;

  // Solver intermediates are RGBA16F (scalars in R — R32F can't be sampled as
  // Float on WebGPU); present writes rgba8 (default, no hint).
  state::registerShaderSPV("height_from_gradient_gradient",   GRADIENT_SPV,   GRADIENT_SPV_SIZE,   "rgba16float", "write");
  state::registerShaderSPV("height_from_gradient_divergence", DIVERGENCE_SPV, DIVERGENCE_SPV_SIZE, "rgba16float", "write");
  state::registerShaderSPV("height_from_gradient_restrict",   RESTRICT_SPV,   RESTRICT_SPV_SIZE,   "rgba16float", "write");
  state::registerShaderSPV("height_from_gradient_jacobi",     JACOBI_SPV,     JACOBI_SPV_SIZE,     "rgba16float", "write");
  state::registerShaderSPV("height_from_gradient_prolong",    PROLONG_SPV,    PROLONG_SPV_SIZE,    "rgba16float", "write");
  state::registerShaderSPV("height_from_gradient_mm_seed",    MM_SEED_SPV,    MM_SEED_SPV_SIZE,    "rgba16float", "write");
  state::registerShaderSPV("height_from_gradient_mm_reduce",  MM_REDUCE_SPV,  MM_REDUCE_SPV_SIZE,  "rgba16float", "write");
  state::registerShaderSPV("height_from_gradient_present",    PRESENT_SPV,    PRESENT_SPV_SIZE);

  auto cs_gradient   = gpu::Device::createShaderModuleByName("height_from_gradient_gradient");
  auto cs_divergence = gpu::Device::createShaderModuleByName("height_from_gradient_divergence");
  auto cs_restrict   = gpu::Device::createShaderModuleByName("height_from_gradient_restrict");
  auto cs_jacobi     = gpu::Device::createShaderModuleByName("height_from_gradient_jacobi");
  auto cs_prolong    = gpu::Device::createShaderModuleByName("height_from_gradient_prolong");
  auto cs_mm_seed    = gpu::Device::createShaderModuleByName("height_from_gradient_mm_seed");
  auto cs_mm_reduce  = gpu::Device::createShaderModuleByName("height_from_gradient_mm_reduce");
  auto cs_present    = gpu::Device::createShaderModuleByName("height_from_gradient_present");
  if (!cs_gradient || !cs_divergence || !cs_restrict || !cs_jacobi ||
      !cs_prolong || !cs_mm_seed || !cs_mm_reduce || !cs_present) return;

  s_pso_gradient = gpu::Device::createComputePSO(cs_gradient, "main", gpu::Bindings()
      .tex2d(0)                                       // input
      .storageTex2d(1, gpu::TextureFormat::RGBA16F)   // gradient (RG)
      .uniform(2));

  s_pso_divergence = gpu::Device::createComputePSO(cs_divergence, "main", gpu::Bindings()
      .tex2d(0)                                       // gradient
      .storageTex2d(1, gpu::TextureFormat::RGBA16F)); // divergence F_0

  s_pso_restrict = gpu::Device::createComputePSO(cs_restrict, "main", gpu::Bindings()
      .tex2d(0)                                       // finer F_k
      .storageTex2d(1, gpu::TextureFormat::RGBA16F)); // coarser F_{k+1}

  s_pso_jacobi = gpu::Device::createComputePSO(cs_jacobi, "main", gpu::Bindings()
      .tex2d(0)                                       // current height
      .tex2d(1)                                       // pre-scaled divergence F
      .storageTex2d(2, gpu::TextureFormat::RGBA16F)); // next height

  s_pso_prolong = gpu::Device::createComputePSO(cs_prolong, "main", gpu::Bindings()
      .tex2d(0)                                       // coarse height
      .storageTex2d(1, gpu::TextureFormat::RGBA16F)); // fine initial guess

  s_pso_mm_seed = gpu::Device::createComputePSO(cs_mm_seed, "main", gpu::Bindings()
      .tex2d(0)                                       // height
      .storageTex2d(1, gpu::TextureFormat::RGBA16F)); // (min,max)

  s_pso_mm_reduce = gpu::Device::createComputePSO(cs_mm_reduce, "main", gpu::Bindings()
      .tex2d(0)                                       // finer (min,max)
      .storageTex2d(1, gpu::TextureFormat::RGBA16F)); // coarser (min,max)

  s_pso_present = gpu::Device::createComputePSO(cs_present, "main", gpu::Bindings()
      .tex2d(0)                                       // height (level 0)
      .tex2d(1)                                       // gradient (debug)
      .tex2d(2)                                       // input (mix)
      .tex2d(3)                                       // 1x1 (min,max)
      .storageTex2d(4)                                // tex_out
      .uniform(5));

  state::log("height_from_gradient: module initialized");
}

void* create() {
  auto* s = new State();
  s->uniform_buf = gpu::Device::createBuffer(sizeof(Uniforms), gpu::BufferUsage::Uniform);
  return s;
}

void destroy(void* self) {
  auto* s = static_cast<State*>(self);
  if (!s) return;
  s->uniform_buf.release();
  s->grad_tex.release();
  for (int l = 0; l < NUM_LEVELS; l++) {
    s->div[l].release();
    s->h_a[l].release();
    s->h_b[l].release();
  }
  for (int l = 0; l < MM_LEVELS; l++) s->mm[l].release();
  s->zero_tex.release();
  delete s;
}

void init(void* self) {
  auto* s = static_cast<State*>(self);
  if (!s) return;
  s->initialized = false;
  s->tex_w = 0;
  s->tex_h = 0;
  if (!s_pso_gradient.valid() || !s_pso_divergence.valid() ||
      !s_pso_restrict.valid() || !s_pso_jacobi.valid() ||
      !s_pso_prolong.valid() || !s_pso_mm_seed.valid() ||
      !s_pso_mm_reduce.valid() || !s_pso_present.valid()) return;
  if (!s->uniform_buf.valid()) return;
  s->initialized = true;
  state::setOnStateReady(&on_state_ready);
}

static void on_state_ready(void* self) {
  auto* s = static_cast<State*>(self);
  if (!s) return;
  // Fired after init + the initial state replay — hide the inactive modes'
  // fields so the IDE never paints a transient "all fields visible" frame.
  apply_visibility(s->source, s->present_mode, s->bias_mode, s->edge_mode);
}

void tick(void* self, double dt) { (void)self; (void)dt; }


void on_state_patched(void* self, int n, const char* pb, const int* off,
                      const int* len, const int* ops) {
  auto* s = static_cast<State*>(self);
  if (!s) return;
  bool mode_changed = false;
  for (int i = 0; i < n; i++) {
    if (ops[i] != state::PatchReplace) continue;
    const char* path = pb + off[i];
    int plen = len[i];
    if      (state::pathIs(path, plen, "source"))        { int v = (int)state::patchFloat(i); if (v != s->source) { s->source = v; mode_changed = true; } }
    else if (state::pathIs(path, plen, "center")) {
      auto v = state::patchVec2(i);
      s->center_x = v.x;
      s->center_y = v.y;
    }
    else if (state::pathIs(path, plen, "grad_gain"))     s->grad_gain = state::patchFloat(i);
    else if (state::pathIs(path, plen, "core_radius"))   s->core_radius = state::patchFloat(i);
    else if (state::pathIs(path, plen, "core_softness")) s->core_softness = state::patchFloat(i);
    else if (state::pathIs(path, plen, "bias_mode"))     { int v = (int)state::patchFloat(i); if (v != s->bias_mode) { s->bias_mode = v; mode_changed = true; } }
    else if (state::pathIs(path, plen, "sweep_angle"))   s->sweep_angle = state::patchFloat(i);
    else if (state::pathIs(path, plen, "edge_mode"))     { int v = (int)state::patchFloat(i); if (v != s->edge_mode) { s->edge_mode = v; mode_changed = true; } }
    else if (state::pathIs(path, plen, "edge_threshold")) s->edge_threshold = state::patchFloat(i);
    else if (state::pathIs(path, plen, "edge_gain"))     s->edge_gain = state::patchFloat(i);
    else if (state::pathIs(path, plen, "channel_mode"))  s->channel_mode = (int)state::patchFloat(i);
    else if (state::pathIs(path, plen, "vector_sign"))   s->vector_sign = (int)state::patchFloat(i);
    else if (state::pathIs(path, plen, "present_mode"))  { int v = (int)state::patchFloat(i); if (v != s->present_mode) { s->present_mode = v; mode_changed = true; } }
    else if (state::pathIs(path, plen, "light_angle"))   s->light_angle = state::patchFloat(i);
    else if (state::pathIs(path, plen, "light_elevation")) s->light_elevation = state::patchFloat(i);
    else if (state::pathIs(path, plen, "relief_scale"))  s->relief_scale = state::patchFloat(i);
    else if (state::pathIs(path, plen, "mix"))           s->mix_amount = state::patchFloat(i);
    else if (state::pathIs(path, plen, "iterations"))    s->iterations = (int)state::patchFloat(i);
    else if (state::pathIs(path, plen, "light_gain"))    s->light_gain = state::patchFloat(i);
    else if (state::pathIs(path, plen, "ambient"))       s->ambient = state::patchFloat(i);
    else if (state::pathIs(path, plen, "tint")) {
      auto v = state::patchVec3(i);
      s->tint_r = v.x;
      s->tint_g = v.y;
      s->tint_b = v.z;
    }
    else if (state::pathIs(path, plen, "height_scale"))  s->height_scale = state::patchFloat(i);
    else if (state::pathIs(path, plen, "height_offset")) s->height_offset = state::patchFloat(i);
    else if (state::pathIs(path, plen, "contour_density")) s->contour_density = state::patchFloat(i);
    else if (state::pathIs(path, plen, "line_width"))    s->line_width = state::patchFloat(i);
    else if (state::pathIs(path, plen, "debug_show_gradient")) s->debug_show_gradient = state::patchFloat(i) != 0.0f;
  }
  if (mode_changed) apply_visibility(s->source, s->present_mode, s->bias_mode, s->edge_mode);
}

static inline int half_up(int x) { return (x + 1) / 2; }

// (Re)allocate the pyramid working set on viewport change. Every texture is
// fully written each frame before it's read (divergence → restrict → cleared
// coarse seed → prolong), so no clears are needed here.
static bool ensure_textures(State* s, int vp_w, int vp_h) {
  if (s->tex_w == vp_w && s->tex_h == vp_h && s->grad_tex.valid() &&
      s->div[0].valid() && s->h_a[0].valid() && s->h_b[0].valid() &&
      s->mm[0].valid())
    return true;

  s->grad_tex.release();
  for (int l = 0; l < NUM_LEVELS; l++) {
    s->div[l].release();
    s->h_a[l].release();
    s->h_b[l].release();
  }
  for (int l = 0; l < MM_LEVELS; l++) s->mm[l].release();

  s->grad_tex = gpu::Device::createTexture(vp_w, vp_h, gpu::TextureFormat::RGBA16F);
  if (!s->grad_tex.valid()) return false;

  int lw = vp_w, lh = vp_h;
  for (int l = 0; l < NUM_LEVELS; l++) {
    s->div[l] = gpu::Device::createTexture(lw, lh, gpu::TextureFormat::RGBA16F);
    s->h_a[l] = gpu::Device::createTexture(lw, lh, gpu::TextureFormat::RGBA16F);
    s->h_b[l] = gpu::Device::createTexture(lw, lh, gpu::TextureFormat::RGBA16F);
    if (!s->div[l].valid() || !s->h_a[l].valid() || !s->h_b[l].valid()) return false;
    lw = half_up(lw);
    lh = half_up(lh);
  }

  // Min/max reduction chain: full res → 1x1 (each level a 2x2 fold).
  int mw = vp_w, mh = vp_h, count = 0;
  while (count < MM_LEVELS) {
    s->mm[count] = gpu::Device::createTexture(mw, mh, gpu::TextureFormat::RGBA16F);
    if (!s->mm[count].valid()) return false;
    count++;
    if (mw == 1 && mh == 1) break;
    mw = half_up(mw);
    mh = half_up(mh);
  }
  s->mm_count = count;

  s->tex_w = vp_w;
  s->tex_h = vp_h;
  return true;
}

// Mirror of local_delay's dispatch helper: up to 4 textures (slot = arg order,
// access 0=read/1=write) + an optional uniform buffer.
static inline void disp(const gpu::ComputePSO& pso, int w, int h,
                        const gpu::Texture* t0, int a0,
                        const gpu::Texture* t1, int a1,
                        const gpu::Texture* t2, int a2,
                        const gpu::Texture* t3, int a3,
                        const gpu::Buffer* ub, int ubslot) {
  auto cp = gpu::ComputePass::begin();
  cp.setPSO(pso);
  if (t0) cp.setTexture(*t0, 0, a0);
  if (t1) cp.setTexture(*t1, 1, a1);
  if (t2) cp.setTexture(*t2, 2, a2);
  if (t3) cp.setTexture(*t3, 3, a3);
  if (ub) cp.setBuffer(*ub, ubslot);
  cp.dispatch((w + 7) / 8, (h + 7) / 8);
  cp.end();
}

// Run `iters` Jacobi sweeps at one pyramid level, ping-ponging h_a/h_b. The
// initial guess must already be in h_a[level]. Returns the texture holding the
// final estimate (parity depends on iters).
static gpu::Texture solve_level(State* s, int level, int w, int h, int iters) {
  gpu::Texture* cur = &s->h_a[level];
  gpu::Texture* nxt = &s->h_b[level];
  for (int i = 0; i < iters; i++) {
    disp(s_pso_jacobi, w, h, cur, 0, &s->div[level], 0, nxt, 1, nullptr, 0, nullptr, 0);
    std::swap(cur, nxt);
  }
  return *cur;
}

void render(void* self, int vp_w, int vp_h) {
  auto* s = static_cast<State*>(self);
  if (!s || !s->initialized || vp_w <= 0 || vp_h <= 0) return;
  auto in  = gpu::Device::textureForField("tex_in");
  auto out = gpu::Device::textureForField("tex_out");
  if (!in.valid() || !out.valid()) return;

  // Contours at density 0 = no iso-lines. Skip the entire solve + present and
  // just clear black (style guide §0 — skip the stage rather than render a
  // degenerate full-white fill). Debug overlay still wants the real pass.
  if (s->present_mode == 3 && s->contour_density <= 0.0f && !s->debug_show_gradient) {
    gpu::Device::clear(out, 0.0f, 0.0f, 0.0f, 1.0f);
    gpu::Device::submit();
    return;
  }

  if (!ensure_textures(s, vp_w, vp_h)) return;

  auto cs = fx::coverSquare(vp_w, vp_h);

  // Hillshade light direction from azimuth + elevation.
  float az = s->light_angle * 2.0f * kPi;
  float el = s->light_elevation * (kPi * 0.5f);
  float ce = std::cos(el);
  float lx = ce * std::cos(az);
  float ly = ce * std::sin(az);
  float lz = std::sin(el);

  // Linear-sweep bias direction (level-curves sign disambiguation).
  float sa = s->sweep_angle * 2.0f * kPi;
  float bias_x = std::cos(sa);
  float bias_y = std::sin(sa);

  Uniforms u = {
    s->grad_gain, (float)s->source, s->center_x, s->center_y,
    cs.ax, cs.ay, (float)s->present_mode, s->relief_scale,
    lx, ly, lz, s->light_gain,
    s->ambient, s->mix_amount, s->height_scale, s->height_offset,
    s->tint_r, s->tint_g, s->tint_b, s->debug_show_gradient ? 1.0f : 0.0f,
    s->core_radius, s->core_softness, (float)s->bias_mode, bias_x,
    bias_y, (float)s->edge_mode, s->edge_threshold, s->edge_gain,
    s->contour_density, s->line_width, (float)s->channel_mode, (float)s->vector_sign,
  };
  s->uniform_buf.writeOne(u);

  int iters = s->iterations;
  if (iters < 1) iters = 1;
  if (iters > 200) iters = 200;

  // Per-level dimensions (half_up cascade).
  int lw[NUM_LEVELS], lh[NUM_LEVELS];
  lw[0] = vp_w; lh[0] = vp_h;
  for (int l = 1; l < NUM_LEVELS; l++) {
    lw[l] = half_up(lw[l - 1]);
    lh[l] = half_up(lh[l - 1]);
  }

  // 1 — gradient: source texture → RG gradient field. The Motion Vectors
  //     source reads the incoming render_outputs/motion rail instead of the
  //     input image (1x1 zero fallback when nothing's wired → flat → no-op).
  gpu::Texture src_tex = in;
  if (s->source == 2) {
    auto motion = gpu::Device::textureForField("render_outputs_in/motion");
    if (motion.valid()) {
      src_tex = motion;
    } else {
      if (!s->zero_tex.valid()) {
        s->zero_tex = gpu::Device::createTexture(1, 1, gpu::TextureFormat::RGBA16F);
        gpu::Device::clear(s->zero_tex, 0.0f, 0.0f, 0.0f, 0.0f);
      }
      src_tex = s->zero_tex;
    }
  }
  disp(s_pso_gradient, vp_w, vp_h, &src_tex, 0, &s->grad_tex, 1,
       nullptr, 0, nullptr, 0, &s->uniform_buf, 2);

  // 2 — divergence: g → F_0 (finest level).
  disp(s_pso_divergence, vp_w, vp_h, &s->grad_tex, 0, &s->div[0], 1,
       nullptr, 0, nullptr, 0, nullptr, 0);

  // 3 — restrict: build the pre-scaled divergence pyramid (2x2 sum).
  for (int k = 0; k < NUM_LEVELS - 1; k++) {
    disp(s_pso_restrict, lw[k + 1], lh[k + 1], &s->div[k], 0, &s->div[k + 1], 1,
         nullptr, 0, nullptr, 0, nullptr, 0);
  }

  // 4 — coarsest solve from a zero initial guess.
  int L = NUM_LEVELS - 1;
  gpu::Device::clear(s->h_a[L], 0.0f, 0.0f, 0.0f, 0.0f);
  gpu::Texture coarse = solve_level(s, L, lw[L], lh[L], iters);

  // 5 — cascade: prolong the coarse solution as the finer level's initial
  //     guess, then relax there. Repeat to full res.
  for (int k = L - 1; k >= 0; k--) {
    disp(s_pso_prolong, lw[k], lh[k], &coarse, 0, &s->h_a[k], 1,
         nullptr, 0, nullptr, 0, nullptr, 0);
    coarse = solve_level(s, k, lw[k], lh[k], iters);
  }

  // 6 — min/max reduction: fold the reconstructed height to a 1x1 global range
  //     so present can normalize the arbitrary Poisson scale.
  disp(s_pso_mm_seed, vp_w, vp_h, &coarse, 0, &s->mm[0], 1,
       nullptr, 0, nullptr, 0, nullptr, 0);
  {
    int mw = vp_w, mh = vp_h;
    for (int k = 0; k + 1 < s->mm_count; k++) {
      int nw = half_up(mw), nh = half_up(mh);
      disp(s_pso_mm_reduce, nw, nh, &s->mm[k], 0, &s->mm[k + 1], 1,
           nullptr, 0, nullptr, 0, nullptr, 0);
      mw = nw; mh = nh;
    }
  }
  const gpu::Texture& mm_global = s->mm[s->mm_count - 1];   // 1x1 (min,max)

  // 7 — present: visualize the reconstructed height. Five textures, so dispatch
  //     explicitly (the disp helper tops out at four).
  {
    auto cp = gpu::ComputePass::begin();
    cp.setPSO(s_pso_present);
    cp.setTexture(coarse, 0, 0);
    cp.setTexture(s->grad_tex, 1, 0);
    cp.setTexture(in, 2, 0);
    cp.setTexture(mm_global, 3, 0);
    cp.setTexture(out, 4, 1);
    cp.setBuffer(s->uniform_buf, 5);
    cp.dispatch((vp_w + 7) / 8, (vp_h + 7) / 8);
    cp.end();
  }

  gpu::Device::submit();
}

} // namespace height_from_gradient
