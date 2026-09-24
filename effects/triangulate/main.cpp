/*
 * filter.mesh.triangulate — "Triangulate".
 *
 * Renders a Delaunay triangulation that FOLLOWS THE TOPOLOGY of an input's
 * density: ridgelines first, then corners, then filling voids. Built entirely
 * on the GPU.
 *
 * Pipeline (per frame, all at an internal proc resolution except present):
 *   downsample → blur → feature (importance field W)
 *   → JFA Voronoi (init, splat, log2 steps)
 *   → score (per-cell mass / centroid / argmax-importance candidate)
 *   → present (input / debug map / mesh) — reads this frame's consistent state
 *   → takeover (stochastic confidence-gated teleport — updates seeds for NEXT
 *               frame; seeds are LOCKED and only jump when a candidate is
 *               confidently a better match → no continuous drift, no swim).
 *
 * P2 status: feature maps + JFA Voronoi + stochastic-takeover seed dynamics.
 * The Delaunay mesh (triple-point edges → instanced lines) lands in P3; for now
 * `present` shows the input, a debug map, the Voronoi cells, or the seed points.
 */

#include <gpu.h>
#include <host.h>
#include <effect_blur.h>
#include "triangulate_shaders.h"

#include <cstdint>
#include <cmath>

namespace triangulate {

// ---- constants -------------------------------------------------------------
static constexpr int   MAX_SEEDS         = 4096;   // pool ceiling (16-bit-packed edge indices)
static constexpr int   MAX_EDGES         = 65536;  // Delaunay edge-buffer ceiling
static constexpr int   SEEN_WORDS        = (MAX_SEEDS * MAX_SEEDS) / 32;  // edge dedup bitmask (2 MB)
static constexpr int   PROC_MAX          = 640;    // internal long-edge cap (< 1024)
static constexpr int   MAX_JFA_STEPS     = 12;
static constexpr float FEATURE_BLUR_SCALE= 0.6f;
static constexpr float STENCIL_MAX_PX    = 7.0f;
// Rough pre-scale gains — the histogram/percentile pass does the real per-frame
// auto-leveling, so these only need to keep the responses off saturation.
static constexpr float RIDGE_GAIN        = 15.0f;
static constexpr float CORNER_GAIN       = 1500.0f;
static constexpr int   HIST_BINS         = 64;
static constexpr int   HIST_WORDS        = 2 * HIST_BINS;   // ridge + corner
static constexpr float FEAT_PERCENTILE   = 0.97f;
static constexpr float POINT_RADIUS_UV   = 0.006f;
static constexpr float LINE_HALF_W_MAX   = 5.0f;   // line_width=1 → 0.5..5.5 px half-width

// ---- uniform layouts (must match the HLSL cbuffers) ------------------------
struct FeatureUniforms {
  float ridge_w, corner_w, void_w, stencil;
  float ridge_gain, corner_gain, _p0, _p1;
};
struct SplatUniforms { uint32_t count, w, h, _p; };
struct HistUniforms  { uint32_t w, h, bins, _p; };
struct CdfUniforms   { uint32_t bins; float percentile; float _p0, _p1; };
struct RemapUniforms { float ridge_w, corner_w, void_w, _p; };
struct StepUniforms  { int32_t step; uint32_t w, h; float aspect; };
struct ClearUniforms { uint32_t count, _p0, _p1, _p2; };
struct ScoreUniforms { uint32_t w, h; float _p0, _p1; };
struct TakeoverUniforms {
  uint32_t count, w, h, frame;
  float dt, churn, confidence, aspect;
  uint32_t mode; float decimation; uint32_t bnd, _p2;
};
struct PresentUniforms {
  uint32_t debug_view, bg_mode, proc_w, proc_h;
  float aspect, point_r; uint32_t count, _p0;
};
struct ClearEdgeUniforms { uint32_t max_edges, seen_words, _p1, _p2; };
struct EdgeUniforms { uint32_t w, h, max, _p; };
struct LineUniforms {
  float vp_x, vp_y, half_w, threshold;
  float cr, cg, cb, _p1;
  float fcr, fcg, fcb, _p2;
};

// ---- per-instance state ----------------------------------------------------
struct State {
  // GPU-resident pool + accumulators.
  gpu::Buffer seed_buf;      // Seed[MAX_SEEDS]
  gpu::Buffer accum_buf;     // uint[MAX_SEEDS * 4]
  gpu::Buffer edge_buf;      // uint[MAX_EDGES] packed (a<<16)|b
  gpu::Buffer edge_count_buf;// uint append counter
  gpu::Buffer edge_args_buf; // uint[4] indirect draw args (edge_args kernel)
  gpu::Buffer seen_buf;      // uint[SEEN_WORDS] per-pair edge dedup bitmask
  gpu::Buffer hist_buf;      // uint[HIST_WORDS] ridge+corner histogram
  gpu::Buffer pct_buf;       // float[4] percentile divisors

  // Uniform buffers (one write per pass, per frame).
  gpu::Buffer feature_buf, splat_buf, clear_buf, score_buf, takeover_buf, present_buf;
  gpu::Buffer edge_clear_buf, edge_uniform_buf, line_buf;
  gpu::Buffer hist_buf_u, cdf_buf_u, remap_buf_u;
  gpu::Buffer step_buf[MAX_JFA_STEPS];

  // Textures (proc-res unless noted).
  gpu::Texture small_tex;    // RGBA8 downsampled input
  gpu::Texture blur_tex;     // RGBA8 blurred
  gpu::Texture feat_raw;     // RGBA16F raw feature maps (pre auto-level)
  gpu::Texture feat_tex;     // RGBA16F balanced feature/importance field
  gpu::Texture jfa_a, jfa_b; // R32F seed-id ping-pong
  gpu::Sampler sampler;
  int proc_w = 0, proc_h = 0;

  uint32_t frame = 0;
  bool initialized = false;
  bool seeded = false;

  // Schema-mirrored params.
  float density = 0.3f, ridge_weight = 0.6f, corner_weight = 0.3f, void_weight = 0.2f;
  float churn = 0.3f, decimation = 0.6f, line_width = 0.3f;
  float line_r = 1.f, line_g = 1.f, line_b = 1.f;
  float feat_r = 1.f, feat_g = 0.45f, feat_b = 0.15f;
  float edge_threshold = 0.5f;
  int   frame_points = 8;
  float feature_scale = 0.4f, confidence = 0.4f;
  int   scoring_mode = 0, bg_mode = 1;
  float fill_opacity = 0.0f, quality = 0.3f;
  int   debug_view = 0;
};

// ---- type-shared PSOs ------------------------------------------------------
static gpu::ComputePSO s_pso_downsample, s_pso_feature, s_pso_jfa_init,
                       s_pso_jfa_splat, s_pso_jfa_step, s_pso_score_clear,
                       s_pso_score, s_pso_seed_prep, s_pso_takeover, s_pso_present,
                       s_pso_edge_clear, s_pso_edges, s_pso_edge_args,
                       s_pso_hist, s_pso_cdf, s_pso_remap;
static gpu::RenderPSO s_pso_lines;
static fx::GaussianBlur s_blur;

void module_init() {
  state::init("filter.mesh.triangulate", {1, 2, 1},
    state::Schema()
      .helpField("intro",
        "## Triangulate\n"
        "Re-draws an input as a **Delaunay mesh that follows its topology** — dense on "
        "ridgelines and corners, coarse across flat voids. Feature maps (ridge, corner, "
        "density) are histogram auto-levelled, a persistent seed pool is partitioned by a "
        "GPU Voronoi, and the vertices *stochastically settle* onto features (no swim) and "
        "*decimate* to coalesce triangles. Works on video frames or sparse point clouds.\n\n"
        "**Try:** push *Decimation* to coarsen the mesh into big feature-hugging triangles; "
        "bias *Ridge* vs *Corner* weight to choose which features keep detail; drop *Feature "
        "Threshold* to light more edges in the *Feature Colour*. Step *Debug View* to watch "
        "the importance field and Voronoi cells while you tune.")

      .group("budget", "Points")
        .groupHelp(
          "The vertex ceiling. *Density* is the maximum number of seed points; *Decimation* "
          "(under Dynamics) thins from it, so density sets the finest detail and decimation "
          "sets how far it coalesces.")
      .floatField("density",      0.3f, 0.f, 1.f, state::PrimaryInput, nullptr, 0.01f,
                  nullptr, "Mesh density — how many seed points populate the triangulation.").label("Density", "Dens")

      .group("features", "Feature Weighting")
        .groupHelp(
          "What counts as a feature to follow. The ridge (crest), corner (peak) and void "
          "(general density) maps are each histogram auto-levelled per frame, then mixed by "
          "these weights into one importance field — so only the RELATIVE mix matters, and it "
          "chooses which features win. *Feature Scale* sets the smoothing before the "
          "derivatives (bigger = coarser, smoother features).")
      .floatField("ridge_weight", 0.6f, 0.f, 1.f, state::PrimaryInput, nullptr, 0.01f,
                  nullptr, "How strongly seeds are pulled onto ridgelines.").label("Ridge Weight", "Ridge")
      .floatField("corner_weight",0.3f, 0.f, 1.f, state::PrimaryInput, nullptr, 0.01f,
                  nullptr, "How strongly seeds are pulled onto corners / maxima.").label("Corner Weight", "Corner")
      .floatField("void_weight",  0.2f, 0.f, 1.f, state::PrimaryInput, nullptr, 0.01f,
                  nullptr, "Baseline coverage of low-feature density (general fill).").label("Void Weight", "Void")
      .floatField("feature_scale",0.4f, 0.f, 1.f, state::SecondaryInput, nullptr, 0.01f,
                  nullptr, "Smoothing radius applied before derivatives (larger = coarser features).").label("Feature Scale", "Scale")

      .group("dynamics", "Dynamics")
        .groupHelp(
          "How the vertices move and thin. *Churn* is how eagerly mismatched vertices "
          "resettle (0 = frozen). *Decimation* removes low-importance vertices to coalesce "
          "triangles — crank it for big, sparse, feature-hugging triangles. *Mode* picks the "
          "placement rule; *Confidence* is the settle deadband for the relaxation modes.")
      .floatField("churn",        0.3f, 0.f, 1.f, state::PrimaryInput, nullptr, 0.01f,
                  nullptr, "Stochastic-takeover rate — how eagerly mismatched vertices jump (0 = frozen).").label("Churn", "Churn")
      .floatField("decimation",   0.6f, 0.f, 1.f, state::PrimaryInput, nullptr, 0.01f,
                  nullptr, "How aggressively low-importance vertices are dropped to coalesce triangles. Higher = sparser mesh, bigger contrast triangles; 0 = full density.").label("Decimation", "Decim")
      .selectField("scoring_mode", 0, state::SecondaryInput,
                   {{"Ridge Protect", 0}, {"Cell Residual", 1}, {"Feature Weight", 2}, {"Weight + Blue-noise", 3}},
                   false, "Where the surviving vertices sit. Ridge Protect (default): relax toward the importance-weighted centre. Others are relaxation variants.").label("Dynamics Mode", "Mode")
      .floatField("confidence",   0.4f, 0.f, 1.f, state::SecondaryInput, nullptr, 0.01f,
                  nullptr, "Takeover margin — a candidate must beat the incumbent by this much to displace it (higher = crisper/stiller). Relaxation modes only.").label("Confidence", "Conf")

      .group("style", "Mesh Style")
        .groupHelp(
          "The drawn mesh. *Line Width/Colour* is the base edge; edges whose feature weight "
          "beats *Feature Threshold* switch to *Feature Colour* (binary) so ridges stand out. "
          "*Frame Anchors* pin points around the border so the mesh triangulates out to the "
          "edges. *Backdrop* is what sits behind the mesh.")
      .floatField("line_width",   0.3f, 0.f, 1.f, state::PrimaryInput, nullptr, 0.01f,
                  nullptr, "Triangulation edge thickness.").label("Line Width", "Width")
      .rgbField  ("line_color",   1.f, 1.f, 1.f, state::PrimaryInput).label("Line Colour", "Line")
      .rgbField  ("feature_color",1.f, 0.45f, 0.15f, state::PrimaryInput).label("Feature Colour", "Feat")
      .floatField("edge_threshold",0.5f, 0.f, 1.f, state::PrimaryInput, nullptr, 0.01f,
                  nullptr, "Binary edge colouring: edges whose feature weight exceeds this get feature_color, the rest line_color.").label("Feature Threshold", "Thresh")
      .intField  ("frame_points", 8, 0, 24, state::PrimaryInput, 1, nullptr,
                  "Fixed anchor points per frame edge — the mesh triangulates out to the border (0 = off).").label("Frame Anchors", "Frame")
      .floatField("fill_opacity", 0.0f, 0.f, 1.f, state::SecondaryInput, nullptr, 0.01f,
                  nullptr, "Opacity of solid triangle fills (0 = wireframe only).").label("Fill Opacity", "Fill")
      .selectField("bg_mode", 1, state::SecondaryInput,
                   {{"Input", 0}, {"Dark", 1}, {"Feature", 2}},
                   false, "Backdrop the mesh is drawn over.").label("Backdrop", "BG")

      .group("debug", "Debug")
        .groupHelp(
          "Inspection aids. *Debug View* shows an internal stage — the density / ridge / "
          "corner maps, the blended importance field, the Voronoi cells or the seed points — "
          "instead of the mesh. *Blur Quality* tunes the feature-map blur sampling.")
      .selectField("debug_view", 0, state::SecondaryInput,
                   {{"Off", 0}, {"Density", 1}, {"Ridge", 2}, {"Corner", 3},
                    {"Importance", 4}, {"Voronoi", 5}, {"Points", 6}},
                   true, "Visualize an internal stage instead of the mesh.").label("Debug View", "Debug")
      .floatField("quality",      0.3f, 0.05f, 1.f, state::SecondaryInput, nullptr, 0.01f,
                  nullptr, "Blur sample density for the feature pass (tuning).").label("Blur Quality", "Qual")

      .endGroup()
      .capability(state::Capability::SeekableApproximate)
      .textureField("tex_in",  state::PrimaryInput)
      .textureField("tex_out", state::PrimaryOutput));

  if (gpu::Device::backend() == gpu::Backend::None) return;

  state::registerShaderSPV("triangulate_downsample", DOWNSAMPLE_SPV, DOWNSAMPLE_SPV_SIZE, "rgba8unorm", "write");
  state::registerShaderSPV("triangulate_feature",    FEATURE_SPV,    FEATURE_SPV_SIZE,    "rgba16float", "write");
  state::registerShaderSPV("triangulate_hist",       HIST_SPV,       HIST_SPV_SIZE);
  state::registerShaderSPV("triangulate_cdf",        CDF_SPV,        CDF_SPV_SIZE);
  state::registerShaderSPV("triangulate_remap",      REMAP_SPV,      REMAP_SPV_SIZE,      "rgba16float", "write");
  // JFA/score/edges pin their r32f id textures with [[vk::image_format]] and
  // bind them read_write (WebGPU rejects r32f write-only / sampled), so no
  // format override is needed. present carries an rgba8 output too, so it takes
  // the rgba8unorm,write override for that (its r32f id stays pinned).
  state::registerShaderSPV("triangulate_jfa_init",   JFA_INIT_SPV,   JFA_INIT_SPV_SIZE);
  state::registerShaderSPV("triangulate_jfa_splat",  JFA_SPLAT_SPV,  JFA_SPLAT_SPV_SIZE);
  state::registerShaderSPV("triangulate_jfa_step",   JFA_STEP_SPV,   JFA_STEP_SPV_SIZE);
  state::registerShaderSPV("triangulate_score_clear",SCORE_CLEAR_SPV,SCORE_CLEAR_SPV_SIZE);
  state::registerShaderSPV("triangulate_score",      SCORE_SPV,      SCORE_SPV_SIZE);
  state::registerShaderSPV("triangulate_seed_prep",  SEED_PREP_SPV,  SEED_PREP_SPV_SIZE);
  state::registerShaderSPV("triangulate_takeover",   TAKEOVER_SPV,   TAKEOVER_SPV_SIZE);
  state::registerShaderSPV("triangulate_present",    PRESENT_SPV,    PRESENT_SPV_SIZE);
  state::registerShaderSPV("triangulate_edge_clear", EDGE_CLEAR_SPV, EDGE_CLEAR_SPV_SIZE);
  state::registerShaderSPV("triangulate_edges",      EDGES_SPV,      EDGES_SPV_SIZE);
  state::registerShaderSPV("triangulate_edge_args",  EDGE_ARGS_SPV,  EDGE_ARGS_SPV_SIZE);
  state::registerShaderSPV("triangulate_line_vs",    LINE_VS_SPV,    LINE_VS_SPV_SIZE);
  state::registerShaderSPV("triangulate_line_fs",    LINE_FS_SPV,    LINE_FS_SPV_SIZE);

  auto cs_ds  = gpu::Device::createShaderModuleByName("triangulate_downsample");
  auto cs_ft  = gpu::Device::createShaderModuleByName("triangulate_feature");
  auto cs_hi  = gpu::Device::createShaderModuleByName("triangulate_hist");
  auto cs_cd  = gpu::Device::createShaderModuleByName("triangulate_cdf");
  auto cs_rm  = gpu::Device::createShaderModuleByName("triangulate_remap");
  auto cs_ji  = gpu::Device::createShaderModuleByName("triangulate_jfa_init");
  auto cs_jp  = gpu::Device::createShaderModuleByName("triangulate_jfa_splat");
  auto cs_js  = gpu::Device::createShaderModuleByName("triangulate_jfa_step");
  auto cs_sc  = gpu::Device::createShaderModuleByName("triangulate_score_clear");
  auto cs_s   = gpu::Device::createShaderModuleByName("triangulate_score");
  auto cs_sp  = gpu::Device::createShaderModuleByName("triangulate_seed_prep");
  auto cs_tk  = gpu::Device::createShaderModuleByName("triangulate_takeover");
  auto cs_pr  = gpu::Device::createShaderModuleByName("triangulate_present");
  auto cs_ec  = gpu::Device::createShaderModuleByName("triangulate_edge_clear");
  auto cs_ed  = gpu::Device::createShaderModuleByName("triangulate_edges");
  auto cs_ea  = gpu::Device::createShaderModuleByName("triangulate_edge_args");
  auto vs_ln  = gpu::Device::createShaderModuleByName("triangulate_line_vs");
  auto fs_ln  = gpu::Device::createShaderModuleByName("triangulate_line_fs");
  if (!cs_ds || !cs_ft || !cs_ji || !cs_jp || !cs_js || !cs_sc || !cs_s || !cs_sp || !cs_tk || !cs_pr
      || !cs_ec || !cs_ed || !cs_ea || !vs_ln || !fs_ln || !cs_hi || !cs_cd || !cs_rm) return;

  s_pso_downsample = gpu::Device::createComputePSO(cs_ds, "main", gpu::Bindings()
      .tex2d(0).sampler(1).storageTex2d(2, gpu::TextureFormat::RGBA8));
  s_pso_feature = gpu::Device::createComputePSO(cs_ft, "main", gpu::Bindings()
      .tex2d(0).storageTex2d(1, gpu::TextureFormat::RGBA16F).uniform(2));
  s_pso_hist = gpu::Device::createComputePSO(cs_hi, "main", gpu::Bindings()
      .tex2d(0).storageRW(1).uniform(2));
  s_pso_cdf = gpu::Device::createComputePSO(cs_cd, "main", gpu::Bindings()
      .storageRW(0).storageRW(1).uniform(2));
  s_pso_remap = gpu::Device::createComputePSO(cs_rm, "main", gpu::Bindings()
      .tex2d(0).storage(1).storageTex2d(2, gpu::TextureFormat::RGBA16F).uniform(3));
  s_pso_jfa_init = gpu::Device::createComputePSO(cs_ji, "main", gpu::Bindings()
      .storageTex2dRW(0, gpu::TextureFormat::R32F));
  s_pso_jfa_splat = gpu::Device::createComputePSO(cs_jp, "main", gpu::Bindings()
      .storage(0).storageTex2dRW(1, gpu::TextureFormat::R32F).uniform(2));
  s_pso_jfa_step = gpu::Device::createComputePSO(cs_js, "main", gpu::Bindings()
      .storageTex2dRW(0, gpu::TextureFormat::R32F).storage(1)
      .storageTex2dRW(2, gpu::TextureFormat::R32F).uniform(3));
  s_pso_score_clear = gpu::Device::createComputePSO(cs_sc, "main", gpu::Bindings()
      .storageRW(0).uniform(1));
  s_pso_score = gpu::Device::createComputePSO(cs_s, "main", gpu::Bindings()
      .storageTex2dRW(0, gpu::TextureFormat::R32F).tex2d(1).storageRW(2).uniform(3));
  s_pso_seed_prep = gpu::Device::createComputePSO(cs_sp, "main", gpu::Bindings()
      .storageRW(0).tex2d(1).uniform(2));
  s_pso_takeover = gpu::Device::createComputePSO(cs_tk, "main", gpu::Bindings()
      .storage(0).tex2d(1).storageRW(2).uniform(3));
  s_pso_present = gpu::Device::createComputePSO(cs_pr, "main", gpu::Bindings()
      .tex2d(0).tex2d(1).storageTex2dRW(2, gpu::TextureFormat::R32F).storage(3)
      .storageTex2d(4).uniform(5));
  s_pso_edge_clear = gpu::Device::createComputePSO(cs_ec, "main", gpu::Bindings()
      .storageRW(0).storageRW(1).storageRW(2).uniform(3));
  s_pso_edges = gpu::Device::createComputePSO(cs_ed, "main", gpu::Bindings()
      .storageTex2dRW(0, gpu::TextureFormat::R32F).storageRW(1).storageRW(2)
      .storageRW(3).uniform(4));
  s_pso_edge_args = gpu::Device::createComputePSO(cs_ea, "main", gpu::Bindings()
      .storageRW(0).storageRW(1).uniform(2));
  s_pso_lines = gpu::Device::createInstancedRenderPSO(
      vs_ln, "main", fs_ln, "main", gpu::TextureFormat::Surface,
      gpu::Bindings().uniform(0).storage(1).storage(2),
      gpu::Device::BlendMode::AlphaOver);
  s_blur.init();

  state::log("triangulate: module initialized");
}

void* create() {
  auto* s = new State();
  s->seed_buf  = gpu::Device::createBuffer(sizeof(float) * 4 * MAX_SEEDS, gpu::BufferUsage::Storage);
  s->accum_buf = gpu::Device::createBuffer(sizeof(uint32_t) * 4 * MAX_SEEDS, gpu::BufferUsage::Storage);
  s->edge_buf  = gpu::Device::createBuffer(sizeof(uint32_t) * MAX_EDGES, gpu::BufferUsage::Storage);
  s->edge_count_buf = gpu::Device::createBuffer(sizeof(uint32_t) * 4, gpu::BufferUsage::Storage);
  s->edge_args_buf  = gpu::Device::createBuffer(sizeof(uint32_t) * 4, gpu::BufferUsage::Storage);
  s->seen_buf  = gpu::Device::createBuffer(sizeof(uint32_t) * SEEN_WORDS, gpu::BufferUsage::Storage);
  s->hist_buf  = gpu::Device::createBuffer(sizeof(uint32_t) * HIST_WORDS, gpu::BufferUsage::Storage);
  s->pct_buf   = gpu::Device::createBuffer(sizeof(float) * 4, gpu::BufferUsage::Storage);
  { uint32_t zero[HIST_WORDS] = {}; s->hist_buf.writeBytes(zero, sizeof(zero), 0); }
  s->feature_buf  = gpu::Device::createBuffer(sizeof(FeatureUniforms),  gpu::BufferUsage::Uniform);
  s->splat_buf    = gpu::Device::createBuffer(sizeof(SplatUniforms),    gpu::BufferUsage::Uniform);
  s->clear_buf    = gpu::Device::createBuffer(sizeof(ClearUniforms),    gpu::BufferUsage::Uniform);
  s->score_buf    = gpu::Device::createBuffer(sizeof(ScoreUniforms),    gpu::BufferUsage::Uniform);
  s->takeover_buf = gpu::Device::createBuffer(sizeof(TakeoverUniforms), gpu::BufferUsage::Uniform);
  s->present_buf  = gpu::Device::createBuffer(sizeof(PresentUniforms),  gpu::BufferUsage::Uniform);
  s->hist_buf_u   = gpu::Device::createBuffer(sizeof(HistUniforms),  gpu::BufferUsage::Uniform);
  s->cdf_buf_u    = gpu::Device::createBuffer(sizeof(CdfUniforms),   gpu::BufferUsage::Uniform);
  s->remap_buf_u  = gpu::Device::createBuffer(sizeof(RemapUniforms), gpu::BufferUsage::Uniform);
  s->edge_clear_buf   = gpu::Device::createBuffer(sizeof(ClearEdgeUniforms), gpu::BufferUsage::Uniform);
  s->edge_uniform_buf = gpu::Device::createBuffer(sizeof(EdgeUniforms),      gpu::BufferUsage::Uniform);
  s->line_buf         = gpu::Device::createBuffer(sizeof(LineUniforms),      gpu::BufferUsage::Uniform);
  for (int i = 0; i < MAX_JFA_STEPS; i++)
    s->step_buf[i] = gpu::Device::createBuffer(sizeof(StepUniforms), gpu::BufferUsage::Uniform);
  s->sampler = gpu::Device::createSampler(gpu::FilterMode::Linear, gpu::AddressMode::ClampToEdge);
  return s;
}

void destroy(void* self) {
  auto* s = static_cast<State*>(self);
  if (!s) return;
  s->seed_buf.release(); s->accum_buf.release();
  s->edge_buf.release(); s->edge_count_buf.release(); s->edge_args_buf.release();
  s->seen_buf.release();
  s->hist_buf.release(); s->pct_buf.release();
  s->feature_buf.release(); s->splat_buf.release(); s->clear_buf.release();
  s->score_buf.release(); s->takeover_buf.release(); s->present_buf.release();
  s->hist_buf_u.release(); s->cdf_buf_u.release(); s->remap_buf_u.release();
  s->edge_clear_buf.release(); s->edge_uniform_buf.release(); s->line_buf.release();
  for (int i = 0; i < MAX_JFA_STEPS; i++) s->step_buf[i].release();
  if (s->small_tex.valid()) s->small_tex.release();
  if (s->blur_tex.valid())  s->blur_tex.release();
  if (s->feat_raw.valid())  s->feat_raw.release();
  if (s->feat_tex.valid())  s->feat_tex.release();
  if (s->jfa_a.valid())     s->jfa_a.release();
  if (s->jfa_b.valid())     s->jfa_b.release();
  s->sampler.release();
  delete s;
}

// Seed the pool with random positions (CPU, chunked to bound the wasm stack).
static void seed_pool(State* s) {
  if (!s->seed_buf.valid()) return;
  constexpr int CHUNK = 256;
  float buf[CHUNK * 4];
  uint32_t rng = 0x1234567u;
  auto next = [&]() { rng ^= rng << 13; rng ^= rng >> 17; rng ^= rng << 5; return (rng & 0xFFFFFF) / 16777216.0f; };
  for (int base = 0; base < MAX_SEEDS; base += CHUNK) {
    int n = (base + CHUNK <= MAX_SEEDS) ? CHUNK : (MAX_SEEDS - base);
    for (int j = 0; j < n; j++) {
      buf[j * 4 + 0] = next();  // pos.x
      buf[j * 4 + 1] = next();  // pos.y
      buf[j * 4 + 2] = 0.0f;    // score
      buf[j * 4 + 3] = 1.0f;    // flags: active
    }
    s->seed_buf.writeBytes(buf, sizeof(float) * 4 * n, sizeof(float) * 4 * base);
  }
  s->seeded = true;
}

// Mode-dependent parameter visibility (style guide §0). Every field is always in
// the schema; we just hide the ones the active mode/view can't use.
static void apply_visibility(const State* s) {
  const bool mesh = (s->debug_view == 0);   // the mesh is only drawn when debug is off
  state::setFieldHidden("line_width",     !mesh);
  state::setFieldHidden("line_color",     !mesh);
  state::setFieldHidden("feature_color",  !mesh);
  state::setFieldHidden("edge_threshold", !mesh);
  state::setFieldHidden("fill_opacity",   !mesh);
  state::setFieldHidden("bg_mode",        !mesh);
  // `confidence` is the takeover deadband for the relaxation modes; Ridge Protect
  // (mode 0) relaxes to a fixed-gate centroid and never reads it.
  state::setFieldHidden("confidence",     s->scoring_mode == 0);
}

static void on_state_ready(void* self) {
  auto* s = static_cast<State*>(self);
  if (s) apply_visibility(s);
}

void init(void* self) {
  auto* s = static_cast<State*>(self);
  if (!s) return;
  if (!s_pso_present.valid()) return;
  s->frame = 0;
  seed_pool(s);
  s->initialized = true;
  state::setOnStateReady(&on_state_ready);
}

void tick(void* self, double dt) { (void)self; (void)dt; }

void on_state_patched(void* self, int n, const char* pb, const int* off,
                      const int* len, const int* ops) {
  auto* s = static_cast<State*>(self);
  if (!s) return;
  bool vis_changed = false;
  for (int i = 0; i < n; i++) {
    if (ops[i] != state::PatchReplace) continue;
    const char* p = pb + off[i];
    int l = len[i];
    if      (state::pathIs(p, l, "density"))       s->density       = state::patchFloat(i);
    else if (state::pathIs(p, l, "ridge_weight"))  s->ridge_weight  = state::patchFloat(i);
    else if (state::pathIs(p, l, "corner_weight")) s->corner_weight = state::patchFloat(i);
    else if (state::pathIs(p, l, "void_weight"))   s->void_weight   = state::patchFloat(i);
    else if (state::pathIs(p, l, "churn"))         s->churn         = state::patchFloat(i);
    else if (state::pathIs(p, l, "decimation"))    s->decimation    = state::patchFloat(i);
    else if (state::pathIs(p, l, "line_width"))    s->line_width    = state::patchFloat(i);
    else if (state::pathIs(p, l, "line_color"))  { auto v = state::patchVec3(i); s->line_r=v.x; s->line_g=v.y; s->line_b=v.z; }
    else if (state::pathIs(p, l, "feature_color")){ auto v = state::patchVec3(i); s->feat_r=v.x; s->feat_g=v.y; s->feat_b=v.z; }
    else if (state::pathIs(p, l, "edge_threshold")) s->edge_threshold = state::patchFloat(i);
    else if (state::pathIs(p, l, "frame_points"))   s->frame_points    = state::patchInt(i);
    else if (state::pathIs(p, l, "feature_scale")) s->feature_scale = state::patchFloat(i);
    else if (state::pathIs(p, l, "confidence"))    s->confidence    = state::patchFloat(i);
    else if (state::pathIs(p, l, "scoring_mode")) { s->scoring_mode = state::patchInt(i); vis_changed = true; }
    else if (state::pathIs(p, l, "bg_mode"))       s->bg_mode       = state::patchInt(i);
    else if (state::pathIs(p, l, "fill_opacity"))  s->fill_opacity  = state::patchFloat(i);
    else if (state::pathIs(p, l, "quality"))       s->quality       = state::patchFloat(i);
    else if (state::pathIs(p, l, "debug_view"))   { s->debug_view   = state::patchInt(i); vis_changed = true; }
  }
  if (vis_changed) apply_visibility(s);
}

static void ensureTextures(State* s, int vp_w, int vp_h) {
  // Proc resolution: cap the long edge at PROC_MAX, keep aspect.
  int longest = vp_w > vp_h ? vp_w : vp_h;
  float k = longest > PROC_MAX ? (float)PROC_MAX / (float)longest : 1.0f;
  int pw = (int)(vp_w * k); if (pw < 1) pw = 1;
  int ph = (int)(vp_h * k); if (ph < 1) ph = 1;
  if (s->small_tex.valid() && s->proc_w == pw && s->proc_h == ph) return;
  if (s->small_tex.valid()) s->small_tex.release();
  if (s->blur_tex.valid())  s->blur_tex.release();
  if (s->feat_raw.valid())  s->feat_raw.release();
  if (s->feat_tex.valid())  s->feat_tex.release();
  if (s->jfa_a.valid())     s->jfa_a.release();
  if (s->jfa_b.valid())     s->jfa_b.release();
  s->small_tex = gpu::Device::createTexture(pw, ph, gpu::TextureFormat::RGBA8);
  s->blur_tex  = gpu::Device::createTexture(pw, ph, gpu::TextureFormat::RGBA8);
  s->feat_raw  = gpu::Device::createTexture(pw, ph, gpu::TextureFormat::RGBA16F);
  s->feat_tex  = gpu::Device::createTexture(pw, ph, gpu::TextureFormat::RGBA16F);
  s->jfa_a     = gpu::Device::createTexture(pw, ph, gpu::TextureFormat::R32F);
  s->jfa_b     = gpu::Device::createTexture(pw, ph, gpu::TextureFormat::R32F);
  s->proc_w = pw; s->proc_h = ph;
}

static inline int boundary_count(const State* s) {
  int fp = s->frame_points;
  if (fp < 0) fp = 0;
  if (fp > 24) fp = 24;
  return 4 * fp;   // evenly spaced around the perimeter; corners included
}

static inline int seed_count(const State* s) {
  int c = (int)lroundf(128.0f + s->density * (float)(MAX_SEEDS - 128));
  int bnd = boundary_count(s);
  if (c < bnd + 16) c = bnd + 16;   // keep room for interior seeds
  if (c > MAX_SEEDS) c = MAX_SEEDS;
  return c;
}

void render(void* self, int vp_w, int vp_h) {
  auto* s = static_cast<State*>(self);
  if (!s || !s->initialized || vp_w <= 0 || vp_h <= 0) return;
  auto in  = gpu::Device::textureForField("tex_in");
  auto out = gpu::Device::textureForField("tex_out");
  if (!in.valid() || !out.valid()) return;

  ensureTextures(s, vp_w, vp_h);
  if (!s->feat_tex.valid()) return;

  const int pw = s->proc_w, ph = s->proc_h;
  const int pgx = (pw + 7) / 8, pgy = (ph + 7) / 8;
  const int count = seed_count(s);
  const int sgx = (count + 63) / 64;
  const float aspect = (float)pw / (float)ph;
  const float dt = (float)host::deltaTime();

  // 1. Downsample the input to proc resolution.
  {
    auto cp = gpu::ComputePass::begin();
    cp.setPSO(s_pso_downsample);
    cp.setTexture(in, 0, 0);
    cp.setSampler(s->sampler, 1);
    cp.setTexture(s->small_tex, 2, 1);
    cp.dispatch(pgx, pgy);
    cp.end();
  }

  // 2. Blur + feature maps → importance field.
  s_blur.applyWithRadius(s->small_tex, s->blur_tex, pw, ph,
                         s->feature_scale * FEATURE_BLUR_SCALE, s->quality);
  {
    FeatureUniforms fu = {};
    fu.ridge_w = s->ridge_weight; fu.corner_w = s->corner_weight; fu.void_w = s->void_weight;
    fu.stencil = 1.0f + s->feature_scale * STENCIL_MAX_PX;
    fu.ridge_gain = RIDGE_GAIN; fu.corner_gain = CORNER_GAIN;
    s->feature_buf.writeOne(fu);
    auto cp = gpu::ComputePass::begin();
    cp.setPSO(s_pso_feature);
    cp.setTexture(s->blur_tex, 0, 0);
    cp.setTexture(s->feat_raw, 1, 1);
    cp.setBuffer(s->feature_buf, 2);
    cp.dispatch(pgx, pgy);
    cp.end();
  }

  // 2b. Histogram auto-level: hist(ridge,corner) → percentile divisors → remap
  // into the balanced importance field feat_tex. Data-driven per frame, so
  // ridges and corners land on a common distribution (no magic gains).
  {
    HistUniforms hu = { (uint32_t)pw, (uint32_t)ph, (uint32_t)HIST_BINS, 0 };
    s->hist_buf_u.writeOne(hu);
    auto cp = gpu::ComputePass::begin();
    cp.setPSO(s_pso_hist);
    cp.setTexture(s->feat_raw, 0, 0);
    cp.setBuffer(s->hist_buf, 1);
    cp.setBuffer(s->hist_buf_u, 2);
    cp.dispatch(pgx, pgy);
    cp.end();
  }
  {
    CdfUniforms cu = { (uint32_t)HIST_BINS, FEAT_PERCENTILE, 0.f, 0.f };
    s->cdf_buf_u.writeOne(cu);
    auto cp = gpu::ComputePass::begin();
    cp.setPSO(s_pso_cdf);
    cp.setBuffer(s->hist_buf, 0);
    cp.setBuffer(s->pct_buf, 1);
    cp.setBuffer(s->cdf_buf_u, 2);
    cp.dispatch(1, 1);
    cp.end();
  }
  {
    RemapUniforms ru = { s->ridge_weight, s->corner_weight, s->void_weight, 0.f };
    s->remap_buf_u.writeOne(ru);
    auto cp = gpu::ComputePass::begin();
    cp.setPSO(s_pso_remap);
    cp.setTexture(s->feat_raw, 0, 0);
    cp.setBuffer(s->pct_buf, 1);
    cp.setTexture(s->feat_tex, 2, 1);
    cp.setBuffer(s->remap_buf_u, 3);
    cp.dispatch(pgx, pgy);
    cp.end();
  }

  // 3. JFA Voronoi over the seed pool. init(jfa_a) → splat(jfa_a) → ping-pong.
  {
    auto cp = gpu::ComputePass::begin();
    cp.setPSO(s_pso_jfa_init);
    cp.setTexture(s->jfa_a, 0, 2);
    cp.dispatch(pgx, pgy);
    cp.end();
  }
  {
    SplatUniforms su = { (uint32_t)count, (uint32_t)pw, (uint32_t)ph, 0 };
    s->splat_buf.writeOne(su);
    auto cp = gpu::ComputePass::begin();
    cp.setPSO(s_pso_jfa_splat);
    cp.setBuffer(s->seed_buf, 0);
    cp.setTexture(s->jfa_a, 1, 2);
    cp.setBuffer(s->splat_buf, 2);
    cp.dispatch(sgx, 1);
    cp.end();
  }
  gpu::Texture* src = &s->jfa_a;
  gpu::Texture* dst = &s->jfa_b;
  int longest = pw > ph ? pw : ph;
  int step0 = 1; while (step0 * 2 < longest) step0 *= 2;
  int sidx = 0;
  for (int step = step0; step >= 1 && sidx < MAX_JFA_STEPS; step >>= 1, ++sidx) {
    StepUniforms st = { step, (uint32_t)pw, (uint32_t)ph, aspect };
    s->step_buf[sidx].writeOne(st);
    auto cp = gpu::ComputePass::begin();
    cp.setPSO(s_pso_jfa_step);
    cp.setTexture(*src, 0, 2);
    cp.setBuffer(s->seed_buf, 1);
    cp.setTexture(*dst, 2, 2);
    cp.setBuffer(s->step_buf[sidx], 3);
    cp.dispatch(pgx, pgy);
    cp.end();
    gpu::Texture* t = src; src = dst; dst = t;
  }
  gpu::Texture* id_tex = src;  // final nearest-seed id texture

  // 4. Score: clear accumulators, then scatter per-cell mass/centroid/candidate.
  {
    ClearUniforms cu = { (uint32_t)count, 0, 0, 0 };
    s->clear_buf.writeOne(cu);
    auto cp = gpu::ComputePass::begin();
    cp.setPSO(s_pso_score_clear);
    cp.setBuffer(s->accum_buf, 0);
    cp.setBuffer(s->clear_buf, 1);
    cp.dispatch(sgx, 1);
    cp.end();
  }
  {
    ScoreUniforms scu = { (uint32_t)pw, (uint32_t)ph, 0.f, 0.f };
    s->score_buf.writeOne(scu);
    auto cp = gpu::ComputePass::begin();
    cp.setPSO(s_pso_score);
    cp.setTexture(*id_tex, 0, 2);
    cp.setTexture(s->feat_tex, 1, 0);
    cp.setBuffer(s->accum_buf, 2);
    cp.setBuffer(s->score_buf, 3);
    cp.dispatch(pgx, pgy);
    cp.end();
  }

  // 4b. Seed prep: stamp each seed's own importance weight + clear its
  // neighbour-max accumulator (feeds the Ridge Protect dynamics). Reuses the
  // splat uniform (same {count, w, h}).
  {
    auto cp = gpu::ComputePass::begin();
    cp.setPSO(s_pso_seed_prep);
    cp.setBuffer(s->seed_buf, 0);
    cp.setTexture(s->feat_tex, 1, 0);
    cp.setBuffer(s->splat_buf, 2);
    cp.dispatch(sgx, 1);
    cp.end();
  }

  // 4c. Delaunay edges + per-seed adjacency. Runs EVERY frame: the mesh render
  // consumes the edge buffer only when shown, but the takeover dynamics always
  // need the neighbour-max weights this pass accumulates.
  const bool show_mesh = (s->debug_view == 0);
  {
    ClearEdgeUniforms ce = { (uint32_t)MAX_EDGES, (uint32_t)SEEN_WORDS, 0, 0 };
    s->edge_clear_buf.writeOne(ce);
    {
      auto cp = gpu::ComputePass::begin();
      cp.setPSO(s_pso_edge_clear);
      cp.setBuffer(s->edge_buf, 0);
      cp.setBuffer(s->edge_count_buf, 1);
      cp.setBuffer(s->seen_buf, 2);
      cp.setBuffer(s->edge_clear_buf, 3);
      cp.dispatch((SEEN_WORDS + 63) / 64, 1);
      cp.end();
    }
    EdgeUniforms eu = { (uint32_t)pw, (uint32_t)ph, (uint32_t)MAX_EDGES, 0 };
    s->edge_uniform_buf.writeOne(eu);
    {
      auto cp = gpu::ComputePass::begin();
      cp.setPSO(s_pso_edges);
      cp.setTexture(*id_tex, 0, 2);
      cp.setBuffer(s->edge_buf, 1);
      cp.setBuffer(s->edge_count_buf, 2);
      cp.setBuffer(s->seen_buf, 3);
      cp.setBuffer(s->edge_uniform_buf, 4);
      cp.dispatch(pgx, pgy);
      cp.end();
    }
    // Fold the append counter into indirect draw args so the mesh pass draws
    // the REAL edge count (typically ~3×seeds) instead of MAX_EDGES worst-case
    // instances. Binds edge_clear_buf: its layout carries u_max_edges.
    {
      auto cp = gpu::ComputePass::begin();
      cp.setPSO(s_pso_edge_args);
      cp.setBuffer(s->edge_count_buf, 0);
      cp.setBuffer(s->edge_args_buf, 1);
      cp.setBuffer(s->edge_clear_buf, 2);
      cp.dispatch(1, 1);
      cp.end();
    }
  }

  // 5. Present (uses this frame's consistent pre-takeover state).
  {
    PresentUniforms pu = {};
    pu.debug_view = (uint32_t)s->debug_view;
    pu.bg_mode = (uint32_t)s->bg_mode;
    pu.proc_w = (uint32_t)pw; pu.proc_h = (uint32_t)ph;
    pu.aspect = aspect; pu.point_r = POINT_RADIUS_UV;
    pu.count = (uint32_t)count;
    s->present_buf.writeOne(pu);
    auto cp = gpu::ComputePass::begin();
    cp.setPSO(s_pso_present);
    cp.setTexture(s->feat_tex, 0, 0);
    cp.setTexture(in, 1, 0);
    cp.setTexture(*id_tex, 2, 2);
    cp.setBuffer(s->seed_buf, 3);
    cp.setTexture(out, 4, 1);
    cp.setBuffer(s->present_buf, 5);
    cp.dispatch((vp_w + 7) / 8, (vp_h + 7) / 8);
    cp.end();
  }

  // 5b. Draw the Delaunay mesh over the present backdrop (wireframe).
  if (show_mesh) {
    LineUniforms lu = {};
    lu.vp_x = (float)vp_w; lu.vp_y = (float)vp_h;
    lu.half_w = 0.5f + s->line_width * LINE_HALF_W_MAX;
    lu.threshold = s->edge_threshold;
    lu.cr = s->line_r; lu.cg = s->line_g; lu.cb = s->line_b;
    lu.fcr = s->feat_r; lu.fcg = s->feat_g; lu.fcb = s->feat_b;
    s->line_buf.writeOne(lu);
    auto rp = gpu::RenderPass::beginLoad(out);
    rp.setPSO(s_pso_lines);
    rp.setBuffer(s->line_buf, 0);
    rp.setBuffer(s->edge_buf, 1);
    rp.setBuffer(s->seed_buf, 2);
    // GPU-fed instance count (edge_args kernel) — was draw(6, MAX_EDGES).
    rp.drawIndirect(s->edge_args_buf);
    rp.end();
  }

  // 6. Takeover: stochastic teleport of mismatched seeds — updates NEXT frame.
  {
    TakeoverUniforms tu = {};
    tu.count = (uint32_t)count; tu.w = (uint32_t)pw; tu.h = (uint32_t)ph; tu.frame = s->frame;
    tu.dt = dt; tu.churn = s->churn; tu.confidence = s->confidence;
    tu.aspect = aspect; tu.mode = (uint32_t)s->scoring_mode; tu.decimation = s->decimation;
    tu.bnd = (uint32_t)boundary_count(s);
    s->takeover_buf.writeOne(tu);
    auto cp = gpu::ComputePass::begin();
    cp.setPSO(s_pso_takeover);
    cp.setBuffer(s->accum_buf, 0);
    cp.setTexture(s->feat_tex, 1, 0);
    cp.setBuffer(s->seed_buf, 2);
    cp.setBuffer(s->takeover_buf, 3);
    cp.dispatch(sgx, 1);
    cp.end();
  }

  gpu::Device::submit();
  s->frame++;
}

} // namespace triangulate
