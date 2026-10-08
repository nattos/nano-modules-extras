/*
 * filter.color.ember_tone — a warm, two-tone colour grade.
 *
 * Part LUT, part saturation non-linearity, part colourise. The pixel's
 * lightness drives everything: it climbs and rolls off onto a warm, full
 * "body", the brightest still sizzle through to white, and a soft threshold
 * low in the range splits it between two tones — a STRONG hue set directly
 * and a WEAK one computed from it (its OkLCh complement, turned by an
 * offset). The crossover walks the hue wheel instead of blending through
 * grey, so every level stays punchy.
 *
 * Optionally the input's own colour steers it: each pixel's hue offset from
 * the image's dominant hue (found the way color.legacy.bicolor_grad finds its
 * major hue) turns the output hue, and the input's saturation can be mixed
 * back in.
 *
 * Three compute passes: hist (hue histogram) -> analyze (single thread: the
 * anchor hue, eased over time, and the two tones in OkLCh) -> render. The
 * analysis buffer persists per instance — the anchor's easing memory — which
 * is why this is SeekableApproximate.
 */

#include <gpu.h>
#include <host.h>
#include "ember_tone_shaders.h"

#include <cmath>
#include <cstdint>

namespace ember_tone {

static constexpr int GRID_SN = 64;          // histogram sample grid (per axis)
static constexpr int BUCKETS = 64;          // hue bins
static constexpr int ANALYSIS_FLOATS = 8;   // see common.hlsl

enum AnchorMode : int { ANCHOR_AUTO = 0, ANCHOR_FIXED = 1 };

struct HistUniforms { float grid, weight_scale, _p0, _p1; };
struct AnalyzeUniforms {
  float dt, follow, anchor_mode, anchor_hue;
  float strong_hue, weak_offset, _p0, _p1;
};
struct RenderUniforms {
  float drive, body, chroma, climb_exp;
  float sizzle, sizzle_point, threshold, threshold_soft;
  float hue_amount, hue_curve_exp, source_sat, mix;
};

// EFFECTS_STYLE_GUIDE §1.3: signed slider -> power-curve exponent.
static float signedSliderExp(float s) { return std::pow(2.0f, -s * 3.0f); }

struct State {
  gpu::Buffer  hist_buf;
  gpu::Buffer  analysis_buf;
  gpu::Buffer  hist_uniform;
  gpu::Buffer  analyze_uniform;
  gpu::Buffer  render_uniform;
  gpu::Sampler sampler;
  bool initialized = false;
  double pending_dt = 0.0;   // accumulated tick dt, consumed by render()

  // Tone
  float drive = 1.0f, body = 0.78f, chroma = 1.0f, rolloff_curve = 0.0f;
  float sizzle = 1.0f, sizzle_point = 0.85f;
  // Two tone
  float strong_hue = 0.07f, weak_offset = 35.0f;
  float threshold = 0.3f, threshold_soft = 0.1f;
  // Source hue
  float hue_amount = 0.0f, hue_curve = 0.0f, hue_follow = 0.5f;
  int   anchor = ANCHOR_AUTO;
  float anchor_hue = 0.0f;
  float source_sat = 0.0f;
  // Output
  float mix = 1.0f;
};

static gpu::ComputePSO s_pso_hist;
static gpu::ComputePSO s_pso_analyze;
static gpu::ComputePSO s_pso_render;

void module_init() {
  state::init("filter.color.ember_tone", {1, 0, 0},
    state::Schema()
      .helpField("intro",
        "## Ember Tone\n"
        "A warm two-tone grade: part LUT, part saturation, part colorize. "
        "Brightness is all it reads. As a pixel gets brighter it **climbs** "
        "and rolls off onto a full, warm **body**, and the very brightest "
        "still **sizzle** through to white.\n\n"
        "Two tones split the range: the **weak** tone in the shadows, the "
        "**strong** one everywhere above a low threshold. Crossing between "
        "them walks round the color wheel rather than through gray, so "
        "every level stays punchy.\n\n"
        "**Try:** *Weak Offset* at -60 for a magenta crossing, or *Hue Amount* "
        "at 0.5 to let the footage's own colors pull the grade apart.")

      // ---------------- Tone ----------------
      .group("tone", "Tone")
        .groupHelp(
          "How brightness becomes color. **Drive** pushes the input up the "
          "climb; **Body** is the lightness it rolls off onto, and **Chroma** "
          "how saturated the body is (as far as the screen can show that "
          "hue). **Roll-off Curve** bends the climb: negative holds the "
          "shadows down, positive lifts them onto the body sooner. Past "
          "**Sizzle Point** a pixel races through a hot tint to white; "
          "**Sizzle** 0 keeps even white input on the body.")
      .floatField("drive", 1.0f, 0.f, 4.f, state::PrimaryInput,
                  nullptr, 0.f, nullptr,
                  "Gain on the input's brightness before the climb.")
        .label("Drive", "Drive")
      .floatField("body", 0.78f, 0.2f, 1.f, state::PrimaryInput,
                  nullptr, 0.f, nullptr,
                  "The lightness the climb rolls off onto (OkLab L).")
        .label("Body", "Body")
      .floatField("chroma", 1.0f, 0.f, 1.5f, state::PrimaryInput,
                  nullptr, 0.f, nullptr,
                  "Saturation of the grade. Held to what the screen can show.")
        .label("Chroma", "Chroma")
      .floatField("rolloff_curve", 0.0f, -1.f, 1.f, state::SecondaryInput,
                  nullptr, 0.f, nullptr,
                  "Shape of the climb. -1 holds the shadows down, +1 lifts "
                  "them onto the body early. 0 is the plain shoulder.")
        .label("Roll-off Curve", "Curve")
      .floatField("sizzle", 1.0f, 0.f, 1.f, state::PrimaryInput,
                  nullptr, 0.f, nullptr,
                  "How far the brightest pixels break through to white.")
        .label("Sizzle", "Sizzle")
      .floatField("sizzle_point", 0.85f, 0.3f, 1.5f, state::SecondaryInput,
                  nullptr, 0.f, nullptr,
                  "Driven brightness where the sizzle starts. White input is 1.")
        .label("Sizzle Point", "SzPt")

      // ---------------- Two Tone ----------------
      .group("two_tone", "Two Tone")
        .groupHelp(
          "**Strong Hue** is the main color. The **weak** color is its "
          "opposite on the wheel, turned by **Weak Offset**; the offset's "
          "sign also picks which way round the wheel the crossing goes. "
          "Below **Threshold** a pixel is weak, above it strong, over a "
          "**Softness** either side.")
      .floatField("strong_hue", 0.07f, 0.f, 1.f, state::PrimaryInput,
                  nullptr, 0.f, nullptr,
                  "The main color, as a hue. 0.07 is ember orange.")
        .label("Strong Hue", "Hue")
      .floatField("weak_offset", 35.0f, -180.f, 180.f, state::PrimaryInput,
                  nullptr, 0.f, "°",
                  "Turns the weak color away from the strong one's opposite.")
        .label("Weak Offset", "WkOff")
      .floatField("threshold", 0.3f, 0.f, 1.f, state::PrimaryInput,
                  nullptr, 0.f, nullptr,
                  "Input lightness where weak turns to strong. Low keeps most "
                  "of the image strong.")
        .label("Threshold", "Thresh")
      .floatField("threshold_soft", 0.1f, 0.f, 0.5f, state::SecondaryInput,
                  nullptr, 0.f, nullptr,
                  "How gradual the crossing is, either side of the threshold.")
        .label("Softness", "Soft")

      // ---------------- Source Hue ----------------
      .group("source_hue", "Source Hue")
        .groupHelp(
          "Let the footage's own color back in. The **anchor** is the "
          "image's dominant hue (or **Anchor Hue**, in Fixed mode). A pixel "
          "whose hue differs from it turns the grade's hue by that "
          "difference times **Hue Amount** — negative turns the other way. "
          "**Hue Curve** bends it: negative leaves near-anchor colors alone "
          "and swings the far ones, positive swings even small differences. "
          "Gray pixels have no hue and never turn. **Follow** is how long "
          "the anchor takes to move with the footage. **Source Saturation** "
          "lets dull input stay dull.")
      .floatField("hue_amount", 0.0f, -1.f, 1.f, state::PrimaryInput,
                  nullptr, 0.f, nullptr,
                  "How much each pixel's hue difference turns the output. 1 "
                  "turns it by the full difference.")
        .label("Hue Amount", "HueAmt")
      .floatField("hue_curve", 0.0f, -1.f, 1.f, state::SecondaryInput,
                  nullptr, 0.f, nullptr,
                  "-1 swings only the far colors, +1 even the near ones. "
                  "0 is linear.")
        .label("Hue Curve", "HueCrv")
      .floatField("hue_follow", 0.5f, 0.f, 4.f, state::SecondaryInput,
                  nullptr, 0.f, "s",
                  "Time the anchor takes to follow the footage's dominant hue.")
        .label("Follow", "Follow")
      .selectField("anchor", ANCHOR_AUTO, state::SecondaryInput,
                   {{"Auto", ANCHOR_AUTO}, {"Fixed", ANCHOR_FIXED}}, false,
                   "Auto finds the dominant hue in the image; Fixed uses "
                   "Anchor Hue.")
        .label("Anchor", "Anchor")
      .floatField("anchor_hue", 0.0f, 0.f, 1.f, state::SecondaryInput,
                  nullptr, 0.f, nullptr,
                  "The anchor in Fixed mode, as a hue.")
        .label("Anchor Hue", "AncHue")
      .floatField("source_sat", 0.0f, 0.f, 1.f, state::PrimaryInput,
                  nullptr, 0.f, nullptr,
                  "Mixes in the input's saturation: 1 leaves gray input gray.")
        .label("Source Saturation", "SrcSat")

      // ---------------- Output ----------------
      .group("output", "Output")
      .floatField("mix", 1.0f, 0.f, 1.f, state::PrimaryInput,
                  nullptr, 0.f, nullptr,
                  "Dry input at 0, the full grade at 1.")
        .label("Mix", "Mix")
      .textureField("tex_in",  state::PrimaryInput)
      .textureField("tex_out", state::PrimaryOutput)
      .capability(state::Capability::SeekableApproximate)
  );

  if (gpu::Device::backend() == gpu::Backend::None) return;

  state::registerShaderSPV("ember_tone_hist",    HIST_SPV,    HIST_SPV_SIZE);
  state::registerShaderSPV("ember_tone_analyze", ANALYZE_SPV, ANALYZE_SPV_SIZE);
  state::registerShaderSPV("ember_tone_render",  RENDER_SPV,  RENDER_SPV_SIZE);

  auto cs_hist    = gpu::Device::createShaderModuleByName("ember_tone_hist");
  auto cs_analyze = gpu::Device::createShaderModuleByName("ember_tone_analyze");
  auto cs_render  = gpu::Device::createShaderModuleByName("ember_tone_render");
  if (!cs_hist || !cs_analyze || !cs_render) return;

  s_pso_hist = gpu::Device::createComputePSO(cs_hist, "main", gpu::Bindings()
      .tex2d(0).sampler(1).storageRW(2).uniform(3));
  s_pso_analyze = gpu::Device::createComputePSO(cs_analyze, "main", gpu::Bindings()
      .storage(0).storageRW(1).uniform(2));
  s_pso_render = gpu::Device::createComputePSO(cs_render, "main", gpu::Bindings()
      .tex2d(0).sampler(1).storage(2).storageTex2d(3).uniform(4));
}

void* create() {
  auto* s = new State();
  s->hist_buf        = gpu::Device::createBuffer(BUCKETS * sizeof(int32_t), gpu::BufferUsage::Storage);
  s->analysis_buf    = gpu::Device::createBuffer(ANALYSIS_FLOATS * sizeof(float), gpu::BufferUsage::Storage);
  s->hist_uniform    = gpu::Device::createBuffer(sizeof(HistUniforms), gpu::BufferUsage::Uniform);
  s->analyze_uniform = gpu::Device::createBuffer(sizeof(AnalyzeUniforms), gpu::BufferUsage::Uniform);
  s->render_uniform  = gpu::Device::createBuffer(sizeof(RenderUniforms), gpu::BufferUsage::Uniform);
  s->sampler         = gpu::Device::createSampler(gpu::FilterMode::Linear, gpu::AddressMode::ClampToEdge);
  return s;
}

void destroy(void* self) {
  auto* s = static_cast<State*>(self);
  if (!s) return;
  s->hist_buf.release();
  s->analysis_buf.release();
  s->hist_uniform.release();
  s->analyze_uniform.release();
  s->render_uniform.release();
  s->sampler.release();
  delete s;
}

void init(void* self) {
  auto* s = static_cast<State*>(self);
  if (!s) return;
  if (!s_pso_hist.valid() || !s_pso_analyze.valid() || !s_pso_render.valid()) return;
  if (!s->analysis_buf.valid()) return;
  // Not initialized (slot 2 = 0): the first analysis snaps to the image.
  float seed[ANALYSIS_FLOATS] = {1.0f, 0.0f};
  s->analysis_buf.write(seed, ANALYSIS_FLOATS);
  s->pending_dt = 0.0;
  s->initialized = true;
}

void tick(void* self, double dt) {
  auto* s = static_cast<State*>(self);
  if (!s) return;
  // Accumulated here, consumed in render() — the runner may tick several
  // times per render, or render without ticking.
  if (dt > 0.0) s->pending_dt += dt;
  if (s->pending_dt > 0.25) s->pending_dt = 0.25;
}

void on_resolume_param(void*, long long, double) {}

void on_state_patched(void* self, int n, const char* pb, const int* off,
                      const int* len, const int* ops) {
  auto* s = static_cast<State*>(self);
  if (!s) return;
  for (int i = 0; i < n; i++) {
    if (ops[i] != state::PatchReplace) continue;
    const char* p = pb + off[i];
    int l = len[i];
    if      (state::pathIs(p, l, "drive"))          s->drive = state::patchFloat(i);
    else if (state::pathIs(p, l, "body"))           s->body = state::patchFloat(i);
    else if (state::pathIs(p, l, "chroma"))         s->chroma = state::patchFloat(i);
    else if (state::pathIs(p, l, "rolloff_curve"))  s->rolloff_curve = state::patchFloat(i);
    else if (state::pathIs(p, l, "sizzle"))         s->sizzle = state::patchFloat(i);
    else if (state::pathIs(p, l, "sizzle_point"))   s->sizzle_point = state::patchFloat(i);
    else if (state::pathIs(p, l, "strong_hue"))     s->strong_hue = state::patchFloat(i);
    else if (state::pathIs(p, l, "weak_offset"))    s->weak_offset = state::patchFloat(i);
    else if (state::pathIs(p, l, "threshold"))      s->threshold = state::patchFloat(i);
    else if (state::pathIs(p, l, "threshold_soft")) s->threshold_soft = state::patchFloat(i);
    else if (state::pathIs(p, l, "hue_amount"))     s->hue_amount = state::patchFloat(i);
    else if (state::pathIs(p, l, "hue_curve"))      s->hue_curve = state::patchFloat(i);
    else if (state::pathIs(p, l, "hue_follow"))     s->hue_follow = state::patchFloat(i);
    else if (state::pathIs(p, l, "anchor"))         s->anchor = state::patchInt(i);
    else if (state::pathIs(p, l, "anchor_hue"))     s->anchor_hue = state::patchFloat(i);
    else if (state::pathIs(p, l, "source_sat"))     s->source_sat = state::patchFloat(i);
    else if (state::pathIs(p, l, "mix"))            s->mix = state::patchFloat(i);
  }
}

void render(void* self, int vp_w, int vp_h) {
  auto* s = static_cast<State*>(self);
  if (!s || !s->initialized || vp_w <= 0 || vp_h <= 0) return;

  auto in  = gpu::Device::textureForField("tex_in");
  auto out = gpu::Device::textureForField("tex_out");
  if (!in.valid() || !out.valid()) return;

  const bool fixed = s->anchor == ANCHOR_FIXED;

  HistUniforms hu = { (float)GRID_SN, 256.0f, 0.f, 0.f };
  s->hist_uniform.writeOne(hu);

  AnalyzeUniforms au = {};
  au.dt          = (float)s->pending_dt;
  au.follow      = s->hue_follow;
  au.anchor_mode = fixed ? 1.0f : 0.0f;
  au.anchor_hue  = s->anchor_hue;
  au.strong_hue  = s->strong_hue;
  au.weak_offset = s->weak_offset / 360.0f;
  s->analyze_uniform.writeOne(au);

  RenderUniforms ru = {};
  ru.drive          = s->drive;
  ru.body           = s->body;
  ru.chroma         = s->chroma;
  ru.climb_exp      = signedSliderExp(s->rolloff_curve);
  ru.sizzle         = s->sizzle;
  ru.sizzle_point   = s->sizzle_point;
  ru.threshold      = s->threshold;
  ru.threshold_soft = s->threshold_soft;
  ru.hue_amount     = s->hue_amount;
  ru.hue_curve_exp  = signedSliderExp(s->hue_curve);
  ru.source_sat     = s->source_sat;
  ru.mix            = s->mix;
  s->render_uniform.writeOne(ru);

  // Pass 1 — histogram. Always run (64x64 samples), even while nothing reads
  // the anchor, so it is already on the footage when Hue Amount comes up.
  int32_t zeros[BUCKETS] = {0};
  s->hist_buf.write(zeros, BUCKETS);
  {
    auto cp = gpu::ComputePass::begin();
    cp.setPSO(s_pso_hist);
    cp.setTexture(in, 0, 0);
    cp.setSampler(s->sampler, 1);
    cp.setBuffer(s->hist_buf, 2);
    cp.setBuffer(s->hist_uniform, 3);
    cp.dispatch((GRID_SN + 7) / 8, (GRID_SN + 7) / 8);
    cp.end();
  }

  // Pass 2 — analyze (single thread).
  {
    auto cp = gpu::ComputePass::begin();
    cp.setPSO(s_pso_analyze);
    cp.setBuffer(s->hist_buf, 0);
    cp.setBuffer(s->analysis_buf, 1);
    cp.setBuffer(s->analyze_uniform, 2);
    cp.dispatch(1, 1, 1);
    cp.end();
  }

  // Pass 3 — the grade.
  {
    auto cp = gpu::ComputePass::begin();
    cp.setPSO(s_pso_render);
    cp.setTexture(in, 0, 0);
    cp.setSampler(s->sampler, 1);
    cp.setBuffer(s->analysis_buf, 2);
    cp.setTexture(out, 3, 1);
    cp.setBuffer(s->render_uniform, 4);
    cp.dispatch((vp_w + 7) / 8, (vp_h + 7) / 8);
    cp.end();
  }

  gpu::Device::submit();
  s->pending_dt = 0.0;
}

} // namespace ember_tone
