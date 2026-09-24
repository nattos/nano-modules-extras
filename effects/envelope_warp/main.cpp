/*
 * warp.envelope — Envelope Warp.
 *
 * Warps the input along an arbitrary user-drawn parametric envelope (the same
 * (x, y, ease) piecewise curve as mod.shaper.envelope, drawn with the same
 * graph editor). The curve maps SOURCE position → WARPED position; the
 * identity diagonal (0,0)→(1,1) is no warp. Symmetry modes mirror the curve
 * about the viewport center (per axis), apply two independent curves, or wrap
 * it radially by distance from a center point.
 *
 * Rendering is geometry-driven, but the quads rasterize 1D COORDINATE MAPS,
 * not the image: each envelope segment becomes one instanced quad in an
 * (N x 1) rgba32float map (R = analytic source coordinate — the per-segment
 * exponential ease is exactly invertible — G = coverage). Non-monotonic curves
 * fold the image over itself via painter's order (later segments overwrite;
 * Replace blend). A single compute resolve then composes the per-axis maps
 * (separable warps compose coordinate-wise) and samples the input ONCE per
 * output pixel — no scratch image, no double resample. Radial reads its map by
 * per-pixel radius (1 = half the longer viewport axis), so it needs no annular
 * geometry at all.
 *
 * Uncovered destination regions (the curve squeezes the image) either smear
 * the nearest covered source (Stretch — matching the envelope's clamp-flat
 * semantics) or clear to transparent. Outside the authored radial domain
 * (r > 1, the viewport corners) the warp continues rigidly at slope 1 so an
 * identity curve leaves the corners untouched.
 *
 * Class-like instance ABI; type-shared PSOs are file-static.
 */

#include <gpu.h>
#include <host.h>
#include <sketch/envelope.h>
#include "envelope_warp_shaders.h"

#include <cmath>
#include <cstring>

namespace envelope_warp {

enum Mode : int {
  MODE_H      = 0,   // X only, mirrored left/right
  MODE_V      = 1,   // Y only, mirrored top/bottom
  MODE_XY     = 2,   // one curve, both axes, mirrored
  MODE_X_AND_Y = 3,  // two curves, both axes, mirrored
  MODE_RECT   = 4,   // two curves, both axes, NOT mirrored
  MODE_RADIAL = 5,   // one curve, radial by distance from center
};

enum Edges : int { EDGES_STRETCH = 0, EDGES_TRANSPARENT = 1 };

static const char kDefaultCurve[] = "[0,0,0,1,1,0]";

// Radial map domain: r = 1 is half the LONGER viewport axis, so the farthest
// corner is at r = sqrt(1 + (short/long)^2) <= sqrt(2) < 1.5.
static constexpr float kRadialDom = 1.5f;
static constexpr int   kRadialMapN = 4096;

// Per curve: 2 stretch edge fills + 2 rigid head/tail continuations + 63
// segments = 67 records, x2 when mirrored.
static constexpr int kMaxRecs = 134;

struct SegRec { float d0, d1, s0, s1, inv_e, _p0, _p1, _p2; };
static_assert(sizeof(SegRec) == 8 * 4, "SegRec layout");

struct VsU      { float dom, _p0, _p1, _p2; };
struct ResolveU {
  float radial, warp_x, warp_y, dom;
  float center_x, center_y, r_scale, _pad0;
};
static_assert(sizeof(VsU) == 4 * 4, "VsU layout");
static_assert(sizeof(ResolveU) == 8 * 4, "ResolveU layout");

struct State {
  // --- Schema-mirrored params ---
  int   mode   = MODE_H;
  float amount = 1.0f;
  int   edges  = EDGES_STRETCH;
  float center[2] = {0.0f, 0.0f};   // cover-square coords (radial only)
  char  curveJson[2048]  = {};
  char  curveYJson[2048] = {};

  // Parsed curves (fallback: the identity line).
  envelope::Point ptsA[envelope::kMaxPoints]; int nA = 0;   // `curve`
  envelope::Point ptsB[envelope::kMaxPoints]; int nB = 0;   // `curve_y`

  // --- GPU resources (per-instance) ---
  gpu::Buffer  seg_buf_a, seg_buf_b;    // quad records: curve / curve_y
  int          seg_count_a = 0, seg_count_b = 0;
  gpu::Buffer  vs_u_axis, vs_u_radial;
  gpu::Buffer  resolve_u;
  gpu::Sampler sampler;
  gpu::Texture mapX, mapY, mapR;        // (vp_w x 1), (vp_h x 1), (4096 x 1)
  int   map_w = 0, map_h = 0;
  bool  maps_dirty = true;              // re-raster the coordinate maps
  bool  initialized = false;
};

static gpu::RenderPSO  s_pso_map;       // rgba32float target, Replace blend
static gpu::ComputePSO s_pso_resolve;

// ---------------------------------------------------------------------------
// Curve handling

static void parseCurve(const char* json, envelope::Point* pts, int& n) {
  n = envelope::parse(json, pts, envelope::kMaxPoints);
  if (n < 2) {   // empty / malformed → identity passthrough
    pts[0] = {0.0f, 0.0f, 0.0f};
    pts[1] = {1.0f, 1.0f, 0.0f};
    n = 2;
  }
}

static bool isIdentityCurve(const envelope::Point* pts, int n) {
  for (int i = 0; i < n; i++) {
    if (std::fabs(pts[i].y - pts[i].x) > 1e-6f) return false;
    if (i + 1 < n && std::fabs(pts[i].ease) > 1e-6f) return false;
  }
  return true;
}

// Build the quad records for one curve. Works in the curve's own domain
// (h ∈ [0, 1], or radius units), then expands mirrored records onto both sides
// of the axis center. Painter's order: stretch edge fills first, then the
// rigid radial head/tail continuations, then the curve segments — so the drawn
// curve wins every fold-over overlap. Returns the record count.
static int buildSegs(const envelope::Point* pts, int n, float amount, bool mirrored,
                     bool radial, bool stretchEdges, SegRec* out) {
  const float dom = radial ? kRadialDom : 1.0f;
  float ex[envelope::kMaxPoints], ey[envelope::kMaxPoints], ee[envelope::kMaxPoints];
  for (int i = 0; i < n; i++) {
    ex[i] = pts[i].x;
    ey[i] = pts[i].x + (pts[i].y - pts[i].x) * amount;   // amount → identity lerp
    ee[i] = pts[i].ease * amount;
  }

  struct Tmp { float d0, d1, s0, s1, inv; };
  Tmp tmp[kMaxRecs / 2];
  int m = 0;

  // Radial rigid continuations: outside the authored [x0, xn] domain the warp
  // continues at slope 1 (a pure translation), so an identity curve leaves the
  // viewport corners (r > 1) untouched.
  Tmp head = {}, tail = {};
  bool hasHead = false, hasTail = false;
  if (radial && ex[0] > 1e-6f) {
    head = {ey[0] - ex[0], ey[0], 0.0f, ex[0], 1.0f};
    hasHead = true;
  }
  if (radial && ex[n - 1] < dom - 1e-6f) {
    tail = {ey[n - 1], ey[n - 1] + (dom - ex[n - 1]), ex[n - 1], dom, 1.0f};
    hasTail = true;
  }

  // Destination coverage extremes (and the source that lands there) over
  // everything drawn — the stretch edge fills smear those source positions
  // into the uncovered bands.
  float dmin = 1e9f, dmax = -1e9f, sAtMin = 0.0f, sAtMax = 0.0f;
  auto track = [&](float d, float s) {
    if (d < dmin) { dmin = d; sAtMin = s; }
    if (d > dmax) { dmax = d; sAtMax = s; }
  };
  if (hasHead) { track(head.d0, head.s0); track(head.d1, head.s1); }
  if (hasTail) { track(tail.d0, tail.s0); track(tail.d1, tail.s1); }
  for (int i = 0; i < n; i++) track(ey[i], ex[i]);

  if (stretchEdges) {
    if (dmin > 1e-6f)       tmp[m++] = {0.0f, dmin, sAtMin, sAtMin, 1.0f};
    if (dmax < dom - 1e-6f) tmp[m++] = {dmax, dom, sAtMax, sAtMax, 1.0f};
  }
  if (hasHead) tmp[m++] = head;
  if (hasTail) tmp[m++] = tail;
  for (int i = 0; i + 1 < n; i++) {
    if (std::fabs(ey[i + 1] - ey[i]) < 1e-6f) continue;   // flat: no dest span
    tmp[m++] = {ey[i], ey[i + 1], ex[i], ex[i + 1], std::pow(2.0f, 3.0f * ee[i])};
  }

  int cnt = 0;
  for (int i = 0; i < m; i++) {
    if (!mirrored) {
      out[cnt++] = {tmp[i].d0, tmp[i].d1, tmp[i].s0, tmp[i].s1, tmp[i].inv, 0, 0, 0};
    } else {
      // h ∈ [0, 1] is the distance from the axis center: u = 0.5 ± h/2.
      out[cnt++] = {0.5f + tmp[i].d0 * 0.5f, 0.5f + tmp[i].d1 * 0.5f,
                    0.5f + tmp[i].s0 * 0.5f, 0.5f + tmp[i].s1 * 0.5f, tmp[i].inv, 0, 0, 0};
      out[cnt++] = {0.5f - tmp[i].d0 * 0.5f, 0.5f - tmp[i].d1 * 0.5f,
                    0.5f - tmp[i].s0 * 0.5f, 0.5f - tmp[i].s1 * 0.5f, tmp[i].inv, 0, 0, 0};
    }
  }
  return cnt;
}

// ---------------------------------------------------------------------------
// Visibility: curve_y only exists in the two-curve modes; center is radial-only.

static void apply_visibility(int mode) {
  const bool twoCurves = (mode == MODE_X_AND_Y || mode == MODE_RECT);
  state::setFieldHidden("curve_y", !twoCurves);
  state::setFieldHidden("center", mode != MODE_RADIAL);
}

void eval_visibility(int n, const char* pb, const int* off, const int* len, const int* ops) {
  int mode = MODE_H;
  for (int i = 0; i < n; i++) {
    if (ops[i] != state::PatchReplace) continue;
    if (state::pathIs(pb + off[i], len[i], "mode")) mode = state::patchInt(i);
  }
  apply_visibility(mode);
}

static void on_state_ready(void* self) {
  auto* s = static_cast<State*>(self);
  if (s) apply_visibility(s->mode);
}

// ---------------------------------------------------------------------------

void module_init() {
  state::init("warp.envelope", {1, 0, 0},
    state::Schema()
      .helpField("intro",
        "## Envelope Warp\n"
        "Warps the image along a curve you **draw by hand** — horizontal "
        "position (or vertical, or distance from the center) is remapped "
        "through the graph. The straight diagonal is no warp; lift it to "
        "bulge, dip it to pinch, and make it **non-monotonic** to fold the "
        "image over itself.\n\n"
        "**Try:** in *XY*, drag the middle of the curve up for a fisheye "
        "bulge. Draw an S for a crunchy center-squeeze. Wire an LFO into "
        "*Amount* to breathe between flat and warped.")
      .group("warp", "Warp")
        .groupHelp(
          "*Symmetry* picks how the curve wraps the frame: mirrored about the "
          "center per axis (Horizontal / Vertical / XY), two independent "
          "curves (X and Y mirrored, or Rect edge-to-edge), or Radial by "
          "distance from *Center* (1 on the graph = half the longer axis). "
          "*Amount* morphs between no warp (0) and the full curve (1). "
          "*Edges* fills regions the curve leaves uncovered: Stretch smears "
          "the nearest edge of the image; Transparent cuts them out.")
      .selectField("mode", MODE_H, state::PrimaryInput, {
        {"Horizontal", MODE_H}, {"Vertical", MODE_V}, {"XY", MODE_XY},
        {"X and Y", MODE_X_AND_Y}, {"Rect", MODE_RECT}, {"Radial", MODE_RADIAL},
      }, false, "How the curve maps onto the frame (mirrored axes, two curves, or radial).")
        .label("Symmetry", "Sym")
      .floatField("amount", 1.0f, 0.0f, 1.0f, state::PrimaryInput, nullptr, 0.01f, nullptr,
                  "Warp strength: 0 = passthrough, 1 = the drawn curve.").label("Amount", "Amt")
      .selectField("edges", EDGES_STRETCH, state::SecondaryInput, {
        {"Stretch", EDGES_STRETCH}, {"Transparent", EDGES_TRANSPARENT},
      }, false, "Fill for destination regions the curve doesn't cover.").label("Edges", "Edge")
      .vec2Field("center", 0.0f, 0.0f, state::SecondaryInput, -1.0f, 1.0f)
        .label("Center", "Ctr")
      .group("curves", "Curves")
        .groupHelp(
          "The curve maps source position (horizontal) to warped position "
          "(vertical). In mirrored modes 0 is the **center** of the frame and "
          "1 the edge; in Rect 0 is the left/top edge. Double-click to add or "
          "remove points; drag between points to bend a segment's easing. "
          "Fold-overs (curve going back down) draw later parts of the image "
          "on top.")
      .textField("curve", kDefaultCurve, state::SecondaryInput).label("Curve", "Crv")
      .textField("curve_y", kDefaultCurve, state::SecondaryInput).label("Y Curve", "CrvY")
      .endGroup()
      .capability(state::Capability::TimeIndependent)
      .textureField("tex_in",  state::PrimaryInput)
      .textureField("tex_out", state::PrimaryOutput)
  );

  if (gpu::Device::backend() == gpu::Backend::None) return;

  state::registerShaderSPV("envelope_warp_vs",      VS_SPV,      VS_SPV_SIZE);
  state::registerShaderSPV("envelope_warp_fs",      FS_SPV,      FS_SPV_SIZE);
  state::registerShaderSPV("envelope_warp_resolve", RESOLVE_SPV, RESOLVE_SPV_SIZE);

  auto vs = gpu::Device::createShaderModuleByName("envelope_warp_vs");
  auto fs = gpu::Device::createShaderModuleByName("envelope_warp_fs");
  auto cs = gpu::Device::createShaderModuleByName("envelope_warp_resolve");
  if (!vs || !fs || !cs) return;

  // Replace blend is load-bearing twice over: rgba32float isn't blendable, and
  // fold-over needs strict painter's-order overwrite.
  s_pso_map = gpu::Device::createInstancedRenderPSO(
      vs, "main", fs, "main",
      gpu::TextureFormat::RGBA32F,
      gpu::Bindings()
          .storage(0)       // SegRec[] (vertex reads)
          .uniform(1),      // VsU
      gpu::Device::BlendMode::Replace);

  s_pso_resolve = gpu::Device::createComputePSO(cs, "main", gpu::Bindings()
      .tex2d(0)             // inputTex
      .sampler(1)
      // The maps are rgba32float → unfilterable-float on WebGPU (read with
      // Load / manual lerp only; the format here is the layout hint).
      .tex2d(2, gpu::TextureFormat::RGBA32F)   // mapX (or the radial map)
      .tex2d(3, gpu::TextureFormat::RGBA32F)   // mapY
      .storageTex2d(4)      // outTex
      .uniform(5));         // ResolveU

  state::log("envelope_warp: module initialized");
}

void* create() {
  auto* s = new State();
  s->seg_buf_a = gpu::Device::createBuffer(kMaxRecs * sizeof(SegRec), gpu::BufferUsage::Storage);
  s->seg_buf_b = gpu::Device::createBuffer(kMaxRecs * sizeof(SegRec), gpu::BufferUsage::Storage);
  s->vs_u_axis   = gpu::Device::createBuffer(sizeof(VsU), gpu::BufferUsage::Uniform);
  s->vs_u_radial = gpu::Device::createBuffer(sizeof(VsU), gpu::BufferUsage::Uniform);
  s->resolve_u   = gpu::Device::createBuffer(sizeof(ResolveU), gpu::BufferUsage::Uniform);
  s->sampler = gpu::Device::createSampler(gpu::FilterMode::Linear, gpu::AddressMode::ClampToEdge);
  return s;
}

void destroy(void* self) {
  auto* s = static_cast<State*>(self);
  if (!s) return;
  s->seg_buf_a.release();
  s->seg_buf_b.release();
  s->vs_u_axis.release();
  s->vs_u_radial.release();
  s->resolve_u.release();
  s->sampler.release();
  if (s->mapX.valid()) s->mapX.release();
  if (s->mapY.valid()) s->mapY.release();
  if (s->mapR.valid()) s->mapR.release();
  delete s;
}

void init(void* self) {
  auto* s = static_cast<State*>(self);
  if (!s) return;
  if (!s_pso_map.valid() || !s_pso_resolve.valid()) return;

  std::strncpy(s->curveJson, kDefaultCurve, sizeof(s->curveJson) - 1);
  std::strncpy(s->curveYJson, kDefaultCurve, sizeof(s->curveYJson) - 1);
  parseCurve(s->curveJson, s->ptsA, s->nA);
  parseCurve(s->curveYJson, s->ptsB, s->nB);

  VsU ua = {1.0f, 0, 0, 0};
  VsU ur = {kRadialDom, 0, 0, 0};
  s->vs_u_axis.writeOne(ua);
  s->vs_u_radial.writeOne(ur);

  state::setOnStateReady(&on_state_ready);
  s->maps_dirty = true;
  s->initialized = true;
}

void tick(void* self, double dt) { (void)self; (void)dt; }
void on_resolume_param(void*, long long, double) {}

int32_t is_identity(void* self) {
  auto* s = static_cast<State*>(self);
  if (!s || !s->initialized) return 0;
  if (s->amount <= 0.0f) return 1;
  const bool aIdent = isIdentityCurve(s->ptsA, s->nA);
  if (s->mode == MODE_X_AND_Y || s->mode == MODE_RECT) {
    return (aIdent && isIdentityCurve(s->ptsB, s->nB)) ? 1 : 0;
  }
  return aIdent ? 1 : 0;
}

void on_state_patched(void* self, int n, const char* pb, const int* off,
                      const int* len, const int* ops) {
  auto* s = static_cast<State*>(self);
  if (!s) return;
  for (int i = 0; i < n; i++) {
    if (ops[i] != state::PatchReplace) continue;
    const char* p = pb + off[i];
    int l = len[i];
    if (state::pathIs(p, l, "mode")) {
      int mode = state::patchInt(i);
      if (mode != s->mode) { s->mode = mode; s->maps_dirty = true; apply_visibility(mode); }
    } else if (state::pathIs(p, l, "amount")) {
      float v = state::patchFloat(i);
      if (v != s->amount) { s->amount = v; s->maps_dirty = true; }
    } else if (state::pathIs(p, l, "edges")) {
      int v = state::patchInt(i);
      if (v != s->edges) { s->edges = v; s->maps_dirty = true; }
    } else if (state::pathIs(p, l, "center")) {
      auto v = state::patchVec2(i);
      s->center[0] = v.x; s->center[1] = v.y;   // uniform-only: no re-raster
    } else if (state::pathIs(p, l, "curve")) {
      char buf[sizeof(State::curveJson)];
      state::patchString(i, buf, sizeof(buf));
      if (std::strcmp(buf, s->curveJson) != 0) {
        std::memcpy(s->curveJson, buf, sizeof(buf));
        parseCurve(s->curveJson, s->ptsA, s->nA);
        s->maps_dirty = true;
      }
    } else if (state::pathIs(p, l, "curve_y")) {
      char buf[sizeof(State::curveYJson)];
      state::patchString(i, buf, sizeof(buf));
      if (std::strcmp(buf, s->curveYJson) != 0) {
        std::memcpy(s->curveYJson, buf, sizeof(buf));
        parseCurve(s->curveYJson, s->ptsB, s->nB);
        s->maps_dirty = true;
      }
    }
  }
}

// Rasterize one curve's records into a 1D coordinate map. The clear IS the
// Transparent edge fill; a zero-record build (possible with Transparent edges
// and an all-flat curve) still clears.
static void rasterMap(gpu::Texture target, gpu::Buffer segBuf, int count, gpu::Buffer vsU) {
  auto rp = gpu::RenderPass::begin(target, 0.0f, 0.0f, 0.0f, 0.0f);
  rp.setPSO(s_pso_map);
  rp.setBuffer(segBuf, 0);
  rp.setBuffer(vsU, 1);
  if (count > 0) rp.draw(6, count);
  rp.end();
}

void render(void* self, int vp_w, int vp_h) {
  auto* s = static_cast<State*>(self);
  if (!s || !s->initialized || vp_w <= 0 || vp_h <= 0) return;

  auto in  = gpu::Device::textureForField("tex_in");
  auto out = gpu::Device::textureForField("tex_out");
  if (!in.valid() || !out.valid()) return;

  const bool radial  = (s->mode == MODE_RADIAL);
  const bool warpX   = (s->mode == MODE_H || s->mode == MODE_XY ||
                        s->mode == MODE_X_AND_Y || s->mode == MODE_RECT);
  const bool warpY   = (s->mode == MODE_V || s->mode == MODE_XY ||
                        s->mode == MODE_X_AND_Y || s->mode == MODE_RECT);
  const bool mirrored = (s->mode != MODE_RECT);   // (radial ignores this)
  const bool stretch  = (s->edges == EDGES_STRETCH);

  // (Re)allocate the per-axis maps at the viewport size. The radial map is a
  // fixed-resolution 1D LUT, allocated on first radial use.
  if (!radial && (s->map_w != vp_w || s->map_h != vp_h)) {
    if (s->mapX.valid()) s->mapX.release();
    if (s->mapY.valid()) s->mapY.release();
    s->mapX = gpu::Device::createTexture(vp_w, 1, gpu::TextureFormat::RGBA32F);
    s->mapY = gpu::Device::createTexture(vp_h, 1, gpu::TextureFormat::RGBA32F);
    s->map_w = vp_w; s->map_h = vp_h;
    s->maps_dirty = true;
  }
  if (radial && !s->mapR.valid()) {
    s->mapR = gpu::Device::createTexture(kRadialMapN, 1, gpu::TextureFormat::RGBA32F);
    s->maps_dirty = true;
  }

  if (s->maps_dirty) {
    SegRec recs[kMaxRecs];
    if (radial) {
      s->seg_count_a = buildSegs(s->ptsA, s->nA, s->amount, /*mirrored=*/false,
                                 /*radial=*/true, stretch, recs);
      s->seg_buf_a.write(recs, s->seg_count_a);
      rasterMap(s->mapR, s->seg_buf_a, s->seg_count_a, s->vs_u_radial);
    } else {
      if (warpX) {
        s->seg_count_a = buildSegs(s->ptsA, s->nA, s->amount, mirrored,
                                   /*radial=*/false, stretch, recs);
        s->seg_buf_a.write(recs, s->seg_count_a);
        rasterMap(s->mapX, s->seg_buf_a, s->seg_count_a, s->vs_u_axis);
      }
      if (warpY) {
        // XY reuses the X curve; the two-curve modes rasterize curve_y.
        const bool useB = (s->mode == MODE_X_AND_Y || s->mode == MODE_RECT);
        const envelope::Point* pts = useB ? s->ptsB : s->ptsA;
        int n = useB ? s->nB : s->nA;
        s->seg_count_b = buildSegs(pts, n, s->amount, mirrored,
                                   /*radial=*/false, stretch, recs);
        s->seg_buf_b.write(recs, s->seg_count_b);
        rasterMap(s->mapY, s->seg_buf_b, s->seg_count_b, s->vs_u_axis);
      }
      // V mode without an X pass still binds mapX in the resolve — make sure
      // it exists (the resolve only READS it when warp_x is set, but the bind
      // group needs a valid texture).
    }
    s->maps_dirty = false;
  }

  const float maxAxis = (float)(vp_w > vp_h ? vp_w : vp_h);
  ResolveU ru = {};
  ru.radial   = radial ? 1.0f : 0.0f;
  ru.warp_x   = warpX ? 1.0f : 0.0f;
  ru.warp_y   = warpY ? 1.0f : 0.0f;
  ru.dom      = kRadialDom;
  ru.center_x = vp_w * 0.5f + s->center[0] * maxAxis * 0.5f;
  ru.center_y = vp_h * 0.5f + s->center[1] * maxAxis * 0.5f;
  ru.r_scale  = 2.0f / maxAxis;
  s->resolve_u.writeOne(ru);

  gpu::Texture mx = radial ? s->mapR : s->mapX;
  gpu::Texture my = radial ? s->mapR : s->mapY;   // unread in radial mode

  auto cp = gpu::ComputePass::begin();
  cp.setPSO(s_pso_resolve);
  cp.setTexture(in, 0, 0);
  cp.setSampler(s->sampler, 1);
  cp.setTexture(mx, 2, 0);
  cp.setTexture(my, 3, 0);
  cp.setTexture(out, 4, 1);
  cp.setBuffer(s->resolve_u, 5);
  cp.dispatch((vp_w + 7) / 8, (vp_h + 7) / 8);
  cp.end();

  gpu::Device::submit();
}

} // namespace envelope_warp
