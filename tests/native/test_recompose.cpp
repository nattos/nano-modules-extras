// test_recompose.cpp — end-to-end GPU render of warp.recompose from nano.wasm
// on a real Metal backend. Exercises the full four-pass pipeline (accumulate →
// weigh → solve → render).
//
// These tests assert the ANALYSIS, not just that pixels moved: the effect
// publishes its measured imbalance as scalar outputs, so `publishedScalar`
// lets us check the measurement directly, and a luminance centroid of the
// output frame lets us check that the correction actually moved the image's
// centre of mass toward the rule-of-thirds power point.

#include <catch2/catch_test_macros.hpp>

#include <cmath>
#include <fstream>
#include <vector>

#include "bridge/param_cache.h"
#include "gpu/gpu_backend.h"
#include "runtime/effect_runtime.h"
#include "sketch/module_registry.h"
#include "wasm/wasm_host.h"

#include "wasm_paths.h"

using bridge::ParamCache;
using wasm::WasmHost;
using wasm::WasmEffectDesc;
using effect_runtime::EffectRuntime;
using effect_runtime::EffectInstance;

#ifndef NANO_WASM_PATH
#error "NANO_WASM_PATH must be defined"
#endif

static std::vector<uint8_t> load_file(const char* path) {
  std::ifstream f(path, std::ios::binary | std::ios::ate);
  if (!f) return {};
  auto size = f.tellg();
  f.seekg(0);
  std::vector<uint8_t> buf(static_cast<size_t>(size));
  f.read(reinterpret_cast<char*>(buf.data()), size);
  return buf;
}

// A bright disc on black at normalized (fx, fy), radius fr (fraction of width).
// A disc fires every saliency term at once — a hard rim (Detail) and a bright
// core far from a mostly-black frame mean (Contrast) — so these tests don't
// depend on any single weight being dominant.
static std::vector<uint8_t> makeBlob(uint32_t W, uint32_t H,
                                     float fx, float fy, float fr) {
  std::vector<uint8_t> px(W * H * 4, 0);
  for (uint32_t y = 0; y < H; ++y)
    for (uint32_t x = 0; x < W; ++x) {
      float dx = ((float)x + 0.5f) / (float)W - fx;
      float dy = ((float)y + 0.5f) / (float)H - fy;
      float d  = std::sqrt(dx * dx + dy * dy);
      // Background is EXACTLY black so it carries zero luminance weight: the
      // lumaCentroid() instrument below must measure the blob and nothing else.
      // A merely-dark background (say 8) spread over the whole frame outweighs
      // a small bright blob and drags the measured centroid to the frame centre.
      uint8_t v = (d <= fr) ? 240 : 0;
      size_t i = (y * W + x) * 4;
      px[i] = v; px[i + 1] = v; px[i + 2] = v; px[i + 3] = 255;
    }
  return px;
}

// Two discs, top-left and top-right — splits the mass across two cells so the
// per-cell redistribution term is non-trivial.
static std::vector<uint8_t> makeTwoBlobs(uint32_t W, uint32_t H) {
  auto a = makeBlob(W, H, 0.22f, 0.25f, 0.09f);
  auto b = makeBlob(W, H, 0.78f, 0.25f, 0.09f);
  for (size_t i = 0; i < a.size(); i += 4)
    for (int c = 0; c < 3; ++c) a[i + c] = (uint8_t)std::max(a[i + c], b[i + c]);
  return a;
}

static std::vector<uint8_t> makeRamp(uint32_t W, uint32_t H) {
  std::vector<uint8_t> px(W * H * 4, 0);
  for (uint32_t y = 0; y < H; ++y)
    for (uint32_t x = 0; x < W; ++x) {
      float t = (float)(x + y) / (float)(W + H - 2);
      uint8_t v = (uint8_t)(20.0f + t * 215.0f);
      size_t i = (y * W + x) * 4;
      px[i] = v; px[i + 1] = v; px[i + 2] = v; px[i + 3] = 255;
    }
  return px;
}

static double meanDiff(const std::vector<uint8_t>& a, const std::vector<uint8_t>& b) {
  double s = 0; size_t n = 0;
  for (size_t i = 0; i + 3 < a.size() && i + 3 < b.size(); i += 4)
    for (int c = 0; c < 3; ++c) { s += std::abs((int)a[i + c] - (int)b[i + c]); n++; }
  return n ? s / n : 0.0;
}

// Luminance-weighted centroid of a frame, in pixels. The measuring instrument
// for "did the correction move the image's centre of mass".
static void lumaCentroid(const std::vector<uint8_t>& px, uint32_t W, uint32_t H,
                         double& cx, double& cy) {
  double sw = 0, sx = 0, sy = 0;
  for (uint32_t y = 0; y < H; ++y)
    for (uint32_t x = 0; x < W; ++x) {
      size_t i = (y * W + x) * 4;
      double l = 0.299 * px[i] + 0.587 * px[i + 1] + 0.114 * px[i + 2];
      sw += l; sx += l * x; sy += l * y;
    }
  if (sw < 1e-6) { cx = W * 0.5; cy = H * 0.5; return; }
  cx = sx / sw; cy = sy / sw;
}

static double dist(double ax, double ay, double bx, double by) {
  double dx = ax - bx, dy = ay - by;
  return std::sqrt(dx * dx + dy * dy);
}

static EffectInstance* setup(WasmHost& host, EffectRuntime& rt,
                             sketch_executor::ModuleRegistry& registry,
                             gpu::GPUBackend* backend, int32_t& moduleId) {
  auto bytecode = load_file(kNanoWasm);
  REQUIRE(!bytecode.empty());
  REQUIRE(host.init());
  moduleId = host.load_module(bytecode.data(), bytecode.size());
  INFO("last_error: " << host.last_error());
  REQUIRE(moduleId >= 0);
  host.set_gpu_backend(moduleId, backend);
  REQUIRE(host.call_function(moduleId, "nano_module_main") == 0);

  const WasmEffectDesc* w = nullptr;
  for (const auto& e : host.registered_effects(moduleId))
    if (e.id == "warp.recompose") { w = &e; break; }
  REQUIRE(w != nullptr);
  REQUIRE(registry.registerWasmEffect("warp.recompose", "Recompose", &host, moduleId, *w));
  return rt.instanceFor("warp.recompose", "k0");
}

// Common baseline: snap the analysis (smooth 0), hold it (update_rate 0 — the
// forced first-frame update still runs), black fills so vacated regions
// contribute no luminance to the centroid measurement, and a full travel budget.
static void applyBase(EffectInstance* inst) {
  inst->setParamFloat("smooth", 0.0f);
  inst->setParamFloat("update_rate", 0.0f);
  inst->setParamFloat("spread", 0.0f);
  inst->setParamFloat("distance", 1.0f);
  inst->setParamFloat("overshoot", 1.0f);
  inst->setParamFloat("axis", 0.0f);
  // Detail-only saliency. The Contrast term (|luma − frame mean|) is area-
  // weighted by design, so on a synthetic "small bright blob on a flat field"
  // the large uniform background contributes as much total weight as the blob
  // and pulls the saliency centroid toward the frame centre — which would make
  // these geometry/sign assertions measure the fixture rather than the code.
  // Edge energy is unambiguously localized on the blob, so the effect's
  // saliency centroid and the test's luma centroid agree.
  inst->setParamFloat("w_grad", 1.0f);
  inst->setParamFloat("w_dev", 0.0f);
  inst->setParamFloat("w_sat", 0.0f);
  inst->setParamFloat("center_bias", 0.5f);
  inst->setParamFloat("rift_fill", 4.0f);    // black
  inst->setParamFloat("edge_fill", 4.0f);    // black
  inst->setParamFloat("overlap_mode", 0.0f); // heaviest on top
  inst->setParamFloat("debug_show", 0.0f);
}

TEST_CASE("recompose: correct = 0 is a passthrough", "[recompose]") {
  auto backend = gpu::createBackend();
  if (!backend) SKIP("No GPU device available");

  ParamCache cache;
  WasmHost host(cache);
  EffectRuntime rt(backend.get());
  sketch_executor::ModuleRegistry registry(&rt);
  int32_t moduleId = -1;
  EffectInstance* inst = setup(host, rt, registry, backend.get(), moduleId);
  REQUIRE(inst != nullptr);

  const uint32_t W = 96, H = 96;
  const int RGBA8 = 1;
  int inTex = backend->createTexture(W, H, RGBA8);
  int outTex = backend->createTexture(W, H, RGBA8);
  auto inPix = makeRamp(W, H);
  backend->writeTexture(inTex, W, H, inPix.data(), (uint32_t)inPix.size());
  inst->setTextureField("tex_in", inTex);
  inst->setTextureField("tex_out", outTex);
  inst->setFieldConnected("tex_in", true, false);

  applyBase(inst);
  inst->setParamFloat("correct", 0.0f);
  inst->doTick(0.05);
  inst->doRender(W, H);
  auto out = backend->readbackTexture(outTex, W, H);

  // Every D_k is zero, so every pixel resamples its own position: the only
  // deviation possible is one linear-sampler round trip. This is the
  // modulation-neutrality contract — a wire at rest must do nothing.
  double d = meanDiff(out, inPix);
  INFO("mean |out-in| at correct=0 = " << d);
  CHECK(d < 2.0);

  host.shutdown();
}

TEST_CASE("recompose: correction moves the centre of mass toward the power point",
          "[recompose]") {
  auto backend = gpu::createBackend();
  if (!backend) SKIP("No GPU device available");

  ParamCache cache;
  WasmHost host(cache);
  EffectRuntime rt(backend.get());
  sketch_executor::ModuleRegistry registry(&rt);
  int32_t moduleId = -1;
  EffectInstance* inst = setup(host, rt, registry, backend.get(), moduleId);
  REQUIRE(inst != nullptr);

  const uint32_t W = 96, H = 96;
  const int RGBA8 = 1;
  int inTex = backend->createTexture(W, H, RGBA8);
  int outTex = backend->createTexture(W, H, RGBA8);
  // Blob up-left of the top-left power point, but not so far that a full
  // negative correction would push it off-frame.
  auto inPix = makeBlob(W, H, 0.22f, 0.22f, 0.08f);
  backend->writeTexture(inTex, W, H, inPix.data(), (uint32_t)inPix.size());
  inst->setTextureField("tex_in", inTex);
  inst->setTextureField("tex_out", outTex);
  inst->setFieldConnected("tex_in", true, false);

  // On a square viewport the cover-square extent is (1,1), so the four power
  // points land exactly on the pixel thirds. The nearest one to this blob is
  // (W/3, H/3).
  const double px = W / 3.0, py = H / 3.0;

  auto runAt = [&](float correct) {
    applyBase(inst);
    inst->setParamFloat("correct", correct);
    inst->setParamFloat("trigger", 1.0f);   // force a fresh analysis
    inst->doTick(0.05);
    inst->doRender(W, H);
    inst->setParamFloat("trigger", 0.0f);
    auto out = backend->readbackTexture(outTex, W, H);
    double cx, cy; lumaCentroid(out, W, H, cx, cy);
    return std::make_pair(cx, cy);
  };

  auto zero = runAt(0.0f);
  auto pos  = runAt(1.0f);
  auto neg  = runAt(-1.0f);

  double dZero = dist(zero.first, zero.second, px, py);
  double dPos  = dist(pos.first,  pos.second,  px, py);
  double dNeg  = dist(neg.first,  neg.second,  px, py);
  INFO("centroid dist to power point: correct=0 " << dZero
       << ", +1 " << dPos << ", -1 " << dNeg);

  // Passing this exercises the whole chain at once and pins every sign
  // convention: saliency weighting, the centroid reduction, power-point
  // selection, G, the correction vectors and the nine-cell inverse map.
  CHECK(dPos < dZero);    // +1 balances
  CHECK(dNeg > dZero);    // -1 deliberately un-balances

  host.shutdown();
}

TEST_CASE("recompose: spread changes the image without disturbing the balance",
          "[recompose]") {
  auto backend = gpu::createBackend();
  if (!backend) SKIP("No GPU device available");

  ParamCache cache;
  WasmHost host(cache);
  EffectRuntime rt(backend.get());
  sketch_executor::ModuleRegistry registry(&rt);
  int32_t moduleId = -1;
  EffectInstance* inst = setup(host, rt, registry, backend.get(), moduleId);
  REQUIRE(inst != nullptr);

  const uint32_t W = 96, H = 96;
  const int RGBA8 = 1;
  int inTex = backend->createTexture(W, H, RGBA8);
  int outTex = backend->createTexture(W, H, RGBA8);
  auto inPix = makeTwoBlobs(W, H);   // mass split across two cells
  backend->writeTexture(inTex, W, H, inPix.data(), (uint32_t)inPix.size());
  inst->setTextureField("tex_in", inTex);
  inst->setTextureField("tex_out", outTex);
  inst->setFieldConnected("tex_in", true, false);

  auto runSpread = [&](float spread) {
    applyBase(inst);
    inst->setParamFloat("correct", 1.0f);
    inst->setParamFloat("spread", spread);
    inst->setParamFloat("trigger", 1.0f);
    inst->doTick(0.05);
    inst->doRender(W, H);
    inst->setParamFloat("trigger", 0.0f);
    return backend->readbackTexture(outTex, W, H);
  };

  auto a = runSpread(0.0f);
  auto b = runSpread(1.0f);

  double cxa, cya, cxb, cyb;
  lumaCentroid(a, W, H, cxa, cya);
  lumaCentroid(b, W, H, cxb, cyb);

  double imgDiff = meanDiff(a, b);
  double comDiff = dist(cxa, cya, cxb, cyb);
  INFO("spread 0 vs 1: image diff " << imgDiff << ", centre-of-mass shift " << comDiff);

  // The images must differ a lot — spread genuinely rearranges the cells...
  CHECK(imgDiff > 5.0);
  // ...but the centre of mass must barely move, because the per-cell
  // redistribution is mean-removed (Sum_k m_k R'_k = 0 by construction). If the
  // mean removal were dropped or mis-signed this is the check that fails.
  CHECK(comDiff < 0.15 * (double)W);

  host.shutdown();
}

TEST_CASE("recompose: axis restrict moves only the enabled axis", "[recompose]") {
  auto backend = gpu::createBackend();
  if (!backend) SKIP("No GPU device available");

  ParamCache cache;
  WasmHost host(cache);
  EffectRuntime rt(backend.get());
  sketch_executor::ModuleRegistry registry(&rt);
  int32_t moduleId = -1;
  EffectInstance* inst = setup(host, rt, registry, backend.get(), moduleId);
  REQUIRE(inst != nullptr);

  const uint32_t W = 96, H = 96;
  const int RGBA8 = 1;
  int inTex = backend->createTexture(W, H, RGBA8);
  int outTex = backend->createTexture(W, H, RGBA8);
  auto inPix = makeBlob(W, H, 0.22f, 0.22f, 0.08f);
  backend->writeTexture(inTex, W, H, inPix.data(), (uint32_t)inPix.size());
  inst->setTextureField("tex_in", inTex);
  inst->setTextureField("tex_out", outTex);
  inst->setFieldConnected("tex_in", true, false);

  auto runAxis = [&](float axis) {
    applyBase(inst);
    inst->setParamFloat("correct", 1.0f);
    inst->setParamFloat("axis", axis);
    inst->setParamFloat("trigger", 1.0f);
    inst->doTick(0.05);
    inst->doRender(W, H);
    inst->setParamFloat("trigger", 0.0f);
    auto out = backend->readbackTexture(outTex, W, H);
    double cx, cy; lumaCentroid(out, W, H, cx, cy);
    return std::make_pair(cx, cy);
  };

  auto base = runAxis(0.0f);   // both
  auto onlyX = runAxis(1.0f);
  auto onlyY = runAxis(2.0f);
  INFO("both (" << base.first << "," << base.second << ") "
       << "X (" << onlyX.first << "," << onlyX.second << ") "
       << "Y (" << onlyY.first << "," << onlyY.second << ")");

  // X Only must move horizontally like the unrestricted run, and leave the
  // vertical centre of mass essentially where correct=0 would have left it.
  CHECK(std::abs(onlyX.first - base.first) < 3.0);
  CHECK(onlyX.second < base.second - 3.0);   // y did NOT get corrected downward
  // ...and symmetrically for Y Only.
  CHECK(std::abs(onlyY.second - base.second) < 3.0);
  CHECK(onlyY.first < base.first - 3.0);

  host.shutdown();
}

TEST_CASE("recompose: publishes a real imbalance measurement", "[recompose]") {
  auto backend = gpu::createBackend();
  if (!backend) SKIP("No GPU device available");

  ParamCache cache;
  WasmHost host(cache);
  EffectRuntime rt(backend.get());
  sketch_executor::ModuleRegistry registry(&rt);
  int32_t moduleId = -1;
  EffectInstance* inst = setup(host, rt, registry, backend.get(), moduleId);
  REQUIRE(inst != nullptr);

  const uint32_t W = 96, H = 96;
  const int RGBA8 = 1;
  int inTex = backend->createTexture(W, H, RGBA8);
  int outTex = backend->createTexture(W, H, RGBA8);
  inst->setTextureField("tex_in", inTex);
  inst->setTextureField("tex_out", outTex);
  inst->setFieldConnected("tex_in", true, false);

  // Analyse a fixture and drain the readback (request in render, poll in the
  // following tick), then read the published scalars.
  auto measure = [&](const std::vector<uint8_t>& pix,
                     double& bx, double& by, double& err) {
    backend->writeTexture(inTex, W, H, pix.data(), (uint32_t)pix.size());
    applyBase(inst);
    inst->setParamFloat("correct", 0.0f);   // measure only; don't move anything
    inst->setParamFloat("trigger", 1.0f);
    for (int f = 0; f < 4; ++f) { inst->doTick(0.05); inst->doRender(W, H); }
    inst->setParamFloat("trigger", 0.0f);
    inst->doTick(0.05);
    REQUIRE(inst->publishedScalar("balance_x", 9, &bx));
    REQUIRE(inst->publishedScalar("balance_y", 9, &by));
    REQUIRE(inst->publishedScalar("cell_error", 10, &err));
  };

  // A blob up-left of the top-left power point needs a correction pointing
  // down-right, so both balance channels should read positive.
  double bx = 0, by = 0, err = 0;
  measure(makeBlob(W, H, 0.15f, 0.15f, 0.08f), bx, by, err);
  INFO("up-left blob: balance = (" << bx << ", " << by << "), cell_error = " << err);
  CHECK(bx > 0.02);
  CHECK(by > 0.02);
  CHECK(err > 0.0);

  // Mirrored blob (down-right of the bottom-right power point) must flip both
  // signs — that is what proves the sign convention rather than a constant.
  double bx2 = 0, by2 = 0, err2 = 0;
  measure(makeBlob(W, H, 0.85f, 0.85f, 0.08f), bx2, by2, err2);
  INFO("down-right blob: balance = (" << bx2 << ", " << by2 << ")");
  CHECK(bx2 < -0.02);
  CHECK(by2 < -0.02);

  // A blob sitting ON a power point is better balanced than one in the corner.
  double bx3 = 0, by3 = 0, err3 = 0;
  measure(makeBlob(W, H, 1.0f / 3.0f, 1.0f / 3.0f, 0.08f), bx3, by3, err3);
  INFO("on-power-point blob: balance = (" << bx3 << ", " << by3 << ")");
  CHECK(std::abs(bx3) < std::abs(bx));
  CHECK(std::abs(by3) < std::abs(by));

  host.shutdown();
}

TEST_CASE("recompose: holds its analysis between updates", "[recompose]") {
  auto backend = gpu::createBackend();
  if (!backend) SKIP("No GPU device available");

  ParamCache cache;
  WasmHost host(cache);
  EffectRuntime rt(backend.get());
  sketch_executor::ModuleRegistry registry(&rt);
  int32_t moduleId = -1;
  EffectInstance* inst = setup(host, rt, registry, backend.get(), moduleId);
  REQUIRE(inst != nullptr);

  const uint32_t W = 96, H = 96;
  const int RGBA8 = 1;
  int inTex = backend->createTexture(W, H, RGBA8);
  int outTex = backend->createTexture(W, H, RGBA8);
  auto inPix = makeBlob(W, H, 0.22f, 0.22f, 0.08f);
  backend->writeTexture(inTex, W, H, inPix.data(), (uint32_t)inPix.size());
  inst->setTextureField("tex_in", inTex);
  inst->setTextureField("tex_out", outTex);
  inst->setFieldConnected("tex_in", true, false);

  applyBase(inst);                            // update_rate 0, smooth 0
  inst->setParamFloat("correct", 1.0f);
  inst->doTick(0.05); inst->doRender(W, H);
  auto a = backend->readbackTexture(outTex, W, H);
  inst->doTick(0.05); inst->doRender(W, H);
  auto b = backend->readbackTexture(outTex, W, H);

  // Frozen rate + snap smoothing: the correction must not drift frame to frame.
  CHECK(a == b);

  host.shutdown();
}
