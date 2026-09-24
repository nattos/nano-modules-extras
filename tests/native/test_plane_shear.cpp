// test_plane_shear.cpp — end-to-end GPU render of warp.plane_shear from
// nano.wasm on a real Metal backend. Exercises the full three-pass pipeline
// (accumulate → solve → render): the effect analyzes a synthetic vertical-edge
// input, latches a dividing plane, and shears the two halves. Verifies the
// shear actually moves pixels and that a transparent rift opens a real gap.

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

// Left half dark, right half bright — a strong vertical edge at x = W/2, so the
// dominant-edge / PCA analysis picks a vertical dividing line through center.
static std::vector<uint8_t> makeEdge(uint32_t W, uint32_t H) {
  std::vector<uint8_t> px(W * H * 4, 0);
  for (uint32_t y = 0; y < H; ++y)
    for (uint32_t x = 0; x < W; ++x) {
      uint8_t v = (x < W / 2) ? 30 : 220;
      size_t i = (y * W + x) * 4;
      px[i] = v; px[i + 1] = v; px[i + 2] = v; px[i + 3] = 255;
    }
  return px;
}

// A vertical edge at an arbitrary column — lets us place the analyzed plane
// OFF-center to exercise center_weight.
static std::vector<uint8_t> makeEdgeAt(uint32_t W, uint32_t H, uint32_t ex) {
  std::vector<uint8_t> px(W * H * 4, 0);
  for (uint32_t y = 0; y < H; ++y)
    for (uint32_t x = 0; x < W; ++x) {
      uint8_t v = (x < ex) ? 30 : 220;
      size_t i = (y * W + x) * 4;
      px[i] = v; px[i + 1] = v; px[i + 2] = v; px[i + 3] = 255;
    }
  return px;
}

// A smooth diagonal ramp — has a clear global gradient (so every algorithm
// finds a plane) AND is NOT translation-invariant, so any real shear changes
// pixels even with an opaque (original / edge-stretch) fill.
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
    if (e.id == "warp.plane_shear") { w = &e; break; }
  REQUIRE(w != nullptr);
  REQUIRE(registry.registerWasmEffect("warp.plane_shear", "Plane Shear", &host, moduleId, *w));
  return rt.instanceFor("warp.plane_shear", "k0");
}

TEST_CASE("plane_shear shears the image and opens a transparent rift", "[plane_shear]") {
  auto backend = gpu::createBackend();
  if (!backend) SKIP("No GPU device available");

  ParamCache cache;
  WasmHost host(cache);
  EffectRuntime rt(backend.get());
  sketch_executor::ModuleRegistry registry(&rt);
  int32_t moduleId = -1;
  EffectInstance* inst = setup(host, rt, registry, backend.get(), moduleId);
  REQUIRE(inst != nullptr);

  const uint32_t W = 64, H = 64;
  const int RGBA8 = 1;
  int inTex = backend->createTexture(W, H, RGBA8);
  int outTex = backend->createTexture(W, H, RGBA8);
  REQUIRE(inTex >= 0); REQUIRE(outTex >= 0);
  auto inPix = makeEdge(W, H);
  backend->writeTexture(inTex, W, H, inPix.data(), (uint32_t)inPix.size());

  inst->setTextureField("tex_in", inTex);
  inst->setTextureField("tex_out", outTex);
  inst->setFieldConnected("tex_in", /*input*/true, /*output*/false);

  // Instant, full-amplitude rift on the analyzed vertical plane. duration 0 →
  // shear amount snaps to 1 on the first tick (no ramp needed).
  inst->setParamFloat("algorithm", 0.0f);      // Dominant Edge
  inst->setParamFloat("direction", -1.0f);     // rift (halves apart)
  inst->setParamFloat("duration", 0.0f);       // instant
  inst->setParamFloat("distance", 0.3f);
  inst->setParamFloat("mult_a", 1.0f);
  inst->setParamFloat("mult_b", 1.0f);
  inst->setParamFloat("rift_fill", 0.0f);      // transparent
  inst->setParamFloat("overlap_mode", 0.0f);

  inst->doTick(0.05);   // compute shear amount (=1) + keep the forced first-frame update
  inst->doRender(W, H);
  auto out = backend->readbackTexture(outTex, W, H);
  REQUIRE(out.size() == W * H * 4);

  // 1) The shear moved pixels: output differs substantially from the input.
  double diff = meanDiff(out, inPix);
  INFO("mean |out-in| = " << diff);
  CHECK(diff > 10.0);

  // 2) A transparent rift opened: some pixels in the central band went to
  //    alpha 0, whereas the input was fully opaque everywhere.
  int transparentCentral = 0;
  for (uint32_t y = 0; y < H; ++y)
    for (uint32_t x = W / 2 - 6; x <= W / 2 + 6; ++x) {
      size_t i = (y * W + x) * 4;
      if (out[i + 3] < 8) transparentCentral++;
    }
  INFO("transparent central pixels = " << transparentCentral);
  CHECK(transparentCentral > 0);

  host.shutdown();
}

TEST_CASE("plane_shear runs every algorithm and stays stiff between updates", "[plane_shear]") {
  auto backend = gpu::createBackend();
  if (!backend) SKIP("No GPU device available");

  ParamCache cache;
  WasmHost host(cache);
  EffectRuntime rt(backend.get());
  sketch_executor::ModuleRegistry registry(&rt);
  int32_t moduleId = -1;
  EffectInstance* inst = setup(host, rt, registry, backend.get(), moduleId);
  REQUIRE(inst != nullptr);

  const uint32_t W = 64, H = 64;
  int inTex = backend->createTexture(W, H, 1);
  int outTex = backend->createTexture(W, H, 1);
  auto inPix = makeRamp(W, H);   // translation-visible (not piecewise-constant)
  backend->writeTexture(inTex, W, H, inPix.data(), (uint32_t)inPix.size());
  inst->setTextureField("tex_in", inTex);
  inst->setTextureField("tex_out", outTex);
  inst->setFieldConnected("tex_in", true, false);

  inst->setParamFloat("direction", -1.0f);
  inst->setParamFloat("duration", 0.0f);
  inst->setParamFloat("update_rate", 0.0f);   // slow (≈8 s) → no retarget within the test
  inst->setParamFloat("rift_fill", 1.0f);     // original (always opaque, easy to compare)

  // Every algorithm should render without error and move pixels.
  for (float alg = 0.0f; alg <= 3.0f; alg += 1.0f) {
    inst->setParamFloat("algorithm", alg);    // forces a fresh analysis
    inst->doTick(0.05);
    inst->doRender(W, H);
    auto out = backend->readbackTexture(outTex, W, H);
    REQUIRE(out.size() == W * H * 4);
    INFO("algorithm " << alg << " mean |out-in| = " << meanDiff(out, inPix));
    CHECK(meanDiff(out, inPix) > 5.0);
  }

  // Stiffness: with the plane held (no retarget) and shear fixed, two
  // consecutive frames must be byte-identical — no rubbery drift.
  inst->setParamFloat("algorithm", 3.0f);
  inst->doTick(0.05);
  inst->doRender(W, H);
  auto a = backend->readbackTexture(outTex, W, H);
  inst->doTick(0.05);   // time advances, but plane is held and shear is at max
  inst->doRender(W, H);
  auto b = backend->readbackTexture(outTex, W, H);
  REQUIRE(a.size() == b.size());
  CHECK(a == b);

  host.shutdown();
}

TEST_CASE("plane_shear center_weight pulls an off-center plane toward the center", "[plane_shear]") {
  auto backend = gpu::createBackend();
  if (!backend) SKIP("No GPU device available");

  ParamCache cache;
  WasmHost host(cache);
  EffectRuntime rt(backend.get());
  sketch_executor::ModuleRegistry registry(&rt);
  int32_t moduleId = -1;
  EffectInstance* inst = setup(host, rt, registry, backend.get(), moduleId);
  REQUIRE(inst != nullptr);

  const uint32_t W = 64, H = 64;
  int inTex = backend->createTexture(W, H, 1);
  int outTex = backend->createTexture(W, H, 1);
  // Edge at 3/4 width → the analyzed vertical plane sits right of center.
  auto inPix = makeEdgeAt(W, H, (3 * W) / 4);
  backend->writeTexture(inTex, W, H, inPix.data(), (uint32_t)inPix.size());
  inst->setTextureField("tex_in", inTex);
  inst->setTextureField("tex_out", outTex);
  inst->setFieldConnected("tex_in", true, false);

  inst->setParamFloat("algorithm", 0.0f);   // Dominant Edge → vertical plane
  inst->setParamFloat("direction", -1.0f);  // rift
  inst->setParamFloat("duration", 0.0f);    // instant
  inst->setParamFloat("update_rate", 0.0f);
  inst->setParamFloat("distance", 0.35f);
  inst->setParamFloat("rift_fill", 0.0f);   // transparent → the rift band is measurable

  // Mean X of the transparent rift band = where the plane sits horizontally.
  auto riftMeanX = [&]() -> double {
    auto out = backend->readbackTexture(outTex, W, H);
    double sx = 0; int n = 0;
    for (uint32_t y = 0; y < H; ++y)
      for (uint32_t x = 0; x < W; ++x) {
        size_t i = (y * W + x) * 4;
        if (out[i + 3] < 8) { sx += x; n++; }
      }
    return n ? sx / n : -1.0;
  };

  inst->setParamFloat("center_weight", 0.0f);
  inst->doTick(0.05);
  inst->doRender(W, H);
  double x0 = riftMeanX();

  // The offset pull only engages above 0.5, and Dominant Edge has no search-time
  // selection bias — so at 0.5 the plane must be essentially where it was at 0.
  inst->setParamFloat("center_weight", 0.5f);
  inst->doTick(0.05);
  inst->doRender(W, H);
  double xHalf = riftMeanX();

  inst->setParamFloat("center_weight", 1.0f);
  inst->doTick(0.05);
  inst->doRender(W, H);
  double x1 = riftMeanX();

  const double centerX = W / 2.0;
  INFO("rift mean X: w0=" << x0 << " w0.5=" << xHalf << " w1=" << x1 << " (center=" << centerX << ")");
  REQUIRE(x0 > 0.0);
  REQUIRE(xHalf > 0.0);
  REQUIRE(x1 > 0.0);
  // weight 0: plane near the edge (right of center).
  CHECK(x0 > centerX + 4.0);
  // weight 0.5: pull not yet engaged → unchanged from weight 0.
  CHECK(std::abs(xHalf - x0) < 3.0);
  // weight 1: pulled to center.
  CHECK(std::abs(x1 - centerX) < std::abs(x0 - centerX));
  CHECK(x1 < x0);

  host.shutdown();
}

TEST_CASE("plane_shear per-side colour tint changes the output", "[plane_shear]") {
  auto backend = gpu::createBackend();
  if (!backend) SKIP("No GPU device available");

  ParamCache cache;
  WasmHost host(cache);
  EffectRuntime rt(backend.get());
  sketch_executor::ModuleRegistry registry(&rt);
  int32_t moduleId = -1;
  EffectInstance* inst = setup(host, rt, registry, backend.get(), moduleId);
  REQUIRE(inst != nullptr);

  const uint32_t W = 64, H = 64;
  int inTex = backend->createTexture(W, H, 1);
  int outTex = backend->createTexture(W, H, 1);
  auto inPix = makeRamp(W, H);
  backend->writeTexture(inTex, W, H, inPix.data(), (uint32_t)inPix.size());
  inst->setTextureField("tex_in", inTex);
  inst->setTextureField("tex_out", outTex);
  inst->setFieldConnected("tex_in", true, false);

  inst->setParamFloat("direction", -1.0f);
  inst->setParamFloat("duration", 0.0f);
  inst->setParamFloat("update_rate", 0.0f);
  inst->setParamFloat("distance", 0.4f);
  inst->setParamFloat("rift_fill", 1.0f);   // original (content present to tint)

  inst->setParamFloat("tint", 0.0f);
  inst->doTick(0.05); inst->doRender(W, H);
  auto untinted = backend->readbackTexture(outTex, W, H);

  inst->setParamFloat("tint", 1.0f);   // default warm/cool per-side colours
  inst->setParamFloat("tint_mode", 0.0f);   // multiply
  inst->doTick(0.05); inst->doRender(W, H);
  auto mul = backend->readbackTexture(outTex, W, H);

  inst->setParamFloat("tint_mode", 1.0f);   // add
  inst->doTick(0.05); inst->doRender(W, H);
  auto add = backend->readbackTexture(outTex, W, H);

  INFO("mul vs untinted = " << meanDiff(mul, untinted)
       << "  add vs mul = " << meanDiff(add, mul));
  CHECK(meanDiff(mul, untinted) > 8.0);   // tint changes the output
  CHECK(meanDiff(add, mul) > 8.0);        // add mode differs from multiply

  host.shutdown();
}

TEST_CASE("plane_shear defaults to opaque-black rift fill", "[plane_shear]") {
  auto backend = gpu::createBackend();
  if (!backend) SKIP("No GPU device available");

  ParamCache cache;
  WasmHost host(cache);
  EffectRuntime rt(backend.get());
  sketch_executor::ModuleRegistry registry(&rt);
  int32_t moduleId = -1;
  EffectInstance* inst = setup(host, rt, registry, backend.get(), moduleId);
  REQUIRE(inst != nullptr);

  const uint32_t W = 64, H = 64;
  int inTex = backend->createTexture(W, H, 1);
  int outTex = backend->createTexture(W, H, 1);
  auto inPix = makeEdge(W, H);
  backend->writeTexture(inTex, W, H, inPix.data(), (uint32_t)inPix.size());
  inst->setTextureField("tex_in", inTex);
  inst->setTextureField("tex_out", outTex);
  inst->setFieldConnected("tex_in", true, false);

  // Do NOT set rift_fill — rely on the Black default. Open a rift.
  inst->setParamFloat("algorithm", 0.0f);
  inst->setParamFloat("direction", -1.0f);
  inst->setParamFloat("duration", 0.0f);
  inst->setParamFloat("update_rate", 0.0f);
  inst->setParamFloat("distance", 0.4f);

  inst->doTick(0.05);
  inst->doRender(W, H);
  auto out = backend->readbackTexture(outTex, W, H);

  // The rift band is OPAQUE black (rgb≈0, alpha≈255) — not transparent.
  int opaqueBlack = 0, transparent = 0;
  for (uint32_t y = 0; y < H; ++y)
    for (uint32_t x = W / 2 - 6; x <= W / 2 + 6; ++x) {
      size_t i = (y * W + x) * 4;
      if (out[i] < 8 && out[i + 1] < 8 && out[i + 2] < 8 && out[i + 3] > 200) opaqueBlack++;
      if (out[i + 3] < 8) transparent++;
    }
  INFO("central opaque-black=" << opaqueBlack << " transparent=" << transparent);
  CHECK(opaqueBlack > 0);
  CHECK(transparent == 0);

  host.shutdown();
}

TEST_CASE("plane_shear edge_fill controls the viewport-border reveal", "[plane_shear]") {
  auto backend = gpu::createBackend();
  if (!backend) SKIP("No GPU device available");

  ParamCache cache;
  WasmHost host(cache);
  EffectRuntime rt(backend.get());
  sketch_executor::ModuleRegistry registry(&rt);
  int32_t moduleId = -1;
  EffectInstance* inst = setup(host, rt, registry, backend.get(), moduleId);
  REQUIRE(inst != nullptr);

  const uint32_t W = 64, H = 64;
  int inTex = backend->createTexture(W, H, 1);
  int outTex = backend->createTexture(W, H, 1);
  auto inPix = makeEdge(W, H);   // vertical plane through center
  backend->writeTexture(inTex, W, H, inPix.data(), (uint32_t)inPix.size());
  inst->setTextureField("tex_in", inTex);
  inst->setTextureField("tex_out", outTex);
  inst->setFieldConnected("tex_in", true, false);

  // Slip (direction 0) slides the halves ALONG the plane, so both halves pull a
  // viewport border into view — the region edge_fill governs. No rift here, so
  // rift_fill is irrelevant; the only exposed pixels are border reveals.
  inst->setParamFloat("algorithm", 0.0f);
  inst->setParamFloat("direction", 0.0f);
  inst->setParamFloat("duration", 0.0f);
  inst->setParamFloat("update_rate", 0.0f);
  inst->setParamFloat("distance", 0.6f);

  auto transparentCount = [&]() {
    auto out = backend->readbackTexture(outTex, W, H);
    int n = 0;
    for (size_t i = 3; i < out.size(); i += 4) if (out[i] < 8) n++;
    return n;
  };

  // Edge stretch (default) clamps the border — nothing transparent.
  inst->setParamFloat("edge_fill", 2.0f);
  inst->doTick(0.05);
  inst->doRender(W, H);
  int stretched = transparentCount();

  // Transparent edge fill leaves the border reveal empty.
  inst->setParamFloat("edge_fill", 0.0f);
  inst->doTick(0.05);
  inst->doRender(W, H);
  int transparent = transparentCount();

  INFO("border transparent px: stretch=" << stretched << " transparent=" << transparent);
  CHECK(stretched == 0);
  CHECK(transparent > 100);

  host.shutdown();
}

TEST_CASE("plane_shear trigger button restarts the shear animation", "[plane_shear]") {
  auto backend = gpu::createBackend();
  if (!backend) SKIP("No GPU device available");

  ParamCache cache;
  WasmHost host(cache);
  EffectRuntime rt(backend.get());
  sketch_executor::ModuleRegistry registry(&rt);
  int32_t moduleId = -1;
  EffectInstance* inst = setup(host, rt, registry, backend.get(), moduleId);
  REQUIRE(inst != nullptr);

  const uint32_t W = 64, H = 64;
  int inTex = backend->createTexture(W, H, 1);
  int outTex = backend->createTexture(W, H, 1);
  auto inPix = makeEdge(W, H);
  backend->writeTexture(inTex, W, H, inPix.data(), (uint32_t)inPix.size());
  inst->setTextureField("tex_in", inTex);
  inst->setTextureField("tex_out", outTex);
  inst->setFieldConnected("tex_in", true, false);

  // One-shot ramp over ~1 s, opening a transparent rift. The transparent-pixel
  // count is a proxy for the current shear amount (bigger shear → wider rift).
  inst->setParamFloat("algorithm", 0.0f);
  inst->setParamFloat("anim_mode", 0.0f);      // one-shot hold
  inst->setParamFloat("direction", -1.0f);     // rift
  inst->setParamFloat("duration", 0.5f);       // ≈1 s ramp (slider² · 4)
  inst->setParamFloat("update_rate", 0.0f);    // plane held
  inst->setParamFloat("distance", 0.5f);
  inst->setParamFloat("rift_fill", 0.0f);      // transparent
  inst->setParamFloat("retrigger", 0.0f);      // isolate the manual trigger

  auto transparentCount = [&]() {
    auto out = backend->readbackTexture(outTex, W, H);
    int n = 0;
    for (size_t i = 3; i < out.size(); i += 4) if (out[i] < 8) n++;
    return n;
  };

  // Ramp to full shear (phase ≫ duration → amount saturates at 1).
  inst->doTick(2.0);
  inst->doRender(W, H);
  int full = transparentCount();
  CHECK(full > 0);

  // Press "trigger now" (rising edge) → the animation restarts from 0, so after
  // a tiny time step the shear (and the rift) is far smaller than at full.
  inst->setParamFloat("trigger", 1.0f);
  inst->doTick(0.02);
  inst->doRender(W, H);
  int afterTrigger = transparentCount();
  INFO("full rift = " << full << "  after trigger = " << afterTrigger);
  CHECK(afterTrigger < full / 2);

  host.shutdown();
}
