// test_envelope_warp.cpp — end-to-end GPU render of warp.envelope from nano.wasm
// on a real Metal backend. Exercises the whole pipeline: per-segment instanced
// quads rasterizing the 1D coordinate maps (rgba32float, Replace blend) and the
// compute resolve that composes them — Metal parity for what the web e2e
// (web/test/envelope-warp.test.ts) pins on WebGPU.

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

// Horizontal ramp: value depends on x only, so the H-mode warp math is exactly
// checkable per pixel (and a V-mode warp leaves it untouched).
static std::vector<uint8_t> makeRampX(uint32_t W, uint32_t H) {
  std::vector<uint8_t> px(W * H * 4, 0);
  for (uint32_t y = 0; y < H; ++y)
    for (uint32_t x = 0; x < W; ++x) {
      uint8_t v = (uint8_t)std::lround(20.0 + 215.0 * (double)x / (double)(W - 1));
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

static int red(const std::vector<uint8_t>& px, uint32_t W, uint32_t x, uint32_t y) {
  return px[(y * W + x) * 4];
}
static int alpha(const std::vector<uint8_t>& px, uint32_t W, uint32_t x, uint32_t y) {
  return px[(y * W + x) * 4 + 3];
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
    if (e.id == "warp.envelope") { w = &e; break; }
  REQUIRE(w != nullptr);
  REQUIRE(registry.registerWasmEffect("warp.envelope", "Envelope Warp", &host, moduleId, *w));
  return rt.instanceFor("warp.envelope", "k0");
}

TEST_CASE("envelope_warp: identity, squeeze, edge fills and fold-over on Metal", "[envelope_warp]") {
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
  int inTex = backend->createTexture(W, H, 1);
  int outTex = backend->createTexture(W, H, 1);
  auto inPix = makeRampX(W, H);
  backend->writeTexture(inTex, W, H, inPix.data(), (uint32_t)inPix.size());
  inst->setTextureField("tex_in", inTex);
  inst->setTextureField("tex_out", outTex);
  inst->setFieldConnected("tex_in", true, false);

  auto renderNow = [&]() {
    inst->doTick(0.05);
    inst->doRender(W, H);
    return backend->readbackTexture(outTex, W, H);
  };

  // Identity curve: the full raster+resolve pipeline reproduces the input
  // (samples land on exact texel centers — bilinear is a no-op).
  inst->setParamFloat("mode", 0.0f);      // Horizontal
  inst->setParamFloat("amount", 1.0f);
  inst->setParamJson("curve", "\"[0,0,0,1,1,0]\"");
  auto ident = renderNow();
  INFO("identity meanDiff = " << meanDiff(ident, inPix));
  CHECK(meanDiff(ident, inPix) < 2.0);

  // Horizontal squeeze y = 0.5x (mirrored): dest u = 0.625 shows source
  // u = 0.75; the center column is fixed; the stretch bands smear the edges.
  inst->setParamJson("curve", "\"[0,0,0,1,0.5,0]\"");
  auto sq = renderNow();
  CHECK(std::abs(red(sq, W, 48, 48) - red(inPix, W, 48, 48)) <= 6);   // center fixed
  CHECK(std::abs(red(sq, W, 60, 48) - red(inPix, W, 72, 48)) <= 6);   // 0.625 <- 0.75
  CHECK(std::abs(red(sq, W, 36, 48) - red(inPix, W, 24, 48)) <= 6);   // mirrored side
  CHECK(std::abs(red(sq, W, 5, 48)  - red(inPix, W, 0, 48))  <= 6);   // left band = left edge
  CHECK(std::abs(red(sq, W, 90, 48) - red(inPix, W, W - 1, 48)) <= 6);
  CHECK(alpha(sq, W, 5, 48) == 255);                                   // stretch is opaque

  // Transparent edges: the uncovered bands really are alpha 0 (native readback
  // has honest alpha — no checkerboard compositing here).
  inst->setParamFloat("edges", 1.0f);
  auto tr = renderNow();
  CHECK(alpha(tr, W, 5, 48) == 0);
  CHECK(alpha(tr, W, 90, 48) == 0);
  CHECK(std::abs(red(tr, W, 60, 48) - red(inPix, W, 72, 48)) <= 6);   // interior unchanged
  inst->setParamFloat("edges", 0.0f);

  // Eased segment (+1): forward d = t^(1/8), so dest h' = 0.5 (u = 0.75)
  // pulls from source h = 0.5^8 ≈ 0.004 — the image center, not h = 0.5.
  inst->setParamJson("curve", "\"[0,0,1,1,1,0]\"");
  auto eased = renderNow();
  CHECK(std::abs(red(eased, W, 72, 48) - red(inPix, W, 48, 48)) <= 6);

  // Fold-over: rise to 1 at x=0.5 then fall to 0.5 — the LATER (falling)
  // segment wins dest h' = 0.9: source h = 0.6 (u = 0.80), not h = 0.45.
  inst->setParamJson("curve", "\"[0,0,0,0.5,1,0,1,0.5,0]\"");
  auto fold = renderNow();
  CHECK(std::abs(red(fold, W, 91, 48) - red(inPix, W, 76, 48)) <= 6);

  // Amount 0 with a non-identity curve: the pipeline itself renders identity
  // (the executor would normally skip via is_identity; here we render live).
  inst->setParamFloat("amount", 0.0f);
  auto flat = renderNow();
  CHECK(meanDiff(flat, inPix) < 2.0);

  host.shutdown();
}

TEST_CASE("envelope_warp: radial mode warps by center distance on Metal", "[envelope_warp]") {
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
  int inTex = backend->createTexture(W, H, 1);
  int outTex = backend->createTexture(W, H, 1);
  auto inPix = makeRampX(W, H);
  backend->writeTexture(inTex, W, H, inPix.data(), (uint32_t)inPix.size());
  inst->setTextureField("tex_in", inTex);
  inst->setTextureField("tex_out", outTex);
  inst->setFieldConnected("tex_in", true, false);

  inst->setParamFloat("mode", 5.0f);      // Radial
  inst->setParamFloat("amount", 1.0f);

  // Identity curve: the rigid slope-1 continuation beyond r=1 keeps the
  // CORNERS intact too, so the whole frame round-trips.
  inst->setParamJson("curve", "\"[0,0,0,1,1,0]\"");
  inst->doTick(0.05);
  inst->doRender(W, H);
  auto ident = backend->readbackTexture(outTex, W, H);
  INFO("radial identity meanDiff = " << meanDiff(ident, inPix));
  CHECK(meanDiff(ident, inPix) < 2.0);

  // Radial squeeze r' = 0.5r: pixel (60, 48) sits at r ≈ 0.26 and pulls from
  // r ≈ 0.52 — source x ≈ 73 on the same row. The exact center is fixed.
  inst->setParamJson("curve", "\"[0,0,0,1,0.5,0]\"");
  inst->doTick(0.05);
  inst->doRender(W, H);
  auto sq = backend->readbackTexture(outTex, W, H);
  CHECK(std::abs(red(sq, W, 48, 48) - red(inPix, W, 48, 48)) <= 8);
  CHECK(std::abs(red(sq, W, 60, 48) - red(inPix, W, 73, 48)) <= 8);
  CHECK(meanDiff(sq, inPix) > 8.0);

  host.shutdown();
}
