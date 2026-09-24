// test_tri_shear.cpp — end-to-end GPU render of warp.tri_shear from nano.wasm on a
// real Metal backend. Exercises the full pipeline (accumulate → 3-line solve → 3
// chained render passes): discovers a triangle and shears the image three times.

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

// A diagonal ramp — clear global gradient (so the discovery finds lines) AND not
// translation-invariant (so a real shear changes pixels under any opaque fill).
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
    if (e.id == "warp.tri_shear") { w = &e; break; }
  REQUIRE(w != nullptr);
  REQUIRE(registry.registerWasmEffect("warp.tri_shear", "Triangle Shear", &host, moduleId, *w));
  return rt.instanceFor("warp.tri_shear", "k0");
}

TEST_CASE("tri_shear shears the ramp and holds stiff", "[tri_shear]") {
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

  inst->setParamFloat("algorithm", 1.0f);     // strongest edges
  inst->setParamFloat("direction", -1.0f);    // rift
  inst->setParamFloat("duration", 0.0f);      // instant
  inst->setParamFloat("update_rate", 0.0f);   // held
  inst->setParamFloat("size", 0.6f);
  inst->setParamFloat("rift_fill", 1.0f);     // original (opaque) so shear is visible on the ramp

  // distance 0 = mA=mB=0 → every pass is identity → output == input.
  inst->setParamFloat("distance", 0.0f);
  inst->doTick(0.05);
  inst->doRender(W, H);
  auto ident = backend->readbackTexture(outTex, W, H);
  INFO("distance0 meanDiff vs input = " << meanDiff(ident, inPix));
  CHECK(meanDiff(ident, inPix) < 2.0);

  // distance 0.4 = three chained shears → the ramp is visibly displaced.
  inst->setParamFloat("distance", 0.4f);
  inst->doTick(0.05);
  inst->doRender(W, H);
  auto shear = backend->readbackTexture(outTex, W, H);
  INFO("distance0.4 meanDiff vs input = " << meanDiff(shear, inPix));
  CHECK(meanDiff(shear, inPix) > 8.0);

  // Stiffness: plane held + shear at max → two consecutive frames byte-identical.
  inst->doTick(0.05);
  inst->doRender(W, H);
  auto a = backend->readbackTexture(outTex, W, H);
  inst->doTick(0.05);
  inst->doRender(W, H);
  auto b = backend->readbackTexture(outTex, W, H);
  CHECK(a == b);

  host.shutdown();
}

TEST_CASE("tri_shear size and algorithm change the discovered triangle", "[tri_shear]") {
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
  inst->setParamFloat("rift_fill", 1.0f);

  auto renderWith = [&](float alg, float size) {
    inst->setParamFloat("algorithm", alg);
    inst->setParamFloat("size", size);      // both force a fresh analysis
    inst->doTick(0.05);
    inst->doRender(W, H);
    return backend->readbackTexture(outTex, W, H);
  };

  auto small = renderWith(1.0f, 0.0f);   // strongest, tiny triangle
  auto large = renderWith(1.0f, 1.0f);   // strongest, large triangle
  auto seam  = renderWith(2.0f, 1.0f);   // low-energy seams, large

  INFO("size diff = " << meanDiff(large, small) << "  alg diff = " << meanDiff(seam, large));
  CHECK(meanDiff(large, small) > 3.0);   // the size param reshapes the triangle
  CHECK(meanDiff(seam, large) > 3.0);    // the two discovery modes differ

  // Obliqueness: equilateral (0) vs freely-oblique (1) yield different triangles.
  inst->setParamFloat("algorithm", 1.0f);
  inst->setParamFloat("size", 0.6f);
  inst->setParamFloat("obliqueness", 0.0f);
  inst->doTick(0.05); inst->doRender(W, H);
  auto equil = backend->readbackTexture(outTex, W, H);
  inst->setParamFloat("obliqueness", 1.0f);
  inst->doTick(0.05); inst->doRender(W, H);
  auto obliq = backend->readbackTexture(outTex, W, H);
  INFO("obliqueness diff = " << meanDiff(obliq, equil));
  CHECK(meanDiff(obliq, equil) > 2.0);

  host.shutdown();
}

TEST_CASE("tri_shear obliqueness_jitter re-rolls per update", "[tri_shear]") {
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

  inst->setParamFloat("algorithm", 1.0f);
  inst->setParamFloat("direction", -1.0f);
  inst->setParamFloat("duration", 0.0f);
  inst->setParamFloat("size", 0.6f);
  inst->setParamFloat("distance", 0.4f);
  inst->setParamFloat("rift_fill", 1.0f);
  inst->setParamFloat("obliqueness", 0.5f);
  inst->setParamFloat("update_rate", 1.0f);   // ≈0.05 s interval → frequent updates

  // Jitter 0: an update reproduces the exact same triangle (deterministic).
  inst->setParamFloat("obliqueness_jitter", 0.0f);
  inst->doTick(0.05); inst->doRender(W, H);
  auto a = backend->readbackTexture(outTex, W, H);
  inst->doTick(0.10); inst->doRender(W, H);   // metronome fires an update
  auto b = backend->readbackTexture(outTex, W, H);
  CHECK(a == b);

  // Jitter 1: successive updates roll different obliqueness → the triangle varies.
  inst->setParamFloat("obliqueness_jitter", 1.0f);
  inst->doTick(0.05); inst->doRender(W, H);
  auto base = backend->readbackTexture(outTex, W, H);
  bool varied = false;
  for (int i = 0; i < 6 && !varied; ++i) {
    inst->doTick(0.10); inst->doRender(W, H);   // each fires a fresh update + re-roll
    auto cur = backend->readbackTexture(outTex, W, H);
    if (meanDiff(cur, base) > 2.0) varied = true;
  }
  CHECK(varied);

  host.shutdown();
}

TEST_CASE("tri_shear: update_rate 0 never auto-updates; trigger re-analyzes", "[tri_shear]") {
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

  inst->setParamFloat("algorithm", 1.0f);
  inst->setParamFloat("direction", -1.0f);
  inst->setParamFloat("duration", 0.0f);
  inst->setParamFloat("distance", 0.4f);
  inst->setParamFloat("rift_fill", 1.0f);
  inst->setParamFloat("obliqueness", 0.5f);
  inst->setParamFloat("obliqueness_jitter", 1.0f);   // would vary IF it re-rolled
  inst->setParamFloat("update_rate", 0.0f);          // never auto-update

  inst->doTick(0.05); inst->doRender(W, H);
  auto a = backend->readbackTexture(outTex, W, H);
  // Lots of time passes, but update_rate 0 means no auto-update → no re-roll.
  inst->doTick(0.5); inst->doRender(W, H);
  auto b = backend->readbackTexture(outTex, W, H);
  CHECK(a == b);

  // A manual trigger re-analyzes (and re-rolls the jitter) → the output varies.
  bool varied = false;
  for (int i = 0; i < 8 && !varied; ++i) {
    inst->setParamFloat("trigger", 0.0f);
    inst->setParamFloat("trigger", 1.0f);   // rising edge
    inst->doTick(0.05); inst->doRender(W, H);
    auto cur = backend->readbackTexture(outTex, W, H);
    if (meanDiff(cur, a) > 2.0) varied = true;
  }
  CHECK(varied);

  host.shutdown();
}

TEST_CASE("tri_shear per-region colour tint changes the output", "[tri_shear]") {
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

  inst->setParamFloat("algorithm", 1.0f);
  inst->setParamFloat("direction", -1.0f);
  inst->setParamFloat("duration", 0.0f);
  inst->setParamFloat("update_rate", 0.0f);
  inst->setParamFloat("distance", 0.3f);
  inst->setParamFloat("size", 0.4f);
  inst->setParamFloat("rift_fill", 1.0f);

  inst->setParamFloat("tint", 0.0f);
  inst->doTick(0.05); inst->doRender(W, H);
  auto untinted = backend->readbackTexture(outTex, W, H);

  inst->setParamFloat("tint", 1.0f);   // default red/green/blue wedges
  inst->setParamFloat("tint_mode", 0.0f);   // multiply
  inst->doTick(0.05); inst->doRender(W, H);
  auto mul = backend->readbackTexture(outTex, W, H);

  inst->setParamFloat("tint_mode", 1.0f);   // add
  inst->doTick(0.05); inst->doRender(W, H);
  auto add = backend->readbackTexture(outTex, W, H);

  INFO("mul vs untinted = " << meanDiff(mul, untinted)
       << "  add vs mul = " << meanDiff(add, mul));
  CHECK(meanDiff(mul, untinted) > 8.0);
  CHECK(meanDiff(add, mul) > 8.0);

  host.shutdown();
}

TEST_CASE("tri_shear defaults to opaque-black fills", "[tri_shear]") {
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

  // Rely on the Black fill defaults. Big rift.
  inst->setParamFloat("algorithm", 1.0f);
  inst->setParamFloat("direction", -1.0f);
  inst->setParamFloat("duration", 0.0f);
  inst->setParamFloat("update_rate", 0.0f);
  inst->setParamFloat("size", 0.8f);
  inst->setParamFloat("distance", 0.5f);

  inst->doTick(0.05);
  inst->doRender(W, H);
  auto out = backend->readbackTexture(outTex, W, H);

  int opaqueBlack = 0, transparent = 0;
  for (size_t i = 0; i + 3 < out.size(); i += 4) {
    if (out[i] < 8 && out[i + 1] < 8 && out[i + 2] < 8 && out[i + 3] > 200) opaqueBlack++;
    if (out[i + 3] < 8) transparent++;
  }
  INFO("opaque-black=" << opaqueBlack << " transparent=" << transparent);
  CHECK(opaqueBlack > 50);      // rifts fill with opaque black by default
  CHECK(transparent == 0);

  host.shutdown();
}
