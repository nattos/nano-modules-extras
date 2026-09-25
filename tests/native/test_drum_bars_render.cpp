// test_drum_bars_render.cpp — source.light.drum_bars through the real
// executor on Metal: the Ch1 field, edge-detected from the replayed state,
// fires a mirror-symmetric kick that dies back to black.

#include <catch2/catch_test_macros.hpp>

#include <algorithm>
#include <array>
#include <cstdlib>
#include <string>
#include <vector>

#include <nlohmann/json.hpp>

#include "gpu/gpu_backend.h"
#include "runtime/effect_runtime.h"
#include "sketch/module_registry.h"
#include "sketch/sketch_executor.h"
#include "sketch/wasm_bundles.h"

#include "wasm_paths.h"

using effect_runtime::EffectRuntime;

#if defined(CORE_WASM_PATH) && defined(LIGHTS_WASM_PATH)

namespace {

constexpr uint32_t W = 64, H = 80;   // 4 bars × 16 px, 10 segments × 8 px
constexpr int kRgba8 = 1;
constexpr double kDt = 1.0 / 60.0;

struct Rig {
  std::unique_ptr<gpu::GPUBackend> backend;
  sketch_executor::WasmEffectBundles bundles;
  std::unique_ptr<EffectRuntime> rt;
  std::unique_ptr<sketch_executor::ModuleRegistry> registry;
  std::unique_ptr<sketch_executor::SketchExecutor> ex;
  int in = -1, out = -1;
  nlohmann::json sketch;

  bool init(const nlohmann::json& state) {
    backend = gpu::createBackend();
    if (!backend) return false;
    if (!bundles.init()) return false;
    rt = std::make_unique<EffectRuntime>(backend.get());
    registry = std::make_unique<sketch_executor::ModuleRegistry>(rt.get());
    if (bundles.loadBundleFile(kCoreWasm, *registry, backend.get(), nullptr) <= 1) return false;
    if (bundles.loadBundleFile(kLightsWasm, *registry, backend.get(), nullptr) <= 0) return false;
    ex = std::make_unique<sketch_executor::SketchExecutor>(rt.get(), registry.get(), backend.get());
    in = backend->createTexture(W, H, kRgba8);
    out = backend->createTexture(W, H, kRgba8);
    sketch = nlohmann::json::parse(R"JSON({
      "wires": [],
      "chain": [
        { "type": "module", "module_type": "source.light.drum_bars", "instance_key": "db" }
      ],
      "instances": { "db": { "module_type": "source.light.drum_bars", "state": {} } }
    })JSON");
    sketch["instances"]["db"]["state"] = state;
    return in >= 0 && out >= 0;
  }

  void set(const char* key, double v) { sketch["instances"]["db"]["state"][key] = v; }

  /// One frame; returns red at the middle of each bar (segments 0 and 9 too,
  /// to see every segment of a bar agree).
  std::array<int, 4> frame(bool dirty, bool* flat = nullptr) {
    int32_t o = ex->execute(sketch, in, out, (int)W, (int)H, kDt, dirty);
    backend->submit();
    auto rb = backend->readbackTexture(o, W, H);
    auto red = [&](int bar, int seg) {
      const size_t y = H - 1 - (size_t)(seg * 8 + 4);
      return (int)rb[(y * W + (size_t)(bar * 16 + 8)) * 4];
    };
    std::array<int, 4> r{};
    bool f = true;
    for (int b = 0; b < 4; ++b) {
      r[b] = red(b, 5);
      if (std::abs(red(b, 0) - r[b]) > 1 || std::abs(red(b, 9) - r[b]) > 1) f = false;
    }
    if (flat) *flat = f;
    return r;
  }
};

}  // namespace

TEST_CASE("drum_bars: Ch1 fires a mirror-symmetric kick that dies to black", "[drum_bars]") {
  Rig rig;
  if (!rig.init({{"color", {1.0, 1.0, 1.0}}, {"flam", 0.0}}))
    SKIP("No GPU device / bundles");

  std::array<int, 4> r{};
  for (int f = 0; f < 10; ++f) r = rig.frame(f == 0);
  for (int b = 0; b < 4; ++b) CHECK(r[b] < 3);

  // Press and release, as the inspector's button does.
  rig.set("ch1", 1.0);
  bool flat = false;
  r = rig.frame(true, &flat);
  rig.set("ch1", 0.0);
  CHECK(flat);
  CHECK(r[0] == r[3]);
  CHECK(r[1] == r[2]);
  CHECK(std::max(r[0], r[1]) > 250);
  CHECK((r[0] > 250 || r[0] < 3));
  CHECK((r[1] > 250 || r[1] < 3));

  for (int f = 0; f < 60; ++f) r = rig.frame(f == 0);
  for (int b = 0; b < 4; ++b) CHECK(r[b] < 3);

  // The replayed 0 never fires again on its own.
  for (int f = 0; f < 30; ++f) r = rig.frame(false);
  for (int b = 0; b < 4; ++b) CHECK(r[b] < 3);
}

#endif
