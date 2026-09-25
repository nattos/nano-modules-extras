// test_sweep_gradient_render.cpp — source.light.sweep_gradient through the
// real executor on Metal: the Trigger field, edge-detected from the replayed
// state, fires the gesture over a knob parked at black.

#include <catch2/catch_test_macros.hpp>

#include <array>
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
        { "type": "module", "module_type": "source.light.sweep_gradient", "instance_key": "sg" }
      ],
      "instances": { "sg": { "module_type": "source.light.sweep_gradient", "state": {} } }
    })JSON");
    sketch["instances"]["sg"]["state"] = state;
    return in >= 0 && out >= 0;
  }

  void set(const char* key, double v) { sketch["instances"]["sg"]["state"][key] = v; }

  /// One frame; returns red at the bottom and top segment of bar 1.
  std::array<int, 2> frame(bool dirty) {
    int32_t o = ex->execute(sketch, in, out, (int)W, (int)H, kDt, dirty);
    backend->submit();
    auto rb = backend->readbackTexture(o, W, H);
    auto red = [&](int seg) {
      const size_t y = H - 1 - (size_t)(seg * 8 + 4);
      return (int)rb[(y * W + 24) * 4];
    };
    return {red(0), red(9)};
  }
};

}  // namespace

TEST_CASE("sweep_gradient: Trigger plays the whole wave over a knob at black", "[sweep_gradient]") {
  Rig rig;
  if (!rig.init({{"sweep", 0.0}, {"trig_side", 0}, {"color", {1.0, 1.0, 1.0}}}))
    SKIP("No GPU device / bundles");

  std::array<int, 2> r{};
  for (int f = 0; f < 30; ++f) r = rig.frame(f == 0);
  CHECK(r[0] < 5);
  CHECK(r[1] < 5);

  // Press and release, as the inspector's button does.
  rig.set("trigger", 1.0);
  r = rig.frame(true);
  rig.set("trigger", 0.0);
  CHECK(r[0] > 250);   // taken over: lit on the fire
  CHECK(r[1] > 250);
  bool bottom_first = false, top_back = false, top_dark = false;
  for (int f = 0; f < 150; ++f) {
    r = rig.frame(f == 0);
    if (r[0] < 100 && r[1] > r[0] + 60) bottom_first = true;
    if (r[1] < 60) top_dark = true;
    if (top_dark && r[1] > 200) top_back = true;
  }
  CHECK(bottom_first);
  CHECK(top_back);
  // And the knob's black is back.
  CHECK(r[0] < 5);
  CHECK(r[1] < 5);

  // The replayed trigger value (0) never fires again on its own.
  for (int f = 0; f < 60; ++f) r = rig.frame(false);
  CHECK(r[0] < 5);
  CHECK(r[1] < 5);
}

#endif
