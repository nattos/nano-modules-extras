// test_side_racks_render.cpp — source.light.side_racks through the real
// executor on Metal: the compact grid lands the way the rig stands (left rack
// on the left half, inner ends meeting in the middle, top bar on top), every
// segment is a flat block, and an edge on the primary gate launches a Wave
// that reaches the inner ends first.

#include <catch2/catch_test_macros.hpp>

#include <algorithm>
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

constexpr uint32_t W = 240, H = 60;   // 120 columns × 2 px, 3 rows × 20 px
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
  std::vector<uint8_t> rb;

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
        { "type": "module", "module_type": "source.light.side_racks", "instance_key": "sr" }
      ],
      "instances": { "sr": { "module_type": "source.light.side_racks", "state": {} } }
    })JSON");
    sketch["instances"]["sr"]["state"] = state;
    return in >= 0 && out >= 0;
  }

  void set(const char* key, double v) { sketch["instances"]["sr"]["state"][key] = v; }

  void frame(bool dirty) {
    int32_t o = ex->execute(sketch, in, out, (int)W, (int)H, kDt, dirty);
    backend->submit();
    rb = backend->readbackTexture(o, W, H);
  }

  /// Green at a pixel of grid column `col`, row `row` (row 0 the top bar),
  /// `dx`/`dy` inside the 2 × 20 px block.
  int green(int col, int row, int dx = 1, int dy = 10) const {
    const size_t x = (size_t)(col * 2 + dx), y = (size_t)(row * 20 + dy);
    return (int)rb[(y * W + x) * 4 + 1];
  }
};

}  // namespace

TEST_CASE("side_racks: the grid stands like the rig, in flat blocks", "[side_racks]") {
  // Parked halfway into the fade, at rest, full tilt kept at rest: the bars
  // the front has passed (the bottom) are darker than the top.
  Rig rig;
  if (!rig.init({{"color", {1.0, 1.0, 1.0}}, {"tilt", 1.0}, {"rest_tilt", 1.0},
                 {"sweep", 0.5 + 0.5 * (0.45 + 0.5 * 0.55)}}))
    SKIP("No GPU device / bundles");
  for (int f = 0; f < 10; ++f) rig.frame(f == 0);

  const int top = rig.green(10, 0), mid = rig.green(10, 1), bot = rig.green(10, 2);
  CHECK(top > mid);
  CHECK(mid > bot);
  CHECK(top > 100);
  // The racks mirror, and every segment of a bar is one block.
  for (int row = 0; row < 3; ++row) {
    const int v = rig.green(0, row);
    for (int col : {0, 30, 59, 60, 90, 119}) CHECK(std::abs(rig.green(col, row) - v) <= 1);
    for (int dy : {1, 10, 18})
      for (int dx : {0, 1}) CHECK(rig.green(45, row, dx, dy) == rig.green(45, row));
  }
}

TEST_CASE("side_racks: the primary gate launches a Wave from the middle", "[side_racks]") {
  Rig rig;
  if (!rig.init({{"color", {1.0, 1.0, 1.0}}, {"primary_mode", 1}, {"wave_start", 0.0},
                 {"wave_speed", 2.0}, {"wave_amount", 1.0}, {"wave_sparkle", 0.0}}))
    SKIP("No GPU device / bundles");
  for (int f = 0; f < 10; ++f) rig.frame(f == 0);
  for (int col : {0, 59, 60, 119}) CHECK(rig.green(col, 1) > 250);

  // Press and release, as the inspector's button does.
  rig.set("primary", 1.0);
  rig.frame(true);
  rig.set("primary", 0.0);
  // 0.7 s later the front is 1.4 m out: past the inner ends, short of the
  // outer ones.
  for (int f = 0; f < 42; ++f) rig.frame(f == 0);
  for (int row = 0; row < 3; ++row) {
    CHECK(rig.green(59, row) < 200);    // left rack, inner end
    CHECK(rig.green(60, row) < 200);    // right rack, inner end
    CHECK(rig.green(0, row) > 250);     // left rack, outer end
    CHECK(rig.green(119, row) > 250);   // right rack, outer end
    CHECK(std::abs(rig.green(59, row) - rig.green(60, row)) <= 1);
  }

  // Long after, it is all back — and the replayed 0 never fires again.
  for (int f = 0; f < 240; ++f) rig.frame(f == 0);
  for (int col : {0, 59, 60, 119}) CHECK(rig.green(col, 1) > 250);
}

#endif
