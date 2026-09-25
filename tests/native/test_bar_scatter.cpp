// test_bar_scatter.cpp — filter.light.bar_scatter under a live, changing
// input, on Metal through the real executor.
//
// The web suite can only feed a solid colour, so the only flash it can fire
// is the image arriving. What matters in a show is the opposite: a strobe
// that keeps hitting, and knobs that move while it does. Here every frame
// writes its own input and the sketch's params can change mid-run, so these
// cases pin exactly the failures a latched design has: a choke that restarts
// itself on every hit and never lets go, fates that never re-roll, and a
// Strength knob that stops doing anything once the bars are busy.

#include <catch2/catch_test_macros.hpp>

#include <array>
#include <set>
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

constexpr uint32_t W = 64, H = 80;   // 4 bars × 16 px, 10 rows × 8 px
constexpr int kRgba8 = 1;
constexpr double kDt = 1.0 / 60.0;
// 120 BPM: a flash every 30 frames, lit for 6.
constexpr int kPeriod = 30, kOn = 6;

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
        { "type": "module", "module_type": "filter.light.bar_scatter", "instance_key": "bs" }
      ],
      "instances": { "bs": { "module_type": "filter.light.bar_scatter", "state": {} } }
    })JSON");
    sketch["instances"]["bs"]["state"] = state;
    return in >= 0 && out >= 0;
  }

  void set(const char* key, double v) { sketch["instances"]["bs"]["state"][key] = v; }

  // One frame: the four bars at `level` (0..1 grey), returns each bar's red
  // at its middle cell.
  std::array<int, 4> frame(const std::array<float, 4>& level, bool dirty) {
    std::vector<uint8_t> px((size_t)W * H * 4, 255);
    for (uint32_t y = 0; y < H; ++y)
      for (uint32_t x = 0; x < W; ++x) {
        uint8_t v = (uint8_t)(level[x * 4 / W] * 255.0f + 0.5f);
        size_t i = ((size_t)y * W + x) * 4;
        px[i] = px[i + 1] = px[i + 2] = v;
      }
    backend->writeTexture(in, W, H, px.data(), (uint32_t)px.size());
    int32_t o = ex->execute(sketch, in, out, (int)W, (int)H, kDt, dirty);
    backend->submit();
    auto rb = backend->readbackTexture(o, W, H);
    std::array<int, 4> r{};
    for (int b = 0; b < 4; ++b) r[b] = rb[((size_t)(H / 2) * W + (size_t)(b * 16 + 8)) * 4];
    return r;
  }
};

bool strobeOn(int f) { return f % kPeriod < kOn; }
std::array<float, 4> all(float v) { return {v, v, v, v}; }
int litCount(const std::array<int, 4>& r) {
  int n = 0;
  for (int v : r) n += v > 200;
  return n;
}

}  // namespace

TEST_CASE("bar_scatter: a 120 BPM strobe re-scatters on every hit", "[bar_scatter]") {
  Rig rig;
  if (!rig.init({{"rows", 10}, {"strength", 1.0}, {"delay_share", 0.0}, {"seed", 3}}))
    SKIP("No GPU device / bundles");

  std::set<int> survivors;
  int flashes = 0;
  for (int f = 0; f < kPeriod * 14; ++f) {
    auto r = rig.frame(all(strobeOn(f) ? 1.0f : 0.0f), f == 0);
    // Two frames into each flash (the first flash is the image arriving too).
    if (f % kPeriod == 2 && f >= kPeriod) {
      ++flashes;
      INFO("flash " << flashes << " bars " << r[0] << " " << r[1] << " " << r[2] << " " << r[3]);
      // Every hit is scattered: exactly one bar survives it...
      CHECK(litCount(r) == 1);
      for (int b = 0; b < 4; ++b) if (r[b] > 200) survivors.insert(b);
      // ...and the rest are choked to black, not left half-lit.
      for (int v : r) CHECK((v > 200 || v < 10));
    }
  }
  // Fresh dice on every hit: the survivor moves around the rig.
  INFO("distinct survivors over " << flashes << " flashes: " << survivors.size());
  CHECK(survivors.size() >= 3);
}

TEST_CASE("bar_scatter: Strength lands mid-strobe, both ways", "[bar_scatter]") {
  Rig rig;
  if (!rig.init({{"rows", 10}, {"strength", 1.0}, {"delay_share", 0.0}, {"seed", 5}}))
    SKIP("No GPU device / bundles");

  int f = 0;
  for (; f < kPeriod * 5 + 2; ++f) rig.frame(all(strobeOn(f) ? 1.0f : 0.0f), f == 0);
  // Mid-flash, three bars choked. Strength to 0: the very next frame lets the
  // flash through everywhere.
  rig.set("strength", 0.0);
  auto open = rig.frame(all(strobeOn(f) ? 1.0f : 0.0f), true);
  ++f;
  CHECK(litCount(open) == 4);
  // It stays open through the following hits.
  for (; f < kPeriod * 8; ++f) {
    auto r = rig.frame(all(strobeOn(f) ? 1.0f : 0.0f), false);
    if (strobeOn(f)) CHECK(litCount(r) == 4);
  }
  // And back up: the next hit is scattered again.
  rig.set("strength", 1.0);
  bool dirty = true;
  for (; f < kPeriod * 9 + 3; ++f) {
    auto r = rig.frame(all(strobeOn(f) ? 1.0f : 0.0f), dirty);
    dirty = false;
    if (f == kPeriod * 9 + 2) CHECK(litCount(r) == 1);
  }
}

TEST_CASE("bar_scatter: chaotic hits never lock it up", "[bar_scatter]") {
  Rig rig;
  if (!rig.init({{"rows", 10}, {"strength", 0.6}, {"seed", 11}}))
    SKIP("No GPU device / bundles");

  // Random flashes, random bars, random lengths: a busy, messy input.
  uint32_t h = 12345;
  auto rnd = [&]() { h = h * 1664525u + 1013904223u; return (h >> 8) * (1.0f / 16777216.0f); };
  std::array<float, 4> lv = all(0.0f);
  for (int f = 0; f < 600; ++f) {
    if (rnd() < 0.15f) {
      bool together = rnd() < 0.6f;
      for (int b = 0; b < 4; ++b) lv[b] = (together || rnd() < 0.5f) ? 1.0f : 0.0f;
    } else if (rnd() < 0.2f) {
      lv = all(0.0f);
    }
    rig.frame(lv, f == 0);
  }
  // Then it all stops. A steady picture comes up — all four bars together,
  // so that is one last scatter — and once its choke and delay have run
  // out, it passes untouched: nothing is left latched from the mess.
  for (int f = 0; f < 60; ++f) rig.frame(all(0.0f), false);
  std::array<int, 4> r{};
  for (int f = 0; f < 90; ++f) r = rig.frame(all(0.5f), false);
  for (int v : r) CHECK(std::abs(v - 128) <= 2);
}

#endif
