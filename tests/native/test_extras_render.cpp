// test_extras_render.cpp — end-to-end renders of the extras' effects (nano,
// lights, legacy) through the real executor on Metal: the cases that used
// to live in nano-modules' native/tests/test_effect_render.cpp. Built by
// nano-modules' CMake when NANO_EXTRAS_DIR names this checkout (see
// native/CMakeLists.txt here); the bundles come from its build/wasm.

#include <array>
#include <catch2/catch_test_macros.hpp>
#include <catch2/catch_approx.hpp>

#include <algorithm>
#include <cmath>
#include <fstream>
#include <vector>

#include <nlohmann/json.hpp>

#include "bridge/param_cache.h"
#include "gpu/gpu_backend.h"
#include "runtime/effect_runtime.h"
#include "runtime/shader_from_spv.h"
#include "runtime/text_host.h"
#include "sketch/module_registry.h"
#include "sketch/sketch_executor.h"
#include "sketch/trigger_bus.h"
#include "sketch/wasm_bundles.h"
#include "wasm/wasm_host.h"

#include "wasm_paths.h"         // kCoreWasm & co. — see there

using bridge::ParamCache;
using wasm::WasmHost;
using wasm::WasmEffectDesc;
using effect_runtime::EffectRuntime;
using effect_runtime::EffectInstance;

static std::vector<uint8_t> load_file(const char* path) {
  std::ifstream f(path, std::ios::binary | std::ios::ate);
  if (!f) return {};
  auto size = f.tellg();
  f.seekg(0);
  std::vector<uint8_t> buf(static_cast<size_t>(size));
  f.read(reinterpret_cast<char*>(buf.data()), size);
  return buf;
}

static double mean_rgb(const std::vector<uint8_t>& px) {
  long sum = 0, n = 0;
  for (size_t i = 0; i + 3 < px.size(); i += 4) {
    sum += px[i] + px[i + 1] + px[i + 2];
    n += 3;
  }
  return n ? static_cast<double>(sum) / n : 0.0;
}

// triangulate — the topology-following GPU triangulation effect (nano bundle).
// Validates the P2 pipeline end-to-end on Metal: downsample→blur→feature (via
// the Density debug view — a left-bright/right-dark input must read brighter on
// the left) and the JFA Voronoi partition (via the Voronoi debug view — the
// random per-cell colouring must produce high spatial variance, which is only
// possible if the seed pool splatted and the jump-flood propagated).
#ifdef NANO_WASM_PATH
static double stddev_luma(const std::vector<uint8_t>& px) {
  double m = mean_rgb(px), s = 0; long n = 0;
  for (size_t i = 0; i + 3 < px.size(); i += 4) {
    double l = (px[i] + px[i + 1] + px[i + 2]) / 3.0;
    s += (l - m / 3.0) * (l - m / 3.0); ++n;   // mean_rgb averages 3 channels
  }
  return n ? std::sqrt(s / n) : 0.0;
}

TEST_CASE("WASM GPU effect renders topology triangulation (triangulate)", "[effect_render]") {
  auto backend = gpu::createBackend();
  if (!backend) SKIP("No GPU device available");

  auto bytecode = load_file(kNanoWasm);
  REQUIRE(!bytecode.empty());

  ParamCache cache;
  WasmHost host(cache);
  REQUIRE(host.init());
  int32_t id = host.load_module(bytecode.data(), bytecode.size());
  INFO("last_error: " << host.last_error());
  REQUIRE(id >= 0);
  host.set_gpu_backend(id, backend.get());
  REQUIRE(host.call_function(id, "nano_module_main") == 0);

  const WasmEffectDesc* w = nullptr;
  for (const auto& e : host.registered_effects(id))
    if (e.id == "filter.mesh.triangulate") { w = &e; break; }
  REQUIRE(w != nullptr);

  EffectRuntime rt(backend.get());
  sketch_executor::ModuleRegistry registry(&rt);
  REQUIRE(registry.registerWasmEffect("filter.mesh.triangulate", "Triangulate", &host, id, *w));
  EffectInstance* inst = rt.instanceFor("filter.mesh.triangulate", "k0");
  REQUIRE(inst != nullptr);

  const uint32_t W = 128, H = 128;
  const int RGBA8 = 1;
  int inTex = backend->createTexture(W, H, RGBA8);
  int outTex = backend->createTexture(W, H, RGBA8);
  REQUIRE(inTex >= 0); REQUIRE(outTex >= 0);

  // Left half bright, right half dark.
  std::vector<uint8_t> inPixels(W * H * 4, 0);
  for (uint32_t y = 0; y < H; ++y)
    for (uint32_t x = 0; x < W; ++x) {
      size_t i = (y * W + x) * 4;
      uint8_t v = (x < W / 2) ? 210 : 25;
      inPixels[i] = inPixels[i + 1] = inPixels[i + 2] = v; inPixels[i + 3] = 255;
    }
  backend->writeTexture(inTex, W, H, inPixels.data(), (uint32_t)inPixels.size());
  inst->setTextureField("tex_in", inTex);
  inst->setTextureField("tex_out", outTex);
  inst->setParamFloat("density", 0.4f);
  inst->setParamFloat("feature_scale", 0.35f);

  auto halves = [&](const std::vector<uint8_t>& px, double& left, double& right) {
    double sl = 0, sr = 0; long nl = 0, nr = 0;
    for (uint32_t y = 0; y < H; ++y)
      for (uint32_t x = 0; x < W; ++x) {
        size_t i = (y * W + x) * 4;
        double l = (px[i] + px[i + 1] + px[i + 2]) / 3.0;
        if (x < W / 2) { sl += l; ++nl; } else { sr += l; ++nr; }
      }
    left = nl ? sl / nl : 0; right = nr ? sr / nr : 0;
  };

  // A. Density debug view → left (bright input) reads brighter than right.
  inst->setParamFloat("debug_view", 1.0f);
  inst->doRender(W, H);
  inst->doRender(W, H);
  auto density = backend->readbackTexture(outTex, W, H);
  REQUIRE(density.size() == W * H * 4);
  double dl = 0, dr = 0; halves(density, dl, dr);
  INFO("density view: left " << dl << "  right " << dr);
  CHECK(dl > dr + 20.0);

  // B. Voronoi debug view → random per-cell colours → high spatial variance.
  inst->setParamFloat("debug_view", 5.0f);
  inst->doRender(W, H);
  auto voronoi = backend->readbackTexture(outTex, W, H);
  double sd = stddev_luma(voronoi);
  INFO("voronoi view stddev " << sd);
  CHECK(sd > 20.0);

  // C. Mesh output (debug off, dark backdrop, white edges): the Delaunay edges
  // must rasterize as lit pixels over the black background — validates edge
  // extraction + the instanced line render pass over the compute backdrop.
  inst->setParamFloat("debug_view", 0.0f);
  inst->setParamFloat("bg_mode", 1.0f);     // dark
  inst->setParamFloat("density", 0.05f);    // sparse enough for gaps on a 128px canvas
  inst->setParamFloat("line_width", 0.0f);  // thin (~1px) lines
  inst->setParamArray("line_color", {1.0f, 1.0f, 1.0f});
  inst->doRender(W, H);
  inst->doRender(W, H);
  auto mesh = backend->readbackTexture(outTex, W, H);
  long lit = 0, dark = 0;
  for (size_t i = 0; i + 3 < mesh.size(); i += 4) {
    double l = (mesh[i] + mesh[i + 1] + mesh[i + 2]) / 3.0;
    if (l > 60.0) ++lit; else ++dark;
  }
  INFO("mesh: lit " << lit << "  dark " << dark << " / " << (W * H));
  CHECK(lit > 40);                          // edges drawn (mesh rasterizes)
  CHECK(dark > (long)(W * H) / 10);         // structured wireframe, not a full-screen fill

  host.shutdown();
}
#endif  // NANO_WASM_PATH

// ---------------------------------------------------------------------------
// Arena crash repro (2026-07-04): three relaunches, three identical SIGSEGVs on
// the Render Thread at a STABLE low address (~0xd432c0) that WAMR's trap
// handler refused as a wasm OOB — the signature of a wasm offset dereferenced
// off a dead/NULL memory base. Live topology at crash time (recovered from the
// composition's nanobarrel://config blobs): two active barrel instances in one
// process — [brutal_fold → auto_level → edges] and [barrel_macros → shape_burst
// + a macro_0→manual wire] — at 1920×1080, with the web client editing (the
// crash always followed a `regenerate`, i.e. a dirty rebuild that destroys and
// re-creates the wasm effect instances). This case replays exactly that:
// alternating executors, per-frame param churn, trigger events, and periodic
// dirty rebuilds.
#ifdef NANO_WASM_PATH
TEST_CASE("arena repro: brutal_fold + shape_burst chains survive regenerate churn",
          "[effect_render][arena_repro]") {
  auto backend = gpu::createBackend();
  if (!backend) SKIP("No GPU device available");

  sketch_executor::WasmEffectBundles bundles;
  REQUIRE(bundles.init());
  EffectRuntime rt(backend.get());
  sketch_executor::ModuleRegistry registry(&rt);
  REQUIRE(bundles.loadBundleFile(kCoreWasm, registry, backend.get(), nullptr) > 1);
  REQUIRE(bundles.loadBundleFile(kNanoWasm, registry, backend.get(), nullptr) > 1);

  sketch_executor::SketchExecutor A(&rt, &registry, backend.get());
  A.setKeyNamespace("A/");
  sketch_executor::SketchExecutor B(&rt, &registry, backend.get());
  B.setKeyNamespace("B/");

  const uint32_t W = 1920, H = 1080;
  const int RGBA8 = 1;
  int inA = backend->createTexture(W, H, RGBA8);
  int outA = backend->createTexture(W, H, RGBA8);
  int inB = backend->createTexture(W, H, RGBA8);
  int outB = backend->createTexture(W, H, RGBA8);
  REQUIRE(inA >= 0); REQUIRE(outA >= 0); REQUIRE(inB >= 0); REQUIRE(outB >= 0);

  // Instance states lifted from the crashing composition (help text dropped).
  auto sketchA = nlohmann::json::parse(R"JSON({
    "chain": [
      { "type": "module", "module_type": "source.brutal_fold",     "instance_key": "bf" },
      { "type": "module", "module_type": "color.tone.auto_level",  "instance_key": "al" },
      { "type": "module", "module_type": "filter.edges",           "instance_key": "ed" }
    ],
    "instances": {
      "bf": { "module_type": "source.brutal_fold", "state": {
        "anim_amount": 1, "complexity": 0.2924, "order": 0.2062, "liveliness": 1,
        "time_speed": 0.12, "scale": 0.7, "extrude": 1, "fog": 5,
        "interp_cells": true, "second_structure": true, "vol_amount": 1,
        "vol_depth": 0.12, "vol_radius": 0.5, "vol_shape": 1, "vol_z": 0.55,
        "diff_hue_hi": 0.586, "diff_hue_lo": 0.283, "diff_hue_mid": 0.815
      } },
      "al": { "module_type": "color.tone.auto_level", "state": {
        "equalize": 0.87, "median_pull": 0.57, "median_target": 0.41 } },
      "ed": { "module_type": "filter.edges", "state": {
        "bg": [0,0,0], "keep_input": 0, "line": [1,1,1],
        "radius": 0.14, "threshold": 0.17 } }
    }
  })JSON");

  auto sketchB = nlohmann::json::parse(R"JSON({
    "chain": [
      { "type": "module", "module_type": "control.barrel_macros", "instance_key": "mac" },
      { "type": "module", "module_type": "source.shape_burst",    "instance_key": "burst" }
    ],
    "instances": {
      "mac": { "module_type": "control.barrel_macros", "state": { "macro_0": 0.0 } },
      "burst": { "module_type": "source.shape_burst", "state": {
        "auto_rate": 0, "composite": 0, "distort": 0.15, "distort_freq": 0.4,
        "duration": 0.3, "manual": 0, "voices": 1, "shape": 0,
        "thickness": 0.03, "min_scale": 0.05, "max_scale": 1.2,
        "motion_strength": 1, "tilt": 0, "trigger": 0 } }
    },
    "wires": [
      { "id": "w0", "combine": "add",
        "src":  { "instanceKey": "mac",   "field": "macro_0" },
        "dest": { "instanceKey": "burst", "field": "manual" } }
    ]
  })JSON");

  for (int f = 0; f < 900; ++f) {
    // The web client editing → periodic full regenerates (plan rebuild:
    // destroy + re-create every wasm effect instance).
    const bool dirty = (f % 60) == 0;
    // Live param churn (macro knob riding, brutal_fold pad drag).
    sketchB["instances"]["mac"]["state"]["macro_0"] = 0.5 + 0.5 * std::sin(f * 0.11);
    sketchA["instances"]["bf"]["state"]["complexity"] = 0.29 + 0.2 * std::sin(f * 0.05);
    // Occasional trigger events firing shape_burst voices.
    if (f % 90 == 30) sketchB["instances"]["burst"]["state"]["trigger"] = f;

    int32_t oA = A.execute(sketchA, inA, outA, (int)W, (int)H, 1.0 / 60.0, dirty);
    backend->submit();
    int32_t oB = B.execute(sketchB, inB, outB, (int)W, (int)H, 1.0 / 60.0, dirty);
    backend->submit();
    REQUIRE(oA > 0);
    REQUIRE(oB > 0);
  }

  // Surviving 900 frames (15 dirty rebuilds) without a signal IS the assertion.
  auto pxA = backend->readbackTexture(outA, W, H);
  auto pxB = backend->readbackTexture(outB, W, H);
  CHECK(pxA.size() == (size_t)W * H * 4);
  CHECK(pxB.size() == (size_t)W * H * 4);
}

// The full Phase A+D native emit path: triggering control.nanolooper's
// trigger_1 (rising edge) must push an on-event onto its "triggers" ring, which
// the executor drains onto the process-global trigger_bus (channel 1). This is
// exactly the barrel path — if this passes, a live "nothing happens" is a setup
// or deployment issue, not the emit code.
TEST_CASE("control.nanolooper emits a trigger onto the rail when fired",
          "[effect_render][nanolooper][trigger_rail]") {
  auto backend = gpu::createBackend();
  if (!backend) SKIP("No GPU device available");

  sketch_executor::WasmEffectBundles bundles;
  REQUIRE(bundles.init());
  EffectRuntime rt(backend.get());
  sketch_executor::ModuleRegistry registry(&rt);
  REQUIRE(bundles.loadBundleFile(kCoreWasm, registry, backend.get(), nullptr) > 1);
  REQUIRE(bundles.loadBundleFile(kNanoWasm, registry, backend.get(), nullptr) > 1);

  sketch_executor::SketchExecutor executor(&rt, &registry, backend.get());
  const uint32_t W = 64, H = 64;
  int inTex = backend->createTexture(W, H, 1);
  int outTex = backend->createTexture(W, H, 1);
  REQUIRE(inTex >= 0); REQUIRE(outTex >= 0);

  // A one-effect sketch: control.nanolooper with send_to_rail on and no trigger
  // pressed yet.
  auto sketch = nlohmann::json::parse(R"JSON({
    "chain": [
      { "type": "module", "module_type": "control.nanolooper", "instance_key": "lp" }
    ],
    "instances": {
      "lp": { "module_type": "control.nanolooper",
              "state": { "send_to_rail": true, "trigger_1": 0 } }
    }
  })JSON");

  trigger_bus::resetForTest();

  // Frame 0: establish the looper (trigger_1 low) — arms the rising-edge
  // detector and baselines the executor's per-instance trigger seq watermark.
  REQUIRE(executor.execute(sketch, inTex, outTex, (int)W, (int)H, 1.0 / 60.0, true) >= 0);
  backend->submit();
  trigger_bus::drain("test");  // consume any baseline

  // Frame 1: press trigger_1 → rising edge → gate_on → ring → executor drain.
  // A trigger via a state edit is a dirty frame (that's how the editor delivers
  // it — maybeApplyState only re-applies persisted params on dirty frames).
  sketch["instances"]["lp"]["state"]["trigger_1"] = 1;
  REQUIRE(executor.execute(sketch, inTex, outTex, (int)W, (int)H, 1.0 / 60.0, true) >= 0);
  backend->submit();

  auto events = trigger_bus::drain("test");
  bool saw_ch1_on = false;
  for (const auto& e : events) {
    INFO("event ch=" << e.channel << " on=" << e.on);
    if (e.channel == 1 && e.on) saw_ch1_on = true;
  }
  CHECK(saw_ch1_on);
}

// strict_deadline > 0 makes each emitted trigger carry the precision subtree,
// which the executor's drainTriggerRing parses onto the bus Event (strict +
// deadline_ms). strict_deadline == 0 emits none (→ "any"). This proves the full
// authoring path effect-schema → publish → drain → bus for the new payload.
TEST_CASE("control.nanolooper strict_deadline authors the precision payload",
          "[effect_render][nanolooper][trigger_rail]") {
  auto backend = gpu::createBackend();
  if (!backend) SKIP("No GPU device available");

  sketch_executor::WasmEffectBundles bundles;
  REQUIRE(bundles.init());
  EffectRuntime rt(backend.get());
  sketch_executor::ModuleRegistry registry(&rt);
  REQUIRE(bundles.loadBundleFile(kCoreWasm, registry, backend.get(), nullptr) > 1);
  REQUIRE(bundles.loadBundleFile(kNanoWasm, registry, backend.get(), nullptr) > 1);

  sketch_executor::SketchExecutor executor(&rt, &registry, backend.get());
  const uint32_t W = 64, H = 64;
  int inTex = backend->createTexture(W, H, 1);
  int outTex = backend->createTexture(W, H, 1);
  REQUIRE(inTex >= 0); REQUIRE(outTex >= 0);

  auto sketch = nlohmann::json::parse(R"JSON({
    "chain": [
      { "type": "module", "module_type": "control.nanolooper", "instance_key": "lp" }
    ],
    "instances": {
      "lp": { "module_type": "control.nanolooper",
              "state": { "send_to_rail": true, "strict_deadline": 120, "trigger_1": 0 } }
    }
  })JSON");

  trigger_bus::resetForTest();
  REQUIRE(executor.execute(sketch, inTex, outTex, (int)W, (int)H, 1.0 / 60.0, true) >= 0);
  backend->submit();
  trigger_bus::drain("test");

  sketch["instances"]["lp"]["state"]["trigger_1"] = 1;
  REQUIRE(executor.execute(sketch, inTex, outTex, (int)W, (int)H, 1.0 / 60.0, true) >= 0);
  backend->submit();

  auto events = trigger_bus::drain("test");
  bool saw_strict = false;
  for (const auto& e : events) {
    if (e.channel == 1 && e.on) {
      CHECK(e.strict == true);
      CHECK(e.deadline_ms == 120);
      saw_strict = true;
    }
  }
  CHECK(saw_strict);
}

// The host musical clock (WasmEffectBundles::setHostClock → the bundle FrameState
// the barrel now feeds from FFGL SetBeatInfo, and a headless ffgl_runner --bpm
// supplies) is what drives the looper. This test manually steps that transport to
// PROVE the looper loops: record a note (press → release), then advance the beat
// clock around the bar back onto the recorded window with NO key held and assert
// the gate re-fires on its own — the exact behavior that was dead when nothing
// fed the wasm effects' host.barPhase.
TEST_CASE("control.nanolooper replays a recorded note as the beat clock loops",
          "[effect_render][nanolooper][trigger_rail]") {
  auto backend = gpu::createBackend();
  if (!backend) SKIP("No GPU device available");

  sketch_executor::WasmEffectBundles bundles;
  REQUIRE(bundles.init());
  EffectRuntime rt(backend.get());
  sketch_executor::ModuleRegistry registry(&rt);
  REQUIRE(bundles.loadBundleFile(kCoreWasm, registry, backend.get(), nullptr) > 1);
  REQUIRE(bundles.loadBundleFile(kNanoWasm, registry, backend.get(), nullptr) > 1);

  sketch_executor::SketchExecutor executor(&rt, &registry, backend.get());
  const uint32_t W = 64, H = 64;
  int inTex = backend->createTexture(W, H, 1);
  int outTex = backend->createTexture(W, H, 1);
  REQUIRE(inTex >= 0); REQUIRE(outTex >= 0);

  auto sketch = nlohmann::json::parse(R"JSON({
    "chain": [
      { "type": "module", "module_type": "control.nanolooper", "instance_key": "lp" }
    ],
    "instances": {
      "lp": { "module_type": "control.nanolooper",
              "state": { "send_to_rail": true, "trigger_1": 0 } }
    }
  })JSON");

  trigger_bus::resetForTest();

  // Step the transport + one executor tick, then report whether a channel-1 ON
  // reached the rail this frame. dirty=true only when we edit trigger state.
  // NOTE: applyState (which delivers a trigger edit → on_param_change) runs
  // BEFORE the module's tick, so a press applied this frame sees the PREVIOUS
  // frame's phase. The sequence below sets the clock on a settle frame first,
  // then edits on the next — exactly how a live host feeds a continuous barPhase.
  auto step = [&](double barPhase, bool dirty) -> bool {
    bundles.setHostClock(0.0, 1.0 / 60.0, barPhase, 120.0, (int)W, (int)H);
    REQUIRE(executor.execute(sketch, inTex, outTex, (int)W, (int)H, 1.0 / 60.0, dirty) >= 0);
    backend->submit();
    bool on = false;
    for (const auto& e : trigger_bus::drain("test"))
      if (e.channel == 1 && e.on) on = true;
    return on;
  };

  // Baseline (clock at 0, nothing held) — arm edge detection + seq watermark.
  step(0.0, true);

  // Record a note across [2, 3): the press lands at phase 2, the release at 3.
  step(0.125, false);                        // settle clock at phase 2
  sketch["instances"]["lp"]["state"]["trigger_1"] = 1;
  CHECK(step(0.125, true));                  // press (sees phase 2) → live ON
  step(0.1875, false);                       // hold; advance clock to phase 3
  sketch["instances"]["lp"]["state"]["trigger_1"] = 0;
  step(0.25, true);                          // release (sees phase 3) → gate [2,3)

  // Advance the beat clock elsewhere in the bar with nothing held — silence.
  CHECK_FALSE(step(0.50, false));            // phase 8, outside [2,3)
  CHECK_FALSE(step(0.90, false));            // phase 14.4, outside

  // Loop back onto the recorded window (still nothing held): the sequencer must
  // replay the gate on its own. THIS is "it loops".
  CHECK(step(0.15625, false));               // phase 2.5, inside [2,3) → ON re-fires
}

// Two ABUTTING recorded notes (one ends exactly as the next begins) must
// RETRIGGER: playback coverage never lapses, but the boundary is a real new hit,
// so the looper emits an off THEN an on at the seam (not one sustained gate).
// This is what strict mode then holds a frame of "off" between.
TEST_CASE("control.nanolooper retriggers at an abutting-note boundary",
          "[effect_render][nanolooper][trigger_rail]") {
  auto backend = gpu::createBackend();
  if (!backend) SKIP("No GPU device available");

  sketch_executor::WasmEffectBundles bundles;
  REQUIRE(bundles.init());
  EffectRuntime rt(backend.get());
  sketch_executor::ModuleRegistry registry(&rt);
  REQUIRE(bundles.loadBundleFile(kCoreWasm, registry, backend.get(), nullptr) > 1);
  REQUIRE(bundles.loadBundleFile(kNanoWasm, registry, backend.get(), nullptr) > 1);

  sketch_executor::SketchExecutor executor(&rt, &registry, backend.get());
  const uint32_t W = 64, H = 64;
  int inTex = backend->createTexture(W, H, 1);
  int outTex = backend->createTexture(W, H, 1);
  REQUIRE(inTex >= 0); REQUIRE(outTex >= 0);

  auto sketch = nlohmann::json::parse(R"JSON({
    "chain": [
      { "type": "module", "module_type": "control.nanolooper", "instance_key": "lp" }
    ],
    "instances": {
      "lp": { "module_type": "control.nanolooper",
              "state": { "send_to_rail": true, "trigger_1": 0 } }
    }
  })JSON");

  trigger_bus::resetForTest();

  // step returns (sawOn, sawOff) for channel 1 this frame.
  auto step = [&](double barPhase, bool dirty) -> std::pair<bool, bool> {
    bundles.setHostClock(0.0, 1.0 / 60.0, barPhase, 120.0, (int)W, (int)H);
    REQUIRE(executor.execute(sketch, inTex, outTex, (int)W, (int)H, 1.0 / 60.0, dirty) >= 0);
    backend->submit();
    bool on = false, off = false;
    for (const auto& e : trigger_bus::drain("test")) {
      if (e.channel != 1) continue;
      if (e.on) on = true; else off = true;
    }
    return {on, off};
  };

  step(0.0, true);  // baseline (phase 0)

  // Record note A over [2,4): press sees phase 2, release sees phase 4.
  step(0.125, false);                            // settle phase 2
  sketch["instances"]["lp"]["state"]["trigger_1"] = 1;
  step(0.125, true);                             // press → phase 2
  step(0.25, false);                             // hold to phase 4
  sketch["instances"]["lp"]["state"]["trigger_1"] = 0;
  step(0.3125, true);                            // release → note A [2,4)

  // Record note B over [4,6): press sees phase 4, release sees phase 6.
  step(0.25, false);                             // settle phase 4
  sketch["instances"]["lp"]["state"]["trigger_1"] = 1;
  step(0.25, true);                              // press → phase 4
  step(0.375, false);                            // hold to phase 6
  sketch["instances"]["lp"]["state"]["trigger_1"] = 0;
  step(0.4375, true);                            // release → note B [4,6)

  // Play through the coverage with NOTHING held. Enter note A → ON.
  step(0.0, false);                              // phase 0, silence, reset gate
  { auto r = step(0.15625, false); CHECK(r.first); }    // phase 2.5 inside A → ON
  { auto r = step(0.21875, false); CHECK_FALSE(r.first); CHECK_FALSE(r.second); } // 3.5, still on

  // Cross the A→B boundary at phase 4: coverage is continuous, but the retrigger
  // must emit BOTH an off and an on this frame.
  auto boundary = step(0.28125, false);          // phase 4.5, crossed onset at 4
  CHECK(boundary.first);                          // ON (note B)
  CHECK(boundary.second);                         // OFF (note A ended) — the retrigger
}

// The looper now DRAWS its debug overlay through the in-effect overlay toolbox
// (overlay.h): solid-quad GPU rects + host-text labels composited onto tex_out
// over tex_in. This is only possible because it declares a tex_out texture
// field — otherwise the executor classifies it a modulation-source passthrough
// and never calls render(). This test proves both halves:
//   show_overlay=true  → the output differs from the passthrough input (the
//                        overlay's panel darkens the top and the playhead paints
//                        a near-white column), while a pixel below the panel
//                        still equals the input (passthrough preserved).
//   show_overlay=false → the output equals the input (clean passthrough — the
//                        effect owns the output texture now and must forward it).
TEST_CASE("control.nanolooper composites its overlay over the passthrough input",
          "[effect_render][nanolooper]") {
  auto backend = gpu::createBackend();
  if (!backend) SKIP("No GPU device available");

  sketch_executor::WasmEffectBundles bundles;
  REQUIRE(bundles.init());
  EffectRuntime rt(backend.get());
  sketch_executor::ModuleRegistry registry(&rt);
  REQUIRE(bundles.loadBundleFile(kCoreWasm, registry, backend.get(), nullptr) > 1);
  REQUIRE(bundles.loadBundleFile(kNanoWasm, registry, backend.get(), nullptr) > 1);

  sketch_executor::SketchExecutor executor(&rt, &registry, backend.get());
  const uint32_t W = 480, H = 270;
  int inTex = backend->createTexture(W, H, 1);
  int outTex = backend->createTexture(W, H, 1);
  REQUIRE(inTex >= 0); REQUIRE(outTex >= 0);

  // Uniform mid-gray input so any overlay pixel is unambiguous.
  std::vector<uint8_t> gray(W * H * 4, 128);
  for (size_t i = 3; i < gray.size(); i += 4) gray[i] = 255;
  backend->writeTexture(inTex, W, H, gray.data(), (uint32_t)gray.size());

  auto sketchFor = [](bool overlay) {
    auto s = nlohmann::json::parse(R"JSON({
      "chain": [ { "type": "module", "module_type": "control.nanolooper", "instance_key": "lp" } ],
      "instances": { "lp": { "module_type": "control.nanolooper", "state": {} } }
    })JSON");
    s["instances"]["lp"]["state"]["show_overlay"] = overlay;
    return s;
  };

  auto pixAt = [&](std::vector<uint8_t>& px, uint32_t x, uint32_t y) {
    size_t i = ((size_t)y * W + x) * 4;
    return std::array<int,4>{ px[i], px[i+1], px[i+2], px[i+3] };
  };

  // --- Overlay ON: panel darkens the top band; playhead paints a bright column.
  bundles.setHostClock(0.0, 1.0 / 60.0, 0.5, 120.0, (int)W, (int)H);  // playhead mid-bar
  {
    auto sketch = sketchFor(true);
    int32_t out = executor.execute(sketch, inTex, outTex, (int)W, (int)H, 1.0 / 60.0, true);
    backend->submit();
    auto px = backend->readbackTexture(out, W, H);

    // Scan the lane band for the darkest and brightest luma we can find.
    int lo = 255, hi = 0;
    for (uint32_t y = 20; y < 70; ++y)
      for (uint32_t x = 4; x < W - 4; ++x) {
        auto p = pixAt(px, x, y);
        int l = (p[0] + p[1] + p[2]) / 3;
        if (l < lo) lo = l;
        if (l > hi) hi = l;
      }
    INFO("overlay band luma range [" << lo << ", " << hi << "] (input was 128)");
    CHECK(lo < 100);   // the semi-transparent panel darkens the input
    // Labels paint clearly-bright pixels. Threshold note: at this quarter-res
    // canvas the overlay labels are a few px tall, so the MSDF glyph shader's
    // screen-px-range (spr = px_range * screen_h / tile_h) is well below 1 and
    // the softened peaks top out ~147 — the same value the web path produces.
    // The old 170 passed only via a first-frame accident (the text GPU cache
    // reset zeroed atlas dims AFTER the atlas was built, so frame 1 hit the
    // shader's spr=1.0 fallback and rendered over-sharp, unlike every frame
    // after it).
    CHECK(hi > 140);

    // A pixel well below the panel is untouched → passthrough preserved.
    auto below = pixAt(px, W / 2, H - 6);
    INFO("below-panel rgba = " << below[0] << "," << below[1] << "," << below[2]);
    CHECK(std::abs(below[0] - 128) <= 6);
    CHECK(std::abs(below[1] - 128) <= 6);
    CHECK(std::abs(below[2] - 128) <= 6);
  }

  // --- Overlay OFF: clean passthrough (the effect still owns tex_out).
  {
    auto sketch = sketchFor(false);
    int32_t out = executor.execute(sketch, inTex, outTex, (int)W, (int)H, 1.0 / 60.0, true);
    backend->submit();
    auto px = backend->readbackTexture(out, W, H);
    double m = mean_rgb(px);
    INFO("overlay-off mean = " << m << " (input 128)");
    CHECK(std::abs(m - 128.0) <= 3.0);
    auto top = pixAt(px, W / 2, 40);   // where the panel WOULD be
    CHECK(std::abs(top[0] - 128) <= 6);
  }
}
#endif  // NANO_WASM_PATH

// Second Arena repro axis: the user isolated a MULTI-CARD PASTE that crashed
// seconds later — a 5-filter chain including two LEGACY-bundle effects
// (color.legacy.bicolor_grad, filter.legacy.subtle_blur). legacy.wasm is the
// ~14MB module, which is exactly where a NULL-membase dereference at the
// crash's stable in-bounds offset (~0xd432c0, 13.9MB) would land. Resolume
// additionally renders one instance at ALTERNATING sizes (composition output
// vs preview panel) — so this case drives the pasted chain over a non-black
// input with per-frame 1920×1080 ↔ 1754×987 viewport thrash and periodic
// dirty rebuilds (the paste itself is a regenerate).
#ifdef LEGACY_WASM_PATH
TEST_CASE("arena repro: pasted legacy filter chain survives viewport thrash",
          "[effect_render][arena_repro]") {
  auto backend = gpu::createBackend();
  if (!backend) SKIP("No GPU device available");

  sketch_executor::WasmEffectBundles bundles;
  REQUIRE(bundles.init());
  EffectRuntime rt(backend.get());
  sketch_executor::ModuleRegistry registry(&rt);
  REQUIRE(bundles.loadBundleFile(kCoreWasm, registry, backend.get(), nullptr) > 1);
  REQUIRE(bundles.loadBundleFile(kLegacyWasm, registry, backend.get(), nullptr) > 1);

  sketch_executor::SketchExecutor ex(&rt, &registry, backend.get());

  const uint32_t W1 = 1920, H1 = 1080, W2 = 1754, H2 = 987;
  const int RGBA8 = 1;
  int in1 = backend->createTexture(W1, H1, RGBA8);
  int out1 = backend->createTexture(W1, H1, RGBA8);
  int in2 = backend->createTexture(W2, H2, RGBA8);
  int out2 = backend->createTexture(W2, H2, RGBA8);
  REQUIRE(in1 >= 0); REQUIRE(out1 >= 0); REQUIRE(in2 >= 0); REQUIRE(out2 >= 0);

  // Non-black, non-uniform input (the filters analyze content — bicolor_grad
  // isolates a mid-band, subtle_blur displaces by hue).
  auto fillGradient = [&](int tex, uint32_t w, uint32_t h) {
    std::vector<uint8_t> px((size_t)w * h * 4);
    for (uint32_t y = 0; y < h; ++y)
      for (uint32_t x = 0; x < w; ++x) {
        size_t i = ((size_t)y * w + x) * 4;
        px[i] = (uint8_t)(255 * x / w);
        px[i + 1] = (uint8_t)(255 * y / h);
        px[i + 2] = (uint8_t)(255 - (255 * x / w));
        px[i + 3] = 255;
      }
    backend->writeTexture(tex, w, h, px.data(), (uint32_t)px.size());
  };
  fillGradient(in1, W1, H1);
  fillGradient(in2, W2, H2);

  // The exact pasted payload (help text dropped; __opacity__ kept — a partial-
  // opacity FIRST stage rides the executor's reserved-key path).
  auto sketch = nlohmann::json::parse(R"JSON({
    "chain": [
      { "type": "module", "module_type": "color.legacy.bicolor_grad",  "instance_key": "bg"  },
      { "type": "module", "module_type": "color.temperature",          "instance_key": "ct"  },
      { "type": "module", "module_type": "color.hsl",                  "instance_key": "hsl" },
      { "type": "module", "module_type": "filter.vignette",            "instance_key": "vig" },
      { "type": "module", "module_type": "filter.legacy.subtle_blur",  "instance_key": "sb"  }
    ],
    "instances": {
      "bg":  { "module_type": "color.legacy.bicolor_grad", "state": {
        "__opacity__": 0.49, "blend": 1, "color_sat": 0.05, "isolation": 0.3,
        "midband": 0.2, "mode": 0, "neutral": [0.05, 0.05, 0.06],
        "neutral_mix": 0.25, "reverse": false, "scale": 1, "smoothing": 0.85 } },
      "ct":  { "module_type": "color.temperature", "state": { "temperature": 1 } },
      "hsl": { "module_type": "color.hsl", "state": {
        "hue_shift": -0.18, "lightness": -0.35, "saturation": 1 } },
      "vig": { "module_type": "filter.vignette", "state": {
        "amount": -0.52, "center": [0, 0], "radius": 0.6, "shape": 0,
        "softness": 0.4, "squash": 0 } },
      "sb":  { "module_type": "filter.legacy.subtle_blur", "state": {
        "amount": 0.15, "blur": 0.09, "hue": 0.22, "movement": 1, "quality": 0.3 } }
    }
  })JSON");

  for (int f = 0; f < 1200; ++f) {
    const bool dirty = (f % 120) == 0;   // periodic re-paste / plan rebuild
    const bool small = (f % 2) == 1;     // per-frame Resolume preview-size thrash
    int32_t out = small
      ? ex.execute(sketch, in2, out2, (int)W2, (int)H2, 1.0 / 60.0, dirty)
      : ex.execute(sketch, in1, out1, (int)W1, (int)H1, 1.0 / 60.0, dirty);
    backend->submit();
    REQUIRE(out > 0);
  }

  auto px = backend->readbackTexture(out1, W1, H1);
  CHECK(px.size() == (size_t)W1 * H1 * 4);
}


// REGRESSION: lut_collection bakes its 13 preset cubes on the first render —
// inside the executor's whole-frame command batch, where effect-called
// gpu::Device::submit() is a no-op and gpu_write_buffer is an immediate CPU
// write. The original bake reused ONE staging buffer with a submit() between
// presets, so every fill dispatch read the LAST preset's bytes: all 13 cubes
// held "Hue Rotate 270" and every preset rendered identically (web was fine —
// its submit really flushes). Fixed with one staging buffer per preset.
// Asserted semantically: "Mono" must be grayscale (Hue270 is wildly colored),
// and distinct presets must render distinct pixels.
TEST_CASE("lut_collection presets bake distinct cubes in one batched frame",
          "[effect_render][lut]") {
  auto backend = gpu::createBackend();
  if (!backend) SKIP("No GPU device available");
  sketch_executor::WasmEffectBundles bundles;
  REQUIRE(bundles.init());
  EffectRuntime rt(backend.get());
  sketch_executor::ModuleRegistry registry(&rt);
  REQUIRE(bundles.loadBundleFile(kLegacyWasm, registry, backend.get(), nullptr) > 1);
  sketch_executor::SketchExecutor executor(&rt, &registry, backend.get());

  const uint32_t W = 64, H = 64; const int RGBA8 = 1;
  int inTex = backend->createTexture(W, H, RGBA8);
  int outTex = backend->createTexture(W, H, RGBA8);
  REQUIRE(inTex >= 0); REQUIRE(outTex >= 0);

  // Gradient input (r=x, g=y, b=0) — sweeps a plane of the LUT cube.
  std::vector<uint8_t> px((size_t)W * H * 4);
  for (uint32_t y = 0; y < H; ++y)
    for (uint32_t x = 0; x < W; ++x) {
      size_t i = ((size_t)y * W + x) * 4;
      px[i] = (uint8_t)(255 * x / (W - 1));
      px[i + 1] = (uint8_t)(255 * y / (H - 1));
      px[i + 2] = 0;
      px[i + 3] = 255;
    }
  backend->writeTexture(inTex, W, H, px.data(), (uint32_t)px.size());

  auto renderPreset = [&](int lut) {
    char buf[512];
    std::snprintf(buf, sizeof(buf), R"JSON({
      "chain": [ { "module_type": "color.legacy.lut_collection", "instance_key": "lut" } ],
      "instances": {
        "lut": { "module_type": "color.legacy.lut_collection",
                 "state": { "lut": %d, "amount": 1.0, "pregain": 0.0 } }
      }
    })JSON", lut);
    auto sk = nlohmann::json::parse(buf);
    int32_t out = executor.execute(sk, inTex, outTex, (int)W, (int)H, 1.0/60.0, true);
    backend->submit();
    REQUIRE(out > 0);
    return backend->readbackTexture(out, W, H);
  };

  // Frame 1 selects "Mono" (preset 6) so the bake AND a bake-dependent apply
  // ride the same batched command buffer.
  auto mono = renderPreset(6);
  REQUIRE(mono.size() == px.size());
  int maxChanDelta = 0;
  for (size_t i = 0; i < mono.size(); i += 4) {
    int r = mono[i], g = mono[i + 1], b = mono[i + 2];
    maxChanDelta = std::max({maxChanDelta, std::abs(r - g), std::abs(r - b)});
  }
  INFO("Mono max |r-g| / |r-b| over all pixels: " << maxChanDelta);
  CHECK(maxChanDelta <= 8);   // bug rendered Hue Rotate 270 here (delta ~230)

  auto process = renderPreset(0);
  auto hue270  = renderPreset(12);
  auto meanAbsDiff = [](const std::vector<uint8_t>& a, const std::vector<uint8_t>& b) {
    long s = 0, n = 0;
    for (size_t i = 0; i < a.size(); i += 4) {   // rgb only
      s += std::abs((int)a[i] - (int)b[i]) + std::abs((int)a[i+1] - (int)b[i+1])
         + std::abs((int)a[i+2] - (int)b[i+2]);
      n += 3;
    }
    return n ? (double)s / n : 0.0;
  };
  INFO("meanAbsDiff mono/process " << meanAbsDiff(mono, process)
       << " mono/hue270 " << meanAbsDiff(mono, hue270)
       << " process/hue270 " << meanAbsDiff(process, hue270));
  CHECK(meanAbsDiff(mono, process) > 15.0);    // bug: all three identical (0)
  CHECK(meanAbsDiff(mono, hue270) > 15.0);
  CHECK(meanAbsDiff(process, hue270) > 15.0);
}

// The motion rail, end to end, through the BARREL's path (SketchExecutor over a
// raw sketch — no editor-side augmentation). double_chamber publishes
// render_outputs/motion only when a downstream sink reads it; motion.blur is
// that sink, and the connection is IMPLICIT (wires:[] → sketch_augment
// synthesises the rail). The two runs are byte-identical apart from the blur
// strength, so a difference can only come from real velocity arriving over the
// rail. The web engine harness covers the same chain via executor.wasm; this is
// the native twin, so a native-only rail regression can't hide.
TEST_CASE("motion rail: double_chamber drives motion.blur natively",
          "[effect_render][motion_rail]") {
  auto backend = gpu::createBackend();
  if (!backend) SKIP("No GPU device available");

  sketch_executor::WasmEffectBundles bundles;
  REQUIRE(bundles.init());
  EffectRuntime rt(backend.get());
  sketch_executor::ModuleRegistry registry(&rt);
  REQUIRE(bundles.loadBundleFile(kCoreWasm, registry, backend.get(), nullptr) > 1);
  REQUIRE(bundles.loadBundleFile(kLegacyWasm, registry, backend.get(), nullptr) > 1);

  const uint32_t W = 128, H = 128;
  const int RGBA8 = 1;

  auto renderChain = [&](double strength) {
    sketch_executor::SketchExecutor ex(&rt, &registry, backend.get());
    int in = backend->createTexture(W, H, RGBA8);
    int out = backend->createTexture(W, H, RGBA8);
    std::vector<uint8_t> black((size_t)W * H * 4, 0);
    for (size_t i = 3; i < black.size(); i += 4) black[i] = 255;
    backend->writeTexture(in, W, H, black.data(), (uint32_t)black.size());

    auto sketch = nlohmann::json::parse(R"JSON({
      "wires": [],
      "chain": [
        { "type": "module", "module_type": "source.legacy.double_chamber", "instance_key": "dc" },
        { "type": "module", "module_type": "motion.blur",                  "instance_key": "mb" }
      ],
      "instances": {
        "dc": { "module_type": "source.legacy.double_chamber", "state": {
          "p_count": 6000, "p_point_size": 1.0, "p_opacity": 1.0, "exposure": 2.0,
          "color_contrib": 0.0, "field_speed": 0.6, "motion_rate": 2.0, "jitter": 0.0,
          "to_big": 0.5, "big_count": 4, "l_count": 8, "l_opacity": 1.0,
          "motion_line_speed": 0.6, "bridger_count": 0 } },
        "mb": { "module_type": "motion.blur", "state": {
          "strength": 0.0, "samples": 16, "quality": 1 } }
      }
    })JSON");
    sketch["instances"]["mb"]["state"]["strength"] = strength;

    for (int f = 0; f < 24; ++f) {
      REQUIRE(ex.execute(sketch, in, out, (int)W, (int)H, 1.0 / 60.0, /*dirty=*/f == 0) > 0);
      backend->submit();
    }
    return backend->readbackTexture(out, W, H);
  };

  auto sharp   = renderChain(0.0);    // pass-through
  auto blurred = renderChain(32.0);   // smeared along the motion vectors
  REQUIRE(sharp.size() == (size_t)W * H * 4);
  REQUIRE(blurred.size() == (size_t)W * H * 4);

  // The cloud must actually be on screen, else "frames differ" proves nothing.
  int lit = 0;
  for (size_t i = 0; i < sharp.size(); i += 4)
    if ((int)sharp[i] + sharp[i + 1] + sharp[i + 2] > 24) lit++;
  INFO("lit pixels " << lit);
  CHECK(lit > 100);

  long diff = 0;
  for (size_t i = 0; i < sharp.size(); i += 4)
    diff += std::abs((int)sharp[i] - (int)blurred[i])
          + std::abs((int)sharp[i + 1] - (int)blurred[i + 1])
          + std::abs((int)sharp[i + 2] - (int)blurred[i + 2]);
  INFO("total rgb delta blurred vs sharp " << diff);
  CHECK(diff > 1000);   // no rail → both runs identical → diff == 0
}
#endif  // LEGACY_WASM_PATH

#if defined(NANO_WASM_PATH) && defined(CORE_WASM_PATH)
// sweep_chamber (double_chamber's successor, nano bundle) native smoke: all
// of its PSOs must build on Metal (storage-texture formats, threadgroup
// carry-through), the sim must run on a real luma-gradient input with the
// sweep mid-band (image coupling + tracers + spawn-on-line live), and the
// motion rail must drive motion.blur exactly like the double_chamber twin
// above. Two byte-identical runs apart from blur strength → a difference can
// only be real velocity over the rail.
TEST_CASE("sweep_chamber renders and drives motion.blur natively",
          "[effect_render][motion_rail][sweep_chamber]") {
  auto backend = gpu::createBackend();
  if (!backend) SKIP("No GPU device available");

  sketch_executor::WasmEffectBundles bundles;
  REQUIRE(bundles.init());
  EffectRuntime rt(backend.get());
  sketch_executor::ModuleRegistry registry(&rt);
  REQUIRE(bundles.loadBundleFile(kCoreWasm, registry, backend.get(), nullptr) > 1);
  REQUIRE(bundles.loadBundleFile(kNanoWasm, registry, backend.get(), nullptr) > 1);

  const uint32_t W = 128, H = 128;
  const int RGBA8 = 1;

  auto renderChain = [&](double strength) {
    sketch_executor::SketchExecutor ex(&rt, &registry, backend.get());
    int in = backend->createTexture(W, H, RGBA8);
    int out = backend->createTexture(W, H, RGBA8);
    // Horizontal luma ramp: every brightness present, so the mid-sweep
    // window captures a clean vertical band for the tracers to grip.
    std::vector<uint8_t> ramp((size_t)W * H * 4, 0);
    for (uint32_t y = 0; y < H; ++y) {
      for (uint32_t x = 0; x < W; ++x) {
        size_t i = ((size_t)y * W + x) * 4;
        uint8_t v = (uint8_t)((x * 255u) / (W - 1u));
        ramp[i] = ramp[i + 1] = ramp[i + 2] = v;
        ramp[i + 3] = 255;
      }
    }
    backend->writeTexture(in, W, H, ramp.data(), (uint32_t)ramp.size());

    auto sketch = nlohmann::json::parse(R"JSON({
      "wires": [],
      "chain": [
        { "type": "module", "module_type": "source.particles.sweep_chamber", "instance_key": "sc" },
        { "type": "module", "module_type": "motion.blur",                    "instance_key": "mb" }
      ],
      "instances": {
        "sc": { "module_type": "source.particles.sweep_chamber", "state": {
          "count": 6000, "shape_kind": 1, "size": 0.8, "opacity": 1.0,
          "exposure": 2.0, "color_blend": 1.0, "speed": 3.0, "noise_speed": 1.0,
          "sweep_center": 0.5, "to_image": 2.0, "input_alpha": 0.0,
          "l_count": 8, "l_opacity": 1.0, "motion_line_speed": 0.6 } },
        "mb": { "module_type": "motion.blur", "state": {
          "strength": 0.0, "samples": 16, "quality": 1 } }
      }
    })JSON");
    sketch["instances"]["mb"]["state"]["strength"] = strength;

    for (int f = 0; f < 24; ++f) {
      REQUIRE(ex.execute(sketch, in, out, (int)W, (int)H, 1.0 / 60.0, /*dirty=*/f == 0) > 0);
      backend->submit();
    }
    return backend->readbackTexture(out, W, H);
  };

  auto sharp   = renderChain(0.0);
  auto blurred = renderChain(32.0);
  REQUIRE(sharp.size() == (size_t)W * H * 4);
  REQUIRE(blurred.size() == (size_t)W * H * 4);

  int lit = 0;
  for (size_t i = 0; i < sharp.size(); i += 4)
    if ((int)sharp[i] + sharp[i + 1] + sharp[i + 2] > 24) lit++;
  INFO("lit pixels " << lit);
  CHECK(lit > 100);

  long diff = 0;
  for (size_t i = 0; i < sharp.size(); i += 4)
    diff += std::abs((int)sharp[i] - (int)blurred[i])
          + std::abs((int)sharp[i + 1] - (int)blurred[i + 1])
          + std::abs((int)sharp[i + 2] - (int)blurred[i + 2]);
  INFO("total rgb delta blurred vs sharp " << diff);
  CHECK(diff > 1000);
}
#endif  // NANO_WASM_PATH && CORE_WASM_PATH

// source.mesh.three_planes' MASK input, on Metal.
//
// This is the leg the web suites cannot reach. The per-effect GPU suite runs on
// both backends but has no way to wire a second texture in, so it only ever
// exercises the mask with nothing bound; the engine e2e wires it properly but
// is WebGPU only. What is untested between them is precisely the thing
// CLAUDE.md warns about — a binding index that lines up on one backend and not
// on the other silently reads the wrong texture, and the symptom is a black
// frame nobody attributes to the binding table.
TEST_CASE("a wired mask cuts the neon on Metal too", "[effect_render]") {
  auto backend = gpu::createBackend();
  if (!backend) {
    SKIP("No GPU device available");
  }

  sketch_executor::WasmEffectBundles bundles;
  REQUIRE(bundles.init());
  EffectRuntime rt(backend.get());
  sketch_executor::ModuleRegistry registry(&rt);
  REQUIRE(bundles.loadBundleFile(kCoreWasm, registry, backend.get(), nullptr) > 1);
  REQUIRE(bundles.loadBundleFile(kLightsWasm, registry, backend.get(), nullptr) > 0);

  sketch_executor::SketchExecutor executor(&rt, &registry, backend.get());

  const uint32_t W = 64, H = 64;
  const int RGBA8 = 1;
  int inTex = backend->createTexture(W, H, RGBA8);
  int outTex = backend->createTexture(W, H, RGBA8);
  REQUIRE(inTex >= 0);
  REQUIRE(outTex >= 0);
  std::vector<uint8_t> blk(W * H * 4, 0);
  for (size_t i = 3; i < blk.size(); i += 4) blk[i] = 255;
  backend->writeTexture(inTex, W, H, blk.data(), (uint32_t)blk.size());

  // A white solid feeds the stack's own input AND its mask, so the whole frame
  // is masked at full weight and the reading is a single number rather than a
  // shape to hunt for.
  auto sketchFor = [](const char* extra, bool wire) {
    std::string w = wire
        ? R"JSON({ "id": "wm", "src": { "instanceKey": "msk", "field": "tex_out" },
                   "dest": { "instanceKey": "tp", "field": "mask_in" } })JSON"
        : "";
    return nlohmann::json::parse(std::string(R"JSON({
      "chain": [
        { "type": "module", "module_type": "source.solid_color", "instance_key": "msk",
          "params": { "color": [1.0, 1.0, 1.0] } },
        { "type": "module", "module_type": "source.mesh.three_planes", "instance_key": "tp",
          "params": { "grain": 0.0, "scanline": 0.0, "chroma_bleed": 0.0,
                      "input_opacity": 1.0, "plane1_emission": 1.0,
                      "plane2_emission": 1.0, "plane3_emission": 1.0)JSON")
      + extra + R"JSON( } }
      ],
      "wires": [)JSON" + w + R"JSON(]
    })JSON");
  };

  auto meanOf = [&](const nlohmann::json& sketch) {
    int32_t out = executor.execute(sketch, inTex, outTex, (int)W, (int)H, 1.0 / 60.0,
                                   /*sketchDirty=*/true);
    backend->submit();
    auto px = backend->readbackTexture(out, W, H);
    double sum = 0;
    int n = 0;
    for (size_t i = 0; i + 3 < px.size(); i += 4) {
      sum += (px[i] + px[i + 1] + px[i + 2]) / 3.0;
      ++n;
    }
    return sum / n;
  };

  const double unwired = meanOf(sketchFor("", false));
  const double cut_all = meanOf(sketchFor(", \"mask_halo\": 1.0", true));
  const double cut_mid = meanOf(sketchFor(", \"mask_halo\": 0.6", true));
  const double cut_none = meanOf(sketchFor(", \"mask_halo\": 0.0", true));
  const double off = meanOf(sketchFor(", \"mask_strength\": 0.0", true));
  INFO("unwired " << unwired << " all " << cut_all << " mid " << cut_mid
                  << " none " << cut_none << " off " << off);

  // A white input under a lit stack is a bright frame.
  CHECK(unwired > 200.0);
  // Masked at full weight with the halo taken too: nothing is left.
  CHECK(cut_all < 4.0);
  // Let the halo through and the glow survives the cut — that spill is the
  // whole difference between a mask in the scene and one on the glass.
  CHECK(cut_none > cut_mid + 10.0);
  CHECK(cut_mid > cut_all + 10.0);
  // ...but the body and the base are gone whatever the halo does.
  CHECK(cut_none < unwired * 0.6);
  // And a mask nobody asked for changes nothing.
  CHECK(off == Catch::Approx(unwired).margin(1.0));
}

// The LED map, on Metal. It is a SECOND compute pass with its own binding
// layout — a storage texture at 0 and a uniform at 1, with no sampled texture
// in front of them — so the slot numbering is the thing at risk here, and a
// mis-bind shows up as a black map rather than as an error. The web side of
// the same pass is web/test/led_bars.test.ts; the arithmetic behind it is
// native/tests/test_led_bars.cpp.
//
// Read the way the web reads a secondary output: wire it into a sidechannel
// send override and put the receive after it, so the sketch output IS the map.
TEST_CASE("the LED map is the same map on Metal", "[effect_render]") {
  auto backend = gpu::createBackend();
  if (!backend) {
    SKIP("No GPU device available");
  }

  sketch_executor::WasmEffectBundles bundles;
  REQUIRE(bundles.init());
  EffectRuntime rt(backend.get());
  sketch_executor::ModuleRegistry registry(&rt);
  REQUIRE(bundles.loadBundleFile(kCoreWasm, registry, backend.get(), nullptr) > 1);
  REQUIRE(bundles.loadBundleFile(kLightsWasm, registry, backend.get(), nullptr) > 0);

  sketch_executor::SketchExecutor executor(&rt, &registry, backend.get());

  // 10 segments over 200 rows is 20 rows each, so every boundary is a whole
  // pixel and a sample can sit either side of one.
  const uint32_t W = 64, H = 200;
  const int RGBA8 = 1;
  int inTex = backend->createTexture(W, H, RGBA8);
  int outTex = backend->createTexture(W, H, RGBA8);
  REQUIRE(inTex >= 0);
  REQUIRE(outTex >= 0);
  std::vector<uint8_t> blk(W * H * 4, 0);
  for (size_t i = 3; i < blk.size(); i += 4) blk[i] = 255;
  backend->writeTexture(inTex, W, H, blk.data(), (uint32_t)blk.size());

  // A primary per floor, so a segment showing the wrong one is a different
  // colour rather than a near miss.
  auto sketch = nlohmann::json::parse(R"JSON({
    "chain": [
      { "type": "module", "module_type": "source.solid_color", "instance_key": "bg",
        "params": { "color": [0.0, 0.0, 0.0] } },
      { "type": "module", "module_type": "source.mesh.three_planes", "instance_key": "tp",
        "params": { "grain": 0.0, "scanline": 0.0, "chroma_bleed": 0.0,
                    "plane1_color": [1.0, 0.0, 0.0],
                    "plane2_color": [0.0, 1.0, 0.0],
                    "plane3_color": [0.0, 0.0, 1.0],
                    "plane1_emission": 1.0, "plane2_emission": 1.0,
                    "plane3_emission": 1.0 } },
      { "type": "module", "module_type": "util.sidechannel_out", "instance_key": "send",
        "params": { "channel": 3 } },
      { "type": "module", "module_type": "util.sidechannel_in", "instance_key": "recv",
        "params": { "channel": 3 } }
    ],
    "wires": [
      { "id": "lw", "src": { "instanceKey": "tp", "field": "led_out" },
        "dest": { "instanceKey": "send", "field": "send_in" } }
    ]
  })JSON");

  int32_t out = executor.execute(sketch, inTex, outTex, (int)W, (int)H, 1.0 / 60.0,
                                 /*sketchDirty=*/true);
  backend->submit();
  auto px = backend->readbackTexture(out, W, H);
  REQUIRE(px.size() >= (size_t)W * H * 4);

  auto at = [&](uint32_t x, uint32_t y) {
    const size_t i = ((size_t)y * W + x) * 4;
    return std::array<int, 3>{px[i], px[i + 1], px[i + 2]};
  };
  // The image row at the middle of segment `seg`, counting from the BOTTOM.
  auto rowOf = [&](int seg) { return (uint32_t)(H - (uint32_t)(seg * 20) - 10); };

  for (int seg = 0; seg <= 2; ++seg) {      // bottom floor, three segments
    INFO("segment " << seg);
    auto c = at(8, rowOf(seg));
    CHECK(c[0] > 240); CHECK(c[1] < 12); CHECK(c[2] < 12);
  }
  for (int seg = 3; seg <= 5; ++seg) {      // middle floor, three segments
    INFO("segment " << seg);
    auto c = at(8, rowOf(seg));
    CHECK(c[0] < 12); CHECK(c[1] > 240); CHECK(c[2] < 12);
  }
  for (int seg = 6; seg <= 9; ++seg) {      // top floor, FOUR segments
    INFO("segment " << seg);
    auto c = at(8, rowOf(seg));
    CHECK(c[0] < 12); CHECK(c[1] < 12); CHECK(c[2] > 240);
  }

  // Every bar shows the same tower.
  for (uint32_t bar = 1; bar < 4; ++bar) {
    INFO("bar " << bar);
    auto c = at(bar * 16 + 8, rowOf(0));
    CHECK(c[0] > 240); CHECK(c[1] < 12); CHECK(c[2] < 12);
  }

  // And the blocks are FLAT: the boundary under the top floor sits at row 80,
  // with no ramp across it. A consumer sampling anywhere inside a segment has
  // to come away with that segment's exact colour.
  auto above = at(8, 78);
  auto below = at(8, 82);
  CHECK(above[2] > 240);
  CHECK(below[1] > 240);
  CHECK(below[2] < 12);
}


// three_planes' impact light, on Metal. A third compute pass on that card, and
// the one whose whole output is a falloff — so a translation that quietly lost
// the grazing term or the per-ring gap would still produce a picture, just a
// wrong one. The shape is pinned in web/test/three_planes_walls.test.ts; what
// is here is that Metal agrees about WHERE the light lands and in what colour.
TEST_CASE("the impact light lands on the same walls on Metal", "[effect_render]") {
  auto backend = gpu::createBackend();
  if (!backend) {
    SKIP("No GPU device available");
  }

  sketch_executor::WasmEffectBundles bundles;
  REQUIRE(bundles.init());
  EffectRuntime rt(backend.get());
  sketch_executor::ModuleRegistry registry(&rt);
  REQUIRE(bundles.loadBundleFile(kCoreWasm, registry, backend.get(), nullptr) > 1);
  REQUIRE(bundles.loadBundleFile(kLightsWasm, registry, backend.get(), nullptr) > 0);

  sketch_executor::SketchExecutor executor(&rt, &registry, backend.get());

  const uint32_t W = 240, H = 270;
  const int RGBA8 = 1;
  int inTex = backend->createTexture(W, H, RGBA8);
  int outTex = backend->createTexture(W, H, RGBA8);
  REQUIRE(inTex >= 0);
  REQUIRE(outTex >= 0);
  std::vector<uint8_t> blk(W * H * 4, 0);
  for (size_t i = 3; i < blk.size(); i += 4) blk[i] = 255;
  backend->writeTexture(inTex, W, H, blk.data(), (uint32_t)blk.size());

  // A primary per floor, and no glimmer: a glint is a particle born from knob
  // motion, and one frame of a static patch must not depend on having any.
  auto sketch = nlohmann::json::parse(R"JSON({
    "chain": [
      { "type": "module", "module_type": "source.solid_color", "instance_key": "bg",
        "params": { "color": [0.0, 0.0, 0.0] } },
      { "type": "module", "module_type": "source.mesh.three_planes", "instance_key": "tp",
        "params": { "grain": 0.0, "scanline": 0.0, "chroma_bleed": 0.0,
                    "glimmer_gain": 0.0, "glimmer_chaos": 0.0,
                    "orbit_azimuth": 0.0, "elevation": 0.0,
                    "plane1_color": [1.0, 0.0, 0.0],
                    "plane2_color": [0.0, 1.0, 0.0],
                    "plane3_color": [0.0, 0.0, 1.0],
                    "plane1_emission": 1.0, "plane2_emission": 1.0,
                    "plane3_emission": 1.0 } },
      { "type": "module", "module_type": "util.sidechannel_out", "instance_key": "send",
        "params": { "channel": 3 } },
      { "type": "module", "module_type": "util.sidechannel_in", "instance_key": "recv",
        "params": { "channel": 3 } }
    ],
    "wires": [
      { "id": "ww", "src": { "instanceKey": "tp", "field": "left_out" },
        "dest": { "instanceKey": "send", "field": "send_in" } }
    ]
  })JSON");

  int32_t out = executor.execute(sketch, inTex, outTex, (int)W, (int)H, 1.0 / 60.0,
                                 /*sketchDirty=*/true);
  backend->submit();
  auto px = backend->readbackTexture(out, W, H);
  REQUIRE(px.size() >= (size_t)W * H * 4);

  auto at = [&](uint32_t x, uint32_t y) {
    const size_t i = ((size_t)y * W + x) * 4;
    return std::array<int, 3>{px[i], px[i + 1], px[i + 2]};
  };
  // Flat on the horizon and square on, so the three pools are horizontal bars
  // down the middle of the wall — the elevation would otherwise tilt each ring
  // and fan the floors along the wall as well as up it, which is its own case
  // in web/test/three_planes_walls.test.ts. Model height through the zoom, into
  // cover-square, into pixels: 31 rows a floor at this frame size. The bottom
  // floor is LOW, which is the one thing a sign error would flip.
  auto bottom = at(W / 2, 166);
  auto middle = at(W / 2, 135);
  auto top    = at(W / 2, 104);
  INFO("bottom " << bottom[0] << "," << bottom[1] << "," << bottom[2]
       << "  middle " << middle[0] << "," << middle[1] << "," << middle[2]
       << "  top " << top[0] << "," << top[1] << "," << top[2]);

  CHECK(bottom[0] > bottom[1] + 40);
  CHECK(bottom[0] > bottom[2] + 40);
  CHECK(middle[1] > middle[0] + 40);
  CHECK(middle[1] > middle[2] + 40);
  CHECK(top[2] > top[0] + 40);
  CHECK(top[2] > top[1] + 40);

  // Flat across the ring, and gone by the edge of the wall: square on, the near
  // edge is at one distance along its whole length, so the pool is a bar rather
  // than a blob.
  auto lum = [](const std::array<int, 3>& c) { return (c[0] + c[1] + c[2]) / 3.0; };
  CHECK(std::abs(lum(at(W / 2 - 25, 104)) - lum(top)) < lum(top) * 0.10);
  CHECK(lum(at(6, 104)) < lum(top) * 0.35);
}

// Instrument for the open persistent-storage-buffer bug (web/KNOWN_ISSUES.md).
// warp.legacy.d_wave carries BOTH mechanisms — a stateful wave field in
// ping-pong textures and a pool of dampening flashes in a RWStructuredBuffer
// the vertex shader splats — so the ratio between "flashes on" and "field
// alone" isolates the buffer. On a healthy backend it falls monotonically with
// damp_count; on Metal it plateaus around 256 particles and stops moving.
//
// Hidden (the leading `.` in the tag) because it ASSERTS NOTHING — it prints
// numbers for a human to compare across backends. Run it by name:
//   ./build/test_effect_render "probe: d_wave*"
TEST_CASE("probe: d_wave persistent storage buffer", "[.probe]") {
  auto backend = gpu::createBackend();
  if (!backend) SKIP("No GPU device available");
  sketch_executor::WasmEffectBundles bundles;
  REQUIRE(bundles.init());
  EffectRuntime rt(backend.get());
  sketch_executor::ModuleRegistry registry(&rt);
  REQUIRE(bundles.loadBundleFile(kLegacyWasm, registry, backend.get(), nullptr) > 1);

  const uint32_t W = 128, H = 128; const int RGBA8 = 1;
  int inTex = backend->createTexture(W, H, RGBA8);
  int outTex = backend->createTexture(W, H, RGBA8);
  backend->clearTexture(inTex, 0, 0, 0, 1);

  auto run = [&](double damp, int count, double rate, int ticks) -> double {
    sketch_executor::SketchExecutor ex(&rt, &registry, backend.get());
    auto sketch = nlohmann::json::parse(R"JSON({
      "chain": [
        { "type": "module", "module_type": "warp.legacy.d_wave", "instance_key": "dw" }
      ],
      "instances": { "dw": { "module_type": "warp.legacy.d_wave", "state": {
        "debug_field": 1.0, "distortion": 0.5, "rate": 0.5, "wave_speed": 0.3
      } } }
    })JSON");
    auto& st = sketch["instances"]["dw"]["state"];
    st["damp"] = damp; st["damp_count"] = count; st["damp_rate"] = rate;
    for (int f = 0; f < ticks; ++f) {
      ex.execute(sketch, inTex, outTex, (int)W, (int)H, 1.0 / 60.0, /*dirty=*/f == 0);
      backend->submit();
    }
    auto px = backend->readbackTexture(outTex, W, H);
    long s = 0, n = 0;
    for (size_t i = 0; i < px.size(); i += 4) { s += px[i]; ++n; }
    return n ? (double)s / n : 0.0;
  };

  for (int ticks : {1, 4, 14, 40}) {
    const double base = run(0.0, 0, 0.5, ticks);
    std::fprintf(stderr, "[probe] ticks=%2d  field-alone mean red %.2f\n", ticks, base);
    if (ticks != 40) continue;
    for (int c : {64, 128, 256, 400, 1500, 4096}) {
      const double v = run(0.5, c, 0.5, ticks);
      std::fprintf(stderr, "[probe]   damp_count %4d -> %.2f  ratio %.3f\n",
                   c, v, base > 0 ? v / base : 0.0);
    }
    for (double r : {0.0, 1.0}) {
      const double v = run(0.5, 1500, r, ticks);
      std::fprintf(stderr, "[probe]   damp_rate %.1f (count 1500) -> %.2f\n", r, v);
    }
  }
}
