// looper_plugin.mm — NanoLooper FFGL plugin (dedicated looper bundle).
//
// A thin shell over the shared effect runtime — the SAME architecture as
// NanoBarrel, but instead of hosting a user-authored, editable, persisted
// sketch it drives ONE fixed effect (`control.nanolooper`) through the shared
// executor. Drop it on a Resolume clip and it *is* the looper: a 4-channel /
// 16-step sequencer with an on-video overlay, a trigger rail that launches
// clips tagged by NanoLooper Ch markers, and (optionally) an audible synth.
//
// It links NONE of the effect engine. The shared runtime (Metal backend + WAMR
// + effect bundles + executor) lives in libbridge_server.dylib as a process
// singleton, dlopen'd next to the bundle via BridgeLoader — the SAME sibling
// path NanoBarrel uses, so both share ONE runtime + WsServer. The effect WASM
// bundles are read from the colocated NanoBarrel bundle's Resources/wasm (the
// looper ships none of its own); if a NanoBarrel is already live the runtime is
// already built and our paths are ignored.
//
// Live-only: nothing is persisted (no FILE/config param, no UUID persistence).
// A fresh instance key is minted each InitGL; the recorded loop lives only for
// the session.
//
// Control: FFGL params write the looper effect's instance-state fields in the
// shared doc + flag the frame dirty; the executor re-fetches on dirty and
// delivers the changes via on_state_patched, where the WASM does its own edge
// detection. Momentary buttons (triggers/delete/undo/redo) are latched so a
// sub-frame press+release can't be diffed away.
//
// Audio: the WASM calls host.trigger_audio(ch) on each gate-on; the shared
// runtime fans that out over the additive audio_bus ABI, tagged with the firing
// instance's key. We register a listener that filters to our own executor
// namespace and plucks a native Synth (kept off the render/runtime thread work).

#include <array>
#include <atomic>
#include <cstring>
#include <memory>
#include <mutex>
#include <string>
#include <vector>

#include <OpenGL/gl3.h>

#import <Metal/Metal.h>
#import <Foundation/Foundation.h>
#import <AppKit/AppKit.h>

#include <dlfcn.h>
#include <nlohmann/json.hpp>

#include <ffgl/FFGLPluginSDK.h>
#include <ffgl/FFGLPluginInfo.h>
#include <ffgl/FFGLLib.h>

#include "plugin/bridge_loader.h"
#include "bridge/bridge_api.h"
#include "platform/resource_root.h"
#include "synth.h"

#include "nano_barrel/interop_texture.h"

namespace {

// FFGL param layout. `field` is the looper effect's state field the knob drives
// (nullptr = handled locally — see `local`). `momentary` fields are edge
// controls the WASM detects rising/falling on; they get the press latch.
// `group` is the Resolume param-panel section (SetParamGroup). `local` routes
// nullptr-field params to the native synth instead of the sketch doc.
enum class Local { None, SynthEnable, SynthGain };
struct ParamDef {
  const char* name;
  const char* group;    // Resolume param-panel section
  unsigned int type;    // FF_TYPE_*
  const char* field;    // control.nanolooper state field, or nullptr (local)
  float def;            // FFGL-normalized default (0..1)
  bool momentary;
  Local local;
  // Doc-write scale: the field's real max (min assumed 0). FFGL STANDARD knobs
  // are normalized 0..1, but a field whose schema range isn't 0..1 must be
  // rescaled before it hits the sketch doc. 1.0 = write the 0..1 value straight
  // through (the case for every 0..1 field). `def` above stays normalized.
  float scale = 1.0f;
};

// Order is the param-panel order; it mirrors the effect schema's field order in
// nanolooper/main.cpp. The WASM is driven by field NAME, not index — field names
// must match field_to_pid there. Grouped for Resolume; synth is plugin-local.
//
// The key PERFORMANCE controls (triggers, delete, mute, undo, redo, loop mode)
// deliberately carry NO group: Resolume renders grouped params inside a collapsed
// section that's a pain to reach live, so these stay at the panel's top level for
// instant access. The editor (nanolooper/main.cpp schema) keeps them grouped —
// this ungrouping is FFGL-only. The remaining, set-once params stay grouped.
const ParamDef kParams[] = {
  {"Trigger 1",       nullptr,     FF_TYPE_BOOLEAN,  "trigger_1",        0.0f,    true,  Local::None},
  {"Trigger 2",       nullptr,     FF_TYPE_BOOLEAN,  "trigger_2",        0.0f,    true,  Local::None},
  {"Trigger 3",       nullptr,     FF_TYPE_BOOLEAN,  "trigger_3",        0.0f,    true,  Local::None},
  {"Trigger 4",       nullptr,     FF_TYPE_BOOLEAN,  "trigger_4",        0.0f,    true,  Local::None},
  {"Delete",          nullptr,     FF_TYPE_BOOLEAN,  "delete",           0.0f,    true,  Local::None},
  {"Mute",            nullptr,     FF_TYPE_BOOLEAN,  "mute",             0.0f,    false, Local::None},
  {"Undo",            nullptr,     FF_TYPE_EVENT,    "undo",             0.0f,    true,  Local::None},
  {"Redo",            nullptr,     FF_TYPE_EVENT,    "redo",             0.0f,    true,  Local::None},
  {"Loop",            nullptr,     FF_TYPE_OPTION,   "loop_mode",        1.0f,    false, Local::None},
  {"Quantize Start",  "Quantize",  FF_TYPE_BOOLEAN,  "quantize_start",   0.0f,    false, Local::None},
  {"Quantize Length", "Quantize",  FF_TYPE_BOOLEAN,  "quantize_length",  0.0f,    false, Local::None},
  {"Grace",           "Quantize",  FF_TYPE_STANDARD, "grace",            0.0625f, false, Local::None},
  {"Send To Rail",    "Output",    FF_TYPE_BOOLEAN,  "send_to_rail",     1.0f,    false, Local::None},
  {"Strict Deadline", "Output",    FF_TYPE_STANDARD, "strict_deadline",  0.0f,    false, Local::None, 250.0f},
  {"Show Overlay",    "Display",   FF_TYPE_BOOLEAN,  "show_overlay",     1.0f,    false, Local::None},
  {"Anchor",          "Display",   FF_TYPE_OPTION,   "anchor",           0.0f,    false, Local::None},
  {"Overlay Opacity", "Display",   FF_TYPE_STANDARD, "overlay_opacity",  1.0f,    false, Local::None},
  {"Synth",           "Synth",     FF_TYPE_BOOLEAN,  nullptr,            0.0f,    false, Local::SynthEnable},
  {"Synth Gain",      "Synth",     FF_TYPE_STANDARD, nullptr,            0.5f,    false, Local::SynthGain},
};

// Option-dropdown elements. Element VALUES double as the field value, matching
// the effect schema's selectField — Resolume delivers the picked element's value
// straight to SetFloatParameter.
const char* const kAnchorLabels[4] = {"Top Left", "Bottom Left", "Top Right", "Bottom Right"};
const char* const kLoopLabels[3]   = {"Off", "Overdub", "Latch"};
constexpr unsigned int P_COUNT = sizeof(kParams) / sizeof(kParams[0]);

// The single effect instance key inside the fixed sketch.
constexpr const char* kLooperInstanceKey = "looper";

}  // namespace

// ============================================================================
class LooperPlugin : public CFFGLPlugin {
 public:
  LooperPlugin() : CFFGLPlugin() {
    SetMinInputs(1);
    SetMaxInputs(1);
    SetTimeSupported(true);
    for (unsigned int i = 0; i < P_COUNT; ++i) {
      if (kParams[i].type == FF_TYPE_OPTION) {
        // Enum dropdowns: pick the element table by field.
        const char* const* labels = nullptr; unsigned int n = 0;
        if (kParams[i].field && std::strcmp(kParams[i].field, "loop_mode") == 0) {
          labels = kLoopLabels; n = 3;
        } else {  // anchor
          labels = kAnchorLabels; n = 4;
        }
        SetOptionParamInfo(i, kParams[i].name, n, kParams[i].def);
        for (unsigned int e = 0; e < n; ++e)
          SetParamElementInfo(i, e, labels[e], (float)e);
      } else {
        SetParamInfo(i, kParams[i].name, kParams[i].type, kParams[i].def);
      }
      if (kParams[i].group) SetParamGroup(i, kParams[i].group);
      param_values_[i] = kParams[i].def;
    }
    // No runtime/WASM/bridge work in the ctor: Resolume builds a throwaway
    // prototype of every plugin at startup just to enumerate params, and those
    // never call InitGL. The param registration above is all the scan needs.
  }

  ~LooperPlugin() override { teardownBridge(); }

  // -- Lifecycle -------------------------------------------------------
  FFResult InitGL(const FFGLViewportStruct* vp) override {
    CFFGLPlugin::InitGL(vp);
    glGenFramebuffers(1, &src_fbo_);
    synth_.init();
    for (unsigned int i = 0; i < P_COUNT; ++i) {
      if (kParams[i].local == Local::SynthEnable) synth_.set_enabled(param_values_[i] >= 0.5f);
      else if (kParams[i].local == Local::SynthGain) synth_.set_gain(param_values_[i]);
    }
    setupBridge();
    return FF_SUCCESS;
  }

  FFResult DeInitGL() override {
    teardownBridge();
    synth_.deinit();
    if (src_fbo_) { glDeleteFramebuffers(1, &src_fbo_); src_fbo_ = 0; }
    input_interop_.reset();
    output_interop_.reset();
    return FF_SUCCESS;
  }

  // -- Parameters ------------------------------------------------------
  FFResult SetFloatParameter(unsigned int idx, float value) override {
    if (idx >= P_COUNT) return FF_SUCCESS;
    param_values_[idx] = value;

    // Native synth controls — local, not part of the sketch.
    const ParamDef& pd = kParams[idx];
    if (pd.local == Local::SynthEnable) { synth_.set_enabled(value >= 0.5f); return FF_SUCCESS; }
    if (pd.local == Local::SynthGain)   { synth_.set_gain(value);            return FF_SUCCESS; }

    if (!pd.field) return FF_SUCCESS;

    std::lock_guard<std::mutex> lock(tick_mu_);
    if (pd.momentary) {
      // Record the edge only — the render thread runs the latch state machine
      // (so a fast press+release can't collapse before a frame carries the 1).
      int v = value >= 0.5f ? 1 : 0;
      MomState& st = mom_[idx];
      if (v == 1 && st.held == 0) st.saw_rise = true;
      st.held = v;
    } else {
      // Level params (toggles + grace): write straight through, rescaled for
      // fields whose schema range isn't 0..1 (e.g. Strict Deadline → 0..250 ms).
      writeFieldLocked(pd.field, (double)value * (double)pd.scale);
      dirty_frame_ = true;
    }
    return FF_SUCCESS;
  }

  float GetFloatParameter(unsigned int idx) override {
    return idx < P_COUNT ? param_values_[idx] : 0.0f;
  }

  // -- Render ----------------------------------------------------------
  FFResult ProcessOpenGL(ProcessOpenGLStruct* pGL) override {
    ++frame_;
    bool dirty;
    {
      std::lock_guard<std::mutex> lock(tick_mu_);
      pumpMomentaryLocked();                 // may writeField + set dirty_frame_
      dirty = dirty_frame_;
      dirty_frame_ = false;
    }

    if (pGL->numInputTextures < 1 || !pGL->inputTextures || !pGL->inputTextures[0]) {
      drawBadgeOnly(pGL);
      return FF_SUCCESS;
    }
    const FFGLTextureStruct* pInput = pGL->inputTextures[0];
    const unsigned int W = currentViewport.width;
    const unsigned int H = currentViewport.height;
    if (W == 0 || H == 0) return FF_SUCCESS;
    if (!rt_ready_ || !bridge_ || !shared_device_ || !loader_.bridge_executor_render) {
      drawBadgeOnly(pGL);
      return FF_SUCCESS;
    }

    ensureInterop((int)pInput->Width, (int)pInput->Height, (int)W, (int)H);
    if (!input_interop_ || !output_interop_) { drawBadgeOnly(pGL); return FF_SUCCESS; }

    blitGlInputToInterop(pGL, pInput);

    double hostT = hostTime / 1000.0;
    if (!time_initialized_) { time_start_ = hostT; time_prev_ = hostT; time_initialized_ = true; }
    double rawDt = hostT - time_prev_;
    double dt    = std::max(0.0, std::min(rawDt, 0.1));
    time_prev_   = hostT;

    // No macro channel — the looper is a single fixed effect. Forward the host
    // musical clock so the sequencer advances (bpm 0 = no transport → 120).
    const double hostBpm = this->bpm > 0.0f ? (double)this->bpm : 120.0;
    int outputUsed = loader_.bridge_executor_render(
        bridge_, barrel_plugin_key_.c_str(),
        input_interop_->getNativeTexture(),
        output_interop_->getNativeTexture(),
        (int)W, (int)H, dt, hostT - time_start_, dirty ? 1 : 0,
        nullptr, 0, (double)this->barPhase, hostBpm);

    blitInteropToGlOutput(pGL, outputUsed != 0);
    return FF_SUCCESS;
  }

 private:
  // -- Momentary press latch (render thread) ---------------------------
  // Each momentary field is guaranteed ≥1 rendered frame at 1 before its 0, so a
  // sub-frame press+release (a fast MIDI/OSC gate) isn't diffed away by the
  // executor's per-field state compare. Runs at frame start, before the render.
  void pumpMomentaryLocked() {
    for (unsigned int i = 0; i < P_COUNT; ++i) {
      if (!kParams[i].momentary || !kParams[i].field) continue;
      MomState& st = mom_[i];
      if (st.committed == 0) {
        if (st.saw_rise || st.held == 1) {
          writeFieldLocked(kParams[i].field, 1.0);
          st.committed = 1;
          st.high_unrendered = true;   // the imminent render delivers the press
          st.saw_rise = false;
          dirty_frame_ = true;
        }
      } else {  // committed == 1
        if (st.high_unrendered) {
          st.high_unrendered = false;  // the previous frame rendered the press
        } else if (st.held == 0 && !st.saw_rise) {
          writeFieldLocked(kParams[i].field, 0.0);
          st.committed = 0;
          dirty_frame_ = true;
        }
        st.saw_rise = false;           // a rise while already high can't be split
      }
    }
  }

  void writeFieldLocked(const char* field, double value) {
    if (!bridge_ || !loader_.bridge_set_at) return;
    std::string path = "/plugins/" + barrel_plugin_key_ +
                       "/state/sketch/instances/" + kLooperInstanceKey +
                       "/state/" + field;
    loader_.bridge_set_at(bridge_, path.c_str(), nlohmann::json(value).dump().c_str());
  }

  // -- Audio -----------------------------------------------------------
  // Fired from the shared runtime's render thread when the looper effect calls
  // host.trigger_audio(ch). Filter to our own executor namespace so several
  // NanoLooper instances in one process each pluck only their own Synth.
  static void audioTrampoline(void* userdata, const char* instance_key, int channel) {
    auto* self = static_cast<LooperPlugin*>(userdata);
    if (!self || self->audio_key_prefix_.empty() || !instance_key) return;
    const std::string& p = self->audio_key_prefix_;
    if (std::strncmp(instance_key, p.c_str(), p.size()) != 0) return;
    if (self->synth_.is_enabled()) self->synth_.trigger(channel);
  }

  // -- Bundle discovery (mirrors nano_barrel_plugin.mm) ----------------
  // libbridge_server.dylib is a SIBLING of the bundle (same path → one shared
  // image across barrel + looper).
  // Anchor for image-relative discovery — an address inside OUR image.
  static void resourceAnchor() {}

  static std::string bundleDylibPath() {
    const std::string self = nano_paths::imagePathContaining(
        reinterpret_cast<const void*>(&resourceAnchor));
    if (self.empty()) return "";
    auto pos = self.rfind(".bundle");
    if (pos == std::string::npos) return "";
    return nano_paths::joinPath(nano_paths::parentDir(self.substr(0, pos)),
                                "libbridge_server.dylib");
  }

  // The looper ships no WASM of its own. It used to reach sideways into a
  // colocated NanoBarrel.bundle's Resources/ by name; now both plugins resolve
  // the SAME shared root, so the looper works wherever it is deployed and the
  // hardcoded "NanoBarrel.bundle" literal is gone.
  static std::string bundleWasmDir() {
    return nano_paths::wasmDir(reinterpret_cast<const void*>(&resourceAnchor));
  }
  static std::string bundleFontPath() {
    return nano_paths::fontPath(reinterpret_cast<const void*>(&resourceAnchor),
                                "default.ttf");
  }

  static std::string generateUuid() {
    @autoreleasepool {
      NSString* u = [[NSUUID UUID] UUIDString];
      return u ? std::string(u.UTF8String) : std::string();
    }
  }

  // -- Shared-bridge lifecycle -----------------------------------------
  void setupBridge() {
    if (bridge_) return;

    std::string dylib = bundleDylibPath();
    if (dylib.empty() || !loader_.load(dylib.c_str())) return;   // render-only badge
    if (!loader_.bridge_init || !loader_.bridge_register_plugin) return;
    bridge_ = loader_.bridge_init();
    if (!bridge_) return;

    if (loader_.bridge_rt_acquire && loader_.bridge_executor_create &&
        loader_.bridge_executor_render && loader_.bridge_rt_gpu_device) {
      std::string wasmDir  = bundleWasmDir();
      std::string fontPath = bundleFontPath();
      rt_ready_ = loader_.bridge_rt_acquire(bridge_, wasmDir.c_str(), fontPath.c_str()) != 0;
      if (rt_ready_)
        shared_device_ = (__bridge id<MTLDevice>)loader_.bridge_rt_gpu_device(bridge_);
    }

    // Fresh identity each session (no persistence).
    std::string uuid = generateUuid();
    char keybuf[128] = {0};
    int n = loader_.bridge_register_plugin(
        bridge_, "com.nano.nanolooper", 0, 1, 0,
        /*schema_json=*/"", uuid.c_str(), keybuf, sizeof(keybuf));
    std::string actual(keybuf, (n > 0 && n < (int)sizeof(keybuf)) ? n : (int)strlen(keybuf));
    barrel_plugin_key_ = actual.empty() ? uuid : actual;

    if (rt_ready_ && loader_.bridge_executor_create)
      loader_.bridge_executor_create(bridge_, barrel_plugin_key_.c_str());

    publishInitialState();

    // Register the audio listener AFTER the key prefix is set so a trigger on
    // the render thread always sees a stable filter.
    audio_key_prefix_ = barrel_plugin_key_ + "/";
    if (loader_.bridge_add_audio_listener)
      audio_token_ = loader_.bridge_add_audio_listener(
          bridge_, &LooperPlugin::audioTrampoline, this);

    dirty_frame_ = true;  // first frame fetches the fixed sketch
  }

  void teardownBridge() {
    if (!bridge_) return;
    if (audio_token_ && loader_.bridge_remove_audio_listener)
      loader_.bridge_remove_audio_listener(bridge_, audio_token_);
    audio_token_ = 0;
    audio_key_prefix_.clear();
    if (rt_ready_ && loader_.bridge_executor_destroy)
      loader_.bridge_executor_destroy(bridge_, barrel_plugin_key_.c_str());
    if (rt_ready_ && loader_.bridge_rt_release)
      loader_.bridge_rt_release(bridge_);
    rt_ready_ = false;
    shared_device_ = nil;
    if (loader_.bridge_unregister_plugin)
      loader_.bridge_unregister_plugin(bridge_, barrel_plugin_key_.c_str());
    if (loader_.bridge_release)
      loader_.bridge_release(bridge_);
    bridge_ = nullptr;
  }

  // Publish this instance's initial state, with the fixed single-effect sketch
  // (editor {chain,instances,wires} form the executor normalizes) inlined and
  // its state seeded to the param defaults.
  void publishInitialState() {
    if (!bridge_ || !loader_.bridge_set_plugin_state) return;

    nlohmann::json looperState = nlohmann::json::object();
    for (unsigned int i = 0; i < P_COUNT; ++i)
      if (kParams[i].field)
        looperState[kParams[i].field] = (double)kParams[i].def * (double)kParams[i].scale;

    nlohmann::json sketch = {
      {"chain", nlohmann::json::array({
        {{"instance_key", kLooperInstanceKey},
         {"module_type", "control.nanolooper"},
         {"type", "module"}},
      })},
      {"instances", {
        {kLooperInstanceKey, {
          {"module_type", "control.nanolooper"},
          {"state", looperState},
        }},
      }},
      {"wires", nlohmann::json::array()},
    };

    nlohmann::json plugin_schemas = nlohmann::json::object();
    if (rt_ready_ && loader_.bridge_rt_schemas && loader_.bridge_free_string) {
      char* raw = loader_.bridge_rt_schemas(bridge_);
      if (raw) {
        auto parsed = nlohmann::json::parse(raw, nullptr, false);
        loader_.bridge_free_string(raw);
        if (parsed.is_object()) plugin_schemas = std::move(parsed);
      }
    }

    nlohmann::json initial = {
      {"sketch", std::move(sketch)},
      {"macros", nlohmann::json::array()},
      {"triggers", nlohmann::json::object()},
      {"host", nlohmann::json::object()},
      {"plugin_schemas", std::move(plugin_schemas)},
      {"preview_requests", nlohmann::json::object()},
    };
    loader_.bridge_set_plugin_state(bridge_, barrel_plugin_key_.c_str(), initial.dump().c_str());
  }

  // -- Interop (verbatim from nano_barrel_plugin.mm) -------------------
  void ensureInterop(int inW, int inH, int outW, int outH) {
    if (!shared_device_) return;
    if (!input_interop_ || input_interop_->getWidth() != inW || input_interop_->getHeight() != inH) {
      input_interop_ = createInteropTexture((__bridge void*)shared_device_, inW, inH);
    }
    if (!output_interop_ || output_interop_->getWidth() != outW || output_interop_->getHeight() != outH) {
      output_interop_ = createInteropTexture((__bridge void*)shared_device_, outW, outH);
    }
  }

  void blitGlInputToInterop(ProcessOpenGLStruct* pGL, const FFGLTextureStruct* pInput) {
    GLenum target = GL_TEXTURE_RECTANGLE;
    if (pInput->HardwareWidth > pInput->Width || pInput->HardwareHeight > pInput->Height)
      target = GL_TEXTURE_2D;

    GLint prevRead = 0, prevDraw = 0;
    glGetIntegerv(GL_READ_FRAMEBUFFER_BINDING, &prevRead);
    glGetIntegerv(GL_DRAW_FRAMEBUFFER_BINDING, &prevDraw);
    glBindFramebuffer(GL_READ_FRAMEBUFFER, src_fbo_);
    glFramebufferTexture2D(GL_READ_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, target, pInput->Handle, 0);
    glBindFramebuffer(GL_DRAW_FRAMEBUFFER, input_interop_->getOpenGLFBO());
    glBlitFramebuffer(0, 0, (GLint)pInput->Width, (GLint)pInput->Height,
                      0, (GLint)input_interop_->getHeight(),
                      (GLint)input_interop_->getWidth(), 0,
                      GL_COLOR_BUFFER_BIT, GL_LINEAR);
    glBindFramebuffer(GL_READ_FRAMEBUFFER, (GLuint)prevRead);
    glBindFramebuffer(GL_DRAW_FRAMEBUFFER, (GLuint)prevDraw);
    glFlush();
  }

  void blitInteropToGlOutput(ProcessOpenGLStruct* pGL, bool outputUsed) {
    if (!output_interop_) return;
    const unsigned int W = currentViewport.width;
    const unsigned int H = currentViewport.height;
    GLint prevRead = 0, prevDraw = 0;
    glGetIntegerv(GL_READ_FRAMEBUFFER_BINDING, &prevRead);
    glGetIntegerv(GL_DRAW_FRAMEBUFFER_BINDING, &prevDraw);
    glBindFramebuffer(GL_READ_FRAMEBUFFER,
                      outputUsed ? output_interop_->getOpenGLFBO() : input_interop_->getOpenGLFBO());
    glBindFramebuffer(GL_DRAW_FRAMEBUFFER, pGL->HostFBO);
    glBlitFramebuffer(0, (GLint)H, (GLint)W, 0, 0, 0, (GLint)W, (GLint)H,
                      GL_COLOR_BUFFER_BIT, GL_NEAREST);
    glBindFramebuffer(GL_READ_FRAMEBUFFER, (GLuint)prevRead);
    glBindFramebuffer(GL_DRAW_FRAMEBUFFER, (GLuint)prevDraw);
  }

  void drawBadgeOnly(ProcessOpenGLStruct* pGL) {
    const unsigned int W = currentViewport.width;
    const unsigned int H = currentViewport.height;
    if (W == 0 || H == 0) return;
    GLint dst_fbo = 0;
    glGetIntegerv(GL_FRAMEBUFFER_BINDING, &dst_fbo);
    glBindFramebuffer(GL_FRAMEBUFFER, dst_fbo);
    const int bx = (int)(W * 0.85f), by = (int)(H * 0.85f);
    const int bw = (int)(W * 0.12f), bh = (int)(H * 0.12f);
    glEnable(GL_SCISSOR_TEST);
    glScissor(bx, by, bw, bh);
    glClearColor(0.1f, 0.6f, 1.0f, 1.0f);
    glClear(GL_COLOR_BUFFER_BIT);
    glDisable(GL_SCISSOR_TEST);
  }

  // -- State -----------------------------------------------------------
  struct MomState {
    int  held = 0;              // latest Resolume value (0/1), set on the UI thread
    bool saw_rise = false;      // a 0→1 arrived since last consumed
    int  committed = 0;         // value currently written into the doc
    bool high_unrendered = false;  // wrote a 1 a render hasn't carried yet
  };

  std::mutex                      tick_mu_;
  plugin::BridgeLoader            loader_;
  BridgeHandle                    bridge_ = nullptr;
  std::string                     barrel_plugin_key_;
  bool                            rt_ready_ = false;
  id<MTLDevice>                   shared_device_ = nil;

  std::unique_ptr<InteropTexture> input_interop_;
  std::unique_ptr<InteropTexture> output_interop_;

  Synth                           synth_;
  std::string                     audio_key_prefix_;   // "<key>/" filter
  uint64_t                        audio_token_ = 0;

  double  time_start_ = 0.0, time_prev_ = 0.0;
  bool    time_initialized_ = false;
  bool    dirty_frame_ = false;

  std::array<float, P_COUNT>      param_values_{};
  std::array<MomState, P_COUNT>   mom_{};
  int     frame_ = 0;
  GLuint  src_fbo_ = 0;
};

// ============================================================================
static CFFGLPluginInfo PluginInfo(
    PluginFactory<LooperPlugin>,
    "NLPR",                 // keep the historical 4CC so existing comps resolve
    "NanoLooper",
    2, 1,
    1, 0,
    FF_EFFECT,
    "NanoLooper — a 4-channel/16-step looper sequencer that launches Resolume "
    "clips on trigger channels, with an on-video overlay",
    "nano");
