#pragma once
/*
 * overlay.h — in-effect debug-overlay toolbox.
 *
 * A small set of render helpers an effect BAKES IN (shared code, but linked
 * into the effect itself) to draw a debug/HUD overlay onto its own output
 * texture. It replaces the old host-side `canvas_*` ABI — which is a no-op in
 * the sketch-executor / barrel render path (nothing wires a DrawList there) —
 * with primitives that call the NORMAL GPU + text ABIs, so the overlay actually
 * shows up wherever the effect renders (native barrel, web, tests).
 *
 * Two mechanisms, split by the user's design:
 *   - Coloured rectangles / borders  → an instanced solid-quad GPU pass
 *     (overlay/{common,vs,fs}.hlsl → overlay_shaders.h). Cheap, per-frame,
 *     no relayout, so dynamic bits (playheads, live bars, flashes) are free.
 *   - Text                           → the host rich-text engine
 *     (text::layout(mode:"html") + text::render). One absolutely-positioned
 *     HTML document composites every label in a single pass; keep the label
 *     set stable frame-to-frame and the host caches the layout.
 *
 * Compositing order (in Canvas::end):
 *   1. tex_out  = copy(tex_in)         // passthrough background (or transparent)
 *   2. tex_out += solid rects          // AlphaOver, drawn in call order
 *   3. tex_out  = text over copy(tex_out)   // all labels on top
 *
 * Requirements on the host effect:
 *   - declare a `tex_out` PrimaryOutput texture field (and usually `tex_in`
 *     PrimaryInput) so the executor binds a writable render target + calls
 *     render(). Without a texture output the effect is a modulation-source
 *     passthrough and render() never runs.
 *   - call overlay::initShaders() once (module_init, after a GPU backend
 *     exists), and drive a Canvas from render().
 *
 * The solid-quad SPV lives in overlay_shaders.h, emitted next to the effect's
 * other shader headers by the bundle build (compile overlay/{vs,fs}.hlsl +
 * _emit_spv_header_var overlay vs fs). Any bundle that uses this header must
 * build those two shaders.
 */

#include <gpu.h>
#include <host.h>

#include <cstdio>
#include <cstring>

#include "overlay_shaders.h"   // OVERLAY_VS_SPV / OVERLAY_FS_SPV (bundle-built)

namespace overlay {

struct Color {
  float r = 1, g = 1, b = 1, a = 1;
};
inline Color rgba(float r, float g, float b, float a = 1.0f) { return Color{r, g, b, a}; }

namespace detail {

// One shared solid-quad PSO per module (function-local static → single
// instance across the effects that share a binary).
inline gpu::RenderPSO& psoRef() { static gpu::RenderPSO pso; return pso; }

// GPU-side rect record. MUST match `struct OverlayRect` in overlay/common.hlsl:
// float4 rect (x,y,w,h px) + float4 color (straight rgba) = 8 floats / 32 bytes.
struct GpuRect { float x, y, w, h, r, g, b, a; };

inline int toByte(float v) {
  int i = (int)(v * 255.0f + 0.5f);
  return i < 0 ? 0 : (i > 255 ? 255 : i);
}

} // namespace detail

/// Register the solid-quad shader + build the shared PSO. Idempotent; safe to
/// call every module_init / every frame (returns immediately once ready).
inline void initShaders() {
  if (detail::psoRef().valid()) return;
  if (gpu::Device::backend() == gpu::Backend::None) return;
  state::registerShaderSPV("overlay_vs", OVERLAY_VS_SPV, OVERLAY_VS_SPV_SIZE);
  state::registerShaderSPV("overlay_fs", OVERLAY_FS_SPV, OVERLAY_FS_SPV_SIZE);
  auto vs = gpu::Device::createShaderModuleByName("overlay_vs");
  auto fs = gpu::Device::createShaderModuleByName("overlay_fs");
  if (!vs || !fs) return;
  detail::psoRef() = gpu::Device::createInstancedRenderPSO(
      vs, "main", fs, "main", gpu::TextureFormat::Surface,
      gpu::Bindings().uniform(0).storage(1),
      gpu::Device::BlendMode::AlphaOver);
}

/// A per-effect overlay drawer. Store one on the effect's State and drive it
/// from render(): begin() → fillRect/border/text… → end().
class Canvas {
public:
  static constexpr int kMaxRects = 512;
  static constexpr int kHtmlCap  = 16384;

  /// Start a frame. `vp_w`/`vp_h` are the output pixel size (rect + text coords
  /// are in these pixels, top-left origin).
  void begin(int vp_w, int vp_h) {
    vp_w_ = vp_w; vp_h_ = vp_h;
    rect_count_ = 0;
    html_len_ = 0;
    html_[0] = 0;
  }

  /// Filled rectangle (output pixels, straight-alpha colour).
  void fillRect(float x, float y, float w, float h, Color c) {
    if (rect_count_ >= kMaxRects || w <= 0 || h <= 0) return;
    rects_[rect_count_++] = {x, y, w, h, c.r, c.g, c.b, c.a};
  }

  /// Rectangle outline of `t` px, drawn as four filled edge rects.
  void border(float x, float y, float w, float h, float t, Color c) {
    if (t <= 0) return;
    fillRect(x, y, w, t, c);                       // top
    fillRect(x, y + h - t, w, t, c);               // bottom
    fillRect(x, y + t, t, h - 2 * t, c);           // left
    fillRect(x + w - t, y + t, t, h - 2 * t, c);   // right
  }

  /// Text label. `x`/`y` is the top-left in output pixels; `size` is the
  /// font-size in px. Rendered through the host text engine as an absolutely
  /// positioned span (so many labels cost one composite pass).
  void text(const char* s, float x, float y, float size, Color c,
            int weight = 400, bool mono = false) {
    if (!s || !*s) return;
    if (html_len_ >= kHtmlCap - 512) return;
    int n = std::snprintf(
        html_ + html_len_, kHtmlCap - html_len_,
        "<span style='position:absolute;left:%.1fpx;top:%.1fpx;"
        "font-size:%.2fpx;line-height:1;color:rgba(%d,%d,%d,%.3f);"
        "font-weight:%d;font-family:%s;white-space:pre'>",
        x, y, size,
        detail::toByte(c.r), detail::toByte(c.g), detail::toByte(c.b), c.a,
        weight, mono ? "monospace" : "sans-serif");
    if (n > 0) html_len_ += n;
    appendHtmlEscaped(s);
    n = std::snprintf(html_ + html_len_, kHtmlCap - html_len_, "</span>");
    if (n > 0) html_len_ += n;
  }

  /// Composite everything onto tex_out (over tex_in) and submit.
  void end() {
    initShaders();

    int out = gpu::Device::textureForField("tex_out").id;
    if (out < 0) out = gpu::Device::renderTarget().id;
    if (out < 0) return;   // no writable target — nothing to draw into
    gpu::Texture outTex{out};

    int bg = gpu::Device::textureForField("tex_in").id;

    // 1 — base layer: passthrough input, or a transparent field.
    if (bg >= 0) gpu::Device::copy(gpu::Texture{bg}, outTex);
    else         gpu::Device::clear(outTex, 0, 0, 0, 0);

    // 2 — solid rects, blended on top in call order.
    if (rect_count_ > 0 && detail::psoRef().valid()) {
      ensureRectBuffers();
      if (rect_buf_.valid() && uni_buf_.valid()) {
        rect_buf_.write(reinterpret_cast<const float*>(rects_), rect_count_ * 8);
        float uni[4] = {(float)vp_w_, (float)vp_h_, 0.f, 0.f};
        uni_buf_.write(uni, 4);
        auto rp = gpu::RenderPass::beginLoad(outTex);
        rp.setPSO(detail::psoRef());
        rp.setBuffer(uni_buf_, 0);
        rp.setBuffer(rect_buf_, 1);
        rp.draw(6, rect_count_);
        rp.end();
      }
    }

    // 3 — text on top: composite the label document over a copy of the result.
    if (html_len_ > 0) {
      ensureScratch();
      if (scratch_.valid()) {
        gpu::Device::copy(outTex, scratch_);
        int id = layoutText();
        if (id > 0) {
          text::render(id, out, "{\"x\":0,\"y\":0}", scratch_.id);
          text::release(id);
        }
      }
    }

    gpu::Device::submit();
  }

  /// Release GPU resources (call from the effect's destroy()).
  void dispose() {
    rect_buf_.release();
    uni_buf_.release();
    scratch_.release();
    scratch_w_ = scratch_h_ = 0;
  }

private:
  void ensureRectBuffers() {
    if (!rect_buf_.valid())
      rect_buf_ = gpu::Device::createBuffer(kMaxRects * 8 * (int)sizeof(float),
                                            gpu::BufferUsage::Storage);
    if (!uni_buf_.valid())
      uni_buf_ = gpu::Device::createBuffer(4 * (int)sizeof(float),
                                           gpu::BufferUsage::Uniform);
  }

  void ensureScratch() {
    if (scratch_.valid() && scratch_w_ == vp_w_ && scratch_h_ == vp_h_) return;
    scratch_.release();
    scratch_ = gpu::Device::createTexture(vp_w_, vp_h_,
                                          gpu::TextureFormat::SketchDefault);
    scratch_w_ = vp_w_; scratch_h_ = vp_h_;
  }

  // Build the JSON text-layout spec from the accumulated spans and lay it out.
  int layoutText() {
    static char spec[kHtmlCap + 4096];   // one at a time; not reentrant
    int p = std::snprintf(spec, sizeof(spec),
        "{\"mode\":\"html\",\"html\":\"<!DOCTYPE html><html><head><style>"
        "*{margin:0;padding:0;box-sizing:border-box}body{position:relative}"
        "</style></head><body>");
    appendJsonEscaped(spec, p, (int)sizeof(spec), html_);
    int n = std::snprintf(spec + p, sizeof(spec) - p,
        "</body></html>\",\"width\":%d,\"height\":%d,\"scale\":1}",
        vp_w_, vp_h_);
    if (n > 0) p += n;
    return text::layout(spec, p);
  }

  // Append `s` to html_ with HTML text-content escaping (& < >).
  void appendHtmlEscaped(const char* s) {
    for (int i = 0; s[i] && html_len_ < kHtmlCap - 8; i++) {
      char c = s[i];
      const char* rep = nullptr;
      if (c == '&') rep = "&amp;";
      else if (c == '<') rep = "&lt;";
      else if (c == '>') rep = "&gt;";
      if (rep) { int n = std::snprintf(html_ + html_len_, kHtmlCap - html_len_, "%s", rep); if (n > 0) html_len_ += n; }
      else html_[html_len_++] = c;
    }
    html_[html_len_] = 0;
  }

  // Append `s` into `dst` as the body of a JSON string literal (escape the
  // characters JSON forbids raw). Mirrors richtext/text appendEscaped.
  static void appendJsonEscaped(char* dst, int& pos, int cap, const char* s) {
    static const char kHex[] = "0123456789abcdef";
    for (int i = 0; s[i] && pos < cap - 7; i++) {
      unsigned char c = (unsigned char)s[i];
      switch (c) {
        case '"':  dst[pos++]='\\'; dst[pos++]='"';  break;
        case '\\': dst[pos++]='\\'; dst[pos++]='\\'; break;
        case '\n': dst[pos++]='\\'; dst[pos++]='n';  break;
        case '\r': dst[pos++]='\\'; dst[pos++]='r';  break;
        case '\t': dst[pos++]='\\'; dst[pos++]='t';  break;
        default:
          if (c < 0x20) {
            dst[pos++]='\\'; dst[pos++]='u'; dst[pos++]='0'; dst[pos++]='0';
            dst[pos++]=kHex[(c >> 4) & 0xF]; dst[pos++]=kHex[c & 0xF];
          } else {
            dst[pos++] = (char)c;
          }
      }
    }
    dst[pos] = 0;
  }

  int vp_w_ = 0, vp_h_ = 0;
  detail::GpuRect rects_[kMaxRects];
  int rect_count_ = 0;
  char html_[kHtmlCap] = {0};
  int html_len_ = 0;

  gpu::Buffer rect_buf_;
  gpu::Buffer uni_buf_;
  gpu::Texture scratch_;
  int scratch_w_ = 0, scratch_h_ = 0;
};

} // namespace overlay
