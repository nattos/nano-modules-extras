#pragma once
/*
 * blur16.h — RGBA16F two-pass separable Gaussian for line_reconstruct.
 *
 * A precision-preserving sibling of fx::GaussianBlur: the intermediate scratch is
 * RGBA16F (fx::GaussianBlur's is RGBA8, which quantizes luma to 8-bit and clamps
 * signed values to [0,1]). Exact fixed sigma in px, weights CPU-computed with a
 * 3.0-sigma half-width (matching the prototype's _gauss1d), integer per-pixel
 * taps → tap locations never slide (no shimmer). Type-shared (file-static), like
 * fx::GaussianBlur. Requires blur16.hlsl compiled into line_reconstruct_shaders.h.
 */

#include <gpu.h>
#include "line_reconstruct_shaders.h"
#include <cmath>

namespace line_reconstruct {

class Blur16 {
public:
  static constexpr int MAX_HALF = 128;

  bool init() {
    if (m_init) return true;
    if (gpu::Device::backend() == gpu::Backend::None) return false;
    state::registerShaderSPV("line_reconstruct_blur16", BLUR16_SPV, BLUR16_SPV_SIZE,
                             "rgba16float", "write");
    auto cs = gpu::Device::createShaderModuleByName("line_reconstruct_blur16");
    if (!cs) return false;
    m_pso = gpu::Device::createComputePSO(cs, "main", gpu::Bindings()
        .tex2d(0).storageTex2d(1, gpu::TextureFormat::RGBA16F).uniform(2).storage(3));
    m_uh   = gpu::Device::createBuffer(sizeof(U), gpu::BufferUsage::Uniform);
    m_uv   = gpu::Device::createBuffer(sizeof(U), gpu::BufferUsage::Uniform);
    m_wbuf = gpu::Device::createBuffer(sizeof(float) * (MAX_HALF + 1), gpu::BufferUsage::Storage);
    m_init = true;
    return true;
  }

  bool valid() const { return m_init; }

  // Blur `in` → `out` at Gaussian sigma_px (both RGBA16F, same w×h). Uses an
  // internal RGBA16F scratch for the horizontal pass.
  void apply(gpu::Texture in, gpu::Texture out, int w, int h, float sigma_px) {
    if (!m_init || w <= 0 || h <= 0 || !in.valid() || !out.valid()) return;
    int half = kernel(sigma_px);
    m_wbuf.write(m_w, MAX_HALF + 1);
    ensureScratch(w, h);
    if (!m_scratch.valid()) return;

    U uh = { 1.0f, 0.0f, 1.0f, half }; m_uh.writeOne(uh);
    U uv = { 0.0f, 1.0f, 1.0f, half }; m_uv.writeOne(uv);
    {
      auto cp = gpu::ComputePass::begin();
      cp.setPSO(m_pso);
      cp.setTexture(in, 0, 0);
      cp.setTexture(m_scratch, 1, 1);
      cp.setBuffer(m_uh, 2);
      cp.setBuffer(m_wbuf, 3);
      cp.dispatch((w + 7) / 8, (h + 7) / 8);
      cp.end();
    }
    {
      auto cp = gpu::ComputePass::begin();
      cp.setPSO(m_pso);
      cp.setTexture(m_scratch, 0, 0);
      cp.setTexture(out, 1, 1);
      cp.setBuffer(m_uv, 2);
      cp.setBuffer(m_wbuf, 3);
      cp.dispatch((w + 7) / 8, (h + 7) / 8);
      cp.end();
    }
  }

  void release() {
    if (m_scratch.valid()) m_scratch.release();
  }

private:
  struct U { float dx, dy, spacing; int half; };

  bool m_init = false;
  gpu::ComputePSO m_pso;
  gpu::Buffer m_uh, m_uv, m_wbuf;
  gpu::Texture m_scratch;
  int m_sw = 0, m_sh = 0;
  float m_w[MAX_HALF + 1];

  void ensureScratch(int w, int h) {
    if (m_scratch.valid() && m_sw == w && m_sh == h) return;
    if (m_scratch.valid()) m_scratch.release();
    m_scratch = gpu::Device::createTexture(w, h, gpu::TextureFormat::RGBA16F);
    m_sw = w; m_sh = h;
  }

  // Normalized Gaussian weights, 3.0-sigma half-width (matches prototype).
  int kernel(float sigma) {
    if (sigma < 1e-3f) {
      m_w[0] = 1.0f;
      for (int i = 1; i <= MAX_HALF; i++) m_w[i] = 0.0f;
      return 0;
    }
    int half = (int)std::ceil(3.0f * sigma);
    if (half < 1) half = 1;
    if (half > MAX_HALF) half = MAX_HALF;
    float sum = 1.0f; m_w[0] = 1.0f;
    for (int i = 1; i <= half; i++) {
      float x = (float)i;
      float wv = std::exp(-(x * x) / (2.0f * sigma * sigma));
      m_w[i] = wv; sum += 2.0f * wv;
    }
    float inv = 1.0f / sum;
    for (int i = 0; i <= half; i++) m_w[i] *= inv;
    for (int i = half + 1; i <= MAX_HALF; i++) m_w[i] = 0.0f;
    return half;
  }
};

} // namespace line_reconstruct
