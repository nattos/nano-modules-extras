// filter.reconstruct.line — RGBA16F separable Gaussian (one axis per dispatch).
//
// fx::GaussianBlur routes its intermediate through an RGBA8 scratch, which would
// 8-bit-quantize the luma pyramid (killing sub-LSB deband) and CLAMP the signed
// structure-tensor products (Jxy) to [0,1]. This does the same separable blur
// entirely in RGBA16F (integer per-pixel taps, weights from a storage buffer),
// preserving precision and sign. Run twice (host: H then V) via Blur16::apply.

// Write-only rgba16f (format set by the registerShaderSPV "rgba16float","write"
// hint — an explicit [[vk::image_format]] pin would force read_write access,
// which WebGPU forbids for rgba16f and which mismatches a write-only layout).
RWTexture2D<float4>     outputTex : register(u1);
Texture2D<float4>       inputTex : register(t0);
StructuredBuffer<float> weights  : register(t3);
cbuffer BlurU : register(b2) {
  float dir_x; float dir_y; float spacing; int half_count;
};

[numthreads(8, 8, 1)]
void main(uint3 gid : SV_DispatchThreadID) {
  uint w, h;
  outputTex.GetDimensions(w, h);
  if (gid.x >= w || gid.y >= h) return;
  int2 p    = int2(gid.xy);
  int2 hi   = int2(w - 1, h - 1);
  int2 step = int2(round(float2(dir_x, dir_y) * spacing));

  float4 acc = inputTex[p] * weights[0];
  for (int i = 1; i <= half_count; i++) {
    int2 o = step * i;
    acc += weights[i] * (inputTex[clamp(p + o, int2(0, 0), hi)]
                       + inputTex[clamp(p - o, int2(0, 0), hi)]);
  }
  outputTex[gid.xy] = acc;
}
