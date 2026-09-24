// filter.blur.lens — RGBA16F separable Gaussian (one axis per dispatch).
//
// The RGBA16F-safe blur (fx::GaussianBlur's RGBA8 scratch clamps HDR). Used for
// the wide flare/glow blurs (veiling glare, halation, bloom). Integer per-pixel
// taps, weights from a storage buffer. Run twice (host: H then V) via Blur16::apply.

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
