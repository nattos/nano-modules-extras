// color.legacy.bicolor_grad — hue histogram pass.
//
// Samples a coarse grid of the input, converts RGB→YIQ to get a hue angle +
// chroma, and atomically scatters a chroma-weighted vote into a 64-bin hue
// histogram. Mirrors the NanoGraph BicolorGrad preprocessing + histogram.

Texture2D<float4>       inputTex : register(t0);
SamplerState           samp     : register(s1);
RWStructuredBuffer<int> hist     : register(u2);   // 64 bins

cbuffer U : register(b3) {
  float grid;          // grid resolution (samples per axis)
  float weight_scale;  // chroma → integer vote scale
  float _p0;
  float _p1;
};

static const float3 kRGBToI = float3(0.596, -0.275, -0.321);
static const float3 kRGBToQ = float3(0.212, -0.523,  0.311);
static const float  PI = 3.14159265358979323846;

[numthreads(8, 8, 1)]
void main(uint3 gid : SV_DispatchThreadID) {
  int g = (int)grid;
  if ((int)gid.x >= g || (int)gid.y >= g) return;

  float2 uv = (float2(gid.xy) + 0.5) / grid;
  float3 rgb = max(inputTex.SampleLevel(samp, uv, 0).rgb, 0.0);

  float I = dot(rgb, kRGBToI);
  float Q = dot(rgb, kRGBToQ);
  float chroma = sqrt(I * I + Q * Q);
  float hue = (abs(I) <= 1e-5 && abs(Q) <= 1e-5) ? 0.0 : atan2(Q, I);
  hue /= (PI * 2.0);
  hue -= floor(hue);                 // [0, 1)

  int bin = (int)(hue * 64.0);
  bin = clamp(bin, 0, 63);

  int w = (int)(chroma * weight_scale);
  if (w > 0) {
    int prev;
    InterlockedAdd(hist[bin], w, prev);
  }
}
