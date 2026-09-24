// color.legacy.lut_collection — pass 2: apply the selected colour LUT.
//
// Faithful to the Resolume Wire "LUT 2" ISF LUTShader, but efficient: the
// original hand-addressed an 8x8 strip with two taps + a manual blend; here a
// real 3D texture does the trilinear lookup in ONE hardware-filtered sample.
//
//   coord  = pow(input.rgb, power)         // pregain push into the LUT
//   graded = pow(LUT(coord), 1/power)      // and unwarp
//   out    = lerp(input, graded, amount)   // ISF "Alpha" mix; keep input alpha
//
// `power` is derived from the [-1,1] Pregain on the C++ side (see main.cpp).
Texture2D<float4>   inputTex  : register(t0);
Texture3D<float4>   lut       : register(t1);
SamplerState        samp      : register(s2);   // Linear + ClampToEdge
RWTexture2D<float4> outputTex : register(u3);

cbuffer Uniforms : register(b4) {
  float amount;     // ISF Alpha — mix between input and graded
  float power;      // input^power before lookup
  float invPower;   // 1/power, applied to the LUT result
  float _pad;
}

static const float LUT_N = 64.0;

[numthreads(8, 8, 1)]
void main(uint3 gid : SV_DispatchThreadID) {
  uint ow, oh;
  outputTex.GetDimensions(ow, oh);
  if (gid.x >= ow || gid.y >= oh) return;

  float4 c = inputTex.Load(int3(int2(gid.xy), 0));

  float3 coord = pow(saturate(c.rgb), power.xxx);
  // Half-texel correction: coord 0 -> first texel centre, 1 -> last centre.
  float3 uvw = (coord * (LUT_N - 1.0) + 0.5) / LUT_N;
  float3 graded = lut.SampleLevel(samp, uvw, 0).rgb;
  graded = pow(saturate(graded), invPower.xxx);

  float3 outc = lerp(c.rgb, graded, amount);
  outputTex[gid.xy] = float4(outc, c.a);
}
