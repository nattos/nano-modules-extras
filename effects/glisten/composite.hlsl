// filter.legacy.glisten — final composite.
//
// out = input × input_alpha + blurredSparkles × tint  (the original's Add
// mix), computed in LINEAR space: the original's textures were all sRGB, so
// its Add mixed decoded values and the output texture re-encoded. Input and
// sparkle-layer texels here hold sRGB codes; decode, add, encode. The layer
// is ≥ 0 (unorm-accumulated), so the composite only ever brightens.

#include "nano_color.hlsl"   // nano_srgb_to_linear / nano_linear_to_srgb

Texture2D<float4>   inputTex : register(t0);
Texture2D<float4>   sparkTex : register(t1);
SamplerState        samp     : register(s2);
RWTexture2D<float4> outTex   : register(u3);

cbuffer U : register(b4) {
  float input_alpha;
  float tint_r, tint_g, tint_b;
};

[numthreads(8, 8, 1)]
void main(uint3 gid : SV_DispatchThreadID) {
  uint w, h;
  outTex.GetDimensions(w, h);
  if (gid.x >= w || gid.y >= h) return;
  float2 uv = (float2(gid.xy) + 0.5) / float2(w, h);
  float4 in_c  = inputTex[gid.xy];
  float4 spark = sparkTex.SampleLevel(samp, uv, 0);
  float3 base_l  = nano_srgb_to_linear(in_c.rgb) * input_alpha;
  float3 spark_l = nano_srgb_to_linear(spark.rgb) * float3(tint_r, tint_g, tint_b);
  float4 o;
  o.rgb = nano_linear_to_srgb(saturate(base_l + spark_l));
  o.a = in_c.a * input_alpha + spark.a;
  outTex[gid.xy] = o;
}
