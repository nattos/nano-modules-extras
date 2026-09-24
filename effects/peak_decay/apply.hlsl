// motion.peak_decay — pass 2: apply. Luma gain from the pixel's age.
//
// Gain holds at 1 through `hold`, then falls along a smoothstep sigmoid over
// `fall` seconds, down by `amount`, and scales the LIVE input (chroma
// direction untouched — the pixel dims toward black like a meter bar
// falling). The catch mode only affects what RESETS the gain (pass 1).
// Alpha passes through.

#include "common.hlsl"

Texture2D<float4>   inTex    : register(t0);
Texture2D<float4>   stateTex : register(t1);   // this frame's meter output
RWTexture2D<float4> outTex   : register(u2);

cbuffer Uniforms : register(b3) {
  float dt;
  float amount;
  float hold;
  float fall;
  float threshold;
  float reset;
  float rgb_balance;
  float catch_mode;   // 0 = Any Change, 1 = Rise Only
};

[numthreads(8, 8, 1)]
void main(uint3 gid : SV_DispatchThreadID) {
  uint W, H;
  outTex.GetDimensions(W, H);
  if (gid.x >= W || gid.y >= H) return;

  float4 c = inTex[gid.xy];
  float age = stateTex[gid.xy].a;
  float g = pd_gain(age, amount, hold, fall);
  outTex[gid.xy] = float4(c.rgb * g, c.a);
}
