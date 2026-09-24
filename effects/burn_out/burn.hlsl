// color.legacy.burn_out — the grade pass.
//
// A per-pixel exposure-blowout grade whose strength is driven by an AR
// envelope computed in main.cpp (tick). At the envelope's peak the image
// blows out: saturation + contrast lift (atan-soft-clipped on the C++ side),
// exposure pushes highlights toward white, and a crossfade carries the result
// to pure white. Optionally the burn also drops alpha for a compositing
// fade-out. The whole thing decays back to the untouched image.

Texture2D<float4>   inputTex  : register(t0);
RWTexture2D<float4> outputTex : register(u1);

cbuffer Uniforms : register(b2) {
  float sat_amt;     // saturation add (>0 boosts; soft-clipped in C++)
  float con_amt;     // contrast add (around mid-grey)
  float brightness;  // tone lift/crush before the fade (optional flash)
  float fade_black;  // crossfade to black at the burn peak (the fade-out)
  float alpha_fade;  // alpha reduction (compositing fade-out)
  float _p0, _p1, _p2;
};

[numthreads(8, 8, 1)]
void main(uint3 gid : SV_DispatchThreadID) {
  uint W, H;
  outputTex.GetDimensions(W, H);
  if (gid.x >= W || gid.y >= H) return;

  float4 c = inputTex[gid.xy];
  float3 rgb = c.rgb;

  // The "burn": saturation + contrast lift (the colours intensify on the way
  // down), an optional brightness flash, then a crossfade to BLACK — Burn Out
  // is an emotional fade-OUT, not a white blow-out.
  float luma = dot(rgb, float3(0.2126, 0.7152, 0.0722));
  rgb = lerp(float3(luma, luma, luma), rgb, 1.0 + sat_amt);
  rgb = (rgb - 0.5) * (1.0 + con_amt) + 0.5;
  rgb += brightness;
  rgb = lerp(rgb, float3(0.0, 0.0, 0.0), saturate(fade_black));
  rgb = saturate(rgb);

  float a = c.a * (1.0 - saturate(alpha_fade));
  outputTex[gid.xy] = float4(rgb, a);
}
