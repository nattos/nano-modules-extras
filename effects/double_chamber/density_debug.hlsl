// source.legacy.double_chamber — density buffer debug view (interactions).
//
// Blits the interaction density buffer to tex_out as a heat map so the crowding
// field (and the effect of interaction_radius) is directly visible. The buffer
// is square and covers uv [0,1]², sampled here at the output uv.

Texture2D<float4>   densityTex : register(t0);
SamplerState        samp       : register(s1);
RWTexture2D<float4> outTex     : register(u2);

[numthreads(8, 8, 1)]
void main(uint3 gid : SV_DispatchThreadID) {
  uint W, H;
  outTex.GetDimensions(W, H);
  if (gid.x >= W || gid.y >= H) return;

  float2 uv = (float2(gid.xy) + 0.5) / float2(W, H);
  float d = densityTex.SampleLevel(samp, uv, 0).r;
  // Soft-normalise so it's readable at any scale (~4 neighbours → mid).
  float v = d / (d + 4.0);
  // Black → red → yellow → white heat ramp.
  float3 col = float3(saturate(v * 1.5), saturate(v * 1.5 - 0.4), saturate(v * 1.5 - 0.8));
  outTex[gid.xy] = float4(col, 1.0);
}
