// filter.legacy.glisten — search-texture downsample.
//
// Resamples the input to the fixed 64×64 search grid with a single linear tap
// per texel (matches the original NanoGraph TextureComputeNode resample). The
// anchor search never touches the full-res input — it runs on a blurred copy
// of this tiny grid.

Texture2D<float4>   inputTex : register(t0);
SamplerState        samp     : register(s1);
RWTexture2D<float4> outTex   : register(u2);

[numthreads(8, 8, 1)]
void main(uint3 gid : SV_DispatchThreadID) {
  uint w, h;
  outTex.GetDimensions(w, h);
  if (gid.x >= w || gid.y >= h) return;
  float2 uv = (float2(gid.xy) + 0.5) / float2(w, h);
  outTex[gid.xy] = inputTex.SampleLevel(samp, uv, 0);
}
