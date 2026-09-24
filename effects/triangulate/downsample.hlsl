// triangulate — downsample the (viewport-res) input to the internal
// processing resolution with a linear sampler.
Texture2D<float4>   inTex   : register(t0);
SamplerState        samp    : register(s1);
RWTexture2D<float4> outTex  : register(u2);

[numthreads(8, 8, 1)]
void main(uint3 gid : SV_DispatchThreadID) {
  uint w, h;
  outTex.GetDimensions(w, h);
  if (gid.x >= w || gid.y >= h) return;
  float2 uv = (float2(gid.xy) + 0.5) / float2(w, h);
  outTex[gid.xy] = inTex.SampleLevel(samp, uv, 0.0);
}
