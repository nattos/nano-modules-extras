// source.particles.sweep_chamber — density buffer debug view (heat map).
// flow_swarm parity.

Texture2D<float4>   densityTex : register(t0);
SamplerState        samp       : register(s1);
RWTexture2D<float4> outTex     : register(u2);

cbuffer DensityUniforms : register(b3) {
  float res;
  float dens_scale;  // screen uv → density uv (the buffer overhangs the frame)
  float dens_off;
  float _pad;
};

[numthreads(8, 8, 1)]
void main(uint3 gid : SV_DispatchThreadID) {
  uint W, H;
  outTex.GetDimensions(W, H);
  if (gid.x >= W || gid.y >= H) return;

  float2 uv = (float2(gid.xy) + 0.5) / float2(W, H);
  float d = densityTex.SampleLevel(samp, uv * dens_scale + dens_off, 0).r;
  float v = d / (d + 4.0);
  float3 col = float3(saturate(v * 1.5), saturate(v * 1.5 - 0.4), saturate(v * 1.5 - 0.8));
  outTex[gid.xy] = float4(col, 1.0);
}
