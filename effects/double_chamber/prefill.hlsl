// source.legacy.double_chamber — prefill pass: tex_in × input_alpha → tex_out.
Texture2D<float4>   inputTex  : register(t0);
RWTexture2D<float4> outputTex : register(u1);
cbuffer U : register(b2) { float sr, sg, sb, sa; };
[numthreads(8, 8, 1)]
void main(uint3 gid : SV_DispatchThreadID) {
  uint w, h; outputTex.GetDimensions(w, h);
  if (gid.x >= w || gid.y >= h) return;
  float4 c = inputTex[gid.xy];
  outputTex[gid.xy] = float4(c.rgb * float3(sr, sg, sb), c.a * sa);
}
