// vcr_halo down — one octave of the glow pyramid.
//
// Same 13-tap kernel as the prefilter (Jimenez, SIGGRAPH 2014), reading the
// previous level and writing the next. Separate textures per level rather
// than mips of one texture: the up chain reads level k AND level k+1 in the
// same dispatch, and distinct textures keep that trivially legal on WebGPU's
// sync-scope validator without any single-mip view juggling.

Texture2D<float4>   srcTex : register(t0);
RWTexture2D<float4> dstTex : register(u1);
SamplerState        samp   : register(s2);

cbuffer Uniforms : register(b3) {
  float src_texel_x;   // 1 / source level width
  float src_texel_y;   // 1 / source level height
  float src_scale;     // unused here (shared uniform block with up.hlsl)
  float add_weight;    // unused here
};

[numthreads(8, 8, 1)]
void main(uint3 gid : SV_DispatchThreadID) {
  uint dw, dh;
  dstTex.GetDimensions(dw, dh);
  if (gid.x >= dw || gid.y >= dh) return;

  float2 uv = (float2(gid.xy) + 0.5) / float2(dw, dh);
  float2 t  = float2(src_texel_x, src_texel_y);

  float4 a = srcTex.SampleLevel(samp, uv + t * float2(-2,  2), 0);
  float4 b = srcTex.SampleLevel(samp, uv + t * float2( 0,  2), 0);
  float4 c = srcTex.SampleLevel(samp, uv + t * float2( 2,  2), 0);
  float4 d = srcTex.SampleLevel(samp, uv + t * float2(-2,  0), 0);
  float4 e = srcTex.SampleLevel(samp, uv,                      0);
  float4 f = srcTex.SampleLevel(samp, uv + t * float2( 2,  0), 0);
  float4 g = srcTex.SampleLevel(samp, uv + t * float2(-2, -2), 0);
  float4 h = srcTex.SampleLevel(samp, uv + t * float2( 0, -2), 0);
  float4 i = srcTex.SampleLevel(samp, uv + t * float2( 2, -2), 0);
  float4 j = srcTex.SampleLevel(samp, uv + t * float2(-1,  1), 0);
  float4 k = srcTex.SampleLevel(samp, uv + t * float2( 1,  1), 0);
  float4 l = srcTex.SampleLevel(samp, uv + t * float2(-1, -1), 0);
  float4 m = srcTex.SampleLevel(samp, uv + t * float2( 1, -1), 0);

  dstTex[gid.xy] = (j + k + l + m) * 0.125
                 + e * 0.125
                 + (b + d + f + h) * 0.0625
                 + (a + c + g + i) * 0.03125;
}
