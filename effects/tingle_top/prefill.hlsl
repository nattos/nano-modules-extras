// source.light.tingle_top — prefill pass. Copies tex_in into tex_out so the instanced
// sparkle quads (next pass) blend ADDITIVELY over the input. Also draws the
// optional debug lines at each active voice's distribution peak.

Texture2D<float4>   inputTex  : register(t0);
RWTexture2D<float4> outputTex : register(u1);

cbuffer Uniforms : register(b2) {
  uint   debug_region;
  uint   _pad0;
  float  _pad1;
  float  _pad2;
  float4 peaks;       // up to 4 voice y_peaks; inactive set to -1
};

[numthreads(8, 8, 1)]
void main(uint3 gid : SV_DispatchThreadID) {
  uint W, H;
  outputTex.GetDimensions(W, H);
  if (gid.x >= W || gid.y >= H) return;
  float4 c = inputTex[gid.xy];
  if (debug_region != 0u) {
    float uy = (float(gid.y) + 0.5) / float(H);
    float t = 1.5 / float(H);
    float4 pk = peaks;
    if (abs(uy - pk.x) < t || abs(uy - pk.y) < t ||
        abs(uy - pk.z) < t || abs(uy - pk.w) < t)
      c.rgb = saturate(c.rgb + float3(0.0, 0.25, 0.25));
  }
  outputTex[gid.xy] = c;
}
