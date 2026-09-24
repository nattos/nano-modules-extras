// color.legacy.bicolor_grad — render pass.
//
// Paints a smooth gradient from MinorColor to MajorColor, oriented along the
// analysed Direction through the analysed Center, with a neutral mid-band, and
// composites it over the input by the chosen blend mode.

Texture2D<float4>         inputTex : register(t0);
SamplerState             samp     : register(s1);
StructuredBuffer<float>   aBuf     : register(t2);   // analysis (read)
RWTexture2D<float4>       outTex   : register(u3);

cbuffer U : register(b4) {
  float scale;       // gradient span (cover fraction)
  float blend;       // 0 = input, 1 = full gradient
  float neutralMix;  // mid-band neutral amount
  float midband;     // mid-band width
  float nr, ng, nb;  // neutral colour
  float mode;        // 0 Mix, 1 Multiply, 2 Screen, 3 Add
};

[numthreads(8, 8, 1)]
void main(uint3 gid : SV_DispatchThreadID) {
  uint W, H;
  outTex.GetDimensions(W, H);
  if (gid.x >= W || gid.y >= H) return;

  float2 uv = (float2(gid.xy) + 0.5) / float2(W, H);

  float3 major  = float3(aBuf[0], aBuf[1], aBuf[2]);
  float3 minor  = float3(aBuf[3], aBuf[4], aBuf[5]);
  float2 center = float2(aBuf[6], aBuf[7]);
  float2 dir    = float2(aBuf[8], aBuf[9]);

  float2 p = uv * 2.0 - 1.0;
  float  s = max(scale, 1e-3);
  float  proj = dot(p - center, dir) / s;
  float  t = saturate(proj * 0.5 + 0.5);

  float3 grad = lerp(minor, major, t);
  float band = (1.0 - smoothstep(0.0, max(midband, 1e-3), abs(proj))) * saturate(neutralMix);
  grad = lerp(grad, float3(nr, ng, nb), band);

  float4 inp = inputTex[gid.xy];
  float3 outc;
  if      (mode < 0.5) outc = lerp(inp.rgb, grad, blend);
  else if (mode < 1.5) outc = lerp(inp.rgb, inp.rgb * grad, blend);
  else if (mode < 2.5) outc = lerp(inp.rgb, 1.0 - (1.0 - inp.rgb) * (1.0 - grad), blend);
  else                 outc = inp.rgb + grad * blend;

  outTex[gid.xy] = float4(outc, inp.a);
}
