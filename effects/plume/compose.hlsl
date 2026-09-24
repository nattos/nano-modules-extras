// plume sculptor — overlay compose: base shell map + simulation overlay
// -> final shell map.
//
// Part of the shared generator (field_gen.h), dispatched TWICE per frame
// (full + coarse resolutions) when a height overlay is active — the plume
// renderer itself never uses it (its sculpted field is pass-through), the
// standalone provider's tracer simulation does. Kept as a separate pass so
// shell.hlsl stays byte-identical for the base field.
//
// Inputs:  base shell at the SAME resolution as the output (texel load),
//          overlay oct map sampled bilinear (256², .r = height in [-1,1]
//          normalized units, .g = flow/traffic density in [0,1]).
// Output:  .r = base h + overlay h * amp (world units), floored so the
//               simulation cannot dig through the body's core,
//          .g = crest emphasis: base crest + streamline trails (the flow
//               channel feeds the renderer's crest-driven material), plus
//               a touch of emphasis where the sim BUILT ridges,
//          .ba reserved (zero), matching shell.hlsl.

#include "common.hlsl"

Texture2D<float4>   baseShell  : register(t0);
Texture2D<float4>   overlayTex : register(t1);
SamplerState        linearSamp : register(s2);
RWTexture2D<float4> outShell   : register(u3);

cbuffer ComposeUniforms : register(b4) {
  float res;       // output resolution (matches baseShell)
  float ov_amp;    // overlay .r -> world units
  float trail;     // flow channel -> crest emphasis gain
  float h_min;     // floor for the composed height (approx -0.5 * R)
};

[numthreads(8, 8, 1)]
void main(uint3 gid : SV_DispatchThreadID) {
  int R = int(res);
  if (gid.x >= (uint)R || gid.y >= (uint)R) return;

  float2 uv = (float2(gid.xy) + 0.5) / res;
  float4 base = baseShell.Load(int3(gid.xy, 0));
  float4 ov = overlayTex.SampleLevel(linearSamp, uv, 0);

  float h = max(base.r + ov.r * ov_amp, h_min);
  // Streamline trails ride the crest channel; built ridges crest a little
  // on their own so deposition reads as fresh material.
  float crest = saturate(base.g + trail * saturate(ov.g)
                         + 0.35 * saturate(ov.r));

  outShell[gid.xy] = float4(h, crest, 0.0, 0.0);
}
