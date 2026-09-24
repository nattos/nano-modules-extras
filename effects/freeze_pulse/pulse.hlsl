// warp.legacy.freeze_pulse — the pulse + blend pass.
//
// Samples the FROZEN frame through a centred scale (the "pop") + jitter
// translation, applies a brightness/contrast pop, then composites it over the
// LIVE input using a selectable blend mode at `blend_phase` (= intensity ×
// alpha × envelope). When blend_phase is 0 the output is the untouched live
// frame, so the effect is a passthrough between pulses. v2 of Wire "Freeze
// Pulse".

Texture2D<float4>   liveTex   : register(t0);
Texture2D<float4>   frozenTex : register(t1);
SamplerState        samp      : register(s2);
RWTexture2D<float4> outputTex : register(u3);

cbuffer Uniforms : register(b4) {
  float scale;       // frozen-frame zoom (the pop)
  float trans_x;
  float trans_y;
  float bright;
  float contrast;
  float blend_phase; // intensity × alpha × envelope
  int   mode;        // 0 RGB, 1 Hard Light, 2 Difference, 3 Difference-I, 4 Lighten
  int   _p0;
};

float3 hardLight(float3 b, float3 s) {
  return float3(
    s.x < 0.5 ? 2.0 * b.x * s.x : 1.0 - 2.0 * (1.0 - b.x) * (1.0 - s.x),
    s.y < 0.5 ? 2.0 * b.y * s.y : 1.0 - 2.0 * (1.0 - b.y) * (1.0 - s.y),
    s.z < 0.5 ? 2.0 * b.z * s.z : 1.0 - 2.0 * (1.0 - b.z) * (1.0 - s.z));
}

[numthreads(8, 8, 1)]
void main(uint3 gid : SV_DispatchThreadID) {
  uint W, H;
  outputTex.GetDimensions(W, H);
  if (gid.x >= W || gid.y >= H) return;

  float2 uv = (float2(gid.xy) + 0.5) / float2(W, H);
  float4 live = liveTex.SampleLevel(samp, uv, 0.0);

  if (blend_phase <= 1e-4) { outputTex[gid.xy] = live; return; }

  // Transform the frozen frame: centred scale + jitter.
  float invs = (scale > 1e-4) ? (1.0 / scale) : 1.0;
  float2 src = (uv - 0.5) * invs + 0.5 + float2(trans_x, trans_y);
  float3 froz = frozenTex.SampleLevel(samp, src, 0.0).rgb;

  // Brightness/contrast pop.
  froz = (froz - 0.5) * (1.0 + contrast) + 0.5 + bright;
  froz = saturate(froz);

  float3 b = live.rgb;
  float3 blended;
  if      (mode == 1) blended = hardLight(b, froz);
  else if (mode == 2) blended = abs(b - froz);
  else if (mode == 3) blended = 1.0 - abs(b - froz);
  else if (mode == 4) blended = max(b, froz);
  else                blended = froz;            // RGB / replace

  float3 outRgb = lerp(b, blended, saturate(blend_phase));
  outputTex[gid.xy] = float4(outRgb, live.a);
}
