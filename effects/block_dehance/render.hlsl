// filter.glitch.block_dehance — render pass. Per pixel, find the (last) alive rect
// covering it and "dehance" the input there: black-fill, mosaic downres, or
// noise — the mode captured per rect at spawn. An optional hard duty-cycle
// flicker stutters each rect on/off out of phase.

#include "common.hlsl"

Texture2D<float4>   inputTex  : register(t0);
SamplerState        linSamp   : register(s1);
RWTexture2D<float4> outputTex : register(u2);
StructuredBuffer<Rect> rects  : register(t4);

cbuffer Uniforms : register(b3) {
  uint  count; uint pool_max; uint tick_index; uint debug_show;
  float time; float flicker_rate_hz; float flicker_duty; float noise_intensity;
  float fill_r; float fill_g; float fill_b; float fill_a;
  uint  noise_temporal; uint noise_color_mode; float _pad0; float _pad1;
};

// Hard duty-cycle flicker for a rect, out of phase per flicker_seed.
bool bd_flicker_on(uint flicker_seed) {
  if (flicker_rate_hz <= 1e-4) return true;          // 0 Hz → continuously on
  float phase = bd_unit(flicker_seed);
  return frac(time * flicker_rate_hz + phase) < saturate(flicker_duty);
}

[numthreads(8, 8, 1)]
void main(uint3 gid : SV_DispatchThreadID) {
  uint W, H;
  outputTex.GetDimensions(W, H);
  if (gid.x >= W || gid.y >= H) return;

  float2 uv = (float2(gid.xy) + 0.5) / float2(W, H);
  float4 base = inputTex[gid.xy];
  float3 outc = base.rgb;
  int    debug_mode = -1;

  uint n = min(count, pool_max);
  for (uint i = 0u; i < n; i++) {
    Rect r = rects[i];
    if (r.state.x <= 0.0) continue;                  // not alive
    float2 c = r.pos_size.xy, hs = r.pos_size.zw * 0.5;
    float2 d = abs(uv - c);
    if (d.x > hs.x || d.y > hs.y) continue;          // not covered

    uint flicker_seed = asuint(r.params.z);
    if (!bd_flicker_on(flicker_seed)) continue;

    uint mode      = uint(r.state.w + 0.5);
    uint mode_seed = asuint(r.params.y);

    if (mode == MODE_BLACK) {
      outc = lerp(outc, float3(fill_r, fill_g, fill_b), saturate(fill_a));
    } else if (mode == MODE_MOSAIC) {
      float cell = max(r.params.x, 1e-4);
      float2 corner = c - hs;
      float2 local  = uv - corner;
      float2 cell_uv = (floor(local / cell) + 0.5) * cell;
      outc = inputTex.SampleLevel(linSamp, corner + cell_uv, 0).rgb;
    } else { // MODE_NOISE
      uint t = (noise_temporal != 0u) ? tick_index : 0u;
      uint hh = bd_pcg3(gid.x + mode_seed, gid.y * 9781u + 0x1234u, t);
      float3 nn = float3(bd_unit(hh), bd_unit(bd_pcg(hh ^ 0xABCDu)), bd_unit(bd_pcg(hh ^ 0x5A5Au)));
      if (noise_color_mode == 1u) {                  // grayscale
        nn = float3(bd_luma(nn), bd_luma(nn), bd_luma(nn));
      } else if (noise_color_mode == 2u) {           // luma_preserve
        float sl = bd_luma(base.rgb);
        nn = nn * sl / max(bd_luma(nn), 1e-3);
      }
      outc = lerp(outc, nn, saturate(noise_intensity));
    }
    debug_mode = int(mode);
  }

  if (debug_show != 0u && debug_mode >= 0) {
    // 1-px outline coloured by mode (red=black, green=mosaic, blue=noise).
    bool edge = false;
    for (uint i = 0u; i < n; i++) {
      Rect r = rects[i];
      if (r.state.x <= 0.0) continue;
      float2 c = r.pos_size.xy, hs = r.pos_size.zw * 0.5;
      float2 d = abs(uv - c);
      float2 px = 1.0 / float2(W, H);
      bool inside = d.x <= hs.x && d.y <= hs.y;
      bool inner  = d.x <= hs.x - px.x && d.y <= hs.y - px.y;
      if (inside && !inner) edge = true;
    }
    if (edge) {
      float3 oc = (debug_mode == 0) ? float3(1, 0, 0)
                : (debug_mode == 1) ? float3(0, 1, 0) : float3(0, 0, 1);
      outc = oc;
    }
  }

  outputTex[gid.xy] = float4(saturate(outc), base.a);
}
