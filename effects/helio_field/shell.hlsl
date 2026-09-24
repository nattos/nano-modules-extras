// source.sdf.helio_field — shell pass: sim state → displacement shell map.
//
// The height is DERIVED from the current magnetic state every frame (no
// height accumulator — the lines ARE the field): ridges sit on the
// contour lines of the potential A (the field lines), with a fixed
// SPATIAL width. Distance to the nearest contour is estimated as
//   s ≈ frac-distance(A·k) / (k·|∇A|)   [radians]
// so the ridge profile exp(−(s/w)²) has a bounded slope amp·0.86/w no
// matter how hard the sim bunches the contours — that bound is what the
// CPU's Lipschitz factor is derived from. Where ∇A → 0 (null regions)
// s → ∞ and the lines vanish, which is also physically right.
//
// Dispatched twice per frame (512² full + 256² coarse for the bake),
// same field: A is bilinearly sampled from the same sim texture and the
// gradient uses the same FIXED sim_eps, so the two maps only differ by
// output resolution (sub-texel error the renderer's handoff band eats).
//
// Channels: .r = h (world units, ≥ 0), .g = crest (line strength + storm
// heat afterglow), .b = storm activation u (the rail leaves .ba to the
// provider; a downstream material can read the live burn), .a = 0.

#include "../plume/common.hlsl"

Texture2D<float4>   auxTex   : register(t0);  // (A, ...)
SamplerState        samp     : register(s1);
RWTexture2D<float4> shellTex : register(u2);
Texture2D<float4>   stormTex : register(t4);  // (u, v, heat)
Texture2D<float4>   dustTex  : register(t5);  // (a, b) — granule chemistry

cbuffer HShellUniforms : register(b3) {
  float res;         // target resolution (full or coarse)
  float amp;         // height amplitude, world units
  float line_k;      // contour density (lines per unit A)
  float line_w;      // ridge half-width, radians

  float base_floor;  // body skin height fraction below the lines
  float heat_gain;   // storm heat → crest glow
  float sim_eps;     // gradient half-step, radians (matches dynamics)
  float ga_cap;      // |∇A| normalizer for line brightness

  float storm_amp;   // storm curtain extrusion, world units
  float dust_amp;    // granule bump height, world units
  float dust_gain;   // granule chemical b → full bump normalization
  float _pad2;
};

void hs_frame(float3 dir, out float3 t1, out float3 t2) {
  float3 a = abs(dir.y) < 0.92 ? float3(0.0, 1.0, 0.0)
                               : float3(1.0, 0.0, 0.0);
  t1 = normalize(cross(a, dir));
  t2 = cross(dir, t1);
}

// .x = raw A (line positions keep their detail), .y = A_smooth (all
// gradient estimates — the raw field's bilinear phase noise would fur
// the walls).
float2 aux_sample(float3 d) {
  return auxTex.SampleLevel(samp, nano_oct_encode(d), 0).xy;
}

[numthreads(8, 8, 1)]
void main(uint3 gid : SV_DispatchThreadID) {
  int r = int(res);
  if (gid.x >= (uint)r || gid.y >= (uint)r) return;
  float2 uv = (float2(gid.xy) + 0.5) / res;
  float3 dir = nano_oct_decode(uv);

  float3 t1, t2;
  hs_frame(dir, t1, t2);
  float A = aux_sample(dir).x;
  float2 uvS = nano_oct_encode(dir);
  float4 storm = stormTex.SampleLevel(samp, uvS, 0);  // (u, v, heat[, kink])
  float Apu = aux_sample(normalize(dir + sim_eps * t1)).y;
  float Amu = aux_sample(normalize(dir - sim_eps * t1)).y;
  float Apv = aux_sample(normalize(dir + sim_eps * t2)).y;
  float Amv = aux_sample(normalize(dir - sim_eps * t2)).y;
  float3 gA = (t1 * (Apu - Amu) + t2 * (Apv - Amv)) / (2.0 * sim_eps);
  float gAl = max(length(gA), 1e-5);

  // Spatial distance to the nearest field line (contour of A at pitch
  // 1/line_k), then a fixed-width ridge profile.
  float tA = frac(A * line_k);
  float dA = min(tA, 1.0 - tA) / line_k;   // A-units to nearest contour
  float s = dA / gAl;                       // ≈ radians
  float ridge = exp(-(s * s) / (line_w * line_w));
  // Where the sim bunches contours tighter than the ridge width (steep
  // A-cliffs — active regions), discrete lines would render as stacked
  // terrace ripples. Blend toward a solid wall as they pack.
  float pack = saturate(2.0 * line_w * line_k * gAl);
  ridge = lerp(ridge, 1.0, pack * pack);

  // Line brightness follows field strength (active regions read hot).
  float w = gAl / (gAl + ga_cap);
  float lines = ridge * (0.35 + 0.65 * w);

  // Granules: the dust layer's b chemical, ridge-gated so the height
  // hierarchy stays lines > storms > grain (the chemistry already
  // starves them near strong field — this just keeps stragglers off
  // the walls).
  float dots = saturate(dustTex.SampleLevel(samp, uvS, 0).y * dust_gain);
  dots = dots * dots * (3.0 - 2.0 * dots);   // rounded caps, not plateaus
  float grain = dots * (1.0 - 0.75 * ridge);

  // Storm curtain: an actively burning storm (u) EXTRUDES the line it
  // rides — tall aurora-like walls along the field line. The afterglow
  // (heat) only lights the crest channel; the relief relaxes back as
  // the burn passes.
  float h = amp * (base_floor + (1.0 - base_floor) * lines)
          + storm_amp * storm.x * ridge
          + dust_amp * grain;
  float crest = saturate(lines + storm.x + heat_gain * storm.z
                         + 0.3 * grain);

  shellTex[gid.xy] = float4(h, crest, storm.x, 0.0);
}
