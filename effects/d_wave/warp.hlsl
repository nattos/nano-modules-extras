// warp.legacy.d_wave — warp + composite pass.
//
// Reads the stateful polar wave field (field.hlsl) and SUBTRACTS the transient
// dampening-flash layer (blob_vs/fs) from it, then radially distorts the input
// by the result: each pixel's centred coordinate is scaled in/out by a factor
// driven by the local (dampened) wave strength, so the propagating ripples pull
// the image inward in concentric arcs while the flashes punch fast streaks of
// calm. Composites the distorted image over the untouched input by render_alpha.

Texture2D<float4>   inputTex : register(t0);
Texture2D<float4>   field    : register(t1);   // stateful wave field
Texture2D<float4>   damp     : register(t2);   // transient dampening flashes
SamplerState        sampIn   : register(s3);   // Linear + ClampToEdge
SamplerState        sampF    : register(s4);   // Linear + Repeat (angle wraps)
RWTexture2D<float4> outTex   : register(u5);

cbuffer Uniforms : register(b6) {
  float aspect;        // vp_w / vp_h — circularizes the polar lookup
  float distortion;    // warp magnitude [0,1]
  float scale;         // ring spatial size: radius → row divisor
  float squeeze;       // radial offset of the ring pattern [-1,1]

  float render_alpha;  // opacity of the distorted layer over the input
  float debug_field;   // overlay the (dampened) field strength [0,1]
  float center_x;      // distortion anchor (uv offset from centre)
  float center_y;

  float damp_amount;   // how strongly the flashes subtract from the field
  float _d0, _d1, _d2;
}

static const float DW_PI    = 3.14159265358979323846;
static const float DW_GAIN  = 5.0;   // matches the original's baked distortion gain

[numthreads(8, 8, 1)]
void main(uint3 gid : SV_DispatchThreadID) {
  uint W, H;
  outTex.GetDimensions(W, H);
  if (gid.x >= W || gid.y >= H) return;

  float2 uv = (float2(gid.xy) + 0.5) / float2(W, H);
  float2 anchor = 0.5 + float2(center_x, center_y);
  float2 c = uv - anchor;            // centred on the anchor
  float  ax = c.x * aspect;          // aspect-correct so rings are round

  // Polar lookup: column = angle, row = radius.
  float ang = atan2(c.y, ax) / (2.0 * DW_PI) + 0.5;   // [0,1)
  float r   = length(float2(ax, c.y));
  float row = clamp((r + squeeze * 0.5) / max(scale, 1e-3), 0.002, 0.998);
  float2 polar = float2(frac(ang), row);

  // Wave field minus the transient dampening flashes (clamped — a flash can
  // calm the distortion to zero but never invert it).
  float wave = field.SampleLevel(sampF, polar, 0).r;
  float flash = damp.SampleLevel(sampF, polar, 0).r;
  float LI = max(wave - damp_amount * flash, 0.0);

  // Radial warp: ripples pull the image inward (factor < 1) proportional to the
  // local field strength. Clamp so a strong accumulation can't fully invert.
  float warpFactor = clamp(1.0 - DW_GAIN * LI * distortion, 0.15, 2.0);
  float2 sampleUV = c * warpFactor + anchor;

  float4 base = inputTex[gid.xy];
  float4 dist = inputTex.SampleLevel(sampIn, sampleUV, 0);
  float4 outc = lerp(base, dist, render_alpha);

  // Debug: overlay the (dampened) field strength as a red ramp.
  outc.rgb = lerp(outc.rgb, float3(saturate(LI), 0.0, 0.0), debug_field);

  outTex[gid.xy] = outc;
}
