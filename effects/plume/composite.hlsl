// source.sdf.plume — final composite (fog pipeline only).
//
// Full-res: body from the scene buffer (rgb, .a = hit distance), fog from
// the half-res fog buffer (rgb = in-scatter, .a = transmittance, bilinear
// upsample — fog is soft), dust from the splat layer (color tex + exact
// depth + aggregate coverage buffers), over the input, faded by the
// global opacity.

#include "nano_hash.hlsl"
#include "common.hlsl"

Texture2D<float4>   sceneTex   : register(t0);
Texture2D<float4>   fogTex     : register(t1);
Texture2D<float4>   bgTex      : register(t2);
SamplerState        linearSamp : register(s3);
RWTexture2D<float4> outTex     : register(u4);
Texture2D<float4>   dustTex    : register(t6);   // splat layer, rgb = color
StructuredBuffer<uint> dustDepth : register(t7); // asuint(t) per pixel
StructuredBuffer<uint> dustCov   : register(t8); // 8.8 coverage sum

cbuffer CompUniforms : register(b5) {
  float opacity;
  float has_bg;
  float exposure;   // linear gain on the fog in-scatter (see below)
  float black;      // black level: >0 lift, <0 crush
  float dust_on;    // dust layer valid this frame
  float t_far;      // approx fog integration end on miss rays (see below)
  float srgb;       // encode plume's color with the sRGB OETF
  float _p1;
};

[numthreads(8, 8, 1)]
void main(uint3 gid : SV_DispatchThreadID) {
  uint W, H;
  outTex.GetDimensions(W, H);
  if (gid.x >= W || gid.y >= H) return;

  float4 bg = has_bg > 0.5 ? bgTex.Load(int3(int(gid.x), int(gid.y), 0))
                           : float4(0.0, 0.0, 0.0, 0.0);
  float4 scene = sceneTex.Load(int3(int(gid.x), int(gid.y), 0));
  float2 uv = (float2(gid.xy) + 0.5) / float2(W, H);
  // 4-tap tent upsample of the half-res fog: averages out the march's IGN
  // jitter dither (a single bilinear tap leaves a faint 2-px diamond
  // lattice in high-banding regions, e.g. backlit fog shadows). Fog is
  // soft by construction, so the extra half-texel of blur costs nothing.
  uint FW, FH;
  fogTex.GetDimensions(FW, FH);
  float2 ft = 0.5 / float2(FW, FH);
  float4 fog = (fogTex.SampleLevel(linearSamp, uv + float2( ft.x,  ft.y), 0)
              + fogTex.SampleLevel(linearSamp, uv + float2(-ft.x,  ft.y), 0)
              + fogTex.SampleLevel(linearSamp, uv + float2( ft.x, -ft.y), 0)
              + fogTex.SampleLevel(linearSamp, uv + float2(-ft.x, -ft.y), 0))
             * 0.25;

  // Miss sentinel is 6e4 — the largest sentinel that survives the RGBA16F
  // scene buffer (f16 tops out at 65504; anything bigger reads back NaN).
  bool hit = scene.a < 1.0e4;
  float3 c = hit ? scene.rgb : bg.rgb;
  float cover = hit ? 1.0 : 0.0;

  // Fog integrates in front of the body (the march stopped at its depth).
  // Exposure applies to the fog's LINEAR in-scatter here — the surface
  // color already took the same gain ahead of its shoulder in march.hlsl.
  c = fog.rgb * exposure + c * fog.a;
  cover = max(cover, 1.0 - fog.a);

  // Dust: blend the splat layer over the fogged scene, seated in the
  // atmosphere by its own depth. The fog buffer holds the FULL path
  // (in-scatter I, transmittance T) integrated to t_ref; the front
  // fraction f = t_dust/t_ref of that path veils the dust, apportioned
  // as a uniform medium: T_front = T^f, I_front = I·(1−T^f)/(1−T). A
  // near speck stays crisp, a deep one dissolves into the haze — per
  // pixel at full res, without the fog march ever seeing point
  // occluders. Alpha is the accumulated coverage: sub-pixel specks are
  // area-faded, overlapping specks build toward opaque.
  if (dust_on > 0.5) {
    uint idx = gid.y * W + gid.x;
    float A = min(float(dustCov[idx]) * (1.0 / 256.0), 1.0);
    if (A > 0.001) {
      float td = asfloat(dustDepth[idx]);
      float tref = min(scene.a, t_far);
      float f = saturate(td / max(tref, 1e-4));
      float Tf = pow(max(fog.a, 1e-4), f);
      float If = (1.0 - Tf) / max(1.0 - fog.a, 1e-4);
      float3 cd = dustTex.Load(int3(int(gid.x), int(gid.y), 0)).rgb * Tf
                + fog.rgb * exposure * If;
      c = lerp(c, cd, A);
      cover = max(cover, A);
    }
  }

  // Black level: >0 lifts the floor toward a filmic pedestal, <0 crushes
  // the darks to true black. Graded before the opacity fade (opacity 0
  // stays pure passthrough) and BEFORE the dither below, so the ±half-LSB
  // amplitude stays exact at the output quantizer no matter how the grade
  // reshapes the gradients.
  c = black >= 0.0 ? black + (1.0 - black) * c
                   : max((c + black) / (1.0 + black), 0.0);
  // Optional sRGB encode — same placement contract as march.hlsl's direct
  // path: after the grade, before the bg crossfade and the dither.
  if (srgb > 0.5) c = plm_srgb_encode(c);

  float3 outc = lerp(bg.rgb, c, opacity);
  // Triangular-PDF output dither, ±1 LSB. The soft looks are built from
  // huge shallow gradients (fog haze, grazing-sun key rolloff), and the
  // downstream 8-bit output quantizes them into visible contour bands —
  // the fog's iso-brightness lines read as "ringing" arcs around the
  // body. Uniform ±half-LSB is the bare minimum and its residual noise
  // modulation still traces the contours — worse, any smoothing in the
  // display/export path (scaled preview, screenshot resample) filters
  // the noise faster than the staircase and the terraces reappear.
  // Triangular (two decorrelated taps) linearizes the quantizer with
  // constant noise variance, so the contours stay hidden with margin.
  // One IGN tap (perceptually spread) + one white tap (IGN alone is
  // quasi-periodic — its own ripple family reads as arcs under gain,
  // and its purely high-frequency energy is the first thing a resample
  // filters out; white noise keeps masking power after smoothing).
  // Static pattern (stable in motion); costs two hashes.
  float dn = nano_ign(float2(gid.xy)) +
             nano_hash21(float2(gid.xy) + float2(97.0, 71.0)) - 1.0;
  outc += dn * (1.0 / 255.0);
  outTex[gid.xy] = float4(outc, max(bg.a, cover * opacity));
}
