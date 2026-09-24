// filter.lights_sim — sample 4 vertical LED bars from the input and render them.
//
// Per pixel:
//   * fade the input by input_opacity (the passthrough background).
//   * find the quarter q = which of the 4 vertical bars this pixel is under.
//   * if the pixel falls inside that quarter's INSET bar rectangle, find its
//     segment, sample the input at (quarter horizontal centre, segment
//     vertical centre), and paint that segment's LED colour over the pixel.

Texture2D<float4>   inputTex      : register(t0);
SamplerState        linearSampler : register(s1);
RWTexture2D<float4> outputTex     : register(u2);

cbuffer Uniforms : register(b3) {
  int   segments;
  float inset_h;
  float inset_v;
  float input_opacity;
};

[numthreads(8, 8, 1)]
void main(uint3 gid : SV_DispatchThreadID) {
  uint W, H;
  outputTex.GetDimensions(W, H);
  if (gid.x >= W || gid.y >= H) return;

  float2 uv = (float2(gid.xy) + 0.5) / float2(W, H);
  float4 base = inputTex[gid.xy];
  float3 outc = base.rgb * saturate(input_opacity);

  // Which LED bar (vertical quarter) and its horizontal centre.
  uint  q   = min(uint(uv.x * 4.0), 3u);
  float qcx = (float(q) + 0.5) * 0.25;

  // Inset bar rectangle: half-extents shrink with inset (1 → collapsed).
  float halfW = 0.125 * saturate(1.0 - inset_h);
  float halfH = 0.5   * saturate(1.0 - inset_v);

  if (abs(uv.x - qcx) <= halfW && abs(uv.y - 0.5) <= halfH) {
    int nseg = max(segments, 1);
    // Vertical position within the bar → segment index.
    float t = saturate((uv.y - (0.5 - halfH)) / max(2.0 * halfH, 1e-6));
    int   s  = min(int(t * float(nseg)), nseg - 1);
    float sy = (float(s) + 0.5) / float(nseg);
    // Resolume-style sample: horizontal centre of the quarter, vertical
    // centre of the segment.
    float3 seg = inputTex.SampleLevel(linearSampler, float2(qcx, sy), 0).rgb;
    outc = seg;
  }

  outputTex[gid.xy] = float4(outc, base.a);
}
