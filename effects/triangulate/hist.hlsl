// triangulate — per-frame histogram of the (rough soft-normalized) ridge and
// corner responses. Feeds the percentile pass that auto-levels them so ridges
// and corners land on a common distribution (no more hand-tuned gains), giving
// coverage of BOTH extended ridges and rare maxima.
Texture2D<float4>        featRaw : register(t0);   // g=ridge_s  b=corner_s
RWStructuredBuffer<uint> hist    : register(u1);   // [0,bins) ridge, [bins,2bins) corner

cbuffer HistUniforms : register(b2) {
  uint u_w;
  uint u_h;
  uint u_bins;
  uint u_pad;
};

[numthreads(8, 8, 1)]
void main(uint3 gid : SV_DispatchThreadID) {
  if (gid.x >= u_w || gid.y >= u_h) return;
  float4 f = featRaw.Load(int3(gid.xy, 0));
  uint br = min((uint)(saturate(f.g) * (float)u_bins), u_bins - 1u);
  uint bc = min((uint)(saturate(f.b) * (float)u_bins), u_bins - 1u);
  uint o;
  InterlockedAdd(hist[br], 1u, o);
  InterlockedAdd(hist[u_bins + bc], 1u, o);
}
