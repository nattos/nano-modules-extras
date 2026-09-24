// filter.reconstruct.line — pass 5c: the shared centerline (fp16-safe).
//
// Per-pixel Newton offsets diverge on a band's inflection ring; averaging the
// projected center positions (confidence-weighted) gives every pixel of a band
// ONE consistent centerline. The prototype blurs ABSOLUTE pixel coordinates,
// which collapses in fp16 at HD (coord ~1920 has ~1px fp16 resolution). We do
// the centroid in RELATIVE (gid-local) coordinates instead — a direct sigma-2.5
// Gaussian gather over the vote weights Wc, all quantities small (≤ window
// radius), so fp16 is exact. cax-xx = Σ K(dp)[w_ctr(q)·dp + w_ctr·delta·n(q)] / Σ.

#include "common.hlsl"

Texture2D<float4>   wc : register(t0);   // (w_ctr, w_ctr*delta*nxr, w_ctr*delta*nyr, 0)
Texture2D<float4>   s0 : register(t1);   // smoothed (cos2t, sin2t, w_est, -)
Texture2D<float4>   m0 : register(t2);   // raw (cos2t, sin2t, w_est, delta)
RWTexture2D<float4> sd : register(u3);   // (delta_shared, trust, 0, 0)

static const int   CTR_R    = 8;         // sigma 2.5 → 3-sigma window
static const float CTR_INV2 = 1.0 / (2.0 * 2.5 * 2.5);

[numthreads(8, 8, 1)]
void main(uint3 gid : SV_DispatchThreadID) {
  uint w, h;
  sd.GetDimensions(w, h);
  if (gid.x >= w || gid.y >= h) return;
  int2 p  = int2(gid.xy);
  int2 hi = int2(w - 1, h - 1);

  float numX = 0.0, numY = 0.0, bw = 0.0;
  [loop] for (int dy = -CTR_R; dy <= CTR_R; dy++)
    [loop] for (int dx = -CTR_R; dx <= CTR_R; dx++) {
      int2 q = clamp(p + int2(dx, dy), int2(0, 0), hi);
      float k = exp(-float(dx * dx + dy * dy) * CTR_INV2);
      float4 W = wc[q];                       // w_ctr, w_ctr*delta*nxr, w_ctr*delta*nyr
      numX += k * (W.x * float(dx) + W.y);    // relative offset + projected center
      numY += k * (W.x * float(dy) + W.z);
      bw   += k * W.x;
    }

  float bwe = bw + 1e-6;
  float caxr = numX / bwe;   // = cax - xx (weighted-mean relative x offset)
  float cayr = numY / bwe;   // = cay - yy

  // Project onto the SMOOTHED line normal.
  float4 S0 = s0[gid.xy];
  float c2s = S0.x, s2s = S0.y;
  float nrm = sqrt(c2s * c2s + s2s * s2s + 1e-12);
  float nxs = sqrt(clamp(0.5 * (1.0 + c2s / nrm), 0.0, 1.0));
  float nys = sign(s2s) * sqrt(clamp(0.5 * (1.0 - c2s / nrm), 0.0, 1.0));
  float delta_shared = caxr * nxs + cayr * nys;

  float trust = bw / (bw + 0.02);            // fall back to raw where unsupported
  float draw  = m0[gid.xy].w;                // raw per-pixel delta
  float outd  = clamp(delta_shared * trust + draw * (1.0 - trust), -6.0, 6.0);

  sd[gid.xy] = float4(outd, trust, 0.0, 0.0);
}
