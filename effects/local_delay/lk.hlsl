// motion.local_delay — pyramidal Lucas-Kanade flow, one level.
//
// Estimates dense optical flow between the current and previous luma at
// one pyramid level, refining the (upsampled) flow from the coarser
// level. This is the windowed structure-tensor solve that replaces the
// old single-pixel normal-flow estimate — it "looks in a larger area"
// (a 5x5 window) and is Tikhonov-regularized so flat / low-gradient
// regions inherit the coarse flow instead of spiking.
//
// Run coarse→fine: eighth (incoming = 1x1 zero), quarter (incoming =
// eighth), half (incoming = quarter). The host binds the right textures
// for each dispatch; the level resolution comes from GetDimensions, so
// one shader/PSO serves all three.
//
// Flow is stored in uv/frame (normalized) — scale-invariant, so the
// coarser level's flow carries to the finer level without rescaling.
// Internally we convert uv↔level-pixels for warping and the solve.

#include "common.hlsl"

Texture2D<float4>   lumaNow  : register(t0);   // current luma (R), this level
Texture2D<float4>   lumaPrev : register(t1);   // previous-frame luma (R), this level
Texture2D<float4>   incoming : register(t2);   // coarser flow (uv), or 1x1 zero
RWTexture2D<float4> flowOut  : register(u3);   // rgba16f: RG = flow (uv/frame)

cbuffer Uniforms : register(b4) {
  float delay_amount;  float noise_weight;     float seed;            float weight_gain;
  float vignette;      float vignette_radius;  float vignette_softness; float squash;
  float max_flow;      float align_amount;     float align_sharpness; float have_history;
  float aspect_x;      float aspect_y;         float debug_show_motion; float history_alpha;
};

#define LK_RADIUS 2        // 5x5 aggregation window
#define LK_LAMBDA 2e-3     // Tikhonov regularizer (flat regions → keep coarse flow)

[numthreads(8, 8, 1)]
void main(uint3 gid : SV_DispatchThreadID) {
  uint w, h;
  flowOut.GetDimensions(w, h);
  if (gid.x >= w || gid.y >= h) return;
  int2 dims = int2(int(w), int(h));

  // First frame: no valid previous luma → emit zero flow.
  if (have_history < 0.5) { flowOut[gid.xy] = float4(0, 0, 0, 0); return; }

  // Incoming flow from the coarser level (uv, bilinear). 1x1 zero → 0.
  uint iw, ih;
  incoming.GetDimensions(iw, ih);
  float2 uv  = (float2(gid.xy) + 0.5) / float2(w, h);
  float2 pin = uv * float2(iw, ih) - 0.5;
  float2 flow_uv_in = ld_bil_flow(incoming, pin, int2(int(iw), int(ih)));
  float2 flow_px_in = flow_uv_in * float2(w, h);   // this level's pixels

  // Windowed structure tensor: aggregate gradient products + temporal
  // term over the window. prev is warped by the incoming flow so each
  // level only solves the small residual displacement.
  float A11 = 0.0, A12 = 0.0, A22 = 0.0, b1 = 0.0, b2 = 0.0;
  [unroll] for (int dy = -LK_RADIUS; dy <= LK_RADIUS; dy++)
  [unroll] for (int dx = -LK_RADIUS; dx <= LK_RADIUS; dx++) {
    int2 q   = clamp(int2(gid.xy) + int2(dx, dy), int2(0, 0), dims - 1);
    int2 qxp = clamp(q + int2(1, 0), int2(0, 0), dims - 1);
    int2 qxm = clamp(q - int2(1, 0), int2(0, 0), dims - 1);
    int2 qyp = clamp(q + int2(0, 1), int2(0, 0), dims - 1);
    int2 qym = clamp(q - int2(0, 1), int2(0, 0), dims - 1);
    float gx = (lumaNow[uint2(qxp)].x - lumaNow[uint2(qxm)].x) * 0.5;
    float gy = (lumaNow[uint2(qyp)].x - lumaNow[uint2(qym)].x) * 0.5;
    // flow is the TRUE content displacement (prev→now), so the content now at q
    // came FROM q - flow: sample the previous frame there.
    float Iprev = ld_bil_r(lumaPrev, float2(q) - flow_px_in, dims);
    float It = lumaNow[uint2(q)].x - Iprev;
    A11 += gx * gx; A12 += gx * gy; A22 += gy * gy;
    b1  += gx * It; b2  += gy * It;
  }
  A11 += LK_LAMBDA; A22 += LK_LAMBDA;

  // Solve A * d = -b for the incremental displacement (level pixels). The sign
  // (paired with the q - flow warp above) makes the result the true content
  // motion: for content drifting +x, the flow comes out +x.
  float det = A11 * A22 - A12 * A12;
  float2 d = float2(0, 0);
  if (abs(det) > 1e-12) {
    d.x = -( A22 * b1 - A12 * b2) / det;
    d.y = -(-A12 * b1 + A11 * b2) / det;
  }

  float2 flow_uv = (flow_px_in + d) / float2(w, h);
  float m = length(flow_uv);
  if (m > max_flow && m > 1e-8) flow_uv *= max_flow / m;
  flowOut[gid.xy] = float4(flow_uv, 0.0, 0.0);
}
