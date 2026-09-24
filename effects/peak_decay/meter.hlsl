// motion.peak_decay — pass 1: the meter. Per-pixel staleness tracking.
//
// State ping-pong (RGBA16F): rgb = the HELD reference colour (what the pixel
// is compared against), a = age in seconds since it last moved. On a catch
// the reference re-latches and the age snaps to 0 (instant peak-meter
// attack; the gain is applied in pass 2). Two catch modes:
//
//  - Any Change: the pixel differs from the held reference — not from last
//    frame, so a slow drift still accumulates and eventually trips. The
//    metric is full-RGB balanced against luma by `rgb_balance`.
//  - Rise Only: only an UPWARD luma edge catches — the live luma rising past
//    the held reference. The reference follows the input DOWN silently (the
//    age keeps running), so a dip-then-recovery measures its rise from the
//    bottom of the dip, while darker or chroma-only changes never reset the
//    fall. (Comparing against the decaying displayed level instead would
//    let a static bright input perpetually re-catch itself.)

#include "common.hlsl"

Texture2D<float4>   inTex     : register(t0);
Texture2D<float4>   statePrev : register(t1);
// Write-only rgba16f (format from the registerShaderSPV "rgba16float","write"
// hint; a [[vk::image_format]] pin would force read_write, forbidden for
// rgba16f — and the hint is per-shader, which is why tex_out lives in pass 2).
RWTexture2D<float4> stateNext : register(u2);

cbuffer Uniforms : register(b3) {
  float dt;           // seconds this frame (CPU-clamped against hitches)
  float amount;       // decay depth (Rise Only compares against the live gain)
  float hold;         // seconds a pixel may sit still before the fall starts
  float fall;         // sigmoid fall duration
  float threshold;    // change needed to count as motion / rise margin
  float reset;        // 1 = seed the state from the input (first frame / resize)
  float rgb_balance;  // Any Change: 0 = luma-only metering, 1 = full RGB
  float catch_mode;   // 0 = Any Change, 1 = Rise Only
};

[numthreads(8, 8, 1)]
void main(uint3 gid : SV_DispatchThreadID) {
  uint W, H;
  stateNext.GetDimensions(W, H);
  if (gid.x >= W || gid.y >= H) return;

  float3 c = inTex[gid.xy].rgb;
  if (reset > 0.5) {
    stateNext[gid.xy] = float4(c, 0.0);
    return;
  }

  float4 st = statePrev[gid.xy];
  bool caught;
  if (catch_mode < 0.5) {
    // Any Change: luma steps count at face value; a pure chroma swap is
    // weighed by rgb_balance (0.5 = the classic half-weight metering).
    float dl = abs(pd_lum(c) - pd_lum(st.rgb));
    float3 dv = abs(c - st.rgb);
    float d = max(dl, rgb_balance * max(dv.x, max(dv.y, dv.z)));
    caught = d > threshold;
  } else {
    float lc = pd_lum(c), lr = pd_lum(st.rgb);
    caught = lc > lr + threshold;
    if (!caught && lc < lr) st.rgb = c;   // follow down, keep the age
  }

  if (caught) {
    stateNext[gid.xy] = float4(c, 0.0);   // re-latch + instant snap back
  } else {
    // Cap just past the end of the fall: half precision stays sharp (dt
    // never stalls against the ulp) and the cap tracks the knobs, so
    // raising hold/fall live revives fully-decayed pixels — performable.
    float cap = hold + fall + 0.25;
    stateNext[gid.xy] = float4(st.rgb, min(st.a + dt, cap));
  }
}
