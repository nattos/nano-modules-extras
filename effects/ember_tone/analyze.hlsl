// filter.color.ember_tone — Pass 2: analysis (single thread).
//
//   * The ANCHOR: the image's dominant hue — the fullest histogram bin, with
//     parabolic sub-bin interpolation (bicolor_grad's major hue). Kept as a
//     unit vector so easing it across the red/magenta wrap doesn't swing the
//     long way round; eased over `follow` seconds, snapped on the first frame.
//     An image with no chroma at all holds the last anchor. In Fixed mode the
//     anchor is the Anchor Hue slider instead.
//   * The TWO TONES, resolved once per frame: the strong hue slider (an
//     ordinary HSV-wheel hue) becomes an OkLCh angle, the weak hue is its
//     OkLCh complement turned by `weak_offset`, and the arc between them is
//     the SHORT way round — so the offset's sign picks the side of the wheel
//     the crossover passes through.

#include "nano_color.hlsl"
#include "nano_oklab.hlsl"
#include "common.hlsl"

StructuredBuffer<int>     hist   : register(t0);
RWStructuredBuffer<float> outBuf : register(u1);

cbuffer U : register(b2) {
  float dt;            // seconds since last analysis; 0 holds
  float follow;        // anchor easing time, s (0 = snap)
  float anchor_mode;   // 0 Auto, 1 Fixed
  float anchor_hue;    // HSV-wheel hue, turns (Fixed)
  float strong_hue;    // HSV-wheel hue, turns
  float weak_offset;   // turns, added to the OkLCh complement
  float _p0, _p1;
};

// An HSV-wheel hue -> the OkLCh hue of that fully saturated colour.
float oklch_hue_of(float hsv_hue) {
  float3 rgb = nano_hsv_to_rgb(float3(frac(hsv_hue), 1.0, 1.0));
  return nano_oklab_to_oklch(nano_linear_srgb_to_oklab(nano_srgb_to_linear(rgb))).z;
}

[numthreads(1, 1, 1)]
void main(uint3 id : SV_DispatchThreadID) {
  bool initialized = outBuf[2] > 0.5;
  float2 prev = float2(outBuf[0], outBuf[1]);

  // ---- anchor ----
  float2 target = prev;
  bool have = false;
  if (anchor_mode > 0.5) {
    float h = et_yiq_hue(nano_hsv_to_rgb(float3(frac(anchor_hue), 1.0, 1.0))).x;
    target = float2(cos(ET_TAU * h), sin(ET_TAU * h));
    have = true;
  } else {
    int best = 0;
    float bestW = 0.0;
    for (int i = 0; i < ET_BUCKETS; ++i) {
      float w = (float)hist[i];
      if (w > bestW) { bestW = w; best = i; }
    }
    if (bestW > 0.0) {
      float w0 = (float)hist[(best - 1) & (ET_BUCKETS - 1)];
      float w2 = (float)hist[(best + 1) & (ET_BUCKETS - 1)];
      float bucket = clamp((w2 - w0) / max(bestW, 1.0), -1.0, 1.0) * 0.5 + (float)best + 0.5;
      float h = bucket / ET_BUCKETS;
      target = float2(cos(ET_TAU * h), sin(ET_TAU * h));
      have = true;
    }
  }

  float2 dir = prev;
  if (!initialized) {
    dir = have ? target : float2(1.0, 0.0);
  } else if (have) {
    float a = (follow <= 1e-4) ? 1.0 : 1.0 - exp(-max(dt, 0.0) / follow);
    dir = lerp(prev, target, a);
    float l = length(dir);
    dir = (l > 1e-4) ? dir / l : target;
  }

  // ---- two tones ----
  float hs = oklch_hue_of(strong_hue);
  float arc = 0.5 + weak_offset;
  arc -= floor(arc + 0.5);            // -> [-0.5, 0.5)
  if (arc <= -0.5) arc += 1.0;        // -> (-0.5, 0.5]; offset 0 takes the + side
  float hw = frac(hs + arc + 1.0);

  outBuf[0] = dir.x;
  outBuf[1] = dir.y;
  outBuf[2] = 1.0;
  outBuf[3] = hs;
  outBuf[4] = hw;
  outBuf[5] = arc;
}
