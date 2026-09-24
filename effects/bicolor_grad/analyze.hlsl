// color.legacy.bicolor_grad — analysis pass (single thread).
//
// Reads the 64-bin hue histogram, picks the dominant "major" hue (with
// parabolic sub-bin interpolation), then an angularly-isolated "minor" hue,
// reconstructs saturated colours for each (blended toward the neutral colour
// by a confidence weight), and locates the spatial centroids of major- vs
// minor-coloured regions to derive a gradient Center + Direction. Results are
// temporally smoothed against the previous frame's analysis. One invocation.
//
// analysis buffer layout (floats):
//   [0..2]  MajorColor.rgb   [3..5]  MinorColor.rgb
//   [6,7]   Center.xy        [8,9]   Direction.xy
//   [10]    confidence       [11]    initialized flag

Texture2D<float4>         inputTex : register(t0);
SamplerState             samp     : register(s1);
StructuredBuffer<int>     hist     : register(t2);   // read-only
RWStructuredBuffer<float> outBuf   : register(u3);

cbuffer U : register(b4) {
  float nr, ng, nb;     // neutral colour
  float smoothing;      // 0 = snap, 1 = (nearly) frozen
  float dirSign;        // +1 or -1 (Reverse)
  float isolation;      // minor-hue isolation + membership window
  float colorSat;       // confidence saturation
  float _p;
};

static const float3 kYIQToR = float3(1.0,  0.956,  0.621);
static const float3 kYIQToG = float3(1.0, -0.272, -0.647);
static const float3 kYIQToB = float3(1.0, -1.107,  1.704);
static const float  PI = 3.14159265358979323846;
static const int    BUCKETS = 64;
static const int    GRID = 16;

float3 colorFromHue(float h01) {
  float hue = h01 * PI * 2.0;
  float Q = sin(hue);
  float I = cos(hue);
  float3 yIQ = float3(1.0, I, Q);          // lightness 1, full chroma
  return saturate(float3(dot(yIQ, kYIQToR), dot(yIQ, kYIQToG), dot(yIQ, kYIQToB)) * 0.5);
}

float circDist(float a, float b) {          // circular distance, [0, 0.5]
  return abs(frac(a - b + 0.5) - 0.5);
}

[numthreads(1, 1, 1)]
void main(uint3 id : SV_DispatchThreadID) {
  // ---- major bucket ----
  int   bestBucket = 0;
  float bestWeight = 0.0;
  for (int i = 0; i < BUCKETS; ++i) {
    float w = (float)hist[i];
    if (w > bestWeight) { bestWeight = w; bestBucket = i; }
  }
  float w0 = (float)hist[(bestBucket - 1) & (BUCKETS - 1)];
  float w1 = max(bestWeight, 1.0);
  float w2 = (float)hist[(bestBucket + 1) & (BUCKETS - 1)];
  float majorBucket = clamp((w2 - w0) / w1, -1.0, 1.0) * 0.5 + (float)bestBucket;
  float majorWeight = bestWeight;

  // ---- minor bucket (isolation-weighted) ----
  int   mBest = 0;
  float mBestW = 0.0;
  for (int j = 0; j < BUCKETS; ++j) {
    float iso = saturate(abs(j - majorBucket) / BUCKETS / max(0.0001, isolation));
    iso *= iso;
    float w = (float)hist[j] * iso;
    if (w > mBestW) { mBestW = w; mBest = j; }
  }
  float mw0 = (float)hist[(mBest - 1) & (BUCKETS - 1)];
  float mw1 = max(mBestW, 1.0);
  float mw2 = (float)hist[(mBest + 1) & (BUCKETS - 1)];
  float minorBucket = clamp((mw2 - mw0) / mw1, -1.0, 1.0) * 0.5 + (float)mBest;
  float minorWeight = mBestW;

  float majorHue = majorBucket / BUCKETS;
  float minorHue = minorBucket / BUCKETS;

  float3 neutral = float3(nr, ng, nb);
  float majorConf = saturate(atan(majorWeight * colorSat) / PI * 2.0);
  float minorConf = saturate(atan(minorWeight * colorSat) / PI * 2.0);
  float3 majorColor = lerp(neutral, colorFromHue(majorHue), majorConf);
  float3 minorColor = lerp(neutral, colorFromHue(minorHue), minorConf);

  // ---- centroids (16x16 grid membership) ----
  float window = max(0.02, isolation);
  float2 majorAcc = 0.0, minorAcc = 0.0;
  float  majorWAcc = 0.0, minorWAcc = 0.0;
  for (int y = 0; y < GRID; ++y) {
    for (int x = 0; x < GRID; ++x) {
      float2 uv = (float2(x, y) + 0.5) / (float)GRID;
      float3 rgb = max(inputTex.SampleLevel(samp, uv, 0).rgb, 0.0);
      float I = dot(rgb, float3(0.596, -0.275, -0.321));
      float Q = dot(rgb, float3(0.212, -0.523, 0.311));
      float chroma = sqrt(I * I + Q * Q);
      float hue = (abs(I) <= 1e-5 && abs(Q) <= 1e-5) ? 0.0 : atan2(Q, I);
      hue = hue / (PI * 2.0); hue -= floor(hue);
      float mjW = chroma * pow(saturate(1.0 - circDist(hue, majorHue) / window), 2.0);
      float mnW = chroma * pow(saturate(1.0 - circDist(hue, minorHue) / window), 2.0);
      majorAcc += uv * mjW; majorWAcc += mjW;
      minorAcc += uv * mnW; minorWAcc += mnW;
    }
  }
  float2 majorC = majorWAcc > 1e-4 ? majorAcc / majorWAcc : float2(0.5, 0.5);
  float2 minorC = minorWAcc > 1e-4 ? minorAcc / minorWAcc : float2(0.5, 0.5);

  float2 center = (majorC + minorC) * 0.5 * 2.0 - 1.0;
  float2 diff = majorC - minorC;
  float  len = length(diff);
  float2 dir = (len > 1e-3) ? (diff / len) : float2(1.0, 0.0);
  dir *= dirSign;

  float conf = majorConf;

  // ---- temporal smoothing against previous frame ----
  float a = (outBuf[11] > 0.5) ? saturate(1.0 - smoothing) : 1.0;
  majorColor = lerp(float3(outBuf[0], outBuf[1], outBuf[2]), majorColor, a);
  minorColor = lerp(float3(outBuf[3], outBuf[4], outBuf[5]), minorColor, a);
  center     = lerp(float2(outBuf[6], outBuf[7]), center, a);
  float2 prevDir = float2(outBuf[8], outBuf[9]);
  dir = lerp(prevDir, dir, a);
  float dl = length(dir); dir = (dl > 1e-4) ? dir / dl : float2(1.0, 0.0);
  conf = lerp(outBuf[10], conf, a);

  outBuf[0] = majorColor.r; outBuf[1] = majorColor.g; outBuf[2] = majorColor.b;
  outBuf[3] = minorColor.r; outBuf[4] = minorColor.g; outBuf[5] = minorColor.b;
  outBuf[6] = center.x;     outBuf[7] = center.y;
  outBuf[8] = dir.x;        outBuf[9] = dir.y;
  outBuf[10] = conf;
  outBuf[11] = 1.0;
}
