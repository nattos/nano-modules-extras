// filter.legacy.sphr_blur — "SPHR Expand" pass (ported from the Wire ISF).
//
// A latitude-dependent HORIZONTAL blur. Treating the frame as an
// equirectangular (lat/long) map of a sphere, a fixed angular step on the
// sphere maps to an ever-WIDER horizontal UV step as you approach the poles
// (top/bottom of the frame) and a narrow one at the equator (middle). So the
// horizontal blur radius grows toward the top and bottom edges — which is what
// keeps a blur "seam-correct" on dome content, and gives a distinctive
// edge-softening look even off-sphere (the team's note: "amazingly useful even
// off-sphere; the math is likely wrong but the look is fine").
//
// Ported nearly verbatim from SPHR Expand.isf. The GLSL mat3 rotations are
// inlined as explicit component arithmetic (column-major GLSL → plain math) to
// avoid any row/column-major confusion. x wraps (equirect seam) — sampled with
// a Repeat sampler after a manual frac().

Texture2D<float4>   inputTex   : register(t0);
SamplerState        wrapSampler : register(s1);   // Repeat / Linear
RWTexture2D<float4> outputTex  : register(u2);

cbuffer Uniforms : register(b3) {
  float blur_size;   // sphere angular blur amount (0 → identity)
  float quality;     // sample-grid density
  float render_w;    // RENDERSIZE.x (major axis, for the pixel-width estimate)
  float _pad;
};

static const float PI  = 3.14159265358979;
static const float PI2 = 6.28318530717958;

float wrap05(float x) { return x - round(x); }

// latlonTranspose(xy, delta): convert a lat/long UV to a sphere point, rotate
// by delta (delta.x → roll about Y, delta.y → pitch about X), reproject to a
// lat/long UV. Verbatim port of the ISF helper.
float2 latlonTranspose(float2 xy, float2 delta) {
  float pitch = (xy.y - 0.5) * PI;
  float yaw   = (xy.x - 0.5) * 2.0 * PI;
  // spos = (cos(pitch)*sin(yaw), sin(pitch), cos(pitch)*cos(yaw))
  float cp0 = cos(pitch), sp0 = sin(pitch);
  float3 spos = float3(cp0 * sin(yaw), sp0, cp0 * cos(yaw));

  float sphereRoll  = delta.x * PI;
  float spherePitch = delta.y * PI;
  float cR = cos(sphereRoll),  sR = sin(sphereRoll);
  float cP = cos(spherePitch), sP = sin(spherePitch);

  // w = rotY(roll) * spos
  float3 w = float3(cR * spos.x - sR * spos.z, spos.y, sR * spos.x + cR * spos.z);
  // out = rotX(pitch) * w
  float3 o = float3(w.x, cP * w.y - sP * w.z, sP * w.y + cP * w.z);

  float phi   = atan2(o.z, o.x);
  float theta = acos(clamp(o.y, -1.0, 1.0));
  float2 xyOut = float2(2.0 - (phi + PI) / (2.0 * PI) * 2.0, 1.0 - theta / PI);
  xyOut.x = xyOut.x * 0.5 - 0.75;
  return xyOut;
}

[numthreads(8, 8, 1)]
void main(uint3 gid : SV_DispatchThreadID) {
  uint W, H;
  outputTex.GetDimensions(W, H);
  if (gid.x >= W || gid.y >= H) return;

  float2 xy = (float2(gid.xy) + 0.5) / float2(W, H);

  if (blur_size <= 0.0) {
    outputTex[gid.xy] = inputTex.SampleLevel(wrapSampler, xy, 0.0);
    return;
  }

  float majorSize = render_w;
  float h = blur_size;

  // Horizontal UV half-width corresponding to an angular step `h` at this
  // latitude (xy.y). standardXY.x is pinned to 0.25 (the ISF does this).
  float2 standardXY = float2(0.25, xy.y);
  float2 dy  = latlonTranspose(standardXY, float2(0.0, h));
  float2 dxy = float2(abs(wrap05(standardXY.x - dy.x)), 0.0);

  float2 xy00 = xy - dxy;
  float2 xy11 = xy + dxy;
  float2 uv00 = xy00 - floor(xy00);
  float2 uv11 = xy11 - floor(xy11);
  float2 uvDelta = uv11 - uv00;
  uvDelta -= round(uvDelta);
  uv11 = uv00 + uvDelta;
  float2 searchSize = (abs(uvDelta) * majorSize - 0.5);

  int maxSampleGrid = int(quality * 16.0);
  float rawGridSize = searchSize.x * quality;
  int coarseGridSize = max(0, min(maxSampleGrid, int(floor(rawGridSize))));
  float fineGridSize = clamp(rawGridSize - float(coarseGridSize), 0.0, 1.0);
  int gridSizeX = 1 + coarseGridSize * 2;

  float4 colorOut;
  if (gridSizeX <= 1) {
    colorOut = inputTex.SampleLevel(wrapSampler, xy, 0.0);
  } else {
    float4 acc = float4(0.0, 0.0, 0.0, 0.0);
    float totalWeight = 0.0;
    [loop] for (int gx = 0; gx < gridSizeX && gx < 64; ++gx) {
      float t = float(gx) / float(max(1, gridSizeX - 1));
      float2 uv = lerp(uv00, uv11, t);
      uv -= floor(uv);
      float4 colorSample = inputTex.SampleLevel(wrapSampler, uv, 0.0);
      float weight = abs(t - 0.5);
      weight = 1.0 - weight * 2.0;
      weight += (fineGridSize * 0.5) * float(gridSizeX - 2) / float(gridSizeX);
      weight = clamp(weight, 0.0, 1.0);
      acc += colorSample * weight;
      totalWeight += weight;
    }
    colorOut = (totalWeight > 1e-5) ? acc / totalWeight
                                    : inputTex.SampleLevel(wrapSampler, xy, 0.0);
  }
  outputTex[gid.xy] = colorOut;
}
