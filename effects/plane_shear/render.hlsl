// warp.plane_shear — render / shear warp pass.
//
// Reads the latched plane (center C, unit normal n) and the CPU-timed shear
// amounts, and warps the image by inverse-mapping each output pixel back
// through the rigid translation of its half. The signed `dir` morphs the per-
// half motion between normal (rift/overlap) and tangent (slip):
//   D_A = mA * (-dir*n + (1-|dir|)*t),  D_B = mB * ( dir*n - (1-|dir|)*t)
// where t = perp(n). dir=+1 → halves together (overlap); dir=-1 → apart (rift);
// dir=0 → opposite slip along the plane.

#include "common.hlsl"

Texture2D<float4>       inputTex : register(t0);
SamplerState            samp     : register(s1);   // Linear / ClampToEdge
StructuredBuffer<float> plane    : register(t2);
RWTexture2D<float4>     outTex    : register(u3);

cbuffer U : register(b4) {
  float aspect_x, aspect_y;
  float dir;            // signed direction, [-1, 1]
  float mA, mB;         // signed per-half translation magnitude (× shear amount)
  float rift_fill;      // 0 transparent / 1 original / 2 edge-stretch / 3 mirror
  float overlap_mode;   // 0 A-on-top / 1 blend / 2 additive
  float debug_show;
  float edge_fill;      // border reveal: 0 transparent / 1 original / 2 stretch / 3 mirror
  float tint;           // 0 = off → 1 = full per-side colour tint
  float tintA_r, tintA_g, tintA_b;   // side A (proj ≥ 0) tint colour
  float tintB_r, tintB_g, tintB_b;   // side B (proj < 0) tint colour
  float tint_mode;      // 0 = multiply / 1 = add / 2 = blend
  float _p0, _p1, _p2;
};

float4 sampleCover(float2 x, float2 aspect) {
  return inputTex.SampleLevel(samp, nano_cover_square_to_uv(x, aspect), 0);
}

// Sample a covered half at source `src`. If the source lands INSIDE the image
// it's real content; if it falls OUTSIDE (the translated half pulled a viewport
// border into view) apply edge_fill: transparent / original@q / edge-stretch
// (clamp) / mirror.
float4 sampleHalf(float2 src, float2 q, float2 aspect) {
  float2 uv = nano_cover_square_to_uv(src, aspect);
  bool inb = uv.x >= 0.0 && uv.x <= 1.0 && uv.y >= 0.0 && uv.y <= 1.0;
  if (inb) return inputTex.SampleLevel(samp, uv, 0);
  int ef = (int)(edge_fill + 0.5);
  if (ef == 0) return float4(0.0, 0.0, 0.0, 0.0);          // transparent
  if (ef == 1) return sampleCover(q, aspect);              // original at this pixel
  if (ef == 3) {                                           // mirror across the border
    float2 muv = 1.0 - abs(frac(uv * 0.5) * 2.0 - 1.0);
    return inputTex.SampleLevel(samp, muv, 0);
  }
  if (ef == 4) return float4(0.0, 0.0, 0.0, 1.0);          // black
  return inputTex.SampleLevel(samp, uv, 0);                // edge stretch (clamp)
}

[numthreads(8, 8, 1)]
void main(uint3 gid : SV_DispatchThreadID) {
  uint W, H;
  outTex.GetDimensions(W, H);
  if (gid.x >= W || gid.y >= H) return;
  float2 aspect = float2(aspect_x, aspect_y);

  float2 uv = (float2(gid.xy) + 0.5) / float2(W, H);
  float2 q  = nano_uv_to_cover_square(uv, aspect);

  float2 C = float2(plane[0], plane[1]);
  float2 n = float2(plane[2], plane[3]);
  float2 t = float2(-n.y, n.x);

  float ad = abs(dir);
  float2 D_A = mA * (-dir * n + (1.0 - ad) * t);
  float2 D_B = mB * ( dir * n - (1.0 - ad) * t);

  // Inverse map: which half's pre-image lands on this output pixel?
  float2 qa = q - D_A; float projA = dot(qa - C, n); bool coverA = projA >= 0.0;
  float2 qb = q - D_B; float projB = dot(qb - C, n); bool coverB = projB <  0.0;

  float4 col;
  if (coverA && coverB) {
    float4 a = sampleHalf(qa, q, aspect), b = sampleHalf(qb, q, aspect);
    int om = (int)(overlap_mode + 0.5);
    if      (om == 1) col = 0.5 * (a + b);                              // blend
    else if (om == 2) col = float4(saturate(a.rgb + b.rgb), max(a.a, b.a)); // additive
    else              col = a;                                          // A on top
  } else if (coverA) {
    col = sampleHalf(qa, q, aspect);
  } else if (coverB) {
    col = sampleHalf(qb, q, aspect);
  } else {
    // Rift band — neither half covers this pixel.
    int rf = (int)(rift_fill + 0.5);
    if (rf == 0) {
      col = float4(0.0, 0.0, 0.0, 0.0);                                 // transparent
    } else if (rf == 1) {
      col = sampleCover(q, aspect);                                             // original
    } else if (rf == 3) {
      col = sampleCover(q - 2.0 * dot(q - C, n) * n, aspect);                   // mirror across plane
    } else if (rf == 4) {
      col = float4(0.0, 0.0, 0.0, 1.0);                                 // black
    } else {
      // Edge-stretch: extend the nearer half's edge column across the gap.
      float distA = -projA;   // how far qa is on the wrong side of the plane
      float distB =  projB;
      if (distA <= distB) col = sampleCover(qa - n * projA, aspect);
      else                col = sampleCover(qb - n * projB, aspect);
    }
  }

  // Per-side colour tint: which side of the plane this output pixel sits on.
  {
    float projq = dot(q - C, n);
    float3 tc = (projq >= 0.0) ? float3(tintA_r, tintA_g, tintA_b)
                               : float3(tintB_r, tintB_g, tintB_b);
    float3 tinted = (tint_mode < 0.5) ? (col.rgb * tc)
                  : (tint_mode < 1.5) ? (col.rgb + tc)
                  : tc;                                // blend toward the flat colour
    col.rgb = lerp(col.rgb, tinted, tint);
  }

  if (debug_show > 0.5) {
    float proj = dot(q - C, n);
    // ~1.5px line in cover-square units (short axis reference).
    float eps = 1.5 * max(aspect.x, aspect.y) / float(min(W, H));
    float onLine = 1.0 - smoothstep(0.0, eps, abs(proj));
    col.rgb = lerp(col.rgb, float3(1.0, 0.15, 0.15), onLine);
    col.a = max(col.a, onLine);
  }

  outTex[gid.xy] = col;
}
