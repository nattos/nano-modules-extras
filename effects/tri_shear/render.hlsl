// warp.tri_shear — render / shear warp pass (one triangle edge per dispatch).
//
// Copy of plane_shear's render, differing only in that the plane is selected from
// a 3-line buffer by `line_index` (the host chains three dispatches, ping-ponging
// textures: in→tmpA→tmpB→out). Shear math, coverage and fill modes are identical.

#include "../plane_shear/common.hlsl"

Texture2D<float4>       inputTex : register(t0);
SamplerState            samp     : register(s1);   // Linear / ClampToEdge
StructuredBuffer<float> plane    : register(t2);   // 3 lines × (center.xy, normal.xy)
RWTexture2D<float4>     outTex    : register(u3);

cbuffer U : register(b4) {
  float aspect_x, aspect_y;
  float dir;            // signed direction, [-1, 1]
  float mA, mB;         // signed per-half translation magnitude (× shear amount)
  float rift_fill;      // 0 transparent / 1 original / 2 edge-stretch / 3 mirror / 4 black
  float overlap_mode;   // 0 A-on-top / 1 blend / 2 additive
  float debug_show;
  float edge_fill;      // border reveal: 0 transparent / 1 original / 2 stretch / 3 mirror / 4 black
  float line_index;     // which of the 3 triangle edges this pass shears
  float tint;           // 0 = off → 1 = full per-region colour tint (multiply)
  float tint0_r, tint0_g, tint0_b;   // wedge outside edge 0
  float tint1_r, tint1_g, tint1_b;   // wedge outside edge 1
  float tint2_r, tint2_g, tint2_b;   // wedge outside edge 2
  float tintC_r, tintC_g, tintC_b;   // inside the triangle (center)
  float tint_mode;      // 0 = multiply / 1 = add / 2 = blend
};

float4 sampleCover(float2 x, float2 aspect) {
  return inputTex.SampleLevel(samp, nano_cover_square_to_uv(x, aspect), 0);
}

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

  int base = (int)(line_index + 0.5) * 4;
  float2 C = float2(plane[base + 0], plane[base + 1]);
  float2 n = float2(plane[base + 2], plane[base + 3]);
  float2 t = float2(-n.y, n.x);

  float ad = abs(dir);
  float2 D_A = mA * (-dir * n + (1.0 - ad) * t);
  float2 D_B = mB * ( dir * n - (1.0 - ad) * t);

  float2 qa = q - D_A; float projA = dot(qa - C, n); bool coverA = projA >= 0.0;
  float2 qb = q - D_B; float projB = dot(qb - C, n); bool coverB = projB <  0.0;

  float4 col;
  if (coverA && coverB) {
    float4 a = sampleHalf(qa, q, aspect), b = sampleHalf(qb, q, aspect);
    int om = (int)(overlap_mode + 0.5);
    if      (om == 1) col = 0.5 * (a + b);
    else if (om == 2) col = float4(saturate(a.rgb + b.rgb), max(a.a, b.a));
    else              col = a;
  } else if (coverA) {
    col = sampleHalf(qa, q, aspect);
  } else if (coverB) {
    col = sampleHalf(qb, q, aspect);
  } else {
    int rf = (int)(rift_fill + 0.5);
    if (rf == 0) {
      col = float4(0.0, 0.0, 0.0, 0.0);
    } else if (rf == 1) {
      col = sampleCover(q, aspect);
    } else if (rf == 3) {
      col = sampleCover(q - 2.0 * dot(q - C, n) * n, aspect);
    } else if (rf == 4) {
      col = float4(0.0, 0.0, 0.0, 1.0);
    } else {
      float distA = -projA;
      float distB =  projB;
      if (distA <= distB) col = sampleCover(qa - n * projA, aspect);
      else                col = sampleCover(qb - n * projB, aspect);
    }
  }

  // Per-region colour tint, on the final pass — classify the pixel against all 3
  // edges: inside all → center; else the most-outer edge's wedge.
  if ((int)(line_index + 0.5) == 2) {
    float2 C0 = float2(plane[0], plane[1]),  n0 = float2(plane[2],  plane[3]);
    float2 C1 = float2(plane[4], plane[5]),  n1 = float2(plane[6],  plane[7]);
    float2 C2 = float2(plane[8], plane[9]),  n2 = float2(plane[10], plane[11]);
    float p0 = dot(q - C0, n0), p1 = dot(q - C1, n1), p2 = dot(q - C2, n2);
    float3 tc;
    if (p0 < 0.0 && p1 < 0.0 && p2 < 0.0)  tc = float3(tintC_r, tintC_g, tintC_b);
    else if (p0 >= p1 && p0 >= p2)         tc = float3(tint0_r, tint0_g, tint0_b);
    else if (p1 >= p2)                     tc = float3(tint1_r, tint1_g, tint1_b);
    else                                   tc = float3(tint2_r, tint2_g, tint2_b);
    float3 tinted = (tint_mode < 0.5) ? (col.rgb * tc)
                  : (tint_mode < 1.5) ? (col.rgb + tc)
                  : tc;                                // blend toward the flat colour
    col.rgb = lerp(col.rgb, tinted, tint);
  }

  // Debug: on the final pass, overlay ALL three discovered edge lines.
  if (debug_show > 0.5 && (int)(line_index + 0.5) == 2) {
    float eps = 1.5 * max(aspect.x, aspect.y) / float(min(W, H));
    float onLine = 0.0;
    [unroll] for (int k = 0; k < 3; ++k) {
      float2 Ck = float2(plane[k * 4 + 0], plane[k * 4 + 1]);
      float2 nk = float2(plane[k * 4 + 2], plane[k * 4 + 3]);
      onLine = max(onLine, 1.0 - smoothstep(0.0, eps, abs(dot(q - Ck, nk))));
    }
    col.rgb = lerp(col.rgb, float3(1.0, 0.15, 0.15), onLine);
    col.a = max(col.a, onLine);
  }

  outTex[gid.xy] = col;
}
