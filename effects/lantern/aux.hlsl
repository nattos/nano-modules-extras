// source.mesh.lantern — the optional outputs, run only while one is wired:
// Aux 1 and Aux 2 (each what its select picks: LTN_AUX_*) and render_outputs'
// depth (camera-space metres in .x; misses at LTN_MISS). It reads what the
// composite reads, so a picture here is that part of the main output.

#include "grade.hlsl"

Texture2D<float4>   sceneTex : register(t2);
Texture2D<float4>   fogI     : register(t3);
Texture2D<float4>   fogT     : register(t4);
Texture2D<float4>   sparkTex : register(t5);   // the flakes: radiance x cov, cov
Texture2D<float4>   glintTex : register(t6);   // the scene's glint charge per pixel
RWTexture2D<float4> aux1Tex  : register(u7);
RWTexture2D<float4> aux2Tex  : register(u8);
RWTexture2D<float4> depthTex : register(u9);

// Aux output `what` at this pixel. Pictures are graded like the main output;
// data (depth, transmittance, glint) is not. Depth: a ramp over 5 m either
// side of the structure's center, white nearest; black where rays miss.
float4 ltn_aux_out(int what, uint2 px, bool hit, float z, float3 surface, float3 I, float3 T,
                   float4 sp, float3 c) {
  if (what == LTN_AUX_DEPTH) {
    float g = hit ? saturate(0.5 - (z - aux.w) / 10.0) : 0.0;
    return float4(g, g, g, 1.0);
  }
  if (what == LTN_AUX_SURFACES) return float4(ltn_agx(hit ? surface * grade.x : 0.0, grade.w), 1.0);
  if (what == LTN_AUX_INSCATTER) return float4(ltn_agx(I * grade.x, grade.w), 1.0);
  if (what == LTN_AUX_TRANSMIT) return float4(T, 1.0);
  if (what == LTN_AUX_FLAKES) return float4(ltn_agx(sp.rgb * grade.x, grade.w), 1.0);
  if (what == LTN_AUX_GLINT) {
    float g = saturate(glintTex.Load(int3(int2(px), 0)).x);
    return float4(g, g, g, 1.0);
  }
  return float4(c * grade.x, 1.0);   // Linear: the picture before the tonemap
}

[numthreads(8, 8, 1)]
void main(uint3 gid : SV_DispatchThreadID) {
  uint W = (uint)vp.x, H = (uint)vp.y;
  if (gid.x >= W || gid.y >= H) return;

  float4 scene = sceneTex.Load(int3(int2(gid.xy), 0));
  float d = scene.a;
  bool hit = d < LTN_MISS * 0.5;
  // Camera-space depth (m) of the block hit.
  float3 ro, rd;
  ltn_camera_ray(float2(gid.xy), vp.xy, ro, rd);
  float z = hit ? d * dot(rd, cam_f.xyz) : LTN_MISS;
  if (aux.z > 0.5) depthTex[gid.xy] = float4(z, 0.0, 0.0, 1.0);
  if (aux.x < 0.5 && aux.y < 0.5) return;

  float3 I, T;
  ltn_media(fogI, fogT, gid.xy, d, I, T);
  float3 surface = hit ? scene.rgb : float3(0.0002, 0.0004, 0.0016);
  float4 sp = dust.w > 0.5 ? sparkTex.Load(int3(int2(gid.xy), 0)) : float4(0.0, 0.0, 0.0, 0.0);
  float3 c = (I + T * surface) * (1.0 - saturate(sp.a)) + sp.rgb;
  if (aux.x > 0.5) aux1Tex[gid.xy] = ltn_aux_out((int)aux.x, gid.xy, hit, z, surface, I, T, sp, c);
  if (aux.y > 0.5) aux2Tex[gid.xy] = ltn_aux_out((int)aux.y, gid.xy, hit, z, surface, I, T, sp, c);
}
