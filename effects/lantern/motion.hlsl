// source.mesh.lantern — the blocks' motion vectors (full res), for the
// render_outputs motion leaf a motion blur reads.
//
// The point each pixel sees is carried back through its block's previous
// pose (which block: the one whose surface it lies on) and projected through
// the previous camera; the motion is where it is now minus where it was, in
// uv per frame. Water and the core get none. The flakes are laid over this
// by the spark raster.

#include "particles.hlsl"

StructuredBuffer<float4> prevBoxes : register(t2);   // 4 rows a box: center, axes
Texture2D<float4>        sceneTex  : register(t3);
RWTexture2D<float4>      motionTex : register(u5);

[numthreads(8, 8, 1)]
void main(uint3 gid : SV_DispatchThreadID) {
  if (gid.x >= (uint)vp.x || gid.y >= (uint)vp.y) return;
  float d = sceneTex.Load(int3(int2(gid.xy), 0)).a;
  float2 mv = float2(0.0, 0.0);
  if (d < LTN_MISS * 0.5) {
    float3 ro, rd;
    ltn_camera_ray(float2(gid.xy), vp.xy, ro, rd);
    float3 p = ro + rd * d;
    int best = -1;
    float bd = 0.05;          // carved points sit a little inside their box
    int n = ltn_box_count();
    [loop] for (int i = 0; i < n; i++) {
      float3 nn;
      float s = abs(ltn_box_sdf(i, p, nn));
      if (s < bd) { bd = s; best = i; }
    }
    if (best >= 0) {
      float4 r0 = boxes[best * LTN_BOX_ROWS + 0];
      float3 o = p - r0.xyz;
      float3 lq = float3(dot(o, boxes[best * LTN_BOX_ROWS + 1].xyz),
                         dot(o, boxes[best * LTN_BOX_ROWS + 2].xyz),
                         dot(o, boxes[best * LTN_BOX_ROWS + 3].xyz));
      float3 pp = prevBoxes[best * 4 + 0].xyz + lq.x * prevBoxes[best * 4 + 1].xyz +
                  lq.y * prevBoxes[best * 4 + 2].xyz + lq.z * prevBoxes[best * 4 + 3].xyz;
      float2 ppx;
      float pw;
      if (ltn_project(prev_r, prev_u, prev_f, pp, ppx, pw))
        mv = ((float2(gid.xy) + 0.5) - ppx) * pview.zw;
    }
  }
  motionTex[gid.xy] = float4(mv, 0.0, 0.0);
}
