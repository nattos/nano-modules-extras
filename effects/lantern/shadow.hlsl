// source.mesh.lantern — core light shadow atlas.
//
// One octahedral tile per light sample (lsamp[k]); each texel stores the
// distance from that sample to the first box along its direction (LTN_MISS
// when nothing is hit; 0 when the sample sits inside a box, which then
// blocks everything). ltn_core_vis() compares against it.

#include "common.hlsl"

RWTexture2D<float4> atlas : register(u2);

[numthreads(8, 8, 1)]
void main(uint3 gid : SV_DispatchThreadID) {
  int res = (int)misc.y, cols = (int)misc.z;
  int count = (int)core_i.w;
  int2 tile = int2(gid.xy) / res;
  int k = tile.y * cols + tile.x;
  if (tile.x >= cols || k >= count) return;

  float2 uv = (float2(int2(gid.xy) - tile * res) + 0.5) / float(res);
  float3 rd = nano_oct_decode(uv);
  float3 ro = lsamp[k].xyz;

  float depth = LTN_MISS;
  int n = ltn_box_count();
  [loop] for (int i = 0; i < n; i++) {
    float tn, tf;
    int ax;
    float3 ld;
    if (ltn_box_ray(i, ro, rd, tn, tf, ax, ld))
      depth = min(depth, max(tn, 0.0));
  }
  atlas[gid.xy] = float4(depth, 0.0, 0.0, 0.0);
}
