// source.mesh.lantern — the core's escape map (for the fog).
//
// An octahedral map seen from the core's CENTER: per texel, 2x2 rays out
// through the boxes. r = the fraction that escape to open water, g = the
// blocked rays' summed depth over the ray count, so (1 - r) * mean blocked
// depth. Both average linearly, which is what vis.hlsl's prefilter needs.

#include "common.hlsl"

RWTexture2D<float4> esc : register(u2);

[numthreads(8, 8, 1)]
void main(uint3 gid : SV_DispatchThreadID) {
  if (gid.x >= (uint)LTN_ESC_RES || gid.y >= (uint)LTN_ESC_RES) return;
  float3 ro = core_p.xyz;
  float open = 0.0, depth_sum = 0.0;
  [unroll] for (int s = 0; s < 4; s++) {
    float2 uv = (float2(gid.xy) + float2(0.25 + 0.5 * float(s & 1), 0.25 + 0.5 * float(s >> 1))) /
                float(LTN_ESC_RES);
    float3 rd = nano_oct_decode(uv);
    float depth = LTN_MISS;
    int n = ltn_box_count();
    [loop] for (int i = 0; i < n; i++) {
      float tn, tf;
      int ax;
      float3 ld;
      if (ltn_box_ray(i, ro, rd, tn, tf, ax, ld))
        depth = min(depth, max(tn, 0.0));
    }
    bool escaped = depth >= LTN_MISS;
    open += escaped ? 1.0 : 0.0;
    depth_sum += escaped ? 0.0 : depth;
  }
  esc[gid.xy] = float4(open * 0.25, depth_sum * 0.25, 0.0, 0.0);
}
