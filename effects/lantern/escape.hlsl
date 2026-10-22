// source.mesh.lantern — the core's escape map (for the fog).
//
// An octahedral map seen from the core's CENTER: per texel, 2x2 rays out
// through the boxes. r = the fraction that escape to open water, g = the
// blocked rays' summed depth over the ray count, so (1 - r) * mean blocked
// depth. Both average linearly, which is what vis.hlsl's prefilter needs.
//
// A block that strays near the center (within Clear) lets the rays through,
// fading out the nearer it comes: one overlapping the center would otherwise
// block every ray at once and mute the whole core.

#include "common.hlsl"

RWTexture2D<float4> esc : register(u2);

// How much of the core's light box i stops: all of it beyond Clear from the
// center (vol_k.z), fading to none as it nears the center or covers it.
float ltn_core_block(int i, float3 c) {
  if (vol_k.z <= 0.0) return 1.0;
  float4 r0 = boxes[i * LTN_BOX_ROWS + 0];
  float3 o = c - r0.xyz;
  float3 q = abs(float3(dot(o, boxes[i * LTN_BOX_ROWS + 1].xyz),
                        dot(o, boxes[i * LTN_BOX_ROWS + 2].xyz),
                        dot(o, boxes[i * LTN_BOX_ROWS + 3].xyz))) -
             float3(boxes[i * LTN_BOX_ROWS + 1].w, boxes[i * LTN_BOX_ROWS + 2].w,
                    boxes[i * LTN_BOX_ROWS + 3].w);
  float d = length(max(q, 0.0)) + min(max(q.x, max(q.y, q.z)), 0.0);
  return smoothstep(0.0, vol_k.z, d);
}

[numthreads(8, 8, 1)]
void main(uint3 gid : SV_DispatchThreadID) {
  if (gid.x >= (uint)LTN_ESC_RES || gid.y >= (uint)LTN_ESC_RES) return;
  float3 ro = core_p.xyz;
  float open = 0.0, depth_sum = 0.0;
  [unroll] for (int s = 0; s < 4; s++) {
    float2 uv = (float2(gid.xy) + float2(0.25 + 0.5 * float(s & 1), 0.25 + 0.5 * float(s >> 1))) /
                float(LTN_ESC_RES);
    float3 rd = nano_oct_decode(uv);
    float depth = LTN_MISS, T = 1.0;
    int n = ltn_box_count();
    [loop] for (int i = 0; i < n; i++) {
      float tn, tf;
      int ax;
      float3 ld;
      if (ltn_box_ray(i, ro, rd, tn, tf, ax, ld)) {
        float a = ltn_core_block(i, ro);
        if (a <= 0.0) continue;
        T *= 1.0 - a;
        depth = min(depth, max(tn, 0.0));
      }
    }
    open += T;
    depth_sum += (1.0 - T) * (depth >= LTN_MISS ? 0.0 : depth);
  }
  esc[gid.xy] = float4(open * 0.25, depth_sum * 0.25, 0.0, 0.0);
}
