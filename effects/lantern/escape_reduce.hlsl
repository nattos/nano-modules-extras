// source.mesh.lantern — the escape map, box-averaged 4x4 (the source the
// widest prefilter levels tap, so their taps never skip over detail).

#include "common.hlsl"

Texture2D<float4>   esc     : register(t2);
RWTexture2D<float4> esc_low : register(u3);

[numthreads(8, 8, 1)]
void main(uint3 gid : SV_DispatchThreadID) {
  if (gid.x >= (uint)LTN_ESC_LOW || gid.y >= (uint)LTN_ESC_LOW) return;
  const int f = LTN_ESC_RES / LTN_ESC_LOW;
  float4 sum = float4(0.0, 0.0, 0.0, 0.0);
  [unroll] for (int y = 0; y < f; y++)
    [unroll] for (int x = 0; x < f; x++)
      sum += esc.Load(int3(int2(gid.xy) * f + int2(x, y), 0));
  esc_low[gid.xy] = sum / float(f * f);
}
