// source.mesh.monolith — final combine + tonemap.
//
// hdr = composite + rays (screen-style blend), shoulder tonemap
// (identity below 1.0, C1 shoulder above), written to tex_out. Pixels
// with no shape coverage AND no ray contribution copy the input
// VERBATIM — the passthrough guarantee. The rays slot is bound to a
// 1x1 zero texture when the pass is skipped (OOB Loads return zero).

Texture2D<float4>   compTex : register(t0);
Texture2D<float4>   raysTex : register(t1);
Texture2D<float4>   inTex   : register(t2);
RWTexture2D<float4> outTex  : register(u3);

cbuffer Uniforms : register(b4) {
  float4 p;   // x = has_rays, yzw unused
};

float shoulder(float x) {
  return x <= 1.0 ? x : 2.0 - 1.0 / x;
}

[numthreads(8, 8, 1)]
void main(uint3 gid : SV_DispatchThreadID) {
  uint W, H;
  outTex.GetDimensions(W, H);
  if (gid.x >= W || gid.y >= H) return;
  int3 ip = int3(int(gid.x), int(gid.y), 0);

  float4 c = compTex.Load(ip);
  // Rays land via a screen-style blend: strong shafts over dark water,
  // never a white-out over an already-bright backdrop.
  float3 add = raysTex.Load(ip).rgb * p.x * saturate(1.0 - c.rgb);
  float4 inp = inTex.Load(ip);
  if (c.a <= 0.0 && (add.x + add.y + add.z) < 1e-5) {
    outTex[gid.xy] = inp;   // bit-exact passthrough
    return;
  }

  float3 hdr = clamp(c.rgb + add, 0.0, 64.0);
  float3 tm = float3(shoulder(hdr.x), shoulder(hdr.y), shoulder(hdr.z));
  outTex[gid.xy] = float4(tm, max(inp.a, saturate(c.a)));
}
