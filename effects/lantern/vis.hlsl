// source.mesh.lantern — the fog's prefiltered core visibility.
//
// A big light's shadow in the water, at a point r out along direction w from
// the core, is the escape map blurred over a disk: a light point offset x
// from the center reaches the point through the shell (~LTN_SHELL out) at
// direction w + x (1 - shell / r) / shell, so the visible fraction of the
// light is the escape map averaged over a disk of angular radius
// rho(r) = LTN_PENUMBRA * R (r - shell) / (r * shell) — sharp at the shell,
// widening to R / shell far away (ltn_fog_rho).
//
// This pass bakes that average at four radii, rho_max / 8, / 4, / 2 and
// rho_max (the far limit), so the march reads a smooth visibility with a
// few filtered samples a step instead of estimating it with random shadow
// lookups (which is where its grain came from). Each level keeps the
// escaped fraction (visF) and the mean depth of what blocked the rest
// (visD): water in front of a blocker is still lit.

#include "common.hlsl"

Texture2D<float4>   esc     : register(t2);
Texture2D<float4>   esc_low : register(t3);
SamplerState        samp    : register(s4);
RWTexture2D<float4> visF    : register(u5);
RWTexture2D<float4> visD    : register(u6);

static const int LTN_VIS_TAPS = 48;

[numthreads(8, 8, 1)]
void main(uint3 gid : SV_DispatchThreadID) {
  if (gid.x >= (uint)LTN_ESC_RES || gid.y >= (uint)LTN_ESC_RES) return;
  float3 w = nano_oct_decode((float2(gid.xy) + 0.5) / float(LTN_ESC_RES));
  float3 t1 = normalize(cross(w, abs(w.z) < 0.9 ? float3(0.0, 0.0, 1.0) : float3(1.0, 0.0, 0.0)));
  float3 t2 = cross(w, t1);
  float rho_max = ltn_fog_rho_max();

  float4 F, D;
  [unroll] for (int l = 0; l < 4; l++) {
    float rho = rho_max * exp2(float(l) - 3.0);
    // Taps ~0.26 rho apart: the 256 map resolves the narrow levels, the
    // 4x-averaged one feeds the wide ones without aliasing.
    bool low = rho > 0.12;
    float2 acc = float2(0.0, 0.0);
    [loop] for (int i = 0; i < LTN_VIS_TAPS; i++) {
      float th = rho * sqrt((float(i) + 0.5) / float(LTN_VIS_TAPS));
      float a = float(i) * 2.39996323;
      float3 d = w * cos(th) + (t1 * cos(a) + t2 * sin(a)) * sin(th);
      float2 uv = nano_oct_encode(d);
      acc += (low ? esc_low.SampleLevel(samp, uv, 0.0) : esc.SampleLevel(samp, uv, 0.0)).rg;
    }
    acc /= float(LTN_VIS_TAPS);
    F[l] = acc.x;
    D[l] = min(acc.y / max(1.0 - acc.x, 1e-3), 1.0e3);
  }
  visF[gid.xy] = F;
  visD[gid.xy] = D;
}
