// warp.legacy.d_wave — dampening-flash vertex shader.
//
// One instanced quad per particle, splatted into the SEPARATE damp texture
// (X = angle, Y = radius) — NOT the wave field. The warp pass subtracts this
// from the wave field, so a blob reads as a fast streak of REDUCED distortion.
// Thin in angle, short-elongated in radius. A flash envelope fades each blob in
// and out across its mid-radius band so it pops and vanishes rather than
// hard-appearing. Idle slots collapse outside clip space.

StructuredBuffer<float4> particles : register(t0);

cbuffer Uniforms : register(b1) {
  uint  count;
  float ang_halfwidth;   // angular half-extent (thinness)
  float rad_halflen;     // radial half-extent (short elongation)
  float grain;           // [0,1] per-particle size/strength jitter
}

struct VsOut {
  float4 pos      : SV_Position;
  float2 local    : TEXCOORD0;                 // quad-local [-1,1]
  nointerpolation float strength : TEXCOORD1;
};

[shader("vertex")]
VsOut main(uint vid : SV_VertexID, uint iid : SV_InstanceID) {
  static const float2 corners[6] = {
    float2(-1.0, -1.0), float2( 1.0, -1.0), float2(-1.0,  1.0),
    float2( 1.0, -1.0), float2( 1.0,  1.0), float2(-1.0,  1.0),
  };
  VsOut o;
  if (iid >= count) {                          // idle → collapse
    o.pos = float4(2.0, 2.0, 2.0, 1.0);
    o.local = float2(0.0, 0.0);
    o.strength = 0.0;
    return o;
  }

  float4 p = particles[iid];
  float ang = p.x, r = p.y, sj = p.z;
  float2 c = corners[vid % 6u];

  // grain jitters each flash's size AND strength (correlated).
  float lenJit = lerp(1.0, 0.4 + 1.2 * sj, grain);
  float2 center = float2(ang, r);
  float2 uv = center + c * float2(ang_halfwidth, rad_halflen * lenJit);
  o.pos = float4(uv * 2.0 - 1.0, 0.0, 1.0);
  o.local = c;

  // Flash envelope: fade in/out across the mid band (matches particles.hlsl
  // BAND_LO=0.12 / BAND_OUT=0.82) so flashes pop and vanish, not pop-in.
  float env = smoothstep(0.10, 0.24, r) * (1.0 - smoothstep(0.64, 0.82, r));
  o.strength = lerp(1.0, 0.3 + 1.4 * sj, grain) * env;
  return o;
}
