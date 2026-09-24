// source.legacy.double_chamber — "Big" attractor update. One thread per Big.
//
// Runs in the same aspect-corrected s-space as the P pass. A few large slow
// bodies the P field-particles orbit: orbital drift + image-ride + radial sink
// + the soft circular boundary. Persistent, with occasional spread-respawn.

#include "common.hlsl"

RWStructuredBuffer<Particle> bigs     : register(u0);
Texture2D<float4>            inputTex : register(t1);
SamplerState                 samp     : register(s2);

cbuffer Uniforms : register(b3) {
  uint  count;
  uint  frame_index;
  float dt;
  float motion_rate;

  float big_speed;
  float big_momentum;
  float big_momentum_decay;
  float drift;

  float repel;
  float direction;
  float curl;
  float curl_dir;

  float sink;
  float boundary;
  float boundary_size;
  float boundary_stiffness;

  float boundary_speed;
  float spread;
  float ttl;
  float aspect_x;

  float aspect_y;
  float image_smoothing;
  float _p1, _p2;
}

static const float DC_BIG_VEL_MAX = 3.0;
float dc_lum(float3 c) { return max(c.r, max(c.g, c.b)); }
float2 dc_clamp_mag(float2 v, float m) { float l = length(v); return (l > m) ? v * (m / l) : v; }

[numthreads(64, 1, 1)]
void main(uint3 gid : SV_DispatchThreadID) {
  uint i = gid.x;
  if (i >= count) return;

  float2 aspect = float2(aspect_x, aspect_y);
  Particle p = bigs[i];
  float2 uv          = p.a.xy;
  float  life_remain = p.a.z;
  float  life_total  = p.a.w;
  float2 vel         = p.b.xy;

  if (life_remain > 0.0) {
    float2 s = (uv - 0.5) / max(aspect, 1e-4);
    float  r = max(length(s), 1e-4);
    float2 rad = s / r;
    float2 force = float2(0.0, 0.0);

    force += dc_perp(s) * drift;       // orbital swirl

    if (repel != 0.0 || curl != 0.0) {
      float e = lerp(0.006, 0.05, saturate(image_smoothing));
      float2 du = e * aspect;
      float vl = dc_lum(inputTex.SampleLevel(samp, saturate(uv - float2(du.x, 0)), 0).rgb);
      float vr = dc_lum(inputTex.SampleLevel(samp, saturate(uv + float2(du.x, 0)), 0).rgb);
      float vd = dc_lum(inputTex.SampleLevel(samp, saturate(uv - float2(0, du.y)), 0).rgb);
      float vu = dc_lum(inputTex.SampleLevel(samp, saturate(uv + float2(0, du.y)), 0).rgb);
      float2 g = float2(vr - vl, vu - vd) * 6.0;
      float2 ed = min(uv, 1.0 - uv);
      float edgeFade = smoothstep(0.0, 0.05, min(ed.x, ed.y));
      force += g * repel * direction * edgeFade;
      force += dc_perp(g) * curl * curl_dir * edgeFade;
    }

    force += rad * sink;

    if (boundary > 1e-6) {
      float over = max((r - boundary_size) * boundary_stiffness, 0.0);
      force += -rad * atan(over) * boundary_speed * boundary;
    }

    vel = lerp(force * big_speed, vel * big_momentum_decay, saturate(big_momentum));
    vel = dc_clamp_mag(vel, DC_BIG_VEL_MAX);
    s += vel * dt * motion_rate;
    uv = 0.5 + s * aspect;
    life_remain -= dt;
  } else {
    uint h = dc_hash3(i + 0x68E31DA4u, frame_index, 0x9Au);
    float ang = dc_unit(dc_hash(h)) * 6.28318530718;
    float rr = spread * (0.4 + 0.6 * dc_unit(dc_hash(h ^ 0x1234u)));
    float2 s = float2(cos(ang), sin(ang)) * rr;
    uv = 0.5 + s * aspect;   // no clamp — see p_update spawn note
    float4 capt = inputTex.SampleLevel(samp, uv, 0);
    uint packed = dc_pack_rgbz(capt.rgb, dc_unit(h));
    p.b.w = asfloat(packed);
    float lj = dc_signed(dc_hash2(i, frame_index));
    life_remain = max(ttl * 20.0 * (1.0 + 0.3 * lj), 0.1);
    life_total  = life_remain;
    vel = float2(0.0, 0.0);
  }

  p.a = float4(uv, life_remain, life_total);
  p.b.xy = vel;
  bigs[i] = p;
}
