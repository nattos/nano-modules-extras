// source.legacy.double_chamber — tracer (L block) pass. One thread per tracer.
//
// Each tracer holds a persistent seed (pos in s-space, drift angle, life). Each
// frame it drifts the seed along the flow, then re-traces a streamline of the
// field (polynomial + optional image gradient) FORWARD and REVERSE from the
// seed, writing oriented line segments into its own fixed slot range of the
// segment buffer (unused slots zeroed → degenerate in the renderer; no atomics,
// no readback). On life expiry it reseeds at a fresh random spot.

#include "common.hlsl"

RWStructuredBuffer<TracerState> tracers : register(u0);
RWStructuredBuffer<Seg>         segs     : register(u1);
Texture2D<float4>               fieldTex : register(t2);
SamplerState                    samp     : register(s3);

cbuffer Uniforms : register(b4) {
  uint  count;
  uint  max_seg;
  uint  frame_index;
  float dt;

  float field_scale;
  float field_skew;
  float field_squash;
  float field_speed;

  float to_image;
  float momentum;
  float step_speed;
  float length01;

  float time_decay;
  float adv_step;
  float color_contrib;
  float l_opacity;

  float aspect_x;
  float aspect_y;
  float tint_r;
  float tint_g;

  float tint_b;
  float reseed_spread;
  float image_smoothing;
  float gradient_descent; // 0 = follow the tangent (level curves), 1 = follow the gradient

  float value_stop;       // stop tracing where luma < this (0 = off)
  float grad_stop;        // stop tracing where field magnitude < this (0 = off)
  float time_stop_decay;  // extra life decay when a tracer stopped early (0 = off)
  float _tp0;
}

float dc_lum(float3 c) { return max(c.r, max(c.g, c.b)); }

float2 flowDir(float2 s, float2 aspect) {
  float2 f = dc_field(clamp(s * field_scale, -3.0, 3.0), field_skew, field_squash) * field_speed;
  if (to_image != 0.0) {
    float2 uv = 0.5 + s * aspect;
    float e = lerp(0.004, 0.04, saturate(image_smoothing));
    float2 du = e * aspect;
    float vl = dc_lum(fieldTex.SampleLevel(samp, saturate(uv - float2(du.x, 0)), 0).rgb);
    float vr = dc_lum(fieldTex.SampleLevel(samp, saturate(uv + float2(du.x, 0)), 0).rgb);
    float vd = dc_lum(fieldTex.SampleLevel(samp, saturate(uv - float2(0, du.y)), 0).rgb);
    float vu = dc_lum(fieldTex.SampleLevel(samp, saturate(uv + float2(0, du.y)), 0).rgb);
    float2 ed = min(uv, 1.0 - uv);
    float edgeFade = smoothstep(0.0, 0.05, min(ed.x, ed.y));
    f += float2(vr - vl, vu - vd) * to_image * 6.0 * edgeFade;
  }
  return f;
}

[numthreads(64, 1, 1)]
void main(uint3 gid : SV_DispatchThreadID) {
  uint i = gid.x;
  if (i >= count) return;

  float2 aspect = float2(aspect_x, aspect_y);
  TracerState t = tracers[i];
  float2 seed = t.a.xy;
  float  time = t.a.z;
  float  ang  = t.a.w;

  bool reseeded = false;
  if (time <= 0.0) {
    // Reseed on a uniform-area disc of radius reseed_spread (s-space).
    uint h = dc_hash3(i + 0x1234u, frame_index, 0x99u);
    float a2 = dc_unit(dc_hash(h)) * 6.28318530718;
    float rr = reseed_spread * sqrt(dc_unit(dc_hash(h ^ 0xABCu)));
    seed = float2(cos(a2), sin(a2)) * rr;
    ang  = dc_unit(dc_hash(h ^ 0x55u)) * 6.28318530718;
    // Staggered initial life so tracers don't all expire/reseed in lockstep —
    // each gets a random time in [0.15, 1], decorrelated per tracer + frame.
    time = lerp(0.15, 1.0, dc_unit(dc_hash(h ^ 0x0033u)));
    reseeded = true;
  } else {
    float2 fd = flowDir(seed, aspect);
    float fl = length(fd);
    if (fl > 1e-5) seed += (fd / fl) * adv_step * dt;
    time -= time_decay * dt;
  }

  uint base = i * max_seg;
  uint half = max_seg / 2u;
  int steps = max(2, (int)(saturate(length01) * (float)half));
  float3 tint = float3(tint_r, tint_g, tint_b);
  uint segIdx = 0u;
  bool didStop = false;

  // Forward then reverse from the seed → a streamline through it.
  [unroll(1)]
  for (int pass = 0; pass < 2; ++pass) {
    float2 pos = seed;
    float2 dir = float2(cos(ang), sin(ang)) * (pass == 0 ? 1.0 : -1.0);
    for (int k = 0; k < steps; ++k) {
      float2 fd = flowDir(pos, aspect);
      float fl = length(fd);
      // Stop conditions (parity with the L tracer): terminate the line where the
      // field is too weak ("off gradient", i.e. near a critical point) or — with
      // image coupling — too dark. grad_stop is normalized by field_speed so it's
      // in field-shape units (a fraction), independent of the speed scale.
      if (grad_stop > 0.0 && fl < grad_stop * field_speed) { didStop = true; break; }
      if (value_stop > 0.0) {
        float lum = dc_lum(fieldTex.SampleLevel(samp, saturate(0.5 + pos * aspect), 0).rgb);
        if (lum < value_stop) { didStop = true; break; }
      }
      if (fl > 1e-5) {
        float2 g = fd / fl;                       // gradient direction
        float2 tangent = float2(g.y, -g.x);       // perpendicular → along the level curve
        if (dot(tangent, dir) < 0.0) tangent = -tangent;  // keep it forward-facing
        // tangent-dominant, with a little descent along the gradient, then momentum.
        float2 target = normalize(lerp(tangent, g, saturate(gradient_descent)));
        dir = normalize(lerp(target, dir, saturate(momentum)));
      }
      float2 nextPos = pos + dir * step_speed;
      if (segIdx < max_seg) {
        float2 uv0 = 0.5 + pos * aspect;
        float2 uv1 = 0.5 + nextPos * aspect;
        float a = (1.0 - (float)k / (float)steps) * l_opacity * saturate(time);
        float3 col = lerp(float3(1, 1, 1),
                          fieldTex.SampleLevel(samp, saturate(uv0), 0).rgb,
                          saturate(color_contrib)) * tint;
        segs[base + segIdx].a = float4(uv0, uv1);
        segs[base + segIdx].b = float4(max(col, 0.0), max(a, 0.0));
        segIdx++;
      }
      pos = nextPos;
    }
  }

  for (uint z = segIdx; z < max_seg; ++z) {
    segs[base + z].a = float4(0, 0, 0, 0);
    segs[base + z].b = float4(0, 0, 0, 0);
  }

  // A tracer that stopped early (stuck in a weak/dark region) dies faster.
  if (didStop && !reseeded) time = max(time - time_stop_decay * dt, 0.0);

  t.a = float4(seed, time, ang);
  tracers[i] = t;
}
