// source.light.tingle_top — update pass. One thread per particle slot.
//
// Spawn position is sampled from a MIXTURE of up to 4 active "voices" (the
// polyphony). Each voice is a y-distribution computed on the CPU and passed
// in `voices[v] = (y_peak, sigma_trail, sigma_lead, weight)`:
//   * a SUSTAIN voice is a small gaussian at the top band,
//   * a RELEASE voice is a split-normal (asymmetric) gaussian whose window
//     bursts/accelerates downward — a wave that drains off screen.
// Per spawn we pick a voice by weight (so density stays constant and splits
// across voices), then sample y via Box–Muller with sign-dependent sigma.
// x is uniform across the target bar's slice. Particles live + fade in place
// (optional per-particle velocity drift on top).

#include "common.hlsl"

RWStructuredBuffer<Particle> parts : register(u0);

cbuffer Uniforms : register(b1) {
  uint  count; uint pool_max; uint frame_index; uint do_reset;
  float dt; float life_s; float respawn_delay_s; float size;
  float size_jitter; float life_jitter; float hue_jitter; float _pad0;
  float vel_x; float vel_y; float vel_x_jitter; float vel_y_jitter;
  uint  respect_bounds; uint seed; uint _pad1; uint _pad2;
  uint4 bar_nv;          // active voice count per bar
  float4 voices[16];     // [bar*4 + slot] = (y_peak, sigma_trail (up), sigma_lead (down), weight)
};

// Sample a spawn y from the particle's OWN bar's voice sub-pool. Per-bar, so
// the loop length is still <= 4 voices. Returns -1 if that bar has no voice.
float sampleVoiceY(uint i, uint bar) {
  uint base = bar * 4u;
  uint nv = (bar == 0u) ? bar_nv.x : (bar == 1u) ? bar_nv.y : (bar == 2u) ? bar_nv.z : bar_nv.w;
  if (nv == 0u) return -1.0;
  float total = 0.0;
  for (uint v = 0u; v < nv; v++) total += voices[base + v].w;
  if (total <= 1e-6) return -1.0;

  float r = tt_unit(tt_pcg3(i + 0x3C6EF35Fu, frame_index, seed)) * total;
  uint vi = 0u; float cum = 0.0;
  for (uint v = 0u; v < nv; v++) { cum += voices[base + v].w; if (r < cum) { vi = v; break; } vi = v; }
  uint idx = base + vi;

  // Standard normal (Box–Muller), then split-normal: spread up by sigma_trail
  // (toward the top), down by sigma_lead (toward the bottom).
  float u1 = tt_unit(tt_pcg3(i + 0x9E3779B9u, frame_index, 0x1u + seed));
  float u2 = tt_unit(tt_pcg3(i + 0x85EBCA6Bu, frame_index, 0x2u + seed));
  float z = sqrt(-2.0 * log(max(u1, 1e-7))) * cos(6.28318530718 * u2);
  float y = voices[idx].x + (z < 0.0 ? z * voices[idx].y : z * voices[idx].z);
  return max(y, 0.0);          // never above the top; below-screen culled by caller
}

// Roll a fresh spawn into `p`. The particle's bar is its slot's partition
// (i & 3); it samples only that bar's voices. Leaves the particle DEAD when
// the bar has no on-screen voice sample.
void respawn(inout Particle p, uint i) {
  uint  bar = i & 3u;
  float py  = sampleVoiceY(i, bar);
  float rx  = tt_unit(tt_pcg3(i + 0xA17F2B91u, frame_index, 0x11u + seed));
  float px  = (float(bar) + rx) * 0.25;

  float sz = max(size * (1.0 + size_jitter * tt_signed(tt_pcg2(i + 0xC2B2AE3Du, frame_index))), 1e-5);
  float lt = max(life_s * (1.0 + life_jitter * tt_signed(tt_pcg2(i + 0x85EBCA77u, frame_index))), 1e-3);
  float rs = max(respawn_delay_s * (1.0 + life_jitter * tt_signed(tt_pcg2(i + 0x27D4EB2Fu, frame_index))), 0.0);
  float ho = tt_signed(tt_pcg2(i + 0x165667B1u, frame_index)) * hue_jitter;
  float vx = vel_x * (1.0 + vel_x_jitter * tt_signed(tt_pcg2(i + 0x6D2B79F5u, frame_index)));
  float vy = vel_y * (1.0 + vel_y_jitter * tt_signed(tt_pcg2(i + 0x1B873593u, frame_index)));

  bool on_screen = (py >= 0.0 && py <= 1.0);
  p.a = float4(px, max(py, 0.0), sz, on_screen ? lt : 0.0);
  p.b = float4(lt, on_screen ? rs : max(rs, 0.02), ho, vx);
  p.c = float4(vy, float(bar), 0.0, 0.0);
}

[numthreads(64, 1, 1)]
void main(uint3 gid : SV_DispatchThreadID) {
  uint i = gid.x;
  if (i >= pool_max) return;

  if (do_reset != 0u) {
    // Analytic prewarm: spawn fresh, then advance by a random elapsed age
    // (life partially spent + position drifted by velocity) so the pool
    // starts in steady state. Skip when no voice placed the particle.
    Particle p;
    respawn(p, i);
    if (p.a.w > 0.0) {
      float life_total = p.b.x;
      float age = tt_unit(tt_pcg2(i + 0x77EEAA11u, seed + 0xABCu)) * life_total;
      p.a.x += p.b.w * age;
      p.a.y += p.c.x * age;
      p.a.w = life_total - age;
      if (respect_bounds != 0u) {
        float bar = p.c.y, bl = bar * 0.25, br = bl + 0.25;
        if (p.a.x < bl || p.a.x > br || p.a.y < 0.0 || p.a.y > 1.0) {
          p.a.w = 0.0;
          p.b.y = tt_unit(tt_pcg2(i + 0x51EDu, seed)) * max(respawn_delay_s, 1e-3);
        }
      }
    }
    parts[i] = p;
    return;
  }
  if (i >= count) return;

  Particle p = parts[i];
  float life_remain = p.a.w;

  if (life_remain > 0.0) {
    float vx = p.b.w, vy = p.c.x;
    p.a.x += vx * dt;
    p.a.y += vy * dt;
    p.a.w = life_remain - dt;
    if (respect_bounds != 0u) {
      float bar = p.c.y, bl = bar * 0.25, br = bl + 0.25;
      if (p.a.x < bl || p.a.x > br || p.a.y < 0.0 || p.a.y > 1.0) p.a.w = -1.0;
    }
    parts[i] = p;
    return;
  }
  if (p.b.y > 0.0) {              // respawn delay
    p.b.y -= dt;
    parts[i] = p;
    return;
  }
  respawn(p, i);
  parts[i] = p;
}
