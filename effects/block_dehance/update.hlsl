// filter.glitch.block_dehance — update pass. One thread per rect slot.
//
// Lifecycle (mirrors flash_particles):
//   life_remain > 0                        — visible; tick life down.
//   life_remain <= 0 && respawn_remain > 0 — invisible; tick respawn delay.
//   both <= 0                              — respawn: bright-seek K mask
//       samples (softmax by mask_temperature), capture position/size/life,
//       sample a dehance mode by its weight, capture per-mode jittered params.
//
// On do_reset every slot (including i >= count, so growing `count` activates
// pre-staggered slots) is seeded dead with a randomized respawn phase.
//
// A vignette-like radial spawn gate (spawn_amount/radius/softness, cover-square
// coords) scales each candidate's probability; if no candidate survives the
// gate the respawn simply defers to the next frame.

#include "common.hlsl"
#include "nano_coords.hlsl"

RWStructuredBuffer<Rect> rects   : register(u0);
Texture2D<float4>        maskTex  : register(t1);
SamplerState             linSamp  : register(s2);

cbuffer Uniforms : register(b3) {
  uint  count; uint pool_max; uint frame_index; uint do_reset;
  float dt; float mask_temperature; float life_s; float respawn_delay_s;
  float life_jitter; float rect_width; float rect_height; float rect_size_jitter;
  float mode_black_w; float mode_mosaic_w; float mode_noise_w; float mosaic_cell_size;
  float mosaic_cell_jitter; uint mask_samples; uint seed; float move_chance;
  float move_amount; float move_delay_max; float spawn_amount; float spawn_radius;
  float spawn_softness; float aspect_x; float aspect_y; float _pad0;
};

// Spawn-probability gate: a vignette-shaped radial mask over where respawns
// may land. spawn_amount is bipolar — positive confines spawning toward the
// centre (rim probability fades to 0 at +1), negative pushes it away from the
// centre (centre probability fades to 0 at -1). 0 = fully open.
float bd_spawn_prob(float2 uv) {
  if (abs(spawn_amount) <= 1e-4) return 1.0;
  float dist = length(nano_uv_to_cover_square(uv, float2(aspect_x, aspect_y)));
  float t = smoothstep(spawn_radius, spawn_radius + max(spawn_softness, 1e-4), dist);
  return (spawn_amount > 0.0) ? lerp(1.0, 1.0 - t, spawn_amount)
                              : lerp(1.0, t, -spawn_amount);
}

[numthreads(64, 1, 1)]
void main(uint3 gid : SV_DispatchThreadID) {
  uint i = gid.x;
  if (i >= pool_max) return;

  if (do_reset != 0u) {
    // Seed every slot dead, with a staggered respawn phase so the pool
    // doesn't pop in all at once.
    float phase = bd_unit(bd_pcg2(i + 0x51EDu, seed));
    Rect r;
    r.pos_size = float4(0.5, 0.5, 0.0, 0.0);
    r.state    = float4(0.0, 1.0, phase * max(respawn_delay_s, 1e-3), float(MODE_BLACK));
    r.params   = float4(mosaic_cell_size, asfloat(bd_pcg2(i, seed)), asfloat(bd_pcg2(i + 0x9E37u, seed)), -1.0);
    rects[i] = r;
    return;
  }

  if (i >= count) return;

  Rect r = rects[i];
  float life_remain    = r.state.x;
  float respawn_remain = r.state.z;

  if (life_remain > 0.0) {
    r.state.x = life_remain - dt;
    // One-time "glitch jump": once the scheduled delay elapses, the rect hops
    // once by a small random offset. params.w holds the countdown: >= 0 means
    // pending, < 0 means not-scheduled / already-moved.
    float move_remain = r.params.w;
    if (move_remain >= 0.0) {
      move_remain -= dt;
      if (move_remain < 0.0) {
        uint  mh  = bd_pcg2(asuint(r.params.y), 0x2545F491u);
        float ang = bd_unit(mh) * 6.28318530718;
        float mag = move_amount * (0.5 + 0.5 * bd_unit(bd_pcg(mh ^ 0xA5A5u)));
        r.pos_size.xy += float2(cos(ang), sin(ang)) * mag;
        move_remain = -1.0;            // fire once
      }
      r.params.w = move_remain;
    }
    rects[i] = r;
    return;
  }
  if (respawn_remain > 0.0) {
    r.state.z = respawn_remain - dt;
    rects[i] = r;
    return;
  }

  // --- Respawn: bright-seek the mask (K candidates, softmax by T). ---
  // Each candidate carries a spawn-gate probability; gated-out candidates
  // never win, and if EVERY candidate is gated out the respawn defers (the
  // slot stays dead and retries next frame with fresh candidates).
  uint K = clamp(mask_samples, 1u, 16u);
  float2 cand_uv[16];
  float  cand_lum[16];
  float  cand_p[16];
  float  max_lum = -1e9;
  for (uint k = 0u; k < K; k++) {
    uint h = bd_pcg3(i + 0xA17F2B91u, frame_index, k + seed);
    float2 uv = float2(float(h & 0xFFFFu), float((h >> 16u) & 0xFFFFu)) * (1.0 / 65536.0);
    float lum = bd_luma(maskTex.SampleLevel(linSamp, uv, 0).rgb);
    cand_uv[k] = uv; cand_lum[k] = lum; cand_p[k] = bd_spawn_prob(uv);
    max_lum = max(max_lum, lum);
  }
  float2 best_uv = cand_uv[0];
  bool   spawn_ok = false;
  if (mask_temperature <= 1e-4) {
    // Argmax: brightest among candidates that individually pass a Bernoulli
    // draw against their gate probability.
    float best = -1e9;
    for (uint k = 0u; k < K; k++) {
      if (bd_unit(bd_pcg3(i + 0x5F356495u, frame_index, k + seed)) >= cand_p[k]) continue;
      if (cand_lum[k] > best) { best = cand_lum[k]; best_uv = cand_uv[k]; spawn_ok = true; }
    }
  } else {
    float inv_t = 1.0 / max(mask_temperature, 1e-4);
    float total = 0.0, w[16];
    for (uint k = 0u; k < K; k++) { w[k] = exp((cand_lum[k] - max_lum) * inv_t) * cand_p[k]; total += w[k]; }
    if (total > 1e-6) {
      float rsel = bd_unit(bd_pcg3(i + 0xBEEF1234u, frame_index, 0xC0FFEEu + seed)) * total;
      best_uv = cand_uv[K - 1u];
      float cum = 0.0;
      for (uint k = 0u; k < K; k++) { cum += w[k]; if (rsel < cum) { best_uv = cand_uv[k]; break; } }
      spawn_ok = true;
    }
  }
  if (!spawn_ok) return;   // fully gated this frame — retry on the next

  // --- Capture geometry with jitter. ---
  float wj = bd_signed(bd_pcg2(i + 0xC2B2AE3Du, frame_index));
  float hj = bd_signed(bd_pcg2(i + 0x27D4EB2Fu, frame_index));
  float cw = max(rect_width  * (1.0 + rect_size_jitter * wj), 1e-4);
  float ch = max(rect_height * (1.0 + rect_size_jitter * hj), 1e-4);

  // --- Lifetime with jitter. ---
  float lj = bd_signed(bd_pcg2(i + 0x9E3779B1u, frame_index));
  float new_life    = max(life_s          * (1.0 + life_jitter * lj), 1e-3);
  float new_respawn = max(respawn_delay_s * (1.0 + life_jitter * bd_signed(bd_pcg2(i + 0x85EBCA77u, frame_index))), 0.0);

  // --- Sample a dehance mode by weight. ---
  float wb = max(mode_black_w, 0.0), wm = max(mode_mosaic_w, 0.0), wn = max(mode_noise_w, 0.0);
  float wtot = wb + wm + wn;
  uint mode = MODE_BLACK;
  if (wtot > 1e-6) {
    float rm = bd_unit(bd_pcg2(i + 0x165667B1u, frame_index)) * wtot;
    if (rm < wb)            mode = MODE_BLACK;
    else if (rm < wb + wm)  mode = MODE_MOSAIC;
    else                    mode = MODE_NOISE;
  }

  // --- Per-mode captured params. ---
  float cj = bd_signed(bd_pcg2(i + 0xD3F6A1C1u, frame_index));
  float cell = max(mosaic_cell_size * (1.0 + mosaic_cell_jitter * cj), 1e-4);
  uint mode_seed    = bd_pcg2(i + 0x7F4A7C15u, frame_index);
  uint flicker_seed = bd_pcg2(i + 0x94D049BBu, frame_index);

  // Roll the one-time move: scheduled (move_remain >= 0) with prob move_chance,
  // firing after a random delay in [0, move_delay_max].
  float move_remain = -1.0;
  if (bd_unit(bd_pcg2(i + 0x6D2B79F5u, frame_index)) < move_chance) {
    move_remain = bd_unit(bd_pcg2(i + 0x1B873593u, frame_index)) * max(move_delay_max, 0.0);
  }

  r.pos_size = float4(best_uv, cw, ch);
  r.state    = float4(new_life, new_life, new_respawn, float(mode));
  r.params   = float4(cell, asfloat(mode_seed), asfloat(flicker_seed), move_remain);
  rects[i] = r;
}
