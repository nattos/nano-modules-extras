// source.particles.flash_particles — update pass.
//
// One thread per particle slot. Three states:
//
//   life_remain > 0  — visible; decay life by dt.
//   life_remain <= 0 && respawn_remain > 0 — invisible; decay respawn.
//   both <= 0        — time to respawn: random-sample mask for the
//                      brightest of K candidates, capture uv as the new
//                      position, sample inputTex for the captured color,
//                      and re-roll all the jittered geometry/color
//                      parameters from the current uniform values.
//
// CPU populates initial state (random respawn phase per slot, zero life)
// when the count grows; this shader handles every subsequent step.

#include "common.hlsl"

RWStructuredBuffer<Particle> particles : register(u0);
Texture2D<float4>            maskTex   : register(t1);
Texture2D<float4>            inputTex  : register(t2);
SamplerState                 linearSampler : register(s3);

cbuffer Uniforms : register(b4) {
  uint  count;
  uint  frame_index;
  float dt;
  // 0 → argmax (greedy: always pick brightest candidate, sharpest
  // adherence to the mask). >0 samples from softmax(luma / T) over
  // the K candidates; very large values → uniform random.
  float mask_temperature;

  // Lifetime
  float life;
  float respawn_delay;
  float life_jitter;     // applies to both life_total and respawn_total
  float _pad_l0;

  // Geometry — captured at spawn
  float width;
  float height;
  float global_scale;
  float width_jitter;

  float height_jitter;
  float rotation_rad;
  float rotation_jitter_rad;
  float _pad_g0;

  // Color jitters — captured at spawn (strength of per-particle randomization)
  float hue_jitter;
  float brightness_jitter;
  float saturation_jitter;
  float alpha_jitter;
};

static const uint MASK_SAMPLES = 8u;

[numthreads(64, 1, 1)]
void main(uint3 gid : SV_DispatchThreadID) {
  uint i = gid.x;
  if (i >= count) return;

  Particle p = particles[i];
  float life_remain    = p.state.y;
  float life_total     = p.state.z;
  float respawn_remain = p.state.w;

  if (life_remain > 0.0) {
    // Phase 1 — alive, decaying.
    p.state.y = life_remain - dt;
    particles[i] = p;
    return;
  }

  if (respawn_remain > 0.0) {
    // Phase 2 — dead, draining respawn delay.
    p.state.w = respawn_remain - dt;
    particles[i] = p;
    return;
  }

  // Phase 3 — respawn. Re-roll everything from current params.
  // Decorrelated hash streams via the slot index + frame counter +
  // a per-stream prime salt, so two adjacent slots don't sample the
  // same RNG sequence.
  const uint stream_pos      = 0xA17F2B91u;
  const uint stream_life     = 0x9E3779B1u;
  const uint stream_respawn  = 0x85EBCA77u;
  const uint stream_geom     = 0xC2B2AE3Du;
  const uint stream_rot      = 0x27D4EB2Fu;
  const uint stream_color    = 0x165667B1u;
  const uint stream_seed     = 0xD3F6A1C1u;

  // --- Random-sample the mask: K candidates, pick one ---
  // First pass collects all candidates and tracks the running max
  // luma. We need both the per-candidate luma (for the softmax
  // weighting) and the max (for numerical stability when exponentiating).
  float2 cand_uv[MASK_SAMPLES];
  float  cand_lum[MASK_SAMPLES];
  float  max_lum = -1e9;
  for (uint k = 0u; k < MASK_SAMPLES; k++) {
    uint h = pf_pcg_hash3(i + stream_pos, frame_index, k);
    // Split the 32-bit hash into two 16-bit u/v draws — uncorrelated
    // enough for K=8 candidates without doing two PCG rounds.
    float ux = float(h & 0xFFFFu) * (1.0 / 65536.0);
    float uy = float((h >> 16u) & 0xFFFFu) * (1.0 / 65536.0);
    float2 uv = float2(ux, uy);
    float3 c = maskTex.SampleLevel(linearSampler, uv, 0).rgb;
    float lum = dot(c, float3(0.299, 0.587, 0.114));
    cand_uv[k]  = uv;
    cand_lum[k] = lum;
    if (lum > max_lum) max_lum = lum;
  }

  float2 best_uv;
  if (mask_temperature <= 1e-4) {
    // T ≈ 0 → greedy argmax. Cheap, exactly preserves the original
    // brightest-of-K behavior (and avoids exp-of-huge-number when
    // the user actually wants strict mask-following).
    float best_lum = cand_lum[0];
    best_uv = cand_uv[0];
    for (uint k = 1u; k < MASK_SAMPLES; k++) {
      if (cand_lum[k] > best_lum) {
        best_lum = cand_lum[k];
        best_uv  = cand_uv[k];
      }
    }
  } else {
    // Softmax over candidates: w[k] = exp((lum[k] - max_lum) / T).
    // Subtracting max keeps the largest exp() at 1 — no overflow at
    // small T, no underflow that wipes the distribution at large.
    // Inverse-CDF sample with a fresh hash decorrelated from the
    // candidate-position draws.
    float weights[MASK_SAMPLES];
    float total = 0.0;
    float inv_t = 1.0 / max(mask_temperature, 1e-4);
    for (uint k = 0u; k < MASK_SAMPLES; k++) {
      float w = exp((cand_lum[k] - max_lum) * inv_t);
      weights[k] = w;
      total += w;
    }
    uint  rh = pf_pcg_hash3(i + 0xBEEF1234u, frame_index, 0xC0FFEEu);
    float r  = pf_unit(rh) * total;
    // Last candidate is the fallback if floating-point sums round us
    // just past `total` on the final entry — shouldn't happen with
    // sensible weights but cheap insurance.
    best_uv  = cand_uv[MASK_SAMPLES - 1u];
    float cum = 0.0;
    for (uint k = 0u; k < MASK_SAMPLES; k++) {
      cum += weights[k];
      if (r < cum) {
        best_uv = cand_uv[k];
        break;
      }
    }
  }

  // --- Capture color at the chosen uv from the COLOR input ---
  // (This deliberately samples inputTex even when mask_in is wired to
  // a different texture — the user wants the captured color tied to the
  // visual frame, not the mask.)
  float4 capt = inputTex.SampleLevel(linearSampler, best_uv, 0);

  // --- Lifetime with jitter ---
  // pf_signed in [-1, +1]. Multiplied by life_jitter so the captured
  // total scales by (1 - life_jitter .. 1 + life_jitter). max() clamps
  // to a tiny positive lifetime so saturation math doesn't divide by 0.
  float lj = pf_signed(pf_pcg_hash2(i + stream_life,    frame_index));
  float rj = pf_signed(pf_pcg_hash2(i + stream_respawn, frame_index));
  float new_life_total    = max(life          * (1.0 + life_jitter * lj), 1e-3);
  float new_respawn_total = max(respawn_delay * (1.0 + life_jitter * rj), 0.0);

  // --- Geometry capture ---
  uint  gh   = pf_pcg_hash2(i + stream_geom, frame_index);
  float gj_w = pf_signed(gh);
  float gj_h = pf_signed(pf_pcg_hash(gh ^ 0xFEEDFACEu));
  float captured_w = max(width  * global_scale * (1.0 + width_jitter  * gj_w), 1e-4);
  float captured_h = max(height * global_scale * (1.0 + height_jitter * gj_h), 1e-4);

  // --- Rotation capture ---
  float rot_jit = pf_signed(pf_pcg_hash2(i + stream_rot, frame_index));
  float captured_rot = rotation_rad + rot_jit * rotation_jitter_rad;

  // --- Color jitter capture ---
  uint ch = pf_pcg_hash2(i + stream_color, frame_index);
  // Hue offset ∈ [-1, +1] then scaled by hue_jitter (effectively in turns).
  float hue_off  = pf_signed(ch)                              * hue_jitter;
  float bri_mult = 1.0 + pf_signed(pf_pcg_hash(ch ^ 0xA1B2u)) * brightness_jitter;
  float sat_mult = 1.0 + pf_signed(pf_pcg_hash(ch ^ 0xC3D4u)) * saturation_jitter;
  float alpha_mult = 1.0 + pf_signed(pf_pcg_hash(ch ^ 0xE5F6u)) * alpha_jitter;

  // --- Per-particle frame-jitter seed (stable across this lifetime) ---
  uint frame_seed = pf_pcg_hash2(i + stream_seed, frame_index);

  // --- Commit ---
  p.pos_size  = float4(best_uv, captured_w, captured_h);
  p.captured  = capt;
  p.state     = float4(captured_rot, new_life_total, new_life_total, new_respawn_total);
  p.jitters   = float4(hue_off, bri_mult, sat_mult, alpha_mult);
  p.meta      = float4(new_respawn_total, asfloat(frame_seed), 0.0, 0.0);
  particles[i] = p;
}
