// source.particles.flow_swarm — update pass. One thread per particle slot.
//
// Alive + in-bounds: sample the flow field at the particle's uv, derive the
// "effective" flow (undertow may re-aim / re-scale it for some depths), then
// advance velocity by the chosen ACCELERATION MODE:
//   Velocity — velocity chases the field (momentum blends the old velocity;
//              momentum 0 = a clean sim of the field).
//   Force    — the field is a FORCE (acceleration); velocity integrates it,
//              divided by the particle's weight (mass). The field becomes a
//              hint and inertia/overshoot emerge.
// Dead / out-of-bounds: respawn at a fresh random uv, capture input color +
// roll a depth, reset lifetime/size, seed velocity from the field.
//
// All timing is dt-accumulated (style guide §2.1) — no time*rate.

#include "common.hlsl"

RWStructuredBuffer<Particle> particles    : register(u0);
Texture2D<float4>            flowTex       : register(t1);  // flow_field velocity (uv/s)
Texture2D<float4>            inputTex      : register(t2);  // color capture at spawn
SamplerState                 linearSampler : register(s3);
Texture2D<float4>            densityTex    : register(t5);  // last frame's crowding (.r)

cbuffer Uniforms : register(b4) {
  uint  count;
  uint  frame_index;
  float dt;
  float speed;          // multiplier on the sampled field

  float momentum;       // velocity mode: 0 = snap to field, →1 = heavy inertia
  float jitter;         // forward spray: ±wobble on speed + slight direction
  float drag;           // velocity decay per second
  float life;           // base lifetime (s)

  float life_jitter;    // ±fraction on lifetime
  float size;           // base particle size (isotropic uv, already curved C++-side)
  float size_jitter;    // ±fraction on size
  uint  seed;           // decorrelates instances

  uint  mode;           // 0 = Velocity, 1 = Force
  float weight;         // force mode: particle mass (accel = force / weight)
  float undertow_split; // 0 = no depths undertow, 1 = all particles undertow
  float undertow_polarity; // 1 = normal dir, -1 = reverse, 2 = 2× speed, …

  float undertow_curl;  // -1 = turn 90° left, 0 = unchanged, +1 = turn 90° right
  float pull;           // settle: pull velocity back toward the field flow [0,1]
  uint  interactions;   // 0 = off (skip density read), 1 = on
  float density_threshold; // crowding (≈ neighbour count) above which death kicks in

  float density_death;  // death-rate scale when over threshold (soft knee)
  float avoid;          // avoid/curl strength away from neighbours
  float avoid_curl;     // -1/+1 rotate the avoidance ±90° (swirl)
  float density_res;    // density buffer resolution (texels per axis)

  float avoid_noise;    // random jitter on the avoidance (breaks flat clumps)
  uint  substeps;       // integration substeps per frame (1 = single step)
  float dens_aspect_x;  // min/W, min/H — density is isotropic in pixels
  float dens_aspect_y;

  float stream;         // +align / -diverge velocity vs the group; 0 = off
  float stream_density; // neighbour density for ~max stream effect
  float _pad_s0;
  float _pad_s1;
};

// Max settle rate (1/s) at pull = 1, used for a framerate-independent approach.
static const float FSW_PULL_RATE  = 20.0;
// Tuning scales for the interaction forces / death.
static const float FSW_DEATH_RATE = 4.0;   // MAX death rate (1/s) at density_death=1
static const float FSW_AVOID_VEL  = 1.5;   // avoidance velocity scale (× speed)
static const float FSW_NOISE_CIRC = 0.2;   // slight isotropic part of avoid noise
static const float FSW_STREAM_RATE = 10.0; // MAX align/diverge rate (1/s) at |stream|=1

[numthreads(64, 1, 1)]
void main(uint3 gid : SV_DispatchThreadID) {
  uint i = gid.x;
  if (i >= count) return;

  Particle p = particles[i];
  float2 pos         = p.a.xy;
  float  life_remain = p.a.z;
  float  life_total  = p.a.w;
  float2 vel         = p.b.xy;
  float  size_cur    = p.b.z;
  uint   packed      = asuint(p.b.w);

  // Substepping: integrate the motion `nsub` times with dt/nsub, re-sampling
  // the field at each refined position. Everything dt-scaled so the result is
  // substep-invariant where it should be: death probability and drag are
  // dt-weighted, `pull` is an exponential approach, velocity-mode momentum is a
  // per-substep blend (msub), and stochastic kicks scale by 1/√nsub so their
  // variance holds. nsub=1 reproduces the single-step path exactly (sub 0 reuses
  // the original RNG seed and dt). The win: force mode + fast flow stop
  // overshooting field/cycle features they used to jump over.
  uint  nsub        = max(substeps, 1u);
  float dt_sub      = dt / float(nsub);
  float noise_scale = rsqrt(float(nsub));
  float msub        = (momentum < 1e-6) ? 0.0 : pow(momentum, 1.0 / float(nsub));

  // --- Density interactions, frozen at the START position ---
  // The density map is a per-frame snapshot (1-frame delayed), and a particle's
  // own halo ("shadow") sits at its previous position. If we re-sampled the
  // density at each substep's MOVED position, the particle would drift off its
  // own shadow within the frame and the avoidance gradient would push it away
  // from itself (self-curl / self-propulsion). So evaluate crowding ONCE here
  // and treat death + avoidance as a constant per-frame force across substeps.
  // (The EXTERNAL flow field IS re-sampled per substep below — it has no self-
  // shadow, and finer field integration is the whole point of substepping.)
  float2 avoid_vec  = float2(0.0, 0.0);
  float2 stream_dir = float2(0.0, 0.0);   // group direction (soft-normalised)
  float  stream_rate = 0.0;               // 1/s, gated by neighbour density
  float  stream_sign = (stream >= 0.0) ? 1.0 : -1.0;
  if (interactions != 0u) {
    if (life_remain > 0.0 && density_death > 1e-5) {
      float dens = densityTex.SampleLevel(linearSampler, saturate(pos), 0).r;
      float others = max(dens - 1.0, 0.0);          // subtract own halo peak (~1)
      float knee = max(density_threshold * 0.5, 1.0);
      float factor = smoothstep(0.0, knee, others - density_threshold);
      float lambda = density_death * FSW_DEATH_RATE * factor;   // 1/s
      float pdie = 1.0 - exp(-lambda * dt);                     // whole frame, once
      float rr = fsw_unit(fsw_hash3(i + 0xDEAD0001u, frame_index, seed));
      if (rr < pdie) life_remain = 0.0;
    }
    if (avoid > 1e-5) {
      // Sample the gradient over EQUAL PIXEL distances (aspect-corrected eps), so
      // the avoidance neighbourhood is round on screen, not squashed. The result
      // is a pixel-space push direction; it's converted back to uv at the end.
      float e = 2.0 / max(density_res, 1.0);
      float ex = e * dens_aspect_x, ey = e * dens_aspect_y;
      float dl = densityTex.SampleLevel(linearSampler, saturate(pos - float2(ex, 0.0)), 0).r;
      float dr = densityTex.SampleLevel(linearSampler, saturate(pos + float2(ex, 0.0)), 0).r;
      float dd = densityTex.SampleLevel(linearSampler, saturate(pos - float2(0.0, ey)), 0).r;
      float du = densityTex.SampleLevel(linearSampler, saturate(pos + float2(0.0, ey)), 0).r;
      float2 away = -float2(dr - dl, du - dd);            // away from crowding (px space)
      float2 awayhat = away / (length(away) + 0.5);       // soft-normalised
      float ang2 = avoid_curl * 1.5707963;                // true ±90° on screen
      float ca2 = cos(ang2), sa2 = sin(ang2);
      float2 av = float2(awayhat.x * ca2 - awayhat.y * sa2,
                         awayhat.x * sa2 + awayhat.y * ca2);
      float2 vec_iso = av * avoid * FSW_AVOID_VEL * speed;
      // Avoidance noise: mostly along `av` (a spray), slight isotropic part.
      // Frozen per frame (coherent), so no per-substep variance scaling.
      if (avoid_noise > 1e-6) {
        uint nh = fsw_hash3(i + 0x51ED2701u, frame_index, seed);
        float mag  = fsw_unit(fsw_hash(nh));                          // [0,1] along +av
        float2 cir = float2(fsw_signed(fsw_hash(nh ^ 0x9E3779B1u)),
                            fsw_signed(fsw_hash(nh ^ 0x85EBCA77u)));  // isotropic
        float2 nv = av * mag + cir * FSW_NOISE_CIRC;
        vec_iso += nv * avoid_noise * FSW_AVOID_VEL * speed;
      }
      // Pixel-space push → uv (so it's isotropic on screen, matching the field).
      avoid_vec = vec_iso * float2(dens_aspect_x, dens_aspect_y);
    }
    // Stream: read the group's mean velocity (.gb = Σ halo·v, .r = Σ halo) and
    // its direction. The per-substep steering toward/away from it is applied in
    // the loop; here we just freeze the direction + a density-gated rate.
    if (abs(stream) > 1e-4) {
      float3 dv = densityTex.SampleLevel(linearSampler, saturate(pos), 0).rgb;
      float2 gmean = dv.gb / max(dv.r, 1e-4);           // halo-weighted mean velocity
      stream_dir = gmean / (length(gmean) + 1e-4);      // group direction
      float others = max(dv.r - 1.0, 0.0);              // neighbours (excl. own halo peak)
      float dfac = saturate(others / max(stream_density, 1e-3));
      stream_rate = abs(stream) * FSW_STREAM_RATE * dfac;
    }
  }

  for (uint sub = 0u; sub < nsub; sub++) {
    // sub 0 → fi == frame_index (identical RNG to the single-step path).
    uint fi = frame_index + sub * 0x9E3779B9u;

    bool oob = pos.x < -0.05 || pos.x > 1.05 || pos.y < -0.05 || pos.y > 1.05;

    if (life_remain > 0.0 && !oob) {
      float depth = fsw_unpack_depth(packed);
      float u = fsw_undertow(depth, undertow_split);

      // Field, with undertow re-aim (curl rotates) + re-scale (polarity).
      // Re-sampled every substep (external field, no self-shadow). The frozen
      // avoidance is added on top.
      float2 fv = flowTex.SampleLevel(linearSampler, saturate(pos), 0).xy;
      float ang = undertow_curl * 1.5707963;           // ±90° at ±1
      float ca = cos(ang), sa = sin(ang);
      float2 rotv = float2(fv.x * ca - fv.y * sa, fv.x * sa + fv.y * ca);
      float2 under = rotv * undertow_polarity;
      float2 eff = lerp(fv, under, u) * speed + avoid_vec;

      // Acceleration mode.
      if (mode == 1u) {
        vel += eff * dt_sub / max(weight, 1e-3);          // force / mass
      } else {
        vel = eff * (1.0 - msub) + vel * msub;            // velocity chase, msub inertia
      }

      // Settle ("pull"): exponential approach toward the field flow (dt-scaled).
      if (pull > 1e-5) {
        float a = 1.0 - exp(-pull * FSW_PULL_RATE * dt_sub);
        vel = lerp(vel, eff, a);
      }

      // Stream: steer velocity toward (align, stream>0) or away from (diverge,
      // stream<0) the frozen group direction, preserving own speed. dt-scaled.
      if (stream_rate > 1e-5) {
        float2 target = stream_dir * length(vel);   // group direction, own speed
        float a = 1.0 - exp(-stream_rate * dt_sub);
        vel += stream_sign * (target - vel) * a;
      }

      // Jitter as a forward SPRAY (not an isotropic cloud): mostly a random
      // magnitude along the direction of motion (speeds the particle up / slows
      // it down), plus a slight directional spread. Scaled by the particle's own
      // speed so it rides the flow — more spray, less cloud. fwd*mag works out to
      // vel*(1 + mag·jitter), i.e. a ±jitter wobble on the forward magnitude.
      if (jitter > 1e-6) {
        float vmag = length(vel);
        float2 fwd = vel / (vmag + 1e-4);
        uint h = fsw_hash3(i + 0x9E3779B1u, fi, seed);
        float mag  = fsw_signed(fsw_hash(h));                          // forward magnitude
        float2 cir = float2(fsw_signed(fsw_hash(h ^ 0x68BC21EBu)),
                            fsw_signed(fsw_hash(h ^ 0xA17F2B91u)));    // direction spread
        float2 kick = fwd * mag + cir * FSW_NOISE_CIRC;
        vel += kick * jitter * vmag * noise_scale;
      }

      // Drag, then integrate.
      vel *= max(1.0 - drag * dt_sub, 0.0);
      pos += vel * dt_sub;
      life_remain -= dt_sub;
    } else {
      // --- Respawn: fresh random uv, capture color, roll depth/size/life ---
      uint hp = fsw_hash3(i + 0x85EBCA77u, fi, seed);
      float ux = float(hp & 0xFFFFu) * (1.0 / 65536.0);
      float uy = float((hp >> 16u) & 0xFFFFu) * (1.0 / 65536.0);
      float2 nuv = float2(ux, uy);

      float4 capt = inputTex.SampleLevel(linearSampler, nuv, 0);
      float depth = fsw_unit(fsw_hash2(i + 0x9E3779B9u, fi + seed));
      packed = fsw_pack_rgbd(capt.rgb, depth);

      float lj = fsw_signed(fsw_hash2(i + 0xC2B2AE3Du, fi + seed));
      life_remain = max(life * (1.0 + life_jitter * lj), 1e-3);
      life_total  = life_remain;
      float sj = fsw_signed(fsw_hash2(i + 0x27D4EB2Fu, fi + seed));
      size_cur = max(size * (1.0 + size_jitter * sj), 1e-6);

      pos = nuv;
      vel = flowTex.SampleLevel(linearSampler, nuv, 0).xy * speed;
    }
  }

  p.a = float4(pos, life_remain, life_total);
  p.b = float4(vel, size_cur, asfloat(packed));
  particles[i] = p;
}
