// source.particles.sweep_chamber — particle update. One thread per slot.
//
// flow_swarm's substepped integrator driving double_chamber's aesthetic:
// the velocity field is ONE bilinear tap into field_b (curl-noise background
// + swept-image gradient), composed per particle so each keeps its own
// z-phase curl factor. Two acceleration modes (Velocity chase / Force on a
// mass), exponential settle, forward-spray jitter, drag.
//
// Containment is double_chamber's soft circular boundary (an inward
// velocity impulse past boundary_size) rather than flow_swarm's edge-cull:
// the sweep-release "fling" must arc past the frame edge and curve back,
// not get truncated at uv 1.05. Runaways past the escape radius respawn.
//
// Respawn is dc's uniform-area central disc in s-space (unclamped uv — an
// oversized disc places particles at their true off-screen position), with
// input color captured at the spawn point. All timing dt-accumulated.
//
// Register map (final; later passes add t5 density, t6 segs, t7 tracers,
// t8 response, t9 field_a):
//   u0 particles · t1 field_b · t2 input · s3 sampler · b4 uniforms

#include "common.hlsl"
#include "nano_sanitize.hlsl"

RWStructuredBuffer<Particle> particles    : register(u0);
Texture2D<float4>            fieldTexB    : register(t1);
Texture2D<float4>            inputTex     : register(t2);
SamplerState                 linearSampler : register(s3);
Texture2D<float4>            densityTex    : register(t5);   // last frame's crowding
StructuredBuffer<Seg>        segs          : register(t6);   // tracer segments (spawn-on-line)
StructuredBuffer<TracerState> tracers      : register(t7);   // per-tracer grip (spawn weighting)
StructuredBuffer<float4>     respBuf       : register(t8);   // [1] = calm↔intense response
Texture2D<float4>            fieldTexA     : register(t9);   // ridge presence (undertow gate)
Texture2D<float4>            fieldTexC     : register(t10);  // .rg ∇luma (curl direction)

cbuffer Uniforms : register(b4) {
  uint  count;
  uint  frame_index;
  float dt;
  float speed;           // multiplier on the composed field velocity

  float momentum;        // velocity mode: 0 = snap to field, →1 = heavy inertia
  float jitter;          // forward spray: ±wobble on speed + slight direction
  float drag;            // velocity decay per second
  float life;            // base lifetime (s)

  float life_jitter;     // ±fraction on lifetime
  float size;            // base particle size (isotropic uv, curved C++-side)
  float size_jitter;     // ±fraction on size
  uint  seed;            // decorrelates instances

  uint  mode;            // 0 = Velocity, 1 = Force
  float weight;          // force mode: particle mass (accel = force / weight)
  float pull;            // settle: pull velocity back toward the field flow [0,1]
  float to_image;        // image-gradient coupling (composed here, not baked)

  float to_image_curl;   // perp-gradient coupling (× per-particle curl factor)
  float undertow_skew;   // curl factor = (z - skew) · squash  (dc parity)
  float undertow_squash;
  float aspect_x;        // min(W,H)/W

  float aspect_y;        // min(W,H)/H
  uint  substeps;        // integration substeps per frame
  float boundary;        // soft circular containment strength [0,1]
  float boundary_size;   // s-space radius of the free zone

  float boundary_stiffness;
  float boundary_death;  // P(die+respawn) past the boundary, ∝ overshoot
  float spawn_size;      // respawn disc radius (s-space)
  float to_line_rate;    // P(respawn onto a tracer line), pre-grip

  float l_count_f;       // tracer count
  float seg_stride;      // segment slots per tracer (its private block size)
  float seg_live;        // slots per tracer actually written (rest zeroed)
  float calm_stretch;    // TTL ×(1 + this) at intensity 0

  float intense_shrink;  // TTL × this at intensity 1
  float respawn_rate;    // forced-respawn hazard (fraction of pool/s at inten 1)
  float line_boost;      // spawn-on-line multiplier at intensity 1
  float jitter_boost;    // jitter multiplier gain at intensity 1

  float fling_boost;     // velocity kick × release envelope
  uint  interactions;    // 0 = off (skip density reads entirely), 1 = on
  float density_threshold;
  float density_death;

  float avoid;           // push away from neighbours (velocity, × speed)
  float avoid_curl;      // -1/+1 rotate the avoidance ±90° (swirl)
  float avoid_noise;     // random jitter on the avoidance (breaks flat clumps)
  float density_res;     // density buffer resolution (texels per axis)

  float stream;          // +align / -diverge velocity vs the group
  float stream_density;  // neighbour density for ~max stream effect
  float field_res;       // field texture resolution (B-spline sampling)
  float dens_scale;      // screen uv → density uv (scale about the centre)
}

// The density buffer covers a MARGIN of screen uv beyond the frame so that
// off-frame particles still crowd their in-frame neighbours; mapping in is a
// pure scale about the centre. Clamping past the margin is harmless — nothing
// lives out there that a particle inside the frame can feel.
float2 swc_dens_uv(float2 p) {
  return saturate(p * dens_scale + 0.5 * (1.0 - dens_scale));
}

// Max settle rate (1/s) at pull = 1 (flow_swarm parity).
static const float SWC_PULL_RATE      = 20.0;
static const float SWC_NOISE_CIRC     = 0.2;   // slight isotropic part of jitter
// Interaction tuning scales (flow_swarm parity).
static const float SWC_DEATH_RATE     = 4.0;   // MAX death rate (1/s) at density_death=1
static const float SWC_AVOID_VEL      = 1.5;   // avoidance velocity scale (× speed)
static const float SWC_STREAM_RATE    = 10.0;  // MAX align/diverge rate (1/s) at |stream|=1
static const float SWC_BOUNDARY_ACCEL = 3.0;   // boundary impulse scale (uv/s²·rad)
static const float SWC_ESCAPE_R       = 1.5;   // s-radius past which a particle dies
// s-space overshoot at which boundary-death probability saturates (dc parity).
static const float SWC_BDEATH_REF     = 0.25;

// Compose the field velocity (uv/s) at one field_b sample for a particle
// with curl factor `cf` and ridge presence `ridge` (field_a's L'max — see
// swc_undertow in common.hlsl for why the undertow is ridge-gated).
//
// Orientation continuity: a tangent field's sign is inherently ambiguous,
// and perp(∇L') flips 180° across a ridge crest — a particle oscillating
// across the line would reverse its along-ridge stream every frame (jerky
// single-frame direction switches). So the tangent sign follows the
// particle's OWN current heading (`vel_prev`), falling back to the signed
// depth phase `cf` at spawn (which keeps the undertow_skew/squash
// population split: half the swarm streams each way).
// The undertow's direction comes from the PLAIN luma gradient `gl`
// (field_c.rg) and its magnitude from swept presence — one smooth hump per
// region as the sweep passes (see swc_undertow); the attraction term keeps
// the raw swept gradient — its reversal IS the trapping well.
float2 swc_field_vel(float4 fb, float ridge, float2 gl, float cf,
                     float2 vel_prev) {
  float2 aspect = float2(aspect_x, aspect_y);
  float2 und = swc_undertow(fb.zw, gl, ridge);
  float aln = dot(und, vel_prev / max(aspect, 1e-4));
  float sgn = (abs(aln) > 1e-6) ? (aln > 0.0 ? 1.0 : -1.0)
                                : (cf >= 0.0 ? 1.0 : -1.0);
  float2 iso = fb.zw * to_image + und * (to_image_curl * abs(cf) * sgn);
  return fb.xy + iso * aspect;
}

[numthreads(64, 1, 1)]
void main(uint3 gid : SV_DispatchThreadID) {
  uint i = gid.x;
  if (i >= count) return;

  float2 aspect = float2(aspect_x, aspect_y);
  Particle p = particles[i];
  float2 pos         = p.a.xy;
  float  life_remain = p.a.z;
  float  life_total  = p.a.w;
  float2 vel         = p.b.xy;
  float  size_cur    = p.b.z;
  uint   packed      = asuint(p.b.w);

  float cf = (swc_unpack_z(packed) - undertow_skew) * undertow_squash;

  // Calm↔intense response, derived from the swept image (see stats.hlsl).
  float4 resp = respBuf[1];
  float inten = nano_sanitize(resp.z, 0.0, 0.0, 1.0);
  float env   = nano_sanitize(resp.w, 0.0, 0.0, 64.0);
  float jitter_eff = jitter * (1.0 + jitter_boost * inten);

  // Substepping (flow_swarm parity): dt-scaled everywhere, noise variance
  // preserved by 1/√nsub, momentum as a per-substep blend. nsub=1 reproduces
  // the single-step path exactly.
  uint  nsub        = max(substeps, 1u);
  float dt_sub      = dt / float(nsub);
  float noise_scale = rsqrt(float(nsub));
  float msub        = (momentum < 1e-6) ? 0.0 : pow(momentum, 1.0 / float(nsub));

  bool respawn = (life_remain <= 0.0);
  bool dens_kill = false;

  // --- Density interactions, frozen at the START position (flow_swarm) ---
  // The density map is a 1-frame-delayed snapshot and a particle's own halo
  // sits at its previous position; evaluating once here keeps each particle
  // on its own "shadow" so the gradient it feels is its NEIGHBOURS', not its
  // own (re-sampling per substep would self-propel it off its halo).
  float2 avoid_vec  = float2(0.0, 0.0);
  float2 stream_dir = float2(0.0, 0.0);
  float  stream_rate = 0.0;
  float  stream_sign = (stream >= 0.0) ? 1.0 : -1.0;
  if (interactions != 0u && !respawn) {
    if (density_death > 1e-5) {
      float dens = densityTex.SampleLevel(linearSampler, swc_dens_uv(pos), 0).r;
      float others = max(dens - 1.0, 0.0);          // subtract own halo peak (~1)
      float knee = max(density_threshold * 0.5, 1.0);
      float factor = smoothstep(0.0, knee, others - density_threshold);
      float lambda = density_death * SWC_DEATH_RATE * factor;   // 1/s
      float pdie = 1.0 - exp(-lambda * dt);
      float rr = swc_unit(swc_hash3(i + 0xDEAD0001u, frame_index, seed));
      if (rr < pdie) { respawn = true; dens_kill = true; }
    }
    if (avoid > 1e-5 && !respawn) {
      // Gradient over equal PIXEL distances → round avoidance on screen.
      // Step = two density texels, in SCREEN uv (the buffer spans 1/dens_scale
      // of the frame, so one texel is 1/(res·dens_scale) of it).
      float e = 2.0 / max(density_res * dens_scale, 1.0);
      float ex = e * aspect_x, ey = e * aspect_y;
      float dl = densityTex.SampleLevel(linearSampler, swc_dens_uv(pos - float2(ex, 0.0)), 0).r;
      float dr = densityTex.SampleLevel(linearSampler, swc_dens_uv(pos + float2(ex, 0.0)), 0).r;
      float dd = densityTex.SampleLevel(linearSampler, swc_dens_uv(pos - float2(0.0, ey)), 0).r;
      float du = densityTex.SampleLevel(linearSampler, swc_dens_uv(pos + float2(0.0, ey)), 0).r;
      float2 away = -float2(dr - dl, du - dd);            // away from crowding
      float2 awayhat = away / (length(away) + 0.5);       // soft-normalised
      float ang2 = avoid_curl * 1.5707963;
      float ca2 = cos(ang2), sa2 = sin(ang2);
      float2 av = float2(awayhat.x * ca2 - awayhat.y * sa2,
                         awayhat.x * sa2 + awayhat.y * ca2);
      float2 vec_iso = av * avoid * SWC_AVOID_VEL * speed;
      if (avoid_noise > 1e-6) {
        uint nh = swc_hash3(i + 0x51ED2701u, frame_index, seed);
        float mag  = swc_unit(swc_hash(nh));
        float2 cir = float2(swc_signed(swc_hash(nh ^ 0x9E3779B1u)),
                            swc_signed(swc_hash(nh ^ 0x85EBCA77u)));
        vec_iso += (av * mag + cir * SWC_NOISE_CIRC) * avoid_noise * SWC_AVOID_VEL * speed;
      }
      avoid_vec = vec_iso * float2(aspect_x, aspect_y);   // pixel push → uv
    }
    if (abs(stream) > 1e-4 && !respawn) {
      float3 dv = densityTex.SampleLevel(linearSampler, swc_dens_uv(pos), 0).rgb;
      float2 gmean = dv.gb / max(dv.r, 1e-4);             // halo-weighted mean vel
      stream_dir = gmean / (length(gmean) + 1e-4);
      float others = max(dv.r - 1.0, 0.0);
      float dfac = saturate(others / max(stream_density, 1e-3));
      stream_rate = abs(stream) * SWC_STREAM_RATE * dfac;
    }
  }

  if (!respawn) {
    for (uint sub = 0u; sub < nsub; sub++) {
      uint fi = frame_index + sub * 0x9E3779B9u;

      // Field velocity, re-sampled every substep. C1 B-spline taps (4
      // bilinear each): plain bilinear is C0 and slow particles trace its
      // per-texel direction kinks as quantized little steps. Frozen
      // avoidance on top.
      float4 fb = swc_sample_bspline(fieldTexB, linearSampler, saturate(pos), field_res);
      float ridge = swc_sample_bspline(fieldTexA, linearSampler, saturate(pos), field_res).a;
      float2 gl = swc_sample_bspline(fieldTexC, linearSampler, saturate(pos), field_res).rg;
      float2 eff = swc_field_vel(fb, ridge, gl, cf, vel) * speed + avoid_vec;

      // Acceleration mode.
      if (mode == 1u) {
        vel += eff * dt_sub / max(weight, 1e-3);          // force / mass
      } else {
        vel = eff * (1.0 - msub) + vel * msub;            // velocity chase
      }

      // Settle ("pull"): exponential approach toward the field flow.
      if (pull > 1e-5) {
        float a = 1.0 - exp(-pull * SWC_PULL_RATE * dt_sub);
        vel = lerp(vel, eff, a);
      }

      // Stream: steer toward (align) / away from (diverge) the frozen group
      // direction, preserving own speed (flow_swarm parity).
      if (stream_rate > 1e-5) {
        float2 target = stream_dir * length(vel);
        float a = 1.0 - exp(-stream_rate * dt_sub);
        vel += stream_sign * (target - vel) * a;
      }

      // Soft circular boundary: inward impulse past boundary_size. A force
      // (not dc's velocity overwrite) so a flung particle ARCS back instead
      // of stopping dead at the wall.
      float2 s = (pos - 0.5) / max(aspect, 1e-4);
      float r = max(length(s), 1e-4);
      if (boundary > 1e-6) {
        float over = max((r - boundary_size) * boundary_stiffness, 0.0);
        if (over > 0.0) {
          float2 rad = s / r;
          vel += (-rad * atan(over) * boundary * SWC_BOUNDARY_ACCEL) * aspect * dt_sub;
        }
      }

      // Forward-spray jitter (flow_swarm parity), boosted by intensity.
      if (jitter_eff > 1e-6) {
        float vmag = length(vel);
        float2 fwd = vel / (vmag + 1e-4);
        uint h = swc_hash3(i + 0x9E3779B1u, fi, seed);
        float mag  = swc_signed(swc_hash(h));
        float2 cir = float2(swc_signed(swc_hash(h ^ 0x68BC21EBu)),
                            swc_signed(swc_hash(h ^ 0xA17F2B91u)));
        float2 kick = fwd * mag + cir * SWC_NOISE_CIRC;
        vel += kick * jitter_eff * vmag * noise_scale;
      }

      // Release FLING: while the release envelope rings (the sweep just let
      // go of a band), kick every particle along its current motion. The
      // dt-scaled sum over the envelope is framerate-independent.
      if (env > 1e-4 && fling_boost > 0.0) {
        float2 fdir = vel / (length(vel) + 0.05);
        vel += fdir * fling_boost * env * dt_sub;
      }

      // Drag, then integrate.
      vel *= max(1.0 - drag * dt_sub, 0.0);
      pos += vel * dt_sub;
      life_remain -= dt_sub;
    }

    // Post-integration kills (once per frame, dt-scaled).
    float2 s_end = (pos - 0.5) / max(aspect, 1e-4);
    float r_end = length(s_end);
    if (r_end > SWC_ESCAPE_R) respawn = true;   // runaway: recycle now
    if (!respawn && boundary_death > 0.0) {
      float over_b = r_end - boundary_size;
      if (over_b > 0.0) {
        float ov = saturate(over_b / SWC_BDEATH_REF);
        float prob = saturate(boundary_death * ov * dt * 60.0);
        uint hd = swc_hash3(i + 0x5151BEEFu, frame_index, 0xD1u);
        if (swc_unit(swc_hash(hd)) < prob) respawn = true;
      }
    }
    // Forced respawn ("intense" bursts): when the sweep is capturing a lot,
    // churn the pool onto the lines fast — the "jumble-ey" bunching. Calm
    // moments leave particles to their long graceful lifetimes.
    if (!respawn && respawn_rate > 0.0 && inten > 1e-4) {
      float lambda = respawn_rate * inten;                 // 1/s
      float pforce = 1.0 - exp(-lambda * dt);
      uint hf = swc_hash3(i + 0x0F0CE001u, frame_index, 0xF1u);
      if (swc_unit(swc_hash(hf)) < pforce) respawn = true;
    }
    if (life_remain <= 0.0) respawn = true;
  }

  if (respawn) {
    // Uniform-area disc about centre in s-space (dc parity: equal density per
    // unit area, round on screen, concentric with the boundary). NO clamp —
    // an oversized disc must place particles at their true position.
    uint h = swc_hash3(i + 0x85EBCA77u, frame_index, seed + 0x55u);
    // A DENSITY kill REDISTRIBUTES across the whole chamber rather than
    // respawning on the (small, central) spawn disc — otherwise thinning a
    // pile-up merely teleports it to the source and the pool collapses onto
    // the spawn point (dc parity; see double_chamber/p_update.hlsl).
    float disc = dens_kill ? max(boundary_size, spawn_size) : spawn_size;
    float rad = disc * sqrt(swc_unit(swc_hash(h)));
    float theta = 6.28318530718 * swc_unit(swc_hash(h ^ 0xA17Fu));
    float2 sp = rad * float2(cos(theta), sin(theta));
    float2 nuv = 0.5 + sp * aspect;

    // Spawn-on-line, GRIP-WEIGHTED: with prob to_line_rate run a few
    // independent trials — each picks a random tracer and accepts it with
    // probability = its grip (how hard the image currently holds it) — then
    // land at a UNIFORM point along a random live segment of its block
    // (vertex-snapping would quantize the cloud into hard rails — dc parity).
    // Multi-trial rejection keeps O(1) cost but makes the slider honest:
    // acceptance is 1-(1-mean grip)^4, so at rate 1.0 nearly every spawn
    // lands on a line once any decent fraction of lines actually grips —
    // while free tracers drifting through black still pull nothing, and a
    // releasing sweep still lets go smoothly (the direct replacement for
    // dc's death-based bunching control).
    // (Skipped for density kills: those are a redistribution, and the lines
    // are exactly where particles bunch — landing them back on one would
    // just re-feed the pile they were culled from.)
    uint lc   = (uint)l_count_f;
    uint live = (uint)seg_live;
    float p_line = saturate(to_line_rate * lerp(1.0, line_boost, inten));
    if (!dens_kill && p_line > 0.0 && lc > 0u && live > 0u
        && swc_unit(swc_hash(h ^ 0x0777u)) < p_line) {
      [unroll]
      for (uint trial = 0u; trial < 4u; ++trial) {
        uint ht = swc_hash(h ^ (0x0999u + trial * 0x9E37u));
        uint li = min(lc - 1u, (uint)(swc_unit(ht) * (float)lc));
        float grip_w = saturate(tracers[li].b.z);
        if (swc_unit(swc_hash(ht ^ 0x0F31u)) < grip_w) {
          uint sk = min(live - 1u, (uint)(swc_unit(swc_hash(ht ^ 0x0BB5u)) * (float)live));
          Seg sg = segs[li * (uint)seg_stride + sk];
          // Geometric liveness (a real segment always advances) — gating on
          // the segment's ALPHA would silently kill spawn-on-line whenever
          // the lines are drawn invisible (l_opacity 0).
          float2 sd = sg.a.zw - sg.a.xy;
          if (dot(sd, sd) > 0.0) {
            nuv = lerp(sg.a.xy, sg.a.zw, swc_unit(swc_hash(ht ^ 0x0CCDu)));
            break;
          }
        }
      }
    }

    // Capture the input color at the spawn point (ClampToEdge tolerates
    // out-of-range uv) + roll a fresh z phase / life / size.
    float4 capt = inputTex.SampleLevel(linearSampler, nuv, 0);
    float zr = swc_unit(swc_hash2(i + 0x27D4EB2Fu, frame_index));
    packed = swc_pack_rgbz(capt.rgb, zr);

    // Intensity-scaled lifetime: calm = long lingering trails, intense =
    // short bursty churn.
    float lj = swc_signed(swc_hash2(i + 0xC2B2AE3Du, frame_index));
    float life_eff = life * lerp(1.0 + calm_stretch, intense_shrink, inten);
    life_remain = max(life_eff * (1.0 + life_jitter * lj), 1e-3);
    life_total  = life_remain;
    float sj = swc_signed(swc_hash2(i + 0x1B873593u, frame_index));
    size_cur = max(size * (1.0 + size_jitter * sj), 1e-6);

    pos = nuv;
    // Newborns stream along the field immediately.
    float4 fb0 = swc_sample_bspline(fieldTexB, linearSampler, saturate(nuv), field_res);
    float ridge0 = swc_sample_bspline(fieldTexA, linearSampler, saturate(nuv), field_res).a;
    float2 gl0 = swc_sample_bspline(fieldTexC, linearSampler, saturate(nuv), field_res).rg;
    float cf0 = (zr - undertow_skew) * undertow_squash;
    vel = swc_field_vel(fb0, ridge0, gl0, cf0, float2(0.0, 0.0)) * speed;
  }

  p.a = float4(pos, life_remain, life_total);
  p.b = float4(vel, size_cur, asfloat(packed));
  particles[i] = p;
}
