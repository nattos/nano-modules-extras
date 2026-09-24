// source.legacy.double_chamber — P field-particle update. One thread per slot.
//
// The simulation runs in aspect-corrected "s-space": s = (uv - 0.5) / aspect,
// so a circle in s is a circle on screen (the field/boundary are isotropic on
// any viewport). Velocity is stored in s-space; the position is kept in uv for
// rendering + image sampling.
//
// Force sum (PetriDish): polynomial field + Big-attractor pull (+curl) + sink +
// jitter + soft circular boundary + optional image gradient (+curl). Forces and
// velocity are clamped so the cubic field term can't blow particles up. Dead /
// escaped particles respawn near centre and capture the input colour. All
// timing is dt-accumulated (style guide §2.1).

#include "common.hlsl"

RWStructuredBuffer<Particle> particles    : register(u0);
StructuredBuffer<Particle>   bigs          : register(t1);
Texture2D<float4>            inputTex      : register(t2);
SamplerState                 samp          : register(s3);
StructuredBuffer<Seg>        segs          : register(t5);   // tracer segments (spawn-on-line)
Texture2D<float4>            densityTex    : register(t6);   // last frame's crowding (.r) + Σhalo·vel (.gb)

cbuffer Uniforms : register(b4) {
  uint  count;
  uint  big_count;
  uint  frame_index;
  float dt;

  float motion_rate;
  float momentum;
  float momentum_decay;
  float field_speed;

  float field_scale;
  float field_skew;
  float field_squash;
  float jitter;

  float to_big;
  float to_big_curl;
  float curl_dir;
  float sink;

  float boundary;
  float boundary_size;
  float boundary_stiffness;
  float boundary_speed;

  float to_image;
  float to_image_curl;
  float undertow_skew;
  float undertow_squash;

  float ttl;
  float spawn_size;
  float aspect_x;        // min/W
  float aspect_y;        // min/H

  float to_big_range;    // s-space radius of Big influence (0 → effectively global)
  float image_smoothing; // matches the blur radius → scales the gradient step
  float to_line_rate;    // P(respawn onto a tracer line)
  float seg_total;       // number of tracer segment slots (l_count * seg_stride)

  float boundary_death;  // P(die+respawn) when over the boundary, ∝ overshoot
  float l_count_f;       // tracer count
  float seg_stride;      // segment slots per tracer (its private block size)
  float seg_live;        // slots per tracer actually written (the rest are zeroed)

  uint  interactions;      // 0 = off (skip the density read entirely), 1 = on
  float density_threshold; // crowding (≈ neighbour count) above which death kicks in
  float density_death;     // death-rate scale when over threshold (soft knee)
  float avoid;             // avoidance force away from neighbours

  float avoid_curl;        // -1/+1 rotate the avoidance ±90° (swirl)
  float avoid_noise;       // random jitter on the avoidance (breaks flat clumps)
  float stream;            // +align / -diverge velocity vs the local group
  float stream_density;    // neighbour density for ~max stream effect

  float density_res;       // density buffer resolution (texels per axis)
  float _pd0;
  float _pd1;
  float _pd2;
}

static const float DC_FORCE_MAX = 6.0;
static const float DC_VEL_MAX    = 6.0;
static const float DC_IMG_GAIN   = 6.0;   // image gradients are small; amplify
static const float DC_ESCAPE_R   = 1.3;   // s-radius past which a particle dies
// s-space overshoot at which the boundary-death proportional factor saturates.
static const float DC_BDEATH_REF = 0.25;
// Interaction tuning scales (ported from flow_swarm; the avoidance is a FORCE
// here, not a velocity, because double_chamber integrates a force sum).
static const float DC_DEATH_RATE  = 4.0;   // MAX death rate (1/s) at density_death = 1
static const float DC_AVOID_FORCE = 2.0;   // avoidance force scale at avoid = 1
static const float DC_NOISE_CIRC  = 0.2;   // slight isotropic part of the avoid noise
static const float DC_STREAM_RATE = 10.0;  // MAX align/diverge rate (1/s) at |stream| = 1

float dc_lum(float3 c) { return max(c.r, max(c.g, c.b)); }
float2 dc_clamp_mag(float2 v, float m) {
  float l = length(v);
  return (l > m) ? v * (m / l) : v;
}

[numthreads(64, 1, 1)]
void main(uint3 gid : SV_DispatchThreadID) {
  uint i = gid.x;
  if (i >= count) return;

  float2 aspect = float2(aspect_x, aspect_y);
  Particle p = particles[i];
  float2 uv          = p.a.xy;
  float  life_remain = p.a.z;
  float  life_total  = p.a.w;
  float2 vel         = p.b.xy;          // s-space velocity
  float  size_cur    = p.b.z;
  uint   packed      = asuint(p.b.w);

  // Respawn this frame if already dead, OR if a boundary/density kill fires
  // below (same-frame recycle — "die and respawn right away"). Natural ttl
  // expiry and the escape kill keep their next-frame respawn (respawn stays
  // false). A density kill additionally REDISTRIBUTES — see the respawn block.
  bool respawn = (life_remain <= 0.0);
  bool dens_kill = false;

  if (life_remain > 0.0) {
    float2 s = (uv - 0.5) / max(aspect, 1e-4);   // aspect-corrected centred coord
    float  zc = dc_unpack_z(packed);
    float  curl_factor = (zc - undertow_skew) * undertow_squash;

    float2 force = float2(0.0, 0.0);

    // ---- Interactions: particle-vs-particle via the density buffer ----
    // The buffer is a 1-frame-delayed snapshot of where everyone was, splatted
    // as soft halos (.r = Σ halo ≈ crowding, .gb = Σ halo·velocity). Read ONCE,
    // at the pre-integration position, so each particle sits on its own halo
    // ("shadow") and the gradient it feels is its NEIGHBOURS', not its own.
    //
    // The gradient is sampled over equal PIXEL steps (aspect-corrected eps), so
    // the resulting direction is isotropic on screen — which is exactly s-space,
    // the space the force sum lives in. No conversion needed on the way out.
    float2 stream_dir  = float2(0.0, 0.0);   // group direction (soft-normalised)
    float  stream_rate = 0.0;                // 1/s, gated by neighbour density
    float  stream_sign = (stream >= 0.0) ? 1.0 : -1.0;
    if (interactions != 0u) {
      // Density death: over `threshold` crowding, a soft-knee chance to die and
      // respawn — thins over-packed regions back toward even coverage. Uses the
      // same-frame recycle path as boundary_death.
      if (density_death > 1e-5) {
        float dens = densityTex.SampleLevel(samp, saturate(uv), 0).r;
        float others = max(dens - 1.0, 0.0);        // subtract own halo peak (~1)
        float knee = max(density_threshold * 0.5, 1.0);
        float factor = smoothstep(0.0, knee, others - density_threshold);
        float lambda = density_death * DC_DEATH_RATE * factor;   // 1/s
        float pdie = 1.0 - exp(-lambda * dt);
        uint hr = dc_hash3(i + 0xDEAD0001u, frame_index, 0xD2u);
        if (dc_unit(dc_hash(hr)) < pdie) { respawn = true; dens_kill = true; }
      }

      // Avoidance: push down the crowding gradient (away from neighbours); curl
      // rotates that push ±90° for a swirling avoidance instead of a straight one.
      if (avoid > 1e-5) {
        float e = 2.0 / max(density_res, 1.0);
        float ex = e * aspect_x, ey = e * aspect_y;
        float dl = densityTex.SampleLevel(samp, saturate(uv - float2(ex, 0.0)), 0).r;
        float dr = densityTex.SampleLevel(samp, saturate(uv + float2(ex, 0.0)), 0).r;
        float dd = densityTex.SampleLevel(samp, saturate(uv - float2(0.0, ey)), 0).r;
        float du = densityTex.SampleLevel(samp, saturate(uv + float2(0.0, ey)), 0).r;
        float2 away = -float2(dr - dl, du - dd);          // away from crowding
        float2 awayhat = away / (length(away) + 0.5);     // soft-normalised
        float ang2 = avoid_curl * 1.5707963;              // true ±90° on screen
        float ca2 = cos(ang2), sa2 = sin(ang2);
        float2 av = float2(awayhat.x * ca2 - awayhat.y * sa2,
                           awayhat.x * sa2 + awayhat.y * ca2);
        float2 push = av * avoid * DC_AVOID_FORCE;
        // Avoidance noise: mostly a spray along `av`, slight isotropic part —
        // without it a symmetric clump has a flat gradient at its centre and
        // just sits there.
        if (avoid_noise > 1e-6) {
          uint nh = dc_hash3(i + 0x51ED2701u, frame_index, 0xA3u);
          float mag  = dc_unit(dc_hash(nh));                            // [0,1] along +av
          float2 cir = float2(dc_signed(dc_hash(nh ^ 0x9E3779B1u)),
                              dc_signed(dc_hash(nh ^ 0x85EBCA77u)));    // isotropic
          push += (av * mag + cir * DC_NOISE_CIRC) * avoid_noise * DC_AVOID_FORCE;
        }
        force += push;
      }

      // Stream: read the group's mean velocity (.gb / .r) and freeze its
      // direction + a density-gated rate; the steering happens post-integration
      // (it acts on velocity, not force, so it preserves each particle's speed).
      if (abs(stream) > 1e-4) {
        float3 dv = densityTex.SampleLevel(samp, saturate(uv), 0).rgb;
        float2 gmean = dv.gb / max(dv.r, 1e-4);
        stream_dir = gmean / (length(gmean) + 1e-4);
        float others = max(dv.r - 1.0, 0.0);     // neighbours (excl. own halo peak)
        float dfac = saturate(others / max(stream_density, 1e-3));
        stream_rate = abs(stream) * DC_STREAM_RATE * dfac;
      }
    }

    // Polynomial field (clamp the input so the cubic term stays finite).
    float2 fx = clamp(s * field_scale, -3.0, 3.0);
    force += dc_field(fx, field_skew, field_squash) * field_speed;

    // Big-attractor pull + curl (Big positions converted to s-space). The pull
    // is a spring with a smooth distance cutoff at to_big_range so it's a LOCAL
    // attractor, not a global tilt; range <= 0 disables the cutoff (global).
    if (to_big != 0.0 || to_big_curl != 0.0) {
      float rng = to_big_range;
      for (uint b = 0u; b < big_count; b++) {
        if (bigs[b].a.z <= 0.0) continue;
        float2 bs = (bigs[b].a.xy - 0.5) / max(aspect, 1e-4);
        float2 d = bs - s;
        float fall = (rng > 1e-4) ? (1.0 - smoothstep(rng * 0.6, rng, length(d))) : 1.0;
        force += d * to_big * fall;
        force += dc_perp(d) * to_big_curl * curl_dir * curl_factor * fall;
      }
    }

    // Radial sink (outward when positive).
    float r = max(length(s), 1e-4);
    float2 rad = s / r;
    force += rad * sink;

    // Jitter.
    if (jitter > 1e-6) {
      uint h = dc_hash3(i + 0x9E3779B1u, frame_index, 0x1234u);
      force += float2(dc_signed(dc_hash(h)), dc_signed(dc_hash(h ^ 0x68BC21EBu))) * jitter;
    }

    // Soft circular boundary: inward restoring force outside boundary_size.
    if (boundary > 1e-6) {
      float over = max((r - boundary_size) * boundary_stiffness, 0.0);
      force += -rad * atan(over) * boundary_speed * boundary;
    }

    // Image gradient (+curl), sampled over equal pixel steps → s-space gradient.
    // The step widens with image_smoothing so it reads the broad (blurred)
    // gradient instead of a near-flat local difference.
    if (to_image != 0.0 || to_image_curl != 0.0) {
      float e = lerp(0.004, 0.04, saturate(image_smoothing));
      float2 du = e * aspect;
      float vl = dc_lum(inputTex.SampleLevel(samp, saturate(uv - float2(du.x, 0)), 0).rgb);
      float vr = dc_lum(inputTex.SampleLevel(samp, saturate(uv + float2(du.x, 0)), 0).rgb);
      float vd = dc_lum(inputTex.SampleLevel(samp, saturate(uv - float2(0, du.y)), 0).rgb);
      float vu = dc_lum(inputTex.SampleLevel(samp, saturate(uv + float2(0, du.y)), 0).rgb);
      float2 g = float2(vr - vl, vu - vd) * DC_IMG_GAIN;
      // Taper the image force to zero at the frame edge: ClampToEdge sampling
      // makes the gradient vanish past the border, which would otherwise trap
      // particles in a pile right at the viewport edge.
      float2 ed = min(uv, 1.0 - uv);
      float edgeFade = smoothstep(0.0, 0.05, min(ed.x, ed.y));
      force += g * to_image * edgeFade;
      force += dc_perp(g) * to_image_curl * curl_dir * curl_factor * edgeFade;
    }

    // Clamp force + velocity so the field can't fling particles to infinity.
    force = dc_clamp_mag(force, DC_FORCE_MAX);
    vel = lerp(force, vel * momentum_decay, saturate(momentum));

    // Stream: steer the velocity toward (align, stream > 0) or away from
    // (diverge, stream < 0) the frozen group direction, preserving the
    // particle's own speed. Exponential approach, so it's dt-independent.
    if (stream_rate > 1e-5) {
      float2 target = stream_dir * length(vel);   // group direction, own speed
      float a = 1.0 - exp(-stream_rate * dt);
      vel += stream_sign * (target - vel) * a;
    }

    vel = dc_clamp_mag(vel, DC_VEL_MAX);

    s += vel * dt * motion_rate;
    uv = 0.5 + s * aspect;
    life_remain -= dt;
    if (length(s) > DC_ESCAPE_R) life_remain = 0.0;   // safety kill for runaways

    // Boundary death: a particle past the boundary has a per-frame chance to
    // die + respawn, PROPORTIONAL to how far it has overshot (and scaled by
    // boundary_death). dt·60 keeps the rate frame-rate independent. This
    // recycles particles that would otherwise pile up against the boundary.
    if (boundary_death > 0.0) {
      float over_b = length(s) - boundary_size;
      if (over_b > 0.0) {
        float ov = saturate(over_b / DC_BDEATH_REF);
        float prob = saturate(boundary_death * ov * dt * 60.0);
        uint hd = dc_hash3(i + 0x5151BEEFu, frame_index, 0xD1u);
        if (dc_unit(dc_hash(hd)) < prob) respawn = true;
      }
    }
  }

  if (respawn) {
    // Respawn on a uniform-area disc about centre (the original's spawn shape:
    // radius = spawn_size·sqrt(rand) gives equal density per unit area; full-
    // circle angle, no directional bias). Disc lives in s-space → round on
    // screen, concentric with the boundary circle.
    uint h = dc_hash3(i + 0x85EBCA77u, frame_index, 0x55u);
    // A DENSITY kill redistributes rather than respawns: it lands uniformly
    // across the whole chamber disc, not on the (small, central) spawn disc.
    // Otherwise thinning a pile-up merely teleports it to the source — the
    // source then goes over threshold itself, kills the arrivals in place, and
    // the entire pool collapses onto the spawn point. (flow_swarm gets this for
    // free: its respawn is already a uniform random uv over the whole field.)
    // Every other respawn path — ttl expiry, escape, boundary death — keeps the
    // spawn disc, so the effect's usual look is untouched.
    float disc = dens_kill ? max(boundary_size, spawn_size) : spawn_size;
    float rad = disc * sqrt(dc_unit(dc_hash(h)));
    float theta = 6.28318530718 * dc_unit(dc_hash(h ^ 0xA17Fu));
    float2 sp = rad * float2(cos(theta), sin(theta));
    // NO saturate: a spawn disc larger than the frame must place particles at
    // their true (possibly off-screen) position, not clamp them onto the edge
    // (which would pile + simulate from the viewport border). Colour capture
    // below tolerates out-of-range uv via the ClampToEdge sampler.
    float2 nuv = 0.5 + sp * aspect;

    // Spawn-on-line: with prob to_line_rate, respawn somewhere on a tracer line.
    //
    // The segment buffer is BLOCKED per tracer — tracer j owns
    // [j*seg_stride, j*seg_stride + seg_live), the tail of each block being
    // zeroed — so the tracer and the slot within it must be drawn separately.
    // (Indexing the flat buffer instead lands most spawns in the last tracer's
    // block, and in its dead tail at that.) Then land at a UNIFORM POINT ALONG
    // the chosen segment rather than on its p0: the vertices are a coarse
    // lattice, and snapping to them quantizes the cloud into hard rails.
    // (Skipped for density kills: those are a REDISTRIBUTION, and the lines are
    // exactly the kind of place particles bunch up — landing them back on one
    // would just re-feed the pile they were culled from.)
    uint lc   = (uint)l_count_f;
    uint live = (uint)seg_live;
    if (!dens_kill && to_line_rate > 0.0 && lc > 0u && live > 0u
        && dc_unit(dc_hash(h ^ 0x0777u)) < to_line_rate) {
      uint li = min(lc   - 1u, (uint)(dc_unit(dc_hash(h ^ 0x0999u)) * (float)lc));
      uint sk = min(live - 1u, (uint)(dc_unit(dc_hash(h ^ 0x0BB5u)) * (float)live));
      Seg sg = segs[li * (uint)seg_stride + sk];
      if (sg.b.w > 0.0) nuv = lerp(sg.a.xy, sg.a.zw, dc_unit(dc_hash(h ^ 0x0CCDu)));
    }

    float4 capt = inputTex.SampleLevel(samp, nuv, 0);
    float zr = dc_unit(dc_hash2(i + 0x27D4EB2Fu, frame_index));
    packed = dc_pack_rgbz(capt.rgb, zr);
    float lj = dc_signed(dc_hash2(i + 0xC2B2AE3Du, frame_index));
    life_remain = max(ttl * 8.0 * (1.0 + 0.3 * lj), 0.05);
    life_total  = life_remain;
    uv = nuv;
    vel = float2(0.0, 0.0);
    size_cur = max(size_cur, 1e-4);
  }

  p.a = float4(uv, life_remain, life_total);
  p.b = float4(vel, size_cur, asfloat(packed));
  particles[i] = p;
}
