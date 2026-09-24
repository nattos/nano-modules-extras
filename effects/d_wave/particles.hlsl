// warp.legacy.d_wave — dampening-flash particle pool (one thread per particle).
//
// These are NOT the wave field — they're a fast-evolving layer that SUBTRACTS
// (dampens) the stateful wave field at warp time (see blob_vs/blob_fs). Each is
// float4(angle, radius, sizeJitter, speedJitter):
//   angle ∈ [0,1)  — position around the circle
//   radius         — distance from the centre; drifts OUTWARD over time
//   sizeJitter     — [0,1) per-spawn random (strength/length jitter, blob_vs)
//   speedJitter    — [0,1) per-spawn random (per-particle speed → break-up)
//
// They live in a mid radius BAND: spawn between BAND_LO and BAND_HI (never at
// the centre) and respawn once past BAND_OUT (before the rim). With fast speeds
// this reads as quick flashes sweeping the middle of the field. Forward time
// integration (style guide §2.1): radius += speed·dt, never absolute time.

RWStructuredBuffer<float4> particles : register(u0);

cbuffer Uniforms : register(b1) {
  uint  count;   // live particle count
  uint  frame;   // frame index → respawn entropy
  uint  seed;    // 1 on the first frame → initialise the whole pool (staggered)
  uint  _u;

  float dt;
  float speed;   // base outward speed (damp_rate · DAMP_SPEED_SCALE)
  float spread;  // [0,1] per-particle speed variation
  float _p;
}

static const float BAND_LO  = 0.12;   // innermost spawn radius (never the centre)
static const float BAND_HI  = 0.50;   // outermost spawn radius
static const float BAND_OUT = 0.82;   // respawn once past here (before the rim)

uint dw_hash(uint x) {
  x ^= x >> 16; x *= 0x7feb352du; x ^= x >> 15; x *= 0x846ca68bu; x ^= x >> 16;
  return x;
}
float dw_unit(uint h) { return (h >> 8) * (1.0 / 16777216.0); }

[numthreads(64, 1, 1)]
void main(uint3 gid : SV_DispatchThreadID) {
  uint i = gid.x;
  if (i >= 4096u) return;                       // MAX_PARTICLES

  // Seed: spread the pool across the active band [BAND_LO, BAND_OUT) so phases
  // are staggered from the first frame.
  if (seed != 0u) {
    uint b = dw_hash(i * 2654435761u + 12345u);
    float ang = dw_unit(b);
    float rad = lerp(BAND_LO, BAND_OUT, dw_unit(dw_hash(b ^ 0xA17Fu)));
    float sj  = dw_unit(dw_hash(b ^ 0xB2C3u));
    float spj = dw_unit(dw_hash(b ^ 0xC3D4u));
    particles[i] = float4(ang, rad, sj, spj);
    return;
  }

  if (i >= count) return;                       // idle slots

  float4 p = particles[i];
  float ang = p.x, r = p.y, sj = p.z, spj = p.w;

  // Forward-integrate outward; per-particle speed gives the temporal break-up.
  float ps = speed * lerp(1.0, 0.25 + 1.5 * spj, spread);
  r += ps * dt;

  // Respawn in the inner band (not the centre) once past the outer edge of the
  // band — so a flash never reaches the rim before it recycles.
  if (r > BAND_OUT) {
    uint hr = dw_hash(i * 40503u ^ frame * 2246822519u);
    ang = dw_unit(hr);
    sj  = dw_unit(dw_hash(hr ^ 0x77u));
    spj = dw_unit(dw_hash(hr ^ 0x99u));
    r   = lerp(BAND_LO, BAND_HI, dw_unit(dw_hash(hr ^ 0x55u)));
  }

  particles[i] = float4(ang, r, sj, spj);
}
