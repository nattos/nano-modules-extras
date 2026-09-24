/*
 * source.light.orthomod — Hadamard-driven, beat-synced bar pattern.
 *
 * Two co-driven code systems sharing one global envelope:
 *
 *   System A — channel envelopes (4 per-bar brightness rails).
 *     8×8 Hadamard, rows sorted by complexity, columns Fisher-Yates
 *     shuffled by seed. Row 0 forced to all-ones.
 *     idx_A = floor((1 - env) * 8) — env=1 picks row 0 → all channels ON.
 *     Each row's 8 bits split into 4 (bar) × 2-bit waveform codes:
 *       (0,0)=OFF, (1,1)=ON, (1,0)=square @ mod_rate, (0,1)=|sin| @ mod_rate.
 *
 *   System B — bar fill pattern (4 bars × hadamard_size segments).
 *     MxM Hadamard grouped into P = M/4 pages of 4 rows each. Pages
 *     sorted ascending by entropy (sum of within-column row transitions).
 *     Equal-entropy pages shuffled by seed.
 *     page_idx = floor(env * P) — env=1 picks highest-entropy page.
 *     bar c, segment r → bit = page[c][r mod M].
 *
 * Beat sync via fx::BeatTick: integer crossings of effective_beats
 * snap linear_env := 1; decay otherwise. Decay time in beats →
 * seconds via host::bpm().
 *
 * Outputs:
 *   tex_out         — colored bar pattern (off bits / outside inset pass through tex_in)
 *   ch1..ch4, env   — float rails for downstream effects to tap.
 *
 * Class-like instance model: module_init() compiles the shared compute
 * PSO + publishes the schema once per type; each chain entry gets its own
 * State (params, BeatTick, envelope/edge-state, cached pattern tables,
 * per-instance uniform + page buffers) via create(). All instance
 * callbacks take `self`.
 */

#include <gpu.h>
#include <host.h>
#include <val.h>
#include <effect_utils.h>
#include <effect_beat_tick.h>
#include "orthomod_shaders.h"

#include <cmath>
#include <cstdint>
#include <cstring>

namespace orthomod {

static constexpr int  BARS                  = 4;
static constexpr int  SYS_A_N               = 8;     // 8x8 Hadamard
static constexpr int  MAX_HADAMARD          = 64;    // System B cap (powers of 2 only)
static constexpr int  MAX_PAGE_BITS         = BARS * MAX_HADAMARD;

// System B codebook — the family of M-bit codes the pages are built from.
// Every codebook produces M distinct codes of M bits; the shared rotation
// + visible-entropy sort + page machinery then order them solid→stripes.
enum Codebook : int {
  CB_WALSH      = 0,   // Sylvester-Hadamard — clean sequency stripes
  CB_RANDOM     = 1,   // per-(code,segment) hash — white-noise stripes
  CB_LFSR       = 2,   // maximal-length LFSR; codes are shifts of one m-sequence
  CB_GRAY       = 3,   // repeating n-bit reflected-Gray words
  CB_BINARY     = 4,   // integer frequency ramp (chirp) — i bands in code i
  CB_THUE_MORSE = 5,   // windows of the Thue–Morse sequence — fractal texture
};

struct Uniforms {
  // row 0 — per-bar channel envelopes (already waveform-shaped on CPU)
  float ch0;
  float ch1;
  float ch2;
  float ch3;

  // row 1
  float env;
  float primary_hue;
  float saturation;
  float intensity;

  // row 2
  float scatter_max;
  float channel_brightness_mod;
  float inset_top;
  float inset_bottom;

  // row 3
  uint32_t hadamard_size;
  uint32_t render_bits;
  uint32_t page_idx;
  uint32_t seed;

  // row 4
  float env_brightness;
  float _pad_r4_0;
  float _pad_r4_1;
  float _pad_r4_2;
};
static_assert(sizeof(Uniforms) == 80, "Uniforms layout mismatch");

// --- Per-instance state. One per chain entry. ---
struct State {
  // GPU resources (per-instance buffers).
  gpu::Buffer uniform_buf;
  gpu::Buffer page_buf;            // 4 × hadamard_size uints
  bool        initialized = false;

  // --- Schema-mirrored params ---
  // Standard
  int   beat_multiplier_id   = 2;       // index into the select table
  float primary_hue          = 0.08f;
  float saturation           = 0.9f;
  float intensity            = 1.0f;
  float decay_time_beats     = 1.0f;
  float decay_curve          = 0.0f;    // signed [-1,+1] via fx::signedSliderToExp
  // Release phase — used after the gate is released. Decay is used while the
  // gate is held, and after a momentary trigger or beat crossing (which have
  // no held state). Separate half-life + curve, typically a faster fall.
  float release_time_beats   = 0.5f;
  float release_curve        = 0.0f;
  float scatter_max          = 0.15f;
  float channel_brightness_mod = 0.5f;
  // Signed power-curve slider shaping how much the envelope influences
  // overall bar brightness (via fx::signedSliderToExp on env). 0 = linear
  // (env used as-is). -1 → exp 8 → env crushed (bars dim fast / less
  // influence across most of the decay). +1 → exp 1/8 → env lifted (bars
  // stay bright longer / more sustained influence).
  float env_brightness_curve = 0.0f;
  float mod_rate_hz          = 15.0f;
  // Tuning
  int   codebook             = CB_WALSH;
  // Start/end of the entropy-sorted page range the envelope sweeps, as
  // fractions of [0, P-1] (0 = orderly end, 1 = entropic end). The sweep
  // runs start→end as the envelope decays; set start > end to sweep in
  // REVERSE (entropic→orderly). keep_flash keeps the all-1s solid flash as
  // the first level regardless of the range.
  float start                = 0.0f;
  float end                  = 1.0f;
  bool  keep_flash           = true;
  int   hadamard_size        = 32;
  int   render_bits          = 13;
  float inset_top            = 0.0f;
  float inset_bottom         = 0.0f;
  // Per-bar decay jitter (0..1). Biases each bar to cross code (page)
  // boundaries at a slightly offset envelope position so they don't all
  // flip codes in lockstep. Deliberately NOT linked to seed — uses a
  // fixed golden-ratio per-bar offset table. Subtractive (delays only) so
  // the env=1 solid flash stays intact.
  float decay_jitter         = 0.0f;
  int   seed                 = 1;

  // --- Runtime state ---
  fx::BeatTick tick;
  double linear_env = 0.0;
  double mod_phase  = 0.0;
  // Manual trigger surface. Both gate (bool) and trigger (event) are
  // momentary in the IDE — the value is 1 while held, 0 on release — and
  // the executor replays that value every frame (style guide §8.2). So
  // BOTH fire only on a 0→1 rising edge of their value; firing on mere
  // patch presence would re-pin the envelope every frame (stuck-on bug).
  bool   gate                = false;
  bool   gate_prev           = false;
  float  trigger_prev        = 0.0f;
  // True while in the decay phase (gate held, or after a momentary fire);
  // false after the gate is released → release phase. Picks which
  // time/curve the envelope fall uses.
  bool   gate_open           = false;

  // --- Cached System A: rows sorted by complexity, columns shuffled by seed ---
  uint8_t sys_a_rows[SYS_A_N][SYS_A_N];   // sorted+shuffled bits, 0/1
  bool    sys_a_dirty   = true;
  int     sys_a_seed    = -1;

  // --- Cached System B: full Hadamard + entropy-sorted code order ---
  // Each Hadamard row is a "code" (M bits indexed by segment). The columns
  // are first cyclically rotated by a seed-derived offset (see rebuild) to
  // move the all-ones DC column off segment 0. We then sort the M codes
  // ascending by VISIBLE (windowed) horizontal entropy — the number of 0↔1
  // transitions across the rendered segments — so sorted[0] is the all-1s
  // solid code and sorted[M-1] is the busiest stripe pattern. Pages of 4
  // consecutive sorted codes hold 4 same-ish-entropy codes; which of the 4
  // lands on which bar is decided per-page by the seed at render time. The
  // rotation makes the sort seed-dependent, so it caches on (M, render_bits,
  // seed).
  uint8_t sys_b_bits[MAX_HADAMARD][MAX_HADAMARD]; // rotated codes [code][segment]
  int     sys_b_sorted[MAX_HADAMARD];             // code indices, ascending visible entropy
  int     sys_b_size_cached = -1;
  int     sys_b_rb_cached   = -1;                 // render_bits the sort assumes
  int     sys_b_seed_cached = -1;                 // seed the rotation assumes
  int     sys_b_cb_cached   = -1;                 // codebook the codes assume
};

// --- Type-shared GPU resources: compiled once in module_init(). ---
static gpu::ComputePSO s_pso;

// Map beat_multiplier_id → ticks per bar. Values match the selectField
// option values. Id 5 = "Off" → 0, which disables beat-synced triggering
// entirely (fx::BeatTick returns no crossings for a non-positive
// multiplier); the manual trigger / gate still fires in that mode.
static inline float beat_multiplier_value(int id) {
  switch (id) {
    case 0: return 0.25f;
    case 1: return 0.5f;
    case 2: return 1.0f;
    case 3: return 2.0f;
    case 4: return 4.0f;
    case 5: return 0.0f;   // Off
    default: return 1.0f;
  }
}

// --- Pure stateless math helpers ---
static inline float clampf(float v, float lo, float hi) {
  return v < lo ? lo : (v > hi ? hi : v);
}
static inline int clampi(int v, int lo, int hi) {
  return v < lo ? lo : (v > hi ? hi : v);
}
static inline uint32_t lcg_next(uint32_t& s) {
  s = s * 1664525u + 1013904223u;
  return s;
}
static inline int round_up_pow2(int v) {
  if (v <= 1) return 1;
  int p = 1;
  while (p < v) p <<= 1;
  return p;
}
static inline int popcount32(uint32_t x) {
  x = x - ((x >> 1) & 0x55555555u);
  x = (x & 0x33333333u) + ((x >> 2) & 0x33333333u);
  x = (x + (x >> 4)) & 0x0F0F0F0Fu;
  return (int)((x * 0x01010101u) >> 24);
}
// 0/1 Hadamard bit at (i, j) for the 2^n × 2^n Sylvester matrix.
// Row 0 = all 1 (popcount(0&j) = 0).
static inline uint8_t hadamard_bit(int i, int j) {
  return (uint8_t)(1 ^ (popcount32((uint32_t)(i & j)) & 1));
}

static inline int log2i(int x) {
  int k = 0;
  while (x > 1) { x >>= 1; k++; }
  return k;
}

// Maximal-length Fibonacci-LFSR feedback masks (primitive polynomials),
// bit positions 0-indexed. n = log2(M) ∈ [2, 6] for our M ∈ [4, 64].
static inline uint32_t lfsr_taps(int n) {
  switch (n) {
    case 2: return 0x3u;   // x^2 + x + 1   [2,1]
    case 3: return 0x6u;   // x^3 + x^2 + 1 [3,2]
    case 4: return 0xCu;   // x^4 + x^3 + 1 [4,3]
    case 5: return 0x14u;  // x^5 + x^3 + 1 [5,3]
    case 6: return 0x30u;  // x^6 + x^5 + 1 [6,5]
    default: return 0x3u;
  }
}

// Stable per-(code, segment) random bit for the Random codebook.
static inline uint8_t cb_rand_bit(uint32_t seed, int i, int j) {
  uint32_t h = seed * 747796405u + (uint32_t)i * 2654435761u
             + (uint32_t)j * 40503u + 0x9E3779B9u;
  h ^= h >> 16; h *= 0x85EBCA6Bu; h ^= h >> 13; h *= 0xC2B2AE35u; h ^= h >> 16;
  return (uint8_t)((h >> 31) & 1u);
}

// Fill `out` with M codes of M bits (UNrotated) for the selected codebook.
// Every codebook yields M distinct M-bit codes; downstream rotation +
// entropy sort + paging are identical regardless of which is chosen.
static void gen_codebook(State& s, int cb, int M, uint8_t out[MAX_HADAMARD][MAX_HADAMARD]) {
  int n = log2i(M);
  switch (cb) {
    case CB_RANDOM:
      for (int i = 0; i < M; i++)
        for (int j = 0; j < M; j++) out[i][j] = cb_rand_bit((uint32_t)s.seed, i, j);
      break;

    case CB_LFSR: {
      // One maximal-length m-sequence; each code is a cyclic shift of it,
      // so all bars share a single noise texture, just phase-offset.
      uint32_t taps = lfsr_taps(n);
      uint32_t mask = (n >= 32) ? 0xFFFFFFFFu : ((1u << n) - 1u);
      uint32_t st = (uint32_t)s.seed & mask;
      if (st == 0u) st = 1u;                       // all-zero state is illegal
      uint8_t base[MAX_HADAMARD];
      for (int k = 0; k < M; k++) {
        base[k] = (uint8_t)(st & 1u);
        uint32_t nb = (uint32_t)(popcount32(st & taps) & 1u);
        st = (st >> 1) | (nb << (n - 1));
      }
      for (int i = 0; i < M; i++)
        for (int j = 0; j < M; j++) out[i][j] = base[(i + j) % M];
      break;
    }

    case CB_GRAY: {
      // Code i = the reflected-Gray word of i, repeated across segments.
      // Consecutive code indices differ by one bit in the repeating unit.
      for (int i = 0; i < M; i++) {
        uint32_t g = (uint32_t)i ^ ((uint32_t)i >> 1);
        for (int j = 0; j < M; j++) out[i][j] = (uint8_t)((g >> (j % n)) & 1u);
      }
      break;
    }

    case CB_BINARY:
      // Integer frequency ramp: code i has ~i bands across the segments —
      // a clean low→high frequency chirp.
      for (int i = 0; i < M; i++)
        for (int j = 0; j < M; j++) out[i][j] = (uint8_t)(((j * (i + 1)) / M) & 1);
      break;

    case CB_THUE_MORSE:
      // Successive length-M windows of the Thue–Morse sequence
      // (t(k) = parity of popcount(k)) — self-similar fractal texture.
      for (int i = 0; i < M; i++)
        for (int j = 0; j < M; j++)
          out[i][j] = (uint8_t)(popcount32((uint32_t)(i * M + j)) & 1u);
      break;

    case CB_WALSH:
    default:
      for (int i = 0; i < M; i++)
        for (int j = 0; j < M; j++) out[i][j] = hadamard_bit(i, j);
      break;
  }
}

static int sys_a_row_complexity(const uint8_t* row) {
  int t = 0;
  for (int j = 0; j + 1 < SYS_A_N; j++) if (row[j] != row[j + 1]) t++;
  return t;
}

// Visible (windowed) horizontal entropy of a code: the number of 0↔1
// transitions across the segments AS RENDERED. Segment r samples column
// (r % M), so this matches exactly what the shader draws — codes that
// look identical in the rendered window sort equally. All-1s → 0.
static int code_window_transitions(State& s, int code, int M, int render_bits) {
  int rb = render_bits < 1 ? 1 : render_bits;
  int t = 0;
  for (int r = 0; r + 1 < rb; r++) {
    if (s.sys_b_bits[code][r % M] != s.sys_b_bits[code][(r + 1) % M]) t++;
  }
  return t;
}

// Seed-driven permutation of {0,1,2,3} (which page slot → which bar).
static void make_perm4(uint32_t h, int perm[4]) {
  for (int i = 0; i < 4; i++) perm[i] = i;
  // Mix once so small seeds don't degenerate.
  h ^= h >> 16; h *= 0x85EBCA6Bu; h ^= h >> 13;
  for (int i = 3; i > 0; i--) {
    h = h * 1664525u + 1013904223u;
    int k = (int)((h >> 8) % (uint32_t)(i + 1));
    int t = perm[i]; perm[i] = perm[k]; perm[k] = t;
  }
}

static void rebuild_sys_a(State& s) {
  // 1. Generate raw 8x8 Hadamard.
  uint8_t raw[SYS_A_N][SYS_A_N];
  for (int i = 0; i < SYS_A_N; i++) {
    for (int j = 0; j < SYS_A_N; j++) raw[i][j] = hadamard_bit(i, j);
  }
  // 2. Sort row indices by complexity ascending. Row 0 (all 1s) has 0
  //    transitions, so it lands first naturally. Tie-break by index.
  int row_order[SYS_A_N];
  for (int i = 0; i < SYS_A_N; i++) row_order[i] = i;
  for (int i = 1; i < SYS_A_N; i++) {
    int v = row_order[i], cv = sys_a_row_complexity(raw[v]), j = i - 1;
    while (j >= 0) {
      int cj = sys_a_row_complexity(raw[row_order[j]]);
      if (cj > cv || (cj == cv && row_order[j] > v)) {
        row_order[j + 1] = row_order[j]; j--;
      } else break;
    }
    row_order[j + 1] = v;
  }
  // 3. Fisher-Yates column permutation seeded from user seed.
  int col_perm[SYS_A_N];
  for (int i = 0; i < SYS_A_N; i++) col_perm[i] = i;
  uint32_t rng = (uint32_t)s.seed ^ 0xA1B2C3D4u;
  lcg_next(rng);
  for (int i = SYS_A_N - 1; i > 0; i--) {
    uint32_t r = lcg_next(rng);
    int k = (int)(r % (uint32_t)(i + 1));
    int t = col_perm[i]; col_perm[i] = col_perm[k]; col_perm[k] = t;
  }
  // 4. Assemble sorted + shuffled output.
  for (int i = 0; i < SYS_A_N; i++) {
    for (int j = 0; j < SYS_A_N; j++) {
      s.sys_a_rows[i][j] = raw[row_order[i]][col_perm[j]];
    }
  }
  s.sys_a_seed = s.seed;
  s.sys_a_dirty = false;
}

static void rebuild_sys_b(State& s, int M, int render_bits) {
  // Seed-derived cyclic column rotation. Sylvester-Hadamard column 0 is
  // all-ones (the DC term) — without this it renders as an always-lit
  // segment 0 shared by every code. Rotating all columns by a seed-derived
  // offset moves that always-on column to a seed-dependent position (and,
  // when render_bits < M, often outside the rendered window entirely).
  // Rotation is a single column permutation applied to every row, so codes
  // stay distinct and patterns stay clean stripes — only their phase shifts.
  //
  // render_bits is folded into the hash so each bit-count picks its OWN
  // rotation. Sweeping render_bits then re-phases the pattern per step
  // instead of smoothly revealing a fixed one (which read as a "zip").
  // (For non-Walsh codebooks there's no DC column to hide, but the same
  // rotation is applied uniformly as a free, seed-driven phase shift.)
  uint32_t hr = (uint32_t)s.seed * 2654435761u;
  hr ^= (uint32_t)render_bits * 0x9E3779B1u;
  hr ^= hr >> 15; hr *= 0x2C1B3C6Du; hr ^= hr >> 12;
  int rot = (int)(hr % (uint32_t)M);

  // Generate the selected codebook (unrotated) into scratch, then store it
  // with the cyclic column rotation applied. `static` to keep the 4 KB
  // matrix off the stack — rebuild runs single-threaded and non-reentrant.
  static uint8_t code[MAX_HADAMARD][MAX_HADAMARD];
  gen_codebook(s, s.codebook, M, code);
  for (int i = 0; i < M; i++) {
    for (int j = 0; j < M; j++) s.sys_b_bits[i][j] = code[i][(j + rot) % M];
  }
  // Sort code indices ascending by visible (windowed) horizontal entropy.
  // Insertion sort, tie-break by code index for determinism. sorted[0] is
  // the all-1s code (0 transitions). Seed is deliberately NOT used here —
  // the entropy order is fixed; only the per-page bar assignment is seeded.
  int ent[MAX_HADAMARD];
  for (int i = 0; i < M; i++) {
    s.sys_b_sorted[i] = i;
    ent[i] = code_window_transitions(s, i, M, render_bits);
  }
  for (int i = 1; i < M; i++) {
    int v = s.sys_b_sorted[i], ev = ent[v], j = i - 1;
    while (j >= 0) {
      int oj = s.sys_b_sorted[j];
      if (ent[oj] > ev || (ent[oj] == ev && oj > v)) {
        s.sys_b_sorted[j + 1] = s.sys_b_sorted[j]; j--;
      } else break;
    }
    s.sys_b_sorted[j + 1] = v;
  }
  s.sys_b_size_cached = M;
  s.sys_b_rb_cached = render_bits;
  s.sys_b_seed_cached = s.seed;
  s.sys_b_cb_cached = s.codebook;
}

static void ensure_caches(State& s) {
  // Hadamard sizes constrained to powers of 2 in [4, MAX_HADAMARD].
  int M = round_up_pow2(s.hadamard_size);
  if (M < 4) M = 4;
  if (M > MAX_HADAMARD) M = MAX_HADAMARD;
  if (M != s.hadamard_size) s.hadamard_size = M;

  int rb = clampi(s.render_bits, 1, 64);
  if (s.sys_a_dirty || s.sys_a_seed != s.seed) rebuild_sys_a(s);
  if (s.sys_b_size_cached != M || s.sys_b_rb_cached != rb
      || s.sys_b_seed_cached != s.seed || s.sys_b_cb_cached != s.codebook)
    rebuild_sys_b(s, M, rb);
}

// Sample the per-channel waveform for a 2-bit code.
//   (0,0) → 0     (1,1) → 1
//   (1,0) → square @ mod_phase   (0,1) → |sin(2π mod_phase)|
static float channel_value(int code_msb, int code_lsb, double mod_phase) {
  if (code_msb == 0 && code_lsb == 0) return 0.0f;
  if (code_msb == 1 && code_lsb == 1) return 1.0f;
  if (code_msb == 1 && code_lsb == 0) {
    return (mod_phase - std::floor(mod_phase)) < 0.5 ? 1.0f : 0.0f;
  }
  // (0,1)
  return (float)std::fabs(std::sin(mod_phase * 2.0 * 3.14159265358979));
}

// Type-level setup: schema + the shared compute PSO. Runs once per type.
void module_init() {
  state::init("source.light.orthomod", {1, 0, 1},
    state::Schema()
      // Top-level manual: high-level "what is this / how to use / what to try".
      .helpField("intro",
        "## Orthomod\n"
        "A beat-synced bar pattern driven by orthogonal code systems (Hadamard/Walsh "
        "and friends). An envelope fires on the beat (or on gate/trigger) and, as it "
        "decays, sweeps the four bars from a solid flash down through progressively "
        "busier stripe patterns. It also publishes **ch1–ch4** and **env** rails to "
        "modulate other effects.\n\n"
        "**Try:** set *Beat Sync* and let it pulse; sweep *Codebook* for different "
        "stripe families; tap the **env** rail into another effect for beat-locked motion.")
      // --- Trigger ---
      .group("trigger", "Trigger")
        .groupHelp(
          "How the envelope is fired. **Beat Sync** locks pulses to the transport (set "
          "*Off* to go manual). **Gate** re-fires on each rising edge and holds the "
          "decay phase; **Trigger** is a one-shot. All three snap the envelope to full "
          "and then let it fall.")
      // Manual trigger surface. gate = rising-edge re-fire; trigger = one
      // shot event. Both snap the envelope to 1 just like a beat crossing,
      // and work even when beat_multiplier is "Off".
      .boolField ("gate",            false,                  state::PrimaryInput).label("Gate", "Gate")
      .eventField("trigger",                                 state::PrimaryInput).label("Trigger", "Trig")
      .selectField("beat_multiplier", 2, state::PrimaryInput,
                   {{"Off", 5}, {"1/4", 0}, {"1/2", 1}, {"1", 2}, {"2", 3}, {"4", 4}}).label("Beat Sync", "Beat")
      // --- Appearance ---
      .group("appearance", "Appearance")
      .floatField("primary_hue",            0.08f, 0.0f, 1.0f,  state::PrimaryInput).label("Hue", "Hue")
      .floatField("saturation",             0.9f,  0.0f, 1.0f,  state::PrimaryInput).label("Saturation", "Sat")
      .floatField("intensity",              1.0f,  0.0f, 2.0f,  state::PrimaryInput).label("Intensity", "Int")
      // --- Envelope ---
      .group("envelope", "Envelope")
        .groupHelp(
          "Shapes the fall after each hit. **Decay** runs while the gate is held (or "
          "after a beat/trigger); **Release** takes over once the gate is let go — give "
          "it a faster time for a snappier tail. The *Curve* knobs bend each phase "
          "(negative = crushed/percussive, positive = sustained).")
      .floatField("decay_time_beats",       1.0f,  0.05f, 4.0f, state::PrimaryInput).label("Decay Time", "Decay")
      .floatField("decay_curve",            0.0f, -1.0f, 1.0f,  state::PrimaryInput).label("Decay Curve", "DecCrv")
      .floatField("release_time_beats",     0.5f,  0.05f, 4.0f, state::PrimaryInput).label("Release Time", "Rel")
      .floatField("release_curve",          0.0f, -1.0f, 1.0f,  state::PrimaryInput).label("Release Curve", "RelCrv")
      // --- Modulation ---
      .group("modulation", "Modulation")
      .floatField("scatter_max",            0.15f, 0.0f, 0.5f,  state::PrimaryInput).label("Scatter", "Scat")
      .floatField("channel_brightness_mod", 0.5f,  0.0f, 1.0f,  state::PrimaryInput).label("Channel Mod", "ChMod")
      .floatField("env_brightness_curve",   0.0f, -1.0f, 1.0f,  state::PrimaryInput).label("Env Brightness", "EnvBri")
      .floatField("mod_rate_hz",            15.0f, 0.0f, 30.0f, state::PrimaryInput).label("Mod Rate", "Rate")
      // --- Pattern ---
      .group("pattern", "Pattern")
        .groupHelp(
          "The code system that draws the bars. **Codebook** picks the stripe family "
          "(Walsh = clean sequency, Thue-Morse = fractal, etc.). *Sweep Start/End* set "
          "the order→chaos range the envelope traverses (set Start > End to sweep in "
          "reverse). **Size**/**Bits** control resolution; **Seed** re-rolls the whole "
          "arrangement.")
      .selectField("codebook", CB_WALSH, state::PrimaryInput,
                   {{"Walsh", CB_WALSH}, {"Random", CB_RANDOM}, {"LFSR", CB_LFSR},
                    {"Gray", CB_GRAY}, {"Binary", CB_BINARY}, {"Thue-Morse", CB_THUE_MORSE}}, /*wrap=*/true).label("Codebook", "Code")
      .floatField("start",         0.0f, 0.0f, 1.0f, state::PrimaryInput).label("Sweep Start", "Start")
      .floatField("end",           1.0f, 0.0f, 1.0f, state::PrimaryInput).label("Sweep End", "End")
      .boolField ("keep_flash",    true,             state::PrimaryInput).label("Keep Flash", "Flash")
      .intField  ("hadamard_size", 32, 4, MAX_HADAMARD, state::PrimaryInput).label("Pattern Size", "Size")
      .intField  ("render_bits",   13, 1, 64,           state::PrimaryInput).label("Render Bits", "Bits")
      .floatField("inset_top",     0.0f, 0.0f, 0.5f,    state::PrimaryInput).label("Inset Top", "Top")
      .floatField("inset_bottom",  0.0f, 0.0f, 0.5f,    state::PrimaryInput).label("Inset Bottom", "Bottom")
      .floatField("decay_jitter",  0.0f, 0.0f, 1.0f,    state::PrimaryInput).label("Decay Jitter", "Jitter")
      .intField  ("seed",          1, 0, 0x7FFFFFFF,    state::PrimaryInput).label("Seed", "Seed")
      // --- Output rails (5 separate floats per meta-question #3) ---
      .floatField("ch1", 0.0f, 0.0f, 1.0f, state::PrimaryOutput)
      .floatField("ch2", 0.0f, 0.0f, 1.0f, state::PrimaryOutput)
      .floatField("ch3", 0.0f, 0.0f, 1.0f, state::PrimaryOutput)
      .floatField("ch4", 0.0f, 0.0f, 1.0f, state::PrimaryOutput)
      .floatField("env", 0.0f, 0.0f, 1.0f, state::PrimaryOutput)
      // --- I/O ---
      .textureField("tex_in",  state::PrimaryInput)
      .textureField("tex_out", state::PrimaryOutput)
        .capability(state::Capability::Generator)
        .capability(state::Capability::SeekableApproximate)
    );

  if (gpu::Device::backend() == gpu::Backend::None) return;

  state::registerShaderSPV("orthomod_render", RENDER_SPV, RENDER_SPV_SIZE);
  auto cs = gpu::Device::createShaderModuleByName("orthomod_render");
  if (!cs) return;

  s_pso = gpu::Device::createComputePSO(cs, "main", gpu::Bindings()
      .tex2d(0)
      .storageTex2d(1)
      .uniform(2)
      .storage(3));
  state::log("orthomod: module initialized");
}

// Per-instance construction: allocate State + its own GPU buffers.
void* create() {
  auto* s = new State();
  s->uniform_buf = gpu::Device::createBuffer(sizeof(Uniforms), gpu::BufferUsage::Uniform);
  s->page_buf    = gpu::Device::createBuffer(sizeof(uint32_t) * MAX_PAGE_BITS,
                                             gpu::BufferUsage::Storage);
  return s;
}

void destroy(void* self) {
  auto* s = static_cast<State*>(self);
  if (!s) return;
  s->uniform_buf.release();
  s->page_buf.release();
  delete s;
}

// Per-instance init tail: reset params/envelope/edge-state, re-arm the
// BeatTick, and invalidate the cache keys so the pattern tables rebuild on
// first use.
void init(void* self) {
  auto* s = static_cast<State*>(self);
  if (!s) return;
  s->initialized = false;
  s->sys_a_dirty = true;
  s->sys_a_seed = -1;
  s->sys_b_size_cached = -1;
  s->sys_b_rb_cached = -1;
  s->sys_b_seed_cached = -1;
  s->sys_b_cb_cached = -1;
  s->linear_env = 0.0;
  s->mod_phase = 0.0;
  s->gate = false;
  s->gate_prev = false;
  s->trigger_prev = 0.0f;
  s->gate_open = false;
  s->tick.reset();
  std::memset(s->sys_a_rows, 0, sizeof(s->sys_a_rows));
  std::memset(s->sys_b_bits, 0, sizeof(s->sys_b_bits));

  if (!s_pso.valid()) return;
  if (!s->uniform_buf.valid() || !s->page_buf.valid()) return;
  s->initialized = true;
  state::log("orthomod: initialized");
}

void tick(void* self, double dt) {
  auto* s = static_cast<State*>(self);
  if (!s) return;
  if (!s->initialized) return;
  ensure_caches(*s);

  float beat_mult = beat_multiplier_value(s->beat_multiplier_id);
  int crossings = s->tick.tick(beat_mult);   // 0 crossings when "Off" (mult 0)
  if (crossings > 0) { s->linear_env = 1.0; s->gate_open = true; }  // beat → decay phase

  // Fall: linear_env loses 1 unit over the active phase's time. Decay while
  // the gate is held (or after a momentary fire); release once let go.
  double bpm = host::bpm();
  if (bpm < 1.0) bpm = 120.0;
  double phase_beats = s->gate_open ? (double)s->decay_time_beats
                                    : (double)s->release_time_beats;
  double phase_seconds = phase_beats * 60.0 / bpm;
  if (phase_seconds > 1e-5) {
    s->linear_env -= dt / phase_seconds;
    if (s->linear_env < 0.0) s->linear_env = 0.0;
  }

  // §2.1 accumulator: phase advances regardless of rate changes.
  if (s->mod_rate_hz > 0.0f) {
    s->mod_phase += dt * (double)s->mod_rate_hz;
    if (s->mod_phase > 1024.0) s->mod_phase -= std::floor(s->mod_phase);
  }
}

// Switch from decay to release phase (gate let go). Remap linear_env so the
// OUTPUT env is continuous across the decay→release curve change — without
// this the brightness pops at release whenever the two curves differ.
static void enter_release(State& s) {
  if (!s.gate_open) return;
  float e_decay   = fx::signedSliderToExp(clampf(s.decay_curve,   -1.0f, 1.0f));
  float e_release = fx::signedSliderToExp(clampf(s.release_curve, -1.0f, 1.0f));
  if (s.linear_env > 0.0 && e_release > 1e-6f) {
    double out = std::pow(s.linear_env, (double)e_decay);     // current output env
    s.linear_env = std::pow(out, 1.0 / (double)e_release);    // matching linear for release curve
    if (s.linear_env > 1.0) s.linear_env = 1.0;
  }
  s.gate_open = false;
}


void on_state_patched(void* self, int n, const char* pb, const int* off,
                      const int* len, const int* ops) {
  auto* s = static_cast<State*>(self);
  if (!s) return;
  for (int i = 0; i < n; i++) {
    const char* path = pb + off[i];
    int plen = len[i];
    int op = ops[i];

    if (op == state::PatchReplace) {
      if (state::pathIs(path, plen, "gate")) {
        bool new_gate = state::patchFloat(i) != 0.0f;
        if (new_gate && !s->gate_prev) {
          s->linear_env = 1.0;     // attack → decay phase
          s->gate_open = true;
        } else if (!new_gate && s->gate_prev) {
          enter_release(*s);       // gate let go → release phase
        }
        s->gate = new_gate;
        s->gate_prev = new_gate;
      }
      else if (state::pathIs(path, plen, "trigger")) {
        // Momentary event value (1 held / 0 released), replayed every
        // frame — fire only on the 0→1 rising edge, exactly like gate.
        float v = state::patchFloat(i);
        if (v != 0.0f && s->trigger_prev == 0.0f) { s->linear_env = 1.0; s->gate_open = true; }
        s->trigger_prev = v;
      }
      else if (state::pathIs(path, plen, "beat_multiplier")) s->beat_multiplier_id   = (int)state::patchFloat(i);
      else if (state::pathIs(path, plen, "primary_hue"))     s->primary_hue          = state::patchFloat(i);
      else if (state::pathIs(path, plen, "saturation"))      s->saturation           = state::patchFloat(i);
      else if (state::pathIs(path, plen, "intensity"))       s->intensity            = state::patchFloat(i);
      else if (state::pathIs(path, plen, "decay_time_beats"))s->decay_time_beats     = state::patchFloat(i);
      else if (state::pathIs(path, plen, "decay_curve"))     s->decay_curve          = state::patchFloat(i);
      else if (state::pathIs(path, plen, "release_time_beats")) s->release_time_beats = state::patchFloat(i);
      else if (state::pathIs(path, plen, "release_curve"))   s->release_curve        = state::patchFloat(i);
      else if (state::pathIs(path, plen, "scatter_max"))     s->scatter_max          = state::patchFloat(i);
      else if (state::pathIs(path, plen, "channel_brightness_mod")) s->channel_brightness_mod = state::patchFloat(i);
      else if (state::pathIs(path, plen, "env_brightness_curve")) s->env_brightness_curve = state::patchFloat(i);
      else if (state::pathIs(path, plen, "mod_rate_hz"))     s->mod_rate_hz          = state::patchFloat(i);
      else if (state::pathIs(path, plen, "codebook"))        s->codebook             = (int)state::patchFloat(i);
      else if (state::pathIs(path, plen, "start"))           s->start                = state::patchFloat(i);
      else if (state::pathIs(path, plen, "end"))             s->end                  = state::patchFloat(i);
      else if (state::pathIs(path, plen, "keep_flash"))      s->keep_flash           = state::patchFloat(i) != 0.0f;
      else if (state::pathIs(path, plen, "hadamard_size")) {
        int v = (int)state::patchFloat(i);
        if (v != s->hadamard_size) { s->hadamard_size = v; s->sys_b_size_cached = -1; }
      }
      else if (state::pathIs(path, plen, "render_bits"))     s->render_bits          = (int)state::patchFloat(i);
      else if (state::pathIs(path, plen, "inset_top"))       s->inset_top            = state::patchFloat(i);
      else if (state::pathIs(path, plen, "inset_bottom"))    s->inset_bottom         = state::patchFloat(i);
      else if (state::pathIs(path, plen, "decay_jitter"))    s->decay_jitter         = state::patchFloat(i);
      else if (state::pathIs(path, plen, "seed")) {
        // Seed re-rolls System A's column shuffle, System B's column
        // rotation (rebuilt via ensure_caches' seed cache key) and System
        // B's per-page bar assignment (applied live in render).
        int v = (int)state::patchFloat(i);
        if (v != s->seed) { s->seed = v; s->sys_a_dirty = true; }
      }
    }
  }
}

static void publish_output(const char* name, float value) {
  auto vh = val::number(value);
  state::setValPath(name, vh);
  val::release(vh);
}

void render(void* self, int vp_w, int vp_h) {
  auto* s = static_cast<State*>(self);
  if (!s) return;
  if (!s->initialized || vp_w <= 0 || vp_h <= 0) return;
  ensure_caches(*s);

  auto in  = gpu::Device::textureForField("tex_in");
  auto out = gpu::Device::textureForField("tex_out");
  if (!in.valid() || !out.valid()) return;

  // Compute env from linear_env with the active phase's curve (decay while
  // the gate is held / after a momentary fire, release once let go).
  float linear = (float)s->linear_env;
  if (linear < 0.0f) linear = 0.0f;
  float phase_curve = s->gate_open ? s->decay_curve : s->release_curve;
  float env = std::pow(linear, fx::signedSliderToExp(clampf(phase_curve, -1.0f, 1.0f)));
  if (env < 0.0f) env = 0.0f;
  if (env > 1.0f) env = 1.0f;

  // System A — pick row by env.
  int idx_A = (int)std::floor((1.0f - env) * (float)SYS_A_N);
  idx_A = clampi(idx_A, 0, SYS_A_N - 1);
  const uint8_t* rowA = s->sys_a_rows[idx_A];

  // Per-bar 2-bit code → RAW channel waveform [0,1] (NOT pre-multiplied by
  // env). The global envelope enters brightness only through env_brightness
  // in the shader, so env_brightness_curve fully governs how much the
  // envelope dims the bars — even via the channel_brightness_mod branch.
  // Scatter likewise uses the raw channel (its env dependence comes from
  // the tent(env) factor). The published ch1..ch4 rails ARE env-scaled
  // below so they decay to 0 with the envelope as modulation sources.
  float ch[BARS];
  for (int b = 0; b < BARS; b++) {
    int code_msb = rowA[b * 2 + 0];
    int code_lsb = rowA[b * 2 + 1];
    ch[b] = channel_value(code_msb, code_lsb, s->mod_phase);
  }

  // System B — pick the entropy level by env.
  //
  // Codes are sorted ascending by visible entropy; pages of 4 consecutive
  // sorted codes hold same-ish entropy. start/end pick the page range the
  // sweep traverses (0 = orderly end, 1 = entropic end): the sweep runs
  // pg_start→pg_end as the envelope decays, so start > end sweeps in
  // REVERSE. When keep_flash is on, a synthetic SOLID page (all-1s) is
  // prepended as level 0 regardless of the range. Within a page the 4
  // codes are dealt to the 4 bars by a per-page seed-driven permutation,
  // so no bar has a fixed personality and every code can land on any bar.
  int M = s->hadamard_size;
  int P = M / 4;
  if (P < 1) P = 1;

  float start_f = clampf(s->start, 0.0f, 1.0f);
  float end_f   = clampf(s->end,   0.0f, 1.0f);
  int pg_start = clampi((int)std::lround(start_f * (float)(P - 1)), 0, P - 1);
  int pg_end   = clampi((int)std::lround(end_f   * (float)(P - 1)), 0, P - 1);
  int step = (pg_end >= pg_start) ? 1 : -1;     // reversed range → sweep backwards
  int cropped = (pg_end >= pg_start ? pg_end - pg_start : pg_start - pg_end) + 1;

  bool flash = s->keep_flash;
  int num_levels = cropped + (flash ? 1 : 0);
  if (num_levels < 1) num_levels = 1;
  float level_pos = (1.0f - env) * (float)num_levels;

  // Per-bar decay jitter — fixed, seed-independent offsets (golden-ratio
  // spread → well-separated, non-monotonic) that DELAY each bar's level
  // crossing. Subtracting (never advancing) keeps the env=1 solid flash
  // intact: at level_pos ≈ 0 every bar still clamps to level 0. Each bar
  // resolves its own level, so under jitter bars can sit in adjacent
  // pages and flip codes at slightly offset envelope positions.
  static const float BAR_JIT[BARS] = { 0.618f, 0.236f, 0.854f, 0.472f };

  uint32_t page_bits[MAX_PAGE_BITS];
  int debug_level = 0;
  for (int b = 0; b < BARS; b++) {
    float jittered = level_pos - s->decay_jitter * BAR_JIT[b];
    int level_b = (int)std::floor(jittered);
    level_b = clampi(level_b, 0, num_levels - 1);
    if (b == 0) debug_level = level_b;            // for the scatter hash key
    if (flash && level_b == 0) {
      // Solid flash — every segment lit for this bar.
      for (int c = 0; c < M; c++) page_bits[b * M + c] = 1u;
    } else {
      int crop_idx = flash ? (level_b - 1) : level_b;   // 0 .. cropped-1
      crop_idx = clampi(crop_idx, 0, cropped - 1);
      int page = pg_start + step * crop_idx;            // pg_start .. pg_end (either direction)
      uint32_t h = ((uint32_t)s->seed * 2654435761u) ^ ((uint32_t)page * 40503u);
      int perm[4];
      make_perm4(h, perm);
      int code = s->sys_b_sorted[page * 4 + perm[b]];
      for (int c = 0; c < M; c++) page_bits[b * M + c] = (uint32_t)s->sys_b_bits[code][c];
    }
  }
  s->page_buf.writeBytes(page_bits, (int)sizeof(uint32_t) * BARS * M);

  // Uniforms.
  Uniforms u = {};
  u.ch0 = ch[0]; u.ch1 = ch[1]; u.ch2 = ch[2]; u.ch3 = ch[3];
  u.env = env;
  u.primary_hue = s->primary_hue;
  u.saturation = s->saturation;
  u.intensity = s->intensity;
  u.scatter_max = s->scatter_max;
  u.channel_brightness_mod = s->channel_brightness_mod;
  u.inset_top = clampf(s->inset_top, 0.0f, 0.5f);
  u.inset_bottom = clampf(s->inset_bottom, 0.0f, 0.5f);
  u.hadamard_size = (uint32_t)M;
  u.render_bits = (uint32_t)clampi(s->render_bits, 1, 64);
  u.page_idx = (uint32_t)debug_level;
  u.seed = (uint32_t)s->seed;
  // Envelope shaped by the brightness power curve (style guide §1.3).
  // Only affects bar brightness in the shader — page selection, channels,
  // and scatter all still use the raw env.
  u.env_brightness = std::pow(env, fx::signedSliderToExp(clampf(s->env_brightness_curve, -1.0f, 1.0f)));
  s->uniform_buf.writeOne(u);

  // Dispatch.
  auto cp = gpu::ComputePass::begin();
  cp.setPSO(s_pso);
  cp.setTexture(in,  0, 0);
  cp.setTexture(out, 1, 1);
  cp.setBuffer(s->uniform_buf, 2);
  cp.setBuffer(s->page_buf,    3);
  cp.dispatch((vp_w + 7) / 8, (vp_h + 7) / 8);
  cp.end();
  gpu::Device::submit();

  // Publish output rails — env-scaled so they decay to 0 with the
  // envelope (the shader uses the raw ch[] for brightness/scatter).
  publish_output("ch1", ch[0] * env);
  publish_output("ch2", ch[1] * env);
  publish_output("ch3", ch[2] * env);
  publish_output("ch4", ch[3] * env);
  publish_output("env", env);
}

} // namespace orthomod
