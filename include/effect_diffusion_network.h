#pragma once
/*
 * effect_diffusion_network.h — 4-bar scalar diffusion network.
 *
 * A row of 4 bars each holds a scalar "value". On each HOP the whole
 * vector is multiplied by a 4×4 mixing matrix M: v ← M·v. Hops happen at
 * `rate` Hz (a visible tempo — NOT an audio-rate substep), and each hop
 * advances through a cycle of `pattern_count` seeded matrices, so the
 * routing evolves instead of locking into one pattern.
 *
 * The matrix has NO spatial structure — connections are random, never
 * "to my neighbour":
 *
 *   spread   — how many bars a source feeds, PER HOP. spread 0 → a seeded
 *              DERANGEMENT (permutation, no fixed points): every bar dumps
 *              its full value onto exactly one OTHER bar this hop. As
 *              spread rises, each column's weight bleeds from that one
 *              target out to other randomly-chosen bars; spread 1 → a fully
 *              random per-column distribution. Convex blend → column sum is
 *              preserved at every spread.
 *
 *   feedback — the matrix column sum = energy retained PER HOP (literal):
 *                feedback 1.0 → conserved → reverberates forever.
 *                feedback <1  → decays (e.g. 0.5 = halve each hop).
 *                feedback >1  → grows (max ~1.2).
 *              Because it's per-hop at a visible rate, low values decay
 *              fast (0.02 ≈ gone next hop) — unlike an audio-rate substep
 *              where the knob is unusably compressed.
 *
 *   seed     — re-rolls the permutations + random spread weights.
 *   pattern_count — number of distinct matrices cycled through (cycle len).
 *   rate     — hops per second (the visible bounce tempo).
 *
 * Each bar also carries a HUE (turns, [0,1)) that diffuses ALONGSIDE the
 * value: when intensity flows j→i, it carries hue_j, so a bar's new hue is
 * the intensity-weighted circular mean of its incoming hues. Hue is
 * circular, so the mean is taken as a 2D (cos,sin) vector sum, not a scalar
 * average. `hue_spread` adds a seeded per-edge hue rotation, so a hue is
 * twisted a little each time it's transferred — colours cycle as they bounce.
 *
 * Header-only, CPU-only, no GPU resources owned. The hosting effect packs
 * the per-bar values + hues into uniforms and renders them.
 */

#include <cmath>
#include <cstdint>

namespace fx {

class DiffusionNetwork4 {
public:
  static constexpr int   N           = 4;
  static constexpr int   kMaxN       = 16;     // max pattern_count
  static constexpr int   kMaxHops    = 256;    // per-frame hop clamp
  static constexpr float kEnvRelease    = 0.98f;  // decay-phase envelope release/hop
  static constexpr float kShapeStrength = 0.80f;  // decay_shaping curve strength
  static constexpr float kTau           = 6.28318530717958648f;

  struct Params {
    float    feedback        = 0.90f;  // 0..1.2, per-hop energy gain
    float    spread          = 0.30f;  // 0..1, permutation → random per hop
    float    spread_contrast = 0.0f;   // 0..1, bimodal-ize the spread weights
    float    decay_shaping   = 0.0f;   // -1..1, ease the decay curve
    float    hue_spread      = 0.0f;   // 0..1, per-edge hue rotation on transfer
    float    hue_converge    = 0.0f;   // 0..1, pull hues to home_hue at the tail
    float    home_hue        = 0.0f;   // target hue (turns) for hue_converge
    uint32_t seed            = 0;      // re-rolls permutations + spread weights
    int      pattern_count   = 4;      // distinct matrices cycled (1..kMaxN)
    float    rate            = 6.0f;   // hops per second (visible tempo)
  };

  void setParams(const Params& p) {
    // Only the matrix STRUCTURE (routing + per-edge hue shifts) needs a
    // rebuild; feedback and decay_shaping are applied as a per-hop scalar
    // gain, so they're free to change every frame.
    if (p.spread != params_.spread || p.seed != params_.seed
        || p.pattern_count != params_.pattern_count
        || p.spread_contrast != params_.spread_contrast
        || p.hue_spread != params_.hue_spread) {
      matrix_dirty_ = true;
    }
    params_ = p;
    if (params_.pattern_count < 1)      params_.pattern_count = 1;
    if (params_.pattern_count > kMaxN)  params_.pattern_count = kMaxN;
    if (params_.rate < 0.0f)            params_.rate = 0.0f;
  }

  /// Inject `amount` of value into one bar at hue `hue` (turns, [0,1)).
  /// The bar's hue becomes the intensity-weighted circular blend of its
  /// existing hue and the injected one (so injecting into a dark bar lands
  /// exactly on `hue`).
  void impulse(int bar, float amount, float hue) {
    if (bar < 0 || bar >= N || amount <= 0.0f) return;
    float a0 = kTau * h_[bar], a1 = kTau * hue;
    float cx = v_[bar] * std::cos(a0) + amount * std::cos(a1);
    float cy = v_[bar] * std::sin(a0) + amount * std::sin(a1);
    v_[bar] += amount;
    if (cx * cx + cy * cy > 1e-12f) h_[bar] = wrapTurns(std::atan2(cy, cx) / kTau);
  }

  /// Inject `amount` into every bar at hue `hue` (a uniform splash).
  void impulseAll(float amount, float hue) {
    for (int b = 0; b < N; b++) impulse(b, amount, hue);
  }

  void step(float dt) {
    if (dt <= 0.0f || params_.rate <= 0.0f) return;
    if (matrix_dirty_) { rebuildMatrices(); matrix_dirty_ = false; }

    accum_ += dt;
    int hops = (int)std::floor(accum_ * params_.rate);
    if (hops <= 0) return;
    if (hops > kMaxHops) { hops = kMaxHops; accum_ = 0.0f; }
    else { accum_ -= (float)hops / params_.rate; }

    int nc = params_.pattern_count;
    for (int s = 0; s < hops; s++) {
      int    kk = hop_idx_ % nc;
      const float (*M)[N]  = M_[kk];
      const float (*CD)[N] = cd_[kk];   // cos of per-edge hue shift
      const float (*SD)[N] = sd_[kk];   // sin of per-edge hue shift

      // Decay shaping: read total energy as a proxy for "where we are in
      // the decay phase" (1 = just peaked, 0 = faded) via a peak-follower
      // envelope, then EASE the decay by reducing the per-hop feedback in
      // one phase region. It only ever lowers feedback (never adds energy,
      // so no runaway), bending the curve:
      //   decay_shaping > 0 → full feedback while bright, faster near the
      //                       tail → sustain then snap.
      //   decay_shaping < 0 → faster while bright, full feedback near the
      //                       tail → pluck then linger.
      float total = 0.0f;
      for (int i = 0; i < N; i++) total += v_[i];
      if (total > env_) env_ = total; else env_ *= kEnvRelease;
      float phase = (env_ > 1e-6f) ? (total / env_) : 0.0f;
      if (phase > 1.0f) phase = 1.0f;
      float fb = params_.feedback;
      if (params_.decay_shaping != 0.0f && params_.feedback < 1.0f) {
        // Ease the decay by scaling the per-hop decay fraction (1-feedback)
        // up or down with the decay phase. Scaling the decay (never the
        // gain) means it can't freeze or grow — it always decays, just on a
        // bent curve:
        //   +shaping → small decay while bright, large near the tail
        //              (sustain then snap).
        //   -shaping → large decay while bright, small near the tail
        //              (pluck then linger).
        float d = 1.0f - params_.feedback;
        float factor = 1.0f + params_.decay_shaping * kShapeStrength * (1.0f - 2.0f * phase);
        float deff = d * factor;
        if (deff < 0.0f) deff = 0.0f; else if (deff > 1.0f) deff = 1.0f;
        fb = 1.0f - deff;
      }

      // Per-bar (cos,sin) of the current hue — for the circular mean below.
      float cj[N], sj[N];
      for (int j = 0; j < N; j++) { float a = kTau * h_[j]; cj[j] = std::cos(a); sj[j] = std::sin(a); }

      // Redistribute value (column sum 1) × feedback, AND carry hue with it:
      // each j's hue (rotated by the edge's hue shift) is summed as a 2D
      // vector weighted by the intensity it sends, then atan2'd back.
      float nv[N], nhx[N], nhy[N];
      for (int i = 0; i < N; i++) {
        float a = 0.0f, hx = 0.0f, hy = 0.0f;
        for (int j = 0; j < N; j++) {
          float w = M[i][j] * v_[j];           // intensity flowing j→i
          a += w;
          // rotate hue_j by the edge shift via angle-addition (no per-hop trig)
          float rc = cj[j] * CD[i][j] - sj[j] * SD[i][j];
          float rs = sj[j] * CD[i][j] + cj[j] * SD[i][j];
          hx += w * rc;
          hy += w * rs;
        }
        nv[i] = a * fb; nhx[i] = hx; nhy[i] = hy;
      }
      // Hue convergence: pull hues toward home_hue, squared in the decay
      // phase so it only bites at the tail (bright → free to wander, faded →
      // settle back to the band colour).
      float conv = 0.0f;
      if (params_.hue_converge > 0.0f) {
        float t = 1.0f - phase;
        conv = params_.hue_converge * t * t;
        if (conv > 1.0f) conv = 1.0f;
      }

      for (int i = 0; i < N; i++) {
        // Clamp purely as inf/NaN insurance for long runs at feedback>1.
        v_[i] = nv[i] < 0.0f ? 0.0f : (nv[i] > 1e6f ? 1e6f : nv[i]);
        // Keep the old hue when ~no intensity arrives (mean is undefined).
        if (nhx[i] * nhx[i] + nhy[i] * nhy[i] > 1e-12f)
          h_[i] = wrapTurns(std::atan2(nhy[i], nhx[i]) / kTau);
        if (conv > 0.0f) {
          // Rotate along the shortest arc toward home_hue by fraction conv.
          float delta = wrapTurns(params_.home_hue - h_[i] + 0.5f) - 0.5f;
          h_[i] = wrapTurns(h_[i] + delta * conv);
        }
      }
      hop_idx_ = (hop_idx_ + 1) % nc;
    }
  }

  void reset() {
    for (int i = 0; i < N; i++) { v_[i] = 0.0f; h_[i] = 0.0f; }
    accum_ = 0.0f;
    hop_idx_ = 0;
    env_ = 0.0f;
    matrix_dirty_ = true;
  }

  float value(int bar) const { return (bar >= 0 && bar < N) ? v_[bar] : 0.0f; }
  float hue(int bar)   const { return (bar >= 0 && bar < N) ? h_[bar] : 0.0f; }

  /// Active-matrix entry, for debug overlays.
  float matrix(int i, int j) const {
    int nc = params_.pattern_count < 1 ? 1 : params_.pattern_count;
    return (i >= 0 && i < N && j >= 0 && j < N) ? M_[hop_idx_ % nc][i][j] : 0.0f;
  }

  /// Number of floats exportMatrices() writes (pattern_count × 48).
  int matrixFloatCount() const { return params_.pattern_count * N * N * 3; }

  /// Export all cycled matrices for upload to a GPU shader that does the
  /// stepping. Layout per pattern k (48 floats): M[16], cos(dHue)[16],
  /// sin(dHue)[16], each row-major i*N+j. Builds the matrices if stale.
  void exportMatrices(float* dst) {
    if (matrix_dirty_) { rebuildMatrices(); matrix_dirty_ = false; }
    for (int k = 0; k < params_.pattern_count; k++) {
      int base = k * 48;
      for (int i = 0; i < N; i++)
        for (int j = 0; j < N; j++) {
          dst[base +  0 + i * N + j] = M_[k][i][j];
          dst[base + 16 + i * N + j] = cd_[k][i][j];
          dst[base + 32 + i * N + j] = sd_[k][i][j];
        }
    }
  }

private:
  Params params_;
  float  v_[N]            = {0.f, 0.f, 0.f, 0.f};
  float  h_[N]            = {0.f, 0.f, 0.f, 0.f};   // per-bar hue (turns)
  float  M_[kMaxN][N][N]  = {{{0}}};
  float  cd_[kMaxN][N][N] = {{{0}}};                // cos of per-edge hue shift
  float  sd_[kMaxN][N][N] = {{{0}}};                // sin of per-edge hue shift
  float  accum_          = 0.0f;
  int    hop_idx_        = 0;
  float  env_            = 0.0f;   // decay-phase envelope follower
  bool   matrix_dirty_   = true;

  static float wrapTurns(float h) { h -= std::floor(h); return h; }

  // Stateless seeded hash → uniform [0, 1). No spatial structure.
  static float hash01(uint32_t seed, uint32_t a, uint32_t b) {
    uint32_t h = seed ^ (a * 0x9E3779B1u) ^ (b * 0x85EBCA77u);
    h ^= h >> 16; h *= 0x7FEB352Du; h ^= h >> 15; h *= 0x846CA68Bu; h ^= h >> 16;
    return (float)(h >> 8) * (1.0f / 16777216.0f);
  }

  // Bimodal contrast on a weight in [0, 1]: a symmetric logistic S-curve
  // that pushes values toward 0 or 1. contrast 0 → unchanged; contrast 1 →
  // near-binary. Used to make a column's spread weights "all or nothing".
  static float contrastWeight(float r, float contrast) {
    if (contrast <= 0.0f) return r;
    if (r <= 0.0f) return 0.0f;
    if (r >= 1.0f) return 1.0f;
    float g = 1.0f + contrast * 8.0f;            // sharpness
    float a = std::pow(r, g), b = std::pow(1.0f - r, g);
    return a / (a + b);
  }

  void rebuildMatrices() {
    float sp = params_.spread < 0.0f ? 0.0f : (params_.spread > 1.0f ? 1.0f : params_.spread);
    float ct = params_.spread_contrast < 0.0f ? 0.0f
             : (params_.spread_contrast > 1.0f ? 1.0f : params_.spread_contrast);

    for (int k = 0; k < params_.pattern_count; k++) {
      uint32_t kseed = params_.seed ^ ((uint32_t)k * 0x9E3779B1u + 0x1234567u);

      // 1. Seeded random DERANGEMENT (no fixed points) via rejection-
      //    sampled Fisher–Yates: perm[j] = the bar column j feeds at spread 0.
      uint32_t rng = kseed * 747796405u + 2891336453u;
      auto next = [&]() -> uint32_t {
        rng = rng * 747796405u + 2891336453u;
        uint32_t x = ((rng >> ((rng >> 28) + 4u)) ^ rng) * 277803737u;
        return (x >> 22) ^ x;
      };
      int perm[N];
      bool ok = false;
      for (int attempt = 0; attempt < 32 && !ok; attempt++) {
        for (int i = 0; i < N; i++) perm[i] = i;
        for (int i = N - 1; i > 0; i--) {
          int r = (int)(next() % (uint32_t)(i + 1));
          int t = perm[i]; perm[i] = perm[r]; perm[r] = t;
        }
        ok = true;
        for (int j = 0; j < N; j++) if (perm[j] == j) { ok = false; break; }
      }
      if (!ok) for (int j = 0; j < N; j++) perm[j] = (j + 1) % N;  // cyclic fallback

      // 2. Per-column random spread distribution q (column-normalized),
      //    no spatial correlation. spread_contrast pushes the raw weights
      //    toward a bimodal (all-or-nothing) shape before normalizing.
      float q[N][N];
      for (int j = 0; j < N; j++) {
        float s = 0.0f;
        for (int i = 0; i < N; i++) {
          float r = hash01(kseed, (uint32_t)i, (uint32_t)j + 97u);
          q[i][j] = contrastWeight(r, ct);
          s += q[i][j];
        }
        if (s > 1e-6f) { float inv = 1.0f / s; for (int i = 0; i < N; i++) q[i][j] *= inv; }
        else           { for (int i = 0; i < N; i++) q[i][j] = 1.0f / (float)N; }  // degenerate → uniform
      }

      // 3. Blend permutation one-hot → random q by spread. Column sum stays
      //    1 (pure redistribution); feedback is applied per-hop in step().
      for (int j = 0; j < N; j++) {
        for (int i = 0; i < N; i++) {
          float onehot = (i == perm[j]) ? 1.0f : 0.0f;
          M_[k][i][j] = (1.0f - sp) * onehot + sp * q[i][j];
        }
      }

      // 4. Per-edge hue shift: a seeded rotation (±hue_spread·½turn) applied
      //    to a hue as it transfers along edge j→i. Stored as cos/sin so the
      //    hop loop rotates via angle-addition. hue_spread 0 → no shift.
      float hs = params_.hue_spread < 0.0f ? 0.0f : (params_.hue_spread > 1.0f ? 1.0f : params_.hue_spread);
      for (int j = 0; j < N; j++) {
        for (int i = 0; i < N; i++) {
          float r = hash01(kseed, (uint32_t)i + 211u, (uint32_t)j + 53u);  // [0,1)
          float dh = (2.0f * r - 1.0f) * hs * 3.14159265358979f;           // ±½turn max
          cd_[k][i][j] = std::cos(dh);
          sd_[k][i][j] = std::sin(dh);
        }
      }
    }
  }
};

} // namespace fx
