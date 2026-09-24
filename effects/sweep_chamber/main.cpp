/*
 * source.particles.sweep_chamber — swept-luma capture/release particle+line sim.
 *
 * Successor to source.legacy.double_chamber, rebuilt on flow_swarm's modern
 * structure. A built-in luma SWEEP (band-pass window over the input's luma)
 * "captures" one level of the image at a time; a coarse per-frame field
 * texture pair carries everything the sim needs:
 *   field_a — swept luma L' (mean + max + intra-cell peak offset)
 *   field_b — curl-noise background velocity + swept-image gradient
 * Particles and line tracers sample the field with single bilinear taps —
 * no full-res convolutions in any per-particle/per-step loop (the old
 * effect's cost). At either end of the sweep the window captures nothing,
 * the image reads as black, and everything free-flows on the noise field;
 * mid-sweep, particles and lines catch onto the captured band, then get
 * flung on release.
 *
 * Pipeline (grows by milestone; currently M1 skeleton):
 *   1. field_b (compute) — curl-noise velocity + ∇L' from field_a.
 *   2. p_update (compute) — substepped advection / age / respawn.
 *   3. prefill (compute) — tex_in × input_alpha → tex_out (or clear).
 *   4. raster  (instanced) — 6 verts × count quads, additive/alpha-over.
 */

#include <gpu.h>
#include <host.h>
#include "sweep_chamber_shaders.h"

#include <cmath>
#include <cstdint>
#include <cstring>

namespace sweep_chamber {

// 1M particles × 32 bytes = 32 MB. Pool pre-sized so `count` dials freely.
static constexpr int MAX_PARTICLES = 1000000;

// Quadratic size mapping (flow_swarm parity): schema `size` is a [0,1]
// slider; on-GPU size = SIZE_SCALE · slider².
static constexpr float SIZE_SCALE = 0.035f;
// Point shape draws a fixed quad this many pixels across.
static constexpr float POINT_PX = 1.5f;

// Coarse field resolution — an abstract square sim field, unrelated to the
// viewport (like the interaction density buffer).
static constexpr int FIELD_RES = 256;

// Interaction density buffer — an abstract square proximity field, resolution
// unrelated to the viewport (flow_swarm parity).
//
// It covers uv [-MARGIN, 1+MARGIN] rather than the bare viewport: particles
// live out to s-radius SWC_ESCAPE_R (well off-screen), and their crowding must
// still reach neighbours just inside the frame. Without the margin the halo of
// every off-frame particle is clipped away and the outermost `radius` of the
// picture under-reports density — a rim where avoidance, density-death and
// streaming all quietly weaken. The margin exceeds the largest
// interaction_radius (0.08), so the whole visible region has full kernel
// support. RES is scaled with the span to keep the on-screen texel pitch.
static constexpr int   DENSITY_RES    = 320;
static constexpr float DENSITY_MARGIN = 0.1f;
static constexpr float DENSITY_SPAN   = 1.0f + 2.0f * DENSITY_MARGIN;
// screen uv → density uv: a scale about the centre (offset = (1-scale)/2).
static constexpr float DENSITY_SCALE  = 1.0f / DENSITY_SPAN;

// Tracers ("lines"): fixed slot ranges in the segment buffer, no atomics.
static constexpr int MAX_TRACERS = 96;
static constexpr int MAX_SEG     = 96;   // segment slots per tracer
// l_width slider [0,1] → uv line width.
static constexpr float LINE_WIDTH_SCALE = 0.02f;
// motion_line_speed slider [0,1] → uv/frame tangent speed on the motion rail.
static constexpr float MOTION_LINE_SCALE = 0.05f;

// 2 vec4 = 32 bytes. Mirror of `Particle` in common.hlsl.
struct GpuParticle {
  float a[4];   // a.xy=pos, a.z=life_remain, a.w=life_total
  float b[4];   // b.xy=vel, b.z=size, b.w=asfloat(packed rgbz)
};
static_assert(sizeof(GpuParticle) == 32, "Particle GPU struct must be 32 bytes");

struct FieldAUniforms {
  uint32_t field_res;
  float    sweep_center;
  float    sweep_width;
  float    sweep_soft;
  float    blend_k;      // temporal EMA factor (1 = replace outright)
  float    _p0, _p1, _p2;
};
static_assert(sizeof(FieldAUniforms) == 32, "FieldAUniforms layout mismatch");

struct FieldUniforms {
  uint32_t field_res;
  float    aspect_x;
  float    aspect_y;
  float    noise_speed;

  float    noise_curl;
  float    eddy_scale;
  float    eddy_detail;
  float    spin_phase;

  float    drift_phase;
  float    drift_dir;
  float    image_smoothing;
  float    blend_k;      // field_c temporal EMA factor (matches field_a's)
};
static_assert(sizeof(FieldUniforms) == 48, "FieldUniforms layout mismatch");

struct UpdateUniforms {
  uint32_t count;
  uint32_t frame_index;
  float    dt;
  float    speed;

  float    momentum;
  float    jitter;
  float    drag;
  float    life;

  float    life_jitter;
  float    size;
  float    size_jitter;
  uint32_t seed;

  uint32_t mode;
  float    weight;
  float    pull;
  float    to_image;

  float    to_image_curl;
  float    undertow_skew;
  float    undertow_squash;
  float    aspect_x;

  float    aspect_y;
  uint32_t substeps;
  float    boundary;
  float    boundary_size;

  float    boundary_stiffness;
  float    boundary_death;
  float    spawn_size;
  float    to_line_rate;

  float    l_count_f;
  float    seg_stride;
  float    seg_live;
  float    calm_stretch;

  float    intense_shrink;
  float    respawn_rate;
  float    line_boost;
  float    jitter_boost;

  float    fling_boost;
  uint32_t interactions;
  float    density_threshold;
  float    density_death;

  float    avoid;
  float    avoid_curl;
  float    avoid_noise;
  float    density_res;

  float    stream;
  float    stream_density;
  float    field_res;
  float    dens_scale;   // screen uv → density uv (scale about the centre)
};
static_assert(sizeof(UpdateUniforms) == 192, "UpdateUniforms layout mismatch");

struct DensityUniforms { float res, dens_scale, dens_off, _pad; };
static_assert(sizeof(DensityUniforms) == 16, "DensityUniforms layout mismatch");

// One axis of the separable density halo (see density_blur.hlsl).
struct BlurUniforms {
  float res, dir_x, dir_y, inv_sigma;
  float taps, _p0, _p1, _p2;
};
static_assert(sizeof(BlurUniforms) == 32, "BlurUniforms layout mismatch");

struct FieldDebugUniforms { float to_image, to_image_curl, _p0, _p1; };
static_assert(sizeof(FieldDebugUniforms) == 16, "FieldDebugUniforms layout mismatch");

struct MotionVsUniforms {
  float aspect_x, aspect_y, point_size, dt;
  float scale, _m0, _m1, _m2;
};
static_assert(sizeof(MotionVsUniforms) == 32, "MotionVsUniforms layout mismatch");
struct MotionFsUniforms { uint32_t shape_kind; float shape_param, _f0, _f1; };
static_assert(sizeof(MotionFsUniforms) == 16, "MotionFsUniforms layout mismatch");
struct LineMotionVsUniforms { float aspect_x, aspect_y, width, line_speed; };
static_assert(sizeof(LineMotionVsUniforms) == 16, "LineMotionVsUniforms layout mismatch");

struct StatsUniforms {
  float field_res;
  float dt;
  float intensity_attack;
  float intensity_decay;

  float intensity_sens;
  float release_gain;
  float release_decay;
  float _pad0;
};
static_assert(sizeof(StatsUniforms) == 32, "StatsUniforms layout mismatch");

struct TraceUniforms {
  uint32_t count;
  uint32_t max_seg;
  uint32_t frame_index;
  float    dt;

  float    aspect_x;
  float    aspect_y;
  float    field_res;
  float    to_image;

  float    to_image_curl;
  float    step_len;
  float    length01;
  float    momentum;

  float    gradient_descent;
  float    snap;
  float    arc;
  float    adv;

  float    grip_attack;
  float    grip_decay;
  float    grip_alpha;
  float    fling_boost;

  float    time_decay;
  float    reseed_spread;
  float    color_contrib;
  float    l_opacity;

  float    tint_r;
  float    tint_g;
  float    tint_b;
  float    seed_rng;
};
static_assert(sizeof(TraceUniforms) == 112, "TraceUniforms layout mismatch");

struct LineVsUniforms { float aspect_x, aspect_y, width, _pad; };
static_assert(sizeof(LineVsUniforms) == 16, "LineVsUniforms layout mismatch");

struct LineFsUniforms { float soft, _a, _b, _c; };
static_assert(sizeof(LineFsUniforms) == 16, "LineFsUniforms layout mismatch");

struct PrefillUniforms { float scale_r, scale_g, scale_b, scale_a; };
static_assert(sizeof(PrefillUniforms) == 16, "PrefillUniforms layout mismatch");

struct VsUniforms {
  float aspect_x, aspect_y, point_size, shape_kind;
};
static_assert(sizeof(VsUniforms) == 16, "VsUniforms layout mismatch");

struct ColorUniforms {
  float    color_blend;
  float    solid_r;
  float    solid_g;
  float    solid_b;

  float    tint_by_flow;
  float    opacity;
  float    alpha_curve;
  float    shape_param;

  uint32_t shape_kind;
  float    exposure;
  float    _pad0;
  float    _pad1;
};
static_assert(sizeof(ColorUniforms) == 48, "ColorUniforms layout mismatch");

enum BlendMode : int { BLEND_ALPHA = 0, BLEND_ADD = 1 };
enum ShapeKind : int { SHAPE_POINT = 0, SHAPE_GAUSSIAN = 1, SHAPE_CIRCLE = 2, SHAPE_SOLID = 3 };
enum Mode      : int { MODE_VELOCITY = 0, MODE_FORCE = 1 };

// Type-shared: compiled once in module_init().
static gpu::ComputePSO s_pso_field_a;
static gpu::ComputePSO s_pso_field_b;
static gpu::ComputePSO s_pso_update;
static gpu::ComputePSO s_pso_trace;
static gpu::ComputePSO s_pso_stats;
static gpu::ComputePSO s_pso_prefill;
static gpu::RenderPSO  s_pso_render_alpha;
static gpu::RenderPSO  s_pso_render_add;
static gpu::RenderPSO  s_pso_line_alpha;
static gpu::RenderPSO  s_pso_line_add;
static gpu::RenderPSO  s_pso_density;        // unit-mass additive scatter
static gpu::ComputePSO s_pso_density_blur;   // separable halo convolution
static gpu::ComputePSO s_pso_density_debug;  // heat-map blit
static gpu::ComputePSO s_pso_field_debug;    // field readout blit
static gpu::ComputePSO s_pso_motion_prefill;
static gpu::RenderPSO  s_pso_motion_point;   // particle velocity → motion (RGBA16F)
static gpu::RenderPSO  s_pso_motion_line;    // line tangent → motion (RGBA16F)

struct State {
  gpu::Buffer  particle_buf;
  gpu::Buffer  tracer_buf;      // MAX_TRACERS × TracerState (32 B)
  gpu::Buffer  seg_buf;         // MAX_TRACERS × MAX_SEG × Seg (32 B)
  gpu::Buffer  stats_buf;       // 2 × float4: raw stats + calm↔intense response
  gpu::Buffer  stats_uniforms;
  gpu::Buffer  field_a_uniforms;
  gpu::Buffer  field_uniforms;
  gpu::Buffer  update_uniforms;
  gpu::Buffer  trace_uniforms;
  gpu::Buffer  prefill_uniforms;
  gpu::Buffer  vs_uniforms;
  gpu::Buffer  color_uniforms;
  gpu::Buffer  line_vs_uniforms;
  gpu::Buffer  line_fs_uniforms;
  gpu::Buffer  density_uniforms;
  gpu::Buffer  blur_h_uniforms;
  gpu::Buffer  blur_v_uniforms;
  gpu::Buffer  field_debug_uniforms;
  gpu::Buffer  motion_vs_uniforms;
  gpu::Buffer  motion_fs_uniforms;
  gpu::Buffer  lm_vs_uniforms;
  gpu::Sampler sampler;
  gpu::Texture field_a_tex;     // FIELD_RES² swept luma + peak offsets (lazy)
  gpu::Texture field_a_tex2;    // ping-pong partner (temporal EMA reads prev)
  gpu::Texture field_or_tex;    // band-side σ + plain luma (field_c's source)
  gpu::Texture field_c_tex;     // ∇luma + σ, EMA'd (curl direction) — ping-pong
  gpu::Texture field_c_tex2;
  bool         field_a_flip = false;   // flips field_a AND field_c pairs
  gpu::Texture field_b_tex;     // FIELD_RES² velocity field (lazy)
  gpu::Texture black_tex;       // 1×1 opaque black — generator fallback input
  gpu::Texture density_tex;     // blurred crowding buffer (1-frame delayed)
  gpu::Texture density_tex2;    // ping-pong partner (this frame's write target)
  gpu::Texture density_splat_tex; // raw unit-mass scatter (pre-blur scratch)
  gpu::Texture density_tmp_tex;   // horizontal-pass scratch
  bool         density_flip = false;
  gpu::Texture zero_density_tex; // 1×1 fallback when interactions are off
  gpu::Texture motion_tex;      // render_outputs/motion (RGBA16F, when a sink reads it)
  gpu::Texture zero_motion_tex; // 1×1 zero upstream-motion fallback
  int motion_w = 0, motion_h = 0;

  bool initialized = false;

  // CPU mirrors of schema params.
  int   count        = 150000;
  int   mode         = MODE_VELOCITY;
  float speed        = 1.0f;
  float momentum     = 0.6f;
  float weight       = 1.0f;
  int   substeps     = 1;
  float pull         = 0.0f;
  float jitter       = 0.05f;
  float drag         = 0.1f;
  float life         = 4.0f;
  float life_jitter  = 0.4f;
  float size         = 0.3f;
  float size_jitter  = 0.5f;
  // Sweep.
  float sweep_center    = 0.5f;
  float sweep_width     = 0.25f;
  float sweep_soft      = 0.3f;
  float sweep_smooth    = 0.12f;  // field temporal EMA time constant (s)
  float image_smoothing = 0.25f;
  // Field.
  float to_image        = 1.2f;
  float to_image_curl   = 1.0f;
  float undertow_skew   = 0.5f;
  float undertow_squash = 1.0f;
  float noise_speed     = 0.25f;
  float noise_curl      = 1.0f;
  float eddy_scale      = 0.4f;
  float eddy_detail     = 0.5f;
  float eddy_evolve     = 0.6f;
  float eddy_drift      = 0.05f;
  float eddy_drift_dir  = 0.0f;
  // Lines.
  int   l_count            = 24;
  float spawn_on_line      = 0.35f;
  float l_length           = 0.6f;
  float l_step             = 0.6f;   // ×0.01 → iso units at use site
  float l_momentum         = 0.5f;
  float l_gradient_descent = 0.0f;
  float l_snap             = 0.5f;
  float l_arc              = 0.4f;
  float l_adv              = 1.0f;
  float l_grip_attack      = 0.15f;
  float l_grip_decay       = 0.6f;
  float l_grip_alpha       = 0.6f;
  float l_fling_boost      = 3.0f;
  float l_time_decay       = 0.1f;
  float l_reseed_spread    = 0.4f;
  float l_color_contrib    = 0.5f;
  float l_width            = 0.2f;
  float l_soft             = 1.0f;
  float l_opacity          = 0.5f;
  // Interactions (1-frame-delayed density buffer).
  bool  interactions       = false;
  float interaction_radius = 0.015f;
  float density_threshold  = 4.0f;
  float density_death      = 0.0f;
  float avoid              = 0.0f;
  float avoid_curl         = 0.0f;
  float avoid_noise        = 0.08f;
  float stream             = 0.0f;
  float stream_density     = 3.0f;
  bool  debug_density      = false;
  bool  debug_field        = false;
  // Motion rail (render_outputs/motion) — only produced when a sink reads it.
  float motion_line_speed     = 0.3f;
  float motion_particle_scale = 1.0f;
  // Intensity & response (shaping only — the axis itself is derived).
  float intensity_sens   = 2.0f;
  float intensity_attack = 0.15f;
  float intensity_decay  = 1.2f;
  float calm_stretch     = 0.75f;
  float intense_shrink   = 0.25f;
  float respawn_rate     = 1.0f;
  float line_boost       = 2.5f;
  float jitter_boost     = 0.5f;
  float fling_boost      = 2.0f;
  float release_gain     = 2.0f;
  float release_decay    = 0.4f;
  // Containment.
  float boundary           = 0.4f;
  float boundary_size      = 0.62f;
  float boundary_stiffness = 4.0f;
  float boundary_death     = 0.25f;
  float spawn_size         = 0.6f;
  // Render.
  float color_blend  = 1.0f;   // 1 = captured input colour (dc parity)
  float solid_r      = 1.0f;
  float solid_g      = 1.0f;
  float solid_b      = 1.0f;
  float tint_by_flow = 0.0f;
  float opacity      = 0.25f;
  float alpha_curve  = 0.6f;
  float exposure     = 1.0f;
  int   shape_kind   = SHAPE_POINT;
  float shape_param  = 0.5f;
  int   blend_mode   = BLEND_ADD;
  float input_alpha  = 1.0f;
  int   seed         = 0;

  // Bookkeeping.
  int      inited_count = 0;
  uint32_t frame_index  = 0;
  uint32_t init_lcg     = 0x51EEB0CDu;
  // Noise phases, CPU-accumulated so rate-param changes stay smooth (and the
  // drift phase stays bounded — raw time × rate would degrade after hours).
  float spin_phase  = 0.0f;
  float drift_phase = 0.0f;
};

static inline uint32_t lcg_next(uint32_t& s) { s = s * 1664525u + 1013904223u; return s; }
static inline float    lcg_unit(uint32_t& s) { return (lcg_next(s) >> 8) * (1.0f / float(1u << 24)); }

// Pack white rgb + an 8-bit z phase into the particle's color slot (matches
// swc_pack_rgbz in common.hlsl). Only bit-reinterpreted, never float math.
static inline float pack_white_z(float z) {
  uint32_t d = (uint32_t)(z * 255.0f + 0.5f);
  if (d > 255u) d = 255u;
  uint32_t packed = 0xFFFFFFu | (d << 24);
  float f;
  std::memcpy(&f, &packed, sizeof(f));
  return f;
}

// Chunked seeding so a 1M pool doesn't blow the wasm stack (256 × 32 B = 8 KB).
static constexpr int INIT_CHUNK = 256;

static void seed_initial_slots(State& s, int from, int to) {
  if (!s.initialized || from >= to) return;
  GpuParticle entries[INIT_CHUNK];
  float size_uv = SIZE_SCALE * s.size * s.size;
  for (int chunk_start = from; chunk_start < to; chunk_start += INIT_CHUNK) {
    int chunk_end = chunk_start + INIT_CHUNK;
    if (chunk_end > to) chunk_end = to;
    int n = chunk_end - chunk_start;
    for (int i = 0; i < n; i++) {
      GpuParticle& p = entries[i];
      float ux = lcg_unit(s.init_lcg);
      float uy = lcg_unit(s.init_lcg);
      float life_remain = lcg_unit(s.init_lcg) * s.life;   // staggered start
      float z = lcg_unit(s.init_lcg);
      p.a[0] = ux; p.a[1] = uy; p.a[2] = life_remain; p.a[3] = s.life;
      p.b[0] = 0.0f; p.b[1] = 0.0f; p.b[2] = size_uv; p.b[3] = pack_white_z(z);
    }
    s.particle_buf.writeBytes(entries, int(sizeof(GpuParticle)) * n,
                              int(sizeof(GpuParticle)) * chunk_start);
  }
}

// Zero the tracer + segment buffers: a zeroed TracerState has time = 0 so
// every tracer self-seeds on its first trace; zeroed segs are degenerate.
static void zero_tracer_buffers(State& s) {
  float zeros[512] = {};   // 2 KB chunks
  int tracer_bytes = 32 * MAX_TRACERS;
  for (int off = 0; off < tracer_bytes; off += (int)sizeof(zeros)) {
    int n = tracer_bytes - off;
    if (n > (int)sizeof(zeros)) n = (int)sizeof(zeros);
    s.tracer_buf.writeBytes(zeros, n, off);
  }
  int seg_bytes = 32 * MAX_TRACERS * MAX_SEG;
  for (int off = 0; off < seg_bytes; off += (int)sizeof(zeros)) {
    int n = seg_bytes - off;
    if (n > (int)sizeof(zeros)) n = (int)sizeof(zeros);
    s.seg_buf.writeBytes(zeros, n, off);
  }
}

static void apply_count_change(State& s) {
  if (s.count > MAX_PARTICLES) s.count = MAX_PARTICLES;
  if (s.count < 1)             s.count = 1;
  if (s.count > s.inited_count) {
    seed_initial_slots(s, s.inited_count, s.count);
    s.inited_count = s.count;
  }
}

static void apply_mode_visibility(int mode, bool interactions) {
  state::setFieldHidden("momentum", mode != MODE_VELOCITY);
  state::setFieldHidden("weight",   mode != MODE_FORCE);
  bool ix = !interactions;
  state::setFieldHidden("interaction_radius", ix);
  state::setFieldHidden("density_threshold",  ix);
  state::setFieldHidden("density_death",      ix);
  state::setFieldHidden("avoid",              ix);
  state::setFieldHidden("avoid_curl",         ix);
  state::setFieldHidden("avoid_noise",        ix);
  state::setFieldHidden("stream",             ix);
  state::setFieldHidden("stream_density",     ix);
  state::setFieldHidden("debug_density",      ix);
}

// Static (self-less) visibility evaluator — pure over state.
void eval_visibility(int n, const char* pb, const int* off, const int* len, const int* ops) {
  int mode = MODE_VELOCITY; bool interactions = false;
  for (int i = 0; i < n; i++) {
    if (ops[i] != state::PatchReplace) continue;
    const char* p = pb + off[i]; int l = len[i];
    if      (state::pathIs(p, l, "mode"))         mode = (int)state::patchFloat(i);
    else if (state::pathIs(p, l, "interactions")) interactions = state::patchFloat(i) != 0.0f;
  }
  apply_mode_visibility(mode, interactions);
}

static void on_state_ready(void* self) {
  auto* s = static_cast<State*>(self);
  if (s) apply_mode_visibility(s->mode, s->interactions);
}

void module_init() {
  state::init("source.particles.sweep_chamber", {0, 1, 0},
    state::Schema()
      .helpField("intro",
        "## Sweep Chamber\n"
        "A particle + line sim with a **built-in luma sweep**. The *Sweep* "
        "window captures one band of the input's brightness at a time: "
        "particles catch onto the captured detail, bunch up along its ridges, "
        "then get **flung** when the sweep releases them. At either end of "
        "the sweep nothing is captured and everything free-flows on a smooth "
        "curl-noise eddy field.\n\n"
        "**Try:** wire a video in, slowly sweep *Center* from 0 to 1, and "
        "watch the swarm catch and release each brightness layer.")
      // ---- Sweep ----
      .group("sweep", "Sweep")
        .groupHelp(
          "The band-pass window over the input's luma. *Center* travels the "
          "window across the brightness range — 0 and 1 are always fully OFF "
          "either end (the image reads as black and the sim free-flows). "
          "*Width*/*Softness* shape the captured band; *Smoothing* widens the "
          "gradient read for broader, calmer attraction.")
      .floatField("sweep_center",    0.5f,  0.0f, 1.0f, state::PrimaryInput).label("Sweep Center", "Sweep")
      .floatField("sweep_width",     0.25f, 0.02f, 1.0f, state::PrimaryInput).label("Sweep Width", "Width")
      .floatField("sweep_soft",      0.3f,  0.0f, 1.0f, state::PrimaryInput).label("Sweep Softness", "Soft")
      .floatField("sweep_smooth",    0.12f, 0.0f, 0.5f, state::PrimaryInput).label("Temporal Smooth", "TSmth")
      .floatField("image_smoothing", 0.25f, 0.0f, 1.0f, state::PrimaryInput).label("Image Smoothing", "Smooth")
      // Debug: render the coarse field itself (velocity hue + swept luma +
      // ridge detector) instead of the sim — the tuning window.
      .boolField ("debug_field",     false,             state::PrimaryInput).label("Debug Field", "DbgFld")
      // ---- Field ----
      .group("field", "Field")
        .groupHelp(
          "The forces. *To Image* pulls along the swept-luma gradient (toward "
          "the captured band's edges); *Image Curl* pushes perpendicular, "
          "scaled per particle by its hidden z phase (*Skew*/*Squash* shape "
          "that variation). The background is a smooth curl-noise eddy field: "
          "*Curl* blends gradient-flow → divergence-free eddies, *Eddy Scale/"
          "Detail* set their size and roughness, *Evolve* churns them and "
          "*Drift* advects them across the frame.")
      .floatField("to_image",        1.2f, -4.0f, 4.0f, state::PrimaryInput).label("To Image", "ToImg")
      .floatField("to_image_curl",   1.0f, -4.0f, 4.0f, state::PrimaryInput).label("Image Curl", "ImgCurl")
      .floatField("undertow_skew",   0.5f,  0.0f, 1.0f, state::PrimaryInput).label("Undertow Skew", "USkew")
      .floatField("undertow_squash", 1.0f, -2.0f, 2.0f, state::PrimaryInput).label("Undertow Squash", "USquash")
      .floatField("noise_speed",     0.25f, 0.0f, 2.0f, state::PrimaryInput).label("Noise Speed", "NSpd")
      .floatField("noise_curl",      1.0f, -1.0f, 1.0f, state::PrimaryInput).label("Noise Curl", "NCurl")
      .floatField("eddy_scale",      0.4f,  0.0f, 1.0f, state::PrimaryInput).label("Eddy Scale", "EddySc")
      .floatField("eddy_detail",     0.5f,  0.0f, 1.0f, state::PrimaryInput).label("Eddy Detail", "EddyDt")
      .floatField("eddy_evolve",     0.6f,  0.0f, 4.0f, state::PrimaryInput).label("Eddy Evolve", "Evolve")
      .floatField("eddy_drift",      0.05f, 0.0f, 0.5f, state::PrimaryInput).label("Eddy Drift", "Drift")
      .floatField("eddy_drift_dir",  0.0f,  0.0f, 1.0f, state::PrimaryInput).label("Drift Direction", "DriftDir")
      // ---- Lines (tracers) ----
      .group("lines", "Lines")
        .groupHelp(
          "Streamline tracers that get **trapped along the ridges** of the "
          "captured band. Lines never die in black: in free space they keep "
          "propagating on a per-line ballistic **Arc**, and instead carry a "
          "**grip** weight (how hard the image holds them) that scales how "
          "strongly particles spawn onto them (*Spawn On Line*) — as the "
          "sweep releases, lines stop attracting and get **flung**, arcing "
          "away. *Descent* 0 follows level curves (trapped), 1 descends the "
          "gradient; *Ridge Snap* locks lines onto sub-cell ridge peaks.")
      .intField  ("l_count",            24,     0,     MAX_TRACERS, state::PrimaryInput).label("Line Count", "Lines")
      .floatField("spawn_on_line",      0.35f,  0.0f,  1.0f,  state::PrimaryInput).label("Spawn On Line", "OnLine")
      .floatField("l_length",           0.6f,   0.0f,  1.0f,  state::PrimaryInput).label("Line Length", "LLen")
      .floatField("l_step",             0.6f,   0.1f,   2.0f,  state::PrimaryInput).label("Line Step", "LStep")
      .floatField("l_momentum",         0.5f,   0.0f,  0.95f, state::PrimaryInput).label("Line Momentum", "LMom")
      .floatField("l_gradient_descent", 0.0f,   0.0f,  1.0f,  state::PrimaryInput).label("Gradient Descent", "Descent")
      .floatField("l_snap",             0.5f,   0.0f,  1.0f,  state::PrimaryInput).label("Ridge Snap", "Snap")
      .floatField("l_arc",              0.4f,   0.0f,  1.0f,  state::PrimaryInput).label("Line Arc", "Arc")
      .floatField("l_adv",              1.0f,   0.0f,  4.0f,  state::PrimaryInput).label("Seed Advection", "LAdv")
      .floatField("l_grip_attack",      0.15f,  0.02f, 2.0f,  state::PrimaryInput).label("Grip Attack", "GripAtk")
      .floatField("l_grip_decay",       0.6f,   0.05f, 5.0f,  state::PrimaryInput).label("Grip Decay", "GripDec")
      .floatField("l_grip_alpha",       0.6f,   0.0f,  1.0f,  state::PrimaryInput).label("Grip Alpha", "GripA")
      .floatField("l_fling_boost",      3.0f,   0.0f,  8.0f,  state::PrimaryInput).label("Line Fling", "LFling")
      .floatField("l_time_decay",       0.1f,   0.0f,  2.0f,  state::PrimaryInput).label("Line Life Decay", "LDecay")
      .floatField("l_reseed_spread",    0.4f,   0.0f,  1.0f,  state::PrimaryInput).label("Reseed Spread", "Reseed")
      .floatField("l_color_contrib",    0.5f,   0.0f,  1.0f,  state::PrimaryInput).label("Line Colour", "LCol")
      .floatField("l_width",            0.2f,   0.0f,  1.0f,  state::PrimaryInput).label("Line Width", "LWidth")
      .floatField("l_soft",             1.0f,   0.1f,  4.0f,  state::PrimaryInput).label("Line Softness", "LSoft")
      .floatField("l_opacity",          0.5f,   0.0f,  1.0f,  state::PrimaryInput).label("Line Opacity", "LOpac")
      // ---- Intensity & response ----
      .group("response", "Intensity & Response")
        .groupHelp(
          "The calm↔intense axis is **derived from the swept image itself** — "
          "how much captured energy the sweep window is currently holding. "
          "Calm (nothing captured): long lingering lifetimes, gentle motion. "
          "Intense (a band captured): shorter bursty lifetimes, forced "
          "respawn churn onto the lines, boosted spawn-on-line and jitter. "
          "When the sweep RELEASES a band, a release envelope fires and "
          "**flings** everything along its current motion. These knobs shape "
          "the response; none of them IS the axis.")
      .floatField("intensity_sens",   2.0f,  0.0f,  8.0f, state::PrimaryInput).label("Intensity Sensitivity", "ISens")
      .floatField("intensity_attack", 0.15f, 0.02f, 2.0f, state::PrimaryInput).label("Intensity Attack", "IAtk")
      .floatField("intensity_decay",  1.2f,  0.1f,  8.0f, state::PrimaryInput).label("Intensity Decay", "IDec")
      .floatField("calm_stretch",     0.75f, 0.0f,  3.0f, state::PrimaryInput).label("Calm Life Stretch", "Calm")
      .floatField("intense_shrink",   0.25f, 0.05f, 1.0f, state::PrimaryInput).label("Intense Life Shrink", "Intense")
      .floatField("respawn_rate",     1.0f,  0.0f,  4.0f, state::PrimaryInput).label("Forced Respawn", "Respawn")
      .floatField("line_boost",       2.5f,  1.0f,  6.0f, state::PrimaryInput).label("Line Spawn Boost", "LBoost")
      .floatField("jitter_boost",     0.5f,  0.0f,  3.0f, state::PrimaryInput).label("Jitter Boost", "JBoost")
      .floatField("fling_boost",      2.0f,  0.0f,  8.0f, state::PrimaryInput).label("Release Fling", "Fling")
      .floatField("release_gain",     2.0f,  0.0f,  8.0f, state::PrimaryInput).label("Release Gain", "RGain")
      .floatField("release_decay",    0.4f,  0.05f, 2.0f, state::PrimaryInput).label("Release Hold", "RHold")
      // ---- Interactions (flow_swarm parity) ----
      .group("interactions", "Interactions")
        .groupHelp(
          "Lets particles feel their neighbours through a 1-frame crowding "
          "buffer (off by default — it adds a splat pass). *Radius* is the "
          "sensing range. **Density Death** thins over-packed areas; "
          "**Avoidance** pushes down the crowd gradient (*Curl* swirls it, "
          "*Noise* unsticks symmetric clumps); **Streaming** aligns (+) or "
          "scatters (−) each particle with its local group. **Debug Density** "
          "shows the buffer as a heat map.")
      .boolField ("interactions",       false,                 state::PrimaryInput).label("Interactions", "Inter")
      .floatField("interaction_radius", 0.015f, 0.002f, 0.08f, state::PrimaryInput).label("Interaction Radius", "Radius")
      .floatField("density_threshold",  4.0f,   0.0f,  32.0f,  state::PrimaryInput).label("Density Threshold", "Thresh")
      .floatField("density_death",      0.0f,   0.0f,  1.0f,   state::PrimaryInput).label("Density Death", "Death")
      .floatField("avoid",              0.0f,   0.0f,  1.0f,   state::PrimaryInput).label("Avoidance", "Avoid")
      .floatField("avoid_curl",         0.0f,  -1.0f,  1.0f,   state::PrimaryInput).label("Avoid Curl", "ACurl")
      .floatField("avoid_noise",        0.08f,  0.0f,  1.0f,   state::PrimaryInput).label("Avoid Noise", "ANoise")
      .floatField("stream",             0.0f,  -1.0f,  1.0f,   state::PrimaryInput).label("Streaming", "Stream")
      .floatField("stream_density",     3.0f,   0.5f,  32.0f,  state::PrimaryInput).label("Stream Density", "StrDen")
      .boolField ("debug_density",      false,                 state::PrimaryInput).label("Debug Density", "Debug")
      // ---- Motion rail output ----
      .group("motion", "Motion Output")
        .groupHelp(
          "Motion vectors for downstream motion-blur-style effects, produced "
          "only when something reads the `render_outputs` rail. Particles "
          "emit their integrated per-frame velocity (× *Particle Scale*); "
          "lines emit a tangent-along-segment speed (*Line Speed*).")
      .floatField("motion_line_speed",     0.3f, 0.0f, 1.0f, state::PrimaryInput).label("Motion Line Speed", "MLine")
      .floatField("motion_particle_scale", 1.0f, 0.0f, 4.0f, state::PrimaryInput).label("Motion Particle Scale", "MPart")
      // ---- Pool / advection ----
      .group("advection", "Pool & Advection")
      .intField  ("count",     150000, 1, MAX_PARTICLES, state::PrimaryInput).label("Count", "Count")
      .selectField("mode",     MODE_VELOCITY, state::PrimaryInput, {
        {"Velocity", MODE_VELOCITY},
        {"Force",    MODE_FORCE},
      }).label("Mode", "Mode")
      .floatField("speed",     1.0f,  0.0f,  8.0f,  state::PrimaryInput).label("Speed", "Spd")
      .floatField("momentum",  0.6f,  0.0f,  0.99f, state::PrimaryInput).label("Momentum", "Mom")
      .floatField("weight",    1.0f,  0.05f, 8.0f,  state::PrimaryInput).label("Weight", "Wt")
      .intField  ("substeps",  1,     1,     16,    state::PrimaryInput).label("Substeps", "Sub")
      .floatField("pull",      0.0f,  0.0f,  1.0f,  state::PrimaryInput).label("Settle", "Pull")
      .floatField("jitter",    0.05f, 0.0f,  1.0f,  state::PrimaryInput).label("Jitter", "Jit")
      .floatField("drag",      0.1f,  0.0f,  4.0f,  state::PrimaryInput).label("Drag", "Drag")
      // ---- Geometry / lifetime / containment ----
      .group("geometry", "Geometry & Lifetime")
      .floatField("size",         0.3f, 0.0f, 1.0f,  state::PrimaryInput).label("Size", "Size")
      .floatField("size_jitter",  0.5f, 0.0f, 1.0f,  state::PrimaryInput).label("Size Jitter", "SzJit")
      .floatField("life",         4.0f, 0.1f, 30.0f, state::PrimaryInput).label("Lifetime", "Life")
      .floatField("life_jitter",  0.4f, 0.0f, 1.0f,  state::PrimaryInput).label("Life Jitter", "LfJit")
      .floatField("spawn_size",   0.6f, 0.0f, 1.2f,  state::PrimaryInput).label("Spawn Size", "Spawn")
      .floatField("boundary",           0.4f,  0.0f, 1.0f,  state::PrimaryInput).label("Boundary", "Bound")
      .floatField("boundary_size",      0.62f, 0.1f, 1.2f,  state::PrimaryInput).label("Boundary Size", "BSize")
      .floatField("boundary_stiffness", 4.0f,  0.5f, 16.0f, state::PrimaryInput).label("Boundary Stiffness", "BStiff")
      .floatField("boundary_death",     0.25f, 0.0f, 1.0f,  state::PrimaryInput).label("Boundary Death", "BDeath")
      // ---- Color ----
      .group("color", "Colour")
      .floatField("color_blend",  1.0f, 0.0f, 1.0f, state::PrimaryInput).label("Image Colour", "ImCol")
      .rgbField  ("solid_color",  1.0f, 1.0f, 1.0f, state::PrimaryInput).label("Solid Colour", "Colour")
      .floatField("tint_by_flow", 0.0f, 0.0f, 1.0f, state::PrimaryInput).label("Tint By Flow", "Tint")
      // ---- Composite ----
      .group("composite", "Composite")
      .selectField("blend_mode",  BLEND_ADD, state::PrimaryInput, {
        {"Add",   BLEND_ADD},
        {"Alpha", BLEND_ALPHA},
      }).label("Blend Mode", "Blend")
      .floatField("opacity",      0.25f, 0.0f, 1.0f, state::PrimaryInput).label("Opacity", "Opac")
      .floatField("input_alpha",  1.0f,  0.0f, 1.0f, state::PrimaryInput).label("Input Alpha", "InAlph")
      // ---- Tuning / shape ----
      .group("shape", "Shape & Tuning")
      .selectField("shape_kind",  SHAPE_POINT, state::PrimaryInput, {
        {"Point",    SHAPE_POINT},
        {"Gaussian", SHAPE_GAUSSIAN},
        {"Circle",   SHAPE_CIRCLE},
        {"Solid",    SHAPE_SOLID},
      }).label("Shape", "Shape")
      .floatField("shape_param",  0.5f, 0.0f,  1.0f, state::PrimaryInput).label("Shape Param", "Param")
      .floatField("alpha_curve",  0.6f, 0.25f, 4.0f, state::PrimaryInput).label("Alpha Curve", "Curve")
      .floatField("exposure",     1.0f, 0.0f,  8.0f, state::PrimaryInput).label("Exposure", "Exp")
      .intField  ("seed",         0,    0,     65535, state::PrimaryInput).label("Seed", "Seed")
      // ---- I/O ----
      .textureField("tex_in",  state::PrimaryInput)
      .textureField("tex_out", state::PrimaryOutput)
      // Canonical render_outputs rail (motion leaf) — produced only when a
      // downstream sink reads it; render_outputs_in composes upstream motion.
      .renderOutputs(state::PrimaryOutput)
      .renderOutputs(state::PrimaryInput, "render_outputs_in")
        .capability(state::Capability::Generator)
    );

  if (gpu::Device::backend() == gpu::Backend::None) return;

  state::registerShaderSPV("sweep_chamber_field_a",  FIELD_A_SPV,  FIELD_A_SPV_SIZE,
                           "rgba16float", "write");
  state::registerShaderSPV("sweep_chamber_field_b",  FIELD_B_SPV,  FIELD_B_SPV_SIZE,
                           "rgba16float", "write");
  state::registerShaderSPV("sweep_chamber_p_update", P_UPDATE_SPV, P_UPDATE_SPV_SIZE);
  state::registerShaderSPV("sweep_chamber_trace",    TRACE_SPV,    TRACE_SPV_SIZE);
  state::registerShaderSPV("sweep_chamber_stats",    STATS_SPV,    STATS_SPV_SIZE);
  state::registerShaderSPV("sweep_chamber_prefill",  PREFILL_SPV,  PREFILL_SPV_SIZE);
  state::registerShaderSPV("sweep_chamber_vs",       VS_SPV,       VS_SPV_SIZE);
  state::registerShaderSPV("sweep_chamber_fs",       FS_SPV,       FS_SPV_SIZE);
  state::registerShaderSPV("sweep_chamber_line_vs",  LINE_VS_SPV,  LINE_VS_SPV_SIZE);
  state::registerShaderSPV("sweep_chamber_line_fs",  LINE_FS_SPV,  LINE_FS_SPV_SIZE);
  state::registerShaderSPV("sweep_chamber_density_vs", DENSITY_VS_SPV, DENSITY_VS_SPV_SIZE);
  state::registerShaderSPV("sweep_chamber_density_fs", DENSITY_FS_SPV, DENSITY_FS_SPV_SIZE);
  state::registerShaderSPV("sweep_chamber_density_blur", DENSITY_BLUR_SPV, DENSITY_BLUR_SPV_SIZE,
                           "rgba16float", "write");
  state::registerShaderSPV("sweep_chamber_density_debug", DENSITY_DEBUG_SPV, DENSITY_DEBUG_SPV_SIZE);
  state::registerShaderSPV("sweep_chamber_field_debug", FIELD_DEBUG_SPV, FIELD_DEBUG_SPV_SIZE);
  state::registerShaderSPV("sweep_chamber_motion_prefill", MOTION_PREFILL_SPV, MOTION_PREFILL_SPV_SIZE,
                           "rgba16float", "write");
  state::registerShaderSPV("sweep_chamber_motion_vs",      MOTION_VS_SPV,      MOTION_VS_SPV_SIZE);
  state::registerShaderSPV("sweep_chamber_motion_fs",      MOTION_FS_SPV,      MOTION_FS_SPV_SIZE);
  state::registerShaderSPV("sweep_chamber_line_motion_vs", LINE_MOTION_VS_SPV, LINE_MOTION_VS_SPV_SIZE);
  state::registerShaderSPV("sweep_chamber_line_motion_fs", LINE_MOTION_FS_SPV, LINE_MOTION_FS_SPV_SIZE);

  auto cs_field_a = gpu::Device::createShaderModuleByName("sweep_chamber_field_a");
  auto cs_field_b = gpu::Device::createShaderModuleByName("sweep_chamber_field_b");
  auto cs_update  = gpu::Device::createShaderModuleByName("sweep_chamber_p_update");
  auto cs_trace   = gpu::Device::createShaderModuleByName("sweep_chamber_trace");
  auto cs_stats   = gpu::Device::createShaderModuleByName("sweep_chamber_stats");
  auto cs_prefill = gpu::Device::createShaderModuleByName("sweep_chamber_prefill");
  auto vs_module  = gpu::Device::createShaderModuleByName("sweep_chamber_vs");
  auto fs_module  = gpu::Device::createShaderModuleByName("sweep_chamber_fs");
  auto line_vs    = gpu::Device::createShaderModuleByName("sweep_chamber_line_vs");
  auto line_fs    = gpu::Device::createShaderModuleByName("sweep_chamber_line_fs");
  auto vs_density = gpu::Device::createShaderModuleByName("sweep_chamber_density_vs");
  auto fs_density = gpu::Device::createShaderModuleByName("sweep_chamber_density_fs");
  auto cs_dblur   = gpu::Device::createShaderModuleByName("sweep_chamber_density_blur");
  auto cs_dbg     = gpu::Device::createShaderModuleByName("sweep_chamber_density_debug");
  auto cs_fdbg    = gpu::Device::createShaderModuleByName("sweep_chamber_field_debug");
  auto cs_mpf     = gpu::Device::createShaderModuleByName("sweep_chamber_motion_prefill");
  auto mvs        = gpu::Device::createShaderModuleByName("sweep_chamber_motion_vs");
  auto mfs        = gpu::Device::createShaderModuleByName("sweep_chamber_motion_fs");
  auto lmvs       = gpu::Device::createShaderModuleByName("sweep_chamber_line_motion_vs");
  auto lmfs       = gpu::Device::createShaderModuleByName("sweep_chamber_line_motion_fs");
  if (!cs_field_a || !cs_field_b || !cs_update || !cs_trace || !cs_stats ||
      !cs_prefill || !vs_module || !fs_module || !line_vs || !line_fs ||
      !vs_density || !fs_density || !cs_dblur || !cs_dbg || !cs_fdbg ||
      !cs_mpf || !mvs || !mfs || !lmvs || !lmfs) return;

  s_pso_field_a = gpu::Device::createComputePSO(cs_field_a, "main", gpu::Bindings()
      .storageTex2d(0, gpu::TextureFormat::RGBA16F)   // field_a out (EMA-blended)
      .tex2d(1)                                       // full-res input
      .sampler(2)
      .uniform(3)
      .tex2d(4)                                       // last frame's field_a
      .storageTex2d(5, gpu::TextureFormat::RGBA16F)); // field_or (band-side σ)

  s_pso_field_b = gpu::Device::createComputePSO(cs_field_b, "main", gpu::Bindings()
      .storageTex2d(0, gpu::TextureFormat::RGBA16F)   // field_b out
      .tex2d(1)                                       // field_a (swept luma)
      .sampler(2)
      .uniform(3)
      .tex2d(4)                                       // field_or (σ + plain luma)
      .storageTex2d(5, gpu::TextureFormat::RGBA16F)   // field_c out (∇luma, EMA)
      .tex2d(6));                                     // last frame's field_c

  s_pso_update = gpu::Device::createComputePSO(cs_update, "main", gpu::Bindings()
      .storageRW(0)   // particles[]
      .tex2d(1)       // field_b
      .tex2d(2)       // input (color capture)
      .sampler(3)
      .uniform(4)
      .tex2d(5)       // density (last frame's crowding)
      .storage(6)     // tracer segments (spawn-on-line)
      .storage(7)     // tracer states (grip weighting)
      .storage(8)     // stats/response (calm↔intense)
      .tex2d(9)       // field_a (ridge presence — undertow gate)
      .tex2d(10));    // field_or (band-side σ — curl orientation)

  s_pso_trace = gpu::Device::createComputePSO(cs_trace, "main", gpu::Bindings()
      .storageRW(0)   // tracers[]
      .storageRW(1)   // segs[]
      .tex2d(2)       // field_a
      .tex2d(3)       // field_b
      .sampler(4)
      .uniform(5)
      .tex2d(6)       // input (line color)
      .tex2d(7));     // field_or (band-side σ)

  s_pso_stats = gpu::Device::createComputePSO(cs_stats, "main", gpu::Bindings()
      .tex2d(0)       // field_a
      .tex2d(1)       // field_b
      .storageRW(2)   // stats/response buffer
      .uniform(3));

  s_pso_prefill = gpu::Device::createComputePSO(cs_prefill, "main", gpu::Bindings()
      .tex2d(0)
      .storageTex2d(1)
      .uniform(2));

  s_pso_render_alpha = gpu::Device::createInstancedRenderPSO(
      vs_module, "main", fs_module, "main", gpu::TextureFormat::Surface,
      gpu::Bindings().storage(0).uniform(1).uniform(2),
      gpu::Device::BlendMode::AlphaOver);
  s_pso_render_add = gpu::Device::createInstancedRenderPSO(
      vs_module, "main", fs_module, "main", gpu::TextureFormat::Surface,
      gpu::Bindings().storage(0).uniform(1).uniform(2),
      gpu::Device::BlendMode::Additive);

  s_pso_line_alpha = gpu::Device::createInstancedRenderPSO(
      line_vs, "main", line_fs, "main", gpu::TextureFormat::Surface,
      gpu::Bindings().storage(0).uniform(1).uniform(2),
      gpu::Device::BlendMode::AlphaOver);
  s_pso_line_add = gpu::Device::createInstancedRenderPSO(
      line_vs, "main", line_fs, "main", gpu::TextureFormat::Surface,
      gpu::Bindings().storage(0).uniform(1).uniform(2),
      gpu::Device::BlendMode::Additive);

  // Density scatter: one unit of mass per particle (2×2 texels), summed into
  // the RGBA16F buffer; the halo itself is the separable blur below.
  s_pso_density = gpu::Device::createInstancedRenderPSO(
      vs_density, "main", fs_density, "main", gpu::TextureFormat::RGBA16F,
      gpu::Bindings().storage(0).uniform(1),
      gpu::Device::BlendMode::Additive);

  s_pso_density_blur = gpu::Device::createComputePSO(cs_dblur, "main", gpu::Bindings()
      .tex2d(0)                                       // source (one axis in)
      .storageTex2d(1, gpu::TextureFormat::RGBA16F)   // destination
      .uniform(2));

  s_pso_density_debug = gpu::Device::createComputePSO(cs_dbg, "main", gpu::Bindings()
      .tex2d(0)
      .sampler(1)
      .storageTex2d(2)
      .uniform(3));

  s_pso_field_debug = gpu::Device::createComputePSO(cs_fdbg, "main", gpu::Bindings()
      .tex2d(0)       // field_a
      .tex2d(1)       // field_b
      .sampler(2)
      .storageTex2d(3)
      .uniform(4)
      .tex2d(5));     // field_or (band-side σ)

  // Motion-vector PSOs: RGBA16F targets, AlphaOver composes each footprint's
  // velocity over the (upstream-seeded) motion field by coverage.
  s_pso_motion_prefill = gpu::Device::createComputePSO(cs_mpf, "main", gpu::Bindings()
      .tex2d(0).storageTex2d(1, gpu::TextureFormat::RGBA16F));
  s_pso_motion_point = gpu::Device::createInstancedRenderPSO(mvs, "main", mfs, "main",
      gpu::TextureFormat::RGBA16F,
      gpu::Bindings().storage(0).uniform(1).uniform(2),
      gpu::Device::BlendMode::AlphaOver);
  s_pso_motion_line = gpu::Device::createInstancedRenderPSO(lmvs, "main", lmfs, "main",
      gpu::TextureFormat::RGBA16F,
      gpu::Bindings().storage(0).uniform(1).uniform(2),
      gpu::Device::BlendMode::AlphaOver);

  state::log("sweep_chamber: module initialized");
}

void* create() {
  auto* s = new State();
  s->particle_buf = gpu::Device::createBuffer(
      sizeof(GpuParticle) * MAX_PARTICLES, gpu::BufferUsage::Storage);
  s->tracer_buf = gpu::Device::createBuffer(
      32 * MAX_TRACERS, gpu::BufferUsage::Storage);
  s->seg_buf = gpu::Device::createBuffer(
      32 * MAX_TRACERS * MAX_SEG, gpu::BufferUsage::Storage);
  s->stats_buf = gpu::Device::createBuffer(32, gpu::BufferUsage::Storage);
  s->stats_uniforms = gpu::Device::createBuffer(sizeof(StatsUniforms), gpu::BufferUsage::Uniform);
  s->field_a_uniforms = gpu::Device::createBuffer(sizeof(FieldAUniforms),  gpu::BufferUsage::Uniform);
  s->field_uniforms   = gpu::Device::createBuffer(sizeof(FieldUniforms),   gpu::BufferUsage::Uniform);
  s->update_uniforms  = gpu::Device::createBuffer(sizeof(UpdateUniforms),  gpu::BufferUsage::Uniform);
  s->trace_uniforms   = gpu::Device::createBuffer(sizeof(TraceUniforms),   gpu::BufferUsage::Uniform);
  s->prefill_uniforms = gpu::Device::createBuffer(sizeof(PrefillUniforms), gpu::BufferUsage::Uniform);
  s->vs_uniforms      = gpu::Device::createBuffer(sizeof(VsUniforms),      gpu::BufferUsage::Uniform);
  s->color_uniforms   = gpu::Device::createBuffer(sizeof(ColorUniforms),   gpu::BufferUsage::Uniform);
  s->line_vs_uniforms = gpu::Device::createBuffer(sizeof(LineVsUniforms),  gpu::BufferUsage::Uniform);
  s->line_fs_uniforms = gpu::Device::createBuffer(sizeof(LineFsUniforms),  gpu::BufferUsage::Uniform);
  s->density_uniforms = gpu::Device::createBuffer(sizeof(DensityUniforms), gpu::BufferUsage::Uniform);
  s->blur_h_uniforms  = gpu::Device::createBuffer(sizeof(BlurUniforms),    gpu::BufferUsage::Uniform);
  s->blur_v_uniforms  = gpu::Device::createBuffer(sizeof(BlurUniforms),    gpu::BufferUsage::Uniform);
  s->field_debug_uniforms = gpu::Device::createBuffer(sizeof(FieldDebugUniforms), gpu::BufferUsage::Uniform);
  s->motion_vs_uniforms = gpu::Device::createBuffer(sizeof(MotionVsUniforms), gpu::BufferUsage::Uniform);
  s->motion_fs_uniforms = gpu::Device::createBuffer(sizeof(MotionFsUniforms), gpu::BufferUsage::Uniform);
  s->lm_vs_uniforms     = gpu::Device::createBuffer(sizeof(LineMotionVsUniforms), gpu::BufferUsage::Uniform);
  s->sampler = gpu::Device::createSampler(gpu::FilterMode::Linear, gpu::AddressMode::ClampToEdge);
  return s;
}

void destroy(void* self) {
  auto* s = static_cast<State*>(self);
  if (!s) return;
  s->particle_buf.release();
  s->tracer_buf.release();
  s->seg_buf.release();
  s->stats_buf.release();
  s->stats_uniforms.release();
  s->field_a_uniforms.release();
  s->field_uniforms.release();
  s->update_uniforms.release();
  s->trace_uniforms.release();
  s->prefill_uniforms.release();
  s->vs_uniforms.release();
  s->color_uniforms.release();
  s->line_vs_uniforms.release();
  s->line_fs_uniforms.release();
  s->density_uniforms.release();
  s->blur_h_uniforms.release();
  s->blur_v_uniforms.release();
  s->field_debug_uniforms.release();
  s->motion_vs_uniforms.release();
  s->motion_fs_uniforms.release();
  s->lm_vs_uniforms.release();
  s->sampler.release();
  s->field_a_tex.release();
  s->field_a_tex2.release();
  s->field_or_tex.release();
  s->field_c_tex.release();
  s->field_c_tex2.release();
  s->field_b_tex.release();
  s->black_tex.release();
  s->density_tex.release();
  s->density_tex2.release();
  s->density_splat_tex.release();
  s->density_tmp_tex.release();
  s->zero_density_tex.release();
  s->motion_tex.release();
  s->zero_motion_tex.release();
  delete s;
}

void init(void* self) {
  auto* s = static_cast<State*>(self);
  if (!s) return;
  if (!s_pso_field_a.valid() || !s_pso_field_b.valid() ||
      !s_pso_update.valid() || !s_pso_trace.valid() || !s_pso_stats.valid() ||
      !s_pso_prefill.valid() ||
      !s_pso_render_alpha.valid() || !s_pso_render_add.valid() ||
      !s_pso_line_alpha.valid() || !s_pso_line_add.valid() ||
      !s_pso_density.valid() || !s_pso_density_blur.valid() ||
      !s_pso_density_debug.valid() ||
      !s_pso_field_debug.valid() ||
      !s_pso_motion_prefill.valid() || !s_pso_motion_point.valid() ||
      !s_pso_motion_line.valid()) return;
  if (!s->particle_buf.valid() || !s->tracer_buf.valid() || !s->seg_buf.valid()) return;

  s->inited_count = 0;
  s->frame_index  = 0;
  s->init_lcg     = 0x51EEB0CDu;
  s->initialized  = true;
  apply_count_change(*s);   // seed the initial pool
  zero_tracer_buffers(*s);  // tracers self-seed on first trace
  float stats_zero[8] = {};
  s->stats_buf.writeBytes(stats_zero, sizeof(stats_zero), 0);
  state::setOnStateReady(&on_state_ready);
}

void tick(void* self, double dt) { (void)self; (void)dt; }   // timing is GPU-side via dt uniform

void on_state_patched(void* self, int n, const char* pb, const int* off,
                      const int* len, const int* ops) {
  auto* s = static_cast<State*>(self);
  if (!s) return;
  for (int i = 0; i < n; i++) {
    if (ops[i] != state::PatchReplace) continue;
    const char* path = pb + off[i];
    int plen = len[i];
    if      (state::pathIs(path, plen, "count"))        { s->count = (int)state::patchFloat(i); apply_count_change(*s); }
    else if (state::pathIs(path, plen, "mode"))         { int v = (int)state::patchFloat(i); if (v != s->mode) { s->mode = v; apply_mode_visibility(s->mode, s->interactions); } }
    else if (state::pathIs(path, plen, "speed"))        s->speed = state::patchFloat(i);
    else if (state::pathIs(path, plen, "momentum"))     s->momentum = state::patchFloat(i);
    else if (state::pathIs(path, plen, "weight"))       s->weight = state::patchFloat(i);
    else if (state::pathIs(path, plen, "substeps"))     s->substeps = (int)state::patchFloat(i);
    else if (state::pathIs(path, plen, "pull"))         s->pull = state::patchFloat(i);
    else if (state::pathIs(path, plen, "jitter"))       s->jitter = state::patchFloat(i);
    else if (state::pathIs(path, plen, "drag"))         s->drag = state::patchFloat(i);
    else if (state::pathIs(path, plen, "size"))         s->size = state::patchFloat(i);
    else if (state::pathIs(path, plen, "size_jitter"))  s->size_jitter = state::patchFloat(i);
    else if (state::pathIs(path, plen, "life"))         s->life = state::patchFloat(i);
    else if (state::pathIs(path, plen, "life_jitter"))  s->life_jitter = state::patchFloat(i);
    else if (state::pathIs(path, plen, "sweep_center"))    s->sweep_center = state::patchFloat(i);
    else if (state::pathIs(path, plen, "sweep_width"))     s->sweep_width = state::patchFloat(i);
    else if (state::pathIs(path, plen, "sweep_soft"))      s->sweep_soft = state::patchFloat(i);
    else if (state::pathIs(path, plen, "sweep_smooth"))    s->sweep_smooth = state::patchFloat(i);
    else if (state::pathIs(path, plen, "image_smoothing")) s->image_smoothing = state::patchFloat(i);
    else if (state::pathIs(path, plen, "to_image"))        s->to_image = state::patchFloat(i);
    else if (state::pathIs(path, plen, "to_image_curl"))   s->to_image_curl = state::patchFloat(i);
    else if (state::pathIs(path, plen, "undertow_skew"))   s->undertow_skew = state::patchFloat(i);
    else if (state::pathIs(path, plen, "undertow_squash")) s->undertow_squash = state::patchFloat(i);
    else if (state::pathIs(path, plen, "noise_speed"))     s->noise_speed = state::patchFloat(i);
    else if (state::pathIs(path, plen, "noise_curl"))      s->noise_curl = state::patchFloat(i);
    else if (state::pathIs(path, plen, "eddy_scale"))      s->eddy_scale = state::patchFloat(i);
    else if (state::pathIs(path, plen, "eddy_detail"))     s->eddy_detail = state::patchFloat(i);
    else if (state::pathIs(path, plen, "eddy_evolve"))     s->eddy_evolve = state::patchFloat(i);
    else if (state::pathIs(path, plen, "eddy_drift"))      s->eddy_drift = state::patchFloat(i);
    else if (state::pathIs(path, plen, "eddy_drift_dir"))  s->eddy_drift_dir = state::patchFloat(i);
    else if (state::pathIs(path, plen, "l_count"))            s->l_count = (int)state::patchFloat(i);
    else if (state::pathIs(path, plen, "spawn_on_line"))      s->spawn_on_line = state::patchFloat(i);
    else if (state::pathIs(path, plen, "l_length"))           s->l_length = state::patchFloat(i);
    else if (state::pathIs(path, plen, "l_step"))             s->l_step = state::patchFloat(i);
    else if (state::pathIs(path, plen, "l_momentum"))         s->l_momentum = state::patchFloat(i);
    else if (state::pathIs(path, plen, "l_gradient_descent")) s->l_gradient_descent = state::patchFloat(i);
    else if (state::pathIs(path, plen, "l_snap"))             s->l_snap = state::patchFloat(i);
    else if (state::pathIs(path, plen, "l_arc"))              s->l_arc = state::patchFloat(i);
    else if (state::pathIs(path, plen, "l_adv"))              s->l_adv = state::patchFloat(i);
    else if (state::pathIs(path, plen, "l_grip_attack"))      s->l_grip_attack = state::patchFloat(i);
    else if (state::pathIs(path, plen, "l_grip_decay"))       s->l_grip_decay = state::patchFloat(i);
    else if (state::pathIs(path, plen, "l_grip_alpha"))       s->l_grip_alpha = state::patchFloat(i);
    else if (state::pathIs(path, plen, "l_fling_boost"))      s->l_fling_boost = state::patchFloat(i);
    else if (state::pathIs(path, plen, "l_time_decay"))       s->l_time_decay = state::patchFloat(i);
    else if (state::pathIs(path, plen, "l_reseed_spread"))    s->l_reseed_spread = state::patchFloat(i);
    else if (state::pathIs(path, plen, "l_color_contrib"))    s->l_color_contrib = state::patchFloat(i);
    else if (state::pathIs(path, plen, "l_width"))            s->l_width = state::patchFloat(i);
    else if (state::pathIs(path, plen, "l_soft"))             s->l_soft = state::patchFloat(i);
    else if (state::pathIs(path, plen, "l_opacity"))          s->l_opacity = state::patchFloat(i);
    else if (state::pathIs(path, plen, "interactions"))       { bool v = state::patchFloat(i) != 0.0f; if (v != s->interactions) { s->interactions = v; apply_mode_visibility(s->mode, s->interactions); } }
    else if (state::pathIs(path, plen, "interaction_radius")) s->interaction_radius = state::patchFloat(i);
    else if (state::pathIs(path, plen, "density_threshold"))  s->density_threshold = state::patchFloat(i);
    else if (state::pathIs(path, plen, "density_death"))      s->density_death = state::patchFloat(i);
    else if (state::pathIs(path, plen, "avoid"))              s->avoid = state::patchFloat(i);
    else if (state::pathIs(path, plen, "avoid_curl"))         s->avoid_curl = state::patchFloat(i);
    else if (state::pathIs(path, plen, "avoid_noise"))        s->avoid_noise = state::patchFloat(i);
    else if (state::pathIs(path, plen, "stream"))             s->stream = state::patchFloat(i);
    else if (state::pathIs(path, plen, "stream_density"))     s->stream_density = state::patchFloat(i);
    else if (state::pathIs(path, plen, "debug_density"))      s->debug_density = state::patchFloat(i) != 0.0f;
    else if (state::pathIs(path, plen, "debug_field"))        s->debug_field = state::patchFloat(i) != 0.0f;
    else if (state::pathIs(path, plen, "motion_line_speed"))     s->motion_line_speed = state::patchFloat(i);
    else if (state::pathIs(path, plen, "motion_particle_scale")) s->motion_particle_scale = state::patchFloat(i);
    else if (state::pathIs(path, plen, "intensity_sens"))     s->intensity_sens = state::patchFloat(i);
    else if (state::pathIs(path, plen, "intensity_attack"))   s->intensity_attack = state::patchFloat(i);
    else if (state::pathIs(path, plen, "intensity_decay"))    s->intensity_decay = state::patchFloat(i);
    else if (state::pathIs(path, plen, "calm_stretch"))       s->calm_stretch = state::patchFloat(i);
    else if (state::pathIs(path, plen, "intense_shrink"))     s->intense_shrink = state::patchFloat(i);
    else if (state::pathIs(path, plen, "respawn_rate"))       s->respawn_rate = state::patchFloat(i);
    else if (state::pathIs(path, plen, "line_boost"))         s->line_boost = state::patchFloat(i);
    else if (state::pathIs(path, plen, "jitter_boost"))       s->jitter_boost = state::patchFloat(i);
    else if (state::pathIs(path, plen, "fling_boost"))        s->fling_boost = state::patchFloat(i);
    else if (state::pathIs(path, plen, "release_gain"))       s->release_gain = state::patchFloat(i);
    else if (state::pathIs(path, plen, "release_decay"))      s->release_decay = state::patchFloat(i);
    else if (state::pathIs(path, plen, "boundary"))           s->boundary = state::patchFloat(i);
    else if (state::pathIs(path, plen, "boundary_size"))      s->boundary_size = state::patchFloat(i);
    else if (state::pathIs(path, plen, "boundary_stiffness")) s->boundary_stiffness = state::patchFloat(i);
    else if (state::pathIs(path, plen, "boundary_death"))     s->boundary_death = state::patchFloat(i);
    else if (state::pathIs(path, plen, "spawn_size"))         s->spawn_size = state::patchFloat(i);
    else if (state::pathIs(path, plen, "color_blend"))  s->color_blend = state::patchFloat(i);
    else if (state::pathIs(path, plen, "solid_color")) {
      auto v = state::patchVec3(i);
      s->solid_r = v.x; s->solid_g = v.y; s->solid_b = v.z;
    }
    else if (state::pathIs(path, plen, "tint_by_flow")) s->tint_by_flow = state::patchFloat(i);
    else if (state::pathIs(path, plen, "blend_mode"))   s->blend_mode = (int)state::patchFloat(i);
    else if (state::pathIs(path, plen, "opacity"))      s->opacity = state::patchFloat(i);
    else if (state::pathIs(path, plen, "input_alpha"))  s->input_alpha = state::patchFloat(i);
    else if (state::pathIs(path, plen, "shape_kind"))   s->shape_kind = (int)state::patchFloat(i);
    else if (state::pathIs(path, plen, "shape_param"))  s->shape_param = state::patchFloat(i);
    else if (state::pathIs(path, plen, "alpha_curve"))  s->alpha_curve = state::patchFloat(i);
    else if (state::pathIs(path, plen, "exposure"))     s->exposure = state::patchFloat(i);
    else if (state::pathIs(path, plen, "seed"))         s->seed = (int)state::patchFloat(i);
  }
}

void render(void* self, int vp_w, int vp_h) {
  auto* s = static_cast<State*>(self);
  if (!s || !s->initialized || vp_w <= 0 || vp_h <= 0) return;

  auto in  = gpu::Device::textureForField("tex_in");
  auto out = gpu::Device::textureForField("tex_out");
  if (!out.valid()) return;

  // Generator fallback: with no input wired, capture from 1×1 black (the sim
  // then free-flows on the noise field alone) and clear the base instead of
  // pre-filling.
  bool has_in = in.valid();
  if (!has_in) {
    if (!s->black_tex.valid()) {
      s->black_tex = gpu::Device::createTexture(1, 1, gpu::TextureFormat::RGBA8);
      gpu::Device::clear(s->black_tex, 0.0f, 0.0f, 0.0f, 1.0f);
    }
    in = s->black_tex;
  }

  // Lazy field textures. field_a is a ping-pong pair: the gen pass reads
  // last frame's field and writes an EMA blend (temporal smoothing).
  if (!s->field_a_tex.valid()) {
    s->field_a_tex = gpu::Device::createTexture(FIELD_RES, FIELD_RES,
                                                gpu::TextureFormat::RGBA16F);
    gpu::Device::clear(s->field_a_tex, 0.0f, 0.0f, 0.0f, 0.0f);
  }
  if (!s->field_a_tex2.valid()) {
    s->field_a_tex2 = gpu::Device::createTexture(FIELD_RES, FIELD_RES,
                                                 gpu::TextureFormat::RGBA16F);
    gpu::Device::clear(s->field_a_tex2, 0.0f, 0.0f, 0.0f, 0.0f);
  }
  if (!s->field_b_tex.valid()) {
    s->field_b_tex = gpu::Device::createTexture(FIELD_RES, FIELD_RES,
                                                gpu::TextureFormat::RGBA16F);
    gpu::Device::clear(s->field_b_tex, 0.0f, 0.0f, 0.0f, 0.0f);
  }
  if (!s->field_or_tex.valid()) {
    s->field_or_tex = gpu::Device::createTexture(FIELD_RES, FIELD_RES,
                                                 gpu::TextureFormat::RGBA16F);
    gpu::Device::clear(s->field_or_tex, 0.0f, 0.0f, 0.0f, 0.0f);
  }
  if (!s->field_c_tex.valid()) {
    s->field_c_tex = gpu::Device::createTexture(FIELD_RES, FIELD_RES,
                                                gpu::TextureFormat::RGBA16F);
    gpu::Device::clear(s->field_c_tex, 0.0f, 0.0f, 0.0f, 0.0f);
  }
  if (!s->field_c_tex2.valid()) {
    s->field_c_tex2 = gpu::Device::createTexture(FIELD_RES, FIELD_RES,
                                                 gpu::TextureFormat::RGBA16F);
    gpu::Device::clear(s->field_c_tex2, 0.0f, 0.0f, 0.0f, 0.0f);
  }
  if (!s->field_a_tex.valid() || !s->field_a_tex2.valid() ||
      !s->field_b_tex.valid() || !s->field_or_tex.valid() ||
      !s->field_c_tex.valid() || !s->field_c_tex2.valid()) return;
  s->field_a_flip = !s->field_a_flip;
  gpu::Texture fa_cur  = s->field_a_flip ? s->field_a_tex2 : s->field_a_tex;
  gpu::Texture fa_prev = s->field_a_flip ? s->field_a_tex  : s->field_a_tex2;
  gpu::Texture fc_cur  = s->field_a_flip ? s->field_c_tex2 : s->field_c_tex;
  gpu::Texture fc_prev = s->field_a_flip ? s->field_c_tex  : s->field_c_tex2;

  // Interactions: persistent density buffer read by the update pass (built
  // from LAST frame's particles → 1-frame delay). 1×1 zero when off.
  // Four buffers: a scatter target + a horizontal-blur scratch, plus a
  // ping-pong pair so this frame's blur never writes the texture the update
  // pass already sampled.
  bool ix = s->interactions;
  // The density chain is the most expensive interaction work; only run it
  // when something downstream actually reads the buffer.
  bool need_density = ix && (s->density_death > 0.0f || s->avoid > 0.0f ||
                             s->stream != 0.0f || s->debug_density);
  gpu::Texture dens_cur, dens_prev;
  gpu::Texture density_in;
  if (need_density) {
    gpu::Texture* dts[4] = { &s->density_tex, &s->density_tex2,
                             &s->density_splat_tex, &s->density_tmp_tex };
    for (auto* t : dts) {
      if (t->valid()) continue;
      *t = gpu::Device::createTexture(DENSITY_RES, DENSITY_RES,
                                      gpu::TextureFormat::RGBA16F);
      gpu::Device::clear(*t, 0.0f, 0.0f, 0.0f, 0.0f);
    }
    for (auto* t : dts) if (!t->valid()) need_density = false;
  }
  if (need_density) {
    s->density_flip = !s->density_flip;
    dens_cur  = s->density_flip ? s->density_tex2 : s->density_tex;
    dens_prev = s->density_flip ? s->density_tex  : s->density_tex2;
    density_in = dens_prev;
  } else {
    if (!s->zero_density_tex.valid()) {
      s->zero_density_tex = gpu::Device::createTexture(1, 1, gpu::TextureFormat::RGBA16F);
      gpu::Device::clear(s->zero_density_tex, 0.0f, 0.0f, 0.0f, 0.0f);
    }
    density_in = s->zero_density_tex;
  }

  s->frame_index++;
  float dt = (float)host::deltaTime();

  // Isotropic-uv aspect (1 unit = min(W,H) px) — shared by every pass.
  float min_dim = float(vp_w < vp_h ? vp_w : vp_h);
  float aspect_x = min_dim / float(vp_w);
  float aspect_y = min_dim / float(vp_h);

  // Noise phases: accumulate at the CURRENT rates so rate edits are smooth.
  // Drift wraps far out (one lattice-hash period is irrelevant at 4096) to
  // stay in float-precision range after hours of runtime.
  s->spin_phase  += s->eddy_evolve * dt;
  s->drift_phase += s->eddy_drift * dt;
  if (s->spin_phase  > 4096.0f) s->spin_phase  -= 4096.0f;
  if (s->drift_phase > 4096.0f) s->drift_phase -= 4096.0f;

  // Temporal EMA: replace outright when smoothing is off OR paused (dt 0 —
  // a paused drag should still show its result immediately).
  float blend_k = (s->sweep_smooth <= 1e-4f || dt <= 0.0f)
                      ? 1.0f
                      : 1.0f - std::exp(-dt / s->sweep_smooth);

  FieldAUniforms fau = {};
  fau.field_res    = (uint32_t)FIELD_RES;
  fau.sweep_center = s->sweep_center;
  fau.sweep_width  = s->sweep_width;
  fau.sweep_soft   = s->sweep_soft;
  fau.blend_k      = blend_k;
  s->field_a_uniforms.writeOne(fau);

  FieldUniforms fu = {};
  fu.field_res       = (uint32_t)FIELD_RES;
  fu.aspect_x        = aspect_x;
  fu.aspect_y        = aspect_y;
  fu.noise_speed     = s->noise_speed;
  fu.noise_curl      = s->noise_curl;
  fu.eddy_scale      = s->eddy_scale;
  fu.eddy_detail     = s->eddy_detail;
  fu.spin_phase      = s->spin_phase;
  fu.drift_phase     = s->drift_phase;
  fu.drift_dir       = s->eddy_drift_dir;
  fu.image_smoothing = s->image_smoothing;
  fu.blend_k         = blend_k;
  s->field_uniforms.writeOne(fu);

  float size_uv = SIZE_SCALE * s->size * s->size;

  UpdateUniforms uu = {};
  uu.count              = (uint32_t)s->count;
  uu.frame_index        = s->frame_index;
  uu.dt                 = dt;
  uu.speed              = s->speed;
  uu.momentum           = s->momentum;
  uu.jitter             = s->jitter;
  uu.drag               = s->drag;
  uu.life               = s->life;
  uu.life_jitter        = s->life_jitter;
  uu.size               = size_uv;
  uu.size_jitter        = s->size_jitter;
  uu.seed               = (uint32_t)s->seed;
  uu.mode               = (uint32_t)s->mode;
  uu.weight             = s->weight;
  uu.pull               = s->pull;
  uu.to_image           = s->to_image;
  uu.to_image_curl      = s->to_image_curl;
  uu.undertow_skew      = s->undertow_skew;
  uu.undertow_squash    = s->undertow_squash;
  uu.aspect_x           = aspect_x;
  uu.aspect_y           = aspect_y;
  uu.substeps           = (uint32_t)(s->substeps < 1 ? 1 : (s->substeps > 16 ? 16 : s->substeps));
  uu.boundary           = s->boundary;
  uu.boundary_size      = s->boundary_size;
  uu.boundary_stiffness = s->boundary_stiffness;
  uu.boundary_death     = s->boundary_death;
  uu.spawn_size         = s->spawn_size;
  // Live segment slots per tracer this frame (2 passes × steps, dc parity).
  int steps = s->l_length <= 0.0f ? 2 : (int)(s->l_length * (MAX_SEG / 2));
  if (steps < 2) steps = 2;
  int seg_live = steps * 2;
  if (seg_live > MAX_SEG) seg_live = MAX_SEG;
  uu.to_line_rate       = (s->l_count > 0) ? s->spawn_on_line : 0.0f;
  uu.l_count_f          = (float)s->l_count;
  uu.seg_stride         = (float)MAX_SEG;
  uu.seg_live           = (float)seg_live;
  uu.calm_stretch       = s->calm_stretch;
  uu.intense_shrink     = s->intense_shrink;
  uu.respawn_rate       = s->respawn_rate;
  uu.line_boost         = s->line_boost;
  uu.jitter_boost       = s->jitter_boost;
  uu.fling_boost        = s->fling_boost;
  uu.interactions       = ix ? 1u : 0u;
  uu.density_threshold  = s->density_threshold;
  uu.density_death      = s->density_death;
  uu.avoid              = s->avoid;
  uu.avoid_curl         = s->avoid_curl;
  uu.avoid_noise        = s->avoid_noise;
  uu.density_res        = (float)DENSITY_RES;
  uu.stream             = s->stream;
  uu.stream_density     = s->stream_density;
  uu.field_res          = (float)FIELD_RES;
  uu.dens_scale         = DENSITY_SCALE;
  s->update_uniforms.writeOne(uu);

  StatsUniforms su = {};
  su.field_res        = (float)FIELD_RES;
  su.dt               = dt;
  su.intensity_attack = s->intensity_attack;
  su.intensity_decay  = s->intensity_decay;
  su.intensity_sens   = s->intensity_sens;
  su.release_gain     = s->release_gain;
  su.release_decay    = s->release_decay;
  s->stats_uniforms.writeOne(su);

  TraceUniforms tu = {};
  tu.count            = (uint32_t)s->l_count;
  tu.max_seg          = (uint32_t)MAX_SEG;
  tu.frame_index      = s->frame_index;
  tu.dt               = dt;
  tu.aspect_x         = aspect_x;
  tu.aspect_y         = aspect_y;
  tu.field_res        = (float)FIELD_RES;
  tu.to_image         = s->to_image;
  tu.to_image_curl    = s->to_image_curl;
  tu.step_len         = s->l_step * 0.01f;  // UI unit = 0.01 iso (survives 0.01 quantize)
  tu.length01         = s->l_length;
  tu.momentum         = s->l_momentum;
  tu.gradient_descent = s->l_gradient_descent;
  tu.snap             = s->l_snap;
  tu.arc              = s->l_arc;
  tu.adv              = s->l_adv;
  tu.grip_attack      = s->l_grip_attack;
  tu.grip_decay       = s->l_grip_decay;
  tu.grip_alpha       = s->l_grip_alpha;
  tu.fling_boost      = s->l_fling_boost;
  tu.time_decay       = s->l_time_decay;
  tu.reseed_spread    = s->l_reseed_spread;
  tu.color_contrib    = s->l_color_contrib;
  tu.l_opacity        = s->l_opacity;
  tu.tint_r           = s->solid_r;
  tu.tint_g           = s->solid_g;
  tu.tint_b           = s->solid_b;
  tu.seed_rng         = (float)s->seed;
  s->trace_uniforms.writeOne(tu);

  PrefillUniforms pu = { s->input_alpha, s->input_alpha, s->input_alpha, 1.0f };
  s->prefill_uniforms.writeOne(pu);

  VsUniforms vu = {};
  vu.aspect_x   = aspect_x;
  vu.aspect_y   = aspect_y;
  vu.point_size = POINT_PX / min_dim;
  vu.shape_kind = (float)s->shape_kind;
  s->vs_uniforms.writeOne(vu);

  ColorUniforms cu = {};
  // Generator mode has no input to capture — captured colour is black, so
  // force the solid colour or the swarm silently vanishes.
  cu.color_blend  = has_in ? s->color_blend : 0.0f;
  cu.solid_r      = s->solid_r;
  cu.solid_g      = s->solid_g;
  cu.solid_b      = s->solid_b;
  cu.tint_by_flow = s->tint_by_flow;
  cu.opacity      = s->opacity;
  cu.alpha_curve  = s->alpha_curve;
  cu.shape_param  = s->shape_param;
  cu.shape_kind   = (uint32_t)s->shape_kind;
  cu.exposure     = s->exposure;
  s->color_uniforms.writeOne(cu);

  // ---- Pass 1a: sweep + downsample the input into field_a ----
  // Always runs (with no input it reduces the 1×1 black fallback → all-zero
  // L', i.e. the free-flow state) so field_a is never stale.
  {
    auto cp = gpu::ComputePass::begin();
    cp.setPSO(s_pso_field_a);
    cp.setTexture(fa_cur, 0, 1);
    cp.setTexture(in, 1, 0);
    cp.setSampler(s->sampler, 2);
    cp.setBuffer(s->field_a_uniforms, 3);
    cp.setTexture(fa_prev, 4, 0);
    cp.setTexture(s->field_or_tex, 5, 1);
    cp.dispatch((FIELD_RES + 7) / 8, (FIELD_RES + 7) / 8);
    cp.end();
  }

  // ---- Pass 1b: build the velocity field (curl noise + ∇L') ----
  {
    auto cp = gpu::ComputePass::begin();
    cp.setPSO(s_pso_field_b);
    cp.setTexture(s->field_b_tex, 0, 1);
    cp.setTexture(fa_cur, 1, 0);
    cp.setSampler(s->sampler, 2);
    cp.setBuffer(s->field_uniforms, 3);
    cp.setTexture(s->field_or_tex, 4, 0);
    cp.setTexture(fc_cur, 5, 1);
    cp.setTexture(fc_prev, 6, 0);
    cp.dispatch((FIELD_RES + 7) / 8, (FIELD_RES + 7) / 8);
    cp.end();
  }

  // ---- Pass 1c: swept-image stats → calm↔intense response (one group) ----
  {
    auto cp = gpu::ComputePass::begin();
    cp.setPSO(s_pso_stats);
    cp.setTexture(fa_cur, 0, 0);
    cp.setTexture(s->field_b_tex, 1, 0);
    cp.setBuffer(s->stats_buf, 2);
    cp.setBuffer(s->stats_uniforms, 3);
    cp.dispatch(1, 1, 1);
    cp.end();
  }

  // ---- Pass 2a: tracers (BEFORE p_update so spawn-on-line sees this
  // frame's segments) ----
  if (s->l_count > 0) {
    auto cp = gpu::ComputePass::begin();
    cp.setPSO(s_pso_trace);
    cp.setBuffer(s->tracer_buf, 0);
    cp.setBuffer(s->seg_buf, 1);
    cp.setTexture(fa_cur, 2, 0);
    cp.setTexture(s->field_b_tex, 3, 0);
    cp.setSampler(s->sampler, 4);
    cp.setBuffer(s->trace_uniforms, 5);
    cp.setTexture(in, 6, 0);
    cp.setTexture(fc_cur, 7, 0);
    cp.dispatch((s->l_count + 63) / 64, 1, 1);
    cp.end();
  }

  // ---- Pass 2b: update particles ----
  {
    auto cp = gpu::ComputePass::begin();
    cp.setPSO(s_pso_update);
    cp.setBuffer(s->particle_buf, 0);
    cp.setTexture(s->field_b_tex, 1, 0);
    cp.setTexture(in, 2, 0);
    cp.setSampler(s->sampler, 3);
    cp.setBuffer(s->update_uniforms, 4);
    cp.setTexture(density_in, 5, 0);
    cp.setBuffer(s->seg_buf, 6);
    cp.setBuffer(s->tracer_buf, 7);
    cp.setBuffer(s->stats_buf, 8);
    cp.setTexture(fa_cur, 9, 0);
    cp.setTexture(fc_cur, 10, 0);
    int groups = (s->count + 63) / 64;
    cp.dispatch(groups, 1, 1);
    cp.end();
  }

  // ---- Pass 3: base (pre-fill from input, or clear when generating) ----
  if (has_in) {
    auto cp = gpu::ComputePass::begin();
    cp.setPSO(s_pso_prefill);
    cp.setTexture(in,  0, 0);
    cp.setTexture(out, 1, 1);
    cp.setBuffer(s->prefill_uniforms, 2);
    cp.dispatch((vp_w + 7) / 8, (vp_h + 7) / 8);
    cp.end();
  } else {
    gpu::Device::clear(out, 0.0f, 0.0f, 0.0f, 1.0f);
  }

  bool debug = (ix && s->debug_density) || s->debug_field;

  // ---- Pass 4: instanced particle raster ----
  if (s->opacity > 0.0f && !debug) {
    auto rp = gpu::RenderPass::beginLoad(out);
    auto pso = (s->blend_mode == BLEND_ADD) ? s_pso_render_add : s_pso_render_alpha;
    rp.setPSO(pso);
    rp.setBuffer(s->particle_buf, 0);
    rp.setBuffer(s->vs_uniforms,  1);
    rp.setBuffer(s->color_uniforms, 2);
    rp.draw(6, s->count);
    rp.end();
  }

  // ---- Pass 5: instanced line raster ----
  if (s->l_count > 0 && s->l_opacity > 0.0f && !debug) {
    LineVsUniforms lvu = { aspect_x, aspect_y, s->l_width * LINE_WIDTH_SCALE, 0.0f };
    s->line_vs_uniforms.writeOne(lvu);
    LineFsUniforms lfu = { s->l_soft, 0.0f, 0.0f, 0.0f };
    s->line_fs_uniforms.writeOne(lfu);
    auto rp = gpu::RenderPass::beginLoad(out);
    auto pso = (s->blend_mode == BLEND_ADD) ? s_pso_line_add : s_pso_line_alpha;
    rp.setPSO(pso);
    rp.setBuffer(s->seg_buf, 0);
    rp.setBuffer(s->line_vs_uniforms, 1);
    rp.setBuffer(s->line_fs_uniforms, 2);
    rp.draw(6, s->l_count * MAX_SEG);
    rp.end();
  }

  // ---- Pass 6: density scatter + halo convolution (after update moved the
  // particles) → next frame's read.
  //
  // The interaction halo is a gaussian, and a gaussian is separable, so the
  // per-particle footprint stays at 2×2 texels (unit mass) and the radius is
  // applied by two 1-D passes over the 256² buffer. Cost is O(N) + O(RES²·σ)
  // instead of the old O(N·radius²) blended fragments — at 700k particles and
  // radius 0.04 that was >100M additive fragments piling onto the same few
  // texels wherever the swarm clustered, which is what tanked the frame rate.
  if (need_density) {
    DensityUniforms du = { (float)DENSITY_RES, DENSITY_SCALE,
                           0.5f * (1.0f - DENSITY_SCALE), 0.f };
    s->density_uniforms.writeOne(du);
    {
      auto rp = gpu::RenderPass::begin(s->density_splat_tex, 0.0f, 0.0f, 0.0f, 0.0f);
      rp.setPSO(s_pso_density);
      rp.setBuffer(s->particle_buf, 0);
      rp.setBuffer(s->density_uniforms, 1);
      rp.draw(6, s->count);
      rp.end();
    }
    // σ per axis in density texels: the halo is round in SCREEN pixels, and
    // the buffer is square, so the aspect factors survive into the kernel.
    // Truncate at 2σ — exactly where the old splat quad's `discard` cut off.
    auto blur_axis = [&](gpu::Texture src, gpu::Texture dst, gpu::Buffer ub,
                         float sigma, float dx, float dy) {
      float taps = std::ceil(2.0f * sigma);
      if (taps > 24.0f) taps = 24.0f;
      if (!(taps > 0.0f)) taps = 0.0f;
      BlurUniforms bu = { (float)DENSITY_RES, dx, dy,
                          1.0f / (sigma > 1e-3f ? sigma : 1e-3f),
                          taps, 0.f, 0.f, 0.f };
      ub.writeOne(bu);
      auto cp = gpu::ComputePass::begin();
      cp.setPSO(s_pso_density_blur);
      cp.setTexture(src, 0, 0);
      cp.setTexture(dst, 1, 1);
      cp.setBuffer(ub, 2);
      cp.dispatch((DENSITY_RES + 7) / 8, (DENSITY_RES + 7) / 8);
      cp.end();
    };
    float half_r = 0.5f * s->interaction_radius * (float)DENSITY_RES * DENSITY_SCALE;
    blur_axis(s->density_splat_tex, s->density_tmp_tex, s->blur_h_uniforms,
              half_r * aspect_x, 1.0f, 0.0f);
    blur_axis(s->density_tmp_tex, dens_cur, s->blur_v_uniforms,
              half_r * aspect_y, 0.0f, 1.0f);
  }

  // ---- Pass 7 (debug): field readout or density heat map into tex_out. ----
  if (s->debug_field) {
    FieldDebugUniforms fdu = { s->to_image, s->to_image_curl, 0.f, 0.f };
    s->field_debug_uniforms.writeOne(fdu);
    auto cp = gpu::ComputePass::begin();
    cp.setPSO(s_pso_field_debug);
    cp.setTexture(fa_cur, 0, 0);
    cp.setTexture(s->field_b_tex, 1, 0);
    cp.setSampler(s->sampler, 2);
    cp.setTexture(out, 3, 1);
    cp.setBuffer(s->field_debug_uniforms, 4);
    cp.setTexture(fc_cur, 5, 0);
    cp.dispatch((vp_w + 7) / 8, (vp_h + 7) / 8);
    cp.end();
  } else if (debug && dens_cur.valid()) {
    auto cp = gpu::ComputePass::begin();
    cp.setPSO(s_pso_density_debug);
    cp.setTexture(dens_cur, 0, 0);
    cp.setBuffer(s->density_uniforms, 3);
    cp.setSampler(s->sampler, 1);
    cp.setTexture(out, 2, 1);
    cp.dispatch((vp_w + 7) / 8, (vp_h + 7) / 8);
    cp.end();
  }

  // ---- Pass 8: motion vectors (render_outputs/motion). Only when a
  // downstream sink reads the rail; pure overhead otherwise.
  if (state::isOutputConnected("render_outputs")) {
    if (!s->motion_tex.valid() || s->motion_w != vp_w || s->motion_h != vp_h) {
      s->motion_tex.release();
      s->motion_tex = gpu::Device::createTexture(vp_w, vp_h, gpu::TextureFormat::RGBA16F);
      s->motion_w = vp_w; s->motion_h = vp_h;
      if (s->motion_tex.valid()) state::setGpuTexture("render_outputs/motion", s->motion_tex.id);
    }
    if (s->motion_tex.valid()) {
      // Seed from upstream motion (1×1 zero fallback → effectively a clear).
      auto up = gpu::Device::textureForField("render_outputs_in/motion");
      if (!up.valid()) {
        if (!s->zero_motion_tex.valid())
          s->zero_motion_tex = gpu::Device::createTexture(1, 1, gpu::TextureFormat::RGBA16F);
        up = s->zero_motion_tex;
      }
      if (up.valid()) {
        auto cp = gpu::ComputePass::begin();
        cp.setPSO(s_pso_motion_prefill);
        cp.setTexture(up, 0, 0);
        cp.setTexture(s->motion_tex, 1, 1);
        cp.dispatch((vp_w + 7) / 8, (vp_h + 7) / 8);
        cp.end();
      }

      float mpoint = (s->shape_kind == SHAPE_POINT) ? (POINT_PX / min_dim) : size_uv;
      MotionVsUniforms mvu = { aspect_x, aspect_y, mpoint, dt,
                               s->motion_particle_scale, 0.f, 0.f, 0.f };
      s->motion_vs_uniforms.writeOne(mvu);
      MotionFsUniforms mfu = { (uint32_t)s->shape_kind, s->shape_param, 0.f, 0.f };
      s->motion_fs_uniforms.writeOne(mfu);
      float line_spd = s->motion_line_speed * MOTION_LINE_SCALE;
      LineMotionVsUniforms lmu = { aspect_x, aspect_y,
                                   s->l_width * LINE_WIDTH_SCALE, line_spd };
      s->lm_vs_uniforms.writeOne(lmu);
      LineFsUniforms lfu2 = { s->l_soft, 0.0f, 0.0f, 0.0f };
      s->line_fs_uniforms.writeOne(lfu2);

      // Motion mirrors VISIBILITY: an element only writes motion when it's
      // actually drawn in the colour pass — otherwise an invisible element
      // would punch "holes" into the motion field (dc parity).
      bool draw_p = s->opacity > 0.0f && !debug;
      bool draw_l = line_spd > 0.0f && s->l_count > 0 && s->l_opacity > 0.0f && !debug;

      if (draw_p || draw_l) {
        auto rp = gpu::RenderPass::beginLoad(s->motion_tex);
        if (draw_p) {
          rp.setPSO(s_pso_motion_point);
          rp.setBuffer(s->particle_buf, 0);
          rp.setBuffer(s->motion_vs_uniforms, 1);
          rp.setBuffer(s->motion_fs_uniforms, 2);
          rp.draw(6, s->count);
        }
        if (draw_l) {
          rp.setPSO(s_pso_motion_line);
          rp.setBuffer(s->seg_buf, 0);
          rp.setBuffer(s->lm_vs_uniforms, 1);
          rp.setBuffer(s->line_fs_uniforms, 2);   // reuse tracer soft falloff
          rp.draw(6, s->l_count * MAX_SEG);
        }
        rp.end();
      }
    }
  }

  gpu::Device::submit();
}

} // namespace sweep_chamber
