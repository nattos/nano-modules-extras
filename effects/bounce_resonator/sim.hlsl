// source.light.bounce_resonator — Stage 1: the diffusion sim (GPU-resident state).
//
// Single-threaded (gid 0 only) — the native Metal backend hardcodes 8×8
// threadgroups, which breaks single-workgroup parallel shaders, so the
// whole 4-bar sim runs serially on one thread (the work is tiny). State
// (per-bar value + hue, plus the decay-phase envelope) lives in a storage
// buffer that persists across frames.
//
// Per frame: run n_hops of the cycling diffusion matrix (CPU-built, uploaded
// in `mats`), each hop carrying value AND colour (hue+saturation as an
// intensity·saturation-weighted hue vector + per-edge hue shift — mixing
// disagreeing hues desaturates). Then inject impulses AFTER the hops (so a
// fresh trigger is a solid flash). In gen mode the impulse is band_color's
// hue/sat + a queued amount; in tex_in mode a trigger samples tex_in per bar
// for its dominant hue + luminance (scaled by tex_in_boost), blended toward
// band_color by tex_in_color.

#include "nano_color.hlsl"

static const float TAU = 6.28318530717958648;

Texture2D<float4> inputTex : register(t0);

struct SimState {
  float v[4];
  float h[4];     // hue, turns
  float s[4];     // saturation
  float env;      // decay-phase envelope follower
  float pad[3];
};
RWStructuredBuffer<SimState> simState : register(u1);

StructuredBuffer<float> mats : register(t2);   // pattern_count × 48 floats

cbuffer SimU : register(b3) {
  float feedback; float decay_shaping; float hue_converge; float home_hue;
  int   pattern_count; int hop_idx_start; int n_hops; int mode;   // mode 1 = tex_in sampling
  float pending0; float pending1; float pending2; float pending3;
  float band_hue; float tex_in_boost; int trigger_fired; int do_reset;
  int   tex_w; int tex_h; int sample_nx; int sample_ny;
  float band_sat; float tex_in_color; float upad0; float upad1;
};

static float wrapT(float h) { return frac(h + 1.0); }   // → [0,1)

// Colour mixing treats each bar as a hue vector of length value·saturation:
// mixing agreeing hues keeps saturation, opposing hues cancel toward grey.
void injectBar(inout SimState st, int b, float amt, float hue, float sat) {
  if (amt <= 0.0) return;
  float a0 = TAU * st.h[b], a1 = TAU * hue;
  float w0 = st.v[b] * st.s[b], w1 = amt * sat;
  float cx = w0 * cos(a0) + w1 * cos(a1);
  float cy = w0 * sin(a0) + w1 * sin(a1);
  st.v[b] += amt;
  float rl = sqrt(cx * cx + cy * cy);
  if (rl > 1e-6) st.h[b] = wrapT(atan2(cy, cx) / TAU);
  st.s[b] = (st.v[b] > 1e-6) ? saturate(rl / st.v[b]) : sat;
}

[numthreads(64, 1, 1)]
void main(uint3 gid : SV_DispatchThreadID) {
  if (gid.x != 0u || gid.y != 0u || gid.z != 0u) return;

  SimState st = simState[0];
  if (do_reset != 0) {
    [unroll] for (int i = 0; i < 4; i++) { st.v[i] = 0.0; st.h[i] = 0.0; st.s[i] = 0.0; }
    st.env = 0.0;
  }

  int nc = max(pattern_count, 1);
  [loop] for (int s = 0; s < n_hops; s++) {
    int base = ((hop_idx_start + s) % nc) * 48;

    // Decay-phase proxy + shaped feedback (mirrors effect_diffusion_network.h).
    float total = st.v[0] + st.v[1] + st.v[2] + st.v[3];
    st.env = (total > st.env) ? total : st.env * 0.98;
    float phase = (st.env > 1e-6) ? min(total / st.env, 1.0) : 0.0;
    float fb = feedback;
    if (decay_shaping != 0.0 && feedback < 1.0) {
      float d = 1.0 - feedback;
      float factor = 1.0 + decay_shaping * 0.80 * (1.0 - 2.0 * phase);
      float deff = clamp(d * factor, 0.0, 1.0);
      fb = 1.0 - deff;
    }

    // Hue vectors scaled by saturation: fully grey energy carries no hue,
    // and mixing disagreeing hues desaturates the receiving bar.
    float cj[4], sj[4];
    [unroll] for (int j = 0; j < 4; j++) { float a = TAU * st.h[j]; cj[j] = st.s[j] * cos(a); sj[j] = st.s[j] * sin(a); }

    float nv[4], nhx[4], nhy[4], na[4];
    [unroll] for (int i = 0; i < 4; i++) {
      float a = 0.0, hx = 0.0, hy = 0.0;
      [unroll] for (int jj = 0; jj < 4; jj++) {
        float w  = mats[base + i * 4 + jj] * st.v[jj];
        a += w;
        float cd = mats[base + 16 + i * 4 + jj];
        float sd = mats[base + 32 + i * 4 + jj];
        float rc = cj[jj] * cd - sj[jj] * sd;
        float rs = sj[jj] * cd + cj[jj] * sd;
        hx += w * rc; hy += w * rs;
      }
      nv[i] = a * fb; nhx[i] = hx; nhy[i] = hy; na[i] = a;
    }

    float conv = 0.0;
    if (hue_converge > 0.0) { float t = 1.0 - phase; conv = min(hue_converge * t * t, 1.0); }

    [unroll] for (int i2 = 0; i2 < 4; i2++) {
      st.v[i2] = clamp(nv[i2], 0.0, 1e6);
      float rl = sqrt(nhx[i2] * nhx[i2] + nhy[i2] * nhy[i2]);
      if (rl > 1e-6)
        st.h[i2] = wrapT(atan2(nhy[i2], nhx[i2]) / TAU);
      if (na[i2] > 1e-6) st.s[i2] = saturate(rl / na[i2]);
      if (conv > 0.0) {
        float delta = frac(home_hue - st.h[i2] + 1.5) - 0.5;
        st.h[i2] = wrapT(st.h[i2] + delta * conv);
      }
    }
  }

  // Inject impulses after the hops (solid, undiffused flash).
  if (mode == 0) {
    float pend[4] = { pending0, pending1, pending2, pending3 };
    [unroll] for (int b = 0; b < 4; b++) injectBar(st, b, pend[b], band_hue, band_sat);
  } else if (trigger_fired != 0) {
    // Sample tex_in per bar: dominant hue = sat·val-weighted circular mean of
    // the sample hues (its resultant length ↔ how coherent the hue is → the
    // injected saturation), impulse amount = mean luminance. tex_in_color
    // blends the injected colour from band_color (0) to the dominant colour
    // (1). Bar b owns x in [b/4, (b+1)/4) over the full height.
    int nx = max(sample_nx, 1), ny = max(sample_ny, 1);
    float cstr = saturate(tex_in_color);
    [loop] for (int b = 0; b < 4; b++) {
      float lum = 0.0, hue_x = 0.0, hue_y = 0.0, vsum = 0.0;
      [loop] for (int gy = 0; gy < ny; gy++) {
        for (int gx = 0; gx < nx; gx++) {
          float ux = (float(b) + (float(gx) + 0.5) / float(nx)) * 0.25;
          float uy = (float(gy) + 0.5) / float(ny);
          int px = clamp(int(ux * float(tex_w)), 0, tex_w - 1);
          int py = clamp(int(uy * float(tex_h)), 0, tex_h - 1);
          float3 rgb = inputTex.Load(int3(px, py, 0)).rgb;
          float3 hsv = nano_rgb_to_hsv(rgb);
          float w = hsv.y * hsv.z;
          float a = TAU * hsv.x;
          hue_x += w * cos(a); hue_y += w * sin(a);
          vsum += hsv.z;
          lum += dot(rgb, float3(0.299, 0.587, 0.114));
        }
      }
      lum /= float(nx * ny);
      float rl = sqrt(hue_x * hue_x + hue_y * hue_y);
      float domHue = (rl > 1e-5) ? wrapT(atan2(hue_y, hue_x) / TAU) : band_hue;
      float domSat = (vsum > 1e-5) ? saturate(rl / vsum) : 0.0;
      float dh = frac(domHue - band_hue + 1.5) - 0.5;   // shortest-path hue lerp
      float hue = wrapT(band_hue + dh * cstr);
      float sat = lerp(band_sat, domSat, cstr);
      injectBar(st, b, lum * tex_in_boost, hue, sat);
    }
  }

  simState[0] = st;
}
