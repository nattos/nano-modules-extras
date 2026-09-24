// source.light.side_jet — Stage 2: 2D plume synthesis (color).
//
// Stateless w.r.t. history → reacts in one frame. For each pixel we map x
// to an axial station, read the 1D solver state (u, p, b, m, phi, lit), and
// paint the plume anatomy that hands off down the axis:
//
//   potential core  — glassy hard-edged cone near the nozzle (low maturity)
//   shock diamonds  — bright re-compression nodes at cos(phi); spacing comes
//                     from the solver, so they breathe with chamber pressure
//   Mach disk       — a bright perpendicular band at the normal shock
//   shear layer     — high-frequency KH vortices churning the cone edge
//   turbulent far   — fbm billows as maturity → 1
//   crackle         — fast luminosity shimmer on the whole plume
//   sparks          — CPU pool, additive, sprayed on ignition (low frequency)

#include "nano_hash.hlsl"

Texture2D<float4>   inputTex  : register(t0);
RWTexture2D<float4> outputTex : register(u1);

cbuffer Uniforms : register(b2) {
  float intensity;        float centerline_y;   float nozzle_radius;  float spread;
  float radial_sharpness; float diamond_amp;    float mach_disk_x;    float mach_disk_amp;
  float mach_disk_width;  float shimmer_phase;  float kh_amp;         float kh_scale;
  float kh_phase;         float crackle_amp;    float crackle_phase;  float mixture;
  float zoom;             float _padc1;         float _padc2;         float aspect;
  float _pade0;           float _pade1;         float _pade2;         float core_brightness;
  uint  cell_count;       uint  spark_count;    uint  debug_show_axis; float motion_scale;
};

struct Cell {
  float u; float p; float b; float m;
  float kappa; float phi; float lit; float _pad;
};
StructuredBuffer<Cell> cells : register(t3);

struct Spark { float x; float y; float vx; float vy; float life; float size; float _p0; float _p1; };
StructuredBuffer<Spark> sparks : register(t4);

static const float TAU = 6.28318530717958;

[numthreads(8, 8, 1)]
void main(uint3 gid : SV_DispatchThreadID) {
  uint w, h;
  outputTex.GetDimensions(w, h);
  if (gid.x >= w || gid.y >= h) return;

  float2 uv = (float2(gid.xy) + 0.5) / float2(w, h);
  float4 base = inputTex[gid.xy];
  float3 add = float3(0.0, 0.0, 0.0);

  // Zoom — magnify the jet anchored at left-center (0, 0.5). Overwriting uv
  // means every downstream jet + spark term scales together. Lets you crank
  // a long apparent core by shrinking nozzle_radius and zooming in.
  uv = float2(uv.x / max(zoom, 1e-3), (uv.y - 0.5) / max(zoom, 1e-3) + 0.5);

  // --- Sample the axial state at this pixel's x (linear between cells). ---
  uint nc = max(cell_count, 1u);
  float fidx = saturate(uv.x) * (float)(nc - 1u);
  uint i0 = (uint)floor(fidx);
  uint i1 = min(i0 + 1u, nc - 1u);
  float fr = fidx - (float)i0;
  Cell a = cells[i0];
  Cell c = cells[i1];
  float u_ax  = lerp(a.u,   c.u,   fr);
  float p_ax  = lerp(a.p,   c.p,   fr);
  float b_ax  = lerp(a.b,   c.b,   fr);
  float m_ax  = lerp(a.m,   c.m,   fr);
  float phi   = lerp(a.phi, c.phi, fr);
  float lit   = lerp(a.lit, c.lit, fr);

  // --- Jet geometry (BIG nozzle at the left edge, axis at centerline_y).
  // A fat, roughly collimated beam (like a real test-stand plume): wide at
  // the nozzle, tapering slightly, with the turbulent edge feathering out
  // downstream as maturity grows. ---
  float dy = uv.y - centerline_y;
  float hw = nozzle_radius * (1.0 - 0.45 * smoothstep(0.0, 0.7, uv.x))
           + spread * uv.x * 0.4 * m_ax;
  float rn = dy / max(hw, 1e-4);
  float litgate = smoothstep(0.0, 0.12, lit);

  if (abs(rn) < 2.5 && litgate > 0.001 && b_ax > 1e-4) {
    float rn2 = rn * rn;
    float src = litgate * b_ax * core_brightness;

    // Two components with a big dynamic range — this is what makes the
    // white-hot core read against the blue body instead of everything
    // blowing out to white:
    //   * core beam — a CRISP-EDGED top-hat (shock-bounded), so the core
    //     keeps sharp characteristics even at low throttle instead of fading
    //     into a soft incandescent bulb. radial_sharpness sets the flat-top
    //     width / edge crispness.
    //   * flame body — a defined (not foggy) blue beam.
    float aedge     = abs(rn);
    float inner     = saturate(1.0 - 2.2 / max(radial_sharpness, 1.0));
    float core_prof = 1.0 - smoothstep(inner, 1.02, aedge);          // crisp top-hat
    float halo_prof = exp(-rn2 * radial_sharpness * 0.42);           // defined beam
    float billow    = 0.7 + 0.5 * nano_fbm2(float2(uv.x * 7.0 + kh_phase * 0.15,
                                                    dy * 9.0), 3);
    // The incandescent core is the POTENTIAL CORE — only where the flow is
    // still coherent (low maturity). Past breakdown it's the blue body.
    float coreness  = pow(saturate(1.0 - m_ax), 1.3);
    float core_beam = src * core_prof * coreness;
    float halo      = src * 0.55 * lerp(core_prof, halo_prof * billow, m_ax);

    // Shock diamonds — sharp bright NODES bulging on the axis. Phase comes
    // from the solver so spacing breathes with pressure; fade with maturity.
    // Diamonds live in the supersonic body (overlap the core AND the near
    // blue), only fading in the far field — so keep them past breakdown.
    float node = pow(max(0.5 + 0.5 * cos(phi + shimmer_phase), 0.0), 2.5);
    float diamonds = src * diamond_amp * (1.0 - 0.55 * m_ax) * node
                   * exp(-rn2 * radial_sharpness * 0.7) * 1.5;

    // Mach disk — bright perpendicular band at the normal shock.
    float dmd = (uv.x - mach_disk_x) / max(mach_disk_width, 1e-3);
    float machdisk = src * mach_disk_amp * exp(-dmd * dmd)
                   * exp(-rn2 * radial_sharpness * 0.8) * (1.0 - 0.6 * m_ax);

    // Incandescent nozzle exit — brilliant white-hot at x≈0, shaped to the
    // crisp beam (not a round bulb).
    float nozzle_glow = b_ax * core_brightness * litgate
                      * exp(-uv.x * uv.x * 240.0) * core_prof * 2.5;

    // Kelvin–Helmholtz shear vortices — high-frequency detail feathering the
    // edge (|rn| ~ 1), convecting downstream, growing with maturity.
    float shear = exp(-(abs(rn) - 1.0) * (abs(rn) - 1.0) * 6.0);
    float khn   = nano_fbm2(float2(uv.x * kh_scale - kh_phase,
                                   dy * kh_scale + kh_phase * 0.3), 3);
    float kh    = 1.0 + kh_amp * m_ax * shear * (khn - 0.5) * 2.0;

    // Crackle — fast whole-plume luminosity shimmer.
    float crackle = 1.0 + crackle_amp *
        (nano_hash21(float2(floor(uv.x * 40.0), floor(crackle_phase))) - 0.5);

    // The "hot" part (core + diamonds + nozzle) drives both colour temperature
    // and the white blow-out; the halo only adds dim blue body.
    float hot   = (core_beam + diamonds + machdisk) * kh * crackle + nozzle_glow;
    float total = max(hot + halo * kh, 0.0);

    // --- Colour: per-pixel temperature ramp white → orange → blue, like the
    // test-stand reference. On-axis near the nozzle the core is white-hot;
    // it cools to orange, and the wide lean supersonic body glows blue
    // (chemiluminescence — not blackbody). `mixture` pushes the body toward a
    // rich orange afterburner. ---
    float heat = saturate(hot * 1.2) * pow(saturate(1.0 - m_ax), 1.3);
    float3 flame_blue = float3(0.28, 0.42, 1.00);
    float3 flame_body = lerp(flame_blue, float3(1.00, 0.42, 0.10), mixture);
    float3 orange     = float3(1.00, 0.55, 0.16);
    float3 white_hot  = float3(1.00, 0.96, 0.88);
    float3 col = (heat > 0.55)
        ? lerp(orange, white_hot, saturate((heat - 0.55) / 0.45))
        : lerp(flame_body, orange, saturate(heat / 0.55));

    add += col * total * intensity;

    if (debug_show_axis != 0u && abs(uv.y - centerline_y) < (1.5 / float(h))) {
      add += float3(1.0, 0.0, 0.5);
    }
  }

  // --- Sparks — oriented streaks (low-frequency rim spray). Work in aspect-
  // corrected space so the dab is round, then stretch ALONG the screen-space
  // velocity so an arcing spark visibly rotates as it flies. ---
  uint ns = min(spark_count, 64u);
  for (uint k = 0u; k < ns; ++k) {
    Spark sp = sparks[k];
    if (sp.life <= 0.0) continue;
    float2 d  = float2(uv.x - sp.x, (uv.y - sp.y) * aspect);
    float2 vv = float2(sp.vx, sp.vy * aspect);
    float vlen = length(vv);
    float2 vn = (vlen > 1e-5) ? (vv / vlen) : float2(1.0, 0.0);
    float2 pn = float2(-vn.y, vn.x);
    float along  = dot(d, vn);
    float across = dot(d, pn);
    float rad = max(sp.size, 1e-4);
    float streak = 1.0 + saturate(vlen * 5.0) * 3.0;   // faster → longer
    float a = along / (rad * streak);
    float b = across / rad;
    float g = exp(-(a * a + b * b));
    // Warm ember, whiter when brighter (hot near the rim).
    float3 col = lerp(float3(1.0, 0.5, 0.18), float3(1.0, 0.85, 0.62), saturate(sp.life));
    add += g * sp.life * col;
  }

  // Soft exposure rolloff (1-e^-x) on the accumulated flame instead of a hard
  // clip: highlights roll smoothly through orange → white and colours stay
  // saturated in the shoulder — the "juicy" look. Added over the (preserved)
  // input, so add → 0 ⇒ output → base.
  float3 outc = base.rgb + (1.0 - exp(-add));
  outputTex[gid.xy] = float4(saturate(outc), base.a);
}
