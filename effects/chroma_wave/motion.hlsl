// source.light.chroma_wave — motion-vector pass.
//
// The motion is the OPTICAL FLOW of the colour-band field
//     t(uv, time) = g·F + grade_phase + band_tilt·qy
// i.e. the velocity that advects the bands: v = −(∂t/∂time)·∇t / |∇t|². This
// captures everything that makes the bands move visually:
//   * the grade-phase scroll (the fold) — d(grade_phase)/dt,
//   * grade_freq F — denser bands move slower per ring (it's in ∇t and ∂t/∂t),
//   * the geometric expansion — ∂g/∂time = −v_geo·∇g with v_geo = rel·ṙ/r,
//   * band_tilt.
// The direction lands along the band gradient (crescent + anisotropy included,
// computed analytically), so it runs outward perpendicular to the rings. We
// weight by the SAME alpha the colour pass uses so the footprint and fade track
// the visible bloom, then blend over upstream motion by that mask.

Texture2D<float4>   upstreamTex : register(t0);   // upstream motion (uv/sec)
RWTexture2D<float4> motionTex   : register(u1);   // rgba16f
StructuredBuffer<float4> voices : register(t3);   // 4 float4 per voice

cbuffer Uniforms : register(b2) {
  float cres_off;
  float motion_scale;
  float alpha_gamma;
  float band_tilt;
  uint  voice_count;
  float motion_warp;
  float motion_edge_mask;
  float _p2;
};

static const float TAU = 6.28318530717958647692;

// Density field for one voice + its uv gradient ∇g (analytic), plus qy and the
// y scale invy (for the band_tilt term). Mirrors the colour pass exactly.
float cw_eval(uint vi, float2 uv, float asp, out float2 grad,
              out float qy_out, out float invy_out) {
  float4 a = voices[vi * 4u + 0u];   // cx, cy, radius, elong
  float4 b = voices[vi * 4u + 1u];   // ycomp, sharp, plateau_p, cres
  float cx = a.x, cy = a.y, radius = a.z, elong = a.w;
  float ycomp = b.x, sharp = b.y, p = b.z, cres = b.w;

  float2 rel = uv - float2(cx, cy);
  float invx = asp / max(radius * elong, 1e-5);
  float invy = 1.0 / max(radius * ycomp, 1e-5);
  float qx = rel.x * invx;
  float qy = rel.y * invy;
  qy_out = qy; invy_out = invy;

  float r2  = qx * qx + qy * qy;
  float qyu = qy + cres_off;
  float ru2 = qx * qx + qyu * qyu;

  float gm = exp(-pow(r2, p) * sharp);
  float gc = exp(-ru2 * sharp * 1.6);
  float graw = gm - cres * gc;

  float dm = sharp * p * pow(max(r2, 1e-12), p - 1.0);
  float2 dgm = -gm * dm * 2.0 * float2(qx * invx, qy * invy);
  float2 dgc = -gc * (sharp * 1.6) * 2.0 * float2(qx * invx, qyu * invy);
  float2 graw_grad = dgm - cres * dgc;

  grad = (graw <= 0.0 || graw >= 1.0) ? float2(0.0, 0.0) : graw_grad;
  return saturate(graw);
}

[numthreads(8, 8, 1)]
void main(uint3 gid : SV_DispatchThreadID) {
  uint W, H;
  motionTex.GetDimensions(W, H);
  if (gid.x >= W || gid.y >= H) return;

  float4 upstream = upstreamTex[gid.xy];
  if (voice_count == 0u) {
    motionTex[gid.xy] = float4(upstream.xy, 0.0, 0.0);
    return;
  }

  float asp = float(W) / float(H);
  float2 uv = (float2(gid.xy) + 0.5) / float2(W, H);

  float2 v_accum = float2(0.0, 0.0);
  float  w_accum = 0.0;
  for (uint vi = 0u; vi < voice_count; vi++) {
    float4 a = voices[vi * 4u + 0u];          // cx, cy, radius, elong
    float4 c = voices[vi * 4u + 2u];          // grade_freq, grade_phase, band_contrast, overlay_alpha
    float  F       = c.x;
    float  gphase  = c.y;
    float  bc      = c.z;                      // band_contrast (band visibility)
    float  overlay = c.w;
    float  growth  = voices[vi * 4u + 3u].y;  // ṙ/r
    float  dP      = voices[vi * 4u + 3u].z;  // d(grade_phase)/dt

    float2 grad; float qy, invy;
    float  g = cw_eval(vi, uv, asp, grad, qy, invy);
    float2 rel = uv - float2(a.x, a.y);

    // ∇t = F·∇g + band_tilt·∇qy   (∇qy = (0, invy)).
    float2 gradt = F * grad + float2(0.0, band_tilt * invy);
    float  gt2 = dot(gradt, gradt);

    // ∂t/∂time = F·∂g/∂t + ∂(grade_phase)/∂t + band_tilt·∂qy/∂t.
    // ∂g/∂t = −v_geo·∇g (geometric expansion, v_geo = rel·growth);
    // ∂qy/∂t ≈ −qy·growth (radius scaling). The fold term (dP) is the band
    // sweep — only VISIBLE in proportion to band_contrast, which also tames
    // the |∇t|→0 singularity as the blob shallows out at the tail.
    float dt_dtime = F * (-growth * dot(rel, grad)) + bc * dP + band_tilt * (-qy * growth);

    float2 vel = (gt2 > 1e-10) ? (-dt_dtime * gradt / gt2) : float2(0.0, 0.0);
    // Safety clamp for the flat plateau core (|∇t| → 0).
    float vl = length(vel);
    if (vl > 4.0) vel *= 4.0 / vl;

    float wt = overlay * pow(g, alpha_gamma);

    // Optionally isolate to the bands' OUTWARD (leading) edges: moving outward
    // t decreases, so the leading edge of each ring is where the band B =
    // ½+½cos(2π t) is falling, i.e. sin(2π t) < 0. max(0,−sin) is that edge.
    if (motion_edge_mask > 0.0) {
      float t = g * F + gphase + band_tilt * qy;
      float edge = saturate(-sin(TAU * t));
      wt *= lerp(1.0, edge, motion_edge_mask);
    }
    v_accum += vel * wt;
    w_accum += wt;
  }

  float mask = saturate(w_accum);
  float2 local = (w_accum > 1e-5) ? (v_accum / w_accum) : float2(0.0, 0.0);

  // Arbitrary wavefront warp: damp the lateral (x) spread so the analytic
  // fan reads as a coherent downward front (the blob always faces down here).
  local.x *= (1.0 - motion_warp);

  float2 mixed = lerp(upstream.xy, local * motion_scale, mask);
  motionTex[gid.xy] = float4(mixed, 0.0, 0.0);
}
