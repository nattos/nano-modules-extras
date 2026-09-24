// source.light.side_jet — Stage 2b: motion-vector emission.
//
// The analytic flow field IS the motion field — far better than a single
// "head velocity" because u(x) varies along the axis (fast at the nozzle,
// decaying downstream), so downstream motion.blur streaks with the
// correct gradient. Per pixel inside the plume we emit (u * motion_scale, 0)
// weighted by the same radial envelope the colour pass uses, blended over
// any upstream motion.

Texture2D<float4>   upstreamTex : register(t0);
RWTexture2D<float4> motionTex   : register(u1);

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

[numthreads(8, 8, 1)]
void main(uint3 gid : SV_DispatchThreadID) {
  uint w, h;
  motionTex.GetDimensions(w, h);
  if (gid.x >= w || gid.y >= h) return;

  float2 uv = (float2(gid.xy) + 0.5) / float2(w, h);
  float4 upstream = upstreamTex[gid.xy];

  // Match color.hlsl's zoom transform so motion stays registered with pixels.
  uv = float2(uv.x / max(zoom, 1e-3), (uv.y - 0.5) / max(zoom, 1e-3) + 0.5);

  uint nc = max(cell_count, 1u);
  float fidx = saturate(uv.x) * (float)(nc - 1u);
  uint i0 = (uint)floor(fidx);
  uint i1 = min(i0 + 1u, nc - 1u);
  float fr = fidx - (float)i0;
  Cell a = cells[i0];
  Cell c = cells[i1];
  float u_ax = lerp(a.u,   c.u,   fr);
  float m_ax = lerp(a.m,   c.m,   fr);
  float b_ax = lerp(a.b,   c.b,   fr);
  float lit  = lerp(a.lit, c.lit, fr);

  // Geometry must match color.hlsl so motion covers the whole visible beam —
  // including the top/bottom edges, not just the tight core.
  float dy = uv.y - centerline_y;
  float hw = nozzle_radius * (1.0 - 0.45 * smoothstep(0.0, 0.7, uv.x))
           + spread * uv.x * 0.4 * m_ax;
  float rn = dy / max(hw, 1e-4);
  float rn2 = rn * rn;

  float weight = 0.0;
  float2 local = float2(0.0, 0.0);
  if (abs(rn) < 2.5 && lit > 0.05 && b_ax > 1e-4) {
    // Cover the full beam: flat-top core OR the wider body profile.
    float cover = max(exp(-pow(rn2, 2.0) * radial_sharpness * 0.4),
                      exp(-rn2 * radial_sharpness * 0.42));
    weight = cover * smoothstep(0.0, 0.12, lit) * saturate(b_ax * 4.0);
    // Mostly downstream, with a little outward spread at the edges so the
    // shear layer streaks correctly. Scale by zoom — magnified features move
    // proportionally faster in screen space.
    float ms = motion_scale * zoom;
    local = float2(u_ax * ms, sign(dy) * u_ax * ms * 0.12 * saturate(abs(rn)));
  }

  float mask = saturate(weight);
  float2 mixed = lerp(upstream.xy, local, mask);
  motionTex[gid.xy] = float4(mixed, 0.0, 0.0);
}
