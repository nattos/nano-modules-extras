// filter.reconstruct.line — pass 6: reconstruction + composite (full).
//
// Reads the analysis textures and repaints crisp lines & points (box-AA, uniform
// size, energy-gain "4K-downsample" look), de-bands smooth gradients, gated by
// the pass-6 tests and hierarchically composited over the input. `strength`
// enters ONLY here (0 = identity). debug_view != 0 shows an internal classifier
// stage. (Faithful port of reconstruct.pass_reconstruct.)

#include "common.hlsl"
#include "nano_color.hlsl"   // nano_hsv_to_rgb (orientation-hue debug)

Texture2D<float4>   inputTex   : register(t0);
Texture2D<float4>   flankTex   : register(t1);   // sep_gauss(img, 0.7)   (bg/ring taps)
Texture2D<float4>   cmnTex     : register(t2);   // rgb 3x3 min
Texture2D<float4>   cmxTex     : register(t3);   // rgb 3x3 max
Texture2D<float4>   cstarTex   : register(t4);   // (c*, -, -, -)
Texture2D<float4>   s0Tex      : register(t5);   // smoothed (cos2t, sin2t, w_est, -)
Texture2D<float4>   s1Tex      : register(t6);   // (w_line_s, w_point_s, w_grad_s, ori_coh)
Texture2D<float4>   sdTex      : register(t7);   // (delta_shared, trust, -, -)
Texture2D<float4>   m2Tex      : register(t8);   // (dbx, dby, r_est, sigma_s)
Texture2D<float4>   m1Tex      : register(t9);   // (w_line, w_point, w_grad, polarity)
Texture2D<float4>   armsTex    : register(t10);  // blur(w_line_s, 2)  (crossing suppress)
Texture2D<float4>   wideTex    : register(t14);  // sep_gauss(img, 5.6) (deband + wide bg)
SamplerState        samp       : register(s12);
RWTexture2D<float4> outputTex  : register(u13);
// b11 rather than the b14 this sequence would otherwise land on: D3D11 gives
// a stage only 14 constant-buffer slots, b0..b13, while t-space has 128. So
// the cbuffer takes the low number and wideTex takes the high one. The flat
// numbering across letters is what the host binds by, so they must still be
// unique across ALL the resources here.
cbuffer Uniforms : register(b11) { LRUniforms u; };

float3 texBilinear(Texture2D<float4> t, float2 pix, float2 dim) {
  return t.SampleLevel(samp, (pix + 0.5) / dim, 0.0).rgb;
}

[numthreads(8, 8, 1)]
void main(uint3 gid : SV_DispatchThreadID) {
  uint w, h;
  outputTex.GetDimensions(w, h);
  if (gid.x >= w || gid.y >= h) return;
  float2 dim = float2(w, h);
  float2 pf  = float2(gid.xy);

  float4 cin = inputTex[gid.xy];
  float4 S0  = s0Tex[gid.xy];
  float4 S1  = s1Tex[gid.xy];
  float  cs  = max(cstarTex[gid.xy].r, 1e-4);
  float3 cmn_g = cmnTex[gid.xy].rgb, cmx_g = cmxTex[gid.xy].rgb;

  // ---- debug views (internal classifier stages) ----
  if (u.debug_view != 0u) {
    float3 dbg = cin.rgb;
    if (u.debug_view == 1u) {
      dbg = saturate(S1.xyz);
    } else if (u.debug_view == 2u) {
      dbg = (S0.z / max(u.max_width, 1e-3)).xxx * lr_smoothstep(0.15, 0.35, S1.x);
    } else if (u.debug_view == 3u) {
      float ang = atan2(S0.y, S0.x);
      dbg = nano_hsv_to_rgb(float3(frac(ang / 6.2831853 + 0.5), 1.0, saturate(S1.x)));
    } else if (u.debug_view == 4u) {
      float d = sdTex[gid.xy].x;
      float g = lr_smoothstep(0.15, 0.35, S1.x);
      dbg = float3(saturate(d), 0.0, saturate(-d)) * g + 0.05 * g;
    } else if (u.debug_view == 5u) {
      dbg = saturate(S1.w).xxx;
    }
    outputTex[gid.xy] = float4(saturate(dbg), cin.a);
    return;
  }

  // ---- line branch ----
  float c2 = S0.x, s2 = S0.y, west = S0.z;
  float nrm = sqrt(c2 * c2 + s2 * s2 + 1e-12);
  float c2n = c2 / nrm, s2n = s2 / nrm;
  float nx = sqrt(clamp(0.5 * (1.0 + c2n), 0.0, 1.0));
  float ny = sign(s2n) * sqrt(clamp(0.5 * (1.0 - c2n), 0.0, 1.0));
  float2 nrml = float2(nx, ny);

  float delta   = sdTex[gid.xy].x;
  float sigma_s = m2Tex[gid.xy].w;
  float2 pc = pf + delta * nrml;

  float f = clamp(west * 0.5 + 1.5 * sigma_s + 1.0, 2.0, 6.0);
  float3 bg_p = texBilinear(flankTex, pc + f * nrml, dim);
  float3 bg_m = texBilinear(flankTex, pc - f * nrml, dim);
  float3 c_center = texBilinear(inputTex, pc, dim);

  float d_signed = -delta;
  float t = saturate((d_signed + f) / (2.0 * f));
  float3 bg_d = lerp(bg_m, bg_p, t);
  float3 bg_0 = 0.5 * (bg_m + bg_p);

  float pull = u.retarget * (1.0 - lr_smoothstep(1.8, 3.0, west));
  float w_t  = max(west + (u.target_width - west) * pull, 0.75);
  float gain = clamp(0.9 * west / w_t, 1.0, 4.0);           // energy re-concentration
  float3 fg_raw = saturate(bg_0 + (c_center - bg_0) * gain);

  float3 cmn_c = texBilinear(cmnTex, pc, dim), cmx_c = texBilinear(cmxTex, pc, dim);
  float3 lo = min(cmn_g, cmn_c) - 2.0 * LSB;
  float3 hi = max(cmx_g, cmx_c) + 2.0 * LSB;
  float3 fg_loc = clamp(fg_raw, lo, hi);
  float3 fg = fg_loc + (fg_raw - fg_loc) * u.recover;

  // solidify: rescue stroke colour from the extreme-coverage sample ALONG the
  // stroke (dash/aliased rescue). Self-limiting — a uniform faint stroke has no
  // brighter neighbour, so gain_avail → 0 and fg is untouched. All candidates
  // are real observed pixels → no colour invention.
  if (u.solidify > 0.0) {
    float2 tang = float2(-ny, nx);
    float beta = 6.0 * m1Tex[gid.xy].w;                    // toward bright (pol>0) / dark
    const float ks[7] = { -4.0, -2.5, -1.0, 0.0, 1.0, 2.5, 4.0 };
    float3 acc = float3(0, 0, 0); float wsum = 1e-6;
    [unroll] for (int i = 0; i < 7; i++) {
      float3 sc = texBilinear(inputTex, pc + ks[i] * tang, dim);   // RAW: keep dot peaks
      float wt = exp(beta * lr_luma709(sc));
      acc += wt * sc; wsum += wt;
    }
    float3 stroke_col = acc / wsum;
    float gain_avail = lr_smoothstep(0.05, 0.35,
                         abs(lr_luma709(stroke_col) - lr_luma709(c_center)) / cs);
    fg = fg + (stroke_col - fg) * (u.solidify * gain_avail);
  }

  float a = lr_band_coverage(d_signed, w_t);
  float3 c_line = lerp(bg_d, fg, a);

  // ---- pass-6 line gates ----
  float yc = lr_luma709(c_center), yp = lr_luma709(bg_p), ym = lr_luma709(bg_m);
  float same = (yc - yp) * (yc - ym) / (cs * cs);
  float gate_ridge = lr_smoothstep(0.0, 0.02, same);
  float disparity = abs(yp - ym) / cs;
  float gate_flank = 1.0 - lr_smoothstep(FLANK0, FLANK1, disparity);
  float y_wide = lr_luma709(wideTex[gid.xy].rgb);           // = blur(luma,5.6), luma commutes
  float gate_bg = 1.0 - lr_smoothstep(0.25, 0.55, abs(0.5 * (yp + ym) - y_wide) / cs);
  float wl = S1.x * gate_ridge * gate_flank * gate_bg;

  // ---- point / blob branch ----
  float2 db = m2Tex[gid.xy].xy;
  float r_est = m2Tex[gid.xy].z;
  float2 pb = pf + db;
  float fb = r_est + 1.5;
  float3 ring = 0.25 * (texBilinear(flankTex, pb + float2(fb, 0), dim)
                      + texBilinear(flankTex, pb + float2(-fb, 0), dim)
                      + texBilinear(flankTex, pb + float2(0, fb), dim)
                      + texBilinear(flankTex, pb + float2(0, -fb), dim));
  float3 c_center_b = texBilinear(inputTex, pb, dim);
  float r_t = max(r_est + (u.point_radius - r_est) * u.retarget, 0.5);
  float gain_b = clamp(0.9 * (r_est / r_t) * (r_est / r_t), 1.0, 4.0);
  float3 fgb_raw = saturate(ring + (c_center_b - ring) * gain_b);
  float3 fgb_loc = clamp(fgb_raw, cmn_g - 2.0 * LSB, cmx_g + 2.0 * LSB);
  float3 fg_b = fgb_loc + (fgb_raw - fgb_loc) * u.recover;
  float r_pix = sqrt(dot(db, db) + 1e-12);
  float a_b = clamp(r_t - r_pix + 0.5, 0.0, 1.0) * saturate(3.14159 * r_t * r_t);
  float3 c_point = lerp(ring, fg_b, a_b);
  float gate_pt = lr_smoothstep(2.0 * LSB, 8.0 * LSB, abs(lr_luma709(c_center_b) - lr_luma709(ring)));
  float arms = clamp(1.5 * armsTex[gid.xy].x, 0.0, 1.0);    // suppress points amid line arms
  float wp = S1.y * gate_pt * (1.0 - arms);

  // ---- deband branch ----
  float3 wide = wideTex[gid.xy].rgb;
  float q = u.deband * 4.0 * LSB;
  float3 c_db = cin.rgb + clamp(wide - cin.rgb, -q, q);
  if (u.deband > 0.0) c_db += lr_ign(pf) * (0.5 * LSB);
  float wg = S1.z;

  // ---- hierarchical composite (renormalized if oversubscribed) ----
  wl = u.strength * lr_smoothstep(0.10, 0.45, wl) * lr_smoothstep(0.50, 0.80, S1.w);
  wp = u.strength * lr_smoothstep(0.10, 0.45, wp);
  wg = u.strength * wg;
  wp = wp * (1.0 - wl);
  wg = wg * (1.0 - wl) * (1.0 - wp);
  float total = wl + wp + wg;
  float scl = max(total, 1.0);
  wl /= scl; wp /= scl; wg /= scl;
  float wpass = 1.0 - wl - wp - wg;
  float3 outc = wl * c_line + wp * c_point + wg * c_db + wpass * cin.rgb;

  outputTex[gid.xy] = float4(saturate(outc), cin.a);
}
