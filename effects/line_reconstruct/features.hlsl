// filter.reconstruct.line — pass 4: features. ONE 3x3 gather over the scale
// stack yields gradient + Hessian at all detection scales; a softmax scale-blend
// picks the dominant scale (no argmax); the Steger steered ridge and the LoG
// blob give subpixel offsets, width, orientation and soft class confidences.
// (Faithful port of pipeline.pass_features.) Writes:
//   M0 = (cos2t, sin2t, w_est, delta)
//   M1 = (w_line, w_point, w_grad, polarity)
//   M2 = (dbx, dby, r_est, sigma_s)
//   M3 = (rho, 0, 0, 0)

#include "common.hlsl"

Texture2D<float4> y0Tex     : register(t0);   // sigma 0.7 (.r)
Texture2D<float4> y1Tex     : register(t1);   // sigma 1.4
Texture2D<float4> y2Tex     : register(t2);   // sigma 2.8
Texture2D<float4> y3Tex     : register(t3);   // sigma 5.6 (wide context, deband grad)
Texture2D<float4> cstarTex  : register(t4);   // (c*, -, -, -)
Texture2D<float4> tensorTex : register(t5);   // (kappa, junction, 0, 0)
Texture2D<float4> statsTex  : register(t6);   // (Y, min3, max3, c)

// Write-only rgba16f (format from the registerShaderSPV "rgba16float","write"
// hint; a [[vk::image_format]] pin would force read_write, forbidden for rgba16f).
RWTexture2D<float4> m0 : register(u7);
RWTexture2D<float4> m1 : register(u8);
RWTexture2D<float4> m2 : register(u9);
RWTexture2D<float4> m3 : register(u10);

cbuffer Uniforms : register(b11) { LRUniforms u; };

// 3x3 finite-difference gradient + Hessian on an (already smoothed) .r channel.
void derivs(Texture2D<float4> tex, int2 p, int2 hi,
            out float yx, out float yy_, out float yxx, out float yyy, out float yxy) {
  #define T(cx, cy) (tex[clamp(int2(cx, cy), int2(0, 0), hi)].r)
  float c  = T(p.x,     p.y);
  float e  = T(p.x + 1, p.y);
  float ww = T(p.x - 1, p.y);
  float s  = T(p.x,     p.y + 1);
  float n  = T(p.x,     p.y - 1);
  float se = T(p.x + 1, p.y + 1);
  float nw = T(p.x - 1, p.y - 1);
  float ne = T(p.x + 1, p.y - 1);
  float sw = T(p.x - 1, p.y + 1);
  #undef T
  yx  = 0.5 * (e - ww);
  yy_ = 0.5 * (s - n);
  yxx = e + ww - 2.0 * c;
  yyy = s + n - 2.0 * c;
  yxy = 0.25 * (se + nw - ne - sw);
}

[numthreads(8, 8, 1)]
void main(uint3 gid : SV_DispatchThreadID) {
  uint w, h;
  m0.GetDimensions(w, h);
  if (gid.x >= w || gid.y >= h) return;
  int2 p  = int2(gid.xy);
  int2 hi = int2(w - 1, h - 1);

  float cs = cstarTex[gid.xy].r;
  const float SIG[3]    = { LR_SIGMA0, LR_SIGMA1, LR_SIGMA2 };
  const float LN_SIG[3] = { log(LR_SIGMA0), log(LR_SIGMA1), log(LR_SIGMA2) };

  // Per-scale derivs.
  float DVyx[3], DVyy[3], DVyxx[3], DVyyy[3], DVyxy[3];
  derivs(y0Tex, p, hi, DVyx[0], DVyy[0], DVyxx[0], DVyyy[0], DVyxy[0]);
  derivs(y1Tex, p, hi, DVyx[1], DVyy[1], DVyxx[1], DVyyy[1], DVyxy[1]);
  derivs(y2Tex, p, hi, DVyx[2], DVyy[2], DVyxx[2], DVyyy[2], DVyxy[2]);

  // Per-scale rotation-invariant ridge / blob responses (Lindeberg gamma-norm).
  float rr[3], bb[3];
  [unroll] for (int i = 0; i < 3; i++) {
    float mS  = 0.5 * (DVyxx[i] + DVyyy[i]);
    float ddS = sqrt(0.25 * (DVyxx[i] - DVyyy[i]) * (DVyxx[i] - DVyyy[i])
                     + DVyxy[i] * DVyxy[i] + 1e-20);
    float abs_lnn = abs(mS) + ddS;
    rr[i] = pow(SIG[i], 1.5) * abs_lnn / cs;
    float blob_gate = clamp(1.0 - ddS / (abs(mS) + 1e-6), 0.0, 1.0);
    bb[i] = (SIG[i] * SIG[i]) * abs(2.0 * mS) * blob_gate / cs;
  }

  // Softmax scale-blend (no argmax): weights, blended sigma, blended response.
  float3 r_all = float3(rr[0], rr[1], rr[2]);
  float3 b_all = float3(bb[0], bb[1], bb[2]);
  float rmax = max(r_all.x, max(r_all.y, r_all.z));
  float bmax = max(b_all.x, max(b_all.y, b_all.z));
  float3 er = exp(BETA_SOFTMAX * r_all / (rmax + 1e-6));
  float3 eb = exp(BETA_SOFTMAX * b_all / (bmax + 1e-6));
  float3 pw_r = er / (er.x + er.y + er.z);
  float3 pw_b = eb / (eb.x + eb.y + eb.z);
  float3 lnsig = float3(LN_SIG[0], LN_SIG[1], LN_SIG[2]);
  float sigma_s = exp(dot(pw_r, lnsig));
  float sigma_b = exp(dot(pw_b, lnsig));
  float r_star = dot(pw_r, r_all);
  float b_star = dot(pw_b, b_all);

  // wblend: blend the k-th derivative across scales by the ridge / blob weights.
  #define WBL(pw, ARR) (pw.x * ARR[0] + pw.y * ARR[1] + pw.z * ARR[2])
  float yx  = WBL(pw_r, DVyx),  yy_ = WBL(pw_r, DVyy);
  float yxx = WBL(pw_r, DVyxx), yyy = WBL(pw_r, DVyyy), yxy = WBL(pw_r, DVyxy);

  // --- ridge branch: steer once (no cross-scale flips) ---
  float m  = 0.5 * (yxx + yyy);
  float dd = sqrt(0.25 * (yxx - yyy) * (yxx - yyy) + yxy * yxy + 1e-20);
  float sgn = (m < 0.0) ? -1.0 : 1.0;
  float lnn = m + sgn * dd;                          // max-magnitude eigenvalue, signed
  float cos2t = sgn * (yxx - yyy) / (2.0 * dd + 1e-12);   // double-angle of line NORMAL
  float sin2t = sgn * (2.0 * yxy) / (2.0 * dd + 1e-12);
  float nx = sqrt(clamp(0.5 * (1.0 + cos2t), 0.0, 1.0));
  float ny = sign(sin2t) * sqrt(clamp(0.5 * (1.0 - cos2t), 0.0, 1.0));
  float ln = nx * yx + ny * yy_;
  float denom = lnn + ((lnn < 0.0) ? -1e-6 : 1e-6);
  float delta = clamp(-ln / denom, -1.5 * sigma_s, 1.5 * sigma_s);
  float polarity = -sgn;                             // +1 bright line on dark bg
  float rho = (abs(lnn) * sigma_s) / (abs(lnn) * sigma_s + abs(ln) + 1e-5);
  float w_est = clamp(AW2 * sigma_s * sigma_s + ALPHA_W * sigma_s + BETA_W, 0.5, u.max_width);

  // --- blob branch: 2-D Newton offset ---
  float bx  = WBL(pw_b, DVyx),  by  = WBL(pw_b, DVyy);
  float bxx = WBL(pw_b, DVyxx), byy = WBL(pw_b, DVyyy), bxy = WBL(pw_b, DVyxy);
  #undef WBL
  float det = bxx * byy - bxy * bxy;
  det = (abs(det) < 1e-6) ? (1e-6 * ((det < 0.0) ? -1.0 : 1.0)) : det;
  float dbx = -(byy * bx - bxy * by) / det;
  float dby = -(-bxy * bx + bxx * by) / det;
  float dbn = sqrt(dbx * dbx + dby * dby + 1e-12);
  float dscale = min(1.5 * sigma_b / dbn, 1.0);
  dbx *= dscale; dby *= dscale;
  float r_est = clamp(ALPHA_B * sigma_b + BETA_B, 0.4, u.max_width);

  // --- soft classification, hierarchical residual allocation ---
  float kappa = tensorTex[gid.xy].r;
  float jn    = tensorTex[gid.xy].g;
  float abs_line = lr_smoothstep(ABS0, ABS1, abs(lnn) * pow(sigma_s, 1.5));
  float abs_blob = lr_smoothstep(ABS0, ABS1, b_star * cs);
  // 1-D structure evidence: tensor coherence OR Hessian anisotropy (kappa dies
  // inside wide strokes; aniso_h ~1 along a ridge interior).
  float aniso_h = clamp(2.0 * dd / (abs(m) + dd + 1e-9), 0.0, 1.0);
  float oned = clamp(kappa + aniso_h * lr_smoothstep(ABS0, ABS1, dd * pow(sigma_s, 1.5)), 0.0, 1.0);
  float conf_line  = lr_smoothstep(R0, R1, r_star) * oned * (1.0 - jn) * abs_line;
  float conf_point = lr_smoothstep(B0, B1, b_star)
                   * (1.0 - lr_smoothstep(0.45, 0.80, kappa)) * abs_blob;

  // deband candidates: fine contrast small, some wide-scale drift present.
  float g3x, g3y, g3xx, g3yy, g3xy;
  derivs(y3Tex, p, hi, g3x, g3y, g3xx, g3yy, g3xy);
  float g3 = sqrt(g3x * g3x + g3y * g3y + 1e-20);
  float statc = statsTex[gid.xy].a;
  float tol = (G0 + (G1 - G0 + 8.0 * u.deband)) * LSB;
  float conf_grad = (1.0 - lr_smoothstep(0.5 * tol, tol, statc))
                  * lr_smoothstep(0.10 * LSB, 0.35 * LSB, g3);

  // pure confidences (strength / sharpening / hierarchy applied once in pass 6).
  float w_line  = conf_line;
  float w_point = conf_point * (1.0 - conf_line);
  float w_grad  = u.deband * conf_grad;

  m0[gid.xy] = float4(cos2t, sin2t, w_est, delta);
  m1[gid.xy] = float4(w_line, w_point, w_grad, polarity);
  m2[gid.xy] = float4(dbx, dby, r_est, sigma_s);
  m3[gid.xy] = float4(rho, 0.0, 0.0, 0.0);
}
