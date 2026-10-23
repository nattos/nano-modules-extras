// source.mesh.lantern — primary visibility + surface shading (full res).
//
// Per pixel: nearest box along the camera ray (exact), then shade the hit:
//   - carved edges: rounded wear, chips and sheared corners cut into each
//     box near its edges, marched by the camera ray (ltn_carve_hit), so they
//     break up the silhouettes too
//   - two materials: a dark glossy SHELL (metal-flecked base + clear coat) and
//     banded STONE whose strata are laid out in the structure's frame, so
//     they run on across neighboring blocks and jutting blocks read as faults
//   - lights: the core (soft shadow from the atlas), two suns and the kicker
//     (hard shadow rays against the boxes), GGX base + coat lobes
//   - reflections: the reflected ray finds the next box (core-lit) and the
//     lit haze in front of it — the glossy blocks mostly read through these
//   - ember: blocks straining furthest out glow from within — stone in its
//     grooves (Ember), shell over the whole block (Shell Glow)
// Writes (linear radiance, hit distance) — LTN_MISS on a miss — for the fog
// march and the composite.

#include "common.hlsl"
#include "nano_hash.hlsl"
#include "nano_noise3.hlsl"

Texture2D<float4>   atlas    : register(t2);
RWTexture2D<float4> sceneOut : register(u3);

// egg_r*.w carry the core's plain color (for ember emission).
float3 ltn_core_color() { return float3(egg_r0.w, egg_r1.w, egg_r2.w); }

float3 ltn_to_egg(float3 p) {
  return float3(dot(egg_r0.xyz, p), dot(egg_r1.xyz, p), dot(egg_r2.xyz, p));
}

float ltn_ggx_d(float ndh, float a) {
  float a2 = a * a;
  float d = ndh * ndh * (a2 - 1.0) + 1.0;
  return a2 / (LTN_PI * d * d);
}

// Height-correlated Smith visibility (includes 1 / (4 ndl ndv)).
float ltn_ggx_vis(float ndl, float ndv, float a) {
  float a2 = a * a;
  float gv = ndl * sqrt(ndv * ndv * (1.0 - a2) + a2);
  float gl = ndv * sqrt(ndl * ndl * (1.0 - a2) + a2);
  return 0.5 / max(gv + gl, 1e-5);
}

float3 ltn_schlick(float3 f0, float c) {
  float k = pow(1.0 - saturate(c), 5.0);
  return f0 + (1.0 - f0) * k;
}

// With a grazing reflectance f90 below 1 (Albedo darkens the base layer
// all the way to grazing).
float3 ltn_schlick90(float3 f0, float f90, float c) {
  float k = pow(1.0 - saturate(c), 5.0);
  return f0 + (f90 - f0) * k;
}

// Blender Wave texture, BANDS along Z, SAW profile: period 2pi / (20 scale).
float ltn_saw_bands(float3 p, float scale, float distortion, float3 off) {
  float n = p.z * scale * 20.0;
  n += distortion * nano_gfbm3(p * scale * 1.4 + off, 3, 0.6);
  n *= 1.0 / (2.0 * LTN_PI);
  return n - floor(n);
}

// Same, SINE profile.
float ltn_sine_bands(float3 p, float scale, float distortion, float3 off) {
  float n = p.z * scale * 20.0;
  n += distortion * nano_gfbm3(p * scale + off, 3, 0.5);
  return 0.5 + 0.5 * sin(n);
}

float3 ltn_stone_ramp(float x) {
  // Slate, umber, slate, a pale band, warm gray (the scene's color ramp).
  float3 c0 = float3(0.022, 0.024, 0.027);
  float3 c1 = float3(0.05, 0.036, 0.024);
  float3 c2 = float3(0.028, 0.03, 0.033);
  float3 c3 = float3(0.17, 0.155, 0.135);
  float3 c4 = float3(0.06, 0.055, 0.05);
  if (x < 0.35) return lerp(c0, c1, x / 0.35);
  if (x < 0.6)  return lerp(c1, c2, (x - 0.35) / 0.25);
  if (x < 0.82) return lerp(c2, c3, (x - 0.6) / 0.22);
  return lerp(c3, c4, (x - 0.82) / 0.18);
}

// --- Carved edges (camera rays only) ------------------------------------------
//
// The scene's blocks are finely divided boxes whose faces are pushed in near
// their edges (part_structure.py _carve): rounded WEAR, chipped scoops and
// sheared-off corners, all varying along each edge, roughened where cut. As
// a solid: a point is inside when, for every face, its depth below that face
// exceeds the face's cut there (which depends on its distance to the face's
// two edges) — ltn_carve_f >= 0. The camera ray marches that near the edges,
// so the cuts show in silhouette (a ray through a notch goes on to the next
// box); flat faces away from every edge hit at once. Shadows, the fog and
// reflections still see plain boxes.

struct LtnCarve {
  float3 h;      // half extents
  float3 seed;
  float  size;   // carving shrinks on small pieces (fragments)
  float  corner; // this octant's corner shear depth (0: none)
  float  rough;  // fracture roughness amplitude (faded where a pixel can't resolve it)
  int    key;
};

LtnCarve ltn_carve_setup(int box) {
  LtnCarve k;
  k.h = float3(boxes[box * LTN_BOX_ROWS + 1].w, boxes[box * LTN_BOX_ROWS + 2].w,
               boxes[box * LTN_BOX_ROWS + 3].w);
  float s = boxes[box * LTN_BOX_ROWS + 4].y;
  k.seed = frac(s * float3(1.0, 57.31, 113.77)) * 200.0 - 100.0;
  k.size = min(1.0, min(min(k.h.x, k.h.y), k.h.z) * 2.0 / 0.95);
  k.corner = 0.0;
  k.rough = 0.25;
  k.key = (int)(s * 65521.0);
  return k;
}

// Corner shear depth for the octant `sp` (each corner: a 50% chance).
float ltn_carve_corner(LtnCarve k, float3 sp) {
  int3 si = int3(sp) + k.key;
  float roll = nano_hash31i(si);
  float depth = nano_hash31i(si * 7 + 13);
  return roll < 0.5 * surf.y ? lerp(0.1, 0.32, depth) * k.size * k.size : 0.0;
}

// How far a face is cut at distance u from an edge whose wear and chip depth
// are `wear`, `chip`: a quarter-circle round, and a concave scoop with a
// crisp lip where it meets the face.
float ltn_edge_profile(float u, float wear, float chip) {
  float cut = 0.0;
  if (u < wear) cut = wear - sqrt(max(wear * wear - (wear - u) * (wear - u), 0.0));
  float width = chip * 1.6;
  if (u < width) cut = max(cut, chip * pow(1.0 - u / width, 1.6));
  return cut;
}

// Inside-ness of a point in the box's frame (>= 0 inside), in metres.
float ltn_carve_f(LtnCarve k, float3 lp, float reach) {
  float3 d = k.h - abs(lp);
  float3 sp = select(lp >= 0.0, 1.0, -1.0);
  // Each edge of this octant (edge e runs along axis e): its wear and chip,
  // from noise at the nearest point ON the edge, so both its faces agree.
  float wear[3], chip[3];
  float chance = 0.22 * surf.y;
  [unroll] for (int e = 0; e < 3; e++) {
    int f1 = (e + 1) % 3, f2 = (e + 2) % 3;
    wear[e] = 0.0;
    chip[e] = 0.0;
    if (d[f1] < reach && d[f2] < reach) {
      float3 ep = lp;
      ep[f1] = sp[f1] * k.h[f1];
      ep[f2] = sp[f2] * k.h[f2];
      wear[e] = shade.x * k.size * smoothstep(0.12, 0.6, nano_gnoise3(ep * 2.2 + k.seed));
      // Chips: one cell in four-and-a-half per metre along the edge may be
      // bitten; its depth tapers to nothing toward the cell's ends, so a
      // bite is a lens-shaped scoop rather than a square notch.
      float3 cp = ep * 4.5 + k.seed;
      float c = nano_hash31i(int3(floor(cp)));
      float along = frac(cp[e]);
      float taper = smoothstep(0.0, 0.4, along) * smoothstep(1.0, 0.6, along);
      if (c < chance) chip[e] = lerp(0.04, 0.15, c / chance) * k.size * sqrt(taper);
    }
  }
  float rough = 1.0 + k.rough * nano_gnoise3(lp * 12.0 + k.seed);
  float F = 1.0e9;
  [unroll] for (int a = 0; a < 3; a++) {
    int b = (a + 1) % 3, c = (a + 2) % 3;
    // Face a meets face b along the edge running along c, and face c along b.
    float cut = max(max(ltn_edge_profile(d[b], wear[c], chip[c]),
                        ltn_edge_profile(d[c], wear[b], chip[b])),
                    k.corner - (d[b] + d[c]));
    F = min(F, d[a] - max(cut, 0.0) * rough);
  }
  return F;
}

// March the camera ray through box `box` (entered at tn, left at tf).
// On a hit: t, and the outward normal in the box's frame (the entry face's
// on a flat stretch). `pa` is a pixel's angular size.
bool ltn_carve_hit(int box, float3 ro, float3 rd, float tn, float tf, int ax, float3 ld,
                   float pa, out float t, out float3 nlocal) {
  LtnCarve k = ltn_carve_setup(box);
  float4 r0 = boxes[box * LTN_BOX_ROWS + 0];
  float3 o = ro - r0.xyz;
  float3 r1 = boxes[box * LTN_BOX_ROWS + 1].xyz, r2 = boxes[box * LTN_BOX_ROWS + 2].xyz,
         r3 = boxes[box * LTN_BOX_ROWS + 3].xyz;
  float3 lo = float3(dot(o, r1), dot(o, r2), dot(o, r3));
  float3 lr = float3(dot(rd, r1), dot(rd, r2), dot(rd, r3));
  t = max(tn, 0.0);
  nlocal = float3(0.0, 0.0, 0.0);
  nlocal[ax] = ld[ax] > 0.0 ? -1.0 : 1.0;

  float reach = max(shade.x, surf.y > 0.0 ? 0.32 : 0.0) * k.size;
  float3 lp = lo + lr * t;
  float3 d = k.h - abs(lp);
  // Away from every edge (the entry face's two other depths both beyond
  // reach): a flat face, hit at entry.
  if (reach <= 0.0 || (d[(ax + 1) % 3] > reach && d[(ax + 2) % 3] > reach)) return true;
  // Fracture roughness (8 cm across — the scene's 35/m only survives its
  // 1.6 cm mesh as a gentle unevenness), where its bumps span 4-12+ pixels;
  // finer, it would only sparkle.
  k.rough = 0.2 * saturate((0.021 - t * pa) / 0.014);

  // March from the entry point (where F = -(the entry face's cut there):
  // nothing cut means a plain face), then refine between the last sample
  // outside and the first inside.
  float tp = t;
  bool hit = false;
  [loop] for (int i = 0; i < 49; i++) {
    lp = lo + lr * t;
    k.corner = ltn_carve_corner(k, select(lp >= 0.0, 1.0, -1.0));
    float F = ltn_carve_f(k, lp, reach);
    if (i == 0 && F >= -1e-4) return true;
    if (F >= 0.0) { hit = true; break; }
    tp = t;
    t += clamp(-F * 0.7, 0.002, 0.05);
    if (t > tf) return false;
  }
  if (!hit) return false;
  float ta = tp, tb = t;
  [loop] for (int j = 0; j < 5; j++) {
    float tm = 0.5 * (ta + tb);
    float3 q = lo + lr * tm;
    k.corner = ltn_carve_corner(k, select(q >= 0.0, 1.0, -1.0));
    if (ltn_carve_f(k, q, reach) >= 0.0) tb = tm; else ta = tm;
  }
  t = tb;
  lp = lo + lr * t;

  // The carved surface's normal: the field's gradient (outward = falling).
  k.corner = ltn_carve_corner(k, select(lp >= 0.0, 1.0, -1.0));
  const float e = 0.004;
  float3 g = float3(0.0, 0.0, 0.0);
  [loop] for (int m = 0; m < 6; m++) {
    float3 off = float3(0.0, 0.0, 0.0);
    off[m >> 1] = (m & 1) ? -e : e;
    g[m >> 1] += ((m & 1) ? -1.0 : 1.0) * ltn_carve_f(k, lp + off, reach);
  }
  if (dot(g, g) > 1e-12) nlocal = -normalize(g);
  return true;
}

struct LtnMat {
  float3 diff;      // diffuse albedo
  float3 f0;        // base specular reflectance
  float  f90;       // ... and at grazing
  float  rough;     // base GGX roughness (perceptual)
  float  coat;      // clear coat weight
  float  coat_rough;
  float3 emit;      // radiance
};

LtnMat ltn_material(int box, float3 p) {
  float4 r0 = boxes[box * LTN_BOX_ROWS + 0];
  float4 r4 = boxes[box * LTN_BOX_ROWS + 4];
  float glow = r0.w;
  LtnMat m;
  if (r4.x < 0.5) {
    // Shell: dark, glossy, stoic.
    float3 base = lerp(float3(0.016, 0.015, 0.015), float3(0.04, 0.036, 0.034), r4.y);
    const float metal = 0.35;
    m.diff = base * (1.0 - metal);
    m.f0 = lerp(float3(0.04, 0.04, 0.04), base, metal);
    m.rough = 0.2;
    m.coat = 0.6;
    m.coat_rough = 0.05;
    m.emit = ltn_core_color() * (glow * glow * glow * 0.15 * surf.x);
  } else {
    // Stone: folded strata in the structure's frame.
    float3 q = ltn_to_egg(p);
    float3 fold = float3(nano_gnoise3(q * 0.35),
                         nano_gnoise3(q * 0.35 + float3(17.3, 5.1, 9.7)),
                         nano_gnoise3(q * 0.35 + float3(41.9, 23.3, 3.1)));
    float3 f = q + 0.45 * fold;
    float major = ltn_saw_bands(f, 0.55, 1.5, float3(3.0, 1.0, 7.0));
    float fine = ltn_saw_bands(f, 6.0, 3.0, float3(11.0, 2.0, 5.0));
    float groove = smoothstep(0.9, 0.995,
                              ltn_sine_bands(f, 2.0, 4.0, float3(1.0, 9.0, 4.0)));
    float mask = saturate((0.5 + 0.5 * nano_gnoise3(f * 1.2) - 0.4) / 0.2);
    float crack = groove * mask;
    float band = frac(fine * 0.5 + major);
    float3 alb = ltn_stone_ramp(band) * lerp(1.0, 0.15, crack);
    m.diff = alb;
    m.f0 = float3(0.04, 0.04, 0.04);
    m.rough = 0.65;
    m.coat = 0.15;
    m.coat_rough = 0.3;
    m.emit = ltn_core_color() * (crack * glow * glow * 1.5 * shade.z);
  }
  // Albedo: scales everything the surface reflects (diffuse, base specular,
  // the coat), not what it emits.
  m.diff *= surf.z;
  m.f0 *= surf.z;
  m.coat *= surf.z;
  m.f90 = min(surf.z, 1.0);
  return m;
}

// One light's contribution: irradiance E (already * n.l and shadowed) from
// direction l. `widen` grows the lobes for large or near lights.
float3 ltn_light(LtnMat m, float3 n, float3 v, float3 l, float3 E, float widen,
                 float coat_keep) {
  float ndl = saturate(dot(n, l));
  if (ndl <= 0.0 || all(E <= 0.0)) return float3(0.0, 0.0, 0.0);
  float ndv = max(dot(n, v), 1e-4);
  float3 h = normalize(l + v);
  float ndh = saturate(dot(n, h));
  float vdh = saturate(dot(v, h));
  float a = saturate(m.rough * m.rough + widen);
  float ac = saturate(m.coat_rough * m.coat_rough + widen);
  float3 base = m.diff / LTN_PI +
                ltn_ggx_d(ndh, a) * ltn_ggx_vis(ndl, ndv, a) * ltn_schlick90(m.f0, m.f90, vdh);
  float coat = m.coat * ltn_ggx_d(ndh, ac) * ltn_ggx_vis(ndl, ndv, ac) *
               ltn_schlick(float3(0.04, 0.04, 0.04), vdh).x;
  return E * (base * coat_keep + coat);
}

// The core as a SPHERE light (Karis 2013): diffuse from its center with the
// horizon wrapped by its angular size, specular toward the point on the
// sphere nearest the reflected ray, its lobe widened by the sphere and
// renormalized — so glossy faces mirror a broad disk of light, even faces
// turned almost edge-on to the core.
float3 ltn_sphere_light(LtnMat m, float3 n, float3 v, float3 to_c, float3 E0,
                        float radius, float coat_keep) {
  float r2 = dot(to_c, to_c);
  float dist = sqrt(r2);
  float3 lc = to_c / dist;
  float sin_a = saturate(radius / dist);
  float ndl_c = dot(n, lc);
  float wrap = saturate((ndl_c + sin_a) / (1.0 + sin_a));
  float3 col = E0 * (wrap * wrap) * m.diff / LTN_PI * coat_keep;

  float3 r = reflect(-v, n);
  float3 c2r = dot(to_c, r) * r - to_c;
  float3 lp = to_c + c2r * saturate(radius / max(length(c2r), 1e-5));
  float3 l = normalize(lp);
  float ndl = saturate(dot(n, l));
  if (ndl <= 0.0) return col;
  float ndv = max(dot(n, v), 1e-4);
  float3 h = normalize(l + v);
  float ndh = saturate(dot(n, h));
  float vdh = saturate(dot(v, h));
  float grow = radius / (2.0 * dist);
  float a = max(m.rough * m.rough, 0.002);
  float ac = max(m.coat_rough * m.coat_rough, 0.002);
  float a2 = saturate(a + grow), ac2 = saturate(ac + grow);
  float norm_b = (a / a2) * (a / a2), norm_c = (ac / ac2) * (ac / ac2);
  float3 spec = ltn_ggx_d(ndh, a) * norm_b * ltn_ggx_vis(ndl, ndv, a) *
                ltn_schlick90(m.f0, m.f90, vdh) * coat_keep;
  float coat = m.coat * ltn_ggx_d(ndh, ac) * norm_c * ltn_ggx_vis(ndl, ndv, ac) *
               ltn_schlick(float3(0.04, 0.04, 0.04), vdh).x;
  return col + E0 * ndl * (spec + coat);
}

// What a reflected ray sees: the core-lit haze along it, then the next box.
float3 ltn_reflection(float3 o, float3 r) {
  float t;
  int ax;
  float3 ld;
  int hit = ltn_trace(o, r, 1e-3, 40.0, t, ax, ld);
  int ns = (int)core_i.w;
  float3 acc = float3(0.0, 0.0, 0.0);
  const float ts[4] = { 0.5, 1.6, 3.6, 7.5 };
  const float dts[4] = { 1.0, 1.2, 2.8, 5.0 };
  [unroll] for (int j = 0; j < 4; j++) {
    if (ts[j] >= t) break;
    float3 q = o + r * ts[j];
    float3 dq = q - core_p.xyz;
    float3 sig = water_s.rgb + ltn_halo_density(q, 0.5) * float3(0.9, 0.9, 0.88);
    float vis = ltn_core_vis(atlas, j % ns, q);
    acc += sig * (dts[j] / (4.0 * LTN_PI)) * ltn_core_irr_vol(dot(dq, dq)) * vis;
  }
  if (hit >= 0) {
    float3 q = o + r * t;
    float3 axes[3] = { boxes[hit * LTN_BOX_ROWS + 1].xyz,
                       boxes[hit * LTN_BOX_ROWS + 2].xyz,
                       boxes[hit * LTN_BOX_ROWS + 3].xyz };
    float3 nh = axes[ax] * (ld[ax] > 0.0 ? -1.0 : 1.0);
    float3 dq = core_p.xyz - q;
    float r2 = dot(dq, dq);
    float vis = ltn_core_vis(atlas, 0, q + nh * 0.01);
    float3 E = ltn_core_irr(r2) * saturate(dot(nh, dq * rsqrt(r2))) * vis;
    acc += 0.05 / LTN_PI * E;
  }
  return acc;
}

[numthreads(8, 8, 1)]
void main(uint3 gid : SV_DispatchThreadID) {
  uint W = (uint)vp.x, H = (uint)vp.y;
  if (gid.x >= W || gid.y >= H) return;

  float3 ro, rd;
  ltn_camera_ray(float2(gid.xy), vp.xy, ro, rd);

  // Nearest box whose carved shape the ray hits (a ray through a carved
  // notch goes on to the boxes behind).
  float t = LTN_MISS;
  int ax = 0;
  float3 ld = float3(0.0, 0.0, 1.0);
  float3 nlocal = float3(0.0, 0.0, 1.0);
  int box = -1;
  {
    float pa = 1.0 / (vp.y * cam_p.z * cam_p.x);   // a pixel's angle (rad)
    float tmin = 0.0;
    [loop] for (int it = 0; it < 4; it++) {
      float tn, tf;
      int bax;
      float3 bld;
      int bx = ltn_trace(ro, rd, tmin, LTN_MISS, tn, bax, bld);
      if (bx < 0) break;
      int ax2;
      float3 ld2;
      float tn2;
      ltn_box_ray(bx, ro, rd, tn2, tf, ax2, ld2);
      float th;
      float3 nl;
      if (ltn_carve_hit(bx, ro, rd, tn, tf, bax, bld, pa, th, nl)) {
        box = bx; t = th; ax = bax; ld = bld; nlocal = nl;
        break;
      }
      tmin = tn + 1e-4;
    }
  }

  // The incandescent heart: a small emissive ball at the core, seen only by
  // the camera (it lights nothing — the core light does that).
  {
    float hr = 0.33 * core_p.w;
    float3 oc = ro - core_p.xyz;
    float b = dot(oc, rd);
    float disc = b * b - (dot(oc, oc) - hr * hr);
    if (disc > 0.0) {
      float th = -b - sqrt(disc);
      if (th > 0.0 && th < t) {
        float3 hn = normalize(oc + rd * th);
        float limb = 0.75 + 0.25 * saturate(dot(hn, -rd));
        sceneOut[gid.xy] = float4(heart.rgb * limb, th);
        return;
      }
    }
  }
  if (box < 0) {
    sceneOut[gid.xy] = float4(0.0, 0.0, 0.0, LTN_MISS);
    return;
  }

  float4 r0 = boxes[box * LTN_BOX_ROWS + 0];
  float4 r1 = boxes[box * LTN_BOX_ROWS + 1];
  float4 r2 = boxes[box * LTN_BOX_ROWS + 2];
  float4 r3 = boxes[box * LTN_BOX_ROWS + 3];
  float4 r4 = boxes[box * LTN_BOX_ROWS + 4];
  float3 axes[3] = { r1.xyz, r2.xyz, r3.xyz };
  float3 half_ext = float3(r1.w, r2.w, r3.w);

  float3 p = ro + rd * t;
  float3 o = p - r0.xyz;
  float3 lp = float3(dot(o, r1.xyz), dot(o, r2.xyz), dot(o, r3.xyz));

  float3 n = normalize(axes[0] * nlocal.x + axes[1] * nlocal.y + axes[2] * nlocal.z);
  float3 v = -rd;
  // Debug view Glint (8): each block flat gray, red as far as its glint
  // charge goes (1 = full red; past it, brighter).
  if (shade.w > 7.5 && shade.w < 8.5) {
    float gc = boxes[box * LTN_BOX_ROWS + 5].x;
    float3 base = float3(0.12, 0.12, 0.13) * (0.35 + 0.65 * saturate(dot(n, v)));
    sceneOut[gid.xy] = float4(lerp(base, float3(1.0, 0.12, 0.04), saturate(gc)) * max(gc, 1.0), t);
    return;
  }
  // Lighting is looked up from just outside the PLAIN box (what the shadow
  // maps and shadow rays see): a carved point sits inside it, so step out
  // through its nearest face.
  float3 dep = half_ext - abs(lp);
  int fa = dep.x < dep.y ? (dep.x < dep.z ? 0 : 2) : (dep.y < dep.z ? 1 : 2);
  float3 ng = axes[fa] * (lp[fa] >= 0.0 ? 1.0 : -1.0);

  LtnMat m = ltn_material(box, p);
  float ndv = max(dot(n, v), 1e-4);
  float coat_f = m.coat * ltn_schlick(float3(0.04, 0.04, 0.04), ndv).x;
  float coat_keep = 1.0 - coat_f;
  float3 pl = p + ng * (dep[fa] + 0.01);

  float3 col = m.emit;

  // Core: soft shadow from every light sample.
  {
    int ns = (int)core_i.w;
    float vis = 0.0;
    [loop] for (int k = 0; k < ns; k++) vis += ltn_core_vis(atlas, k, pl);
    vis /= float(max(ns, 1));
    float3 dl = core_p.xyz - p;
    float3 E0 = ltn_core_irr(dot(dl, dl)) * vis;
    col += ltn_sphere_light(m, n, v, dl, E0, core_p.w, coat_keep);
  }
  // Ambient sun (soft) and the rim sun.
  {
    float vis = ltn_visible(pl, amb_d.xyz, 200.0);
    float3 E = amb_e.rgb * saturate(dot(n, amb_d.xyz)) * lerp(0.35, 1.0, vis);
    col += ltn_light(m, n, v, amb_d.xyz, E, 0.35, coat_keep);
  }
  {
    float vis = ltn_visible(pl, rim_d.xyz, 200.0);
    float3 E = rim_e.rgb * saturate(dot(n, rim_d.xyz)) * vis;
    col += ltn_light(m, n, v, rim_d.xyz, E, 0.0, coat_keep);
  }
  // Kicker: a large disk behind, left and above.
  {
    float3 dl = kick_p.xyz - p;
    float d2 = dot(dl, dl);
    float d = sqrt(d2);
    float3 l = dl / d;
    float vis = ltn_visible(pl, l, d);
    float3 E = kick_i.rgb * saturate(dot(kick_d.xyz, -l)) * saturate(dot(n, l)) /
               d2 * vis;
    col += ltn_light(m, n, v, l, E, 0.5 * kick_p.w / d, coat_keep);
  }
  // Reflections.
  if (shade.y > 0.0) {
    float3 r = reflect(rd, n);
    float3 env = ltn_reflection(pl, r);
    float3 fb = ltn_schlick90(m.f0, m.f90, ndv) * saturate(1.0 - m.rough * 1.4);
    col += shade.y * env * (coat_f + coat_keep * fb);
  }

  sceneOut[gid.xy] = float4(col, t);
}
