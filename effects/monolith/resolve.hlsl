// source.mesh.monolith — deferred resolve (one dispatch per copy round).
//
// Reads the round's G-buffer and shades the covered pixels: Schlick
// fresnel env reflection (equirect env_in or screen-space tex_in
// fallback), diffuse lambert (black diffuse still reads via the spec
// terms), screen-space refraction of the background, height + depth fog.
// Composites over `bgTex` (round 0: tex_in; later rounds: the previous
// composite — outer shells refract/blend over inner ones) into the
// RGBA16F ping-pong target. Alpha channel accumulates occlusion for the
// god-ray pass.
//
// PASSTHROUGH PURITY: uncovered pixels are copied VERBATIM (Load, no
// filtering, no fog) so an idle region round-trips the input exactly.

#include "caustics.hlsl"

Texture2D<float4>   gbufA    : register(t0);  // n.xyz, coverage
Texture2D<float4>   gbufB    : register(t1);  // world_y, view_z, world_x, world_z
Texture2D<float4>   bgTex    : register(t2);  // tex_in (seed) or comp[prev]
Texture2D<float4>   envSharp : register(t3);
Texture2D<float4>   envBlur  : register(t4);  // = envSharp when blur skipped
RWTexture2D<float4> outTex   : register(u5);  // comp[next], rgba16float
SamplerState        clampS   : register(s6);
SamplerState        wrapS    : register(s7);  // equirect u wrap
Texture2D<float4>   tintTex  : register(t9);  // sun tint source (1x1 zero when off)

cbuffer Uniforms : register(b8) {
  float4 sun_view;     // xyz dir toward light (view space, unit), w = sun intensity
  float4 sun_color;    // light tint rgb
  float4 cam;          // focal, cover_ax, cover_ay, phi (camera pitch)
  float4 material;     // reflect, roughness, refract, opacity
  float4 color_shade;  // diffuse rgb, shading amount
  float4 fog_p;        // fog_amount, fog_y0, inv_fog_h, fog_depth_k
  float4 round_p;      // copy_weight, is_seed, env_mode (1 = equirect), fog_z0
  float4 vp;           // w, h, 1/w, 1/h
  float4 caustic;      // amount, world scale, water_t, 0
  float4 sun_env;      // sun sample uv.xy, mode (1 = hue from env), 0
  float4 fog_color;    // medium tint rgb (white = neutral), w = inv zoom
};

static const float INV_TAU = 0.15915494309;
static const float INV_PI = 0.31830988618;

float2 equirectUV(float3 d) {
  float u = atan2(d.x, d.z) * INV_TAU + 0.5;
  float v = acos(clamp(d.y, -1.0, 1.0)) * INV_PI;
  return float2(u, clamp(v, 0.004, 0.996));
}

[numthreads(8, 8, 1)]
void main(uint3 gid : SV_DispatchThreadID) {
  uint W = (uint)vp.x, H = (uint)vp.y;
  if (gid.x >= W || gid.y >= H) return;
  int3 ip = int3(int(gid.x), int(gid.y), 0);

  float4 A = gbufA.Load(ip);
  float4 bg4 = bgTex.Load(ip);
  // Seed round: tex_in's own alpha is content, not occlusion.
  float occ = round_p.y > 0.5 ? 0.0 : bg4.a;
  float3 bg = bg4.rgb;
  if (A.a < 0.5) {
    outTex[gid.xy] = float4(bg, occ);
    return;
  }

  float4 B = gbufB.Load(ip);
  float world_y = B.x;
  float view_z = max(B.y, 0.1);
  float3 N = A.xyz;
  float nl = sqrt(dot(N, N));
  if (nl < 1e-5) {
    outTex[gid.xy] = float4(bg, occ);
    return;
  }
  N /= nl;

  const float focal = cam.x, ax = cam.y, ay = cam.z, phi = cam.w;
  float2 uv = (float2(gid.xy) + 0.5) * vp.zw;
  // Pixel -> y-up square NDC -> view ray (inverse of the CPU projection).
  float2 ndc = float2(uv.x * 2.0 - 1.0, 1.0 - uv.y * 2.0);
  float3 V = normalize(float3(ndc.x / (2.0 * ax * focal),
                              ndc.y / (2.0 * ay * focal), 1.0));

  float ndv = saturate(dot(N, -V));
  float F = material.x * (0.06 + 0.94 * pow(1.0 - ndv, 5.0));

  float3 sunD = sun_view.xyz;
  float sun_i = sun_view.w;
  // Effective sun tint: the authored color, optionally multiplied by the
  // hue of a texture at the sun's position (Sun Source: 1 = env/input,
  // 2 = the dedicated tint tex; 5-tap cross for stability — every pixel
  // samples the same point, cache-hot).
  float3 sun_col = sun_color.rgb;
  if (sun_env.z > 1.5) {
    float2 su = sun_env.xy;
    float3 e = tintTex.SampleLevel(clampS, su, 0).rgb
             + tintTex.SampleLevel(clampS, su + float2(0.02, 0.0), 0).rgb
             + tintTex.SampleLevel(clampS, su - float2(0.02, 0.0), 0).rgb
             + tintTex.SampleLevel(clampS, su + float2(0.0, 0.02), 0).rgb
             + tintTex.SampleLevel(clampS, su - float2(0.0, 0.02), 0).rgb;
    sun_col *= nano_sun_chroma(e * 0.2);
  } else if (sun_env.z > 0.5) {
    float2 su = sun_env.xy;
    float3 e;
    if (round_p.z > 0.5) {
      e = envBlur.SampleLevel(wrapS, su, 0).rgb
        + envBlur.SampleLevel(wrapS, su + float2(0.02, 0.0), 0).rgb
        + envBlur.SampleLevel(wrapS, su - float2(0.02, 0.0), 0).rgb
        + envBlur.SampleLevel(wrapS, su + float2(0.0, 0.02), 0).rgb
        + envBlur.SampleLevel(wrapS, su - float2(0.0, 0.02), 0).rgb;
    } else {
      e = envBlur.SampleLevel(clampS, su, 0).rgb
        + envBlur.SampleLevel(clampS, su + float2(0.02, 0.0), 0).rgb
        + envBlur.SampleLevel(clampS, su - float2(0.02, 0.0), 0).rgb
        + envBlur.SampleLevel(clampS, su + float2(0.0, 0.02), 0).rgb
        + envBlur.SampleLevel(clampS, su - float2(0.0, 0.02), 0).rgb;
    }
    sun_col *= nano_sun_chroma(e * 0.2);
  }

  float lam = saturate(dot(N, sunD));
  float shade = 1.0 + color_shade.w * ((0.30 + 0.70 * lam) - 1.0);
  float3 diffuse = color_shade.rgb * shade * sun_i * sun_col;

  // Environment reflection.
  float3 Rv = reflect(V, N);
  float cphi = cos(phi), sphi = sin(phi);
  float3 env;
  if (round_p.z > 0.5) {
    // Equirect env_in: rotate the reflected ray view -> world (Rx(-phi)).
    float3 Rw = float3(Rv.x, cphi * Rv.y + sphi * Rv.z,
                       -sphi * Rv.y + cphi * Rv.z);
    float2 uvE = equirectUV(Rw);
    env = lerp(envSharp.SampleLevel(wrapS, uvE, 0),
               envBlur.SampleLevel(wrapS, uvE, 0), material.y).rgb;
  } else {
    // Screen-space fallback: project the reflection into the input.
    float2 uvE = uv + float2(Rv.x, -Rv.y) * 0.35;
    env = lerp(envSharp.SampleLevel(clampS, uvE, 0),
               envBlur.SampleLevel(clampS, uvE, 0), material.y).rgb;
  }
  float3 glint = pow(saturate(dot(Rv, sunD)), 48.0) * sun_i * sun_col;

  // Water caustics: dapple projected from the SURFACE ABOVE — world-top,
  // independent of the sun (which may well sit "underwater"). The field
  // lives on the world XZ plane with a small fixed refraction slant (the
  // y shear — also keeps vertical faces from reading as flat stripes).
  // Weighted like projected irradiance: cosine falloff from above (tops
  // full, oblique faces attenuated), a faint grazing spill on near-
  // vertical walls, and NOTHING below horizontal — undersides are in
  // their own shadow. Rebalanced around 1.0 so the amount knob doesn't
  // shift mean brightness.
  if (caustic.x > 0.0) {
    float2 cp = float2(B.z + 0.35 * world_y, B.w - 0.27 * world_y) * caustic.y;
    float c = nano_caustic2(cp, caustic.z);
    float nw_y = cphi * N.y + sphi * N.z;   // world-space normal y
    float up = saturate(nw_y);
    float w_c = up * (0.45 + 0.55 * up);                       // cosine, mid-lifted
    w_c += 0.08 * saturate(nw_y * 5.0 + 1.0) * (1.0 - up);     // grazing spill, 0 by -0.2
    float dap = 1.0 + caustic.x * (c * 2.2 - 0.45) * w_c;
    diffuse *= dap;
    glint *= dap;
    env *= 1.0 + 0.35 * (dap - 1.0);
  }

  // Screen-space refraction: the background seen through the surface.
  float2 uvR = uv + float2(-N.x, N.y) * material.z * 0.12;
  float3 refr = bgTex.SampleLevel(clampS, uvR, 0).rgb;
  refr *= lerp(float3(1.0, 1.0, 1.0), color_shade.rgb, 0.35);

  float3 surf = lerp(refr, diffuse, material.w) + (env + glint) * F;

  // Atmosphere: a smooth bottom-to-top gradient (smoothstep spanning past
  // both ends of the body — no plateau, no knee) plus mild depth haze.
  // Fog color is the HEAVILY BLURRED env (guaranteed blurred whenever fog
  // is on), desaturated and milk-lifted: it must read as a scattering
  // medium, never as transparency toward the backdrop.
  if (fog_p.x > 0.0) {
    float fh = saturate((world_y - fog_p.y) * fog_p.z);
    fh = fh * fh * (3.0 - 2.0 * fh);
    float fd = 1.0 - exp(-max(0.0, view_z - round_p.w) * fog_p.w);
    float f = fog_p.x * saturate(fh + 0.6 * fd);
    // The scene sample feeding the medium is spatially ZOOMED about the
    // center (fog_color.w = inverse zoom): at 1 it aligns with the
    // backdrop, zoomed in it draws from the scene's middle — decoupling
    // the haze from whatever sits directly behind each pixel.
    float3 fogc;
    if (round_p.z > 0.5) {
      float3 Vw = float3(V.x, cphi * V.y + sphi * V.z,
                         -sphi * V.y + cphi * V.z);
      float2 uvF = 0.5 + (equirectUV(Vw) - 0.5) * fog_color.w;
      uvF.y = clamp(uvF.y, 0.004, 0.996);
      fogc = envBlur.SampleLevel(wrapS, uvF, 0).rgb;
    } else {
      float2 uvF = 0.5 + (uv - 0.5) * fog_color.w;
      fogc = envBlur.SampleLevel(clampS, uvF, 0).rgb;
    }
    float fl = dot(fogc, float3(0.299, 0.587, 0.114));
    fogc = lerp(fogc, float3(fl, fl, fl), 0.45);
    fogc = fogc * 0.85 + 0.10 * (0.4 + 0.6 * sun_i) *
           lerp(float3(1.0, 1.0, 1.0), sun_col, 0.5);
    fogc *= fog_color.rgb;   // authored medium tint (white = neutral)
    surf = lerp(surf, fogc, f);
  }

  float w = round_p.x;
  outTex[gid.xy] = float4(lerp(bg, surf, w), max(occ, w));
}
