// source.sdf.plume_field — tracer simulation, resolve pass.
//
// Folds the frame's fixed-point deposits into the persistent overlay map
// (ping-pong: reads last frame's overlay, writes the next), then zeroes
// the deposit bins for the next step. One thread per overlay texel, so
// the read-then-clear is race-free.
//
// The overlay is the simulation's memory:
//   .r = accumulated height delta, normalized [-1, 1] (world scale is
//        applied downstream by the compose pass / height sampling),
//   .g = flow (traffic) density — decays much faster than the height, so
//        streamline trails read as motion, not as a permanent stain.
// A light 4-neighbor diffusion keeps carvings smooth (and keeps the
// field's slopes inside the Lipschitz budget the CPU assumes). Oct seams
// are ignored by the stencil — sub-texel error at the map border, same
// stance as the shell sampler.

RWStructuredBuffer<int> deposit     : register(u0);
Texture2D<float4>       overlayPrev : register(t1);
RWTexture2D<float4>     overlayNext : register(u2);

cbuffer ResolveUniforms : register(b3) {
  float res;       // overlay resolution
  float rate_h;    // height-deposit gain per frame
  float rate_f;    // flow-deposit gain per frame
  float blur;      // diffusion blend [0, ~0.3]

  float decay_h;   // height retention this frame (exp(-fade*dt))
  float decay_f;   // flow retention this frame
  float _pad0;
  float _pad1;
};

[numthreads(8, 8, 1)]
void main(uint3 gid : SV_DispatchThreadID) {
  int r = int(res);
  if (gid.x >= (uint)r || gid.y >= (uint)r) return;
  int2 pc = int2(gid.xy);

  int idx = (pc.y * r + pc.x) * 2;
  float dh = float(deposit[idx]) / 65536.0;
  float fl = float(deposit[idx + 1]) / 4096.0;
  deposit[idx] = 0;
  deposit[idx + 1] = 0;

  float4 prev = overlayPrev.Load(int3(pc, 0));
  // 4-neighbor diffusion (clamped at the map edge).
  int2 xm = int2(max(pc.x - 1, 0), pc.y);
  int2 xp = int2(min(pc.x + 1, r - 1), pc.y);
  int2 ym = int2(pc.x, max(pc.y - 1, 0));
  int2 yp = int2(pc.x, min(pc.y + 1, r - 1));
  float hn = (overlayPrev.Load(int3(xm, 0)).r + overlayPrev.Load(int3(xp, 0)).r +
              overlayPrev.Load(int3(ym, 0)).r + overlayPrev.Load(int3(yp, 0)).r)
             * 0.25;

  float h = lerp(prev.r, hn, blur);
  // Deposition tapers as the overlay grows: carving/building stays gentle,
  // approaching a soft ceiling instead of racing to the clamp.
  h = clamp((h + dh * rate_h * (1.0 - 0.6 * abs(h))) * decay_h, -1.0, 1.0);
  float f = saturate(prev.g * decay_f + fl * rate_f);

  overlayNext[pc] = float4(h, f, 0.0, 0.0);
}
