// source.particles.flash_particles — pre-fill compute pass.
//
// Seeds the framebuffer-sized target with the upstream content the
// raster pass needs to blend on top of. Used twice with different
// PSOs:
//   color  — copies tex_in × input_alpha into tex_out (rgba8).
//   motion — copies upstream render_outputs/motion into motionTex
//            (rgba16f). For an unwired upstream the host binds a 1×1
//            zero fallback; out-of-bounds Loads return zero per
//            WebGPU spec, so the unconditional read is safe.
//
// `scale` is a per-channel multiplier (used for input_alpha on color;
// passed as 1.0 for motion). The same SPV is registered twice with
// different storage-tex format hints so each PSO gets the right
// substitution.

Texture2D<float4>   srcTex : register(t0);
RWTexture2D<float4> dstTex : register(u1);

cbuffer Uniforms : register(b2) {
  float4 scale;
};

[numthreads(8, 8, 1)]
void main(uint3 gid : SV_DispatchThreadID) {
  uint W, H;
  dstTex.GetDimensions(W, H);
  if (gid.x >= W || gid.y >= H) return;
  float4 src = srcTex.Load(int3(int(gid.x), int(gid.y), 0));
  dstTex[gid.xy] = src * scale;
}
