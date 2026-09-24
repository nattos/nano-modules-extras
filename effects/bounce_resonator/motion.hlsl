// source.light.bounce_resonator — motion-vector pass.
//
// The diffusion network has no vertical motion (brightness sloshes, the
// band doesn't travel), so for now this is a clean passthrough of any
// upstream motion. Reserved for a future per-bar flux → motion mapping
// (value flowing j→i implies a horizontal drift).

Texture2D<float4>   upstreamTex : register(t0);    // upstream motion
RWTexture2D<float4> motionTex   : register(u1);    // rgba16f

cbuffer Uniforms : register(b2) {
  float band_val; float intensity; float input_opacity; float _pad;
};

[numthreads(8, 8, 1)]
void main(uint3 gid : SV_DispatchThreadID) {
  uint w, h;
  motionTex.GetDimensions(w, h);
  if (gid.x >= w || gid.y >= h) return;

  float4 upstream = upstreamTex[gid.xy];
  motionTex[gid.xy] = float4(upstream.xy, 0.0, 0.0);
}
