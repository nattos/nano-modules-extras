// triangulate — clear the JFA id texture to -1 (empty) before splatting seeds.
// r32f read_write storage (WebGPU rejects r32f as write-only / sampled).
[[vk::image_format("r32f")]] RWTexture2D<float> idTex : register(u0);

[numthreads(8, 8, 1)]
void main(uint3 gid : SV_DispatchThreadID) {
  uint w, h;
  idTex.GetDimensions(w, h);
  if (gid.x >= w || gid.y >= h) return;
  idTex[gid.xy] = -1.0;
}
