// source.pixel.descent — render pass.
//
// The screen is split into a cols × rows grid; exactly one cell per column is
// lit. WHICH row is lit is decided CPU-side per column (beat-locked step
// scheduling with jitter — see main.cpp) and passed in as an int4[4] array,
// so per pixel we just compare this cell's row against its column's entry.

Texture2D<float4>   inputTex  : register(t0);
RWTexture2D<float4> outputTex : register(u1);

cbuffer Uniforms : register(b2) {
  int    u_cols;
  int    u_rows;
  int    u_composite;   // 0 black, 1 transparent, 2 custom, 3 input
  float  u_intensity;
  float4 u_color;       // rgb = pixel colour
  float4 u_bg;          // custom background
  int4   u_lit[4];      // lit row per column, 16 columns max
};

// cbuffer int arrays have 16-byte stride, so the 16 rows travel as int4[4];
// select the component without dynamic vector subscripts (portability).
int litRowFor(int col) {
  int4 v = u_lit[col >> 2];
  int k = col & 3;
  return k == 0 ? v.x : (k == 1 ? v.y : (k == 2 ? v.z : v.w));
}

[numthreads(8, 8, 1)]
void main(uint3 gid : SV_DispatchThreadID) {
  uint w, h;
  outputTex.GetDimensions(w, h);
  if (gid.x >= w || gid.y >= h) return;

  int col = clamp(int(float(gid.x) * float(u_cols) / float(w)), 0, u_cols - 1);
  int row = clamp(int(float(gid.y) * float(u_rows) / float(h)), 0, u_rows - 1);

  float4 base;
  if      (u_composite == 1) base = float4(0.0, 0.0, 0.0, 0.0);
  else if (u_composite == 2) base = u_bg;
  else if (u_composite == 3) base = inputTex[gid.xy];
  else                       base = float4(0.0, 0.0, 0.0, 1.0);

  if (row == litRowFor(col)) {
    base.rgb = saturate(base.rgb + u_color.rgb * u_intensity);
    base.a = 1.0;
  }
  outputTex[gid.xy] = base;
}
