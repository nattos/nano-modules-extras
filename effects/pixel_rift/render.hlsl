// source.pixel.rift — render pass.
//
// The screen splits into a coarse cols × rows cell grid (pixel_descent look),
// but waves live on a WIDER virtual grid: `rift` extra columns are spliced in
// between the left and right halves of the visible columns. Those rift columns
// exist for motion and never map to a pixel — a wave slides in from the left,
// vanishes into the rift, and re-emerges on the right.
//
// All motion/animation is CPU-side (see main.cpp); per pixel we just map the
// visible cell to its virtual column and test it against ≤12 wave sprites
// (tiny dot / omega remakes of pixel_ocean's art). Both axes wrap as a torus
// here so sprites straddling an edge stay contiguous.

Texture2D<float4>   inputTex  : register(t0);
RWTexture2D<float4> outputTex : register(u1);

cbuffer Uniforms : register(b2) {
  int    u_cols;
  int    u_rows;
  int    u_rift;        // hidden virtual columns spliced into the middle
  int    u_composite;   // 0 black, 1 transparent, 2 custom, 3 input
  float4 u_color;       // rgb = wave colour
  float4 u_bg;          // custom background
  float  u_intensity;
  int    u_nwaves;
  float2 u_pad;
  int4   u_waves[12];   // (torus col, row, type*4+frame, active)
};

// Wave sprites, drawn TURNED ON THEIR SIDES: with only ~2 visible columns per
// half-grid there's no room to read a shape horizontally, so the art's long
// axis runs down the ROWS — on screen each sprite occupies a 2-wide × 3-tall
// box, with the art's +x pointing DOWN (art-right = screen-down) and its +y
// toward the trailing (left) edge — so the omega's humps trail and its peak
// leads. The masks below are stored in art space: bit = ay*3 + ax (ax=0 left,
// ay=0 top). 3 frames per type, ping-ponged 0,1,2,1 by the CPU clock.
// Placeholder art in the pixel_ocean spirit — hand-tune freely.
static const uint PR_SPRITES[6] = {
  // type 0 — dot (the ocean's fleck): one · / pair ·· / split ·.·
  //   one ...   pair ...   split ...
  //       .#.        ##.         #.#
  0x10u, 0x18u, 0x28u,
  // type 1 — omega crest, breathing flat → normal → sharp:
  //   flat ...   normal .#.   sharp .#.
  //        #.#          #.#         .#.
  0x28u, 0x2Au, 0x12u,
};

[numthreads(8, 8, 1)]
void main(uint3 gid : SV_DispatchThreadID) {
  uint w, h;
  outputTex.GetDimensions(w, h);
  if (gid.x >= w || gid.y >= h) return;

  int col = clamp(int(float(gid.x) * float(u_cols) / float(w)), 0, u_cols - 1);
  int row = clamp(int(float(gid.y) * float(u_rows) / float(h)), 0, u_rows - 1);

  // Splice the rift in: visible columns [0, left) keep their index, the right
  // half shifts up by the rift width. The torus also carries a 4-column
  // off-grid margin so sprites slide fully out before re-entering.
  int left  = (u_cols + 1) / 2;
  int vcol  = col < left ? col : col + u_rift;
  int torus = u_cols + u_rift + 4;

  float4 base;
  if      (u_composite == 1) base = float4(0.0, 0.0, 0.0, 0.0);
  else if (u_composite == 2) base = u_bg;
  else if (u_composite == 3) base = inputTex[gid.xy];
  else                       base = float4(0.0, 0.0, 0.0, 1.0);

  bool lit = false;
  for (int i = 0; i < u_nwaves && !lit; i++) {
    int4 wv = u_waves[i];
    if (wv.w == 0) continue;
    // Torus-wrapped offsets from the sprite anchor, so a sprite crossing the
    // seam (or the top row while rising) stays contiguous. Screen box is
    // 2 wide × 3 tall; rotate into art space (art-right = screen-DOWN,
    // art-down = screen-left): ax = dy, ay = 1 - dx.
    int dx = vcol - wv.x; if (dx < 0) dx += torus;
    int dy = row  - wv.y; if (dy < 0) dy += u_rows;
    if (dx >= 2 || dy >= 3) continue;
    uint bits = PR_SPRITES[(uint(wv.z) >> 2) * 3u + (uint(wv.z) & 3u)];
    lit = ((bits >> uint((1 - dx) * 3 + dy)) & 1u) != 0u;
  }
  if (lit) {
    base.rgb = saturate(base.rgb + u_color.rgb * u_intensity);
    base.a = 1.0;
  }
  outputTex[gid.xy] = base;
}
