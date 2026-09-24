import { runGpuTest, runGpuEffectTest, forEachBackend } from '@nano/web/test/gpu-test-helpers';

// Per-effect tests for `source.light.strobe_channel` against the `lights` bundle.
//
// The effect:
//   - reads a triangle-wave ping-pong of (seed_low, seed_high) over time
//   - iterates the logistic map (x = r*x*(1-x)) `iterations` times
//   - the final x ∈ [0, 1) is split into `bar_count` regions
//   - the matching bar lights up with `flash_color * intensity`
//   - other bars are black
//
// To get reliable pixel assertions we pin r = 0 (logistic map → 0 after
// one iter) so the active bar is always bar 0 regardless of seed, OR
// use `seed_low = seed_high` to pin the seed directly. We can also pin
// `iterations` to make the math simpler.

forEachBackend((backend) => {
describe(`Strobe Channel Effect E2E (${backend})`, () => {
  jest.setTimeout(30000);

  it('declares metadata and I/O', async () => {
    const frame = await runGpuTest({
      module: 'source.light.strobe_channel',
      bundle: 'lights',
      dumpName: 'strobe_channel_metadata',
    });
    expect(frame.success).toBe(true);
    expect(frame.metadata?.id).toBe('source.light.strobe_channel');
  });

  it('intensity=0 passes input through unchanged', async () => {
    const frame = await runGpuEffectTest({
      module: 'source.light.strobe_channel',
      bundle: 'lights',
      inputColor: [0.4, 0.2, 0.6, 1.0],
      params: [['intensity', 0.0]],
      dumpName: 'strobe_channel_passthrough',
    });
    expect(frame.success).toBe(true);
    // With intensity 0 the additive overlay contributes nothing.
    frame.expectPixelAt(32, 32, { r: 102, g: 51, b: 153, a: 255 }, 4);
  });

  it('pins bar 0 with r=0 and bar_count=4', async () => {
    // r=0 → x' = 0 after first iteration, so active_bar = floor(0 * 4) = 0.
    // Force solid input to make the additive math easy: black input →
    // bar 0 region shows the flash color, others stay black.
    const frame = await runGpuEffectTest({
      module: 'source.light.strobe_channel',
      bundle: 'lights',
      width: 64, height: 64,
      inputColor: [0.0, 0.0, 0.0, 1.0],
      params: [
        ['r', 0.0],
        ['iterations', 4],
        ['bar_count', 4],
        ['flash_color', [1.0, 1.0, 1.0]],
        ['intensity', 1.0],
      ],
      dumpName: 'strobe_channel_bar0',
    });
    expect(frame.success).toBe(true);

    // Bar 0: x ∈ [0, 16). Expect lit (white).
    frame.expectPixelAt(8, 32,  { r: 255, g: 255, b: 255 }, 5);
    // Bar 1, 2, 3: dark.
    frame.expectPixelAt(24, 32, { r: 0, g: 0, b: 0 }, 5);
    frame.expectPixelAt(40, 32, { r: 0, g: 0, b: 0 }, 5);
    frame.expectPixelAt(56, 32, { r: 0, g: 0, b: 0 }, 5);
  });

  it('exactly one bar is lit at default params', async () => {
    // With default params + black input, we should see exactly one bar
    // of width 16px lit and the other three dark. Don't pin which one
    // — the chaotic dynamics depend on elapsed time.
    const frame = await runGpuEffectTest({
      module: 'source.light.strobe_channel',
      bundle: 'lights',
      width: 64, height: 64,
      inputColor: [0.0, 0.0, 0.0, 1.0],
      params: [
        ['flash_color', [1.0, 1.0, 1.0]],
        ['intensity', 1.0],
      ],
      dumpName: 'strobe_channel_exactly_one_lit',
    });
    expect(frame.success).toBe(true);

    // Sample the middle row of each bar at x = 8, 24, 40, 56.
    const samples = [8, 24, 40, 56].map((x) => frame.pixelAt(x, 32));
    const lit = samples.filter(p => p.r > 200);
    const dark = samples.filter(p => p.r < 30);
    expect(lit.length).toBe(1);
    expect(dark.length).toBe(3);
  });

  it('respects custom flash_color and intensity', async () => {
    const frame = await runGpuEffectTest({
      module: 'source.light.strobe_channel',
      bundle: 'lights',
      width: 64, height: 64,
      inputColor: [0.0, 0.0, 0.0, 1.0],
      params: [
        ['r', 0.0],
        ['iterations', 4],
        ['bar_count', 4],
        ['flash_color', [0.0, 1.0, 0.0]],   // pure green
        ['intensity', 0.5],                  // half brightness
      ],
      dumpName: 'strobe_channel_green_half',
    });
    expect(frame.success).toBe(true);

    // Bar 0 lit at green*0.5 ≈ 128.
    frame.expectPixelAt(8, 32, { r: 0, g: 128, b: 0 }, 4);
  });

  it('bar_count=8 splits the canvas into 8 regions', async () => {
    const frame = await runGpuEffectTest({
      module: 'source.light.strobe_channel',
      bundle: 'lights',
      width: 64, height: 64,
      inputColor: [0.0, 0.0, 0.0, 1.0],
      params: [
        ['r', 0.0],
        ['iterations', 4],
        ['bar_count', 8],
        ['flash_color', [1.0, 1.0, 1.0]],
        ['intensity', 1.0],
      ],
      dumpName: 'strobe_channel_8bars',
    });
    expect(frame.success).toBe(true);

    // 8 bars of width 8px each. r=0 → bar 0 lit, others dark.
    frame.expectPixelAt(4, 32,  { r: 255, g: 255, b: 255 }, 5); // bar 0
    frame.expectPixelAt(12, 32, { r: 0, g: 0, b: 0 }, 5);       // bar 1
    frame.expectPixelAt(60, 32, { r: 0, g: 0, b: 0 }, 5);       // bar 7
  });
});
});
