import { runGpuEffectTest, Frame, forEachBackend } from '@nano/web/test/gpu-test-helpers';

// Per-effect tests for filter.glitch.block_dehance — a GPU rect pool that "dehances"
// the input (black / mosaic / noise) inside bright-seeking rectangles.
//
// The pool is GPU-resident and cycles over time, so these use
// renderEachTick + a small respawn_delay so the rects are alive by the
// snapshot. A solid input means bright-seek lands rects anywhere (uniform
// mask) — fine for asserting coverage/mode, not specific positions.

forEachBackend((backend) => {
describe(`Block Dehance Effect E2E (${backend})`, () => {
  jest.setTimeout(60000);

  const W = 128, H = 128;
  const luma = (p: { r: number; g: number; b: number }) => 0.299 * p.r + 0.587 * p.g + 0.114 * p.b;

  // Fill the pool fast: long life, tiny respawn delay, many rects.
  const base = (extra: [string, number][]): [string, number][] => [
    ['count', 60], ['pool_max', 64], ['life_s', 5.0], ['respawn_delay_s', 0.05],
    ['rect_width', 0.18], ['rect_height', 0.12], ['mask_temperature', 1.0], ['seed', 7],
    ...extra,
  ];

  it('declares metadata and I/O', async () => {
    const frame = await runGpuEffectTest({
      module: 'filter.glitch.block_dehance', bundle: 'lights',
      inputColor: [0, 0, 0, 1], dumpName: 'block_dehance_metadata',
    });
    expect(frame.success).toBe(true);
    expect(frame.metadata?.id).toBe('filter.glitch.block_dehance');
  });

  it('count 0 is a pure passthrough', async () => {
    const frame = await runGpuEffectTest({
      module: 'filter.glitch.block_dehance', bundle: 'lights',
      width: W, height: H, inputColor: [0.5, 0.5, 0.5, 1], renderEachTick: true,
      ticks: 5, params: [['count', 0], ['mode_black_weight', 1]],
      dumpName: 'block_dehance_passthrough',
    });
    expect(frame.success).toBe(true);
    frame.expectUniformColor({ r: 128, g: 128, b: 128 }, 4);
  });

  it('black mode punches dark rectangles into the input', async () => {
    const frame = await runGpuEffectTest({
      module: 'filter.glitch.block_dehance', bundle: 'lights',
      width: W, height: H, inputColor: [0.5, 0.5, 0.5, 1], renderEachTick: true,
      ticks: 20,
      params: base([['mode_black_weight', 1], ['mode_mosaic_weight', 0], ['mode_noise_weight', 0]]),
      dumpName: 'block_dehance_black',
    });
    expect(frame.success).toBe(true);
    // Lots of near-black coverage (the grey background is luma 128).
    frame.expectCoverage(c => luma(c) < 30, { min: 0.05 });
    // But not the whole frame — the background grey still shows.
    frame.expectCoverage(c => Math.abs(luma(c) - 128) < 10, { min: 0.1 });
  });

  it('move_chance makes a rect jump once after spawning', async () => {
    // One black rect, fixed seed. The bright-seek + lifecycle are
    // deterministic, so the ONLY difference between move 0 and move 1 is the
    // one-time glitch jump — the dark rect's centroid shifts by ~move_amount.
    const run = (move_chance: number) => runGpuEffectTest({
      module: 'filter.glitch.block_dehance', bundle: 'lights',
      width: W, height: H, inputColor: [0.5, 0.5, 0.5, 1], renderEachTick: true,
      ticks: 24,
      params: [['count', 1], ['pool_max', 8], ['life_s', 10.0], ['respawn_delay_s', 0.02],
               ['rect_width', 0.1], ['rect_height', 0.1], ['seed', 3],
               ['mode_black_weight', 1], ['mode_mosaic_weight', 0], ['mode_noise_weight', 0],
               ['move_chance', move_chance], ['move_amount', 0.25], ['move_delay_max', 0.1]],
      dumpName: `block_dehance_move_${move_chance}`,
    });
    const darkCentroid = (f: Frame) => {
      let sx = 0, sy = 0, n = 0;
      f.forEachPixel((c, x, y) => { if (luma(c) < 30) { sx += x; sy += y; n++; } });
      return n > 0 ? { x: sx / n, y: sy / n } : null;
    };
    const still = await run(0.0);
    const moved = await run(1.0);
    expect(still.success && moved.success).toBe(true);
    const a = darkCentroid(still), b = darkCentroid(moved);
    expect(a && b).toBeTruthy();
    const dist = Math.hypot(a!.x - b!.x, a!.y - b!.y);
    expect(dist).toBeGreaterThan(8);
  });

  // Spawn-area gate geometry: 128×128 viewport → cover-square dist =
  // pixel dist from (64,64) / 64. Rects are 0.1×0.1 uv (half-extent 6.4 px),
  // so a rect reaches at most ~9.1 px (half diagonal) past its centre.
  const spawnParams = (bias: number, radius: number, temp: number): [string, number][] => [
    ['count', 60], ['pool_max', 64], ['life_s', 5.0], ['respawn_delay_s', 0.05],
    ['rect_width', 0.1], ['rect_height', 0.1], ['rect_size_jitter', 0],
    ['mask_temperature', temp], ['seed', 7],
    ['mode_black_weight', 1], ['mode_mosaic_weight', 0], ['mode_noise_weight', 0],
    ['spawn_amount', bias], ['spawn_radius', radius], ['spawn_softness', 0],
  ];

  it('spawn bias +1 confines rects to the centre', async () => {
    // Hard gate at cover-square radius 0.4 → rect centres within 25.6 px of
    // the viewport centre; with the half-diagonal, no dark pixel past ~35 px.
    // Temperature 0 exercises the argmax/Bernoulli gate path.
    const frame = await runGpuEffectTest({
      module: 'filter.glitch.block_dehance', bundle: 'lights',
      width: W, height: H, inputColor: [0.5, 0.5, 0.5, 1], renderEachTick: true,
      ticks: 24, params: spawnParams(1.0, 0.4, 0.0),
      dumpName: 'block_dehance_spawn_center',
    });
    expect(frame.success).toBe(true);
    frame.expectCoverage(c => luma(c) < 30, { min: 0.02 });
    let stray = 0;
    frame.forEachPixel((c, x, y) => {
      if (luma(c) < 30 && Math.hypot(x - 64, y - 64) > 40) stray++;
    });
    expect(stray).toBe(0);
  });

  it('spawn bias -1 keeps the centre clear', async () => {
    // Hard gate at cover-square radius 0.5 → rect centres at least 32 px out;
    // nearest dark pixel ≥ ~23 px from the centre. Temperature 1 exercises
    // the softmax-weighting path.
    const frame = await runGpuEffectTest({
      module: 'filter.glitch.block_dehance', bundle: 'lights',
      width: W, height: H, inputColor: [0.5, 0.5, 0.5, 1], renderEachTick: true,
      ticks: 24, params: spawnParams(-1.0, 0.5, 1.0),
      dumpName: 'block_dehance_spawn_rim',
    });
    expect(frame.success).toBe(true);
    frame.expectCoverage(c => luma(c) < 30, { min: 0.03 });
    let inner = 0;
    frame.forEachPixel((c, x, y) => {
      if (luma(c) < 30 && Math.hypot(x - 64, y - 64) < 20) inner++;
    });
    expect(inner).toBe(0);
  });

  it('noise mode replaces covered pixels with varied colour', async () => {
    const frame = await runGpuEffectTest({
      module: 'filter.glitch.block_dehance', bundle: 'lights',
      width: W, height: H, inputColor: [0.5, 0.5, 0.5, 1], renderEachTick: true,
      ticks: 20,
      params: base([['mode_black_weight', 0], ['mode_mosaic_weight', 0], ['mode_noise_weight', 1],
                    ['noise_intensity', 1.0]]),
      dumpName: 'block_dehance_noise',
    });
    expect(frame.success).toBe(true);
    // Noise scatters pixels far from the flat grey in BOTH directions.
    frame.expectCoverage(c => luma(c) > 180, { min: 0.03 });
    frame.expectCoverage(c => luma(c) < 70, { min: 0.03 });
  });
});
});
