import { runGpuEffectTest, Frame, forEachBackend } from '@nano/web/test/gpu-test-helpers';

// Per-effect tests for source.light.tingle_top — sparkles bundled at the top of each
// bar while gated, draining downward on release. The GPU particle pool cycles
// over time, so these use renderEachTick. With a black input the sparkles are
// just bright pixels, so the gated-at-top vs released-fills-down envelope is
// visible as where the bright coverage lands vertically.

forEachBackend((backend) => {
describe(`Tingle Top Effect E2E (${backend})`, () => {
  jest.setTimeout(60000);

  const W = 128, H = 128;
  const luma = (p: { r: number; g: number; b: number }) => 0.299 * p.r + 0.587 * p.g + 0.114 * p.b;

  // Fraction of bright pixels in the vertical band [y0, y1).
  const brightBand = (f: Frame, y0: number, y1: number): number => {
    let bright = 0, n = 0;
    for (let y = Math.floor(y0 * H); y < Math.floor(y1 * H); y++) {
      for (let x = 0; x < W; x++) { if (luma(f.pixelAt(x, y)) > 40) bright++; n++; }
    }
    return n > 0 ? bright / n : 0;
  };

  const params = (extra: [string, number][]): [string, number][] => [
    ['density', 300], ['intensity', 2.0], ['size', 0.012], ['hue', 0.12],
    ['top_band_height', 0.1], ['particle_life_ms', 300], ['respawn_delay_ms', 10],
    ['frame_alpha_jitter', 0.0], ['seed', 5], ...extra,
  ];

  it('declares metadata and I/O', async () => {
    const frame = await runGpuEffectTest({
      module: 'source.light.tingle_top', bundle: 'lights',
      inputColor: [0, 0, 0, 1], dumpName: 'tingle_top_metadata',
    });
    expect(frame.success).toBe(true);
    expect(frame.metadata?.id).toBe('source.light.tingle_top');
  });

  it('gated: sparkles stay bundled at the top band', async () => {
    // default_gate_state locks the region at top_band_height.
    const frame = await runGpuEffectTest({
      module: 'source.light.tingle_top', bundle: 'lights',
      width: W, height: H, inputColor: [0, 0, 0, 1], renderEachTick: true,
      ticks: 20, params: params([['default_gate_state', 1]]),
      dumpName: 'tingle_top_gated',
    });
    expect(frame.success).toBe(true);
    expect(brightBand(frame, 0.0, 0.2)).toBeGreaterThan(0.01);   // sparkles up top
    expect(brightBand(frame, 0.6, 1.0)).toBeLessThan(0.003);     // bottom stays dark
  });

  it('prewarm: velocity drifts particles down the bar on the very first frame', async () => {
    // Region LOCKED at the top 5% (gated), but downward velocity. The analytic
    // prewarm gives each particle a random age and drifts it by velocity×age,
    // so the bar is already populated top-to-bottom on frame 1 — impossible
    // without the velocity-based prewarm (fresh spawns would all be at the top).
    const run = (velY: number) => runGpuEffectTest({
      module: 'source.light.tingle_top', bundle: 'lights',
      width: W, height: H, inputColor: [0, 0, 0, 1], renderEachTick: true,
      ticks: 1,
      params: params([['default_gate_state', 1], ['top_band_height', 0.05],
                      ['particle_velocity_y', velY], ['particle_life_ms', 1500],
                      ['respawn_delay_ms', 0]]),
      dumpName: `tingle_top_prewarm_${velY}`,
    });
    const moving = await run(1.0);
    const still = await run(0.0);
    expect(moving.success && still.success).toBe(true);
    // Mid-bar is populated on frame 1 only with velocity (drift); without it,
    // everything is stuck in the top band.
    expect(brightBand(moving, 0.4, 0.6)).toBeGreaterThan(0.01);
    expect(brightBand(still, 0.4, 0.6)).toBeLessThan(0.002);
  });

  it('bar_target_mode one_bar spawns only in the chosen bar', async () => {
    const frame = await runGpuEffectTest({
      module: 'source.light.tingle_top', bundle: 'lights',
      width: W, height: H, inputColor: [0, 0, 0, 1], renderEachTick: true,
      ticks: 12,
      params: params([['default_gate_state', 1], ['bar_target_mode', 0], ['one_bar_target', 2]]),
      dumpName: 'tingle_top_one_bar',
    });
    expect(frame.success).toBe(true);
    // Bright coverage in bar k's column, top band.
    const barCov = (k: number) => {
      const x0 = Math.floor((k + 0.1) * W / 4), x1 = Math.floor((k + 0.9) * W / 4);
      let bright = 0, n = 0;
      for (let y = 0; y < Math.floor(H * 0.25); y++)
        for (let x = x0; x < x1; x++) { if (luma(frame.pixelAt(x, y)) > 40) bright++; n++; }
      return n > 0 ? bright / n : 0;
    };
    expect(barCov(2)).toBeGreaterThan(0.01);   // chosen bar lit
    expect(barCov(0)).toBeLessThan(0.002);     // others stay dark
    expect(barCov(3)).toBeLessThan(0.002);
  });

  it('bar_target_mode random_bar puts a note in a single bar (not spread)', async () => {
    // One sustaining voice in random mode → all its sparkles land in ONE
    // random bar, unlike all_bars which fills every bar.
    const frame = await runGpuEffectTest({
      module: 'source.light.tingle_top', bundle: 'lights',
      width: W, height: H, inputColor: [0, 0, 0, 1], renderEachTick: true,
      ticks: 12,
      params: params([['default_gate_state', 1], ['bar_target_mode', 1]]),
      dumpName: 'tingle_top_random_bar',
    });
    expect(frame.success).toBe(true);
    const barCov = (k: number) => {
      const x0 = Math.floor((k + 0.1) * W / 4), x1 = Math.floor((k + 0.9) * W / 4);
      let bright = 0, n = 0;
      for (let y = 0; y < Math.floor(H * 0.25); y++)
        for (let x = x0; x < x1; x++) { if (luma(frame.pixelAt(x, y)) > 40) bright++; n++; }
      return n > 0 ? bright / n : 0;
    };
    const covs = [0, 1, 2, 3].map(barCov).sort((a, b) => b - a);
    expect(covs[0]).toBeGreaterThan(0.01);    // exactly one bar lit
    expect(covs[1]).toBeLessThan(0.002);      // the rest dark
  });

  it('idle (no note) spawns nothing', async () => {
    // No gate / trigger / level / auto_rate and default_gate_state false → no
    // voice → no spawns. (The trigger model now drives everything.)
    const frame = await runGpuEffectTest({
      module: 'source.light.tingle_top', bundle: 'lights',
      width: W, height: H, inputColor: [0, 0, 0, 1], renderEachTick: true,
      ticks: 10, params: params([]),
      dumpName: 'tingle_top_idle',
    });
    expect(frame.success).toBe(true);
    frame.expectUniformColor({ r: 0, g: 0, b: 0, a: 255 }, 4);
  });

  it('release: a trigger sustains at top, then the wave descends on note-off', async () => {
    // A trigger holds for min_sustain (sparkles at the top), then releases —
    // the spawn distribution becomes a wave that bursts/accelerates downward.
    const at = (ticks: number) => runGpuEffectTest({
      module: 'source.light.tingle_top', bundle: 'lights',
      width: W, height: H, inputColor: [0, 0, 0, 1], renderEachTick: true,
      ticks,
      params: [...params([['top_band_height', 0.1], ['min_sustain_s', 0.1],
                          ['release_s', 0.6], ['release_curve', 1.5],
                          ['particle_life_ms', 150], ['respawn_delay_ms', 5]]),
               ['trigger', 1]],
      dumpName: `tingle_top_release_${ticks}`,
    });
    const sustaining = await at(4);   // still held → bundled at the top
    const released = await at(28);    // wave has moved down the bar
    expect(sustaining.success && released.success).toBe(true);
    expect(brightBand(sustaining, 0.0, 0.2)).toBeGreaterThan(0.01);
    expect(brightBand(released, 0.5, 0.85)).toBeGreaterThan(0.01);
  });

  it('auto_rate fires notes on its own (distinct timed voices)', async () => {
    // Each Poisson event is a discrete note held for min_sustain then released,
    // so the effect self-animates with no gate wired.
    const frame = await runGpuEffectTest({
      module: 'source.light.tingle_top', bundle: 'lights',
      width: W, height: H, inputColor: [0, 0, 0, 1], renderEachTick: true,
            // auto_mode Random: `auto_rate` on its own is inert, since
      // fx::AutoTrigger defaults the mode to Off (a fresh drop stays quiet).
      ticks: 30, params: params([['auto_mode', 1], ['auto_rate', 0.6],
                                 ['min_sustain_s', 0.2]]),
      dumpName: 'tingle_top_auto',
    });
    expect(frame.success).toBe(true);
    expect(brightBand(frame, 0.0, 0.6)).toBeGreaterThan(0.005);
  });
});
});
