import { runGpuEffectTest, forEachBackend } from '@nano/web/test/gpu-test-helpers';

// Per-effect tests for filter.light.flicker_grid — reduces the input to a grid
// of flat cells and turns each column's luma into a per-column flicker rate
// (pulses capped at on/off every frame; optional overflow fill; low/high
// thresholds → black / solid-on; HSL neutral pull + column leveling).
//
// The harness ticks with a FIXED dt = 0.016 s and renderEachTick interleaves
// tick()+render(), so the phase accumulator is exactly computable: the cap
// (0.5 cycles/frame) engages at rate ≥ 31.25 Hz, and at the cap the
// accumulator hits 1.0 on EVEN frames (0.5, 1.0→pulse, 0.5, 1.0→pulse, …),
// so ticks=8 lands on an ON frame and ticks=7 on an OFF frame.
//
// The harness only supplies a SOLID input colour, so peak-vs-average and
// column leveling (both need luma variation WITHIN a column) aren't testable
// here — verify those by eye in-app. A grey (g,g,g) input has Rec.601 luma
// exactly g, which makes the threshold/rate math below exact.

forEachBackend((backend) => {
describe(`Flicker Grid Effect E2E (${backend})`, () => {
  jest.setTimeout(120000);

  const W = 160, H = 100;
  const MOD = 'filter.light.flicker_grid';
  // Pin the color-shaping knobs off unless a test exercises them, so the
  // rendered brightness is exactly the input grey level times the gate.
  const flat: Array<[string, number]> = [['neutral_pull', 0], ['level_strength', 0]];

  it('declares metadata and I/O', async () => {
    const frame = await runGpuEffectTest({
      module: MOD, bundle: 'lights',
      inputColor: [0, 0, 0, 1], dumpName: 'flicker_grid_metadata',
    });
    expect(frame.success).toBe(true);
    expect(frame.metadata?.id).toBe(MOD);
  });

  it('below the low threshold the output is plain black', async () => {
    const frame = await runGpuEffectTest({
      module: MOD, bundle: 'lights', renderEachTick: true,
      width: W, height: H, inputColor: [0.03, 0.03, 0.03, 1], ticks: 8,
      params: [...flat, ['min_thr', 0.1], ['max_thr', 0.9], ['rate_max', 120], ['fill', 1]],
      dumpName: 'flicker_grid_below_min',
    });
    expect(frame.success).toBe(true);
    frame.expectUniformColor({ r: 0, g: 0, b: 0, a: 255 }, 6);
  });

  it('at/above the high threshold the output holds solid on (no flicker)', async () => {
    const at = (ticks: number) => runGpuEffectTest({
      module: MOD, bundle: 'lights', renderEachTick: true,
      width: W, height: H, inputColor: [1, 1, 1, 1], ticks,
      params: [...flat, ['min_thr', 0.05], ['max_thr', 0.9], ['rate_max', 120], ['fill', 0]],
      dumpName: `flicker_grid_solid_${ticks}`,
    });
    const even = await at(8);
    const odd = await at(7);
    expect(even.success && odd.success).toBe(true);
    // Solid on both frame parities — a capped flicker would blank one of them.
    expect(even.pixelAt(W / 2, H / 2).r).toBeGreaterThan(200);
    expect(odd.pixelAt(W / 2, H / 2).r).toBeGreaterThan(200);
  });

  it('a hot column flickers at the cap: on/off alternating every frame', async () => {
    // 0.8 grey → t = (0.8-0.05)/0.85 ≈ 0.88, rate ≈ 106 Hz — far past the
    // 31.25 Hz cap, so the gate strictly alternates: even frames on.
    const at = (ticks: number) => runGpuEffectTest({
      module: MOD, bundle: 'lights', renderEachTick: true,
      width: W, height: H, inputColor: [0.8, 0.8, 0.8, 1], ticks,
      params: [...flat, ['min_thr', 0.05], ['max_thr', 0.9], ['rate_max', 120], ['fill', 0]],
      dumpName: `flicker_grid_cap_${ticks}`,
    });
    const on = await at(8);
    const off = await at(7);
    expect(on.success && off.success).toBe(true);
    expect(on.pixelAt(W / 2, H / 2).r).toBeGreaterThan(180);   // ≈ 0.8 → 204
    expect(off.pixelAt(W / 2, H / 2).r).toBeLessThan(20);      // off frame: black
    on.expectDifferentFrom(off);
  });

  it('overflow fill brightens the off frames by the beyond-cap demand', async () => {
    // rate_max 60 → rate ≈ 52.9 Hz → inc_raw = 0.847 cycles/frame →
    // fill = (0.847-0.5)/0.5 ≈ 0.69. Off frame ≈ 0.8 · 0.69 ≈ 0.56 → ~142.
    const at = (fill: number) => runGpuEffectTest({
      module: MOD, bundle: 'lights', renderEachTick: true,
      width: W, height: H, inputColor: [0.8, 0.8, 0.8, 1], ticks: 7,
      params: [...flat, ['min_thr', 0.05], ['max_thr', 0.9], ['rate_max', 60], ['fill', fill]],
      dumpName: `flicker_grid_fill_${fill}`,
    });
    const filled = await at(1);
    const dark = await at(0);
    expect(filled.success && dark.success).toBe(true);
    const fr = filled.pixelAt(W / 2, H / 2).r;
    expect(fr).toBeGreaterThan(117);   // 142 ± 25
    expect(fr).toBeLessThan(167);
    expect(dark.pixelAt(W / 2, H / 2).r).toBeLessThan(20);
  });

  it('neutral pull drags cell lightness toward 0.5', async () => {
    // 0.3 grey with max_thr 0.25 → solid on, so brightness is purely the
    // shaped cell colour: pulled → HSL L 0.5 (≈127), unpulled → 0.3 (≈77).
    const at = (pull: number) => runGpuEffectTest({
      module: MOD, bundle: 'lights', renderEachTick: true,
      width: W, height: H, inputColor: [0.3, 0.3, 0.3, 1], ticks: 8,
      params: [['neutral_pull', pull], ['level_strength', 0],
               ['min_thr', 0.05], ['max_thr', 0.25], ['rate_max', 120], ['fill', 0]],
      dumpName: `flicker_grid_pull_${pull}`,
    });
    const pulled = await at(1);
    const raw = await at(0);
    expect(pulled.success && raw.success).toBe(true);
    expect(pulled.pixelAt(W / 2, H / 2).r).toBeGreaterThan(raw.pixelAt(W / 2, H / 2).r + 30);
  });

  it('both luma modes render (peak vs average needs in-column variation to differ)', async () => {
    const at = (mode: number) => runGpuEffectTest({
      module: MOD, bundle: 'lights', renderEachTick: true,
      width: W, height: H, inputColor: [1, 1, 1, 1], ticks: 8,
      params: [...flat, ['mode', mode], ['min_thr', 0.05], ['max_thr', 0.9], ['rate_max', 0], ['fill', 0]],
      dumpName: `flicker_grid_mode_${mode}`,
    });
    const peak = await at(0);
    const avg = await at(1);
    expect(peak.success && avg.success).toBe(true);
    // Solid white input reduces identically under both modes → solid on.
    expect(peak.pixelAt(W / 2, H / 2).r).toBeGreaterThan(200);
    expect(avg.pixelAt(W / 2, H / 2).r).toBeGreaterThan(200);
  });
});
});
