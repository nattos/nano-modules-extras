import { runGpuTest, runGpuEffectTest, Frame, forEachBackend } from '@nano/web/test/gpu-test-helpers';

/**
 * `filter.light.bar_scatter` — co-triggering bars are, at random, passed,
 * choked or delayed, with one always let through.
 *
 * The sim is GPU-resident, so every case uses `renderEachTick: true` (the
 * harness ticks 0.016 s per frame). The input is a solid colour and can't
 * change mid-run, so the flash under test is the image ARRIVING: reset zeroes
 * each cell's onset reference, so all four bars fire together on frame 1.
 */

// 4 bars x 16 px, 10 segments x 8 px: every cell boundary lands on a pixel.
const W = 64, H = 80;
const ROWS = 10;
const cellX = (bar: number) => Math.round((bar + 0.5) * W / 4);
const cellY = (row: number) => Math.round(H - (row + 0.5) * H / ROWS);   // row 0 at the bottom
const cellR = (f: Frame, bar: number, row: number) => f.pixelAt(cellX(bar), cellY(row)).r;
const barR = (f: Frame, bar: number) => cellR(f, bar, 4);
const lit = (f: Frame) => [0, 1, 2, 3].map((b) => barR(f, b) > 200);
const litCount = (f: Frame) => lit(f).filter(Boolean).length;

forEachBackend((backend) => {
describe(`Bar Scatter (${backend})`, () => {
  jest.setTimeout(60000);

  const run = (dumpName: string, ticks: number, params: [string, unknown][], grey = 1.0) =>
    runGpuEffectTest({
      module: 'filter.light.bar_scatter',
      bundle: 'lights',
      width: W, height: H,
      inputColor: [grey, grey, grey, 1.0],
      renderEachTick: true,
      ticks,
      params: [['rows', ROWS], ['seed', 0], ...params] as any,
      dumpName,
    });
  const chokeOnly: [string, unknown][] = [['strength', 1], ['delay_share', 0]];

  it('declares metadata', async () => {
    const frame = await runGpuTest({
      module: 'filter.light.bar_scatter',
      bundle: 'lights',
      dumpName: 'bar_scatter_metadata',
    });
    expect(frame.success).toBe(true);
    expect(frame.metadata?.id).toBe('filter.light.bar_scatter');
  });

  it('Strength 0 passes the flash straight through', async () => {
    for (const ticks of [2, 40]) {
      const f = await run(`bar_scatter_dry_${ticks}`, ticks, [['strength', 0]]);
      expect(f.success).toBe(true);
      for (let bar = 0; bar < 4; bar++)
        for (let row = 0; row < ROWS; row++) expect(cellR(f, bar, row)).toBeGreaterThanOrEqual(254);
    }
  });

  it('a jump under Sensitivity fires nothing', async () => {
    const f = await run('bar_scatter_quiet', 3, [...chokeOnly, ['sensitivity', 1]], 0.3);
    expect(f.success).toBe(true);
    for (let bar = 0; bar < 4; bar++) expect(Math.abs(barR(f, bar) - 77)).toBeLessThanOrEqual(2);
  });

  it('choke: one bar survives the flash, the rest go dark, then release', async () => {
    const early = await run('bar_scatter_choke_early', 3, chokeOnly);
    expect(early.success).toBe(true);
    expect(litCount(early)).toBe(1);
    for (let bar = 0; bar < 4; bar++) if (!lit(early)[bar]) expect(barR(early, bar)).toBeLessThan(10);

    // Held 0.25 s, then released over 0.15 s: back to white well within 1.3 s.
    const late = await run('bar_scatter_choke_late', 80, chokeOnly);
    for (let bar = 0; bar < 4; bar++) expect(barR(late, bar)).toBeGreaterThanOrEqual(250);
  });

  it('Tilt +1 keeps the centre pair, -1 the outer pair', async () => {
    const plus = await run('bar_scatter_tilt_plus', 3, [...chokeOnly, ['tilt', 1]]);
    const minus = await run('bar_scatter_tilt_minus', 3, [...chokeOnly, ['tilt', -1]]);
    expect(lit(plus)).toEqual([false, true, true, false]);
    expect(lit(minus)).toEqual([true, false, false, true]);
  });

  it('delay: the hits arrive staggered, one step apart', async () => {
    // Steps of 0.08 s, unjittered: the delayed bars land at ~0.1, 0.18, 0.26 s.
    const p: [string, unknown][] = [['strength', 1], ['delay_share', 1], ['delay_jitter', 0]];
    const counts: number[] = [];
    for (const ticks of [3, 8, 13, 20]) {
      const f = await run(`bar_scatter_delay_${ticks}`, ticks, p);
      expect(f.success).toBe(true);
      counts.push(litCount(f));
    }
    expect(counts).toEqual([1, 2, 3, 4]);
  });

  it('Zones: each band of the bars rolls its own dice', async () => {
    const f = await run('bar_scatter_zones', 3, [...chokeOnly, ['zones', ROWS]]);
    expect(f.success).toBe(true);
    let split = false;
    for (let row = 0; row < ROWS; row++) {
      // Each row is its own event, with its own survivor.
      const rowLit = [0, 1, 2, 3].filter((bar) => cellR(f, bar, row) > 200).length;
      expect(rowLit).toBe(1);
    }
    for (let bar = 0; bar < 4; bar++) {
      const rs = Array.from({ length: ROWS }, (_, row) => cellR(f, bar, row) > 200);
      if (rs.some(Boolean) && !rs.every(Boolean)) split = true;
    }
    expect(split).toBe(true);
  });
});
});
