import { runGpuTest, runGpuEffectTest, Frame, forEachBackend } from '@nano/web/test/gpu-test-helpers';

/**
 * `filter.light.bounce_resonator` — the passive, input-excited "fx" version
 * of Bounce Resonator: 4 bars × `rows` cells, each plucked continuously by the
 * input (random grains at a rate set by luma, and kicks on rising luma), all
 * rows stepping through the same cycling diffusion pattern, shown A/C (the
 * swing about each cell's running mean — above lights, below chokes to black).
 *
 * The sim is GPU-resident, so every case uses `renderEachTick: true` (the
 * harness ticks 0.016 s per frame). The input is a solid colour and can't
 * change mid-run, so the rising edge under test is the image ARRIVING: reset
 * zeroes each cell's remembered luma, and the first frame's input is the rise.
 */

// 4 bars x 16 px, 10 segments x 8 px: every cell boundary lands on a pixel.
const W = 64, H = 80;
const ROWS = 10;
const cellX = (bar: number) => Math.round((bar + 0.5) * W / 4);
const cellY = (row: number) => Math.round(H - (row + 0.5) * H / ROWS);   // row 0 at the bottom
const cellR = (f: Frame, bar: number, row: number) => f.pixelAt(cellX(bar), cellY(row)).r;
const allCells = (f: Frame) => {
  const out: number[] = [];
  for (let row = 0; row < ROWS; row++)
    for (let bar = 0; bar < 4; bar++) out.push(cellR(f, bar, row));
  return out;
};

forEachBackend((backend) => {
describe(`Bounce Resonator FX (${backend})`, () => {
  jest.setTimeout(60000);

  // Neutral kicks (band colour white, input colour grey) so r tracks energy.
  const run = (dumpName: string, grey: number, ticks: number, params: [string, unknown][]) =>
    runGpuEffectTest({
      module: 'filter.light.bounce_resonator',
      bundle: 'lights',
      width: W, height: H,
      inputColor: [grey, grey, grey, 1.0],
      renderEachTick: true,
      ticks,
      params: [
        ['band_color', [1.0, 1.0, 1.0]], ['rows', ROWS], ['seed', 0],
        ['grain_rate', 0], ['edge_excite', 0],
        ...params,
      ] as any,
      dumpName,
    });

  it('declares metadata', async () => {
    const frame = await runGpuTest({
      module: 'filter.light.bounce_resonator',
      bundle: 'lights',
      dumpName: 'bounce_resonator_fx_metadata',
    });
    expect(frame.success).toBe(true);
    expect(frame.metadata?.id).toBe('filter.light.bounce_resonator');
  });

  it('with nothing exciting it, the image passes through dry', async () => {
    const frame = await run('bounce_resonator_fx_dry', 0.3, 6, []);
    expect(frame.success).toBe(true);
    for (const r of allCells(frame)) expect(Math.abs(r - 77)).toBeLessThanOrEqual(2);
  });

  it('the image arriving is an up-edge that lights every cell', async () => {
    const frame = await run('bounce_resonator_fx_edge', 0.3, 4, [
      ['edge_excite', 2.0], ['cycle_rate', 0], ['ac_amount', 0],
    ]);
    expect(frame.success).toBe(true);
    // 0.3 dry + (1 - e^-0.6) of kick ≈ 0.75.
    for (const r of allCells(frame)) expect(r).toBeGreaterThan(160);
  });

  it('A/C cancels a held level back to the dry image', async () => {
    const frame = await run('bounce_resonator_fx_ac', 0.3, 90, [
      ['edge_excite', 2.0], ['cycle_rate', 0], ['ac_amount', 1.0], ['ac_time', 0.05],
    ]);
    expect(frame.success).toBe(true);
    for (const r of allCells(frame)) expect(Math.abs(r - 77)).toBeLessThanOrEqual(4);
  });

  it('a busy waveguide overshoots into black and flips frame to frame', async () => {
    const params: [string, unknown][] = [
      ['grain_rate', 40], ['grain_size', 1.0], ['tail', 0.3], ['spread', 0.0],
      ['cycle_rate', 120], ['ac_amount', 1.0], ['ac_time', 0.3], ['black_depth', 3.0],
    ];
    // Long enough for every cell's running mean to settle onto its level.
    const a = await run('bounce_resonator_fx_strobe_a', 0.5, 200, params);
    const b = await run('bounce_resonator_fx_strobe_b', 0.5, 201, params);
    expect(a.success).toBe(true);
    expect(b.success).toBe(true);
    const ca = allCells(a), cb = allCells(b);
    // Below its mean a cell chokes the grey (128) image well toward black.
    expect(Math.min(...ca)).toBeLessThan(60);
    // And one frame later many cells have swung the other way.
    const flipped = ca.filter((r, i) => Math.abs(r - cb[i]) > 60).length;
    expect(flipped).toBeGreaterThanOrEqual(6);
  });

  it('bright input sparkles cell by cell instead of flooding white', async () => {
    const frame = await run('bounce_resonator_fx_grains', 0.8, 200, [
      ['grain_rate', 30], ['grain_size', 0.6], ['ac_amount', 1.0], ['ac_time', 0.3],
    ]);
    expect(frame.success).toBe(true);
    const cells = allCells(frame);
    const mean = cells.reduce((s, r) => s + r, 0) / cells.length;
    const sd = Math.sqrt(cells.reduce((s, r) => s + (r - mean) ** 2, 0) / cells.length);
    expect(sd).toBeGreaterThan(20);
    expect(cells.filter(r => r >= 250).length).toBeLessThan(cells.length / 2);
    // Cells ring on their own: the rows of one bar don't collapse to one value.
    for (let bar = 0; bar < 4; bar++) {
      const col = [...Array(ROWS).keys()].map(row => cellR(frame, bar, row));
      expect(Math.max(...col) - Math.min(...col)).toBeGreaterThan(20);
    }
  });

  it('black input excites nothing', async () => {
    const frame = await run('bounce_resonator_fx_black', 0.0, 20, [
      ['grain_rate', 60], ['grain_size', 2.0], ['edge_excite', 8.0],
    ]);
    expect(frame.success).toBe(true);
    for (const r of allCells(frame)) expect(r).toBeLessThanOrEqual(1);
  });
});
});
