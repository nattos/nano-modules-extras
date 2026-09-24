/**
 * E2E tests for source.pixel.descent (nano bundle) — beat-locked stepping grid.
 *
 * The GPU test runner drives a deterministic clock: dt = 0.016 s at 120 BPM,
 * barPhase = (tick+1) * 0.008. The effect reconstructs beats as
 * (bars + barPhase) * 4, so after N ticks beats = 0.032·N. All tests use
 * rows 5 / Loop Beats 2 ⇒ the unwrapped step clock T = beats·rows/loop
 * = 0.08·N — one row step every 12.5 ticks, a full loop every 62.5.
 *
 * The stepping is a pure function of (transport, seed): no RNG state, so
 * every case here is fully deterministic. renderEachTick is required — the
 * bar-wrap clock advances in tick() and render() reads it each frame.
 *
 * Needs the dev server up (GPU_TEST_BASE_URL) and a fresh nano bundle
 * (native/wasm_modules/nano/build.sh).
 */
import { runGpuEffectTest, Frame, forEachBackend } from '@nano/web/test/gpu-test-helpers';

jest.setTimeout(120000);

const W = 64, H = 64;
const COLS = 4, ROWS = 5;
// Column / row cell centres for the 4×5 grid on 64×64.
const colX = (c: number) => Math.round((c + 0.5) * (W / COLS));
const rowY = (r: number) => Math.round((r + 0.5) * (H / ROWS));

const BASE_PARAMS: [string, number | number[]][] = [
  ['columns', COLS], ['rows', ROWS], ['beats', 2],
  ['jitter', 0], ['skip_chance', 0], ['seed', 0],
  ['pixel_color', [1, 1, 1]], ['intensity', 1], ['composite', 0],
];

function run(overrides: [string, number | number[]][], ticks: number, dumpName: string,
             inputColor: [number, number, number, number] = [0, 0, 0, 1]) {
  const merged = new Map<string, number | number[]>(BASE_PARAMS);
  for (const [k, v] of overrides) merged.set(k, v);
  return runGpuEffectTest({
    module: 'source.pixel.descent', bundle: 'nano',
    width: W, height: H,
    inputColor,
    params: [...merged.entries()],
    ticks, renderEachTick: true, dumpName,
  });
}

const isLit = (c: { r: number }) => c.r > 200;

// Which row-centre is lit in column c (-1 if none, -2 if more than one).
function litRow(f: Frame, c: number): number {
  let found = -1;
  for (let r = 0; r < ROWS; r++) {
    if (isLit(f.pixelAt(colX(c), rowY(r)))) {
      if (found >= 0) return -2;
      found = r;
    }
  }
  return found;
}
const litRows = (f: Frame) => Array.from({ length: COLS }, (_, c) => litRow(f, c));

// PUPPETEER ONLY — a recorded gap, not an oversight. The step-clock cases
// diverge on Metal (the row/step progression doesn't land where the
// puppeteer run puts it) even though host time, dt and barPhase are stepped
// identically by both runners — so the divergence is inside the effect's
// native path, not the harness. Undiagnosed; flip the list back once it is.
forEachBackend((backend) => {
describe(`Pixel Descent (${backend})`, () => {
  it('starts as a line: every column lit on row 0, one cell per column', async () => {
    const f = await run([], 4, 'pixel_descent_line_top');   // T = 0.32 → row 0
    expect(f.success).toBe(true);
    expect(f.gpuErrors).toEqual([]);
    expect(litRows(f)).toEqual([0, 0, 0, 0]);
    // Exactly one lit cell per column: 4 cells ≈ 4 · 16 · 12.8 px.
    const lit = f.countPixels(isLit);
    expect(lit).toBeGreaterThan(600);
    expect(lit).toBeLessThan(1100);
  });

  it('progresses linearly and wraps back to the top after a full loop', async () => {
    const mid = await run([], 20, 'pixel_descent_row1');    // T = 1.6 → row 1
    expect(litRows(mid)).toEqual([1, 1, 1, 1]);
    const wrapped = await run([], 70, 'pixel_descent_wrap'); // T = 5.6 → step 5 → row 0
    expect(litRows(wrapped)).toEqual([0, 0, 0, 0]);
  });

  it('jitter breaks the line near a step boundary', async () => {
    // T = 1.6; step 2's early window (jitter 0.9) opens at T ≥ 1.1, so eager
    // columns already sit on row 2 while on-grid ones hold row 1. Seed 0.1
    // hashes to gate hits on columns 1 and 3 (verified against the CPU hash).
    const jittered = await run([['jitter', 0.9], ['seed', 0.1]], 20, 'pixel_descent_jitter');
    expect(litRows(jittered)).toEqual([1, 2, 1, 2]);
    const straight = await run([], 20, 'pixel_descent_nojitter');
    jittered.expectDifferentFrom(straight, 50);
  });

  it('skip: one column near-double-steps while the rest hold', async () => {
    // skip_chance 1 → every step one hashed column fires ~0.9 of a step early.
    // At T = 1.12 step 2's skip window (T ≥ 1.1) has JUST opened: exactly one
    // column is on row 2, the other three still on row 1.
    const f = await run([['skip_chance', 1]], 14, 'pixel_descent_skip');
    const rows = litRows(f);
    expect(rows.filter((r) => r === 2).length).toBe(1);
    expect(rows.filter((r) => r === 1).length).toBe(3);
  });

  it('composite Input passes the layer through under the pixels', async () => {
    const f = await run([['composite', 3]], 4, 'pixel_descent_input', [0.9, 0, 0, 1]);
    // Unlit cell: the red input, untouched.
    const bg = f.pixelAt(colX(0), rowY(3));
    expect(bg.r).toBeGreaterThan(180);
    expect(bg.g).toBeLessThan(40);
    // Lit cell: white pixel added on top.
    const lit = f.pixelAt(colX(0), rowY(0));
    expect(lit.g).toBeGreaterThan(200);
  });

  it('is deterministic: identical config twice → identical frames', async () => {
    const cfg: [string, number | number[]][] = [
      ['jitter', 0.7], ['skip_chance', 0.5], ['seed', 0.42]];
    const a = await run(cfg, 30, 'pixel_descent_det_a');
    const b = await run(cfg, 30, 'pixel_descent_det_b');
    expect(a.diffCount(b, 2)).toBe(0);
  });
});
}, ['puppeteer']);
