/**
 * E2E tests for source.pixel.rift (nano bundle) — coarse-grid ocean waves
 * crossing a hidden mid-rift.
 *
 * The GPU test runner drives a fixed dt = 0.016 s/tick, and the effect's wave
 * pool is a pure function of (params, seed, tick stream) — the xorshift RNG is
 * seeded from `seed` on the first tick — so runs are fully deterministic and
 * repeatable. renderEachTick is required (motion/spawning advance in tick()).
 *
 * Spawning is rate-limited (~2 waves/sec), so tests that need a settled pool
 * use a small density (target = round(density·12) waves) and enough ticks for
 * it to fill: density 0.1 → 1 wave, first spawn at t=0.5s (tick ~32).
 *
 * Wave placement is random on the virtual torus, so exact positions aren't
 * asserted here — shape art, the rift gap, and the rise are eyeballed via the
 * dump PNGs. Needs the dev server up (GPU_TEST_BASE_URL) and a fresh nano
 * bundle (native/wasm_modules/nano/build.sh).
 */
import { runGpuEffectTest, Frame, forEachBackend } from '@nano/web/test/gpu-test-helpers';

jest.setTimeout(120000);

const W = 64, H = 64;

const BASE_PARAMS: [string, number | number[]][] = [
  ['columns', 4], ['rows', 10], ['rift_cols', 4],
  ['drift_rate', 0.4], ['rise', 0.3], ['drift_jitter', 1],
  ['anim_rate', 0.5], ['anim_jitter', 1],
  ['density', 0.4], ['dot_weight', 0.55], ['omega_weight', 0.45], ['seed', 0],
  ['pixel_color', [1, 1, 1]], ['intensity', 1], ['composite', 0],
];

function run(overrides: [string, number | number[]][], ticks: number, dumpName: string) {
  const merged = new Map<string, number | number[]>(BASE_PARAMS);
  for (const [k, v] of overrides) merged.set(k, v);
  return runGpuEffectTest({
    module: 'source.pixel.rift', bundle: 'nano',
    width: W, height: H,
    inputColor: [0, 0, 0, 1],
    params: [...merged.entries()],
    ticks, renderEachTick: true, dumpName,
  });
}

const isLit = (c: { r: number }) => c.r > 200;

// Coarse lit-cell map (visible grid resolution) for frame comparisons.
function litMap(f: Frame, cols = 4, rows = 10): string {
  let out = '';
  for (let r = 0; r < rows; r++)
    for (let c = 0; c < cols; c++)
      out += isLit(f.pixelAt(Math.round((c + 0.5) * (W / cols)),
                             Math.round((r + 0.5) * (H / rows)))) ? '#' : '.';
  return out;
}

forEachBackend((backend) => {
describe(`Pixel Rift (${backend})`, () => {
  it('renders clean with defaults', async () => {
    const f = await run([], 60, 'pixel_rift_defaults');
    expect(f.success).toBe(true);
    expect(f.gpuErrors).toEqual([]);
  });

  it('density 0 stays fully black', async () => {
    const f = await run([['density', 0]], 90, 'pixel_rift_empty');
    expect(f.success).toBe(true);
    expect(f.countPixels(isLit)).toBe(0);
  });

  it('density 1 puts waves on screen, and sparsely', async () => {
    // 240 ticks = 3.84 s → ~7 spawns of a 12-wave target; sprites are ≤ 3×2
    // virtual cells and half the torus is off-grid/rift, so lit coverage stays
    // well under half the frame.
    const f = await run([['density', 1]], 240, 'pixel_rift_dense');
    expect(f.success).toBe(true);
    const lit = f.countPixels(isLit);
    expect(lit).toBeGreaterThan(0);
    expect(lit).toBeLessThan((W * H) / 2);
  });

  it('is deterministic: identical runs produce identical frames', async () => {
    const a = await run([['density', 1]], 150, 'pixel_rift_det_a');
    const b = await run([['density', 1]], 150, 'pixel_rift_det_b');
    expect(a.countPixels(isLit)).toBe(b.countPixels(isLit));
    expect(litMap(a)).toBe(litMap(b));
    expect(a.countPixels(isLit)).toBeGreaterThan(0);   // comparing something real
  });

  it('drift moves a frozen wave; all-zero rates freeze the frame', async () => {
    // One wave (density 0.1), anim frozen (rate 0 → captured 0 → follows the
    // live 0), jitter 0 so steps land on exact accumulator integers. Drift 1
    // = 4 steps/s: 94 ticks ≈ 6 columns — half the 12-col torus — so the wave
    // cannot sit in the 4-col hidden margin at BOTH samples; wider grid
    // (8 visible, rift 0) keeps most of the torus on screen.
    const moving = [['columns', 8], ['rift_cols', 0], ['density', 0.1],
      ['anim_rate', 0], ['drift_jitter', 0], ['rise', 0], ['drift_rate', 1]] as
      [string, number][];
    const m1 = await run(moving, 120, 'pixel_rift_move_a');
    const m2 = await run(moving, 214, 'pixel_rift_move_b');
    expect(litMap(m1, 8)).not.toBe(litMap(m2, 8));

    // Frozen: no drift, no rise, no anim — after the single wave has spawned
    // (tick ~32) nothing else ever changes.
    const frozen = [['columns', 8], ['rift_cols', 0], ['density', 0.1],
      ['anim_rate', 0], ['drift_jitter', 0], ['rise', 0], ['drift_rate', 0]] as
      [string, number][];
    const f1 = await run(frozen, 120, 'pixel_rift_frozen_a');
    const f2 = await run(frozen, 214, 'pixel_rift_frozen_b');
    expect(f1.countPixels(isLit)).toBeGreaterThan(0);
    expect(litMap(f1, 8)).toBe(litMap(f2, 8));
  });

  it('drift jitter 0 locks every wave to the same global step instants', async () => {
    // Two waves (density 0.2 → target 2; spawned at ticks ~32 and ~63), anim
    // frozen, rise 0. Drift 0.5 ⇒ 0.780776 steps/s = 0.0124924 steps/tick, so
    // the SHARED clock crosses step 1 between ticks 80→81 and step 2 between
    // 160→161. Lock-step means motion happens ONLY at those crossings: both
    // waves jump together at 80→81, then nothing moves through tick 155.
    // (Discriminating: per-wave spawn-residue clocks would tick near ticks
    // ~112 and ~143 instead — failing both assertions below.)
    const p = [['columns', 8], ['rift_cols', 0], ['density', 0.2],
      ['anim_rate', 0], ['drift_jitter', 0], ['rise', 0], ['drift_rate', 0.5]] as
      [string, number][];
    const before = await run(p, 80, 'pixel_rift_lockstep_before');
    const after  = await run(p, 81, 'pixel_rift_lockstep_after');
    const quiet  = await run(p, 155, 'pixel_rift_lockstep_quiet');
    expect(before.countPixels(isLit)).toBeGreaterThan(0);
    // The crossing moves the whole sea at once...
    expect(litMap(before, 8)).not.toBe(litMap(after, 8));
    // ...and between crossings every wave holds its cell.
    expect(litMap(after, 8)).toBe(litMap(quiet, 8));
  });

  it('rift smoke: max rift on a wide grid renders clean', async () => {
    const f = await run([['columns', 16], ['rift_cols', 16], ['density', 1]],
      120, 'pixel_rift_max_rift');
    expect(f.success).toBe(true);
    expect(f.gpuErrors).toEqual([]);
  });
});
});
