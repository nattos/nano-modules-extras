import { runGpuEffectTest, forEachBackend } from '@nano/web/test/gpu-test-helpers';
import { runEngineTest } from '@nano/web/test/engine-test-helpers';
import type { Sketch } from '@nano/web/src/sketch-types';

/**
 * E2E for warp.legacy.wobble_master — "Wobble Master", the v2 port of the
 * Resolume Wire "Wobble Master 2" family (triggered shockwave wobble +
 * chromatic afterglow). Warping a solid is invisible, so the transform is
 * exercised via a source.grid chain; `amount` drives the standing-wobble
 * floor manually, `gate` launches the traveling pulses.
 */
/** The solid probe colour both halves of this suite feed in. */
const SOLID: [number, number, number, number] = [0.15, 0.30, 0.55, 1.0];

// The metadata case is effect-level and runs on BOTH backends — it pins the
// schema this effect publishes, which is exactly where a host-side derivation
// can diverge. Everything below drives runEngineTest (the engine harness page:
// executor.wasm, wires, trace points), which has no native runner; the comp
// runner is the native equivalent and a native sketch host is a follow-up.
forEachBackend((backend) => {
describe(`Wobble Master (warp.legacy.wobble_master) E2E schema (${backend})`, () => {
  jest.setTimeout(60000);

  it('declares metadata and its parameters', async () => {
    const frame = await runGpuEffectTest({
      module: 'warp.legacy.wobble_master', bundle: 'legacy',
      inputColor: SOLID,
      dumpName: 'wm_metadata',
    });
    expect(frame.success).toBe(true);
    expect(frame.metadata?.id).toBe('warp.legacy.wobble_master');
    const names = frame.params.map(p => p.name);
    for (const n of ['trigger', 'gate', 'amount', 'amplitude', 'frequency',
                     'wave_speed', 'width', 'ripple', 'ripple_speed', 'chroma',
                     'hue', 'release']) {
      expect(names).toContain(n);
    }
  });
});
});

describe('Wobble Master (warp.legacy.wobble_master) E2E', () => {
  jest.setTimeout(60000);

  const runChain = (id: string, params: Record<string, number>, dump: string) => {
    const sketch: Sketch = {
      anchor: null,
      wires: [],
      chain: [
        { type: 'module', module_type: 'source.grid', instance_key: 'grid@0', params: {} },
        { type: 'module', module_type: 'warp.legacy.wobble_master', instance_key: 'wm@0', params },
      ],
    };
    return runEngineTest({
      width: 128, height: 128,
      modules: ['com.nano.core', 'com.nano.legacy'],
      commands: [
        { type: 'createSketch', sketchId: id, sketch },
        { type: 'setTracePoints', tracePoints: [
          { id: 'out', target: { type: 'sketch_output', sketchId: id } },
        ]},
      ],
      waitFrames: 8,
      captureTraceIds: ['out'],
      dumpName: dump,
    });
  };

  const runGridOnly = (id: string, dump: string, waitFrames = 8) => runEngineTest({
    width: 128, height: 128,
    modules: ['com.nano.core', 'com.nano.legacy'],
    commands: [
      { type: 'createSketch', sketchId: id, sketch: {
        anchor: null, wires: [],
        chain: [{ type: 'module', module_type: 'source.grid', instance_key: 'grid@0', params: {} }],
      }},
      { type: 'setTracePoints', tracePoints: [
        { id: 'out', target: { type: 'sketch_output', sketchId: id } },
      ]},
    ],
    waitFrames,
    captureTraceIds: ['out'],
    dumpName: dump,
  });

  it('the radial ripple warps the structured input', async () => {
    const still = await runChain('wm_still', { amount: 0.0 }, 'wm_still');
    const rippled = await runChain('wm_rip',
      { amount: 1.0, amplitude: 1.0, frequency: 0.6, chroma: 0.3, wave_speed: 0.0 }, 'wm_rippled');
    expect(still.success).toBe(true);
    expect(rippled.success).toBe(true);
    let lit = 0;
    still.trace('out').forEachPixel((c) => { if (c.r + c.g + c.b > 24) lit++; });
    expect(lit).toBeGreaterThan(100);
    rippled.trace('out').expectDifferentFrom(still.trace('out'), 100);
  });

  it('the chromatic dispersion alone changes the image', async () => {
    const none = await runChain('wm_cn', { amount: 1.0, amplitude: 1.0, frequency: 0.6, chroma: 0.0 }, 'wm_chroma_off');
    const split = await runChain('wm_cs', { amount: 1.0, amplitude: 1.0, frequency: 0.6, chroma: 1.0 }, 'wm_chroma_on');
    expect(none.success).toBe(true);
    expect(split.success).toBe(true);
    split.trace('out').expectDifferentFrom(none.trace('out'), 100);
  });

  it('a gated pulse emanates from the centre — distorted inside, untouched corners', async () => {
    // Slow, wide, long-tailed pulse so the front provably sits in r ∈ [~0.1,
    // 0.6] across the whole headless rAF pacing range (4–20 ms/frame × 60):
    // wave_speed 0.18 → 15·0.18² ≈ 0.5 r-units/s. Corners (r > 0.65) must be
    // untouched —
    // the wave distorts ONLY what it has reached — while the interior behind
    // the front (long 3 s tail) differs from the resting grid.
    const grid = await runGridOnly('wm_pgrid', 'wm_pulse_grid', 60);
    const pulsed = await runEngineTest({
      width: 128, height: 128,
      modules: ['com.nano.core', 'com.nano.legacy'],
      commands: [
        { type: 'createSketch', sketchId: 'wm_pulse', sketch: {
          anchor: null, wires: [],
          chain: [
            { type: 'module', module_type: 'source.grid', instance_key: 'grid@0', params: {} },
            { type: 'module', module_type: 'warp.legacy.wobble_master', instance_key: 'wm@0',
              params: { amount: 0.0, amplitude: 1.0, wave_speed: 0.18, width: 0.3,
                        release: 3.0, chroma: 1.0 } },
          ],
          instances: {
            'wm@0': { module_type: 'warp.legacy.wobble_master', state: { gate: true } },
          },
        } as unknown as Sketch },
        { type: 'setTracePoints', tracePoints: [
          { id: 'out', target: { type: 'sketch_output', sketchId: 'wm_pulse' } },
        ]},
      ],
      waitFrames: 60,
      captureTraceIds: ['out'],
      dumpName: 'wm_pulse',
    });
    expect(grid.success).toBe(true);
    expect(pulsed.success).toBe(true);

    const a = grid.trace('out');
    const b = pulsed.trace('out');
    let inner = 0, corner = 0;
    for (let y = 0; y < 128; y++) {
      for (let x = 0; x < 128; x++) {
        const r = Math.hypot(x - 63.5, y - 63.5) / 64;
        const p = a.pixelAt(x, y), q = b.pixelAt(x, y);
        const diff = Math.abs(p.r - q.r) + Math.abs(p.g - q.g) + Math.abs(p.b - q.b) > 24;
        if (diff && r < 0.6) inner++;
        if (diff && r > 0.65) corner++;
      }
    }
    expect(inner).toBeGreaterThan(10);   // the wave visibly warps the interior
    expect(corner).toBeLessThan(15);     // ...and hasn't touched the corners yet
  });

  it('is a passthrough at rest (amount=0)', async () => {
    const grid = await runGridOnly('wm_grid', 'wm_gridonly');
    const rest = await runChain('wm_rest', { amount: 0.0 }, 'wm_rest');
    expect(grid.success).toBe(true);
    expect(rest.success).toBe(true);
    rest.trace('out').expectSameAs(grid.trace('out'), 2);
  });
});
