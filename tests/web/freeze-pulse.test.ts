import { runGpuEffectTest, forEachBackend } from '@nano/web/test/gpu-test-helpers';
import { runEngineTest } from '@nano/web/test/engine-test-helpers';
import type { Sketch } from '@nano/web/src/sketch-types';

/**
 * E2E for warp.legacy.freeze_pulse — "Freeze Pulse", the v2 port of the Resolume
 * Wire "Freeze Pulse" patch (frame-freeze + randomized-blend stutter pulse).
 *
 * The FREEZE itself is invisible against the harness's static grid (a frozen
 * grid == the live grid), so that fidelity is IDE-only. What's testable here:
 * the pulse's scale/blend transforms the frame, different blend modes differ,
 * and with no trigger it's a clean passthrough.
 */
/** The solid probe colour both halves of this suite feed in. */
const SOLID: [number, number, number, number] = [0.15, 0.30, 0.55, 1.0];

// The metadata case is effect-level and runs on BOTH backends — it pins the
// schema this effect publishes, which is exactly where a host-side derivation
// can diverge. Everything below drives runEngineTest (the engine harness page:
// executor.wasm, wires, trace points), which has no native runner; the comp
// runner is the native equivalent and a native sketch host is a follow-up.
forEachBackend((backend) => {
describe(`Freeze Pulse (warp.legacy.freeze_pulse) E2E schema (${backend})`, () => {
  jest.setTimeout(60000);

  it('declares metadata and its parameters', async () => {
    const frame = await runGpuEffectTest({
      module: 'warp.legacy.freeze_pulse', bundle: 'legacy',
      inputColor: SOLID,
      dumpName: 'fp_metadata',
    });
    expect(frame.success).toBe(true);
    expect(frame.metadata?.id).toBe('warp.legacy.freeze_pulse');
    const names = frame.params.map(p => p.name);
    for (const n of ['trigger', 'gate', 'time', 'intensity', 'alpha', 'start_offset',
                     'max_scale', 'jitter', 'contrast', 'blend_mode', 'random_mode', 'seed']) {
      expect(names).toContain(n);
    }
  });
});
});

describe('Freeze Pulse (warp.legacy.freeze_pulse) E2E', () => {
  jest.setTimeout(60000);

  const runChain = (id: string, params: Record<string, number>, dump: string) => {
    const sketch: Sketch = {
      anchor: null,
      wires: [],
      chain: [
        { type: 'module', module_type: 'source.grid', instance_key: 'grid@0', params: {} },
        { type: 'module', module_type: 'warp.legacy.freeze_pulse', instance_key: 'fp@0', params },
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

  const runGridOnly = (id: string, dump: string) => runEngineTest({
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
    waitFrames: 8,
    captureTraceIds: ['out'],
    dumpName: dump,
  });

  it('a triggered pulse transforms/blends the frame', async () => {
    const off = await runChain('fp_off', {}, 'fp_off');
    const on = await runChain('fp_on',
      { trigger: 1, random_mode: 0, blend_mode: 0, max_scale: 2.0, intensity: 1.0, time: 2.0 }, 'fp_on');
    expect(off.success).toBe(true);
    expect(on.success).toBe(true);
    let lit = 0;
    off.trace('out').forEachPixel((c) => { if (c.r + c.g + c.b > 24) lit++; });
    expect(lit).toBeGreaterThan(100);
    on.trace('out').expectDifferentFrom(off.trace('out'), 100);
  });

  it('different blend modes differ', async () => {
    const rgb = await runChain('fp_rgb',
      { trigger: 1, random_mode: 0, blend_mode: 0, max_scale: 1.0, intensity: 1.0, time: 2.0 }, 'fp_rgb');
    const diff = await runChain('fp_diff',
      { trigger: 1, random_mode: 0, blend_mode: 2, max_scale: 1.0, intensity: 1.0, time: 2.0 }, 'fp_diff');
    expect(rgb.success).toBe(true);
    expect(diff.success).toBe(true);
    diff.trace('out').expectDifferentFrom(rgb.trace('out'), 100);
  });

  it('is a passthrough with no trigger', async () => {
    const grid = await runGridOnly('fp_grid', 'fp_gridonly');
    const rest = await runChain('fp_rest', {}, 'fp_rest');
    expect(grid.success).toBe(true);
    expect(rest.success).toBe(true);
    rest.trace('out').expectSameAs(grid.trace('out'), 2);
  });
});
