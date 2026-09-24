import { runGpuEffectTest, forEachBackend } from '@nano/web/test/gpu-test-helpers';
import { runEngineTest } from '@nano/web/test/engine-test-helpers';
import type { Sketch } from '@nano/web/src/sketch-types';

/**
 * E2E for filter.legacy.subtle_blur — "Subtle Blur", the v2 port of the
 * Resolume Wire "Subtle Blur" patch (light Gaussian blur + drifting chromatic
 * colour offset).
 *
 * A solid input is invisible through this effect (blurring a solid → the same
 * solid; resampling a solid at offsets → the same solid), so the transform is
 * exercised by chaining a deterministic structured generator (source.grid) →
 * subtle_blur and comparing parameter settings.
 */
/** The solid probe colour both halves of this suite feed in. */
const SOLID: [number, number, number, number] = [0.15, 0.30, 0.55, 1.0];

// The metadata case is effect-level and runs on BOTH backends — it pins the
// schema this effect publishes, which is exactly where a host-side derivation
// can diverge. Everything below drives runEngineTest (the engine harness page:
// executor.wasm, wires, trace points), which has no native runner; the comp
// runner is the native equivalent and a native sketch host is a follow-up.
forEachBackend((backend) => {
describe(`Subtle Blur (filter.legacy.subtle_blur) E2E schema (${backend})`, () => {
  jest.setTimeout(60000);

  it('declares metadata and its parameters', async () => {
    const frame = await runGpuEffectTest({
      module: 'filter.legacy.subtle_blur', bundle: 'legacy',
      inputColor: SOLID,
      dumpName: 'subtle_metadata',
    });
    expect(frame.success).toBe(true);
    expect(frame.metadata?.id).toBe('filter.legacy.subtle_blur');
    const names = frame.params.map(p => p.name);
    for (const n of ['blur', 'amount', 'movement', 'hue', 'quality']) {
      expect(names).toContain(n);
    }
  });
});
});

describe('Subtle Blur (filter.legacy.subtle_blur) E2E', () => {
  jest.setTimeout(60000);

  // source.grid → subtle_blur, traced output. params override the defaults.
  const runChain = (id: string, params: Record<string, number>, dump: string) => {
    const sketch: Sketch = {
      anchor: null,
      wires: [],
      chain: [
        { type: 'module', module_type: 'source.grid', instance_key: 'grid@0', params: {} },
        { type: 'module', module_type: 'filter.legacy.subtle_blur', instance_key: 'sb@0',
          params },
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

  // Just the grid, no effect — the passthrough reference.
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

  it('blur softens the structured input', async () => {
    const sharp = await runChain('sb_sharp', { blur: 0.0, amount: 0.0 }, 'subtle_sharp');
    const blurred = await runChain('sb_blur', { blur: 1.0, amount: 0.0 }, 'subtle_blurred');
    expect(sharp.success).toBe(true);
    expect(blurred.success).toBe(true);
    // The grid must have rendered (so "differ" is meaningful).
    let lit = 0;
    sharp.trace('out').forEachPixel((c) => { if (c.r + c.g + c.b > 24) lit++; });
    expect(lit).toBeGreaterThan(100);
    blurred.trace('out').expectDifferentFrom(sharp.trace('out'), 100);
  });

  it('chromatic offset shifts the colour channels', async () => {
    // movement=0 → deterministic basis (no drift); amount carves an RGB fringe.
    const none = await runChain('sb_cn', { blur: 0.0, amount: 0.0, movement: 0.0 }, 'subtle_chroma_off');
    const split = await runChain('sb_cs', { blur: 0.0, amount: 1.0, movement: 0.0 }, 'subtle_chroma_on');
    expect(none.success).toBe(true);
    expect(split.success).toBe(true);
    split.trace('out').expectDifferentFrom(none.trace('out'), 100);
  });

  it('is a passthrough at blur=0, amount=0, movement=0 (is_identity)', async () => {
    const grid = await runGridOnly('sb_grid', 'subtle_gridonly');
    const passthrough = await runChain('sb_pass', { blur: 0.0, amount: 0.0, movement: 0.0 }, 'subtle_passthrough');
    expect(grid.success).toBe(true);
    expect(passthrough.success).toBe(true);
    passthrough.trace('out').expectSameAs(grid.trace('out'), 2);
  });
});
