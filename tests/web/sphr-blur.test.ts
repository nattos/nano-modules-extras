import { runGpuEffectTest, forEachBackend } from '@nano/web/test/gpu-test-helpers';
import { runEngineTest } from '@nano/web/test/engine-test-helpers';
import type { Sketch } from '@nano/web/src/sketch-types';

/**
 * E2E for filter.legacy.sphr_blur — "SPHR Blur", the v2 port of the Resolume
 * Wire "SPHR Blur" patch (latitude-dependent sphere-aware horizontal blur +
 * Gaussian softening).
 *
 * Blurring a solid is invisible, so the transform is exercised by chaining a
 * deterministic structured generator (source.grid) → sphr_blur.
 */
/** The solid probe colour both halves of this suite feed in. */
const SOLID: [number, number, number, number] = [0.15, 0.30, 0.55, 1.0];

// The metadata case is effect-level and runs on BOTH backends — it pins the
// schema this effect publishes, which is exactly where a host-side derivation
// can diverge. Everything below drives runEngineTest (the engine harness page:
// executor.wasm, wires, trace points), which has no native runner; the comp
// runner is the native equivalent and a native sketch host is a follow-up.
forEachBackend((backend) => {
describe(`SPHR Blur (filter.legacy.sphr_blur) E2E schema (${backend})`, () => {
  jest.setTimeout(60000);

  it('declares metadata and its parameters', async () => {
    const frame = await runGpuEffectTest({
      module: 'filter.legacy.sphr_blur', bundle: 'legacy',
      inputColor: SOLID,
      dumpName: 'sphr_metadata',
    });
    expect(frame.success).toBe(true);
    expect(frame.metadata?.id).toBe('filter.legacy.sphr_blur');
    const names = frame.params.map(p => p.name);
    for (const n of ['strength', 'gaussian', 'quality']) {
      expect(names).toContain(n);
    }
  });
});
});

describe('SPHR Blur (filter.legacy.sphr_blur) E2E', () => {
  jest.setTimeout(60000);

  const runChain = (id: string, params: Record<string, number>, dump: string) => {
    const sketch: Sketch = {
      anchor: null,
      wires: [],
      chain: [
        { type: 'module', module_type: 'source.grid', instance_key: 'grid@0', params: {} },
        { type: 'module', module_type: 'filter.legacy.sphr_blur', instance_key: 'sb@0', params },
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

  it('sphere-aware expand blurs the structured input', async () => {
    const sharp = await runChain('sphr_sharp', { strength: 0.0, gaussian: 0.0 }, 'sphr_sharp');
    const expanded = await runChain('sphr_exp', { strength: 1.0, gaussian: 0.0, quality: 0.6 }, 'sphr_expanded');
    expect(sharp.success).toBe(true);
    expect(expanded.success).toBe(true);
    let lit = 0;
    sharp.trace('out').forEachPixel((c) => { if (c.r + c.g + c.b > 24) lit++; });
    expect(lit).toBeGreaterThan(100);
    expanded.trace('out').expectDifferentFrom(sharp.trace('out'), 100);
  });

  it('Gaussian stage softens on top (gated by strength)', async () => {
    // gaussian is scaled by strength, so both need strength>0; the extra
    // gaussian softening is what differs.
    const none = await runChain('sphr_gn', { strength: 0.3, gaussian: 0.0, quality: 0.6 }, 'sphr_gauss_off');
    const soft = await runChain('sphr_gs', { strength: 0.3, gaussian: 1.0, quality: 0.6 }, 'sphr_gauss_on');
    expect(none.success).toBe(true);
    expect(soft.success).toBe(true);
    soft.trace('out').expectDifferentFrom(none.trace('out'), 100);
  });

  it('is a passthrough at strength=0, gaussian=0 (is_identity)', async () => {
    const grid = await runGridOnly('sphr_grid', 'sphr_gridonly');
    const passthrough = await runChain('sphr_pass', { strength: 0.0, gaussian: 0.0 }, 'sphr_passthrough');
    expect(grid.success).toBe(true);
    expect(passthrough.success).toBe(true);
    passthrough.trace('out').expectSameAs(grid.trace('out'), 2);
  });
});
