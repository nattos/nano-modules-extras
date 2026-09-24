import { runGpuEffectTest, forEachBackend } from '@nano/web/test/gpu-test-helpers';
import { runEngineTest } from '@nano/web/test/engine-test-helpers';
import type { Sketch } from '@nano/web/src/sketch-types';

/**
 * E2E for warp.legacy.chroma_wobble — "Chroma Wobble", the v2 port of the
 * Resolume Wire "ChromaWobble" patch (triggered fbm-noise UV wobble + chromatic
 * split). Warping a solid is invisible, so the transform is exercised via a
 * source.grid chain. `amount` drives the wobble manually (no timing).
 */
/** The solid probe colour both halves of this suite feed in. */
const SOLID: [number, number, number, number] = [0.15, 0.30, 0.55, 1.0];

// The metadata case is effect-level and runs on BOTH backends — it pins the
// schema this effect publishes, which is exactly where a host-side derivation
// can diverge. Everything below drives runEngineTest (the engine harness page:
// executor.wasm, wires, trace points), which has no native runner; the comp
// runner is the native equivalent and a native sketch host is a follow-up.
forEachBackend((backend) => {
describe(`Chroma Wobble (warp.legacy.chroma_wobble) E2E schema (${backend})`, () => {
  jest.setTimeout(60000);

  it('declares metadata and its parameters', async () => {
    const frame = await runGpuEffectTest({
      module: 'warp.legacy.chroma_wobble', bundle: 'legacy',
      inputColor: SOLID,
      dumpName: 'cw_metadata',
    });
    expect(frame.success).toBe(true);
    expect(frame.metadata?.id).toBe('warp.legacy.chroma_wobble');
    const names = frame.params.map(p => p.name);
    for (const n of ['trigger', 'gate', 'amount', 'intensity', 'chroma', 'warp',
                     'frequency', 'speed', 'hue', 'attack', 'release']) {
      expect(names).toContain(n);
    }
  });
});
});

describe('Chroma Wobble (warp.legacy.chroma_wobble) E2E', () => {
  jest.setTimeout(60000);

  const runChain = (id: string, params: Record<string, number>, dump: string) => {
    const sketch: Sketch = {
      anchor: null,
      wires: [],
      chain: [
        { type: 'module', module_type: 'source.grid', instance_key: 'grid@0', params: {} },
        { type: 'module', module_type: 'warp.legacy.chroma_wobble', instance_key: 'cw@0', params },
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

  it('the wobble warps the structured input', async () => {
    const still = await runChain('cw_still', { amount: 0.0 }, 'cw_still');
    const wobbled = await runChain('cw_wob',
      { amount: 1.0, intensity: 1.0, chroma: 0.5, warp: 0.7, speed: 0.0 }, 'cw_wobbled');
    expect(still.success).toBe(true);
    expect(wobbled.success).toBe(true);
    let lit = 0;
    still.trace('out').forEachPixel((c) => { if (c.r + c.g + c.b > 24) lit++; });
    expect(lit).toBeGreaterThan(100);
    wobbled.trace('out').expectDifferentFrom(still.trace('out'), 100);
  });

  it('the chromatic split alone changes the image', async () => {
    const none = await runChain('cw_cn', { amount: 1.0, intensity: 1.0, chroma: 0.0, warp: 0.0 }, 'cw_chroma_off');
    const split = await runChain('cw_cs', { amount: 1.0, intensity: 1.0, chroma: 1.0, warp: 0.0 }, 'cw_chroma_on');
    expect(none.success).toBe(true);
    expect(split.success).toBe(true);
    split.trace('out').expectDifferentFrom(none.trace('out'), 100);
  });

  it('is a passthrough at rest (is_identity)', async () => {
    const grid = await runGridOnly('cw_grid', 'cw_gridonly');
    const rest = await runChain('cw_rest', { amount: 0.0 }, 'cw_rest');
    expect(grid.success).toBe(true);
    expect(rest.success).toBe(true);
    rest.trace('out').expectSameAs(grid.trace('out'), 2);
  });
});
