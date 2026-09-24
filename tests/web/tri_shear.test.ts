import { runEngineTest } from '@nano/web/test/engine-test-helpers';
import type { Sketch } from '@nano/web/src/sketch-types';

/**
 * E2E coverage for warp.tri_shear (nano bundle) — the three-plane triangle shear.
 * It discovers 3 natural lines biased to enclose a large triangle, then chains the
 * single-plane shear three times (once per edge). Fed a deterministic diagonal
 * gradient. Under test:
 *  1. Registers + renders.
 *  2. The shear moves pixels: distance>0 differs from distance=0 (identity).
 *  3. The `size` param reshapes the triangle (size 1 vs 0 differ).
 *  4. Stiffness: a held triangle + max shear is identical across frame counts.
 */
function buildSketch(params: Record<string, unknown>): Sketch {
  return {
    anchor: null,
    chain: [
      {
        type: 'module',
        module_type: 'source.gradient',
        instance_key: 'grad@0',
        params: { angle: 0.3, offset: 0.0, softness: 1.0,
                  color_a: [1, 1, 1], color_b: [0, 0, 0] },
      },
      {
        type: 'module',
        module_type: 'warp.tri_shear',
        instance_key: 'ts@0',
        params,
      },
    ],
  };
}

async function render(sketchId: string, params: Record<string, unknown>, dumpName: string,
                      waitFrames = 6) {
  const result = await runEngineTest({
    width: 96, height: 96,
    modules: ['com.nano.testonly', 'com.nano.core', 'com.nano.nano'],
    commands: [
      { type: 'createSketch', sketchId, sketch: buildSketch(params) },
      { type: 'setTracePoints', tracePoints: [
        { id: 'out', target: { type: 'sketch_output', sketchId } },
      ]},
    ],
    waitFrames,
    captureTraceIds: ['out'],
    dumpName,
  });
  expect(result.success).toBe(true);
  return result;
}

// Instant, held triangle; opaque (original) fill so ramp displacement is directly visible.
const BASE = { algorithm: 1, duration: 0.0, update_rate: 0.0, distance: 0.4,
               size: 0.6, direction: -1.0, rift_fill: 1 };

describe('warp.tri_shear E2E', () => {
  jest.setTimeout(60000);

  it('registers and renders', async () => {
    const result = await render('ts_smoke', BASE, 'ts_smoke');
    result.trace('out').expectNotSolidColor({ r: 0, g: 0, b: 0 }, 5);
    const ts = result.state.plugins.find((p: any) => p.id === 'warp.tri_shear');
    expect(ts).toBeTruthy();
  });

  it('shears the image (three chained shears differ from identity)', async () => {
    const none = await render('ts_none', { ...BASE, distance: 0.0 }, 'ts_none');
    const shear = await render('ts_shear', { ...BASE, distance: 0.4 }, 'ts_shear');
    shear.trace('out').expectDifferentFrom(none.trace('out'), 50);
  });

  it('size reshapes the triangle (changes the output)', async () => {
    const small = await render('ts_small', { ...BASE, size: 0.0 }, 'ts_small');
    const large = await render('ts_large', { ...BASE, size: 1.0 }, 'ts_large');
    large.trace('out').expectDifferentFrom(small.trace('out'), 30);
  });

  it('is stiff: identical across frame counts', async () => {
    const early = await render('ts_stiff_a', BASE, 'ts_stiff_a', 6);
    const late  = await render('ts_stiff_b', BASE, 'ts_stiff_b', 12);
    late.trace('out').expectSameAs(early.trace('out'), 2);
  });
});
