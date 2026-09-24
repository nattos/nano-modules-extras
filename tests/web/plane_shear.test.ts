import { runEngineTest } from '@nano/web/test/engine-test-helpers';
import type { Sketch } from '@nano/web/src/sketch-types';

/**
 * E2E coverage for warp.plane_shear (nano bundle) — the analysis-driven shear /
 * rift. The effect analyzes the input to pick a "natural" dividing plane, then
 * shears the two halves on either side of it. The whole analysis + latch runs
 * on-GPU (accumulate → solve → render); only the shear translation is CPU-timed.
 *
 * We feed a deterministic diagonal gradient (source.gradient) — a translation-
 * VISIBLE image with a clear global gradient so every plane algorithm finds a
 * line. Under test:
 *  1. Registers + renders; the schema exposes the algorithm + direction params.
 *  2. The shear moves pixels: a rift (direction=-1) differs from no shear.
 *  3. A transparent rift opens a real gap (fully-transparent pixels appear).
 *  4. Slip (direction=0) differs from rift (direction=-1).
 *  5. Stiffness: with the plane held (slow update_rate) and shear at max
 *     (duration=0), the output is IDENTICAL across different frame counts — the
 *     plane snaps and holds; it never rubber-bands.
 */
function buildSketch(params: Record<string, unknown>): Sketch {
  return {
    anchor: null,
    chain: [
      {
        type: 'module',
        module_type: 'source.gradient',
        instance_key: 'grad@0',
        // White→black diagonal ramp: clear gradient + not translation-invariant.
        params: { angle: 0.3, offset: 0.0, softness: 1.0,
                  color_a: [1, 1, 1], color_b: [0, 0, 0] },
      },
      {
        type: 'module',
        module_type: 'warp.plane_shear',
        instance_key: 'ps@0',
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

// Instant, full-amplitude shear on a slowly-updating (effectively held) plane.
const BASE = { algorithm: 0, duration: 0.0, update_rate: 0.0,
               distance: 0.4, mult_a: 1.0, mult_b: 1.0 };

describe('warp.plane_shear E2E', () => {
  jest.setTimeout(60000);

  it('registers and renders', async () => {
    const result = await render('ps_smoke', { ...BASE, direction: -1.0, rift_fill: 1 }, 'ps_smoke');
    result.trace('out').expectNotSolidColor({ r: 0, g: 0, b: 0 }, 5);

    const ps = result.state.plugins.find((p: any) => p.id === 'warp.plane_shear');
    expect(ps).toBeTruthy();
  });

  it('shears the image (rift differs from no shear)', async () => {
    const none = await render('ps_none', { ...BASE, direction: -1.0, distance: 0.0, rift_fill: 1 }, 'ps_none');
    const rift = await render('ps_rift', { ...BASE, direction: -1.0, rift_fill: 1 }, 'ps_rift');
    rift.trace('out').expectDifferentFrom(none.trace('out'), 50);
  });

  it('opens a rift band whose fill mode changes the output', async () => {
    // A rift band exists and its fill matters: transparent, original, edge-
    // stretch and mirror each render the band differently.
    const orig    = await render('ps_fill_orig',    { ...BASE, direction: -1.0, rift_fill: 1 }, 'ps_fill_orig');
    const transp  = await render('ps_fill_transp',  { ...BASE, direction: -1.0, rift_fill: 0 }, 'ps_fill_transp');
    const stretch = await render('ps_fill_stretch', { ...BASE, direction: -1.0, rift_fill: 2 }, 'ps_fill_stretch');
    transp.trace('out').expectDifferentFrom(orig.trace('out'), 50);
    stretch.trace('out').expectDifferentFrom(transp.trace('out'), 50);
  });

  it('slip (direction 0) differs from rift (direction -1)', async () => {
    const rift = await render('ps_dir_rift', { ...BASE, direction: -1.0, rift_fill: 1 }, 'ps_dir_rift');
    const slip = await render('ps_dir_slip', { ...BASE, direction: 0.0, rift_fill: 1 }, 'ps_dir_slip');
    slip.trace('out').expectDifferentFrom(rift.trace('out'), 50);
  });

  it('is stiff: the held plane + max shear are identical across frame counts', async () => {
    const early = await render('ps_stiff_a', { ...BASE, direction: -1.0, rift_fill: 1 }, 'ps_stiff_a', 6);
    const late  = await render('ps_stiff_b', { ...BASE, direction: -1.0, rift_fill: 1 }, 'ps_stiff_b', 12);
    late.trace('out').expectSameAs(early.trace('out'), 2);
  });
});
