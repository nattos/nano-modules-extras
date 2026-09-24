import { runEngineTest } from '@nano/web/test/engine-test-helpers';
import type { Sketch } from '@nano/web/src/sketch-types';

/**
 * E2E coverage for warp.recompose (nano bundle) — the rule-of-thirds
 * compositional rebalancer. The effect analyzes the input's visual weight,
 * measures how far it sits from the nearest thirds intersection, then slices
 * the frame into the nine thirds cells and translates each one toward balance.
 * The whole analysis + warp runs on-GPU (accumulate → weigh → solve → render);
 * only the three published imbalance scalars take a CPU readback.
 *
 * Web covers the WIRING; native (tests/test_recompose.cpp) covers the math —
 * that the centroid actually converges on the power point, that `spread` does
 * not disturb the balance guarantee, and that the published signs are right.
 *
 * We feed a diagonal gradient (source.gradient) — a translation-VISIBLE image
 * with a clear global gradient, so the saliency analysis always finds an
 * off-centre centre of mass. Under test:
 *  1. Registers + renders, and exposes the three modulation outputs.
 *  2. Correction 0 is an exact passthrough (the modulation-neutrality contract).
 *  3. Correction moves pixels.
 *  4. Spread changes the character of the rearrangement.
 *  5. Axis restrict changes the result.
 *  6. Rift fill modes render differently.
 *  7. Stiffness: frozen update rate + snap smoothing is identical across
 *     different frame counts — it never rubber-bands.
 */
const GRAD = {
  type: 'module' as const,
  module_type: 'source.gradient',
  instance_key: 'grad@0',
  // Off-centre diagonal ramp: a clear gradient and not translation-invariant.
  params: { angle: 0.3, offset: 0.25, softness: 1.0,
            color_a: [1, 1, 1], color_b: [0, 0, 0] },
};

function buildSketch(params: Record<string, unknown> | null): Sketch {
  const chain: Sketch['chain'] = [GRAD];
  if (params) {
    chain.push({
      type: 'module',
      module_type: 'warp.recompose',
      instance_key: 'rc@0',
      params,
    });
  }
  return { anchor: null, chain };
}

async function render(sketchId: string, params: Record<string, unknown> | null,
                      dumpName: string, waitFrames = 6) {
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

// Frozen analysis (update_rate 0 still runs the forced first-frame solve),
// snap smoothing, full travel budget, opaque fills.
const BASE = {
  update_rate: 0.0, smooth: 0.0, spread: 0.0, distance: 1.0,
  overshoot: 1.0, axis: 0, w_grad: 1.0, w_dev: 0.5, w_sat: 0.0,
  center_bias: 0.5, rift_fill: 1, edge_fill: 2, overlap_mode: 0, debug_show: 0,
};

describe('warp.recompose E2E', () => {
  jest.setTimeout(60000);

  it('registers, renders, and exposes its modulation outputs', async () => {
    const result = await render('rc_smoke', { ...BASE, correct: 0.8 }, 'rc_smoke');
    result.trace('out').expectNotSolidColor({ r: 0, g: 0, b: 0 }, 5);

    const rc = result.state.plugins.find((p: any) => p.id === 'warp.recompose');
    expect(rc).toBeTruthy();

    // The three imbalance channels must appear as data outputs (kind 2) so they
    // are wireable as modulation sources.
    for (const name of ['balance_x', 'balance_y', 'cell_error']) {
      expect(rc.io.find((io: any) => io.name === name && io.kind === 2)).toBeTruthy();
    }
    // ...while the inputs must NOT be data outputs.
    expect(rc.io.find((io: any) => io.name === 'correct' && io.kind === 2)).toBeUndefined();
  });

  it('correction 0 is a passthrough', async () => {
    const bare = await render('rc_bare', null, 'rc_bare');
    const zero = await render('rc_zero', { ...BASE, correct: 0.0 }, 'rc_zero');
    // Every cell offset is zero, so each pixel resamples its own position.
    zero.trace('out').expectSameAs(bare.trace('out'), 3);
  });

  it('correction moves pixels', async () => {
    const zero = await render('rc_c0', { ...BASE, correct: 0.0 }, 'rc_c0');
    const full = await render('rc_c1', { ...BASE, correct: 1.0 }, 'rc_c1');
    full.trace('out').expectDifferentFrom(zero.trace('out'), 50);
  });

  it('spread changes the character of the rearrangement', async () => {
    const s0 = await render('rc_s0', { ...BASE, correct: 1.0, spread: 0.0 }, 'rc_s0');
    const s1 = await render('rc_s1', { ...BASE, correct: 1.0, spread: 1.0 }, 'rc_s1');
    s1.trace('out').expectDifferentFrom(s0.trace('out'), 50);
  });

  it('axis restrict changes the result', async () => {
    const both  = await render('rc_ax0', { ...BASE, correct: 1.0, axis: 0 }, 'rc_ax0');
    const onlyX = await render('rc_ax1', { ...BASE, correct: 1.0, axis: 1 }, 'rc_ax1');
    const onlyY = await render('rc_ax2', { ...BASE, correct: 1.0, axis: 2 }, 'rc_ax2');
    onlyX.trace('out').expectDifferentFrom(both.trace('out'), 20);
    onlyY.trace('out').expectDifferentFrom(both.trace('out'), 20);
    onlyY.trace('out').expectDifferentFrom(onlyX.trace('out'), 20);
  });

  it('rift fill modes render the gaps differently', async () => {
    // A big spread opens real gaps between separated cells.
    const gap = { ...BASE, correct: 1.0, spread: 1.0 };
    const orig    = await render('rc_f1', { ...gap, rift_fill: 1 }, 'rc_f1');
    const transp  = await render('rc_f0', { ...gap, rift_fill: 0 }, 'rc_f0');
    const black   = await render('rc_f4', { ...gap, rift_fill: 4 }, 'rc_f4');
    transp.trace('out').expectDifferentFrom(orig.trace('out'), 20);
    black.trace('out').expectDifferentFrom(transp.trace('out'), 20);
  });

  it('draws its debug overlays', async () => {
    const off  = await render('rc_dbg0', { ...BASE, correct: 0.0, debug_show: 0 }, 'rc_dbg0');
    const grid = await render('rc_dbg1', { ...BASE, correct: 0.0, debug_show: 1 }, 'rc_dbg1');
    const mass = await render('rc_dbg2', { ...BASE, correct: 0.0, debug_show: 2 }, 'rc_dbg2');
    grid.trace('out').expectDifferentFrom(off.trace('out'), 50);
    mass.trace('out').expectDifferentFrom(off.trace('out'), 50);
    mass.trace('out').expectDifferentFrom(grid.trace('out'), 50);
  });

  it('is stiff: a frozen analysis is identical across frame counts', async () => {
    const early = await render('rc_stiff_a', { ...BASE, correct: 1.0 }, 'rc_stiff_a', 6);
    const late  = await render('rc_stiff_b', { ...BASE, correct: 1.0 }, 'rc_stiff_b', 12);
    late.trace('out').expectSameAs(early.trace('out'), 2);
  });
});
