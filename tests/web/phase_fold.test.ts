import { runEngineTest, runEngineMultiPhaseTest } from '@nano/web/test/engine-test-helpers';
import type { Sketch } from '@nano/web/src/sketch-types';

/**
 * E2E coverage for source.phase_fold (nano bundle) — the limit-cycle phase-
 * portrait generator. A baked atlas of level-set limit-cycle fields is uploaded
 * to the GPU; the backdrop (blended height field), the streamline tracer (with
 * animated arrows) and the limit-cycle integrator all run as GPU compute passes,
 * rasterized as soft line quads. Streamlines and the limit cycle are separate,
 * independently toggleable stages.
 *
 * Under test:
 *  1. Registers + renders: a valid cell produces a non-solid banded backdrop
 *     (the compute → raster pipeline dispatches cleanly on WebGPU) and the
 *     autopilot_x/y broadcast outputs are declared (kind=2).
 *  2. wind (z) changes the output — the non-potential force distorts the flow.
 *  3. show_streamlines / show_limit_cycle each toggle a real, separable stage:
 *     turning a stage off changes the output (its lines disappear).
 *  4. autopilot is a live, non-destructive override: with the flow clock frozen
 *     (flow_speed=0) the output STILL animates across frames when autopilot is
 *     on (the epicycle moves the effective XY) and is STATIC when it's off.
 *
 * The generator ignores its input, so the chain is just
 * texture_input → phase_fold → texture_output.
 */
function buildSketch(params: Record<string, unknown>): Sketch {
  return {
    anchor: null,
    chain: [
      {
        type: 'module',
        module_type: 'source.phase_fold',
        instance_key: 'pf@0',
        params,
      },
    ],
  };
}

async function render(sketchId: string, params: Record<string, unknown>, dumpName: string,
                      waitFrames = 6) {
  const result = await runEngineTest({
    width: 96, height: 96,
    modules: ['com.nano.testonly', 'com.nano.nano'],
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

// A valid cell with both stages on, flow clock frozen for determinism.
const BASE = { eccentricity: 0.5, lobedness: 0.3, flow_speed: 0.0 };

describe('source.phase_fold E2E', () => {
  jest.setTimeout(60000);

  it('registers and renders a non-solid phase portrait', async () => {
    const result = await render('pf_smoke', BASE, 'pf_smoke');
    result.trace('out').expectNotSolidColor({ r: 0, g: 0, b: 0 }, 5);

    const pf = result.state.plugins.find((p: any) => p.id === 'source.phase_fold');
    expect(pf).toBeTruthy();
    expect(pf.io.find((io: any) => io.name === 'autopilot_x' && io.kind === 2)).toBeTruthy();
    expect(pf.io.find((io: any) => io.name === 'autopilot_y' && io.kind === 2)).toBeTruthy();
  });

  it('wind (z) changes the flow', async () => {
    const calm = await render('pf_calm', { ...BASE, wind: 0.0 }, 'pf_calm');
    const windy = await render('pf_windy', { ...BASE, wind: 0.9 }, 'pf_windy');
    windy.trace('out').expectDifferentFrom(calm.trace('out'), 20);
  });

  it('show_streamlines toggles a separable stage', async () => {
    const on = await render('pf_sl_on', { ...BASE, show_streamlines: true, show_limit_cycle: false }, 'pf_sl_on');
    const off = await render('pf_sl_off', { ...BASE, show_streamlines: false, show_limit_cycle: false }, 'pf_sl_off');
    on.trace('out').expectDifferentFrom(off.trace('out'), 10);
  });

  it('show_limit_cycle toggles a separable stage', async () => {
    const on = await render('pf_lc_on', { ...BASE, show_streamlines: false, show_limit_cycle: true }, 'pf_lc_on');
    const off = await render('pf_lc_off', { ...BASE, show_streamlines: false, show_limit_cycle: false }, 'pf_lc_off');
    on.trace('out').expectDifferentFrom(off.trace('out'), 5);
  });

  it('limit-cycle solver explores — the ring jitters across frames', async () => {
    // explore>0 random-walks the stateful ring tangentially each frame, so the
    // cycle is NOT static even with a frozen flow clock and fixed XY.
    const r = await runEngineMultiPhaseTest({
      width: 96, height: 96,
      modules: ['com.nano.testonly', 'com.nano.nano'],
      dumpName: 'pf_jitter',
      phases: [
        {
          commands: [
            { type: 'createSketch', sketchId: 'pf_jitter',
              sketch: buildSketch({ ...BASE, show_streamlines: false, show_limit_cycle: true, explore: 0.8 }) },
            { type: 'setTracePoints', tracePoints: [
              { id: 'out', target: { type: 'sketch_output', sketchId: 'pf_jitter' } },
            ]},
          ],
          waitFrames: 4, captureTraceIds: ['out'],
        },
        { waitFrames: 8, captureTraceIds: ['out'] },
      ],
    });
    expect(r.success).toBe(true);
    r.phases[1].trace('out').expectDifferentFrom(r.phases[0].trace('out'), 5);
  });

  it('Tracer cycle mode renders a cycle (CPU flow tracer + momentum ring)', async () => {
    // Give the CPU tracer time to trace a loop and pull the ring onto it.
    const tracer = await render('pf_tracer', { ...BASE, show_streamlines: false, cycle_mode: 1 }, 'pf_tracer', 80);
    tracer.trace('out').expectNotSolidColor({ r: 0, g: 0, b: 0 }, 5);
    const relax = await render('pf_relax', { ...BASE, show_streamlines: false, cycle_mode: 0 }, 'pf_relax', 12);
    relax.trace('out').expectNotSolidColor({ r: 0, g: 0, b: 0 }, 5);
    // Trace mode draws the tracer's raw trajectory directly.
    const traceViz = await render('pf_traceviz', { ...BASE, show_streamlines: false, cycle_mode: 2 }, 'pf_traceviz', 40);
    traceViz.trace('out').expectNotSolidColor({ r: 0, g: 0, b: 0 }, 5);
    // Contour mode draws the height field's zero level-set (no particles).
    const contour = await render('pf_contour', { ...BASE, show_streamlines: false, cycle_mode: 3 }, 'pf_contour', 4);
    contour.trace('out').expectNotSolidColor({ r: 0, g: 0, b: 0 }, 5);
  });

  it('jitter orbits the XY (output drifts across frames, autopilot off)', async () => {
    // jitter chaotically moves the effective XY → the selected cell/field drifts
    // across frames even with a frozen flow clock and no autopilot.
    const r = await runEngineMultiPhaseTest({
      width: 96, height: 96,
      modules: ['com.nano.testonly', 'com.nano.nano'],
      dumpName: 'pf_jitterxy',
      phases: [
        {
          commands: [
            { type: 'createSketch', sketchId: 'pf_jitterxy',
              sketch: buildSketch({ eccentricity: 0.5, lobedness: 0.5, flow_speed: 0.0,
                                    show_streamlines: false, show_limit_cycle: false,
                                    autopilot: false, jitter: 0.8, jitter_speed: 0.8 }) },
            { type: 'setTracePoints', tracePoints: [
              { id: 'out', target: { type: 'sketch_output', sketchId: 'pf_jitterxy' } },
            ]},
          ],
          waitFrames: 4, captureTraceIds: ['out'],
        },
        { waitFrames: 40, captureTraceIds: ['out'] },
      ],
    });
    expect(r.success).toBe(true);
    r.phases[1].trace('out').expectDifferentFrom(r.phases[0].trace('out'), 20);
  });

  it('wind jitter wobbles the field (output drifts across frames, XY jitter off)', async () => {
    // wind_jitter chaotically modulates the wind value → the Bands backdrop tilt
    // drifts across frames even with a frozen flow clock, static XY and no autopilot.
    const r = await runEngineMultiPhaseTest({
      width: 96, height: 96,
      modules: ['com.nano.testonly', 'com.nano.nano'],
      dumpName: 'pf_jitterwind',
      phases: [
        {
          commands: [
            { type: 'createSketch', sketchId: 'pf_jitterwind',
              sketch: buildSketch({ eccentricity: 0.5, lobedness: 0.5, flow_speed: 0.0,
                                    show_streamlines: false, show_limit_cycle: false,
                                    autopilot: false, jitter: 0.0,
                                    wind_jitter: 0.8, wind_jitter_speed: 0.8 }) },
            { type: 'setTracePoints', tracePoints: [
              { id: 'out', target: { type: 'sketch_output', sketchId: 'pf_jitterwind' } },
            ]},
          ],
          waitFrames: 4, captureTraceIds: ['out'],
        },
        { waitFrames: 40, captureTraceIds: ['out'] },
      ],
    });
    expect(r.success).toBe(true);
    r.phases[1].trace('out').expectDifferentFrom(r.phases[0].trace('out'), 20);
  });

  it('scale (domain zoom) changes the output', async () => {
    const near = await render('pf_scale1', { ...BASE, scale: 1.0 }, 'pf_scale1');
    const far = await render('pf_scale4', { ...BASE, scale: 4.0 }, 'pf_scale4');
    far.trace('out').expectDifferentFrom(near.trace('out'), 30);
  });

  it('Bands shading is wind-aware (downwind force-potential tilt)', async () => {
    // Bands folds in the along-wind force-potential ramp W·p, so the height-field
    // terrace tilts downwind — turning wind up must change the Bands backdrop.
    const calm = await render('pf_bands_calm', { ...BASE, show_streamlines: false, show_limit_cycle: false, shading_mode: 0, wind: 0.0 }, 'pf_bands_calm');
    const windy = await render('pf_bands_windy', { ...BASE, show_streamlines: false, show_limit_cycle: false, shading_mode: 0, wind: 0.8 }, 'pf_bands_windy');
    windy.trace('out').expectDifferentFrom(calm.trace('out'), 30);
    // Visual sign-check dump: bands + cycle so the zero band can be compared to
    // the integrated gold cycle.
    await render('pf_bands_cyc', { ...BASE, show_streamlines: false, show_limit_cycle: true, shading_mode: 0, wind: 0.7 }, 'pf_bands_cyc');
  });

  it('Gradient shading mode differs from Bands and is wind-aware', async () => {
    // Backdrop only (lines off) so we isolate the shading mode.
    const bands = await render('pf_bands', { ...BASE, show_streamlines: false, show_limit_cycle: false, shading_mode: 0 }, 'pf_bands');
    const grad = await render('pf_grad', { ...BASE, show_streamlines: false, show_limit_cycle: false, shading_mode: 1 }, 'pf_grad');
    grad.trace('out').expectDifferentFrom(bands.trace('out'), 30);
    // Matplotlib colormap (Magma) differs from the diverging Bands.
    const magma = await render('pf_magma2', { ...BASE, show_streamlines: false, show_limit_cycle: false, shading_mode: 2 }, 'pf_magma2');
    magma.trace('out').expectDifferentFrom(bands.trace('out'), 20);
    // The Gradient reads the flow field (level-set flow + WIND), so adding wind
    // must change it — Bands (height field, wind-independent) would not.
    const gradWind = await render('pf_grad_wind', { ...BASE, show_streamlines: false, show_limit_cycle: false, shading_mode: 1, wind: 0.9 }, 'pf_grad_wind');
    gradWind.trace('out').expectDifferentFrom(grad.trace('out'), 20);
  });

  it('autopilot drives a live, non-destructive XY override (flow clock frozen)', async () => {
    // flow_speed=0 freezes the arrow/marker animation, so any change across
    // phases is the autopilot epicycle moving the effective XY — never inputs.
    // show_limit_cycle off: the stateful solver random-walks every frame (by
    // design), so we isolate the deterministic backdrop to test autopilot.
    const moving = await runEngineMultiPhaseTest({
      width: 96, height: 96,
      modules: ['com.nano.testonly', 'com.nano.nano'],
      dumpName: 'pf_ap_on',
      phases: [
        {
          commands: [
            { type: 'createSketch', sketchId: 'pf_ap_on',
              sketch: buildSketch({ eccentricity: 0.5, lobedness: 0.5, flow_speed: 0.0,
                                    show_limit_cycle: false, autopilot: true, ap_speed: 1.0 }) },
            { type: 'setTracePoints', tracePoints: [
              { id: 'out', target: { type: 'sketch_output', sketchId: 'pf_ap_on' } },
            ]},
          ],
          waitFrames: 2, captureTraceIds: ['out'],
        },
        { waitFrames: 40, captureTraceIds: ['out'] },
      ],
    });
    expect(moving.success).toBe(true);
    moving.phases[1].trace('out').expectDifferentFrom(moving.phases[0].trace('out'), 20);

    // Control: autopilot off + frozen flow → static across the same span.
    const still = await runEngineMultiPhaseTest({
      width: 96, height: 96,
      modules: ['com.nano.testonly', 'com.nano.nano'],
      dumpName: 'pf_ap_off',
      phases: [
        {
          commands: [
            { type: 'createSketch', sketchId: 'pf_ap_off',
              sketch: buildSketch({ eccentricity: 0.5, lobedness: 0.5, flow_speed: 0.0,
                                    show_limit_cycle: false, autopilot: false }) },
            { type: 'setTracePoints', tracePoints: [
              { id: 'out', target: { type: 'sketch_output', sketchId: 'pf_ap_off' } },
            ]},
          ],
          waitFrames: 2, captureTraceIds: ['out'],
        },
        { waitFrames: 40, captureTraceIds: ['out'] },
      ],
    });
    expect(still.success).toBe(true);
    still.phases[1].trace('out').expectSameAs(still.phases[0].trace('out'), 2);
  });
});
