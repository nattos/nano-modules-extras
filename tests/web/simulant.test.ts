import { runEngineTest } from '@nano/web/test/engine-test-helpers';
import type { Sketch } from '@nano/web/src/sketch-types';

/**
 * E2E for filter.sim.simulant (nano bundle) — the faithful port of the Resolume
 * Wire "Simulant" patch. It is a DIFFERENCE-BLEND + BLUR-DIFFUSION feedback loop
 * (abs(fadedPrev - input), diffused by a per-frame blur) whose churning
 * accumulator is traced into Sobel lines.
 *
 * The load-bearing FAITHFUL QUIRK: with the stock knobs the flicker envelope is
 * SUBTRACTED (invert off) and Flicker Min/Max = 0, so the injection opacity
 * clamps to 0 — a fresh drop just decays to a uniform field with NO edges (black
 * output). Turning up Const Alpha injects the input, the difference-feedback
 * churns, and Sobel lines appear. Both behaviors are asserted here.
 */

describe('filter.sim.simulant E2E', () => {
  jest.setTimeout(60000);

  const MODULES = ['com.nano.testonly', 'com.nano.nano'];
  const W = 128, H = 128;

  // A localized white rect on black; `speed` optionally makes it move (which
  // supplies the per-frame CHANGE the difference-feedback reacts to).
  const rectSketch = (pr: Record<string, unknown>, rectSpeed = 0.0): Sketch => ({
    anchor: null,
    chain: [
      { type: 'module', module_type: 'source.solid_color', instance_key: 'bg@0',
        params: { color: [0, 0, 0] } },
      { type: 'module', module_type: 'debug.motion_rect', instance_key: 'rc@0',
        params: { size: 0.25, speed: rectSpeed, color: [1, 1, 1] } },
      { type: 'module', module_type: 'filter.sim.simulant', instance_key: 'sm@0', params: pr },
    ],
  } as Sketch);

  const run = (id: string, sketch: Sketch, waitFrames = 30) =>
    runEngineTest({
      width: W, height: H,
      modules: MODULES,
      commands: [
        { type: 'createSketch', sketchId: id, sketch },
        { type: 'setTracePoints', tracePoints: [
          { id: 'out', target: { type: 'sketch_output', sketchId: id } },
        ]},
      ],
      waitFrames, captureTraceIds: ['out'], dumpName: id,
    });

  it('stock defaults decay to a lineless field (the faithful quirk)', async () => {
    // Flicker env subtracted + Min/Max 0 + Const Alpha 0 → injection clamps to 0
    // → the accumulator creeps to a uniform field → Sobel finds no edges → black.
    const off = await run('sm_default', rectSketch({
      const_alpha: 0.0, flicker_min: 0.0, flicker_max: 0.0, flicker_invert: false,
    }));
    expect(off.success).toBe(true);
    // Essentially no bright line pixels.
    off.trace('out').expectCoverage(c => c.r > 120 && c.g > 120 && c.b > 120, { max: 0.002 });
  });

  it('flicker pulses drive a static image into expanding Sobel contours', async () => {
    // The canonical static-image path: flicker (Invert on so the env ADDS) pulses
    // the input into the retained feedback; each pulse diffuses outward as a clean
    // expanding ring → the signature Simulant concentric contours.
    const on = await run('sm_flick_on', rectSketch({
      const_alpha: 0.0, flicker_max: 0.6, flicker_invert: true, flicker_rate: 0.5,
      wave_speed: 0.5, choke: 0.0, line_strength: 0.8, levels: 0.3,
    }));
    const dead = await run('sm_flick_off', rectSketch({
      const_alpha: 0.0, flicker_min: 0.0, flicker_max: 0.0, flicker_invert: false, flicker_rate: 0.5,
    }));
    expect(on.success && dead.success).toBe(true);
    on.trace('out').expectDifferentFrom(dead.trace('out'), 100);
    on.trace('out').expectCoverage(c => c.r > 150 && c.g > 150 && c.b > 150, { min: 0.003 });
  });

  it('Const Alpha turns a MOVING input into contour lines', async () => {
    // Steady Const Alpha reaches a smooth equilibrium on a static image (no
    // edges), but a MOVING input supplies per-frame change the difference-
    // feedback reacts to → contours appear.
    // Flicker disabled to isolate the const_alpha + motion mechanism.
    const noFlick = { flicker_min: 0.0, flicker_max: 0.0 };
    const still = await run('sm_move_still', rectSketch({ ...noFlick, const_alpha: 0.6, wave_speed: 0.5 }, 0.0));
    const moving = await run('sm_move_on', rectSketch({
      ...noFlick, const_alpha: 0.6, wave_speed: 0.5, choke: 0.0, line_strength: 0.8, levels: 0.3,
    }, 0.4));
    expect(still.success && moving.success).toBe(true);
    moving.trace('out').expectDifferentFrom(still.trace('out'), 50);
    moving.trace('out').expectCoverage(c => c.r > 150 && c.g > 150 && c.b > 150, { min: 0.002 });
  });
});
