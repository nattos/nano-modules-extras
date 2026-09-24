import { runEngineTest, runEngineMultiPhaseTest } from '@nano/web/test/engine-test-helpers';
import type { Sketch } from '@nano/web/src/sketch-types';

/**
 * E2E coverage for source.brutal_fold (nano bundle) — the brutalist
 * axonometric-prism generator. A baked control surface is resolved on the CPU
 * to two structures' terms (the solid threshold is CPU-resolved too), then a
 * single present compute pass composites the receding prism layers with depth
 * fog into a grayscale image.
 *
 * Under test:
 *  1. Registers + renders: a busy cell produces a structured (non-solid) field,
 *     and the output is grayscale (R≈G≈B — the fog only lifts toward the sky
 *     tone, no chroma).
 *  2. balance plays the two co-folded structures' parallax against each other —
 *     it changes the output.
 *  3. autopilot is a live, non-destructive override: with the time clock frozen
 *     (time_speed=0) the output STILL animates across frames when autopilot is
 *     on (the epicycle moves the effective XY) and is STATIC when it's off.
 *  4. Broadcast wiring: autopilot_x / autopilot_y are declared as data_output
 *     fields (kind=2) — the channel the custom XY-pad editor reads live.
 *
 * The generator ignores its input, so the chain is just
 * texture_input → brutal_fold → texture_output.
 */
function buildSketch(params: Record<string, unknown>): Sketch {
  return {
    anchor: null,
    chain: [
      {
        type: 'module',
        module_type: 'source.brutal_fold',
        instance_key: 'bf@0',
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

// A busy cell, frozen in time for determinism.
const BUSY = { complexity: 0.75, order: 0.35, time_speed: 0.0 };

describe('source.brutal_fold E2E', () => {
  jest.setTimeout(60000);

  it('registers and renders a structured grayscale field', async () => {
    const result = await render('bf_smoke', BUSY, 'bf_smoke');
    // The composite produced a structured (non-uniform) field.
    result.trace('out').expectNotSolidColor({ r: 0, g: 0, b: 0 }, 5);

    // Grayscale: every pixel is achromatic (the fog lifts toward the sky tone,
    // never adds chroma). Tolerate rounding to rgba8.
    let maxChroma = 0;
    result.trace('out').forEachPixel((c: any) => {
      const chroma = Math.max(Math.abs(c.r - c.g), Math.abs(c.g - c.b), Math.abs(c.r - c.b));
      if (chroma > maxChroma) maxChroma = chroma;
    });
    expect(maxChroma).toBeLessThanOrEqual(4);

    // Registration: the effect is present with its broadcast outputs declared.
    const bf = result.state.plugins.find((p: any) => p.id === 'source.brutal_fold');
    expect(bf).toBeTruthy();
    expect(bf.io.find((io: any) => io.name === 'autopilot_x' && io.kind === 2)).toBeTruthy();
    expect(bf.io.find((io: any) => io.name === 'autopilot_y' && io.kind === 2)).toBeTruthy();
  });

  it('balance plays the two structures parallax (changes the output)', async () => {
    const a = await render('bf_bal0', { ...BUSY, balance: 0.0 }, 'bf_bal0');
    const b = await render('bf_bal1', { ...BUSY, balance: 1.0 }, 'bf_bal1');
    b.trace('out').expectDifferentFrom(a.trace('out'), 30);
  });

  // Max per-pixel chroma (channel spread) over lit pixels — 0 = pure grayscale.
  const maxChroma = (trace: any): number => {
    let m = 0;
    trace.forEachPixel((c: any) => {
      const chroma = Math.max(Math.abs(c.r - c.g), Math.abs(c.g - c.b), Math.abs(c.r - c.b));
      if (chroma > m) m = chroma;
    });
    return m;
  };

  it('diffuse tint colours the panels (and 0 stays grayscale)', async () => {
    // diff_sat=0 → grayscale; cranked → the panel tones pick up the graded hues.
    const gray = await render('bf_diff0', { ...BUSY, diff_sat: 0.0 }, 'bf_diff0');
    expect(maxChroma(gray.trace('out'))).toBeLessThanOrEqual(4);
    const tinted = await render('bf_diff1',
      { ...BUSY, diff_sat: 1.0, diff_hue_lo: 0.6, diff_hue_mid: 0.05, diff_hue_hi: 0.12 },
      'bf_diff1');
    expect(maxChroma(tinted.trace('out'))).toBeGreaterThan(30);
    tinted.trace('out').expectDifferentFrom(gray.trace('out'), 40);
  });

  it('fog tint colours the atmosphere by depth', async () => {
    // fog_sat cranked → the depth-faded fog (incl. the background) takes a hue,
    // while diffuse stays gray (diff_sat=0). The frame gains chroma.
    const gray = await render('bf_fog0', { ...BUSY, fog: 1.5, fog_sat: 0.0 }, 'bf_fog0');
    const tinted = await render('bf_fog1',
      { ...BUSY, fog: 1.5, fog_sat: 1.0, fog_hue_lo: 0.0, fog_hue_mid: 0.5, fog_hue_hi: 0.7 },
      'bf_fog1');
    expect(maxChroma(tinted.trace('out'))).toBeGreaterThan(20);
    tinted.trace('out').expectDifferentFrom(gray.trace('out'), 30);
  });

  it('volumetric blob shapes the fog (differs from uniform depth fog)', async () => {
    // vol_amount=0 → uniform depth fog; cranked with a tight sphere → the fog is
    // concentrated into the blob, so the frame changes.
    const uniform = await render('bf_vol0', { ...BUSY, fog: 1.4, vol_amount: 0.0 }, 'bf_vol0');
    const blob = await render('bf_vol1',
      { ...BUSY, fog: 1.4, vol_amount: 1.0, vol_shape: 1.0, vol_radius: 0.35,
        vol_anchor_x: -0.4, vol_z: 0.5 },
      'bf_vol1');
    blob.trace('out').expectDifferentFrom(uniform.trace('out'), 30);
  });

  it('vol_depth controls Z selectivity (thin slice differs from broad)', async () => {
    // Same blob centred at z=0.5; a thin Z extent only fogs the mid slices, a
    // broad one fogs across depth → the two frames differ.
    const base = { ...BUSY, fog: 1.4, vol_amount: 1.0, vol_shape: 1.0,
                   vol_radius: 0.4, vol_softness_xy: 0.05, vol_softness_z: 0.05, vol_z: 0.5 };
    const thin = await render('bf_depth_thin', { ...base, vol_depth: 0.1 }, 'bf_depth_thin');
    const broad = await render('bf_depth_broad', { ...base, vol_depth: 0.95 }, 'bf_depth_broad');
    thin.trace('out').expectDifferentFrom(broad.trace('out'), 20);
  });

  it('renders with drift enabled and declares the live blob broadcasts', async () => {
    // Drift is a wall-clock-RATE bounded random walk on the blob — it advances in
    // real time (verified live in the app). The engine harness steps frames
    // flat-out (negligible real dt), so the per-frame motion is sub-pixel here;
    // rather than assert pixel motion the harness can't produce, verify the
    // effect renders cleanly with drift params set AND that the live (drifted)
    // blob values are broadcast as data outputs — the channel the fog-preview
    // widget reads to show the drift.
    const result = await render('bf_drift',
      { ...BUSY, vol_amount: 1.0, drift_xy: 1.0, drift_z: 0.5, drift_shape: 0.5,
        drift_angle: 0.5, drift_speed: 1.0 }, 'bf_drift');
    result.trace('out').expectNotSolidColor({ r: 0, g: 0, b: 0 }, 5);
    const bf = result.state.plugins.find((p: any) => p.id === 'source.brutal_fold');
    for (const name of ['vol_x_live', 'vol_y_live', 'vol_z_live', 'vol_shape_live', 'vol_angle_live']) {
      expect(bf.io.find((io: any) => io.name === name && io.kind === 2)).toBeTruthy();
    }
  });

  it('autopilot drives a live, non-destructive XY override (clock frozen)', async () => {
    // time_speed=0 freezes the loop, so any change across phases is the
    // autopilot epicycle moving the effective XY — never the inputs.
    const moving = await runEngineMultiPhaseTest({
      width: 96, height: 96,
      modules: ['com.nano.testonly', 'com.nano.nano'],
      dumpName: 'bf_ap_on',
      phases: [
        {
          commands: [
            { type: 'createSketch', sketchId: 'bf_ap_on',
              sketch: buildSketch({ complexity: 0.5, order: 0.5, time_speed: 0.0,
                                    autopilot: true, ap_speed: 1.0 }) },
            { type: 'setTracePoints', tracePoints: [
              { id: 'out', target: { type: 'sketch_output', sketchId: 'bf_ap_on' } },
            ]},
          ],
          waitFrames: 2, captureTraceIds: ['out'],
        },
        { waitFrames: 40, captureTraceIds: ['out'] },
      ],
    });
    expect(moving.success).toBe(true);
    moving.phases[1].trace('out').expectDifferentFrom(moving.phases[0].trace('out'), 20);

    // Control: autopilot off + frozen clock → static across the same span.
    const still = await runEngineMultiPhaseTest({
      width: 96, height: 96,
      modules: ['com.nano.testonly', 'com.nano.nano'],
      dumpName: 'bf_ap_off',
      phases: [
        {
          commands: [
            { type: 'createSketch', sketchId: 'bf_ap_off',
              sketch: buildSketch({ complexity: 0.5, order: 0.5, time_speed: 0.0,
                                    autopilot: false }) },
            { type: 'setTracePoints', tracePoints: [
              { id: 'out', target: { type: 'sketch_output', sketchId: 'bf_ap_off' } },
            ]},
          ],
          waitFrames: 2, captureTraceIds: ['out'],
        },
        { waitFrames: 40, captureTraceIds: ['out'] },
      ],
    });
    expect(still.success).toBe(true);
    still.phases[1].trace('out').expectSameAs(still.phases[0].trace('out'), 2); // frozen → identical
  });

  // Key-moment mode renders from the SEPARATE 32×32 KM atlas (its own scenes +
  // curated windows). This XY snaps to a scored, non-poppy cell (peak ≈ 0.33)
  // that animates across its window. liveliness is ignored in KM mode.
  const KM_CELL = { complexity: 0.548, order: 0.129, anim_amount: 1.5 };

  it('key moment mode scrubs the curated window (Time mode)', async () => {
    // Time mode maps km_time directly onto the window playhead (0 = window start,
    // 1 = settled on the centre peak) — deterministic, no dt. The two ends of the
    // window render different frames.
    const start = await render('bf_km_t0',
      { ...KM_CELL, key_moment: true, km_time_mode: 1, km_time: 0.0 }, 'bf_km_t0');
    const peak = await render('bf_km_t1',
      { ...KM_CELL, key_moment: true, km_time_mode: 1, km_time: 1.0 }, 'bf_km_t1');
    start.trace('out').expectNotSolidColor({ r: 0, g: 0, b: 0 }, 5);
    peak.trace('out').expectDifferentFrom(start.trace('out'), 20);

    // The live playhead is broadcast for the editor readout.
    const bf = peak.state.plugins.find((p: any) => p.id === 'source.brutal_fold');
    expect(bf.io.find((io: any) => io.name === 'km_phase' && io.kind === 2)).toBeTruthy();
  });

  it('key moment mode renders a different scene than continuous mode', async () => {
    // Same XY, KM off vs on → the KM atlas has different scenes, so the frame
    // changes (proves the atlas swap, not just a time shift).
    const cont = await render('bf_km_off', { ...KM_CELL, key_moment: false, time_speed: 0 }, 'bf_km_off');
    const kmf = await render('bf_km_on',
      { ...KM_CELL, key_moment: true, km_time_mode: 1, km_time: 0.8 }, 'bf_km_on');
    kmf.trace('out').expectDifferentFrom(cont.trace('out'), 30);
  });

  it('sky threshold snaps away from an unreachable (low-sky) cell', async () => {
    // This XY lands on a low-sky cell; raising the threshold drops it and snaps to
    // the nearest reachable cell, changing the rendered scene.
    const base = { complexity: 0.548, order: 0.839, key_moment: true, km_time_mode: 1, km_time: 0.8 };
    const keep = await render('bf_sky0', { ...base, sky_threshold: 0.0 }, 'bf_sky0');
    const snap = await render('bf_sky1', { ...base, sky_threshold: 0.5 }, 'bf_sky1');
    snap.trace('out').expectDifferentFrom(keep.trace('out'), 20);
  });

  it('key moment Loop mode advances the playhead over time', async () => {
    // Loop replays the window over km_duration seconds, so across a span of frames
    // (real dt, like the autopilot test) the output moves. Short duration → the
    // window advances clearly within the captured frames.
    const r = await runEngineMultiPhaseTest({
      width: 96, height: 96,
      modules: ['com.nano.testonly', 'com.nano.nano'],
      dumpName: 'bf_km_loop',
      phases: [
        {
          commands: [
            { type: 'createSketch', sketchId: 'bf_km_loop',
              sketch: buildSketch({ ...KM_CELL, key_moment: true, km_time_mode: 2, km_duration: 0.4 }) },
            { type: 'setTracePoints', tracePoints: [
              { id: 'out', target: { type: 'sketch_output', sketchId: 'bf_km_loop' } },
            ]},
          ],
          waitFrames: 2, captureTraceIds: ['out'],
        },
        { waitFrames: 40, captureTraceIds: ['out'] },
      ],
    });
    expect(r.success).toBe(true);
    r.phases[1].trace('out').expectDifferentFrom(r.phases[0].trace('out'), 20);
  });

  it('key moment Trigger mode holds until triggered, then plays the window', async () => {
    // Trigger is a one-shot: parked at the window start until a rising edge on
    // km_trigger fires it; it then plays out to the settled peak and holds.
    const r = await runEngineMultiPhaseTest({
      width: 96, height: 96,
      modules: ['com.nano.testonly', 'com.nano.nano'],
      dumpName: 'bf_km_trig',
      phases: [
        {
          commands: [
            { type: 'createSketch', sketchId: 'bf_km_trig',
              sketch: buildSketch({ ...KM_CELL, key_moment: true, km_time_mode: 0, km_duration: 0.4 }) },
            { type: 'setTracePoints', tracePoints: [
              { id: 'out', target: { type: 'sketch_output', sketchId: 'bf_km_trig' } },
            ]},
          ],
          waitFrames: 3, captureTraceIds: ['out'],
        },
        // No trigger yet: the one-shot stays parked at the window start.
        { waitFrames: 30, captureTraceIds: ['out'] },
        // Fire the trigger: the window plays out toward the peak.
        {
          commands: [
            { type: 'setParam', sketchId: 'bf_km_trig', chainIdx: 0, paramKey: 'km_trigger', value: 1 },
          ],
          waitFrames: 40, captureTraceIds: ['out'],
        },
      ],
    });
    expect(r.success).toBe(true);
    r.phases[1].trace('out').expectSameAs(r.phases[0].trace('out'), 2);        // parked → no auto-advance
    r.phases[2].trace('out').expectDifferentFrom(r.phases[0].trace('out'), 20); // triggered → played
  });
});
