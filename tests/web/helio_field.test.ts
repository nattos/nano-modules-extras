import { runEngineTest, runEngineMultiPhaseTest } from '@nano/web/test/engine-test-helpers';
import type { Sketch } from '@nano/web/src/sketch-types';

/**
 * E2E coverage for source.sdf.helio_field (nano bundle) — the simulated-sun
 * SDF provider (2D MHD-lite: fluid + magnetic potential on an oct map,
 * self-igniting storms), rendered through Plume via the sdf_field rail.
 *
 * Determinism notes: the sim clock is wall-time dt, so frame content is NOT
 * reproducible across separate runs once the sim is moving — assertions
 * compare frames WITHIN one run, except the frozen test (sim_rate 0 zeroes
 * the sim dt exactly, so nothing advances) and the variation test (different
 * seeds diverge far beyond any timing wobble). Plume look: orbit/morph 0
 * freeze the camera accumulators and bounce 0 avoids the known GI wall-clock
 * jitter, so a frozen field renders pixel-stable.
 */

const LOOK = {
  orbit: 0.0, morph: 0.0, tilt: 0.1, zoom: 0.25,
  albedo: [0.75, 0.75, 0.8], sun: 0.85, azimuth: 40, elevation: 20,
  bounce: 0, opacity: 1.0, fog: 0.0, room_fog: 0.0,
  radius: 0, ridge_depth: 0,
};

const BG = { r: 178, g: 178, b: 191 };
const lum = (c: { r: number, g: number, b: number }) => (c.r + c.g + c.b) / 3;

function helioSketch(params: Record<string, unknown>,
                     look: Record<string, unknown> = {}): Sketch {
  return {
    anchor: null, wires: [],
    chain: [
      { type: 'module', module_type: 'source.sdf.helio_field',
        instance_key: 'sun@0', params },
      { type: 'module', module_type: 'source.sdf.plume',
        instance_key: 'look@0', params: { ...LOOK, ...look } },
    ],
  } as Sketch;
}

function phasesFor(sketchId: string, params: Record<string, unknown>,
                   waits: number[],
                   look: Record<string, unknown> = {}): any[] {
  const phases: any[] = [{
    commands: [
      { type: 'createSketch', sketchId, sketch: helioSketch(params, look) },
      { type: 'setTracePoints', tracePoints: [
        { id: 'out', target: { type: 'sketch_output', sketchId } },
      ]},
    ],
    waitFrames: waits[0], captureTraceIds: ['out'],
  }];
  for (const w of waits.slice(1)) {
    phases.push({ waitFrames: w, captureTraceIds: ['out'] });
  }
  return phases;
}

async function run(sketchId: string, params: Record<string, unknown>,
                   waits: number[], look: Record<string, unknown> = {}) {
  const r = await runEngineMultiPhaseTest({
    width: 96, height: 96,
    modules: ['com.nano.core', 'com.nano.nano'],
    dumpName: sketchId,
    phases: phasesFor(sketchId, params, waits, look),
  });
  expect(r.success).toBe(true);
  return r;
}

describe('source.sdf.helio_field E2E', () => {
  jest.setTimeout(180000);

  it('renders a living sun through the rail', async () => {
    const r = await run('helio_live', { radius: 0.8, relief: 0.6 },
                        [10, 180, 180]);
    // Body renders at center; the frame corner stays off-body.
    const late = r.phases[2].trace('out');
    expect(lum(late.pixelAt(48, 48))).toBeGreaterThan(25);
    expect(late.pixelAt(48, 48).a).toBe(255);
    // The simulation moves the surface: the line relief shifts.
    late.expectDifferentFrom(r.phases[1].trace('out'), 100);
    r.phases[1].trace('out').expectDifferentFrom(r.phases[0].trace('out'), 100);
  });

  it('freezes exactly at Sim Rate 0', async () => {
    const r = await run('helio_frozen',
      { radius: 0.8, relief: 0.6, sim_rate: 0 }, [10, 60]);
    // sim dt is exactly zero: the initial field renders pixel-stable
    // (also the determinism smoke: no wall-clock leaks anywhere).
    r.phases[1].trace('out').expectSameAs(r.phases[0].trace('out'), 1);
  });

  it('variation reseeds the sun', async () => {
    const a = await run('helio_var_a',
      { radius: 0.8, relief: 0.6, variation: 0.15 }, [30]);
    const b = await run('helio_var_b',
      { radius: 0.8, relief: 0.6, variation: 0.85 }, [30]);
    // Different seed = a different initial multipole = a different sun.
    b.phases[0].trace('out').expectDifferentFrom(a.phases[0].trace('out'), 500);
  });

  it('granules add relief in the flatlands', async () => {
    // Frozen sim (deterministic): the granule chemistry's seeded initial
    // state still renders as bumps, so on-vs-off is a pure static diff.
    const off = await run('helio_grain_off',
      { radius: 0.8, relief: 0.6, sim_rate: 0, granules: 0 }, [10]);
    const on = await run('helio_grain_on',
      { radius: 0.8, relief: 0.6, sim_rate: 0, granules: 1 }, [10]);
    on.phases[0].trace('out').expectDifferentFrom(off.phases[0].trace('out'), 200);
  });

  it('dust motes render sharp above the surface', async () => {
    // Frozen sim: mote placement is a pure function of the (static)
    // chemistry + seed, so on-vs-off is a deterministic static diff.
    const off = await run('helio_dust_off',
      { radius: 0.8, relief: 0.6, sim_rate: 0, dust: 0 }, [10]);
    const on = await run('helio_dust_on',
      { radius: 0.8, relief: 0.6, sim_rate: 0, dust: 0.9 }, [10]);
    on.phases[0].trace('out').expectDifferentFrom(off.phases[0].trace('out'), 100);
  });

  it('dust survives the fog pipeline', async () => {
    // With fog on, the splat runs inside the scene-buffer route (march →
    // dust → fog → composite) — this pins that integration: dust must
    // still land on screen through the fog composite.
    const off = await run('helio_dustfog_off',
      { radius: 0.8, relief: 0.6, sim_rate: 0, dust: 0 }, [10], { fog: 0.4 });
    const on = await run('helio_dustfog_on',
      { radius: 0.8, relief: 0.6, sim_rate: 0, dust: 0.9 }, [10], { fog: 0.4 });
    on.phases[0].trace('out').expectDifferentFrom(off.phases[0].trace('out'), 100);
  });

  it('self-resonant storms keep the surface churning', async () => {
    const r = await run('helio_storms',
      { radius: 0.8, relief: 0.5, sim_rate: 0.7, excite: 0.95,
        storm_h: 0.9, glow: 1.0 }, [10, 300, 200, 200]);
    // High Excitability: storms self-ignite and keep burning — every
    // late sample differs substantially from the previous one.
    r.phases[2].trace('out').expectDifferentFrom(r.phases[1].trace('out'), 200);
    r.phases[3].trace('out').expectDifferentFrom(r.phases[2].trace('out'), 200);
  });

  it('passes video through untouched when nothing consumes the rail', async () => {
    const r = await runEngineTest({
      width: 96, height: 96,
      modules: ['com.nano.core', 'com.nano.nano'],
      commands: [
        { type: 'createSketch', sketchId: 'helio_pass', sketch: {
          anchor: null, wires: [],
          chain: [
            { type: 'module', module_type: 'source.solid_color',
              instance_key: 'bg@0', params: { color: [0.7, 0.7, 0.75] } },
            { type: 'module', module_type: 'source.sdf.helio_field',
              instance_key: 'sun@0', params: {} },
          ],
        } as Sketch },
        { type: 'setTracePoints', tracePoints: [
          { id: 'out', target: { type: 'sketch_output', sketchId: 'helio_pass' } },
        ]},
      ],
      waitFrames: 6,
      captureTraceIds: ['out'],
      dumpName: 'helio_pass',
    });
    expect(r.success).toBe(true);
    for (const [x, y] of [[3, 3], [48, 48], [92, 92]]) {
      r.trace('out').expectPixelAt(x, y, BG, 2);
    }
  });
});
