import { runEngineTest, runEngineMultiPhaseTest } from '@nano/web/test/engine-test-helpers';
import type { Sketch } from '@nano/web/src/sketch-types';

/**
 * E2E coverage for source.sdf.dust_halo (nano bundle) — the first
 * INTERMEDIATE sdf_field stage: consumes an upstream provider (helio),
 * republishes the field with shaped halo dust merged in, rendered by
 * Plume downstream.
 *
 * Determinism notes: halo generation is stateless (pure hash + two
 * accumulated clocks), so with drift 0 + tumble 0 the cloud is a static
 * function of the knobs. The upstream sun is frozen (sim_rate 0, its
 * clock zeroes exactly), so every assertion is a deterministic static
 * diff within or across runs.
 */

const LOOK = {
  orbit: 0.0, morph: 0.0, tilt: 0.1, zoom: 0.25,
  albedo: [0.75, 0.75, 0.8], sun: 0.85, azimuth: 40, elevation: 20,
  bounce: 0, opacity: 1.0, fog: 0.0, room_fog: 0.0,
  radius: 0, ridge_depth: 0,
};

const SUN = { radius: 0.8, relief: 0.6, sim_rate: 0, dust: 0 };
const STILL = { drift: 0, tumble: 0 };

const BG = { r: 178, g: 178, b: 191 };

function haloSketch(halo: Record<string, unknown>,
                    sun: Record<string, unknown> = {},
                    look: Record<string, unknown> = {}): Sketch {
  return {
    anchor: null, wires: [],
    chain: [
      { type: 'module', module_type: 'source.sdf.helio_field',
        instance_key: 'sun@0', params: { ...SUN, ...sun } },
      { type: 'module', module_type: 'source.sdf.dust_halo',
        instance_key: 'halo@0', params: { ...STILL, ...halo } },
      { type: 'module', module_type: 'source.sdf.plume',
        instance_key: 'look@0', params: { ...LOOK, ...look } },
    ],
  } as Sketch;
}

async function run(sketchId: string, halo: Record<string, unknown>,
                   waits: number[], sun: Record<string, unknown> = {},
                   look: Record<string, unknown> = {}) {
  const phases: any[] = [{
    commands: [
      { type: 'createSketch', sketchId, sketch: haloSketch(halo, sun, look) },
      { type: 'setTracePoints', tracePoints: [
        { id: 'out', target: { type: 'sketch_output', sketchId } },
      ]},
    ],
    waitFrames: waits[0], captureTraceIds: ['out'],
  }];
  for (const w of waits.slice(1)) {
    phases.push({ waitFrames: w, captureTraceIds: ['out'] });
  }
  const r = await runEngineMultiPhaseTest({
    width: 96, height: 96,
    modules: ['com.nano.core', 'com.nano.nano'],
    dumpName: sketchId,
    phases,
  });
  expect(r.success).toBe(true);
  return r;
}

describe('source.sdf.dust_halo E2E', () => {
  jest.setTimeout(180000);

  it('adds a beret of motes above the surface', async () => {
    const off = await run('halo_off', { amount: 0 }, [10]);
    const on = await run('halo_on', { amount: 0.8 }, [10]);
    on.phases[0].trace('out').expectDifferentFrom(off.phases[0].trace('out'), 100);
  });

  it('is stateless: a still halo renders pixel-stable', async () => {
    const r = await run('halo_still', { amount: 0.8 }, [10, 30]);
    // drift 0 + tumble 0: both clocks stay at exactly 0, and generation
    // is a pure function of (knobs, clocks) — no pool to evolve.
    r.phases[1].trace('out').expectSameAs(r.phases[0].trace('out'), 1);
  });

  it('drift carries the motes', async () => {
    const r = await run('halo_drift', { amount: 0.8, drift: 0.8 }, [10, 30]);
    // The drift clock runs on wall dt (the sun stays frozen — its own
    // clock is sim-scaled): the same cloud, orbited between captures.
    r.phases[1].trace('out').expectDifferentFrom(r.phases[0].trace('out'), 60);
  });

  it('a ring is not a beret', async () => {
    const beret = await run('halo_beret', { amount: 0.7 }, [10]);
    const ring = await run('halo_ring',
      { amount: 0.7, arc: 1, width: 0.05, gap: 0.6, thick: 1, tilt: 0.35 },
      [10]);
    ring.phases[0].trace('out').expectDifferentFrom(beret.phases[0].trace('out'), 100);
  });

  it('merges upstream dust instead of replacing it', async () => {
    const haloOnly = await run('halo_merge_h', { amount: 0.6 }, [10]);
    const both = await run('halo_merge_b', { amount: 0.6 }, [10],
                           { dust: 0.9 });
    // Upstream motes must survive the relay: with the halo unchanged,
    // turning the sun's own dust on still changes the frame.
    both.phases[0].trace('out').expectDifferentFrom(haloOnly.phases[0].trace('out'), 60);
  });

  it('halo dust survives the fog pipeline and thickens it', async () => {
    const off = await run('halo_fog_off', { amount: 0 }, [10], {},
                          { fog: 0.4 });
    const on = await run('halo_fog_on', { amount: 0.9, thick: 0.6 }, [10], {},
                         { fog: 0.4 });
    // Pins the full soft path: halo motes fold into the published grid's
    // .a, and the splat runs inside the scene-buffer fog route.
    on.phases[0].trace('out').expectDifferentFrom(off.phases[0].trace('out'), 100);
  });

  it('passes video through untouched when nothing consumes the rail', async () => {
    const r = await runEngineTest({
      width: 96, height: 96,
      modules: ['com.nano.core', 'com.nano.nano'],
      commands: [
        { type: 'createSketch', sketchId: 'halo_pass', sketch: {
          anchor: null, wires: [],
          chain: [
            { type: 'module', module_type: 'source.solid_color',
              instance_key: 'bg@0', params: { color: [0.7, 0.7, 0.75] } },
            { type: 'module', module_type: 'source.sdf.dust_halo',
              instance_key: 'halo@0', params: {} },
          ],
        } as Sketch },
        { type: 'setTracePoints', tracePoints: [
          { id: 'out', target: { type: 'sketch_output', sketchId: 'halo_pass' } },
        ]},
      ],
      waitFrames: 6,
      captureTraceIds: ['out'],
      dumpName: 'halo_pass',
    });
    expect(r.success).toBe(true);
    for (const [x, y] of [[3, 3], [48, 48], [92, 92]]) {
      r.trace('out').expectPixelAt(x, y, BG, 2);
    }
  });
});
