import * as fs from 'fs';
import * as path from 'path';
import * as zlib from 'zlib';
import { runEngineTest, runEngineMultiPhaseTest } from '@nano/web/test/engine-test-helpers';
import { runGpuEffectTest, forEachBackend } from '@nano/web/test/gpu-test-helpers';
import type { Sketch } from '@nano/web/src/sketch-types';

/**
 * E2E coverage for source.die_fold (nano bundle) — the endless fake-chip
 * generator ported from the die-fold prototype. One compute pass draws the
 * whole die from a packed parameter buffer the CPU samples out of a 15x15
 * parameter atlas each frame.
 *
 * Under test:
 *  1. Registers + renders; its broadcasts (pad_x/pad_y/phase) are data outputs.
 *  2. PARITY with the prototype, on WebGPU AND natively on Metal: frames
 *     rendered by the prototype's own Python (`effects/die_fold/gen_refs.py` →
 *     fixtures/die_fold) at atlas cells, several depths, an off-origin camera
 *     and a second seed, against the gate the prototype's WebGPU port was held
 *     to (mean ≤ 2/255, ≤ 2% past 24/255).
 *  3. The controls do what they say: seed, look, the pad, a non-square frame.
 *  4. The descent loop closes: depth 0 and depth → 1 are (nearly) one frame.
 *  5. Fly owns the pad and the camera; engaging the orbit is not a jump.
 *
 * Speed 0 freezes every clock, so each render is deterministic.
 */
function buildSketch(params: Record<string, unknown>): Sketch {
  return {
    anchor: null,
    chain: [
      {
        type: 'module',
        module_type: 'source.die_fold',
        instance_key: 'df@0',
        params,
      },
    ],
  };
}

async function render(sketchId: string, params: Record<string, unknown>,
                      width = 96, height = 96) {
  const result = await runEngineTest({
    width, height,
    modules: ['com.nano.testonly', 'com.nano.nano'],
    commands: [
      { type: 'createSketch', sketchId, sketch: buildSketch({ speed: 0, ...params }) },
      { type: 'setTracePoints', tracePoints: [
        { id: 'out', target: { type: 'sketch_output', sketchId } },
      ]},
    ],
    waitFrames: 6,
    captureTraceIds: ['out'],
    dumpName: sketchId,
  });
  expect(result.success).toBe(true);
  return result;
}

/** Mean |Δ| over RGB (0..255) and the fraction of channels past 24. */
function compare(a: any, b: any): { mean: number; tail: number } {
  let sum = 0, tail = 0, n = 0;
  const bp: { r: number; g: number; b: number }[] = [];
  b.forEachPixel((c: any) => bp.push(c));
  let i = 0;
  a.forEachPixel((c: any) => {
    const o = bp[i++];
    for (const k of ['r', 'g', 'b'] as const) {
      const d = Math.abs(c[k] - o[k]);
      sum += d;
      if (d > 24) tail++;
      n++;
    }
  });
  return { mean: sum / n, tail: tail / n };
}

const FIXTURES = path.join(__dirname, 'fixtures', 'die_fold');

describe('source.die_fold E2E', () => {
  jest.setTimeout(90000);

  it('registers and renders a die', async () => {
    const result = await render('df_smoke', {});
    result.trace('out').expectNotSolidColor({ r: 0, g: 0, b: 0 }, 5);
    // Structure, not a wash: plenty of distinct values in the frame.
    const seen = new Set<number>();
    result.trace('out').forEachPixel((c: any) => seen.add((c.r << 16) | (c.g << 8) | c.b));
    expect(seen.size).toBeGreaterThan(200);

    const df = result.state.plugins.find((p: any) => p.id === 'source.die_fold');
    expect(df).toBeTruthy();
    for (const name of ['pad_x', 'pad_y', 'phase']) {
      expect(df.io.find((io: any) => io.name === name && io.kind === 2)).toBeTruthy();
    }
  });

  it('the seed re-rolls the die', async () => {
    const a = await render('df_seed_a', { seed: 11 });
    const b = await render('df_seed_b', { seed: 12 });
    b.trace('out').expectDifferentFrom(a.trace('out'), 500);
  });

  it('look runs from layout to micrograph', async () => {
    const lay = await render('df_look_lo', { look: -1 });
    const sem = await render('df_look_hi', { look: 1 });
    sem.trace('out').expectDifferentFrom(lay.trace('out'), 500);
    // The micrograph is grayscale.
    let chroma = 0;
    sem.trace('out').forEachPixel((c: any) => { chroma += Math.max(c.r, c.g, c.b) - Math.min(c.r, c.g, c.b); });
    expect(chroma / (96 * 96)).toBeLessThan(6);
  });

  it('the pad moves through the fold', async () => {
    const a = await render('df_pad_a', { grain: 0.1, netness: 0.2 });
    const b = await render('df_pad_b', { grain: 0.9, netness: 0.85 });
    b.trace('out').expectDifferentFrom(a.trace('out'), 500);
  });

  it('covers a non-square viewport (no letterbox bars)', async () => {
    const result = await render('df_wide', {}, 128, 64);
    // Under cover both edge bands carry structure, not one flat color.
    const left = new Set<number>(), right = new Set<number>();
    result.trace('out').forEachPixel((c: any, x: number) => {
      const key = (c.r << 16) | (c.g << 8) | c.b;
      if (x < 6) left.add(key);
      if (x >= 122) right.add(key);
    });
    expect(left.size).toBeGreaterThan(20);
    expect(right.size).toBeGreaterThan(20);
  });

  it('the descent loop closes', async () => {
    const start = await render('df_loop_0', { depth: 0 });
    const end = await render('df_loop_1', { depth: 0.9999 });
    const mid = await render('df_loop_h', { depth: 0.5 });
    const seam = compare(start.trace('out'), end.trace('out'));
    const away = compare(start.trace('out'), mid.trace('out'));
    expect(seam.mean).toBeLessThan(away.mean * 0.15);
  });

  it('fly owns the pad and the camera', async () => {
    const parked = await render('df_fly_off', { fly: false });
    const flyA = await render('df_fly_a', { fly: true, grain: 0.1, netness: 0.1 });
    const flyB = await render('df_fly_b', { fly: true, grain: 0.9, netness: 0.9 });
    // The path's camera is not the parked one...
    flyA.trace('out').expectDifferentFrom(parked.trace('out'), 500);
    // ...and while it flies, the pad inputs are not what's drawn.
    flyB.trace('out').expectSameAs(flyA.trace('out'), 2);
  });

  it('engaging the orbit is not a jump; it then moves the camera', async () => {
    const off = await render('df_orbit_off', { orbit: false });
    const on = await render('df_orbit_on', { orbit: true, orb_span: 6 });
    on.trace('out').expectSameAs(off.trace('out'), 2);

    // Running, the orbit carries the camera away from where it engaged.
    const r = await runEngineMultiPhaseTest({
      width: 96, height: 96,
      modules: ['com.nano.testonly', 'com.nano.nano'],
      dumpName: 'df_orbit_run',
      phases: [
        {
          commands: [
            { type: 'createSketch', sketchId: 'df_orbit_run',
              sketch: buildSketch({ speed: 0.6, slowness: 1, orbit: true, orb_span: 6 }) },
            { type: 'setTracePoints', tracePoints: [
              { id: 'out', target: { type: 'sketch_output', sketchId: 'df_orbit_run' } },
            ]},
          ],
          waitFrames: 2, captureTraceIds: ['out'],
        },
        { waitFrames: 30, captureTraceIds: ['out'] },
      ],
    });
    expect(r.success).toBe(true);
    r.phases[1].trace('out').expectDifferentFrom(r.phases[0].trace('out'), 500);
  });
});

// The prototype's own Python is the reference on both backends: the HLSL goes
// to WGSL through naga on one and to MSL through SPIRV-Cross on the other.
forEachBackend((backend) => {
  describe(`source.die_fold parity with the prototype (${backend})`, () => {
    jest.setTimeout(90000);
    const manifest = JSON.parse(fs.readFileSync(path.join(FIXTURES, 'manifest.json'), 'utf8'));
    for (const c of manifest.cases) {
      it(`matches the Python reference: ${c.name}`, async () => {
        const res = manifest.res as number;
        const ref = zlib.gunzipSync(fs.readFileSync(path.join(FIXTURES, `${c.name}.rgb.gz`)));
        expect(ref.length).toBe(res * res * 3);
        const out = await runGpuEffectTest({
          module: 'source.die_fold',
          bundle: 'nano',
          width: res, height: res,
          inputColor: [0, 0, 0, 1],
          params: [['speed', 0], ['grain', c.grain], ['netness', c.netness], ['depth', c.depth],
                   ['pan_x', c.pan_x], ['pan_y', c.pan_y], ['seed', c.seed]],
          ticks: 2,
          dumpName: `df_parity_${backend}_${c.name}`,
        });
        expect(out.success).toBe(true);
        let sum = 0, tail = 0;
        out.forEachPixel((px: any, x: number, y: number) => {
          const o = (y * res + x) * 3;
          for (const [k, v] of [[px.r, ref[o]], [px.g, ref[o + 1]], [px.b, ref[o + 2]]]) {
            const d = Math.abs(k - v);
            sum += d;
            if (d > 24) tail++;
          }
        });
        const n = res * res * 3;
        // The prototype's own gate for its WebGPU port.
        expect(sum / n).toBeLessThanOrEqual(2.0);
        expect(tail / n).toBeLessThanOrEqual(0.02);
      });
    }
  });
});
