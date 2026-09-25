import { runGpuTest, runGpuEffectTest, forEachBackend } from '@nano/web/test/gpu-test-helpers';
import { runEngineTest } from '@nano/web/test/engine-test-helpers';
import type { Sketch } from '@nano/web/src/sketch-types';

/**
 * `source.light.sweep_choke` — one knob that sweeps and chokes the LED bars.
 *
 * The behaviour itself (the tilt law, the throw, the tail, the stutter) is
 * pinned host-free in native/tests/test_sweep_choke.cpp, on the same header
 * the effect calls. What only a real render can show is here: that the map
 * lands on the texture the right way up and round, as flat blocks, at the
 * colour asked for — and that the gray mask reaches `mask_out`.
 *
 * The bars show the gesture (glints, the bounce's flash), which a parked knob
 * never makes, so the position tests put the mask on the bars with Base 1.
 * Parked, the mask is always flat — the roll that makes it a gradient only
 * happens while it changes, which the native goldens drive.
 *
 * The knob is PARKED from the first frame in every case. knob_rate seeds at
 * wherever the knob already is, so a stored value is never read as motion —
 * nothing is charged, nothing stutters, and the picture is a pure function of
 * position whatever the harness's frame timing.
 */

// 4 bars x 16 px, 10 segments x 8 px: every boundary lands on a whole pixel.
const W = 64, H = 80;
const SEG = H / 10;
const BAR = W / 4;
const rowOfSegment = (seg: number) => Math.round(H - (seg + 0.5) * SEG);
const colOfBar = (bar: number) => Math.round((bar + 0.5) * BAR);

forEachBackend((backend) => {
describe(`Sweep Choke (${backend})`, () => {
  jest.setTimeout(30000);

  const render = (dumpName: string, params: [string, unknown][]) =>
    runGpuEffectTest({
      module: 'source.light.sweep_choke',
      bundle: 'lights',
      width: W, height: H,
      inputColor: [0.3, 0.3, 0.3, 1.0],
      params: [['color', [1.0, 0.0, 0.5]], ['base', 1.0], ...params],
      dumpName,
    });

  it('declares metadata', async () => {
    const frame = await runGpuTest({
      module: 'source.light.sweep_choke',
      bundle: 'lights',
      dumpName: 'sweep_choke_metadata',
    });
    expect(frame.success).toBe(true);
    expect(frame.metadata?.id).toBe('source.light.sweep_choke');
  });

  it('parked, the bars are dark by default: they show the gesture, not the position', async () => {
    const frame = await render('sweep_choke_dark_home', [['sweep', 0.5], ['base', 0.0]]);
    expect(frame.success).toBe(true);
    for (let bar = 0; bar < 4; bar++)
      for (const seg of [0, 5, 9])
        frame.expectPixelAt(colOfBar(bar), rowOfSegment(seg), { r: 0, g: 0, b: 0, a: 255 }, 2);
  });

  it('home is the solid colour on every segment, over any input', async () => {
    const frame = await render('sweep_choke_home', [['sweep', 0.5]]);
    expect(frame.success).toBe(true);
    for (let bar = 0; bar < 4; bar++)
      for (const seg of [0, 5, 9])
        frame.expectPixelAt(colOfBar(bar), rowOfSegment(seg), { r: 255, g: 0, b: 128, a: 255 }, 4);
  });

  it('parked in the fade, the base is one flat level: a gradient only passes through', async () => {
    const frame = await render('sweep_choke_parked_fade', [['sweep', 0.86], ['angle', 45]]);
    expect(frame.success).toBe(true);
    const ref = frame.pixelAt(colOfBar(0), rowOfSegment(0));
    expect(ref.r).toBeGreaterThan(30);
    expect(ref.r).toBeLessThan(225);
    for (let bar = 0; bar < 4; bar++)
      for (const seg of [0, 3, 6, 9]) {
        const px = frame.pixelAt(colOfBar(bar), rowOfSegment(seg));
        expect(Math.abs(px.r - ref.r)).toBeLessThanOrEqual(1);
      }
  });

  it('both ends choke to black', async () => {
    for (const sweep of [0.0, 1.0]) {
      const frame = await render(`sweep_choke_end_${sweep}`, [['sweep', sweep]]);
      expect(frame.success).toBe(true);
      for (let bar = 0; bar < 4; bar++)
        for (const seg of [0, 5, 9])
          frame.expectPixelAt(colOfBar(bar), rowOfSegment(seg), { r: 0, g: 0, b: 0, a: 255 }, 3);
    }
  });
});
});

// mask_out, through a sidechannel: wiring it into the send's `send_in` makes
// the bus carry the mask, and the receive REPLACES the stage output with it.
describe('Sweep Choke mask_out (engine path)', () => {
  jest.setTimeout(60000);

  const run = (id: string, params: Record<string, unknown>) => runEngineTest({
    width: W, height: H,
    modules: ['com.nano.lights', 'com.nano.core'],
    commands: [
      { type: 'createSketch', sketchId: id, sketch: {
        anchor: null,
        wires: [{ id: 'mw', src: { instanceKey: 'sc@0', field: 'mask_out' },
                  dest: { instanceKey: 'send@0', field: 'send_in' } }],
        chain: [
          { type: 'module', module_type: 'source.light.sweep_choke', instance_key: 'sc@0',
            params: { color: [1.0, 0.0, 0.5], ...params } },
          { type: 'module', module_type: 'util.sidechannel_out', instance_key: 'send@0',
            params: { channel: 5 } },
          { type: 'module', module_type: 'util.sidechannel_in', instance_key: 'recv@0',
            params: { channel: 5 } },
        ],
      } as Sketch },
      { type: 'setTracePoints', tracePoints: [
        { id: 'out', target: { type: 'sketch_output', sketchId: id } },
      ]},
    ],
    waitFrames: 20,
    captureTraceIds: ['out'],
    dumpName: `sweep_choke_${id}`,
  });

  it('home is a white mask, whatever the bars are doing', async () => {
    const r = await run('mask_home', { sweep: 0.5 });
    expect(r.success).toBe(true);
    for (let bar = 0; bar < 4; bar++)
      for (const seg of [0, 5, 9])
        r.trace('out').expectPixelAt(colOfBar(bar), rowOfSegment(seg), { r: 255, g: 255, b: 255 }, 3);
  });

  it('parked in the fade the mask is one flat gray', async () => {
    const r = await run('mask_parked', { sweep: 0.86, angle: 90 });
    expect(r.success).toBe(true);
    const ref = r.trace('out').pixelAt(colOfBar(0), rowOfSegment(0));
    expect(ref.r).toBeGreaterThan(30);
    expect(ref.r).toBeLessThan(225);
    for (let bar = 0; bar < 4; bar++)
      for (const seg of [0, 5, 9]) {
        const px = r.trace('out').pixelAt(colOfBar(bar), rowOfSegment(seg));
        expect(Math.abs(px.r - ref.r)).toBeLessThanOrEqual(1);
        expect(Math.abs(px.r - px.g)).toBeLessThanOrEqual(1);
        expect(Math.abs(px.r - px.b)).toBeLessThanOrEqual(1);
      }
  });
});
