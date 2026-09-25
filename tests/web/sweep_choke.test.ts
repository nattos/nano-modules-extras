import { runGpuTest, runGpuEffectTest, forEachBackend } from '@nano/web/test/gpu-test-helpers';

/**
 * `source.light.sweep_choke` — one knob that sweeps and chokes the LED bars.
 *
 * The behaviour itself (the tilt law, the throw, the tail, the stutter) is
 * pinned host-free in native/tests/test_sweep_choke.cpp, on the same header
 * the effect calls. What only a real render can show is here: that the map
 * lands on the texture the right way up and round, as flat blocks, at the
 * colour asked for.
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
      params: [['color', [1.0, 0.0, 0.5]], ...params],
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

  it('home is the solid colour on every segment, over any input', async () => {
    const frame = await render('sweep_choke_home', [['sweep', 0.5]]);
    expect(frame.success).toBe(true);
    for (let bar = 0; bar < 4; bar++)
      for (const seg of [0, 5, 9])
        frame.expectPixelAt(colOfBar(bar), rowOfSegment(seg), { r: 255, g: 0, b: 128, a: 255 }, 4);
  });

  it('parked in the fade at 90 deg, the floor is out and the top still lit', async () => {
    const frame = await render('sweep_choke_tilt_up', [
      ['sweep', 0.86], ['angle', 90], ['tilt', 1.0],
    ]);
    expect(frame.success).toBe(true);
    const bottom = frame.pixelAt(colOfBar(1), rowOfSegment(0));
    const top = frame.pixelAt(colOfBar(1), rowOfSegment(9));
    expect(bottom.r).toBeLessThan(40);
    expect(top.r).toBeGreaterThan(200);
  });

  it('Ping-Pong turns it over on the other half of the knob', async () => {
    const frame = await render('sweep_choke_tilt_down', [
      ['sweep', 0.14], ['angle', 90], ['tilt', 1.0], ['side_mode', 0],
    ]);
    expect(frame.success).toBe(true);
    const bottom = frame.pixelAt(colOfBar(1), rowOfSegment(0));
    const top = frame.pixelAt(colOfBar(1), rowOfSegment(9));
    expect(bottom.r).toBeGreaterThan(200);
    expect(top.r).toBeLessThan(40);
  });

  it('at 0 deg the dark sweeps left to right', async () => {
    const frame = await render('sweep_choke_tilt_right', [
      ['sweep', 0.86], ['angle', 0], ['tilt', 1.0],
    ]);
    expect(frame.success).toBe(true);
    const left = frame.pixelAt(colOfBar(0), rowOfSegment(5));
    const right = frame.pixelAt(colOfBar(3), rowOfSegment(5));
    expect(left.r).toBeLessThan(right.r - 100);
  });

  it('segments are flat blocks', async () => {
    const frame = await render('sweep_choke_flat', [
      ['sweep', 0.86], ['angle', 45], ['tilt', 1.0],
    ]);
    expect(frame.success).toBe(true);
    // Corners of one cell agree with its centre.
    const x0 = BAR + 2, x1 = 2 * BAR - 3;
    const y = rowOfSegment(4);
    const c = frame.pixelAt(colOfBar(1), y);
    for (const [x, yy] of [[x0, y - 3], [x1, y - 3], [x0, y + 3], [x1, y + 3]]) {
      const p = frame.pixelAt(x, yy);
      expect(Math.abs(p.r - c.r)).toBeLessThan(2);
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
