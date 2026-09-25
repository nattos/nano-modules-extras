import { runGpuTest, runGpuEffectTest, forEachBackend } from '@nano/web/test/gpu-test-helpers';

/**
 * `source.light.sweep_gradient` — Sweep Choke's rolling gradient, played by the
 * knob or fired by a trigger.
 *
 * The roll and the triggered gesture are pinned host-free in
 * tests/native/test_sweep_gradient.cpp, on the same header the effect calls.
 * What only a real render can show is here: that the map lands on the texture
 * the right way up, as flat blocks, at the colour and level asked for. The knob
 * is PARKED from the first frame, and the core seeds settled at wherever it
 * already is, so each picture is a pure function of position.
 */

// 4 bars x 16 px, 10 segments x 8 px: every boundary lands on a whole pixel.
const W = 64, H = 80;
const SEG = H / 10;
const BAR = W / 4;
const rowOfSegment = (seg: number) => Math.round(H - (seg + 0.5) * SEG);
const colOfBar = (bar: number) => Math.round((bar + 0.5) * BAR);

forEachBackend((backend) => {
describe(`Sweep Gradient (${backend})`, () => {
  jest.setTimeout(30000);

  const render = (dumpName: string, params: [string, unknown][]) =>
    runGpuEffectTest({
      module: 'source.light.sweep_gradient',
      bundle: 'lights',
      width: W, height: H,
      inputColor: [0.3, 0.3, 0.3, 1.0],
      params: [['color', [1.0, 0.0, 0.5]], ...params],
      dumpName,
    });

  it('declares metadata', async () => {
    const frame = await runGpuTest({
      module: 'source.light.sweep_gradient',
      bundle: 'lights',
      dumpName: 'sweep_gradient_metadata',
    });
    expect(frame.success).toBe(true);
    expect(frame.metadata?.id).toBe('source.light.sweep_gradient');
  });

  it('home is the solid colour on every segment, over any input', async () => {
    const frame = await render('sweep_gradient_home', [['sweep', 0.5]]);
    expect(frame.success).toBe(true);
    for (let bar = 0; bar < 4; bar++)
      for (const seg of [0, 5, 9])
        frame.expectPixelAt(colOfBar(bar), rowOfSegment(seg), { r: 255, g: 0, b: 128, a: 255 }, 4);
  });

  it('either end is black', async () => {
    for (const x of [0, 1]) {
      const frame = await render(`sweep_gradient_end_${x}`, [['sweep', x]]);
      expect(frame.success).toBe(true);
      for (let bar = 0; bar < 4; bar++)
        for (const seg of [0, 9])
          frame.expectPixelAt(colOfBar(bar), rowOfSegment(seg), { r: 0, g: 0, b: 0, a: 255 }, 2);
    }
  });

  it('parked in the fade, one flat level', async () => {
    const frame = await render('sweep_gradient_parked_fade', [['sweep', 0.86], ['angle', 45]]);
    expect(frame.success).toBe(true);
    const ref = frame.pixelAt(colOfBar(0), rowOfSegment(0));
    expect(ref.r).toBeGreaterThan(30);
    expect(ref.r).toBeLessThan(225);
    for (let bar = 0; bar < 4; bar++)
      for (const seg of [0, 3, 6, 9]) {
        const px = frame.pixelAt(colOfBar(bar), rowOfSegment(seg));
        expect(Math.abs(px.r - ref.r)).toBeLessThanOrEqual(2);
      }
  });

  it('Intensity 0 is black', async () => {
    const frame = await render('sweep_gradient_off', [['intensity', 0]]);
    expect(frame.success).toBe(true);
    frame.expectPixelAt(colOfBar(1), rowOfSegment(5), { r: 0, g: 0, b: 0, a: 255 }, 2);
  });
});
});
