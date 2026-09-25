import { runGpuTest, runGpuEffectTest, forEachBackend } from '@nano/web/test/gpu-test-helpers';

/**
 * `source.light.drum_bars` — three drum roles on the four LED bars.
 *
 * The hits, the pattern cycle, the flams, the strobes and the XOR are pinned
 * host-free in tests/native/test_drum_bars.cpp, and a real Ch1 press through
 * the executor in test_drum_bars_render.cpp. What is left for here is that the
 * card loads and draws the map: nothing fired, the bars sit at black, as flat
 * blocks, over any input.
 */

const W = 64, H = 80;
const SEG = H / 10;
const BAR = W / 4;
const rowOfSegment = (seg: number) => Math.round(H - (seg + 0.5) * SEG);
const colOfBar = (bar: number) => Math.round((bar + 0.5) * BAR);

forEachBackend((backend) => {
describe(`Drum Bars (${backend})`, () => {
  jest.setTimeout(30000);

  it('declares metadata', async () => {
    const frame = await runGpuTest({
      module: 'source.light.drum_bars',
      bundle: 'lights',
      dumpName: 'drum_bars_metadata',
    });
    expect(frame.success).toBe(true);
    expect(frame.metadata?.id).toBe('source.light.drum_bars');
  });

  it('idle, every bar is black over a bright input', async () => {
    const frame = await runGpuEffectTest({
      module: 'source.light.drum_bars',
      bundle: 'lights',
      width: W, height: H,
      inputColor: [0.9, 0.9, 0.9, 1.0],
      params: [['color', [1.0, 1.0, 1.0]], ['cycle', 0.37]],
      dumpName: 'drum_bars_idle',
    });
    expect(frame.success).toBe(true);
    for (let bar = 0; bar < 4; bar++)
      for (const seg of [0, 5, 9])
        frame.expectPixelAt(colOfBar(bar), rowOfSegment(seg), { r: 0, g: 0, b: 0, a: 255 }, 2);
  });
});
});
