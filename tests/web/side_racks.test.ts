import { runGpuTest, runGpuEffectTest, forEachBackend } from '@nano/web/test/gpu-test-helpers';

/**
 * `source.light.side_racks` — the six-bar rig's whole show on one card.
 *
 * The sweep, the tilt, every mix-in, the lift, the strobe and Mix Post are
 * pinned host-free in tests/native/test_side_racks.cpp, and a real primary
 * press through the executor in test_side_racks_render.cpp. What is left for
 * here is that the card loads and draws the compact grid on this backend: the
 * left rack on the left half, the top bar on top, flat blocks, opaque over any
 * input.
 */

const W = 240, H = 60;          // 120 columns × 2 px, 3 rows × 20 px
const colX = (col: number) => col * 2 + 1;
const rowY = (row: number) => row * 20 + 10;

forEachBackend((backend) => {
describe(`Side Racks (${backend})`, () => {
  jest.setTimeout(30000);

  it('declares metadata', async () => {
    const frame = await runGpuTest({
      module: 'source.light.side_racks',
      bundle: 'lights',
      dumpName: 'side_racks_metadata',
    });
    expect(frame.success).toBe(true);
    expect(frame.metadata?.id).toBe('source.light.side_racks');
  });

  it('at home, every segment of both racks is the color, over a dark input', async () => {
    const frame = await runGpuEffectTest({
      module: 'source.light.side_racks',
      bundle: 'lights',
      width: W, height: H,
      inputColor: [0.1, 0.1, 0.1, 1.0],
      params: [['color', [1.0, 0.0, 1.0]]],
      dumpName: 'side_racks_home',
    });
    expect(frame.success).toBe(true);
    for (let row = 0; row < 3; row++)
      for (const col of [0, 59, 60, 119])
        frame.expectPixelAt(colX(col), rowY(row), { r: 255, g: 0, b: 255, a: 255 }, 2);
  });

  it('parked in the fade at full rest tilt, the top bar is brightest', async () => {
    const frame = await runGpuEffectTest({
      module: 'source.light.side_racks',
      bundle: 'lights',
      width: W, height: H,
      inputColor: [0.0, 0.0, 0.0, 1.0],
      params: [
        ['color', [1.0, 1.0, 1.0]],
        ['tilt', 1.0],
        ['rest_tilt', 1.0],
        ['sweep', 0.5 + 0.5 * (0.45 + 0.5 * 0.55)],
      ],
      dumpName: 'side_racks_tilt',
    });
    expect(frame.success).toBe(true);
    const g = (col: number, row: number) => frame.pixelAt(colX(col), rowY(row)).g;
    expect(g(10, 0)).toBeGreaterThan(g(10, 1));
    expect(g(10, 1)).toBeGreaterThan(g(10, 2));
    // Mirrored, and one flat block along a whole bar.
    for (let row = 0; row < 3; row++)
      for (const col of [0, 30, 59, 60, 119])
        expect(Math.abs(g(col, row) - g(10, row))).toBeLessThanOrEqual(1);
  });

  it('lift brings a choked rig up in the accent', async () => {
    const frame = await runGpuEffectTest({
      module: 'source.light.side_racks',
      bundle: 'lights',
      width: W, height: H,
      inputColor: [0.9, 0.9, 0.9, 1.0],
      params: [
        ['sweep', 1.0],
        ['lift_color', [1.0, 0.0, 0.0]],
        ['lift', 0.5],
      ],
      dumpName: 'side_racks_lift',
    });
    expect(frame.success).toBe(true);
    for (let row = 0; row < 3; row++)
      for (const col of [0, 60, 119])
        frame.expectPixelAt(colX(col), rowY(row), { r: 128, g: 0, b: 0, a: 255 }, 3);
  });
});
});
