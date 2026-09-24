import { runGpuEffectTest, forEachBackend } from '@nano/web/test/gpu-test-helpers';

// Per-effect tests for source.light.side_jet — the fixed-engine JPL test plume.
//
// side_jet's 1D axial solver carries persistent GPU state that develops
// across rendered frames, so these tests set `renderEachTick: true` (the
// runner interleaves tick()+render() instead of ticking N times then
// rendering once). Without it the plume only advances a single frame and
// barely leaves the nozzle.

forEachBackend((backend) => {
describe(`Side Jet Effect E2E (${backend})`, () => {
  jest.setTimeout(60000);

  const W = 192, H = 108;

  it('declares metadata and I/O', async () => {
    const frame = await runGpuEffectTest({
      module: 'source.light.side_jet', bundle: 'lights',
      inputColor: [0, 0, 0, 1], dumpName: 'side_jet_metadata',
    });
    expect(frame.success).toBe(true);
    expect(frame.metadata?.id).toBe('source.light.side_jet');
  });

  it('ignited engine renders a visible plume', async () => {
    const frame = await runGpuEffectTest({
      module: 'source.light.side_jet', bundle: 'lights',
      width: W, height: H, inputColor: [0, 0, 0, 1],
      ticks: 60, renderEachTick: true,
      params: [['ignition', 1.0], ['throttle', 0.85], ['mixture', 0.2]],
      dumpName: 'side_jet_plume',
    });
    expect(frame.success).toBe(true);
    frame.expectNotSolidColor({ r: 0, g: 0, b: 0 }, 5);
  });

  it('engine off (ignition=0) stays dark', async () => {
    const frame = await runGpuEffectTest({
      module: 'source.light.side_jet', bundle: 'lights',
      width: W, height: H, inputColor: [0, 0, 0, 1],
      ticks: 60, renderEachTick: true,
      params: [['ignition', 0.0], ['throttle', 0.85]],
      dumpName: 'side_jet_off',
    });
    expect(frame.success).toBe(true);
    // No flame: the plume should be essentially black everywhere.
    frame.expectUniformColor({ r: 0, g: 0, b: 0, a: 255 }, 6);
  });

  it('higher throttle produces a longer/brighter plume', async () => {
    const lit = (throttle: number) => runGpuEffectTest({
      module: 'source.light.side_jet', bundle: 'lights',
      width: W, height: H, inputColor: [0, 0, 0, 1],
      ticks: 60, renderEachTick: true,
      params: [['ignition', 1.0], ['throttle', throttle], ['mixture', 0.2]],
      dumpName: `side_jet_throttle_${Math.round(throttle * 100)}`,
    });

    const lo = await lit(0.3);
    const hi = await lit(0.95);
    expect(lo.success && hi.success).toBe(true);
    // Integrate luminance over a downstream band (40–95% along the axis,
    // around the centerline): the high-throttle plume reaches farther and
    // brighter, so its integrated energy here exceeds the low-throttle one.
    // A single pixel is unreliable — it can land between shock nodes.
    const bandLum = (f: typeof lo) => {
      let s = 0;
      for (let x = Math.floor(W * 0.4); x < Math.floor(W * 0.95); x++) {
        for (let y = Math.floor(H * 0.35); y < Math.floor(H * 0.65); y++) {
          const p = f.pixelAt(x, y);
          s += p.r + p.g + p.b;
        }
      }
      return s;
    };
    expect(bandLum(hi)).toBeGreaterThan(bandLum(lo) * 1.2);
  });
});
});
