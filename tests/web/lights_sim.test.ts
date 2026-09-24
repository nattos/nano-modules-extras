import { runGpuEffectTest, Frame, forEachBackend } from '@nano/web/test/gpu-test-helpers';

// Per-effect tests for filter.lights_sim — samples the input into 4 vertical LED
// bars (Resolume-style) and renders them inset into their quarters.
//
// The harness only supplies a SOLID input colour, so these can't verify the
// per-segment vertical sampling (every segment samples the same colour). They
// cover the geometry instead: bars show the sampled colour, render inset into
// each quarter, and the background obeys input_opacity.

forEachBackend((backend) => {
describe(`Lights Sim Effect E2E (${backend})`, () => {
  jest.setTimeout(60000);

  const W = 160, H = 108;
  const lum = (f: Frame, fx: number, fy: number) => {
    const p = f.pixelAt(Math.floor(fx * W), Math.floor(fy * H));
    return p.r + p.g + p.b;
  };

  it('declares metadata and I/O', async () => {
    const frame = await runGpuEffectTest({
      module: 'filter.lights_sim', bundle: 'lights',
      inputColor: [0, 0, 0, 1], dumpName: 'lights_sim_metadata',
    });
    expect(frame.success).toBe(true);
    expect(frame.metadata?.id).toBe('filter.lights_sim');
  });

  it('renders inset bars showing the sampled input colour', async () => {
    // White input, opacity 0 → background black, bars white. inset_h 0.3 means
    // the bar spans the middle 70% of each quarter, leaving dark side gaps.
    const frame = await runGpuEffectTest({
      module: 'filter.lights_sim', bundle: 'lights',
      width: W, height: H, inputColor: [1, 1, 1, 1], ticks: 1,
      params: [['segments', 8], ['inset_h', 0.3], ['inset_v', 0.05], ['input_opacity', 0.0]],
      dumpName: 'lights_sim_render',
    });
    expect(frame.success).toBe(true);
    // Quarter 0 centre (x≈0.125) is inside the bar → white.
    expect(lum(frame, 0.125, 0.5)).toBeGreaterThan(720);
    // Near the quarter edge (x≈0.23) is in the inset side-gap → black.
    expect(lum(frame, 0.23, 0.5)).toBeLessThan(20);
    // Above the vertical inset (y≈0.01) is outside the bar → black.
    expect(lum(frame, 0.125, 0.01)).toBeLessThan(20);
  });

  it('input_opacity fades the background behind the bars', async () => {
    const at = (input_opacity: number) => runGpuEffectTest({
      module: 'filter.lights_sim', bundle: 'lights',
      width: W, height: H, inputColor: [0.5, 0.5, 0.5, 1], ticks: 1,
      params: [['segments', 8], ['inset_h', 0.3], ['input_opacity', input_opacity]],
      dumpName: `lights_sim_opacity_${Math.round(input_opacity * 100)}`,
    });
    const opaque = await at(1.0);
    const black = await at(0.0);
    expect(opaque.success && black.success).toBe(true);
    // A side-gap pixel shows the faded input: ~grey at 1.0, ~black at 0.0.
    expect(lum(opaque, 0.23, 0.5)).toBeGreaterThan(lum(black, 0.23, 0.5) + 120);
  });

  it('inset_h controls how far the bar is inset (narrower bar at higher inset)', async () => {
    const at = (inset_h: number) => runGpuEffectTest({
      module: 'filter.lights_sim', bundle: 'lights',
      width: W, height: H, inputColor: [1, 1, 1, 1], ticks: 1,
      params: [['segments', 8], ['inset_h', inset_h], ['input_opacity', 0.0]],
      dumpName: `lights_sim_inset_${Math.round(inset_h * 100)}`,
    });
    const wide = await at(0.0);    // bar fills the whole quarter
    const narrow = await at(0.6);  // bar is the middle 40%
    expect(wide.success && narrow.success).toBe(true);
    // x≈0.225 is inside the full-width bar but in the side-gap once inset.
    expect(lum(wide, 0.225, 0.5)).toBeGreaterThan(720);
    expect(lum(narrow, 0.225, 0.5)).toBeLessThan(20);
  });
});
});
