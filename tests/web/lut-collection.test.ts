import { runGpuEffectTest, forEachBackend } from '@nano/web/test/gpu-test-helpers';

/**
 * E2E for color.legacy.lut_collection — "LUT Collection 1", the v2 port of the
 * Resolume Wire "LUT 2" preset LUT grader.
 *
 * A LUT on a SOLID input colour IS visible (every pixel maps the same input to
 * the same graded output), so the single-effect harness with a flat inputColor
 * is the natural fit — no structured generator needed.
 *
 * LUT indices (selectField order): 0 Process, 1 Instant, 2 Fade, 3 Chrome,
 * 4 Transfer, 5 Tonal, 6 Mono, 7 Noir, 8 Sat+Contrast, 9 Sat/Contrast More,
 * 10/11/12 Hue Rotate 90/180/270.
 */
forEachBackend((backend) => {
describe(`LUT Collection 1 (color.legacy.lut_collection) E2E (${backend})`, () => {
  jest.setTimeout(60000);

  const COLOR: [number, number, number, number] = [0.8, 0.2, 0.4, 1.0];

  it('declares metadata and its parameters', async () => {
    const frame = await runGpuEffectTest({
      module: 'color.legacy.lut_collection', bundle: 'legacy',
      inputColor: COLOR,
      dumpName: 'lut_metadata',
    });
    expect(frame.success).toBe(true);
    expect(frame.metadata?.id).toBe('color.legacy.lut_collection');
    const names = frame.params.map(p => p.name);
    expect(names).toContain('lut');
    expect(names).toContain('amount');
    expect(names).toContain('pregain');
    // The lut field is a 13-option selector (indices 0..12).
    const lut = frame.params.find(p => p.name === 'lut');
    expect(lut?.min).toBe(0);
    expect(lut?.max).toBe(12);
  });

  it('Mono LUT desaturates the input to grayscale', async () => {
    // The "Mono" preset maps any colour to gray, so the (uniform) output must
    // have r == g == b. This is also the strongest correctness signal that the
    // baked cube addressing is faithful (the generator validated Mono spread=0).
    const frame = await runGpuEffectTest({
      module: 'color.legacy.lut_collection', bundle: 'legacy',
      inputColor: COLOR,
      params: [['lut', 6], ['amount', 1.0], ['pregain', 0.0]],
      dumpName: 'lut_mono',
    });
    expect(frame.success).toBe(true);
    const m = frame.averageColor();
    expect(Math.abs(m.r - m.g)).toBeLessThanOrEqual(4);
    expect(Math.abs(m.g - m.b)).toBeLessThanOrEqual(4);
    // And it actually changed the colour (input was saturated red/magenta).
    expect(Math.abs(m.r - 204)).toBeGreaterThan(10);
  });

  it('amount = 0 is a pure passthrough', async () => {
    const frame = await runGpuEffectTest({
      module: 'color.legacy.lut_collection', bundle: 'legacy',
      inputColor: COLOR,
      params: [['lut', 0], ['amount', 0.0]],
      dumpName: 'lut_passthrough',
    });
    expect(frame.success).toBe(true);
    // Input [0.8,0.2,0.4] -> [204,51,102] within rounding.
    frame.expectUniformColor({ r: 204, g: 51, b: 102 }, 3);
  });

  it('different presets produce different grades', async () => {
    const run = (lut: number, dump: string) => runGpuEffectTest({
      module: 'color.legacy.lut_collection', bundle: 'legacy',
      inputColor: COLOR,
      params: [['lut', lut], ['amount', 1.0]],
      dumpName: dump,
    });
    const process = await run(0, 'lut_process');
    const mono    = await run(6, 'lut_mono_cmp');
    expect(process.success).toBe(true);
    expect(mono.success).toBe(true);
    // Two distinct LUTs on the same input -> uniform but different frames.
    process.expectDifferentFrom(mono, 100);
  });

  it('trilinearly interpolates between LUT cells (not nearest-neighbour)', async () => {
    // Two input greys that sit INSIDE a single LUT cell (64^3 cell width along
    // an axis = 1/63 ≈ 0.0159; nodes 31=0.4921 and 32=0.5079 bracket both
    // samples). Nearest-neighbour would snap them to the same node -> identical
    // output. Hardware trilinear varies the output smoothly across the cell.
    const grade = (v: number) => runGpuEffectTest({
      module: 'color.legacy.lut_collection', bundle: 'legacy',
      inputColor: [v, v, v, 1.0],
      params: [['lut', 0], ['amount', 1.0]],
    }).then(f => { expect(f.success).toBe(true); return f.averageColor(); });

    const lo = await grade(0.498);
    const hi = await grade(0.505);
    const spread = Math.abs(lo.r - hi.r) + Math.abs(lo.g - hi.g) + Math.abs(lo.b - hi.b);
    expect(spread).toBeGreaterThan(1);   // 0 would mean nearest-neighbour
  });

  it('pregain reshapes the grade', async () => {
    const run = (pregain: number, dump: string) => runGpuEffectTest({
      module: 'color.legacy.lut_collection', bundle: 'legacy',
      inputColor: [0.5, 0.5, 0.5, 1.0],
      params: [['lut', 0], ['amount', 1.0], ['pregain', pregain]],
      dumpName: dump,
    });
    const neutral = await run(0.0, 'lut_pregain_0');
    const pushed  = await run(0.8, 'lut_pregain_hi');
    expect(neutral.success).toBe(true);
    expect(pushed.success).toBe(true);
    const a = neutral.averageColor();
    const b = pushed.averageColor();
    const delta = Math.abs(a.r - b.r) + Math.abs(a.g - b.g) + Math.abs(a.b - b.b);
    expect(delta).toBeGreaterThan(5);
  });
});
});
