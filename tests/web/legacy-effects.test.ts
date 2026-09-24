import { runGpuEffectTest, forEachBackend } from '@nano/web/test/gpu-test-helpers';

// Per-effect E2E for the `legacy` bundle (com.nano.legacy) — ports of shipped
// NanoGraph effects. Point at a running dev server via GPU_TEST_BASE_URL.

forEachBackend((backend) => {
describe(`Bicolor Gradient (color.legacy.bicolor_grad) E2E (${backend})`, () => {
  jest.setTimeout(30000);

  it('declares metadata and its parameters', async () => {
    const frame = await runGpuEffectTest({
      module: 'color.legacy.bicolor_grad',
      bundle: 'legacy',
      inputColor: [0.6, 0.2, 0.1, 1.0],
      dumpName: 'bicolor_grad_metadata',
    });
    expect(frame.success).toBe(true);
    expect(frame.metadata?.id).toBe('color.legacy.bicolor_grad');
    // `neutral` is a vec3/colour field, so it lives in the schema, not the
    // scalar params[] list — assert on the scalar knobs.
    const names = frame.params.map(p => p.name);
    expect(names).toContain('scale');
    expect(names).toContain('blend');
    expect(names).toContain('smoothing');
    expect(names).toContain('mode');
  });

  it('blend=0 is a passthrough (output equals input)', async () => {
    const frame = await runGpuEffectTest({
      module: 'color.legacy.bicolor_grad',
      bundle: 'legacy',
      width: 64, height: 64,
      inputColor: [0.6, 0.2, 0.1, 1.0],
      params: [['blend', 0.0], ['mode', 0]],
      samplePoints: [[32, 32]],
      dumpName: 'bicolor_grad_passthrough',
    });
    expect(frame.success).toBe(true);
    const s = frame.samples[0];
    expect(Math.abs(s.r - 153)).toBeLessThan(12);
    expect(Math.abs(s.g - 51)).toBeLessThan(12);
    expect(Math.abs(s.b - 26)).toBeLessThan(12);
  });

  it('a chroma-free (grey) input collapses the gradient to the neutral colour', async () => {
    // Grey input → no dominant hue → major/minor confidence ~0 → both ends of
    // the gradient fall back to the neutral colour, so the whole frame paints
    // neutral at blend=1.
    const frame = await runGpuEffectTest({
      module: 'color.legacy.bicolor_grad',
      bundle: 'legacy',
      width: 64, height: 64,
      inputColor: [0.5, 0.5, 0.5, 1.0],
      params: [['blend', 1.0], ['mode', 0], ['neutral', [0.2, 0.4, 0.8]]],
      samplePoints: [[10, 10], [54, 54]],
      dumpName: 'bicolor_grad_neutral',
    });
    expect(frame.success).toBe(true);
    for (const s of frame.samples) {
      expect(Math.abs(s.r - 51)).toBeLessThan(20);   // 0.2 * 255
      expect(Math.abs(s.g - 102)).toBeLessThan(20);  // 0.4 * 255
      expect(Math.abs(s.b - 204)).toBeLessThan(20);  // 0.8 * 255
    }
  });
});
});

forEachBackend((backend) => {
describe(`Glisten (filter.legacy.glisten) E2E (${backend})`, () => {
  jest.setTimeout(30000);

  it('declares metadata and its parameters', async () => {
    const frame = await runGpuEffectTest({
      module: 'filter.legacy.glisten',
      bundle: 'legacy',
      inputColor: [0.3, 0.3, 0.3, 1.0],
      dumpName: 'glisten_metadata',
    });
    expect(frame.success).toBe(true);
    expect(frame.metadata?.id).toBe('filter.legacy.glisten');
    const names = frame.params.map(p => p.name);
    expect(names).toContain('size');
    expect(names).toContain('blades');
    expect(names).toContain('levels');
    expect(names).toContain('intensity');
  });

  it('adds light near the anchor (top-left) and leaves the far corner untouched', async () => {
    // On a uniform input the brightest-spot search settles on the first cell
    // (top-left). The additive sparkle brightens around it; the opposite
    // corner stays at the input value.
    const frame = await runGpuEffectTest({
      module: 'filter.legacy.glisten',
      bundle: 'legacy',
      width: 64, height: 64,
      inputColor: [0.3, 0.3, 0.3, 1.0],
      params: [['intensity', 1.5], ['size', 0.3], ['flicker_sustain', 1.0]],
      samplePoints: [[2, 2], [62, 62]],
      ticks: 2,
      renderEachTick: true,
      dumpName: 'glisten_anchor',
    });
    expect(frame.success).toBe(true);
    const near = frame.samples.find(s => s.x === 2)!;
    const far = frame.samples.find(s => s.x === 62)!;
    expect(near.r).toBeGreaterThan(far.r + 10);  // sparkle brightened the anchor
    expect(Math.abs(far.r - 77)).toBeLessThan(12); // far corner ~ input (0.3*255)
  });

  it('never darkens the input (unorm sparkle layer only adds light)', async () => {
    // The sparkle layer accumulates in RGBA8 unorm like the original
    // (GlobalBitDepth=Int8): its unclamped negative vertex colours carve the
    // layer's own glow, but the layer itself is clamped ≥ 0, so the composite
    // in + layer can only brighten. Crank the digging knobs and check no
    // sample drops below the input.
    const frame = await runGpuEffectTest({
      module: 'filter.legacy.glisten',
      bundle: 'legacy',
      width: 64, height: 64,
      inputColor: [0.5, 0.4, 0.3, 1.0],
      params: [['intensity', 2.0], ['flicker_sustain', 1.0],
               ['color_grad_power', 1.0], ['color_grad_sharp', 1.0], ['size', 0.8]],
      samplePoints: [[4, 4], [16, 16], [32, 32], [48, 48], [62, 62]],
      ticks: 2,
      renderEachTick: true,
      dumpName: 'glisten_no_darken',
    });
    expect(frame.success).toBe(true);
    for (const s of frame.samples) {
      expect(s.r).toBeGreaterThanOrEqual(127 - 3);
      expect(s.g).toBeGreaterThanOrEqual(102 - 3);
      expect(s.b).toBeGreaterThanOrEqual(76 - 3);
    }
  });

  it('sustain 0 at rest gates the layer to zero (passthrough)', async () => {
    // The flicker envelope starts at 0; with sustain 0 the blur-pass gain is
    // (contrast+1)·mix(env, 1, 0) = 0, so the sparkle layer contributes
    // nothing and the output is exactly input × input_alpha.
    const frame = await runGpuEffectTest({
      module: 'filter.legacy.glisten',
      bundle: 'legacy',
      width: 64, height: 64,
      inputColor: [0.3, 0.5, 0.7, 1.0],
      params: [['flicker_sustain', 0.0], ['flicker_rate', 0.0]],
      samplePoints: [[4, 4], [32, 32]],
      dumpName: 'glisten_gated',
    });
    expect(frame.success).toBe(true);
    for (const s of frame.samples) {
      expect(Math.abs(s.r - 77)).toBeLessThan(4);
      expect(Math.abs(s.g - 128)).toBeLessThan(4);
      expect(Math.abs(s.b - 179)).toBeLessThan(4);
    }
  });
});
});

forEachBackend((backend) => {
describe(`Double Chamber (source.legacy.double_chamber) E2E (${backend})`, () => {
  jest.setTimeout(30000);

  it('declares metadata and its parameters', async () => {
    const frame = await runGpuEffectTest({
      module: 'source.legacy.double_chamber',
      bundle: 'legacy',
      inputColor: [0.0, 0.0, 0.0, 1.0],
      dumpName: 'double_chamber_metadata',
    });
    expect(frame.success).toBe(true);
    expect(frame.metadata?.id).toBe('source.legacy.double_chamber');
    const names = frame.params.map(p => p.name);
    expect(names).toContain('p_count');
    expect(names).toContain('field_speed');
    expect(names).toContain('to_big');
    expect(names).toContain('big_count');
  });

  it('renders additive particles over a black input', async () => {
    // Particles seed as white points, blend additively over the black input.
    // After a few ticks the cloud has clearly brightened the frame.
    const frame = await runGpuEffectTest({
      module: 'source.legacy.double_chamber',
      bundle: 'legacy',
      width: 128, height: 128,
      inputColor: [0.0, 0.0, 0.0, 1.0],
      params: [
        ['p_count', 8000],
        ['p_point_size', 1.0],    // [0,1] slider → 0.01 uv effective
        ['p_opacity', 1.0],
        ['exposure', 2.0],
        ['color_contrib', 0.0],   // pure white points
      ],
      samplePoints: [[64, 64], [48, 48], [80, 80], [64, 44], [44, 64], [84, 64]],
      ticks: 8,
      renderEachTick: true,
      dumpName: 'double_chamber_particles',
    });
    expect(frame.success).toBe(true);
    // The cloud's exact per-pixel layout is stochastic; assert it brightened
    // the frame somewhere rather than at one fixed pixel.
    const maxR = Math.max(...frame.samples.map(s => s.r));
    expect(maxR).toBeGreaterThan(8);  // particles added light
  });

  it('renders tracer lines (no particles)', async () => {
    // Particles off, tracers on: the field streamlines should draw bright
    // lines over the black input.
    const frame = await runGpuEffectTest({
      module: 'source.legacy.double_chamber',
      bundle: 'legacy',
      width: 128, height: 128,
      inputColor: [0.0, 0.0, 0.0, 1.0],
      params: [
        ['p_count', 0],
        ['big_opacity', 0.0],
        ['l_count', 48],
        ['l_opacity', 1.0],
        ['l_length', 0.8],
        ['l_width', 0.4],
        ['color_contrib', 0.0],   // white lines
        ['field_speed', 0.3],
      ],
      samplePoints: [[64, 64], [40, 50], [88, 64], [50, 80], [64, 30]],
      ticks: 16,
      renderEachTick: true,
      dumpName: 'double_chamber_tracers',
    });
    expect(frame.success).toBe(true);
    const maxR = Math.max(...frame.samples.map(s => s.r));
    expect(maxR).toBeGreaterThan(8);  // tracer lines drew light
  });

  it('renders bridger chords between particles', async () => {
    // Particles on but invisible (size ~0), bridgers on: each bridger draws a
    // chord between two particles, so the only light in the frame comes from
    // the connector lines.
    const frame = await runGpuEffectTest({
      module: 'source.legacy.double_chamber',
      bundle: 'legacy',
      width: 128, height: 128,
      inputColor: [0.0, 0.0, 0.0, 1.0],
      params: [
        ['p_count', 3000],
        ['p_opacity', 0.0],       // hide the particles themselves
        ['big_opacity', 0.0],
        ['l_count', 0],           // no tracer lines
        ['bridger_count', 384],
        ['bridger_opacity', 1.0],
        ['bridger_width', 0.6],
        ['bridger_rate', 0.2],
        ['bridger_color_contrib', 0.0],  // hue-driven (white-ish) chords
        ['exposure', 2.0],
      ],
      samplePoints: [[64, 64], [40, 50], [88, 64], [50, 80], [64, 30], [80, 80]],
      ticks: 6,
      renderEachTick: true,
      dumpName: 'double_chamber_bridgers',
    });
    expect(frame.success).toBe(true);
    const maxR = Math.max(...frame.samples.map(s => s.r));
    expect(maxR).toBeGreaterThan(8);  // bridger chords drew light
  });
});
});

// PUPPETEER ONLY — the same recorded gap double-chamber-interactions.test.ts
// carries, and the only case in this file that needs cross-frame sim state to
// ADVANCE rather than merely exist. Natively the two runs below come back
// byte-identical (2756 lit pixels each, max radius 43 of a 55 threshold), i.e.
// boundary_death changes nothing because the particles never drift out to the
// boundary in the first place — the persistent particle buffer isn't carrying
// state across ticks. The four cases above still run on both backends and prove
// the effect loads, renders particles, tracers and bridgers; what doesn't
// survive natively is multi-frame integration. This effect keeps its particle,
// tracer and bridger pools in BufferUsage::Storage buffers advanced in place
// each frame, which is the documented native gap — see web/KNOWN_ISSUES.md,
// "Persistent GPU storage buffers don't carry state across frames on Metal"
// (d_wave is the cleanest measurement of it). Flip to the default list once
// that's fixed.
forEachBackend((backend) => {
describe(`Double Chamber multi-frame integration (${backend})`, () => {
  jest.setTimeout(30000);

  it('boundary_death confines the cloud (proportional kill at the boundary)', async () => {
    // Isolate boundary_death by turning the soft boundary FORCE off, so the
    // only thing containing the cloud is the kill. A mild outward sink pushes
    // particles past boundary_size; with death off they stream to the edges,
    // with death on they die there and recycle to centre → confined disc.
    const W = 128, H = 128;
    const common: [string, number | number[]][] = [
      ['p_count', 12000],
      ['p_point_size', 0.6],
      ['p_opacity', 1.0],
      ['exposure', 2.0],
      ['color_contrib', 0.0],
      ['boundary', 0.0],         // soft boundary FORCE off
      ['boundary_size', 0.3],
      ['field_speed', 0.4],
      ['motion_rate', 2.0],
      ['sink', 0.3],             // gentle outward drift toward the boundary
      ['jitter', 0.05],
      ['to_big', 0.0],
      ['big_opacity', 0.0],
      ['l_count', 0],
    ];
    const outerLit = (frame: { forEachPixel: (fn: (c: { r: number; g: number; b: number }, x: number, y: number) => void) => void }) => {
      const cx = W / 2, cy = H / 2; let n = 0;
      frame.forEachPixel((c, x, y) => {
        const dx = x - cx, dy = y - cy;
        if (Math.sqrt(dx * dx + dy * dy) > 55 && c.r + c.g + c.b > 24) n++;
      });
      return n;
    };

    const off = await runGpuEffectTest({
      module: 'source.legacy.double_chamber', bundle: 'legacy', width: W, height: H,
      inputColor: [0, 0, 0, 1],
      params: [...common, ['boundary_death', 0.0]],
      ticks: 24, renderEachTick: true,
      dumpName: 'double_chamber_bdeath_off',
    });
    const on = await runGpuEffectTest({
      module: 'source.legacy.double_chamber', bundle: 'legacy', width: W, height: H,
      inputColor: [0, 0, 0, 1],
      params: [...common, ['boundary_death', 1.0]],
      ticks: 24, renderEachTick: true,
      dumpName: 'double_chamber_bdeath_on',
    });
    expect(off.success && on.success).toBe(true);

    const offOuter = outerLit(off), onOuter = outerLit(on);  // ~906 vs ~229
    expect(offOuter).toBeGreaterThan(150);            // death-off spreads outward
    expect(onOuter).toBeLessThan(offOuter * 0.5);     // death-on confines it
  });
});
}, ['puppeteer']);
