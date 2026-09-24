import { runGpuEffectTest, runGpuChainTest, Frame, forEachBackend } from '@nano/web/test/gpu-test-helpers';

// Per-effect tests for filter.glow.vcr_halo — the convolution sibling of
// source.mesh.three_planes: a multi-octave glow pyramid plus the shared
// nano_vcr grade.
//
// The effect is stateless (the grain seed comes off absolute host time), so
// one render per case is enough — no ticks, no renderEachTick.
//
// Two properties carry most of the coverage, and both are exact rather than
// statistical:
//
//   * A blur of a CONSTANT field is that constant. The pyramid's per-level
//     weights are normalised to sum to 1, so on a flat input "halo only"
//     must land on the same pixels as "source only" — at ANY radius, which
//     also pins down that changing the level count doesn't change DC.
//   * Outline band-passes the emitter, so on a flat input it must produce
//     NOTHING. That is the whole difference between blooming a shape and
//     blooming its edge.

forEachBackend((backend) => {
describe(`VCR Halo E2E (${backend})`, () => {
  jest.setTimeout(60000);

  const W = 192, H = 108;
  const MODULE = 'filter.glow.vcr_halo';
  const BUNDLE = 'lights' as const;

  // Grade knobs that would fight a pixel assertion. Everything that remains
  // (drive, toe, shoulder, ...) is a pure per-pixel function, so it cancels
  // when two runs are compared against each other.
  const QUIET: [string, number][] = [
    ['grain', 0], ['scanline', 0], ['chroma_bleed', 0],
  ];

  // A pass-everything emitter: no threshold, no chroma boost, white tint.
  const PASSTHRU: [string, number][] = [
    ['threshold', 0], ['knee', 0.01], ['glow_saturation', 1], ['outline', 0],
  ];

  const luma = (p: { r: number; g: number; b: number }) => (p.r + p.g + p.b) / 3;

  const colMeans = (f: Frame): number[] => {
    const out: number[] = [];
    for (let x = 0; x < f.width; x++) {
      let s = 0;
      for (let y = 0; y < f.height; y++) s += luma(f.pixelAt(x, y));
      out.push(s / f.height);
    }
    return out;
  };

  /** Fraction of pixels that are essentially black. */
  const darkFraction = (f: Frame, thresh = 8): number => {
    let n = 0;
    for (let y = 0; y < f.height; y++)
      for (let x = 0; x < f.width; x++) if (luma(f.pixelAt(x, y)) < thresh) n++;
    return n / (f.width * f.height);
  };

  const frameMean = (f: Frame) => {
    let s = 0;
    for (let y = 0; y < f.height; y++)
      for (let x = 0; x < f.width; x++) s += luma(f.pixelAt(x, y));
    return s / (f.width * f.height);
  };

  /** Largest per-pixel |r - b| anywhere in the frame. */
  const maxSplit = (f: Frame): number => {
    let m = 0;
    for (let y = 0; y < f.height; y++)
      for (let x = 0; x < f.width; x++) {
        const p = f.pixelAt(x, y);
        m = Math.max(m, Math.abs(p.r - p.b));
      }
    return m;
  };

  /** Four bright bars on black — structure to spread, with hard dark gaps. */
  const barsThen = (params: [string, number][]) => runGpuChainTest({
    chain: [
      { module: 'filter.lights_sim', params: [['input_opacity', 0]] },
      { module: MODULE, params },
    ],
    bundle: BUNDLE, width: W, height: H, inputColor: [1, 1, 1, 1],
  });

  it('declares its metadata', async () => {
    const frame = await runGpuEffectTest({
      module: MODULE, bundle: BUNDLE, inputColor: [0.5, 0.5, 0.5, 1],
      dumpName: 'vcr_halo_metadata',
    });
    expect(frame.success).toBe(true);
    expect(frame.metadata?.id).toBe(MODULE);
  });

  // The normalisation test. If the per-level weights ever stop summing to 1 —
  // or a level is dropped without renormalising — this is what catches it.
  it.each([0.1, 0.5, 0.95])('halo of a flat field reproduces it (radius %p)', async (r) => {
    // Radius compensation deliberately breaks energy conservation (that is
    // its whole job), so this test — which is about the weights summing to 1
    // — turns it off.
    const common: [string, number][] = [
      ...QUIET, ...PASSTHRU, ['halo_radius', r], ['halo_compensate', 0],
    ];

    const src = await runGpuEffectTest({
      module: MODULE, bundle: BUNDLE, width: W, height: H,
      inputColor: [0.45, 0.35, 0.6, 1],
      params: [...common, ['input_gain', 1], ['halo_gain', 0]],
      dumpName: `vcr_halo_flat_src_${r}`,
    });
    const halo = await runGpuEffectTest({
      module: MODULE, bundle: BUNDLE, width: W, height: H,
      inputColor: [0.45, 0.35, 0.6, 1],
      params: [...common, ['input_gain', 0], ['halo_gain', 1]],
      dumpName: `vcr_halo_flat_halo_${r}`,
    });
    expect(src.success && halo.success).toBe(true);

    // Half-float storage plus a bilinear chain down and back up: a couple of
    // LSBs of drift is expected, a shifted DC level is not.
    for (const [x, y] of [[W >> 2, H >> 2], [W >> 1, H >> 1], [W - 3, H - 3]]) {
      const a = src.pixelAt(x, y), b = halo.pixelAt(x, y);
      expect(Math.abs(a.r - b.r)).toBeLessThanOrEqual(3);
      expect(Math.abs(a.g - b.g)).toBeLessThanOrEqual(3);
      expect(Math.abs(a.b - b.b)).toBeLessThanOrEqual(3);
    }
  });

  it('outline band-passes: a flat field has no edges, so it cannot glow', async () => {
    const common: [string, number][] = [
      ...QUIET, ['threshold', 0], ['knee', 0.01], ['glow_saturation', 1],
      ['input_gain', 0], ['halo_gain', 2], ['halo_radius', 0.4],
    ];
    const body = await runGpuEffectTest({
      module: MODULE, bundle: BUNDLE, width: W, height: H,
      inputColor: [0.7, 0.7, 0.7, 1], params: [...common, ['outline', 0]],
      dumpName: 'vcr_halo_outline_off',
    });
    const edge = await runGpuEffectTest({
      module: MODULE, bundle: BUNDLE, width: W, height: H,
      inputColor: [0.7, 0.7, 0.7, 1], params: [...common, ['outline', 1]],
      dumpName: 'vcr_halo_outline_on',
    });
    expect(body.success && edge.success).toBe(true);
    expect(frameMean(body)).toBeGreaterThan(100);
    expect(frameMean(edge)).toBeLessThan(4);
  });

  it('spreads light into the dark gaps between bars', async () => {
    const off = await barsThen([...QUIET, ['halo_gain', 0], ['input_gain', 1]]);
    const on  = await barsThen([...QUIET, ['halo_gain', 1.5], ['input_gain', 1],
                                ['halo_radius', 0.95], ['threshold', 0.2]]);
    expect(off.success && on.success).toBe(true);

    // Most of the frame is the black gaps between bars. The halo has to put
    // light where there was none — measured as area recovered, not as peak
    // brightness, because a normalised halo trades height for width.
    expect(darkFraction(off)).toBeGreaterThan(0.5);
    expect(darkFraction(on)).toBeLessThan(darkFraction(off) - 0.15);
    expect(Math.min(...colMeans(on))).toBeGreaterThan(Math.min(...colMeans(off)));
  });

  it('a wider radius reaches further and sits flatter', async () => {
    const mk = (r: number) => barsThen([
      ...QUIET, ['input_gain', 0], ['halo_gain', 1.5],
      ['threshold', 0.2], ['halo_radius', r],
    ]);
    const tight = await mk(0.35);
    const wide  = await mk(0.9);
    expect(tight.success && wide.success).toBe(true);

    const t = colMeans(tight), w = colMeans(wide);
    const contrast = (c: number[]) => Math.max(...c) - Math.min(...c);
    // Energy is conserved as the halo widens, so the profile flattens rather
    // than brightening: the gaps come up and the peaks come down.
    expect(Math.min(...w)).toBeGreaterThan(Math.min(...t));
    expect(contrast(w)).toBeLessThan(contrast(t));
  });

  // The reason the band-pass lives in the up chain rather than in the
  // prefilter: a fixed-offset high-pass would measure the same ~2px rim at
  // every radius and every resolution, and vanish the moment the halo got
  // wide. A Laplacian octave measures a rim as wide as the octave itself.
  it('the outline band tracks the radius instead of collapsing', async () => {
    const mk = (o: number, r: number) => barsThen([
      ...QUIET, ['debug_show_halo', 1], ['halo_gain', 1], ['halo_compensate', 0],
      ['threshold', 0.2], ['outline', o], ['halo_radius', r],
    ]);
    const [loT, loW, hiT, hiW] = await Promise.all([]).then(async () => [
      await mk(0, 0.3), await mk(0, 0.9), await mk(1, 0.3), await mk(1, 0.9),
    ]);

    // Low pass, energy-normalised: total light is radius-independent.
    expect(Math.abs(frameMean(loW) - frameMean(loT))).toBeLessThan(
      0.15 * frameMean(loT));
    // Band pass: the octave's own band grows as the octave does, so total
    // light stays roughly put across a 10x radius sweep. A fixed-offset
    // high-pass would have measured the same 2px rim and then spread it ten
    // times wider — landing near a TENTH of this. That collapse is the
    // failure this test exists to catch, so the band is checked both ways.
    expect(frameMean(hiW)).toBeGreaterThan(frameMean(hiT) * 0.7);
    expect(frameMean(hiW)).toBeLessThan(frameMean(hiT) * 1.6);
  });

  it('chroma bleed splits the channels horizontally', async () => {
    const off = await barsThen([['grain', 0], ['scanline', 0], ['chroma_bleed', 0],
                                ['halo_gain', 1.2], ['halo_radius', 0.3]]);
    const on  = await barsThen([['grain', 0], ['scanline', 0], ['chroma_bleed', 0.9],
                                ['halo_gain', 1.2], ['halo_radius', 0.3]]);
    expect(off.success && on.success).toBe(true);
    // Relative, not absolute: nano_vcr_softclip runs a slightly different
    // transfer per channel by design (film dye layers), so even a white bar
    // carries some r/b separation at bleed 0.
    expect(maxSplit(on)).toBeGreaterThan(maxSplit(off) + 20);
  });

  // The highlight tint lives in the shared nano_vcr grade, so these two cases
  // cover it for source.mesh.three_planes as well. What makes it worth an
  // exact test is that the obvious implementation does NOT work: the soft clip
  // saturates around 1.1, so a tint that merely scales a blown pixel keeps
  // every channel over the knee and still resolves to white. The target has to
  // be absolute.
  it('the highlight tint recolours what is near clipping', async () => {
    // input_gain 2.5 on a 0.9 grey puts the accumulator at 2.25 — a stop over
    // the pivot, so the tint arrives in full.
    const common: [string, number | number[]][] = [
      ...QUIET, ['halo_gain', 0], ['input_gain', 2.5],
      ['warmth', 0], ['drive', 0], ['toe', 0], ['shoulder', 0],
      ['highlight_tint', [0.9, 0.05, 0.15]],
    ];
    const off = await runGpuEffectTest({
      module: MODULE, bundle: BUNDLE, width: W, height: H,
      inputColor: [0.9, 0.9, 0.9, 1],
      params: [...common, ['highlight_tint_amount', 0]] as any,
      dumpName: 'vcr_halo_tint_off',
    });
    const on = await runGpuEffectTest({
      module: MODULE, bundle: BUNDLE, width: W, height: H,
      inputColor: [0.9, 0.9, 0.9, 1],
      params: [...common, ['highlight_tint_amount', 1]] as any,
      dumpName: 'vcr_halo_tint_on',
    });
    expect(off.success && on.success).toBe(true);

    const a = off.pixelAt(W >> 1, H >> 1), b = on.pixelAt(W >> 1, H >> 1);
    // Untinted, 2.25 clips to white on every channel.
    expect(a.r).toBeGreaterThan(240);
    expect(a.b).toBeGreaterThan(240);
    // Tinted, the weak channels have to come down BELOW the soft clip's knee,
    // which is the entire point.
    expect(b.r).toBeGreaterThan(200);
    expect(b.b).toBeLessThan(140);
    expect(b.r - b.b).toBeGreaterThan(80);
  });

  it('...and leaves anything under the pivot alone', async () => {
    // Same knobs, but the accumulator now lands at 0.5 — well under the pivot,
    // so the tint must be a no-op rather than a global colour cast.
    const common: [string, number | number[]][] = [
      ...QUIET, ['halo_gain', 0], ['input_gain', 1.0],
      ['warmth', 0], ['drive', 0], ['toe', 0], ['shoulder', 0],
      ['highlight_tint', [0.9, 0.05, 0.15]],
    ];
    const off = await runGpuEffectTest({
      module: MODULE, bundle: BUNDLE, width: W, height: H,
      inputColor: [0.5, 0.5, 0.5, 1],
      params: [...common, ['highlight_tint_amount', 0]] as any,
      dumpName: 'vcr_halo_tint_lo_off',
    });
    const on = await runGpuEffectTest({
      module: MODULE, bundle: BUNDLE, width: W, height: H,
      inputColor: [0.5, 0.5, 0.5, 1],
      params: [...common, ['highlight_tint_amount', 1]] as any,
      dumpName: 'vcr_halo_tint_lo_on',
    });
    expect(off.success && on.success).toBe(true);
    const a = off.pixelAt(W >> 1, H >> 1), b = on.pixelAt(W >> 1, H >> 1);
    expect(Math.abs(a.r - b.r)).toBeLessThanOrEqual(1);
    expect(Math.abs(a.g - b.g)).toBeLessThanOrEqual(1);
    expect(Math.abs(a.b - b.b)).toBeLessThanOrEqual(1);
  });

  it('debug views isolate the halo and the emitter', async () => {
    const halo = await barsThen([...QUIET, ['debug_show_halo', 1],
                                 ['halo_gain', 1], ['halo_compensate', 0],
                                 ['halo_radius', 0.5], ['threshold', 0.2]]);
    const emit = await barsThen([...QUIET, ['debug_show_emitter', 1],
                                 ['threshold', 0.2]]);
    expect(halo.success && emit.success).toBe(true);

    // Same light, differently distributed: at unity gain a blur can only
    // lower the peaks and raise the floor. Both halves have to hold, or the
    // pyramid is scaling energy rather than spreading it.
    const he = colMeans(halo), em = colMeans(emit);
    expect(Math.max(...em)).toBeGreaterThan(Math.max(...he));
    expect(Math.min(...he)).toBeGreaterThan(Math.min(...em));
  });
});
});
