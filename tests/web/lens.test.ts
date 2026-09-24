import { runEngineTest } from '@nano/web/test/engine-test-helpers';
import type { Sketch } from '@nano/web/src/sketch-types';

/**
 * E2E for filter.blur.lens — "Lens" (nano bundle).
 *
 * A full photographic-lens sim: shaped-aperture bokeh gather + flare stack +
 * filmic finish. Multi-pass compute, linear-HDR, TimeIndependent. Exercised by
 * chaining a deterministic generator (source.grid, from core) → lens and
 * comparing param settings; we trace the effect's own INPUT ('in',
 * chain_entry side:'input') and the final output ('out') in one render.
 *
 * STAGE 1: only registration + the passthrough skeleton are asserted. The
 * per-pass behavioural cases land as the real passes are implemented.
 */
describe('Lens (filter.blur.lens) E2E', () => {
  jest.setTimeout(60000);

  const W = 128, H = 128;
  const MODULES = ['com.nano.testonly', 'com.nano.core', 'com.nano.nano'];
  const LENS = 'filter.blur.lens';

  const runChain = (id: string, params: Record<string, number | number[]>, dump: string,
                    gridParams: Record<string, number> = {}, waitFrames = 8) => {
    const sketch: Sketch = {
      anchor: null,
      wires: [],
      chain: [
        { type: 'module', module_type: 'source.grid', instance_key: 'grid@0', params: gridParams },
        { type: 'module', module_type: LENS, instance_key: 'lens@0', params },
      ],
    };
    return runEngineTest({
      width: W, height: H, modules: MODULES,
      commands: [
        { type: 'createSketch', sketchId: id, sketch },
        { type: 'setTracePoints', tracePoints: [
          { id: 'in',  target: { type: 'chain_entry', sketchId: id, colIdx: 0, chainIdx: 1, side: 'input' } },
          { id: 'out', target: { type: 'sketch_output', sketchId: id } },
        ]},
      ],
      waitFrames, captureTraceIds: ['in', 'out'], dumpName: dump,
    });
  };

  it('registers with the right id and declares the param surface', async () => {
    const r = await runChain('lens_reg', {}, 'lens_reg');
    expect(r.success).toBe(true);
    const p = r.state.plugins.find((x: any) => x.id === LENS);
    expect(p).toBeTruthy();
    expect(p.io.find((io: any) => io.name === 'tex_in')).toBeTruthy();
    expect(p.io.find((io: any) => io.name === 'tex_out')).toBeTruthy();
    // Scalar/int/select fields surface in the flat params array.
    const names = p.params.map((x: any) => x.name);
    for (const n of ['preset', 'in_brightness', 'in_contrast', 'blur_amount', 'blades',
                     'cats_eye', 'coating', 'sun_intensity', 'halation', 'distortion',
                     'tca', 'exposure', 'tone', 'grain', 'quality', 'debug_view']) {
      expect(names).toContain(n);
    }
    // vec2/vec3(color) fields surface in the schema (the inspector binds via it).
    const sch = typeof p.schema === 'string' ? JSON.parse(p.schema) : p.schema;
    expect(sch.focus_center?.type).toBe('float2');
    expect(sch.sun_color?.type).toBe('float3');
    expect(sch.sun_color?.hint).toBe('color');
    expect(sch.halation_color?.type).toBe('float3');
    expect(p.capabilities).toContain('time_independent');
  });

  const GRID = { cell_size: 0.22, line_width: 0.12 };

  // Larger frame so the reduced-resolution tiers actually engage (bokeh ds>1 when
  // the CoC exceeds work_radius; flare fds>1 when the long edge >~540px).
  const runBig = (id: string, params: Record<string, number>, dump: string) => {
    const sketch: Sketch = {
      anchor: null, wires: [],
      chain: [
        { type: 'module', module_type: 'source.grid', instance_key: 'grid@0', params: GRID },
        { type: 'module', module_type: LENS, instance_key: 'lens@0', params },
      ],
    };
    return runEngineTest({
      width: 640, height: 640, modules: MODULES,
      commands: [
        { type: 'createSketch', sketchId: id, sketch },
        { type: 'setTracePoints', tracePoints: [{ id: 'out', target: { type: 'sketch_output', sketchId: id } }] },
      ],
      waitFrames: 8, captureTraceIds: ['out'], dumpName: dump,
    });
  };

  it('renders a non-solid frame (the pipeline dispatches cleanly)', async () => {
    const r = await runChain('lens_render', {}, 'lens_render', GRID);
    expect(r.success).toBe(true);
    r.trace('out').expectNotSolidColor({ r: 0, g: 0, b: 0 }, 5);
  });

  it('bokeh softens the image (blur amount changes the look)', async () => {
    const sharp = await runChain('lens_b0', { blur_amount: 0.0 }, 'lens_blur_0', GRID);
    const soft  = await runChain('lens_b1', { blur_amount: 0.6 }, 'lens_blur_1', GRID);
    expect(sharp.success).toBe(true);
    expect(soft.success).toBe(true);
    soft.trace('out').expectDifferentFrom(sharp.trace('out'), 100);
  });

  it('quality tier changes the gather (Cheap vs Max differ)', async () => {
    // Cheap thins taps + downsamples more; Max adds taps + downsamples less. With a
    // big blur and no anti-stipple fill, the two tiers produce visibly different
    // discs. (quality: 0 = Cheap, 2 = Max.)
    const cheap = await runChain('lens_q0', { blur_amount: 0.6, fill: 0.0, quality: 0 }, 'lens_qual_cheap', GRID);
    const max   = await runChain('lens_q2', { blur_amount: 0.6, fill: 0.0, quality: 2 }, 'lens_qual_max', GRID);
    expect(cheap.success).toBe(true);
    expect(max.success).toBe(true);
    max.trace('out').expectDifferentFrom(cheap.trace('out'), 50);
  });

  it('onion ring textures the bokeh (0 vs 1 differ on the default coating)', async () => {
    // Onion modulates the gather tap weights radially → concentric rings in
    // defocused discs. Coating-gated but floored, so it bites even on SMC (coating
    // 0, the default). Needs real defocus (a large blur) to manifest.
    const none = await runChain('lens_o0', { blur_amount: 0.6, onion_ring: 0.0, apodize: 0.0 }, 'lens_onion_0', GRID);
    const full = await runChain('lens_o1', { blur_amount: 0.6, onion_ring: 1.0, apodize: 0.0 }, 'lens_onion_1', GRID);
    expect(none.success).toBe(true);
    expect(full.success).toBe(true);
    full.trace('out').expectDifferentFrom(none.trace('out'), 50);
  });

  it('grain bites (deterministic film grain differs from clean)', async () => {
    const clean = await runChain('lens_g0', { blur_amount: 0.3, grain: 0.0 }, 'lens_grain_0', GRID);
    const noisy = await runChain('lens_g1', { blur_amount: 0.3, grain: 1.0 }, 'lens_grain_1', GRID);
    expect(clean.success).toBe(true);
    expect(noisy.success).toBe(true);
    noisy.trace('out').expectDifferentFrom(clean.trace('out'), 50);
  });

  it('coating changes the colour grade (SMC vs Uncoated)', async () => {
    const smc = await runChain('lens_cs', { coating: 0, blur_amount: 0.2 }, 'lens_coat_smc', GRID);
    const unc = await runChain('lens_cu', { coating: 2, blur_amount: 0.2 }, 'lens_coat_unc', GRID);
    expect(smc.success).toBe(true);
    expect(unc.success).toBe(true);
    unc.trace('out').expectDifferentFrom(smc.trace('out'), 100);
  });

  it('distortion warps the geometry', async () => {
    const flat = await runChain('lens_d0', { distortion: 0.0, blur_amount: 0.1 }, 'lens_dist_0', GRID);
    const bent = await runChain('lens_d1', { distortion: 0.9, blur_amount: 0.1 }, 'lens_dist_1', GRID);
    expect(flat.success).toBe(true);
    expect(bent.success).toBe(true);
    bent.trace('out').expectDifferentFrom(flat.trace('out'), 100);
  });

  it('chromatic aberration fringes high-contrast edges', async () => {
    const clean  = await runChain('lens_t0', { tca: 0.0, blur_amount: 0.1 }, 'lens_tca_0', GRID);
    const fringe = await runChain('lens_t1', { tca: 1.0, blur_amount: 0.1 }, 'lens_tca_1', GRID);
    expect(clean.success).toBe(true);
    expect(fringe.success).toBe(true);
    fringe.trace('out').expectDifferentFrom(clean.trace('out'), 50);
  });

  it('veiling glare (hood) lifts the image when the hood retracts', async () => {
    // Highlights (grid lines) above a low threshold scatter into a wide bloom when
    // the hood is retracted (extension 0); a shaded hood (extension 1) stays clean.
    const clean = await runChain('lens_h1', { hood_extension: 1.0, flare_strength: 1.0, hl_threshold: 0.3, blur_amount: 0.1 }, 'lens_hood_clean', GRID);
    const flary = await runChain('lens_h0', { hood_extension: 0.0, flare_strength: 1.0, hl_threshold: 0.3, blur_amount: 0.1 }, 'lens_hood_flary', GRID);
    expect(clean.success).toBe(true);
    expect(flary.success).toBe(true);
    flary.trace('out').expectDifferentFrom(clean.trace('out'), 50);
  });

  it('sun / stray light adds flare when enabled and admitted', async () => {
    // The hood must be retracted (extension 0) to admit the oblique source.
    const off = await runChain('lens_s0', { sun_intensity: 0.0, hood_extension: 0.0, blur_amount: 0.1 }, 'lens_sun_off', GRID);
    const on  = await runChain('lens_s1', { sun_intensity: 1.0, hood_extension: 0.0, sun_glow: 1.0, sun_ghost: 1.0, blur_amount: 0.1 }, 'lens_sun_on', GRID);
    expect(off.success).toBe(true);
    expect(on.success).toBe(true);
    on.trace('out').expectDifferentFrom(off.trace('out'), 50);
  });

  it('halation + bloom bleed a glow from the highlights', async () => {
    // The glow is additive (it bleeds highlight energy into the surrounding dark
    // areas), so it lifts the total brightness.
    const dry = await runChain('lens_gl0', { halation: 0.0, bloom: 0.0, blur_amount: 0.0 }, 'lens_glow_off', GRID);
    const wet = await runChain('lens_gl1', { halation: 1.0, bloom: 1.0, blur_amount: 0.0 }, 'lens_glow_on', GRID);
    expect(dry.success).toBe(true);
    expect(wet.success).toBe(true);
    let sumDry = 0, sumWet = 0;
    dry.trace('out').forEachPixel((c) => { sumDry += c.r + c.g + c.b; });
    wet.trace('out').forEachPixel((c) => { sumWet += c.r + c.g + c.b; });
    expect(sumWet).toBeGreaterThan(sumDry);
  });

  it('reduced-res tiers render correctly at a large frame (bokeh ds + flare fds)', async () => {
    // Big CoC → bokeh downsamples; large frame → flare downsamples. Still must
    // produce a sensible, non-solid image and glow must still add energy.
    // Boost highlights (low threshold + gain) so the grid lines survive the flare
    // downsample as bright sources for the glow to bleed.
    const hl = { hl_threshold: 0.3, hl_boost: 1.0 };
    const dry = await runBig('lens_big0', { ...hl, blur_amount: 0.1, halation: 0.0, bloom: 0.0 }, 'lens_big_dry');
    const wet = await runBig('lens_big1', { ...hl, blur_amount: 0.1, halation: 1.0, bloom: 1.0 }, 'lens_big_wet');
    expect(dry.success).toBe(true);
    expect(wet.success).toBe(true);
    dry.trace('out').expectNotSolidColor({ r: 0, g: 0, b: 0 }, 5);
    let sumDry = 0, sumWet = 0;
    dry.trace('out').forEachPixel((c) => { sumDry += c.r + c.g + c.b; });
    wet.trace('out').forEachPixel((c) => { sumWet += c.r + c.g + c.b; });
    expect(sumWet).toBeGreaterThan(sumDry);
  });

  it('debug views each render a distinct internal stage', async () => {
    // A rich setup so every stage has content: bokeh (blur), a CoC gradient
    // (field curvature), boosted highlights (mask), and active glow (flare).
    const P = { blur_amount: 0.3, field_curvature: 0.6, hl_threshold: 0.3,
                hl_boost: 1.0, halation: 1.0, bloom: 1.0 };
    const off   = await runChain('lens_dv0', { ...P, debug_view: 0 }, 'lens_dbg_off', GRID);
    const mask  = await runChain('lens_dv1', { ...P, debug_view: 1 }, 'lens_dbg_mask', GRID);
    const coc   = await runChain('lens_dv2', { ...P, debug_view: 2 }, 'lens_dbg_coc', GRID);
    const bokeh = await runChain('lens_dv3', { ...P, debug_view: 3 }, 'lens_dbg_bokeh', GRID);
    const flare = await runChain('lens_dv4', { ...P, debug_view: 4 }, 'lens_dbg_flare', GRID);
    for (const r of [off, mask, coc, bokeh, flare]) expect(r.success).toBe(true);
    // Each debug view must differ from the normal composited output...
    mask.trace('out').expectDifferentFrom(off.trace('out'), 100);
    coc.trace('out').expectDifferentFrom(off.trace('out'), 100);
    bokeh.trace('out').expectDifferentFrom(off.trace('out'), 100);
    flare.trace('out').expectDifferentFrom(off.trace('out'), 100);
    // ...and from each OTHER (proves they're distinct plumbed stages, not one
    // fallback path). Mask (grayscale) vs CoC (grayscale gradient) vs bokeh.
    mask.trace('out').expectDifferentFrom(coc.trace('out'), 100);
    mask.trace('out').expectDifferentFrom(bokeh.trace('out'), 100);
    coc.trace('out').expectDifferentFrom(bokeh.trace('out'), 100);
    // Non-solid where the content guarantees variation (mask edges, CoC bowl).
    mask.trace('out').expectNotSolidColor({ r: 0, g: 0, b: 0 }, 5);
    coc.trace('out').expectNotSolidColor({ r: 0, g: 0, b: 0 }, 5);
  });

  it('input levels condition the image before the pipeline (default lift vs raw)', async () => {
    // The default lifts blacks (in_brightness 0.15, in_contrast -0.13) so the
    // filmic finish doesn't crush already-processed footage. Zeroing both is a
    // rawer, darker response — so the default renders brighter overall.
    const raw    = await runChain('lens_lv0', { in_brightness: 0.0, in_contrast: 0.0, grain: 0.0 }, 'lens_lvl_raw', GRID);
    const lifted = await runChain('lens_lv1', { in_brightness: 0.15, in_contrast: -0.13, grain: 0.0 }, 'lens_lvl_lift', GRID);
    expect(raw.success).toBe(true);
    expect(lifted.success).toBe(true);
    let sumRaw = 0, sumLift = 0;
    raw.trace('out').forEachPixel((c) => { sumRaw += c.r + c.g + c.b; });
    lifted.trace('out').forEachPixel((c) => { sumLift += c.r + c.g + c.b; });
    expect(sumLift).toBeGreaterThan(sumRaw);
  });

  it('exposure changes the output brightness', async () => {
    const dim    = await runChain('lens_e0', { exposure: -0.5 }, 'lens_exp_0', GRID);
    const bright = await runChain('lens_e1', { exposure: 0.5 },  'lens_exp_1', GRID);
    expect(dim.success).toBe(true);
    expect(bright.success).toBe(true);
    let sumDim = 0, sumBright = 0;
    dim.trace('out').forEachPixel((c) => { sumDim += c.r + c.g + c.b; });
    bright.trace('out').forEachPixel((c) => { sumBright += c.r + c.g + c.b; });
    expect(sumBright).toBeGreaterThan(sumDim);
  });
});
