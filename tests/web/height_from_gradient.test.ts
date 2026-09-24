import { runEngineTest } from '@nano/web/test/engine-test-helpers';
import type { Sketch } from '@nano/web/src/sketch-types';

/**
 * E2E coverage for filter.height_from_gradient (nano bundle) — GPU gradient-
 * domain height reconstruction (multigrid Poisson solve).
 *
 * The effect is DETERMINISTIC per frame (no cross-frame state, no RNG), so
 * two separate captures with different params are directly comparable.
 *
 * Under test:
 *  1. Registers + renders: the six-pass pipeline (gradient → divergence →
 *     restrict → jacobi → prolong → present) dispatches cleanly and produces
 *     a non-black image from a solid input. (A solid input still yields a
 *     STRUCTURED height: the radial gradient field's divergence is non-trivial,
 *     so the reconstruction is a relief, not a flat plane.)
 *  2. Relief carries reconstructed structure: relief_scale=0 flattens the
 *     surface (uniform shade); relief_scale>0 lights the reconstructed slopes.
 *     The two must differ — proving the solver produced spatial height.
 *  3. present_mode changes the look: Hillshade vs Normals differ.
 *  4. debug_show_gradient overlays the source gradient field (differs from
 *     the normal hillshade output).
 */

// A solid mid-grey background — a deterministic, static input. Its luma is
// constant, but the radial gradient field still has a non-trivial divergence,
// so the height reconstruction is a smooth radial relief.
function buildSketch(sketchId: string, params: Record<string, unknown>): Sketch {
  return {
    anchor: null,
    chain: [
      {
        type: 'module',
        module_type: 'source.solid_color',
        instance_key: 'bg@0',
        params: { color: [0.4, 0.55, 0.7] },
      },
      {
        type: 'module',
        module_type: 'filter.height_from_gradient',
        instance_key: 'hfg@0',
        params,
      },
    ],
  };
}

async function render(sketchId: string, params: Record<string, unknown>, dumpName: string) {
  const result = await runEngineTest({
    width: 64, height: 64,
    modules: ['com.nano.testonly', 'com.nano.nano'],
    commands: [
      { type: 'createSketch', sketchId, sketch: buildSketch(sketchId, params) },
      { type: 'setTracePoints', tracePoints: [
        { id: 'out', target: { type: 'sketch_output', sketchId } },
      ]},
    ],
    waitFrames: 6,
    captureTraceIds: ['out'],
    dumpName,
  });
  expect(result.success).toBe(true);
  return result.trace('out');
}

describe('filter.height_from_gradient E2E', () => {
  jest.setTimeout(40000);

  it('registers and renders a non-black reconstruction', async () => {
    const out = await render('hfg_smoke', {}, 'hfg_smoke');
    // The six-pass solve produced a lit relief — not an all-black frame.
    out.expectNotSolidColor({ r: 0, g: 0, b: 0 }, 5);
  });

  it('reconstructs relief from the radial gradient (relief_scale drives it)', async () => {
    // relief_scale=0 → flat surface → uniform shade.
    const flat = await render('hfg_flat', { relief_scale: 0.0 }, 'hfg_flat');
    flat.expectUniformColor({}, 4);
    // relief_scale>0 → the reconstructed slopes light up → spatial variation.
    const relief = await render('hfg_relief', { relief_scale: 0.7 }, 'hfg_relief');
    relief.expectDifferentFrom(flat, 50);
  });

  it('core_radius smooths the anchor singularity (changes the reconstruction)', async () => {
    // The radial field is singular at the anchor (div ~ 1/r). core_radius
    // suppresses the field magnitude near the center, replacing the spike with
    // a smooth dome — so the reconstructed relief differs from the raw field.
    const raw    = await render('hfg_core_off', { core_radius: 0.0, relief_scale: 0.7 }, 'hfg_core_off');
    const smooth = await render('hfg_core_on',  { core_radius: 0.4, relief_scale: 0.7 }, 'hfg_core_on');
    smooth.expectDifferentFrom(raw, 50);
  });

  it('present_mode changes the visualization (Hillshade vs Normals)', async () => {
    const hill = await render('hfg_hill', { present_mode: 0, relief_scale: 0.7 }, 'hfg_hill');
    const norm = await render('hfg_norm', { present_mode: 2, relief_scale: 0.7 }, 'hfg_norm');
    norm.expectDifferentFrom(hill, 50);
  });

  it('Contours mode draws iso-lines of the reconstructed height', async () => {
    // Contour lines are sparse (mostly black bg with thin tinted lines), so it
    // differs strongly from the filled hillshade.
    const hill = await render('hfg_cont_hill', { present_mode: 0, relief_scale: 0.7 }, 'hfg_cont_hill');
    const cont = await render('hfg_cont', { present_mode: 3, contour_density: 0.4 }, 'hfg_cont');
    cont.expectNotSolidColor({ r: 0, g: 0, b: 0 }, 5);   // some lines drawn
    cont.expectDifferentFrom(hill, 50);
  });

  it('Contours at density 0 are skipped → solid black', async () => {
    const cont = await render('hfg_cont_d0', { present_mode: 3, contour_density: 0.0 }, 'hfg_cont_d0');
    cont.expectUniformColor({ r: 0, g: 0, b: 0 }, 2);
  });

  it('line_width thins the contour lines (razor at the low end)', async () => {
    const thin  = await render('hfg_cont_thin',  { present_mode: 3, contour_density: 0.4, line_width: 0.05 }, 'hfg_cont_thin');
    const thick = await render('hfg_cont_thick', { present_mode: 3, contour_density: 0.4, line_width: 0.9 }, 'hfg_cont_thick');
    thin.expectNotSolidColor({ r: 0, g: 0, b: 0 }, 5);   // still draws razor lines
    thin.expectDifferentFrom(thick, 50);                 // width visibly changes coverage
  });

  it('debug_show_gradient overlays the source gradient field', async () => {
    const normal = await render('hfg_dbg_off', { debug_show_gradient: false }, 'hfg_dbg_off');
    const debug  = await render('hfg_dbg_on',  { debug_show_gradient: true },  'hfg_dbg_on');
    debug.expectDifferentFrom(normal, 50);
  });
});

// A static rectangle gives a clean closed CONTOUR (its border) — the input
// shape a level-curves reconstruction is built for. Its edges should integrate
// into a stepped height (a plateau), so the relief is non-flat.
function buildContourSketch(sketchId: string, params: Record<string, unknown>): Sketch {
  return {
    anchor: null,
    chain: [
      {
        type: 'module',
        module_type: 'source.solid_color',
        instance_key: 'bg@0',
        params: { color: [0.1, 0.1, 0.12] },
      },
      {
        type: 'module',
        module_type: 'debug.motion_rect',
        instance_key: 'rect@0',
        // speed 0 → a static, centered rect: a deterministic closed contour.
        params: { size: 0.45, speed: 0.0, color: [0.9, 0.9, 0.9] },
      },
      {
        type: 'module',
        module_type: 'filter.height_from_gradient',
        instance_key: 'hfg@0',
        params,
      },
    ],
  };
}

async function renderContour(sketchId: string, params: Record<string, unknown>, dumpName: string) {
  const result = await runEngineTest({
    width: 64, height: 64,
    modules: ['com.nano.testonly', 'com.nano.nano'],
    commands: [
      { type: 'createSketch', sketchId, sketch: buildContourSketch(sketchId, params) },
      { type: 'setTracePoints', tracePoints: [
        { id: 'out', target: { type: 'sketch_output', sketchId } },
      ]},
    ],
    waitFrames: 6,
    captureTraceIds: ['out'],
    dumpName,
  });
  expect(result.success).toBe(true);
  return result.trace('out');
}

describe('filter.height_from_gradient — Level Curves source', () => {
  jest.setTimeout(40000);

  it('reconstructs a stepped height from a contour (edges → relief)', async () => {
    // source=1 (Level Curves). The rect's border is a contour; integrating its
    // across-curve gradient yields a stepped plateau → non-flat relief.
    const flat   = await renderContour('hfg_lc_flat',
      { source: 1, relief_scale: 0.0, edge_threshold: 0.05 }, 'hfg_lc_flat');
    flat.expectUniformColor({}, 4);
    const relief = await renderContour('hfg_lc_relief',
      { source: 1, relief_scale: 0.7, edge_threshold: 0.05 }, 'hfg_lc_relief');
    relief.expectDifferentFrom(flat, 40);
  });

  it('Level Curves differs from Radial on the same input', async () => {
    const radial = await renderContour('hfg_lc_radial',
      { source: 0, relief_scale: 0.7 }, 'hfg_lc_radial');
    const curves = await renderContour('hfg_lc_curves',
      { source: 1, relief_scale: 0.7, edge_threshold: 0.05 }, 'hfg_lc_curves');
    curves.expectDifferentFrom(radial, 40);
  });

  it('bias_mode (Radial vs Linear) changes the reconstruction', async () => {
    const radialBias = await renderContour('hfg_lc_bias_r',
      { source: 1, bias_mode: 0, relief_scale: 0.7, edge_threshold: 0.05 }, 'hfg_lc_bias_r');
    const linearBias = await renderContour('hfg_lc_bias_l',
      { source: 1, bias_mode: 1, sweep_angle: 0.0, relief_scale: 0.7, edge_threshold: 0.05 }, 'hfg_lc_bias_l');
    linearBias.expectDifferentFrom(radialBias, 30);
  });
});

describe('filter.height_from_gradient — vector sources', () => {
  jest.setTimeout(40000);

  // Grayscale present normalizes the height, so any reconstructed structure is
  // visible regardless of the source field's absolute magnitude.
  it('Normal Map source integrates the input differently than Radial', async () => {
    const radial = await renderContour('hfg_nm_radial', { source: 0, present_mode: 1 }, 'hfg_nm_radial');
    const normal = await renderContour('hfg_nm',        { source: 3, present_mode: 1 }, 'hfg_nm');
    normal.expectNotSolidColor({ r: 0, g: 0, b: 0 }, 5);
    normal.expectDifferentFrom(radial, 40);
  });

  it('Gradient Field source interprets the input differently than Normal Map', async () => {
    const grad   = await renderContour('hfg_gf',    { source: 4, present_mode: 1 }, 'hfg_gf');
    const normal = await renderContour('hfg_gf_nm', { source: 3, present_mode: 1 }, 'hfg_gf_nm');
    grad.expectNotSolidColor({ r: 0, g: 0, b: 0 }, 5);
    grad.expectDifferentFrom(normal, 30);
  });

  it('vector_sign (Signed vs Unsigned) changes the decode', async () => {
    // Hillshade present (NOT normalized) so the sign change shows: the unsigned
    // remap (2v-1) shifts the gradient field, which changes the reconstructed
    // slope → different shading. (Under grayscale's normalization a Gradient
    // Field's affine remap would cancel out — a neat invariant — so test the
    // slope-sensitive hillshade instead.)
    const signed   = await renderContour('hfg_vs_s', { source: 4, vector_sign: 0, present_mode: 0, relief_scale: 0.7 }, 'hfg_vs_s');
    const unsigned = await renderContour('hfg_vs_u', { source: 4, vector_sign: 1, present_mode: 0, relief_scale: 0.7 }, 'hfg_vs_u');
    unsigned.expectDifferentFrom(signed, 40);
  });

  it('Motion Vectors source reads the incoming render_outputs/motion rail', async () => {
    // debug.motion_rect publishes render_outputs/motion; height_from_gradient
    // (source = Motion Vectors) integrates it. Rail wired vs not must differ:
    // unwired → zero motion → flat height → uniform black grayscale.
    // Wire model: hfg's render_outputs_in auto-connects to the motion_rect
    // producer above. Negative case omits the producer → zero motion → flat.
    const buildChain = (withProducer: boolean): Sketch => ({
      anchor: null,
      wires: [],
      chain: [
        { type: 'module', module_type: 'source.solid_color', instance_key: 'bg@0', params: { color: [0.05, 0.05, 0.1] } },
        ...(withProducer ? [{
          type: 'module', module_type: 'debug.motion_rect', instance_key: 'rect@0',
          params: { size: 0.3, speed: 2.0, color: [0.9, 0.4, 0.8] },
        }] : []),
        {
          type: 'module', module_type: 'filter.height_from_gradient', instance_key: 'hfg@0',
          params: { source: 2, present_mode: 1, grad_gain: 1.0 },
        },
      ],
    } as Sketch);

    const run = (id: string, withProducer: boolean) => runEngineTest({
      width: 64, height: 64,
      modules: ['com.nano.testonly', 'com.nano.nano'],
      commands: [
        { type: 'createSketch', sketchId: id, sketch: buildChain(withProducer) },
        { type: 'setTracePoints', tracePoints: [{ id: 'out', target: { type: 'sketch_output', sketchId: id } }] },
      ],
      waitFrames: 8, captureTraceIds: ['out'], dumpName: id,
    });

    const withRail = await run('hfg_motion_rail', true);
    const noRail   = await run('hfg_motion_norail', false);
    expect(withRail.success).toBe(true);
    expect(noRail.success).toBe(true);
    // No rail → zero motion → flat → uniform black.
    noRail.trace('out').expectUniformColor({ r: 0, g: 0, b: 0 }, 4);
    // Rail wired → the motion field integrates to a non-flat height.
    withRail.trace('out').expectDifferentFrom(noRail.trace('out'), 30);
  });
});
