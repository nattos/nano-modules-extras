import { runEngineTest } from '@nano/web/test/engine-test-helpers';
import type { Sketch } from '@nano/web/src/sketch-types';

/**
 * E2E coverage for source.shape_burst (nano bundle) — the triggered expanding-
 * ring generator. Each trigger fires a ring (circle / square / triangle) that
 * grows min→max scale over a duration; a `manual` 0..1 knob drives one always-on
 * ring directly, so tests can render a deterministic ring with no trigger timing.
 *
 * Under test:
 *  1. Registers + renders: manual=0.5 over a Black composite draws a non-black ring.
 *  2. All three shapes render (circle/square/triangle), and differ from each other.
 *  3. Composite=Input passes the input through everywhere but the ring.
 *  4. auto_rate self-fires: over several frames the generator produces output.
 */
function burstChain(params: Record<string, unknown>, withInput = false): Sketch['chain'] {
  const chain: Sketch['chain'] = [];
  if (withInput) {
    chain.push({
      type: 'module',
      module_type: 'source.solid_color',
      instance_key: 'bg@0',
      params: { color: [0.2, 0.4, 0.8] },
    });
  }
  chain.push({
    type: 'module',
    module_type: 'source.shape_burst',
    instance_key: 'burst@0',
    params,
  });
  return chain;
}

function buildSketch(params: Record<string, unknown>, withInput = false): Sketch {
  return { anchor: null, chain: burstChain(params, withInput) };
}

async function render(sketchId: string, params: Record<string, unknown>, dumpName: string,
                      withInput = false, waitFrames = 4) {
  const result = await runEngineTest({
    width: 96, height: 96,
    modules: ['com.nano.testonly', 'com.nano.nano'],
    commands: [
      { type: 'createSketch', sketchId, sketch: buildSketch(params, withInput) },
      { type: 'setTracePoints', tracePoints: [
        { id: 'out', target: { type: 'sketch_output', sketchId } },
      ]},
    ],
    waitFrames,
    captureTraceIds: ['out'],
    dumpName,
  });
  expect(result.success).toBe(true);
  return result;
}

// Visible ring: thick stroke, mid scale, opaque white.
const RING = { thickness: 0.1, min_scale: 0.1, max_scale: 1.0, color: [1, 1, 1, 1], manual: 0.5 };

describe('source.shape_burst E2E', () => {
  jest.setTimeout(60000);

  it('registers and renders a manual-driven ring', async () => {
    const r = await render('burst_smoke', { ...RING, composite: 0 /* Black */ }, 'burst_smoke');
    r.trace('out').expectNotSolidColor({ r: 0, g: 0, b: 0 }, 5);

    const burst = r.state.plugins.find((p: any) => p.id === 'source.shape_burst');
    expect(burst).toBeTruthy();
  });

  it('renders all three shapes, each distinct', async () => {
    const circle = await render('burst_circle', { ...RING, shape: 0, composite: 0 }, 'burst_circle');
    const square = await render('burst_square', { ...RING, shape: 1, composite: 0 }, 'burst_square');
    const triangle = await render('burst_triangle', { ...RING, shape: 2, composite: 0 }, 'burst_triangle');
    for (const f of [circle, square, triangle]) {
      f.trace('out').expectNotSolidColor({ r: 0, g: 0, b: 0 }, 5);
    }
    circle.trace('out').expectDifferentFrom(square.trace('out'), 20);
    square.trace('out').expectDifferentFrom(triangle.trace('out'), 20);
  });

  it('rotation spins squares/triangles but not circles', async () => {
    const sq0 = await render('burst_sq_r0', { ...RING, shape: 1, rotation: 0.0, composite: 0 }, 'burst_sq_r0');
    const sq1 = await render('burst_sq_r1', { ...RING, shape: 1, rotation: 0.25, composite: 0 }, 'burst_sq_r1');
    sq0.trace('out').expectDifferentFrom(sq1.trace('out'), 20);   // 45° rotated square differs

    const c0 = await render('burst_c_r0', { ...RING, shape: 0, rotation: 0.0, composite: 0 }, 'burst_c_r0');
    const c1 = await render('burst_c_r1', { ...RING, shape: 0, rotation: 0.25, composite: 0 }, 'burst_c_r1');
    c0.trace('out').expectSameAs(c1.trace('out'), 2);             // circle is rotation-invariant
  });

  it('distort warps the outline via the twitch-masked noise', async () => {
    const clean = await render('burst_clean',
      { ...RING, shape: 0, distort: 0.0, composite: 0 }, 'burst_clean');
    const warped = await render('burst_warp',
      { ...RING, shape: 0, distort: 1.0, distort_radius: 1.0, distort_freq: 0.5, composite: 0 },
      'burst_warp');
    warped.trace('out').expectNotSolidColor({ r: 0, g: 0, b: 0 }, 5);
    // A pushed/pulled outline differs from the clean circle.
    warped.trace('out').expectDifferentFrom(clean.trace('out'), 20);
  });

  it('Input composite passes the input through around the ring', async () => {
    // Thin ring so most of the frame is the untouched input colour (0.2,0.4,0.8).
    const r = await render('burst_input',
      { ...RING, thickness: 0.03, composite: 3 /* Input */ }, 'burst_input', /*withInput=*/true);
    // A clear majority of pixels should still read as the blue input.
    r.trace('out').expectCoverage(
      (c) => Math.abs(c.r - 51) < 25 && Math.abs(c.g - 102) < 25 && Math.abs(c.b - 204) < 25,
      { min: 0.6 });
    // ...but the ring itself puts some near-white pixels on screen.
    r.trace('out').expectCoverage((c) => c.r > 200 && c.g > 200 && c.b > 200, { min: 0.001 });
  });

  it('square corners are sharp (no rounding from the stroke offset)', async () => {
    // Fixed square: min=max=0.55, thickness 0.2 → outer boundary 0.65. At
    // 256×256, 1 cover unit = 128 px, centre (128,128). The corner-tip region
    // beyond the old SDF-offset rounding arc is cover (0.628..0.65) on both
    // axes → pixels 208..210. Sharp corners fill it; rounded left it black.
    const r = await runEngineTest({
      width: 256, height: 256,
      modules: ['com.nano.testonly', 'com.nano.nano'],
      commands: [
        { type: 'createSketch', sketchId: 'burst_corner', sketch: buildSketch({
          shape: 1, min_scale: 0.55, max_scale: 0.55, thickness: 0.2,
          color: [1, 1, 1, 1], manual: 0.5, composite: 0,
        }) },
        { type: 'setTracePoints', tracePoints: [
          { id: 'out', target: { type: 'sketch_output', sketchId: 'burst_corner' } },
        ]},
      ],
      waitFrames: 4,
      captureTraceIds: ['out'],
      dumpName: 'burst_corner',
    });
    expect(r.success).toBe(true);
    const f = r.trace('out');
    let bright = 0;
    for (let y = 208; y <= 210; y++)
      for (let x = 208; x <= 210; x++)
        if (f.pixelAt(x, y).r > 150) bright++;
    expect(bright).toBeGreaterThanOrEqual(6);
  });

  it('gradient shading dims the stroke edges but keeps a bright core', async () => {
    const thick = { ...RING, thickness: 0.2, shape: 0, composite: 0 };
    const solid = await render('burst_shade_solid', { ...thick, shading: 0 }, 'burst_shade_solid');
    const grad = await render('burst_shade_grad', { ...thick, shading: 1 }, 'burst_shade_grad');
    const white = (f: any) => { let n = 0; f.forEachPixel((c: any) => { if (c.r > 240) n++; }); return n; };
    const solidWhite = white(solid.trace('out'));
    const gradWhite = white(grad.trace('out'));
    expect(gradWhite).toBeGreaterThan(0);                 // the bell peak still hits full alpha
    expect(gradWhite).toBeLessThan(solidWhite * 0.6);     // ...but the flanks fall off
    grad.trace('out').expectDifferentFrom(solid.trace('out'), 20);
  });

  it('shade tilt slides the bright core inner <-> outer', async () => {
    // Circle, centre (48,48), scale 0.55 → mid radius ~26 px, half-thickness
    // ~5 px. The mean radius of bright pixels must move with the tilt sign
    // (positive = inner, matching Motion's Tilt convention).
    const thick = { ...RING, thickness: 0.2, shape: 0, composite: 0, shading: 1 };
    const inner = await render('burst_stilt_in', { ...thick, shade_tilt: 1.0 }, 'burst_stilt_in');
    const outer = await render('burst_stilt_out', { ...thick, shade_tilt: -1.0 }, 'burst_stilt_out');
    const meanRadius = (f: any) => {
      let sum = 0, n = 0;
      f.forEachPixel((c: any, x: number, y: number) => {
        if (c.r > 200) { sum += Math.hypot(x - 48, y - 48); n++; }
      });
      expect(n).toBeGreaterThan(0);
      return sum / n;
    };
    const rIn = meanRadius(inner.trace('out'));
    const rOut = meanRadius(outer.trace('out'));
    expect(rIn).toBeLessThan(rOut - 2);
  });

  it('auto_rate self-fires and produces output', async () => {
    // No manual voice; rely on Poisson auto-trigger over several frames.
    const r = await render('burst_auto',
      { thickness: 0.1, min_scale: 0.1, max_scale: 1.0, color: [1, 1, 1, 1],
        manual: 0, auto_mode: 1 /* Random */, auto_rate: 1.0, composite: 0 },
      'burst_auto', /*withInput=*/false, /*waitFrames=*/16);
    r.trace('out').expectNotSolidColor({ r: 0, g: 0, b: 0 }, 5);
  });

  // shape_burst -> motion.blur: the expanding rings write a radial motion rail
  // that the blur consumes. RNG is seeded identically per instance, so with the
  // same params the ring (color) frames are identical between runs — only
  // motion_strength changes the motion rail, so any output difference is proof
  // the motion vectors are produced AND consumed downstream.
  function motionSketch(motion_strength: number, tilt = 0, thickness = 0.08): any {
    return {
      anchor: null,
      wires: [],   // opt into wire mode → motion.blur auto-connects the rail above it
      chain: [
        {
          type: 'module',
          module_type: 'source.shape_burst',
          instance_key: 'burst@0',
          params: {
            thickness, min_scale: 0.1, max_scale: 1.3, color: [1, 1, 1, 1],
            manual: 0, auto_mode: 1 /* Random */, auto_rate: 1.0, duration: 0.2,
            composite: 0, motion_strength, tilt,
          },
        },
        {
          type: 'module',
          module_type: 'motion.blur',
          instance_key: 'blur@0',
          params: { strength: 32.0, samples: 16, quality: 1 },
        },
      ],
    };
  }

  async function renderMotion(sketchId: string, motion_strength: number,
                              tilt = 0, thickness = 0.08) {
    const r = await runEngineTest({
      width: 96, height: 96,
      modules: ['com.nano.testonly', 'com.nano.nano'],
      commands: [
        { type: 'createSketch', sketchId, sketch: motionSketch(motion_strength, tilt, thickness) },
        { type: 'setTracePoints', tracePoints: [
          { id: 'out', target: { type: 'sketch_output', sketchId } },
        ]},
      ],
      waitFrames: 22,
      captureTraceIds: ['out'],
      dumpName: sketchId,
    });
    expect(r.success).toBe(true);
    return r;
  }

  it('emits a motion rail that drives a downstream motion blur', async () => {
    const withMotion = await renderMotion('burst_mot_on', 4.0);
    const noMotion = await renderMotion('burst_mot_off', 0.0);
    withMotion.trace('out').expectNotSolidColor({ r: 0, g: 0, b: 0 }, 5);
    // Same rings, different motion rail → the blur output must diverge.
    withMotion.trace('out').expectDifferentFrom(noMotion.trace('out'), 20);
  });

  it('tilt redistributes motion magnitude across the ring band', async () => {
    // Thick stroke so the inner/outer magnitude split is resolvable. Same rings
    // both runs; only the tilt sign flips the inner<->outer weighting, so the
    // resulting blur differs.
    const inner = await renderMotion('burst_tilt_in', 4.0, 1.0, 0.16);
    const outer = await renderMotion('burst_tilt_out', 4.0, -1.0, 0.16);
    inner.trace('out').expectDifferentFrom(outer.trace('out'), 20);
  });
});
