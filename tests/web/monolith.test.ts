import { runEngineTest, runEngineMultiPhaseTest } from '@nano/web/test/engine-test-helpers';
import type { Sketch } from '@nano/web/src/sketch-types';

/**
 * E2E coverage for source.mesh.monolith (nano bundle) — the deferred env-lit
 * 3D primitive generator (1:4:9 slab / regular pyramid, ≤3 concentric
 * shells, fresnel env reflections, refraction glass, fog, god rays).
 *
 * Determinism trick: `motion: Arc` with `arc: 0` pins the pose to a slow
 * eased sweep start (yaw ≈ -20°, drift far below a pixel over test spans).
 * Static comparisons also pin `vantage: 0, loom: 0` (neutral camera) and
 * zero the atmosphere unless it is under test. Animation is asserted
 * structurally (phases differ), never pixel-exact — headless frame pacing
 * varies 4–20 ms.
 *
 * Probe rig: bright solid input (0.7, 0.7, 0.75) ≈ (178, 178, 191) so the
 * default VOID-BLACK slab reads against it. Engine traces are
 * checkerboard-composited (alpha always 255) — assert by color.
 */

const BG = { r: 178, g: 178, b: 191 };
const lum = (c: { r: number, g: number, b: number }) => (c.r + c.g + c.b) / 3;

// Frozen pose + neutral camera + no atmosphere: the baseline for A/B tests.
const STATIC = {
  motion: 0 /* Arc */, arc: 0.0, tilt: 0.2, size: 0.8, opacity: 1.0,
  vantage: 0.0, loom: 0.0, fog: 0.0, rays: 0.0,
  // The caustic water clock is free-running, so A/B comparisons pin it off.
  caustics: 0.0,
};

function buildSketch(params: Record<string, unknown>, opts?: {
  gradientInput?: boolean,   // insert a gradient as mono's tex_in
  envWire?: boolean,         // wire the SOLID bg into mono's env_in
  bgColor?: number[],        // override the solid input color
  tintColor?: number[],      // add a solid producer wired into tint_in
}): Sketch {
  const chain: any[] = [];
  const wires: any[] = [];
  if (opts?.tintColor) {
    // First in the chain so it does NOT become mono's tex_in (adjacency
    // picks the entry directly above).
    chain.push({ type: 'module', module_type: 'source.solid_color',
                 instance_key: 'tint@0', params: { color: opts.tintColor } });
    wires.push({ id: 'wt', src: { instanceKey: 'tint@0', field: 'tex_out' },
                 dest: { instanceKey: 'mono@0', field: 'tint_in' } });
  }
  chain.push({ type: 'module', module_type: 'source.solid_color',
               instance_key: 'bg@0',
               params: { color: opts?.bgColor ?? [0.7, 0.7, 0.75] } });
  if (opts?.gradientInput) {
    // Chain adjacency makes the LAST entry above mono its tex_in.
    chain.push({ type: 'module', module_type: 'source.gradient',
                 instance_key: 'grad@0', params: {} });
  }
  if (opts?.envWire) {
    wires.push({ id: 'we', src: { instanceKey: 'bg@0', field: 'tex_out' },
                 dest: { instanceKey: 'mono@0', field: 'env_in' } });
  }
  chain.push({ type: 'module', module_type: 'source.mesh.monolith',
               instance_key: 'mono@0', params });
  return { anchor: null, chain, wires } as Sketch;
}

async function render(sketchId: string, params: Record<string, unknown>,
                      opts?: { gradientInput?: boolean, envWire?: boolean,
                               bgColor?: number[], tintColor?: number[],
                               waitFrames?: number }) {
  const result = await runEngineTest({
    width: 96, height: 96,
    modules: ['com.nano.core', 'com.nano.nano'],
    commands: [
      { type: 'createSketch', sketchId, sketch: buildSketch(params, opts) },
      { type: 'setTracePoints', tracePoints: [
        { id: 'out', target: { type: 'sketch_output', sketchId } },
      ]},
    ],
    waitFrames: opts?.waitFrames ?? 4,
    captureTraceIds: ['out'],
    dumpName: sketchId,
  });
  expect(result.success).toBe(true);
  return result;
}

describe('source.mesh.monolith E2E', () => {
  jest.setTimeout(120000);

  it('void-black default slab reads against a bright input', async () => {
    // Default near-black color: the body must sit clearly darker than the
    // input while the corners stay untouched.
    const r = await render('mono_smoke', { ...STATIC });
    const center = r.trace('out').pixelAt(48, 48);
    expect(lum(center)).toBeLessThan(lum(BG) - 40);
    r.trace('out').expectPixelAt(3, 3, BG, 8);
    r.trace('out').expectPixelAt(92, 92, BG, 8);
    const plugin = r.state.plugins.find((p: any) => p.id === 'source.mesh.monolith');
    expect(plugin).toBeTruthy();
  });

  it('opacity 0 is a bit-exact passthrough even with atmosphere cranked', async () => {
    const r = await render('mono_passthrough',
      { ...STATIC, opacity: 0, fog: 1, rays: 1 });
    for (const [x, y] of [[48, 48], [30, 30], [66, 66], [48, 20], [3, 3]]) {
      r.trace('out').expectPixelAt(x, y, BG, 4);
    }
  });

  it('a wired env_in switches the reflection source', async () => {
    // Same chain both runs: gradient is mono's tex_in (adjacency). Run A
    // reflects the gradient via the screen-space fallback; run B wires the
    // SOLID bg into env_in (equirect). Reflections change ON the shape;
    // the untouched corners (the gradient) stay identical across runs.
    const base = { ...STATIC, color: [0.3, 0.3, 0.3], reflect: 1.0, roughness: 0 };
    const fallback = await render('mono_env_off', base, { gradientInput: true });
    const wired = await render('mono_env_on', base,
      { gradientInput: true, envWire: true });
    wired.trace('out').expectDifferentFrom(fallback.trace('out'), 8);
    const cA = fallback.trace('out').pixelAt(3, 3);
    const cB = wired.trace('out').pixelAt(3, 3);
    expect(Math.abs(cA.r - cB.r)).toBeLessThanOrEqual(4);
    expect(Math.abs(cA.g - cB.g)).toBeLessThanOrEqual(4);
    expect(Math.abs(cA.b - cB.b)).toBeLessThanOrEqual(4);
  });

  it('glass (low opacity + refract) differs from the solid slab', async () => {
    const solid = await render('mono_solid', { ...STATIC });
    const glass = await render('mono_glass',
      { ...STATIC, opacity: 0.15, refract: 0.9 });
    glass.trace('out').expectDifferentFrom(solid.trace('out'), 15);
    glass.trace('out').expectPixelAt(3, 3, BG, 8);
    // Glass over a uniform input is BRIGHTER than the black solid at center.
    const gc = glass.trace('out').pixelAt(48, 48);
    const sc = solid.trace('out').pixelAt(48, 48);
    expect(lum(gc)).toBeGreaterThan(lum(sc) + 20);
  });

  it('sun azimuth moves the shading', async () => {
    const base = { ...STATIC, color: [0.5, 0.5, 0.5], reflect: 0.3 };
    const left = await render('mono_sun_l', { ...base, azimuth: -60, elevation: 20 });
    const right = await render('mono_sun_r', { ...base, azimuth: 60, elevation: 20 });
    left.trace('out').expectDifferentFrom(right.trace('out'), 8);
  });

  it('fog melts the structure, strongest at the top', async () => {
    const base = { ...STATIC, size: 1.0 };
    const clear = await render('mono_fog_off', base);
    const foggy = await render('mono_fog_on', { ...base, fog: 1.0 });
    // Upper body pixel pulls toward the haze (backdrop) — a real change.
    const upperDelta = Math.abs(lum(foggy.trace('out').pixelAt(48, 26)) -
                                lum(clear.trace('out').pixelAt(48, 26)));
    expect(upperDelta).toBeGreaterThan(6);
    // Uncovered pixels never fog.
    foggy.trace('out').expectPixelAt(3, 3, BG, 8);
    foggy.trace('out').expectPixelAt(92, 92, BG, 8);
  });

  it('fog color tints the medium', async () => {
    const base = { ...STATIC, size: 1.0, fog: 1.0 };
    const white = await render('mono_fogcol_white', base);
    const red = await render('mono_fogcol_red', { ...base, fog_color: [1.0, 0.15, 0.1] });
    // Upper body (heavily fogged): the red medium shows a clear red excess.
    const pw = white.trace('out').pixelAt(48, 26);
    const pr = red.trace('out').pixelAt(48, 26);
    expect(pr.r - pr.b).toBeGreaterThan(pw.r - pw.b + 8);
    // Uncovered pixels never fog, tinted or not.
    red.trace('out').expectPixelAt(3, 3, BG, 8);
  });

  it('fog scale decouples the haze from the backdrop', async () => {
    // Gradient input, so the fog's scene sample varies spatially (bright at
    // the left edge → dark at the right). Aligned, the haze carries that
    // ramp straight through the body; zoomed, every body pixel samples
    // nearer the gradient's CENTRE, so the ramp COLLAPSES — that flattening
    // is the decoupling. Probe the ramp across the body rather than one
    // pixel: fog fraction is thin near the top of the frame (the height ramp
    // is thinnest there and depth haze hasn't built up), so a single upper
    // pixel moves by ~2 — under the noise floor — while the ramp signal is
    // a consistent ~7 across the whole lower body.
    const base = { ...STATIC, size: 1.0, fog: 1.0 };
    const aligned = await render('mono_fogscale_0', base, { gradientInput: true });
    const zoomed = await render('mono_fogscale_1',
      { ...base, fog_scale: 1.0 }, { gradientInput: true });
    // x = 36/58 sit well inside the slab (it spans x ≈ 31…64) at these rows.
    const ramp = (r: any, y: number) =>
      lum(r.trace('out').pixelAt(36, y)) - lum(r.trace('out').pixelAt(58, y));
    for (const y of [50, 60, 74]) {
      // The haze reads the backdrop 1:1 — bright left, dark right.
      expect(ramp(aligned, y)).toBeGreaterThan(8);
      // Zooming the sample flattens it toward the centre value.
      expect(ramp(aligned, y) - ramp(zoomed, y)).toBeGreaterThan(4);
    }
    // Uncovered pixels identical between the two runs.
    const ca = aligned.trace('out').pixelAt(3, 3);
    const cz = zoomed.trace('out').pixelAt(3, 3);
    expect(Math.abs(ca.r - cz.r)).toBeLessThanOrEqual(4);
  });

  it('vantage makes the verticals converge (towering)', async () => {
    const width = (r: any, y: number) => {
      let n = 0;
      for (let x = 0; x < 96; x++) {
        if (lum(r.trace('out').pixelAt(x, y)) < lum(BG) - 40) n++;
      }
      return n;
    };
    const flat = await render('mono_vant0', { ...STATIC, size: 1.0 });
    const worm = await render('mono_vant1', { ...STATIC, size: 1.0, vantage: 1.0 });
    const flatRatio = width(flat, 22) / Math.max(1, width(flat, 74));
    const wormRatio = width(worm, 22) / Math.max(1, width(worm, 74));
    expect(width(worm, 74)).toBeGreaterThan(0);
    expect(wormRatio).toBeLessThan(flatRatio - 0.05);
  });

  it('god rays bleed around the silhouette when the sun is behind', async () => {
    // Sun dead behind the shape (azimuth 180): rays radiate from center.
    const base = { ...STATIC, azimuth: 180, elevation: 5, sun: 1.0 };
    const dark = await render('mono_rays_off', base);
    const lit = await render('mono_rays_on', { ...base, rays: 1.0 });
    // Just outside the slab's right edge: scattered light raises brightness.
    const probe = { x: 72, y: 48 };
    const dLit = lum(lit.trace('out').pixelAt(probe.x, probe.y));
    const dDark = lum(dark.trace('out').pixelAt(probe.x, probe.y));
    expect(dLit).toBeGreaterThan(dDark + 3);
    lit.trace('out').expectDifferentFrom(dark.trace('out'), 4);
  });

  it('caustics dapple up-facing surfaces', async () => {
    // Gray body leaning BACK (tilt -0.6 → the big front face points up
    // toward the water surface), no rays: dapple is the only change.
    const base = { ...STATIC, color: [0.5, 0.5, 0.5], reflect: 0.2,
                   tilt: -0.6, azimuth: -30, elevation: 30 };
    const still = await render('mono_caust_off', base);
    const dappled = await render('mono_caust_on', { ...base, caustics: 1.0 });
    dappled.trace('out').expectDifferentFrom(still.trace('out'), 6);
    // Dapple is a lit-surface term — the background never ripples.
    dappled.trace('out').expectPixelAt(3, 3, BG, 8);
    dappled.trace('out').expectPixelAt(92, 92, BG, 8);
  });

  it('caustics never reach downward-facing surfaces', async () => {
    // Same body leaning FORWARD (tilt 0.6): the big front face points
    // below horizontal → projected-irradiance weight is zero, so cranking
    // caustics changes nothing on its pixels (probes sit mid-face, away
    // from the up-facing top-cap strip and the vertical side face).
    const base = { ...STATIC, color: [0.5, 0.5, 0.5], reflect: 0.2,
                   tilt: 0.6, azimuth: -30, elevation: 30 };
    const off = await render('mono_caust_down_off', base);
    const on = await render('mono_caust_down_on', { ...base, caustics: 1.0 });
    for (const [x, y] of [[44, 40], [44, 56], [48, 48]]) {
      const a = off.trace('out').pixelAt(x, y);
      const b = on.trace('out').pixelAt(x, y);
      expect(Math.abs(a.r - b.r)).toBeLessThanOrEqual(3);
      expect(Math.abs(a.g - b.g)).toBeLessThanOrEqual(3);
      expect(Math.abs(a.b - b.b)).toBeLessThanOrEqual(3);
    }
  });

  it('caustic speed 0 freezes the water; default speed evolves it', async () => {
    // Multi-phase on ONE instance: frozen pose (STATIC) + caustics on.
    const phased = async (id: string, caustic_speed: number) => {
      const r = await runEngineMultiPhaseTest({
        width: 96, height: 96,
        modules: ['com.nano.core', 'com.nano.nano'],
        dumpName: id,
        phases: [
          {
            commands: [
              { type: 'createSketch', sketchId: id,
                sketch: buildSketch({ ...STATIC, color: [0.5, 0.5, 0.5],
                                      tilt: -0.6, caustics: 1.0, caustic_speed }) },
              { type: 'setTracePoints', tracePoints: [
                { id: 'out', target: { type: 'sketch_output', sketchId: id } },
              ]},
            ],
            waitFrames: 3, captureTraceIds: ['out'],
          },
          { waitFrames: 40, captureTraceIds: ['out'] },
        ],
      });
      expect(r.success).toBe(true);
      return r;
    };
    // Frozen: mid-face probes identical (edges excluded — the eased Arc
    // start still drifts the pose sub-degree between phases).
    const frozen = await phased('mono_cspd_frozen', 0.0);
    for (const [x, y] of [[48, 40], [44, 52], [52, 60]]) {
      const a = frozen.phases[0].trace('out').pixelAt(x, y);
      const b = frozen.phases[1].trace('out').pixelAt(x, y);
      expect(Math.abs(a.r - b.r)).toBeLessThanOrEqual(3);
      expect(Math.abs(a.g - b.g)).toBeLessThanOrEqual(3);
      expect(Math.abs(a.b - b.b)).toBeLessThanOrEqual(3);
    }
    const flowing = await phased('mono_cspd_flow', 0.9);
    flowing.phases[1].trace('out').expectDifferentFrom(flowing.phases[0].trace('out'), 4);
  });

  it('caustics band the god rays', async () => {
    // Backlit with rays on: caustics modulate the shafts by angle, so the
    // region outside the silhouette changes; without rays or coverage the
    // pixel is untouched.
    const base = { ...STATIC, azimuth: 180, elevation: 5, sun: 1.0, rays: 1.0 };
    const smooth = await render('mono_caust_rays_off', base);
    const banded = await render('mono_caust_rays_on', { ...base, caustics: 1.0 });
    banded.trace('out').expectDifferentFrom(smooth.trace('out'), 3);
    // Rays (banded or not) only ADD light over the background.
    expect(lum(banded.trace('out').pixelAt(3, 3))).toBeGreaterThan(lum(BG) - 4);
  });

  it('sun color tints the light and the rays', async () => {
    // Red sun over the backlit ray field: the shafts outside the shape
    // must carry the tint (r >> b at a ray-lit probe).
    const base = { ...STATIC, azimuth: 180, elevation: 5, sun: 1.0, rays: 1.0,
                   sun_color: [1.0, 0.25, 0.15] };
    const red = await render('mono_suncol_red', base);
    const white = await render('mono_suncol_white',
      { ...base, sun_color: [1, 1, 1] });
    red.trace('out').expectDifferentFrom(white.trace('out'), 5);
    const p = red.trace('out').pixelAt(72, 48);
    expect(p.r).toBeGreaterThan(p.b + 8);
  });

  it('sun source From Input steals the ray hue from the input', async () => {
    // Deep-blue input; the sun sample point lands on it (solid color, so
    // anywhere works). From Input rays carry blue chroma; manual white
    // rays stay neutral — compare the blue-minus-red excess at a ray-lit
    // probe between the two modes.
    const base = { ...STATIC, azimuth: 180, elevation: 5, sun: 1.0, rays: 1.0 };
    const bg = { bgColor: [0.15, 0.3, 0.85] };
    const manual = await render('mono_sunsrc_col', { ...base, sun_source: 0 }, bg);
    const fromEnv = await render('mono_sunsrc_env', { ...base, sun_source: 1 }, bg);
    const pm = manual.trace('out').pixelAt(72, 48);
    const pe = fromEnv.trace('out').pixelAt(72, 48);
    expect(pe.b - pe.r).toBeGreaterThan(pm.b - pm.r + 5);
  });

  it('sun source From Tint Tex reads the wired tint, not the input', async () => {
    // Blue input + RED tint tex: From Input rays go blue-heavy, From Tint
    // Tex rays go red-heavy — opposite signs at the same ray probe.
    const base = { ...STATIC, azimuth: 180, elevation: 5, sun: 1.0, rays: 1.0 };
    const opts = { bgColor: [0.15, 0.3, 0.85], tintColor: [0.9, 0.12, 0.1] };
    const fromInput = await render('mono_tint_input', { ...base, sun_source: 1 }, opts);
    const fromTint = await render('mono_tint_tex', { ...base, sun_source: 2 }, opts);
    const pi = fromInput.trace('out').pixelAt(72, 48);
    const pt = fromTint.trace('out').pixelAt(72, 48);
    expect(pi.b - pi.r).toBeGreaterThan(5);    // input hue: blue excess
    expect(pt.r - pt.b).toBeGreaterThan(5);    // tint hue: red excess
  });

  it('extended spread grows colossal shells without breaking the core', async () => {
    const near = await render('mono_spread_mid',
      { ...STATIC, copies: 3, spread: 0.5, falloff: 0.3 });
    const huge = await render('mono_spread_max',
      { ...STATIC, copies: 3, spread: 1.0, falloff: 0.3 });
    huge.trace('out').expectDifferentFrom(near.trace('out'), 10);
    // The core slab still reads at center regardless of shell scale.
    const c = huge.trace('out').pixelAt(48, 48);
    expect(lum(c)).toBeLessThan(lum(BG) - 20);
  });

  it('tumble animates across frames', async () => {
    const moving = await runEngineMultiPhaseTest({
      width: 96, height: 96,
      modules: ['com.nano.core', 'com.nano.nano'],
      dumpName: 'mono_anim',
      phases: [
        {
          commands: [
            { type: 'createSketch', sketchId: 'mono_anim',
              sketch: buildSketch({ ...STATIC, motion: 1 /* Tumble */, speed: 0.9 }) },
            { type: 'setTracePoints', tracePoints: [
              { id: 'out', target: { type: 'sketch_output', sketchId: 'mono_anim' } },
            ]},
          ],
          waitFrames: 3, captureTraceIds: ['out'],
        },
        { waitFrames: 30, captureTraceIds: ['out'] },
      ],
    });
    expect(moving.success).toBe(true);
    moving.phases[1].trace('out').expectDifferentFrom(moving.phases[0].trace('out'), 20);
  });

  it('concentric copies add shells around the core', async () => {
    const one = await render('mono_copies1', { ...STATIC, copies: 1 });
    const three = await render('mono_copies3',
      { ...STATIC, copies: 3, spread: 1.0, falloff: 0.2 });
    three.trace('out').expectDifferentFrom(one.trace('out'), 15);
  });

  it('pyramid and monolith silhouettes differ', async () => {
    const slab = await render('mono_shape_slab', { ...STATIC, shape: 0 });
    const pyramid = await render('mono_shape_pyramid', { ...STATIC, shape: 1 });
    pyramid.trace('out').expectNotSolidColor(BG, 5);
    pyramid.trace('out').expectDifferentFrom(slab.trace('out'), 20);
  });
});
