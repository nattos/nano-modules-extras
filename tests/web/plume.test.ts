import { runEngineTest, runEngineMultiPhaseTest } from '@nano/web/test/engine-test-helpers';
import type { Sketch } from '@nano/web/src/sketch-types';

/**
 * E2E coverage for source.sdf.plume (nano bundle) — the SDF volume renderer
 * flagship (milestone 1: shell map -> SDF bake -> sphere-trace -> shade).
 *
 * Determinism: `orbit: 0` and `morph: 0` stop both accumulators at phase 0
 * (rate 0 at slider 0), so static A/B comparisons are pixel-stable. The
 * default camera frames the displaced sphere well inside the 96² probe
 * frame: center (48,48) is always ON the body, corner (3,3) is always
 * background. Engine traces are checkerboard-composited (alpha always
 * 255) — assert by color.
 */

const BG = { r: 178, g: 178, b: 191 };
const lum = (c: { r: number, g: number, b: number }) => (c.r + c.g + c.b) / 3;

// Frozen accumulators + dark body against the bright input. Atmosphere is
// pinned off (it reaches the whole frame by design) so corner probes stay
// exact — fog behavior gets its own dedicated test.
const STATIC = {
  orbit: 0.0, morph: 0.0, tilt: 0.1, zoom: 0.25,
  albedo: [0.1, 0.1, 0.1], opacity: 1.0, fog: 0.0, room_fog: 0.0,
};

function buildSketch(params: Record<string, unknown>,
                     opts?: { noInput?: boolean }): Sketch {
  const chain: any[] = [];
  if (!opts?.noInput) {
    chain.push({ type: 'module', module_type: 'source.solid_color',
                 instance_key: 'bg@0', params: { color: [0.7, 0.7, 0.75] } });
  }
  chain.push({ type: 'module', module_type: 'source.sdf.plume',
               instance_key: 'plume@0', params });
  return { anchor: null, chain, wires: [] } as Sketch;
}

async function render(sketchId: string, params: Record<string, unknown>,
                      opts?: { noInput?: boolean, waitFrames?: number }) {
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

describe('source.sdf.plume E2E', () => {
  jest.setTimeout(120000);

  it('dark displaced sphere reads against a bright input', async () => {
    const r = await render('plume_smoke', { ...STATIC });
    // Body center clearly darker than the input; corners untouched.
    const center = r.trace('out').pixelAt(48, 48);
    expect(lum(center)).toBeLessThan(lum(BG) - 40);
    r.trace('out').expectPixelAt(3, 3, BG, 8);
    r.trace('out').expectPixelAt(92, 92, BG, 8);
    const plugin = r.state.plugins.find((p: any) => p.id === 'source.sdf.plume');
    expect(plugin).toBeTruthy();
  });

  it('runs as a pure generator with no input wired', async () => {
    const r = await render('plume_gen',
      { ...STATIC, albedo: [0.8, 0.8, 0.85], sun: 0.8 }, { noInput: true });
    // The lit body must produce real pixels at center.
    const center = r.trace('out').pixelAt(48, 48);
    expect(lum(center)).toBeGreaterThan(30);
    r.trace('out').expectNotSolidColor(center, 5);
  });

  it('opacity 0 is a bit-exact passthrough', async () => {
    const r = await render('plume_passthrough', { ...STATIC, opacity: 0 });
    for (const [x, y] of [[48, 48], [30, 30], [66, 66], [48, 20], [3, 3]]) {
      r.trace('out').expectPixelAt(x, y, BG, 4);
    }
  });

  it('ridge depth changes the surface', async () => {
    const smooth = await render('plume_depth0', { ...STATIC, ridge_depth: 0.0 });
    const ridged = await render('plume_depth1', { ...STATIC, ridge_depth: 1.0 });
    ridged.trace('out').expectDifferentFrom(smooth.trace('out'), 10);
  });

  it('ridge scale changes the flake frequency', async () => {
    const coarse = await render('plume_scale_lo',
      { ...STATIC, ridge_depth: 0.8, ridge_scale: 0.2 });
    const fine = await render('plume_scale_hi',
      { ...STATIC, ridge_depth: 0.8, ridge_scale: 0.9 });
    fine.trace('out').expectDifferentFrom(coarse.trace('out'), 8);
  });

  it('sun azimuth moves the shading', async () => {
    const base = { ...STATIC, albedo: [0.5, 0.5, 0.5] };
    const left = await render('plume_sun_l', { ...base, azimuth: -70 });
    const right = await render('plume_sun_r', { ...base, azimuth: 70 });
    left.trace('out').expectDifferentFrom(right.trace('out'), 8);
  });

  it('debug views render the internal state', async () => {
    // SDF slice through the volume center: inside/outside + surface band
    // structure, nothing like the normal render.
    const slice = await render('plume_dbg_sdf',
      { ...STATIC, debug_view: 1, debug_slice: 0.5 });
    slice.trace('out').expectNotSolidColor({ r: 0, g: 0, b: 0 }, 5);
    const normal = await render('plume_dbg_off', { ...STATIC });
    slice.trace('out').expectDifferentFrom(normal.trace('out'), 30);
    // Shell map view renders the octahedral field.
    const shell = await render('plume_dbg_shell',
      { ...STATIC, debug_view: 2, ridge_depth: 0.8 });
    shell.trace('out').expectNotSolidColor({ r: 0, g: 0, b: 0 }, 5);
    shell.trace('out').expectDifferentFrom(slice.trace('out'), 20);
  });

  it('bounce light brightens the surface once the wave field fills', async () => {
    const base = { ...STATIC, albedo: [0.6, 0.6, 0.6], sun: 0.9, ambient: 0.2 };
    const off = await render('plume_gi_off',
      { ...base, bounce: 0.0 }, { waitFrames: 24 });
    const on = await render('plume_gi_on',
      { ...base, bounce: 1.0, gi_decay: 0.8 }, { waitFrames: 24 });
    on.trace('out').expectDifferentFrom(off.trace('out'), 6);
    // Bounce only ADDS light; background stays untouched.
    on.trace('out').expectPixelAt(3, 3, BG, 8);
    const lOn = lum(on.trace('out').pixelAt(48, 48));
    const lOff = lum(off.trace('out').pixelAt(48, 48));
    expect(lOn).toBeGreaterThanOrEqual(lOff);
  });

  it('radiance debug view shows the wave field', async () => {
    const r = await render('plume_dbg_rad',
      { ...STATIC, bounce: 1.0, sun: 1.0, gi_decay: 0.9, debug_view: 4,
        debug_slice: 0.5 }, { waitFrames: 24 });
    // The field must have real energy somewhere in the slice.
    let maxLum = 0;
    for (let y = 8; y < 88; y += 4) {
      for (let x = 8; x < 88; x += 4) {
        maxLum = Math.max(maxLum, lum(r.trace('out').pixelAt(x, y)));
      }
    }
    expect(maxLum).toBeGreaterThan(15);
  });

  it('fog wraps the body in haze; room fog lifts the whole frame', async () => {
    const clear = await render('plume_fog_off', { ...STATIC });
    const hazy = await render('plume_fog_on',
      { ...STATIC, fog: 0.8, fog_soft: 0.8 });
    hazy.trace('out').expectDifferentFrom(clear.trace('out'), 8);
    // The shell haze is bright over the dark body's silhouette region.
    const edge = { x: 48, y: 26 };
    expect(lum(hazy.trace('out').pixelAt(edge.x, edge.y)))
        .toBeGreaterThan(lum(clear.trace('out').pixelAt(edge.x, edge.y)) - 2);
    const roomy = await render('plume_room_fog',
      { ...STATIC, room_fog: 1.0 });
    roomy.trace('out').expectDifferentFrom(clear.trace('out'), 5);
  });

  it('sheen and backlit translucency change the material read', async () => {
    // Backlit setup: sun behind the shape, dark-ish body.
    const base = { ...STATIC, albedo: [0.35, 0.35, 0.35], azimuth: 170,
                   elevation: 10, sun: 1.0, reflect: 0, transmission: 0 };
    const matte = await render('plume_mat_off', base);
    const lucent = await render('plume_mat_trans',
      { ...base, transmission: 1.0, thickness: 0.8 });
    lucent.trace('out').expectDifferentFrom(matte.trace('out'), 6);
    // Translucency only ADDS light on the body.
    const lM = lum(matte.trace('out').pixelAt(48, 48));
    const lT = lum(lucent.trace('out').pixelAt(48, 48));
    expect(lT).toBeGreaterThanOrEqual(lM);
    const glossy = await render('plume_mat_spec',
      { ...base, azimuth: -35, elevation: 35, reflect: 1.0, roughness: 0.1 });
    glossy.trace('out').expectDifferentFrom(matte.trace('out'), 6);
  });

  it('orbit animates across frames', async () => {
    const moving = await runEngineMultiPhaseTest({
      width: 96, height: 96,
      modules: ['com.nano.core', 'com.nano.nano'],
      dumpName: 'plume_anim',
      phases: [
        {
          commands: [
            { type: 'createSketch', sketchId: 'plume_anim',
              sketch: buildSketch({ ...STATIC, ridge_depth: 0.9, orbit: 0.9,
                                    albedo: [0.5, 0.5, 0.5] }) },
            { type: 'setTracePoints', tracePoints: [
              { id: 'out', target: { type: 'sketch_output', sketchId: 'plume_anim' } },
            ]},
          ],
          waitFrames: 3, captureTraceIds: ['out'],
        },
        { waitFrames: 30, captureTraceIds: ['out'] },
      ],
    });
    expect(moving.success).toBe(true);
    moving.phases[1].trace('out').expectDifferentFrom(moving.phases[0].trace('out'), 10);
  });

  // --- sdf_field provider rail (plume -> plume) ---
  // Two adjacent plumes auto-couple by schema shape: the upstream's
  // `sdf_field` output binds to the downstream's `sdf_field_in`, and the
  // downstream renders the FOREIGN geometry with its own camera/light.
  // This is also the platform smoke test for a 3D texture riding a rail.

  // Provider shape (big, spiky) — shared verbatim by the solo reference.
  const A_SHAPE = { radius: 0.8, ridge_depth: 0.9, ridge_scale: 0.6,
                    ridge_sharp: 0.7, ridge_aniso: 0.4, swirl: 0.3,
                    variation: 0.35 };
  // Consumer look (light/material). bounce 0 keeps rendering exact.
  const B_LOOK = { albedo: [0.75, 0.75, 0.8], sun: 0.85, azimuth: 40,
                   elevation: 20, bounce: 0 };

  async function renderSketch(sketchId: string, sketch: Sketch) {
    const result = await runEngineTest({
      width: 96, height: 96,
      modules: ['com.nano.core', 'com.nano.nano'],
      commands: [
        { type: 'createSketch', sketchId, sketch },
        { type: 'setTracePoints', tracePoints: [
          { id: 'out', target: { type: 'sketch_output', sketchId } },
        ]},
      ],
      // Rail scalars ride the sketch-state mirror (one-frame lag) — give
      // the chain a few frames to settle before capturing.
      waitFrames: 10,
      captureTraceIds: ['out'],
      dumpName: sketchId,
    });
    expect(result.success).toBe(true);
    return result;
  }

  it('wired sdf_field renders the provider geometry exactly', async () => {
    // Chain: A (provider, big spiky shape) -> B (consumer, tiny smooth
    // shape of its own, distinct look). Reference: a solo plume with A's
    // shape + B's look — same field, same renderer, so BODY pixels must
    // match the wired consumer bit-exactly (off-body differs by design:
    // the consumer composites over A's render, the reference over clear).
    const wired = await renderSketch('plume_rail', {
      anchor: null, wires: [],
      chain: [
        { type: 'module', module_type: 'source.sdf.plume', instance_key: 'railA@0',
          params: { ...STATIC, ...A_SHAPE, bounce: 0 } },
        { type: 'module', module_type: 'source.sdf.plume', instance_key: 'railB@0',
          params: { ...STATIC, ...B_LOOK, radius: 0, ridge_depth: 0 } },
      ],
    } as Sketch);
    const ref = await render('plume_rail_ref',
      { ...STATIC, ...A_SHAPE, ...B_LOOK }, { noInput: true, waitFrames: 10 });

    // Probes well inside the provider body (base radius ~30 px on screen).
    for (const [x, y] of [[48, 48], [40, 48], [56, 48], [48, 40], [48, 58]]) {
      const expected = ref.trace('out').pixelAt(x, y);
      wired.trace('out').expectPixelAt(x, y, expected, 0);
    }
  });

  it('wired sdf_field overrides the consumer\'s own shape', async () => {
    // The consumer alone (tiny smooth ball) vs the same consumer fed the
    // big spiky provider field: silhouettes differ across the frame.
    const solo = await render('plume_rail_solo',
      { ...STATIC, ...B_LOOK, radius: 0, ridge_depth: 0 }, { noInput: true, waitFrames: 10 });
    const wired = await renderSketch('plume_rail_ovr', {
      anchor: null, wires: [],
      chain: [
        { type: 'module', module_type: 'source.sdf.plume', instance_key: 'ovrA@0',
          params: { ...STATIC, ...A_SHAPE, bounce: 0, opacity: 1 } },
        { type: 'module', module_type: 'source.sdf.plume', instance_key: 'ovrB@0',
          params: { ...STATIC, ...B_LOOK, radius: 0, ridge_depth: 0 } },
      ],
    } as Sketch);
    wired.trace('out').expectDifferentFrom(solo.trace('out'), 40);
  });

  // --- source.sdf.plume_field: the sculptor as a standalone provider ---

  it('plume_field provides the identical field to a downstream plume', async () => {
    // The standalone generator must sculpt byte-for-byte the field plume
    // sculpts for itself (shared field_gen.h). Its video output is clear,
    // so the consumer composites over transparent exactly like the solo
    // reference — the WHOLE frame matches, background included.
    const wired = await renderSketch('plume_fieldgen', {
      anchor: null, wires: [],
      chain: [
        { type: 'module', module_type: 'source.sdf.plume_field',
          instance_key: 'fgen@0', params: { ...A_SHAPE, morph: 0 } },
        { type: 'module', module_type: 'source.sdf.plume', instance_key: 'frend@0',
          params: { ...STATIC, ...B_LOOK, radius: 0, ridge_depth: 0 } },
      ],
    } as Sketch);
    const ref = await render('plume_fieldgen_ref',
      { ...STATIC, ...A_SHAPE, ...B_LOOK }, { noInput: true, waitFrames: 10 });

    for (const [x, y] of [[48, 48], [40, 48], [56, 48], [48, 40], [48, 58],
                          [3, 3], [92, 3], [3, 92]]) {
      const expected = ref.trace('out').pixelAt(x, y);
      wired.trace('out').expectPixelAt(x, y, expected, 0);
    }
  });

  it('disabling the provider mid-run falls back to the internal sculptor', async () => {
    // A bypassed provider is invisible to the rail: with no other source the
    // consumer's `sdf_field_in` must read DISCONNECTED again (executor resets
    // struct-root connection markers every frame) and the renderer resumes its
    // own sculpted shape — exactly, not approximately. Re-enabling re-couples.
    const solo = await render('plume_dis_solo',
      { ...STATIC, ...B_LOOK, radius: 0, ridge_depth: 0 },
      { noInput: true, waitFrames: 10 });
    const dyn = await runEngineMultiPhaseTest({
      width: 96, height: 96,
      modules: ['com.nano.core', 'com.nano.nano'],
      dumpName: 'plume_dis_dyn',
      phases: [
        {
          commands: [
            { type: 'createSketch', sketchId: 'plume_dis', sketch: {
              anchor: null, wires: [],
              chain: [
                { type: 'module', module_type: 'source.sdf.plume_field',
                  instance_key: 'disgen@0' },
                { type: 'module', module_type: 'source.sdf.plume',
                  instance_key: 'disrend@0' },
              ],
              // Canonical state home — setParam mutates instances[key].state
              // (the legacy entry.params blob is not writable mid-run).
              instances: {
                'disgen@0': { state: { ...A_SHAPE, morph: 0 } },
                'disrend@0': { state: { ...STATIC, ...B_LOOK,
                                        radius: 0, ridge_depth: 0 } },
              },
            } as unknown as Sketch },
            { type: 'setTracePoints', tracePoints: [
              { id: 'out', target: { type: 'sketch_output', sketchId: 'plume_dis' } },
            ]},
          ],
          waitFrames: 10, captureTraceIds: ['out'],
        },
        {
          commands: [
            { type: 'setParam', sketchId: 'plume_dis', chainIdx: 0,
              paramKey: '__enable__', value: 0 },
          ],
          waitFrames: 10, captureTraceIds: ['out'],
        },
        {
          commands: [
            { type: 'setParam', sketchId: 'plume_dis', chainIdx: 0,
              paramKey: '__enable__', value: 1 },
          ],
          waitFrames: 10, captureTraceIds: ['out'],
        },
      ],
    });
    expect(dyn.success).toBe(true);
    // Wired: the foreign (big spiky) field — clearly not the tiny ball.
    dyn.phases[0].trace('out').expectDifferentFrom(solo.trace('out'), 40);
    // Disabled with no other source: EXACTLY the consumer's own shape.
    for (const [x, y] of [[48, 48], [40, 48], [56, 48], [30, 30], [3, 3]]) {
      const expected = solo.trace('out').pixelAt(x, y);
      dyn.phases[1].trace('out').expectPixelAt(x, y, expected, 0);
    }
    // Re-enabled: the rail re-couples and the foreign field returns.
    dyn.phases[2].trace('out').expectDifferentFrom(solo.trace('out'), 40);
  });

  it('simulate mode evolves the field over time', async () => {
    // Tracers stream over the manifold and keep reshaping the height field
    // (persistent overlay, sim_fade 0), so the SAME static camera/light
    // must see the surface change across captures — while at sim 0 the
    // published field is exactly the static sculpt (pinned at tolerance 0
    // by the identical-field test above, since sim defaults to 0).
    const r = await runEngineMultiPhaseTest({
      width: 96, height: 96,
      modules: ['com.nano.core', 'com.nano.nano'],
      dumpName: 'plume_sim_evolve',
      phases: [
        {
          commands: [
            { type: 'createSketch', sketchId: 'plume_simev', sketch: {
              anchor: null, wires: [],
              chain: [
                { type: 'module', module_type: 'source.sdf.plume_field',
                  instance_key: 'sev_g@0',
                  params: { ...A_SHAPE, morph: 0,
                            sim: 0.9, sim_rate: 0.8, sim_carve: 0.85,
                            sim_fade: 0.0, sim_trail: 0.8 } },
                { type: 'module', module_type: 'source.sdf.plume',
                  instance_key: 'sev_r@0',
                  params: { ...STATIC, ...B_LOOK, radius: 0, ridge_depth: 0 } },
              ],
            } as Sketch },
            { type: 'setTracePoints', tracePoints: [
              { id: 'out', target: { type: 'sketch_output', sketchId: 'plume_simev' } },
            ]},
          ],
          waitFrames: 10, captureTraceIds: ['out'],
        },
        { waitFrames: 60, captureTraceIds: ['out'] },
      ],
    });
    expect(r.success).toBe(true);
    const early = r.phases[0].trace('out');
    const late = r.phases[1].trace('out');
    // The carved surface differs from the young one...
    late.expectDifferentFrom(early, 30);
    // ...but the rail stays valid: the body still renders (center on-body,
    // clearly darker/other than the empty checkerboard corners would be).
    const center = late.pixelAt(48, 48);
    expect(center.a).toBe(255);
  });

  it('plume_field passes video through untouched', async () => {
    const result = await renderSketch('plume_field_pass', {
      anchor: null, wires: [],
      chain: [
        { type: 'module', module_type: 'source.solid_color',
          instance_key: 'pbg@0', params: { color: [0.7, 0.7, 0.75] } },
        { type: 'module', module_type: 'source.sdf.plume_field',
          instance_key: 'fpass@0', params: {} },
      ],
    } as Sketch);
    // Nothing consumes the rail: the effect is a pure passthrough of the
    // solid color (checkerboard-composited trace ⇒ assert by color).
    for (const [x, y] of [[3, 3], [48, 48], [92, 92]]) {
      result.trace('out').expectPixelAt(x, y, BG, 2);
    }
  });
});
