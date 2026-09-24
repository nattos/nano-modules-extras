import { runEngineTest } from '@nano/web/test/engine-test-helpers';
import type { Sketch } from '@nano/web/src/sketch-types';

/**
 * The MASK input on `source.mesh.three_planes` — a second image laid over the
 * stack in screen space, cutting the neon out of the picture the way a plane
 * at Fill -1 cuts.
 *
 * These need the ENGINE harness rather than the per-effect one, because the
 * whole subject is a second texture arriving over a real wire, and
 * runGpuEffectTest has no way to wire one. That makes them WebGPU only — the
 * Metal side of the same binding is pinned in
 * native/tests/test_effect_render.cpp ("a wired mask cuts the neon on Metal
 * too"), which is where the ladder of cut strengths lives. What is here is
 * what only a real sketch shows: WHERE the mask lands, and what counts as
 * mask at all.
 *
 * Input opacity is 1 throughout and the mask source doubles as the backdrop,
 * so every pixel stays opaque. Engine traces are checkerboard-composited, and
 * a mask that cuts to transparent would otherwise read as BRIGHT.
 */
describe('Three Planes mask E2E', () => {
  jest.setTimeout(60000);

  const W = 384, H = 216;
  const MODULES = ['com.nano.core', 'com.nano.lights'];

  const sketch = (maskModule: string, maskParams: Record<string, unknown>,
                  tp: Record<string, unknown>, wire: boolean): Sketch => ({
    anchor: null,
    chain: [
      { type: 'module', module_type: maskModule, instance_key: 'msk@0', params: maskParams },
      { type: 'module', module_type: 'source.mesh.three_planes', instance_key: 'tp@0',
        params: { grain: 0, scanline: 0, chroma_bleed: 0, input_opacity: 1,
                  plane1_emission: 1, plane2_emission: 1, plane3_emission: 1, ...tp } },
    ],
    wires: wire ? [{ id: 'w1', src: { instanceKey: 'msk@0', field: 'tex_out' },
                     dest: { instanceKey: 'tp@0', field: 'mask_in' } }] : [],
  } as Sketch);

  const run = (id: string, s: Sketch) => runEngineTest({
    width: W, height: H, modules: MODULES,
    commands: [{ type: 'createSketch', sketchId: id, sketch: s }],
    tracePoints: [{ id: 'out', target: { type: 'sketch_output', sketchId: id } }],
    captureTraceIds: ['out'], waitFrames: 20, dumpName: id,
  });

  /** Mean luma of a column band — the mask is a screen-space thing. */
  const band = (r: any, x0: number, x1: number) => {
    const f = r.trace('out');
    let s = 0, n = 0;
    for (let y = 0; y < H; y += 3)
      for (let x = x0; x < x1; x += 3) {
        const p = f.pixelAt(x, y);
        s += (p.r + p.g + p.b) / 3;
        n++;
      }
    return s / n;
  };

  it('cuts the neon, and lets some of the halo spill past the edge', async () => {
    const white = { color: [1.0, 1.0, 1.0] };
    const plain = await run('tpm_plain', sketch('source.solid_color', white, {}, false));
    const all   = await run('tpm_all',   sketch('source.solid_color', white, { mask_halo: 1 }, true));
    const mid   = await run('tpm_mid',   sketch('source.solid_color', white, { mask_halo: 0.6 }, true));
    const none  = await run('tpm_none',  sketch('source.solid_color', white, { mask_halo: 0 }, true));
    expect(plain.success && all.success && mid.success && none.success).toBe(true);

    const lum = (r: any) => band(r, 0, W);
    // A white input under a lit stack is a bright frame...
    expect(lum(plain)).toBeGreaterThan(200);
    // ...and masked at full weight, with the halo taken too, nothing is left.
    expect(lum(all)).toBeLessThan(4);
    // Let the halo through and the glow survives the cut. That spill is the
    // whole difference between a mask in the scene and one on the glass.
    expect(lum(none)).toBeGreaterThan(lum(mid) + 10);
    expect(lum(mid)).toBeGreaterThan(lum(all) + 10);
    // But the body and the base are gone whatever the halo does.
    expect(lum(none)).toBeLessThan(lum(plain) * 0.6);
  });

  it('present but black is not a mask, and Amount 0 is not one either', async () => {
    // The weight is ALPHA times LUMA and it wants both, which is what lets an
    // ordinary rendered logo work with no separate matte. A black shape is
    // fully present and masks nothing.
    const black = { color: [0.0, 0.0, 0.0] };
    const wired   = await run('tpm_black_on',  sketch('source.solid_color', black, {}, true));
    const unwired = await run('tpm_black_off', sketch('source.solid_color', black, {}, false));
    expect(wired.success && unwired.success).toBe(true);
    expect(band(wired, 0, W)).toBeCloseTo(band(unwired, 0, W), 0);

    // And a wired mask turned down does nothing at all.
    const white = { color: [1.0, 1.0, 1.0] };
    const off  = await run('tpm_off',  sketch('source.solid_color', white, { mask_strength: 0 }, true));
    const bare = await run('tpm_bare', sketch('source.solid_color', white, {}, false));
    expect(band(off, 0, W)).toBeCloseTo(band(bare, 0, W), 0);
  });

  it('lands in SCREEN space, where the mask input puts it', async () => {
    // A gradient is bright at one end and dark at the other, so it is its own
    // control: whichever side it masks hardest must be the side it is brightest
    // on. Nothing here assumes which side that is — only that the mask INVERTS
    // the relationship between the two, which no global dimming could do.
    const on  = await run('tpm_grad_on',  sketch('source.gradient', {}, { mask_halo: 1 }, true));
    const off = await run('tpm_grad_off', sketch('source.gradient', {}, {}, false));
    expect(on.success && off.success).toBe(true);

    const L = 0, LW = Math.floor(W / 3), RW = W - LW;
    const offL = band(off, L, LW), offR = band(off, RW, W);
    const onL  = band(on,  L, LW), onR  = band(on,  RW, W);

    // The gradient really does have two ends.
    expect(Math.abs(offL - offR)).toBeGreaterThan(60);
    // The bright end is the masked end, so the order of the two flips.
    expect(Math.sign(onL - onR)).toBe(-Math.sign(offL - offR));
    // ...and it is a cut, not a tilt: the bright end lost most of what it had.
    const brightBefore = Math.max(offL, offR);
    const brightAfter = offL > offR ? onL : onR;
    expect(brightAfter).toBeLessThan(brightBefore * 0.5);
  });
});

/**
 * The editor's own view of the same input. A module's texture INPUT is whichever
 * texture input its schema declares FIRST (schema-channels.ts, firstFieldOfType,
 * which sorts on declaration order) — so a second one declared ahead of `tex_in`
 * silently becomes THE input, and the aux port is the one that vanishes off the
 * card. That is not visible from any render: the picture still comes out right,
 * and the port is simply not there to wire.
 */
describe('Three Planes mask port', () => {
  jest.setTimeout(60000);
  const BASE = process.env.GPU_TEST_BASE_URL || 'http://localhost:5173';
  const WALK = `function* walk(root){for(const el of root.querySelectorAll('*')){yield el; if(el.shadowRoot) yield* walk(el.shadowRoot);}}`;

  it('shows Mask In, and leaves tex_in as the chain input', async () => {
    page.removeAllListeners('console');
    await page.goto(`${BASE}/resolume/index.html?playground`, { waitUntil: 'networkidle0' });
    await new Promise(r => setTimeout(r, 3000));
    await page.evaluate(`(async () => {
      const ac = window.appController;
      ac.mutate('s', d => {
        d.sketches['sk_mask_port'] = { anchor: null,
          chain: [{ type: 'module', module_type: 'source.mesh.three_planes', instance_key: 'tp@0' }],
          wires: [],
          instances: { 'tp@0': { module_type: 'source.mesh.three_planes', state: {} } } };
      });
      ac.setActiveTab('edit');
      ac.editSketch('sk_mask_port');
    })()`);
    await new Promise(r => setTimeout(r, 2500));

    const info: any = await page.evaluate(`(() => {
      const plugins = (window.appState?.local?.engine?.plugins)
                   || (window.appState?.local?.plugins) || [];
      const p = plugins.find(x => (x.id || x.module_type) === 'source.mesh.three_planes');
      const sch = p && (p.schema || p.fields);
      const tex = sch ? Object.entries(sch)
        .filter(([, d]) => d && d.type === 'texture')
        .map(([n, d]) => ({ name: n, io: d.io, order: d.order })) : [];
      ${WALK}
      const labels = [];
      for (const el of walk(document)) {
        const t = (el.textContent || '').trim();
        if (el.children.length === 0 && t.length < 30) labels.push(t);
      }
      return { tex, labels: [...new Set(labels)] };
    })()`);

    const ins = info.tex.filter((f: any) => (f.io & 1) !== 0)
                        .sort((a: any, b: any) => a.order - b.order);
    expect(ins.length).toBe(2);
    // tex_in first — the chain's image still belongs to the chain.
    expect(ins[0].name).toBe('tex_in');
    expect(ins[1].name).toBe('mask_in');
    // ...and the aux one is on the card, where it can be wired.
    expect(info.labels).toContain('Mask In');
  });
});
