/**
 * Custom-inspector E2E for mod.shaper.spectral (resolume shell, playground mode).
 *
 * Verifies the registered inspector mounts: the reused spectral morph XY pad +
 * curve preview, plus the `input` slider. The preview parks its playhead at the
 * live input (the remap lookup position), like the envelope shaper.
 */
const BASE = process.env.GPU_TEST_BASE_URL || 'http://localhost:5173';
const WALK = `function* walk(root){for(const el of root.querySelectorAll('*')){yield el; if(el.shadowRoot) yield* walk(el.shadowRoot);}}`;

describe('mod.shaper.spectral custom inspector', () => {
  jest.setTimeout(60000);

  it('mounts the morph pad + curve preview + input slider', async () => {
    await page.goto(`${BASE}/resolume/index.html?playground`, { waitUntil: 'networkidle0' });
    await new Promise(r => setTimeout(r, 3500));   // boot + nano bundle discovery

    await page.evaluate(`(async () => {
      const ac = window.appController;
      ac.mutate('s', d => {
        d.sketches['sk_sp'] = {
          anchor: null,
          chain: [{ type: 'module', module_type: 'mod.shaper.spectral', instance_key: 'sp@0' }],
          wires: [],
          instances: { 'sp@0': { module_type: 'mod.shaper.spectral', state: { morph_x: 0.5, morph_y: 0.5, input: 0.3 } } },
        };
      });
      ac.setActiveTab('edit');
      ac.editSketch('sk_sp');
    })()`);
    await new Promise(r => setTimeout(r, 2500));   // mount + data fetch + rAF draw

    const info = await page.evaluate(`(() => {
      ${WALK}
      let insp = null, pad = null, preview = null, inputSlider = null;
      for (const el of walk(document)) {
        if (el.tagName === 'MOD-SPECTRAL-INSPECTOR') insp = el;
        if (el.tagName === 'SPECTRAL-LFO-XY-PAD') pad = el;
        if (el.tagName === 'SPECTRAL-LFO-PREVIEW') preview = el;
        if (el.tagName === 'SCALAR-SLIDER' && el.fieldPath === 'input') inputSlider = el;
      }
      let curvePixels = 0;
      if (preview) {
        const canvas = preview.shadowRoot.querySelector('canvas');
        if (canvas && canvas.width > 0) {
          const data = canvas.getContext('2d').getImageData(0, 0, canvas.width, canvas.height).data;
          for (let i = 0; i < data.length; i += 4) {
            // bg is (17,17,34); count clearly-different pixels (the curve strokes).
            if (Math.abs(data[i]-17) + Math.abs(data[i+1]-17) + Math.abs(data[i+2]-34) > 60) curvePixels++;
          }
        }
      }
      return { insp: !!insp, pad: !!pad, preview: !!preview, inputSlider: !!inputSlider, curvePixels };
    })()`) as any;

    expect(info.insp).toBe(true);          // the custom inspector mounted
    expect(info.pad).toBe(true);           // reused morph XY pad
    expect(info.preview).toBe(true);       // reused curve preview
    expect(info.inputSlider).toBe(true);   // the input slider (scrub / wire port)
    expect(info.curvePixels).toBeGreaterThan(50);   // the spectral curve is drawn
  });
});
