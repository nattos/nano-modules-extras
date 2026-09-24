/**
 * Custom-inspector E2E for warp.envelope (resolume shell, playground mode).
 *
 * Verifies the registered inspector mounts: the reused <envelope-graph> curve
 * editor (drawing the curve), the Symmetry/Edges tab bars and Amount slider;
 * that switching Symmetry to "X and Y" shows a SECOND labeled graph and
 * "Radial" shows the Center pad; and that a double-click on the graph writes a
 * new node into the `curve` field.
 */
const BASE = process.env.GPU_TEST_BASE_URL || 'http://localhost:5173';
const WALK = `function* walk(root){for(const el of root.querySelectorAll('*')){yield el; if(el.shadowRoot) yield* walk(el.shadowRoot);}}`;

const PROBE = `(() => {
  ${WALK}
  let insp = null, graphs = 0, tabbars = 0, sliders = 0, vec = false, curvePixels = 0;
  for (const el of walk(document)) {
    if (el.tagName === 'ENVELOPE-WARP-INSPECTOR') insp = el;
    if (el.tagName === 'WARP-CURVE-FIELD') graphs++;
    if (el.tagName === 'FIELD-TAB-BAR') tabbars++;
    if (el.tagName === 'SCALAR-SLIDER') sliders++;
    if (el.tagName === 'FIELD-VEC') vec = true;
    if (el.tagName === 'ENVELOPE-GRAPH' && curvePixels === 0) {
      const canvas = el.shadowRoot.querySelector('canvas');
      if (canvas && canvas.width > 0) {
        const data = canvas.getContext('2d').getImageData(0, 0, canvas.width, canvas.height).data;
        for (let i = 0; i < data.length; i += 4) {
          if (data[i] + data[i + 1] + data[i + 2] > 90 && data[i + 3] > 40) curvePixels++;
        }
      }
    }
  }
  return { insp: !!insp, graphs, tabbars, sliders, vec, curvePixels };
})()`;

describe('warp.envelope custom inspector', () => {
  jest.setTimeout(60000);

  it('mounts the curve editor, follows the mode, and writes curve edits', async () => {
    await page.goto(`${BASE}/resolume/index.html?playground`, { waitUntil: 'networkidle0' });
    await new Promise(r => setTimeout(r, 3500));   // boot + nano bundle discovery

    await page.evaluate(`(async () => {
      const ac = window.appController;
      ac.mutate('s', d => {
        d.sketches['sk_ew'] = {
          anchor: null,
          chain: [{ type: 'module', module_type: 'warp.envelope', instance_key: 'ew@0' }],
          wires: [],
          instances: { 'ew@0': { module_type: 'warp.envelope', state: { mode: 0 } } },
        };
      });
      ac.setActiveTab('edit');
      ac.editSketch('sk_ew');
    })()`);
    await new Promise(r => setTimeout(r, 2500));   // mount + rAF draw

    // Single-curve mode: one graph, mode+edges tab bars, Amount, no Center.
    const h = await page.evaluate(PROBE) as any;
    expect(h.insp).toBe(true);
    expect(h.graphs).toBe(1);
    expect(h.tabbars).toBeGreaterThanOrEqual(2);   // Symmetry + Edges
    expect(h.sliders).toBeGreaterThanOrEqual(1);   // Amount
    expect(h.vec).toBe(false);
    expect(h.curvePixels).toBeGreaterThan(50);     // the identity diagonal is drawn

    // "X and Y" shows a second labeled graph.
    await page.evaluate(`window.appController.mutate('m', d => {
      d.sketches['sk_ew'].instances['ew@0'].state.mode = 3;
    })`);
    await new Promise(r => setTimeout(r, 600));
    const xy = await page.evaluate(PROBE) as any;
    expect(xy.graphs).toBe(2);
    expect(xy.vec).toBe(false);

    // "Radial" shows the Center pad and goes back to one graph.
    await page.evaluate(`window.appController.mutate('m', d => {
      d.sketches['sk_ew'].instances['ew@0'].state.mode = 5;
    })`);
    await new Promise(r => setTimeout(r, 600));
    const rad = await page.evaluate(PROBE) as any;
    expect(rad.graphs).toBe(1);
    expect(rad.vec).toBe(true);

    // Double-click the middle of the graph → a node is added → the `curve`
    // field now holds 3 points (9 numbers).
    const rect = await page.evaluate(`(() => {
      ${WALK}
      for (const el of walk(document)) {
        if (el.tagName === 'ENVELOPE-GRAPH') {
          const r = el.getBoundingClientRect();
          return { x: r.x + r.width / 2, y: r.y + r.height / 2 };
        }
      }
      return null;
    })()`) as any;
    expect(rect).not.toBeNull();
    await page.mouse.click(rect.x, rect.y, { clickCount: 2 });
    await new Promise(r => setTimeout(r, 400));
    const curve = await page.evaluate(`(() => {
      const st = window.appState.database.sketches['sk_ew'].instances['ew@0'].state;
      return st.curve || '';
    })()`) as string;
    const nums = String(curve).match(/-?\d+(\.\d+)?([eE]-?\d+)?/g) || [];
    expect(nums.length).toBe(9);
  });
});
