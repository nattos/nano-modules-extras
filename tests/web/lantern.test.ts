import { runGpuEffectTest, forEachBackend, setBackend } from '@nano/web/test/gpu-test-helpers';

/**
 * E2E coverage for source.mesh.lantern (nano bundle) on both backends: the
 * HLSL reaches WebGPU through naga and Metal through SPIRV-Cross.
 *
 * Determinism: every rate defaults to stopped except Drift (fragments bob)
 * and Breathe Rate, whose amplitude (Breathe) defaults to 0 — so with Drift
 * 0 a frame depends only on the tick count. The default camera is the
 * poster's "intimate" shot; at 2:3 portrait the structure fills the middle
 * and the top-left corner is open water.
 */

const W = 120, H = 180;
const STATIC: [string, unknown][] = [['drift', 0]];
const lum = (c: { r: number, g: number, b: number }) => 0.2126 * c.r + 0.7152 * c.g + 0.0722 * c.b;

async function render(name: string, params: [string, unknown][] = []) {
  const out = await runGpuEffectTest({
    module: 'source.mesh.lantern',
    bundle: 'nano',
    width: W, height: H,
    params: [...STATIC, ...params],
    ticks: 2,
    dumpName: name,
  });
  expect(out.success).toBe(true);
  return out;
}

function meanAbsDiff(a: any, b: any): number {
  let sum = 0;
  a.forEachPixel((p: any, x: number, y: number) => {
    const q = b.pixelAt(x, y);
    sum += Math.abs(p.r - q.r) + Math.abs(p.g - q.g) + Math.abs(p.b - q.b);
  });
  return sum / (W * H * 3);
}

forEachBackend((backend) => {
  describe(`source.mesh.lantern (${backend})`, () => {
    jest.setTimeout(120000);

    it('renders the poster: deep blue water, a glowing core between dark blocks', async () => {
      const f = await render(`lantern_${backend}_poster`);
      // Open water in the corner: dark, blue.
      const corner = f.pixelAt(4, 4);
      expect(corner.b).toBeGreaterThan(corner.r + 15);
      expect(lum(corner)).toBeLessThan(110);
      // The light escaping between the blocks is the brightest thing in frame.
      let peak = 0;
      f.forEachPixel((c: any) => { peak = Math.max(peak, lum(c)); });
      expect(peak).toBeGreaterThan(200);
      // ...and dark blocks stand against it: a real spread of values.
      let dark = 0;
      f.forEachPixel((c: any) => { if (lum(c) < 50) dark++; });
      expect(dark).toBeGreaterThan(W * H * 0.03);
    });

    it('breathing, wandering and spinning move the picture', async () => {
      const still = await render(`lantern_${backend}_still`);
      const moved = await render(`lantern_${backend}_moved`, [
        ['breathe', 1.0], ['breathe_rate', 1.0], ['wander', 1.0], ['wander_rate', 1.0],
        ['spin', 1.0],
      ]);
      expect(meanAbsDiff(still, moved)).toBeGreaterThan(1.0);
    });

    it('grows other layouts from a seed', async () => {
      const ref = await render(`lantern_${backend}_layout0`);
      const grown = await render(`lantern_${backend}_layout7`, [['layout', 7]]);
      expect(meanAbsDiff(ref, grown)).toBeGreaterThan(2.0);
      let peak = 0;
      grown.forEachPixel((c: any) => { peak = Math.max(peak, lum(c)); });
      expect(peak).toBeGreaterThan(150);
    });

    it('the beams hold still: a static scene barely changes frame to frame', async () => {
      // The fog's core visibility is prefiltered (vis.hlsl), so only the
      // march's step dither varies per frame. With random shadow lookups the
      // haze boiled: ~1.06 here, against ~0.34 now. (Without the flakes,
      // which are meant to move.)
      const a = await runGpuEffectTest({
        module: 'source.mesh.lantern', bundle: 'nano', width: W, height: H,
        params: [...STATIC, ['flicker', 0], ['flakes', 0]], ticks: 2,
      });
      const b = await runGpuEffectTest({
        module: 'source.mesh.lantern', bundle: 'nano', width: W, height: H,
        params: [...STATIC, ['flicker', 0], ['flakes', 0]], ticks: 3,
      });
      expect(a.success && b.success).toBe(true);
      expect(meanAbsDiff(a, b)).toBeLessThan(0.6);
    });

    it('the air wall clears thick water in front of the structure', async () => {
      // At Water 0.25 the structure is buried (mean ~21); Air clears the
      // near field and the glow comes back (mean ~85).
      const meanLum = (f: any) => {
        let sum = 0;
        f.forEachPixel((c: any) => { sum += lum(c); });
        return sum / (W * H);
      };
      const buried = await render(`lantern_${backend}_water_deep`, [['water', 0.25]]);
      const cleared = await render(`lantern_${backend}_water_air`, [['water', 0.25], ['air', 1]]);
      expect(meanLum(buried)).toBeLessThan(45);
      expect(meanLum(cleared)).toBeGreaterThan(60);
    });

    it('carves worn, chipped edges into the blocks', async () => {
      // Plain boxes vs the carved default: ~1.8 apart, against ~0.3 of
      // frame-to-frame dither.
      const plain = await render(`lantern_${backend}_plain`, [['chips', 0], ['bevel', 0]]);
      const carved = await render(`lantern_${backend}_carved`);
      expect(meanAbsDiff(plain, carved)).toBeGreaterThan(1.0);
    });

    it('Albedo 0 turns the blocks to silhouettes against the lit water', async () => {
      // Mean luminance ~66 by default, ~59 with nothing reflected (the
      // water's glow is most of the frame).
      const meanLum = (f: any) => {
        let sum = 0;
        f.forEachPixel((c: any) => { sum += lum(c); });
        return sum / (W * H);
      };
      const lit = await render(`lantern_${backend}_albedo1`);
      const dark = await render(`lantern_${backend}_albedo0`, [['albedo', 0]]);
      expect(meanLum(lit) - meanLum(dark)).toBeGreaterThan(3);
    });

    it('draws the flakes: specks over black in the Flakes view, and they fly', async () => {
      // The Flakes debug view is the spark layer alone. Seeded on the ring,
      // a few hundred flakes land as specks; a few frames later they have
      // moved.
      const flakes = async (ticks: number, name: string) => {
        const out = await runGpuEffectTest({
          module: 'source.mesh.lantern', bundle: 'nano', width: 320, height: 180,
          params: [...STATIC, ['debug_view', 5], ['flake_size', 3], ['sparkle', 4]],
          ticks, renderEachTick: true, dumpName: name,
        });
        expect(out.success).toBe(true);
        expect(out.gpuErrors).toEqual([]);
        return out;
      };
      const a = await flakes(2, `lantern_${backend}_flakes_a`);
      let lit = 0;
      a.forEachPixel((c: any) => { if (lum(c) > 12) lit++; });
      expect(lit).toBeGreaterThan(40);
      expect(lit).toBeLessThan(320 * 180 * 0.2);
      const b = await flakes(20, `lantern_${backend}_flakes_b`);
      let moved = 0;
      a.forEachPixel((c: any, x: number, y: number) => {
        if (Math.abs(lum(c) - lum(b.pixelAt(x, y))) > 12) moved++;
      });
      expect(moved).toBeGreaterThan(20);
    });

    it('opacity 0 passes the input through', async () => {
      const out = await runGpuEffectTest({
        module: 'source.mesh.lantern', bundle: 'nano', width: 32, height: 32,
        inputColor: [0.2, 0.6, 0.4, 1], params: [['opacity', 0]], ticks: 1,
      });
      expect(out.success).toBe(true);
      out.expectUniformColor({ r: 51, g: 153, b: 102, a: 255 }, 2);
    });
  });
});

describe('source.mesh.lantern backend parity', () => {
  jest.setTimeout(120000);
  afterAll(() => setBackend('puppeteer'));

  it('both draw the flakes where the blocks are', async () => {
    // The flakes go through a raster pass, whose clip space has its own y
    // convention (spark_vs.hlsl) — got wrong, they draw mirrored about the
    // frame's middle, away from the blocks they follow. The Anchors view pins
    // each flake to a block's center, which always projects inside that
    // block: nearly every dot must land on geometry (Surfaces view: lit),
    // not on the black water.
    const frame = async (view: number, flakes: number) => {
      const out = await runGpuEffectTest({
        module: 'source.mesh.lantern', bundle: 'nano', width: 320, height: 180,
        params: [...STATIC, ['debug_view', view], ['flakes', flakes], ['flake_size', 2],
                 ['distance', 45], ['albedo', 2]],
        ticks: 1, renderEachTick: true,
      });
      expect(out.success).toBe(true);
      return out;
    };
    for (const backend of ['puppeteer', 'metal'] as const) {
      setBackend(backend);
      const dots = await frame(6, 512);
      const surf = await frame(2, 0);
      let on = 0, off = 0;
      dots.forEachPixel((c: any, x: number, y: number) => {
        if (lum(c) < 60) return;
        // Within a pixel of a surface counts (a dot's rim overhangs).
        let hit = false;
        for (let dy = -1; dy <= 1 && !hit; dy++)
          for (let dx = -1; dx <= 1 && !hit; dx++) {
            const q = surf.pixelAt(Math.min(319, Math.max(0, x + dx)), Math.min(179, Math.max(0, y + dy)));
            if (lum(q) > 4) hit = true;
          }
        if (hit) on++; else off++;
      });
      expect(on).toBeGreaterThan(50);
      expect(off / (on + off)).toBeLessThan(0.05);
    }
  });

  it('WebGPU and Metal agree', async () => {
    setBackend('puppeteer');
    const web = await render('lantern_parity_web');
    setBackend('metal');
    const metal = await render('lantern_parity_metal');
    // The fog march dithers per pixel and the two compilers round
    // differently; the pictures agree to a few levels on average.
    expect(meanAbsDiff(web, metal)).toBeLessThan(6);
  });
});

// The aux outputs on WebGPU, in the editor: Aux 1 (Depth) wired out through a
// sidechannel and received back as the sketch's picture, so the monitor shows
// it -- gray blocks on black. (The Metal side, and that an unwired aux is
// never made, are pinned in the native render tests.)
describe('source.mesh.lantern aux outputs (editor, WebGPU)', () => {
  jest.setTimeout(120000);
  const BASE = process.env.GPU_TEST_BASE_URL || 'http://localhost:5173';
  const WALK = `function* walk(root){for(const el of root.querySelectorAll('*')){yield el; if(el.shadowRoot) yield* walk(el.shadowRoot);}}`;

  it('Aux 1 Depth reaches a wire: gray blocks, black water', async () => {
    await page.goto(`${BASE}/resolume/index.html?playground`, { waitUntil: 'networkidle0' });
    await new Promise(r => setTimeout(r, 2000));
    await page.evaluate(`(() => {
      const ac = window.appController;
      ac.mutate('seed', d => {
        d.sketches['sk_aux'] = {
          anchor: null,
          chain: [
            { type: 'module', module_type: 'source.mesh.lantern', instance_key: 'ln@0' },
            { type: 'module', module_type: 'util.sidechannel_out', instance_key: 'so@0' },
            { type: 'module', module_type: 'util.sidechannel_in', instance_key: 'si@0' },
          ],
          wires: [{ id: 'aw', src: { instanceKey: 'ln@0', field: 'aux1_out' },
                    dest: { instanceKey: 'so@0', field: 'send_in' } }],
          instances: {
            'ln@0': { module_type: 'source.mesh.lantern', state: { drift: 0, flakes: 0, aux1: 1 } },
            'so@0': { module_type: 'util.sidechannel_out', state: { channel: 9 } },
            'si@0': { module_type: 'util.sidechannel_in', state: { channel: 9 } },
          },
        };
      });
      ac.setActiveTab('edit');
      ac.editSketch('sk_aux');
      ac.setTappingMode(false);
    })()`);
    // The monitor: the largest texture monitor on screen. Its canvas is
    // WebGPU, so read it from a screenshot decoded in the page.
    const sample = async () => {
      const box: any = await page.evaluate(`(() => { ${WALK} let best = null;
        for (const el of walk(document)) if (el.tagName === 'TEXTURE-MONITOR') {
          const b = el.getBoundingClientRect();
          if (b.top >= 0 && b.bottom <= innerHeight && (!best || b.width * b.height > best.w * best.h))
            best = { x: b.x, y: b.y, w: b.width, h: b.height };
        } return best; })()`);
      if (!box) return null;
      const png = await page.screenshot({ clip: { x: box.x, y: box.y, width: box.w, height: box.h }, encoding: 'base64' });
      return page.evaluate(async (b64: string) => {
        const img = new Image();
        img.src = 'data:image/png;base64,' + b64;
        await img.decode();
        const c = document.createElement('canvas');
        c.width = img.width; c.height = img.height;
        const ctx = c.getContext('2d')!;
        ctx.drawImage(img, 0, 0);
        const d = ctx.getImageData(0, 0, c.width, c.height).data;
        let lit = 0, gray = 0, blue = 0;
        for (let i = 0; i < d.length; i += 4) {
          if (d[i + 2] > d[i] + 25) blue++;
          if (d[i] > 40) {
            lit++;
            if (Math.abs(d[i] - d[i + 1]) < 6 && Math.abs(d[i] - d[i + 2]) < 6) gray++;
          }
        }
        return { lit, gray, blue, n: d.length / 4 };
      }, png as string);
    };
    let s: any = null;
    for (let i = 0; i < 40; i++) {
      await new Promise(r => setTimeout(r, 250));
      s = await sample();
      if (s && s.lit > 0.03 * s.n) break;
    }
    expect(s).toBeTruthy();
    expect(s.lit).toBeGreaterThan(0.03 * s.n);    // the blocks...
    expect(s.gray).toBeGreaterThan(0.95 * s.lit); // ...in gray
    expect(s.blue).toBeLessThan(0.01 * s.n);      // no water: not the main picture
  });
});
