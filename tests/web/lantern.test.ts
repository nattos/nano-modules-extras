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
      // haze boiled: ~1.06 here, against ~0.34 now.
      const a = await runGpuEffectTest({
        module: 'source.mesh.lantern', bundle: 'nano', width: W, height: H,
        params: [...STATIC, ['flicker', 0]], ticks: 2,
      });
      const b = await runGpuEffectTest({
        module: 'source.mesh.lantern', bundle: 'nano', width: W, height: H,
        params: [...STATIC, ['flicker', 0]], ticks: 3,
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
