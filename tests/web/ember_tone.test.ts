import { runGpuTest, runGpuEffectTest, Frame, forEachBackend } from '@nano/web/test/gpu-test-helpers';

/**
 * `filter.color.ember_tone` — the warm two-tone grade. Every input is a solid
 * colour, so each case reads one pixel: the grade is a pure function of the
 * pixel (plus the anchor hue, which for a solid image is the image's own hue).
 */

const W = 32, H = 32;

type RGB = { r: number; g: number; b: number };
const px = (f: Frame): RGB => f.pixelAt(W / 2, H / 2);

// HSV hue in degrees [0, 360) and saturation [0, 1], from 8-bit channels.
const hsv = ({ r, g, b }: RGB) => {
  const mx = Math.max(r, g, b), mn = Math.min(r, g, b), d = mx - mn;
  let h = 0;
  if (d > 0) {
    if (mx === r) h = ((g - b) / d) % 6;
    else if (mx === g) h = (b - r) / d + 2;
    else h = (r - g) / d + 4;
    h = (h * 60 + 360) % 360;
  }
  return { h, s: mx > 0 ? d / mx : 0, v: mx / 255 };
};
// Signed shortest hue difference a - b, degrees.
const hueDiff = (a: number, b: number) => ((a - b + 540) % 360) - 180;

forEachBackend((backend) => {
describe(`Ember Tone (${backend})`, () => {
  jest.setTimeout(60000);

  const run = (dumpName: string, rgb: [number, number, number], params: [string, unknown][] = []) =>
    runGpuEffectTest({
      module: 'filter.color.ember_tone',
      bundle: 'lights',
      width: W, height: H,
      inputColor: [...rgb, 1.0] as [number, number, number, number],
      params: params as any,
      dumpName,
    });
  const grey = (v: number): [number, number, number] => [v, v, v];

  it('declares metadata', async () => {
    const frame = await runGpuTest({
      module: 'filter.color.ember_tone',
      bundle: 'lights',
      dumpName: 'ember_tone_metadata',
    });
    expect(frame.success).toBe(true);
    expect(frame.metadata?.id).toBe('filter.color.ember_tone');
  });

  it('black stays black', async () => {
    const f = await run('ember_tone_black', grey(0));
    expect(f.success).toBe(true);
    const p = px(f);
    expect(Math.max(p.r, p.g, p.b)).toBeLessThanOrEqual(2);
  });

  it('a midtone lands on the strong hue: warm, saturated, bright', async () => {
    const f = await run('ember_tone_mid', grey(0.5));
    expect(f.success).toBe(true);
    const c = hsv(px(f));
    expect(c.h).toBeGreaterThan(10);
    expect(c.h).toBeLessThan(45);
    expect(c.s).toBeGreaterThan(0.6);
    expect(c.v).toBeGreaterThan(0.7);
  });

  it('the shadows take the weak tone, across the wheel', async () => {
    const f = await run('ember_tone_shadow', grey(0.08));
    expect(f.success).toBe(true);
    const p = px(f);
    const c = hsv(p);
    expect(p.b).toBeGreaterThan(p.r);
    expect(Math.abs(hueDiff(c.h, 25))).toBeGreaterThan(120);
    expect(c.s).toBeGreaterThan(0.5);
  });

  it('the crossing is a saturated colour, not grey', async () => {
    // Input lightness right at the threshold (OkLab L 0.3 ≈ sRGB 0.18).
    const f = await run('ember_tone_crossing', grey(0.18));
    expect(f.success).toBe(true);
    expect(hsv(px(f)).s).toBeGreaterThan(0.5);
  });

  it('white sizzles to white; Sizzle 0 keeps it on the body', async () => {
    const hot = await run('ember_tone_white', grey(1.0));
    const body = await run('ember_tone_white_nosizzle', grey(1.0), [['sizzle', 0]]);
    expect(hot.success).toBe(true);
    expect(body.success).toBe(true);
    const h = px(hot);
    expect(Math.min(h.r, h.g, h.b)).toBeGreaterThan(215);
    expect(hsv(px(body)).s).toBeGreaterThan(0.5);
  });

  it('Strong Hue sets the colour', async () => {
    const f = await run('ember_tone_green', grey(0.5), [['strong_hue', 0.33]]);
    expect(f.success).toBe(true);
    const c = hsv(px(f));
    expect(Math.abs(hueDiff(c.h, 120))).toBeLessThan(25);
  });

  it('Weak Offset turns the weak tone, and its sign picks the way round', async () => {
    const plus = await run('ember_tone_weak_plus', grey(0.08), [['weak_offset', 60]]);
    const minus = await run('ember_tone_weak_minus', grey(0.08), [['weak_offset', -60]]);
    expect(plus.success).toBe(true);
    expect(minus.success).toBe(true);
    expect(Math.abs(hueDiff(hsv(px(plus)).h, hsv(px(minus)).h))).toBeGreaterThan(60);
  });

  describe('source hue', () => {
    it('Auto: a solid image is its own anchor, so Hue Amount does nothing', async () => {
      const off = await run('ember_tone_auto_off', [0.2, 0.3, 0.8]);
      const on = await run('ember_tone_auto_on', [0.2, 0.3, 0.8], [['hue_amount', 1]]);
      const a = px(off), b = px(on);
      expect(Math.abs(hueDiff(hsv(a).h, hsv(b).h))).toBeLessThan(8);
    });

    const fixed: [string, unknown][] = [['anchor', 1], ['anchor_hue', 0]];
    it('Fixed: a hue away from the anchor turns the grade, both ways', async () => {
      const blue: [number, number, number] = [0.2, 0.3, 0.8];
      const base = hsv(px(await run('ember_tone_fixed_0', blue, fixed))).h;
      const pos = hsv(px(await run('ember_tone_fixed_pos', blue, [...fixed, ['hue_amount', 1]]))).h;
      const neg = hsv(px(await run('ember_tone_fixed_neg', blue, [...fixed, ['hue_amount', -1]]))).h;
      const dp = hueDiff(pos, base), dn = hueDiff(neg, base);
      expect(Math.abs(dp)).toBeGreaterThan(30);
      expect(Math.abs(dn)).toBeGreaterThan(30);
      expect(Math.sign(dp)).toBe(-Math.sign(dn));
    });

    it('Hue Curve: +1 swings a near colour much further than -1', async () => {
      // A mid-bright orange-red, a small way from the red anchor.
      const near: [number, number, number] = [0.8, 0.35, 0.2];
      const base = hsv(px(await run('ember_tone_curve_0', near, fixed))).h;
      const lift = hsv(px(await run('ember_tone_curve_pos', near,
        [...fixed, ['hue_amount', 1], ['hue_curve', 1]]))).h;
      const crush = hsv(px(await run('ember_tone_curve_neg', near,
        [...fixed, ['hue_amount', 1], ['hue_curve', -1]]))).h;
      expect(Math.abs(hueDiff(lift, base))).toBeGreaterThan(Math.abs(hueDiff(crush, base)) + 30);
      expect(Math.abs(hueDiff(crush, base))).toBeLessThan(8);
    });

    it('Source Saturation 1 leaves grey input grey', async () => {
      const f = await run('ember_tone_srcsat', grey(0.5), [['source_sat', 1]]);
      const p = px(f);
      expect(Math.max(p.r, p.g, p.b) - Math.min(p.r, p.g, p.b)).toBeLessThanOrEqual(3);
    });
  });

  it('Mix 0 passes the input through', async () => {
    const f = await run('ember_tone_dry', [0.2, 0.6, 0.4], [['mix', 0]]);
    const p = px(f);
    expect(Math.abs(p.r - 51)).toBeLessThanOrEqual(2);
    expect(Math.abs(p.g - 153)).toBeLessThanOrEqual(2);
    expect(Math.abs(p.b - 102)).toBeLessThanOrEqual(2);
  });
});
});
