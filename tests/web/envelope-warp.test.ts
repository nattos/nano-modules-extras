import { runEngineTest } from '@nano/web/test/engine-test-helpers';
import type { Sketch } from '@nano/web/src/sketch-types';

/**
 * E2E coverage for warp.envelope (nano bundle) — warp along a hand-drawn
 * parametric envelope. The curve maps source position → warped position
 * (identity diagonal = no warp); segments rasterize 1D coordinate maps whose
 * fragment inverts the per-segment exponential ease analytically, then one
 * compute resolve samples the input through the composed maps.
 *
 * All expectations are derived from a BASELINE render of the same gradient
 * with no warp, so no assumption about source.gradient's exact ramp shape is
 * baked in. 96×96 square viewport: cover-square == viewport, and a pixel at
 * x shows uv (x+0.5)/96.
 *
 * Under test:
 *  1. Registers; identity curve == baseline (and is_identity via amount=0).
 *  2. Horizontal squeeze (y = 0.5x, mirrored): center fixed, interior pixels
 *     pull from 2x the center distance, Stretch bands smear the image edge.
 *  3. Transparent edges cut the uncovered bands instead.
 *  4. Ease is inverted exactly: a +1-eased segment lands where the analytic
 *     inverse (t = td^8) says.
 *  5. Fold-over: a non-monotonic curve draws LATER segments on top.
 *  6. Vertical / Rect (two curves, unmirrored) / XY / Radial modes move pixels
 *     where the geometry says.
 */

const W = 96, H = 96;

const IDENTITY = '[0,0,0,1,1,0]';
const SQUEEZE  = '[0,0,0,1,0.5,0]';           // y = 0.5x

function buildSketch(gradAngle: number, params: Record<string, unknown>): Sketch {
  return {
    anchor: null,
    chain: [
      {
        type: 'module',
        module_type: 'source.gradient',
        instance_key: 'grad@0',
        // Full-softness white→black ramp along `gradAngle` (0 = left→right,
        // 0.5 = top→bottom, 0.25 = 45° diagonal).
        params: { angle: gradAngle, offset: 0.0, softness: 1.0,
                  color_a: [1, 1, 1], color_b: [0, 0, 0] },
      },
      {
        type: 'module',
        module_type: 'warp.envelope',
        instance_key: 'ew@0',
        params,
      },
    ],
  };
}

async function render(sketchId: string, gradAngle: number,
                      params: Record<string, unknown>, dumpName: string) {
  const result = await runEngineTest({
    width: W, height: H,
    modules: ['com.nano.testonly', 'com.nano.core', 'com.nano.nano'],
    commands: [
      { type: 'createSketch', sketchId, sketch: buildSketch(gradAngle, params) },
      { type: 'setTracePoints', tracePoints: [
        { id: 'out', target: { type: 'sketch_output', sketchId } },
      ]},
    ],
    waitFrames: 6,
    captureTraceIds: ['out'],
    dumpName,
  });
  expect(result.success).toBe(true);
  return result;
}

// pixel x for a uv coordinate (x shows uv (x+0.5)/W).
const px = (u: number, extent = W) => Math.round(u * extent - 0.5);

function expectPixelNear(frame: any, x: number, y: number,
                         ref: any, rx: number, ry: number, tol = 8) {
  const want = ref.pixelAt(rx, ry);
  frame.expectPixelAt(x, y, { r: want.r, g: want.g, b: want.b }, tol);
}

describe('warp.envelope E2E', () => {
  jest.setTimeout(120000);

  it('registers; identity curve and amount=0 both pass through', async () => {
    const base = await render('ew_base', 0, { mode: 0, amount: 0.0, curve: SQUEEZE }, 'ew_base');
    expect(base.state.plugins.find((p: any) => p.id === 'warp.envelope')).toBeTruthy();

    const ident = await render('ew_ident', 0, { mode: 0, amount: 1.0, curve: IDENTITY }, 'ew_ident');
    ident.trace('out').expectSameAs(base.trace('out'), 3);
  });

  it('horizontal squeeze: center fixed, pixels pull outward, edges stretch', async () => {
    const base = await render('ew_h_base', 0, { mode: 0, amount: 0.0 }, 'ew_h_base');
    const sq   = await render('ew_h_sq', 0,
      { mode: 0, amount: 1.0, edges: 0, curve: SQUEEZE }, 'ew_h_sq');
    const b = base.trace('out'), s = sq.trace('out');

    // Mirror fixed point: the center column is unmoved.
    expectPixelNear(s, px(0.5), 48, b, px(0.5), 48);
    // dest u=0.625 (h'=0.25) shows source h=0.5 → u=0.75.
    expectPixelNear(s, px(0.625), 48, b, px(0.75), 48);
    // dest u=0.375 mirrors to source u=0.25.
    expectPixelNear(s, px(0.375), 48, b, px(0.25), 48);
    // Stretch bands smear the image's own edges: left band == left edge color.
    expectPixelNear(s, px(0.1), 48, b, 0, 48);
    expectPixelNear(s, px(0.9), 48, b, W - 1, 48);
    // And the squeeze actually moved things.
    s.expectDifferentFrom(b, 50);
  });

  it('transparent edges cut the uncovered bands', async () => {
    const stretch = await render('ew_e_st', 0,
      { mode: 0, amount: 1.0, edges: 0, curve: SQUEEZE }, 'ew_e_st');
    const transp  = await render('ew_e_tr', 0,
      { mode: 0, amount: 1.0, edges: 1, curve: SQUEEZE }, 'ew_e_tr');
    const st = stretch.trace('out'), tr = transp.trace('out');

    // The covered middle band is identical...
    expectPixelNear(tr, px(0.6), 48, st, px(0.6), 48, 3);
    // ...but the uncovered bands are not the stretched edge color anymore
    // (traces are checkerboard-composited, so probe color difference, not alpha).
    const stL = st.pixelAt(px(0.1), 48);
    const trL = tr.pixelAt(px(0.1), 48);
    const diff = Math.abs(stL.r - trL.r) + Math.abs(stL.g - trL.g) + Math.abs(stL.b - trL.b);
    expect(diff).toBeGreaterThan(30);
  });

  it('inverts the segment ease analytically', async () => {
    const base = await render('ew_ease_base', 0, { mode: 0, amount: 0.0 }, 'ew_ease_base');
    // One +1-eased segment: forward d = t^(1/8), inverse t = td^8. The dest
    // pixel at h'=0.5 (u=0.75) shows source h = 0.5^8 ≈ 0.004 — i.e. the
    // center of the image, not h=0.5.
    const eased = await render('ew_ease', 0,
      { mode: 0, amount: 1.0, curve: '[0,0,1,1,1,0]' }, 'ew_ease');
    const b = base.trace('out'), e = eased.trace('out');
    expectPixelNear(e, px(0.75), 48, b, px(0.502), 48);
  });

  it('fold-over: later segments draw on top', async () => {
    const base = await render('ew_fold_base', 0, { mode: 0, amount: 0.0 }, 'ew_fold_base');
    // Rise to 1 at x=0.5, fall back to 0.5 at x=1. Dest h'=0.9 is covered by
    // BOTH segments; the later (falling) one wins: source h=0.6, not h=0.45.
    const fold = await render('ew_fold', 0,
      { mode: 0, amount: 1.0, curve: '[0,0,0,0.5,1,0,1,0.5,0]' }, 'ew_fold');
    const b = base.trace('out'), f = fold.trace('out');
    const want = b.pixelAt(px(0.5 + 0.6 / 2), 48);   // u=0.80 (later segment)
    const not  = b.pixelAt(px(0.5 + 0.45 / 2), 48);  // u=0.725 (earlier segment)
    f.expectPixelAt(px(0.5 + 0.9 / 2), 48, { r: want.r, g: want.g, b: want.b }, 8);
    const got = f.pixelAt(px(0.5 + 0.9 / 2), 48);
    expect(Math.abs(got.r - not.r)).toBeGreaterThan(10);
  });

  it('vertical mode warps Y', async () => {
    const base = await render('ew_v_base', 0.5, { mode: 1, amount: 0.0 }, 'ew_v_base');
    const sq   = await render('ew_v_sq', 0.5,
      { mode: 1, amount: 1.0, edges: 0, curve: SQUEEZE }, 'ew_v_sq');
    const b = base.trace('out'), s = sq.trace('out');
    expectPixelNear(s, 48, px(0.5, H), b, 48, px(0.5, H));
    expectPixelNear(s, 48, px(0.625, H), b, 48, px(0.75, H));
  });

  it('rect mode: two independent curves, no mirroring', async () => {
    const base = await render('ew_r_base', 0.5, { mode: 4, amount: 0.0 }, 'ew_r_base');
    // X identity; Y compresses into the middle band [0.25, 0.75] edge-to-edge
    // (v' = 0.25 + 0.5v — NOT mirrored: v=0 lands at 0.25, not at the center).
    const rect = await render('ew_r', 0.5,
      { mode: 4, amount: 1.0, edges: 0, curve: IDENTITY, curve_y: '[0,0.25,0,1,0.75,0]' },
      'ew_r');
    const b = base.trace('out'), r = rect.trace('out');
    // dest v=0.375 shows source v=0.25.
    expectPixelNear(r, 48, px(0.375, H), b, 48, px(0.25, H));
    // top band stretches the TOP edge row (unmirrored).
    expectPixelNear(r, 48, px(0.1, H), b, 48, 0);
    // center row shows source v=0.5 (fixed point of this particular curve).
    expectPixelNear(r, 48, px(0.5, H), b, 48, px(0.5, H));
  });

  it('xy mode warps both axes with one curve', async () => {
    const base = await render('ew_xy_base', 0.25, { mode: 2, amount: 0.0 }, 'ew_xy_base');
    const xy   = await render('ew_xy', 0.25,
      { mode: 2, amount: 1.0, edges: 0, curve: SQUEEZE }, 'ew_xy');
    const b = base.trace('out'), x = xy.trace('out');
    // Both axes squeezed: dest (0.625, 0.625) shows source (0.75, 0.75).
    expectPixelNear(x, px(0.625), px(0.625, H), b, px(0.75), px(0.75, H));
    expectPixelNear(x, px(0.5), px(0.5, H), b, px(0.5), px(0.5, H));
  });

  it('radial mode warps by distance from the center', async () => {
    const base = await render('ew_rad_base', 0, { mode: 5, amount: 0.0 }, 'ew_rad_base');
    const rad  = await render('ew_rad', 0,
      { mode: 5, amount: 1.0, edges: 0, curve: SQUEEZE }, 'ew_rad');
    const b = base.trace('out'), r = rad.trace('out');
    // Center pixel is the radial fixed point.
    expectPixelNear(r, 48, 48, b, 48, 48, 10);
    // Pixel at x=60 (r≈0.26 of the 48px half-axis) pulls from 2× the radius:
    // source ≈ x=73 on the same row.
    expectPixelNear(r, 60, 48, b, 73, 48);
    r.expectDifferentFrom(b, 50);
  });
});
