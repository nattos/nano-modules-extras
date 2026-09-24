import { runEngineTest } from '@nano/web/test/engine-test-helpers';
import type { Sketch } from '@nano/web/src/sketch-types';

/**
 * E2E for filter.blur.smear — "Smear", a directional Pixulant (nano bundle).
 *
 * A flat solid is invisible through a blur/scatter, so the transform is exercised
 * by chaining a deterministic structured generator (source.grid, from core) → smear
 * and comparing parameter settings. Each run also traces the smear node's own INPUT
 * (chain_entry side:'input') so passthrough / dissolve can be checked against the
 * exact input in a single render. motion=0 pins the scatter salt for stable frames.
 */
describe('Smear (filter.blur.smear) E2E', () => {
  jest.setTimeout(60000);

  const W = 128, H = 128;
  const MODULES = ['com.nano.testonly', 'com.nano.core', 'com.nano.nano'];

  // source.grid → smear. Traces the smear node's input ('in') and the final
  // output ('out'). params override the schema defaults.
  const runChain = (id: string, params: Record<string, number>, dump: string,
                    waitFrames = 8) => {
    const sketch: Sketch = {
      anchor: null,
      wires: [],
      chain: [
        { type: 'module', module_type: 'source.grid', instance_key: 'grid@0', params: {} },
        { type: 'module', module_type: 'filter.blur.smear', instance_key: 'sm@0', params },
      ],
    };
    return runEngineTest({
      width: W, height: H, modules: MODULES,
      commands: [
        { type: 'createSketch', sketchId: id, sketch },
        { type: 'setTracePoints', tracePoints: [
          { id: 'in',  target: { type: 'chain_entry', sketchId: id, colIdx: 0, chainIdx: 1, side: 'input' } },
          { id: 'out', target: { type: 'sketch_output', sketchId: id } },
        ]},
      ],
      waitFrames, captureTraceIds: ['in', 'out'], dumpName: dump,
    });
  };

  it('registers and renders a blurred grid (differs from its input)', async () => {
    const r = await runChain('sm_r', { length: 0.6, width: 0.4, samples: 16 }, 'smear_render');
    expect(r.success).toBe(true);
    r.trace('out').expectNotSolidColor({ r: 0, g: 0, b: 0 }, 5);
    r.trace('out').expectDifferentFrom(r.trace('in'), 100);  // blur actually softened the grid
  });

  it('blur is directional — the streak axis follows angle', async () => {
    // Long thin streak so the axis dominates: horizontal (angle 0) vs vertical (0.5 → θ=π/2).
    const horiz = await runChain('sm_h', { angle: 0.0, length: 0.8, width: 0.03, tail: 0.0, samples: 16 }, 'smear_angle_h');
    const vert  = await runChain('sm_v', { angle: 0.5, length: 0.8, width: 0.03, tail: 0.0, samples: 16 }, 'smear_angle_v');
    expect(horiz.success).toBe(true);
    expect(vert.success).toBe(true);
    vert.trace('out').expectDifferentFrom(horiz.trace('out'), 100);
  });

  it('tail biases the kernel (symmetric blob vs one-sided streak)', async () => {
    const blob   = await runChain('sm_t0', { angle: 0.25, length: 0.8, width: 0.05, tail: 0.0, samples: 16 }, 'smear_tail_0');
    const streak = await runChain('sm_t1', { angle: 0.25, length: 0.8, width: 0.05, tail: 1.0, samples: 16 }, 'smear_tail_1');
    expect(blob.success).toBe(true);
    expect(streak.success).toBe(true);
    streak.trace('out').expectDifferentFrom(blob.trace('out'), 100);
  });

  it('perspective tilt ramps the minor width across the frame', async () => {
    // Wide minor axis so the per-pixel width scaling is visible; short major reach.
    const flat = await runChain('sm_k0', { angle: 0.0, length: 0.1, width: 0.6, tilt: 0.0, samples: 16 }, 'smear_tilt_0');
    const tilt = await runChain('sm_k1', { angle: 0.0, length: 0.1, width: 0.6, tilt: 1.0, samples: 16 }, 'smear_tilt_1');
    expect(flat.success).toBe(true);
    expect(tilt.success).toBe(true);
    tilt.trace('out').expectDifferentFrom(flat.trace('out'), 100);
  });

  it('master strength scales the smear (0 ≈ passthrough, 1 = full)', async () => {
    const off  = await runChain('sm_x0', { length: 0.8, width: 0.4, strength: 0.0, samples: 16 }, 'smear_strength_0');
    const full = await runChain('sm_x1', { length: 0.8, width: 0.4, strength: 1.0, samples: 16 }, 'smear_strength_1');
    expect(off.success).toBe(true);
    expect(full.success).toBe(true);
    // strength 0 collapses both reaches → passthrough of the input.
    expect(off.trace('out').diffCount(off.trace('in'), 6)).toBeLessThan(W * H * 0.02);
    // strength 1 actually smears (differs from input), and from the strength-0 case.
    full.trace('out').expectDifferentFrom(full.trace('in'), 100);
    full.trace('out').expectDifferentFrom(off.trace('out'), 100);
  });

  it('zero reach is a passthrough (≈ the input)', async () => {
    const r = await runChain('sm_p', { length: 0.0, width: 0.0, samples: 16 }, 'smear_passthrough');
    expect(r.success).toBe(true);
    // A zero-reach separable pass samples texel centres → output ≈ input.
    const diff = r.trace('out').diffCount(r.trace('in'), 6);
    expect(diff).toBeLessThan(W * H * 0.02);
  });

  it('scatter mode dissolves the grid into differenced grain', async () => {
    const r = await runChain('sm_s', { mode: 1, length: 0.5, width: 0.2, dive: 1.0, motion: 0.0 }, 'smear_scatter');
    expect(r.success).toBe(true);
    r.trace('out').expectNotSolidColor({ r: 0, g: 0, b: 0 }, 5);
    r.trace('out').expectDifferentFrom(r.trace('in'), 100);
  });

  it('scatter masks off-frame taps so edges do not bloom (the corner-flash fix)', async () => {
    // Same class as Pixulant: a directional throw lands the heavy tap off-frame,
    // where ClampToEdge replicates the edge/corner texel and its abs-difference
    // blooms. The fix fades the dive ONLY where a tap leaves the frame; edge_artifacts
    // =1 restores that off-frame weight (the old behaviour). So the mask must act
    // exactly at the frame edge and nowhere else. With a MODERATE throw and a VERTICAL
    // axis, the top band's taps go off the top (mask active) while the interior stays
    // on-frame (mask inactive → clean ≡ bloom). The bottom-only white injection can't
    // reach the top, so the top-vs-center split cleanly isolates the mask.
    const clean = await runChain('sm_mask_c', { mode: 1, angle: 0.5, length: 0.35, width: 0.1, dive: 1.0, motion: 0.0, edge_artifacts: 0.0 }, 'smear_mask_clean');
    const bloom = await runChain('sm_mask_b', { mode: 1, angle: 0.5, length: 0.35, width: 0.1, dive: 1.0, motion: 0.0, edge_artifacts: 1.0 }, 'smear_mask_bloom');
    expect(clean.success).toBe(true);
    expect(bloom.success).toBe(true);
    const a = clean.trace('out'), b = bloom.trace('out');
    let topDiff = 0, centerDiff = 0;
    a.forEachPixel((c, x, y) => {
      const o = b.pixelAt(x, y);
      const d = Math.abs(c.r - o.r) + Math.abs(c.g - o.g) + Math.abs(c.b - o.b);
      if (d > 30) {
        if (y < H * 0.12) topDiff++;                       // off-frame taps → mask active
        else if (y >= H * 0.4 && y < H * 0.6) centerDiff++; // on-frame taps → mask inert
      }
    });
    expect(topDiff).toBeGreaterThan(20);        // the mask reshapes the off-frame edge
    expect(centerDiff).toBeLessThan(topDiff);   // and leaves the on-frame interior alone
  });

  it('exposure scales the smeared output up (brighter)', async () => {
    // A blurred uniform gray stays that gray, so exposure scales it cleanly (no
    // clamping ambiguity): 0.15 → ~0.6 at exposure 4.
    const solid = (id: string, exposure: number, dump: string) => runEngineTest({
      width: W, height: H, modules: MODULES,
      commands: [
        { type: 'createSketch', sketchId: id, sketch: {
          anchor: null, wires: [],
          chain: [
            { type: 'module', module_type: 'source.solid_color', instance_key: 'sc@0', params: { color: [0.15, 0.15, 0.15] } },
            { type: 'module', module_type: 'filter.blur.smear', instance_key: 'sm@0', params: { length: 0.5, width: 0.3, exposure, samples: 16 } },
          ],
        }},
        { type: 'setTracePoints', tracePoints: [{ id: 'out', target: { type: 'sketch_output', sketchId: id } }] },
      ],
      waitFrames: 8, captureTraceIds: ['out'], dumpName: dump,
    });
    const dim    = await solid('sm_e1', 1.0, 'smear_exp_1');
    const bright = await solid('sm_e4', 4.0, 'smear_exp_4');
    expect(dim.success).toBe(true);
    expect(bright.success).toBe(true);
    dim.trace('out').expectUniformColor({ r: 38, g: 38, b: 38 }, 12);     // 0.15·255
    bright.trace('out').expectUniformColor({ r: 153, g: 153, b: 153 }, 16); // 0.6·255
  });

  it('softness reshapes the tail falloff', async () => {
    const boxy = await runChain('sm_sf0', { angle: 0.0, length: 0.7, width: 0.04, tail: 1.0, softness: 0.0, samples: 24 }, 'smear_soft_0');
    const soft = await runChain('sm_sf1', { angle: 0.0, length: 0.7, width: 0.04, tail: 1.0, softness: 1.0, samples: 24 }, 'smear_soft_1');
    expect(boxy.success).toBe(true);
    expect(soft.success).toBe(true);
    soft.trace('out').expectDifferentFrom(boxy.trace('out'), 100);
  });

  it('scatter softness reshapes the sampling distribution (smooths the hard edges)', async () => {
    const boxy = await runChain('sm_scb', { mode: 1, angle: 0.0, length: 0.6, width: 0.1, tail: 1.0, dive: 1.0, motion: 0.0, softness: 0.0 }, 'smear_scatter_boxy');
    const soft = await runChain('sm_scs', { mode: 1, angle: 0.0, length: 0.6, width: 0.1, tail: 1.0, dive: 1.0, motion: 0.0, softness: 1.0 }, 'smear_scatter_soft');
    expect(boxy.success).toBe(true);
    expect(soft.success).toBe(true);
    soft.trace('out').expectDifferentFrom(boxy.trace('out'), 100);
  });

  it('scatter motion animates the field over time', async () => {
    const early = await runChain('sm_m0', { mode: 1, length: 0.5, width: 0.2, dive: 1.0, motion: 1.0 }, 'smear_motion_early', 4);
    const late  = await runChain('sm_m1', { mode: 1, length: 0.5, width: 0.2, dive: 1.0, motion: 1.0 }, 'smear_motion_late', 60);
    expect(early.success).toBe(true);
    expect(late.success).toBe(true);
    late.trace('out').expectDifferentFrom(early.trace('out'), 50);
  });
});
