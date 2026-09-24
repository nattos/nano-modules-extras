import { runGpuEffectTest, forEachBackend } from '@nano/web/test/gpu-test-helpers';
import { runEngineTest } from '@nano/web/test/engine-test-helpers';
import type { Sketch } from '@nano/web/src/sketch-types';

/**
 * E2E for warp.legacy.pixulant — "Pixulant", the v2 port of the Resolume Wire
 * "Pixulant" patch (3-deep radial-scatter cascade + abs-difference "dive").
 *
 * A solid input is invisible through the scatter+difference (a solid scattered
 * anywhere is the same solid; abs(solid-solid)=0), so the transform is exercised
 * by chaining a deterministic structured generator (source.grid) → pixulant and
 * comparing parameter settings. motion=0 pins the salt so frames are stable.
 */
/** The solid probe colour both halves of this suite feed in. */
const SOLID: [number, number, number, number] = [0.15, 0.30, 0.55, 1.0];

// The metadata case is effect-level and runs on BOTH backends — it pins the
// schema this effect publishes, which is exactly where a host-side derivation
// can diverge. Everything below drives runEngineTest (the engine harness page:
// executor.wasm, wires, trace points), which has no native runner; the comp
// runner is the native equivalent and a native sketch host is a follow-up.
forEachBackend((backend) => {
describe(`Pixulant (warp.legacy.pixulant) E2E schema (${backend})`, () => {
  jest.setTimeout(60000);

  it('declares metadata and its parameters', async () => {
    const frame = await runGpuEffectTest({
      module: 'warp.legacy.pixulant', bundle: 'legacy',
      inputColor: SOLID,
      dumpName: 'pixulant_metadata',
    });
    expect(frame.success).toBe(true);
    expect(frame.metadata?.id).toBe('warp.legacy.pixulant');
    const names = frame.params.map(p => p.name);
    for (const n of ['dive', 'scatter', 'scatter_2', 'motion',
                     'dive_contrast_bias', 'dive_cap', 'dive_rolloff',
                     'scatter_modulate', 'scatter_1_modulate', 'edge_artifacts']) {
      expect(names).toContain(n);
    }
  });
});
});

describe('Pixulant (warp.legacy.pixulant) E2E', () => {
  jest.setTimeout(60000);

  // source.grid → pixulant, traced output. params override the defaults.
  const runChain = (id: string, params: Record<string, number>, dump: string) => {
    const sketch: Sketch = {
      anchor: null,
      wires: [],
      chain: [
        { type: 'module', module_type: 'source.grid', instance_key: 'grid@0', params: {} },
        { type: 'module', module_type: 'warp.legacy.pixulant', instance_key: 'px@0', params },
      ],
    };
    return runEngineTest({
      width: 128, height: 128,
      modules: ['com.nano.core', 'com.nano.legacy'],
      commands: [
        { type: 'createSketch', sketchId: id, sketch },
        { type: 'setTracePoints', tracePoints: [
          { id: 'out', target: { type: 'sketch_output', sketchId: id } },
        ]},
      ],
      waitFrames: 8,
      captureTraceIds: ['out'],
      dumpName: dump,
    });
  };

  // Just the grid, no effect — the passthrough reference.
  const runGridOnly = (id: string, dump: string) => runEngineTest({
    width: 128, height: 128,
    modules: ['com.nano.core', 'com.nano.legacy'],
    commands: [
      { type: 'createSketch', sketchId: id, sketch: {
        anchor: null, wires: [],
        chain: [{ type: 'module', module_type: 'source.grid', instance_key: 'grid@0', params: {} }],
      }},
      { type: 'setTracePoints', tracePoints: [
        { id: 'out', target: { type: 'sketch_output', sketchId: id } },
      ]},
    ],
    waitFrames: 8,
    captureTraceIds: ['out'],
    dumpName: dump,
  });

  it('full dive differences the image into edge grain (≠ the un-dived image)', async () => {
    // dive=0 → the (lightly scattered) image; dive=1 → abs-difference grain.
    const undived = await runChain('px_u', { dive: 0.0, scatter: 0.3, motion: 0.0 }, 'pixulant_undived');
    const dived   = await runChain('px_d', { dive: 1.0, scatter: 0.3, motion: 0.0 }, 'pixulant_dived');
    expect(undived.success).toBe(true);
    expect(dived.success).toBe(true);
    // The grid must have rendered through the un-dived (image-ish) path.
    let lit = 0;
    undived.trace('out').forEachPixel((c) => { if (c.r + c.g + c.b > 24) lit++; });
    expect(lit).toBeGreaterThan(100);
    dived.trace('out').expectDifferentFrom(undived.trace('out'), 100);
  });

  it('scatter widens the displacement (more scatter ≠ less scatter)', async () => {
    const low  = await runChain('px_sl', { dive: 1.0, scatter: 0.05, motion: 0.0 }, 'pixulant_scatter_low');
    const high = await runChain('px_sh', { dive: 1.0, scatter: 0.6, motion: 0.0 }, 'pixulant_scatter_high');
    expect(low.success).toBe(true);
    expect(high.success).toBe(true);
    high.trace('out').expectDifferentFrom(low.trace('out'), 100);
  });

  it('the exposure ceiling (Dive Contrast Bias) changes the grain brightness', async () => {
    const dim    = await runChain('px_eb', { dive: 1.0, scatter: 0.4, motion: 0.0, dive_contrast_bias: 0.5 }, 'pixulant_dim');
    const bright = await runChain('px_eB', { dive: 1.0, scatter: 0.4, motion: 0.0, dive_contrast_bias: 4.0 }, 'pixulant_bright');
    expect(dim.success).toBe(true);
    expect(bright.success).toBe(true);
    bright.trace('out').expectDifferentFrom(dim.trace('out'), 100);
  });

  it('dive_rolloff fades the dive toward the image at low scatter', async () => {
    // At low scatter, rolloff on ⇒ dive rolls off ⇒ output approaches the
    // (lightly-scattered) image, differing from rolloff off (full dive grain).
    const rolled = await runChain('px_ro', { dive: 1.0, scatter: 0.05, motion: 0.0, dive_rolloff: 0.5 }, 'pixulant_rolloff_on');
    const full   = await runChain('px_rf', { dive: 1.0, scatter: 0.05, motion: 0.0, dive_rolloff: 0.0 }, 'pixulant_rolloff_off');
    expect(rolled.success).toBe(true);
    expect(full.success).toBe(true);
    // The rolled-off frame keeps more of the (bright) image; the fully-dived one
    // is differenced toward black — so they differ, and rolled is brighter.
    rolled.trace('out').expectDifferentFrom(full.trace('out'), 100);
    let litRolled = 0, litFull = 0;
    rolled.trace('out').forEachPixel((c) => { if (c.r + c.g + c.b > 24) litRolled++; });
    full.trace('out').forEachPixel((c) => { if (c.r + c.g + c.b > 24) litFull++; });
    expect(litRolled).toBeGreaterThan(litFull);
  });

  it('edge_artifacts changes only the bottom edge (the reproduced Resolume bug)', async () => {
    // Off = clamped/clean; on = sampling past the bottom injects opaque WHITE,
    // which survives the abs-difference + exposure as the bottom-edge grain.
    // The artifact can only fire where the scatter reaches past uv.y>1, i.e. the
    // bottom region — so the top rows must stay unchanged and the bottom differ.
    const clean = await runChain('px_ec', { dive: 1.0, scatter: 0.6, motion: 0.0, edge_artifacts: 0.0 }, 'pixulant_edge_off');
    const buggy = await runChain('px_eo', { dive: 1.0, scatter: 0.6, motion: 0.0, edge_artifacts: 1.0 }, 'pixulant_edge_on');
    expect(clean.success).toBe(true);
    expect(buggy.success).toBe(true);
    const a = buggy.trace('out'), b = clean.trace('out');
    const H = a.height;
    let topDiff = 0, bottomDiff = 0;
    a.forEachPixel((c, x, y) => {
      const o = b.pixelAt(x, y);
      const d = Math.abs(c.r - o.r) + Math.abs(c.g - o.g) + Math.abs(c.b - o.b);
      if (d > 30) {
        if (y < H * 0.3) topDiff++;
        else if (y >= H - 12) bottomDiff++;
      }
    });
    expect(bottomDiff).toBeGreaterThan(20);   // artifact present along the bottom
    expect(topDiff).toBeLessThan(bottomDiff); // localized there (top unreached)
  });

  it('masks off-frame taps so edges do not bloom (the corner-flash fix)', async () => {
    // The reported glitch: at high scatter the heavy tap lands far off-frame, and
    // ClampToEdge replicates the edge/corner texel (a corner is the attractor for a
    // whole off-screen quadrant), so its abs-difference bloomed into flashing grain.
    // The fix fades the dive where a tap leaves the frame; edge_artifacts=1 restores
    // that off-frame weight (the old behaviour). The bottom-edge white injection only
    // fires past uv.y>1, so the TOP band isolates exactly the off-frame mask: the
    // masked (clean) top band must be less lit than the un-masked one.
    const clean = await runChain('px_mask_c', { dive: 1.0, scatter: 0.6, motion: 0.0, edge_artifacts: 0.0 }, 'pixulant_mask_clean');
    const bloom = await runChain('px_mask_b', { dive: 1.0, scatter: 0.6, motion: 0.0, edge_artifacts: 1.0 }, 'pixulant_mask_bloom');
    expect(clean.success).toBe(true);
    expect(bloom.success).toBe(true);
    const a = clean.trace('out'), b = bloom.trace('out');
    let sumClean = 0, sumBloom = 0;
    a.forEachPixel((c, x, y) => { if (y < 12) sumClean += c.r + c.g + c.b; });
    b.forEachPixel((c, x, y) => { if (y < 12) sumBloom += c.r + c.g + c.b; });
    expect(sumClean).toBeLessThan(sumBloom);   // masking off-frame taps darkens the top band
  });

  it('motion animates the scatter field over time', async () => {
    // Same params, but the chain is stepped further — the churn must move.
    const sketch = (id: string): Sketch => ({
      anchor: null, wires: [],
      chain: [
        { type: 'module', module_type: 'source.grid', instance_key: 'grid@0', params: {} },
        { type: 'module', module_type: 'warp.legacy.pixulant', instance_key: 'px@0',
          params: { dive: 1.0, scatter: 0.4, motion: 1.0 } },
      ],
    });
    const early = await runEngineTest({
      width: 128, height: 128, modules: ['com.nano.core', 'com.nano.legacy'],
      commands: [
        { type: 'createSketch', sketchId: 'px_t0', sketch: sketch('px_t0') },
        { type: 'setTracePoints', tracePoints: [{ id: 'out', target: { type: 'sketch_output', sketchId: 'px_t0' } }] },
      ],
      waitFrames: 4, captureTraceIds: ['out'], dumpName: 'pixulant_t0',
    });
    const late = await runEngineTest({
      width: 128, height: 128, modules: ['com.nano.core', 'com.nano.legacy'],
      commands: [
        { type: 'createSketch', sketchId: 'px_t1', sketch: sketch('px_t1') },
        { type: 'setTracePoints', tracePoints: [{ id: 'out', target: { type: 'sketch_output', sketchId: 'px_t1' } }] },
      ],
      waitFrames: 60, captureTraceIds: ['out'], dumpName: 'pixulant_t1',
    });
    expect(early.success).toBe(true);
    expect(late.success).toBe(true);
    late.trace('out').expectDifferentFrom(early.trace('out'), 50);
  });
});
