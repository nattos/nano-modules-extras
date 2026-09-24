import { runEngineTest } from '@nano/web/test/engine-test-helpers';
import { runGpuEffectTest, forEachBackend } from '@nano/web/test/gpu-test-helpers';
import type { Sketch } from '@nano/web/src/sketch-types';

/**
 * E2E for filter.reconstruct.line — "Line Reconstruct" (nano bundle).
 *
 * An SMAA-like morphological reconstructor: classify each pixel (line / point /
 * step-edge / junction / smooth-gradient) then re-render crisp uniform-width
 * strokes + de-band. Multi-pass compute; TimeIndependent; strength 0 = identity.
 *
 * A flat solid has no structure to reconstruct, so the transform is exercised by
 * chaining a deterministic structured generator (source.grid, from core) → line
 * reconstruct and comparing param settings. We trace the effect's own INPUT
 * ('in', chain_entry side:'input') and the final output ('out') in one render so
 * passthrough / identity can be checked against the exact input.
 */
describe('Line Reconstruct (filter.reconstruct.line) E2E', () => {
  jest.setTimeout(60000);

  const W = 128, H = 128;
  const MODULES = ['com.nano.testonly', 'com.nano.core', 'com.nano.nano'];
  const LR = 'filter.reconstruct.line';

  // source.grid → line_reconstruct. Traces the effect input ('in') and output.
  const runChain = (id: string, params: Record<string, number>, dump: string,
                    gridParams: Record<string, number> = {}, waitFrames = 8) => {
    const sketch: Sketch = {
      anchor: null,
      wires: [],
      chain: [
        { type: 'module', module_type: 'source.grid', instance_key: 'grid@0', params: gridParams },
        { type: 'module', module_type: LR, instance_key: 'lr@0', params },
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

  it('registers with the right id and declares the param surface', async () => {
    const r = await runChain('lr_reg', { strength: 1.0 }, 'line_reconstruct_reg');
    expect(r.success).toBe(true);
    const lr = r.state.plugins.find((p: any) => p.id === LR);
    expect(lr).toBeTruthy();
    // I/O texture rails present (kind 0 = input, kind 1 = output).
    expect(lr.io.find((io: any) => io.name === 'tex_in')).toBeTruthy();
    expect(lr.io.find((io: any) => io.name === 'tex_out')).toBeTruthy();
    // The full authored param set is declared.
    const names = lr.params.map((p: any) => p.name).sort();
    expect(names).toEqual(['deband', 'debug_view', 'max_width', 'point_radius',
                           'recover', 'retarget', 'sensitivity', 'solidify',
                           'strength', 'target_width']);
  });

  it('renders a non-solid frame (the multi-pass pipeline dispatches cleanly)', async () => {
    const r = await runChain('lr_render', { strength: 1.0 }, 'line_reconstruct_render');
    expect(r.success).toBe(true);
    r.trace('out').expectNotSolidColor({ r: 0, g: 0, b: 0 }, 5);
  });

  // The line branch re-renders soft/degraded strokes into crisp uniform lines
  // (the effect's actual use case — clean sharp lines are already optimal and
  // correctly pass through, and dense parallel lines fall back per the flank
  // gate). So exercise it on a DEGRADED grid: source.grid → a soft blur → the
  // reconstructor, traced against its own (blurred) input.
  const runDegraded = (id: string, params: Record<string, number>, dump: string) => {
    const sketch: Sketch = {
      anchor: null, wires: [],
      chain: [
        { type: 'module', module_type: 'source.grid', instance_key: 'grid@0', params: { cell_size: 0.22, line_width: 0.12 } },
        { type: 'module', module_type: 'filter.blur.gaussian', instance_key: 'blur@0', params: { radius: 0.09 } },
        { type: 'module', module_type: LR, instance_key: 'lr@0', params },
      ],
    };
    return runEngineTest({
      width: 160, height: 160, modules: MODULES,
      commands: [
        { type: 'createSketch', sketchId: id, sketch },
        { type: 'setTracePoints', tracePoints: [
          { id: 'in',  target: { type: 'chain_entry', sketchId: id, colIdx: 0, chainIdx: 2, side: 'input' } },
          { id: 'out', target: { type: 'sketch_output', sketchId: id } },
        ]},
      ],
      waitFrames: 8, captureTraceIds: ['in', 'out'], dumpName: dump,
    });
  };

  it('reconstructs degraded lines: strength>0 crisps them, strength 0 = identity', async () => {
    const on  = await runDegraded('lr_recon_on',  { strength: 1.0, retarget: 1.0, target_width: 0.25, sensitivity: 0.7 }, 'line_reconstruct_recon_on');
    const off = await runDegraded('lr_recon_off', { strength: 0.0 }, 'line_reconstruct_recon_off');
    expect(on.success && off.success).toBe(true);
    on.trace('out').expectNotSolidColor({ r: 0, g: 0, b: 0 }, 5);
    on.trace('out').expectDifferentFrom(on.trace('in'), 100);   // it actually reshaped the soft lines
    off.trace('out').expectSameAs(off.trace('in'), 2);          // strength 0 is untouched
  });

  it('strength 0 is a pass-through (output == input)', async () => {
    const r = await runChain('lr_identity', { strength: 0.0 }, 'line_reconstruct_identity');
    expect(r.success).toBe(true);
    r.trace('out').expectSameAs(r.trace('in'), 2);
  });

  it('deband reshapes a smooth gradient (the deband branch engages)', async () => {
    // A near-flat gradient is the deband branch's target: fine contrast is small
    // but wide-scale drift is present, so deband's clamp-bounded correction +
    // dither engages. deband 1 must differ from deband 0 on the same gradient.
    const grad = (id: string, deband: number, dump: string) => {
      const sketch: Sketch = {
        anchor: null, wires: [],
        chain: [
          { type: 'module', module_type: 'source.gradient', instance_key: 'grad@0', params: { softness: 1.0 } },
          { type: 'module', module_type: LR, instance_key: 'lr@0', params: { strength: 1.0, deband } },
        ],
      };
      return runEngineTest({
        width: 96, height: 96, modules: MODULES,
        commands: [
          { type: 'createSketch', sketchId: id, sketch },
          { type: 'setTracePoints', tracePoints: [{ id: 'out', target: { type: 'sketch_output', sketchId: id } }] },
        ],
        waitFrames: 8, captureTraceIds: ['out'], dumpName: dump,
      });
    };
    const off = await grad('lr_db_off', 0.0, 'line_reconstruct_deband_off');
    const on  = await grad('lr_db_on',  1.0, 'line_reconstruct_deband_on');
    expect(off.success && on.success).toBe(true);
    on.trace('out').expectDifferentFrom(off.trace('out'), 20);
  });

  // Detection sanity: on a structured grid the classifier fires — the Class /
  // Width / Orientation debug views are non-solid and differ from passthrough
  // and from each other. (Validates stats→pyramid→tensor→features end-to-end.)
  it('debug views expose a non-trivial classification of the grid', async () => {
    const grid = { cell_size: 0.2, line_width: 0.15 };
    const off  = await runChain('lr_dbg_off',  { strength: 1.0, debug_view: 0 }, 'line_reconstruct_dbg_off', grid);
    const cls  = await runChain('lr_dbg_cls',  { strength: 1.0, debug_view: 1 }, 'line_reconstruct_dbg_class', grid);
    const wid  = await runChain('lr_dbg_wid',  { strength: 1.0, debug_view: 2 }, 'line_reconstruct_dbg_width', grid);
    const ori  = await runChain('lr_dbg_ori',  { strength: 1.0, debug_view: 3 }, 'line_reconstruct_dbg_orient', grid);
    const ctr  = await runChain('lr_dbg_ctr',  { strength: 1.0, debug_view: 4 }, 'line_reconstruct_dbg_centerline', grid);
    const coh  = await runChain('lr_dbg_coh',  { strength: 1.0, debug_view: 5 }, 'line_reconstruct_dbg_coherence', grid);
    for (const r of [off, cls, wid, ori, ctr, coh]) expect(r.success).toBe(true);
    // The classifier lit up (not a black frame) and reads differently from input.
    cls.trace('out').expectNotSolidColor({ r: 0, g: 0, b: 0 }, 5);
    cls.trace('out').expectDifferentFrom(off.trace('out'), 100);
    wid.trace('out').expectDifferentFrom(cls.trace('out'), 100);
    ori.trace('out').expectDifferentFrom(cls.trace('out'), 100);
    // The smooth + centerline passes produce a non-trivial coherence / centerline.
    coh.trace('out').expectNotSolidColor({ r: 0, g: 0, b: 0 }, 5);
    ctr.trace('out').expectDifferentFrom(coh.trace('out'), 100);
  });
});

// Cross-backend parity: the whole multi-pass pipeline (≈20 dispatches, RGBA16F
// intermediates, samplers, a big neighbour-gather centerline) must register and
// dispatch on WebGPU AND native Metal (via the AOT sidecar — regenerate with
// `native/wasm_modules/build_aot.sh nano` after any bundle rebuild). Single
// effect on a solid input — enough to exercise every pass without depending on a
// cross-bundle generator.
forEachBackend((backend) => describe(`Line Reconstruct backend parity (${backend})`, () => {
  jest.setTimeout(60000);
  it('registers and dispatches every pass', async () => {
    const f = await runGpuEffectTest({
      module: 'filter.reconstruct.line', bundle: 'nano',
      inputColor: [0.6, 0.3, 0.2, 1.0],
      params: [['strength', 1.0], ['deband', 0.6], ['solidify', 0.6]],
      dumpName: `line_reconstruct_parity_${backend}`,
    });
    expect(f.success).toBe(true);
    expect(f.metadata?.id).toBe('filter.reconstruct.line');
    expect(f.gpuErrors).toEqual([]);
  });
}));
