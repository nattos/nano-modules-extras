import { runGpuEffectTest, forEachBackend } from '@nano/web/test/gpu-test-helpers';
import { runEngineTest } from '@nano/web/test/engine-test-helpers';
import type { Sketch } from '@nano/web/src/sketch-types';

/**
 * E2E for warp.legacy.stutter_scale — "Stutter Scale", the v2 port of the
 * Resolume Wire "Stutter Scale 2" patch (beat-stutter scale/flip/hue/invert
 * glitch). Exercised on a source.grid chain; the effect is a pure function of
 * `sweep`, so a fixed sweep pins one deterministic step.
 */
/** The solid probe colour both halves of this suite feed in. */
const SOLID: [number, number, number, number] = [0.15, 0.30, 0.55, 1.0];

// The metadata case is effect-level and runs on BOTH backends — it pins the
// schema this effect publishes, which is exactly where a host-side derivation
// can diverge. Everything below drives runEngineTest (the engine harness page:
// executor.wasm, wires, trace points), which has no native runner; the comp
// runner is the native equivalent and a native sketch host is a follow-up.
forEachBackend((backend) => {
describe(`Stutter Scale (warp.legacy.stutter_scale) E2E schema (${backend})`, () => {
  jest.setTimeout(60000);

  it('declares metadata and its parameters', async () => {
    const frame = await runGpuEffectTest({
      module: 'warp.legacy.stutter_scale', bundle: 'legacy',
      inputColor: SOLID,
      dumpName: 'ss_metadata',
    });
    expect(frame.success).toBe(true);
    expect(frame.metadata?.id).toBe('warp.legacy.stutter_scale');
    const names = frame.params.map(p => p.name);
    for (const n of ['sweep', 'levels', 'min_scale', 'max_scale', 'jitter',
                     'hue', 'boost', 'intensity', 'deadzone', 'start_deadzone',
                     'end_deadzone', 'fill', 'flip', 'color_invert', 'seed']) {
      expect(names).toContain(n);
    }
  });
});
});

describe('Stutter Scale (warp.legacy.stutter_scale) E2E', () => {
  jest.setTimeout(60000);

  const runChain = (id: string, params: Record<string, number>, dump: string) => {
    const sketch: Sketch = {
      anchor: null,
      wires: [],
      chain: [
        { type: 'module', module_type: 'source.grid', instance_key: 'grid@0', params: {} },
        { type: 'module', module_type: 'warp.legacy.stutter_scale', instance_key: 'ss@0', params },
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
      waitFrames: 6,
      captureTraceIds: ['out'],
      dumpName: dump,
    });
  };

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
    waitFrames: 6,
    captureTraceIds: ['out'],
    dumpName: dump,
  });

  it('a stutter step transforms the structured input', async () => {
    const off = await runChain('ss_off', { intensity: 0.0 }, 'ss_off');
    const on = await runChain('ss_on',
      { intensity: 1.0, sweep: 0.5, min_scale: 2.0, max_scale: 8.0, jitter: 0.3 }, 'ss_on');
    expect(off.success).toBe(true);
    expect(on.success).toBe(true);
    let lit = 0;
    off.trace('out').forEachPixel((c) => { if (c.r + c.g + c.b > 24) lit++; });
    expect(lit).toBeGreaterThan(100);
    on.trace('out').expectDifferentFrom(off.trace('out'), 100);
  });

  it('different sweep positions land on different stutter steps', async () => {
    const a = await runChain('ss_a',
      { intensity: 1.0, sweep: 0.15, min_scale: 2.0, max_scale: 8.0, jitter: 0.4 }, 'ss_step_a');
    const b = await runChain('ss_b',
      { intensity: 1.0, sweep: 0.85, min_scale: 2.0, max_scale: 8.0, jitter: 0.4 }, 'ss_step_b');
    expect(a.success).toBe(true);
    expect(b.success).toBe(true);
    b.trace('out').expectDifferentFrom(a.trace('out'), 100);
  });

  it('the scale sweeps progressively from min_scale to max_scale', async () => {
    // With the randoms neutralized (jitter/flip/hue off), the step only picks
    // the zoom — and the zoom must be the PROGRESSIVE min→max walk, not a
    // random draw. Pin both ends against runs whose range is collapsed to a
    // single value: sweep=0 must render exactly min_scale (== a min==max==2
    // run at any sweep), sweep→1 must render exactly max_scale.
    const base = { intensity: 1.0, jitter: 0.0, flip: 0, hue: 0.0 };
    const atStart = await runChain('ss_p0',
      { ...base, sweep: 0.0, min_scale: 2.0, max_scale: 8.0 }, 'ss_prog_start');
    const pinned2 = await runChain('ss_p2',
      { ...base, sweep: 0.77, min_scale: 2.0, max_scale: 2.0 }, 'ss_pinned_2');
    const atEnd = await runChain('ss_p1',
      { ...base, sweep: 0.999, min_scale: 2.0, max_scale: 8.0, end_deadzone: 0 }, 'ss_prog_end');
    const pinned8 = await runChain('ss_p8',
      { ...base, sweep: 0.3, min_scale: 8.0, max_scale: 8.0 }, 'ss_pinned_8');
    for (const r of [atStart, pinned2, atEnd, pinned8]) expect(r.success).toBe(true);
    atStart.trace('out').expectSameAs(pinned2.trace('out'), 2);
    atEnd.trace('out').expectSameAs(pinned8.trace('out'), 2);
    // And the two ends genuinely differ (2x vs 8x zoom).
    atEnd.trace('out').expectDifferentFrom(atStart.trace('out'), 100);
  });

  it('jitter does not touch the start endpoint', async () => {
    // The Wire patch gates the jitter by phase^0.1 — exactly zero at sweep=0.
    // With everything else neutral (scale range collapsed to 1, flip/hue/boost
    // off), full jitter at sweep=0 must be a clean passthrough of the grid.
    const grid = await runGridOnly('ss_jgrid', 'ss_jitter_grid');
    const start = await runChain('ss_jstart',
      { intensity: 1.0, sweep: 0.0, jitter: 1.0, min_scale: 1.0, max_scale: 1.0,
        flip: 0, hue: 0.0, boost: 0.0 }, 'ss_jitter_start');
    expect(grid.success).toBe(true);
    expect(start.success).toBe(true);
    start.trace('out').expectSameAs(grid.trace('out'), 2);
  });

  it('the end deadzone turns the stutter off (≠ the active stutter)', async () => {
    // sweep=1 sits in the end deadzone (default on) → output goes transparent/
    // off, which differs from the active stutter mid-sweep. (Transparency
    // itself isn't assertable — the GPU runner reads back opaque — so we
    // compare states.)
    const dead = await runChain('ss_dead',
      { intensity: 1.0, sweep: 1.0, min_scale: 2.0, max_scale: 8.0 }, 'ss_dead');
    const active = await runChain('ss_active',
      { intensity: 1.0, sweep: 0.5, min_scale: 2.0, max_scale: 8.0 }, 'ss_active');
    expect(dead.success).toBe(true);
    expect(active.success).toBe(true);
    active.trace('out').expectDifferentFrom(dead.trace('out'), 100);
  });

  it('the fill style picks the deadzone and out-of-bounds look', async () => {
    // Deadzone: fill=Black (0) → an opaque black frame; the default Edge (2)
    // stays transparent (renders as the checkerboard backdrop in traces).
    const deadBlack = await runChain('ss_f_db',
      { intensity: 1.0, sweep: 1.0, fill: 0 }, 'ss_fill_dead_black');
    const deadEdge = await runChain('ss_f_de',
      { intensity: 1.0, sweep: 1.0 }, 'ss_fill_dead_edge');
    expect(deadBlack.success).toBe(true);
    expect(deadEdge.success).toBe(true);
    let dark = 0;
    deadBlack.trace('out').forEachPixel((c) => { if (c.r + c.g + c.b < 30) dark++; });
    expect(dark).toBe(128 * 128);
    deadBlack.trace('out').expectDifferentFrom(deadEdge.trace('out'), 100);

    // Out-of-bounds: a jittered step samples past the border; black fill vs
    // the edge clamp-smear must differ on at least one of two probed steps
    // (the per-step jitter is seeded, so both being ~zero is implausible).
    const probe = (id: string, sweep: number, fill: number, dump: string) => runChain(id,
      { intensity: 1.0, sweep, min_scale: 1.0, max_scale: 1.0, jitter: 1.0,
        flip: 0, fill }, dump);
    const eA = await probe('ss_f_ea', 0.35, 2, 'ss_fill_edge_a');
    const bA = await probe('ss_f_ba', 0.35, 0, 'ss_fill_black_a');
    const eB = await probe('ss_f_eb', 0.55, 2, 'ss_fill_edge_b');
    const bB = await probe('ss_f_bb', 0.55, 0, 'ss_fill_black_b');
    for (const r of [eA, bA, eB, bB]) expect(r.success).toBe(true);
    const diffs = bA.trace('out').diffCount(eA.trace('out'))
                + bB.trace('out').diffCount(eB.trace('out'));
    expect(diffs).toBeGreaterThan(50);
  });

  it('the start deadzone is off by default and toggles on', async () => {
    // Default: sweep=0 is ACTIVE (renders the min_scale step — differs from
    // the plain grid). With start_deadzone on, the same sweep goes off.
    const grid = await runGridOnly('ss_grid', 'ss_gridonly');
    const activeStart = await runChain('ss_s_on',
      { intensity: 1.0, sweep: 0.0, min_scale: 2.0, max_scale: 8.0, jitter: 0.0, flip: 0 },
      'ss_start_active');
    const deadStart = await runChain('ss_s_off',
      { intensity: 1.0, sweep: 0.0, min_scale: 2.0, max_scale: 8.0, jitter: 0.0, flip: 0,
        start_deadzone: 1 },
      'ss_start_dead');
    expect(grid.success).toBe(true);
    expect(activeStart.success).toBe(true);
    expect(deadStart.success).toBe(true);
    activeStart.trace('out').expectDifferentFrom(grid.trace('out'), 100);
    deadStart.trace('out').expectDifferentFrom(activeStart.trace('out'), 100);
  });
});
