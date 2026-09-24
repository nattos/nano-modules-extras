import { runEngineTest, runEngineMultiPhaseTest } from '@nano/web/test/engine-test-helpers';
import type { Sketch } from '@nano/web/src/sketch-types';

/**
 * E2E coverage for motion.peak_decay (nano bundle) — the per-pixel "peak
 * meter": static pixels' luma falls down a smoothstep sigmoid after a hold;
 * any change snaps the pixel back to full brightness instantly.
 *
 * Determinism: headless frame pacing varies 4–20 ms per frame, so tests are
 * dt-invariant — "no decay yet" assertions use a LONG hold (4 s, far beyond
 * any test span) and "fully decayed" assertions use a SHORT hold+fall with a
 * generous frame wait (the sigmoid has fully landed whatever the pacing).
 * Mid-fall pixel values are never asserted.
 *
 * Input rig: a solid colour (static by construction) through
 * brightness_contrast — patching `brightness` mid-run changes every pixel in
 * one frame, which is the "pixel moved → snap back" stimulus.
 */

const SOLID = [0.7, 0.7, 0.7];              // ≈ (178, 178, 178)
const IN = { r: 178, g: 178, b: 178 };

// Fully-decayed-fast profile: hold+fall = 0.15 s ≪ any 60-frame wait.
const FAST = { hold: 0.05, fall: 0.1, threshold: 0.05 };
// Never-decays-during-test profile.
const HELD = { hold: 4.0, fall: 0.1, threshold: 0.05 };

function buildSketch(params: Record<string, unknown>, color?: number[]): Sketch {
  const chain: any[] = [
    { type: 'module', module_type: 'source.solid_color',
      instance_key: 'bg@0', params: { color: color ?? SOLID } },
    { type: 'module', module_type: 'color.tone.brightness_contrast',
      instance_key: 'bc@0', params: {} },
    { type: 'module', module_type: 'motion.peak_decay',
      instance_key: 'pd@0', params },
  ];
  return { anchor: null, chain, wires: [] } as Sketch;
}

async function render(sketchId: string, params: Record<string, unknown>,
                      opts?: { color?: number[], waitFrames?: number }) {
  const result = await runEngineTest({
    width: 96, height: 96,
    modules: ['com.nano.core', 'com.nano.nano'],
    commands: [
      { type: 'createSketch', sketchId, sketch: buildSketch(params, opts?.color) },
      { type: 'setTracePoints', tracePoints: [
        { id: 'out', target: { type: 'sketch_output', sketchId } },
      ]},
    ],
    waitFrames: opts?.waitFrames ?? 60,
    captureTraceIds: ['out'],
    dumpName: sketchId,
  });
  expect(result.success).toBe(true);
  return result;
}

const PROBES: Array<[number, number]> = [[48, 48], [10, 10], [86, 86], [48, 8], [8, 84]];

describe('motion.peak_decay E2E', () => {
  jest.setTimeout(120000);

  it('registers and appears in plugin state', async () => {
    const r = await render('pd_smoke', { ...FAST }, { waitFrames: 8 });
    const plugin = r.state.plugins.find((p: any) => p.id === 'motion.peak_decay');
    expect(plugin).toBeTruthy();
  });

  it('a static input fully decays to black at amount 1', async () => {
    const r = await render('pd_decay', { ...FAST, amount: 1.0 });
    for (const [x, y] of PROBES) {
      const c = r.trace('out').pixelAt(x, y);
      expect(c.r).toBeLessThan(12);
      expect(c.g).toBeLessThan(12);
      expect(c.b).toBeLessThan(12);
    }
  });

  it('amount 0 passes the input through untouched', async () => {
    const r = await render('pd_amount0', { ...FAST, amount: 0.0 });
    for (const [x, y] of PROBES) {
      r.trace('out').expectPixelAt(x, y, IN, 3);
    }
  });

  it('a long hold keeps static pixels at full brightness', async () => {
    const r = await render('pd_held', { ...HELD, amount: 1.0 }, { waitFrames: 30 });
    for (const [x, y] of PROBES) {
      r.trace('out').expectPixelAt(x, y, IN, 3);
    }
  });

  it('amount is the decay depth: 0.5 lands at half the input luma', async () => {
    const r = await render('pd_half', { ...FAST, amount: 0.5 });
    // Fully decayed gain = 1 - 0.5 → 178 * 0.5 = 89.
    for (const [x, y] of PROBES) {
      const c = r.trace('out').pixelAt(x, y);
      expect(Math.abs(c.r - 89)).toBeLessThanOrEqual(6);
      expect(Math.abs(c.g - 89)).toBeLessThanOrEqual(6);
      expect(Math.abs(c.b - 89)).toBeLessThanOrEqual(6);
    }
  });

  it('chroma is preserved while luma decays', async () => {
    // A saturated orange decayed by half keeps its R:G ratio.
    const r = await render('pd_chroma', { ...FAST, amount: 0.5 },
                           { color: [0.8, 0.4, 0.1] });
    const c = r.trace('out').pixelAt(48, 48);
    // 0.8/0.4/0.1 * 0.5 → ≈ (102, 51, 13).
    expect(Math.abs(c.r - 102)).toBeLessThanOrEqual(7);
    expect(Math.abs(c.g - 51)).toBeLessThanOrEqual(7);
    expect(Math.abs(c.b - 13)).toBeLessThanOrEqual(7);
  });

  it('a changed pixel snaps back to full brightness instantly', async () => {
    // Phase 1: decay the static solid to black (150 frames ≥ 0.6 s at the
    // fastest pacing, past hold+fall = 0.45 s). Phase 2: an instant
    // brightness step changes every pixel — the capture 6 frames later
    // (≤ 120 ms at the slowest pacing) lands well inside the fresh 0.3 s
    // hold, so the snapped-back gain is still 1. hold itself is NOT patched:
    // raising it live revives decayed pixels by design, which would mask
    // what this test proves.
    const r = await runEngineMultiPhaseTest({
      width: 96, height: 96,
      modules: ['com.nano.core', 'com.nano.nano'],
      dumpName: 'pd_snap',
      phases: [
        {
          commands: [
            { type: 'createSketch', sketchId: 'pd_snap',
              sketch: buildSketch({ hold: 0.3, fall: 0.15, threshold: 0.05,
                                    amount: 1.0 }) },
            { type: 'setTracePoints', tracePoints: [
              { id: 'out', target: { type: 'sketch_output', sketchId: 'pd_snap' } },
            ]},
          ],
          waitFrames: 150, captureTraceIds: ['out'],
        },
        {
          commands: [
            { type: 'setParam', sketchId: 'pd_snap', colIdx: 0, chainIdx: 1,
              paramKey: 'brightness', value: -0.3 },
          ],
          waitFrames: 6, captureTraceIds: ['out'],
        },
      ],
    });
    expect(r.success).toBe(true);
    // Phase 1: decayed to black.
    const dark = r.phases[0].trace('out').pixelAt(48, 48);
    expect(dark.r).toBeLessThan(12);
    // Phase 2: the input is darker now (0.7 - 0.3 → ≈ 102) but at FULL
    // gain — clearly brighter than the decayed phase-1 frame.
    const snapped = r.phases[1].trace('out').pixelAt(48, 48);
    expect(snapped.r).toBeGreaterThan(60);
    r.phases[1].trace('out').expectDifferentFrom(r.phases[0].trace('out'), 20);
  });

  it('rgb_balance picks what the meter hears: chroma swap trips at 1, not at 0', async () => {
    // Stimulus: an HSL hue rotation of a low-saturation red along the
    // luma-quiet red↔blue axis — a chroma-heavy, near-luma-neutral change.
    // hue_shift -0.667 (measured: → 240°) takes (0.625,0.375,0.375) to
    // (0.375,0.375,0.625): Δluma ≈ 0.035, max channel Δ = 0.25. threshold
    // 0.1 sits cleanly between: balance 0 meters max(0.035, 0) → stays
    // decayed; balance 1 meters 0.25 → catches and snaps back. (A rotation
    // toward GREEN would NOT do: Δluma 0.124 trips even luma-only metering.)
    const phased = async (id: string, rgb_balance: number) => {
      const chain: any[] = [
        { type: 'module', module_type: 'source.solid_color',
          instance_key: 'bg@0', params: { color: [0.625, 0.375, 0.375] } },
        { type: 'module', module_type: 'color.hsl',
          instance_key: 'hsl@0', params: {} },
        { type: 'module', module_type: 'motion.peak_decay',
          instance_key: 'pd@0',
          params: { hold: 0.3, fall: 0.15, threshold: 0.1, amount: 1.0,
                    rgb_balance } },
      ];
      const r = await runEngineMultiPhaseTest({
        width: 96, height: 96,
        modules: ['com.nano.core', 'com.nano.nano'],
        dumpName: id,
        phases: [
          {
            commands: [
              { type: 'createSketch', sketchId: id,
                sketch: { anchor: null, chain, wires: [] } as Sketch },
              { type: 'setTracePoints', tracePoints: [
                { id: 'out', target: { type: 'sketch_output', sketchId: id } },
              ]},
            ],
            waitFrames: 150, captureTraceIds: ['out'],
          },
          {
            commands: [
              { type: 'setParam', sketchId: id, colIdx: 0, chainIdx: 1,
                paramKey: 'hue_shift', value: -0.667 },
            ],
            waitFrames: 6, captureTraceIds: ['out'],
          },
        ],
      });
      expect(r.success).toBe(true);
      expect(r.phases[0].trace('out').pixelAt(48, 48).r).toBeLessThan(12);
      return r.phases[1].trace('out').pixelAt(48, 48);
    };
    const lumaOnly = await phased('pd_bal0', 0.0);
    expect(lumaOnly.r).toBeLessThan(12);       // deaf to the chroma swap
    const fullRgb = await phased('pd_bal1', 1.0);
    expect(fullRgb.b).toBeGreaterThan(100);    // caught: rotated blue at gain 1
  });

  it('Rise Only: an upward luma step takes over the decayed peak', async () => {
    const r = await runEngineMultiPhaseTest({
      width: 96, height: 96,
      modules: ['com.nano.core', 'com.nano.nano'],
      dumpName: 'pd_rise_up',
      phases: [
        {
          commands: [
            { type: 'createSketch', sketchId: 'pd_rise_up',
              sketch: buildSketch({ hold: 0.3, fall: 0.15, threshold: 0.05,
                                    amount: 1.0, catch: 1 },
                                  [0.4, 0.4, 0.4]) },
            { type: 'setTracePoints', tracePoints: [
              { id: 'out', target: { type: 'sketch_output', sketchId: 'pd_rise_up' } },
            ]},
          ],
          waitFrames: 150, captureTraceIds: ['out'],
        },
        {
          commands: [
            { type: 'setParam', sketchId: 'pd_rise_up', colIdx: 0, chainIdx: 1,
              paramKey: 'brightness', value: 0.4 },
          ],
          waitFrames: 6, captureTraceIds: ['out'],
        },
      ],
    });
    expect(r.success).toBe(true);
    expect(r.phases[0].trace('out').pixelAt(48, 48).r).toBeLessThan(12);
    // 0.4 + 0.4 → ≈ 204 at the snapped-back gain of 1.
    expect(r.phases[1].trace('out').pixelAt(48, 48).r).toBeGreaterThan(150);
  });

  it('Rise Only: a downward step does not catch — the fall continues', async () => {
    // amount 0.6 → the decayed gain floor is 0.4 (solid 0.7 → ≈ 71).
    // Dropping the input to 0.2 must NOT reset the gain: the display shows
    // the darker input still at floor gain (0.2·0.4 → ≈ 20), where an Any
    // Change reset would show it at gain 1 (≈ 51). Nothing resets after the
    // step, so the phase-2 capture timing is unconstrained.
    const r = await runEngineMultiPhaseTest({
      width: 96, height: 96,
      modules: ['com.nano.core', 'com.nano.nano'],
      dumpName: 'pd_rise_down',
      phases: [
        {
          commands: [
            { type: 'createSketch', sketchId: 'pd_rise_down',
              sketch: buildSketch({ hold: 0.05, fall: 0.1, threshold: 0.05,
                                    amount: 0.6, catch: 1 }) },
            { type: 'setTracePoints', tracePoints: [
              { id: 'out', target: { type: 'sketch_output', sketchId: 'pd_rise_down' } },
            ]},
          ],
          waitFrames: 60, captureTraceIds: ['out'],
        },
        {
          commands: [
            { type: 'setParam', sketchId: 'pd_rise_down', colIdx: 0, chainIdx: 1,
              paramKey: 'brightness', value: -0.5 },
          ],
          waitFrames: 10, captureTraceIds: ['out'],
        },
      ],
    });
    expect(r.success).toBe(true);
    const floor = r.phases[0].trace('out').pixelAt(48, 48);
    expect(Math.abs(floor.r - 71)).toBeLessThanOrEqual(8);
    const held = r.phases[1].trace('out').pixelAt(48, 48);
    expect(Math.abs(held.r - 20)).toBeLessThanOrEqual(7);   // floor gain, no reset
  });

  it('sub-threshold changes do not reset the decay', async () => {
    // Stimulus is a small brightness nudge (~0.02 luma step) against a high
    // threshold: pixels must STAY dark. Guards against over-eager metering.
    const r = await runEngineMultiPhaseTest({
      width: 96, height: 96,
      modules: ['com.nano.core', 'com.nano.nano'],
      dumpName: 'pd_subthresh',
      phases: [
        {
          commands: [
            { type: 'createSketch', sketchId: 'pd_subthresh',
              sketch: buildSketch({ hold: 0.05, fall: 0.1, threshold: 0.2,
                                    amount: 1.0 }) },
            { type: 'setTracePoints', tracePoints: [
              { id: 'out', target: { type: 'sketch_output', sketchId: 'pd_subthresh' } },
            ]},
          ],
          waitFrames: 60, captureTraceIds: ['out'],
        },
        {
          commands: [
            { type: 'setParam', sketchId: 'pd_subthresh', colIdx: 0, chainIdx: 1,
              paramKey: 'brightness', value: -0.02 },
          ],
          waitFrames: 6, captureTraceIds: ['out'],
        },
      ],
    });
    expect(r.success).toBe(true);
    const before = r.phases[0].trace('out').pixelAt(48, 48);
    const after = r.phases[1].trace('out').pixelAt(48, 48);
    expect(before.r).toBeLessThan(12);
    expect(after.r).toBeLessThan(12);
  });
});
