import { runEngineMultiPhaseTest } from '@nano/web/test/engine-test-helpers';
import type { Sketch } from '@nano/web/src/sketch-types';

/**
 * E2E for mod.shaper.spectral — the Spectral Curve shaper. It builds the SAME
 * spectrally-morphed LFO curve as mod.source.spectral_lfo (shared spectral_curve.cpp)
 * but indexes it by the `input` modulation value instead of time, so the morphed
 * envelope becomes a remapping curve. Probe: white solid → mod.shaper.spectral → bc,
 * with mod.shaper.spectral.output wired into bc.brightness. Sweeping `input` across the
 * curve produces a varying remapped output (the curve is a non-trivial LFO
 * shape, not an identity passthrough).
 */
describe('mod.shaper.spectral shaper node E2E', () => {
  jest.setTimeout(40000);

  const sketch: Sketch = {
    anchor: null,
    chain: [
      { type: 'module', module_type: 'source.solid_color', instance_key: 'src@0',
        params: { color: [1.0, 1.0, 1.0] } },
      { type: 'module', module_type: 'mod.shaper.spectral', instance_key: 'sp@0',
        params: { morph_x: 0.5, morph_y: 0.5, metric: 0, interpolation: 1, input: 0.0 } },
      { type: 'module', module_type: 'color.tone.brightness_contrast', instance_key: 'bc@0',
        params: { brightness: 1.0, contrast: 0.25 } },
    ],
    wires: [
      { id: 'w1', src: { instanceKey: 'sp@0', field: 'output' },
        dest: { instanceKey: 'bc@0', field: 'brightness' }, combine: 'replace' },
    ],
  } as Sketch;

  it('remaps the input through the spectral curve (output varies across the sweep)', async () => {
    const inputs = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0];
    const result = await runEngineMultiPhaseTest({
      width: 64, height: 64,
      modules: ['source.solid_color', 'color.tone.brightness_contrast', 'com.nano.nano'],
      phases: [
        { commands: [
            { type: 'createSketch', sketchId: 'sp', sketch },
            { type: 'setTracePoints', tracePoints: [{ id: 'out', target: { type: 'sketch_output', sketchId: 'sp' } }] },
          ],
          waitFrames: 20, captureTraceIds: ['out'] },
        ...inputs.slice(1).map(v => ({
          commands: [{ type: 'setParam', sketchId: 'sp', colIdx: 0, chainIdx: 1, paramKey: 'input', value: v }],
          waitFrames: 8, captureTraceIds: ['out'],
        })),
      ],
      dumpName: 'sp_sweep',
    });
    expect(result.success).toBe(true);

    const vals = result.phases.map(p => p.trace('out').averageColor().r);
    // Every sample renders a valid grey value.
    for (const v of vals) { expect(v).toBeGreaterThanOrEqual(0); expect(v).toBeLessThanOrEqual(255); }
    // The curve remaps non-trivially: sweeping the input spans a real range.
    const span = Math.max(...vals) - Math.min(...vals);
    expect(span).toBeGreaterThan(25);
  });
});
