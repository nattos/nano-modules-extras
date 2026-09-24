import { runEngineTest, runEngineMultiPhaseTest } from '@nano/web/test/engine-test-helpers';
import type { Sketch } from '@nano/web/src/sketch-types';

/**
 * mod.source.spectral_lfo (nano bundle) E2E.
 *
 * Schema introspection + functional checks via the proven passthrough-wire
 * pattern (see engine-wires "forward scalar wire from a passthrough modulation
 * source"): a white solid → spectral_lfo (passthrough data module) →
 * brightness_contrast, with the LFO's scalar `output` wired into
 * bc.brightness. The center pixel is then a monotonic readout of the live LFO
 * value, so it changes as the manifold position or the phase advances.
 */

const W = 64, H = 64;

function buildSketch(lfoParams: Record<string, unknown>, outField = 'output'): Sketch {
  return {
    anchor: null,
    chain: [
      { type: 'module', module_type: 'source.solid_color', instance_key: 'src@0',
        params: { color: [1.0, 1.0, 1.0] } },
      { type: 'module', module_type: 'mod.source.spectral_lfo', instance_key: 'lfo@0',
        params: lfoParams },
      { type: 'module', module_type: 'color.tone.brightness_contrast', instance_key: 'bc@0',
        params: { brightness: 1.0, contrast: -0.5 } },
    ],
    wires: [
      { id: 'w0', src: { instanceKey: 'lfo@0', field: outField },
        dest: { instanceKey: 'bc@0', field: 'brightness' } },
    ],
  } as Sketch;
}

describe('mod.source.spectral_lfo', () => {
  jest.setTimeout(30000);

  it('declares the schema: data_output + metric/interpolation/morph params', async () => {
    const result = await runEngineTest({
      width: W, height: H,
      modules: ['source.solid_color', 'color.tone.brightness_contrast', 'com.nano.nano'],
      commands: [{ type: 'createSketch', sketchId: 'sl_schema',
                   sketch: buildSketch({ rate: 0.0 }) }],
      waitFrames: 5,
      dumpName: 'spectral_lfo_schema',
    });
    expect(result.success).toBe(true);

    const lfo = result.state.plugins.find((p: any) => p.id === 'mod.source.spectral_lfo');
    expect(lfo).toBeTruthy();

    // `output` is a data_output (io kind=2) AND a schema field.
    expect(lfo.io.find((io: any) => io.name === 'output' && io.kind === 2)).toBeTruthy();
    expect(lfo.params.find((p: any) => p.name === 'output')).toBeTruthy();

    // Exposed controls are present.
    for (const name of ['rate', 'amplitude', 'morph_x', 'morph_y', 'metric', 'interpolation',
                        'satellites', 'sat_spread', 'sat_rotation']) {
      expect(lfo.params.find((p: any) => p.name === name)).toBeTruthy();
    }

    // The three satellite taps are data outputs (io kind=2).
    for (const name of ['output_a', 'output_b', 'output_c']) {
      expect(lfo.io.find((io: any) => io.name === name && io.kind === 2)).toBeTruthy();
    }

    // The autopilot live-position broadcasts are SecondaryOutputs, not inputs.
    expect(lfo.io.find((io: any) => io.name === 'autopilot_x')).toBeTruthy();

    // Inputs are not mistaken for data outputs.
    expect(lfo.io.find((io: any) => io.name === 'rate' && io.kind === 2)).toBeUndefined();
  });

  it('oscillates over time when rate > 0 (phase accumulator advances)', async () => {
    const r = await runEngineMultiPhaseTest({
      width: W, height: H,
      modules: ['source.solid_color', 'color.tone.brightness_contrast', 'com.nano.nano'],
      dumpName: 'spectral_lfo_osc',
      phases: [
        {
          commands: [
            { type: 'createSketch', sketchId: 'sl_osc',
              sketch: buildSketch({ rate: 0.6, amplitude: 1.0, morph_x: 0.4, morph_y: 0.4 }) },
            { type: 'setTracePoints', tracePoints: [
              { id: 'out', target: { type: 'sketch_output', sketchId: 'sl_osc' } },
            ]},
          ],
          waitFrames: 4, captureTraceIds: ['out'],
        },
        { waitFrames: 10, captureTraceIds: ['out'] },
        { waitFrames: 16, captureTraceIds: ['out'] },
      ],
    });
    expect(r.success).toBe(true);

    const vals = [0, 1, 2].map(i => r.phases[i].trace('out').pixelAt(W / 2, H / 2).r);
    // Every readout is a valid grayscale level…
    for (const v of vals) { expect(v).toBeGreaterThanOrEqual(0); expect(v).toBeLessThanOrEqual(255); }
    // …and the LFO is moving: the three samples are not all identical.
    const spread = Math.max(...vals) - Math.min(...vals);
    expect(spread).toBeGreaterThan(2);
  });

  it('responds to the manifold position (different shapes → different signal)', async () => {
    // Two far-apart manifold positions select very different LFO shapes. Run
    // both at the same rate and compare their wired output across several phases
    // (a single frame can coincide — e.g. both shapes pass through 0 — so we
    // sample three phases and require the two sequences to differ somewhere).
    const r = await runEngineMultiPhaseTest({
      width: W, height: H,
      modules: ['source.solid_color', 'color.tone.brightness_contrast', 'com.nano.nano'],
      dumpName: 'spectral_lfo_morph',
      phases: [
        {
          commands: [
            { type: 'createSketch', sketchId: 'sl_a',
              sketch: buildSketch({ rate: 0.5, morph_x: 0.12, morph_y: 0.12 }) },
            { type: 'createSketch', sketchId: 'sl_b',
              sketch: buildSketch({ rate: 0.5, morph_x: 0.88, morph_y: 0.88 }) },
            { type: 'setTracePoints', tracePoints: [
              { id: 'a', target: { type: 'sketch_output', sketchId: 'sl_a' } },
              { id: 'b', target: { type: 'sketch_output', sketchId: 'sl_b' } },
            ]},
          ],
          waitFrames: 4, captureTraceIds: ['a', 'b'],
        },
        { waitFrames: 8, captureTraceIds: ['a', 'b'] },
        { waitFrames: 12, captureTraceIds: ['a', 'b'] },
      ],
    });
    expect(r.success).toBe(true);

    const maxDiff = Math.max(...[0, 1, 2].map(i =>
      Math.abs(r.phases[i].trace('a').pixelAt(W / 2, H / 2).r
             - r.phases[i].trace('b').pixelAt(W / 2, H / 2).r)));
    expect(maxDiff).toBeGreaterThan(2);
  });

  it('satellites: output_a is live when off and selects an offset shape when on', async () => {
    // Three sketches at the same center/rate. `center` taps `output`; `off`
    // taps `output_a` with satellites OFF (mirrors the center curve — it must
    // still be live, not stuck); `sat` taps `output_a` with satellites ON +
    // wide spread (an offset manifold position → a different shape).
    // (Cross-sketch phase isn't bit-exact, so we assert behavior over phases
    // rather than per-frame equality — same idiom as the morph test above.)
    const params = { rate: 0.5, morph_x: 0.5, morph_y: 0.5 };
    const r = await runEngineMultiPhaseTest({
      width: W, height: H,
      modules: ['source.solid_color', 'color.tone.brightness_contrast', 'com.nano.nano'],
      dumpName: 'spectral_lfo_satellites',
      phases: [
        {
          commands: [
            { type: 'createSketch', sketchId: 'sl_center',
              sketch: buildSketch({ ...params }, 'output') },
            { type: 'createSketch', sketchId: 'sl_off',
              sketch: buildSketch({ ...params, satellites: false }, 'output_a') },
            { type: 'createSketch', sketchId: 'sl_sat',
              sketch: buildSketch({ ...params, satellites: true, sat_spread: 0.7, sat_rotation: 0.0 },
                                  'output_a') },
            { type: 'setTracePoints', tracePoints: [
              { id: 'center', target: { type: 'sketch_output', sketchId: 'sl_center' } },
              { id: 'off', target: { type: 'sketch_output', sketchId: 'sl_off' } },
              { id: 'sat', target: { type: 'sketch_output', sketchId: 'sl_sat' } },
            ]},
          ],
          waitFrames: 4, captureTraceIds: ['center', 'off', 'sat'],
        },
        { waitFrames: 8, captureTraceIds: ['center', 'off', 'sat'] },
        { waitFrames: 12, captureTraceIds: ['center', 'off', 'sat'] },
      ],
    });
    expect(r.success).toBe(true);

    const seq = (id: string) => [0, 1, 2].map(i => r.phases[i].trace(id).pixelAt(W / 2, H / 2).r);
    const center = seq('center'), off = seq('off'), sat = seq('sat');
    for (const v of [...center, ...off, ...sat]) { expect(v).toBeGreaterThanOrEqual(0); expect(v).toBeLessThanOrEqual(255); }

    // Satellites off → output_a is published and tracks the live (moving) curve.
    expect(Math.max(...off) - Math.min(...off)).toBeGreaterThan(2);

    // Satellites on → the offset tap selects a different shape than the center.
    const satDiff = Math.max(...[0, 1, 2].map(i => Math.abs(sat[i] - center[i])));
    expect(satDiff).toBeGreaterThan(2);
  });
});
