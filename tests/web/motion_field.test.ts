/**
 * motion.field (nano) feeding core's motion.blur through the engine — moved
 * out of nano-modules' render-outputs suite with the effect.
 */
import { runEngineTest } from '@nano/web/test/engine-test-helpers';
import type { Sketch } from '@nano/web/src/sketch-types';

describe('motion.field', () => {
  jest.setTimeout(60000);

  it('motion_field generates motion vectors from input luma', async () => {
    // Bright generator → motion_field → motion_blur. Default
    // rotation_weight=1 makes every above-threshold pixel emit a
    // +x velocity. motion_blur then smears the bright background
    // horizontally. We confirm: (a) the chain runs cleanly, (b) the
    // output has many distinct colors (the smear creates a brightness
    // gradient instead of two flat colors), and (c) the visualization
    // path doesn't crash when vis_opacity > 0.
    const sketch: Sketch = {
      anchor: null,
      wires: [],
      chain: [
        {
          type: 'module',
          module_type: 'source.solid_color',
          instance_key: 'bg@0',
          // Bright enough to trip the default 0.5 threshold.
          params: { color: [0.9, 0.7, 0.4] },
        },
        {
          type: 'module',
          module_type: 'motion.field',
          instance_key: 'mf@0',
          params: {
            threshold: 0.4,
            softness: 0.05,
            magnitude: 0.008,
            mag_jitter: 0.4,
            mag_noise_scale: 12.0,
            rotation: 30.0,
            rotation_weight: 1.0,
            radial_weight: 0.0,
            radial_anchor: [0.5, 0.5],
            gradient_weight: 0.0,
            gradient_bias: 90.0,
            angle_jitter: 0.1,
            angle_noise_scale: 24.0,
            seed: 7,
            // Visualization off — production usage. The motion
            // vectors flow downstream regardless.
            // Vis on so the smoke test sees per-pixel hue/value
            // variation in the output even when tex_in is uniform —
            // confirms the velocity field is actually being computed.
            vis_opacity: 1.0,
            vis_scale: 200.0,
          },
        },
        {
          type: 'module',
          module_type: 'motion.blur',
          instance_key: 'blur@0',
          params: { strength: 24.0, samples: 12, quality: 1 },
        },
      ],
    };

    const result = await runEngineTest({
      width: 128, height: 128,
      modules: ['com.nano.testonly', 'com.nano.nano'],
      commands: [
        { type: 'createSketch', sketchId: 'mf_chain', sketch },
        { type: 'setTracePoints', tracePoints: [
          { id: 'out', target: { type: 'sketch_output', sketchId: 'mf_chain' } },
        ]},
      ],
      waitFrames: 20,
      captureTraceIds: ['out'],
      dumpName: 'render_outputs_motion_field',
    });
    expect(result.success).toBe(true);

    // motion_field's per-pixel jitter combined with motion_blur's
    // gather should produce many subtly different colors. A pure
    // pass-through (no motion field engaged) would be nearly uniform.
    const frame = result.trace('out');
    const colors = new Set<string>();
    frame.forEachPixel((c) => { colors.add(`${c.r},${c.g},${c.b}`); });
    expect(colors.size).toBeGreaterThan(20);
  });
});
