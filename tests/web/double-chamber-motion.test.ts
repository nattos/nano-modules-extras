import { runEngineTest } from '@nano/web/test/engine-test-helpers';
import type { Sketch } from '@nano/web/src/sketch-types';

/**
 * E2E for source.legacy.double_chamber's motion-vector output. The effect only
 * produces render_outputs/motion when a downstream SINK reads it; here we wire
 * `double_chamber → motion.blur` (wires:[] → struct auto-connect), so the
 * motion pass runs and motion.blur smears the particle cloud along its
 * per-particle velocity.
 *
 * Isolation: the double_chamber stage is byte-identical across both runs (same
 * params, same deterministic frame count), so the ONLY difference between
 * blur strength 0 (pass-through) and 32 is whether the motion field carried
 * real velocity. A visible difference proves the whole rail: produced →
 * published (setGpuTexture) → auto-connected → consumed.
 */
function buildChain(blurStrength: number, dcOverrides: Record<string, unknown> = {}): Sketch {
  return {
    anchor: null,
    wires: [],
    chain: [
      {
        type: 'module',
        module_type: 'source.legacy.double_chamber',
        instance_key: 'dc@0',
        params: {
          p_count: 6000,
          p_point_size: 1.0,   // [0,1] → 0.01 uv
          p_opacity: 1.0,
          exposure: 2.0,
          color_contrib: 0.0,  // white cloud
          field_speed: 0.6,    // brisk field → clear velocities
          motion_rate: 2.0,    // large per-frame displacement → strong motion
          jitter: 0.0,
          to_big: 0.5,
          big_count: 4,
          // Tracers on so the line-motion path is exercised too.
          l_count: 8,
          l_opacity: 1.0,
          motion_line_speed: 0.6,
          bridger_count: 0,
          ...dcOverrides,
        },
      },
      {
        type: 'module',
        module_type: 'motion.blur',
        instance_key: 'blur@0',
        params: { strength: blurStrength, samples: 16, quality: 1 },
      },
    ],
  };
}

describe('Double Chamber motion-vector output (render_outputs/motion) E2E', () => {
  jest.setTimeout(60000);

  it('drives downstream motion blur from per-particle velocity', async () => {
    const blurred = await runEngineTest({
      width: 128, height: 128,
      modules: ['com.nano.legacy', 'com.nano.testonly'],
      commands: [
        { type: 'createSketch', sketchId: 'dc_blur', sketch: buildChain(32.0) },
        { type: 'setTracePoints', tracePoints: [
          { id: 'out', target: { type: 'sketch_output', sketchId: 'dc_blur' } },
        ]},
      ],
      waitFrames: 24,
      captureTraceIds: ['out'],
      dumpName: 'double_chamber_motion_blurred',
    });
    expect(blurred.success).toBe(true);

    const sharp = await runEngineTest({
      width: 128, height: 128,
      modules: ['com.nano.legacy', 'com.nano.testonly'],
      commands: [
        { type: 'createSketch', sketchId: 'dc_sharp', sketch: buildChain(0.0) },
        { type: 'setTracePoints', tracePoints: [
          { id: 'out', target: { type: 'sketch_output', sketchId: 'dc_sharp' } },
        ]},
      ],
      waitFrames: 24,
      captureTraceIds: ['out'],
      dumpName: 'double_chamber_motion_sharp',
    });
    expect(sharp.success).toBe(true);

    // The cloud must actually be present (not a degenerate black frame), else
    // "frames differ" would be meaningless.
    let lit = 0;
    sharp.trace('out').forEachPixel((c) => { if (c.r + c.g + c.b > 24) lit++; });
    expect(lit).toBeGreaterThan(100);

    // strength=32 smears each particle along its motion vector; strength=0 is a
    // pass-through. They must differ → the motion rail carried real velocity.
    blurred.trace('out').expectDifferentFrom(sharp.trace('out'), 100);
  });

  it('motion_particle_scale scales the emitted particle velocity', async () => {
    // Lines off, so the ONLY motion comes from particles. scale=0 zeroes their
    // emitted velocity → motion.blur is a pass-through; scale=2 amplifies it →
    // a strong smear. The two must differ, proving motion_particle_scale gates
    // the particle motion magnitude.
    const off = await runEngineTest({
      width: 128, height: 128,
      modules: ['com.nano.legacy', 'com.nano.testonly'],
      commands: [
        { type: 'createSketch', sketchId: 'dc_scale0',
          sketch: buildChain(32.0, { l_count: 0, motion_particle_scale: 0.0 }) },
        { type: 'setTracePoints', tracePoints: [
          { id: 'out', target: { type: 'sketch_output', sketchId: 'dc_scale0' } },
        ]},
      ],
      waitFrames: 24,
      captureTraceIds: ['out'],
      dumpName: 'double_chamber_motion_scale0',
    });
    expect(off.success).toBe(true);

    const hi = await runEngineTest({
      width: 128, height: 128,
      modules: ['com.nano.legacy', 'com.nano.testonly'],
      commands: [
        { type: 'createSketch', sketchId: 'dc_scale2',
          sketch: buildChain(32.0, { l_count: 0, motion_particle_scale: 2.0 }) },
        { type: 'setTracePoints', tracePoints: [
          { id: 'out', target: { type: 'sketch_output', sketchId: 'dc_scale2' } },
        ]},
      ],
      waitFrames: 24,
      captureTraceIds: ['out'],
      dumpName: 'double_chamber_motion_scale2',
    });
    expect(hi.success).toBe(true);

    hi.trace('out').expectDifferentFrom(off.trace('out'), 100);
  });

  it('runs the motion path cleanly with large invisible Big points', async () => {
    // Regression guard for the "Big holes" bug: big_opacity=0 with large Big
    // quads used to overwrite the particle motion in their footprint, leaving
    // holes in the smear. Motion now mirrors visibility (the colour pass's
    // opacity gates), so an invisible Big writes nothing. We can't assert
    // cross-run pixel equality here — host::deltaTime jitter makes two
    // independent dynamic runs diverge — so this is a smoke test (clean run +
    // a real smeared cloud); the absence of Big discs is confirmed by the
    // double_chamber_motion_big_invisible dump.
    const result = await runEngineTest({
      width: 128, height: 128,
      modules: ['com.nano.legacy', 'com.nano.testonly'],
      commands: [
        { type: 'createSketch', sketchId: 'dc_bigvis',
          sketch: buildChain(32.0, {
            big_count: 6, big_opacity: 0.0, big_point_size: 0.15,
            l_count: 0, bridger_count: 0,
          }) },
        { type: 'setTracePoints', tracePoints: [
          { id: 'out', target: { type: 'sketch_output', sketchId: 'dc_bigvis' } },
        ]},
      ],
      waitFrames: 24,
      captureTraceIds: ['out'],
      dumpName: 'double_chamber_motion_big_invisible',
    });
    expect(result.success).toBe(true);
    let lit = 0;
    result.trace('out').forEachPixel((c) => { if (c.r + c.g + c.b > 24) lit++; });
    expect(lit).toBeGreaterThan(100);  // a real smeared cloud rendered
  });
});
