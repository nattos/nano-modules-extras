import { runGpuTest, runGpuEffectTest, forEachBackend } from '@nano/web/test/gpu-test-helpers';
import { runEngineTest } from '@nano/web/test/engine-test-helpers';
import type { Sketch } from '@nano/web/src/sketch-types';

// Per-effect tests for `source.light.soft_glow` against the `lights` bundle.
//
// Soft glow accumulates gaussian contributions from a pool of drifting
// blobs and runs the accumulated intensity through a hue-shifting ramp.
// Drift is non-deterministic on time, so pixel-perfect assertions are
// limited — we focus on:
//   - intensity=0 → exact passthrough
//   - blob_count=0 → exact passthrough
//   - default params on a black input → SOME bright pixels exist
//   - very small blobs in a big canvas → mostly black with bright spots

forEachBackend((backend) => {
describe(`Soft Glow Effect E2E (${backend})`, () => {
  jest.setTimeout(30000);
  // Engine-level tests below (using runEngineTest with sketches /
  // rails) run via Puppeteer only — the native runner is per-effect
  // only in Phase 1. The metal-mode pass early-returns past them.
  const skipEngine = backend === 'metal';

  it('declares metadata and I/O', async () => {
    const frame = await runGpuTest({
      module: 'source.light.soft_glow',
      bundle: 'lights',
      dumpName: 'soft_glow_metadata',
    });
    expect(frame.success).toBe(true);
    expect(frame.metadata?.id).toBe('source.light.soft_glow');
  });

  it('intensity=0 passes input through unchanged', async () => {
    const frame = await runGpuEffectTest({
      module: 'source.light.soft_glow',
      bundle: 'lights',
      inputColor: [0.3, 0.6, 0.9, 1.0],
      params: [['intensity', 0.0]],
      dumpName: 'soft_glow_passthrough_intensity',
    });
    expect(frame.success).toBe(true);
    // With intensity 0 the gaussian accumulation produces 0 → hsv→rgb
    // yields black → additive blend leaves input unchanged.
    frame.expectUniformColor({ r: 76, g: 153, b: 230, a: 255 }, 5);
  });

  it('blob_count=1 with intensity=0 also passes through', async () => {
    const frame = await runGpuEffectTest({
      module: 'source.light.soft_glow',
      bundle: 'lights',
      inputColor: [0.5, 0.5, 0.5, 1.0],
      params: [
        ['blob_count', 1],
        ['intensity', 0.0],
      ],
      dumpName: 'soft_glow_one_blob_dark',
    });
    expect(frame.success).toBe(true);
    frame.expectUniformColor({ r: 128, g: 128, b: 128 }, 5);
  });

  it('produces glow over a black background at default-ish params', async () => {
    const frame = await runGpuEffectTest({
      module: 'source.light.soft_glow',
      bundle: 'lights',
      width: 128, height: 128,
      inputColor: [0.0, 0.0, 0.0, 1.0],
      params: [
        ['blob_count', 12],
        ['blob_size', 0.3],
        ['intensity', 1.0],
        ['hue',  0.13],
        ['hue_shift', -0.13],   // hue ranges [0, 0.13] across amp 0..1
        ['saturation', 1.0],
        ['white_point', 1.5],
        // Pin amplitude modulators so brightness assertions don't drift
        // with the new pulse/drift defaults.
        ['pulse_depth', 0.0],
        ['amp_drift_depth', 0.0],
      ],
      dumpName: 'soft_glow_visible',
    });
    expect(frame.success).toBe(true);
    // Some pixels should be lit (non-black). Don't pin specific bars
    // since blob positions are seeded but the placement isn't trivial
    // to recompute here.
    frame.expectCoverage((c) => c.r > 30 || c.g > 30 || c.b > 30, { min: 0.05 });
  });

  it('warm hue range produces warm pixels (red/orange dominant)', async () => {
    const frame = await runGpuEffectTest({
      module: 'source.light.soft_glow',
      bundle: 'lights',
      width: 128, height: 128,
      inputColor: [0.0, 0.0, 0.0, 1.0],
      params: [
        ['blob_count', 16],
        ['blob_size', 0.5],
        ['intensity', 1.0],
        // Pin to a tight warm band (deep red ↔ orange).
        ['hue',  0.04],
        ['hue_shift', 0.0],
        ['saturation', 1.0],
        ['white_point', 3.0],   // crank white_point so we never crush to white
        ['ramp_curve', 0.0],
        ['pulse_depth', 0.0],
        ['amp_drift_depth', 0.0],
      ],
      dumpName: 'soft_glow_warm',
    });
    expect(frame.success).toBe(true);

    // Of pixels that are bright enough to register, the red channel
    // should dominate (R > G > B is the expected ordering for warm
    // hues across [0, 0.08]).
    const litPixels: { r: number; g: number; b: number }[] = [];
    frame.forEachPixel((c) => {
      if (c.r > 50 || c.g > 50 || c.b > 50) litPixels.push(c);
    });
    expect(litPixels.length).toBeGreaterThan(50);
    // At least 80% of lit pixels should be R-dominant.
    const rDominant = litPixels.filter(p => p.r >= p.g && p.r >= p.b);
    expect(rDominant.length / litPixels.length).toBeGreaterThan(0.8);
  });

  it('motion vectors flow through render_outputs rail into motion_blur', async () => {
    if (skipEngine) return;
    // Chain: solid bg → soft_glow → motion_blur. With the canonical
    // render_outputs rail wired, soft_glow's per-blob velocity reaches
    // motion_blur and produces directional smearing of the glow. Without
    // the rail, motion_blur falls back to a pass-through copy.
    //
    // We assert the two final frames differ — confirming both that
    // soft_glow IS publishing motion vectors and that they actually
    // reshape the downstream output. blob_count=4 + large blob_size
    // keeps the lit footprint big enough to make smearing visible at
    // the test viewport.
    // Wire model: motion_blur's render_outputs input auto-connects to the
    // soft_glow producer above. Negative case omits soft_glow → no producer →
    // motion_blur falls back to a copy of the black background.
    const buildSoftGlowChain = (withProducer: boolean): Sketch => ({
      anchor: null,
      wires: [],
      chain: [
        {
          type: 'module',
          module_type: 'source.solid_color',
          instance_key: 'bg@0',
          params: { color: [0.0, 0.0, 0.0] },
        },
        ...(withProducer ? [{
          type: 'module',
          module_type: 'source.light.soft_glow',
          instance_key: 'glow@0',
          params: {
            blob_count: 4,
            blob_size: 0.5,
            blob_size_jitter: 0.0,
            intensity: 1.5,
            drift_rate: 0.8,
            motion_strength: 8.0,
            motion_skew: 0.0,
            hue: 0.0, hue_shift: 0.0, saturation: 1.0,
            ramp_curve: 0.0, white_point: 1.0,
            pulse_depth: 0.0, amp_drift_depth: 0.0,
            seed: 17,
          },
        }] : []),
        {
          type: 'module',
          module_type: 'motion.blur',
          instance_key: 'blur@0',
          params: { strength: 32.0, samples: 16, quality: 1 },
        },
      ],
    } as Sketch);

    const withProducer = await runEngineTest({
      width: 128, height: 128,
      modules: ['com.nano.lights', 'com.nano.core'],
      commands: [
        {
          type: 'createSketch',
          sketchId: 'glow_with_producer',
          sketch: buildSoftGlowChain(true),
        },
        { type: 'setTracePoints', tracePoints: [
          { id: 'out', target: { type: 'sketch_output', sketchId: 'glow_with_producer' } },
        ]},
      ],
      waitFrames: 30,
      captureTraceIds: ['out'],
      dumpName: 'soft_glow_motion_with_producer',
    });
    expect(withProducer.success).toBe(true);

    const noProducer = await runEngineTest({
      width: 128, height: 128,
      modules: ['com.nano.lights', 'com.nano.core'],
      commands: [
        {
          type: 'createSketch',
          sketchId: 'glow_no_producer',
          sketch: buildSoftGlowChain(false),
        },
        { type: 'setTracePoints', tracePoints: [
          { id: 'out', target: { type: 'sketch_output', sketchId: 'glow_no_producer' } },
        ]},
      ],
      waitFrames: 30,
      captureTraceIds: ['out'],
      dumpName: 'soft_glow_motion_no_producer',
    });
    expect(noProducer.success).toBe(true);

    // With soft_glow present, it emits motion vectors that auto-connect into
    // motion_blur and smear the bloom; with no producer the blur copies the
    // black background. The frames must differ by a non-trivial pixel count.
    withProducer.trace('out').expectDifferentFrom(noProducer.trace('out'), 100);
  });

  it('motion_skew=1 (wavefront-only) differs from isotropic motion_skew=0', async () => {
    if (skipEngine) return;
    // Same chain run twice — only `motion_skew` changes. Skew biases
    // motion-vector emission toward each blob's leading edge, so the
    // downstream motion_blur should smear asymmetrically vs. the
    // isotropic baseline. Asserting a frame-difference confirms the
    // skew param IS feeding through (not silently ignored).
    // Both ends connected via auto-connect (soft_glow above motion_blur); only
    // motion_skew differs between the two runs.
    const buildSkewChain = (skew: number): Sketch => ({
      anchor: null,
      wires: [],
      chain: [
        {
          type: 'module', module_type: 'source.solid_color', instance_key: 'bg@0',
          params: { color: [0.0, 0.0, 0.0] },
        },
        {
          type: 'module', module_type: 'source.light.soft_glow', instance_key: 'glow@0',
          params: {
            blob_count: 4, blob_size: 0.5, blob_size_jitter: 0.0,
            intensity: 1.5, drift_rate: 0.8,
            motion_strength: 8.0, motion_skew: skew,
            hue: 0.0, hue_shift: 0.0, saturation: 1.0,
            ramp_curve: 0.0, white_point: 1.0,
            pulse_depth: 0.0, amp_drift_depth: 0.0,
            seed: 23,
          },
        },
        {
          type: 'module', module_type: 'motion.blur', instance_key: 'blur@0',
          params: { strength: 32.0, samples: 16, quality: 1 },
        },
      ],
    } as Sketch);

    const isotropic = await runEngineTest({
      width: 128, height: 128,
      modules: ['com.nano.lights', 'com.nano.core'],
      commands: [
        { type: 'createSketch', sketchId: 'iso', sketch: buildSkewChain(0.0) },
        { type: 'setTracePoints', tracePoints: [
          { id: 'out', target: { type: 'sketch_output', sketchId: 'iso' } },
        ]},
      ],
      waitFrames: 30,
      captureTraceIds: ['out'],
      dumpName: 'soft_glow_motion_skew_0',
    });
    expect(isotropic.success).toBe(true);

    const wavefront = await runEngineTest({
      width: 128, height: 128,
      modules: ['com.nano.lights', 'com.nano.core'],
      commands: [
        { type: 'createSketch', sketchId: 'wf', sketch: buildSkewChain(1.0) },
        { type: 'setTracePoints', tracePoints: [
          { id: 'out', target: { type: 'sketch_output', sketchId: 'wf' } },
        ]},
      ],
      waitFrames: 30,
      captureTraceIds: ['out'],
      dumpName: 'soft_glow_motion_skew_1',
    });
    expect(wavefront.success).toBe(true);

    isotropic.trace('out').expectDifferentFrom(wavefront.trace('out'), 50);
  });

  it('hue_shift=0 pins the glow to a single hue band', async () => {
    // Lock hue around 0.33 (green). Expect green-dominant lit pixels.
    const frame = await runGpuEffectTest({
      module: 'source.light.soft_glow',
      bundle: 'lights',
      width: 128, height: 128,
      inputColor: [0.0, 0.0, 0.0, 1.0],
      params: [
        ['blob_count', 16],
        ['blob_size', 0.4],
        ['intensity', 1.0],
        ['hue',  0.33],
        ['hue_shift', 0.0],
        ['saturation', 1.0],
        ['white_point', 3.0],
        ['ramp_curve', 0.0],
        ['pulse_depth', 0.0],
        ['amp_drift_depth', 0.0],
      ],
      dumpName: 'soft_glow_green',
    });
    expect(frame.success).toBe(true);

    const litPixels: { r: number; g: number; b: number }[] = [];
    frame.forEachPixel((c) => {
      if (c.r > 50 || c.g > 50 || c.b > 50) litPixels.push(c);
    });
    expect(litPixels.length).toBeGreaterThan(50);
    const gDominant = litPixels.filter(p => p.g >= p.r && p.g >= p.b);
    expect(gDominant.length / litPixels.length).toBeGreaterThan(0.8);
  });
});
});
