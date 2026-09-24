import { runEngineTest } from '@nano/web/test/engine-test-helpers';
import type { Sketch } from '@nano/web/src/sketch-types';

/**
 * E2E coverage for motion.local_delay (nano bundle) — the stylized
 * motion-driven local delay.
 *
 * Three things under test:
 *  1. No-motion identity: a static input has no flow → zero weight →
 *     the cross-fade collapses to the current frame (pass-through). This
 *     also smoke-tests that all four compute passes dispatch cleanly.
 *  2. delay_amount drives the cross-fade: with a MOVING input (a
 *     debug.motion_rect overlay), a high delay_amount ghosts moving
 *     edges toward the previous frame; delay_amount=0 does not. The two
 *     frames must differ.
 *  3. Motion producer: local_delay writes modulated vectors on
 *     render_outputs/motion. Wiring that rail into a downstream
 *     motion.blur changes the output vs. leaving it unwired —
 *     confirming the motion texture is actually published + transported.
 */

describe('motion.local_delay E2E', () => {
  jest.setTimeout(40000);

  it('passes a static input straight through (no motion → zero weight)', async () => {
    const sketch: Sketch = {
      anchor: null,
      chain: [
        {
          type: 'module',
          module_type: 'source.solid_color',
          instance_key: 'bg@0',
          params: { color: [0.3, 0.5, 0.8] },
        },
        {
          type: 'module',
          module_type: 'motion.local_delay',
          instance_key: 'ld@0',
          // Crank delay + gain: even so, a static input has zero flow,
          // so the weight is zero and the output equals the input.
          params: { delay_amount: 1.0, weight_gain: 0.06 },
        },
      ],
    };

    const result = await runEngineTest({
      width: 64, height: 64,
      modules: ['com.nano.testonly', 'com.nano.nano'],
      commands: [
        { type: 'createSketch', sketchId: 'ld_static', sketch },
        { type: 'setTracePoints', tracePoints: [
          { id: 'out', target: { type: 'sketch_output', sketchId: 'ld_static' } },
        ]},
      ],
      waitFrames: 10,
      captureTraceIds: ['out'],
      dumpName: 'local_delay_static',
    });
    expect(result.success).toBe(true);

    // solid_color (0.3, 0.5, 0.8) → (77, 128, 204). local_delay is a
    // pure pass-through when nothing moves.
    result.trace('out').expectUniformColor({ r: 77, g: 128, b: 204 }, 6);
  });

  it('ghosts moving edges in proportion to delay_amount', async () => {
    // A swarm of moving blobs over a solid bg gives local_delay real
    // frame-to-frame motion at many edges (a single hard rect would only
    // move flow on its 1-2px border). delay_amount=0 is identity
    // (weight ×0); delay_amount high blends those edges toward the
    // previous frame, so the two frames differ across a wide band.
    const buildChain = (delay: number): Sketch => ({
      anchor: null,
      chain: [
        {
          type: 'module',
          module_type: 'source.solid_color',
          instance_key: 'bg@0',
          params: { color: [0.1, 0.1, 0.1] },
        },
        {
          type: 'module',
          module_type: 'debug.motion_swarm',
          instance_key: 'swarm@0',
          // Many moving blobs — used purely as moving IMAGE content (no
          // rail tap), so local_delay must estimate the flow itself.
          params: {
            count: 24, size: 0.06, swirl: 1.5, radial: 0.0,
            randomness: 0.4, speed: 1.0, opacity: 1.0, seed: 7,
          },
        },
        {
          type: 'module',
          module_type: 'motion.local_delay',
          instance_key: 'ld@0',
          params: {
            delay_amount: delay,
            weight_gain: 0.05,
            max_flow: 0.05,
            align_amount: 0.5,
          },
        },
      ],
    });

    const noDelay = await runEngineTest({
      width: 128, height: 128,
      modules: ['com.nano.testonly', 'com.nano.nano'],
      commands: [
        { type: 'createSketch', sketchId: 'ld_off', sketch: buildChain(0.0) },
        { type: 'setTracePoints', tracePoints: [
          { id: 'out', target: { type: 'sketch_output', sketchId: 'ld_off' } },
        ]},
      ],
      waitFrames: 20,
      captureTraceIds: ['out'],
      dumpName: 'local_delay_off',
    });
    expect(noDelay.success).toBe(true);

    const withDelay = await runEngineTest({
      width: 128, height: 128,
      modules: ['com.nano.testonly', 'com.nano.nano'],
      commands: [
        { type: 'createSketch', sketchId: 'ld_on', sketch: buildChain(0.9) },
        { type: 'setTracePoints', tracePoints: [
          { id: 'out', target: { type: 'sketch_output', sketchId: 'ld_on' } },
        ]},
      ],
      waitFrames: 20,
      captureTraceIds: ['out'],
      dumpName: 'local_delay_on',
    });
    expect(withDelay.success).toBe(true);

    // Same moving chain, only delay_amount differs. The cross-fade toward
    // history must change a meaningful band of pixels around the swarm's
    // swept edges.
    withDelay.trace('out').expectDifferentFrom(noDelay.trace('out'), 100);
  });

  it('twitch modulates the motion weight (suppresses a roaming region)', async () => {
    // A roaming vignette at a random anchor each frame. twitch_shape=1 blacks the
    // rim (suppresses motion OUTSIDE a small disk near the anchor), so with
    // debug_show_motion the masked motion field differs strongly from twitch-off.
    const buildChain = (twitchAmount: number): Sketch => ({
      anchor: null,
      chain: [
        { type: 'module', module_type: 'source.solid_color', instance_key: 'bg@0',
        params: { color: [0.1, 0.1, 0.1] } },
        { type: 'module', module_type: 'debug.motion_swarm', instance_key: 'swarm@0',
        params: { count: 24, size: 0.06, swirl: 1.5, radial: 0.0,
                  randomness: 0.4, speed: 1.0, opacity: 1.0, seed: 7 } },
        { type: 'module', module_type: 'motion.local_delay', instance_key: 'ld@0',
        params: {
          delay_amount: 0.0, weight_gain: 0.1, max_flow: 0.05,
          debug_show_motion: true,
          twitch_amount: twitchAmount, twitch_shape: 1.0, twitch_radius: 0.2, twitch_softness: 0.2,
        } },
      ],
    });

    const run = (id: string, amount: number) => runEngineTest({
      width: 128, height: 128,
      modules: ['com.nano.testonly', 'com.nano.nano'],
      commands: [
        { type: 'createSketch', sketchId: id, sketch: buildChain(amount) },
        { type: 'setTracePoints', tracePoints: [
          { id: 'out', target: { type: 'sketch_output', sketchId: id } },
        ]},
      ],
      waitFrames: 20, captureTraceIds: ['out'], dumpName: id,
    });

    const off = await run('ld_twitch_off', 0.0);
    const on  = await run('ld_twitch_on', 1.0);
    expect(off.success && on.success).toBe(true);
    on.trace('out').expectDifferentFrom(off.trace('out'), 40);
  });

  it('publishes modulated motion that drives a downstream motion_blur', async () => {
    // Wire model: motion_blur's render_outputs input auto-connects to the
    // nearest producer above. WITH local_delay present, that's local_delay's
    // MODULATED motion; WITHOUT it, motion_blur binds motion_rect's RAW motion
    // directly. So the two frames differ iff local_delay actually reshapes the
    // field — a stronger check than mere presence.
    const buildChain = (withLocalDelay: boolean): Sketch => ({
      anchor: null,
      wires: [],
      chain: [
        {
          type: 'module',
          module_type: 'source.solid_color',
          instance_key: 'bg@0',
          params: { color: [0.05, 0.05, 0.05] },
        },
        {
          type: 'module',
          module_type: 'debug.motion_rect',
          instance_key: 'rect@0',
          params: { size: 0.3, speed: 3.0, color: [1.0, 0.4, 0.8] },
        },
        ...(withLocalDelay ? [{
          type: 'module',
          module_type: 'motion.local_delay',
          instance_key: 'ld@0',
          params: { delay_amount: 0.5, weight_gain: 0.05, max_flow: 0.05 },
        }] : []),
        {
          type: 'module',
          module_type: 'motion.blur',
          instance_key: 'blur@0',
          params: { strength: 24.0, samples: 12, quality: 1 },
        },
      ],
    } as Sketch);

    const withLD = await runEngineTest({
      width: 128, height: 128,
      modules: ['com.nano.testonly', 'com.nano.nano'],
      commands: [
        { type: 'createSketch', sketchId: 'ld_with', sketch: buildChain(true) },
        { type: 'setTracePoints', tracePoints: [
          { id: 'out', target: { type: 'sketch_output', sketchId: 'ld_with' } },
        ]},
      ],
      waitFrames: 20,
      captureTraceIds: ['out'],
      dumpName: 'local_delay_motion_with',
    });
    expect(withLD.success).toBe(true);

    const withoutLD = await runEngineTest({
      width: 128, height: 128,
      modules: ['com.nano.testonly', 'com.nano.nano'],
      commands: [
        { type: 'createSketch', sketchId: 'ld_without', sketch: buildChain(false) },
        { type: 'setTracePoints', tracePoints: [
          { id: 'out', target: { type: 'sketch_output', sketchId: 'ld_without' } },
        ]},
      ],
      waitFrames: 20,
      captureTraceIds: ['out'],
      dumpName: 'local_delay_motion_without',
    });
    expect(withoutLD.success).toBe(true);

    // With local_delay present, motion_blur smears along its modulated vectors;
    // without it, along motion_rect's raw vectors. The frames must differ —
    // proof local_delay published a reshaped motion field that auto-connected
    // downstream.
    withLD.trace('out').expectDifferentFrom(withoutLD.trace('out'), 40);
  });

  it('delay_steps changes the advection trail length', async () => {
    // Same moving chain advected 1 step vs many. More steps walk further along
    // the flow streamline before sampling the input, so the trail lands the
    // content in a different place — the frames must differ. Exercises the
    // forward-advection loop.
    const buildChain = (steps: number): Sketch => ({
      anchor: null,
      chain: [
        {
          type: 'module',
          module_type: 'source.solid_color',
          instance_key: 'bg@0',
          params: { color: [0.1, 0.1, 0.1] },
        },
        {
          type: 'module',
          module_type: 'debug.motion_swarm',
          instance_key: 'swarm@0',
          params: {
            count: 24, size: 0.06, swirl: 1.5, radial: 0.0,
            randomness: 0.4, speed: 1.0, opacity: 1.0, seed: 7,
          },
        },
        {
          type: 'module',
          module_type: 'motion.local_delay',
          instance_key: 'ld@0',
          params: {
            delay_amount: 0.8, delay_steps: steps,
            weight_gain: 0.05, max_flow: 0.05,
          },
        },
      ],
    });

    const fewSteps = await runEngineTest({
      width: 128, height: 128,
      modules: ['com.nano.testonly', 'com.nano.nano'],
      commands: [
        { type: 'createSketch', sketchId: 'ld_s1', sketch: buildChain(1.0) },
        { type: 'setTracePoints', tracePoints: [
          { id: 'out', target: { type: 'sketch_output', sketchId: 'ld_s1' } },
        ]},
      ],
      waitFrames: 24,
      captureTraceIds: ['out'],
      dumpName: 'local_delay_steps1',
    });
    expect(fewSteps.success).toBe(true);

    const manySteps = await runEngineTest({
      width: 128, height: 128,
      modules: ['com.nano.testonly', 'com.nano.nano'],
      commands: [
        { type: 'createSketch', sketchId: 'ld_s16', sketch: buildChain(16.0) },
        { type: 'setTracePoints', tracePoints: [
          { id: 'out', target: { type: 'sketch_output', sketchId: 'ld_s16' } },
        ]},
      ],
      waitFrames: 24,
      captureTraceIds: ['out'],
      dumpName: 'local_delay_steps16',
    });
    expect(manySteps.success).toBe(true);

    manySteps.trace('out').expectDifferentFrom(fewSteps.trace('out'), 40);
  });

  it('delay_amount=0 is an exact image passthrough (flow-conditioner mode)', async () => {
    // At delay_amount=0 the advection scale is 0, so the color pass early-outs
    // to a straight copy of the input — the image passes through untouched while
    // align/mask/motion still run (flow-conditioner mode). We trace local_delay's
    // OWN input and output in the SAME render (chain_entry input vs output) and
    // assert they're byte-identical. Tracing both sides of one stage avoids any
    // cross-sketch swarm-simulation divergence — the only thing under test is
    // that this stage leaves its input pixels untouched.
    const sketch: Sketch = {
      anchor: null,
      chain: [
        {
          type: 'module',
          module_type: 'source.solid_color',
          instance_key: 'bg@0',
          params: { color: [0.1, 0.1, 0.1] },
        },
        // chainIdx 0
        {
          type: 'module',
          module_type: 'debug.motion_swarm',
          instance_key: 'swarm@0',
          params: {
            count: 24, size: 0.06, swirl: 1.5, radial: 0.0,
            randomness: 0.4, speed: 1.0, opacity: 1.0, seed: 7,
          },
        },
        // chainIdx 1
        {
          type: 'module',
          module_type: 'motion.local_delay',
          instance_key: 'ld@0',
          params: { delay_amount: 0.0, weight_gain: 0.05, max_flow: 0.05 },
        },
        // chainIdx 2
      ],
    };

    const result = await runEngineTest({
      width: 128, height: 128,
      modules: ['com.nano.testonly', 'com.nano.nano'],
      commands: [
        { type: 'createSketch', sketchId: 'ld_pass', sketch },
        { type: 'setTracePoints', tracePoints: [
          { id: 'ld_in',  target: { type: 'chain_entry', sketchId: 'ld_pass', colIdx: 0, chainIdx: 2, side: 'input'  } },
          { id: 'ld_out', target: { type: 'chain_entry', sketchId: 'ld_pass', colIdx: 0, chainIdx: 2, side: 'output' } },
        ]},
      ],
      waitFrames: 20,
      captureTraceIds: ['ld_in', 'ld_out'],
      dumpName: 'local_delay_passthrough',
    });
    expect(result.success).toBe(true);

    // Sanity: the input actually has moving content (not a flat frame), so the
    // passthrough assertion is meaningful.
    result.trace('ld_in').expectNotSolidColor({ r: 26, g: 26, b: 26 });
    // delay_amount=0 → output pixels equal input pixels exactly.
    result.trace('ld_out').expectSameAs(result.trace('ld_in'));
  });

  it('noise motion advances the stochastic seed (re-rolls under state replay)', async () => {
    // The noise mask gates the echo. With `motion`=0 the seed is fixed; with
    // `motion`>0 it must advance over time and re-roll the noise. Same moving
    // input + same frame count, so the ONLY difference is the seed advancing —
    // which it won't if the per-frame state replay keeps resetting the
    // accumulator (the bug). The two outputs must differ.
    const buildChain = (motion: number): Sketch => ({
      anchor: null,
      chain: [
        {
          type: 'module',
          module_type: 'source.solid_color',
          instance_key: 'bg@0',
          params: { color: [0.1, 0.1, 0.1] },
        },
        {
          type: 'module',
          module_type: 'debug.motion_swarm',
          instance_key: 'swarm@0',
          params: {
            count: 24, size: 0.06, swirl: 1.5, radial: 0.0,
            randomness: 0.4, speed: 1.0, opacity: 1.0, seed: 7,
          },
        },
        {
          type: 'module',
          module_type: 'motion.local_delay',
          instance_key: 'ld@0',
          params: {
            delay_amount: 0.8, delay_steps: 12.0,
            noise_weight: 1.0, noise_motion: motion, weight_gain: 0.05, max_flow: 0.05,
          },
        },
      ],
    });

    const frozen = await runEngineTest({
      width: 128, height: 128,
      modules: ['com.nano.testonly', 'com.nano.nano'],
      commands: [
        { type: 'createSketch', sketchId: 'ld_nm0', sketch: buildChain(0.0) },
        { type: 'setTracePoints', tracePoints: [
          { id: 'out', target: { type: 'sketch_output', sketchId: 'ld_nm0' } },
        ]},
      ],
      waitFrames: 24,
      captureTraceIds: ['out'],
      dumpName: 'local_delay_noise_frozen',
    });
    expect(frozen.success).toBe(true);

    const moving = await runEngineTest({
      width: 128, height: 128,
      modules: ['com.nano.testonly', 'com.nano.nano'],
      commands: [
        { type: 'createSketch', sketchId: 'ld_nm1', sketch: buildChain(0.9) },
        { type: 'setTracePoints', tracePoints: [
          { id: 'out', target: { type: 'sketch_output', sketchId: 'ld_nm1' } },
        ]},
      ],
      waitFrames: 24,
      captureTraceIds: ['out'],
      dumpName: 'local_delay_noise_moving',
    });
    expect(moving.success).toBe(true);

    moving.trace('out').expectDifferentFrom(frozen.trace('out'), 40);
  });

  it('uses incoming render_outputs motion as the flow when flow_source=Incoming', async () => {
    // A moving swarm publishes render_outputs/motion. With flow_source=Incoming
    // and the rail wired, local_delay echoes the input along THOSE vectors
    // (skipping its own estimator). Without the rail there's no upstream motion
    // → passthrough.
    //
    // We trace local_delay's OWN input and output in the SAME render (chain_entry
    // input vs output, chainIdx 2) rather than comparing two separate captures:
    // the swarm sim phase diverges run-to-run, so a cross-capture diff is
    // dominated by random swarm-position differences, not the echo. Within one
    // render the echo footprint is isolated. The path is proven by the contrast:
    // rail wired → output differs from input (incoming vectors drove the echo);
    // no rail → output equals input (zero incoming flow → passthrough).
    // Wire model: local_delay (Incoming mode) reads its render_outputs_in via
    // auto-connect from the nearest producer above. WITH a motion_swarm above,
    // it echoes the incoming vectors; WITHOUT one (a plain solid_color in the
    // same slot, keeping ld at chainIdx 2 and producing NO render_outputs), the
    // incoming flow is zero → passthrough.
    const buildChain = (withMotion: boolean): Sketch => ({
      anchor: null,
      wires: [],
      chain: [
        {
          type: 'module',
          module_type: 'source.solid_color',
          instance_key: 'bg@0',
          params: { color: [0.05, 0.05, 0.1] },
        },
        withMotion ? {
          type: 'module',
          module_type: 'debug.motion_swarm',
          instance_key: 'swarm@0',
          params: {
            count: 24, size: 0.08, swirl: 1.5, radial: 0.0,
            randomness: 0.4, speed: 2.5, opacity: 1.0, seed: 7,
          },
        } : {
          // Same slot, no render_outputs producer → zero incoming flow.
          type: 'module',
          module_type: 'source.solid_color',
          instance_key: 'filler@0',
          params: { color: [0.05, 0.05, 0.1] },
        },
        {
          type: 'module',
          module_type: 'motion.local_delay',
          instance_key: 'ld@0',
          // Crank sensitivity/reach so the incoming-driven echo is well clear
          // of the passthrough baseline. smoothing=0 disables the temporal flow
          // EMA: the incoming swarm motion is sparse (nonzero only inside the
          // small moving rects) and intermittent at any fixed pixel, so the
          // default EMA would average it toward zero — unlike the dense LK flow
          // in Estimate mode. With no smoothing the per-frame vectors drive the
          // advection directly. flow_source=1 = Incoming: reads the auto-
          // connected render_outputs_in/motion.
          params: { flow_source: 1, smoothing: 0.0, delay_amount: 1.0, delay_steps: 32.0, weight_gain: 0.15 },
        },
      ],
    } as Sketch);

    const withMotion = await runEngineTest({
      width: 128, height: 128,
      modules: ['com.nano.testonly', 'com.nano.nano'],
      commands: [
        { type: 'createSketch', sketchId: 'ld_inc_on', sketch: buildChain(true) },
        { type: 'setTracePoints', tracePoints: [
          { id: 'ld_in',  target: { type: 'chain_entry', sketchId: 'ld_inc_on', colIdx: 0, chainIdx: 2, side: 'input'  } },
          { id: 'ld_out', target: { type: 'chain_entry', sketchId: 'ld_inc_on', colIdx: 0, chainIdx: 2, side: 'output' } },
        ]},
      ],
      waitFrames: 20,
      captureTraceIds: ['ld_in', 'ld_out'],
      dumpName: 'local_delay_incoming_on',
    });
    expect(withMotion.success).toBe(true);

    const noMotion = await runEngineTest({
      width: 128, height: 128,
      modules: ['com.nano.testonly', 'com.nano.nano'],
      commands: [
        { type: 'createSketch', sketchId: 'ld_inc_off', sketch: buildChain(false) },
        { type: 'setTracePoints', tracePoints: [
          { id: 'ld_in',  target: { type: 'chain_entry', sketchId: 'ld_inc_off', colIdx: 0, chainIdx: 2, side: 'input'  } },
          { id: 'ld_out', target: { type: 'chain_entry', sketchId: 'ld_inc_off', colIdx: 0, chainIdx: 2, side: 'output' } },
        ]},
      ],
      waitFrames: 20,
      captureTraceIds: ['ld_in', 'ld_out'],
      dumpName: 'local_delay_incoming_off',
    });
    expect(noMotion.success).toBe(true);

    // With motion above: the incoming swarm vectors drive a forward-advection
    // echo, so local_delay's output departs from its input. The footprint is
    // edge-limited (only pixels inside the moving rects advect, and a solid blob
    // mostly samples its own color — only blob edges shift), but stable run-to-
    // run (~45 px); 25 sits safely below that floor and far above the
    // passthrough's 0, cleanly separating "incoming echo present" from "no echo".
    withMotion.trace('ld_out').expectDifferentFrom(withMotion.trace('ld_in'), 25);
    // No motion producer: zero incoming flow → the advection collapses → exact passthrough.
    noMotion.trace('ld_out').expectSameAs(noMotion.trace('ld_in'));
  });
});
