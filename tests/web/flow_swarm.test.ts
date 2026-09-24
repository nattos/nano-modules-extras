import { runEngineTest, runEngineMultiPhaseTest } from '@nano/web/test/engine-test-helpers';
import type { Sketch } from '@nano/web/src/sketch-types';

/**
 * E2E coverage for source.particles.flow_swarm (nano bundle) + the `flow_field` struct
 * handoff. flow_swarm consumes a velocity field (produced here by phase_fold)
 * and advects a GPU particle pool along it — connected via STRUCT AUTO-CONNECT
 * (flow_swarm's unwired flow_field_in binds the phase_fold producer above it),
 * not an explicit rail.
 *
 * Chain: phase_fold (produces flow_field) → flow_swarm (flow_field_in).
 *
 * Under test:
 *  1. The field transports end-to-end: a swarm over the auto-connected flow
 *     differs from the bare phase_fold portrait (the swarm painted over it).
 *  2. The flow drives the swarm: a chain WITH the phase_fold producer differs
 *     from one WITHOUT — with flow the particles advect, without they sit at
 *     their seed positions.
 *  3. The swarm is live: its output drifts across frames as particles flow.
 *  4. No-producer fallback: with no flow producer the swarm still renders.
 */

// phase_fold cell with a clear limit cycle, flow clock frozen for determinism.
const PF: Record<string, unknown> = {
  eccentricity: 0.5, lobedness: 0.3, flow_speed: 0.0,
  show_streamlines: false, show_limit_cycle: false,
};

// Swarm isolated over a black backdrop (input_alpha=0) so assertions key on
// the particles themselves. Pure flow motion (no jitter/drag), captured color.
// Gaussian shape at a visible size (size is now a quadratic [0,1] slider).
const SWARM: Record<string, unknown> = {
  count: 3000, mode: 0 /* Velocity */, shape_kind: 1 /* Gaussian */, size: 0.8,
  speed: 4.0, momentum: 0.0, jitter: 0.0, drag: 0.0, life: 6.0, life_jitter: 0.2,
  color_blend: 0.0, blend_mode: 0 /* Add */, opacity: 1.0, input_alpha: 0.0,
  seed: 1,
};

// Wire model: flow_swarm's flow_field_in auto-connects to the phase_fold
// producer above it. `withFlow` now gates the producer's PRESENCE — with it the
// swarm advects along the auto-connected field; without it (no producer above)
// the swarm falls back to its seed positions (zero field).
function buildChain(withFlow: boolean, swarm: Record<string, unknown> = {},
                   pf: Record<string, unknown> = {}): Sketch {
  const chain: any[] = [];
  if (withFlow) {
    chain.push({
      type: 'module',
      module_type: 'source.phase_fold',
      instance_key: 'pf@0',
      params: { ...PF, ...pf },
    });
  }
  chain.push({
    type: 'module',
    module_type: 'source.particles.flow_swarm',
    instance_key: 'sw@0',
    params: { ...SWARM, ...swarm },
  });
  return {
    anchor: null,
    wires: [],
    chain,
  };
}

// Bare phase_fold portrait (no swarm) — the comparison baseline for test 1.
function buildGeneratorOnly(): Sketch {
  return {
    anchor: null,
    chain: [
      { type: 'module', module_type: 'source.phase_fold', instance_key: 'pf@0', params: PF },
    ],
  };
}

const isActive = (c: { r: number; g: number; b: number }) => c.r + c.g + c.b > 24;

describe('source.particles.flow_swarm + flow_field auto-connect E2E', () => {
  jest.setTimeout(60000);

  it('renders a swarm over the flow_field (wired) distinct from the bare portrait', async () => {
    const swarm = await runEngineTest({
      width: 96, height: 96,
      modules: ['com.nano.testonly', 'com.nano.nano'],
      commands: [
        { type: 'createSketch', sketchId: 'fs_wired', sketch: buildChain(true) },
        { type: 'setTracePoints', tracePoints: [
          { id: 'out', target: { type: 'sketch_output', sketchId: 'fs_wired' } },
        ]},
      ],
      waitFrames: 10,
      captureTraceIds: ['out'],
      dumpName: 'flow_swarm_wired',
    });
    expect(swarm.success).toBe(true);

    // With input_alpha=0 the backdrop is black, so any lit pixels are
    // particles. A healthy 3000-particle swarm covers a meaningful area.
    const active = swarm.trace('out').countPixels(isActive);
    expect(active).toBeGreaterThan(150);

    // And it must NOT look like the bare phase_fold portrait.
    const gen = await runEngineTest({
      width: 96, height: 96,
      modules: ['com.nano.testonly', 'com.nano.nano'],
      commands: [
        { type: 'createSketch', sketchId: 'fs_gen', sketch: buildGeneratorOnly() },
        { type: 'setTracePoints', tracePoints: [
          { id: 'out', target: { type: 'sketch_output', sketchId: 'fs_gen' } },
        ]},
      ],
      waitFrames: 10,
      captureTraceIds: ['out'],
      dumpName: 'flow_swarm_generator',
    });
    expect(gen.success).toBe(true);
    swarm.trace('out').expectDifferentFrom(gen.trace('out'), 100);
  });

  it('the auto-connected flow drives the swarm (vs no-producer fallback)', async () => {
    // The ONLY difference is whether a phase_fold producer sits above the swarm.
    // With it, flow_swarm's flow_field_in auto-connects and the particles advect
    // along the field; without it (no producer) they fall back to a zero field
    // and sit at their seed positions.
    const withFlow = await runEngineTest({
      width: 96, height: 96,
      modules: ['com.nano.testonly', 'com.nano.nano'],
      commands: [
        { type: 'createSketch', sketchId: 'fs_on', sketch: buildChain(true) },
        { type: 'setTracePoints', tracePoints: [
          { id: 'out', target: { type: 'sketch_output', sketchId: 'fs_on' } },
        ]},
      ],
      waitFrames: 24,
      captureTraceIds: ['out'],
      dumpName: 'flow_swarm_rail_on',
    });
    expect(withFlow.success).toBe(true);

    const noFlow = await runEngineTest({
      width: 96, height: 96,
      modules: ['com.nano.testonly', 'com.nano.nano'],
      commands: [
        { type: 'createSketch', sketchId: 'fs_off', sketch: buildChain(false) },
        { type: 'setTracePoints', tracePoints: [
          { id: 'out', target: { type: 'sketch_output', sketchId: 'fs_off' } },
        ]},
      ],
      waitFrames: 24,
      captureTraceIds: ['out'],
      dumpName: 'flow_swarm_rail_off',
    });
    // Fallback path must still render cleanly (no crash, particles present).
    expect(noFlow.success).toBe(true);
    expect(noFlow.trace('out').countPixels(isActive)).toBeGreaterThan(100);

    // The flow moved the particles → the two frames diverge.
    withFlow.trace('out').expectDifferentFrom(noFlow.trace('out'), 60);
  });

  it('the swarm is live — output drifts across frames', async () => {
    const r = await runEngineMultiPhaseTest({
      width: 96, height: 96,
      modules: ['com.nano.testonly', 'com.nano.nano'],
      dumpName: 'flow_swarm_drift',
      phases: [
        {
          commands: [
            { type: 'createSketch', sketchId: 'fs_drift', sketch: buildChain(true) },
            { type: 'setTracePoints', tracePoints: [
              { id: 'out', target: { type: 'sketch_output', sketchId: 'fs_drift' } },
            ]},
          ],
          waitFrames: 6, captureTraceIds: ['out'],
        },
        { waitFrames: 30, captureTraceIds: ['out'] },
      ],
    });
    expect(r.success).toBe(true);
    r.phases[1].trace('out').expectDifferentFrom(r.phases[0].trace('out'), 40);
  });

  it('phase_fold still feeds the swarm at opacity 0 (flow bake runs in tick)', async () => {
    // phase_fold invisible (__opacity__=0 → render skipped) but used purely as a
    // flow source. The bake moved to tick(), so flow is still produced; the
    // swarm advects on it. Solid white particles so they're visible regardless
    // of the (passthrough/black) input the invisible phase_fold leaves.
    const sw = {
      count: 3000, shape_kind: 1, size: 0.8, speed: 4.0, momentum: 0.0,
      color_blend: 1.0, solid_color: [1, 1, 1], blend_mode: 0, opacity: 1.0,
      input_alpha: 0.0, seed: 3,
    };
    const run = (id: string, withFlow: boolean) => runEngineTest({
      width: 96, height: 96,
      modules: ['com.nano.testonly', 'com.nano.nano'],
      commands: [
        { type: 'createSketch', sketchId: id,
          sketch: buildChain(withFlow, sw, { __opacity__: 0 }) },
        { type: 'setTracePoints', tracePoints: [
          { id: 'out', target: { type: 'sketch_output', sketchId: id } },
        ]},
      ],
      waitFrames: 24, captureTraceIds: ['out'], dumpName: id,
    });

    const withFlow = await run('fs_op0_on', true);
    const noFlow   = await run('fs_op0_off', false);
    expect(withFlow.success).toBe(true);
    expect(noFlow.success).toBe(true);
    // The swarm renders (phase_fold's own pixels are gone — opacity 0).
    expect(withFlow.trace('out').countPixels(isActive)).toBeGreaterThan(100);
    // Flow produced at opacity 0 drives motion → differs from the zero-field
    // fallback. If the bake hadn't moved to tick(), both would be identical.
    withFlow.trace('out').expectDifferentFrom(noFlow.trace('out'), 40);
  });

  it('jitter sprays along the flow (changes the live swarm)', async () => {
    // Jitter is now a forward spray (wobble on speed + slight direction). With a
    // moving, field-driven swarm it perturbs the trajectories → frame differs.
    const sw = {
      count: 3000, shape_kind: 1, size: 0.8, speed: 4.0, momentum: 0.2,
      color_blend: 1.0, solid_color: [1, 1, 1], blend_mode: 0, opacity: 1.0,
      input_alpha: 0.0, seed: 5,
    };
    const run = (id: string, jitter: number) => runEngineTest({
      width: 96, height: 96,
      modules: ['com.nano.testonly', 'com.nano.nano'],
      commands: [
        { type: 'createSketch', sketchId: id, sketch: buildChain(true, { ...sw, jitter }) },
        { type: 'setTracePoints', tracePoints: [
          { id: 'out', target: { type: 'sketch_output', sketchId: id } },
        ]},
      ],
      waitFrames: 24, captureTraceIds: ['out'], dumpName: id,
    });
    const off = await run('fs_jit_off', 0.0);
    const on  = await run('fs_jit_on', 0.8);
    expect(off.success).toBe(true);
    expect(on.success).toBe(true);
    expect(on.trace('out').countPixels(isActive)).toBeGreaterThan(100);
    on.trace('out').expectDifferentFrom(off.trace('out'), 40);
  });

  it('force mode advects the swarm (field as acceleration on a mass)', async () => {
    // mode=Force, light weight: particles integrate the field as acceleration.
    // Confirm the chain runs and the swarm is live (drifts across frames).
    const r = await runEngineMultiPhaseTest({
      width: 96, height: 96,
      modules: ['com.nano.testonly', 'com.nano.nano'],
      dumpName: 'flow_swarm_force',
      phases: [
        {
          commands: [
            { type: 'createSketch', sketchId: 'fs_force',
              sketch: buildChain(true, { mode: 1, weight: 0.5, drag: 0.2 }) },
            { type: 'setTracePoints', tracePoints: [
              { id: 'out', target: { type: 'sketch_output', sketchId: 'fs_force' } },
            ]},
          ],
          waitFrames: 8, captureTraceIds: ['out'],
        },
        { waitFrames: 30, captureTraceIds: ['out'] },
      ],
    });
    expect(r.success).toBe(true);
    expect(r.phases[1].trace('out').countPixels(isActive)).toBeGreaterThan(100);
    r.phases[1].trace('out').expectDifferentFrom(r.phases[0].trace('out'), 30);
  });

  it('substeps refine the integration (force mode, 1 vs 8 substeps differ)', async () => {
    // Force mode + fast flow overshoots with one big step; substepping
    // re-samples the field along the path, so the trajectory (and frame) differ.
    const force = { mode: 1, weight: 0.4, drag: 0.1, speed: 6.0, momentum: 0.0 };
    const run = (id: string, substeps: number) => runEngineTest({
      width: 96, height: 96,
      modules: ['com.nano.testonly', 'com.nano.nano'],
      commands: [
        { type: 'createSketch', sketchId: id, sketch: buildChain(true, { ...force, substeps }) },
        { type: 'setTracePoints', tracePoints: [
          { id: 'out', target: { type: 'sketch_output', sketchId: id } },
        ]},
      ],
      waitFrames: 24, captureTraceIds: ['out'], dumpName: id,
    });
    const one  = await run('fs_sub1', 1);
    const many = await run('fs_sub8', 8);
    expect(one.success).toBe(true);
    expect(many.success).toBe(true);
    expect(many.trace('out').countPixels(isActive)).toBeGreaterThan(100);
    many.trace('out').expectDifferentFrom(one.trace('out'), 40);
  });

  it('pull settles the swarm onto the field (force mode, pull on vs off)', async () => {
    // Force mode lets particles overshoot the stable zone. `pull` bleeds their
    // velocity back toward the field flow each frame, settling them onto the
    // limit cycle — so the dynamics (and the resulting frame) clearly differ.
    const force = { mode: 1, weight: 0.4, drag: 0.1, speed: 5.0 };
    const run = (id: string, pull: number) => runEngineTest({
      width: 96, height: 96,
      modules: ['com.nano.testonly', 'com.nano.nano'],
      commands: [
        { type: 'createSketch', sketchId: id, sketch: buildChain(true, { ...force, pull }) },
        { type: 'setTracePoints', tracePoints: [
          { id: 'out', target: { type: 'sketch_output', sketchId: id } },
        ]},
      ],
      waitFrames: 28, captureTraceIds: ['out'], dumpName: id,
    });

    const free = await run('fs_pull_off', 0.0);
    const glued = await run('fs_pull_on', 1.0);
    expect(free.success).toBe(true);
    expect(glued.success).toBe(true);
    // Both render a live swarm; the pull changes where particles end up.
    expect(glued.trace('out').countPixels(isActive)).toBeGreaterThan(100);
    glued.trace('out').expectDifferentFrom(free.trace('out'), 50);
  });

  it('undertow changes the look (depth-gated tint + reversed flow)', async () => {
    // split=0 → no undertow (portrait-colored particles flowing forward).
    // split=1 → all particles undertow: blue tint + reversed/curled motion.
    const UNDERTOW = {
      undertow_polarity: -1.0, undertow_curl: 1.0,
      undertow_tint: [0.1, 0.4, 1.0], undertow_alpha: 1.0,
      color_blend: 0.0,
    };
    const off = await runEngineTest({
      width: 96, height: 96,
      modules: ['com.nano.testonly', 'com.nano.nano'],
      commands: [
        { type: 'createSketch', sketchId: 'fs_ut_off',
          sketch: buildChain(true, { ...UNDERTOW, undertow_split: 0.0 }) },
        { type: 'setTracePoints', tracePoints: [
          { id: 'out', target: { type: 'sketch_output', sketchId: 'fs_ut_off' } },
        ]},
      ],
      waitFrames: 20, captureTraceIds: ['out'], dumpName: 'flow_swarm_ut_off',
    });
    expect(off.success).toBe(true);

    const on = await runEngineTest({
      width: 96, height: 96,
      modules: ['com.nano.testonly', 'com.nano.nano'],
      commands: [
        { type: 'createSketch', sketchId: 'fs_ut_on',
          sketch: buildChain(true, { ...UNDERTOW, undertow_split: 1.0 }) },
        { type: 'setTracePoints', tracePoints: [
          { id: 'out', target: { type: 'sketch_output', sketchId: 'fs_ut_on' } },
        ]},
      ],
      waitFrames: 20, captureTraceIds: ['out'], dumpName: 'flow_swarm_ut_on',
    });
    expect(on.success).toBe(true);
    expect(on.trace('out').countPixels(isActive)).toBeGreaterThan(100);

    // Tint (blue) + reversed/curled motion → a clearly different frame.
    on.trace('out').expectDifferentFrom(off.trace('out'), 80);
    // The undertow tint is blue — the "on" frame should carry more blue-dominant
    // pixels than the "off" frame (which is portrait-colored).
    const blueish = (c: { r: number; g: number; b: number }) =>
      c.b > 80 && c.b > c.r + 20 && c.b > c.g + 10;
    expect(on.trace('out').countPixels(blueish))
      .toBeGreaterThan(off.trace('out').countPixels(blueish) + 20);
  });

  // Pull glues particles onto the limit cycle → they crowd there, giving the
  // interactions something to act on.
  const CROWD = {
    count: 4000, mode: 0, pull: 1.0, speed: 3.0, shape_kind: 1, size: 0.8,
    color_blend: 0.0, input_alpha: 0.0, blend_mode: 0, opacity: 1.0, seed: 2,
    interactions: true, interaction_radius: 0.03,
  };

  const runChain = (id: string, swarm: Record<string, unknown>, frames: number) =>
    runEngineTest({
      width: 96, height: 96,
      modules: ['com.nano.testonly', 'com.nano.nano'],
      commands: [
        { type: 'createSketch', sketchId: id, sketch: buildChain(true, swarm) },
        { type: 'setTracePoints', tracePoints: [
          { id: 'out', target: { type: 'sketch_output', sketchId: id } },
        ]},
      ],
      waitFrames: frames, captureTraceIds: ['out'], dumpName: id,
    });

  it('avoid_noise scatters particles where avoidance goes flat', async () => {
    // avoid_noise lives inside the avoidance mechanism, so it needs avoid on.
    // It adds a random kick that breaks up flat-gradient clumps.
    const base = { ...CROWD, avoid: 1.0 };
    const off = await runChain('fs_noise_off', { ...base, avoid_noise: 0.0 }, 24);
    const on  = await runChain('fs_noise_on',  { ...base, avoid_noise: 0.6 }, 24);
    expect(off.success).toBe(true);
    expect(on.success).toBe(true);
    expect(on.trace('out').countPixels(isActive)).toBeGreaterThan(80);
    on.trace('out').expectDifferentFrom(off.trace('out'), 40);
  });

  it('density debug renders at a non-square viewport (aspect path)', async () => {
    // Exercises the aspect-corrected splat/gradient on a wide viewport (the
    // density map used to look squashed; now the splat is round in pixels).
    const r = await runEngineTest({
      width: 160, height: 90,
      modules: ['com.nano.testonly', 'com.nano.nano'],
      commands: [
        { type: 'createSketch', sketchId: 'fs_aspect',
          sketch: buildChain(true, { ...CROWD, debug_density: true, interaction_radius: 0.04 }) },
        { type: 'setTracePoints', tracePoints: [
          { id: 'out', target: { type: 'sketch_output', sketchId: 'fs_aspect' } },
        ]},
      ],
      waitFrames: 20, captureTraceIds: ['out'], dumpName: 'fs_aspect',
    });
    expect(r.success).toBe(true);
    expect(r.trace('out').countPixels(isActive)).toBeGreaterThan(200);
  });

  it('debug view renders the density buffer and reflects interaction_radius', async () => {
    // With the density buffer actually accumulating, the heat map is non-empty
    // and a bigger interaction_radius spreads/sums the halos → more coverage.
    // (This is the regression guard for the additive-blend alpha fix: a broken
    // splat leaves an empty buffer, so radius would have no effect.)
    const base = { ...CROWD, debug_density: true };
    const small = await runChain('fs_dbg_small', { ...base, interaction_radius: 0.008 }, 20);
    const large = await runChain('fs_dbg_large', { ...base, interaction_radius: 0.06 }, 20);
    expect(small.success).toBe(true);
    expect(large.success).toBe(true);
    const smallActive = small.trace('out').countPixels(isActive);
    const largeActive = large.trace('out').countPixels(isActive);
    expect(largeActive).toBeGreaterThan(200);
    expect(largeActive).toBeGreaterThan(smallActive + 100);
  });

  it('stream aligns vs diverges velocities (boids-style)', async () => {
    // A crowded swarm with velocity variety (jitter spray) + freedom to steer
    // (low momentum, no pull). stream=+1 aligns headings to the group mean,
    // stream=-1 amplifies divergence → the two frames clearly differ.
    const sw = {
      count: 4000, mode: 0, momentum: 0.3, pull: 0.0, speed: 3.0,
      shape_kind: 1, size: 0.8, color_blend: 0.0, input_alpha: 0.0,
      blend_mode: 0, opacity: 1.0, seed: 4, jitter: 0.3,
      interactions: true, interaction_radius: 0.04, stream_density: 2.0,
    };
    const align   = await runChain('fs_align',   { ...sw, stream: 1.0 }, 28);
    const diverge = await runChain('fs_diverge', { ...sw, stream: -1.0 }, 28);
    expect(align.success).toBe(true);
    expect(diverge.success).toBe(true);
    expect(align.trace('out').countPixels(isActive)).toBeGreaterThan(100);
    align.trace('out').expectDifferentFrom(diverge.trace('out'), 40);
  });

  it('density death thins crowded regions (interactions)', async () => {
    // Same crowding setup; death culls particles where the density buffer says
    // it's crowded (they respawn elsewhere) → the frame changes vs death off.
    const off = await runChain('fs_death_off', { ...CROWD, density_death: 0.0 }, 28);
    const on  = await runChain('fs_death_on',
      { ...CROWD, density_death: 1.0, density_threshold: 1.0 }, 28);
    expect(off.success).toBe(true);
    expect(on.success).toBe(true);
    expect(on.trace('out').countPixels(isActive)).toBeGreaterThan(80);
    on.trace('out').expectDifferentFrom(off.trace('out'), 40);
  });

  it('avoid/curl pushes particles apart (interactions)', async () => {
    // Avoidance reads the density gradient and pushes particles down it (curl
    // swirls the push) → the swarm spreads, changing the frame vs avoid off.
    const off = await runChain('fs_avoid_off', { ...CROWD, avoid: 0.0 }, 28);
    const on  = await runChain('fs_avoid_on', { ...CROWD, avoid: 1.0, avoid_curl: 0.5 }, 28);
    expect(off.success).toBe(true);
    expect(on.success).toBe(true);
    expect(on.trace('out').countPixels(isActive)).toBeGreaterThan(80);
    on.trace('out').expectDifferentFrom(off.trace('out'), 40);
  });
});
