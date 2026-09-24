import { runGpuEffectTest, Frame, forEachBackend } from '@nano/web/test/gpu-test-helpers';
import { runEngineTest } from '@nano/web/test/engine-test-helpers';
import type { Sketch } from '@nano/web/src/sketch-types';

// Per-effect tests for source.mesh.three_walls — three neon frames rushing down
// a tunnel, rendered from three cameras into three texture outputs.
//
// What is tested WHERE, and why:
//
//   * The moves themselves (the pulse train, the rate ramps, the gates, and
//     above all the frame-locked advance) are pinned host-free by
//     native/tests/test_three_walls_show.cpp, which drives them at an exact dt.
//     Nothing here re-tests that arithmetic.
//   * The main output runs on BOTH backends with plain `ticks`. That works
//     because every bit of this effect's state is CPU-side — there is no
//     persistent GPU state — so N ticks then one render is identical to N
//     tick/render pairs, and `renderEachTick` (which the Metal single-module
//     path silently drops) is not needed.
//   * The two side views can only be read through a real sketch, so they use
//     the engine harness and the sidechannel trick from chroma_wave.test.ts.
//     Engine dt is wall clock, so those cases assert GEOMETRY, never timing.

forEachBackend((backend) => {
describe(`Three Walls E2E (${backend})`, () => {
  jest.setTimeout(90000);

  const W = 240, H = 160;
  const MODULE = 'source.mesh.three_walls';
  const BUNDLE = 'lights' as const;

  // A quiet grade, so the assertions are about the projection rather than the
  // analogue tail.
  const QUIET: [string, number][] = [['grain', 0], ['scanline', 0], ['chroma_bleed', 0]];

  const luma = (p: { r: number; g: number; b: number }) => (p.r + p.g + p.b) / 3;

  /** How many pixels carry real light. The blunt "is anything there" measure. */
  const litCount = (f: Frame, threshold = 40) => {
    let n = 0;
    f.forEachPixel((p) => { if (luma(p) > threshold) n++; });
    return n;
  };

  /** Half-width of the lit region, in pixels — how big the frame has grown. */
  const litHalfWidth = (f: Frame, threshold = 60) => {
    let lo = W, hi = -1;
    f.forEachPixel((p, x) => {
      if (luma(p) > threshold) { if (x < lo) lo = x; if (x > hi) hi = x; }
    });
    return hi < lo ? 0 : (hi - lo) / 2;
  };

  /** Is there a clearly cyan pixel? Frame 3's colour, and nothing else's. */
  const hasCyan = (f: Frame) => {
    let found = false;
    f.forEachPixel((p) => { if (p.b > 120 && p.g > 100 && p.r + 60 < p.b) found = true; });
    return found;
  };
  /** Is there a clearly magenta pixel? Frame 1's colour. */
  const hasMagenta = (f: Frame) => {
    let found = false;
    f.forEachPixel((p) => { if (p.r > 140 && p.g + 60 < p.r && p.b + 20 < p.r) found = true; });
    return found;
  };

  const run = (name: string, params: any[], ticks = 0) => runGpuEffectTest({
    module: MODULE, bundle: BUNDLE, width: W, height: H,
    inputColor: [0, 0, 0, 1],
    params: [...QUIET, ...params] as any,
    ticks,
    dumpName: name,
  });

  it('declares metadata', async () => {
    const f = await run('three_walls_metadata', []);
    expect(f.success).toBe(true);
    expect(f.metadata?.id).toBe(MODULE);
  });

  // The rest pose is NOTHING — the frames do not exist until a move puts them
  // there. That is the contract the whole card is built on, so it is the first
  // thing worth pinning.
  it('is black until a move fires', async () => {
    const idle = await run('three_walls_idle', [], 20);
    expect(idle.success).toBe(true);
    expect(litCount(idle)).toBeLessThan(20);

    const fired = await run('three_walls_fired', [['pulse', 1]], 20);
    expect(litCount(fired)).toBeGreaterThan(500);
  });

  it('a frame grows as it arrives', async () => {
    // One frame, launched alone, sampled twice on its way in. The tunnel maps
    // depth geometrically, so this is the check that the perspective divide is
    // actually happening rather than a fixed-size quad being drawn.
    const p = [['pulse', 1], ['pulse_time', 2.0], ['pulse_stagger', 5.0]];
    const early = await run('three_walls_grow_early', p, 8);
    const late  = await run('three_walls_grow_late', p, 40);
    expect(early.success && late.success).toBe(true);
    expect(litHalfWidth(early)).toBeGreaterThan(2);
    expect(litHalfWidth(late)).toBeGreaterThan(litHalfWidth(early) * 1.5);
  });

  it('Pulse leads with the highlight', async () => {
    // Frame 3 is cyan (the highlight) and frame 1 is magenta, so which one is
    // in front is readable straight off the colours. The train runs 3, 2, 1:
    // the bright one is the hit and the others are its tail, which only reads
    // that way if the hit gets there first.
    //
    // core_whiten off: the neon core deliberately blows to white, and a small
    // frame is nothing BUT core, so the hue only survives out in the halo.
    const p = [['pulse', 1], ['pulse_time', 1.2], ['pulse_stagger', 0.35],
               ['core_whiten', 0]];
    const first = await run('three_walls_train_first', p, 6);    // ~0.1 s
    const all   = await run('three_walls_train_all', p, 50);     // ~0.8 s
    expect(first.success && all.success).toBe(true);
    expect(hasCyan(first)).toBe(true);        // the highlight is out in front
    expect(hasMagenta(first)).toBe(false);    // and frame 1 has not left yet
    expect(hasMagenta(all)).toBe(true);       // by now it has
  });

  it('Pulse plays out and leaves the card black again', async () => {
    // 2*stagger + travel = 0.5 s, so 60 ticks at 16 ms is well past the end.
    const done = await run('three_walls_train_done',
                           [['pulse', 1], ['pulse_time', 0.3], ['pulse_stagger', 0.1]], 60);
    expect(done.success).toBe(true);
    expect(litCount(done)).toBeLessThan(20);
  });

  // A gate stays up, and puts all three frames in the room. Frozen at rate 0 so
  // the pose is exact: the point here is PRESENCE, and the case below covers
  // the motion.
  it('Resonate holds all three frames while the trigger is high', async () => {
    // Small frames, so all three of the start pose are still on the BACK wall
    // — at the default size the nearest has already overflowed onto the sides
    // and is by design not in this output at all.
    const f = await run('three_walls_resonate',
                        [['resonate', 1], ['resonate_f0', 0], ['resonate_f1', 0],
                         ['quad_size', 0.5], ['core_whiten', 0]], 30);
    expect(f.success).toBe(true);
    expect(hasMagenta(f)).toBe(true);
    expect(hasCyan(f)).toBe(true);
    expect(litCount(f)).toBeGreaterThan(1000);
  });

  // Also the check that a held event value does not re-arm the move every frame
  // off the executor's replay: a re-arming Resonate would snap back to its start
  // pose on every patch and these two frames would be identical.
  // The glow carries as much of the depth as the size does. A real tube has a
  // fixed thickness, so what reaches the eye scales with everything else: near,
  // a fat bar with a wide bloom; far, a hairline with almost none. Hold the
  // width constant and a frame coming at you reads as a flat rectangle being
  // scaled up rather than an object arriving.
  //
  // Measured as INK, not as core width. The core is one to three pixels at any
  // sane size, so a real change rounds away; the halo is where nearly all of
  // the cue actually lives, and it is what the eye reads.
  it('Depth Glow grows the neon as a frame arrives', async () => {
    const p: any[] = [['pulse', 1], ['pulse_time', 1.0], ['pulse_stagger', 5.0],
                      ['quad_size', 1.0], ['core_whiten', 0]];
    const BW = 480, BH = 320;
    const big = (name: string, extra: any[], ticks: number) => runGpuEffectTest({
      module: MODULE, bundle: BUNDLE, width: BW, height: BH,
      inputColor: [0, 0, 0, 1],
      params: [...QUIET, ...p, ...extra] as any, ticks, dumpName: name,
    });

    // The SAME tick with the scaling off is the control: identical geometry,
    // so the ratio between them is purely how much tube and bloom the frame
    // was given at that depth.
    const ratioAt = async (ticks: number, tag: string) => {
      const off = await big(`three_walls_dg_${tag}_off`, [['depth_scale', 0]], ticks);
      const on  = await big(`three_walls_dg_${tag}_on`,  [['depth_scale', 1]], ticks);
      expect(off.success && on.success).toBe(true);
      return litCount(on, 30) / Math.max(1, litCount(off, 30));
    };

    const far = await ratioAt(6, 'far');
    const near = await ratioAt(30, 'near');
    // Deep in the tunnel the frame is drawn with far less ink than its authored
    // width would give it; by the time it is on top of you it has most of it
    // back.
    expect(far).toBeLessThan(0.5);
    expect(near).toBeGreaterThan(far * 1.8);
  });

  it('the frames keep moving under a held gate', async () => {
    const p: any[] = [['resonate', 1], ['resonate_f0', 0.5], ['resonate_f1', 0.5]];
    const a = await run('three_walls_moving_a', p, 10);
    const b = await run('three_walls_moving_b', p, 40);
    expect(a.success && b.success).toBe(true);
    a.expectDifferentFrom(b, 60);
  });
});
});

// --- The walls --------------------------------------------------------------
//
// The three outputs are three WALLS of one room, and they PARTITION it: a
// frame's vertical edge is either still on the back wall or already on a side
// wall, never in two pictures at once. That partition is the contract this
// whole card is built on, so most of what follows is checking it holds.
//
// Only reachable through a real sketch: wire the side output into a
// util.sidechannel_out override and put a util.sidechannel_in after it, and the
// sketch output IS that texture (the trick from chroma_wave.test.ts:393).
// Engine dt is wall clock, so every case here freezes the pose with rate 0 and
// asserts geometry — nothing rides on timing.
describe('Three Walls walls', () => {
  jest.setTimeout(150000);
  const W = 240, H = 160;

  const luma = (p: { r: number; g: number; b: number }) => (p.r + p.g + p.b) / 3;
  const litCount = (f: any, threshold = 40) => {
    let n = 0;
    f.forEachPixel((p: any) => { if (luma(p) > threshold) n++; });
    return n;
  };
  /** Centre of mass of the lit pixels, in x. */
  const litCentroidX = (f: any, threshold = 60) => {
    let sum = 0, n = 0;
    f.forEachPixel((p: any, x: number) => {
      if (luma(p) > threshold) { sum += x; n++; }
    });
    return n > 0 ? sum / n : W / 2;
  };

  // A leading solid_color matters: three_walls reads tex_in, and an effect at
  // the head of a chain has no input texture to read.
  const build = (field: string, params: Record<string, unknown>): Sketch => ({
    anchor: null,
    wires: [{ id: 'ww', src: { instanceKey: 'tw@0', field },
              dest: { instanceKey: 'send@0', field: 'send_in' } }],
    chain: [
      { type: 'module', module_type: 'source.solid_color', instance_key: 'bg@0',
        params: { color: [0, 0, 0] } },
      { type: 'module', module_type: 'source.mesh.three_walls', instance_key: 'tw@0',
        params: { grain: 0, scanline: 0, chroma_bleed: 0, ...params } },
      { type: 'module', module_type: 'util.sidechannel_out', instance_key: 'send@0',
        params: { channel: 3 } },
      { type: 'module', module_type: 'util.sidechannel_in', instance_key: 'recv@0',
        params: { channel: 3 } },
    ],
  } as Sketch);

  const view = (id: string, field: string, params: Record<string, unknown>) =>
    runEngineTest({
      width: W, height: H,
      modules: ['com.nano.lights', 'com.nano.core'],
      commands: [
        { type: 'createSketch', sketchId: id, sketch: build(field, params) },
        { type: 'setTracePoints', tracePoints: [
          { id: 'out', target: { type: 'sketch_output', sketchId: id } }]},
      ],
      waitFrames: 25, captureTraceIds: ['out'], dumpName: id,
    });

  // Rate 0 freezes the three frames at phases 0, 1/3 and 2/3 forever, so every
  // case below samples an EXACT pose however long the wall clock took.
  const FROZEN = { resonate: 1, resonate_f0: 0, resonate_f1: 0 };

  it('a frame on the back wall is not on the side walls', async () => {
    // Small frames: all three are well inside the back wall, so the side walls
    // have nothing on them at all.
    const p = { ...FROZEN, quad_size: 0.4 };
    const main = await view('tw_part_main', 'tex_out', p);
    const left = await view('tw_part_left', 'left_out', p);
    const right = await view('tw_part_right', 'right_out', p);
    expect(main.success && left.success && right.success).toBe(true);
    expect(litCount(main.trace('out'))).toBeGreaterThan(500);
    expect(litCount(left.trace('out'))).toBeLessThan(60);
    expect(litCount(right.trace('out'))).toBeLessThan(60);
  });

  it('a frame that overflows leaves the back wall and lands on both sides',
     async () => {
    // Big frames: the nearest has crossed the back wall's edge, so it is gone
    // from the main output and present on BOTH side walls — one vertical edge
    // each, which is the whole partition in one assertion.
    const p = { ...FROZEN, quad_size: 2.0 };
    const left = await view('tw_flow_left', 'left_out', p);
    const right = await view('tw_flow_right', 'right_out', p);
    expect(left.success && right.success).toBe(true);
    expect(litCount(left.trace('out'))).toBeGreaterThan(300);
    expect(litCount(right.trace('out'))).toBeGreaterThan(300);
  });

  it('the two side walls mirror each other about the back wall', async () => {
    const p = { ...FROZEN, quad_size: 2.0 };
    const left = await view('tw_mirror_left', 'left_out', p);
    const right = await view('tw_mirror_right', 'right_out', p);
    expect(left.success && right.success).toBe(true);

    // Each wall's FAR end abuts the back wall, so it is on the inner side of
    // that output — right edge for the left wall, left edge for the right one.
    // A bar partway down therefore sits on opposite sides of the two pictures,
    // at mirrored distances from the centre.
    const lx = litCentroidX(left.trace('out'));
    const rx = litCentroidX(right.trace('out'));
    expect(lx).toBeLessThan(W / 2);
    expect(rx).toBeGreaterThan(W / 2);
    expect(Math.abs((lx + rx) / 2 - W / 2)).toBeLessThan(W * 0.06);
  });

  it('a frame travels down the side wall as it keeps going', async () => {
    // Further along means nearer the camera, which is the OUTER end of the
    // side output. Two sizes put the same frame at two points on that run.
    // Both sizes keep the MIDDLE frame on the back wall, so only the nearest
    // one is on the side and the centre of mass is that single bar. Push
    // further and a second bar arrives and averages the measurement away.
    const near = await view('tw_run_a', 'left_out', { ...FROZEN, quad_size: 1.2 });
    const far  = await view('tw_run_b', 'left_out', { ...FROZEN, quad_size: 1.9 });
    expect(near.success && far.success).toBe(true);
    // The left wall's far end is its right edge, so travelling means moving
    // LEFT across the picture.
    expect(litCentroidX(far.trace('out')))
      .toBeLessThan(litCentroidX(near.trace('out')) - W * 0.1);
  });

  /** The brightest column in a picture, and whether it runs top to bottom. */
  const brightestColumn = (f: any) => {
    let best = -1, bx = 0;
    const colMean: number[] = new Array(W).fill(0);
    f.forEachPixel((p: any, x: number) => { colMean[x] += luma(p) / H; });
    for (let x = 0; x < W; x++) if (colMean[x] > best) { best = colMean[x]; bx = x; }
    return { x: bx, top: luma(f.pixelAt(bx, 2)), bottom: luma(f.pixelAt(bx, H - 3)) };
  };

  // THE SEAM HOLDS AT ANY KEYSTONE. A frame straddling the threshold has its
  // edge on the back wall and on the side wall in the same instant. At the far
  // end the side wall IS the back wall's edge, so the two must be the same
  // height — a frame's bar arrives full height and stays there, whatever the
  // keystone is. Get the taper backwards and a frame crossing the corner
  // collapses to a stub and the room tears.
  it('a frame keeps its height crossing the corner, at any keystone', async () => {
    // Just past the threshold, so the bar is at the far end of the side wall.
    const at = (k: number, id: string) =>
      view(id, 'left_out', { ...FROZEN, quad_size: 1.05, wall_keystone: k });
    const flat = await at(0, 'tw_seam_flat');
    const keyed = await at(1, 'tw_seam_keyed');
    expect(flat.success && keyed.success).toBe(true);

    for (const r of [flat, keyed]) {
      const bar = brightestColumn(r.trace('out'));
      expect(bar.top).toBeGreaterThan(120);
      expect(bar.bottom).toBeGreaterThan(120);
    }
  });

  it('Keystone moves where along the wall a frame sits', async () => {
    // Same frame, two pictures of the same wall: flat is linear in depth,
    // keystoned is linear in apparent position, which crowds the far end.
    const flat = await view('tw_keys_flat', 'left_out',
                            { ...FROZEN, quad_size: 1.5, wall_keystone: 0 });
    const keyed = await view('tw_keys_on', 'left_out',
                             { ...FROZEN, quad_size: 1.5, wall_keystone: 1 });
    expect(flat.success && keyed.success).toBe(true);
    // The left wall's far end is its RIGHT edge, so "crowded toward the far
    // end" means further right.
    expect(brightestColumn(keyed.trace('out')).x)
      .toBeGreaterThan(brightestColumn(flat.trace('out')).x + 10);
  });

  // The point of raking the wall is that a frame which passes you is GONE. An
  // earlier version pinned both ends of the run so nothing could ever leave,
  // and frames piled up against the outer edge and sat there instead.
  it('Stretch runs frames off the near end of the wall', async () => {
    // Far enough along that at rest the frame is well inside the near third...
    const slow = await view('tw_exit_slow', 'left_out',
                            { ...FROZEN, quad_size: 3.2, wall_stretch: 1 });
    // ...and raked, it has already left the picture entirely.
    const fast = await view('tw_exit_fast', 'left_out',
                            { ...FROZEN, quad_size: 3.2, wall_stretch: 3 });
    expect(slow.success && fast.success).toBe(true);

    // The left wall's NEAR end is its left edge, so "gone" means nothing lit in
    // the outer third.
    const nearThird = (f: any) => {
      let n = 0;
      f.forEachPixel((p: any, x: number) => { if (x < W / 3 && luma(p) > 60) n++; });
      return n;
    };
    expect(nearThird(slow.trace('out'))).toBeGreaterThan(40);
    expect(nearThird(fast.trace('out'))).toBeLessThan(10);
  });

  it('Stretch runs a frame down the wall faster', async () => {
    const slow = await view('tw_str_slow', 'left_out',
                            { ...FROZEN, quad_size: 1.5, wall_stretch: 1 });
    const fast = await view('tw_str_fast', 'left_out',
                            { ...FROZEN, quad_size: 1.5, wall_stretch: 3 });
    expect(slow.success && fast.success).toBe(true);
    // Further along means further from the far end, i.e. further LEFT.
    expect(brightestColumn(fast.trace('out')).x)
      .toBeLessThan(brightestColumn(slow.trace('out')).x - W * 0.1);
  });

  it('the main view is unaffected by whether the side walls are wired',
     async () => {
    // The side dispatches are gated on connectivity, and that gate must not be
    // able to change what the back wall looks like.
    const p = { ...FROZEN, quad_size: 0.6 };
    const alone = await view('tw_gate_alone', 'tex_out', p);
    const withAux = await runEngineTest({
      width: W, height: H,
      modules: ['com.nano.lights', 'com.nano.core'],
      commands: [
        { type: 'createSketch', sketchId: 'tw_gate_both', sketch: {
          anchor: null,
          wires: [
            { id: 'w1', src: { instanceKey: 'tw@0', field: 'tex_out' },
              dest: { instanceKey: 'send@0', field: 'send_in' } },
            // A consumer for a side wall too, so its dispatch actually runs.
            { id: 'w2', src: { instanceKey: 'tw@0', field: 'left_out' },
              dest: { instanceKey: 'send2@0', field: 'send_in' } },
          ],
          chain: [
            { type: 'module', module_type: 'source.solid_color', instance_key: 'bg@0',
              params: { color: [0, 0, 0] } },
            { type: 'module', module_type: 'source.mesh.three_walls', instance_key: 'tw@0',
              params: { grain: 0, scanline: 0, chroma_bleed: 0, ...p } },
            { type: 'module', module_type: 'util.sidechannel_out', instance_key: 'send2@0',
              params: { channel: 4 } },
            { type: 'module', module_type: 'util.sidechannel_out', instance_key: 'send@0',
              params: { channel: 3 } },
            { type: 'module', module_type: 'util.sidechannel_in', instance_key: 'recv@0',
              params: { channel: 3 } },
          ],
        } as Sketch },
        { type: 'setTracePoints', tracePoints: [
          { id: 'out', target: { type: 'sketch_output', sketchId: 'tw_gate_both' } }]},
      ],
      waitFrames: 25, captureTraceIds: ['out'], dumpName: 'tw_gate_both',
    });
    expect(alone.success && withAux.success).toBe(true);
    // Same frozen pose either way, so the two back walls agree closely.
    expect(Math.abs(litCount(withAux.trace('out')) - litCount(alone.trace('out'))))
      .toBeLessThan(litCount(alone.trace('out')) * 0.15);
  });
});
