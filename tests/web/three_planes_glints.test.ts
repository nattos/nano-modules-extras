import { runEngineMultiPhaseTest } from '@nano/web/test/engine-test-helpers';
import type { Sketch } from '@nano/web/src/sketch-types';

/**
 * E2E for `source.mesh.three_planes`' glints.
 *
 * These live on the ENGINE harness rather than the per-effect GPU one for a
 * structural reason: a glint is thrown by a GESTURE — the sweep knob crossing
 * the middle of its throw — and the per-effect harness sets its params once
 * and then ticks, so a knob there is parked forever and nothing is ever
 * launched. (That the parked case renders nothing at all is pinned over in
 * three_planes.test.ts.) Only a real engine can move a param between frames.
 *
 * Engine dt is wall clock, so nothing here asserts a TIME. The invariant —
 * sweep the knob across its range at a constant speed and one glint crosses
 * the picture at exactly that rate — is stated as an equality in
 * native/tests/test_three_planes_glints.cpp, which owns the clock. What only a
 * real engine shows is that a moving param reaches the particle system at all,
 * and that a live glint draws as one slash and travels the right way.
 *
 * Ratio is turned right down throughout so a launched glint crosses over
 * seconds: that makes "is it still up there" independent of how long a
 * waitFrames window actually took.
 */
describe('Three Planes glints E2E', () => {
  jest.setTimeout(120000);

  const W = 240, H = 160;
  const MODULES = ['com.nano.core', 'com.nano.lights'];

  // Cover-square coords -> pixel. Mirrors fx::coverSquare / nano_coords.hlsl.
  const ax = Math.max(W, H) / (2 * W);
  const ay = Math.max(W, H) / (2 * H);
  const toPx = (sx: number, sy: number): [number, number] => [
    Math.round((sx * ax + 0.5) * W),
    Math.round((sy * ay + 0.5) * H),
  ];
  const luma = (p: { r: number; g: number; b: number }) => (p.r + p.g + p.b) / 3;

  // Travel direction at the default 45 deg. Cover-square y grows DOWNWARD, so
  // "up-right" is (+, −).
  const R2 = Math.SQRT1_2;

  /**
   * The frame projected onto the travel axis: the brightest pixel at each
   * distance along it.
   *
   * Projecting the WHOLE frame rather than walking its centre diagonal is not
   * fussiness — a glint is a band square across the travel, and a fresh one
   * sits out past where that diagonal runs out of picture, so a line scan
   * reports a flat field for the first part of every glint's life.
   */
  const AXIS_LO = -1.25, AXIS_HI = 1.25, AXIS_STEP = 0.01;
  const AXIS: number[] = [];
  for (let t = AXIS_LO; t <= AXIS_HI + 1e-6; t += AXIS_STEP) AXIS.push(t);

  const scan = (f: any) => {
    const out = new Array<number>(AXIS.length).fill(-1);
    f.forEachPixel((p: { r: number; g: number; b: number }, x: number, y: number) => {
      // Pixel centre -> cover-square -> distance along the travel axis.
      const sx = ((x + 0.5) / W - 0.5) / ax;
      const sy = ((y + 0.5) / H - 0.5) / ay;
      const t = sx * R2 - sy * R2;
      const i = Math.round((t - AXIS_LO) / AXIS_STEP);
      if (i < 0 || i >= out.length) return;
      const l = luma(p);
      if (l > out[i]) out[i] = l;
    });
    // Trim the ends no pixel reached: the travel axis runs past the corners of
    // the frame, and an empty bucket beside a real one is a cliff every peak
    // test would fall off.
    let lo = 0, hi = out.length - 1;
    while (lo < out.length && out[lo] < 0) lo++;
    while (hi >= 0 && out[hi] < 0) hi--;
    return { v: out.slice(lo, hi + 1), lo };
  };

  /**
   * Local maxima standing clear of the field around them — i.e. glints.
   *
   * Two things have to be true, and the second is not obvious. The standoff
   * has to clear a glint's own core, which is flat-topped (a super-Gaussian)
   * and a good fraction of the axis wide. And a candidate has to beat the
   * field's own BASELINE, not just its neighbours: every glint drags a dark
   * wake, so an untouched stretch of picture sitting between two wakes is a
   * local maximum too, and counting those would find a glint between every
   * pair of real ones.
   */
  const PEAK_SPAN = 16;
  const peaks = (s: { v: number[]; lo: number }, prominence: number) => {
    const v = s.v;
    const sorted = [...v].sort((a, b) => a - b);
    const base = sorted[Math.floor(sorted.length / 2)];
    const out: number[] = [];
    for (let i = PEAK_SPAN; i < v.length - PEAK_SPAN; i++) {
      const isTop = v[i] >= v[i - 1] && v[i] >= v[i + 1] &&
                    v[i] > base + prominence &&
                    v[i] > v[i - PEAK_SPAN] + prominence &&
                    v[i] > v[i + PEAK_SPAN] + prominence;
      if (isTop && (out.length === 0 ||
                    i - out[out.length - 1] > PEAK_SPAN)) out.push(i);
    }
    return out;
  };

  /** Axis coordinate of a peak index from `peaks`. */
  const axisOf = (s: { v: number[]; lo: number }, i: number) =>
    AXIS_LO + (s.lo + i) * AXIS_STEP;

  // A flat lit target: the three planes collapsed onto one another, filled and
  // grown past the frame edges, graded neutrally. Everything visible is then
  // ONE uniform emission, so the glint field reads straight off a pixel with
  // no geometry underneath it to confound the scan.
  const FLAT = {
    grain: 0, scanline: 0, chroma_bleed: 0,
    drive: 0, toe: 0, shoulder: 0, warmth: 0, asymmetry: 0, highlight_desat: 0,
    plane_spacing: 0, zoom: 1.5, plane_size: 1.5,
    plane1_emission: 0.30, plane1_fill: 1, fill_gain: 1,
    plane2_emission: 0, plane3_emission: 0,
    line_width: 0, halo_gain: 0,
    glimmer_gain: 1.6, glimmer_shadow: 0.5,
    // Slow enough that a launched glint is up there for several seconds.
    glimmer_ratio: 0.15, glimmer_chaos: 0,
  };

  const sketch = (params: Record<string, unknown>): Sketch => ({
    anchor: null,
    wires: [],
    chain: [
      { type: 'module', module_type: 'source.solid_color', instance_key: 'bg@0',
        params: { color: [0, 0, 0] } },
      { type: 'module', module_type: 'source.mesh.three_planes',
        instance_key: 'tp@0', params: { ...FLAT, ...params } },
    ],
  } as Sketch);

  /** Park the knob, then sweep it across the middle, sampling as it goes. */
  const gesture = (id: string, params: Record<string, unknown>,
                   from: number, to: number, waits: number[]) =>
    runEngineMultiPhaseTest({
      width: W, height: H, modules: MODULES,
      phases: [
        { commands: [
            { type: 'createSketch', sketchId: id,
              sketch: sketch({ ...params, glimmer_sweep: from }) },
            { type: 'setTracePoints', tracePoints: [
                { id: 'out', target: { type: 'sketch_output', sketchId: id } }] },
          ],
          waitFrames: 20, captureTraceIds: ['out'] },
        // The gesture: one param write across the middle of the throw.
        { commands: [{ type: 'setParam', sketchId: id, colIdx: 0, chainIdx: 1,
                       paramKey: 'glimmer_sweep', value: to }],
          waitFrames: waits[0], captureTraceIds: ['out'] },
        { commands: [], waitFrames: waits[1], captureTraceIds: ['out'] },
      ],
      dumpName: id,
    });

  it('crossing the middle of the throw launches one glint', async () => {
    const r = await gesture('glint_launch', {}, 0.95, 0.05, [6, 30]);
    expect(r.success).toBe(true);

    // Parked: nothing at all. The field is one flat colour.
    const before = scan(r.phases[0].trace('out'));
    expect(peaks(before, 8).length).toBe(0);

    // After the gesture: exactly one slash, and only one — Chaos is off, so a
    // single traverse is a single glint.
    const after = scan(r.phases[1].trace('out'));
    expect(peaks(after, 8).length).toBe(1);
  });

  it('the glint travels, and the sweep does not reach back into it', async () => {
    // Keep the knob moving between the two readings, so the speed never falls
    // to a drift and nothing is puttering: the leader must be further along
    // and just as bright as it was, even though the drive has been all over
    // the place since it was thrown.
    const r = await runEngineMultiPhaseTest({
      width: W, height: H, modules: MODULES,
      phases: [
        { commands: [
            { type: 'createSketch', sketchId: 'glint_travel',
              sketch: sketch({ glimmer_sweep: 0.95 }) },
            { type: 'setTracePoints', tracePoints: [
                { id: 'out', target: { type: 'sketch_output', sketchId: 'glint_travel' } }] },
          ],
          waitFrames: 20, captureTraceIds: ['out'] },
        { commands: [{ type: 'setParam', sketchId: 'glint_travel', colIdx: 0,
                       chainIdx: 1, paramKey: 'glimmer_sweep', value: 0.05 }],
          waitFrames: 6, captureTraceIds: ['out'] },
        // Keep sweeping. This throws a second glint behind the first, which is
        // why the leader is read as the LAST peak rather than the only one.
        { commands: [{ type: 'setParam', sketchId: 'glint_travel', colIdx: 0,
                       chainIdx: 1, paramKey: 'glimmer_sweep', value: 0.95 }],
          waitFrames: 10, captureTraceIds: ['out'] },
      ],
      dumpName: 'glint_travel',
    });
    expect(r.success).toBe(true);
    const a = scan(r.phases[1].trace('out'));
    const b = scan(r.phases[2].trace('out'));
    const pa = peaks(a, 8), pb = peaks(b, 8);
    expect(pa.length).toBe(1);
    expect(pb.length).toBeGreaterThanOrEqual(1);
    const lead = pb[pb.length - 1];
    // Moved along the travel axis, in the +dir sense...
    expect(axisOf(b, lead)).toBeGreaterThan(axisOf(a, pa[0]) + 0.03);
    // ...and just as bright as it was.
    expect(Math.abs(b.v[lead] - a.v[pa[0]])).toBeLessThan(14);
  });

  it('let go and the glint putters out where it is', async () => {
    // The other death. A glint that sailed on at the drift speed forever would
    // outlive the gesture that made it, so the envelope running the speed down
    // runs the glint down with it — still moving, but shrinking and dimming
    // until there is nothing left.
    const r = await runEngineMultiPhaseTest({
      width: W, height: H, modules: MODULES,
      phases: [
        { commands: [
            { type: 'createSketch', sketchId: 'glint_putter',
              sketch: sketch({ glimmer_sweep: 0.95 }) },
            { type: 'setTracePoints', tracePoints: [
                { id: 'out', target: { type: 'sketch_output', sketchId: 'glint_putter' } }] },
          ],
          waitFrames: 20, captureTraceIds: ['out'] },
        { commands: [{ type: 'setParam', sketchId: 'glint_putter', colIdx: 0,
                       chainIdx: 1, paramKey: 'glimmer_sweep', value: 0.05 }],
          waitFrames: 6, captureTraceIds: ['out'] },
        // ...and then nothing. The knob is not touched again.
        { commands: [], waitFrames: 60, captureTraceIds: ['out'] },
        { commands: [], waitFrames: 90, captureTraceIds: ['out'] },
      ],
      dumpName: 'glint_putter',
    });
    expect(r.success).toBe(true);

    const born = scan(r.phases[1].trace('out'));
    const pb = peaks(born, 8);
    expect(pb.length).toBe(1);
    const bright = born.v[pb[0]];

    // Part way through the run-down: still there, and dimmer. (Or already
    // gone, if the frames were long — either way it is on its way out, which
    // is the claim. Engine dt is wall clock, so this cannot assert a time.)
    const mid = scan(r.phases[2].trace('out'));
    const pm = peaks(mid, 8);
    if (pm.length > 0) expect(mid.v[pm[0]]).toBeLessThan(bright - 4);

    // And gone — without ever having reached the far side of the picture.
    const late = scan(r.phases[3].trace('out'));
    expect(peaks(late, 8).length).toBe(0);
  });

  it('the direction is fixed: reversing the knob does not turn it round',
     async () => {
    // Sign is thrown away. The knob goes down through the middle, then back up
    // through it — and the first glint carries on exactly as it was while the
    // second one is thrown behind it.
    const r = await runEngineMultiPhaseTest({
      width: W, height: H, modules: MODULES,
      phases: [
        { commands: [
            { type: 'createSketch', sketchId: 'glint_rev',
              sketch: sketch({ glimmer_sweep: 0.95 }) },
            { type: 'setTracePoints', tracePoints: [
                { id: 'out', target: { type: 'sketch_output', sketchId: 'glint_rev' } }] },
          ],
          waitFrames: 20, captureTraceIds: ['out'] },
        { commands: [{ type: 'setParam', sketchId: 'glint_rev', colIdx: 0,
                       chainIdx: 1, paramKey: 'glimmer_sweep', value: 0.05 }],
          waitFrames: 8, captureTraceIds: ['out'] },
        { commands: [{ type: 'setParam', sketchId: 'glint_rev', colIdx: 0,
                       chainIdx: 1, paramKey: 'glimmer_sweep', value: 0.95 }],
          waitFrames: 30, captureTraceIds: ['out'] },
      ],
      dumpName: 'glint_rev',
    });
    expect(r.success).toBe(true);
    const sa = scan(r.phases[1].trace('out'));
    const sb = scan(r.phases[2].trace('out'));
    const a = peaks(sa, 8), b = peaks(sb, 8);
    expect(a.length).toBe(1);
    // Two of them now — the return trip is its own gesture — and the leader is
    // further along than it was, not back where it came from.
    expect(b.length).toBe(2);
    expect(axisOf(sb, b[b.length - 1])).toBeGreaterThan(axisOf(sa, a[0]));
  });

  it('Chaos scuffs a fast sweep up with extra, smaller glints', async () => {
    // The small ones only start arriving once the sweep is BRISK — measured in
    // crossings per second, so the ratio has to be up for a one-frame param
    // jump to count as fast. At 0.15 the same jump is a gentle sweep and the
    // result is one clean glint, which is the point of the threshold.
    const fast = { glimmer_ratio: 0.45 };
    const clean = await gesture('glint_clean', { ...fast, glimmer_chaos: 0 },
                                0.95, 0.05, [10, 10]);
    const messy = await gesture('glint_chaos', { ...fast, glimmer_chaos: 14 },
                                0.95, 0.05, [10, 10]);
    expect(clean.success && messy.success).toBe(true);
    const nClean = peaks(scan(clean.phases[1].trace('out')), 6).length;
    const nMessy = peaks(scan(messy.phases[1].trace('out')), 6).length;
    expect(nClean).toBe(1);
    expect(nMessy).toBeGreaterThan(nClean);
  });

  it('a glint lifts the halo, not just the line core', async () => {
    // HALF THE CLAIM THIS EFFECT MAKES. The glint multiplies emission, and
    // emission scales the core, the halo and the fill together — so one
    // crossing a tube brightens the glow around it as well, which is what
    // stops it reading as a highlight pasted on top of the picture.
    const only2 = {
      grain: 0, scanline: 0, chroma_bleed: 0,
      plane1_emission: 0, plane3_emission: 0,
      // Dim enough that the probe has headroom BOTH ways: at the effect's
      // default levels a wide halo clips flat at the top, and a glint crossing
      // a clipped pixel is invisible.
      plane2_emission: 0.30, halo_radius: 1.0, halo_gain: 0.7,
      glimmer_gain: 1.6, glimmer_shadow: 0.6,
      glimmer_ratio: 0.45, glimmer_chaos: 0,
    };

    // Probe a RING just outside plane 2's outline rather than one pixel of it.
    // The plane is a diamond with vertices at (±0.482, 0) and (0, ∓0.279);
    // 1.15x that is outside the line but well inside the halo, all the way
    // round. A glint is a band across the whole picture, so wherever it has
    // got to it crosses this ring somewhere — which is what makes the reading
    // a fact about the halo rather than a bet on the glint's position.
    const ring: [number, number][] = [];
    for (let i = 0; i < 24; i++) {
      const a = (i / 24) * Math.PI * 2;
      const c = Math.cos(a), sn = Math.sin(a);
      const k = 1.15 / (Math.abs(c) / 0.482 + Math.abs(sn) / 0.279);
      ring.push([c * k, sn * k]);
    }
    const ringMax = (f: any) =>
      Math.max(...ring.map(([x, y]) => luma(f.pixelAt(...toPx(x, y)))));

    // Flip the knob back and forth so there is always something in flight.
    const flips = [0.05, 0.95, 0.05];
    const r = await runEngineMultiPhaseTest({
      width: W, height: H, modules: MODULES,
      phases: [
        { commands: [
            { type: 'createSketch', sketchId: 'glint_halo',
              sketch: sketch({ ...only2, glimmer_sweep: 0.95 }) },
            { type: 'setTracePoints', tracePoints: [
                { id: 'out', target: { type: 'sketch_output', sketchId: 'glint_halo' } }] },
          ],
          waitFrames: 20, captureTraceIds: ['out'] },
        ...flips.map((v) => ({
          commands: [{ type: 'setParam' as const, sketchId: 'glint_halo', colIdx: 0,
                       chainIdx: 1, paramKey: 'glimmer_sweep', value: v }],
          waitFrames: 8, captureTraceIds: ['out'],
        })),
      ],
      dumpName: 'glint_halo',
    });
    expect(r.success).toBe(true);

    const rest = ringMax(r.phases[0].trace('out'));
    expect(rest).toBeGreaterThan(0);   // the ring really is in the halo
    const lit = Math.max(...[1, 2, 3].map((i) => ringMax(r.phases[i].trace('out'))));
    expect(lit).toBeGreaterThan(rest + 6);
  });

  it('a glint cannot touch a pixel that is not emitting', async () => {
    // THE OTHER HALF. It is a multiplier ON EMISSION, not a layer over the
    // finished frame: with every plane dark the incoming image survives
    // untouched however hard the glints are driven. An overlay would tint it.
    const dark = {
      grain: 0, scanline: 0, chroma_bleed: 0,
      plane1_emission: 0, plane2_emission: 0, plane3_emission: 0,
      glimmer_gain: 3, glimmer_shadow: 1, glimmer_chaos: 14, glimmer_ratio: 0.15,
    };
    const r = await runEngineMultiPhaseTest({
      width: W, height: H, modules: MODULES,
      phases: [
        { commands: [
            { type: 'createSketch', sketchId: 'glint_dark', sketch: {
                anchor: null, wires: [],
                chain: [
                  { type: 'module', module_type: 'source.solid_color',
                    instance_key: 'bg@0', params: { color: [0.45, 0.20, 0.30] } },
                  { type: 'module', module_type: 'source.mesh.three_planes',
                    instance_key: 'tp@0',
                    params: { ...dark, glimmer_sweep: 0.95 } },
                ],
              } as Sketch },
            { type: 'setTracePoints', tracePoints: [
                { id: 'out', target: { type: 'sketch_output', sketchId: 'glint_dark' } }] },
          ],
          waitFrames: 20, captureTraceIds: ['out'] },
        { commands: [{ type: 'setParam', sketchId: 'glint_dark', colIdx: 0,
                       chainIdx: 1, paramKey: 'glimmer_sweep', value: 0.05 }],
          waitFrames: 24, captureTraceIds: ['out'] },
      ],
      dumpName: 'glint_dark',
    });
    expect(r.success).toBe(true);
    const a = r.phases[0].trace('out');
    const b = r.phases[1].trace('out');
    let worst = 0;
    a.forEachPixel((p: any, x: number, y: number) => {
      worst = Math.max(worst, Math.abs(luma(p) - luma(b.pixelAt(x, y))));
    });
    expect(worst).toBe(0);
  });

  // ---------------------------------------------------------------- Release
  // The throw. Reaching either end of the sweep mutes the tower — and flings
  // it outward as three rings that open up and go dull as they fly. The
  // envelope is the rig's (pinned in native/tests/test_three_planes_rig.cpp);
  // what only a render shows is that the rings actually leave the stack, and
  // that they are LIGHT rather than a brightening of what was already there.

  // NOT the flat target the glint cases use: that one blows the quads up past
  // the frame so their interiors fill it, and a ring is only an OUTLINE — off
  // the edge of the picture, where nothing can see it. The stack's own camera,
  // and a muted tower exactly as the rig leaves it at either end, so anything
  // in the frame is the throw and nothing else.
  const RING = {
    grain: 0, scanline: 0, chroma_bleed: 0,
    plane1_emission: 0, plane2_emission: 0, plane3_emission: 0,
    glimmer_chaos: 0, glimmer_sweep: 0.5,
    // Kept inside the frame so "further out" stays measurable instead of
    // saturating against the corners.
    release_expand: 0.5,
  };

  const ringSketch = (params: Record<string, unknown>): Sketch => ({
    anchor: null,
    wires: [],
    chain: [
      { type: 'module', module_type: 'source.solid_color', instance_key: 'bg@0',
        params: { color: [0, 0, 0] } },
      { type: 'module', module_type: 'source.mesh.three_planes',
        instance_key: 'tp@0', params: { ...RING, ...params } },
    ],
  } as Sketch);

  /**
   * Where the light IS: its distance from the centre, weighted by brightness.
   *
   * Not "the furthest lit pixel", which is the obvious measure and the wrong
   * one — a ring dims as it flies, so its faint outer skirt drops under any
   * threshold you pick and the reading comes back SMALLER for a ring that has
   * travelled further. Normalising by the total cancels the dimming out and
   * leaves only where the light sits.
   */
  const spreadOf = (f: any) => {
    let sum = 0, weighted = 0;
    f.forEachPixel((p: { r: number; g: number; b: number }, x: number, y: number) => {
      const l = luma(p);
      if (l < 4) return;
      const sx = ((x + 0.5) / W - 0.5) / ax;
      const sy = ((y + 0.5) / H - 0.5) / ay;
      sum += l;
      weighted += l * Math.sqrt(sx * sx + sy * sy);
    });
    return sum > 0 ? weighted / sum : 0;
  };

  it('the throw flings the stack outward as rings that open and go dull',
     async () => {
    // Driven straight, rather than through the rig: the envelope has its own
    // goldens, and what is under test here is the geometry it is spent on.
    const at = (v: number) => ({
      type: 'setParam' as const, sketchId: 'ring', colIdx: 0, chainIdx: 1,
      paramKey: 'release', value: v,
    });
    const r = await runEngineMultiPhaseTest({
      width: W, height: H, modules: MODULES,
      phases: [
        { commands: [
            { type: 'createSketch', sketchId: 'ring', sketch: ringSketch({}) },
            { type: 'setTracePoints', tracePoints: [
                { id: 'out', target: { type: 'sketch_output', sketchId: 'ring' } }] },
          ],
          waitFrames: 12, captureTraceIds: ['out'] },
        { commands: [at(1.0)], waitFrames: 4, captureTraceIds: ['out'] },   // the throw
        { commands: [at(0.6)], waitFrames: 4, captureTraceIds: ['out'] },   // flying
        { commands: [at(0.3)], waitFrames: 4, captureTraceIds: ['out'] },   // spent
        { commands: [at(0.02)], waitFrames: 4, captureTraceIds: ['out'] },  // over
      ],
      dumpName: 'ring_throw',
    });
    expect(r.success).toBe(true);
    const f = (i: number) => r.phases[i].trace('out');

    // Release 0 shows NOTHING over a muted tower — so an unwired card, and a
    // sweep sitting anywhere but an end, cost exactly nothing.
    let painted = 0;
    f(0).forEachPixel((p: { r: number; g: number; b: number }) => {
      if (luma(p) > 4) painted++;
    });
    expect(painted).toBe(0);

    // Thrown: light, over a tower that is emitting none.
    expect(spreadOf(f(1))).toBeGreaterThan(0);
    // ...and it travels outward as the throw is spent.
    expect(spreadOf(f(2))).toBeGreaterThan(spreadOf(f(1)) + 0.03);
    expect(spreadOf(f(3))).toBeGreaterThan(spreadOf(f(2)) + 0.03);

    // And the tail ENDS. A release that trailed away asymptotically would
    // leave the muted picture permanently not-quite-black.
    let left = 0;
    f(4).forEachPixel((p: { r: number; g: number; b: number }) => {
      if (luma(p) > 4) left++;
    });
    expect(left).toBe(0);
  });

  it('the rings dim as they open rather than blooming', async () => {
    // The trap this guards: the halo profile peaks at 1 whatever its radius,
    // so opening a ring out WITHOUT paying for it spreads the same peak over
    // more picture and the thing gets brighter as it dissipates — which is the
    // exact opposite of a release.
    const at = (v: number) => ({
      type: 'setParam' as const, sketchId: 'ring_dim', colIdx: 0, chainIdx: 1,
      paramKey: 'release', value: v,
    });
    const r = await runEngineMultiPhaseTest({
      width: W, height: H, modules: MODULES,
      phases: [
        { commands: [
            { type: 'createSketch', sketchId: 'ring_dim',
              sketch: ringSketch({ release: 1.0 }) },
            { type: 'setTracePoints', tracePoints: [
                { id: 'out', target: { type: 'sketch_output', sketchId: 'ring_dim' } }] },
          ],
          waitFrames: 12, captureTraceIds: ['out'] },
        { commands: [at(0.45)], waitFrames: 4, captureTraceIds: ['out'] },
        { commands: [at(0.12)], waitFrames: 4, captureTraceIds: ['out'] },
      ],
      dumpName: 'ring_dim',
    });
    expect(r.success).toBe(true);
    const peak = (i: number) => {
      let m = 0;
      r.phases[i].trace('out').forEachPixel(
        (p: { r: number; g: number; b: number }) => { m = Math.max(m, luma(p)); });
      return m;
    };
    expect(peak(0)).toBeGreaterThan(peak(1));
    expect(peak(1)).toBeGreaterThan(peak(2));
  });

  it('a Strobe throw keeps the glints in the picture; Grow slings them out',
     async () => {
    // A throw does two things to whatever is in the air, and they come apart.
    // Both modes HOLD it at full strength while the throw rings out — letting
    // go of the knob to reach a mute is what fires one, so a glint being
    // thrown must not start running down. Only Grow also SLINGS it along, and
    // that belongs to a throw whose whole idea is light leaving the frame.
    // Strobe keeps everything inside, so its glints hang about and drift.
    const run = (id: string, mode: number) => runEngineMultiPhaseTest({
      width: W, height: H, modules: MODULES,
      phases: [
        { commands: [
            { type: 'createSketch', sketchId: id,
              sketch: sketch({ glimmer_sweep: 0.95, release_mode: mode }) },
            { type: 'setTracePoints', tracePoints: [
                { id: 'out', target: { type: 'sketch_output', sketchId: id } }] },
          ],
          waitFrames: 20, captureTraceIds: ['out'] },
        // The gesture, which launches one...
        { commands: [{ type: 'setParam', sketchId: id, colIdx: 0, chainIdx: 1,
                       paramKey: 'glimmer_sweep', value: 0.70 }],
          waitFrames: 6, captureTraceIds: ['out'] },
        // ...and then the throw, held rather than decayed: there is no rig
        // here, so `release` stays where it is put and the hold never lifts.
        // Three even windows after it, to watch where the glint gets to.
        { commands: [{ type: 'setParam', sketchId: id, colIdx: 0, chainIdx: 1,
                       paramKey: 'release', value: 1 }],
          waitFrames: 40, captureTraceIds: ['out'] },
        { commands: [], waitFrames: 40, captureTraceIds: ['out'] },
        { commands: [], waitFrames: 40, captureTraceIds: ['out'] },
      ],
      dumpName: id,
    });

    // Sequentially: the engine runner drives one page, and two at once abort
    // each other.
    const grow = await run('glint_throw_grow', 0);
    const strobe = await run('glint_throw_strobe', 1);
    expect(grow.success && strobe.success).toBe(true);

    const where = (r: any, phase: number) => {
      const s = scan(r.phases[phase].trace('out'));
      const p = peaks(s, 8);
      return p.length > 0 ? axisOf(s, p[0]) : null;
    };
    const S = [2, 3, 4].map((i) => where(strobe, i));
    const G = [2, 3, 4].map((i) => where(grow, i));

    // Still up there through all three windows, one slash the whole way, and
    // moving between every pair: held, not parked. Nothing here is ever frozen.
    for (const p of S) expect(p).not.toBeNull();
    expect(S[1]!).toBeGreaterThan(S[0]!);
    expect(S[2]!).toBeGreaterThan(S[1]!);

    // Slung: over the same window it covers several times the ground — or it
    // is simply GONE, having crossed the picture and died inside one window,
    // which is that same claim in its strongest form. Engine dt is wall clock
    // and this glint is the fast one, so which of the two you get is a matter
    // of frame pacing; the contrast with the held one, still up there and
    // barely moved over the very same windows, is what the case is about.
    expect(G[0]).not.toBeNull();
    if (G[1] !== null) {
      expect(G[1] - G[0]!).toBeGreaterThan((S[1]! - S[0]!) * 2);
      // Past where the held one gets to two windows later.
      expect(G[1]).toBeGreaterThan(S[2]!);
    } else {
      // Off the far side already, inside one window. The contrast is the whole
      // assertion: over that same window the held one is still in the picture
      // — and still there a window after that, and a window after that.
      expect(S[1]).not.toBeNull();
      expect(S[2]).not.toBeNull();
    }
  });

  it('a glint lights the throw itself, not just the tower', async () => {
    // The point of a Strobe throw is that the tower is muted and the ghosts
    // ARE the picture. If glints only multiplied the planes' emission they
    // would have nothing to act on in the state they were just taught to hang
    // around for — up there, drifting, touching nothing.
    //
    // Held inside the ARRIVAL, where all three floors are up: the roll is
    // slowed right down so one step is far longer than this window, which is
    // what makes two runs comparable at all when engine dt is wall clock.
    const run = (id: string, gesture: boolean) => runEngineMultiPhaseTest({
      width: W, height: H, modules: MODULES,
      phases: [
        { commands: [
            { type: 'createSketch', sketchId: id, sketch: ringSketch({
                release_mode: 1, strobe_rate: 4, strobe_duty: 1,
                // Wire Glow wide open, so the ghost is a broad glow rather
                // than the hairline the mode normally draws: a glint then
                // crosses a lot of it and the reading is a real fraction
                // rather than two per cent of a thin outline. The glint keeps
                // its nominal WIDTH — widening one moves its birth margin out
                // with it, and it would still be on its way in.
                strobe_glow: 1, glimmer_gain: 3,
                glimmer_ratio: 0.15, glimmer_shadow: 0,
                glimmer_sweep: 0.95 }) },
            { type: 'setTracePoints', tracePoints: [
                { id: 'out', target: { type: 'sketch_output', sketchId: id } }] },
          ],
          waitFrames: 20, captureTraceIds: ['out'] },
        // One run sweeps the knob through the middle and launches a glint; the
        // other leaves it exactly where it is and never has one.
        { commands: gesture ? [{ type: 'setParam', sketchId: id, colIdx: 0,
                                 chainIdx: 1, paramKey: 'glimmer_sweep',
                                 value: 0.70 }] : [],
          waitFrames: 6, captureTraceIds: ['out'] },
        { commands: [{ type: 'setParam', sketchId: id, colIdx: 0, chainIdx: 1,
                       paramKey: 'release', value: 1 }],
          waitFrames: 6, captureTraceIds: ['out'] },
      ],
      dumpName: id,
    });

    // Sequentially: the engine runner drives one page.
    const bare = await run('strobe_glint_bare', false);
    const lit = await run('strobe_glint_lit', true);
    expect(bare.success && lit.success).toBe(true);

    // Neither the brightest pixel nor the total will do. A neon core is
    // already blown to white in both runs, so a peak reading saturates; and an
    // engine trace comes back checkerboard-composited, so a frame total is
    // mostly background and a real flare moves it by a couple of per cent.
    //
    // So: the light the two frames differ BY, against the light the wireframe
    // brought on its own. Same sketch, same phase, grain and scanlines off —
    // the only thing that differs between them is the glint.
    const bareF = bare.phases[2].trace('out');
    const litF = lit.phases[2].trace('out');

    const lumas: number[] = [];
    bareF.forEachPixel((p: { r: number; g: number; b: number }) => lumas.push(luma(p)));
    lumas.sort((x, y) => x - y);
    const ground = lumas[Math.floor(lumas.length / 2)];   // the checkerboard

    // The extra light, projected onto the travel axis the same way a glint is
    // — because WHERE it lands is the claim. A glint is a band square across
    // that axis, so if the ghosts are taking it the difference is a band too,
    // and not a general lift.
    const diff = new Array<number>(AXIS.length).fill(0);
    let ink = 0, added = 0;
    bareF.forEachPixel((p: { r: number; g: number; b: number }, x: number, y: number) => {
      const d = Math.max(0, luma(litF.pixelAt(x, y)) - luma(p));
      ink += Math.max(0, luma(p) - ground);
      added += d;
      const sx = ((x + 0.5) / W - 0.5) / ax;
      const sy = ((y + 0.5) / H - 0.5) / ay;
      const i = Math.round((sx * R2 - sy * R2 - AXIS_LO) / AXIS_STEP);
      if (i >= 0 && i < diff.length && d > diff[i]) diff[i] = d;
    });

    // The wireframe is there in both — the throw does not need a glint.
    expect(ink).toBeGreaterThan(0);
    // ...and the glint lights a real fraction of it. (Shadow is off here, so a
    // glint can only ADD; the wake punching holes in a tube is the same field,
    // and it is covered on the tower.)
    expect(added).toBeGreaterThan(ink * 0.03);
    // ...in a BAND, which is what says it came from the glint and not from the
    // two runs having drifted apart. A glint is narrow against the whole axis,
    // so its peak stands far clear of the median.
    const sorted = [...diff].sort((x, y) => x - y);
    const mid = sorted[Math.floor(sorted.length / 2)];
    expect(Math.max(...diff)).toBeGreaterThan(mid * 8 + 20);
  });

  it('a lit plane holds its own ring down while the ring is still on it',
     async () => {
    // FAKED LOCAL CONTRAST. Sweeping straight back after a throw relights the
    // tower UNDERNEATH a ring that has barely left it — two bright things in
    // the same place, which read as one bright thing and lose the ring. So a
    // lit plane damps its own ring, and only while the ring is still close.
    const mk = (id: string, lit: number, release: number) =>
      runEngineMultiPhaseTest({
        width: W, height: H, modules: MODULES,
        phases: [
          { commands: [
              { type: 'createSketch', sketchId: id, sketch: ringSketch({
                  release, plane1_emission: lit, plane2_emission: lit,
                  plane3_emission: lit }) },
              { type: 'setTracePoints', tracePoints: [
                  { id: 'out', target: { type: 'sketch_output', sketchId: id } }] },
            ],
            // Two settled frames, so the one-frame hold below is long over.
            waitFrames: 12, captureTraceIds: ['out'] },
        ],
        dumpName: id,
      });

    /** Light in the frame, over and above whatever the tower itself is doing. */
    const litSum = async (id: string, lit: number, rel: number) => {
      // Sequentially: the engine runner drives one page, and two at once
      // abort each other.
      const withRing = await mk(`${id}_on`, lit, rel);
      const without = await mk(`${id}_off`, lit, 0);
      expect(withRing.success && without.success).toBe(true);
      let sum = 0;
      withRing.phases[0].trace('out').forEachPixel(
        (p: { r: number; g: number; b: number }, x: number, y: number) => {
          sum += luma(p) - luma(without.phases[0].trace('out').pixelAt(x, y));
        });
      return sum;
    };

    // A ring still sitting on the stack, over a dark tower and over a lit one.
    const onDark = await litSum('damp_close_dark', 0, 0.95);
    const onLit = await litSum('damp_close_lit', 1, 0.95);
    expect(onDark).toBeGreaterThan(0);
    expect(onLit).toBeLessThan(onDark * 0.6);

    // ...and the same comparison once it has flown clear: the damping is gone,
    // so the tower's brightness stops mattering.
    const farDark = await litSum('damp_far_dark', 0, 0.25);
    const farLit = await litSum('damp_far_lit', 1, 0.25);
    expect(farDark).toBeGreaterThan(0);
    expect(farLit).toBeGreaterThan(farDark * 0.8);
  });

});
