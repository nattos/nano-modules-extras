import { runEngineTest } from '@nano/web/test/engine-test-helpers';
import type { Sketch } from '@nano/web/src/sketch-types';

/**
 * `left_out` / `right_out` — the impact light `source.mesh.three_planes` throws
 * onto the walls of the room it stands in.
 *
 * A LIGHT MODEL, not a second picture of the tower: nothing in the pass draws a
 * quad. Each floor is FOUR TUBES in the room, turned by the orbit, so what
 * lands depends on how the ring is facing: square on, the near edge does all
 * the work and lays a flat bar; turned off it, one corner is nearest and the
 * pool leans that way. Everything below is checking that the SHAPE is the one
 * the geometry gives rather than an authored gradient, because that is the
 * whole difference between light on a wall and bad lighting in an old game.
 *
 * Most cases pin the orbit SQUARE ON, where the shape is simple enough to make
 * claims about. The orbit's own effect gets its own cases at the bottom.
 *
 * Engine harness, because these are secondary texture outputs and only a wire
 * can reach one (chroma_wave's sidechannel trick). That makes them WebGPU
 * only; the Metal side of the same pass is pinned in
 * native/tests/test_effect_render.cpp.
 *
 * The GLINTS also light the walls, and because they cross the room rather than
 * the picture they reach the two walls at different moments — which is the one
 * thing that makes the two outputs different pictures. That is deliberately NOT
 * asserted here: a glint is a particle born from knob MOTION, so a wall-clock
 * engine run cannot hold one still. What is asserted is the symmetry it breaks
 * — with the glimmer off, the two walls agree exactly.
 */
describe('Three Planes impact light', () => {
  jest.setTimeout(120000);

  const W = 480, H = 270;
  const MODULES = ['com.nano.core', 'com.nano.lights'];

  // Where a floor's pool sits. The elevation tilts each ring about its OWN
  // centre, so a centre lands at height y*cos(phi) — the picture's own height,
  // which is what keeps the three outputs one room — and on the vertical axis
  // at any elevation. That last part is the deliberate cheat: tilting the stack
  // as one body would carry the upper floors off along the wall by y*sin(phi)
  // and string the pools out diagonally.
  const SPACING = 0.42, ZOOM = 0.55;
  const ELEV = 35.264389682754654;
  const ASPECT_X = Math.max(W, H) / (2 * W);
  const ASPECT_Y = Math.max(W, H) / (2 * H);
  const floorAt = (i: number, elevDeg = ELEV) => {
    const y0 = (i - 1) * SPACING * ZOOM;
    return {
      x: Math.round(W / 2),
      y: Math.round((-y0 * Math.cos((elevDeg * Math.PI) / 180) * ASPECT_Y + 0.5) * H),
    };
  };

  const luma = (p: { r: number; g: number; b: number }) => (p.r + p.g + p.b) / 3;
  // How bright the light IS, rather than how bright a grey of the same value
  // would be. The floors here are pure primaries, so a luma reading is a third
  // of the level and every threshold written against it would be a lie.
  const level = (p: { r: number; g: number; b: number }) =>
    Math.max(p.r, Math.max(p.g, p.b));

  const build = (field: string, params: Record<string, unknown>,
                 wired = true): Sketch => ({
    anchor: null,
    wires: wired ? [{ id: 'ww', src: { instanceKey: 'tp@0', field },
                      dest: { instanceKey: 'send@0', field: 'send_in' } }] : [],
    chain: [
      { type: 'module', module_type: 'source.solid_color', instance_key: 'bg@0',
        params: { color: [0, 0, 0] } },
      { type: 'module', module_type: 'source.mesh.three_planes', instance_key: 'tp@0',
        params: {
          grain: 0, scanline: 0, chroma_bleed: 0,
          // A primary per floor, so a bar in the wrong place is a different
          // colour rather than a near miss.
          plane1_color: [1, 0, 0], plane2_color: [0, 1, 0], plane3_color: [0, 0, 1],
          plane1_emission: 1, plane2_emission: 1, plane3_emission: 1,
          // No glimmer: glints are particles born from knob motion, and a
          // wall-clock run cannot hold one still.
          glimmer_gain: 0, glimmer_chaos: 0,
          // Square on unless a case says otherwise — the ring's near edge then
          // faces the wall outright and lays the simple bar these claims are
          // about. (The card's own default is 45, a corner pointing at each
          // wall, which is a different picture entirely.)
          orbit_azimuth: 0,
          ...params,
        } },
      { type: 'module', module_type: 'util.sidechannel_out', instance_key: 'send@0',
        params: { channel: 3 } },
      { type: 'module', module_type: 'util.sidechannel_in', instance_key: 'recv@0',
        params: { channel: 3 } },
    ],
  } as Sketch);

  const view = (id: string, field: string, params: Record<string, unknown>,
                wired = true) =>
    runEngineTest({
      width: W, height: H, modules: MODULES,
      commands: [
        { type: 'createSketch', sketchId: id, sketch: build(field, params, wired) },
        { type: 'setTracePoints', tracePoints: [
          { id: 'out', target: { type: 'sketch_output', sketchId: id } }]},
      ],
      waitFrames: 20, captureTraceIds: ['out'], dumpName: id,
    });

  it('lays one bar per floor, at its own height and in its own colour',
     async () => {
    const r = await view('tpw_floors', 'left_out', {});
    expect(r.success).toBe(true);
    const f = r.trace('out');

    const at = (i: number) => { const p = floorAt(i); return f.pixelAt(p.x, p.y); };
    const bottom = at(0), middle = at(1), top = at(2);

    // Each floor's own colour dominates at its own height. The bottom floor is
    // LOW on the wall, which is the one thing a sign error would flip.
    expect(bottom.r).toBeGreaterThan(bottom.g + 40);
    expect(bottom.r).toBeGreaterThan(bottom.b + 40);
    expect(middle.g).toBeGreaterThan(middle.r + 40);
    expect(middle.g).toBeGreaterThan(middle.b + 40);
    expect(top.b).toBeGreaterThan(top.r + 40);
    expect(top.b).toBeGreaterThan(top.g + 40);
  });

  it('a dark floor throws nothing', async () => {
    const r = await view('tpw_dark', 'left_out', { plane2_emission: 0 });
    expect(r.success).toBe(true);
    const f = r.trace('out');
    const m = floorAt(1), b = floorAt(0);
    // The middle pool is now only what its neighbours' bounce puts there...
    expect(luma(f.pixelAt(m.x, m.y)))
      .toBeLessThan(luma(f.pixelAt(b.x, b.y)) * 0.6);
    // ...and no green survives in it, so nothing of that floor is left.
    const mid = f.pixelAt(m.x, m.y);
    expect(mid.g).toBeLessThan(mid.r + 12);
  });

  it('lays a bar with hot ends across the ring, and nothing past it',
     async () => {
    // The shape the four tubes give, and none of it is drawn. Square on, the
    // near edge is at one distance along its whole length, so the pool is FLAT
    // across the middle — an emitter with width, not a blob centred on a point.
    // At the ends the two edges running away from the wall come close enough to
    // add, so the bar brightens into its corners before it goes. That structure
    // is the difference between this and a gradient.
    // Flat on the horizon, so the bar is horizontal and a row IS the bar. At
    // any elevation the pool slants and a row cuts across it instead.
    const r = await view('tpw_bar', 'left_out', { elevation: 0 });
    expect(r.success).toBe(true);
    const f = r.trace('out');
    const y = floorAt(2, 0).y;

    // The ring is 0.62 stack-units wide at 0.55 zoom, so its ends land about
    // 82px either side of centre.
    const mid = level(f.pixelAt(W / 2, y));
    for (const x of [W / 2 - 40, W / 2 + 40])
      expect(Math.abs(level(f.pixelAt(x, y)) - mid)).toBeLessThan(mid * 0.08);

    // The corners, hotter than the middle and on both sides of it.
    for (const x of [W / 2 - 80, W / 2 + 80])
      expect(level(f.pixelAt(x, y))).toBeGreaterThan(mid * 1.15);

    // ...and then it goes. Out at the edge of the wall there is bounce and
    // little else.
    expect(level(f.pixelAt(12, y))).toBeLessThan(mid * 0.15);
  });

  it('Distance is the size of the light, not a softness', async () => {
    // Half brightness lands about two thirds of a gap out, so the pool's own
    // height IS the distance to the wall. Nothing else in the pass sets it.
    const halfHeight = (f: any, y0: number) => {
      const peak = luma(f.pixelAt(W / 2, y0));
      for (let d = 1; d < H / 2; d++)
        if (luma(f.pixelAt(W / 2, y0 - d)) < peak * 0.5) return d;
      return H / 2;
    };
    // Flat on the horizon, so walking UP the column walks straight out of the
    // pool rather than along a slant.
    const flat = { elevation: 0, wall_bounce: 0 };
    const near = await view('tpw_near', 'left_out', { ...flat, wall_gap: 0.15 });
    const far = await view('tpw_far', 'left_out', { ...flat, wall_gap: 0.8 });
    expect(near.success && far.success).toBe(true);

    const a = halfHeight(near.trace('out'), floorAt(2, 0).y);
    const b = halfHeight(far.trace('out'), floorAt(2, 0).y);
    expect(b).toBeGreaterThan(a * 2.5);
  });

  it('a throw floods the wall', async () => {
    // The impact. Mid-flight the rings have opened out, so what lands is a wide
    // soft flare over the bars rather than three more bars — which is what it
    // looked like before the ghosts' own opening was carried out here.
    const area = (f: any, t: number) => {
      let n = 0;
      for (let y = 0; y < H; y += 2)
        for (let x = 0; x < W; x += 2) if (level(f.pixelAt(x, y)) > t) n++;
      return n;
    };
    // On a MUTED tower, which is what a throw actually is: the rig spends the
    // latched charge as the floors go dark, and a lit floor holds its own ring
    // down besides. Comparing against a blazing tower measures the tower.
    const muted = { plane1_emission: 0.15, plane2_emission: 0.15, plane3_emission: 0.15 };
    const rest = await view('tpw_rest', 'left_out', { ...muted, release: 0 });
    const thrown = await view('tpw_throw', 'left_out',
                              { ...muted, release: 0.65, release_gain: 2.2 });
    expect(rest.success && thrown.success).toBe(true);

    // A muted tower puts almost nothing on the wall; the throw floods it. Stated
    // as two absolutes rather than a ratio, because the resting figure is
    // nearly zero and a ratio against nearly zero says nothing.
    expect(area(rest.trace('out'), 40)).toBeLessThan(300);
    expect(area(thrown.trace('out'), 40)).toBeGreaterThan(2000);
    // And the flare is SOFT: it reaches well above the top floor's own bar,
    // where a resting tower puts almost nothing.
    const y = floorAt(2).y - 34;
    expect(level(thrown.trace('out').pixelAt(W / 2, y)))
      .toBeGreaterThan(level(rest.trace('out').pixelAt(W / 2, y)) + 25);
  });

  it('the two walls agree with the ring square on', async () => {
    // Square on, the ring faces both walls identically and there is nothing
    // that could tell them apart. Turned off square there is — see below.
    const l = await view('tpw_sym_l', 'left_out', {});
    const r = await view('tpw_sym_r', 'right_out', {});
    expect(l.success && r.success).toBe(true);
    // Sampled at each floor's own place on ITS wall — the depth axis runs
    // outward from the back wall on each side, so the two are mirrored in x.
    for (const i of [0, 1, 2]) {
      const p = floorAt(i);
      expect(Math.abs(luma(l.trace('out').pixelAt(p.x, p.y))
                    - luma(r.trace('out').pixelAt(p.x, p.y)))).toBeLessThan(3);
    }
  });

  it('an off-square orbit puts the pools at opposite ends of the room',
     async () => {
    // The orbit's own effect. Turned off square, one corner of the ring is
    // nearest the wall and the pool leans to it — and because a square is
    // symmetric through its centre, the corner nearest THIS wall and the one
    // nearest the far wall are at opposite ends of the room. So the two pools
    // sit at opposite depths, by the same amount.
    //
    // Read in ROOM depth rather than in pixels: each wall's picture runs
    // outward from the back wall, so the same screen side is the far end on one
    // and the near end on the other, and comparing columns compares nothing.
    const centroid = (f: any) => {
      let sum = 0, w = 0;
      for (let y = 0; y < H; y += 2)
        for (let x = 0; x < W; x += 2) {
          const v = level(f.pixelAt(x, y));
          if (v > 60) { sum += v * x; w += v; }
        }
      return w > 0 ? sum / w : W / 2;
    };
    // The MIDDLE floor alone, so the reading is one pool's lean and not three
    // slants averaged together.
    const p = { orbit_azimuth: 0.06, plane1_emission: 0, plane3_emission: 0 };
    const l = await view('tpw_orb_l', 'left_out', p);
    const r = await view('tpw_orb_r', 'right_out', p);
    expect(l.success && r.success).toBe(true);

    // Screen column back into the room's own depth. The left wall reads its
    // picture left-to-right as depth; the right wall reads it the other way.
    const depthOf = (c: number, side: 'l' | 'r') =>
      ((c / W - 0.5) / ASPECT_X) * (side === 'l' ? 1 : -1);
    const zl = depthOf(centroid(l.trace('out')), 'l');
    const zr = depthOf(centroid(r.trace('out')), 'r');

    // Each well off the middle of the room...
    expect(Math.abs(zl)).toBeGreaterThan(0.05);
    expect(Math.abs(zr)).toBeGreaterThan(0.05);
    // ...at opposite ends of it...
    expect(Math.sign(zl)).not.toBe(Math.sign(zr));
    // ...and by the same amount, which is the central symmetry of a square and
    // not a coincidence.
    expect(Math.abs(Math.abs(zl) - Math.abs(zr))).toBeLessThan(0.03);
  });

  it('Depth draws the far side in, spreading the pool', async () => {
    // The knob for when the honest geometry gets too point-like: it pulls the
    // far side of each ring toward the wall, so more of the ring lights it and
    // the pool spreads. The near side does not move, so the gap — and the
    // exposure normalised against it — stay put.
    const area = (f: any) => {
      let n = 0;
      for (let y = 0; y < H; y += 2)
        for (let x = 0; x < W; x += 2) if (level(f.pixelAt(x, y)) > 100) n++;
      return n;
    };
    const p = { orbit_azimuth: 0.06 };
    const full = await view('tpw_dep_1', 'left_out', { ...p, wall_depth: 1 });
    const some = await view('tpw_dep_h', 'left_out', { ...p, wall_depth: 0.3 });
    const none = await view('tpw_dep_0', 'left_out', { ...p, wall_depth: 0 });
    expect(full.success && some.success && none.success).toBe(true);

    const a = area(full.trace('out')), b = area(some.trace('out')),
          c = area(none.trace('out'));
    expect(b).toBeGreaterThan(a * 1.3);
    expect(c).toBeGreaterThan(b * 1.3);
  });

  it('at Depth 0 the lean is gone and the two walls agree again', async () => {
    // Flat out, every part of a ring is the same distance off whatever the
    // orbit is doing — so there is no nearest corner to lean toward, and the
    // thing that told the two walls apart is gone with it. That is the check
    // that Depth is acting on the DEPTH and not just softening everything.
    const centroid = (f: any) => {
      let sum = 0, w = 0;
      for (let y = 0; y < H; y += 2)
        for (let x = 0; x < W; x += 2) {
          const v = level(f.pixelAt(x, y));
          if (v > 60) { sum += v * x; w += v; }
        }
      return w > 0 ? sum / w : W / 2;
    };
    const p = { orbit_azimuth: 0.06, wall_depth: 0 };
    const l = await view('tpw_dep0_l', 'left_out', p);
    const r = await view('tpw_dep0_r', 'right_out', p);
    expect(l.success && r.success).toBe(true);
    // Centred, where at full depth it leant more than 20px off (above).
    expect(Math.abs(centroid(l.trace('out')) - W / 2)).toBeLessThan(6);
    expect(Math.abs(centroid(r.trace('out')) - W / 2)).toBeLessThan(6);
  });

  it('a corner pointing straight at the wall is symmetric again', async () => {
    // At 45 the ring's corner faces the wall square on, so there is nothing to
    // lean toward and the two walls agree again. Between the two the lean grows
    // and falls — this is what says the asymmetry is the GEOMETRY and not a
    // constant offset someone added to one side.
    const l = await view('tpw_45_l', 'left_out', { orbit_azimuth: 0.125 });
    const r = await view('tpw_45_r', 'right_out', { orbit_azimuth: 0.125 });
    expect(l.success && r.success).toBe(true);
    for (const i of [0, 1, 2]) {
      const p = floorAt(i);
      expect(Math.abs(level(l.trace('out').pixelAt(p.x, p.y))
                    - level(r.trace('out').pixelAt(p.x, p.y)))).toBeLessThan(4);
    }
  });

  it('flat on the horizon a ring lays a thin hard line', async () => {
    // At elevation 0 the deck is edge-on: every part of a ring is at one
    // height, so what lands is a strip barely wider than the gap. Only the
    // middle floor is lit, so the column belongs to it alone.
    const only = { plane1_emission: 0, plane2_emission: 1, plane3_emission: 0 };
    const r = await view('tpw_flat', 'left_out',
                         { ...only, elevation: 0, wall_bounce: 0 });
    expect(r.success).toBe(true);
    const f = r.trace('out');

    const c = floorAt(1, 0);
    const peak = level(f.pixelAt(c.x, c.y));
    let thick = 0;
    for (let y = 0; y < H; y++)
      if (level(f.pixelAt(c.x, y)) > peak * 0.5) thick++;
    // Thin, and hard — the pool's height is set by the gap alone, with nothing
    // about the ring spreading it. Tilted, the same cut is far wider (below).
    expect(thick).toBeLessThan(34);
    expect(peak).toBeGreaterThan(130);
  });

  it('elevation tilts the ring, so the pool slants at tan of it', async () => {
    // The picture at elevation is a picture of a TILTED DECK — a raised camera
    // over a flat stack and a level camera on a tilted one draw the same main
    // output. Only the second has anything for a side wall to see, and what it
    // sees is the ring's near edge running uphill at exactly tan(phi).
    //
    // In pixels that IS the slope: the aspect that turns cover-square into
    // rows cancels against the one that turns it into columns.
    const only = { plane1_emission: 0, plane2_emission: 1, plane3_emission: 0 };
    const ridge = (f: any, x: number) => {
      let best = -1, at = 0;
      for (let y = 0; y < H; y++) {
        const v = level(f.pixelAt(x, y));
        if (v > best) { best = v; at = y; }
      }
      return at;
    };
    const slope = (f: any) =>
      (ridge(f, W / 2 - 30) - ridge(f, W / 2 + 30)) / 60;

    const flat = await view('tpw_tilt_0', 'left_out',
                            { ...only, elevation: 0, wall_bounce: 0 });
    const steep = await view('tpw_tilt_60', 'left_out',
                             { ...only, elevation: 60, wall_bounce: 0 });
    expect(flat.success && steep.success).toBe(true);

    // Level deck, level pool.
    expect(Math.abs(slope(flat.trace('out')))).toBeLessThan(0.15);
    // ...and at 60 degrees it climbs at tan 60.
    const want = Math.tan((60 * Math.PI) / 180);
    expect(slope(steep.trace('out'))).toBeGreaterThan(want * 0.8);
    expect(slope(steep.trace('out'))).toBeLessThan(want * 1.25);
  });

  it('keeps the floors in one vertical column at any elevation', async () => {
    // THE CHEAT, and the thing a later correctness pass would quietly undo.
    // Each ring tilts about its own centre rather than the stack tilting as one
    // body, so the centres stay on the vertical however far over the deck goes.
    // Tilted honestly, the top floor would sit at a depth of -y*sin(phi) — at
    // 60 degrees that is a fifth of the room, some 48px off centre here — and
    // the three pools would string out diagonally instead of stacking.
    const only = { plane1_emission: 0, plane2_emission: 0, plane3_emission: 1 };
    const centroidX = (f: any) => {
      let sum = 0, w = 0;
      for (let y = 0; y < H; y += 2)
        for (let x = 0; x < W; x += 2) {
          const v = level(f.pixelAt(x, y));
          if (v > 60) { sum += v * x; w += v; }
        }
      return w > 0 ? sum / w : -1;
    };
    for (const elevation of [0, 30, 60]) {
      const r = await view(`tpw_col_${elevation}`, 'left_out',
                           { ...only, elevation, wall_bounce: 0 });
      expect(r.success).toBe(true);
      expect(Math.abs(centroidX(r.trace('out')) - W / 2)).toBeLessThan(8);
    }
  });

  it('a floor still lands at the height it is drawn at', async () => {
    // The tilt is the same rotation the picture is drawn through, so a ring's
    // CENTRE comes out at y * cos(phi) — exactly where the picture puts that
    // floor. That is what keeps a triptych of left / main / right one room, and
    // with the centres pinned on the vertical it holds at any elevation and in
    // the same column.
    const r = await view('tpw_heights', 'left_out', { elevation: 60 });
    expect(r.success).toBe(true);
    const f = r.trace('out');
    const pb = floorAt(0, 60), pt = floorAt(2, 60);
    const bottom = f.pixelAt(pb.x, pb.y);
    const top = f.pixelAt(pt.x, pt.y);
    expect(bottom.r).toBeGreaterThan(bottom.b + 40);
    expect(top.b).toBeGreaterThan(top.r + 40);
  });

  it('draws nothing at all when nobody is wired to it', async () => {
    const wired = await view('tpw_wired', 'left_out', {});
    const unwired = await view('tpw_unwired', 'left_out', {}, false);
    expect(wired.success && unwired.success).toBe(true);
    // Wired, the sketch output IS the wall. Unwired, the send publishes its
    // chain input instead — the tower's own picture, which at the top floor's
    // height is nothing like a flat bar across the frame.
    const p = floorAt(2);
    expect(level(wired.trace('out').pixelAt(p.x, p.y))).toBeGreaterThan(150);
    // Unwired, the send publishes its chain input instead — the tower's own
    // picture, which is a different image throughout rather than a dimmer one.
    let differing = 0, n = 0;
    for (let yy = 0; yy < H; yy += 4)
      for (let xx = 0; xx < W; xx += 4, n++)
        if (Math.abs(luma(wired.trace('out').pixelAt(xx, yy))
                   - luma(unwired.trace('out').pixelAt(xx, yy))) > 24) differing++;
    expect(differing).toBeGreaterThan(n * 0.05);
  });
});
