import { runEngineTest } from '@nano/web/test/engine-test-helpers';
import type { Sketch } from '@nano/web/src/sketch-types';

/**
 * `led_out` — the pixel map for the house rig, published by both neon-quad
 * instruments. Four vertical bars, ten segments each, as a 4x10 grid of flat
 * blocks.
 *
 * ENGINE HARNESS, not the per-effect one: this is a SECONDARY texture output,
 * and the only way to look at one is chroma_wave's sidechannel trick — wire it
 * into a `util.sidechannel_out` override, put a `util.sidechannel_in` after it,
 * and the sketch output IS the map. That makes these WebGPU-only. The mapping
 * arithmetic itself is pinned host-free in native/tests/test_led_bars.cpp,
 * which runs on the same header the effects call; nothing here re-derives it.
 *
 * What only a real sketch can show is what is here: that the grid lands the
 * right way up and the right way round on an actual texture, that the blocks
 * are FLAT, and that the pass is not drawn at all when nobody is listening.
 *
 * Engine dt is wall clock, so nothing below rides on timing. three_walls is
 * frozen into an exact pose with a rate-0 gate (the trick three_walls.test.ts
 * uses), and three_planes has no clock in this path at all.
 */

const MODULES = ['com.nano.lights', 'com.nano.core'];

/** Mean colour of a rectangle of the trace. */
function meanOf(f: any, x0: number, x1: number, y0: number, y1: number) {
  let r = 0, g = 0, b = 0, n = 0;
  for (let y = y0; y < y1; y++)
    for (let x = x0; x < x1; x++) {
      const p = f.pixelAt(x, y);
      r += p.r; g += p.g; b += p.b; n++;
    }
  return { r: r / n, g: g / n, b: b / n };
}

const lumaOf = (c: { r: number; g: number; b: number }) => (c.r + c.g + c.b) / 3;

describe('LED bars — three_planes', () => {
  jest.setTimeout(90000);

  // 10 segments over 200 rows is exactly 20 rows each, so every boundary lands
  // on a whole pixel and the assertions can sit either side of one.
  const W = 240, H = 200;
  const SEG = H / 10;      // rows per segment
  const BAR = W / 4;       // columns per bar

  /** The image row at the middle of segment `seg`, counting from the BOTTOM. */
  const rowOfSegment = (seg: number) => Math.round(H - (seg + 0.5) * SEG);

  const build = (params: Record<string, unknown>, wired = true): Sketch => ({
    anchor: null,
    wires: wired ? [{ id: 'lw', src: { instanceKey: 'tp@0', field: 'led_out' },
                      dest: { instanceKey: 'send@0', field: 'send_in' } }] : [],
    chain: [
      { type: 'module', module_type: 'source.solid_color', instance_key: 'bg@0',
        params: { color: [0, 0, 0] } },
      { type: 'module', module_type: 'source.mesh.three_planes', instance_key: 'tp@0',
        params: {
          grain: 0, scanline: 0, chroma_bleed: 0,
          // Primaries per floor, so a segment showing the wrong one is not a
          // near miss but a different colour entirely.
          plane1_color: [1, 0, 0], plane2_color: [0, 1, 0], plane3_color: [0, 0, 1],
          plane1_emission: 1, plane2_emission: 1, plane3_emission: 1,
          ...params,
        } },
      { type: 'module', module_type: 'util.sidechannel_out', instance_key: 'send@0',
        params: { channel: 3 } },
      { type: 'module', module_type: 'util.sidechannel_in', instance_key: 'recv@0',
        params: { channel: 3 } },
    ],
  } as Sketch);

  const view = (id: string, params: Record<string, unknown>, wired = true) =>
    runEngineTest({
      width: W, height: H, modules: MODULES,
      commands: [
        { type: 'createSketch', sketchId: id, sketch: build(params, wired) },
        { type: 'setTracePoints', tracePoints: [
          { id: 'out', target: { type: 'sketch_output', sketchId: id } }]},
      ],
      waitFrames: 20, captureTraceIds: ['out'], dumpName: id,
    });

  it('stacks the three floors up the bar, three / three / four', async () => {
    const r = await view('led_tp_split', {});
    expect(r.success).toBe(true);
    const f = r.trace('out');

    // Bottom three segments are the bottom floor, in red.
    for (const seg of [0, 1, 2]) f.expectPixelAt(30, rowOfSegment(seg), { r: 255, g: 0, b: 0 }, 8);
    // The next three are the middle floor, in green.
    for (const seg of [3, 4, 5]) f.expectPixelAt(30, rowOfSegment(seg), { r: 0, g: 255, b: 0 }, 8);
    // And the top FOUR are the top floor, in blue. The spare segment lands
    // here, which is the one thing about this split that no render can show
    // you by looking plausible.
    for (const seg of [6, 7, 8, 9]) f.expectPixelAt(30, rowOfSegment(seg), { r: 0, g: 0, b: 255 }, 8);
  });

  it('every bar shows the same tower', async () => {
    const r = await view('led_tp_bars', {});
    expect(r.success).toBe(true);
    const f = r.trace('out');
    for (const seg of [0, 4, 9]) {
      const y = rowOfSegment(seg);
      const first = meanOf(f, 4, BAR - 4, y, y + 1);
      for (let bar = 1; bar < 4; bar++) {
        const c = meanOf(f, bar * BAR + 4, (bar + 1) * BAR - 4, y, y + 1);
        expect(Math.abs(c.r - first.r)).toBeLessThan(4);
        expect(Math.abs(c.g - first.g)).toBeLessThan(4);
        expect(Math.abs(c.b - first.b)).toBeLessThan(4);
      }
    }
  });

  it('a dark floor is dark, and its neighbours are untouched', async () => {
    const r = await view('led_tp_hole', { plane2_emission: 0 });
    expect(r.success).toBe(true);
    const f = r.trace('out');
    for (const seg of [3, 4, 5]) f.expectPixelAt(30, rowOfSegment(seg), { r: 0, g: 0, b: 0 }, 6);
    // The segments either side of the hole keep their own colour at full: the
    // cells are independent, so nothing bleeds across the boundary.
    f.expectPixelAt(30, rowOfSegment(2), { r: 255, g: 0, b: 0 }, 8);
    f.expectPixelAt(30, rowOfSegment(6), { r: 0, g: 0, b: 255 }, 8);
  });

  it('a dimmed floor keeps its hue and loses its level', async () => {
    // ledLevel is the picture's own dimmer curve, normalised: emission^1.8.
    // At 0.6 that is 0.398, so the segment is that fraction of full red.
    const r = await view('led_tp_dim', { plane1_emission: 0.6 });
    expect(r.success).toBe(true);
    const c = meanOf(r.trace('out'), 10, BAR - 10,
                     rowOfSegment(1) - 4, rowOfSegment(1) + 4);
    expect(c.r).toBeGreaterThan(80);
    expect(c.r).toBeLessThan(125);
    expect(c.g).toBeLessThan(6);
    expect(c.b).toBeLessThan(6);
  });

  it('the blocks are flat, and the boundary between them is one pixel wide',
     async () => {
    // The whole point of the map. A consumer sampling anywhere inside a
    // segment has to come away with that segment's exact colour, so the
    // boundary at row 0.4H must be a step and not a ramp.
    const r = await view('led_tp_quantized', {});
    expect(r.success).toBe(true);
    const f = r.trace('out');
    // The top floor is FOUR segments, so it owns the upper 40% of the image
    // and the boundary under it sits at row 4 * SEG. Getting that wrong is
    // exactly the 3/3/4 mistake this file exists to catch.
    const edge = 4 * SEG;
    f.expectPixelAt(30, edge - 2, { r: 0, g: 0, b: 255 }, 8);   // still top
    f.expectPixelAt(30, edge + 2, { r: 0, g: 255, b: 0 }, 8);   // already middle
    // And flat well inside each: no gradient anywhere in the block.
    const a = meanOf(f, 10, BAR - 10, edge - 18, edge - 14);
    const b = meanOf(f, 10, BAR - 10, edge - 6, edge - 2);
    expect(Math.abs(lumaOf(a) - lumaOf(b))).toBeLessThan(4);
  });

  it('turning quantize off smears the boundary, which is why it is on',
     async () => {
    const r = await view('led_tp_smooth', { led_quantize: false });
    expect(r.success).toBe(true);
    const f = r.trace('out');
    const edge = 4 * SEG;
    // Two rows the quantised map keeps pure now carry both colours, because
    // they sit between two segment CENTRES.
    const c = f.pixelAt(30, edge + 2);
    expect(c.b).toBeGreaterThan(60);
    expect(c.g).toBeGreaterThan(60);
  });

  // --- Solid Mix -----------------------------------------------------------
  //
  // The bars are a fixture standing in the room, not a picture of the picture.
  // Three Planes Rig drives the six LED Source rails with its METER whatever
  // the screen is doing, and this knob is how much of each end reaches them.
  // The rails are rotated one floor against the planes below, so a segment
  // showing the wrong source is a different colour rather than a near miss.
  const SOURCES = {
    led1_color: [0, 0, 1], led2_color: [1, 0, 0], led3_color: [0, 1, 0],
    led1_emission: 1, led2_emission: 1, led3_emission: 1,
  };

  it('Solid Mix 1 keeps the bars on the picture', async () => {
    // The default, and the whole of the old behaviour: the sources are wired
    // and loud, and they change nothing.
    const r = await view('led_tp_mix_solid', { ...SOURCES, led_solid: 1 });
    expect(r.success).toBe(true);
    const f = r.trace('out');
    f.expectPixelAt(30, rowOfSegment(1), { r: 255, g: 0, b: 0 }, 8);
    f.expectPixelAt(30, rowOfSegment(4), { r: 0, g: 255, b: 0 }, 8);
    f.expectPixelAt(30, rowOfSegment(8), { r: 0, g: 0, b: 255 }, 8);
  });

  it('Solid Mix 0 hands the bars over to the LED sources', async () => {
    const r = await view('led_tp_mix_led', { ...SOURCES, led_solid: 0 });
    expect(r.success).toBe(true);
    const f = r.trace('out');
    // Each floor now wears its SOURCE colour, and the picture's is gone.
    f.expectPixelAt(30, rowOfSegment(1), { r: 0, g: 0, b: 255 }, 8);
    f.expectPixelAt(30, rowOfSegment(4), { r: 255, g: 0, b: 0 }, 8);
    f.expectPixelAt(30, rowOfSegment(8), { r: 0, g: 255, b: 0 }, 8);
  });

  it('half way carries both, per floor', async () => {
    const r = await view('led_tp_mix_half', { ...SOURCES, led_solid: 0.5 });
    expect(r.success).toBe(true);
    const f = r.trace('out');
    // The bottom floor is red in the picture and blue on the rails, so half
    // way is both at half — and no green anywhere near it.
    const c = meanOf(f, 10, BAR - 10, rowOfSegment(1) - 4, rowOfSegment(1) + 4);
    expect(c.r).toBeGreaterThan(100);
    expect(c.r).toBeLessThan(155);
    expect(c.b).toBeGreaterThan(100);
    expect(c.b).toBeLessThan(155);
    expect(c.g).toBeLessThan(6);
  });

  it('a dark source pulls its floor down without touching the others',
     async () => {
    // The mix carries the LEVEL as well as the colour, and it is per floor: a
    // meter that has fallen off the top floor takes those segments down with
    // it and leaves the ones below at whatever the picture is holding.
    //
    // The top floor's source is given the picture's own blue here, so the only
    // thing left to read on it is the level. Everywhere else the two ends are
    // still different colours, which is what the second half checks.
    const r = await view('led_tp_mix_level',
                         { ...SOURCES, led3_color: [0, 0, 1], led3_emission: 0,
                           led_solid: 0.5 });
    expect(r.success).toBe(true);
    const f = r.trace('out');
    // Top floor: blue either way, and half as bright because one end is dark.
    const top = meanOf(f, 10, BAR - 10, rowOfSegment(8) - 4, rowOfSegment(8) + 4);
    expect(top.b).toBeGreaterThan(100);
    expect(top.b).toBeLessThan(155);
    expect(top.g).toBeLessThan(6);
    // Middle floor is untouched by any of that: green in the picture, red on
    // the rails, both at full level, so half way is half of each.
    const mid = meanOf(f, 10, BAR - 10, rowOfSegment(4) - 4, rowOfSegment(4) + 4);
    expect(mid.g).toBeGreaterThan(100);
    expect(mid.r).toBeGreaterThan(100);
  });

  it('draws nothing at all when nobody is wired to it', async () => {
    const wired = await view('led_tp_wired', {});
    const unwired = await view('led_tp_unwired', {}, false);
    expect(wired.success && unwired.success).toBe(true);
    // Wired, the sketch output IS the map: the top-left corner is the top
    // floor at full blue. Unwired, the send publishes its chain input instead
    // — three_planes' own picture, which is near black out at the corner.
    wired.trace('out').expectPixelAt(6, 6, { r: 0, g: 0, b: 255 }, 10);
    const corner = unwired.trace('out').pixelAt(6, 6);
    expect(corner.b).toBeLessThan(60);
  });
});

describe('LED bars — three_walls', () => {
  jest.setTimeout(120000);

  const W = 240, H = 160;
  const BAR = W / 4;

  // A rate-0 resonate gate freezes the three frames at phases 0, 1/3 and 2/3
  // forever, so every case samples an EXACT pose however long the wall clock
  // took to get there (the trick three_walls.test.ts leans on).
  //
  // At the default depth range that puts the nearest frame at an apparent size
  // of 1.7705 back-wall half-widths. A ring stands at 1 + spacing * (ring + 1),
  // so a spacing of 0.77 parks the INNER pair exactly on it and leaves the
  // outer pair 0.77 further out — more than two pass-widths away, which is
  // nothing.
  const FROZEN = { resonate: 1, resonate_f0: 0, resonate_f1: 0 };
  const LIT = {
    ...FROZEN,
    quad1_color: [0.2, 0.9, 0.4], quad2_color: [0.2, 0.9, 0.4],
    quad3_color: [0.2, 0.9, 0.4],
    quad1_emission: 1, quad2_emission: 1, quad3_emission: 1,
  };

  const build = (params: Record<string, unknown>, wired = true): Sketch => ({
    anchor: null,
    wires: wired ? [{ id: 'lw', src: { instanceKey: 'tw@0', field: 'led_out' },
                      dest: { instanceKey: 'send@0', field: 'send_in' } }] : [],
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

  const view = (id: string, params: Record<string, unknown>, wired = true) =>
    runEngineTest({
      width: W, height: H, modules: MODULES,
      commands: [
        { type: 'createSketch', sketchId: id, sketch: build(params, wired) },
        { type: 'setTracePoints', tracePoints: [
          { id: 'out', target: { type: 'sketch_output', sketchId: id } }]},
      ],
      waitFrames: 25, captureTraceIds: ['out'], dumpName: id,
    });

  /** Mean colour of bar `i`, sampled well inside it. */
  const bar = (f: any, i: number) =>
    meanOf(f, i * BAR + 8, (i + 1) * BAR - 8, 8, H - 8);

  it('the rig is dark until a move is running', async () => {
    // three_walls' rest pose is nothing at all — no frames in the room, so
    // nothing for the bars to see.
    const r = await view('led_tw_idle', {});
    expect(r.success).toBe(true);
    for (let i = 0; i < 4; i++) expect(lumaOf(bar(r.trace('out'), i))).toBeLessThan(6);
  });

  it('a frame on the inner ring lights the inner pair and not the outer',
     async () => {
    const r = await view('led_tw_inner', { ...LIT, led_spacing: 0.77 });
    expect(r.success).toBe(true);
    const f = r.trace('out');
    // Bars 1 and 2 are the pair nearest the screen (the columns run
    // outer-inner | inner-outer across the room).
    expect(lumaOf(bar(f, 1))).toBeGreaterThan(80);
    expect(lumaOf(bar(f, 2))).toBeGreaterThan(80);
    expect(lumaOf(bar(f, 0))).toBeLessThan(12);
    expect(lumaOf(bar(f, 3))).toBeLessThan(12);
    // And they carry the frame's own colour, not a white key.
    const c = bar(f, 1);
    expect(c.g).toBeGreaterThan(c.r * 2);
    expect(c.g).toBeGreaterThan(c.b * 1.5);
  });

  it('the spacing knob moves the rings, and the outer pair takes the pass',
     async () => {
    // Half the spacing and the OUTER ring is the one standing at 1.77.
    const r = await view('led_tw_outer', { ...LIT, led_spacing: 0.385 });
    expect(r.success).toBe(true);
    const f = r.trace('out');
    expect(lumaOf(bar(f, 0))).toBeGreaterThan(80);
    expect(lumaOf(bar(f, 3))).toBeGreaterThan(80);
    expect(lumaOf(bar(f, 1))).toBeLessThan(12);
    expect(lumaOf(bar(f, 2))).toBeLessThan(12);
  });

  it('the two sides agree', async () => {
    const r = await view('led_tw_mirror', { ...LIT, led_spacing: 0.77 });
    expect(r.success).toBe(true);
    const f = r.trace('out');
    // The room is symmetric about its centre line and the frames are axis
    // aligned, so there is nothing that could tell the halves apart.
    expect(Math.abs(lumaOf(bar(f, 0)) - lumaOf(bar(f, 3)))).toBeLessThan(4);
    expect(Math.abs(lumaOf(bar(f, 1)) - lumaOf(bar(f, 2)))).toBeLessThan(4);
  });

  it('a bar is one colour all the way up', async () => {
    const r = await view('led_tw_flat', { ...LIT, led_spacing: 0.77 });
    expect(r.success).toBe(true);
    const f = r.trace('out');
    // Ten segments, and every one of them the same: the frames are axis
    // aligned and so are the bars, so a vertical bar sees the same thing at
    // every height. The segments are there because the rig has ten.
    const top = meanOf(f, BAR + 8, 2 * BAR - 8, 4, 12);
    const bottom = meanOf(f, BAR + 8, 2 * BAR - 8, H - 12, H - 4);
    expect(Math.abs(lumaOf(top) - lumaOf(bottom))).toBeLessThan(4);
  });

  it('draws nothing at all when nobody is wired to it', async () => {
    const wired = await view('led_tw_wired', { ...LIT, led_spacing: 0.77 });
    const unwired = await view('led_tw_unwired', { ...LIT, led_spacing: 0.77 }, false);
    expect(wired.success && unwired.success).toBe(true);
    // Wired, bar 1's column is a solid block of the frame's colour. Unwired,
    // the send publishes the chain image instead — three_walls' own back wall,
    // which at rate 0 has the near frame's outline in it and is nowhere near
    // a flat column.
    const w = bar(wired.trace('out'), 1);
    const u = bar(unwired.trace('out'), 1);
    expect(lumaOf(w)).toBeGreaterThan(80);
    expect(Math.abs(lumaOf(w) - lumaOf(u))).toBeGreaterThan(20);
  });
});

/**
 * The editor's own view of the new port. A module's texture OUTPUT is whichever
 * texture output its schema declares FIRST (schema-channels.ts, firstFieldOfType,
 * which sorts on declaration order) — so an aux output declared ahead of
 * `tex_out` silently becomes THE output and the chain's picture stops there.
 * Nothing about a render shows you that, which is why it is asserted here: the
 * same trap that hid `mask_in` (three_planes_mask.test.ts) reads the other way
 * round for an output.
 */
describe('LED bars port', () => {
  jest.setTimeout(60000);
  const BASE = process.env.GPU_TEST_BASE_URL || 'http://localhost:5173';
  const WALK = `function* walk(root){for(const el of root.querySelectorAll('*')){yield el; if(el.shadowRoot) yield* walk(el.shadowRoot);}}`;

  it('shows LED Out on both cards, and leaves tex_out as the chain output',
     async () => {
    page.removeAllListeners('console');
    await page.goto(`${BASE}/resolume/index.html?playground`, { waitUntil: 'networkidle0' });
    await new Promise(r => setTimeout(r, 3000));
    await page.evaluate(`(async () => {
      const ac = window.appController;
      ac.mutate('s', d => {
        d.sketches['sk_led_port'] = { anchor: null,
          chain: [
            { type: 'module', module_type: 'source.mesh.three_planes', instance_key: 'tp@0' },
            { type: 'module', module_type: 'source.mesh.three_walls', instance_key: 'tw@0' },
          ],
          wires: [],
          instances: {
            'tp@0': { module_type: 'source.mesh.three_planes', state: {} },
            'tw@0': { module_type: 'source.mesh.three_walls', state: {} },
          } };
      });
      ac.setActiveTab('edit');
      ac.editSketch('sk_led_port');
    })()`);
    await new Promise(r => setTimeout(r, 2500));

    const info: any = await page.evaluate(`(() => {
      const plugins = (window.appState?.local?.engine?.plugins)
                   || (window.appState?.local?.plugins) || [];
      const texOf = (id) => {
        const p = plugins.find(x => (x.id || x.module_type) === id);
        const sch = p && (p.schema || p.fields);
        return sch ? Object.entries(sch)
          .filter(([, d]) => d && d.type === 'texture')
          .map(([n, d]) => ({ name: n, io: d.io, order: d.order })) : [];
      };
      ${WALK}
      const labels = [];
      for (const el of walk(document)) {
        const t = (el.textContent || '').trim();
        if (el.children.length === 0 && t.length < 30) labels.push(t);
      }
      return { planes: texOf('source.mesh.three_planes'),
               walls: texOf('source.mesh.three_walls'),
               labels: [...new Set(labels)] };
    })()`);

    const outs = (tex: any[]) => tex.filter((f: any) => (f.io & 2) !== 0)
                                    .sort((a: any, b: any) => a.order - b.order);

    // tex_out first, then the aux outputs — three_planes has grown a pair of
    // impact-light walls since, and the only thing that matters is that none
    // of them got in front of the chain's own picture.
    const planeOuts = outs(info.planes);
    expect(planeOuts[0].name).toBe('tex_out');
    expect(planeOuts.map((f: any) => f.name)).toContain('led_out');

    const wallOuts = outs(info.walls);
    // tex_out first, and led_out after the two side cameras.
    expect(wallOuts[0].name).toBe('tex_out');
    expect(wallOuts.map((f: any) => f.name)).toContain('led_out');
    expect(wallOuts[wallOuts.length - 1].name).toBe('led_out');

    // ...and the port is on the card, where it can be wired.
    expect(info.labels).toContain('LED Out');
  });
});
