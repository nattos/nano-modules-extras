import { runGpuEffectTest, Frame, forEachBackend } from '@nano/web/test/gpu-test-helpers';

// Per-effect tests for source.mesh.three_planes — three isometric quads
// stacked vertically, shaded from an exact SDF in one fullscreen pass.
//
// The effect is stateless (every envelope lives outside it), so a single
// render is enough for every case; no renderEachTick, no tick counting.
//
// Geometry the assertions lean on, with the defaults below:
//   plane_y[i] = -(i - 1) * spacing * cos(elevation) * zoom
//   half_h     =  sin(elevation) * zoom * size * (|sin(az)| + |cos(az)|)
// Both are closed forms in the effect (main.cpp `projectPlanes`), so the
// tests recompute them here rather than hard-coding pixel rows.

forEachBackend((backend) => {
describe(`Three Planes E2E (${backend})`, () => {
  jest.setTimeout(60000);

  const W = 160, H = 108;
  const MODULE = 'source.mesh.three_planes';
  const BUNDLE = 'lights' as const;

  // Defaults mirrored from main.cpp's State.
  const SPACING = 0.42, ZOOM = 0.55, SIZE = 0.62;
  const ELEV_DEG = 35.264389682754654;

  const planeY = (i: number) =>
    -((i - 1) * SPACING) * Math.cos((ELEV_DEG * Math.PI) / 180) * ZOOM;
  const halfH = (azimuth: number) => {
    const th = azimuth * 2 * Math.PI;
    return Math.sin((ELEV_DEG * Math.PI) / 180) * ZOOM * SIZE *
           (Math.abs(Math.sin(th)) + Math.abs(Math.cos(th)));
  };

  // Cover-square coords -> pixel. Mirrors fx::coverSquare / nano_coords.hlsl.
  const ax = Math.max(W, H) / (2 * W);
  const ay = Math.max(W, H) / (2 * H);
  const toPx = (sx: number, sy: number): [number, number] => [
    Math.round((sx * ax + 0.5) * W),
    Math.round((sy * ay + 0.5) * H),
  ];

  const luma = (p: { r: number; g: number; b: number }) => (p.r + p.g + p.b) / 3;
  const meanRows = (f: Frame, y0: number, y1: number) => {
    let s = 0, n = 0;
    for (let y = y0; y < y1; y++)
      for (let x = 0; x < W; x++) { s += luma(f.pixelAt(x, y)); n++; }
    return n > 0 ? s / n : 0;
  };

  // A quiet grade: no grain / scanlines, so assertions are about the geometry
  // and the resolve rather than about the analogue tail.
  const QUIET: [string, number][] = [
    ['grain', 0], ['scanline', 0], ['chroma_bleed', 0],
  ];

  it('declares metadata and its published rails', async () => {
    const frame = await runGpuEffectTest({
      module: MODULE, bundle: BUNDLE, inputColor: [0, 0, 0, 1],
      dumpName: 'three_planes_metadata',
    });
    expect(frame.success).toBe(true);
    expect(frame.metadata?.id).toBe(MODULE);
  });

  // The highlight tint is the shared nano_vcr knob (see vcr_halo.test.ts for
  // its exact behaviour); this only checks it is wired through here — the
  // white-hot line cores must take the tint while the halos keep their own
  // per-plane colour.
  it('the highlight tint colours the white-hot line cores', async () => {
    const mk = (amount: number, name: string) => runGpuEffectTest({
      module: MODULE, bundle: BUNDLE, width: W, height: H,
      inputColor: [0, 0, 0, 1],
      params: [...QUIET, ['highlight_tint_amount', amount],
               ['highlight_tint', [0.10, 0.85, 0.25]]] as any,
      dumpName: name,
    });
    const off = await mk(0, 'three_planes_tint_off');
    const on  = await mk(1, 'three_planes_tint_on');
    expect(off.success && on.success).toBe(true);

    // Probe the brightest pixel of the UNTINTED frame — that is a line core,
    // and it is the pixel guaranteed to be over the pivot. Picking the
    // brightest of the tinted frame instead would find whatever stayed white.
    let best = -1, bx = 0, by = 0;
    off.forEachPixel((p, x, y) => {
      if (luma(p) > best) { best = luma(p); bx = x; by = y; }
    });
    const a = off.pixelAt(bx, by), b = on.pixelAt(bx, by);
    expect(Math.abs(a.r - a.b)).toBeLessThan(40);   // white-hot to start with
    expect(b.g).toBeGreaterThan(b.r + 60);
    expect(b.g).toBeGreaterThan(b.b + 60);
  });

  it('renders three planes stacked in the middle of the frame', async () => {
    const frame = await runGpuEffectTest({
      module: MODULE, bundle: BUNDLE, width: W, height: H,
      inputColor: [0, 0, 0, 1], params: QUIET,
      dumpName: 'three_planes_default',
    });
    expect(frame.success).toBe(true);

    // The stack occupies the vertical middle; the extreme top and bottom rows
    // are outside every plane's halo.
    const band = meanRows(frame, Math.round(H * 0.3), Math.round(H * 0.7));
    const top = meanRows(frame, 0, 6);
    const bot = meanRows(frame, H - 6, H);
    expect(band).toBeGreaterThan(top + 15);
    expect(band).toBeGreaterThan(bot + 15);
  });

  it('debug plane keys land in stacking order (bottom=plane1)', async () => {
    // Flat per-plane keys, no glow or grade: isolates the projection and the
    // bottom-to-top ordering from everything else.
    const frame = await runGpuEffectTest({
      module: MODULE, bundle: BUNDLE, width: W, height: H,
      inputColor: [0, 0, 0, 1],
      params: [...QUIET, ['debug_show_planes', 1]],
      dumpName: 'three_planes_keys',
    });
    expect(frame.success).toBe(true);

    // Plane 1 keys red and sits LOW (cover-square y grows downward);
    // plane 3 keys blue and sits HIGH.
    const [, y1] = toPx(0, planeY(0));
    const [, y3] = toPx(0, planeY(2));
    expect(y1).toBeGreaterThan(y3);

    const p1 = frame.pixelAt(...toPx(0, planeY(0)));
    const p3 = frame.pixelAt(...toPx(0, planeY(2)));
    expect(p1.r).toBeGreaterThan(p1.b);   // red key at the bottom plane
    expect(p3.b).toBeGreaterThan(p3.r);   // blue key at the top plane
  });

  it('is dark when every plane is unlit', async () => {
    const frame = await runGpuEffectTest({
      module: MODULE, bundle: BUNDLE, width: W, height: H,
      inputColor: [0, 0, 0, 1],
      params: [...QUIET,
        ['plane1_emission', 0], ['plane2_emission', 0], ['plane3_emission', 0]],
      dumpName: 'three_planes_unlit',
    });
    expect(frame.success).toBe(true);
    expect(meanRows(frame, 0, H)).toBeLessThan(3);
  });

  it('a masking plane occludes the halo beneath it but keeps its own outline',
     async () => {
    // THE core semantic. Plane 1 (bottom) glows; planes 2 and 3 are dark.
    // Turning plane 2 into a black mask must eat plane 1's glow wherever
    // plane 2's body covers it — while plane 2's own outline still emits.
    const base: [string, number][] = [
      ...QUIET,
      ['plane1_emission', 1], ['plane2_emission', 0.6], ['plane3_emission', 0],
      ['halo_gain', 1.2], ['halo_radius', 0.5],
    ];

    const open = await runGpuEffectTest({
      module: MODULE, bundle: BUNDLE, width: W, height: H,
      inputColor: [0, 0, 0, 1], params: [...base, ['plane2_fill', 0]],
      dumpName: 'three_planes_mask_open',
    });
    const masked = await runGpuEffectTest({
      module: MODULE, bundle: BUNDLE, width: W, height: H,
      inputColor: [0, 0, 0, 1], params: [...base, ['plane2_fill', -1]],
      dumpName: 'three_planes_mask_closed',
    });
    expect(open.success).toBe(true);
    expect(masked.success).toBe(true);

    // Deep inside plane 2's body, on the axis — plane 1's glow reaches here.
    const [mx, my] = toPx(0, planeY(1));
    expect(luma(masked.pixelAt(mx, my)))
      .toBeLessThan(luma(open.pixelAt(mx, my)) - 20);

    // But the mask is not a blackout: plane 2's outline still emits, so the
    // frame keeps a bright peak.
    let peak = 0;
    masked.forEachPixel((c) => { peak = Math.max(peak, luma(c)); });
    expect(peak).toBeGreaterThan(120);
  });

  // Published scalars only come back on the browser path: the native runner
  // (native/tools/native_test_runner.mm) hardcodes `pluginState` to an empty
  // object, so there is nothing to assert against under `metal`. The pixel
  // cases above still cover both backends.
  const itRails = backend === 'puppeteer' ? it : it.skip;

  itRails('publishes plane Y invariant under orbit while half-height swings',
     async () => {
    const read = async (azimuth: number) => {
      const f = await runGpuEffectTest({
        module: MODULE, bundle: BUNDLE, width: W, height: H,
        inputColor: [0, 0, 0, 1],
        params: [...QUIET, ['orbit_azimuth', azimuth]],
        dumpName: `three_planes_orbit_${azimuth}`,
      });
      expect(f.success).toBe(true);
      return f.pluginState as Record<string, number>;
    };

    const a = await read(0.0);      // |sin| + |cos| = 1
    const b = await read(0.125);    // 45 deg -> sqrt(2)

    // The plane centres sit ON the orbit axis, so azimuth cannot move them.
    for (const [k, i] of [['plane1_y', 0], ['plane2_y', 1], ['plane3_y', 2]] as const) {
      expect(a[k]).toBeCloseTo(planeY(i), 3);
      expect(b[k]).toBeCloseTo(a[k], 5);
    }
    // The silhouette height does swing, though.
    expect(a['plane2_half_h']).toBeCloseTo(halfH(0.0), 3);
    expect(b['plane2_half_h']).toBeCloseTo(halfH(0.125), 3);
    expect(b['plane2_half_h']).toBeGreaterThan(a['plane2_half_h'] + 0.05);
  });

  it('chroma bleed separates the channels on a white outline', async () => {
    // White planes, so the plane's own hue can't account for an r/b split.
    // Note the grade is NOT channel-neutral even at bleed 0: nano_vcr_softclip
    // saturates R sooner than B on purpose (film dye layers, style guide 3.1),
    // so white picks up a slight tint. The assertion is therefore relative —
    // what the split adds on top of that baseline.
    const white: [string, number | number[]][] = [
      ['grain', 0], ['scanline', 0], ['warmth', 0],
      ['plane1_color', [1, 1, 1]], ['plane2_color', [1, 1, 1]],
      ['plane3_color', [1, 1, 1]],
    ];
    const maxSplit = (f: Frame) => {
      let m = 0;
      f.forEachPixel((c) => { m = Math.max(m, Math.abs(c.r - c.b)); });
      return m;
    };

    const off = await runGpuEffectTest({
      module: MODULE, bundle: BUNDLE, width: W, height: H,
      inputColor: [0, 0, 0, 1],
      params: [...white, ['chroma_bleed', 0]] as any,
      dumpName: 'three_planes_chroma_off',
    });
    const on = await runGpuEffectTest({
      module: MODULE, bundle: BUNDLE, width: W, height: H,
      inputColor: [0, 0, 0, 1],
      params: [...white, ['chroma_bleed', 0.8]] as any,
      dumpName: 'three_planes_chroma_on',
    });
    expect(off.success).toBe(true);
    expect(on.success).toBe(true);

    expect(maxSplit(on)).toBeGreaterThan(maxSplit(off) + 25);
  });

  // THE MEDIAL AXIS. A halo built on the nearest edge carries that construction's
  // skeleton into the picture: the distance to the outline ridges along both
  // diagonals of a quad — the locus where the nearest edge switches — and the
  // softmin's own bias brightens along the same locus, so the interior reads as
  // a dark four-pointed star or a bright centre ringed by a dark contour
  // depending on which of the two wins. Normally each plane's interior is washed
  // out by its neighbours' halos and you never see it; set the spacing to zero,
  // so all three coincide and there are no neighbours, and it is the only thing
  // in the frame. The interior is therefore a SUM over the four edges instead —
  // light from four tubes, no nearest-edge structure to inherit (render.hlsl,
  // kInteriorBlend).
  //
  // Probed as a DIP: on a smooth field a point on the axis sits at about the
  // mean of its two neighbours either side; across a kink it sits well below
  // them. Rendered larger than the rest of the suite because the probe needs a
  // few pixels of standoff to straddle.
  it('no medial-axis seam inside a plane', async () => {
    const BW = 480, BH = 320;
    const bax = Math.max(BW, BH) / (2 * BW), bay = Math.max(BW, BH) / (2 * BH);
    const bpx = (sx: number, sy: number): [number, number] =>
      [Math.round((sx * bax + 0.5) * BW), Math.round((sy * bay + 0.5) * BH)];

    const frame = await runGpuEffectTest({
      module: MODULE, bundle: BUNDLE, width: BW, height: BH,
      inputColor: [0, 0, 0, 1],
      params: [...QUIET, ['plane_spacing', 0]] as any,
      dumpName: 'three_planes_medial',
    });
    expect(frame.success).toBe(true);

    // Well inside the rhombus, on the horizontal diagonal — the long arm of the
    // skeleton, and the one with the most room around it to measure.
    const on = luma(frame.pixelAt(...bpx(0.125, 0)));
    const up = luma(frame.pixelAt(...bpx(0.125, -0.025)));
    const dn = luma(frame.pixelAt(...bpx(0.125, 0.025)));
    const dip = (up + dn) / 2 - on;

    // A nearest-edge field reads about 0.47 here; the light-sum about 0.07.
    expect(on).toBeGreaterThan(4);          // the probe is on the lit interior
    expect(dip / on).toBeLessThan(0.25);
  });

  // ---------------------------------------------------------------- Glimmer
  // The glints are thrown by a GESTURE — a knob crossing the middle of its
  // throw — so they cannot be reached from this harness at all: it sets its
  // params once and then ticks, and a parked knob throws nothing forever. That
  // is exactly what the case below pins, and what the effect promises an
  // unwired card. Everything about a glint in flight is in
  // web/test/three_planes_glints.test.ts, which drives the knob through the
  // engine, and in native/tests/test_three_planes_glints.cpp, which owns the
  // clock the invariant is stated against.

  it('a knob nobody moves throws nothing, ever', async () => {
    const mk = (ticks: number, sweep: number, name: string) => runGpuEffectTest({
      module: MODULE, bundle: BUNDLE, width: W, height: H,
      inputColor: [0, 0, 0, 1], ticks,
      params: [...QUIET, ['glimmer_sweep', sweep], ['glimmer_chaos', 16],
               ['glimmer_gain', 3]] as any,
      dumpName: name,
    });
    // Parked in the middle — where an untouched card sits — for two hundred
    // frames. A design that fired on being IN the band rather than on entering
    // it would have thrown one at boot with no gesture behind it.
    const home = await mk(0, 0.5, 'three_planes_glint_home_0');
    const homeLater = await mk(200, 0.5, 'three_planes_glint_home_200');
    // ...and parked off centre, which is the other way to get it wrong.
    const off = await mk(200, 0.92, 'three_planes_glint_off_200');
    expect(home.success && homeLater.success && off.success).toBe(true);

    let worst = 0;
    home.forEachPixel((p, x, y) => {
      worst = Math.max(worst, Math.abs(luma(p) - luma(homeLater.pixelAt(x, y))));
      worst = Math.max(worst, Math.abs(luma(p) - luma(off.pixelAt(x, y))));
    });
    // Not "close to": IDENTICAL. This also pins the shader's branchless idle
    // path — the obvious early `return` compiles into a local naga rejects,
    // and the whole effect silently renders nothing on WebGPU.
    expect(worst).toBe(0);
  });

  // ---------------------------------------------------------------- Release
  // The throw is a plain scalar input, so unlike the glints it IS reachable
  // from this harness — and the beat before it needs exact tick counting,
  // which is the one thing an engine run cannot give (wall-clock pacing puts
  // a single frame below its resolution).

  it('the rings wait one frame before appearing', async () => {
    // The frame that fires a throw is already black — the tower muted, which
    // is what fired it — so the rings are held back for exactly one more and
    // the picture lands on nothing before it lands on the release. A hit
    // reads harder for the silence in front of it.
    const mk = (ticks: number, name: string) => runGpuEffectTest({
      module: MODULE, bundle: BUNDLE, width: W, height: H,
      inputColor: [0, 0, 0, 1], ticks,
      params: [...QUIET, ['release', 1.0],
               // Muted, exactly as the rig leaves the tower at either end, so
               // anything in the frame is the throw and nothing else.
               ['plane1_emission', 0], ['plane2_emission', 0],
               ['plane3_emission', 0]] as any,
      dumpName: name,
    });
    const painted = (f: Frame) => {
      let n = 0;
      f.forEachPixel((p) => { if (luma(p) > 4) n++; });
      return n;
    };
    const firing = await mk(1, 'three_planes_ring_beat_1');
    const after = await mk(2, 'three_planes_ring_beat_2');
    expect(firing.success && after.success).toBe(true);
    expect(painted(firing)).toBe(0);
    expect(painted(after)).toBeGreaterThan(0);
  });

  // ------------------------------------------------------------ Strobe throw
  // The other mode. Nothing flies: the stack comes straight back exactly where
  // it was, as bare wireframe, and then breaks up — one floor at a time on a
  // roll whose window closes as the tail runs down.
  //
  // The roll's TIMING is reachable from this harness and from nowhere else. It
  // advances on a fixed dt of 0.016 per tick, so which floor is lit on tick N
  // is exactly computable, and the counts below are worked out rather than
  // observed. (The puttering needs a release that DECAYS, which only the rig
  // produces; that half is in native/tests/test_three_planes_strobe.cpp.)

  // The stock stack is taller than its own spacing, so the three diamonds
  // overlap in rows and no band belongs to one floor. Shrink and spread them
  // and each floor gets a row band of its own to be measured in.
  const R_SIZE = 0.30, R_SPACING = 0.55;
  const ELEV = (ELEV_DEG * Math.PI) / 180;
  const rPlaneY = (i: number) => -((i - 1) * R_SPACING) * Math.cos(ELEV) * ZOOM;
  // Half-height and floor pitch, in pixels. Azimuth is the default 45 deg, so
  // |sin| + |cos| is exactly sqrt(2).
  const rHalfPx  = Math.sin(ELEV) * ZOOM * R_SIZE * Math.SQRT2 * ay * H;
  const rPitchPx = R_SPACING * Math.cos(ELEV) * ZOOM * ay * H;
  // Wide enough to catch the floor's own side vertices, narrow enough that no
  // neighbour reaches in.
  const bandH = Math.max(2, Math.min(Math.floor(rHalfPx), Math.floor(rPitchPx / 2) - 2));
  const bandOf = (i: number): [number, number] => {
    const cy = toPx(0, rPlaneY(i))[1];
    return [Math.max(0, cy - bandH), Math.min(H, cy + bandH + 1)];
  };
  const bandLuma = (f: Frame, i: number) => {
    const [y0, y1] = bandOf(i);
    return meanRows(f, y0, y1);
  };

  // Floor 0 is the bottom one — `planeY` puts i = 0 at positive y, and y grows
  // downward. That is also where the roll starts.
  const ROLL: [string, any][] = [
    ...QUIET,
    ['release', 1.0], ['release_mode', 1],
    ['strobe_rate', 20], ['strobe_duty', 0.5], ['strobe_grace', 0],
    // Muted, exactly as the rig leaves the tower at either end, so everything
    // in the frame is the throw and nothing else.
    ['plane1_emission', 0], ['plane2_emission', 0], ['plane3_emission', 0],
    ['plane_size', R_SIZE], ['plane_spacing', R_SPACING],
  ];

  it('the whole stack arrives, then the roll takes it one floor at a time',
     async () => {
    // At 20 steps/s and 0.016 per tick the roll advances 0.32 of a step a
    // frame, and the clock does not start until the held frame is over:
    //
    //   tick 1        held — the beat before the hit
    //   ticks 2..5    step 0, the arrival: the whole stack
    //   tick 6        step 1, frac 0.28 -> floor 0 (bottom)
    //   tick 9        step 2, frac 0.24 -> floor 1 (middle)
    //   tick 12       step 3, frac 0.20 -> floor 2 (top)
    //
    // Which is the bounce, one floor at a time, in order.
    const mk = (ticks: number, name: string) => runGpuEffectTest({
      module: MODULE, bundle: BUNDLE, width: W, height: H,
      inputColor: [0, 0, 0, 1], ticks, params: ROLL as any, dumpName: name,
    });
    const held    = await mk(1, 'three_planes_strobe_held');
    const arrival = await mk(3, 'three_planes_strobe_arrival');
    const lo      = await mk(6, 'three_planes_strobe_floor0');
    const mid     = await mk(9, 'three_planes_strobe_floor1');
    const hi      = await mk(12, 'three_planes_strobe_floor2');
    expect(held.success && arrival.success).toBe(true);
    expect(lo.success && mid.success && hi.success).toBe(true);

    // The beat before the hit applies to this mode too — same gate.
    let painted = 0;
    held.forEachPixel((p) => { if (luma(p) > 4) painted++; });
    expect(painted).toBe(0);

    // The arrival is the one moment all three are up.
    const arr = [0, 1, 2].map((i) => bandLuma(arrival, i));
    for (const v of arr) expect(v).toBeGreaterThan(2);

    // ...and then it is strictly one at a time, ascending.
    const frames = [lo, mid, hi];
    frames.forEach((f, want) => {
      const bands = [0, 1, 2].map((i) => bandLuma(f, i));
      for (const other of [0, 1, 2]) {
        if (other === want) continue;
        expect(bands[want]).toBeGreaterThan(bands[other] * 4 + 1);
      }
    });
  });

  it('nothing flies: the wireframe comes back where the stack was', async () => {
    // Half spent, so Grow has thrown its rings most of the way out while
    // Strobe is drawing the same arrival in place. Measured on the topmost lit
    // row, which is the top floor's own apex in one mode and well above the
    // picture in the other.
    const mk = (mode: number, name: string) => runGpuEffectTest({
      module: MODULE, bundle: BUNDLE, width: W, height: H,
      inputColor: [0, 0, 0, 1], ticks: 3,
      params: [...ROLL, ['release', 0.5], ['release_mode', mode],
               ['strobe_duty', 1.0]] as any,
      dumpName: name,
    });
    const grow = await mk(0, 'three_planes_throw_grow');
    const strobe = await mk(1, 'three_planes_throw_strobe');
    expect(grow.success && strobe.success).toBe(true);

    const topLit = (f: Frame) => {
      let top = H;
      f.forEachPixel((p, _x, y) => { if (luma(p) > 8 && y < top) top = y; });
      return top;
    };
    // The stack's own apex: the top floor's centre, one half-height up.
    const apex = toPx(0, rPlaneY(2))[1] - rHalfPx;
    expect(topLit(strobe)).toBeGreaterThan(apex - 4);
    // Grow at half spent is 1 + 1.8 * 0.5 = 1.9x out, and the floors have
    // spread apart as well, so its apex is far above the stack's.
    expect(topLit(grow)).toBeLessThan(apex - 10);
  });

  it('Wire Glow is what makes it a wireframe rather than the picture again',
     async () => {
    const mk = (glow: number, name: string) => runGpuEffectTest({
      module: MODULE, bundle: BUNDLE, width: W, height: H,
      inputColor: [0, 0, 0, 1], ticks: 3,
      params: [...ROLL, ['strobe_glow', glow]] as any,
      dumpName: name,
    });
    const bare = await mk(0, 'three_planes_strobe_glow_0');
    const full = await mk(1, 'three_planes_strobe_glow_1');
    expect(bare.success && full.success).toBe(true);

    const lit = (f: Frame) => {
      let n = 0;
      f.forEachPixel((p) => { if (luma(p) > 8) n++; });
      return n;
    };
    // Same lines, same gain — all that changes is how far the light around
    // them carries.
    expect(lit(bare)).toBeGreaterThan(0);
    expect(lit(full)).toBeGreaterThan(lit(bare) * 2);
  });

  it('a lit floor loses its flam early instead of losing its brightness',
     async () => {
    // Local contrast, spent on the WINDOW. Nothing in this mode ever dims: a
    // floor that is lit underneath gets a SHORTER hit at exactly the same
    // strength, and once its window is gone the flam simply stops landing.
    //
    // The arrival is the clean place to read that. It runs for one whole step,
    // scaled by the floor's weight — at 20 steps/s and 0.016 a tick that is
    // frac 0.32 on tick 3 and 0.64 on tick 4, and a floor at emission 0.6 with
    // the default 0.8 contrast has a window of 0.52. So tick 3 lands for
    // everyone and tick 4 lands only for the undamped stack.
    const mk = (emission: number, release: number, ticks: number, name: string) =>
      runGpuEffectTest({
        module: MODULE, bundle: BUNDLE, width: W, height: H,
        inputColor: [0, 0, 0, 1], ticks,
        params: [...ROLL, ['release', release], ['strobe_duty', 1.0],
                 ['plane1_emission', emission], ['plane2_emission', emission],
                 ['plane3_emission', emission]] as any,
        dumpName: name,
      });
    // What the throw ADDS, which cancels whatever the base is doing.
    const added = (off: Frame, on: Frame) => {
      let s = 0;
      on.forEachPixel((p, x, y) => { s += Math.max(0, luma(p) - luma(off.pixelAt(x, y))); });
      return s;
    };
    const at = async (emission: number, ticks: number, tag: string) => {
      const off = await mk(emission, 0, ticks, `three_planes_strobe_damp_${tag}_off`);
      const on = await mk(emission, 1, ticks, `three_planes_strobe_damp_${tag}_on`);
      expect(off.success && on.success).toBe(true);
      return added(off, on);
    };
    const dark3 = await at(0, 3, 'dark3');
    const dark4 = await at(0, 4, 'dark4');
    const lit3 = await at(0.6, 3, 'lit3');
    const lit4 = await at(0.6, 4, 'lit4');

    // An undamped stack is still going a frame later...
    expect(dark3).toBeGreaterThan(0);
    expect(dark4).toBeGreaterThan(dark3 * 0.5);
    // ...where the lit one has already stopped outright.
    expect(lit4).toBeLessThan(lit3 * 0.1);
    // And what it did play, it played at strength — not at the 0.2 a dimmer
    // would have left. (Under, not equal, because the ghost lands on a lit
    // picture and the grade is not linear up there.)
    expect(lit3).toBeGreaterThan(dark3 * 0.4);
  });
});
});
