import { runGpuEffectTest, Frame, forEachBackend } from '@nano/web/test/gpu-test-helpers';

// Per-effect tests for source.light.bounce_resonator — a 4-bar scalar diffusion
// network with NO spatial structure. Each hop multiplies the 4-vector by a
// seeded mixing matrix (v ← M·v), cycling through `pattern_count` matrices
// at `cycle_rate` hops/sec:
//   spread 0 → seeded random derangement (each bar dumps its full energy
//              onto one other bar each hop);
//   spread 1 → fully random per-column distribution (energy fans out).
//   feedback → per-hop energy gain (1.0 conserves, <1 decays).
// Rendering: each bar fills its whole 1/4 column (hsv2rgb of its hue/value).
//
// The sim is GPU-resident (state in a persistent buffer, stepped in a
// compute shader), so these tests use `renderEachTick: true` — the hop
// accumulator advances per tick and the buffer carries frame to frame.
// A single rising-edge gate kick + auto_rate 0 keeps it deterministic;
// tests assert on conservation/concentration, not which specific bar lights.

forEachBackend((backend) => {
describe(`Bounce Resonator (diffusion) E2E (${backend})`, () => {
  jest.setTimeout(60000);

  const W = 160, H = 108;

  // Mean luminance of bar `k`'s centered band.
  const barBrightness = (f: Frame, k: number): number => {
    const x0 = Math.floor((k + 0.3) * W / 4), x1 = Math.floor((k + 0.7) * W / 4);
    const y0 = Math.floor(H * 0.45), y1 = Math.floor(H * 0.55);
    let s = 0, n = 0;
    for (let y = y0; y < y1; y++) {
      for (let x = x0; x < x1; x++) {
        const p = f.pixelAt(x, y); s += p.r + p.g + p.b; n++;
      }
    }
    return n > 0 ? s / n : 0;
  };
  const bars = (f: Frame) => [0, 1, 2, 3].map(k => barBrightness(f, k));
  const total = (f: Frame) => bars(f).reduce((a, b) => a + b, 0);
  const concentration = (f: Frame) => {
    const b = bars(f), t = b.reduce((a, c) => a + c, 0);
    return t > 1e-3 ? Math.max(...b) / t : 0;
  };

  // One rising-edge kick into bar 0 (gate 0→1), no auto-fire. Low strength
  // + intensity so brightness stays out of the 255 clamp. cycle_rate fixed
  // at 10 Hz so a given tick count maps to a known number of hops.
  // impulse_mode: 0 tex_in, 1 one_bar, 2 random_bar, 3 all_bars.
  const kickBar0 = (extra: [string, number][]): [string, number][] => [
    ['gate', 1], ['auto_rate', 0.0], ['impulse_mode', 1], ['one_bar_target', 0],
    ['impulse_strength', 0.6], ['intensity', 1.0],
    ['seed', 0], ['pattern_count', 4], ['cycle_rate', 10.0],
    ...extra,
  ];

  it('declares metadata and I/O', async () => {
    const frame = await runGpuEffectTest({
      module: 'source.light.bounce_resonator', bundle: 'lights',
      inputColor: [0, 0, 0, 1], dumpName: 'bounce_resonator_metadata',
    });
    expect(frame.success).toBe(true);
    expect(frame.metadata?.id).toBe('source.light.bounce_resonator');
  });

  it('a kick lights the network', async () => {
    const frame = await runGpuEffectTest({
      module: 'source.light.bounce_resonator', bundle: 'lights',
      width: W, height: H, inputColor: [0, 0, 0, 1], renderEachTick: true,
      ticks: 5, params: kickBar0([['feedback', 1.0], ['spread', 0.3]]),
      dumpName: 'bounce_resonator_kick',
    });
    expect(frame.success).toBe(true);
    expect(total(frame)).toBeGreaterThan(30);
  });

  it('fills the whole 1/4 column (not a centered band)', async () => {
    // spread 0, feedback 1.0 → energy parks on one bar; that bar's entire
    // vertical strip should be lit top-to-bottom, equally.
    const frame = await runGpuEffectTest({
      module: 'source.light.bounce_resonator', bundle: 'lights',
      width: W, height: H, inputColor: [0, 0, 0, 1], renderEachTick: true,
      ticks: 8, params: kickBar0([['feedback', 1.0], ['spread', 0.0]]),
      dumpName: 'bounce_resonator_fill',
    });
    expect(frame.success).toBe(true);
    // Find the lit bar, then check top, middle, bottom of its column match.
    const lum = (k: number, fy: number) => {
      const x = Math.floor((k + 0.5) * W / 4), y = Math.floor(fy * H);
      const p = frame.pixelAt(x, y); return p.r + p.g + p.b;
    };
    const lit = [0, 1, 2, 3].reduce((m, k) => lum(k, 0.5) > lum(m, 0.5) ? k : m, 0);
    const top = lum(lit, 0.08), mid = lum(lit, 0.5), bot = lum(lit, 0.92);
    expect(mid).toBeGreaterThan(20);
    expect(top).toBeGreaterThan(mid * 0.9);
    expect(bot).toBeGreaterThan(mid * 0.9);
  });

  // Sum a channel over a bar's column.
  const barChan = (f: Frame, k: number, ch: 'r' | 'g' | 'b'): number => {
    const x0 = Math.floor((k + 0.3) * W / 4), x1 = Math.floor((k + 0.7) * W / 4);
    const y0 = Math.floor(H * 0.4), y1 = Math.floor(H * 0.6);
    let s = 0, n = 0;
    for (let y = y0; y < y1; y++)
      for (let x = x0; x < x1; x++) { s += f.pixelAt(x, y)[ch]; n++; }
    return n > 0 ? s / n : 0;
  };

  it('a kick paints its bar with band_color’s hue', async () => {
    // Pure green band_color → the freshly kicked (undiffused) bar is green.
    const frame = await runGpuEffectTest({
      module: 'source.light.bounce_resonator', bundle: 'lights',
      width: W, height: H, inputColor: [0, 0, 0, 1], renderEachTick: true,
      ticks: 3,
      params: [...kickBar0([['feedback', 1.0], ['spread', 0.0], ['hue_spread', 0.0]]),
               ['band_color', [0, 1, 0]]],
      dumpName: 'bounce_resonator_hue_green',
    });
    expect(frame.success).toBe(true);
    const lit = [0, 1, 2, 3].reduce((m, k) => barBrightness(frame, k) > barBrightness(frame, m) ? k : m, 0);
    expect(barChan(frame, lit, 'g')).toBeGreaterThan(60);
    expect(barChan(frame, lit, 'r')).toBeLessThan(20);
    expect(barChan(frame, lit, 'b')).toBeLessThan(20);
  });

  it('hue_spread rotates hue away from band_color as it transfers', async () => {
    // Kick green; with hue_spread the carried hue twists each hop, so the
    // lit bar drifts off pure green (gains red/blue).
    const lit = (hue_spread: number) => runGpuEffectTest({
      module: 'source.light.bounce_resonator', bundle: 'lights',
      width: W, height: H, inputColor: [0, 0, 0, 1], renderEachTick: true,
      ticks: 40,
      params: [...kickBar0([['feedback', 1.0], ['spread', 0.2], ['hue_spread', hue_spread]]),
               ['band_color', [0, 1, 0]]],
      dumpName: `bounce_resonator_huespread_${Math.round(hue_spread * 100)}`,
    });
    const none = await lit(0.0);
    const lots = await lit(0.8);
    expect(none.success && lots.success).toBe(true);
    // Off-green (red+blue) energy across all bars: hue_spread pushes it up.
    const offGreen = (f: Frame) =>
      [0, 1, 2, 3].reduce((s, k) => s + barChan(f, k, 'r') + barChan(f, k, 'b'), 0);
    expect(offGreen(lots)).toBeGreaterThan(offGreen(none) + 30);
  });

  it('impulse_mode tex_in takes the impulse colour from tex_in (not band_color)', async () => {
    // RED input, GREEN band_color. In tex_in mode each bar samples the input
    // (red); in all_bars mode the bars use band_color (green). So the green
    // added by the bars is high in all_bars mode and ~0 in tex_in mode.
    const run = (mode: number, extra: [string, number][]) => runGpuEffectTest({
      module: 'source.light.bounce_resonator', bundle: 'lights',
      width: W, height: H, inputColor: [0.8, 0, 0, 1], renderEachTick: true,
      ticks: 3,
      params: [...kickBar0([['feedback', 1.0], ['spread', 0.0], ['hue_spread', 0.0]]),
               ['band_color', [0, 1, 0]], ['impulse_mode', mode], ...extra],
      dumpName: `bounce_resonator_mode_${mode}`,
    });
    const sampled = await run(0, [['tex_in_boost', 3.0]]);     // tex_in → red
    const banded  = await run(3, [['impulse_strength', 2.0]]); // all_bars → green
    expect(sampled.success && banded.success).toBe(true);
    const maxG = (f: Frame) => Math.max(...[0, 1, 2, 3].map(k => barChan(f, k, 'g')));
    expect(maxG(banded)).toBeGreaterThan(maxG(sampled) + 40);
  });

  it('tex_in mode takes on the input’s dominant hue; tex_in_color scales it', async () => {
    // Saturated RED input with the DEFAULT near-white bar colour (sat ≈ 0.22).
    // At tex_in_color 1 the sampled saturation rides the sim state, so the
    // bars go strongly red; at 0 the injected colour is Bar Color's pastel
    // (green channel nearly as bright as red). input_opacity 0 isolates the
    // bars from the red passthrough.
    const run = (strength: number) => runGpuEffectTest({
      module: 'source.light.bounce_resonator', bundle: 'lights',
      width: W, height: H, inputColor: [0.9, 0, 0, 1], renderEachTick: true,
      ticks: 3,
      params: [...kickBar0([['feedback', 1.0], ['spread', 0.0], ['hue_spread', 0.0]]),
               ['impulse_mode', 0], ['tex_in_boost', 3.0], ['input_opacity', 0.0],
               ['tex_in_color', strength]],
      dumpName: `bounce_resonator_texcolor_${Math.round(strength * 100)}`,
    });
    const full = await run(1.0);
    const off = await run(0.0);
    expect(full.success && off.success).toBe(true);
    const lit = (f: Frame) =>
      [0, 1, 2, 3].reduce((m, k) => barBrightness(f, k) > barBrightness(f, m) ? k : m, 0);
    const lf = lit(full), lo = lit(off);
    // Full strength: pure red bars — green stays dark.
    expect(barChan(full, lf, 'r')).toBeGreaterThan(80);
    expect(barChan(full, lf, 'g')).toBeLessThan(25);
    // Strength 0: band-colour pastel — green nearly as bright as red.
    expect(barChan(off, lo, 'g')).toBeGreaterThan(80);
    expect(barChan(off, lo, 'g')).toBeGreaterThan(barChan(off, lo, 'r') * 0.7);
  });

  it('input_opacity fades the passed-through input', async () => {
    // Grey input, no bars lit (engine off) → output tracks input_opacity.
    const run = (input_opacity: number) => runGpuEffectTest({
      module: 'source.light.bounce_resonator', bundle: 'lights',
      width: W, height: H, inputColor: [0.5, 0.5, 0.5, 1], renderEachTick: true,
      ticks: 2,
      params: [['auto_rate', 0.0], ['cycle_rate', 0.0], ['intensity', 0.0],
               ['input_opacity', input_opacity]],
      dumpName: `bounce_resonator_inop_${Math.round(input_opacity * 100)}`,
    });
    const opaque = await run(1.0);
    const black = await run(0.0);
    expect(opaque.success && black.success).toBe(true);
    // With intensity 0 the bars add nothing, so the frame is just the faded
    // input: ~grey at 1.0, ~black at 0.0.
    const mid = (f: Frame) => { const p = f.pixelAt(W / 2, H / 2); return p.r + p.g + p.b; };
    expect(mid(opaque)).toBeGreaterThan(300);
    expect(mid(black)).toBeLessThan(15);
  });

  it('chroma_hold keeps overdriven bars saturated instead of washing to white', async () => {
    // Pure red bars driven hot (all_bars strength 2 × intensity 6 → lin.r ≈ 12).
    // At chroma_hold 0 the overflow spills as warm white (green nearly pegs);
    // at 1 the peak-mapped rolloff keeps the r:g:b ratio — red pegs, green and
    // blue stay dark.
    const run = (hold: number) => runGpuEffectTest({
      module: 'source.light.bounce_resonator', bundle: 'lights',
      width: W, height: H, inputColor: [0, 0, 0, 1], renderEachTick: true,
      ticks: 3,
      params: [...kickBar0([['feedback', 1.0], ['spread', 0.0], ['hue_spread', 0.0],
                            ['impulse_mode', 3], ['impulse_strength', 2.0],
                            ['intensity', 6.0], ['chroma_hold', hold]]),
               ['band_color', [1, 0, 0]]],
      dumpName: `bounce_resonator_chroma_${Math.round(hold * 100)}`,
    });
    const washed = await run(0.0);
    const held = await run(1.0);
    expect(washed.success && held.success).toBe(true);
    // Both drives peg red; only the washed one drags green/blue up with it.
    expect(barChan(washed, 0, 'r')).toBeGreaterThan(200);
    expect(barChan(washed, 0, 'g')).toBeGreaterThan(150);
    expect(barChan(held, 0, 'r')).toBeGreaterThan(200);
    expect(barChan(held, 0, 'g')).toBeLessThan(20);
    expect(barChan(held, 0, 'b')).toBeLessThan(20);
  });

  it('feedback conserves vs decays total energy', async () => {
    const after = (feedback: number) => runGpuEffectTest({
      module: 'source.light.bounce_resonator', bundle: 'lights',
      width: W, height: H, inputColor: [0, 0, 0, 1], renderEachTick: true,
      ticks: 40, params: kickBar0([['feedback', feedback], ['spread', 0.5]]),
      dumpName: `bounce_resonator_fb_${Math.round(feedback * 100)}`,
    });
    const hi = await after(1.0);   // reverberates forever → energy retained
    const lo = await after(0.3);   // ~0.3×/sec → much dimmer total by now
    expect(hi.success && lo.success).toBe(true);
    expect(total(hi)).toBeGreaterThan(total(lo) + 40);
  });

  it('spread fans energy out across bars (concentration drops)', async () => {
    const at = (spread: number) => runGpuEffectTest({
      module: 'source.light.bounce_resonator', bundle: 'lights',
      width: W, height: H, inputColor: [0, 0, 0, 1], renderEachTick: true,
      ticks: 30, params: kickBar0([['feedback', 1.0], ['spread', spread]]),
      dumpName: `bounce_resonator_spread_${Math.round(spread * 100)}`,
    });
    const narrow = await at(0.0);  // permutation → energy on a single bar
    const wide = await at(1.0);    // random fan-out → spread across bars
    expect(narrow.success && wide.success).toBe(true);
    // Single-bar concentration is ~1.0; fanned-out is ~0.3-0.4.
    expect(concentration(narrow)).toBeGreaterThan(concentration(wide) + 0.3);
  });
});
});
