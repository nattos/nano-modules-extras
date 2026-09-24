import { runGpuEffectTest, forEachBackend, Frame } from '@nano/web/test/gpu-test-helpers';
import { runEngineTest } from '@nano/web/test/engine-test-helpers';
import type { Sketch } from '@nano/web/src/sketch-types';

// Per-effect tests for source.light.chroma_wave — a charge-and-burst prismatic blob
// that grows from the top-center while gated, then bursts outward on release.
// The grade scrolls across the burst (renderEachTick advances grade_phase and
// the phase machine), so envelope-dependent tests use renderEachTick. With a
// black input the blob's generated colour is just bright pixels, so "where the
// bloom lands vertically" reads the grow → burst-expand envelope.

/** Shared by the effect half and the engine half below. */
const luma = (p: { r: number; g: number; b: number }) => 0.299 * p.r + 0.587 * p.g + 0.114 * p.b;

forEachBackend((backend) => {
describe(`Chroma Wave Effect E2E (${backend})`, () => {
  jest.setTimeout(60000);

  const W = 128, H = 128;

  // Fraction of bright pixels in the vertical band [y0, y1).
  const brightBand = (f: Frame, y0: number, y1: number): number => {
    let bright = 0, n = 0;
    for (let y = Math.floor(y0 * H); y < Math.floor(y1 * H); y++) {
      for (let x = 0; x < W; x++) { if (luma(f.pixelAt(x, y)) > 30) bright++; n++; }
    }
    return n > 0 ? bright / n : 0;
  };

  // Circular variance of hue over coloured pixels — a proxy for "prismatic"
  // (many distinct hues banded across the blob) rather than a single flat hue.
  // 0 = all one hue; → 1 = hues spread around the wheel. A high grade_freq
  // (burst) spreads many hue cycles across the density ramp; a low one (hold)
  // is nearly monochromatic.
  const hueVariance = (f: Frame): number => {
    let sx = 0, sy = 0, n = 0;
    for (let y = 0; y < H; y++) for (let x = 0; x < W; x++) {
      const p = f.pixelAt(x, y);
      const mx = Math.max(p.r, p.g, p.b), mn = Math.min(p.r, p.g, p.b);
      const d = mx - mn;
      if (mx <= 30 || d < 12) continue;   // skip dark / near-grey pixels
      let h = 0;
      if (mx === p.r) h = ((p.g - p.b) / d) % 6;
      else if (mx === p.g) h = (p.b - p.r) / d + 2;
      else h = (p.r - p.g) / d + 4;
      h = (h / 6 + 1) % 1;                 // [0,1)
      const a = h * 2 * Math.PI;
      sx += Math.cos(a); sy += Math.sin(a); n++;
    }
    if (n === 0) return 0;
    const r = Math.sqrt(sx * sx + sy * sy) / n;
    return 1 - r;                          // circular variance
  };

  // Circular mean hue [0,1) over coloured pixels.
  const meanHue = (f: Frame): number => {
    let sx = 0, sy = 0, n = 0;
    for (let y = 0; y < H; y++) for (let x = 0; x < W; x++) {
      const p = f.pixelAt(x, y);
      const mx = Math.max(p.r, p.g, p.b), mn = Math.min(p.r, p.g, p.b);
      const d = mx - mn;
      if (mx <= 30 || d < 12) continue;
      let h = 0;
      if (mx === p.r) h = ((p.g - p.b) / d) % 6;
      else if (mx === p.g) h = (p.b - p.r) / d + 2;
      else h = (p.r - p.g) / d + 4;
      h = (h / 6 + 1) % 1;
      const a = h * 2 * Math.PI;
      sx += Math.cos(a); sy += Math.sin(a); n++;
    }
    if (n === 0) return 0;
    return ((Math.atan2(sy, sx) / (2 * Math.PI)) + 1) % 1;
  };
  // Shortest distance between two hues on the [0,1) wheel.
  const hueDist = (a: number, b: number): number => {
    const d = Math.abs(a - b) % 1;
    return Math.min(d, 1 - d);
  };

  const params = (extra: [string, number][]): [string, number][] => [
    ['auto_rate', 0], ['intensity', 1.5], ['saturation', 0.9],
    ['base_radius', 0.12], ['charge_expand', 2.2], ['position_y', -0.7],
    ['gaussian_sharpness', 4], ['overlay_alpha_hold', 0.5], ['seed', 7],
    // Keep one-shot voices centered/clean so single-voice tests are
    // deterministic; the polyphony test overrides these.
    ['voice_pos_jitter', 0], ['voice_hue_jitter', 0],
    ...extra,
  ];

  it('declares metadata and I/O', async () => {
    const frame = await runGpuEffectTest({
      module: 'source.light.chroma_wave', bundle: 'lights',
      inputColor: [0, 0, 0, 1], dumpName: 'chroma_wave_metadata',
    });
    expect(frame.success).toBe(true);
    expect(frame.metadata?.id).toBe('source.light.chroma_wave');
  });

  it('schema JSON round-trips (not truncated past the buffer)', async () => {
    // chroma_wave's ~40 params push the schema JSON well past the old 4 KB
    // cap. On overflow the JSON truncates → strict JSON.parse fails → the host
    // leaves params empty (the inspector shows NOTHING). metadata.id is set
    // before the parse, so it can't catch this — assert the derived params.
    const frame = await runGpuEffectTest({
      module: 'source.light.chroma_wave', bundle: 'lights',
      inputColor: [0, 0, 0, 1], dumpName: 'chroma_wave_schema',
    });
    expect(frame.success).toBe(true);
    expect(frame.params.length).toBeGreaterThan(30);            // all fields present
    const names = frame.params.map((p: any) => p.name);
    expect(names).toContain('gate');                            // first field
    expect(names).toContain('motion_warp');                     // last-added field
  });

  it('idle (no gate / trigger / auto) renders pure passthrough', async () => {
    const frame = await runGpuEffectTest({
      module: 'source.light.chroma_wave', bundle: 'lights',
      width: W, height: H, inputColor: [0, 0, 0, 1], renderEachTick: true,
      ticks: 8, params: params([]),
      dumpName: 'chroma_wave_idle',
    });
    expect(frame.success).toBe(true);
    frame.expectUniformColor({ r: 0, g: 0, b: 0, a: 255 }, 4);
  });

  it('charge: held gate blooms near the top, bottom stays dark', async () => {
    // default_gate_state locks the charge; the blob sits at the top band.
    const frame = await runGpuEffectTest({
      module: 'source.light.chroma_wave', bundle: 'lights',
      width: W, height: H, inputColor: [0, 0, 0, 1], renderEachTick: true,
      ticks: 30, params: params([['default_gate_state', 1], ['charge_s', 0.3]]),
      dumpName: 'chroma_wave_charge',
    });
    expect(frame.success).toBe(true);
    expect(brightBand(frame, 0.0, 0.35)).toBeGreaterThan(0.02);  // bloom up top
    expect(brightBand(frame, 0.75, 1.0)).toBeLessThan(0.01);     // bottom dark
  });

  it('burst: a trigger sustains at top, then the bloom expands downward', async () => {
    // Trigger holds for min_sustain_s (bloom at top), then bursts — the radius
    // expands rapidly so coverage spreads down the canvas.
    const at = (ticks: number) => runGpuEffectTest({
      module: 'source.light.chroma_wave', bundle: 'lights',
      width: W, height: H, inputColor: [0, 0, 0, 1], renderEachTick: true,
      ticks,
      params: [...params([['min_sustain_s', 0.1], ['release_s', 0.7],
                          ['release_expand', 4.0], ['charge_s', 0.1]]),
               ['trigger', 1]],
      dumpName: `chroma_wave_burst_${ticks}`,
    });
    const charging = await at(4);    // still held → bundled near the top
    const bursting = await at(26);   // expanded → reaches lower band
    expect(charging.success && bursting.success).toBe(true);
    const chargeLow = brightBand(charging, 0.55, 0.95);
    const burstLow = brightBand(bursting, 0.55, 0.95);
    expect(burstLow).toBeGreaterThan(chargeLow + 0.02);
  });

  it('burst goes prismatic: the bloom is more colourful during the burst', async () => {
    // Hold has few bands (gentle); burst ramps grade_freq high → rainbow.
    const charge = await runGpuEffectTest({
      module: 'source.light.chroma_wave', bundle: 'lights',
      width: W, height: H, inputColor: [0, 0, 0, 1], renderEachTick: true,
      ticks: 16, params: params([['default_gate_state', 1], ['charge_s', 0.2],
                                  ['hue_span', 0.25], ['grade_freq_hold', 0.5],
                                  ['band_contrast', 0.2]]),
      dumpName: 'chroma_wave_prismatic_hold',
    });
    const burst = await runGpuEffectTest({
      module: 'source.light.chroma_wave', bundle: 'lights',
      width: W, height: H, inputColor: [0, 0, 0, 1], renderEachTick: true,
      ticks: 18,
      params: [...params([['min_sustain_s', 0.05], ['release_s', 1.0],
                          ['hue_span', 0.25], ['grade_freq_burst', 10],
                          ['band_contrast', 0.2], ['charge_s', 0.05]]),
               ['trigger', 1]],
      dumpName: 'chroma_wave_prismatic_burst',
    });
    expect(charge.success && burst.success).toBe(true);
    expect(hueVariance(burst)).toBeGreaterThan(hueVariance(charge) + 0.05);
  });

  it('extended ranges: off-canvas origin + large base_radius floods the canvas', async () => {
    // position_y = -1.6 puts the center above the top edge; a large base_radius
    // (charge onset, slow charge_s so it stays big) reaches well into frame.
    const frame = await runGpuEffectTest({
      module: 'source.light.chroma_wave', bundle: 'lights',
      width: W, height: H, inputColor: [0, 0, 0, 1], renderEachTick: true,
      ticks: 3, params: params([['default_gate_state', 1], ['charge_s', 2.0],
                                 ['position_y', -1.6], ['base_radius', 2.0],
                                 ['gaussian_sharpness', 1.5]]),
      dumpName: 'chroma_wave_extended',
    });
    expect(frame.success).toBe(true);
    // Mid-canvas lit despite the center sitting off the top edge.
    expect(brightBand(frame, 0.3, 0.6)).toBeGreaterThan(0.02);
  });

  it('band_tilt skews the grade along the wavefront (shifts the mean hue)', async () => {
    // band_tilt adds band_tilt*qy to the transfer, so equal-and-opposite tilts
    // shift the blob's mean hue in opposite directions — a deterministic,
    // measurable asymmetry along the (vertical) wavefront axis.
    const run = (tilt: number) => runGpuEffectTest({
      module: 'source.light.chroma_wave', bundle: 'lights',
      width: W, height: H, inputColor: [0, 0, 0, 1], renderEachTick: true,
      ticks: 14, params: params([['default_gate_state', 1], ['charge_s', 0.15],
                                  ['hue_span', 0.3], ['grade_freq_hold', 3],
                                  ['band_tilt', tilt]]),
      dumpName: `chroma_wave_tilt_${tilt}`,
    });
    const pos = await run(1.5);
    const neg = await run(-1.5);
    expect(pos.success && neg.success).toBe(true);
    expect(hueDist(meanHue(pos), meanHue(neg))).toBeGreaterThan(0.03);
  });

  it('intensity crank brightens the bloom (soft rolloff)', async () => {
    const run = (gain: number) => runGpuEffectTest({
      module: 'source.light.chroma_wave', bundle: 'lights',
      width: W, height: H, inputColor: [0, 0, 0, 1], renderEachTick: true,
      ticks: 14, params: params([['default_gate_state', 1], ['charge_s', 0.15],
                                  ['intensity', gain]]),
      dumpName: `chroma_wave_crank_${gain}`,
    });
    const lo = await run(1.0);
    const hi = await run(6.0);
    expect(lo.success && hi.success).toBe(true);
    expect(brightBand(hi, 0.0, 0.5)).toBeGreaterThan(brightBand(lo, 0.0, 0.5) + 0.01);
  });

  it('auto_rate fires one-shots on its own (silent at 0)', async () => {
    // auto_mode Random + auto_rate > 0 self-animates with no gate wired; each
    // Poisson event is a one-shot that fully charges then bursts. The shared
    // auto-trigger block defaults to Off, so Random must be selected too.
    const on = await runGpuEffectTest({
      module: 'source.light.chroma_wave', bundle: 'lights',
      width: W, height: H, inputColor: [0, 0, 0, 1], renderEachTick: true,
      ticks: 40, params: params([['auto_mode', 1], ['auto_rate', 0.6],
                                  ['charge_s', 0.15],
                                  ['min_sustain_s', 0.1], ['release_s', 0.5]]),
      dumpName: 'chroma_wave_auto_on',
    });
    expect(on.success).toBe(true);
    expect(brightBand(on, 0.0, 1.0)).toBeGreaterThan(0.005);
  });

  it('a trigger value of 0 never fires (rising-edge, replay-safe)', async () => {
    // The executor replays every stored field — including `trigger` — as a
    // PatchReplace each frame. The handler must fire only on a 0->1 rising
    // edge, not on any patch arrival, or the cycle re-arms forever at idle.
    // Here trigger sits at 0 with auto_rate 0: nothing should ever light up.
    const frame = await runGpuEffectTest({
      module: 'source.light.chroma_wave', bundle: 'lights',
      width: W, height: H, inputColor: [0, 0, 0, 1], renderEachTick: true,
      ticks: 12, params: [...params([]), ['trigger', 0]],
      dumpName: 'chroma_wave_trigger0',
    });
    expect(frame.success).toBe(true);
    frame.expectUniformColor({ r: 0, g: 0, b: 0, a: 255 }, 4);
  });

  it('polyphony: a higher voice_limit yields more simultaneous bloom coverage', async () => {
    // With auto_rate cranked, voices spawn faster than they die. voice_limit 8
    // lets up to 8 overlap; limit 1 keeps only one alive at a time → markedly
    // less total bright coverage. Spread them so they don't all stack.
    const run = (limit: number) => runGpuEffectTest({
      module: 'source.light.chroma_wave', bundle: 'lights',
      width: W, height: H, inputColor: [0, 0, 0, 1], renderEachTick: true,
      ticks: 45, params: params([['auto_mode', 1], ['auto_rate', 0.9],
                                  ['charge_s', 0.1],
                                  ['min_sustain_s', 0.05], ['release_s', 0.6],
                                  ['voice_limit', limit], ['voice_pos_jitter', 0.7],
                                  ['base_radius', 0.1]]),
      dumpName: `chroma_wave_poly_${limit}`,
    });
    const one = await run(1);
    const eight = await run(8);
    expect(one.success && eight.success).toBe(true);
    expect(brightBand(eight, 0.0, 1.0)).toBeGreaterThan(brightBand(one, 0.0, 1.0) + 0.02);
  });

  it('hue_interact rotates overlapping voices further (more hue spread)', async () => {
    // Two co-located voices (a held one + a one-shot, no position jitter) fully
    // overlap. At hue_interact 0 their band phases average → one blended hue;
    // cranked up they SUM → the combined phase rotates further round the wheel
    // → more distinct hues. (A lone voice is unaffected: avg == sum.)
    const run = (hi: number) => runGpuEffectTest({
      module: 'source.light.chroma_wave', bundle: 'lights',
      width: W, height: H, inputColor: [0, 0, 0, 1], renderEachTick: true,
      ticks: 16,
      params: [...params([['default_gate_state', 1], ['charge_s', 0.1],
                          ['grade_freq_hold', 3], ['hue_span', 0.3],
                          ['band_contrast', 0.3], ['hue_interact', hi]]),
               ['trigger', 1]],
      dumpName: `chroma_wave_interact_${hi}`,
    });
    const lo = await run(0.0);
    const high = await run(1.8);
    expect(lo.success && high.success).toBe(true);
    expect(hueVariance(high)).toBeGreaterThan(hueVariance(lo) + 0.03);
  });

  it('hue twist shifts the hue where it lands on a primary', async () => {
    // A nearly-constant-hue blob parked at red (base_hue 0, tiny hue_span).
    // hue_shift_r twists the wheel at the red point, so the bloom's mean hue
    // moves away from red. (g/b shifts are 0, so off-red hues stay put.)
    const run = (shiftR: number) => runGpuEffectTest({
      module: 'source.light.chroma_wave', bundle: 'lights',
      width: W, height: H, inputColor: [0, 0, 0, 1], renderEachTick: true,
      ticks: 14, params: params([['default_gate_state', 1], ['charge_s', 0.2],
                                  ['base_hue', 0.0], ['hue_span', 0.02],
                                  ['grade_freq_hold', 0.5], ['saturation', 0.9],
                                  ['hue_shift_r', shiftR]]),
      dumpName: `chroma_wave_huetwist_${shiftR}`,
    });
    const off = await run(0.0);
    const twisted = await run(0.33);
    expect(off.success && twisted.success).toBe(true);
    expect(hueDist(meanHue(off), meanHue(twisted))).toBeGreaterThan(0.1);
  });

  it('intensity 0 renders passthrough even while charging', async () => {
    const frame = await runGpuEffectTest({
      module: 'source.light.chroma_wave', bundle: 'lights',
      width: W, height: H, inputColor: [0, 0, 0, 1], renderEachTick: true,
      ticks: 10, params: params([['default_gate_state', 1], ['intensity', 0]]),
      dumpName: 'chroma_wave_intensity0',
    });
    expect(frame.success).toBe(true);
    frame.expectUniformColor({ r: 0, g: 0, b: 0, a: 255 }, 4);
  });
});
});


// The two engine cases, PUPPETEER ONLY: both go through runEngineTest — the
// engine harness page (executor.wasm, wires, sidechannel bus, trace points) —
// which has no native runner. They are the ones that need chroma_wave WIRED to
// something downstream, which is engine-level rather than effect-level; the
// comp runner is the native equivalent and a native sketch host is a follow-up.
describe('Chroma Wave Effect E2E (engine path)', () => {
  jest.setTimeout(60000);

  it('emits motion vectors that reshape a downstream motion_blur', async () => {
    // Chain: black bg → chroma_wave → motion_blur. The bursting blobs expand,
    // so chroma_wave publishes radial-outward velocity on render_outputs/motion.
    // With the rail wired, motion_blur smears the bloom along those vectors;
    // without it, it falls back to a pass-through. The two final frames must
    // differ — proving chroma_wave emits motion AND motion_blur consumes it.
    // Wire model: motion_blur's render_outputs input auto-connects to the
    // chroma_wave producer above it. The negative case omits the producer so the
    // blur falls back to a pass-through (no motion to consume).
    const buildChain = (withProducer: boolean): Sketch => ({
      anchor: null,
      wires: [],
      chain: [
        { type: 'module', module_type: 'source.solid_color', instance_key: 'bg@0',
        params: { color: [0.0, 0.0, 0.0] } },
        ...(withProducer ? [{ type: 'module', module_type: 'source.light.chroma_wave', instance_key: 'cw@0',
          params: {
            auto_mode: 1, auto_rate: 0.9, charge_s: 0.1, min_sustain_s: 0.05,
            release_s: 0.5,
            release_expand: 4.0, base_radius: 0.15, intensity: 2.0,
            voice_pos_jitter: 0.3, voice_hue_jitter: 0.0, motion_scale: 1.0, seed: 3,
          },
        }] : []),
        { type: 'module', module_type: 'motion.blur', instance_key: 'blur@0',
          params: { strength: 32.0, samples: 16, quality: 1 },
        },
      ],
    } as Sketch);

    const run = (id: string, withProducer: boolean) => runEngineTest({
      width: 128, height: 128,
      modules: ['com.nano.lights', 'com.nano.core'],
      commands: [
        { type: 'createSketch', sketchId: id, sketch: buildChain(withProducer) },
        { type: 'setTracePoints', tracePoints: [
          { id: 'out', target: { type: 'sketch_output', sketchId: id } },
        ]},
      ],
      waitFrames: 30,
      captureTraceIds: ['out'],
      dumpName: `chroma_wave_motion_${withProducer ? 'with' : 'without'}_producer`,
    });

    const withProducer = await run('cw_motion_with', true);
    const noProducer = await run('cw_motion_without', false);
    expect(withProducer.success && noProducer.success).toBe(true);
    withProducer.trace('out').expectDifferentFrom(noProducer.trace('out'), 100);
  });

  it('wave_out publishes the wave alone on opaque black (only when wired)', async () => {
    // Chain: gray solid → chroma_wave (held charge) → sidechannel send →
    // sidechannel receive. Wiring cw@0.wave_out into the send's `send_in`
    // override makes the bus carry the isolated wave layer, and the receive
    // REPLACES the stage output with it — so the sketch output IS wave_out.
    // Without the wire the send publishes its chain input instead (the
    // composite over gray). Wired: corners are opaque BLACK (the wave layer's
    // background — alpha 1, so no checkerboard) with a bloom up top. Unwired:
    // corners show the gray input — proving both the isolation ("on black")
    // and the connection gating.
    const buildChain = (wired: boolean): Sketch => ({
      anchor: null,
      wires: wired ? [{ id: 'ww', src: { instanceKey: 'cw@0', field: 'wave_out' },
                        dest: { instanceKey: 'send@0', field: 'send_in' } }] : [],
      chain: [
        { type: 'module', module_type: 'source.solid_color', instance_key: 'bg@0',
          params: { color: [0.45, 0.45, 0.45] } },
        { type: 'module', module_type: 'source.light.chroma_wave', instance_key: 'cw@0',
          params: { default_gate_state: 1, charge_s: 0.15, intensity: 2.0,
                    auto_rate: 0, voice_pos_jitter: 0, voice_hue_jitter: 0 } },
        { type: 'module', module_type: 'util.sidechannel_out', instance_key: 'send@0',
          params: { channel: 3 } },
        { type: 'module', module_type: 'util.sidechannel_in', instance_key: 'recv@0',
          params: { channel: 3 } },
      ],
    } as Sketch);

    const run = (id: string, wired: boolean) => runEngineTest({
      width: 128, height: 128,
      modules: ['com.nano.lights', 'com.nano.core'],
      commands: [
        { type: 'createSketch', sketchId: id, sketch: buildChain(wired) },
        { type: 'setTracePoints', tracePoints: [
          { id: 'out', target: { type: 'sketch_output', sketchId: id } },
        ]},
      ],
      waitFrames: 30,
      captureTraceIds: ['out'],
      dumpName: `chroma_wave_waveout_${wired ? 'wired' : 'unwired'}`,
    });

    const wired = await run('cw_wave_wired', true);
    const unwired = await run('cw_wave_unwired', false);
    expect(wired.success && unwired.success).toBe(true);

    // Wired: the isolated layer — black corners, bloom near the top-center.
    wired.trace('out').expectPixelAt(3, 3, { r: 0, g: 0, b: 0 }, 6);
    wired.trace('out').expectPixelAt(124, 124, { r: 0, g: 0, b: 0 }, 6);
    let peak = 0;
    for (let y = 4; y < 48; y += 4) for (let x = 32; x < 96; x += 4) {
      peak = Math.max(peak, luma(wired.trace('out').pixelAt(x, y)));
    }
    expect(peak).toBeGreaterThan(30);

    // Unwired: the bus falls back to the chain image — gray corners (the
    // composite over the input), so the gate demonstrably decided the content.
    const bgCorner = unwired.trace('out').pixelAt(3, 3);
    expect(bgCorner.r).toBeGreaterThan(60);
  });
});
