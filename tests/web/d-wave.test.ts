import { runGpuEffectTest, forEachBackend } from '@nano/web/test/gpu-test-helpers';
import { runEngineTest } from '@nano/web/test/engine-test-helpers';
import type { Sketch } from '@nano/web/src/sketch-types';

/**
 * E2E for warp.legacy.d_wave — the "D wave" radial-ripple distortion field
 * ported from the shipped NanoGraph Darkburst (the one block used live).
 *
 * Warping a SOLID input is invisible (every displaced sample reads the same
 * colour), so the field itself is verified via the debug overlay, and the real
 * warp is verified by chaining a deterministic structured generator
 * (source.grid) → d_wave and comparing distortion 0 (passthrough) vs 1.
 */
forEachBackend((backend) => {
describe(`D Wave (warp.legacy.d_wave) E2E (${backend})`, () => {
  jest.setTimeout(60000);

  it('declares metadata and its parameters', async () => {
    const frame = await runGpuEffectTest({
      module: 'warp.legacy.d_wave',
      bundle: 'legacy',
      inputColor: [0.2, 0.4, 0.6, 1.0],
      dumpName: 'd_wave_metadata',
    });
    expect(frame.success).toBe(true);
    expect(frame.metadata?.id).toBe('warp.legacy.d_wave');
    const names = frame.params.map(p => p.name);
    expect(names).toContain('distortion');
    expect(names).toContain('rate');
    expect(names).toContain('wave_speed');
    expect(names).toContain('scale');
    expect(names).toContain('density');
    expect(names).toContain('grain');
    expect(names).toContain('damp');
    expect(names).toContain('damp_count');
    expect(names).toContain('damp_rate');
  });

  it('builds a structured, propagating wave field (debug overlay)', async () => {
    // debug_field=1 paints the raw polar wave field (red) regardless of input,
    // isolating the stateful inject+advect field. damp=0 removes the flash layer
    // so this is the pure wave. Expect a structured (non-uniform) grainy field.
    const frame = await runGpuEffectTest({
      module: 'warp.legacy.d_wave',
      bundle: 'legacy',
      width: 128, height: 128,
      inputColor: [0.0, 0.0, 0.0, 1.0],
      params: [
        ['distortion', 0.0],   // overlay only — no warp
        ['rate', 0.8],
        ['density', 0.6],
        ['wave_speed', 0.3],
        ['damp', 0.0],         // wave only
        ['debug_field', 1.0],
      ],
      ticks: 12,
      renderEachTick: true,
      dumpName: 'd_wave_field',
    });
    expect(frame.success).toBe(true);

    let maxR = 0, lit = 0, dark = 0;
    frame.forEachPixel((c) => {
      if (c.r > maxR) maxR = c.r;
      if (c.r > 40) lit++;
      if (c.r < 8) dark++;
    });
    expect(maxR).toBeGreaterThan(60);    // waves present
    expect(lit).toBeGreaterThan(50);     // a real field, not a stray pixel
    expect(dark).toBeGreaterThan(50);    // gaps → structured grain
  });

  it('low density still covers all directions (no dead axis)', async () => {
    // At very low density the wave noise has few angular cells; without a
    // per-frame rotation the dim cell-midpoints sit at fixed angles → a dead
    // axis (e.g. only left/right fire, top/bottom black). The rotation sweeps
    // them, so over a handful of frames every cardinal direction sees waves.
    const frame = await runGpuEffectTest({
      module: 'warp.legacy.d_wave', bundle: 'legacy', width: 128, height: 128,
      inputColor: [0.0, 0.0, 0.0, 1.0],
      params: [
        ['distortion', 0.0], ['rate', 0.6], ['density', 0.02],
        ['wave_speed', 0.5], ['damp', 0.0], ['debug_field', 1.0],
      ],
      ticks: 16, renderEachTick: true,
      dumpName: 'd_wave_lowdensity',
    });
    expect(frame.success).toBe(true);

    // Lit pixels in the four cardinal mid-radius regions (R/L/T/B).
    const reg = { R: 0, L: 0, T: 0, B: 0 };
    frame.forEachPixel((c, x, y) => {
      const dx = x - 64, dy = y - 64, d = Math.sqrt(dx * dx + dy * dy);
      if (d < 24 || d > 56 || c.r <= 30) return;
      if (Math.abs(dx) > Math.abs(dy)) (dx > 0 ? reg.R++ : reg.L++);
      else (dy > 0 ? reg.B++ : reg.T++);
    });
    // Every direction must have waves — none of the four can be dead.
    for (const k of ['R', 'L', 'T', 'B'] as const) {
      expect(reg[k]).toBeGreaterThan(15);
    }
  });

  it('dampening flashes subtract from the wave field', async () => {
    // A near-full wave (rate=1) lit via the debug overlay. Turning the flash
    // layer on (damp + many fast flashes) carves streaks of reduced strength,
    // so the field's mean brightness drops vs damp=0.
    const meanRed = async (damp: number, dampCount: number) => {
      const f = await runGpuEffectTest({
        module: 'warp.legacy.d_wave', bundle: 'legacy', width: 128, height: 128,
        inputColor: [0.0, 0.0, 0.0, 1.0],
        params: [
          ['distortion', 0.0], ['rate', 1.0], ['density', 0.4], ['wave_speed', 0.25],
          ['grain', 0.0],                       // even wave (no sparkle) → clean measure
          ['damp', damp], ['damp_count', dampCount], ['damp_rate', 0.4],
          ['debug_field', 1.0],
        ],
        ticks: 14, renderEachTick: true,
        dumpName: `d_wave_damp_${damp}`,
      });
      expect(f.success).toBe(true);
      let sum = 0, n = 0;
      f.forEachPixel((c) => { sum += c.r; n++; });
      return sum / n;
    };
    const off = await meanRed(0.0, 0);
    const on  = await meanRed(1.0, 1500);
    expect(off).toBeGreaterThan(10);       // the wave actually lit the field
    // The flashes dampen on BOTH backends, but not equally: WebGPU reaches
    // ~0.8× here and Metal ~0.91× (114.0 → 104.0). The direction is the claim
    // this case exists for and it holds on both, so the threshold is per-backend
    // rather than tuned down to one number that asserts nothing.
    //
    // The gap is real and characterised — d_wave is the cleanest measurement of
    // it, because ONE effect holds both mechanisms: its wave field lives in
    // ping-pong textures and matches WebGPU exactly at every tick count, while
    // these flashes live in a storage buffer advanced in place per frame and
    // don't integrate natively at all (`damp_rate` 0 and 1.0 give identical
    // output). Metal plateaus at ~256 particles where WebGPU keeps scaling.
    // Full write-up incl. what's already been ruled out: web/KNOWN_ISSUES.md,
    // "Persistent GPU storage buffers don't carry state across frames on Metal".
    // Collapse to a single threshold once that's fixed.
    expect(on).toBeLessThan(off * (backend === 'metal' ? 0.95 : 0.85));
  });

});
});

// The cases below drive runEngineTest — the engine harness page (executor.wasm,
// wires, trace points) — which has no native runner, so they stay puppeteer-only.
// The comp runner is the native equivalent for engine-level work, and a native
// sketch host is an explicit follow-up.
describe('D Wave (warp.legacy.d_wave) E2E (engine path)', () => {
  jest.setTimeout(60000);

  it('radially warps a structured input', async () => {
    // source.grid → d_wave. distortion=0 is a pure passthrough (warpFactor==1),
    // so run A is the clean grid (deterministic, static generator). Run B warps
    // it with strong ripples. The two must differ → the warp moved pixels.
    const buildChain = (overrides: Record<string, unknown>): Sketch => ({
      anchor: null,
      wires: [],
      chain: [
        { type: 'module', module_type: 'source.grid', instance_key: 'grid@0', params: {} },
        {
          type: 'module', module_type: 'warp.legacy.d_wave', instance_key: 'dw@0',
          params: { render_alpha: 1.0, ...overrides },
        },
      ],
    });

    const flat = await runEngineTest({
      width: 128, height: 128,
      modules: ['com.nano.core', 'com.nano.legacy'],
      commands: [
        { type: 'createSketch', sketchId: 'dw_flat', sketch: buildChain({ distortion: 0.0 }) },
        { type: 'setTracePoints', tracePoints: [
          { id: 'out', target: { type: 'sketch_output', sketchId: 'dw_flat' } },
        ]},
      ],
      waitFrames: 16,
      captureTraceIds: ['out'],
      dumpName: 'd_wave_warp_flat',
    });
    expect(flat.success).toBe(true);

    const warped = await runEngineTest({
      width: 128, height: 128,
      modules: ['com.nano.core', 'com.nano.legacy'],
      commands: [
        { type: 'createSketch', sketchId: 'dw_warp',
          sketch: buildChain({ distortion: 1.0, rate: 0.7, wave_speed: 0.4, damp: 0.0 }) },
        { type: 'setTracePoints', tracePoints: [
          { id: 'out', target: { type: 'sketch_output', sketchId: 'dw_warp' } },
        ]},
      ],
      waitFrames: 16,
      captureTraceIds: ['out'],
      dumpName: 'd_wave_warp_on',
    });
    expect(warped.success).toBe(true);

    // The grid must actually have rendered (so "differ" is meaningful).
    let lit = 0;
    flat.trace('out').forEachPixel((c) => { if (c.r + c.g + c.b > 24) lit++; });
    expect(lit).toBeGreaterThan(100);

    // The ripples displaced the grid lines → the frames differ.
    warped.trace('out').expectDifferentFrom(flat.trace('out'), 100);
  });

  it('drives downstream motion blur via render_outputs/motion', async () => {
    // grid → d_wave → motion.blur. d_wave only emits render_outputs/motion when a
    // sink reads it; motion.blur is that sink (wires:[] struct auto-connect). The
    // d_wave stage is identical across both runs, so the ONLY difference between
    // blur strength 0 (pass-through) and 32 is whether the motion rail carried
    // the warp's per-frame velocity. A visible difference proves the whole rail.
    const build = (blur: number): Sketch => ({
      anchor: null,
      wires: [],
      chain: [
        { type: 'module', module_type: 'source.grid', instance_key: 'grid@0', params: {} },
        { type: 'module', module_type: 'warp.legacy.d_wave', instance_key: 'dw@0',
          params: { distortion: 1.0, wave_speed: 0.6, damp: 0.0, motion_scale: 3.0, render_alpha: 1.0 } },
        { type: 'module', module_type: 'motion.blur', instance_key: 'blur@0',
          params: { strength: blur, samples: 16, quality: 1 } },
      ],
    });
    const run = (id: string, blur: number, dump: string) => runEngineTest({
      width: 128, height: 128,
      modules: ['com.nano.core', 'com.nano.testonly', 'com.nano.legacy'],
      commands: [
        { type: 'createSketch', sketchId: id, sketch: build(blur) },
        { type: 'setTracePoints', tracePoints: [
          { id: 'out', target: { type: 'sketch_output', sketchId: id } },
        ]},
      ],
      waitFrames: 24,
      captureTraceIds: ['out'],
      dumpName: dump,
    });

    const blurred = await run('dwm_b', 32.0, 'd_wave_motion_blurred');
    expect(blurred.success).toBe(true);
    const sharp = await run('dwm_s', 0.0, 'd_wave_motion_sharp');
    expect(sharp.success).toBe(true);

    let lit = 0;
    sharp.trace('out').forEachPixel((c) => { if (c.r + c.g + c.b > 24) lit++; });
    expect(lit).toBeGreaterThan(100);   // the warped grid actually rendered

    // strength=32 smears along the warp's motion; strength=0 is a pass-through.
    blurred.trace('out').expectDifferentFrom(sharp.trace('out'), 100);
  });
});
