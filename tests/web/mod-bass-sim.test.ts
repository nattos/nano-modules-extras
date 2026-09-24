import { runEngineTest } from '@nano/web/test/engine-test-helpers';
import type { Sketch } from '@nano/web/src/sketch-types';

/**
 * E2E SMOKE for mod.source.bass_sim — the synthetic Resolume-FFT low-band
 * modulation source. The signal model itself (shared sketch/fft_bass_sim.h)
 * is pinned by the transient shaper's Catch2 goldens; this suite verifies
 * the wiring: registration, the deterministic floor level, that kicks
 * actually fire off the engine's beat clock, and the flagship integration —
 * bass_sim auto-connected into mod.shaper.transient_shaper builds real
 * confidence over a couple of live bars.
 *
 * Probe rig: white solid → (mod modules) → brightness_contrast with a mod
 * output wired into bc.brightness (combine:'replace'): display ≈ output*255.
 * Engine clock is the fixed 120 BPM test transport (bar = 2 s).
 */
describe('mod.source.bass_sim source node E2E', () => {
  jest.setTimeout(90000);

  const build = (params: Record<string, number>, opts?: {
    shaper?: boolean, field?: string,
  }): Sketch => {
    const chain: any[] = [
      { type: 'module', module_type: 'source.solid_color', instance_key: 'src@0',
        params: { color: [1.0, 1.0, 1.0] } },
      { type: 'module', module_type: 'mod.source.bass_sim', instance_key: 'bs@0',
        params },
    ];
    let probeKey = 'bs@0';
    let probeField = opts?.field ?? 'output';
    if (opts?.shaper) {
      chain.push({ type: 'module', module_type: 'mod.shaper.transient_shaper',
                   instance_key: 'ts@0', params: {} });
      probeKey = 'ts@0';
    }
    chain.push({ type: 'module', module_type: 'color.tone.brightness_contrast',
                 instance_key: 'bc@0', params: { brightness: 1.0, contrast: -0.5 } });
    return {
      anchor: null,
      chain,
      wires: [
        { id: 'w1', src: { instanceKey: probeKey, field: probeField },
          dest: { instanceKey: 'bc@0', field: 'brightness' }, combine: 'replace' },
      ],
    } as Sketch;
  };

  const run = (id: string, params: Record<string, number>, waitFrames: number,
               opts?: { shaper?: boolean, field?: string }) => runEngineTest({
    width: 64, height: 64,
    modules: ['com.nano.core', 'com.nano.nano'],
    commands: [{ type: 'createSketch', sketchId: id, sketch: build(params, opts) }],
    tracePoints: [{ id: 'out', target: { type: 'sketch_output', sketchId: id } }],
    captureTraceIds: ['out'],
    waitFrames,
    dumpName: id,
  });

  it('Silence pattern rests exactly on the floor level', async () => {
    const r = await run('bs_floor', { pattern: 0, wobble: 0.0, floor: 0.4 }, 20);
    expect(r.success).toBe(true);
    // Output == floor 0.4 → gray ~102, deterministic (no kicks, no wobble).
    r.trace('out').expectPixelAt(32, 32, { r: 102, g: 102, b: 102 }, 10);
  });

  it('kicks fire off the beat clock and hold the level above the floor', async () => {
    // Slow fall (0.1/s) + 4-floor kicks every 500 ms at the 120 BPM test
    // clock: once the first kick lands the level saw-tooths in ~[0.57, 0.62]
    // — always well above the 0.4 floor, whatever the frame pacing.
    const r = await run('bs_kicks', { fall: 0.1, wobble: 0.0 }, 150);
    expect(r.success).toBe(true);
    const p = r.trace('out').pixelAt(32, 32);
    expect(p.r).toBeGreaterThan(120);
    expect(p.r).toBeLessThan(175);
  });

  it('integration: feeds the transient shaper, confidence builds over live bars',
     async () => {
    // bass_sim auto-connects into the shaper (both are mod modules); probe
    // the shaper's `confidence` output. After ~2.5 live bars of 4-floor at
    // default adapt (4 bars), the current slot's confidence passes ~0.25.
    const r = await run('bs_shaper', { wobble: 0.0 }, 700,
                        { shaper: true, field: 'confidence' });
    expect(r.success).toBe(true);
    const p = r.trace('out').pixelAt(32, 32);
    expect(p.r).toBeGreaterThan(60);
  });
});
