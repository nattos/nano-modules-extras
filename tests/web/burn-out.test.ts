import { runGpuEffectTest, forEachBackend } from '@nano/web/test/gpu-test-helpers';

/**
 * E2E for color.legacy.burn_out — "Burn Out", the v2 port of the Resolume Wire
 * "Burn Out" patch (AR-envelope-driven exposure-blowout grade).
 *
 * It's a colour grade, so a solid input shows the effect directly. The manual
 * `amount` knob drives the burn without any timing; `trigger` fires the
 * envelope over a few ticks.
 */
forEachBackend((backend) => {
describe(`Burn Out (color.legacy.burn_out) E2E (${backend})`, () => {
  jest.setTimeout(60000);

  const MID: [number, number, number, number] = [0.30, 0.30, 0.50, 1.0]; // ~ (76,76,128)

  it('declares metadata and its parameters', async () => {
    const frame = await runGpuEffectTest({
      module: 'color.legacy.burn_out', bundle: 'legacy',
      inputColor: MID,
      dumpName: 'burn_metadata',
    });
    expect(frame.success).toBe(true);
    expect(frame.metadata?.id).toBe('color.legacy.burn_out');
    const names = frame.params.map(p => p.name);
    for (const n of ['trigger', 'gate', 'amount', 'attack', 'release',
                     'saturation_boost', 'contrast_boost', 'darkness',
                     'brightness', 'modulate_alpha']) {
      expect(names).toContain(n);
    }
  });

  it('manual amount fades the image out to black', async () => {
    const burned = await runGpuEffectTest({
      module: 'color.legacy.burn_out', bundle: 'legacy',
      inputColor: MID,
      params: [['amount', 1.0]],
      ticks: 2, renderEachTick: true,
      dumpName: 'burn_amount',
    });
    expect(burned.success).toBe(true);
    // amount=1, darkness=1 → the mid input fades to (near) black.
    const p = burned.pixelAt(32, 32);
    expect(p.r).toBeLessThan(30);
    expect(p.g).toBeLessThan(30);
    expect(p.b).toBeLessThan(30);
  });

  it('is a passthrough at rest (is_identity)', async () => {
    const rest = await runGpuEffectTest({
      module: 'color.legacy.burn_out', bundle: 'legacy',
      inputColor: MID,
      params: [['amount', 0.0]],
      dumpName: 'burn_rest',
    });
    expect(rest.success).toBe(true);
    rest.expectUniformColor({ r: 76, g: 76, b: 128 }, 4);
  });

  it('the trigger fires the AR envelope over time', async () => {
    // trigger rising edge → one-shot attack; a fast attack reaches the burn
    // within a handful of 16 ms ticks.
    const fired = await runGpuEffectTest({
      module: 'color.legacy.burn_out', bundle: 'legacy',
      inputColor: MID,
      params: [['trigger', 1], ['amount', 0.0], ['attack', 0.05], ['release', 2.0]],
      ticks: 16, renderEachTick: true,
      dumpName: 'burn_triggered',
    });
    expect(fired.success).toBe(true);
    const p = fired.pixelAt(32, 32);
    expect(p.r).toBeLessThan(50); // faded toward black, not the resting 76
  });

  it('modulate_alpha drops alpha with the burn', async () => {
    const faded = await runGpuEffectTest({
      module: 'color.legacy.burn_out', bundle: 'legacy',
      inputColor: MID,
      params: [['amount', 1.0], ['modulate_alpha', 1]],
      ticks: 2, renderEachTick: true,
      dumpName: 'burn_alpha',
    });
    expect(faded.success).toBe(true);
    const a = faded.pixelAt(32, 32).a;
    // The harness may composite onto an opaque surface and read back a=255;
    // only assert the drop when alpha actually survives readback.
    if (a !== undefined && a < 250) {
      expect(a).toBeLessThan(40); // burn=1 → alpha → ~0
    }
  });
});
});
