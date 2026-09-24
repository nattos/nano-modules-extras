import { runGpuTest, runGpuEffectTest, forEachBackend } from '@nano/web/test/gpu-test-helpers';

// Per-effect tests for `source.light.plasma_beam_cannon` against the `lights` bundle.
//
// V1 scaffold: an ADSR phase machine drives the beam half-height. Idle
// → Attack (held at seed_height) → Decay (ramps to full bar) → Sustain
// (full bar) → Release (shrinks back) → Idle. Triggered by `gate` or
// `trigger`.
//
// Note: the test runner ticks at dt=0.016s for `ticks` iterations.

forEachBackend((backend) => {
describe(`Plasma Beam Cannon Effect E2E (${backend})`, () => {
  jest.setTimeout(30000);

  it('declares metadata and I/O', async () => {
    const frame = await runGpuTest({
      module: 'source.light.plasma_beam_cannon',
      bundle: 'lights',
      dumpName: 'plasma_beam_cannon_metadata',
    });
    expect(frame.success).toBe(true);
    expect(frame.metadata?.id).toBe('source.light.plasma_beam_cannon');
  });

  it('idle with auto_rate=0 passes input through unchanged', async () => {
    const frame = await runGpuEffectTest({
      module: 'source.light.plasma_beam_cannon',
      bundle: 'lights',
      inputColor: [0.4, 0.4, 0.4, 1.0],
      // Mute the Poisson auto-trigger so we can verify pristine
      // idle behavior. Without this, the default auto_rate=0.2
      // makes the effect cycle on its own.
      params: [['auto_rate', 0.0]],
      dumpName: 'plasma_beam_cannon_idle',
    });
    expect(frame.success).toBe(true);
    frame.expectUniformColor({ r: 102, g: 102, b: 102, a: 255 }, 4);
  });

  it('a trigger value of 0 never fires (rising-edge, replay-safe)', async () => {
    // The executor replays every stored field — including `trigger` — every
    // frame. The handler must fire only on a 0->1 rising edge of the value,
    // not on any patch arrival, or the cycle re-arms forever. With trigger at
    // 0 and auto_rate 0, the beam must never light: pristine passthrough.
    const frame = await runGpuEffectTest({
      module: 'source.light.plasma_beam_cannon',
      bundle: 'lights',
      width: 64, height: 64,
      inputColor: [0.4, 0.4, 0.4, 1.0],
      renderEachTick: true,
      ticks: 20,
      params: [['auto_rate', 0.0], ['trigger', 0]],
      dumpName: 'plasma_beam_cannon_trigger0',
    });
    expect(frame.success).toBe(true);
    frame.expectUniformColor({ r: 102, g: 102, b: 102, a: 255 }, 4);
  });

  it('auto_rate fires without external trigger (IDE preview demo path)', async () => {
    // With auto_rate cranked, Poisson fires nearly every frame, so
    // within a few ticks the effect is reliably in a beam-visible
    // phase. This verifies the path that makes the IDE preview "feel
    // alive" when the effect is dropped in without any gate wiring.
    const frame = await runGpuEffectTest({
      module: 'source.light.plasma_beam_cannon',
      bundle: 'lights',
      width: 64, height: 64,
      inputColor: [0.0, 0.0, 0.0, 1.0],
      ticks: 30,                   // ~0.5s of ticking
      params: [
        // auto_mode Random — `auto_rate` alone does nothing. The shared
        // fx::AutoTrigger fields default `auto_mode` to Off so a freshly
        // dropped effect stays quiet until something is wired in, and this
        // suite predates that selector: it set the rate on an effect that was
        // never going to self-fire, so the frame stayed solid black.
        ['auto_mode', 1],
        ['auto_rate', 1.0],        // ~59 Hz Poisson → triggers near-immediately
        ['beam_color', [1.0, 1.0, 1.0]],
        ['intensity', 1.0],
        ['bar_target_all', 1.0],
        // Short ADSR so 30 ticks reaches sustain.
        ['attack_s',  0.05],
        ['decay_s',   0.05],
      ],
      dumpName: 'plasma_beam_cannon_auto_fire',
    });
    expect(frame.success).toBe(true);
    frame.expectNotSolidColor({ r: 0, g: 0, b: 0 }, 5);
  });

  it('gate=true + enough ticks → sustain fills the canvas with beam color', async () => {
    // Default ADSR: attack 0.15s, decay 0.10s. After 0.25s of held
    // gate (~16 frames @ dt=0.016) we should be in Sustain → full
    // bar height, every pixel lit with beam color.
    const frame = await runGpuEffectTest({
      module: 'source.light.plasma_beam_cannon',
      bundle: 'lights',
      width: 64, height: 64,
      inputColor: [0.0, 0.0, 0.0, 1.0],
      ticks: 30,                   // ~0.48s — well into Sustain
      params: [
        ['gate', 1.0],
        ['beam_color', [1.0, 1.0, 1.0]],
        ['intensity', 1.0],
        ['bar_target_all', 1.0],
        ['attack_s',  0.05],       // quick attack
        ['decay_s',   0.05],       // quick decay
      ],
      dumpName: 'plasma_beam_cannon_sustain',
    });
    expect(frame.success).toBe(true);
    // In sustain the beam extent is [seed_y - 0.5, seed_y + 0.5] = [0, 1]
    // (full bar). All bars all-lit (bar_target_all = true).
    frame.expectCoverage((c) => c.r > 200, { min: 0.95 });
  });

  it('gate=true with bar_target_all=false lights only the targeted bar', async () => {
    const frame = await runGpuEffectTest({
      module: 'source.light.plasma_beam_cannon',
      bundle: 'lights',
      width: 64, height: 64,
      inputColor: [0.0, 0.0, 0.0, 1.0],
      ticks: 30,
      params: [
        ['gate', 1.0],
        ['beam_color', [1.0, 1.0, 1.0]],
        ['intensity', 1.0],
        ['bar_target_all', 0.0],
        ['bar_target', 2],          // light bar 2 only
        ['attack_s',  0.05],
        ['decay_s',   0.05],
      ],
      dumpName: 'plasma_beam_cannon_single_bar',
    });
    expect(frame.success).toBe(true);

    // Bar 2 covers x ∈ [32, 48). Sample mid-column of each bar.
    frame.expectPixelAt(8, 32,  { r: 0, g: 0, b: 0 }, 5);       // bar 0 dark
    frame.expectPixelAt(24, 32, { r: 0, g: 0, b: 0 }, 5);       // bar 1 dark
    frame.expectPixelAt(40, 32, { r: 255, g: 255, b: 255 }, 5); // bar 2 lit
    frame.expectPixelAt(56, 32, { r: 0, g: 0, b: 0 }, 5);       // bar 3 dark
  });

  it('release phase paints break gaps in the beam (some pixels dark)', async () => {
    // Trigger, then run long enough that we're deep into release.
    // IMPORTANT: ADSR timing params must be set BEFORE `gate` because
    // the gate rising-edge handler computes the trigger pulse hold
    // using the current values of attack/decay/sustain_s — and the
    // runner applies params in array order.
    const frame = await runGpuEffectTest({
      module: 'source.light.plasma_beam_cannon',
      bundle: 'lights',
      width: 64, height: 64,
      inputColor: [0.0, 0.0, 0.0, 1.0],
      ticks: 40,
      params: [
        ['auto_rate', 0.0],
        ['attack_s', 0.05],
        ['decay_s', 0.05],
        ['sustain_s', 0.10],
        ['release_s', 1.0],
        ['beam_color', [1.0, 1.0, 1.0]],
        ['intensity', 1.0],
        ['bar_target_all', 1.0],
        // Cranked break params: many solid breaks, big target length
        // → very likely to find dark pixels during release.
        ['break_count_per_bar', 16],
        ['attractor_fraction', 0.5],
        ['spacer_fraction', 0.0],
        ['min_break_size', 0.05],
        ['max_break_size', 0.2],
        ['length_target_start', 0.4],
        ['length_target_end', 0.7],
        ['flicker_start_t', 1.0],   // disable flicker so we isolate breaks
        // gate LAST so trigger pulse uses the small ADSR values above.
        ['gate', 1.0],
      ],
      dumpName: 'pbc_release_breaks',
    });
    expect(frame.success).toBe(true);
    // Expect a meaningful fraction of pixels to be eaten by breaks
    // (i.e. passthrough black). At least 5% dark on a fully-lit
    // sustain-into-release beam means breaks are doing visible work.
    const total = 64 * 64;
    let dark = 0;
    frame.forEachPixel((c) => { if (c.r < 30) dark++; });
    const darkFraction = dark / total;
    expect(darkFraction).toBeGreaterThan(0.05);
  });

  it('growth_fraction=0 → every break stays at min size (no large gaps)', async () => {
    // With growth_fraction=0, all breaks get personal_max=min_break_size,
    // so no break can grow regardless of length-controller error.
    // Even with a high target_end the visible break coverage stays
    // bounded by `count * min_break_size`.
    const frame = await runGpuEffectTest({
      module: 'source.light.plasma_beam_cannon',
      bundle: 'lights',
      width: 64, height: 64,
      inputColor: [0.0, 0.0, 0.0, 1.0],
      ticks: 40,
      params: [
        ['auto_rate', 0.0],
        ['attack_s', 0.05],
        ['decay_s', 0.05],
        ['sustain_s', 0.10],
        ['release_s', 1.0],
        ['beam_color', [1.0, 1.0, 1.0]],
        ['intensity', 1.0],
        ['bar_target_all', 1.0],
        ['break_count_per_bar', 16],
        ['attractor_fraction', 0.5],
        ['spacer_fraction', 0.0],
        ['min_break_size', 0.02],
        ['max_break_size', 0.2],   // big — would normally allow huge breaks
        ['length_target_end', 1.0],  // try to push to lots of coverage
        ['grow_response', 4.0],      // aggressive growth pressure
        ['growth_fraction', 0.0],    // ...but NO break is allowed to grow
        ['flicker_start_t', 1.0],
        ['gate', 1.0],
      ],
      dumpName: 'pbc_growth_fraction_zero',
    });
    expect(frame.success).toBe(true);
    // With 16 breaks per bar × 4 bars × min=0.02 size each, max
    // possible dark coverage per bar = 16 * 0.02 = 0.32 of bar height.
    // Realistic coverage with overlap and y-axis-only-checking ≈ 25%.
    // Test: must be substantially less than the unbounded case.
    let dark = 0;
    frame.forEachPixel((c) => { if (c.r < 30) dark++; });
    const darkFraction = dark / (64 * 64);
    expect(darkFraction).toBeLessThan(0.4);
  });

  it('cycle_seed produces different break patterns across sequential triggers', async () => {
    // Run two separate test instances (each one starts with cycle
    // count = 0), but the second one runs longer so it's deeper into
    // the SECOND cycle. With cycle_seed=true the two patterns SHOULD
    // differ — frames won't be identical.
    const common: any = {
      module: 'source.light.plasma_beam_cannon',
      bundle: 'lights' as const,
      width: 64, height: 64,
      inputColor: [0.0, 0.0, 0.0, 1.0] as [number, number, number, number],
      params: [
        ['auto_mode', 1],            // Random — see the auto_rate note above
        ['auto_rate', 1.0],          // fast auto-fire → many cycles
        ['attack_s', 0.02],
        ['decay_s', 0.02],
        ['sustain_s', 0.05],
        ['release_s', 0.3],          // short cycle = many cycles in fixed ticks
        ['beam_color', [1.0, 1.0, 1.0]],
        ['intensity', 1.0],
        ['bar_target_all', 1.0],
        ['break_count_per_bar', 12],
        ['attractor_fraction', 0.3],
        ['spacer_fraction', 0.0],
        ['min_break_size', 0.04],
        ['max_break_size', 0.15],
        ['growth_fraction', 0.5],
        ['cycle_seed', 1.0],         // ← cycle is ON
        ['break_seed', 12345],
        ['flicker_start_t', 1.0],
      ] as any[],
    };
    // Capture two frames at different deep-into-multi-cycle tick counts.
    const a = await runGpuEffectTest({ ...common, ticks: 200, dumpName: 'pbc_cycle_a' });
    const b = await runGpuEffectTest({ ...common, ticks: 250, dumpName: 'pbc_cycle_b' });
    expect(a.success).toBe(true);
    expect(b.success).toBe(true);
    // The two frames are from different cycle indices → break patterns
    // differ → significant pixel-diff (well above the trivial noise
    // floor from float rounding).
    a.expectDifferentFrom(b, 50);
  });

  it('activation_min=1.0 prevents all breaks from activating', async () => {
    // Each break gets a threshold in [activation_min, 1.0]. With
    // activation_min=1.0, every threshold is 1.0 — breaks never become
    // active for any release_t < 1.0. Captured mid-release the beam
    // should be fully lit (zero breaks visible) even though attractor
    // breaks were configured.
    const frame = await runGpuEffectTest({
      module: 'source.light.plasma_beam_cannon',
      bundle: 'lights',
      width: 64, height: 64,
      inputColor: [0.0, 0.0, 0.0, 1.0],
      ticks: 40,
      params: [
        ['auto_rate', 0.0],
        ['attack_s', 0.05],
        ['decay_s', 0.05],
        ['sustain_s', 0.10],
        ['release_s', 1.0],
        ['beam_color', [1.0, 1.0, 1.0]],
        ['intensity', 1.0],
        ['bar_target_all', 1.0],
        ['break_count_per_bar', 16],
        ['attractor_fraction', 0.5],
        ['spacer_fraction', 0.0],
        ['min_break_size', 0.05],
        ['max_break_size', 0.2],
        ['activation_min', 1.0],     // no break can activate
        ['flicker_start_t', 1.0],    // disable flicker
        ['gate', 1.0],
      ],
      dumpName: 'pbc_activation_min_one',
    });
    expect(frame.success).toBe(true);
    frame.expectCoverage((c) => c.r > 200, { min: 0.95 });
  });

  it('release with spacer_fraction=1.0 has no visible breaks', async () => {
    // All "breaks" are spacers (invisible). Beam should be fully lit
    // throughout release, no break-induced dark pixels.
    const frame = await runGpuEffectTest({
      module: 'source.light.plasma_beam_cannon',
      bundle: 'lights',
      width: 64, height: 64,
      inputColor: [0.0, 0.0, 0.0, 1.0],
      ticks: 40,
      params: [
        ['auto_rate', 0.0],
        ['attack_s', 0.05],
        ['decay_s', 0.05],
        ['sustain_s', 0.10],
        ['release_s', 1.0],
        ['beam_color', [1.0, 1.0, 1.0]],
        ['intensity', 1.0],
        ['bar_target_all', 1.0],
        ['break_count_per_bar', 16],
        ['attractor_fraction', 0.0],
        ['spacer_fraction', 1.0],   // all spacers = invisible
        ['flicker_start_t', 1.0],   // disable flicker
        ['gate', 1.0],
      ],
      dumpName: 'pbc_release_no_breaks',
    });
    expect(frame.success).toBe(true);
    // All pixels should be lit since spacers don't eat the beam.
    frame.expectCoverage((c) => c.r > 200, { min: 0.95 });
  });

  it('attack phase shows only the seed_height band around seed_y', async () => {
    // 2 ticks: 1st transitions IDLE→ATTACK (time_in_phase=0), 2nd
    // accumulates time_in_phase=0.016. attack_curve = +1 → exp 1/8
    // → pow(0.016, 1/8) ≈ 0.524 → half_height ≈ 0.052 → beam covers
    // uv.y ∈ [~0.448, ~0.552], visible.
    const frame = await runGpuEffectTest({
      module: 'source.light.plasma_beam_cannon',
      bundle: 'lights',
      width: 64, height: 64,
      inputColor: [0.0, 0.0, 0.0, 1.0],
      ticks: 2,
      params: [
        ['attack_s', 1.0],
        ['attack_curve', 1.0],      // snap to near-full seed quickly
        ['seed_y', 0.5],
        ['seed_height', 0.2],
        ['beam_color', [1.0, 1.0, 1.0]],
        ['intensity', 1.0],
        ['bar_target_all', 1.0],
        ['gate', 1.0],
      ],
      dumpName: 'plasma_beam_cannon_attack_band',
    });
    expect(frame.success).toBe(true);

    // Inside the band (y near 32): lit.
    frame.expectPixelAt(32, 32, { r: 255, g: 255, b: 255 }, 5);
    // Well outside the band (y near 8 or 56): dark.
    frame.expectPixelAt(32, 8,  { r: 0, g: 0, b: 0 }, 5);
    frame.expectPixelAt(32, 56, { r: 0, g: 0, b: 0 }, 5);
  });

  it('attack_curve at default (linear) grows the seed from 0', async () => {
    // 2 ticks into a 1.0s attack with linear curve → t_in_phase=0.016
    // → half_height ≈ 0.1 * 0.016 = 0.0016 → essentially zero. The
    // beam shouldn't be visible yet, input passes through.
    const frame = await runGpuEffectTest({
      module: 'source.light.plasma_beam_cannon',
      bundle: 'lights',
      width: 64, height: 64,
      inputColor: [0.4, 0.4, 0.4, 1.0],
      ticks: 2,
      params: [
        ['attack_s', 1.0],
        ['attack_curve', 0.0],      // linear — slow start
        ['seed_y', 0.5],
        ['seed_height', 0.2],
        ['beam_color', [1.0, 1.0, 1.0]],
        ['intensity', 1.0],
        ['bar_target_all', 1.0],
        ['gate', 1.0],
      ],
      dumpName: 'plasma_beam_cannon_attack_linear_early',
    });
    expect(frame.success).toBe(true);
    frame.expectUniformColor({ r: 102, g: 102, b: 102, a: 255 }, 6);
  });
});
});
