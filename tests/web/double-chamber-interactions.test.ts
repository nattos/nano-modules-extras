import { runGpuEffectTest, Frame, forEachBackend } from '@nano/web/test/gpu-test-helpers';

/**
 * E2E for source.legacy.double_chamber's `interactions` layer (ported from
 * flow_swarm): P particles splat soft halos into a persistent square density
 * buffer, and the NEXT frame's p_update reads local crowding back out of it —
 * so particles can feel each other with no neighbour search.
 *
 * The sim is stochastic (hash-seeded per frame_index) and the harness's dt is
 * not perfectly stable across runs, so these assert on aggregate, causal
 * properties rather than exact pixels. The behavioural cases measure crowding
 * DIRECTLY, off the debug heat map (which IS the density buffer), rather than
 * inferring it from the rendered cloud — that's the quantity avoidance and
 * density-death exist to reduce.
 *
 * Scenario: no polynomial field, an outward `sink`, and a stiff boundary — so
 * the particles pile into a dense ring against the boundary, well away from the
 * (small, central) respawn disc. That separation matters: unlike flow_swarm,
 * double_chamber respawns onto a small centred disc, so a pile-up that sits ON
 * the spawn disc would just be re-fed by its own deaths.
 */
const RING = {
  p_count: 20000,
  p_point_size: 0.35,
  p_opacity: 1.0,
  exposure: 1.5,
  color_contrib: 0.0,      // white cloud
  field_speed: 0.0,        // no field — isolate the interaction forces
  jitter: 0.0,
  to_big: 0.0,
  big_count: 0,
  l_count: 0,
  bridger_count: 0,
  momentum: 0.0,
  ttl: 1.0,
  spawn_size: 0.1,         // tight central respawn disc
  sink: 0.6,               // outward → pile against the boundary
  boundary: 1.0,
  boundary_size: 0.42,
  boundary_stiffness: 16.0,
  boundary_speed: 2.0,
  interaction_radius: 0.04,
};

// Heat-ramp thresholds (density_debug.hlsl: v = d/(d+4), b = v*1.5 - 0.8), so
// the blue channel only lifts off where the field is genuinely packed:
//   b > 100 ≈ 15 overlapping neighbours — the crowded BAND (how wide the pile is)
//   b > 150 ≈ 56 overlapping neighbours — a real PILE-UP (how deep it stacks)
// Peak and area move independently: spreading a pile lowers the peak while
// widening the band, so the two together describe what a force did to the crowd.
const pileUp = (f: Frame) => f.countPixels(c => c.b > 150);
const band = (f: Frame) => f.countPixels(c => c.b > 100);

const ring = (over: Record<string, number>) =>
  Object.entries({ ...RING, ...over }) as [string, number][];

const runRing = (name: string, over: Record<string, number>) =>
  runGpuEffectTest({
    module: 'source.legacy.double_chamber',
    bundle: 'legacy',
    width: 128, height: 128,
    inputColor: [0.0, 0.0, 0.0, 1.0],
    params: ring(over),
    ticks: 30,
    renderEachTick: true,
    dumpName: name,
  });

// PUPPETEER ONLY — a recorded gap, not an oversight. On Metal the persistent
// density buffer reads back all zeros, so every behavioural case degenerates to
// 0-vs-0 (the metadata and "interactions off" cases still pass, which is why the
// gap is easy to miss). The two cases that DO pass natively prove the effect
// loads and renders; what doesn't survive is the cross-frame density feedback.
// Not this suite's bug, and not a one-off: it's the persistent-storage-buffer
// class — an effect that advances a sim buffer in place each frame doesn't
// integrate natively, while its persistent TEXTURES do. Characterised (with the
// ruled-out list) in web/KNOWN_ISSUES.md, "Persistent GPU storage buffers don't
// carry state across frames on Metal". Flip the list back to the default when
// that's fixed.
forEachBackend((backend) => {
describe(`Double Chamber interactions (density buffer) E2E (${backend})`, () => {
  jest.setTimeout(90000);

  it('declares the interaction parameters', async () => {
    const frame = await runGpuEffectTest({
      module: 'source.legacy.double_chamber',
      bundle: 'legacy',
      inputColor: [0.0, 0.0, 0.0, 1.0],
      dumpName: 'dc_interactions_metadata',
    });
    expect(frame.success).toBe(true);
    const names = frame.params.map(p => p.name);
    for (const n of ['interactions', 'interaction_radius', 'density_threshold',
                     'density_death', 'avoid', 'avoid_curl', 'avoid_noise',
                     'stream', 'stream_density', 'debug_density']) {
      expect(names).toContain(n);
    }
  });

  it('debug_density renders the crowding buffer as a heat map', async () => {
    // The most direct proof the splat pass runs and the buffer carries content:
    // with debug on, tex_out IS the heat map (black → red → yellow → white), so
    // a lit frame means particles actually accumulated into the density buffer.
    // The ramp is monotone per channel, so red >= green >= blue at every pixel.
    const frame = await runRing('dc_interactions_debug_density',
                                { interactions: 1, debug_density: 1 });
    expect(frame.success).toBe(true);

    expect(frame.countPixels(c => c.r > 40)).toBeGreaterThan(100);  // real crowding

    let rampViolations = 0;
    frame.forEachPixel(c => { if (c.r < c.g - 2 || c.g < c.b - 2) rampViolations++; });
    expect(rampViolations).toBe(0);     // heat ramp: r >= g >= b
  });

  it('avoidance breaks up the crowding it feels', async () => {
    // Avoidance pushes each particle down the local crowding gradient, so the
    // packed ring fans out and the density peaks flatten.
    const base = await runRing('dc_interactions_avoid_off',
                               { interactions: 1, debug_density: 1 });
    const avoided = await runRing('dc_interactions_avoid_on',
                                  { interactions: 1, debug_density: 1, avoid: 1.0, avoid_noise: 0.1 });
    expect(base.success && avoided.success).toBe(true);

    // eslint-disable-next-line no-console
    console.log('[avoid] pileUp off/on:', pileUp(base), pileUp(avoided),
                '| band off/on:', band(base), band(avoided));

    // The pile flattens...
    expect(pileUp(avoided)).toBeLessThan(pileUp(base) * 0.8);
    // ...by spreading, so the crowded band gets WIDER even as its peak drops.
    // (Both directions matter: a force that merely deleted particles would drop
    // the peak too, but would shrink the band rather than widen it.)
    expect(band(avoided)).toBeGreaterThan(band(base));
  });

  it('density death thins an over-packed region', async () => {
    // Over `density_threshold` crowding, particles get a chance to die — and a
    // density kill REDISTRIBUTES them uniformly across the chamber disc, so the
    // ring's pile-up drains into the interior instead of stacking deeper.
    const base = await runRing('dc_interactions_death_off',
                               { interactions: 1, debug_density: 1 });
    const thinned = await runRing('dc_interactions_death_on',
                                  { interactions: 1, debug_density: 1,
                                    density_death: 1.0, density_threshold: 2.0 });
    expect(base.success && thinned.success).toBe(true);

    // eslint-disable-next-line no-console
    console.log('[death] pileUp off/on:', pileUp(base), pileUp(thinned),
                '| band off/on:', band(base), band(thinned));

    // The deepest stacks thin out, and the culled particles show up spread
    // through the chamber. Guarding the second half matters: an earlier version
    // respawned density kills onto the (small, central) spawn disc, which drained
    // the ring but collapsed the whole pool onto the spawn point — it would have
    // passed a peak-only assertion while looking completely wrong.
    expect(pileUp(thinned)).toBeLessThan(pileUp(base) * 0.9);
    expect(band(thinned)).toBeGreaterThan(band(base));
  });

  it('streaming steers particles with their local group', async () => {
    // Stream reads the group's mean velocity out of the density buffer's .gb
    // channels and steers each particle toward (+) or away from (-) it. Paired
    // with the interactions-off case below: the same knob must do nothing when
    // the bool is off, and something when it is on.
    const base = await runRing('dc_interactions_stream_off', { interactions: 1 });
    const aligned = await runRing('dc_interactions_stream_on',
                                  { interactions: 1, stream: 1.0, stream_density: 2.0 });
    expect(base.success && aligned.success).toBe(true);

    aligned.expectDifferentFrom(base, 100);
  });

  it('interactions off leaves the sim untouched by the interaction knobs', async () => {
    // With the bool off, the update binds a 1x1 zero density buffer and skips
    // the reads entirely — so cranking avoid/death must change nothing. Compare
    // the RENDERED cloud (debug off) via its coverage, which the interaction
    // forces would visibly alter if they were live.
    const off = await runRing('dc_interactions_off_baseline', { interactions: 0 });
    const offCranked = await runRing('dc_interactions_off_cranked',
                                     { interactions: 0, avoid: 1.0, density_death: 1.0,
                                       density_threshold: 1.0, stream: 1.0 });
    expect(off.success && offCranked.success).toBe(true);

    const lit = (f: Frame) => f.countPixels(c => c.r + c.g + c.b > 24);
    // eslint-disable-next-line no-console
    console.log('[off] lit baseline/cranked:', lit(off), lit(offCranked));

    // Same sim → same coverage, up to the run-to-run stochastic wobble.
    expect(Math.abs(lit(offCranked) - lit(off))).toBeLessThan(lit(off) * 0.1);
  });
});
}, ['puppeteer']);
