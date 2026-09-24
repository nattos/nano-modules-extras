/**
 * The extras' bundles through nano-modules' node-side WasmHost harness
 * (@nano/web/src/testing/node-wasm-host): the looper's host-level behaviour,
 * and the schema metadata the nano and lights effects publish. Needs the
 * bundles in nano-modules' build/wasm (build_all.sh --extras <this repo>).
 */

import { captureSchemas, loadHost, wasmPath } from '@nano/web/src/testing/node-wasm-host';

describe('WasmHost: control.nanolooper', () => {
  const loadLooper = () => loadHost(wasmPath('nano'), 'control.nanolooper');
  it('on_param_change triggers audio callback', async () => {
    const { host, module } = await loadLooper();
    module.init();
    host.frameState.barPhase = 0.1;

    let triggeredChannel = -1;
    host.onAudioTrigger = (ch) => { triggeredChannel = ch; };

    host.notifyStatePatched(module, [{ op: 'replace', path: 'trigger_1', value: 1.0 }]);
    expect(triggeredChannel).toBe(0);
  });

  it('on_state_patched reads grid from canonical state', async () => {
    const { host, module } = await loadLooper();
    module.init();
    host.frameState.barPhase = 0.1;

    // Trigger some events normally
    host.notifyStatePatched(module, [{ op: 'replace', path: 'trigger_1', value: 1.0 }]);
    host.notifyStatePatched(module, [{ op: 'replace', path: 'trigger_1', value: 0.0 }]);
    module.tick(0.016);

    // Now externally modify the canonical state (simulating a client edit)
    host.pluginState = {
      phase: 0,
      recording: false,
      event_count: 3,
      grid: [[0, 4], [8], [], []]
    };

    // Notify the module via state patches
    host.notifyStatePatched(module as any, [{ op: 'replace', path: 'grid', value: host.pluginState.grid }]);

    // Tick to publish updated state — the module should now reflect the edited grid
    host.frameState.viewportW = 1920;
    host.frameState.viewportH = 1080;
    module.tick(0.016);

    // After tick, the module publishes its internal state which should match the edit
    expect(host.pluginState.event_count).toBe(3);
    expect(host.pluginState.grid[0]).toEqual([0, 4]);
    expect(host.pluginState.grid[1]).toEqual([8]);
    expect(host.pluginState.grid[2]).toEqual([]);
    expect(host.pluginState.grid[3]).toEqual([]);
  });

  it('on_state_patched preserves all channels when editing one', async () => {
    const { host, module } = await loadLooper();
    module.init();
    host.frameState.barPhase = 0.0;

    // Set up events on all 4 channels via the state
    host.pluginState = {
      phase: 0, recording: false, event_count: 4,
      grid: [[1], [3], [5], [7]]
    };
    host.notifyStatePatched(module as any, [{ op: 'replace', path: 'grid', value: host.pluginState.grid }]);
    module.tick(0.016);

    // Verify all 4 channels loaded
    expect(host.pluginState.event_count).toBe(4);
    expect(host.pluginState.grid).toEqual([[1], [3], [5], [7]]);

    // Now edit: remove only channel 0's event
    host.pluginState = {
      phase: 0, recording: false, event_count: 3,
      grid: [[], [3], [5], [7]]
    };
    host.notifyStatePatched(module as any, [{ op: 'replace', path: 'grid', value: host.pluginState.grid }]);
    module.tick(0.016);

    // Channels 1-3 must still have their events
    expect(host.pluginState.event_count).toBe(3);
    expect(host.pluginState.grid[0]).toEqual([]);
    expect(host.pluginState.grid[1]).toEqual([3]);
    expect(host.pluginState.grid[2]).toEqual([5]);
    expect(host.pluginState.grid[3]).toEqual([7]);
  });
});

describe('schema metadata of the extras (groups / names / help)', () => {
  it('brutal_fold emits groups, per-field name/short/group, and a help field', async () => {
    const schemas = await captureSchemas(wasmPath('nano'), ['source.brutal_fold']);
    const schema = schemas.get('source.brutal_fold');
    expect(schema).toBeTruthy();

    // First-class groups with metadata.
    expect(schema.groups?.shape?.name).toBe('Form');
    expect(typeof schema.groups?.shape?.help).toBe('string');
    expect(schema.groups.shape.help).toContain('atlas');
    expect(schema.groups?.volumetrics?.name).toBe('Volumetrics');
    expect(schema.groups?.autopilot?.name).toBe('Autopilot');

    // Per-field display name + short name + group id.
    expect(schema.fields?.complexity?.name).toBe('Complexity');
    expect(schema.fields?.complexity?.short).toBe('Cplx');
    expect(schema.fields?.complexity?.group).toBe('shape');
    expect(schema.fields?.vol_softness_xy?.name).toBe('Screen Softness');
    expect(schema.fields?.vol_softness_xy?.group).toBe('volumetrics');

    // Help field — a help slot with no instance-state backing.
    expect(schema.fields?.intro?.type).toBe('help');
    expect(schema.fields?.intro?.io).toBe(0);
    expect(schema.fields?.intro?.default).toContain('Brutal Fold');
  });

  it('nano bundle: edited effects emit VALID schema JSON with groups/labels/intro help', async () => {
    const ids = [
      'source.particles.flash_particles', 'source.particles.flow_swarm', 'filter.height_from_gradient',
      'motion.local_delay', 'mod.shaper.spectral', 'motion.field', 'source.phase_fold',
      'source.shape_fold', 'mod.source.spectral_lfo', 'source.brutal_fold',
    ];
    const schemas = await captureSchemas(wasmPath('nano'), ids);
    expect(schemas.size).toBe(ids.length);   // all valid JSON (no truncation on the 41-field phase_fold)
    for (const id of ids) {
      const s = schemas.get(id);
      expect(s?.fields?.intro?.type).toBe('help');
      expect(Object.keys(s?.groups ?? {}).length).toBeGreaterThan(0);
    }
  });

  it('lights bundle: every effect emits VALID schema JSON with groups + labels + intro help', async () => {
    const ids = [
      'source.light.chroma_wave', 'source.light.orthomod', 'warp.dispersion',
      'source.light.plasma_beam_cannon', 'source.light.motion_blobs', 'filter.lights_sim',
      'source.light.side_jet', 'source.light.bounce_resonator', 'source.light.strobe_channel',
      'source.light.soft_glow', 'source.light.tingle_top', 'filter.glitch.block_dehance',
    ];
    // captureSchemas JSON.parses each — a truncated/corrupt schema would throw here.
    const schemas = await captureSchemas(wasmPath('lights'), ids);
    expect(schemas.size).toBe(ids.length);   // all present, all valid JSON

    for (const [id, schema] of schemas) {
      // Every lights effect gained groups + an intro help field + labelled inputs.
      expect(Object.keys(schema.groups ?? {}).length).toBeGreaterThan(0);
      expect(schema.fields?.intro?.type).toBe('help');
      const labelled = Object.values(schema.fields ?? {})
        .filter((f: any) => f?.type !== 'help' && (f?.io & 1) && typeof f?.name === 'string');
      expect(labelled.length).toBeGreaterThan(0);
      // Every labelled field points at a declared group.
      for (const f of labelled) {
        if ((f as any).group !== undefined) {
          expect(schema.groups?.[(f as any).group]).toBeTruthy();
        }
      }
    }
  });
});
