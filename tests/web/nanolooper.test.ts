import { runEngineTest } from '@nano/web/test/engine-test-helpers';

// Per-effect tests for `control.nanolooper` against the shipping `nano`
// bundle. nanolooper has no texture I/O — its render output goes through
// the canvas draw_list, not a GPU texture. UI-side behaviour
// (keyboard triggers, state edits, audio callback) is covered by
// `e2e.test.ts` and `wasm-host.test.ts`. This test verifies that the
// effect registers and ticks correctly inside the engine worker.

describe('NanoLooper Effect E2E (engine worker)', () => {
  jest.setTimeout(30000);

  it('registers in the engine and reports its schema', async () => {
    const result = await runEngineTest({
      width: 32, height: 32,
      modules: ['com.nano.nano'],
      commands: [
        { type: 'instantiateEffect', effectId: 'control.nanolooper' },
      ],
      waitFrames: 5,
      dumpName: 'nanolooper_metadata',
    });

    expect(result.success).toBe(true);
    const nl = result.state.plugins.find((p: any) => p.id === 'control.nanolooper');
    expect(nl).toBeDefined();
    // Key carries an instance-counter suffix (@N); the bundle warmup registers
    // the schema-only instance and instantiateEffect makes the live one, so the
    // exact N isn't load-bearing — just assert the id@N shape.
    expect(nl!.key).toMatch(/^control\.nanolooper@\d+$/);

    // Spot-check a few schema fields — full coverage lives in wasm-host.test.ts.
    const paramNames = nl!.params.map((p: any) => p.name);
    expect(paramNames).toContain('trigger_1');
    expect(paramNames).toContain('mute');
    expect(paramNames).toContain('show_overlay');
    expect(paramNames).toContain('synth_gain');
  });

  it('publishes initial pluginState with empty grid', async () => {
    const result = await runEngineTest({
      width: 32, height: 32,
      modules: ['com.nano.nano'],
      commands: [
        { type: 'instantiateEffect', effectId: 'control.nanolooper' },
      ],
      waitFrames: 10,
      dumpName: 'nanolooper_initial',
    });

    expect(result.success).toBe(true);
    const nl = result.state.plugins.find((p: any) => p.id === 'control.nanolooper');
    const pluginState = result.state.pluginStates?.[nl!.key];
    expect(pluginState).toBeDefined();
    // Grid is the canonical "what's recorded" surface — should exist and
    // start empty for each of the 4 channels.
    expect(Array.isArray(pluginState.grid)).toBe(true);
    expect(pluginState.grid.length).toBe(4);
    for (const channel of pluginState.grid) {
      expect(Array.isArray(channel)).toBe(true);
      expect(channel.length).toBe(0);
    }
    // Default loop length: 1 bar of 16 steps.
    expect(pluginState.loop_steps).toBe(16);
  });

  it('extends the loop to 4 bars via the bars param and publishes loop_steps', async () => {
    const result = await runEngineTest({
      width: 32, height: 32,
      modules: ['com.nano.nano'],
      commands: [
        {
          type: 'createSketch',
          sketchId: 'nl_sketch',
          sketch: {
            anchor: null,
            chain: [
              {
                type: 'module',
                module_type: 'control.nanolooper',
                instance_key: 'nl@0',
                params: {},
              },
            ],
          },
        },
        {
          type: 'setParam',
          sketchId: 'nl_sketch', colIdx: 0, chainIdx: 0,
          paramKey: 'bars', value: 4,
        },
      ],
      waitFrames: 10,
      dumpName: 'nanolooper_bars',
    });

    expect(result.success).toBe(true);
    const pluginState = result.state.pluginStates?.['nl@0']
      ?? result.state.pluginStates?.['control.nanolooper@0'];
    expect(pluginState).toBeDefined();
    // The published loop length follows the param (4 bars × 16 steps)...
    expect(pluginState.loop_steps).toBe(64);
    // ...and the sequencer keeps ticking against the longer loop.
    expect(pluginState.grid.length).toBe(4);
    expect(pluginState.phase).toBeLessThan(64);
  });

  it('records a trigger into the grid via setParam', async () => {
    // Trigger channel 1 at bar phase 0.0625 → step 1 (0.0625 = 1/16).
    // The trigger is an event field; firing it once should record into channel 0.
    const result = await runEngineTest({
      width: 32, height: 32,
      modules: ['com.nano.nano'],
      commands: [
        {
          type: 'createSketch',
          sketchId: 'nl_sketch',
          sketch: {
            anchor: null,
            chain: [
              {
                type: 'module',
                module_type: 'control.nanolooper',
                instance_key: 'nl@0',
                // Start with record on so the trigger lands in the grid.
                params: { record: 1.0 },
              },
            ],
          },
        },
        {
          type: 'setParam',
          sketchId: 'nl_sketch', colIdx: 0, chainIdx: 0,
          paramKey: 'trigger_1', value: 1.0,
        },
      ],
      waitFrames: 10,
      dumpName: 'nanolooper_trigger',
    });

    expect(result.success).toBe(true);
    const pluginState = result.state.pluginStates?.['nl@0']
      ?? result.state.pluginStates?.['control.nanolooper@0'];
    expect(pluginState).toBeDefined();
    // Some step on channel 0 should have been recorded.
    expect(pluginState.grid[0].length).toBeGreaterThan(0);
  });
});
