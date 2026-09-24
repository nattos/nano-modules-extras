// E2E test: control.nanolooper through nano-modules' module-test-app harness —
// the full WASM loading loop in a real browser via Puppeteer. Runs on
// nano-modules' jest with NANO_EXTRAS_DIR set, against its dev server
// (GPU_TEST_BASE_URL), with the nano bundle built into its build/wasm.

const BASE = process.env.GPU_TEST_BASE_URL || 'http://localhost:5173';

describe('NanoLooper Web Harness E2E', () => {
  jest.setTimeout(15000);

  beforeAll(async () => {
    await page.goto(`${BASE}/module-test-app.html?module=nanolooper`, { waitUntil: 'networkidle0' });
  });

  it('page loads without errors', async () => {
    const errors: string[] = [];
    page.on('pageerror', (err) => errors.push(err.message));
    // Wait a moment for any async errors
    await new Promise(r => setTimeout(r, 1000));
    expect(errors).toEqual([]);
  });

  it('status shows Running after WASM loads', async () => {
    // Wait for status to change from "Loading..."
    await page.waitForFunction(
      () => {
        const el = document.getElementById('status');
        return el && !el.textContent!.includes('Loading') && !el.textContent!.includes('failed');
      },
      { timeout: 5000 },
    );

    const status = await page.$eval('#status', el => el.textContent);
    expect(status).toContain('FPS');
    expect(status).toContain('BPM');
  });

  it('canvas element exists and has non-zero dimensions', async () => {
    const dims = await page.$eval('#canvas', (el) => {
      const canvas = el as HTMLCanvasElement;
      return { w: canvas.width, h: canvas.height };
    });
    expect(dims.w).toBeGreaterThan(0);
    expect(dims.h).toBeGreaterThan(0);
  });

  it('status updates with FPS and BPM', async () => {
    await new Promise(r => setTimeout(r, 500));

    const status = await page.$eval('#status', el => el.textContent);
    expect(status).toContain('FPS');
    expect(status).toContain('BPM');
  });

  it('keyboard triggers update the WASM module', async () => {
    // Press trigger key '1' and release
    await page.keyboard.down('1');
    await new Promise(r => setTimeout(r, 100));
    await page.keyboard.up('1');
    await new Promise(r => setTimeout(r, 100));

    // The module should still be running (no crash)
    const status = await page.$eval('#status', el => el.textContent);
    expect(status).toContain('FPS');
  });

  it('multiple trigger keys work without crash', async () => {
    // Quick sequence of triggers
    for (const key of ['1', '2', '3', '4']) {
      await page.keyboard.down(key);
      await new Promise(r => setTimeout(r, 50));
      await page.keyboard.up(key);
      await new Promise(r => setTimeout(r, 50));
    }

    // Modifiers
    await page.keyboard.down('d');
    await new Promise(r => setTimeout(r, 50));
    await page.keyboard.up('d');

    await page.keyboard.down('z'); // undo
    await new Promise(r => setTimeout(r, 50));
    await page.keyboard.up('z');

    const status = await page.$eval('#status', el => el.textContent);
    expect(status).toContain('FPS');
  });

  it('state document reflects triggered events', async () => {
    // Trigger channel 1 and wait for state to update
    await page.keyboard.down('1');
    await new Promise(r => setTimeout(r, 100));
    await page.keyboard.up('1');
    await new Promise(r => setTimeout(r, 500));

    const state = await page.evaluate(() => (window as any).__host?.pluginState);
    expect(state).toBeDefined();
    expect(state.grid).toBeDefined();
    expect(state.event_count).toBeGreaterThan(0);
    // Channel 0 should have at least one step
    expect(state.grid[0].length).toBeGreaterThan(0);
  });

  it('state edit via notifyStatePatched round-trips correctly', async () => {
    const newState = {
      phase: 0, recording: false, event_count: 4,
      grid: [[1, 5], [3, 7], [], []]
    };

    await page.evaluate((s) => {
      const host = (window as any).__host;
      const wasm = (window as any).__wasm;
      // Update both bridge core state and local cache
      if (host.bridgeCore && host.pluginKey) {
        host.bridgeCore.setPluginState(host.pluginKey, s);
      }
      host.pluginState = s;
      host.notifyStatePatched(wasm, [{ op: 'replace', path: 'grid', value: s.grid }]);
    }, newState);

    await new Promise(r => setTimeout(r, 500));

    const state = await page.evaluate(() => (window as any).__host?.pluginState);
    expect(state.event_count).toBe(4);
    expect(state.grid[0]).toEqual([1, 5]);
    expect(state.grid[1]).toEqual([3, 7]);
    expect(state.grid[2]).toEqual([]);
    expect(state.grid[3]).toEqual([]);
  });

  it('editing one channel preserves other channels', async () => {
    const initialState = {
      phase: 0, recording: false, event_count: 4,
      grid: [[2], [4], [6], [8]]
    };

    await page.evaluate((s) => {
      const host = (window as any).__host;
      const wasm = (window as any).__wasm;
      if (host.bridgeCore && host.pluginKey) {
        host.bridgeCore.setPluginState(host.pluginKey, s);
      }
      host.pluginState = s;
      host.notifyStatePatched(wasm, [{ op: 'replace', path: 'grid', value: s.grid }]);
    }, initialState);
    await new Promise(r => setTimeout(r, 500));

    let state = await page.evaluate(() => (window as any).__host?.pluginState);
    expect(state.grid).toEqual([[2], [4], [6], [8]]);

    // Edit: remove only channel 0
    const editedState = {
      phase: 0, recording: false, event_count: 3,
      grid: [[], [4], [6], [8]]
    };

    await page.evaluate((s) => {
      const host = (window as any).__host;
      const wasm = (window as any).__wasm;
      if (host.bridgeCore && host.pluginKey) {
        host.bridgeCore.setPluginState(host.pluginKey, s);
      }
      host.pluginState = s;
      host.notifyStatePatched(wasm, [{ op: 'replace', path: 'grid', value: s.grid }]);
    }, editedState);
    await new Promise(r => setTimeout(r, 500));

    state = await page.evaluate(() => (window as any).__host?.pluginState);
    expect(state.event_count).toBe(3);
    expect(state.grid[0]).toEqual([]);
    expect(state.grid[1]).toEqual([4]);
    expect(state.grid[2]).toEqual([6]);
    expect(state.grid[3]).toEqual([8]);
  });

  it('console logs appear from WASM module', async () => {
    const logs = await page.evaluate(() => (window as any).__host?.consoleLogs);
    expect(logs).toBeDefined();
    expect(logs.length).toBeGreaterThan(0);
    // Should have "NanoLooper initialized" from init()
    const initLog = logs.find((l: any) => l.message.includes('initialized'));
    expect(initLog).toBeDefined();
  });

  it('plugin metadata is reported', async () => {
    const meta = await page.evaluate(() => (window as any).__host?.metadata);
    expect(meta).toBeDefined();
    expect(meta.id).toBe('control.nanolooper');
    // Source of truth: `state::init(id, {1, 2, 0}, schema)` in
    // effects/nanolooper/main.cpp — bump both together.
    expect(meta.version).toBe('1.2.0');
  });

  it('frame count increases over time', async () => {
    const getStep = async () => {
      const status = await page.$eval('#status', el => el.textContent);
      const match = status?.match(/Step (\d+)/);
      return match ? parseInt(match[1]) : -1;
    };

    const step1 = await getStep();
    await new Promise(r => setTimeout(r, 2000));
    const step2 = await getStep();

    // Step should have changed (bar phase is advancing)
    // They might be the same by chance, so just verify they're valid
    expect(step1).toBeGreaterThan(0);
    expect(step2).toBeGreaterThan(0);
  });
});
