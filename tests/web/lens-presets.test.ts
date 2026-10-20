/**
 * The Lens inspector's PRESET strip (web/src/editors/lens-inspector.ts), in the
 * IDE: a <bi-seg> whose pick applies a whole look -- the preset plus every
 * character param it bundles -- as one undo point. bi-seg picks by a PRESS (a
 * continuous edit a held press may drag across segments), so the strip must
 * apply looks on that path: each segment crossed previews its look, the last
 * one undone first, and the release keeps the one it ends on.
 */
const BASE = process.env.GPU_TEST_BASE_URL || 'http://localhost:5173';
const DEEP_ALL = `(sel) => {
  const out = [];
  const walk = (root) => { for (const el of root.querySelectorAll('*')) { if (el.matches(sel)) out.push(el); if (el.shadowRoot) walk(el.shadowRoot); } };
  walk(document);
  return out;
}`;

async function until<T>(fn: () => Promise<T>, ok: (v: T) => boolean, label: string, ms = 30_000): Promise<T> {
  const deadline = Date.now() + ms;
  let last: T | undefined;
  while (Date.now() < deadline) {
    last = await fn();
    if (ok(last)) return last;
    await new Promise((r) => setTimeout(r, 100));
  }
  throw new Error(`timeout: ${label}; last=${JSON.stringify(last)}`);
}

describe('Lens preset strip (IDE)', () => {
  jest.setTimeout(90_000);

  it('a press applies the look it ends on as one undo point', async () => {
    await page.setViewport({ width: 1500, height: 950 });
    await page.goto(`${BASE}/ide.html`, { waitUntil: 'domcontentloaded' });
    await until(() => page.evaluate(`(${DEEP_ALL})('[data-ide-effect="filter.blur.lens"]').length`) as Promise<number>,
      (n) => n > 0, 'Lens in the effects list');
    await page.evaluate(`(${DEEP_ALL})('[data-ide-effect="filter.blur.lens"]')[0].click()`);
    await until(() => page.evaluate(() => (window as any).arrangementStore.composition.playground?.effectId ?? null),
      (id) => id === 'filter.blur.lens', 'the Lens playground');

    const lens = () => page.evaluate(() => {
      const st = (window as any).arrangementStore;
      const p = st.composition.playground;
      const s = st.trackById(p.trackId).sketch.devices.find((d: any) => d.id === p.deviceId).state ?? {};
      // (Absent = the schema's default.)
      return { preset: s.preset ?? 0, swirl: s.swirl ?? 0, anamorphic: s.anamorphic ?? 0, blades: s.blades ?? 7 };
    });
    const before = await lens();
    expect(before.preset).toBe(0);

    // The strip's segments, by label.
    const seg = (label: string) => page.evaluate(`(() => {
      const bar = (${DEEP_ALL})('lens-inspector')[0].shadowRoot.querySelector('bi-seg[path="preset"]');
      const b = bar.shadowRoot.querySelector('button[data-choice=${JSON.stringify(label)}]');
      const r = b.getBoundingClientRect();
      return { x: r.left + r.width / 2, y: r.top + r.height / 2 };
    })()`) as Promise<{ x: number; y: number }>;
    await until(() => page.evaluate(`(${DEEP_ALL})('lens-inspector').length`) as Promise<number>, (n) => n > 0, 'the inspector');

    // Press Anamorphic: its look shows while held.
    const ana = await seg('Anamorphic');
    await page.mouse.move(ana.x, ana.y);
    await page.mouse.down();
    const held = await until(lens, (s) => s.preset === 3, 'Anamorphic previewed');
    expect(held.anamorphic).toBe(0.5);
    // Slide on to Vintage and let go there: Vintage stays, Anamorphic's own
    // params go back to what they were.
    const vin = await seg('Vintage');
    await page.mouse.move(vin.x, vin.y, { steps: 6 });
    await page.mouse.up();
    const after = await until(lens, (s) => s.preset === 2, 'Vintage applied');
    expect(after).toEqual({ preset: 2, swirl: 0.75, anamorphic: before.anamorphic, blades: 8 });

    // One undo takes the whole look back.
    await page.evaluate(() => (window as any).arrangementStore.undo());
    expect(await until(lens, (s) => s.preset === 0, 'undone')).toEqual(before);
  });
});
