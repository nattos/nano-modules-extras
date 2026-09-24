import { runEngineTest } from '@nano/web/test/engine-test-helpers';
import type { Sketch } from '@nano/web/src/sketch-types';

// source.mesh.three_walls through core's util.triptych — the layout the
// triptych card exists for. Moved from nano-modules' triptych suite with the
// effect.
describe('Three Walls in a Triptych', () => {
  jest.setTimeout(120000);

  const MODULES = ['com.nano.core', 'com.nano.lights'];
  // Stretch, so each panel fills its third exactly.
  const STRETCH = { fit_mode: 1 };

  // What the card is for. Three Walls' three outputs are three walls of one
  // room; laid out in this order they are the room, flat.
  it('lays out a Three Walls room', async () => {
    const sketch: Sketch = {
      anchor: null,
      wires: [
        { id: 'wl', src: { instanceKey: 'tw@0', field: 'left_out' },
          dest: { instanceKey: 'tp@0', field: 'left_in' } },
        { id: 'wr', src: { instanceKey: 'tw@0', field: 'right_out' },
          dest: { instanceKey: 'tp@0', field: 'right_in' } },
      ],
      chain: [
        { type: 'module', module_type: 'source.solid_color', instance_key: 'bg@0',
          params: { color: [0, 0, 0] } },
        { type: 'module', module_type: 'source.mesh.three_walls', instance_key: 'tw@0',
          params: { grain: 0, scanline: 0, chroma_bleed: 0,
                    // Frozen pose, and a size that puts the nearest frame onto
                    // the side walls while the others are still on the back.
                    resonate: 1, resonate_f0: 0, resonate_f1: 0, quad_size: 1.3 } },
        { type: 'module', module_type: 'util.triptych', instance_key: 'tp@0',
          params: STRETCH },
      ],
    } as Sketch;

    const r = await runEngineTest({
      width: 600, height: 200, modules: MODULES,
      commands: [
        { type: 'createSketch', sketchId: 'trip_room', sketch },
        { type: 'setTracePoints', tracePoints: [
          { id: 'out', target: { type: 'sketch_output', sketchId: 'trip_room' } }]},
      ],
      waitFrames: 25, captureTraceIds: ['out'], dumpName: 'trip_room',
    });
    expect(r.success).toBe(true);

    // All three panels carry light, and the two side ones are mirror images —
    // which is the check that they arrived in the right slots and neither got
    // dropped.
    const f = r.trace('out');
    const third = 600 / 3;
    const lit = [0, 0, 0];
    f.forEachPixel((p, x) => {
      const col = Math.min(2, Math.floor(x / third));
      if ((p.r + p.g + p.b) / 3 > 40) lit[col]++;
    });
    expect(lit[0]).toBeGreaterThan(50);
    expect(lit[1]).toBeGreaterThan(50);
    expect(lit[2]).toBeGreaterThan(50);
    expect(Math.abs(lit[0] - lit[2])).toBeLessThan(Math.max(lit[0], lit[2]) * 0.25);
  });
});
