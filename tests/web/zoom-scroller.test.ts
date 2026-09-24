import { runGpuEffectTest, forEachBackend } from '@nano/web/test/gpu-test-helpers';
import { runEngineTest } from '@nano/web/test/engine-test-helpers';
import type { Sketch } from '@nano/web/src/sketch-types';

/**
 * E2E for warp.legacy.zoom_scroller — "Zoom Scroller", the v2 port of the
 * Resolume Wire "ZoomScroller" procedural pan/zoom sequence camera.
 *
 * Two things are observable in the harness:
 *  - The GIZMO box draws white pixels over a solid input while the camera is
 *    mid-pan (the pan/zoom of a solid colour is itself invisible).
 *  - The pan/zoom TRANSFORM is verified by chaining a deterministic structured
 *    generator (source.grid) → zoom_scroller and comparing different zoom
 *    levels / time points (the grid must move).
 *
 * The state machine advances on tick(dt); the runner uses dt = 0.016 s, and
 * `renderEachTick` advances persistent state per frame.
 */
forEachBackend((backend) => {
describe(`Zoom Scroller (warp.legacy.zoom_scroller) E2E (${backend})`, () => {
  jest.setTimeout(60000);

  const SOLID: [number, number, number, number] = [0.15, 0.30, 0.55, 1.0];

  it('declares metadata and its parameters', async () => {
    const frame = await runGpuEffectTest({
      module: 'warp.legacy.zoom_scroller', bundle: 'legacy',
      inputColor: SOLID,
      dumpName: 'zoom_metadata',
    });
    expect(frame.success).toBe(true);
    expect(frame.metadata?.id).toBe('warp.legacy.zoom_scroller');
    const names = frame.params.map(p => p.name);
    // (gizmo_color is an rgba/vec4 field — the harness flat param list omits
    //  vector/colour fields, so it isn't asserted here; it is in the schema.)
    for (const n of ['min_scale', 'max_scale', 'scale_variance', 'target_min_radius',
                     'target_max_radius', 'sub_steps', 'sub_step_frames', 'flicker_rate',
                     'sub_delay', 'sequence_delay', 'show_gizmo', 'retrigger', 'filter_mode',
                     'target_step_distance', 'origin_center_bias', 'gizmo_size',
                     'gizmo_alpha', 'gizmo_motion_scale']) {
      expect(names).toContain(n);
    }
    // sub_steps is an int selector-like field with the declared range.
    const ss = frame.params.find(p => p.name === 'sub_steps');
    expect(ss?.min).toBe(1);
    expect(ss?.max).toBe(30);
  });

  it('draws the motion gizmo box over the frame while panning', async () => {
    // Mid-pan (a handful of metronome ticks in) the white outline box is drawn.
    // 16 render-ticks ≈ 0.256 s × 15 Hz ≈ 3.8 metronome ticks → frame_counter≈3,
    // still inside the first pan (total = 6×4 = 24) → gizmo visible.
    const on = await runGpuEffectTest({
      module: 'warp.legacy.zoom_scroller', bundle: 'legacy', width: 128, height: 128,
      inputColor: SOLID,
      params: [['show_gizmo', 1], ['gizmo_alpha', 1.0], ['gizmo_width', 1.0]],
      ticks: 16, renderEachTick: true,
      dumpName: 'zoom_gizmo_on',
    });
    expect(on.success).toBe(true);

    // Without the gizmo, a solid input pans/zooms to the same solid colour.
    const off = await runGpuEffectTest({
      module: 'warp.legacy.zoom_scroller', bundle: 'legacy', width: 128, height: 128,
      inputColor: SOLID,
      params: [['show_gizmo', 0]],
      ticks: 16, renderEachTick: true,
      dumpName: 'zoom_gizmo_off',
    });
    expect(off.success).toBe(true);

    // Gizmo-off frame is uniform (the input colour); gizmo-on has bright box pixels.
    let brightOn = 0;
    on.forEachPixel((c) => { if (c.r > 200 && c.g > 200 && c.b > 200) brightOn++; });
    let brightOff = 0;
    off.forEachPixel((c) => { if (c.r > 200 && c.g > 200 && c.b > 200) brightOff++; });
    expect(brightOn).toBeGreaterThan(20);   // a real box outline
    expect(brightOff).toBe(0);              // solid pan/zoom adds nothing
  });

  it('gizmo_width scales from bold to hairline to gone', async () => {
    // The line width must reach a true hairline and fully disappear at 0 (for
    // "playability"), with a bolder line at the top — no fixed minimum width.
    const litCount = async (gw: number, dump: string) => {
      const f = await runGpuEffectTest({
        module: 'warp.legacy.zoom_scroller', bundle: 'legacy', width: 128, height: 128,
        inputColor: SOLID,
        params: [['show_gizmo', 1], ['gizmo_alpha', 1.0], ['gizmo_width', gw]],
        ticks: 16, renderEachTick: true, dumpName: dump,
      });
      expect(f.success).toBe(true);
      // Count pixels lifted toward the white box vs the blue background.
      let lit = 0;
      f.forEachPixel((c) => { if ((c.r - 38) + (c.g - 76) > 15) lit++; });
      return lit;
    };
    const bold     = await litCount(1.0,  'zoom_gw_bold');
    const hairline = await litCount(0.05, 'zoom_gw_hairline');
    const gone     = await litCount(0.0,  'zoom_gw_gone');
    expect(bold).toBeGreaterThan(hairline);   // bolder at the top
    expect(hairline).toBeGreaterThan(0);      // a faint hairline still draws
    expect(gone).toBe(0);                     // width 0 → fully disappears
  });

});
});

// The cases below drive runEngineTest — the engine harness page (executor.wasm,
// wires, trace points) — which has no native runner, so they stay puppeteer-only.
// The comp runner is the native equivalent for engine-level work, and a native
// sketch host is an explicit follow-up.
describe('Zoom Scroller (warp.legacy.zoom_scroller) E2E (engine path)', () => {
  jest.setTimeout(60000);

  it('zooms a structured input (different scales differ)', async () => {
    // source.grid → zoom_scroller. Pin the zoom to two very different levels via
    // min==max; the grid magnified 5× must differ from 1× (gizmo off so only the
    // transform is measured).
    const buildChain = (lo: number, hi: number): Sketch => ({
      anchor: null,
      wires: [],
      chain: [
        { type: 'module', module_type: 'source.grid', instance_key: 'grid@0', params: {} },
        { type: 'module', module_type: 'warp.legacy.zoom_scroller', instance_key: 'zs@0',
          params: { show_gizmo: 0, min_scale: lo, max_scale: hi, scale_variance: 0.0,
                    flicker_rate: 15.0 } },
      ],
    });
    const run = (id: string, lo: number, hi: number, dump: string) => runEngineTest({
      width: 128, height: 128,
      modules: ['com.nano.core', 'com.nano.legacy'],
      commands: [
        { type: 'createSketch', sketchId: id, sketch: buildChain(lo, hi) },
        { type: 'setTracePoints', tracePoints: [
          { id: 'out', target: { type: 'sketch_output', sketchId: id } },
        ]},
      ],
      waitFrames: 16,
      captureTraceIds: ['out'],
      dumpName: dump,
    });

    const z1 = await run('zs_scale1', 1.0, 1.0, 'zoom_scale1');
    const z5 = await run('zs_scale5', 5.0, 5.0, 'zoom_scale5');
    expect(z1.success).toBe(true);
    expect(z5.success).toBe(true);

    // The grid must have rendered (so "differ" is meaningful).
    let lit = 0;
    z1.trace('out').forEachPixel((c) => { if (c.r + c.g + c.b > 24) lit++; });
    expect(lit).toBeGreaterThan(100);

    // 5× zoom magnifies the grid cells → the frames differ.
    z5.trace('out').expectDifferentFrom(z1.trace('out'), 100);
  });

  it('filter_mode changes the zoom reconstruction (bicubic ≠ linear)', async () => {
    // Smooth (bicubic) softens the magnified pixel grid vs plain Linear, so on a
    // strongly-zoomed high-frequency grid the two reconstructions differ.
    const buildChain = (fm: number): Sketch => ({
      anchor: null,
      wires: [],
      chain: [
        { type: 'module', module_type: 'source.grid', instance_key: 'grid@0', params: {} },
        { type: 'module', module_type: 'warp.legacy.zoom_scroller', instance_key: 'zs@0',
          params: { show_gizmo: 0, min_scale: 6, max_scale: 6, scale_variance: 0.0,
                    flicker_rate: 15.0, filter_mode: fm } },
      ],
    });
    const run = (id: string, fm: number, dump: string) => runEngineTest({
      width: 256, height: 256,
      modules: ['com.nano.core', 'com.nano.legacy'],
      commands: [
        { type: 'createSketch', sketchId: id, sketch: buildChain(fm) },
        { type: 'setTracePoints', tracePoints: [
          { id: 'out', target: { type: 'sketch_output', sketchId: id } },
        ]},
      ],
      waitFrames: 16,
      captureTraceIds: ['out'],
      dumpName: dump,
    });
    const linear  = await run('zs_fm_lin', 1, 'zoom_fm_linear');
    const bicubic = await run('zs_fm_bic', 2, 'zoom_fm_bicubic');
    expect(linear.success).toBe(true);
    expect(bicubic.success).toBe(true);
    bicubic.trace('out').expectDifferentFrom(linear.trace('out'), 50);
  });

  it('pans over time (the camera moves)', async () => {
    // Same sketch, two different times: the quantized pan advances, so the
    // magnified grid is framed differently → the frames differ.
    const build = (): Sketch => ({
      anchor: null,
      wires: [],
      chain: [
        { type: 'module', module_type: 'source.grid', instance_key: 'grid@0', params: {} },
        { type: 'module', module_type: 'warp.legacy.zoom_scroller', instance_key: 'zs@0',
          params: { show_gizmo: 0, min_scale: 4.0, max_scale: 4.0, scale_variance: 0.0,
                    flicker_rate: 15.0, sub_delay: 0.0, sequence_delay: 0.0 } },
      ],
    });
    const run = (id: string, frames: number, dump: string) => runEngineTest({
      width: 128, height: 128,
      modules: ['com.nano.core', 'com.nano.legacy'],
      commands: [
        { type: 'createSketch', sketchId: id, sketch: build() },
        { type: 'setTracePoints', tracePoints: [
          { id: 'out', target: { type: 'sketch_output', sketchId: id } },
        ]},
      ],
      waitFrames: frames,
      captureTraceIds: ['out'],
      dumpName: dump,
    });

    const early = await run('zs_t_early', 6, 'zoom_t_early');
    const late  = await run('zs_t_late', 40, 'zoom_t_late');
    expect(early.success).toBe(true);
    expect(late.success).toBe(true);
    late.trace('out').expectDifferentFrom(early.trace('out'), 80);
  });
});
