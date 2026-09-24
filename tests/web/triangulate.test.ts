import { runEngineTest } from '@nano/web/test/engine-test-helpers';
import type { Sketch } from '@nano/web/src/sketch-types';

/**
 * E2E coverage for filter.mesh.triangulate (nano bundle) — the topology-
 * following GPU Delaunay triangulation. The full pipeline runs on WebGPU
 * (naga-translated): downsample → blur → feature (ridge/corner/density
 * importance field) → JFA Voronoi over a persistent seed pool → per-cell
 * atomic scoring → stochastic confidence-gated takeover → triple-point
 * Delaunay edge extraction → instanced-line mesh render.
 *
 * The seed pool is stateful (it settles over frames), so every capture uses
 * multiple waitFrames. A solid input gives a uniform importance field, which is
 * enough to exercise the whole compute→raster pipeline and the debug views.
 *
 * Under test:
 *  1. Registers + renders: with debug off + dark backdrop + bright edges the
 *     mesh rasterizes a non-black wireframe (the whole pipeline dispatches on
 *     WebGPU).
 *  2. The debug views are real, distinct internal stages: the Voronoi cell view
 *     differs from the Importance-field view.
 *  3. density changes the mesh (more seeds → a visibly different triangulation).
 *  4. scoring_mode is accepted and dispatches for every mode.
 */
function buildSketch(params: Record<string, unknown>): Sketch {
  return {
    anchor: null,
    chain: [
      {
        type: 'module',
        module_type: 'source.solid_color',
        instance_key: 'bg@0',
        params: { color: [0.35, 0.5, 0.65] },
      },
      {
        type: 'module',
        module_type: 'filter.mesh.triangulate',
        instance_key: 'tri@0',
        params,
      },
    ],
  };
}

async function render(sketchId: string, params: Record<string, unknown>, dumpName: string,
                      waitFrames = 8) {
  const result = await runEngineTest({
    width: 96, height: 96,
    modules: ['com.nano.testonly', 'com.nano.nano'],
    commands: [
      { type: 'createSketch', sketchId, sketch: buildSketch(params) },
      { type: 'setTracePoints', tracePoints: [
        { id: 'out', target: { type: 'sketch_output', sketchId } },
      ]},
    ],
    waitFrames,
    captureTraceIds: ['out'],
    dumpName,
  });
  expect(result.success).toBe(true);
  return result;
}

describe('filter.mesh.triangulate E2E', () => {
  jest.setTimeout(60000);

  it('registers and renders a non-black mesh', async () => {
    const result = await render('tri_smoke',
      { debug_view: 0, bg_mode: 1, density: 0.4, line_color: [1, 1, 1] }, 'tri_smoke');
    result.trace('out').expectNotSolidColor({ r: 0, g: 0, b: 0 }, 5);

    const tri = result.state.plugins.find((p: any) => p.id === 'filter.mesh.triangulate');
    expect(tri).toBeTruthy();
  });

  it('Voronoi and Importance debug views are distinct stages', async () => {
    const voronoi = await render('tri_voronoi', { debug_view: 5, density: 0.4 }, 'tri_voronoi');
    const importance = await render('tri_importance', { debug_view: 4, density: 0.4 }, 'tri_importance');
    voronoi.trace('out').expectDifferentFrom(importance.trace('out'), 20);
  });

  it('density changes the triangulation', async () => {
    const sparse = await render('tri_sparse', { debug_view: 5, density: 0.1 }, 'tri_sparse');
    const dense = await render('tri_dense', { debug_view: 5, density: 0.9 }, 'tri_dense');
    dense.trace('out').expectDifferentFrom(sparse.trace('out'), 20);
  });

  it('every scoring_mode dispatches and renders', async () => {
    for (const mode of [0, 1, 2]) {
      const r = await render(`tri_mode${mode}`,
        { debug_view: 0, bg_mode: 1, density: 0.4, scoring_mode: mode, line_color: [1, 1, 1] },
        `tri_mode${mode}`);
      r.trace('out').expectNotSolidColor({ r: 0, g: 0, b: 0 }, 5);
    }
  });
});
