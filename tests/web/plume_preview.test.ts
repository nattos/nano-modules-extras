import { runEngineTest } from '@nano/web/test/engine-test-helpers';
import type { Sketch } from '@nano/web/src/sketch-types';

// Throwaway high-res preview renders for visual tuning (not CI coverage).

function sketch(params: Record<string, unknown>): Sketch {
  return { anchor: null, chain: [
    { type: 'module', module_type: 'source.solid_color', instance_key: 'bg@0',
      params: { color: [0.02, 0.02, 0.025] } },
    { type: 'module', module_type: 'source.sdf.plume',
      instance_key: 'plume@0', params },
  ], wires: [] } as Sketch;
}

describe('plume preview', () => {
  jest.setTimeout(120000);

  it('renders reference-look candidates', async () => {
    const base = {
      orbit: 0.0, morph: 0.0, tilt: 0.05, zoom: 0.3,
      radius: 0.6, ridge_depth: 0.7, ridge_scale: 0.65, ridge_sharp: 0.7,
      ridge_aniso: 0.8, swirl: 0.1,
      albedo: [0.85, 0.85, 0.86], sun: 0.7, shadow: 0.8, ao: 0.8,
      ambient: 0.55, azimuth: -35, elevation: 35,
    };
    for (const [name, params] of [
      ['plume_ref_a', base],
      ['plume_ref_b', { ...base, ridge_scale: 0.85, ridge_depth: 0.55, ridge_aniso: 1.0 }],
      ['plume_ref_c', { ...base, ridge_sharp: 1.0, ridge_aniso: 0.4, swirl: 0.4 }],
      ['plume_gi_off_hi', { ...base, ridge_sharp: 1.0, ridge_aniso: 0.4, swirl: 0.4,
                            bounce: 0.0, fog: 0.0 }],
      ['plume_gi_on_hi', { ...base, ridge_sharp: 1.0, ridge_aniso: 0.4, swirl: 0.4,
                           bounce: 1.0, gi_decay: 0.75, sun: 0.85, fog: 0.0 }],
      ['plume_fog_hi', { ...base, ridge_sharp: 1.0, ridge_aniso: 0.4, swirl: 0.4,
                         bounce: 0.8, fog: 0.6, fog_soft: 0.6, room_fog: 0.15 }],
      ['plume_backlit_hi', { ...base, ridge_sharp: 1.0, ridge_aniso: 0.4, swirl: 0.4,
                             bounce: 0.6, fog: 0.7, fog_soft: 0.7, phase: 0.9,
                             azimuth: 170, elevation: 10, sun: 1.0 }],
      ['plume_lucent_hi', { ...base, ridge_sharp: 1.0, ridge_aniso: 0.4, swirl: 0.4,
                            bounce: 0.5, fog: 0.25, fog_soft: 0.4, phase: 0.7,
                            azimuth: 155, elevation: 15, sun: 1.0,
                            transmission: 0.9, thickness: 0.7,
                            reflect: 0.5, roughness: 0.25 }],
    ] as const) {
      const r = await runEngineTest({
        width: 512, height: 512,
        modules: ['com.nano.core', 'com.nano.nano'],
        commands: [
          { type: 'createSketch', sketchId: name, sketch: sketch(params as any) },
          { type: 'setTracePoints', tracePoints: [
            { id: 'out', target: { type: 'sketch_output', sketchId: name } },
          ]},
        ],
        waitFrames: 24,
        captureTraceIds: ['out'],
        dumpName: name,
      });
      expect(r.success).toBe(true);
    }
  });
});
