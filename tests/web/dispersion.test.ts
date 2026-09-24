import { runGpuTest, runGpuEffectTest, forEachBackend } from '@nano/web/test/gpu-test-helpers';

// Per-effect tests for `warp.dispersion` against the `lights` bundle.
//
// Dispersion samples the input at (block_center + random_offset). With
// a uniform input texture, every sample returns the same color → output
// must equal input regardless of offset / block size. We use that
// invariant heavily; the visible "crunchy blur" only shows up against
// non-uniform input, which we don't easily produce in the per-effect
// runner (it ships a solid-color input).

forEachBackend((backend) => {
describe(`Dispersion Effect E2E (${backend})`, () => {
  jest.setTimeout(30000);

  it('declares metadata and I/O', async () => {
    const frame = await runGpuTest({
      module: 'warp.dispersion',
      bundle: 'lights',
      dumpName: 'dispersion_metadata',
    });
    expect(frame.success).toBe(true);
    expect(frame.metadata?.id).toBe('warp.dispersion');
  });

  it('intensity=0 passes input through unchanged', async () => {
    const frame = await runGpuEffectTest({
      module: 'warp.dispersion',
      bundle: 'lights',
      inputColor: [0.4, 0.5, 0.6, 1.0],
      params: [['intensity', 0.0]],
      dumpName: 'dispersion_passthrough',
    });
    expect(frame.success).toBe(true);
    frame.expectUniformColor({ r: 102, g: 128, b: 153, a: 255 }, 5);
  });

  it('uniform input → uniform output regardless of offset_max', async () => {
    // Sampling a uniform texture at any offset returns the same color,
    // so the output must equal the input exactly (modulo lerp(intensity)
    // which is also a no-op on identical values).
    const frame = await runGpuEffectTest({
      module: 'warp.dispersion',
      bundle: 'lights',
      width: 64, height: 64,
      inputColor: [0.7, 0.3, 0.1, 1.0],
      params: [
        ['intensity', 1.0],
        ['offset_max', 0.4],            // big offset
        ['vertical_block_norm', 0.5],
        ['horizontal_block_norm', 0.5],
      ],
      dumpName: 'dispersion_uniform_invariant',
    });
    expect(frame.success).toBe(true);
    frame.expectUniformColor({ r: 179, g: 76, b: 26, a: 255 }, 5);
  });

  it('offset_max=0 with full intensity also passes through (no jitter)', async () => {
    const frame = await runGpuEffectTest({
      module: 'warp.dispersion',
      bundle: 'lights',
      inputColor: [0.5, 0.5, 0.5, 1.0],
      params: [
        ['intensity', 1.0],
        ['offset_max', 0.0],
      ],
      dumpName: 'dispersion_zero_offset',
    });
    expect(frame.success).toBe(true);
    // With offset_max=0 the sample lands at block_center, which for a
    // uniform input is the same color.
    frame.expectUniformColor({ r: 128, g: 128, b: 128 }, 5);
  });

  it('output is uniform across the frame with uniform input', async () => {
    const frame = await runGpuEffectTest({
      module: 'warp.dispersion',
      bundle: 'lights',
      width: 96, height: 96,
      inputColor: [0.2, 0.4, 0.8, 1.0],
      params: [
        ['intensity', 1.0],
        ['offset_max', 0.2],
        ['vertical_block_norm', 0.3],
        ['horizontal_block_norm', 0.3],
      ],
      samplePoints: [[0, 0], [95, 0], [0, 95], [95, 95], [48, 48]],
      dumpName: 'dispersion_uniform_grid',
    });
    expect(frame.success).toBe(true);
    frame.expectUniformColor({ r: 51, g: 102, b: 204, a: 255 }, 5);
  });
});
});
