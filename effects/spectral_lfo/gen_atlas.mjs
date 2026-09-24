#!/usr/bin/env node
/*
 * gen_atlas.mjs — bake the nano-lfo spectral-morph atlas into a C++ header.
 *
 * Run manually (like shape_fold/gen_atlas.py); the generated
 * spectral_lfo_atlas.h is committed. Reuses the nano-lfo web project's own
 * `delaunator` dependency so the triangulation is byte-faithful to the app.
 *
 * Usage:
 *   node gen_atlas.mjs [path-to-nano-lfo/web]
 * Default source dir: /Users/nattos/Code/nano-lfo/web
 *
 * Source data (in <web>/assets):
 *   lfo_tsne.json            -> shared control points + metric 0 (FFT Magnitude) tsne
 *   lfo_tsne_phase_coh.json  -> metric 1 tsne
 *   lfo_tsne_roughness.json  -> metric 2 tsne
 *   lfo_tsne_spec_td.json    -> metric 3 tsne
 *   lfo_tsne_combined.json   -> metric 4 tsne
 *
 * The LFO shapes (control points) are identical across all 5 files; only the
 * t-SNE embedding differs. So control points are baked once.
 */

import { readFileSync, writeFileSync, mkdirSync } from 'node:fs';
import { fileURLToPath } from 'node:url';
import { dirname, join } from 'node:path';
import { createRequire } from 'node:module';

const __dirname = dirname(fileURLToPath(import.meta.url));
const WEB_DIR = process.argv[2] || '/Users/nattos/Code/nano-lfo/web';
const ASSETS = join(WEB_DIR, 'assets');
const OUT = join(__dirname, 'spectral_lfo_atlas.h');

// Pull Delaunator from the web project's node_modules (faithful to the app).
const require = createRequire(join(WEB_DIR, 'package.json'));
const Delaunator = require('delaunator').default ?? require('delaunator');

// ─── Fixed-point decode (mirror of lfo_spectral_morph.ts) ──────────────
const FP_SCALE = 1 << 30;
function decodeFixedPointArray(b64) {
  const buf = Buffer.from(b64, 'base64');
  const ints = new Int32Array(buf.buffer, buf.byteOffset, buf.byteLength >> 2);
  const out = new Float64Array(ints.length);
  for (let i = 0; i < ints.length; i++) out[i] = ints[i] / FP_SCALE;
  return out;
}

// Metric enum order must match the selectField in main.cpp.
const METRICS = [
  { key: 0, file: 'lfo_tsne.json' },           // FFT Magnitude (baseline)
  { key: 1, file: 'lfo_tsne_phase_coh.json' }, // Phase Coherence
  { key: 2, file: 'lfo_tsne_roughness.json' }, // Roughness
  { key: 3, file: 'lfo_tsne_spec_td.json' },   // Spectral vs TD
  { key: 4, file: 'lfo_tsne_combined.json' },  // Combined
];

function loadFile(file) {
  return JSON.parse(readFileSync(join(ASSETS, file), 'utf8'));
}

// ─── Control points (shared across metrics) — from the baseline file ───
const base = loadFile(METRICS[0].file);
const N = base.entries.length;

const cpX = [];
const cpY = [];
const cpF = [];
const entryOffset = new Int32Array(N);
const entryNcp = new Int32Array(N);
for (let i = 0; i < N; i++) {
  const e = base.entries[i];
  const xs = decodeFixedPointArray(e.x);
  const ys = decodeFixedPointArray(e.y);
  const fs = decodeFixedPointArray(e.f);
  entryOffset[i] = cpX.length;
  entryNcp[i] = xs.length;
  for (let j = 0; j < xs.length; j++) {
    cpX.push(xs[j]); cpY.push(ys[j]); cpF.push(fs[j]);
  }
}

// ─── Per-metric: tsne coords + Delaunay triangulation ──────────────────
// Mirror of buildTriangulation() in lfo_explorer_app.ts: append 4 virtual
// corner points (mapped to their nearest real point) so the mesh covers the
// whole [0,1] scatter, triangulate, and keep coords + corner->data mapping.
const CORNERS = [[-0.5, -0.5], [1.5, -0.5], [1.5, 1.5], [-0.5, 1.5]];

function buildMetric(file) {
  const d = loadFile(file);
  const tsne = decodeFixedPointArray(d.tsne); // flat [x0,y0,x1,y1,...], length N*2
  const total = N + CORNERS.length;
  const coords = new Float64Array(total * 2);
  const mapping = new Int32Array(total);
  for (let i = 0; i < N; i++) {
    coords[i * 2] = tsne[i * 2];
    coords[i * 2 + 1] = tsne[i * 2 + 1];
    mapping[i] = i;
  }
  for (let c = 0; c < CORNERS.length; c++) {
    const idx = N + c;
    const [cx, cy] = CORNERS[c];
    coords[idx * 2] = cx;
    coords[idx * 2 + 1] = cy;
    let bestDist = Infinity, bestIdx = 0;
    for (let i = 0; i < N; i++) {
      const dx = tsne[i * 2] - cx;
      const dy = tsne[i * 2 + 1] - cy;
      const dd = dx * dx + dy * dy;
      if (dd < bestDist) { bestDist = dd; bestIdx = i; }
    }
    mapping[idx] = bestIdx;
  }
  const del = new Delaunator(coords);
  return { coords, mapping, triangles: del.triangles };
}

const metrics = METRICS.map(m => ({ key: m.key, ...buildMetric(m.file) }));

// ─── Emit the header ───────────────────────────────────────────────────
function fnum(v) {
  // Compact float literal with enough precision for curve fidelity.
  let s = Number(v).toPrecision(8);
  if (s.indexOf('.') >= 0 && s.indexOf('e') < 0 && s.indexOf('E') < 0) {
    s = s.replace(/0+$/, '').replace(/\.$/, '.0');
  }
  return s + 'f';
}

function emitFloatArray(name, arr) {
  const parts = [`static const float ${name}[${arr.length}] = {`];
  let line = '  ';
  for (let i = 0; i < arr.length; i++) {
    const tok = fnum(arr[i]) + ',';
    if (line.length + tok.length > 110) { parts.push(line); line = '  '; }
    line += tok;
  }
  if (line.trim().length) parts.push(line);
  parts.push('};');
  return parts.join('\n');
}

function emitIntArray(name, arr) {
  const parts = [`static const int ${name}[${arr.length}] = {`];
  let line = '  ';
  for (let i = 0; i < arr.length; i++) {
    const tok = arr[i] + ',';
    if (line.length + tok.length > 110) { parts.push(line); line = '  '; }
    line += tok;
  }
  if (line.trim().length) parts.push(line);
  parts.push('};');
  return parts.join('\n');
}

const out = [];
out.push('// AUTO-GENERATED by gen_atlas.mjs from the nano-lfo spectral-morph atlas. Do not edit.');
out.push('// Baked LFO-shape atlas: shared control points (Serum-style points) +');
out.push('// per-metric t-SNE embedding and Delaunay triangulation. CPU-only data.');
out.push('#ifndef SPECTRAL_LFO_ATLAS_H');
out.push('#define SPECTRAL_LFO_ATLAS_H');
out.push('');
out.push('namespace spectral_lfo {');
out.push('');
out.push(`static const int SL_NUM_ENTRIES = ${N};`);
out.push(`static const int SL_NUM_METRICS = ${metrics.length};`);
out.push(`static const int SL_TOTAL_CP    = ${cpX.length};`);
out.push('');
out.push('// ── Shared control points (flat; per-entry offset/count) ──');
out.push(emitFloatArray('SL_CP_X', cpX));
out.push(emitFloatArray('SL_CP_Y', cpY));
out.push(emitFloatArray('SL_CP_F', cpF));
out.push(emitIntArray('SL_ENTRY_OFFSET', entryOffset));
out.push(emitIntArray('SL_ENTRY_NCP', entryNcp));
out.push('');

const totalPts = N + CORNERS.length;
for (const m of metrics) {
  out.push(`// ── Metric ${m.key} ──`);
  out.push(emitFloatArray(`SL_COORDS_${m.key}`, m.coords));
  out.push(emitIntArray(`SL_TRITODATA_${m.key}`, m.mapping));
  out.push(emitIntArray(`SL_TRIS_${m.key}`, m.triangles));
  out.push(`static const int SL_NTRIS_${m.key} = ${m.triangles.length / 3};`);
  out.push('');
}

// Lookup tables indexed by metric enum.
out.push(`static const int SL_TOTAL_PTS = ${totalPts};`);
out.push(`static const float* const SL_COORDS[${metrics.length}] = { ${metrics.map(m => `SL_COORDS_${m.key}`).join(', ')} };`);
out.push(`static const int* const SL_TRITODATA[${metrics.length}] = { ${metrics.map(m => `SL_TRITODATA_${m.key}`).join(', ')} };`);
out.push(`static const int* const SL_TRIS[${metrics.length}] = { ${metrics.map(m => `SL_TRIS_${m.key}`).join(', ')} };`);
out.push(`static const int SL_NTRIS[${metrics.length}] = { ${metrics.map(m => `SL_NTRIS_${m.key}`).join(', ')} };`);
out.push('');
out.push('} // namespace spectral_lfo');
out.push('');
out.push('#endif // SPECTRAL_LFO_ATLAS_H');

writeFileSync(OUT, out.join('\n'));
console.error(`Wrote ${OUT}`);
console.error(`  entries=${N} total_cp=${cpX.length} metrics=${metrics.length}`);
for (const m of metrics) console.error(`  metric ${m.key}: ntris=${m.triangles.length / 3}`);

// ─── Editor backdrop: compact quantized t-SNE scatter (web asset) ───────
// The custom inspector draws the manifold as a faint scatter behind the XY
// pad. Ship the per-metric points (normalized [0,1], quantized to 1 byte/axis)
// as a tiny base64 TS module the editor imports directly (no fetch).
const SCATTER_OUT = join(__dirname, '../../../web/src/editors/spectral-lfo-scatter-data.ts');
const scatter = new Uint8Array(metrics.length * N * 2);
let so = 0;
for (const m of metrics) {
  for (let i = 0; i < N; i++) {
    const qx = Math.round(Math.min(1, Math.max(0, m.coords[i * 2])) * 255);
    const qy = Math.round(Math.min(1, Math.max(0, m.coords[i * 2 + 1])) * 255);
    scatter[so++] = qx;
    scatter[so++] = qy;
  }
}
const scatterB64 = Buffer.from(scatter.buffer).toString('base64');
const scatterTs = [
  '// AUTO-GENERATED by gen_atlas.mjs from the nano-lfo t-SNE embeddings. Do not edit.',
  '// Per-metric scatter points (normalized [0,1], quantized to 1 byte/axis) for the',
  '// spectral-lfo inspector XY-pad backdrop. Layout: metric-major, then point, [x,y].',
  `export const SCATTER_NUM_METRICS = ${metrics.length};`,
  `export const SCATTER_NUM_POINTS = ${N};`,
  `export const SCATTER_B64 = '${scatterB64}';`,
  '',
  '/** Decode to a Uint8Array of length NUM_METRICS*NUM_POINTS*2 (xy interleaved). */',
  'export function decodeScatter(): Uint8Array {',
  '  const bin = atob(SCATTER_B64);',
  '  const out = new Uint8Array(bin.length);',
  '  for (let i = 0; i < bin.length; i++) out[i] = bin.charCodeAt(i);',
  '  return out;',
  '}',
  '',
].join('\n');
writeFileSync(SCATTER_OUT, scatterTs);
console.error(`Wrote ${SCATTER_OUT} (${scatterB64.length} b64 chars)`);

// ─── Editor data asset: control points + triangulation (fetched binary) ──
// The inspector draws the Delaunay mesh + active triangle on the pad and a
// morphed-envelope preview below it — all client-side, like the web prototype.
// That needs the control points (to evaluate source shapes) and the per-metric
// triangulation. Ship it as one binary fetched lazily on first inspector mount
// (kept out of the main JS bundle). Little-endian; the loader copies slices so
// no alignment constraints. Layout:
//   i32  [MAGIC, numMetrics, numEntries, totalCP]
//   u32  entryOffset[numEntries]
//   u16  entryNcp[numEntries]
//   u16  cpX[totalCP]   (x * 65535)
//   u8   cpY[totalCP]   (y * 255)
//   u8   cpF[totalCP]   (f * 255)
//   per metric: u32 [numPts, numTris]; f32 coords[numPts*2];
//               u16 triToData[numPts]; u16 tris[numTris*3]
const BIN_OUT = join(__dirname, '../../../web/public/data/spectral_lfo_editor.bin');
mkdirSync(dirname(BIN_OUT), { recursive: true });

const u8 = (v) => Math.max(0, Math.min(255, Math.round(v * 255)));
const u16 = (v) => Math.max(0, Math.min(65535, Math.round(v * 65535)));

const sections = [];
const pushTyped = (ta) => sections.push(Buffer.from(ta.buffer, ta.byteOffset, ta.byteLength));

pushTyped(Int32Array.from([0x31464C53, metrics.length, N, cpX.length])); // 'SLF1'
pushTyped(Uint32Array.from(entryOffset));
pushTyped(Uint16Array.from(entryNcp));
pushTyped(Uint16Array.from(cpX, u16));
pushTyped(Uint8Array.from(cpY, u8));
pushTyped(Uint8Array.from(cpF, u8));
for (const m of metrics) {
  const numPts = m.coords.length / 2;
  const numTris = m.triangles.length / 3;
  pushTyped(Uint32Array.from([numPts, numTris]));
  pushTyped(Float32Array.from(m.coords));
  pushTyped(Uint16Array.from(m.mapping));
  pushTyped(Uint16Array.from(m.triangles));
}
const bin = Buffer.concat(sections);
writeFileSync(BIN_OUT, bin);
console.error(`Wrote ${BIN_OUT} (${bin.length} bytes)`);
