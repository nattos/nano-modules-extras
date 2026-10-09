/*
 * source.die_fold — Die Fold: an infinite fake chip, descended layer by layer.
 *
 * A port of the die-fold prototype (nano-fx-prototypes/die-fold). A die is
 * CONSTRUCTED implicitly, never stored: each pixel walks a nine-level guillotine
 * floorplan on a global track grid, reads the fabric of the block it lands in
 * (arrays, rows, logic, datapaths, combs, macros, motifs), the corridor it is
 * nearest to (wires, jogs, vias) and the power straps, and does that once per
 * visible layer while the camera falls through the stack. The plane is infinite
 * in both axes and zoomable without bound. Two registers of one geometry: an EDA
 * layout and an SEM micrograph, blended by Look.
 *
 * The CPU's share per frame: a bilinear sample of a 15x15 PARAMETER atlas
 * (X = grain, Y = netness; die_fold_atlas.h), the lever offsets, the camera
 * (an int super-cell plus a fraction, so a million super-cells out is as sharp
 * as the origin), and — when flying — a closed trajectory through the atlas with
 * the camera on the same curve plus an optional Kepler orbit (paths.h). The GPU
 * runs ONE compute pass (present.hlsl) over the packed parameter buffer.
 *
 * One seed for the whole atlas: a blend of two hash salts is a third unrelated
 * die, so the seed SNAPS while every folded parameter interpolates. That is what
 * lets a trajectory cross the atlas without popping.
 *
 * Generator: only tex_out, no texture input.
 */

#include <gpu.h>
#include <preview_scenario.h>
#include <host.h>
#include <val.h>
#include "die_fold_shaders.h"
#include "die_fold_atlas.h"
#include "paths.h"

#include <cmath>
#include <cstdint>
#include <algorithm>

namespace die_fold {

// --- the levers: bipolar offsets on folded parameters, 0 = the fold's value ---
struct Lever { const char* field; int slot; };
static const Lever kLevers[] = {
  {"look",         S_LOOK},
  {"cell_size",    S_CELL_LOG2},
  {"occupancy",    S_OCCUPANCY},
  {"connectivity", S_CONT_P},
  {"motifs",       S_W_MOTIF},
  {"variety",      S_VARIETY},
  {"stagger",      S_STAGGER},
  {"erode",        S_ERODE},
  {"haze",         S_HAZE},
  {"depth_cue",    S_DEPTH_CUE},
  {"defocus",      S_DEFOCUS},
  {"saturation",   S_SAT},
};
static constexpr int kNumLevers = (int)(sizeof(kLevers) / sizeof(kLevers[0]));
static constexpr int kParamBase = DF_PRM_SIZE - DF_NPARAM;   // first parameter slot

// The camera is an int super-cell plus a fraction. |cell| stays far below 2^24
// (exact as a float in the buffer) and cell * 128 stays inside int32 in the
// shader's hash keys.
static constexpr double kCellLimit = 1e7;
static constexpr double kZoomLo = -1.0, kZoomHi = 6.0;

struct State {
  gpu::Buffer prm_buf;
  bool initialized = false;

  // --- Schema-mirrored params ---
  float grain = 0.5f, netness = 0.5f;
  int   seed = DF_ATLAS_SEED;
  float speed = 0.12f;           // descents (8-layer loops) per second
  float depth = 0.0f;            // offsets the descent phase
  float zoom = DF_PIN_ZOOM;
  float pan_x = 0.0f, pan_y = 0.0f;
  bool  fly = false;
  int   path = 0;
  float slowness = 48.0f;        // descents per traverse
  int   laps = 1;                // traverses per `slowness` descents
  float cam_span = 6.0f, zoom_lo = 0.6f, zoom_hi = 2.4f;
  bool  orbit = false;
  float orb_span = 2.5f, orb_e = 0.6f, orb_zoom = 0.35f;
  int   orb_k = 1;
  float lever[kNumLevers] = {};
  float hue = 0.0f;              // wrapped hue lever

  // --- clocks ---
  double t = 0.0;                // descent phase accumulator, [0,1)
  double p = 0.0;                // path phase accumulator, [0,1)

  // --- the path, resolved when its knobs change ---
  Spec spec;
  double fscale = 1.0;
  bool spec_dirty = true;

  // The orbit with no path: the anchor is where the camera is parked, so the
  // orbit's offset at the moment it engages is subtracted (no jump).
  bool  orbit_was = false;
  double orb_ref_x = 0, orb_ref_y = 0, orb_ref_z = 0;

  float eff_x = 0.5f, eff_y = 0.5f;   // the XY the atlas is sampled at
};

static void apply_visibility(bool fly, bool orbit) {
  state::setFieldHidden("path",     !fly);
  state::setFieldHidden("cam_span", !fly);
  state::setFieldHidden("zoom_lo",  !fly);
  state::setFieldHidden("zoom_hi",  !fly);
  // The path owns the zoom while it flies.
  state::setFieldHidden("zoom",     fly);
  // The traverse clock paces the orbit too.
  state::setFieldHidden("slowness", !(fly || orbit));
  state::setFieldHidden("laps",     !(fly || orbit));
  state::setFieldHidden("orb_span", !orbit);
  state::setFieldHidden("orb_e",    !orbit);
  state::setFieldHidden("orb_k",    !orbit);
  state::setFieldHidden("orb_zoom", !orbit);
}

void eval_visibility(int n, const char* pb, const int* off, const int* len, const int* ops) {
  bool fly = false, orbit = false;
  for (int i = 0; i < n; i++) {
    if (ops[i] != state::PatchReplace) continue;
    const char* p = pb + off[i]; int l = len[i];
    if      (state::pathIs(p, l, "fly"))   fly   = state::patchFloat(i) != 0.0f;
    else if (state::pathIs(p, l, "orbit")) orbit = state::patchFloat(i) != 0.0f;
  }
  apply_visibility(fly, orbit);
}

static void on_state_ready(void* self);

static gpu::ComputePSO s_pso_present;

void module_init() {
  state::init("source.die_fold", {1, 0, 0},
    state::Schema()
      .helpField("intro",
        "## Die Fold\n"
        "An endless fake **chip**: floorplans of blocks and cells, routing with wires "
        "and vias, power straps over everything — and the camera falls through it "
        "**layer by layer**, each layer eroding away to uncover the one below. Drag the "
        "XY pad to explore the fold: **X** is *grain* (more, smaller cells per block), "
        "**Y** is *netness* (from cell fields toward dense metal networks).\n\n"
        "**Try:** turn on **Fly** to let a trajectory carry the die through the fold "
        "while the camera flies the same curve; sweep **Look** from a false-color layout "
        "to an electron-microscope grayscale; re-roll the **Seed** for a different die "
        "with the same character.")
      // --- The fold: the XY pad + the one seed ---
      .group("die", "Die")
        .groupHelp(
          "The fold: a 15x15 control plane of whole dies, blended continuously. *Grain* "
          "(X) moves from a few large blocks to fine confetti; *Netness* (Y) from cell "
          "fields with incidental wiring to dense metal webs. *Seed* re-rolls the die "
          "without changing its character — it snaps rather than blends.")
      .floatField("grain", 0.5f, 0.0f, 1.0f, state::PrimaryInput).label("Grain", "Grain")
      .floatField("netness", 0.5f, 0.0f, 1.0f, state::PrimaryInput).label("Netness", "Net")
      .intField("seed", DF_ATLAS_SEED, 0, 999, state::PrimaryInput).label("Seed", "Seed")
      // --- The descent ---
      .group("descent", "Descent")
        .groupHelp(
          "The fall through the stack. *Speed* is how many 8-layer descents per second "
          "(the loop closes exactly); *Depth* offsets where in the stack you are — at "
          "Speed 0 it scrubs the descent by hand. *Zoom* reveals another level of the "
          "hierarchy rather than enlarging the last; *Pan* moves over the endless die.")
      .floatField("speed", 0.12f, 0.0f, 0.6f, state::PrimaryInput,
                  nullptr, /*step=*/0.005f, /*units=*/"/s",
                  "Descents per second; one descent falls through all 8 layers and "
                  "returns exactly where it started.").label("Speed", "Spd")
      .floatField("depth", 0.0f, 0.0f, 1.0f, state::PrimaryInput,
                  nullptr, /*step=*/0.001f, /*units=*/nullptr,
                  "Offsets the descent phase: where in the 8-layer stack the camera is.")
                  .wireCombine("replace").label("Depth", "Dpth")
      .floatField("zoom", DF_PIN_ZOOM, (float)kZoomLo, (float)kZoomHi, state::PrimaryInput,
                  nullptr, /*step=*/0.01f, /*units=*/"oct").label("Zoom", "Zoom")
      .floatField("pan_x", 0.0f, -64.0f, 64.0f, state::PrimaryInput,
                  nullptr, /*step=*/0.01f, /*units=*/"cells").label("Pan X", "PanX")
      .floatField("pan_y", 0.0f, -64.0f, 64.0f, state::PrimaryInput,
                  nullptr, /*step=*/0.01f, /*units=*/"cells").label("Pan Y", "PanY")
      // --- Fly: a closed trajectory through the fold ---
      .group("fly", "Fly")
        .groupHelp(
          "**Fly** carries the pad along a closed curve through the fold, and the camera "
          "flies the same curve over the die — one move, not two. *Path* picks one of six "
          "discovered journeys. *Slowness* is how many descents one traverse takes "
          "(the trajectory's only speed control), *Laps* how many traverses per that "
          "span. *Span* sets how far the camera travels; *Zoom Low/High* the zoom it "
          "swings between. Everything closes, so the whole composition loops.")
      .boolField("fly", false, state::PrimaryInput).label("Fly", "Fly")
      .selectField("path", 0, state::PrimaryInput,
                   {{"Lissajous", 0}, {"Rose", 1}, {"Trochoid", 2},
                    {"Knot", 3}, {"Harmonograph", 4}, {"Fourier", 5}},
                   /*wrap=*/true).label("Path", "Path")
      .floatField("slowness", 48.0f, 1.0f, 480.0f, state::PrimaryInput,
                  nullptr, /*step=*/1.0f, /*units=*/nullptr,
                  "Descents per traverse of the path (and per orbit cycle).")
                  .logScale().label("Slowness", "Slow")
      .intField("laps", 1, 1, 6, state::PrimaryInput).label("Laps", "Laps")
      .floatField("cam_span", 6.0f, 0.0f, 24.0f, state::PrimaryInput,
                  nullptr, /*step=*/0.05f, /*units=*/"cells").label("Span", "Span")
      .floatField("zoom_lo", 0.6f, -0.5f, 3.0f, state::PrimaryInput,
                  nullptr, /*step=*/0.01f, /*units=*/"oct").label("Zoom Low", "ZLo")
      .floatField("zoom_hi", 2.4f, 0.0f, 5.0f, state::PrimaryInput,
                  nullptr, /*step=*/0.01f, /*units=*/"oct").label("Zoom High", "ZHi")
      // --- Orbit: an elastic loop around the anchor ---
      .group("orbit", "Orbit")
        .groupHelp(
          "Loops the camera around an anchor — where the path puts it, or where you "
          "parked it. The anchor sits at the orbit's **focus**, so *Eccentricity* is the "
          "elasticity: the camera lolls far out, falls, whips past and climbs away. "
          "*Orbits* per traverse keeps it closing with everything else; *Breathe* swings "
          "the zoom with the fall.")
      .boolField("orbit", false, state::PrimaryInput).label("Orbit", "Orb")
      .floatField("orb_span", 2.5f, 0.0f, 12.0f, state::PrimaryInput,
                  nullptr, /*step=*/0.05f, /*units=*/"cells").label("Orbit Span", "OSpan")
      .floatField("orb_e", 0.6f, 0.0f, 0.9f, state::PrimaryInput).label("Eccentricity", "Ecc")
      .intField("orb_k", 1, 1, 8, state::PrimaryInput).label("Orbits", "Orbs")
      .floatField("orb_zoom", 0.35f, 0.0f, 1.5f, state::PrimaryInput).label("Breathe", "Brth")
      // --- Levers: bipolar offsets on the folded parameters ---
      .group("levers", "Levers")
        .groupHelp(
          "Push the fold's own values up or down: 0 is exactly what the pad gives, ±1 "
          "spans the parameter's whole range. *Look* runs from a false-color layout to "
          "an SEM micrograph; *Hue* turns the palette around the wheel.")
      .floatField("look", 0.0f, -1.0f, 1.0f, state::PrimaryInput).label("Look", "Look")
      .floatField("hue", 0.0f, -1.0f, 1.0f, state::PrimaryInput).label("Hue", "Hue")
      .floatField("saturation", 0.0f, -1.0f, 1.0f, state::PrimaryInput).label("Saturation", "Sat")
      .floatField("cell_size", 0.0f, -1.0f, 1.0f, state::PrimaryInput).label("Cell Size", "Cell")
      .floatField("occupancy", 0.0f, -1.0f, 1.0f, state::PrimaryInput).label("Wires", "Wire")
      .floatField("connectivity", 0.0f, -1.0f, 1.0f, state::PrimaryInput).label("Connectivity", "Conn")
      .floatField("motifs", 0.0f, -1.0f, 1.0f, state::PrimaryInput).label("Motifs", "Motf")
      .floatField("variety", 0.0f, -1.0f, 1.0f, state::PrimaryInput).label("Layer Variety", "Vary")
      .floatField("stagger", 0.0f, -1.0f, 1.0f, state::PrimaryInput).label("Stagger", "Stag")
      .floatField("erode", 0.0f, -1.0f, 1.0f, state::PrimaryInput).label("Erode", "Erod")
      .floatField("defocus", 0.0f, -1.0f, 1.0f, state::PrimaryInput).label("Defocus", "Dfoc")
      .floatField("haze", 0.0f, -1.0f, 1.0f, state::PrimaryInput).label("Haze", "Haze")
      .floatField("depth_cue", 0.0f, -1.0f, 1.0f, state::PrimaryInput).label("Depth Cue", "Cue")
      // --- Broadcasts: the live pad position and the descent phase ---
      .floatField("pad_x", 0.5f, 0.0f, 1.0f, state::SecondaryOutput).wireCombine("replace")
      .floatField("pad_y", 0.5f, 0.0f, 1.0f, state::SecondaryOutput).wireCombine("replace")
      .floatField("phase", 0.0f, 0.0f, 1.0f, state::SecondaryOutput).wireCombine("replace")
      // --- I/O: pure generator ---
      .textureField("tex_out", state::PrimaryOutput)
        .capability(state::Capability::Generator)
        .capability(state::Capability::SeekableApproximate)
    );

  if (gpu::Device::backend() == gpu::Backend::None) return;

  state::registerShaderSPV("die_fold_present", PRESENT_SPV, PRESENT_SPV_SIZE);
  auto cs_present = gpu::Device::createShaderModuleByName("die_fold_present");
  if (!cs_present) return;
  s_pso_present = gpu::Device::createComputePSO(cs_present, "main", gpu::Bindings()
      .uniform(0)          // the packed parameter block
      .storageTex2d(1));   // tex_out

  state::log("die_fold: module initialized");
}

void* create() {
  auto* s = new State();
  s->prm_buf = gpu::Device::createBuffer(DF_PRM_FLOATS * sizeof(float), gpu::BufferUsage::Uniform);
  return s;
}

void destroy(void* self) {
  auto* s = static_cast<State*>(self);
  if (!s) return;
  s->prm_buf.release();
  delete s;
}

void init(void* self) {
  auto* s = static_cast<State*>(self);
  if (!s) return;
  s->initialized = s_pso_present.valid() && s->prm_buf.valid();
  state::setOnStateReady(&on_state_ready);
}

static void on_state_ready(void* self) {
  auto* s = static_cast<State*>(self);
  if (!s) return;
  apply_visibility(s->fly, s->orbit);
}

static inline double clampd(double v, double lo, double hi) { return v < lo ? lo : (v > hi ? hi : v); }
static inline double wrap01(double v) { return v - std::floor(v); }

// The path's knobs: the preset, with the exposed camera and orbit knobs over it.
static void resolve_spec(State* s) {
  s->spec = preset_spec(s->path);
  s->spec.m[M_CAM_SPAN] = s->cam_span;
  s->spec.m[M_ZOOM_LO] = s->zoom_lo;
  s->spec.m[M_ZOOM_HI] = s->zoom_hi;
  s->spec.m[M_ORB_SPAN] = s->orbit ? s->orb_span : 0.0;
  s->spec.m[M_ORB_E] = s->orb_e;
  s->spec.m[M_ORB_K] = s->orb_k;
  s->spec.m[M_ORB_ZOOM] = s->orb_zoom;
  s->fscale = (s->spec.family == FAM_FOURIER) ? fourier_scale(s->spec) : 1.0;
  s->spec_dirty = false;
}

static inline void broadcast(const char* path, float v) {
  auto x = val::number(v);
  state::setValPath(path, x);
  val::release(x);
}

void tick(void* self, double dt) {
  auto* s = static_cast<State*>(self);
  if (!s) return;
  if (s->spec_dirty) resolve_spec(s);

  // TWO CLOCKS AT A RATIO OF INTEGERS. The descent runs at `speed`; the path at
  // speed * laps / slowness. Integrated separately (rather than as multiples of
  // one master phase), so turning a knob never jumps either; with the knobs
  // still they are exactly the prototype's frac(T * dives) / frac(T * laps), and
  // the whole composition closes.
  double dives = std::max(1.0, std::round((double)s->slowness));
  double laps = std::max(1, s->laps);
  s->t = wrap01(s->t + dt * s->speed);
  s->p = wrap01(s->p + dt * s->speed * laps / dives);

  if (s->fly) {
    double x, y;
    path_xy(s->spec, s->p, s->fscale, x, y);
    s->eff_x = (float)x;
    s->eff_y = (float)y;
  } else {
    s->eff_x = s->grain;
    s->eff_y = s->netness;
  }

  broadcast("pad_x", s->eff_x);
  broadcast("pad_y", s->eff_y);
  broadcast("phase", (float)wrap01(s->t + s->depth));
}

void on_state_patched(void* self, int n, const char* pb, const int* off,
                      const int* len, const int* ops) {
  auto* s = static_cast<State*>(self);
  if (!s) return;
  bool vis_changed = false;
  for (int i = 0; i < n; i++) {
    if (ops[i] != state::PatchReplace) continue;
    const char* path = pb + off[i];
    int plen = len[i];
    if      (state::pathIs(path, plen, "grain"))    s->grain = state::patchFloat(i);
    else if (state::pathIs(path, plen, "netness"))  s->netness = state::patchFloat(i);
    else if (state::pathIs(path, plen, "seed"))     s->seed = state::patchInt(i);
    else if (state::pathIs(path, plen, "speed"))    s->speed = state::patchFloat(i);
    else if (state::pathIs(path, plen, "depth"))    s->depth = state::patchFloat(i);
    else if (state::pathIs(path, plen, "zoom"))     s->zoom = state::patchFloat(i);
    else if (state::pathIs(path, plen, "pan_x"))    s->pan_x = state::patchFloat(i);
    else if (state::pathIs(path, plen, "pan_y"))    s->pan_y = state::patchFloat(i);
    else if (state::pathIs(path, plen, "fly"))      { bool v = state::patchFloat(i) != 0.0f; if (v != s->fly) { s->fly = v; vis_changed = true; } }
    else if (state::pathIs(path, plen, "path"))     { s->path = state::patchInt(i); s->spec_dirty = true; }
    else if (state::pathIs(path, plen, "slowness")) s->slowness = state::patchFloat(i);
    else if (state::pathIs(path, plen, "laps"))     s->laps = state::patchInt(i);
    else if (state::pathIs(path, plen, "cam_span")) { s->cam_span = state::patchFloat(i); s->spec_dirty = true; }
    else if (state::pathIs(path, plen, "zoom_lo"))  { s->zoom_lo = state::patchFloat(i); s->spec_dirty = true; }
    else if (state::pathIs(path, plen, "zoom_hi"))  { s->zoom_hi = state::patchFloat(i); s->spec_dirty = true; }
    else if (state::pathIs(path, plen, "orbit"))    { bool v = state::patchFloat(i) != 0.0f; if (v != s->orbit) { s->orbit = v; vis_changed = true; s->spec_dirty = true; } }
    else if (state::pathIs(path, plen, "orb_span")) { s->orb_span = state::patchFloat(i); s->spec_dirty = true; }
    else if (state::pathIs(path, plen, "orb_e"))    { s->orb_e = state::patchFloat(i); s->spec_dirty = true; }
    else if (state::pathIs(path, plen, "orb_k"))    { s->orb_k = state::patchInt(i); s->spec_dirty = true; }
    else if (state::pathIs(path, plen, "orb_zoom")) { s->orb_zoom = state::patchFloat(i); s->spec_dirty = true; }
    else if (state::pathIs(path, plen, "hue"))      s->hue = state::patchFloat(i);
    else {
      for (int k = 0; k < kNumLevers; k++) {
        if (state::pathIs(path, plen, kLevers[k].field)) { s->lever[k] = state::patchFloat(i); break; }
      }
    }
  }
  if (vis_changed) apply_visibility(s->fly, s->orbit);
}

// Where the camera is, in super-cells, and its zoom.
static void camera(State* s, double& wx, double& wy, double& zoom) {
  if (s->spec_dirty) resolve_spec(s);
  if (s->fly) {
    // The path owns the camera (the orbit rides inside it), around the pan.
    path_camera(s->spec, s->p, s->fscale, s->pan_x, s->pan_y, wx, wy, zoom);
    s->orbit_was = false;
    return;
  }
  wx = s->pan_x;
  wy = s->pan_y;
  zoom = s->zoom;
  if (s->orbit) {
    double dx, dy, dz;
    orbit_offset(s->spec, s->p, dx, dy, dz);
    // Engaging the orbit is not a jump: the parked view is taken to be the
    // current point of the ellipse, and the ellipse is drawn around it.
    if (!s->orbit_was) { s->orb_ref_x = dx; s->orb_ref_y = dy; s->orb_ref_z = dz; }
    wx += dx - s->orb_ref_x;
    wy += dy - s->orb_ref_y;
    zoom += dz - s->orb_ref_z;
  }
  s->orbit_was = s->orbit;
}

void render(void* self, int vp_w, int vp_h) {
  auto* s = static_cast<State*>(self);
  if (!s || !s->initialized || vp_w <= 0 || vp_h <= 0) return;
  auto out = gpu::Device::textureForField("tex_out");
  if (!out.valid()) return;

  float prm[DF_PRM_FLOATS] = {};
  sample_atlas(s->eff_x, s->eff_y, prm);

  // The levers: offsets in each parameter's own range, clamped to it.
  for (int k = 0; k < kNumLevers; k++) {
    if (s->lever[k] == 0.0f) continue;
    int slot = kLevers[k].slot;
    int pi = slot - kParamBase;
    float lo = DF_PARAM_LO[pi], hi = DF_PARAM_HI[pi];
    prm[slot] = (float)clampd(prm[slot] + s->lever[k] * (hi - lo), lo, hi);
  }
  // Hue wraps: ±1 turns the palette half the wheel either way.
  prm[S_HUE] = (float)wrap01(prm[S_HUE] + 0.5 * s->hue);

  double wx, wy, zoom;
  camera(s, wx, wy, zoom);
  wx = clampd(wx, -kCellLimit, kCellLimit);
  wy = clampd(wy, -kCellLimit, kCellLimit);
  double cx = std::floor(wx), cy = std::floor(wy);
  double fx = wx - cx, fy = wy - cy;
  if (fx >= 1.0) { cx += 1.0; fx -= 1.0; }
  if (fy >= 1.0) { cy += 1.0; fy -= 1.0; }

  prm[S_T] = (float)wrap01(s->t + s->depth);
  prm[S_SURF_W] = (float)vp_w;
  prm[S_SURF_H] = (float)vp_h;
  prm[S_PROBE] = 0.0f;
  prm[S_CAM_CELL_X] = (float)cx;
  prm[S_CAM_CELL_Y] = (float)cy;
  prm[S_CAM_FRAC_X] = (float)fx;
  prm[S_CAM_FRAC_Y] = (float)fy;
  prm[S_SEED] = (float)s->seed;
  prm[S_PX_SCALE] = host::pxScale(vp_h);
  prm[S_ZOOM] = (float)clampd(zoom, kZoomLo, kZoomHi);
  s->prm_buf.write(prm, DF_PRM_FLOATS);

  auto cp = gpu::ComputePass::begin();
  cp.setPSO(s_pso_present);
  cp.setBuffer(s->prm_buf, 0);
  cp.setTexture(out, 1, 1);
  cp.dispatch((vp_w + 7) / 8, (vp_h + 7) / 8);
  cp.end();
  gpu::Device::submit();
}

// Effect store preview: flying the default path.
void preview(nano::PreviewScenario& s) {
  s.input("none")
      .param("fly", 1.0f)
      .capture(1.0f);
}

}  // namespace die_fold
