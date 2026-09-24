/*
 * mod.shaper.spectral — Spectral Curve modulation remapper.
 *
 * A unary modulation shaper that is to mod.source.spectral_lfo what mod.shaper.envelope is to
 * a hand-drawn curve: it builds the SAME spectrally-morphed LFO curve from a
 * manifold position (morph_x, morph_y, metric, interpolation) — sharing the atlas
 * + morph code via spectral_curve.h — but instead of sweeping it over TIME with a
 * phase accumulator, it indexes the curve by the `input` modulation value. The
 * morphed envelope becomes an arbitrary remapping curve.
 *
 * Pure data module — no GPU, no texture I/O (like the other mod.* shapers). The
 * curve is recomputed only when the morph inputs change (cached on State).
 */

#include <host.h>
#include <val.h>
#include "../spectral_lfo/spectral_curve.h"

#include <cmath>

namespace mod_spectral {

static inline float clampf(float v, float lo, float hi) {
  return v < lo ? lo : (v > hi ? hi : v);
}

struct State {
  float input = 0.0f;
  float morph_x = 0.5f, morph_y = 0.5f;
  int   metric = 0;
  bool  interpolation = true;
  float amplitude = 1.0f;
  bool  initialized = false;
  spectral_lfo::CurveCache curve;
};

void module_init() {
  state::init("mod.shaper.spectral", {1, 0, 1},
    state::Schema()
      // Top-level manual: high-level "what is this / how to use / what to try".
      .helpField("intro",
        "## Spectral Curve\n"
        "A **modulation shaper**: it bends an incoming modulation signal through a "
        "spectrally-morphed curve. Instead of sweeping the curve over time (like the "
        "Spectral LFO), it uses your **Input** value to *look up* the curve — so the "
        "curve becomes an arbitrary remapping of the signal.\n\n"
        "**Try:** wire an LFO or envelope into *Input*, then drag the **Morph** pad to "
        "reshape the response; switch the **Metric** for a wholly different family of "
        "curves; drop **Amplitude** to soften the effect toward a flat passthrough.")
      // --- Signal ---
      .group("signal", "Signal")
      // The signal to remap (wire target). The `magnitude` decl marks this as
      // THE modulation INPUT channel (so the shaper auto-connect locates it).
      .floatField("input", 0.0f, 0.f, 1.f, state::PrimaryInput, "unsigned").label("Input", "In")
      // --- Curve shape (spectral morph manifold) ---
      .group("shape", "Curve Shape")
        .groupHelp(
          "The remapping curve is a snapshot of the **Spectral LFO** morph atlas. "
          "*Morph X/Y* pick and blend which shape you land on; *Metric* selects which "
          "spectral feature the atlas is built from (each is a different curve family). "
          "Turn **Interpolate** off to snap to a single crisp shape instead of blending.")
      // Manifold position — picks/blends the LFO shape that becomes the curve.
      .floatField("morph_x", 0.5f, 0.f, 1.f, state::PrimaryInput).label("Morph X", "MrphX")
      .floatField("morph_y", 0.5f, 0.f, 1.f, state::PrimaryInput).label("Morph Y", "MrphY")
      .selectField("metric", 0, state::PrimaryInput,
                   {{"FFT Magnitude", 0}, {"Phase Coherence", 1}, {"Roughness", 2},
                    {"Spectral vs TD", 3}, {"Combined", 4}}, /*wrap=*/true).label("Metric", "Metric")
      .boolField("interpolation", true, state::PrimaryInput).label("Interpolate", "Interp")   // off = snap to one shape
      // --- Amount ---
      .group("amount", "Amount")
      .floatField("amplitude", 1.0f, 0.f, 1.f, state::SecondaryInput).label("Amplitude", "Amp")  // scales around 0.5
      // Remapped value. Rectifies into [0,1] (the curve's y window), so unsigned —
      // same convention as mod.shaper.remap / mod.shaper.envelope.
      .floatField("output", 0.0f, 0.f, 1.f, state::PrimaryOutput, "unsigned")
      // A unary modulation shaper: 1 modulation value in -> 1 remapped value out.
      .capability(state::Capability::ModulationShaper)
      .capability(state::Capability::ModulationShaperUnary)
      .capability(state::Capability::TimeIndependent)
  );
}

void* create() { return new State(); }

void destroy(void* self) {
  auto* s = static_cast<State*>(self);
  if (!s) return;
  delete s;
}

void init(void* self) {
  auto* s = static_cast<State*>(self);
  if (!s) return;
  *s = State{};
  s->initialized = true;
}

void tick(void* self, double dt) {
  auto* s = static_cast<State*>(self);
  if (!s || !s->initialized) return;
  (void)dt;
  // Rebuild the morphed curve only when the manifold inputs change.
  spectral_lfo::ensureCurve(s->curve, s->metric, s->morph_x, s->morph_y, s->interpolation);
  // Index the curve by the input value (not by time) → remapped output.
  const float v = spectral_lfo::sampleCurveAt(s->curve.curve, s->input);
  const float out = clampf((v - 0.5f) * s->amplitude + 0.5f, 0.0f, 1.0f);
  auto vh = val::number(out);
  state::setValPath("output", vh);
  val::release(vh);
}


void on_state_patched(void* self, int n, const char* pb, const int* off,
                      const int* len, const int* ops) {
  auto* s = static_cast<State*>(self);
  if (!s) return;
  for (int i = 0; i < n; i++) {
    if (ops[i] != state::PatchReplace) continue;
    const char* p = pb + off[i];
    int l = len[i];
    if      (state::pathIs(p, l, "input"))         s->input = state::patchFloat(i);
    else if (state::pathIs(p, l, "morph_x"))       s->morph_x = state::patchFloat(i);
    else if (state::pathIs(p, l, "morph_y"))       s->morph_y = state::patchFloat(i);
    else if (state::pathIs(p, l, "metric"))        s->metric = (int)state::patchFloat(i);
    else if (state::pathIs(p, l, "interpolation")) s->interpolation = state::patchFloat(i) != 0.0f;
    else if (state::pathIs(p, l, "amplitude"))     s->amplitude = state::patchFloat(i);
  }
  if (s->metric < 0) s->metric = 0;
  if (s->metric >= spectral_lfo::SPECTRAL_NUM_METRICS)
    s->metric = spectral_lfo::SPECTRAL_NUM_METRICS - 1;
}

void render(void* self, int vp_w, int vp_h) {
  (void)self; (void)vp_w; (void)vp_h;  // pure data module
}

} // namespace mod_spectral
