/*
 * mod.source.bass_sim — Synthetic Resolume-FFT low-band modulation source.
 *
 * Simulates the signal Arena's FFT bass band actually produces (modeled on
 * real captures): a level riding an elevated floor, kicks stepping a peak
 * hold up by a modest gain on beat-grid slots, a LINEAR fall (Resolume's
 * "Fall" smoothing), a one-pole rise smear for the analyzer's attack lag,
 * and an optional 8th-note wobble for the between-kick bass groove. Locked
 * to the host beat grid, so downstream beat-aware modules (the transient
 * shaper!) see exactly what they'd see live — no audio routing needed.
 *
 * The signal model is the shared sketch/fft_bass_sim.h — the SAME generator
 * that drives the transient shaper's Catch2 goldens, so this effect's output
 * is pinned by those tests.
 *
 * Class-like instance model: module_init() publishes the schema once per
 * type; each chain entry gets its own State via create(). Params arrive as
 * state patches; the sim advances in tick() and republishes `output` every
 * tick.
 *
 * Pure data module — no GPU, no texture I/O.
 */

#include <host.h>
#include <val.h>
#include <cmath>

#include "sketch/fft_bass_sim.h"

namespace mod_bass_sim {

// Pattern presets, 16-bit slot masks (bit k = kick on 16th slot k).
enum Pattern : int {
  PatFourFloor = 0x1111,
  Pat8ths = 0x5555,
  Pat16ths = 0xFFFF,
  PatOffbeat = 0x4444,
  PatOncePerBar = 0x0001,
  PatSilence = 0x0000,
};

// Per-instance state. One per chain entry.
struct State {
  fft_bass_sim::Params params;
  fft_bass_sim::Sim sim;
};

// Type-level setup: schema. Runs once per type. No GPU work.
void module_init() {
  state::init("mod.source.bass_sim", {1, 0, 0},
    state::Schema()
      .helpField("intro",
        "## Bass Sim\n"
        "A stand-in for Resolume's FFT **bass band** — the same shape real "
        "Arena captures have, locked to the host beat grid, with no audio "
        "routing: the level rides a sustained *Floor* (a rolling bassline "
        "never lets it drop to zero), each kick in the *Pattern* steps it up "
        "by *Kick*, the level falls back **linearly** at *Fall* (Resolume's "
        "smoothing), *Smooth* smears the rise like the analyzer's lag, and "
        "*Wobble* adds the little 8th-note groove wiggle real captures "
        "show.\n\n"
        "**Try:** feed it into *mod.shaper.transient_shaper* and watch the "
        "confidence build over a few bars; flip the *Pattern* to Silence to "
        "fake a breakdown, or crank *Smooth* to torture-test onset "
        "detection.")
      // --- Pattern: what plays ---
      .group("pattern", "Pattern")
        .groupHelp(
          "*Pattern* picks which 16th-note slots of the bar carry a kick "
          "(4/4, host clock). Silence keeps just the floor + wobble — a "
          "breakdown you can switch to live.")
      .selectField("pattern", PatFourFloor, state::PrimaryInput,
                   {{"4 Floor", PatFourFloor}, {"8ths", Pat8ths},
                    {"16ths", Pat16ths}, {"Offbeat", PatOffbeat},
                    {"Once/Bar", PatOncePerBar}, {"Silence", PatSilence}},
                   /*wrap=*/true).label("Pattern", "Pat")
      // --- Signal: the analyzer model ---
      .group("signal", "Signal")
        .groupHelp(
          "*Floor* is the sustained bass level between kicks; *Kick* is how "
          "far a kick steps above it. *Fall* is the linear release rate "
          "(Resolume's \"Fall\" — slower = smoother, laggier). *Smooth* "
          "smears the rise (the analyzer's attack lag — the thing a "
          "transient shaper has to fight). *Wobble* is the between-kick "
          "bass-groove wiggle, useful for false-positive testing.")
      .floatField("floor", 0.40f, 0.f, 0.8f, state::PrimaryInput).label("Floor", "Flr")
      .floatField("kick", 0.22f, 0.f, 0.6f, state::PrimaryInput).label("Kick", "Kick")
      .floatField("fall", 0.8f, 0.1f, 4.f, state::PrimaryInput,
                  nullptr, 0.f, "/s").label("Fall", "Fall")
      .floatField("smooth", 0.02f, 0.f, 0.15f, state::PrimaryInput,
                  nullptr, 0.f, "s").label("Smooth", "Smth")
      .floatField("wobble", 0.04f, 0.f, 0.2f, state::PrimaryInput).label("Wobble", "Wob")
      // --- Output ---
      .group("output", "Output")
      .floatField("output", 0.0f, 0.f, 1.f, state::PrimaryOutput, "unsigned").label("Output", "Out")
      // A single-channel modulation source. Stateful (peak hold, smear), so
      // seeks are approximate — the sim advances without firing phantom
      // kicks across a jump and settles within a fall time.
      .capability(state::Capability::ModulationSource)
      .capability(state::Capability::ModulationSourceSingle)
      .capability(state::Capability::SeekableApproximate)
  );
}

void* create() {
  return new State();
}

void destroy(void* self) {
  auto* s = static_cast<State*>(self);
  if (!s) return;
  delete s;
}

void init(void* self) {
  auto* s = static_cast<State*>(self);
  if (!s) return;
  s->params = fft_bass_sim::Params{};
  s->params.wobble = 0.04f;   // schema default (Params{} defaults to 0)
  s->sim.reset();
}

void tick(void* self, double dt) {
  auto* s = static_cast<State*>(self);
  if (!s) return;
  const double phase = host::barPhase();
  const double u = (phase - std::floor(phase)) * 16.0;
  const float level = s->sim.step(u, dt, s->params);
  auto oh = val::number(level);
  state::setValPath("output", oh);
  val::release(oh);
}

void on_state_patched(void* self, int n, const char* pb, const int* off,
                      const int* len, const int* ops) {
  auto* s = static_cast<State*>(self);
  if (!s) return;
  for (int i = 0; i < n; i++) {
    if (ops[i] != state::PatchReplace) continue;
    const char* p = pb + off[i];
    int l = len[i];
    if      (state::pathIs(p, l, "pattern")) s->params.pattern = (unsigned)state::patchInt(i);
    else if (state::pathIs(p, l, "floor"))   s->params.base = state::patchFloat(i);
    else if (state::pathIs(p, l, "kick"))    s->params.kick_gain = state::patchFloat(i);
    else if (state::pathIs(p, l, "fall"))    s->params.fall = state::patchFloat(i);
    else if (state::pathIs(p, l, "smooth"))  s->params.rise_tau = state::patchFloat(i);
    else if (state::pathIs(p, l, "wobble"))  s->params.wobble = state::patchFloat(i);
  }
}

void render(void* self, int vp_w, int vp_h) {
  (void)self; (void)vp_w; (void)vp_h;
  // No rendering — pure data module.
}

} // namespace mod_bass_sim
