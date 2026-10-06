/*
 * source.light.drum_bars — three drum roles lighting the four LED bars.
 *
 * ch1 is the kick: a mirror-symmetric pattern (outer pair, inner pair, or all
 * four) at full. ch2 is the backbeat: the converse of whatever the kick would
 * light, then weaker flams rippling out to the outer bars. ch3 is decoration:
 * one random bar, a staccato strobe burst. Overlaps combine XOR-style. Which
 * pattern a hit fires cycles smoothly with the Cycle knob (and Drift) — noise
 * over a loop, cut crisp. All of the behaviour lives in
 * <sketch/drum_bars_core.h>, host-free so the goldens can drive it at an exact
 * dt; this file is the schema, the patch plumbing, the groove's clock and one
 * small pass.
 *
 * One output, the house rig's pixel map — four bars, ten segments, flat blocks
 * — drawn by the shared nano_led_bars.hlsl pass: each bar one solid level of
 * the colour. It is a control signal, so it is opaque and ignores the input.
 */

#include <gpu.h>
#include <preview_scenario.h>
#include <host.h>
#include <val.h>
#include <effect_beat_tick.h>
#include <sketch/drum_bars_core.h>
#include <sketch/led_bars.h>
#include "drum_bars_shaders.h"

#include <cmath>
#include <cstdint>

namespace drum_bars {

// Mirrors `cbuffer LedUniforms` in shaders/nano_led_bars.hlsl.
struct LedUniforms {
  float cell[led_bars::kCells][4];   // bar-major, segment 0 at the bottom
  float ctl[4];                      // quantize, -, -, -
};
static_assert(sizeof(LedUniforms) == 16 * (led_bars::kCells + 1),
              "LedUniforms layout mismatch with nano_led_bars.hlsl");

enum AutoMode { AutoOff = 0, AutoGroove = 1 };

struct State {
  drum_bars_core::Params p;
  drum_bars_core::Core core;
  float color[3] = {1.0f, 0.55f, 0.18f};
  float intensity = 1.0f;
  bool quantize = true;

  int auto_mode = AutoOff;
  fx::BeatTick beat;
  float ch_prev[drum_bars_core::kChannels] = {};

  bool initialized = false;
  gpu::Buffer led_buf;
};

static gpu::ComputePSO s_pso;

static void publish(const char* name, float value) {
  auto vh = val::number(value);
  state::setValPath(name, vh);
  val::release(vh);
}

static void publishRails(const State& s) {
  static const char* kBarOut[led_bars::kBars] = {"bar1_out", "bar2_out", "bar3_out", "bar4_out"};
  for (int b = 0; b < led_bars::kBars; ++b) publish(kBarOut[b], s.core.level[b]);
  publish("level_out", s.core.mean);
  publish("cycle_out", s.core.cycle_out);
}

void module_init() {
  state::init("source.light.drum_bars", {1, 0, 0},
    state::Schema()
      .helpField("intro",
        "## Drum Bars\n"
        "Three drum roles on the four LED bars, each bar one solid level of "
        "the colour.\n\n"
        "**Ch1, the kick**, lights a mirror pair — the outer bars, the inner "
        "bars — or all four, at full. **Ch2, the backbeat**, lights the "
        "converse of what the kick would: inner for outer, outer for inner, "
        "and all four again but dimmer; then it flams, weaker repeats "
        "rippling out to the outer bars. **Ch3, decoration**, strobes one "
        "random bar in a short staccato burst.\n\n"
        "Where they land on the same bar they cancel, XOR-style: a strobe "
        "over a lit kick flickers it dark.\n\n"
        "**Cycle** picks the patterns. Each position always gives the same "
        "ones, nearby positions mostly agree, and 0 and 1 are the same place, "
        "so a slow ramp on it walks the show through its variations and "
        "loops. **Drift** walks it for you.\n\n"
        "**Try:** *Auto* Groove, *Drift* 0.02 — kick on 1 and 3, backbeat on "
        "2 and 4, decoration scattered, slowly changing shape.")

      // ---------------- Triggers ----------------
      .group("triggers", "Triggers")
        .groupHelp(
          "Momentary: each fires on a rising edge. Wire a kick, a snare and a "
          "hat (or anything) in.")
      .eventField("ch1", state::PrimaryInput).label("Ch1 Kick", "Ch1")
      .eventField("ch2", state::PrimaryInput).label("Ch2 Back", "Ch2")
      .eventField("ch3", state::PrimaryInput).label("Ch3 Deco", "Ch3")

      // ---------------- Cycle ----------------
      .group("cycle", "Cycle")
        .groupHelp(
          "Where in the pattern loop the hits come from. **Variety** is how "
          "many changes one loop holds; **Seed** is a different loop.")
      .floatField("cycle", 0.0f, 0.f, 1.f, state::PrimaryInput,
                  nullptr, 0.f, nullptr,
                  "Position in the pattern loop. Wraps: 1 is 0 again.").wireCombine("replace")
        .label("Cycle", "Cyc")
      .floatField("drift", 0.0f, -1.f, 1.f, state::PrimaryInput,
                  nullptr, 0.f, nullptr,
                  "Loops per second added on top of Cycle. 0 holds.")
        .label("Drift", "Drift")
      .floatField("variety", 4.0f, 1.f, 12.f, state::PrimaryInput,
                  nullptr, 0.f, nullptr,
                  "How many times the picks change across one loop, roughly.")
        .label("Variety", "Var")
      .intField("seed", 0, 0, 0x7FFFFFFF, state::PrimaryInput).label("Seed", "Seed")

      // ---------------- Kick ----------------
      .group("kick", "Kick")
      .floatField("kick_decay", 0.35f, 0.02f, 2.f, state::PrimaryInput,
                  nullptr, 0.f, "s", "Seconds a kick takes to die.")
        .label("Kick Decay", "KDec")

      // ---------------- Backbeat ----------------
      .group("backbeat", "Backbeat")
        .groupHelp(
          "The backbeat lights what the kick would not. **Under** is how "
          "bright it is when the kick would take all four. **Flam** repeats "
          "it, weaker each time (**Flam Fade**), **Flam Gap** apart, on the "
          "outer bars — how many varies with Cycle, up to three.")
      .floatField("back_level", 0.85f, 0.f, 1.f, state::PrimaryInput,
                  nullptr, 0.f, nullptr, "The backbeat's level.")
        .label("Level", "BLvl")
      .floatField("under", 0.5f, 0.f, 1.f, state::PrimaryInput,
                  nullptr, 0.f, nullptr,
                  "Its level, as a share, when it lights all four.")
        .label("Under", "Under")
      .floatField("back_decay", 0.2f, 0.02f, 2.f, state::PrimaryInput,
                  nullptr, 0.f, "s", "Seconds a backbeat takes to die.")
        .label("Back Decay", "BDec")
      .floatField("flam", 0.6f, 0.f, 1.f, state::PrimaryInput,
                  nullptr, 0.f, nullptr, "How many repeats: 0 none, 1 up to three.")
        .label("Flam", "Flam")
      .floatField("flam_gap", 0.05f, 0.005f, 0.5f, state::PrimaryInput,
                  nullptr, 0.f, "s", "Seconds between repeats.")
        .label("Flam Gap", "Gap")
      .floatField("flam_fade", 0.5f, 0.f, 1.f, state::PrimaryInput,
                  nullptr, 0.f, nullptr, "Each repeat this much of the last.")
        .label("Flam Fade", "Fade")

      // ---------------- Deco ----------------
      .group("deco", "Deco")
        .groupHelp(
          "A burst of 2–6 hard pulses on one bar, each at its own level "
          "between **Min** and full. **Staccato** is how much of each pulse "
          "is lit; **Rate** speeds the whole burst up.")
      .floatField("deco_min", 0.3f, 0.f, 1.f, state::PrimaryInput,
                  nullptr, 0.f, nullptr, "The quietest a pulse gets.")
        .label("Min", "Min")
      .floatField("staccato", 0.35f, 0.05f, 1.f, state::PrimaryInput,
                  nullptr, 0.f, nullptr, "Lit share of each pulse.")
        .label("Staccato", "Stac")
      .floatField("deco_rate", 1.0f, 0.25f, 4.f, state::PrimaryInput,
                  nullptr, 0.f, nullptr, "Strobe speed.")
        .label("Rate", "Rate")

      // ---------------- Mix ----------------
      .group("mix", "Mix")
      .selectField("combine", drum_bars_core::CombineXor, state::PrimaryInput,
                   {{"XOR", drum_bars_core::CombineXor},
                    {"Max", drum_bars_core::CombineMax},
                    {"Add", drum_bars_core::CombineAdd}}, false,
                   "How hits on the same bar meet. XOR cancels: two at full "
                   "is dark.")
        .label("Combine", "Comb")

      // ---------------- Auto ----------------
      .group("auto", "Auto")
        .groupHelp(
          "**Groove** plays the three on the transport's 16ths: kick on 1 and "
          "3 (with a pickup at some places in the loop), backbeat on 2 and 4, "
          "decoration at random at **Deco Chance**. Wired triggers still fire "
          "on top.")
      .selectField("auto", AutoOff, state::PrimaryInput,
                   {{"Off", AutoOff}, {"Groove", AutoGroove}}, false,
                   "Self-play a groove on the transport.")
        .label("Auto", "Auto")
      .floatField("auto_deco", 0.2f, 0.f, 1.f, state::PrimaryInput,
                  nullptr, 0.f, nullptr, "Groove: chance of a decoration on each 16th.")
        .label("Deco Chance", "DChn")

      // ---------------- Light ----------------
      .group("light", "Light")
      .rgbField("color", 1.0f, 0.55f, 0.18f, state::PrimaryInput)
        .label("Colour", "Col")
      .floatField("intensity", 1.0f, 0.f, 2.f, state::PrimaryInput,
                  nullptr, 0.f, nullptr, "Overall output level.")
        .label("Intensity", "Int")

      // ---------------- Output ----------------
      .group("output", "Output")
      .boolField("quantize", true, state::SecondaryInput,
                 "Each segment a flat block, so a pixel mapper lands on its "
                 "exact colour. Off interpolates between segment centres.")
        .label("Quantize", "Quant")
      .floatField("bar1_out", 0.f, 0.f, 1.f, state::PrimaryOutput, "unsigned",
                  0.f, nullptr, "Bar 1's level (left, outer).").label("Bar 1", "B1")
      .floatField("bar2_out", 0.f, 0.f, 1.f, state::PrimaryOutput, "unsigned",
                  0.f, nullptr, "Bar 2's level (left, inner).").label("Bar 2", "B2")
      .floatField("bar3_out", 0.f, 0.f, 1.f, state::PrimaryOutput, "unsigned",
                  0.f, nullptr, "Bar 3's level (right, inner).").label("Bar 3", "B3")
      .floatField("bar4_out", 0.f, 0.f, 1.f, state::PrimaryOutput, "unsigned",
                  0.f, nullptr, "Bar 4's level (right, outer).").label("Bar 4", "B4")
      .floatField("level_out", 0.f, 0.f, 1.f, state::PrimaryOutput, "unsigned",
                  0.f, nullptr, "Mean level across the bars.")
        .label("Level", "Lvl")
      .floatField("cycle_out", 0.f, 0.f, 1.f, state::PrimaryOutput, "unsigned",
                  0.f, nullptr, "The Cycle position in effect, Drift included.").wireCombine("replace")
        .label("Cycle Out", "CycO")
      .textureField("tex_in",  state::PrimaryInput)
      .textureField("tex_out", state::PrimaryOutput)

      .capability(state::Capability::Generator)
      .capability(state::Capability::ModulationSource)
      .capability(state::Capability::ModulationSourceMulti)
  );

  if (gpu::Device::backend() == gpu::Backend::None) return;

  state::registerShaderSPV("drum_bars_led", LED_SPV, LED_SPV_SIZE);
  if (auto led = gpu::Device::createShaderModuleByName("drum_bars_led")) {
    s_pso = gpu::Device::createComputePSO(led, "main", gpu::Bindings()
        .storageTex2d(0)
        .uniform(1));
  }
  state::log("drum_bars: module initialized");
}

void* create() {
  auto* s = new State();
  s->led_buf = gpu::Device::createBuffer(sizeof(LedUniforms), gpu::BufferUsage::Uniform);
  return s;
}

void destroy(void* self) {
  auto* s = static_cast<State*>(self);
  if (!s) return;
  s->led_buf.release();
  delete s;
}

void init(void* self) {
  auto* s = static_cast<State*>(self);
  if (!s) return;
  s->core.reset();
  s->beat.reset();
  for (float& v : s->ch_prev) v = 0.0f;
  s->initialized = false;
  if (!s_pso.valid() || !s->led_buf.valid()) return;
  s->initialized = true;
}

// The core advances here and ONLY here: render() may run without a tick
// (thumbnails, off-playhead previews), and stepping from there would play
// hits every time somebody looked at the card.
void tick(void* self, double dt) {
  auto* s = static_cast<State*>(self);
  if (!s) return;
  if (s->auto_mode == AutoGroove) {
    // BeatTick counts 16ths from a bar line, so floor(beats) mod 16 is the
    // step in the bar; a stall that crosses several plays each in turn.
    const int n = s->beat.tick(16.0f);
    const long last = (long)std::floor(s->beat.effectiveBeats());
    for (int i = 0; i < n; ++i) s->core.groove(s->p, (int)((last - (n - 1 - i)) % 16));
  }
  s->core.tick(s->p, (float)dt);
  publishRails(*s);
}

void on_state_patched(void* self, int n, const char* pb, const int* off,
                      const int* len, const int* ops) {
  auto* s = static_cast<State*>(self);
  if (!s) return;
  auto& p = s->p;
  static const char* kCh[drum_bars_core::kChannels] = {"ch1", "ch2", "ch3"};
  for (int i = 0; i < n; i++) {
    if (ops[i] != state::PatchReplace) continue;
    const char* path = pb + off[i];
    const int l = len[i];
    bool handled = false;
    // Rising edge only: every value is replayed every frame.
    for (int ch = 0; ch < drum_bars_core::kChannels; ++ch) {
      if (!state::pathIs(path, l, kCh[ch])) continue;
      const float v = state::patchFloat(i);
      if (v != 0.0f && s->ch_prev[ch] == 0.0f) s->core.fire(ch);
      s->ch_prev[ch] = v;
      handled = true;
    }
    if (handled) continue;
    if (state::pathIs(path, l, "cycle"))            p.cycle = state::patchFloat(i);
    else if (state::pathIs(path, l, "drift"))       p.drift = state::patchFloat(i);
    else if (state::pathIs(path, l, "variety"))     p.variety = state::patchFloat(i);
    else if (state::pathIs(path, l, "seed"))        p.seed = state::patchInt(i);
    else if (state::pathIs(path, l, "kick_decay"))  p.kick_decay = state::patchFloat(i);
    else if (state::pathIs(path, l, "back_level"))  p.back_level = state::patchFloat(i);
    else if (state::pathIs(path, l, "under"))       p.under = state::patchFloat(i);
    else if (state::pathIs(path, l, "back_decay"))  p.back_decay = state::patchFloat(i);
    else if (state::pathIs(path, l, "flam"))        p.flam = state::patchFloat(i);
    else if (state::pathIs(path, l, "flam_gap"))    p.flam_gap = state::patchFloat(i);
    else if (state::pathIs(path, l, "flam_fade"))   p.flam_fade = state::patchFloat(i);
    else if (state::pathIs(path, l, "deco_min"))    p.deco_min = state::patchFloat(i);
    else if (state::pathIs(path, l, "staccato"))    p.staccato = state::patchFloat(i);
    else if (state::pathIs(path, l, "deco_rate"))   p.deco_rate = state::patchFloat(i);
    else if (state::pathIs(path, l, "combine"))     p.combine = state::patchInt(i);
    else if (state::pathIs(path, l, "auto_deco"))   p.auto_deco = state::patchFloat(i);
    else if (state::pathIs(path, l, "auto")) {
      const int m = state::patchInt(i);
      // Re-arm so switching into Groove seeds from where the bar is now.
      if (m == AutoGroove && s->auto_mode != AutoGroove) s->beat.reset();
      s->auto_mode = m;
    }
    else if (state::pathIs(path, l, "color")) {
      auto v = state::patchVec3(i);
      s->color[0] = v.x; s->color[1] = v.y; s->color[2] = v.z;
    }
    else if (state::pathIs(path, l, "intensity"))   s->intensity = state::patchFloat(i);
    else if (state::pathIs(path, l, "quantize"))    s->quantize = state::patchBool(i);
  }
}

void render(void* self, int vp_w, int vp_h) {
  auto* s = static_cast<State*>(self);
  if (!s || !s->initialized || vp_w <= 0 || vp_h <= 0) return;
  auto out = gpu::Device::textureForField("tex_out");
  if (!out.valid()) return;

  // A ZERO-dt tick: ages nothing, only re-reads the knobs.
  s->core.tick(s->p, 0.0f);
  publishRails(*s);

  const auto cells = drum_bars_core::toCells(s->core, s->color, s->intensity);
  LedUniforms u = {};
  for (int i = 0; i < led_bars::kCells; i++) {
    u.cell[i][0] = cells.rgb[i][0];
    u.cell[i][1] = cells.rgb[i][1];
    u.cell[i][2] = cells.rgb[i][2];
    u.cell[i][3] = 1.0f;
  }
  u.ctl[0] = s->quantize ? 1.0f : 0.0f;
  s->led_buf.writeOne(u);
  auto cp = gpu::ComputePass::begin();
  cp.setPSO(s_pso);
  cp.setTexture(out, 0, 1);
  cp.setBuffer(s->led_buf, 1);
  cp.dispatch((vp_w + 7) / 8, (vp_h + 7) / 8);
  cp.end();

  gpu::Device::submit();
}

}  // namespace drum_bars

namespace drum_bars {

// Effect store preview: the auto groove playing the four bars.
void preview(nano::PreviewScenario& s) {
  s.input("black")
      .param("auto", 1.0f)
      .capture(1.6f);
}

}  // namespace drum_bars
