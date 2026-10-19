/*
 * source.light.side_racks — the whole show for the six-bar rig on one card.
 *
 * Two racks of three horizontal LED bars stand either side of the stage, 60
 * segments a bar (see <sketch/rack_map.h>). This card is everything they do:
 * the house sweep with a Tilt that rolls the bars like a chord, a primary
 * mix-in on bouncy-abode's strong channel (Flam, Wave or Beating), pattern
 * flashes on the other three channels, and a built-in lift and strobe — with
 * one knob for whether the mix-ins land under the lift and strobe or over
 * them. All of the behaviour lives in <sketch/side_racks_core.h>, host-free
 * so the goldens can drive it at an exact dt; this file is the schema, the
 * patch plumbing, the transport's step clock and one small pass.
 *
 * One output, the rig's compact grid: the left half of the frame is the left
 * rack, the right half the right rack, three rows with the top bar on top, 60
 * flat blocks a row running left to right as the bars stand. It is a control
 * signal, so it is opaque and ignores the input.
 */

#include <gpu.h>
#include <preview_scenario.h>
#include <host.h>
#include <val.h>
#include <effect_beat_tick.h>
#include <sketch/side_racks_core.h>
#include <sketch/rack_map.h>
#include "side_racks_shaders.h"

#include <cstdint>

namespace side_racks {

using namespace side_racks_core;

// Mirrors `cbuffer RackUniforms` in led.hlsl.
struct RackUniforms {
  float cell[rack_map::kCells][4];   // grid order: row * 120 + column
  float ctl[4];                      // quantize, -, -, -
};
static_assert(sizeof(RackUniforms) == 16 * (rack_map::kCells + 1),
              "RackUniforms layout mismatch with led.hlsl");

struct State {
  Params p;
  Core core;
  bool quantize = true;
  fx::BeatTick step_tick;

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
  publish("speed_out", s.core.knob.speed);
  publish("release_out", s.core.knob.release);
  publish("level_out", clamp01(s.core.mean_level));
  publish("primary_env_out", clamp01(s.core.primary_env));
  publish("sweep_out", s.core.knob.sweep);
}

// --- Visibility -------------------------------------------------------------
//
// One rule, from values, for both the live path and the static one: each
// primary mode shows its own fields; the duck envelope belongs to Flam and
// Beating, the Wave plays out on its own clock.

static void applyVisibility(int primary_mode) {
  const bool flam = primary_mode == PrimaryFlam;
  const bool wave = primary_mode == PrimaryWave;
  const bool beat = primary_mode == PrimaryBeating;
  for (const char* f : {"env_attack", "env_decay", "env_sustain", "env_release"})
    state::setFieldHidden(f, wave);
  for (const char* f : {"flam_amount", "flam_rate", "flam_floor"})
    state::setFieldHidden(f, !flam);
  for (const char* f : {"wave_amount", "wave_speed", "wave_start", "wave_recover",
                        "wave_width", "wave_crush", "wave_sparkle", "wave_sparkle_rate",
                        "wave_color"})
    state::setFieldHidden(f, !wave);
  for (const char* f : {"beat_amount", "beat_rate", "beat_detune", "beat_spread_a",
                        "beat_spread_b"})
    state::setFieldHidden(f, !beat);
}

void eval_visibility(int n, const char* pb, const int* off, const int* len, const int* ops) {
  int mode = PrimaryFlam;
  for (int i = 0; i < n; i++) {
    if (ops[i] != state::PatchReplace) continue;
    if (state::pathIs(pb + off[i], len[i], "primary_mode")) mode = state::patchInt(i);
  }
  applyVisibility(mode);
}

static void on_state_ready(void* self) {
  auto* s = static_cast<State*>(self);
  if (!s) return;
  applyVisibility(s->p.primary_mode);
}

void module_init() {
  state::init("source.light.side_racks", {1, 0, 0},
    state::Schema()
      .helpField("intro",
        "## Side Racks\n"
        "The whole show for the six-bar rig — two racks of three 1 m LED bars "
        "either side of the stage, 60 segments a bar — on one card.\n\n"
        "**Sweep** is the house knob: full in the middle, choking to black at "
        "either end, bouncing past full when you come back in fast, and "
        "throwing a rolling tail when a hard pass lands in the choke. **Tilt** "
        "rolls the bars like a chord: while the knob moves, the light gathers "
        "on the bar the choke is climbing through; at rest it relaxes to a "
        "soft gradient.\n\n"
        "**Primary** is one gate — wire bouncy-abode's *gate strong*. Pick "
        "**Flam** (a sidechain duck that shudders), **Wave** (a pulse from "
        "the stage center that eats the light as it crosses the room, crushed "
        "and sparkling on its front) or **Beating** (two frame-locked rates "
        "interfering up the racks). **Secondary** is three more gates — "
        "*low*, *mid*, *high* — flashing a bar each on one side, both sides "
        "when all three hit together.\n\n"
        "**Lift** and **Strobe** are built in, in their own accent color. "
        "**Mix Post** decides whether the mix-ins land under them or cut "
        "through them.\n\n"
        "**Rig:** the output is a compact grid — left rack on the left half, "
        "right rack on the right half, top bar on top, three rows of 60 "
        "segments running left to right as the bars stand. Lay each bar's "
        "fixture over its row: x 0–0.5 (left) or 0.5–1 (right), centered at "
        "y 1/6, 1/2, 5/6 from the top bar down.\n\n"
        "**Try:** *Tilt* 1, *Rest Tilt* 0.3, then rock the knob out and back; "
        "set *Primary* to Wave and wire a kick into *Primary Gate*.")

      // ---------------- Sweep ----------------
      .group("sweep", "Sweep")
        .groupHelp(
          "**Sweep** is the knob; 0.5 is home. **Deadzone** is how much of "
          "each half stays fully lit, **Depth** how far the ends go.\n\n"
          "**Tilt** is how much the three bars roll instead of moving "
          "together. In **Up** the choke is a front climbing the rack as you "
          "turn out and coming back down as you turn in. While the knob moves "
          "a tilted rack lights only around that front — at 1, the bar it is "
          "passing. At rest the tilt falls back to **Rest Tilt** of itself "
          "and the front blurs over **Rest Blur** bars: the bars it has passed "
          "darker, the rest brighter. **Focus** is the moving spot's width. "
          "Riding inside the deadzone never tilts anything.")
      .floatField("sweep", 0.5f, 0.f, 1.f, state::PrimaryInput,
                  nullptr, 0.f, nullptr,
                  "The knob. 0.5 is home: full color, no motion.").wireCombine("replace")
        .label("Sweep", "Sweep")
      .selectField("sweep_mode", SweepUp, state::PrimaryInput,
                   {{"Up", SweepUp}}, false,
                   "Which way the choke travels through the bars.")
        .label("Sweep Mode", "Mode")
      .floatField("deadzone", 0.45f, 0.f, 0.95f, state::PrimaryInput,
                  nullptr, 0.f, nullptr,
                  "Fraction of each half of the knob that stays fully lit.")
        .label("Deadzone", "DZ")
      .floatField("depth", 1.0f, 0.f, 1.f, state::PrimaryInput,
                  nullptr, 0.f, nullptr,
                  "How far the ends fade. 1 chokes to black.")
        .label("Depth", "Depth")
      .floatField("tilt", 0.5f, 0.f, 1.f, state::PrimaryInput,
                  nullptr, 0.f, nullptr,
                  "0 moves all three bars together; 1 lights only the bar the "
                  "front is passing while the knob moves.")
        .label("Tilt", "Tilt")
      .floatField("rest_tilt", 0.35f, 0.f, 1.f, state::PrimaryInput,
                  nullptr, 0.f, nullptr,
                  "How much of the tilt is kept once the knob stops.")
        .label("Rest Tilt", "Rest")
      .floatField("rest_blur", 1.0f, 0.05f, 3.f, state::SecondaryInput,
                  nullptr, 0.f, nullptr,
                  "How soft the front is at rest, in bars.")
        .label("Rest Blur", "Blur")
      .floatField("focus", 0.45f, 0.05f, 2.f, state::SecondaryInput,
                  nullptr, 0.f, nullptr,
                  "The moving spot's half-width, in bars.")
        .label("Focus", "Focus")
      .floatField("bounce", 0.55f, 0.f, 1.f, state::PrimaryInput,
                  nullptr, 0.f, nullptr,
                  "Coming back in fast, the light leads your hand past full and "
                  "springs back. How hard. Inward only.")
        .label("Bounce", "Bnce")
      .floatField("sense", 2.0f, 0.1f, 8.f, state::SecondaryInput,
                  nullptr, 0.f, nullptr,
                  "Knob ranges per second that count as a full-speed sweep.")
        .label("Sensitivity", "Sens")
      .floatField("decay", 0.18f, 0.01f, 1.f, state::SecondaryInput,
                  nullptr, 0.f, nullptr,
                  "How long the motion reading trails after the knob stops, s.")
        .label("Motion Decay", "Decay")
      .floatField("window", 0.09f, 0.f, 0.4f, state::SecondaryInput,
                  nullptr, 0.f, nullptr,
                  "Span the knob's speed is measured over, s. Long enough to "
                  "span a MIDI encoder's steps.")
        .label("Rate Window", "Win")

      // ---------------- Tail ----------------
      .group("tail", "Tail")
        .groupHelp(
          "The reverb. A pass through the middle charges by how HARD it was "
          "(**Charge Drive**), and the charge leaks away over **Charge "
          "Hold**. Reaching a choke throws it: all three bars land at once, "
          "then the hits roll up and down the rack one bar at a time, getting "
          "shorter until they sputter out over **Ring Time**. **Tail Level** "
          "is how bright the hits are.")
      .floatField("latch_drive", 2.0f, 0.f, 6.f, state::PrimaryInput,
                  nullptr, 0.f, nullptr,
                  "How readily a pass through the middle charges a throw.")
        .label("Charge Drive", "Drive")
      .floatField("latch_decay", 1.2f, 0.05f, 6.f, state::SecondaryInput,
                  nullptr, 0.f, nullptr,
                  "Seconds a charge survives before it bleeds away.")
        .label("Charge Hold", "Hold")
      .floatField("ring_time", 1.4f, 0.05f, 8.f, state::PrimaryInput,
                  nullptr, 0.f, nullptr,
                  "Seconds a full throw takes to ring out.")
        .label("Ring Time", "Ring")
      .floatField("tail_level", 0.6f, 0.f, 1.5f, state::PrimaryInput,
                  nullptr, 0.f, nullptr,
                  "How bright the tail's hits are. 0 turns the tail off.")
        .label("Tail Level", "Lvl")
      .floatField("tail_rate", 22.f, 4.f, 60.f, state::SecondaryInput,
                  nullptr, 0.f, nullptr,
                  "Hits per second. Constant through the tail.")
        .label("Roll Rate", "Rate")
      .floatField("tail_duty", 0.55f, 0.f, 1.f, state::SecondaryInput,
                  nullptr, 0.f, nullptr,
                  "How long each hit is at a full throw, as a fraction of a step.")
        .label("Roll Duty", "Duty")
      .floatField("tail_grace", 0.35f, 0.f, 1.f, state::SecondaryInput,
                  nullptr, 0.f, nullptr,
                  "A quiet grace stroke on the next bar just before each hit. "
                  "0 for a plain roll.")
        .label("Grace", "Grace")

      // ---------------- Primary ----------------
      .group("primary", "Primary")
        .groupHelp(
          "One gate, three ways to play it. **Amount** trims the whole mix-in "
          "live; each mode has its own amount for gain staging.\n\n"
          "**Flam** — a sidechain duck. The gate rising starts a smooth "
          "**Attack** to the full cut, which backs off to **Sustain** over "
          "**Decay** while the gate holds, then **Release** after it falls. "
          "The cut shudders: chopped at **Shudder Rate**, the open frames "
          "keep **Shudder Floor** of it. At high rates it is one frame on, "
          "one off.\n\n"
          "**Wave** — a pulse born at the stage center, its front running out "
          "across the room at **Speed** (m/s), so it reaches the racks' inner "
          "ends first and their outer ends last. **Start** is where it is "
          "born: 1 m is the inner ends, so a kick lands at once; 0 is the "
          "middle of the stage and travels the gap. The front eats the light "
          "and it comes back over **Recover**. On the front itself it is "
          "bit-crushed — blocks of **Crush** segments — and **Sparkle**s in "
          "the **Wave Color**, a new pattern **Sparkle Rate** times a second "
          "(seeded by *Seed*).\n\n"
          "**Beating** — two rates stepped once per FRAME, **Rate** and Rate + "
          "**Detune**. Near half the frame rate they alias into standing "
          "flicker, and their interference beats up and down the racks: "
          "**Spread A/B** are how far each one's phase turns per bar. Each bar "
          "stays one solid color. The envelope is Flam's.")
      .eventField("primary", state::PrimaryInput)
        .label("Primary Gate", "Prim")
      .selectField("primary_mode", PrimaryFlam, state::PrimaryInput,
                   {{"Flam", PrimaryFlam}, {"Wave", PrimaryWave},
                    {"Beating", PrimaryBeating}}, false,
                   "What the primary gate plays.")
        .label("Primary Mode", "Mode")
      .floatField("primary_amount", 1.0f, 0.f, 1.f, state::PrimaryInput,
                  nullptr, 0.f, nullptr, "Live trim for the primary mix-in.")
        .label("Primary Amount", "Amt")
      .floatField("env_attack", 0.03f, 0.f, 1.f, state::PrimaryInput,
                  nullptr, 0.f, "s", "Seconds to the full cut.")
        .label("Attack", "Atk")
      .floatField("env_decay", 0.25f, 0.f, 2.f, state::PrimaryInput,
                  nullptr, 0.f, "s", "Seconds from the full cut back to the sustain.")
        .label("Decay", "Dec")
      .floatField("env_sustain", 0.4f, 0.f, 1.f, state::PrimaryInput,
                  nullptr, 0.f, nullptr, "The cut held while the gate is up.")
        .label("Sustain", "Sus")
      .floatField("env_release", 0.2f, 0.f, 3.f, state::PrimaryInput,
                  nullptr, 0.f, "s", "Seconds back to nothing once the gate falls.")
        .label("Release", "Rel")
      .floatField("flam_amount", 0.85f, 0.f, 1.f, state::PrimaryInput,
                  nullptr, 0.f, nullptr, "How deep the duck cuts.")
        .label("Flam Amount", "Flam")
      .floatField("flam_rate", 18.f, 0.f, 60.f, state::PrimaryInput,
                  nullptr, 0.f, "Hz",
                  "Shudder cycles per second. 0 is a plain duck; at half the "
                  "frame rate and up, one frame on, one off.")
        .label("Shudder Rate", "Rate")
      .floatField("flam_floor", 0.35f, 0.f, 1.f, state::SecondaryInput,
                  nullptr, 0.f, nullptr,
                  "How much of the cut the shudder's open frames keep.")
        .label("Shudder Floor", "Floor")
      .floatField("wave_amount", 0.9f, 0.f, 1.f, state::PrimaryInput,
                  nullptr, 0.f, nullptr, "How much light the front eats.")
        .label("Wave Amount", "Wave")
      .floatField("wave_speed", 6.f, 0.5f, 40.f, state::PrimaryInput,
                  nullptr, 0.f, "m/s", "How fast the front crosses the room.")
        .logScale()
        .label("Speed", "Spd")
      .floatField("wave_start", 1.0f, 0.f, 2.f, state::PrimaryInput,
                  nullptr, 0.f, "m",
                  "Where the front is born, from the stage center. 1 is the "
                  "racks' inner ends.")
        .label("Start", "Start")
      .floatField("wave_recover", 0.35f, 0.02f, 3.f, state::PrimaryInput,
                  nullptr, 0.f, "s", "How long the light takes to come back behind it.")
        .label("Recover", "Rec")
      .floatField("wave_width", 0.25f, 0.02f, 1.f, state::SecondaryInput,
                  nullptr, 0.f, "m", "How far either side of the front is crushed.")
        .label("Front Width", "Width")
      .intField("wave_crush", 4, 1, 15, state::SecondaryInput,
                0, nullptr, "Segments per block on the front.")
        .label("Crush", "Crush")
      .floatField("wave_sparkle", 1.2f, 0.f, 3.f, state::PrimaryInput,
                  nullptr, 0.f, nullptr,
                  "How bright the front's sparkle is. Past 1 it blows out.")
        .label("Sparkle", "Spark")
      .floatField("wave_sparkle_rate", 30.f, 1.f, 60.f, state::SecondaryInput,
                  nullptr, 0.f, "Hz",
                  "New sparkle patterns per second; past half the frame rate, "
                  "every frame.")
        .logScale()
        .label("Sparkle Rate", "SpRate")
      .rgbField("wave_color", 0.30f, 0.85f, 1.00f, state::PrimaryInput)
        .label("Wave Color", "WCol")
      .floatField("beat_amount", 0.8f, 0.f, 1.f, state::PrimaryInput,
                  nullptr, 0.f, nullptr, "How deep the beating cuts.")
        .label("Beat Amount", "Beat")
      .floatField("beat_rate", 29.f, 0.f, 60.f, state::PrimaryInput,
                  nullptr, 0.f, "Hz",
                  "The first rate, stepped per frame. Near half the frame rate "
                  "it aliases into standing flicker.")
        .label("Rate", "Rate")
      .floatField("beat_detune", 1.5f, -8.f, 8.f, state::PrimaryInput,
                  nullptr, 0.f, "Hz",
                  "How far the second rate is off the first — the beat's speed.")
        .label("Detune", "Det")
      .floatField("beat_spread_a", 0.0f, -1.f, 1.f, state::SecondaryInput,
                  nullptr, 0.f, nullptr,
                  "Cycles the first rate's phase turns per bar up the rack.")
        .label("Spread A", "SprA")
      .floatField("beat_spread_b", 0.33f, -1.f, 1.f, state::SecondaryInput,
                  nullptr, 0.f, nullptr,
                  "Cycles the second rate's phase turns per bar. A difference "
                  "from Spread A makes the beat travel up or down.")
        .label("Spread B", "SprB")

      // ---------------- Secondary ----------------
      .group("secondary", "Secondary")
        .groupHelp(
          "Three more gates — wire bouncy-abode's *gate low*, *mid* and "
          "*high*. A **Pattern** deals each channel one bar on the left and "
          "one on the right, dealt again every **Step** (seeded by **Seed**). "
          "A channel hitting alone flashes its bar in its own color on ONE "
          "side; one alternator shared by all three picks which, so two hits "
          "landing together go to opposite sides. All three within **Heavy "
          "Window** of each other is a heavy hit: every channel's bar, both "
          "sides. Flashes fall over **Flash Decay**; **Flash Flam** chops "
          "them.")
      .eventField("sec_low", state::PrimaryInput)
        .label("Low Gate", "Low")
      .eventField("sec_mid", state::PrimaryInput)
        .label("Mid Gate", "Mid")
      .eventField("sec_high", state::PrimaryInput)
        .label("High Gate", "High")
      .selectField("secondary_mode", SecondaryPattern, state::PrimaryInput,
                   {{"Pattern", SecondaryPattern}}, false,
                   "What the secondary gates play.")
        .label("Secondary Mode", "Mode")
      .floatField("secondary_amount", 1.0f, 0.f, 1.f, state::PrimaryInput,
                  nullptr, 0.f, nullptr, "Live trim for the secondary flashes.")
        .label("Secondary Amount", "Amt")
      .selectField("step_mode", StepBar, state::PrimaryInput,
                   {{"1 Beat", StepBeat}, {"2 Beats", Step2Beats}, {"1 Bar", StepBar},
                    {"2 Bars", Step2Bars}, {"4 Bars", Step4Bars},
                    {"Primary", StepPrimary}}, true,
                   "How often the pattern is dealt again: on the transport's "
                   "grid, or on every primary hit.")
        .label("Step", "Step")
      .intField("seed", 0, 0, 0x7FFFFFFF, state::SecondaryInput,
                0, nullptr, "Which pattern, and which sparkle.")
        .label("Seed", "Seed")
      .floatField("sec_decay", 0.18f, 0.01f, 2.f, state::PrimaryInput,
                  nullptr, 0.f, "s", "Seconds a flash takes to fall.")
        .label("Flash Decay", "Dec")
      .floatField("sec_flam_rate", 0.f, 0.f, 60.f, state::SecondaryInput,
                  nullptr, 0.f, "Hz",
                  "Chops a flash into a flam. 0 is a plain flash.")
        .label("Flash Flam", "Flam")
      .floatField("heavy_window", 0.04f, 0.f, 0.25f, state::SecondaryInput,
                  nullptr, 0.f, "s",
                  "How close together all three must land to count as heavy.")
        .label("Heavy Window", "Heavy")
      .rgbField("low_color", 1.00f, 0.45f, 0.10f, state::PrimaryInput)
        .label("Low Color", "LCol")
      .rgbField("mid_color", 0.30f, 0.85f, 1.00f, state::PrimaryInput)
        .label("Mid Color", "MCol")
      .rgbField("high_color", 0.75f, 0.40f, 1.00f, state::PrimaryInput)
        .label("High Color", "HCol")

      // ---------------- Lift + strobe ----------------
      .group("lift", "Lift & Strobe")
        .groupHelp(
          "**Lift** brings every segment up in the **Accent** color, even a "
          "choked rig. **Strobe** flashes the accent and blacks out between: "
          "just above 0 it runs at **Strobe Min**, at 1 at **Strobe Max**, "
          "and past half the frame rate it is exactly one frame on, one off.")
      .floatField("lift", 0.f, 0.f, 1.f, state::PrimaryInput,
                  nullptr, 0.f, nullptr, "How far everything is lifted.")
        .label("Lift", "Lift")
      .floatField("strobe", 0.f, 0.f, 1.f, state::PrimaryInput,
                  nullptr, 0.f, nullptr, "Strobe depth: 0 off, up from there faster.")
        .label("Strobe", "Strb")
      .floatField("strobe_min_hz", 4.f, 0.5f, 60.f, state::SecondaryInput,
                  nullptr, 0.f, "Hz", "Flashes per second just above 0.")
        .logScale()
        .label("Strobe Min", "Min")
      .floatField("strobe_max_hz", 30.f, 0.5f, 60.f, state::SecondaryInput,
                  nullptr, 0.f, "Hz", "Flashes per second at 1.")
        .logScale()
        .label("Strobe Max", "Max")
      .rgbField("lift_color", 1.f, 1.f, 1.f, state::PrimaryInput)
        .label("Accent", "Acc")

      // ---------------- Output ----------------
      .group("output", "Output")
        .groupHelp(
          "**Color** is the sweep's. **Mix Post** moves the mix-ins from "
          "under the lift and strobe (0) to over them (1): at 1 a duck cuts "
          "the strobe too and a wave eats through the lift.")
      .rgbField("color", 1.00f, 0.22f, 0.62f, state::PrimaryInput)
        .label("Color", "Col")
      .floatField("intensity", 1.0f, 0.f, 2.f, state::PrimaryInput,
                  nullptr, 0.f, nullptr, "Overall output level.")
        .label("Intensity", "Int")
      .floatField("mix_post", 0.f, 0.f, 1.f, state::PrimaryInput,
                  nullptr, 0.f, nullptr,
                  "0: mix-ins under the lift and strobe. 1: over them.")
        .label("Mix Post", "Post")
      .floatField("whiten", 0.6f, 0.f, 1.f, state::SecondaryInput,
                  nullptr, 0.f, nullptr,
                  "How far anything past full pushes toward white.")
        .label("Overdrive White", "White")
      .boolField("quantize", true, state::SecondaryInput,
                 "Each segment a flat block, so a fixture lands on its exact "
                 "color. Off interpolates along each bar.")
        .label("Quantize", "Quant")
      .floatField("speed_out", 0.f, 0.f, 1.f, state::PrimaryOutput, "unsigned",
                  0.f, nullptr, "How hard the knob is moving, 0..1.")
        .label("Speed", "Spd")
      .floatField("release_out", 0.f, 0.f, 1.f, state::PrimaryOutput, "unsigned",
                  0.f, nullptr, "The thrown tail, falling to 0 as it rings out.")
        .label("Release", "Rel")
      .floatField("level_out", 1.f, 0.f, 1.f, state::PrimaryOutput, "unsigned",
                  0.f, nullptr, "The sweep's mean level over the three bars.")
        .label("Level", "Lvl")
      .floatField("primary_env_out", 0.f, 0.f, 1.f, state::PrimaryOutput, "unsigned",
                  0.f, nullptr, "The primary mix-in's envelope.")
        .label("Primary Env", "PEnv")
      .floatField("sweep_out", 0.5f, 0.f, 1.f, state::PrimaryOutput, "unsigned",
                  0.f, nullptr, "The knob, passed through.").wireCombine("replace")
        .label("Sweep Out", "SwOut")
      .textureField("tex_in",  state::PrimaryInput)
      .textureField("tex_out", state::PrimaryOutput)

      .capability(state::Capability::Generator)
      .capability(state::Capability::ModulationSource)
      .capability(state::Capability::ModulationSourceMulti)
  );
  state::setOnStateReady(&on_state_ready);

  if (gpu::Device::backend() == gpu::Backend::None) return;

  state::registerShaderSPV("side_racks_led", LED_SPV, LED_SPV_SIZE);
  if (auto led = gpu::Device::createShaderModuleByName("side_racks_led")) {
    s_pso = gpu::Device::createComputePSO(led, "main", gpu::Bindings()
        .storageTex2d(0)
        .uniform(1));
  }
  state::log("side_racks: module initialized");
}

void* create() {
  auto* s = new State();
  s->led_buf = gpu::Device::createBuffer(sizeof(RackUniforms), gpu::BufferUsage::Uniform);
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
  s->step_tick.reset();
  s->initialized = false;
  if (!s_pso.valid() || !s->led_buf.valid()) return;
  s->initialized = true;
}

// The core advances here and ONLY here: render() may run without a tick
// (thumbnails, off-playhead previews), and stepping from there would age the
// show every time somebody looked at the card.
void tick(void* self, double dt) {
  auto* s = static_cast<State*>(self);
  if (!s) return;
  s->core.advanceStep(s->step_tick.tick(stepsPerBar(s->p.step_mode)));
  s->core.tick(s->p, (float)dt);
  publishRails(*s);
}

static void readRgb(int i, float out[3]) {
  auto v = state::patchVec3(i);
  out[0] = v.x; out[1] = v.y; out[2] = v.z;
}

void on_state_patched(void* self, int n, const char* pb, const int* off,
                      const int* len, const int* ops) {
  auto* s = static_cast<State*>(self);
  if (!s) return;
  auto& p = s->p;
  auto& k = p.knob;
  const int mode_before = p.primary_mode;
  for (int i = 0; i < n; i++) {
    if (ops[i] != state::PatchReplace) continue;
    const char* path = pb + off[i];
    const int l = len[i];
    // The gates are read as levels; the core finds their edges itself.
    if      (state::pathIs(path, l, "primary"))           p.primary = state::patchFloat(i);
    else if (state::pathIs(path, l, "sec_low"))           p.sec[0] = state::patchFloat(i);
    else if (state::pathIs(path, l, "sec_mid"))           p.sec[1] = state::patchFloat(i);
    else if (state::pathIs(path, l, "sec_high"))          p.sec[2] = state::patchFloat(i);
    else if (state::pathIs(path, l, "sweep"))             k.sweep = state::patchFloat(i);
    else if (state::pathIs(path, l, "sweep_mode"))        p.sweep_mode = state::patchInt(i);
    else if (state::pathIs(path, l, "deadzone"))          k.deadzone = state::patchFloat(i);
    else if (state::pathIs(path, l, "depth"))             k.depth = state::patchFloat(i);
    else if (state::pathIs(path, l, "tilt"))              p.tilt = state::patchFloat(i);
    else if (state::pathIs(path, l, "rest_tilt"))         p.rest_tilt = state::patchFloat(i);
    else if (state::pathIs(path, l, "rest_blur"))         p.rest_blur = state::patchFloat(i);
    else if (state::pathIs(path, l, "focus"))             p.focus = state::patchFloat(i);
    else if (state::pathIs(path, l, "bounce"))            k.bounce = state::patchFloat(i);
    else if (state::pathIs(path, l, "sense"))             k.sense = state::patchFloat(i);
    else if (state::pathIs(path, l, "decay"))             k.decay = state::patchFloat(i);
    else if (state::pathIs(path, l, "window"))            k.window = state::patchFloat(i);
    else if (state::pathIs(path, l, "latch_drive"))       k.latch_drive = state::patchFloat(i);
    else if (state::pathIs(path, l, "latch_decay"))       k.latch_decay = state::patchFloat(i);
    else if (state::pathIs(path, l, "ring_time"))         k.ring_time = state::patchFloat(i);
    else if (state::pathIs(path, l, "tail_level"))        p.tail_level = state::patchFloat(i);
    else if (state::pathIs(path, l, "tail_rate"))         p.tail_rate = state::patchFloat(i);
    else if (state::pathIs(path, l, "tail_duty"))         p.tail_duty = state::patchFloat(i);
    else if (state::pathIs(path, l, "tail_grace"))        p.tail_grace = state::patchFloat(i);
    else if (state::pathIs(path, l, "primary_mode"))      p.primary_mode = state::patchInt(i);
    else if (state::pathIs(path, l, "primary_amount"))    p.primary_amount = state::patchFloat(i);
    else if (state::pathIs(path, l, "env_attack"))        p.env_attack = state::patchFloat(i);
    else if (state::pathIs(path, l, "env_decay"))         p.env_decay = state::patchFloat(i);
    else if (state::pathIs(path, l, "env_sustain"))       p.env_sustain = state::patchFloat(i);
    else if (state::pathIs(path, l, "env_release"))       p.env_release = state::patchFloat(i);
    else if (state::pathIs(path, l, "flam_amount"))       p.flam_amount = state::patchFloat(i);
    else if (state::pathIs(path, l, "flam_rate"))         p.flam_rate = state::patchFloat(i);
    else if (state::pathIs(path, l, "flam_floor"))        p.flam_floor = state::patchFloat(i);
    else if (state::pathIs(path, l, "wave_amount"))       p.wave_amount = state::patchFloat(i);
    else if (state::pathIs(path, l, "wave_speed"))        p.wave_speed = state::patchFloat(i);
    else if (state::pathIs(path, l, "wave_start"))        p.wave_start = state::patchFloat(i);
    else if (state::pathIs(path, l, "wave_recover"))      p.wave_recover = state::patchFloat(i);
    else if (state::pathIs(path, l, "wave_width"))        p.wave_width = state::patchFloat(i);
    else if (state::pathIs(path, l, "wave_crush"))        p.wave_crush = state::patchInt(i);
    else if (state::pathIs(path, l, "wave_sparkle"))      p.wave_sparkle = state::patchFloat(i);
    else if (state::pathIs(path, l, "wave_sparkle_rate")) p.wave_sparkle_rate = state::patchFloat(i);
    else if (state::pathIs(path, l, "wave_color"))        readRgb(i, p.wave_color);
    else if (state::pathIs(path, l, "beat_amount"))       p.beat_amount = state::patchFloat(i);
    else if (state::pathIs(path, l, "beat_rate"))         p.beat_rate = state::patchFloat(i);
    else if (state::pathIs(path, l, "beat_detune"))       p.beat_detune = state::patchFloat(i);
    else if (state::pathIs(path, l, "beat_spread_a"))     p.beat_spread_a = state::patchFloat(i);
    else if (state::pathIs(path, l, "beat_spread_b"))     p.beat_spread_b = state::patchFloat(i);
    else if (state::pathIs(path, l, "secondary_mode"))    p.secondary_mode = state::patchInt(i);
    else if (state::pathIs(path, l, "secondary_amount"))  p.secondary_amount = state::patchFloat(i);
    else if (state::pathIs(path, l, "step_mode"))         p.step_mode = state::patchInt(i);
    else if (state::pathIs(path, l, "seed"))              p.seed = state::patchInt(i);
    else if (state::pathIs(path, l, "sec_decay"))         p.sec_decay = state::patchFloat(i);
    else if (state::pathIs(path, l, "sec_flam_rate"))     p.sec_flam_rate = state::patchFloat(i);
    else if (state::pathIs(path, l, "heavy_window"))      p.heavy_window = state::patchFloat(i);
    else if (state::pathIs(path, l, "low_color"))         readRgb(i, p.ch_color[0]);
    else if (state::pathIs(path, l, "mid_color"))         readRgb(i, p.ch_color[1]);
    else if (state::pathIs(path, l, "high_color"))        readRgb(i, p.ch_color[2]);
    else if (state::pathIs(path, l, "lift"))              p.lift = state::patchFloat(i);
    else if (state::pathIs(path, l, "strobe"))            p.strobe = state::patchFloat(i);
    else if (state::pathIs(path, l, "strobe_min_hz"))     p.strobe_min_hz = state::patchFloat(i);
    else if (state::pathIs(path, l, "strobe_max_hz"))     p.strobe_max_hz = state::patchFloat(i);
    else if (state::pathIs(path, l, "lift_color"))        readRgb(i, p.lift_color);
    else if (state::pathIs(path, l, "color"))             readRgb(i, p.color);
    else if (state::pathIs(path, l, "intensity"))         p.intensity = state::patchFloat(i);
    else if (state::pathIs(path, l, "mix_post"))          p.mix_post = state::patchFloat(i);
    else if (state::pathIs(path, l, "whiten"))            p.whiten = state::patchFloat(i);
    else if (state::pathIs(path, l, "quantize"))          s->quantize = state::patchBool(i);
  }
  if (p.primary_mode != mode_before) applyVisibility(p.primary_mode);
}

void render(void* self, int vp_w, int vp_h) {
  auto* s = static_cast<State*>(self);
  if (!s || !s->initialized || vp_w <= 0 || vp_h <= 0) return;
  auto out = gpu::Device::textureForField("tex_out");
  if (!out.valid()) return;

  // A ZERO-dt tick: ages nothing — every envelope, chop, pulse and flash is
  // gated on dt > 0 — but seeds a card rendered before its first tick
  // (thumbnails, previews) at the knob's own level.
  s->core.tick(s->p, 0.0f);
  publishRails(*s);

  const rack_map::Cells cells = s->core.cells(s->p);
  RackUniforms u = {};
  for (int side = 0; side < rack_map::kSides; ++side)
    for (int bar = 0; bar < rack_map::kBars; ++bar)
      for (int seg = 0; seg < rack_map::kSegments; ++seg) {
        const int from = rack_map::cellIndex(side, bar, seg);
        const int to = rack_map::gridIndex(side, bar, seg);
        u.cell[to][0] = cells.rgb[from][0];
        u.cell[to][1] = cells.rgb[from][1];
        u.cell[to][2] = cells.rgb[from][2];
        u.cell[to][3] = 1.0f;
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

}  // namespace side_racks

namespace side_racks {

// Effect store preview: an LFO rocking the sweep in and out of the choke at
// full tilt, and a beat on the primary gate playing the Wave.
void preview(nano::PreviewScenario& s) {
  s.input("black")
      .param("tilt", 1.0f)
      .param("primary_mode", (float)PrimaryWave)
      .aux("lfo", "mod.source.lfo")
      .auxParam("lfo", "rate", 0.6f)
      .wire("lfo.output", "$self.sweep")
      .aux("t", "mod.trigger.beat")
      .auxParam("t", "single_frame", 1.0f)
      .wire("t.output", "$self.primary")
      .capture(1.6f);
}

}  // namespace side_racks
