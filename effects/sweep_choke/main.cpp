/*
 * source.light.sweep_choke — one knob that sweeps and chokes the four LED bars.
 *
 * Three Planes Rig's sweep, taken off the tower and put straight onto the rig:
 * the knob sets a level, full in the middle and black at the ends, and every
 * change to it rolls across the grid at a set angle; a hard sweep into a choke
 * leaves a flickering tail ringing behind it. All of the behaviour lives in
 * <sketch/sweep_choke_core.h>, host-free so the goldens can drive it at an exact
 * dt; this file is the schema, the patch plumbing and one small pass.
 *
 * Two outputs, both the house rig's pixel map — four bars, ten segments, flat
 * blocks — drawn by the shared nano_led_bars.hlsl pass. `tex_out` is what the
 * bars show: the glints, the thrown tail and the bounce's flash, over an
 * optional base. `mask_out` is the level itself, gray, for choking and
 * revealing other effects; it is only drawn while something is wired to it.
 * Both are control signals, so they are opaque and ignore the input.
 */

#include <gpu.h>
#include <preview_scenario.h>
#include <host.h>
#include <val.h>
#include <sketch/sweep_choke_core.h>
#include <sketch/led_bars.h>
#include "sweep_choke_shaders.h"

#include <cstdint>

namespace sweep_choke {

// Mirrors `cbuffer LedUniforms` in shaders/nano_led_bars.hlsl.
struct LedUniforms {
  float cell[led_bars::kCells][4];   // bar-major, segment 0 at the bottom
  float ctl[4];                      // quantize, -, -, -
};
static_assert(sizeof(LedUniforms) == 16 * (led_bars::kCells + 1),
              "LedUniforms layout mismatch with nano_led_bars.hlsl");

struct State {
  sweep_choke_core::Params p;
  sweep_choke_core::Core core;
  float color[3] = {1.00f, 0.22f, 0.62f};
  float intensity = 1.0f;
  float whiten = 0.6f;
  bool quantize = true;

  bool initialized = false;
  gpu::Buffer led_buf;
  gpu::Buffer mask_buf;
  gpu::Texture mask_tex;
  int mask_w = 0, mask_h = 0;
};

static gpu::ComputePSO s_pso;

static void publish(const char* name, float value) {
  auto vh = val::number(value);
  state::setValPath(name, vh);
  val::release(vh);
}

static void publishRails(const State& s) {
  publish("speed_out", s.core.speed);
  publish("release_out", s.core.release);
  publish("level_out", sweep_choke_core::clamp01(s.core.mean_level));
  publish("sweep_out", s.core.sweep_out);
}

void module_init() {
  state::init("source.light.sweep_choke", {1, 0, 0},
    state::Schema()
      .helpField("intro",
        "## Sweep Choke\n"
        "One knob that sweeps and chokes the four LED bars. Map **Sweep** to a "
        "bipolar MIDI knob and perform it.\n\n"
        "**Two outputs.** The bars show the **gesture**: every pass through "
        "the middle throws a **glint** across the grid along **Angle**, at "
        "the speed you moved, always the same way whichever way you went, "
        "gliding on after you stop. Come back in fast and the whole grid "
        "**flashes** past full and springs back. Past full, a segment blows "
        "out toward white.\n\n"
        "**Mask** is the **position**, in gray, for choking and revealing "
        "other effects. Parked in the middle it is white; out past the "
        "deadzone it dims toward black at either end — the **choke**. Park "
        "anywhere and it settles solid at that level. Only a CHANGE is a "
        "gradient: it rolls across the grid along **Angle**, one side first, "
        "and passes.\n\n"
        "**How you swept is what you hear afterwards.** A hard pass through "
        "the middle charges; reaching a choke throws that charge as a "
        "flickering tail on the bars, gathered where the light went out "
        "last, and slings the glints off with it.\n\n"
        "**Try:** *Sides* One-Way and *Angle* 90, *Base* at 0.3, then rock the "
        "knob to the beat — glints and every change to the mask climb the "
        "bars, in and out.")

      // ---------------- Sweep ----------------
      .group("sweep", "Sweep")
        .groupHelp(
          "**Sweep** is the knob; the rest shape what it does. **Deadzone** is "
          "how much of each half stays fully lit — ride inside it without "
          "touching the mask. Entering it is what launches a glint. **Depth** is "
          "how far the ends go; below full the "
          "ends never choke, so nothing is thrown. **Sensitivity** and "
          "**Motion Decay** set how the knob's speed is read, which is what "
          "charges a throw.")
      .floatField("sweep", 0.5f, 0.f, 1.f, state::PrimaryInput,
                  nullptr, 0.f, nullptr,
                  "The knob. 0.5 is home: full color, no motion.").wireCombine("replace")
        .label("Sweep", "Sweep")
      .floatField("deadzone", 0.45f, 0.f, 0.95f, state::PrimaryInput,
                  nullptr, 0.f, nullptr,
                  "Fraction of each half of the knob that stays fully lit.")
        .label("Deadzone", "DZ")
      .floatField("depth", 1.0f, 0.f, 1.f, state::PrimaryInput,
                  nullptr, 0.f, nullptr,
                  "How far the ends fade. 1 chokes to black.")
        .label("Depth", "Depth")
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
                  "span a MIDI encoder's steps; 0 differences per frame.")
        .label("Rate Window", "Win")

      // ---------------- Gradient ----------------
      .group("gradient", "Gradient")
        .groupHelp(
          "How the mask gets to where the knob asks. It always settles to one "
          "solid level; a change **rolls** across the grid to get there, so "
          "one side moves before the other and the gradient is the change "
          "passing through.\n\n"
          "**Angle** is the way it rolls, counter-clockwise from "
          "left-to-right: 90 climbs the bars, 0 runs left to right — the same "
          "way whether the mask is getting darker or lighter. **Sides**: "
          "**Ping-Pong** turns it round on the knob's lower half, so which way "
          "you turned shows in the light; **One-Way** keeps it. **Roll** is "
          "how long a change takes to cross the grid — 0 changes it all at "
          "once. **Soften** eases each segment on, so a jump rolls through as "
          "a smooth gradient rather than a hard edge.")
      .floatField("angle", 90.f, 0.f, 360.f, state::PrimaryInput,
                  nullptr, 0.f, nullptr,
                  "Direction changes roll, degrees CCW from +x.")
        .label("Angle", "Ang")
      .selectField("side_mode", sweep_choke_core::SidePingPong, state::PrimaryInput,
                   {{"Ping-Pong", sweep_choke_core::SidePingPong},
                    {"One-Way", sweep_choke_core::SideOneWay}}, false,
                   "Ping-Pong turns the roll round for the lower half of the "
                   "knob; One-Way rolls along the angle both ways.")
        .label("Sides", "Sides")
      .floatField("roll", 0.35f, 0.f, 4.f, state::PrimaryInput,
                  nullptr, 0.f, "s",
                  "Seconds a change takes to cross the grid. 0 all at once.")
        .label("Roll", "Roll")
      .floatField("soften", 0.12f, 0.f, 2.f, state::PrimaryInput,
                  nullptr, 0.f, "s",
                  "Seconds each segment eases toward what reaches it. Higher is "
                  "a softer, longer gradient.")
        .label("Soften", "Soft")

      // ---------------- Light ----------------
      .group("light", "Light")
        .groupHelp(
          "What the bars show. **Base** puts the mask under the glints, in "
          "color — 0 leaves the bars black between gestures, 1 is the full "
          "mask. **Bounce** is the flash coming back in fast; **Overdrive "
          "White** is how far anything past full — a bright glint, a hard "
          "bounce — blows out toward white.")
      .rgbField("color", 1.00f, 0.22f, 0.62f, state::PrimaryInput)
        .label("Color", "Col")
      .floatField("intensity", 1.0f, 0.f, 2.f, state::PrimaryInput,
                  nullptr, 0.f, nullptr, "Overall output level.")
        .label("Intensity", "Int")
      .floatField("base", 0.0f, 0.f, 1.f, state::PrimaryInput,
                  nullptr, 0.f, nullptr,
                  "How much of the mask the bars show under the glints.")
        .label("Base", "Base")
      .floatField("bounce", 0.55f, 0.f, 1.f, state::PrimaryInput,
                  nullptr, 0.f, nullptr,
                  "Coming back in fast, the grid flashes ahead of your hand, "
                  "past full, and springs back. How hard. Inward only.")
        .label("Bounce", "Bnce")
      .floatField("whiten", 0.6f, 0.f, 1.f, state::SecondaryInput,
                  nullptr, 0.f, nullptr,
                  "How far anything past full pushes toward white.")
        .label("Overdrive White", "White")

      // ---------------- Glint ----------------
      .group("glint", "Glint")
        .groupHelp(
          "Three Planes' glints, on the bars. **The glint is the gesture**: "
          "each time the knob enters the deadzone one launches, and at *Ratio* "
          "1 a steady sweep across the whole knob carries it across the whole "
          "grid in the same time. Which way you moved is thrown away — they "
          "always travel along the Gradient's **Angle**, the way the dark "
          "leaves — and once launched only their speed follows the knob. Stop "
          "and they glide on for **Linger**, shrinking and dimming.\n\n"
          "Sweep hard and **Chaos** adds small ones underneath. A throw holds "
          "them at full while its tail rings, and **Sling** is how hard it "
          "hurries them off. **Shadow** is the dark wake behind each one, "
          "which only shows on a *Base*.")
      .floatField("glint_ratio", 1.0f, 0.f, 4.f, state::PrimaryInput,
                  nullptr, 0.f, nullptr,
                  "Crossings of the grid per full range of the knob.")
        .label("Ratio", "Ratio")
      .floatField("glint_chaos", 4.0f, 0.f, 16.f, state::PrimaryInput,
                  nullptr, 0.f, "/s",
                  "Small extra glints per second, once the sweep is brisk. 0 "
                  "leaves one clean glint per pass.")
        .label("Chaos", "Chaos")
      .floatField("glint_gain", 1.6f, 0.f, 4.f, state::PrimaryInput,
                  nullptr, 0.f, nullptr,
                  "A glint's peak. Past 1 it blows out toward white.")
        .label("Glint Gain", "Gain")
      .floatField("glint_shadow", 0.55f, 0.f, 1.f, state::PrimaryInput,
                  nullptr, 0.f, nullptr,
                  "The dark wake trailing each glint, on the base.")
        .label("Shadow", "Shadow")
      .floatField("glint_linger", 0.22f, 0.05f, 3.f, state::PrimaryInput,
                  nullptr, 0.f, "s",
                  "How long a glint glides on after the knob stops, s. Long "
                  "ones coast across the grid; short ones stop and go out.")
        .label("Linger", "Linger")
      .floatField("glint_width", 0.1f, 0.02f, 0.5f, state::SecondaryInput,
                  nullptr, 0.f, nullptr,
                  "Half-width, as a fraction of the grid it crosses. Small is "
                  "a hard slash, large a soft sheen.")
        .label("Glint Width", "GlWid")
      .floatField("glint_sling", 1.0f, 0.f, 1.f, state::SecondaryInput,
                  nullptr, 0.f, nullptr,
                  "How much a throw speeds the glints off. 0 holds them and "
                  "lets them drift.")
        .label("Sling", "Sling")

      // ---------------- Tail ----------------
      .group("tail", "Tail")
        .groupHelp(
          "The reverberant tail, on the bars. A pass through the middle charges by how HARD "
          "it was (**Charge Drive**), and the charge leaks away over **Charge "
          "Hold** — so a flick then a dawdle throws a soft one. Reaching a "
          "choke throws it, and it rings out over **Ring Time** as segments "
          "blinking where the roll reaches last (**Bias**).\n\n"
          "The tail never dims. Blinks keep arriving at **Blink Rate**; what "
          "runs down is how LONG each one is (**Blink Length**), until they "
          "start missing frames and it sputters out. **Density** is segments "
          "per blink. **Chunk** makes them bigger and fewer: runs up the bar, "
          "to whole bars at 1.")
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
      .floatField("tail_rate", 18.f, 1.f, 60.f, state::PrimaryInput,
                  nullptr, 0.f, nullptr,
                  "Blinks per second. Constant through the tail.")
        .label("Blink Rate", "Rate")
      .floatField("tail_duty", 2.0f, 0.f, 6.f, state::PrimaryInput,
                  nullptr, 0.f, nullptr,
                  "Blink length at full release, in blink intervals. Above 1 "
                  "they overlap.")
        .label("Blink Length", "Len")
      .floatField("tail_density", 2.5f, 0.f, 6.f, state::PrimaryInput,
                  nullptr, 0.f, nullptr,
                  "Segments lit per blink.")
        .label("Density", "Dens")
      .floatField("tail_bias", 0.75f, 0.f, 1.f, state::PrimaryInput,
                  nullptr, 0.f, nullptr,
                  "0 blinks anywhere; 1 only where the roll reaches last.")
        .label("Bias", "Bias")
      .floatField("tail_chunk", 0.0f, 0.f, 1.f, state::PrimaryInput,
                  nullptr, 0.f, nullptr,
                  "Blink size. 0 single segments; up from there runs of random "
                  "length up the bar, fewer of them; 1 whole bars.")
        .label("Chunk", "Chunk")

      // ---------------- Output ----------------
      .group("output", "Output")
      .boolField("quantize", true, state::SecondaryInput,
                 "Each segment a flat block, so a pixel mapper lands on its "
                 "exact color. Off interpolates between segment centers.")
        .label("Quantize", "Quant")
      .floatField("speed_out", 0.f, 0.f, 1.f, state::PrimaryOutput, "unsigned",
                  0.f, nullptr, "How hard the knob is moving, 0..1.")
        .label("Speed", "Spd")
      .floatField("release_out", 0.f, 0.f, 1.f, state::PrimaryOutput, "unsigned",
                  0.f, nullptr, "The thrown tail: its charge at the choke, "
                  "falling to 0 across the ring-out.")
        .label("Release", "Rel")
      .floatField("level_out", 1.f, 0.f, 1.f, state::PrimaryOutput, "unsigned",
                  0.f, nullptr, "Mean level of the mask.")
        .label("Level", "Lvl")
      .floatField("sweep_out", 0.5f, 0.f, 1.f, state::PrimaryOutput, "unsigned",
                  0.f, nullptr, "The knob, passed through — wire it to Three "
                  "Planes' Glint Sweep.").wireCombine("replace")
        .label("Sweep Out", "SwOut")
      .textureField("tex_in",  state::PrimaryInput)
      .textureField("tex_out", state::PrimaryOutput)
      .textureField("mask_out", state::SecondaryOutput)

      .capability(state::Capability::Generator)
      .capability(state::Capability::ModulationSource)
      .capability(state::Capability::ModulationSourceMulti)
  );

  if (gpu::Device::backend() == gpu::Backend::None) return;

  state::registerShaderSPV("sweep_choke_led", LED_SPV, LED_SPV_SIZE);
  if (auto led = gpu::Device::createShaderModuleByName("sweep_choke_led")) {
    s_pso = gpu::Device::createComputePSO(led, "main", gpu::Bindings()
        .storageTex2d(0)
        .uniform(1));
  }
  state::log("sweep_choke: module initialized");
}

void* create() {
  auto* s = new State();
  s->led_buf = gpu::Device::createBuffer(sizeof(LedUniforms), gpu::BufferUsage::Uniform);
  s->mask_buf = gpu::Device::createBuffer(sizeof(LedUniforms), gpu::BufferUsage::Uniform);
  return s;
}

void destroy(void* self) {
  auto* s = static_cast<State*>(self);
  if (!s) return;
  s->led_buf.release();
  s->mask_buf.release();
  s->mask_tex.release();
  delete s;
}

void init(void* self) {
  auto* s = static_cast<State*>(self);
  if (!s) return;
  s->core.reset();
  s->initialized = false;
  if (!s_pso.valid() || !s->led_buf.valid() || !s->mask_buf.valid()) return;
  s->initialized = true;
}

// The core advances here and ONLY here: render() may run without a tick
// (thumbnails, off-playhead previews), and stepping from there would age the
// tail every time somebody looked at the card.
void tick(void* self, double dt) {
  auto* s = static_cast<State*>(self);
  if (!s) return;
  s->core.tick(s->p, (float)dt);
  publishRails(*s);
}

void on_state_patched(void* self, int n, const char* pb, const int* off,
                      const int* len, const int* ops) {
  auto* s = static_cast<State*>(self);
  if (!s) return;
  auto& p = s->p;
  for (int i = 0; i < n; i++) {
    if (ops[i] != state::PatchReplace) continue;
    const char* path = pb + off[i];
    const int l = len[i];
    if      (state::pathIs(path, l, "sweep"))        p.sweep = state::patchFloat(i);
    else if (state::pathIs(path, l, "deadzone"))     p.deadzone = state::patchFloat(i);
    else if (state::pathIs(path, l, "depth"))        p.depth = state::patchFloat(i);
    else if (state::pathIs(path, l, "sense"))        p.sense = state::patchFloat(i);
    else if (state::pathIs(path, l, "decay"))        p.decay = state::patchFloat(i);
    else if (state::pathIs(path, l, "window"))       p.window = state::patchFloat(i);
    else if (state::pathIs(path, l, "angle"))        p.angle_deg = state::patchFloat(i);
    else if (state::pathIs(path, l, "side_mode"))    p.side_mode = state::patchInt(i);
    else if (state::pathIs(path, l, "roll"))         p.roll = state::patchFloat(i);
    else if (state::pathIs(path, l, "soften"))       p.soften = state::patchFloat(i);
    else if (state::pathIs(path, l, "bounce"))       p.bounce = state::patchFloat(i);
    else if (state::pathIs(path, l, "latch_drive"))  p.latch_drive = state::patchFloat(i);
    else if (state::pathIs(path, l, "latch_decay"))  p.latch_decay = state::patchFloat(i);
    else if (state::pathIs(path, l, "ring_time"))    p.ring_time = state::patchFloat(i);
    else if (state::pathIs(path, l, "tail_rate"))    p.tail_rate = state::patchFloat(i);
    else if (state::pathIs(path, l, "tail_duty"))    p.tail_duty = state::patchFloat(i);
    else if (state::pathIs(path, l, "tail_density")) p.tail_density = state::patchFloat(i);
    else if (state::pathIs(path, l, "tail_bias"))    p.tail_bias = state::patchFloat(i);
    else if (state::pathIs(path, l, "base"))         p.base = state::patchFloat(i);
    else if (state::pathIs(path, l, "glint_ratio"))  p.glint_ratio = state::patchFloat(i);
    else if (state::pathIs(path, l, "glint_chaos"))  p.glint_chaos = state::patchFloat(i);
    else if (state::pathIs(path, l, "glint_gain"))   p.glint_gain = state::patchFloat(i);
    else if (state::pathIs(path, l, "glint_shadow")) p.glint_shadow = state::patchFloat(i);
    else if (state::pathIs(path, l, "glint_linger")) p.glint_linger = state::patchFloat(i);
    else if (state::pathIs(path, l, "tail_chunk"))   p.tail_chunk = state::patchFloat(i);
    else if (state::pathIs(path, l, "glint_width"))  p.glint_width = state::patchFloat(i);
    else if (state::pathIs(path, l, "glint_sling"))  p.glint_sling = state::patchFloat(i);
    else if (state::pathIs(path, l, "color")) {
      auto v = state::patchVec3(i);
      s->color[0] = v.x; s->color[1] = v.y; s->color[2] = v.z;
    }
    else if (state::pathIs(path, l, "intensity"))    s->intensity = state::patchFloat(i);
    else if (state::pathIs(path, l, "whiten"))       s->whiten = state::patchFloat(i);
    else if (state::pathIs(path, l, "quantize"))     s->quantize = state::patchBool(i);
  }
}

void render(void* self, int vp_w, int vp_h) {
  auto* s = static_cast<State*>(self);
  if (!s || !s->initialized || vp_w <= 0 || vp_h <= 0) return;
  auto out = gpu::Device::textureForField("tex_out");
  if (!out.valid()) return;

  // A ZERO-dt tick: ages nothing — every clock, blink, spring and the mask's
  // roll are gated on dt > 0 — but it seeds a card rendered before its first
  // tick (thumbnails, previews) at the knob's own level, settled.
  s->core.tick(s->p, 0.0f);
  publishRails(*s);

  const float quant = s->quantize ? 1.0f : 0.0f;
  auto draw = [&](gpu::Buffer& buf, const gpu::Texture& tex, const led_bars::Cells& cells) {
    LedUniforms u = {};
    for (int i = 0; i < led_bars::kCells; i++) {
      u.cell[i][0] = cells.rgb[i][0];
      u.cell[i][1] = cells.rgb[i][1];
      u.cell[i][2] = cells.rgb[i][2];
      u.cell[i][3] = 1.0f;
    }
    u.ctl[0] = quant;
    buf.writeOne(u);
    auto cp = gpu::ComputePass::begin();
    cp.setPSO(s_pso);
    cp.setTexture(tex, 0, 1);
    cp.setBuffer(buf, 1);
    cp.dispatch((vp_w + 7) / 8, (vp_h + 7) / 8);
    cp.end();
  };

  draw(s->led_buf, out,
       sweep_choke_core::toCells(s->core, s->color, s->intensity, s->whiten));

  // The mask, only while something is listening to it.
  if (state::isOutputConnected("mask_out")) {
    if (!s->mask_tex.valid() || s->mask_w != vp_w || s->mask_h != vp_h) {
      s->mask_tex.release();
      s->mask_tex = gpu::Device::createTexture(vp_w, vp_h);
      s->mask_w = vp_w;
      s->mask_h = vp_h;
      if (s->mask_tex.valid()) state::setGpuTexture("mask_out", s->mask_tex.id);
    }
    if (s->mask_tex.valid()) draw(s->mask_buf, s->mask_tex, sweep_choke_core::maskCells(s->core));
  }

  gpu::Device::submit();
}

}  // namespace sweep_choke

namespace sweep_choke {

// Effect store preview: an LFO swinging the sweep so the choke and glints
// fire.
void preview(nano::PreviewScenario& s) {
  s.input("black")
      .aux("lfo", "mod.source.lfo")
      .auxParam("lfo", "rate", 0.6f)
      .wire("lfo.output", "$self.sweep")
      .capture(1.6f);
}

}  // namespace sweep_choke
