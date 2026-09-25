/*
 * source.light.sweep_gradient — Sweep Choke's rolling gradient on its own,
 * played by the knob or fired by a trigger.
 *
 * The knob sets one level, full in the middle and dimming past a deadzone to
 * black at the ends; every change to it rolls across the four LED bars at a set
 * angle. A trigger plays the same thing as a gesture — out toward an end and
 * back on an attack / hold / release envelope — and while it runs it owns the
 * sweep: monophonic, the knob takes back over when it ends. All of the
 * behaviour lives in <sketch/sweep_gradient_core.h>, host-free so the goldens
 * can drive it at an exact dt; this file is the schema, the patch plumbing and
 * one small pass.
 *
 * One output, the house rig's pixel map — four bars, ten segments, flat blocks
 * — drawn by the shared nano_led_bars.hlsl pass: the colour at each segment's
 * level. It is a control signal, so it is opaque and ignores the input.
 */

#include <gpu.h>
#include <host.h>
#include <val.h>
#include <effect_auto_trigger.h>
#include <sketch/sweep_gradient_core.h>
#include <sketch/led_bars.h>
#include "sweep_gradient_shaders.h"

#include <cstdint>

namespace sweep_gradient {

// Mirrors `cbuffer LedUniforms` in shaders/nano_led_bars.hlsl.
struct LedUniforms {
  float cell[led_bars::kCells][4];   // bar-major, segment 0 at the bottom
  float ctl[4];                      // quantize, -, -, -
};
static_assert(sizeof(LedUniforms) == 16 * (led_bars::kCells + 1),
              "LedUniforms layout mismatch with nano_led_bars.hlsl");

struct State {
  sweep_gradient_core::Params p;
  sweep_gradient_core::Core core;
  float color[3] = {1.00f, 0.22f, 0.62f};
  float intensity = 1.0f;
  bool quantize = true;

  fx::AutoTrigger auto_trig;
  bool gate_prev = false;
  float trigger_prev = 0.0f;

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
  publish("level_out", sweep_gradient_core::clamp01(s.core.mean_level));
  publish("env_out", s.core.env);
  publish("sweep_out", s.core.sweep_out);
}

void eval_visibility(int n, const char* pb, const int* off, const int* len, const int* ops) {
  fx::AutoTrigger::evalVisibility(n, pb, off, len, ops);
}

static void on_state_ready(void* self) {
  auto* s = static_cast<State*>(self);
  if (!s) return;
  fx::AutoTrigger::applyVisibility(s->auto_trig.mode, s->auto_trig.div);
}

void module_init() {
  state::init("source.light.sweep_gradient", {1, 0, 0},
    fx::AutoTrigger::fields(
    state::Schema()
      .helpField("intro",
        "## Sweep Gradient\n"
        "Sweep Choke's rolling gradient on its own: the four LED bars in one "
        "colour, dimmed by a knob **or** by a trigger.\n\n"
        "**The knob.** Parked in the middle the bars are full; out past the "
        "deadzone they dim toward black at either end. Park anywhere and they "
        "settle solid at that level. Only a CHANGE is a gradient: it rolls "
        "across the grid along **Angle**, one side first, and passes.\n\n"
        "**The trigger** plays the same thing as a gesture — out toward an "
        "end over **Attack**, held, and home over **Release** — so it goes "
        "dark rolling one way and comes back rolling the same way. While it "
        "runs it owns the bars and the knob waits; fire again mid-gesture and "
        "it picks up from where it is.\n\n"
        "**Try:** *Auto Mode* Beats at 1/4, *Side* Alternate, *Roll* 0.3 — a "
        "choke that washes up the bars and back down on every beat.")

      // ---------------- Sweep ----------------
      .group("sweep", "Sweep")
        .groupHelp(
          "**Sweep** is the knob. **Deadzone** is how much of each half stays "
          "fully lit. **Depth** is how far the ends go — below 1 they never "
          "reach black. A trigger's gesture goes through these too.")
      .floatField("sweep", 0.5f, 0.f, 1.f, state::PrimaryInput,
                  nullptr, 0.f, nullptr,
                  "The knob. 0.5 is home: full colour. Ignored while a "
                  "triggered gesture runs.")
        .label("Sweep", "Sweep")
      .floatField("deadzone", 0.45f, 0.f, 0.95f, state::PrimaryInput,
                  nullptr, 0.f, nullptr,
                  "Fraction of each half of the knob that stays fully lit.")
        .label("Deadzone", "DZ")
      .floatField("depth", 1.0f, 0.f, 1.f, state::PrimaryInput,
                  nullptr, 0.f, nullptr,
                  "How far the ends fade. 1 chokes to black.")
        .label("Depth", "Depth")

      // ---------------- Trigger ----------------
      .group("trigger", "Trigger")
        .groupHelp(
          "A trigger sweeps for you: out from home toward one end, **Reach** "
          "of the way, over **Attack**; held for **Hold**; back over "
          "**Release**. **Gate** and **Trigger** both fire on a rising edge, "
          "and *Auto Mode* can fire it on its own. One gesture at a time: a "
          "new fire restarts it from wherever it is, and the knob takes back "
          "over when it ends.\n\n"
          "**Side** picks the end — with *Sides* Ping-Pong, the lower end "
          "rolls the other way, so **Alternate** sends every other hit back "
          "down the bars.")
      .boolField("gate", false, state::PrimaryInput).label("Gate", "Gate")
      .eventField("trigger", state::PrimaryInput).label("Trigger", "Trig")
    )   // ← fx::AutoTrigger::fields: auto_mode + auto_rate + auto_beats + custom
      .selectField("trig_side", sweep_gradient_core::TrigAlternate, state::PrimaryInput,
                   {{"Upper", sweep_gradient_core::TrigUpper},
                    {"Lower", sweep_gradient_core::TrigLower},
                    {"Alternate", sweep_gradient_core::TrigAlternate},
                    {"Random", sweep_gradient_core::TrigRandom}}, false,
                   "Which end of the knob a gesture sweeps toward.")
        .label("Side", "Side")
      .floatField("reach", 1.0f, 0.f, 1.f, state::PrimaryInput,
                  nullptr, 0.f, nullptr,
                  "How far toward the end a gesture goes. 1 is the end itself; "
                  "at or under the deadzone it never dims.")
        .label("Reach", "Reach")
      .floatField("attack", 0.08f, 0.f, 4.f, state::PrimaryInput,
                  nullptr, 0.f, "s", "Seconds out to Reach.")
        .label("Attack", "Atk")
      .floatField("hold", 0.15f, 0.f, 4.f, state::PrimaryInput,
                  nullptr, 0.f, "s", "Seconds held there.")
        .label("Hold", "Hold")
      .floatField("release", 0.35f, 0.f, 4.f, state::PrimaryInput,
                  nullptr, 0.f, "s", "Seconds back home.")
        .label("Release", "Rel")

      // ---------------- Gradient ----------------
      .group("gradient", "Gradient")
        .groupHelp(
          "How the bars get to where the sweep asks. They always settle to one "
          "solid level; a change **rolls** across the grid to get there, so "
          "one side moves before the other and the gradient is the change "
          "passing through.\n\n"
          "**Angle** is the way it rolls, counter-clockwise from "
          "left-to-right: 90 climbs the bars, 0 runs left to right — the same "
          "way whether it is getting darker or lighter. **Sides**: "
          "**Ping-Pong** turns it round on the knob's lower half; **One-Way** "
          "keeps it. **Roll** is how long a change takes to cross the grid — "
          "0 changes it all at once. **Soften** eases each segment on, so a "
          "jump rolls through as a smooth gradient rather than a hard edge.")
      .floatField("angle", 90.f, 0.f, 360.f, state::PrimaryInput,
                  nullptr, 0.f, nullptr,
                  "Direction changes roll, degrees CCW from +x.")
        .label("Angle", "Ang")
      .selectField("side_mode", sweep_gradient_core::SidePingPong, state::PrimaryInput,
                   {{"Ping-Pong", sweep_gradient_core::SidePingPong},
                    {"One-Way", sweep_gradient_core::SideOneWay}}, false,
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
      .rgbField("color", 1.00f, 0.22f, 0.62f, state::PrimaryInput)
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
      .floatField("level_out", 1.f, 0.f, 1.f, state::PrimaryOutput, "unsigned",
                  0.f, nullptr, "Mean level across the bars.")
        .label("Level", "Lvl")
      .floatField("env_out", 0.f, 0.f, 1.f, state::PrimaryOutput, "unsigned",
                  0.f, nullptr, "The trigger's envelope, 0..1.")
        .label("Envelope", "Env")
      .floatField("sweep_out", 0.5f, 0.f, 1.f, state::PrimaryOutput, "unsigned",
                  0.f, nullptr, "The sweep in effect — the knob, or the "
                  "gesture while one runs. Wire it to another Sweep.")
        .label("Sweep Out", "SwOut")
      .textureField("tex_in",  state::PrimaryInput)
      .textureField("tex_out", state::PrimaryOutput)

      .capability(state::Capability::Generator)
      .capability(state::Capability::ModulationSource)
      .capability(state::Capability::ModulationSourceMulti)
  );
  state::setOnStateReady(&on_state_ready);

  if (gpu::Device::backend() == gpu::Backend::None) return;

  state::registerShaderSPV("sweep_gradient_led", LED_SPV, LED_SPV_SIZE);
  if (auto led = gpu::Device::createShaderModuleByName("sweep_gradient_led")) {
    s_pso = gpu::Device::createComputePSO(led, "main", gpu::Bindings()
        .storageTex2d(0)
        .uniform(1));
  }
  state::log("sweep_gradient: module initialized");
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
  s->gate_prev = false;
  s->trigger_prev = 0.0f;
  s->initialized = false;
  if (!s_pso.valid() || !s->led_buf.valid()) return;
  s->initialized = true;
}

// The core advances here and ONLY here: render() may run without a tick
// (thumbnails, off-playhead previews), and stepping from there would play the
// gesture every time somebody looked at the card.
void tick(void* self, double dt) {
  auto* s = static_cast<State*>(self);
  if (!s) return;
  for (int i = 0, nf = s->auto_trig.fires(dt); i < nf; i++) s->core.fire();
  s->core.tick(s->p, (float)dt);
  publishRails(*s);
}

void on_state_patched(void* self, int n, const char* pb, const int* off,
                      const int* len, const int* ops) {
  auto* s = static_cast<State*>(self);
  if (!s) return;
  auto& p = s->p;
  bool vis_changed = false;
  for (int i = 0; i < n; i++) {
    if (ops[i] != state::PatchReplace) continue;
    const char* path = pb + off[i];
    const int l = len[i];
    if (s->auto_trig.patch(path, l, i, &vis_changed)) continue;
    // Both fire on a rising edge only: every value is replayed every frame.
    if (state::pathIs(path, l, "gate")) {
      const bool g = state::patchFloat(i) != 0.0f;
      if (g && !s->gate_prev) s->core.fire();
      s->gate_prev = g;
    }
    else if (state::pathIs(path, l, "trigger")) {
      const float v = state::patchFloat(i);
      if (v != 0.0f && s->trigger_prev == 0.0f) s->core.fire();
      s->trigger_prev = v;
    }
    else if (state::pathIs(path, l, "sweep"))        p.sweep = state::patchFloat(i);
    else if (state::pathIs(path, l, "deadzone"))     p.deadzone = state::patchFloat(i);
    else if (state::pathIs(path, l, "depth"))        p.depth = state::patchFloat(i);
    else if (state::pathIs(path, l, "trig_side"))    p.trig_side = state::patchInt(i);
    else if (state::pathIs(path, l, "reach"))        p.reach = state::patchFloat(i);
    else if (state::pathIs(path, l, "attack"))       p.attack = state::patchFloat(i);
    else if (state::pathIs(path, l, "hold"))         p.hold = state::patchFloat(i);
    else if (state::pathIs(path, l, "release"))      p.release = state::patchFloat(i);
    else if (state::pathIs(path, l, "angle"))        p.angle_deg = state::patchFloat(i);
    else if (state::pathIs(path, l, "side_mode"))    p.side_mode = state::patchInt(i);
    else if (state::pathIs(path, l, "roll"))         p.roll = state::patchFloat(i);
    else if (state::pathIs(path, l, "soften"))       p.soften = state::patchFloat(i);
    else if (state::pathIs(path, l, "color")) {
      auto v = state::patchVec3(i);
      s->color[0] = v.x; s->color[1] = v.y; s->color[2] = v.z;
    }
    else if (state::pathIs(path, l, "intensity"))    s->intensity = state::patchFloat(i);
    else if (state::pathIs(path, l, "quantize"))     s->quantize = state::patchBool(i);
  }
  if (vis_changed)
    fx::AutoTrigger::applyVisibility(s->auto_trig.mode, s->auto_trig.div);
}

void render(void* self, int vp_w, int vp_h) {
  auto* s = static_cast<State*>(self);
  if (!s || !s->initialized || vp_w <= 0 || vp_h <= 0) return;
  auto out = gpu::Device::textureForField("tex_out");
  if (!out.valid()) return;

  // A ZERO-dt tick: ages nothing, but seeds a card rendered before its first
  // tick (thumbnails, previews) at the knob's own level, settled.
  s->core.tick(s->p, 0.0f);
  publishRails(*s);

  const auto cells = sweep_gradient_core::toCells(s->core, s->color, s->intensity);
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

}  // namespace sweep_gradient
