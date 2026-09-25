/*
 * source.light.sweep_choke — one knob that sweeps and chokes the four LED bars.
 *
 * Three Planes Rig's sweep, taken off the tower and put straight onto the rig:
 * the centre is a solid colour at full, moving out tilts the fade across the
 * grid at a set angle, the ends choke to black, and a hard sweep into a choke
 * leaves a flickering tail ringing behind it. All of the behaviour lives in
 * <sketch/sweep_choke_core.h>, host-free so the goldens can drive it at an exact
 * dt; this file is the schema, the patch plumbing and one small pass.
 *
 * The output IS the house rig's pixel map — four bars, ten segments, flat
 * blocks — drawn by the shared nano_led_bars.hlsl pass straight onto tex_out.
 * It is a control signal, so it is opaque and ignores the input.
 */

#include <gpu.h>
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
        "Parked in the middle, the bars are one **solid colour at full**. Move "
        "out past the deadzone and the fade does not dim evenly: a soft front "
        "travels across the grid at **Angle**, so one side goes out before the "
        "other and the light tilts. Keep going and the tilt slides off into "
        "black at either end — the **choke**.\n\n"
        "**How you swept is what you hear afterwards.** A hard pass through "
        "the middle charges; reaching a choke throws that charge as a "
        "flickering tail that rings out on its own clock, gathered where the "
        "light went out last. Sweep straight back and the grid relights over a "
        "tail still running. Come back in fast and the light overshoots and "
        "springs back.\n\n"
        "**Try:** *Angle* 90 and Ping-Pong, then rock the knob to the beat — "
        "the dark rises from the floor one way and falls from the top the "
        "other. Crank *Tilt* for a hard edge, drop it to 0 for a plain dimmer.")

      // ---------------- Sweep ----------------
      .group("sweep", "Sweep")
        .groupHelp(
          "**Sweep** is the knob; the rest shape what it does. **Deadzone** is "
          "how much of each half stays fully lit — ride inside it without "
          "touching the look. **Depth** is how far the ends go; below full the "
          "ends never choke, so nothing is thrown. **Sensitivity** and "
          "**Motion Decay** set how the knob's speed is read, which is what "
          "charges a throw.")
      .floatField("sweep", 0.5f, 0.f, 1.f, state::PrimaryInput,
                  nullptr, 0.f, nullptr,
                  "The knob. 0.5 is home: full colour, no motion.")
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
          "The shape of the fade. **Angle** is the way the dark travels across "
          "the grid, counter-clockwise from left-to-right: 90 rises from the "
          "floor, 0 sweeps left to right. **Sides** decides the other half of "
          "the knob — **Ping-Pong** mirrors it, so which way you turned shows "
          "in the light; **Forward** keeps one direction. **Tilt** is how much "
          "of a front it is: 0 dims every segment together, 1 is a narrow "
          "edge.")
      .floatField("angle", 90.f, 0.f, 360.f, state::PrimaryInput,
                  nullptr, 0.f, nullptr,
                  "Direction the dark travels, degrees CCW from +x.")
        .label("Angle", "Ang")
      .selectField("side_mode", sweep_choke_core::SidePingPong, state::PrimaryInput,
                   {{"Ping-Pong", sweep_choke_core::SidePingPong},
                    {"Forward", sweep_choke_core::SideForward}}, false,
                   "Ping-Pong mirrors the angle for the lower half of the "
                   "knob; Forward uses the same angle both ways.")
        .label("Sides", "Sides")
      .floatField("tilt", 0.6f, 0.f, 1.f, state::PrimaryInput,
                  nullptr, 0.f, nullptr,
                  "0 an even dimmer, 1 a narrow travelling edge.")
        .label("Tilt", "Tilt")

      // ---------------- Light ----------------
      .group("light", "Light")
      .rgbField("color", 1.00f, 0.22f, 0.62f, state::PrimaryInput)
        .label("Colour", "Col")
      .floatField("intensity", 1.0f, 0.f, 2.f, state::PrimaryInput,
                  nullptr, 0.f, nullptr, "Overall output level.")
        .label("Intensity", "Int")
      .floatField("bounce", 0.55f, 0.f, 1.f, state::PrimaryInput,
                  nullptr, 0.f, nullptr,
                  "Coming back in fast, the light runs ahead of your hand, past "
                  "full, and springs back. How far. Inward only.")
        .label("Bounce", "Bnce")
      .floatField("whiten", 0.6f, 0.f, 1.f, state::SecondaryInput,
                  nullptr, 0.f, nullptr,
                  "How far the bounce's overshoot pushes toward white.")
        .label("Overdrive White", "White")

      // ---------------- Tail ----------------
      .group("tail", "Tail")
        .groupHelp(
          "The reverberant tail. A pass through the middle charges by how HARD "
          "it was (**Charge Drive**), and the charge leaks away over **Charge "
          "Hold** — so a flick then a dawdle throws a soft one. Reaching a "
          "choke throws it, and it rings out over **Ring Time** as segments "
          "blinking where the light went out last (**Bias**).\n\n"
          "The tail never dims. Blinks keep arriving at **Blink Rate**; what "
          "runs down is how LONG each one is (**Blink Length**), until they "
          "start missing frames and it sputters out. **Density** is segments "
          "per blink.\n\n"
          "Separately, any motion inside the fade wakes a **Stutter** in the "
          "half-lit segments that **Settle** lets die away once the knob "
          "stops.")
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
                  "0 blinks anywhere; 1 only where the light went out last.")
        .label("Bias", "Bias")
      .floatField("flicker", 0.6f, 0.f, 1.f, state::PrimaryInput,
                  nullptr, 0.f, nullptr,
                  "How hard half-lit segments stutter while the knob moves.")
        .label("Stutter", "Stut")
      .floatField("settle", 1.2f, 0.05f, 6.f, state::SecondaryInput,
                  nullptr, 0.f, nullptr,
                  "Seconds the stutter carries on after the knob stops.")
        .label("Settle", "Settle")

      // ---------------- Output ----------------
      .group("output", "Output")
      .boolField("quantize", true, state::SecondaryInput,
                 "Each segment a flat block, so a pixel mapper lands on its "
                 "exact colour. Off interpolates between segment centres.")
        .label("Quantize", "Quant")
      .floatField("speed_out", 0.f, 0.f, 1.f, state::PrimaryOutput, "unsigned",
                  0.f, nullptr, "How hard the knob is moving, 0..1.")
        .label("Speed", "Spd")
      .floatField("release_out", 0.f, 0.f, 1.f, state::PrimaryOutput, "unsigned",
                  0.f, nullptr, "The thrown tail: its charge at the choke, "
                  "falling to 0 across the ring-out.")
        .label("Release", "Rel")
      .floatField("level_out", 1.f, 0.f, 1.f, state::PrimaryOutput, "unsigned",
                  0.f, nullptr, "Mean segment level.")
        .label("Level", "Lvl")
      .floatField("sweep_out", 0.5f, 0.f, 1.f, state::PrimaryOutput, "unsigned",
                  0.f, nullptr, "The knob, passed through — wire it to Three "
                  "Planes' Glint Sweep.")
        .label("Sweep Out", "SwOut")
      .textureField("tex_in",  state::PrimaryInput)
      .textureField("tex_out", state::PrimaryOutput)

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
  s->initialized = false;
  if (!s_pso.valid() || !s->led_buf.valid()) return;
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
    else if (state::pathIs(path, l, "tilt"))         p.tilt = state::patchFloat(i);
    else if (state::pathIs(path, l, "bounce"))       p.bounce = state::patchFloat(i);
    else if (state::pathIs(path, l, "latch_drive"))  p.latch_drive = state::patchFloat(i);
    else if (state::pathIs(path, l, "latch_decay"))  p.latch_decay = state::patchFloat(i);
    else if (state::pathIs(path, l, "ring_time"))    p.ring_time = state::patchFloat(i);
    else if (state::pathIs(path, l, "tail_rate"))    p.tail_rate = state::patchFloat(i);
    else if (state::pathIs(path, l, "tail_duty"))    p.tail_duty = state::patchFloat(i);
    else if (state::pathIs(path, l, "tail_density")) p.tail_density = state::patchFloat(i);
    else if (state::pathIs(path, l, "tail_bias"))    p.tail_bias = state::patchFloat(i);
    else if (state::pathIs(path, l, "flicker"))      p.flicker = state::patchFloat(i);
    else if (state::pathIs(path, l, "settle"))       p.settle = state::patchFloat(i);
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

  // A ZERO-dt tick: re-reads where the knob is now without ageing anything —
  // every clock, blink and spring in the core is gated on dt > 0. So a host
  // that renders without ticking (thumbnails, previews) still sees the right
  // position rather than whatever the last tick left.
  s->core.tick(s->p, 0.0f);
  publishRails(*s);

  const led_bars::Cells cells =
      sweep_choke_core::toCells(s->core, s->color, s->intensity, s->whiten);
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

}  // namespace sweep_choke
