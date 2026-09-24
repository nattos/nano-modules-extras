/*
 * motion.peak_decay — "Peak Decay".
 *
 * A per-pixel peak meter: any pixel whose colour sits still for longer than
 * `hold` starts to fall — its luma eases down a smoothstep sigmoid over
 * `fall` seconds, dimming toward black by `amount`. The instant the pixel
 * moves again it snaps back to full brightness (classic meter ballistics:
 * instant up, smooth down). Change metering is full-RGB balanced against
 * luma by `rgb_balance`, and the comparison is against the HELD reference
 * (latched when the pixel last moved), so slow drifts accumulate and
 * eventually trip the threshold too. The `catch` select switches to Rise
 * Only: only an upward luma edge (the live luma rising past the held
 * reference) resets the fall — the reference follows the input down
 * silently, so darker or chroma-only changes can't keep pixels awake and a
 * dip-then-recovery measures its rise from the bottom of the dip.
 *
 * Two compute passes over an RGBA16F state ping-pong (rgb = held reference,
 * a = age in seconds): meter (update state) then apply (gain the input).
 * Split because the storage-format hint is per-shader — the state is
 * rgba16f while tex_out is the sketch working format.
 *
 * Stateful per-pixel history → no temporal capability. Deliberately NO
 * is_identity (see burn_out): the executor permanently sidelines an
 * identity stage, and the meter must keep running at amount 0 so the ages
 * are honest when amount comes back up.
 */

#include <gpu.h>
#include <host.h>
#include "peak_decay_shaders.h"

#include <cmath>
#include <cstdint>

namespace peak_decay {

// Layout MUST match cbuffer Uniforms in meter.hlsl / apply.hlsl.
struct Uniforms {
  float dt;
  float amount;
  float hold;
  float fall;
  float threshold;
  float reset;
  float rgb_balance;
  float catch_mode;
};
static_assert(sizeof(Uniforms) == 32, "Uniforms layout mismatch");

struct State {
  gpu::Buffer uniform_buf;
  gpu::Texture state_tex[2];   // RGBA16F ping-pong: rgb = held ref, a = age
  int st_w = 0, st_h = 0;
  int ping = 0;                // index the NEXT meter pass reads from
  bool need_seed = true;       // first frame / resize: latch input, age 0
  bool initialized = false;

  // Schema-mirrored params.
  float amount      = 1.0f;
  float hold        = 0.5f;   // seconds
  float fall        = 1.0f;   // seconds
  float threshold   = 0.05f;
  float rgb_balance = 0.5f;   // 0 = luma-only metering, 1 = full RGB
  int   catch_mode  = 0;      // 0 = Any Change, 1 = Rise Only

  // This frame's dt (tick → render).
  float frame_dt = 0.0f;
};

// Type-shared PSOs: compiled once in module_init(), reused by every instance.
static gpu::ComputePSO s_pso_meter;
static gpu::ComputePSO s_pso_apply;

void module_init() {
  state::init("motion.peak_decay", {1, 0, 1},
    state::Schema()
      .helpField("intro",
        "## Peak Decay\n"
        "A per-pixel peak meter. Any pixel that holds still for longer than "
        "*Hold* starts to fall: its brightness eases down a smooth sigmoid "
        "over *Fall*, dimming toward black. The instant the pixel changes it "
        "snaps back to full brightness — moving material stays vivid while "
        "static regions sink away.\n\n"
        "**Try:** a short *Hold* and a long *Fall* over mostly-still footage "
        "so only the motion survives; raise *Threshold* until noisy video "
        "reads as static and whole regions breathe down between hits; or "
        "switch *Catch* to Rise Only so only rising luma wakes a pixel — "
        "fades and darkening drift keep falling undisturbed.")
      .group("decay", "Decay")
        .groupHelp(
          "The envelope every stale pixel rides: full brightness through "
          "*Hold*, then a sigmoid fall over *Fall*, down by *Amount*. Any "
          "change resets a pixel to full brightness instantly.")
      .floatField("amount", 1.0f, 0.0f, 1.0f, state::PrimaryInput, nullptr, 0.01f,
                  nullptr, "How far a stale pixel falls (1 = all the way to black).")
          .label("Amount", "Amt")
      .floatField("hold", 0.5f, 0.0f, 4.0f, state::PrimaryInput, nullptr, 0.01f,
                  "s", "How long a pixel may sit unchanged before the fall starts.")
          .label("Hold", "Hold")
      .floatField("fall", 1.0f, 0.0f, 5.0f, state::PrimaryInput, nullptr, 0.01f,
                  "s", "Duration of the sigmoid fall (0 = a hard cliff after the hold).")
          .label("Fall", "Fall")
      .group("meter", "Metering")
        .groupHelp(
          "What counts as \"the pixel changed\". Metering compares against "
          "the value latched when the pixel last moved — so slow drifts "
          "eventually count too. *Catch* picks the ballistics: Any Change "
          "resets on any sufficient move (balanced luma↔RGB by *RGB "
          "Balance*); Rise Only resets only on an upward luma edge — the "
          "reference follows the input down silently, so fades and darkening "
          "can't keep pixels awake.")
      .floatField("threshold", 0.05f, 0.0f, 0.5f, state::PrimaryInput, nullptr, 0.01f,
                  nullptr, "Minimum change that counts as motion (in Rise Only: "
                           "the margin the input must rise above the decaying "
                           "peak). Raise it so sensor noise or dithering "
                           "doesn't hold pixels awake.")
          .label("Threshold", "Thr")
      .floatField("rgb_balance", 0.5f, 0.0f, 1.0f, state::PrimaryInput, nullptr, 0.01f,
                  nullptr, "Luma↔RGB balance of the change meter: 0 hears only "
                           "luma steps, 1 hears any channel move at full "
                           "weight. (Rise Only catches on luma by nature.)")
          .label("RGB Balance", "RGB")
      .selectField("catch", 0, state::PrimaryInput,
                   {{"Any Change", 0}, {"Rise Only", 1}}, false,
                   "Any Change: any sufficient move resets a pixel to full "
                   "brightness. Rise Only: only an upward luma edge resets — "
                   "darker or chroma-only changes let the fall continue.")
          .label("Catch", "Ctch")
      .textureField("tex_in",  state::PrimaryInput)
      .textureField("tex_out", state::PrimaryOutput));

  if (gpu::Device::backend() == gpu::Backend::None) return;

  state::registerShaderSPV("peak_decay_meter", METER_SPV, METER_SPV_SIZE,
                           "rgba16float", "write");
  state::registerShaderSPV("peak_decay_apply", APPLY_SPV, APPLY_SPV_SIZE);
  auto cs_m = gpu::Device::createShaderModuleByName("peak_decay_meter");
  auto cs_a = gpu::Device::createShaderModuleByName("peak_decay_apply");
  if (!cs_m || !cs_a) return;
  s_pso_meter = gpu::Device::createComputePSO(cs_m, "main", gpu::Bindings()
      .tex2d(0).tex2d(1).storageTex2d(2, gpu::TextureFormat::RGBA16F).uniform(3));
  s_pso_apply = gpu::Device::createComputePSO(cs_a, "main", gpu::Bindings()
      .tex2d(0).tex2d(1).storageTex2d(2).uniform(3));

  state::log("peak_decay: module initialized");
}

void* create() {
  auto* s = new State();
  s->uniform_buf = gpu::Device::createBuffer(sizeof(Uniforms), gpu::BufferUsage::Uniform);
  return s;
}

void destroy(void* self) {
  auto* s = static_cast<State*>(self);
  if (!s) return;
  s->uniform_buf.release();
  for (auto& t : s->state_tex) if (t.valid()) t.release();
  delete s;
}

void init(void* self) {
  auto* s = static_cast<State*>(self);
  if (!s) return;
  s->need_seed = true;
  if (!s_pso_meter.valid() || !s_pso_apply.valid() || !s->uniform_buf.valid()) return;
  s->initialized = true;
}

void tick(void* self, double dt) {
  auto* s = static_cast<State*>(self);
  if (!s) return;
  // Clamp hitches: a transport stall shouldn't age the whole frame past the
  // hold in one step and slam everything dark on resume.
  if (dt < 0.0) dt = 0.0;
  if (dt > 0.25) dt = 0.25;
  s->frame_dt = (float)dt;
}

void on_state_patched(void* self, int n, const char* pb, const int* off,
                      const int* len, const int* ops) {
  auto* s = static_cast<State*>(self);
  if (!s) return;
  for (int i = 0; i < n; i++) {
    if (ops[i] != state::PatchReplace) continue;
    const char* p = pb + off[i];
    int l = len[i];
    if      (state::pathIs(p, l, "amount"))    s->amount    = state::patchFloat(i);
    else if (state::pathIs(p, l, "hold"))      s->hold      = state::patchFloat(i);
    else if (state::pathIs(p, l, "fall"))      s->fall      = state::patchFloat(i);
    else if (state::pathIs(p, l, "threshold")) s->threshold = state::patchFloat(i);
    else if (state::pathIs(p, l, "rgb_balance")) s->rgb_balance = state::patchFloat(i);
    else if (state::pathIs(p, l, "catch"))     s->catch_mode = state::patchInt(i);
  }
}

void on_resolume_param(void*, long long, double) {}

static void ensureState(State* s, int w, int h) {
  if (s->state_tex[0].valid() && s->st_w == w && s->st_h == h) return;
  for (auto& t : s->state_tex) if (t.valid()) t.release();
  s->state_tex[0] = gpu::Device::createTexture(w, h, gpu::TextureFormat::RGBA16F);
  s->state_tex[1] = gpu::Device::createTexture(w, h, gpu::TextureFormat::RGBA16F);
  s->st_w = w; s->st_h = h;
  s->need_seed = true;
}

void render(void* self, int vp_w, int vp_h) {
  auto* s = static_cast<State*>(self);
  if (!s || !s->initialized || vp_w <= 0 || vp_h <= 0) return;
  auto in  = gpu::Device::textureForField("tex_in");
  auto out = gpu::Device::textureForField("tex_out");
  if (!in.valid() || !out.valid()) return;
  ensureState(s, vp_w, vp_h);
  if (!s->state_tex[0].valid() || !s->state_tex[1].valid()) return;

  Uniforms u = {};
  u.dt        = s->frame_dt;
  u.amount    = s->amount;
  u.hold      = s->hold;
  u.fall      = s->fall;
  u.threshold = s->threshold;
  u.reset     = s->need_seed ? 1.0f : 0.0f;
  u.rgb_balance = s->rgb_balance;
  u.catch_mode  = (float)s->catch_mode;
  s->uniform_buf.writeOne(u);

  const int prev = s->ping, next = 1 - s->ping;
  const int gx = (vp_w + 7) / 8, gy = (vp_h + 7) / 8;

  {   // Meter: advance ages / re-latch references into state[next].
    auto cp = gpu::ComputePass::begin();
    cp.setPSO(s_pso_meter);
    cp.setTexture(in, 0, 0);
    cp.setTexture(s->state_tex[prev], 1, 0);
    cp.setTexture(s->state_tex[next], 2, 1);
    cp.setBuffer(s->uniform_buf, 3);
    cp.dispatch(gx, gy);
    cp.end();
  }
  {   // Apply: gain the input by this frame's ages.
    auto cp = gpu::ComputePass::begin();
    cp.setPSO(s_pso_apply);
    cp.setTexture(in, 0, 0);
    cp.setTexture(s->state_tex[next], 1, 0);
    cp.setTexture(out, 2, 1);
    cp.setBuffer(s->uniform_buf, 3);
    cp.dispatch(gx, gy);
    cp.end();
  }

  s->ping = next;
  s->need_seed = false;
  gpu::Device::submit();
}

} // namespace peak_decay
