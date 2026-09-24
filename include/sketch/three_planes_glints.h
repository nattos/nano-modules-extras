#pragma once
/*
 * three_planes_glints.h — the glints `source.mesh.three_planes` flashes across
 * its quads: the glare off metal in an old cel-animated show.
 *
 * THE GLINT IS THE GESTURE. This does not watch a level and decide when to
 * sparkle; it watches the sweep knob itself, and the invariant it is built
 * around is:
 *
 *     move the knob at a constant speed across its whole range, and ONE glint
 *     crosses the planes at exactly that rate.
 *
 * Everything else falls out of that. The launch is the moment the knob ENTERS
 * the middle band, which is guaranteed to happen once per traverse and is
 * guaranteed to happen while the knob is moving — so there is always a real
 * speed for the glint to take. `ratio` is the exchange rate, and at its
 * default of 1 the sentence above is literal: one knob range, one crossing.
 *
 * SIGN IS THROWN AWAY. Only how fast the knob is moving matters, never which
 * way. Reverse the knob mid-gesture and the glints already out there carry on
 * exactly as they were — they are objects in flight, not a readout.
 *
 * A THROW DOES TWO THINGS TO THEM, and they are separable. When the sweep
 * reaches a mute the rig spends whatever it had latched, and that release
 *
 *   HOLDS them at full strength while it rings out — reaching a mute means
 *   letting go of the knob, so without this they would start running down at
 *   the exact moment they are being thrown. Always on;
 *
 *   SLINGS them along, four times their own speed at a full throw, so they
 *   leave with everything else. That one is `sling`, and it is a choice: it
 *   belongs to a throw that is energy LEAVING the frame. A throw that keeps
 *   everything inside it wants the opposite — glints that hang around and
 *   drift while the rest of it rattles — and passes 0.
 *
 * Above a brisk sweep, small extra glints start arriving on their own to
 * scuff up the result. Those ARE random (exponentially spaced, so they read as
 * independent events rather than as a metronome), but they are deliberately
 * dimmer and narrower than the launched one: the gesture stays legible, and
 * the chaos sits underneath it.
 *
 * WHAT A LIVE GLINT KEEPS. Brightness, width and wake are drawn once at birth
 * and belong to it. The only thing the knob still reaches is SPEED, which is
 * shared by every glint at once — and it has to be shared, because at their
 * own speeds two of them would eventually cross, and the moment they overlap
 * they stop being two things.
 *
 * AND HOW IT GOES OUT. When the sweep comes to rest the speed decays away and
 * the glints run down with it, on the same exponential: still drifting
 * forward, but shrinking and dimming until there is nothing left. That is the
 * glint running out of push, not the knob reaching back into it — which is why
 * it only ever runs DOWN, and a fresh sweep starts a new glint rather than
 * reviving a dying one. Keep sweeping and a glint crosses the whole picture
 * instead; both deaths are real, and which one you get is a fact about the
 * gesture.
 *
 * `pos` is measured in CROSSINGS: 0 is the leading edge of the lit part of the
 * picture and 1 is the trailing edge, so advancing it by `speed * dt` is what
 * makes the invariant exact. The effect supplies `lead` and `trail` — the
 * margins outside that, in the same units, where a glint is fading in or its
 * wake is still finishing — because only the effect knows how big the stack
 * is on screen.
 *
 * Host-free, like three_planes_rig.h and three_walls_show.h: no effect ABI, no
 * GPU, so the Catch2 goldens drive the whole thing at an exact dt.
 */

#include <cmath>

#include <sketch/knob_rate.h>

namespace three_planes_glints {

/// How many glints can be in flight at once.
constexpr int kMaxLive = 8;

/// A frame longer than this is a transport stall, not slow motion. Same clamp,
/// and the same reason, as three_planes_rig.h's.
constexpr float kMaxDt = 0.25f;

/// The window the knob's speed is measured over. Not a knob: it is a property
/// of how MIDI encoders send (a quantized step every few frames), not a taste
/// setting. See knob_rate.h.
constexpr float kRateWindow = 0.09f;

/// How long the speed takes to fall away once the knob stops, in seconds. The
/// attack is instant — a glint launches at exactly the speed of the gesture
/// that threw it — and this is only about how it coasts afterwards.
constexpr float kSpeedRelease = 0.22f;

/// Floor on the travel, in crossings per second. NOT zero: a glint that has
/// stopped being driven still drifts, so letting go of the knob does not park
/// one in the middle of the picture.
constexpr float kMinSpeed = 0.22f;

/// Below this speed a glint is no longer being SWEPT, only drifting — so it
/// starts to putter out. Deliberately the same value as the travel floor:
/// that floor is exactly the drift, so it is exactly the point where "still
/// being pushed" becomes "running down".
constexpr float kPutterFrom = kMinSpeed;

/// How far gone a glint has to be before it is retired. Not zero — the tail of
/// an exponential is forever, and a glint nobody can see is still one of eight
/// slots' worth of nothing.
constexpr float kVitDead = 0.05f;

/// How much of a glint's width survives at the end of the putter. It narrows
/// as it goes, but the BRIGHTNESS is what actually takes it out: a tube going
/// down dims long before it becomes a hairline.
constexpr float kPutterWidthFloor = 0.40f;

/// Where the small extra glints begin and where they are in full flow, in
/// crossings per second. Below the first there are none at all: an unhurried
/// sweep is one clean glint, and nothing else.
constexpr float kChaosFrom = 1.20f;
constexpr float kChaosFull = 4.00f;

/// How much faster a glint travels at a full throw, before `sling` scales it.
/// The release is the sweep arriving at a mute and spending everything it had
/// held; whatever glints are in the air get slung along with it. Shared like
/// every other speed here, so it cannot reorder them.
constexpr float kFlingBoost = 3.0f;

/// Minimum separation, in crossings, between any two live glints. Two on top
/// of each other read as one fat one, which is what these stopped being a
/// pattern to avoid.
constexpr float kMinGap = 0.06f;

/// The spread of per-glint widths drawn at birth, as multiples of the effect's
/// base width. The LAUNCHED glint is the gesture and stays close to nominal;
/// the chaos ones are visibly smaller so they never compete with it.
///
/// The effect needs the top of the whole range to size the margins that hide a
/// glint's entry and its wake's exit — and those margins have to be the same
/// for every glint, or `pos` would map to a different place on screen for each
/// of them and two could cross.
constexpr float kLaunchWidthLo = 0.90f, kLaunchWidthHi = 1.15f;
constexpr float kChaosWidthLo = 0.22f, kChaosWidthHi = 0.50f;
constexpr float kMaxWidthFactor = kLaunchWidthHi;

/// Read fresh each tick.
struct Params {
  float sweep = 0.5f;    ///< the knob, 0..1. Its MOTION is the whole input.
  float band = 0.45f;    ///< launch band, as a fraction of each half of the throw
  float ratio = 1.0f;    ///< crossings per knob range. 1 = the invariant, literally
  float chaos = 4.0f;    ///< small extra glints per second, at kChaosFull and above
  /// The rig's throw, 0..1. Holds the glints already in flight at full
  /// strength while it rings out — a glint being thrown is not a glint running
  /// down, whatever the knob has stopped doing — and, through `sling`, carries
  /// them along with it.
  float fling = 0.0f;
  /// How much of that throw is spent on SPEED, 0..1. At 1 they are slung off
  /// with everything else; at 0 they are held but not hurried, so they stay in
  /// the picture and drift for as long as the throw lasts. The hold is not
  /// negotiable either way — a throw never runs a glint down.
  float sling = 1.0f;

  /// Margins outside the lit picture, in crossings, where a glint is fading in
  /// (`lead`) or its wake is still finishing (`trail`). The effect computes
  /// these from the stack's projected size; the defaults here are only so a
  /// golden can run without one.
  float lead = 0.05f;
  float trail = 0.15f;
};

/// One glint in flight. The three look values are drawn at birth and never
/// touched again — that is what "under its own power" means here. What DOES
/// change is `vit`, which is the glint running down rather than the knob
/// reaching back into it.
struct Glint {
  float pos = 0.0f;    ///< crossings: 0 the leading edge of the picture, 1 the trailing
  float gain = 0.0f;   ///< brightness factor, at birth
  float width = 1.0f;  ///< width factor, at birth
  float shade = 0.0f;  ///< depth of the dark wake behind it, at birth
  bool live = false;
  bool launched = false;  ///< thrown by the knob, rather than chaos

  /// How much push is left in it: 1 while the sweep is still driving it,
  /// falling in step with the speed once the knob comes to rest.
  ///
  /// RATCHETED — it only ever decreases. A fresh sweep must not re-inflate a
  /// glint that has already started to go out; that reads as a rewind, and the
  /// new gesture has its own glint to be seen in.
  float vit = 1.0f;

  // What the effect actually draws. Kept here rather than at the call site so
  // the shape of the putter is one decision in one place.
  float drawGain() const { return gain * vit; }
  float drawShade() const { return shade * vit; }
  float drawWidth() const {
    return width * (kPutterWidthFloor + (1.0f - kPutterWidthFloor) * vit);
  }
};

struct Core {
  Glint glints[kMaxLive];
  knob_rate::KnobRate rate;

  /// Starts TRUE so a knob already parked in the middle — which is where an
  /// untouched card sits — does not read as an entry and throw a glint at boot
  /// with no gesture behind it.
  bool inside = true;
  /// The knob as of the previous tick. A hand at 60 fps cannot get clean
  /// through the band between two samples, but a stalled frame or a jumped
  /// automation lane can — and a gesture that silently does nothing is not a
  /// control, so that counts as a traverse too.
  float last = 0.5f;
  float speed = 0.0f;     ///< crossings per second, shared by every live glint
  float chaos_c = 1.0f;   ///< countdown to the next small one, in expected arrivals
  unsigned rng = 0x2545f491u;
  /// Gestures thrown since the last reset. Nothing reads it but the goldens
  /// and anyone debugging — a live glint can die before you get to count it.
  unsigned launches = 0;

  void reset() { *this = Core(); }

  int liveCount() const {
    int n = 0;
    for (int i = 0; i < kMaxLive; ++i) if (glints[i].live) ++n;
    return n;
  }

  void tick(const Params& p, float dt) {
    if (dt < 0.0f) dt = 0.0f;
    if (dt > kMaxDt) dt = kMaxDt;

    // A NaN patch holds the middle rather than poisoning the ring and the
    // band test alike.
    const float sw = (p.sweep == p.sweep) ? p.sweep : 0.5f;

    // --- 1. How fast the knob is moving, in crossings per second. The sign is
    //        dropped here and nowhere else: which way you are going never
    //        reaches a glint, so reversing mid-gesture leaves the ones already
    //        out there travelling exactly as they were.
    const float r = rate.sample(sw, dt, kRateWindow);
    const float v = (r < 0.0f ? -r : r) * (p.ratio > 0.0f ? p.ratio : 0.0f);

    // Instant attack so a launch takes the speed of the gesture that threw it
    // exactly; a timed release so letting go coasts instead of stopping dead.
    if (dt > 0.0f) {
      const float held = speed * std::exp(-dt / kSpeedRelease);
      speed = v > held ? v : held;
      if (speed < 1e-4f) speed = 0.0f;
    }

    // --- 2. Travel, and the putter.
    //
    //        One speed for everyone, which is what keeps the order they were
    //        born in — and therefore their separation — intact forever. Below
    //        the floor they keep DRIFTING, so nothing is ever left parked.
    //
    //        But drifting is not being swept, and a glint that sailed on at a
    //        crawl forever would outlive the gesture that made it. So the same
    //        envelope that is running the speed down runs them down with it:
    //        as the knob comes to rest they shrink and dim on exactly that
    //        exponential, still moving, until there is nothing left to see.
    const float fling = p.fling < 0.0f ? 0.0f : (p.fling > 1.0f ? 1.0f : p.fling);
    const float sling = p.sling < 0.0f ? 0.0f : (p.sling > 1.0f ? 1.0f : p.sling);
    const float travel = (speed > kMinSpeed ? speed : kMinSpeed)
                       * (1.0f + kFlingBoost * fling * sling);
    // A throw holds the putter off for as long as it lasts. Letting go of the
    // knob to reach a mute is exactly the gesture that fires one, so without
    // this the glints would start running down at the very moment they are
    // being flung — which is backwards.
    float vitality = speed >= kPutterFrom ? 1.0f : speed / kPutterFrom;
    if (fling > vitality) vitality = fling;
    const float death = 1.0f + (p.trail > 0.0f ? p.trail : 0.0f);
    for (int i = 0; i < kMaxLive; ++i) {
      Glint& g = glints[i];
      if (!g.live) continue;
      g.pos += travel * dt;
      if (g.vit > vitality) g.vit = vitality;   // ratchet: down only
      // Two ways out, and both are real: cross the picture, or run down where
      // you are. Which one happens is a fact about the gesture — keep sweeping
      // and a glint makes it all the way over.
      if (g.pos > death || g.vit <= kVitDead) g = Glint();
    }

    // --- 3. The launch. Entering the middle band is the event, and it is the
    //        right one for two reasons: a traverse of the knob crosses it
    //        exactly once, and you cannot enter it standing still — so there
    //        is always a real speed for the new glint to take.
    const float half = sw - 0.5f;
    const float mag = (half < 0.0f ? -half : half) * 2.0f;   // 0 centre, 1 either end
    const bool inside_now = mag <= (p.band < 0.0f ? 0.0f : p.band);
    // Entering the band — or clearing it outright between two samples, which
    // shows up as the knob changing sides without ever being seen inside.
    const bool jumped = !inside_now && !inside &&
                        ((last - 0.5f) < 0.0f) != (half < 0.0f);
    if ((inside_now && !inside) || jumped) launch(p, vitality);
    inside = inside_now;
    last = sw;

    // --- 4. Chaos. Small ones, and only once the sweep is brisk: an unhurried
    //        gesture is one clean glint and nothing else. Exponentially spaced,
    //        so they read as independent events rather than as a second
    //        metronome running underneath the first.
    float drive = (speed - kChaosFrom) / (kChaosFull - kChaosFrom);
    drive = drive < 0.0f ? 0.0f : (drive > 1.0f ? 1.0f : drive);
    const float crate = (p.chaos > 0.0f ? p.chaos : 0.0f) * drive;
    if (crate > 0.0f && dt > 0.0f) {
      chaos_c -= crate * dt;
      for (int guard = 0; chaos_c <= 0.0f && guard < kMaxLive; ++guard) {
        spawn(p, false, vitality);
        float u = rand01();
        if (u < 1e-6f) u = 1e-6f;
        chaos_c += -std::log(u);   // mean 1, so `chaos` really is per second
      }
      if (chaos_c <= 0.0f) chaos_c = 1e-3f;
    }
  }

 private:
  /// xorshift32 — deterministic given the same dt sequence, which is what lets
  /// a golden pin any of this at all.
  float rand01() {
    rng ^= rng << 13;
    rng ^= rng >> 17;
    rng ^= rng << 5;
    return (float)(rng & 0xFFFFFFu) * (1.0f / 16777216.0f);
  }

  /// The knob's own glint. GUARANTEED: unlike a chaos arrival it is never
  /// refused, because the whole design is that a traverse throws exactly one.
  void launch(const Params& p, float vitality) {
    ++launches;
    spawn(p, true, vitality);
  }

  void spawn(const Params& p, bool launched, float vitality) {
    const float birth = -(p.lead > 0.0f ? p.lead : 0.0f);

    // Not on top of the youngest. Separation is established here and preserved
    // for life by the shared speed, so this is the only place it can be got
    // wrong — and a cluster is indistinguishable from one fat glint.
    float youngest = 1e9f;
    for (int i = 0; i < kMaxLive; ++i)
      if (glints[i].live && glints[i].pos < youngest) youngest = glints[i].pos;

    float pos = birth;
    if (youngest < birth + kMinGap) {
      // A chaos arrival with no room is simply DROPPED. A launch is not: it is
      // placed further back instead, so it arrives a beat later but it always
      // arrives.
      if (!launched) return;
      pos = youngest - kMinGap;
    }

    Glint* slot = nullptr;
    for (int i = 0; i < kMaxLive; ++i) {
      if (glints[i].live) continue;
      slot = &glints[i];
      break;
    }
    if (!slot) {
      // Every slot busy. A chaos arrival gives up — stealing one would make a
      // glint vanish in mid-flight, the one thing a particle here may not do.
      // A launch takes the OLDEST, which is the one about to die anyway.
      if (!launched) return;
      float best = -1e9f;
      for (int i = 0; i < kMaxLive; ++i)
        if (glints[i].pos > best) { best = glints[i].pos; slot = &glints[i]; }
    }

    slot->live = true;
    slot->launched = launched;
    slot->pos = pos;
    // Born at whatever push there is right now, rather than at full and
    // dropping a frame later — a gesture too faint to carry a glint should
    // make a faint one, not a bright one that pops.
    slot->vit = vitality;
    // Drawn once, kept for life. The launched glint stays near nominal — it is
    // the gesture, and it has to read the same every time you make it. The
    // chaos ones scatter, and are small enough never to be mistaken for it.
    if (launched) {
      slot->gain = 0.85f + 0.15f * rand01();
      slot->width = kLaunchWidthLo + (kLaunchWidthHi - kLaunchWidthLo) * rand01();
      slot->shade = 0.75f + 0.25f * rand01();
    } else {
      slot->gain = 0.22f + 0.28f * rand01();
      slot->width = kChaosWidthLo + (kChaosWidthHi - kChaosWidthLo) * rand01();
      slot->shade = 0.20f + 0.35f * rand01();
    }
  }
};

}  // namespace three_planes_glints
