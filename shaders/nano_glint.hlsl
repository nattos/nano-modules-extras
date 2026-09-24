// nano_glint.hlsl — one travelling glint, as a function of where you are on
// its travel axis.
//
// Lifted out of three_planes' render pass so the wall pass can light itself
// with the SAME glints rather than a lookalike. A glint is a property of the
// room, not of a picture: the thing that brightens a tube has to brighten what
// that tube throws on the wall in the same instant, or the two read as two
// different events.
//
// The host owns the glints' lives (<sketch/three_planes_glints.h>) and hands
// over however many are in flight, already projected onto the travel axis. All
// that is left is adding up what they look like.

#ifndef NANO_GLINT_HLSL
#define NANO_GLINT_HLSL

// The wake trails, so it sits at a LOWER axis coordinate than the glint itself
// (they travel toward +axis) — hence `d + wake` rather than `d - wake`.
static const float kNanoGlintWake  = 1.5;   // wake offset, in glint half-widths
static const float kNanoGlintWakeW = 1.8;   // wake width, likewise

/// One glint's signed contribution at axis coordinate `axis`.
///
/// `g` is the row the host fills: x = where it sits on the axis, y = its
/// half-width there, z = the brightness and w = the wake depth it was born
/// with. A DEAD SLOT IS ZERO GAIN AND ZERO SHADE, which is why callers can
/// unroll a fixed bank and never test a count.
///
/// The glint is a SUPER-Gaussian (d^4, not d^2): a flatter top with much
/// faster shoulders, so it reads as a hard-edged slash — an object with a
/// boundary — where a plain Gaussian reads as a soft wash sliding past. The
/// wake stays Gaussian, because a wake IS a soft thing. Together they sweep
/// CONTRAST rather than brightness, which is what the eye reads as a moving
/// highlight on a surface rather than a lamp being turned up.
float nano_glint_at(float axis, float4 g) {
  float w = max(g.y, 1e-4);
  float d = (axis - g.x) / w;
  float k = (d + kNanoGlintWake) / kNanoGlintWakeW;
  float d2 = d * d;
  return g.z * exp(-d2 * d2) - g.w * exp(-k * k);
}

#endif  // NANO_GLINT_HLSL
