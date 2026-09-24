/*
 * legacy — ports of shipped dnode / NanoGraph effects (com.nano.legacy).
 *
 * These are re-implementations ("v2"s) of effects we ran at live events in
 * the old NanoGraph framework, brought onto the nano-modules compute/render
 * model. The catalogue of source effects lives in import/EFFECTS_CATALOG.md.
 *
 * To add an effect: drop the source under wasm_modules/<name>/, add a
 * forward declaration + registerEffect call below, and reference its
 * sources from build.sh.
 */

#include <module_api.h>
#include <cstddef>

NANO_DECLARE_INSTANCE_EFFECT(bicolor_grad)
NANO_DECLARE_INSTANCE_EFFECT(glisten)
NANO_DECLARE_INSTANCE_EFFECT(double_chamber)
namespace double_chamber {
void eval_visibility(int n, const char* pb, const int* off, const int* len, const int* ops);
}
NANO_DECLARE_INSTANCE_EFFECT(d_wave)
NANO_DECLARE_INSTANCE_EFFECT(lut_collection)
namespace lut_collection { int32_t is_identity(void* self); }
NANO_DECLARE_INSTANCE_EFFECT(zoom_scroller)
NANO_DECLARE_INSTANCE_EFFECT(subtle_blur)
namespace subtle_blur { int32_t is_identity(void* self); }
NANO_DECLARE_INSTANCE_EFFECT(sphr_blur)
namespace sphr_blur { int32_t is_identity(void* self); }
NANO_DECLARE_INSTANCE_EFFECT(burn_out)
NANO_DECLARE_INSTANCE_EFFECT(chroma_wobble)
NANO_DECLARE_INSTANCE_EFFECT(wobble_master)
NANO_DECLARE_INSTANCE_EFFECT(stutter_scale)
namespace stutter_scale { int32_t is_identity(void* self); }
NANO_DECLARE_INSTANCE_EFFECT(freeze_pulse)
NANO_DECLARE_INSTANCE_EFFECT(pixulant)

extern "C" {

NANO_EXPORT_ABI_VERSION()

__attribute__((export_name("nano_module_main")))
void nano_module_main() {
    nano::registerEffect({
        2,
        "color.legacy.bicolor_grad",
        "Bicolor Gradient",
        "Content-adaptive two-colour gradient. Analyses the input for its two "
        "dominant complementary hues and where they sit, then paints a smooth "
        "gradient between them, oriented along the image's own colour layout. "
        "Port of the shipped NanoGraph BicolorGrad.",
        "color",
        "gradient,palette,color,analysis,legacy,bicolor",
        "la-fill",
        NANO_INSTANCE_LIFECYCLE(bicolor_grad),
    });

    nano::registerEffect({
        2,
        "filter.legacy.glisten",
        "Glisten",
        "Image-anchored sparkle. Locates the brightest region on a blurred "
        "search grid, then stacks layered polygon-disc glints there, shaded "
        "hard by the local colour gradient (colours run negative and dig "
        "complementary hues out of the image), blurred into a glow layer "
        "whose gain twinkles with a random flicker envelope. Faithful port "
        "of NanoGraph Glisten.",
        "filter",
        "sparkle,glint,glisten,lens,star,legacy",
        "la-star",
        NANO_INSTANCE_LIFECYCLE(glisten),
    });

    nano::registerEffect({
        2,
        "source.legacy.double_chamber",
        "Double Chamber",
        "Particle field-chamber: a pool of particles flows through a chaotic "
        "polynomial vector field, pulled and curled around a few drifting "
        "attractors, bounded in a soft disc, coloured from the input image. "
        "Optional interactions let the particles feel each other through a "
        "crowding buffer — thinning packed regions, avoiding neighbours, or "
        "streaming with the local group. v2 "
        "of the shipped NanoGraph DoubleChamber (the used subset — no charged "
        "accelerator, no laser output).",
        "source",
        "particles,field,chamber,attractor,curl,interactions,flocking,generative,legacy",
        "la-columns",
        NANO_INSTANCE_LIFECYCLE(double_chamber),
        nullptr, nullptr, nullptr, &double_chamber::eval_visibility,
    });

    nano::registerEffect({
        2,
        "warp.legacy.d_wave",
        "D Wave",
        "Radial-ripple distortion field. Ripples are stochastically spawned at "
        "the centre, expand outward as concentric arcs, and decay — radially "
        "warping the input image as they pass. A trigger fires a full-circle "
        "shock ripple. v2 port of the distortion field (\"D wave\") from the "
        "shipped NanoGraph Darkburst (the one block we actually used live).",
        "warp",
        "distortion,ripple,wave,warp,radial,darkburst,legacy",
        "la-wave-square",
        NANO_INSTANCE_LIFECYCLE(d_wave),
    });

    nano::registerEffect({
        2,
        "color.legacy.lut_collection",
        "LUT Collection 1",
        "Preset colour LUT grader. Applies one of 13 baked film/stylize look-up "
        "tables with a pregain tone push and a mix amount. The LUTs are baked "
        "into real 32³ 3D textures and applied in a single hardware-trilinear "
        "sample. v2 of the shipped Resolume Wire \"LUT 2\" patch.",
        "color",
        "lut,color,grade,film,look,preset,cube,legacy",
        "la-table",
        NANO_INSTANCE_LIFECYCLE(lut_collection),
        &lut_collection::is_identity,
    });

    nano::registerEffect({
        2,
        "warp.legacy.zoom_scroller",
        "Zoom Scroller",
        "Procedural pan-and-zoom sequence camera. Generates randomized tours "
        "across the frame — picking a far target, a zoom level, and an L-shaped "
        "grid path — then pans through it in quantized, choppy sub-steps with a "
        "white box gizmo that juts in the direction of motion. Good for idle "
        "moments. v2 port of the Resolume Wire \"ZoomScroller\" patch.",
        "warp",
        "zoom,pan,scroll,camera,sequence,gizmo,idle,legacy",
        "la-search-plus",
        NANO_INSTANCE_LIFECYCLE(zoom_scroller),
    });

    nano::registerEffect({
        2,
        "filter.legacy.subtle_blur",
        "Subtle Blur",
        "Light Gaussian blur with a slowly-drifting chromatic colour offset — a "
        "soft bloom with a faint, shifting RGB fringe on edges. Good for "
        "breaking up sharp edges. v2 of the Resolume Wire \"Subtle Blur\" patch.",
        "filter",
        "blur,soft,bloom,chroma,fringe,aberration,legacy",
        "la-water",
        NANO_INSTANCE_LIFECYCLE(subtle_blur),
        &subtle_blur::is_identity,
    });

    nano::registerEffect({
        2,
        "filter.legacy.sphr_blur",
        "SPHR Blur",
        "Sphere-aware blur. Treats the frame as an equirectangular map and "
        "blurs horizontally with a radius that grows toward the top and bottom "
        "edges, then adds an isotropic Gaussian softening on top — a "
        "seam-correct blur for dome content that's also a distinctive "
        "edge-softener off-sphere. v2 of the Resolume Wire \"SPHR Blur\" patch.",
        "filter",
        "blur,sphere,equirect,dome,sphr,soft,legacy",
        "la-globe",
        NANO_INSTANCE_LIFECYCLE(sphr_blur),
        &sphr_blur::is_identity,
    });

    nano::registerEffect({
        2,
        "color.legacy.burn_out",
        "Burn Out",
        "Triggered exposure-blowout grade for emotional fade-outs. Tap the "
        "trigger (or hold the gate) and the image blows out — saturation and "
        "contrast lift, exposure pushes highlights toward white, then it decays "
        "back over the release. Can also drop alpha for a compositing fade-out. "
        "v2 of the Resolume Wire \"Burn Out\" patch.",
        "color",
        "burn,blowout,exposure,fade,envelope,grade,trigger,legacy",
        "la-fire",
        NANO_INSTANCE_LIFECYCLE(burn_out),
    });

    nano::registerEffect({
        2,
        "warp.legacy.chroma_wobble",
        "Chroma Wobble",
        "Triggered chromatic-aberration wobble. An animated fractal-noise field "
        "warps the image with RGB fringing on a trigger (or held gate), then "
        "decays via an Attack/Release envelope. v2 of the Resolume Wire "
        "\"ChromaWobble\" patch (analytic noise, no feedback texture).",
        "warp",
        "wobble,chroma,aberration,noise,glitch,trigger,legacy",
        "la-water",
        NANO_INSTANCE_LIFECYCLE(chroma_wobble),
    });

    nano::registerEffect({
        2,
        "warp.legacy.wobble_master",
        "Wobble Master",
        "Triggered shockwave wobble with chromatic afterglow. Each pulse "
        "emanates from a centre and travels outward, distorting only what's "
        "under the wave (a radial push with an oscillating shimmer); the "
        "colour split lingers behind the front (red out, blue in). "
        "v2 of the Resolume Wire \"Wobble Master\" family.",
        "warp",
        "wobble,ripple,chroma,aberration,beat,pulse,radial,legacy",
        "la-water",
        NANO_INSTANCE_LIFECYCLE(wobble_master),
    });

    nano::registerEffect({
        2,
        "warp.legacy.stutter_scale",
        "Stutter Scale",
        "Beat-stutter scale glitch. A phase is quantized into discrete steps; "
        "the zoom sweeps progressively min to max across them while each step "
        "re-rolls a jitter shift, an optional flip and colour inversion, plus "
        "a hue and contrast push — crossfaded with the input. Great for "
        "stuttering overlays and logos. "
        "v2 of the Resolume Wire \"Stutter Scale 2\" patch.",
        "warp",
        "stutter,scale,zoom,glitch,beat,jitter,flip,legacy",
        "la-expand",
        NANO_INSTANCE_LIFECYCLE(stutter_scale),
        &stutter_scale::is_identity,
    });

    nano::registerEffect({
        2,
        "warp.legacy.freeze_pulse",
        "Freeze Pulse",
        "Stutter-freeze pulse. On a trigger it freezes the current frame, then "
        "scale-pops, jitters and grades that frozen frame and composites it over "
        "the still-running live video using a randomly-chosen blend mode, fading "
        "out over the set time. v2 of the Resolume Wire \"Freeze Pulse\" patch.",
        "warp",
        "freeze,pulse,stutter,glitch,beat,blend,snapshot,legacy",
        "la-snowflake",
        NANO_INSTANCE_LIFECYCLE(freeze_pulse),
    });

    nano::registerEffect({
        2,
        "warp.legacy.pixulant",
        "Pixulant",
        "Roiling pixel-scatter dive. The image is randomly scattered three times "
        "(light, mid, heavy) and the heavy copy is abs-differenced against the "
        "light one, leaving coloured edge halos that bloom out of flat regions "
        "and churn over time. Turn up Dive to push the picture into the grain, "
        "Scatter to widen it, Motion to set the churn rate. v2 of the Resolume "
        "Wire \"Pixulant\" patch.",
        "warp",
        "scatter,dive,difference,feedback,grain,glitch,pixulant,legacy",
        "la-th",
        NANO_INSTANCE_LIFECYCLE(pixulant),
    });
}

} // extern "C"
