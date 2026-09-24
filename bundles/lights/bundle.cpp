/*
 * lights — LED-bar show effects bundle.
 *
 * All effects for the 4-bar performance live here. See
 * docs/SHOW_EFFECTS_PLAN.md for the design.
 *
 * Bundle ID:  com.nano.lights
 * Effect IDs: gen.*  (generators), fx.*  (post-process complicators)
 */

#include <module_api.h>
#include <cstddef>

// All effects are converted to the class-like per-instance ABI.
NANO_DECLARE_INSTANCE_EFFECT(strobe_channel)
NANO_DECLARE_INSTANCE_EFFECT(soft_glow)
NANO_DECLARE_INSTANCE_EFFECT(dispersion)
NANO_DECLARE_INSTANCE_EFFECT(plasma_beam_cannon)
NANO_DECLARE_INSTANCE_EFFECT(orthomod)
NANO_DECLARE_INSTANCE_EFFECT(bounce_resonator)
NANO_DECLARE_INSTANCE_EFFECT(side_jet)
NANO_DECLARE_INSTANCE_EFFECT(motion_blobs)
NANO_DECLARE_INSTANCE_EFFECT(lights_sim)
NANO_DECLARE_INSTANCE_EFFECT(block_dehance)
NANO_DECLARE_INSTANCE_EFFECT(tingle_top)
NANO_DECLARE_INSTANCE_EFFECT(chroma_wave)
NANO_DECLARE_INSTANCE_EFFECT(flicker_grid)
NANO_DECLARE_INSTANCE_EFFECT(three_planes)
NANO_DECLARE_INSTANCE_EFFECT(three_walls)
NANO_DECLARE_INSTANCE_EFFECT(vcr_halo)

extern "C" {

NANO_EXPORT_ABI_VERSION()

__attribute__((export_name("nano_module_main")))
void nano_module_main() {
    nano::registerEffect({
        2,
        "source.light.strobe_channel",
        "Strobe Channel",
        "Logistic-map-driven single-bar selector. A smooth ping-pong seed value is iterated through the chaotic logistic map; the final value selects which bar lights up. Cranking r toward 4 gives maximum chaos / rapid strobe-like switching.",
        "source",
        "strobe,chaos,logistic-map,bar,trigger",
        "la-bolt",
        NANO_INSTANCE_LIFECYCLE(strobe_channel),
    });

    nano::registerEffect({
        2,
        "source.light.soft_glow",
        "Soft Glow",
        "Continuous warm-blob atmosphere bed. Slowly-drifting gaussian blobs across the bars accumulate into a hue-shifting blackbody-ish color ramp. Designed to sit underneath everything as the show's ambient layer.",
        "source",
        "atmosphere,glow,blobs,bed,warm",
        "la-sun",
        NANO_INSTANCE_LIFECYCLE(soft_glow),
    });

    nano::registerEffect({
        2,
        "warp.dispersion",
        "Dispersion",
        "Block-quantized UV-jitter sampler. Tiles the canvas into discrete blocks (size quantized internally to avoid sweeping boundaries), picks a stable random offset per block, samples the input at (block_center + offset). Small blocks → crunchy grain; large blocks → mosaic downres.",
        "warp",
        "dispersion,grain,mosaic,glitch,jitter",
        "la-splotch",
        NANO_INSTANCE_LIFECYCLE(dispersion),
    });

    nano::registerEffect({
        2,
        "source.light.plasma_beam_cannon",
        "Plasma Beam Cannon",
        "90s-anime power-up beam. Attack seed snaps small at a target Y, decay rapidly expands to fill the bar, sustain holds, release breaks up (break particles deferred for v2). All four bars share one linked ADSR timeline.",
        "source",
        "plasma,beam,cannon,trigger,adsr,drama,beat",
        "la-fire",
        NANO_INSTANCE_LIFECYCLE(plasma_beam_cannon),
        nullptr, nullptr, nullptr, &plasma_beam_cannon::eval_visibility,
    });

    nano::registerEffect({
        2,
        "source.light.orthomod",
        "Orthomod",
        "Hadamard-driven beat-synced bar pattern. Two co-driven code systems share a global envelope: an 8x8 Hadamard sorted by row complexity drives 4 per-bar channel envelopes (square / sine / on / off waveforms per 2-bit code), while an MxM Hadamard grouped into pages of 4 rows drives the per-bar segment fill pattern. Triggers via the host bar clock. Exposes ch1..ch4 + env as float rails for downstream effects.",
        "source",
        "atmosphere,hadamard,beat,pattern,bar,bed",
        "la-border-all",
        NANO_INSTANCE_LIFECYCLE(orthomod),
    });

    nano::registerEffect({
        2,
        "source.light.bounce_resonator",
        "Bounce Resonator",
        "4-bar scalar diffusion network: fire an impulse into one bar (or all, or sampled from the input video) and its energy bounces between the bars through a seeded cycling exchange matrix, ringing out per feedback. Hops carry colour — hue spread/converge let it wander or home in on the bar colour. Each bar fills its 1/4 column; chroma hold trades the white-hot overdrive bloom for a hue-preserving limiter. Motion-vector passthrough.",
        "source",
        "resonator,bounce,coupled,trigger,physics,bar",
        "la-broadcast-tower",
        NANO_INSTANCE_LIFECYCLE(bounce_resonator),
        nullptr, nullptr, nullptr, &bounce_resonator::eval_visibility,
    });

    nano::registerEffect({
        2,
        "source.light.side_jet",
        "Side Jet",
        "JPL-style horizontal jet trail. Trigger spawns a procedural jet that traverses the canvas; the shape is a diverging cone with Mach-diamond pulsation along the axis and Fbm-modulated turbulent edges. Pool of up to 16 concurrent jets; direction selectable LtoR / RtoL / random. Emits motion vectors so a downstream motion.blur picks up the head naturally.",
        "source",
        "jet,trail,plume,trigger,motion,bar",
        "la-wind",
        NANO_INSTANCE_LIFECYCLE(side_jet),
    });

    nano::registerEffect({
        2,
        "source.light.motion_blobs",
        "Motion Blobs",
        "Pool of traveling soft blobs that drive motion vectors AND/OR color darkening. motion_strength=1, shadow_darkness=0 is pure motion rain (invisible blobs feeding render_outputs/motion for a downstream motion.blur smear). motion_strength=0, shadow_darkness>0 is shadow flyover (dark sweeping shapes). Both at once gives moving shadows that also blur the underlying scene. Edge-spawning blobs traverse INTO the canvas with parallel drift; the field auto-tops up to density × blob_count_max alive.",
        "source",
        "blobs,motion,shadow,flyover,rain,bar",
        "la-cloud",
        NANO_INSTANCE_LIFECYCLE(motion_blobs),
    });

    nano::registerEffect({
        2,
        "filter.lights_sim",
        "Lights Sim",
        "Samples the input into 4 vertical LED bars (Resolume-style fixture sampling). Each quarter of the input is one bar, divided into `segments` LED segments; a segment's colour is sampled at the horizontal centre of its quarter and the vertical centre of its segment. The bars render inset into their quarters (separate horizontal / vertical inset) over the input faded by input_opacity.",
        "filter",
        "led,bar,sample,resolume,fixture,segments",
        "la-lightbulb",
        NANO_INSTANCE_LIFECYCLE(lights_sim),
    });

    nano::registerEffect({
        2,
        "filter.glitch.block_dehance",
        "Block Dehance",
        "Glitch rectangles that 'dehance' the input in one of three modes — black-fill (dropout), mosaic downres, or noise — sampled probabilistically per rect at spawn, so one instance mixes all three. A GPU rect pool cycles continuously; each rect bright-seeks the mask for its position. Weights control the mode mix; optional per-rect hard-duty flicker for the aggressive glitch feel.",
        "filter",
        "glitch,dropout,mosaic,noise,dehance,block",
        "la-th",
        NANO_INSTANCE_LIFECYCLE(block_dehance),
    });

    nano::registerEffect({
        2,
        "source.light.tingle_top",
        "Tingle Top",
        "Sparkles bundled at the top of each bar while gated, released downward on an envelope when ungated. Particles live and die in place; the cascade is a spawn-region animation (region_y_max snaps to a thin top band while held, ramps to full bar on release). Single hue + jitter, per-frame alpha shimmer. Optional per-particle velocity unlocks the downward-sparkle fountain preset.",
        "source",
        "sparkle,tingle,particles,cut-in,trigger,bar",
        "la-snowflake",
        NANO_INSTANCE_LIFECYCLE(tingle_top),
        nullptr, nullptr, nullptr, &tingle_top::eval_visibility,
    });

    nano::registerEffect({
        2,
        "source.light.chroma_wave",
        "Chroma Wave",
        "Charge-and-burst prismatic wave bloom. A soft super-gaussian blob grows from the top-center while gated; as pressure builds the top flattens into a plateau, the blob elongates in X and hollows out at the top so the mass piles into a downward crescent (max pressure). On release it bursts — rapidly expanding while the colour-grade transfer folds, sending prismatic bands travelling down the density gradient (dominant) and washing back up the inner edge (secondary). Additive bloom composited over the input; a secondary wave_out texture output carries the wave alone on black when wired.",
        "source",
        "chroma,prismatic,wave,bloom,trigger,charge,burst,beat",
        "la-water",
        NANO_INSTANCE_LIFECYCLE(chroma_wave),
        nullptr, nullptr, nullptr, &chroma_wave::eval_visibility,
    });

    nano::registerEffect({
        2,
        "filter.light.flicker_grid",
        "Flicker Grid",
        "Per-column luma-to-flicker-rate LED grid. Reduces the input to a grid (default 4x10) of flat box-averaged cells; each column's luma (peak or average) sets a per-column pulse rate — brighter is faster, capped at on/off every frame, with optional overflow fill pouring beyond-cap rate into the off frames. Dimmer columns keep 1-frame pulses with growing gaps; below the low threshold a column is black, at/above the high threshold it holds solid. Cell colours can be pulled toward neutral HSL lightness (the flicker carries the brightness) and levelled up toward the column max on a curve that leaves near-black alone. Built for LEDs: dodges their low-brightness weakness and adds temporal contrast.",
        "filter",
        "flicker,grid,led,column,strobe,pulse,luma,temporal",
        "la-th-large",
        NANO_INSTANCE_LIFECYCLE(flicker_grid),
    });

    nano::registerEffect({
        2,
        "source.mesh.three_planes",
        "Three Planes",
        "Three isometric planes stacked like a 3D chess board, shaded as VCR-era neon. The CPU projects twelve corner points per frame (orthographic, so orbiting never adds perspective) and one fullscreen pass shades every plane at once from an exact signed distance field — so the halo is a smooth function of true distance with correctly rounded corners, and its radius is free to widen. Each plane is empty, neon-filled, or a black mask that eats the glow of everything beneath it while keeping its own outline. Publishes per-plane screen Y (azimuth-independent) and silhouette half-height as rails. Travelling diagonal glints — the glare off metal in an old cel-animated show — are particles thrown by a GESTURE: wire Glint Sweep from Three Planes Rig's Sweep Out and every pass of the knob through the middle of its throw launches exactly one, crossing the picture at exactly the speed you moved. Which way you moved is thrown away, so reversing mid-gesture leaves the glints already in flight alone; each keeps the brightness and width it was born with, and only their shared speed still follows the knob, so two can never cross. Sweep hard and smaller ones start arriving at random underneath. Come to rest and they run down with the sweep — still drifting forward, but shrinking and dimming on the same curve the speed falls on — while keeping on sweeping carries one clean across the picture. Reaching either end of the sweep is a THROW rather than just a mute: wire Release from the rig and the stack is flung outward as three expanding rings — outline only, opening out and going dull as they fly, a filter closing drawn in space — ringing down over the muted picture on their own clock, so sweeping straight back relights the tower over a tail still running. Glints in the air are slung along with them. They multiply each plane's emission rather than the finished picture, so a glint lights the halo along with the core.",
        "source",
        "isometric,neon,vcr,glow,stack,layer,sdf,show,meter,glimmer,glint",
        "la-layer-group",
        NANO_INSTANCE_LIFECYCLE(three_planes),
    });

    // The frontal companion to three_planes: the same neon-quad motif pointed
    // INTO the screen, and the only effect here with three camera outputs.
    nano::registerEffect({
        2,
        "source.mesh.three_walls",
        "Three Walls",
        "Three neon frames rushing at you down a tunnel — the frontal companion to Three Planes, sharing its quad field and VCR grade but perspective rather than orthographic. Quiet until a move fires: Pulse throws the three frames at you one after another; Cycles, Resonate and Resonate Rev are HELD, running while the trigger is high and ramping their rate the whole time. Resonate is frame-locked rather than dt-integrated, so past roughly half the frame rate it stops reading as motion and lands on standing strobe patterns. Three texture outputs are three cameras on the same tunnel: the main one looks down its throat, and the two auxiliaries watch it from either side, so a frame that fills the main output is the same instant a bar sweeping across the sides. The auxiliaries cost nothing unless wired.",
        "source",
        "tunnel,neon,vcr,glow,perspective,depth,pulse,resonate,strobe,frame,show,corridor",
        "la-expand",
        NANO_INSTANCE_LIFECYCLE(three_walls),
    });

    nano::registerEffect({
        2,
        "filter.glow.vcr_halo",
        "VCR Halo",
        "The Three Planes look applied to an arbitrary image: a multi-octave neon glow pyramid plus the shared VCR dehancement tail (highlight bleach, asymmetric soft clip, filmic toe/shoulder, horizontal chroma split, scanlines, grain). Halo Radius slides weight across the pyramid's octaves rather than switching levels, so it modulates continuously and a wide halo costs barely more than a narrow one. Outline band-passes the emitter so a filled shape glows at its edge like neon tubing instead of blooming as a soft lump. Shares nano_vcr.hlsl with source.mesh.three_planes, so matching the grade knobs matches the look.",
        "filter",
        "glow,bloom,halo,neon,vcr,vhs,dehance,grade,scanline,grain,show",
        "la-lightbulb",
        NANO_INSTANCE_LIFECYCLE(vcr_halo),
    });
}

} // extern "C"
