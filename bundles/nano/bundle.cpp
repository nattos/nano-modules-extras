/*
 * nano — "Weirder" effects bundle.
 *
 * Sibling to `core` for effects that aren't quite the standard
 * brightness/blend/etc. flavour but are still part of the shipping
 * product. The Effect IDE loads this bundle alongside `core`.
 *
 * Today: just `nanolooper`. Future weird/experimental-but-shipping
 * effects land here.
 */

#include <module_api.h>
#include <cstddef>

NANO_DECLARE_INSTANCE_EFFECT(nanolooper)
NANO_DECLARE_INSTANCE_EFFECT(motion_field)
NANO_DECLARE_INSTANCE_EFFECT(flash_particles)
NANO_DECLARE_INSTANCE_EFFECT(local_delay)
NANO_DECLARE_INSTANCE_EFFECT(peak_decay)
NANO_DECLARE_INSTANCE_EFFECT(height_from_gradient)
NANO_DECLARE_INSTANCE_EFFECT(shape_fold)
NANO_DECLARE_INSTANCE_EFFECT(brutal_fold)
NANO_DECLARE_INSTANCE_EFFECT(phase_fold)
NANO_DECLARE_INSTANCE_EFFECT(flow_swarm)
NANO_DECLARE_INSTANCE_EFFECT(sweep_chamber)
NANO_DECLARE_INSTANCE_EFFECT(spectral_lfo)
NANO_DECLARE_INSTANCE_EFFECT(mod_spectral)
NANO_DECLARE_INSTANCE_EFFECT(mod_bass_sim)
NANO_DECLARE_INSTANCE_EFFECT(triangulate)
NANO_DECLARE_INSTANCE_EFFECT(plane_shear)
NANO_DECLARE_INSTANCE_EFFECT(tri_shear)
NANO_DECLARE_INSTANCE_EFFECT(recompose)
NANO_DECLARE_INSTANCE_EFFECT(shape_burst)
NANO_DECLARE_INSTANCE_EFFECT(pixel_ocean)
NANO_DECLARE_INSTANCE_EFFECT(pixel_descent)
NANO_DECLARE_INSTANCE_EFFECT(pixel_rift)
NANO_DECLARE_INSTANCE_EFFECT(simulant)
NANO_DECLARE_INSTANCE_EFFECT(smear)
NANO_DECLARE_INSTANCE_EFFECT(line_reconstruct)
NANO_DECLARE_INSTANCE_EFFECT(lens)
NANO_DECLARE_INSTANCE_EFFECT(envelope_warp)
NANO_DECLARE_INSTANCE_EFFECT(monolith)
NANO_DECLARE_INSTANCE_EFFECT(plume)
NANO_DECLARE_INSTANCE_EFFECT(plume_field)
NANO_DECLARE_INSTANCE_EFFECT(helio_field)
NANO_DECLARE_INSTANCE_EFFECT(dust_halo)
// is_identity is not part of NANO_DECLARE_INSTANCE_EFFECT; declare it so the
// registration can pass &line_reconstruct::is_identity (strength 0 = bypass).
namespace line_reconstruct { int32_t is_identity(void* self); }
namespace lens { int32_t is_identity(void* self); }
namespace envelope_warp { int32_t is_identity(void* self); }

extern "C" {

NANO_EXPORT_ABI_VERSION()

__attribute__((export_name("nano_module_main")))
void nano_module_main() {
    nano::registerEffect({
        2,
        "control.nanolooper",
        "Nano Looper",
        "4-channel 16-step looper sequencer with visual overlay",
        "control",
        "loop,trigger,beat,midi",
        "la-redo",
        NANO_INSTANCE_LIFECYCLE(nanolooper),
    });

    nano::registerEffect({
        2,
        "motion.field",
        "Motion Field",
        "Per-pixel motion vector generator. Soft-thresholds the input by luma, then composes a velocity field from a static rotation, a radial outward direction, and the luma gradient (each weighted), with magnitude and angular jitter on top.",
        "motion",
        "motion,vectors,render-outputs,producer,luma,gradient",
        "la-arrows-alt",
        NANO_INSTANCE_LIFECYCLE(motion_field),
    });

    nano::registerEffect({
        2,
        "source.particles.flash_particles",
        "Flash Particles",
        "Mask-driven particle compositor. Spawns particles at bright spots in the (optional) mask texture, captures the input color at each spawn, and composites alpha-masked oriented quads (solid / squircle / gaussian) using a power-curve life decay with frame jitter. Emits motion vectors along each particle's rotation; chains via render_outputs_in.",
        "source",
        "particles,motion,vectors,render-outputs,producer,mask,jitter",
        "la-star",
        NANO_INSTANCE_LIFECYCLE(flash_particles),
    });

    nano::registerEffect({
        2,
        "motion.local_delay",
        "Local Delay",
        "Stylized motion-driven local delay. Keeps one frame of history, estimates a crude per-pixel flow, smooths it where locally colinear, then masks it by a stochastic-noise field and a signed center vignette. The masked magnitude is power-squashed into a blend weight that cross-fades current->history (moving/unmasked regions ghost toward the delayed frame) and modulates the motion vectors written on render_outputs/motion for a downstream motion blur to clean up.",
        "motion",
        "delay,echo,motion,vectors,flow,render-outputs,producer,stylized",
        "la-history",
        NANO_INSTANCE_LIFECYCLE(local_delay),
    });

    nano::registerEffect({
        2,
        "motion.peak_decay",
        "Peak Decay",
        "Per-pixel peak meter. Any pixel that holds still for longer than the hold time starts to fall: its brightness eases down a smoothstep sigmoid over the fall time, dimming toward black by the amount. The instant a pixel changes it snaps back to full brightness (meter ballistics: instant up, smooth down), so moving material stays vivid while static regions sink away. Change metering compares against the reference latched when the pixel last moved (slow drifts eventually count), balanced luma-to-RGB by rgb_balance, gated by the threshold. The catch select switches to Rise Only: only an upward luma edge (the live luma rising past the held reference) resets the fall — the reference follows the input down silently, so fades, darkening drift and chroma-only changes can't keep pixels awake.",
        "motion",
        "decay,peak,meter,luma,fade,static,motion,temporal,stale",
        "la-chart-bar",
        NANO_INSTANCE_LIFECYCLE(peak_decay),
    });

    nano::registerEffect({
        2,
        "filter.height_from_gradient",
        "Height From Gradient",
        "Reconstructs a height field from a gradient field on the GPU. The gradient comes from one of several sources — Radial (outward from a center, magnitude = luma), Level Curves (treats the input as a contour map: the across-curve normal, sign-resolved by a global bias), or an existing vector field: Motion Vectors (the incoming render_outputs/motion), a Normal Map, or a Gradient Field. It takes the divergence and solves the Poisson equation laplacian(h)=div(g) with a coarse-to-fine multigrid (FMG-lite) cascade for the least-squares height. The field is generally non-conservative so there's no exact reconstruction, but the solve produces the best try. Visualizes the result as shaded hillshade relief, grayscale height, surface normals, or its own contour lines.",
        "filter",
        "height,gradient,poisson,reconstruction,relief,normals,multigrid,math",
        "la-mountain",
        NANO_INSTANCE_LIFECYCLE(height_from_gradient),
        nullptr, nullptr, nullptr, &height_from_gradient::eval_visibility,
    });

    nano::registerEffect({
        2,
        "source.shape_fold",
        "Shape Fold",
        "Evolving-shape generator. A baked 3D atlas of resolved shape parameters — axes frequency (x), simplicity (y), and temporal complexity (z) — is interpolated each frame down to a few terms and evaluated as a scalar SDF field on the GPU. An internal clock animates a seamless loop (with an easing/time-warp lever and a soft 'birth' gate for fading edges); an optional autopilot spirals the XY automatically and broadcasts its live position (autopilot_x/y) without mutating the inputs. The field is histogram auto-leveled (median→0) every frame, driven by an exposure, and output as grayscale or a colormap grade (magma/inferno/viridis/plasma/turbo) — the raw field, the square covering the viewport with a domain scale that reveals the periodic structure. Pure generator (no input).",
        "source",
        "generator,sdf,shape,evolving,autopilot,procedural,math",
        "la-shapes",
        NANO_INSTANCE_LIFECYCLE(shape_fold),
        nullptr, nullptr, nullptr, &shape_fold::eval_visibility,
    });

    nano::registerEffect({
        2,
        "source.brutal_fold",
        "Brutal Fold",
        "Brutalist axonometric-prism generator. A baked control surface — axes complexity (x), order (y), and liveliness (z), with a co-folded second structure — is interpolated each frame into an algebraic occupancy field, rendered as solid oblique-depth prisms (\"3D without a vanishing point\") in grayscale, with fog fading distant receding layers toward the light sky tone. The XY pad picks the cell; balance plays the two structures' parallax against each other; extrude sets the recession depth. A seamless bounded loop animates prisms birthing in and out (with an easing/time-warp lever); speeds map through a quadratic bend onto a low max, so it reads best very slow. An optional autopilot spirals the XY and broadcasts its live position (autopilot_x/y) without mutating the inputs. Pure generator (no input).",
        "source",
        "generator,brutalist,architecture,prisms,axonometric,fog,depth,volumetric,autopilot,procedural",
        "la-random",
        NANO_INSTANCE_LIFECYCLE(brutal_fold),
        nullptr, nullptr, nullptr, &brutal_fold::eval_visibility,
    });

    nano::registerEffect({
        2,
        "source.phase_fold",
        "Phase Fold",
        "Emergent limit-cycle phase-portrait generator. The XY pad picks a cell in a baked atlas of level-set limit-cycle fields (x = eccentricity, y = lobedness); the blended scalar field H is shown as a muted diverging height-field backdrop. Over it, the GPU traces streamlines through the induced vector field v = level-set flow + WIND(z) with arrowheads that animate down each line, and integrates the limit cycle itself from a seed on the resting orbit — both as separate, independently toggleable stages. Wind (z) is a non-potential force that distorts the cycle and, past a SNIC bifurcation, kills it (the orbit collapses to a fixed point); bias slides the cycle across contours. An optional autopilot spirals the XY and broadcasts its live position (autopilot_x/y) without mutating the inputs. Pure generator (no input).",
        "source",
        "generator,phase-portrait,limit-cycle,streamlines,flow,vector-field,autopilot,procedural,math",
        "la-project-diagram",
        NANO_INSTANCE_LIFECYCLE(phase_fold),
        nullptr, nullptr, nullptr, &phase_fold::eval_visibility,
    });

    nano::registerEffect({
        2,
        "source.particles.flow_swarm",
        "Flow Swarm",
        "Flow-field-driven GPU particle swarm. Consumes a flow_field rail (the canonical velocity texture produced by phase_fold or any flow generator / modifier) and advects a GPU-resident pool of up to a million particles along it: each particle chases the sampled field velocity with momentum (inertia), captures the input color where it spawns, and respawns at a fresh position when its lifetime expires or it drifts off-field. This separates field GENERATION from field RENDERING — drop a flow modifier between the generator and the swarm to reshape the motion. Particles rasterize as instanced soft quads (solid / circle / gaussian), additive or alpha-over, colored by the captured input blended with a tunable solid (optionally tinted by flow direction). With no flow wired the swarm still renders, drifting only by jitter.",
        "source",
        "particles,swarm,flow,vector-field,advection,gpu,instanced,generator,renderer",
        "la-atom",
        NANO_INSTANCE_LIFECYCLE(flow_swarm),
        nullptr, nullptr, nullptr, &flow_swarm::eval_visibility,
    });

    nano::registerEffect({
        2,
        "source.particles.sweep_chamber",
        "Sweep Chamber",
        "Swept-luma capture/release particle + line sim (successor to Double Chamber). A built-in band-pass SWEEP window travels the input's luma range, capturing one brightness layer at a time into a coarse per-frame field: particles and streamline tracers catch onto the captured detail's ridges, bunch up along them, then get FLUNG when the sweep releases — at either sweep endpoint nothing is captured and everything free-flows on a smooth, seam-free curl-noise eddy field. Single-pixel additive points capture the input colour where they spawn; the whole sim samples one coarse field texture (no per-particle convolutions), so it runs far lighter than its ancestor.",
        "source",
        "particles,sweep,luma,lines,tracers,curl,eddy,flow,advection,gpu,generator",
        "la-meteor",
        NANO_INSTANCE_LIFECYCLE(sweep_chamber),
        nullptr, nullptr, nullptr, &sweep_chamber::eval_visibility,
    });

    nano::registerEffect({
        2,
        "mod.source.spectral_lfo",
        "Spectral LFO",
        "Spectral-morph LFO generator. A baked atlas of ~2178 Serum LFO shapes is laid out per metric by a t-SNE embedding; the manifold position (morph_x, morph_y) selects the surrounding Delaunay triangle and the 3 nearest shapes are spectrally morphed (FFT -> barycentric blend -> IFFT -> geometric straighten) into one LFO curve. A phase accumulator advances at 'rate' (exponential -> Hz; 0 freezes) and samples the curve, publishing a scalar 'output' in [0,1] scaled by amplitude. The 'metric' selector re-lays-out the manifold; 'interpolation' off snaps to a single shape. An optional autopilot orbits the manifold and broadcasts its live position (autopilot_x/y). Pure data module (no GPU, no input).",
        "mod",
        "lfo,oscillator,modulation,automation,morph,spectral,generator",
        "la-signal",
        NANO_INSTANCE_LIFECYCLE(spectral_lfo),
        nullptr, nullptr, nullptr, &spectral_lfo::eval_visibility,
    });

    nano::registerEffect({
        2,
        "mod.shaper.spectral",
        "Spectral Curve",
        "Unary modulation shaper: builds the same spectrally-morphed LFO curve as mod.source.spectral_lfo (atlas of ~2178 Serum shapes, manifold position morph_x/morph_y + metric, FFT->barycentric blend->IFFT->geometric straighten), but indexes it by the 'input' modulation value instead of time. The morphed envelope becomes an arbitrary remapping curve. Pure data module.",
        "mod",
        "modulation,spectral,morph,remap,curve,shaper,envelope",
        "la-chart-area",
        NANO_INSTANCE_LIFECYCLE(mod_spectral),
    });

    nano::registerEffect({
        2,
        "mod.source.bass_sim",
        "Bass Sim",
        "Synthetic Resolume-FFT low-band source, locked to the host beat grid: kicks on a selectable 16th-slot pattern step a peak hold up from a sustained floor, fall back linearly (Resolume 'Fall'), get smeared by an analyzer-lag rise smooth, and wiggle with an optional 8th-note bass groove. Build and test beat-reactive patches (the transient shaper!) without routing audio. Same signal model as the transient shaper's native goldens (sketch/fft_bass_sim.h). Pure data module.",
        "mod",
        "modulation,bass,fft,audio,band,kick,beat,simulator,test,source",
        "la-drum",
        NANO_INSTANCE_LIFECYCLE(mod_bass_sim),
    });

    nano::registerEffect({
        2,
        "filter.mesh.triangulate",
        "Triangulate",
        "Renders a Delaunay triangulation that follows the topology of an input's density — accentuating ridgelines first, then corners, then filling voids. Convolutions build ridge/corner/density feature maps from the input; a persistent seed pool is partitioned by a GPU Jump-Flood Voronoi pass and relaxed by stochastic confidence-gated takeover (seeds stay locked and only teleport when a candidate is a confidently better match — no continuous drift, no swim); Voronoi triple-points give the Delaunay edges, drawn as instanced line quads. Works on video frames or sparse point-cloud inputs alike.",
        "filter",
        "triangulation,delaunay,voronoi,mesh,topology,ridges,stippling,jfa,gpu,stylize",
        "la-draw-polygon",
        NANO_INSTANCE_LIFECYCLE(triangulate),
    });

    nano::registerEffect({
        2,
        "warp.plane_shear",
        "Plane Shear",
        "Analysis-driven shear / rift. Picks a \"natural\" dividing plane (a 2D line) from the input via one of four algorithms — Dominant Edge (global structure tensor), Strongest Edge (Hough), Low-energy Seam, or Content Centroid (PCA) — then shears the two halves on either side of it. Any algorithm can run at a fixed angle (it then only picks the position). The plane is stiff: held between updates and hard-snapped (never lerped) when it retargets at the configurable rate. Only the shear translation animates (CPU-timed one-shot hold / ping-pong / loop, with an optional retrigger on retarget). A signed direction morphs the per-half motion between rift (halves apart), overlap (halves together), and slip (halves sliding along the plane), with separate signed translation + multiplier per half. Rift gaps and overlaps each have selectable fills/blends.",
        "warp",
        "shear,rift,split,plane,slice,warp,analysis,structure-tensor,hough,seam,pca,glitch",
        "la-arrows-alt-h",
        NANO_INSTANCE_LIFECYCLE(plane_shear),
        nullptr, nullptr, nullptr, &plane_shear::eval_visibility,
    });

    nano::registerEffect({
        2,
        "warp.tri_shear",
        "Triangle Shear",
        "Three-plane triangle shear. Discovers THREE natural lines (strongest edges, or lowest-energy seams) biased to enclose a large triangle — a size param weights the large-area reward — then shears the image by chaining the single-plane shear three times, once per triangle edge. Like warp.plane_shear, the triangle is stiff (held then hard-snapped at the update rate) and only the shear translation animates (one-shot / ping-pong / loop, with a retrigger and a trigger-now button). Signed direction morphs each edge's motion between rift, overlap, and slip; rift gaps, border reveals, and overlaps each have selectable fills/blends (defaulting to black).",
        "warp",
        "shear,rift,triangle,tri,three,plane,split,slice,warp,glitch,hough,seam",
        "la-play",
        NANO_INSTANCE_LIFECYCLE(tri_shear),
    });

    nano::registerEffect({
        2,
        "warp.recompose",
        "Recompose",
        "Rule-of-thirds compositional rebalancer. Analyzes the incoming frame on the GPU for where its visual weight actually sits — a saliency field blending edge detail, deviation from the frame's average brightness, and colour saturation — then measures how far that centre of mass sits from the nearest rule-of-thirds intersection, and how far the 3x3 mass distribution sits from an ideal layout. It slices the frame into the nine thirds cells and translates each one independently to push the composition toward balance: a shared correction that provably lands the centre of mass on the power point at full strength, plus a mass-weighted-mean-removed per-cell redistribution, so the Spread control changes the character without ever disturbing that guarantee. Correction is signed — 0 is an exact passthrough, 1 is fully balanced, negative deliberately un-balances — with a separate overshoot multiplier and an axis restrict. The analysis is stiff: held between updates and re-solved at a configurable rate (0 = frozen, manual Trigger only), with temporal smoothing on the solved offsets. Rift gaps, viewport-border reveals, and cell overlaps each have selectable fills/blends. The measured imbalance is published as three scalar modulation outputs.",
        "warp",
        "composition,rule-of-thirds,thirds,balance,rebalance,saliency,analysis,slice,grid,reframe,warp,modulation",
        "la-th",
        NANO_INSTANCE_LIFECYCLE(recompose),
    });

    nano::registerEffect({
        2,
        "source.shape_burst",
        "Shape Burst",
        "Triggered expanding-shape generator — an ADSR 'decay only' you can see. Each trigger fires a ring (circle / square / triangle) that grows from a min to a max scale over a duration, shaped by an easing curve, drawn hard-cut solid then gone; all bursts are concentric about a center pivot. Shares mod.source.adsr's trigger surface (a gate rising edge, a momentary trigger, an 'auto_mode' self-fire that is Off by default — Random is a Poisson stream, Beats locks to the host transport on a beat division — plus voices + Reset/Legato/Poly retrigger for overlapping shockwaves). A manual 0..1 knob directly drives one highest-priority ring for a hands-free pulse (wire a modulation source into it). Composites over black, transparent, a custom colour, or the input.",
        "source",
        "generator,shape,burst,ring,circle,square,triangle,trigger,adsr,decay,shockwave,ripple,pulse,beat",
        "la-bullseye",
        NANO_INSTANCE_LIFECYCLE(shape_burst),
        nullptr, nullptr, nullptr, &shape_burst::eval_visibility,
    });

    nano::registerEffect({
        2,
        "source.pixel.ocean",
        "Pixel Ocean",
        "Pixel-art ocean generator, world-map style. A rotatable coarse pixel grid of flat blue dotted with sparse tiny wave sprites (dot-line / omega / unrolling wind-curl), each hosted at a hashed jittered position inside a stratified spawn-cell lattice — a very uniform spread with no clumping. Shape animation and forward drift run on separate discrete step clocks with per-wave jitter (0 = the whole sea ticks in lock step, 1 = fully staggered); a backwards probability sends some waves against the current. Density gates births per life-cycle so waves always appear at the start of their animation. Fully procedural and stateless per pixel — no particle pool.",
        "source",
        "generator,ocean,water,waves,sea,pixel,pixel-art,retro,map,monster-hunter,procedural",
        "la-water",
        NANO_INSTANCE_LIFECYCLE(pixel_ocean),
    });

    nano::registerEffect({
        2,
        "source.pixel.descent",
        "Pixel Descent",
        "Beat-locked stepping grid: the screen splits into a coarse pixel grid (default 4 columns x 10 rows) with exactly one lit pixel per column, all starting at the top on the downbeat and stepping linearly to the bottom over an N-beat loop (default 8). Unjittered it reads as a solid line sweeping down; per-column timing jitter breaks it up — each cycle every column draws a random eagerness (the chance any given step rushes ahead of the clock by a Jitter-fraction of a step), and a per-step Skip Chance lets one random column near-double-step, a stutter that 'skips a beat'. Eager columns also pop back to the top slightly BEFORE the beat. Deterministic given the transport + seed; composites over black, transparent, a custom colour, or the input.",
        "source",
        "generator,pixel,grid,beat,step,sequencer,descent,line,jitter,skip,clock,loop",
        "la-braille",
        NANO_INSTANCE_LIFECYCLE(pixel_descent),
        nullptr, nullptr, nullptr, &pixel_descent::eval_visibility,
    });

    nano::registerEffect({
        2,
        "source.pixel.rift",
        "Pixel Rift",
        "Coarse-grid ocean waves crossing a hidden mid-rift. The pixel_descent look (a hard cols x rows cell grid, default 4 x 10) hosting pixel_ocean-style wave life: tiny dot and omega sprites (1-3 cells) drift left to right and slightly up in whole-cell steps, spawning and dying on staggered per-wave animation clocks (rate/type/lifespan captured at spawn; density latches like the ocean's — it gates births, never culls). The signature feature is the RIFT: extra virtual columns spliced between the left and right halves of the grid that waves cross at full speed but that never render — a wave slides in from the left, vanishes into the middle, and re-emerges on the right. Free-running dt clocks (not beat-locked); composites over black, transparent, a custom colour, or the input.",
        "source",
        "generator,pixel,grid,waves,ocean,rift,drift,rise,retro,pixel-art,led,dot,omega",
        "la-water",
        NANO_INSTANCE_LIFECYCLE(pixel_rift),
        nullptr, nullptr, nullptr, &pixel_rift::eval_visibility,
    });

    nano::registerEffect({
        2,
        "filter.sim.simulant",
        "Simulant",
        "A faithful re-creation of the original Resolume Wire 'Simulant' patch (quirks intact). NOT a wave equation and NOT a zoom-feedback drift — it is a DIFFERENCE-BLEND + BLUR-DIFFUSION feedback loop: each frame the image is differenced (abs(A-B), which never goes pure black — the load-bearing Simulant/Pixulant quirk) against a blurred copy of the previous frame, and that blur is the outward propagation. The churning accumulator is traced into Sobel lines (Levels → posterize → edge → crop). A Poisson flicker + manual trigger pulse an Attack/Release envelope into the injection. FAITHFUL QUIRK: with stock knobs the envelope is SUBTRACTED and Flicker Min/Max = 0, so a fresh drop just decays — bring it alive with Const Alpha, Flicker Max, or Flicker Invert. Wave Speed / Choke shape the medium; Levels / Line Strength / Line Width shape the lines.",
        "filter",
        "simulant,feedback,difference,blur,diffusion,reaction,lines,edge,sobel,flicker,contour,growth,resolume,wire",
        "la-water",
        NANO_INSTANCE_LIFECYCLE(simulant),
    });

    nano::registerEffect({
        2,
        "filter.blur.smear",
        "Smear",
        "A directional Pixulant. A separable blur along a user-defined axis (angle) then its perpendicular minor axis, with a tilted/asymmetric kernel: the `tail` knob shrinks the forward (head) reach so a bright point trails a streak behind it instead of a symmetric blob, while `perspective` (tilt) ramps the minor-blur width across the whole frame along the major axis — narrow on the head side, wide on the rear (tilt-shift, not a per-streak comet). Two modes: Blur (two separable compute passes) and Scatter, which reuses the SAME tilted footprint but samples Pixulant-style — salted random displacement + the dive / abs-difference / exposure 'strange colours' cascade animated by `motion`.",
        "filter",
        "smear,blur,directional,motion,angle,tail,perspective,scatter,pixulant,dive,streak",
        "la-wind",
        NANO_INSTANCE_LIFECYCLE(smear),
        nullptr, nullptr, nullptr, &smear::eval_visibility,
    });

    nano::registerEffect({
        2,
        "filter.reconstruct.line",
        "Line Reconstruct",
        "SMAA-like morphological line reconstructor. Classifies every pixel as line / point / step-edge / junction / smooth-gradient (with subpixel center, width and orientation) in small fixed-footprint passes, then re-renders lines and points as crisp, uniform-width, box-AA strokes (the clean '4K-downsampled' look) and de-bands smooth gradients. Classify-then-resolve (SMAA), contrast-normalized (CAS). Great for de-crunching aliased or lightly-blurred line art and graphics; Uniformity trades honest per-line width against forcing every stroke to a target Line Width, Solidify rescues aliased/dashed stroke colour, and Deband collapses staircased gradients. Debug View exposes the classifier stages. Strength 0 is a pass-through.",
        "filter",
        "reconstruct,line,edge,smaa,antialias,deband,detail,morphology,ridge,width,clarity,points",
        "la-bezier-curve",
        NANO_INSTANCE_LIFECYCLE(line_reconstruct),
        &line_reconstruct::is_identity,
    });

    nano::registerEffect({
        2,
        "warp.envelope",
        "Envelope Warp",
        "Warps the image along an arbitrary hand-drawn parametric envelope — the same (x, y, ease) curve editor as mod.shaper.envelope, mapping source position to warped position (the straight diagonal = no warp). Symmetry modes mirror one curve about the center per axis (Horizontal / Vertical / XY), apply two independent curves (X and Y mirrored, or Rect edge-to-edge), or wrap the curve radially by distance from a movable center (1 on the graph = half the longer axis). Non-monotonic curves FOLD the image over itself (painter's order — later curve segments draw on top). Each envelope segment becomes one instanced quad rasterizing a 1D coordinate map (the per-segment exponential ease is analytically inverted in the fragment), then a single compute resolve composes the axis maps and samples the input once. Amount morphs identity <-> full curve (wire an LFO to breathe); uncovered regions either stretch the image edge or cut to transparent.",
        "warp",
        "warp,envelope,curve,remap,distort,bulge,pinch,fisheye,squeeze,mirror,radial,fold,stretch,displace",
        "la-bezier-curve",
        NANO_INSTANCE_LIFECYCLE(envelope_warp),
        &envelope_warp::is_identity,
        nullptr, nullptr, &envelope_warp::eval_visibility,
    });

    nano::registerEffect({
        2,
        "filter.blur.lens",
        "Lens",
        "A full photographic-lens model: shaped-aperture depth-of-field bokeh (Vogel-disc gather with cat's-eye/swirl/anamorphic/LoCA), lens flare & glare (in-frame veiling glare plus an off-frame spectral sun with veil, directional glow, aperture-blade diffraction streaks and a ghost chain), halation & bloom, geometric distortion + transverse chromatic aberration, and a filmic finish (mechanical vignette, highlight desaturation, Reinhard→ACES tonemap, film grain). Everything composites in linear HDR before the film curve. Presets (SMC prime, vintage swirl, anamorphic, dreamy, clinical) set a whole look at once from the UI.",
        "filter",
        "lens,bokeh,dof,depth,blur,flare,glare,halation,bloom,chromatic,vignette,tonemap,film,anamorphic,swirl,vintage,grain",
        "la-camera",
        NANO_INSTANCE_LIFECYCLE(lens),
        &lens::is_identity,
    });

    nano::registerEffect({
        2,
        "source.mesh.monolith",
        "Monolith",
        "Massive env-lit 3D structure — the 1:4:9 monolith slab from 2001 or "
        "a regular triangular pyramid — deferred-shaded over the input with "
        "up to three concentric echo shells. The default material is void-"
        "black obsidian that reads entirely through fresnel reflections of "
        "the environment (wire any texture into Env, or it reflects the "
        "input itself), a movable sun with a specular glint, height/depth "
        "fog that swallows the top, and god rays carving around the "
        "silhouette. Opacity runs solid to clear refracting glass. Vantage "
        "and Loom give the worm's-eye towering camera. Motion is an eased "
        "one-way yaw Arc that snaps back, a two-axis incommensurate Tumble, "
        "or an Arcing Tumble; Sync locks one cycle to N bars.",
        "source",
        "generator,3d,mesh,monolith,pyramid,tetrahedron,glass,reflection,env,fresnel,fog,godrays,rotate,tumble,concentric,2001,massive",
        "la-cube",
        NANO_INSTANCE_LIFECYCLE(monolith),
    });

    nano::registerEffect({
        2,
        "source.sdf.plume",
        "Plume",
        "Raymarched volumetric shape generator — a sphere sheathed in "
        "ridged, morphing flakes, authored as a live displacement field on "
        "an octahedral shell map and rendered by sphere-tracing a signed-"
        "distance volume. The first effect on the SDF volume renderer; "
        "resonant bounce lighting, two-tier atmosphere and materials land "
        "milestone by milestone. Composites over the input (or generates "
        "on transparent black).",
        "source",
        "generator,3d,sdf,raymarching,volume,sphere,ridges,feather,morph,procedural",
        "la-feather",
        NANO_INSTANCE_LIFECYCLE(plume),
    });

    nano::registerEffect({
        2,
        "source.sdf.plume_field",
        "Plume Field",
        "The plume sculptor as a standalone SDF provider: authors the same "
        "ridged, morphing displaced-sphere field (octahedral shell map "
        "baked into a signed-distance volume) and publishes it on the "
        "`sdf_field` rail instead of rendering it. Wire into an SDF "
        "renderer downstream (e.g. Plume) to split the geometry from the "
        "camera/light/atmosphere. Its own Simulate mode runs tracers over "
        "the surface — streaming downhill, carving valleys deeper and "
        "building ridges, leaving glowing streamline trails. Video passes "
        "through untouched.",
        "source",
        "generator,3d,sdf,field,provider,producer,sdf-field,shell,ridges,morph,procedural,simulation,erosion,tracers,streamlines",
        "la-globe",
        NANO_INSTANCE_LIFECYCLE(plume_field),
    });

    nano::registerEffect({
        2,
        "source.sdf.helio_field",
        "Helio Field",
        "Simulated sun as an SDF provider: a 2D magnetohydrodynamic "
        "simulation on the sphere — a fluid stirred by granulation and "
        "sheared by differential rotation carries a frozen-in magnetic "
        "field whose lines ARE the surface: ridges ride the live field "
        "lines, combed into filaments by magnetic tension, bunched and "
        "swirled into eddies. Storms self-ignite where a line kinks "
        "harder than it is strong, burn along the line as tall glowing "
        "curtains, and quench by reconnecting the kink away — one "
        "Excitability dial spans quiet sun to self-resonant. Publishes "
        "on the `sdf_field` rail; wire into an SDF renderer downstream "
        "(e.g. Plume). Video passes through untouched.",
        "source",
        "generator,3d,sdf,field,provider,producer,sdf-field,simulation,"
        "sun,solar,magnetic,mhd,field-lines,eddies,advection,fluid,"
        "storms,flare,excitable,criticality,aurora",
        "la-sun",
        NANO_INSTANCE_LIFECYCLE(helio_field),
    });

    nano::registerEffect({
        2,
        "source.sdf.dust_halo",
        "Dust Halo",
        "Intermediate stage on the `sdf_field` rail: wires between an "
        "SDF provider and its renderer and adds shaped dust to the "
        "field passing through — a cloud of sharp glinting motes "
        "hovering off the body with a tunable gap. One shape control "
        "set spans a tilted hovering cap (a beret), a full spherical "
        "shell, and a planetary ring. Motes orbit the halo axis with "
        "Keplerian shear and tumble for twinkle; their aggregate "
        "density scatters the downstream fog and dims the sun softly. "
        "Upstream dust is merged, not replaced. Video passes through "
        "untouched.",
        "source",
        "3d,sdf,field,sdf-field,dust,particles,motes,halo,ring,rings,"
        "planetary,shell,orbit,glint,sparkle,beret,cloud,band",
        "la-circle-notch",
        NANO_INSTANCE_LIFECYCLE(dust_halo),
    });
}

} // extern "C"
