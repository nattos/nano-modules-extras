# Show Effects Plan

Spec doc for the effects shipping for the first show. **Living document — flesh out before writing code, so the effects can inform each other.**

## Show context

- **Output device**: 4 LED bars, vertical, ~13 segments each, ~1m tall, arranged roughly linearly across a DJ.
- **Mapping**: Resolume pixel-maps the canvas into 4 vertical slices; each bar reads the center column of its slice. Effects don't need to know the exact pixel-mapping geometry — they just need to treat the canvas as "4 vertical slices, one per bar." Sharp horizontal boundaries between slices are invisible after pixel-mapping.
- **Resolution budget**: ~13 visible "rows" per bar × 4 bars. Detail finer than ~1/13 of canvas-height won't read. Big shapes, strong contrast, slow vertical motion.
- **Composition layers** (rendered bottom-up):
  1. **Atmosphere generators** — always-on bed. Continuous, slowly evolving, low-contrast palette.
  2. **Cut-in generators** — dramatic, often triggered. Fire on cues. Stack on top of atmosphere.
  3. **Complicator FX** — post-process. Distort / degrade / re-color what's underneath. Optionally chained.

## Cross-cutting concerns

### Bundle + effect IDs

All show effects ship under the **`com.nano.lights`** namespace. So the fully-qualified IDs are e.g. `com.nano.lights.gen.soft_glow`, `com.nano.lights.fx.block_dehance`. Throughout this doc the short form (`source.light.soft_glow`, `filter.glitch.block_dehance`) is the **relative ID** that appears in C++ `state::init(...)`; the engine prefixes it with the bundle qualifier at registration time (same pattern as the existing `com.nano.nano` / `com.nano.core` bundles).

Practical naming rules:
- Effects prefixed `gen.*` are generators (the "atmosphere" + "cut-in" layers).
- Effects prefixed `fx.*` are post-process (the "complicator" layer).
- Shared helpers (`fx::BeatTick`, `fx::RandomLfo`, etc) live in headers and don't carry the namespace prefix — only registered effects do.

### Per-bar layout

Convention used by every effect that's bar-aware:
- `BAR_COUNT = 4` (compile-time constant in a shared header — `wasm_modules/shaders_common/nano_bars.hlsl`).
- `bar_index = floor(uv.x * BAR_COUNT)` (clamped to `[0, BAR_COUNT-1]`).
- `bar_local_u = frac(uv.x * BAR_COUNT)` — the within-bar horizontal coord in `[0, 1)`.
- Hard step at bar boundaries is fine (invisible after pixel-mapping). Per-bar properties don't need to soft-lerp across the boundary unless an effect specifically wants a continuous look that spans bars (e.g. soft_glow blobs that drift across).
- For effects with per-bar variants ("which bar do I target?"), expose `bar_target` as an `int` (0..3) plus a `bar_target_all` boolean. When `bar_target_all` is true, the effect runs on every bar; otherwise on just the selected one.

### Trigger model

Cut-in effects need a way to be "fired". Three flavors of input — effects pick the subset that fits.

- **`gate` (bool, PrimaryInput)** — held-gate. Rising edge starts the envelope (attack → decay → sustain); the system *holds in sustain* as long as `gate` stays true. Falling edge starts release. Use for held-note style cues (MIDI note-on/note-off, performer holding a knob). Effects with an ADSR shape (e.g. `source.light.plasma_beam_cannon`) should expose this.
- **`trigger` (event, PrimaryInput)** — one-shot. Rising edge fires the full envelope through to its natural end. For an ADSR effect: synthesizes an internal `(attack + decay + sustain_s)` gate pulse and then auto-falls into release, so an instantaneous cue (MIDI button tap, audio onset) still gets a sustain. For non-ADSR effects (bounce_resonator's impulse kick), this is the only trigger needed.
- **`level` (float, 0..1, PrimaryInput)** — continuous "current intensity" input for effects whose visible amplitude tracks an envelope from upstream (audio envelope follower etc). When connected (non-zero), it overrides internal envelope state. Effects without an internal ADSR can use this as their primary drive.

If both `gate` and `trigger` are wired and `gate` is high while a `trigger` event arrives, the trigger is ignored — gate is already driving the envelope. Once gate falls, future triggers are honored.

Auto-trigger fallback:
- **`auto_rate` (float, 0..1)** — Poisson self-trigger per §4.1 of the style guide. When > 0, the effect synthesizes its own `trigger` events at the mapped rate. Useful for atmosphere fill when nobody's at the desk.

For the first show we'll mostly drive `gate` / `trigger` from MIDI through Resolume.

### Beat sync (assumed)

For this show, Ableton Link is plumbed through Resolume into the engine, so `host::barPhase()` / `host::bpm()` carry stable, musically-aligned timing. Effects that need rhythm read them — never `host::elapsed()`.

Standard pattern (style guide §2.2):
```cpp
// per-frame
double dphase = (barPhase - last_bar_phase + 1.0);
if (dphase >= 1.0) dphase -= 1.0;
last_bar_phase = barPhase;
beats_elapsed += dphase * beat_multiplier;  // beats since instance init
```

Triggers fire on integer crossings of `beats_elapsed`. `beat_multiplier` is the standard knob (selectable as `1/4, 1/2, 1, 2, 4` — multiplicative on rate, so `2` = twice as often). Worth packaging as `fx::BeatTick` once we have 2+ effects using it.

### Random-walk LFO helper

Several effects want the §4.2 pattern (sinusoidal LFO whose rate is itself a random walk). Rather than re-inlining in each effect, extract once into:
- C++ helper `fx::RandomLfo` in `<effect_random_lfo.h>` — holds state (`phase`, `rate`, `rate_target`, `walk_phase`), one `tick(dt)` call per frame, `value()` returns the current sine value.
- Params it exposes (the effect can wire to its own schema fields or expose them verbatim): `mean_rate_hz`, `rate_spread` (log-stops), `walk_rate_hz`, `rate_smoothing_time_s`.

Each effect instantiates one per modulator slot. For blobs / particle effects, an array of these.

### Bar hue palette

Effects that vary appearance per bar (`soft_glow` divergence, future per-bar color tints) share a convention:
- A `bar_palette_spread` float (0..0.5) describes how far apart the 4 bars' hues are pulled.
- Per-bar hue offset is `(bar_index - 1.5) * (bar_palette_spread / 1.5)` so the offsets are symmetric around the central pair. (`-spread, -spread/3, +spread/3, +spread` for the 4 bars.)
- Effects sharing this convention can later be driven from a single global "palette" param (out of scope for now; we'll add when we feel the friction).

### Color space

Default to HSV for hue manipulation. `nano_color.hlsl` already has `nano_rgb_to_hsv` / `nano_hsv_to_rgb`. For perceptually-uniform ramps reach for Oklab; for the Niagara/blackbody look, a hand-crafted ramp in HSV is fine (and faster).

### HDR

Atmosphere and cut-in generators that accumulate (soft_glow's blob sum, plasma cannon's beam) should render into an `rgba16float` intermediate so the warm-roll-off / tone-mapped look has headroom. Final output to the surface format is via a tone map at the last step — never a clamp.

### Tap-driven modulation

Every "intensity"-style param ships with a paired `_mod` companion field that's an additive `[-1, +1]` input. Lets the user wire an LFO / envelope rail without overwriting the base level. Pattern:
```
.floatField("intensity",     1.0f,  0.0f, 2.0f, state::PrimaryInput)
.floatField("intensity_mod", 0.0f, -1.0f, 1.0f, state::PrimaryInput)  // tap target
```
The effect uses `clamp(intensity + intensity_mod, 0, intensity_max)`.

---

## Atmosphere layer

### source.light.soft_glow

Continuous warm-blob bed. Slowly-drifting Gaussian blobs that the eye reads as "warm light dappling across the bars." Each pixel accumulates contributions from all blobs and looks up a hue-shifting color ramp ("Niagara blackbody" look — deep red edges, white-yellow cores). A divergence param emphasizes per-bar hue differences when it spikes.

**Visual primitive**: ~8–16 Gaussian blobs, each with a position that random-walks across the canvas (toroidal wrap), a slowly-evolving brightness, and a captured hue offset.

**Motion**: each blob's velocity comes from two `fx::RandomLfo` instances (vx, vy). Not pure sines — random-walk on the LFO rate so two blobs that started in sync slowly drift apart.

**Color ramp (the warmth)**: accumulated intensity at a pixel → HSV color via a curved lookup. Low intensity = deep saturated red; mid = orange / yellow; high = pale yellow → white. Saturation crushes toward 0 as intensity approaches a `white_point` (the Niagara additive-blowout look).

**Divergence**: when `divergence + divergence_mod > divergence_threshold`, each bar's hue lookup is offset by its bar-palette entry. Smooth via `smoothstep(threshold - softness, threshold + softness, …)`. Below threshold all bars share the same ramp.

**Params**

Standard:
- `intensity` (0..2, default 1.0)
- `intensity_mod` (-1..1, tap target)
- `hue_low` (0..1, default 0.0 — deep red), `hue_high` (0..1, default 0.13 — yellow)
- `saturation` (0..1, default 0.95)
- `divergence` (0..1, default 0)
- `divergence_mod` (-1..1, tap target)

Tuning:
- `blob_count` (4..32, default 12)
- `blob_size` (0.05..1.0 isotropic-uv, default 0.4)
- `blob_size_jitter` (0..1, default 0.3)
- `drift_mean_rate_hz` (0.05..1.0, default 0.2) — both axes
- `drift_rate_spread` (0..2 log-stops, default 1.0) — feeds `RandomLfo`'s rate spread
- `drift_walk_rate_hz` (0.01..0.5, default 0.05) — how often each LFO rerolls its rate target
- `drift_x_bias` (-1..1, default 0) — overall sideways flow on top of the random walk
- `ramp_curve` (0.25..4, default 1.5) — power on accumulated intensity → ramp position
- `white_point` (0.5..3, default 1.5) — accumulated intensity at which color crushes to white
- `divergence_threshold` (0..1, default 0.6)
- `divergence_softness` (0.01..0.3, default 0.1)
- `bar_palette_spread` (0..0.5, default 0.2) — shared convention

Debug:
- `debug_show_blob_centers` (bool) — overlays a 1-pixel cross at each blob's center, colored by its assigned hue.

I/O:
- `tex_in` PrimaryInput — atmosphere is additive over input (so you can stack solid_color below for a clean black bed and place soft_glow on top).
- `tex_out` PrimaryOutput
- `render_outputs` PrimaryOutput (motion vectors = velocity at each blob's center, falloff with mask — useful for downstream motion blur)
- `render_outputs_in` PrimaryInput (chain upstream motion)

**Implementation sketch**
- Blob pool on the CPU side (`blob_count` entries). Each blob owns 2× `fx::RandomLfo` for vx/vy, position state, captured hue offset (rerolled every N seconds — `hue_reroll_period`).
- Per tick, integrate velocity → position, wrap toroidally.
- Push positions + sizes + hue offsets to a GPU storage buffer.
- Compute shader, per pixel, into an rgba16f intermediate: sum gaussian contributions, look up ramp at `pow(accumulated, ramp_curve)`, apply bar-palette hue shift weighted by `divergence_strength`, crush to white as accumulated → `white_point`.
- Tone-map + additive-composite onto `tex_in` in a second pass (or one fused pass writing to rgba8 directly when input_alpha is 1 and we don't need HDR storage).
- Motion vector pass (optional, only when `render_outputs` is read downstream): each blob writes its velocity into a small quad around its center; falloff matches the blob mask so the motion field looks like the visible field.

**Open questions** (call before coding)
- Blob lifetime: pure infinite drift with toroidal wrap, or do blobs slowly fade and respawn so the pattern isn't predictable over a long set? Suggest: very slow fade (`life_period` ~30s with `life_jitter`), respawn position uniformly distributed.
- Hue offset re-roll: each blob keeps a captured hue offset for its lifetime, or hue drifts continuously? Suggest: captured at spawn + tiny continuous LFO around it.

---

### source.light.orthomod

Beat-triggered, Hadamard-driven visual pattern generator. Direct descendant of the original Repatch `nicepattern.orthomod` node (`~/Code/nano-repatch/src/customnodes/nicepattern/orthomod.ts`), reframed for our 4-bar canvas and extended with a second matrix that drives the per-segment fill pattern.

**Two co-driven code systems, one global envelope.**

#### System A — channel envelopes (preserves the original behavior)

Generates 4 per-bar **brightness modulation** signals.

- An 8×8 Hadamard matrix (`generateHadamard(8)`) sorted ascending by row complexity (`getComplexity` = within-row 0↔1 transitions). Row 0 is forced to `[1,1,1,1,1,1,1,1]`. Columns are Fisher-Yates shuffled per `seed`. (Same construction as the existing system, ported verbatim.)
- Global envelope position → row: `idx = floor((1 - env) * resolution)`. **Direction preserved from the original**: `env=1 → idx=0 = all-ones solid flash`; as env decays, `idx` walks through increasing-complexity rows.
- Each row's 8 bits split as 4 channels × 2 bits. Bit-pair → waveform:
  - `(0,0)` → OFF (constant 0)
  - `(1,1)` → ON (constant 1)
  - `(1,0)` → square @ `mod_rate_hz` (default 15)
  - `(0,1)` → rectified sine @ `mod_rate_hz`
- Output: `ch[0..3]`, scaled by the global envelope.

The "solid flash on trigger" behavior is the central reason this direction stays as-is: trigger fires → all-ones row at full env → every channel pinned to 1 → brightest possible state. Decay introduces complexity (some channels start wobbling between sin/sqr/off).

#### System B — bar fill pattern (new)

Determines, per bar, which segments are lit.

- Build an MxM Hadamard matrix (M = `hadamard_size`, tuning param, default 32; constrained to powers of 2).
- Group rows into pages of 4: each consecutive 4-row block forms one page (4 bars × M segments). So `P = M / 4` pages from one Hadamard.
- Sort pages ascending by **page entropy** = sum of within-column transitions across the 4 columns. Equal-entropy pages shuffled by `seed`.
- Global envelope → page index: `page_idx = floor(env * P)`. **Direction inverted from System A**: `env=1 → highest-entropy page`, env decays toward lower entropy. At env=0 the rendered output is gated to black regardless of page (see Render below), so the "anticlimactic dark fade" question dissolves.
- Per-bar segment fill: for bar `c`, segment `r`, the visible bit is `page[c][r mod hadamard_size]` — codes **wrap and repeat** when `render_bits > hadamard_size`.

#### Render

For each pixel:
1. Find `(bar, segment)` based on canvas uv. `bar = floor(uv.x * 4)`. Segment maps to the visible vertical range defined by `inset_top` / `inset_bottom` and `render_bits` — see "Bar geometry" below.
2. Look up System B's bit at `(bar, segment)`. If 0 → black, done.
3. If 1, compute final color:
   - `brightness = lerp(env, env * ch[bar], channel_brightness_mod)` — at `channel_brightness_mod=0` every lit cell is uniformly at `env`; at 1, brightness is fully gated by System A's per-bar channel envelope.
   - `scatter_amount = scatter_max * tent(env) * ch[bar]` where `tent(x) = 4 * x * (1 - x)`. Hue scatter peaks when env=0.5; channels with ch=0 contribute no scatter. Each cell gets a stable per-cell scatter via `hash(page_idx, bar, segment, seed)` so the same page always looks the same — not frame-jittery.
   - `hue = primary_hue + (hash_signed) * scatter_amount`.
   - Output: `hsv_to_rgb(hue, saturation, brightness)`.

#### Bar geometry

For sanity around Resolume pixel mapping that's never exactly stretched to fill the viewport.

- `inset_top` (0..0.5 canvas-uv, default 0) — blank space at top of each bar.
- `inset_bottom` (0..0.5 canvas-uv, default 0) — blank space at bottom.
- `render_bits` (1..64, default 13) — how many segments to render in the visible vertical range. Segment `r ∈ [0, render_bits)` maps to `uv.y ∈ [inset_top, 1 - inset_bottom]`.

Bars are full-width-of-their-slice horizontally (no horizontal inset — `floor(uv.x * 4)` already gives clean per-bar columns).

#### Beat sync

- Read `host::barPhase()` and accumulate dphase per the style guide §2.2 pattern.
- `effective_beats_elapsed += dphase * beat_multiplier`
- Trigger fires on every integer crossing of `effective_beats_elapsed`.
- On trigger: `linear_env := 1`, `gate_open := true` (sustain), `active := true`.
- Decay: `linear_env -= dt / decay_time_beats_to_seconds(bpm, decay_time_beats)`. Curve applied at output: `env = pow(linear_env, decay_curve)` (or `release_curve` during gate-off fast release, carried over from the original).

#### Params

Standard:
- `beat_multiplier` (select: 1/4, 1/2, 1, 2, 4 — default 1) — number of triggers per bar.
- `primary_hue` (0..1)
- `saturation` (0..1, default 0.9)
- `intensity` (0..2, default 1) — final brightness scalar. `intensity_mod` tap.
- `decay_time_beats` (0.05..4, default 1.0) — half-life in beats.
- `scatter_max` (0..0.5, default 0.15)
- `channel_brightness_mod` (0..1, default 0.5) — how much System A channels gate per-bar brightness. 0 = uniform across bars, 1 = full channel-gated.

Tuning:
- `seed` (int)
- `resolution` (2..8, default 8) — System A's row count (matches original).
- `hadamard_size` (4, 8, 16, 32, default 8) — System B's matrix size. Affects pool size (P = size / 4) and base segment count.
- `render_bits` (1..64, default 13)
- `inset_top` (0..0.5, default 0)
- `inset_bottom` (0..0.5, default 0)
- `decay_curve` (0.1..4, default 1.5) — sustain/idle envelope curve (style: §1.3 perceptual mapping).
- `release_curve` (0.5..20, default 12) — fast-release curve (gate-off).
- `mod_rate_hz` (1..60, default 15) — sin/sqr modulator rate for System A.
- `manual_phase` (-1..1, default -1) — when ≥ 0, overrides envelope (useful for IDE preview / debugging).

Debug:
- `debug_show_system_a` (bool) — overlay the current System A row as a strip of 8 dots.
- `debug_show_page_index` (bool) — overlay current System B page index as a number.

I/O:
- `tex_out` PrimaryOutput
- `env` (float PrimaryOutput) — current global envelope value.
- `ch1`..`ch4` (float PrimaryOutput) — per-bar channel envelopes (other effects can tap these).
- *(no `tex_in` — pure generator)*

#### Implementation sketch

- CPU-side state holds: `linear_env`, `gate_open`, `phase_accumulator`, last seed+sizes, generated codes (System A rows + System B pages).
- On seed/size change: rebuild both code sets.
- Per tick: advance beat accumulator, check for tick crossings, update envelope.
- Pack per-frame uniforms: env, ch[0..3], page_idx, primary_hue, saturation, scatter_max, channel_brightness_mod, inset_top, inset_bottom, render_bits, intensity, plus a reference to the current System B page bits (as a uint8 array of size `4 * hadamard_size`, packed into a small uniform buffer or a tiny storage buffer).
- One compute shader, per pixel, into `tex_out`. Compute (bar, segment), look up bit, compute color, output.

#### Open questions

- **Hadamard sizes that don't cleanly give 4-row blocks.** `hadamard_size = 4` gives only 1 page (one 4×4 block). 8 gives 2 pages. 16 gives 4 pages. 32 gives 8 pages. The user wants chaos → clean transitions to be perceptible; fewer pages = chunkier transitions. Suggest 32 as a better default than 8 if rendering quality matters; reserve 8 for "punchy" feel.
- **Resolution mismatch between System A and System B.** System A's resolution param (2..8) limits how many Hadamard rows it walks through. System B has its own pool size. These are independent, which is fine, but the user might want them coupled (one "complexity" knob). Defer until we feel friction.
- **Inset units.** Canvas-uv vs pixels? Canvas-uv is consistent with the rest of the codebase (style guide §1.2/§1.4) but pixels would be more intuitive when matching Resolume's mapping. Suggest canvas-uv with a tooltip noting the conversion.
- Is there value in a "chord" output that combines all 4 columns into one rail (e.g. RMS or max)? Defer until we need it.

---

## Cut-in layer

### source.light.tingle_top

Sparkles **bundled at the top of each bar while held**, **released downward on an envelope when ungated**. Individual sparkles don't move — they live and die in place. The visible "cascade" comes from the spawn region growing over time, so newly-born particles appear progressively lower in the bar while older top-region particles age out.

#### The held / released model

A single envelope-driven scalar — **`region_y_max`** — controls the lower edge of the spawn region. Spawn position is uniform random within `y ∈ [0, region_y_max]` (top of bar = 0).

- **Gate ON (rising edge)**: `region_y_max` snaps to `top_band_height` (default 0.1 of bar). Newly-spawning particles are constrained to a thin slice at the top.
- **While gate held**: `region_y_max` stays at `top_band_height`. Older particles further down the bar gradually age out; the visible distribution converges to "bundled at top" over `~particle_life` seconds.
- **Gate OFF (falling edge)**: `region_y_max` ramps from `top_band_height` → 1.0 (full bar) over `release_s`, shaped by `release_curve`. New particles spawn progressively further down. Old top-region particles age out normally. Visible effect: the sparkle cloud "drains" downward.
- **After release completes**: `region_y_max` stays at 1.0 — particles spawn uniformly across the full bar. (Re-gating snaps back to the top.)

Because particles don't move, the "release" is purely a spawn-region animation. Cheaper than a particle-with-velocity simulation, and the visible cascade reads correctly as long as `release_s` is comparable to or longer than `particle_life`.

#### Trigger model

Three inputs, same as the standardized cut-in pattern:
- **`gate` (bool, PrimaryInput)** — the primary control. Rising edge → snap region to top; falling edge → release ramp.
- **`trigger` (event, PrimaryInput)** — one-shot. Synthesizes a `(min_sustain_s)` gate pulse, then auto-falls into release. Use for instantaneous cues that should still produce the bundle-then-cascade gesture.
- **`level` (float 0..1, PrimaryInput)** — continuous alternative. `level >= 0.5` treated as gated; `level < 0.5` treated as released. Provides smooth control from envelope followers without needing event semantics.

Conflict resolution mirrors plasma cannon: gate priority > trigger > level. `auto_rate` (Poisson) synthesizes triggers when no other input is wired.

#### Visual primitives

Same particle pool as `flash_particles` but with different spawn logic (uniform within the dynamic region, not mask-sampled from a texture) and tighter defaults (smaller, faster, simpler color).

- No HSV mixing from captured input color — single `hue` + `hue_jitter` for each sparkle.
- Per-frame alpha jitter on for the "tingle" shimmer.
- Hard or soft mask shape selectable (same `shape_kind` as flash_particles); default Gaussian for the soft sparkle look.

#### Params

**Standard**
- `gate` (bool)
- `trigger` (event)
- `level` (float 0..1)
- `auto_rate` (0..1)
- `top_band_height` (0.01..0.5 of bar, default 0.1) — the held position of the spawn region.
- `release_s` (0.05..4, default 0.8) — duration of the region's downward expansion.
- `release_curve` (0.25..4, default 1.5) — power on the release ramp. 1.0 = linear; >1 = slow start fast end; <1 = fast start slow end.
- `min_sustain_s` (0..2, default 0.3) — for `trigger`-event one-shots, how long to hold gate-on before auto-releasing.
- `default_gate_state` (bool, default false) — when no `gate` / `trigger` / `level` input is wired (or all are at rest), what state to assume. Default `false` = released = region at 1.0 (the current tingle_top behavior). Set `true` to lock the region at `top_band_height` indefinitely (the downward_sparkle preset — particles always spawn at top, regardless of any input).
- `intensity` (0..2) + `intensity_mod` (-1..1).
- `hue` (0..1, default 0.12 — yellow-white) + `hue_jitter` (0..0.5).
- `density` (1..400 active particles per bar, default 60).

**Tuning**
- `bar_target` (0..3, default 0), `bar_target_all` (bool, default true).
- `particle_life_ms` (10..1000, default 200).
- `respawn_delay_ms` (0..500, default 30).
- `life_jitter` (0..1, default 0.4).
- `size` (0.001..0.05 cover-square, default 0.008) — sparkle visual size.
- `size_jitter` (0..1, default 0.5).
- `frame_alpha_jitter` (0..1, default 0.6) — shimmer strength.
- `shape_kind` (select: solid / circle / gaussian, default gaussian).
- `shape_param` (0..1, default 0.7) — same meaning as flash_particles.
- `alpha_curve` (0.25..4, default 1.5).
- `pool_max` (count × 4-bars cap, default 1024).
- `seed` (int).

**Tuning — particle velocity** (defaults zero → stationary tingle_top behavior; non-zero unlocks the downward_sparkle use case)
- `particle_velocity_y` (-2..2 canvas-uv/sec, default 0) — captured at spawn. Positive = falls down. Per-particle integration in the update shader: `pos.y += velocity_y * dt`.
- `particle_velocity_x` (-2..2 canvas-uv/sec, default 0) — sideways drift.
- `velocity_y_jitter` (0..1, default 0) — per-particle multiplicative variance on `velocity_y`. With jitter > 0, some particles fall faster than others — no metronome feel.
- `velocity_x_jitter` (0..1, default 0) — same for x velocity. The "horizontal spray" knob for the downward_sparkle preset.
- `respect_position_bounds` (bool, default true) — when true, a particle dies immediately when its uv leaves `[0, 1]` on either axis or exits its target bar's slice. When false, only lifetime determines death (lets velocity-driven particles linger off-screen until life expires — useful for slow drifts).

**Debug**
- `debug_show_region` (bool) — overlay a faint line at `region_y_max` per bar so you can see the envelope state during tuning.

**I/O**
- `tex_in` PrimaryInput, `tex_out` PrimaryOutput (additive over input).
- `render_outputs/motion` PrimaryOutput, `render_outputs_in` PrimaryInput — passthrough when `particle_velocity_*` is all zero (stationary particles). When velocity > 0 (downward_sparkle preset), each particle emits a motion vector matching its `(velocity_x, velocity_y)` so downstream `motion.blur` creates natural streaks/trails.

#### Implementation sketch

- This is structurally a `flash_particles` variant with **no mask texture**, **a dynamic per-frame spawn region**, and **optional per-particle velocity**.
- CPU state: `region_y_max` (driven by ADSR-style state machine: gate edge → snap/release).
- Per frame: tick the region envelope; push to uniform.
- Update shader (per particle):
  - Same lifecycle as flash_particles (alive → respawn-delay → respawn).
  - On respawn: `spawn_x = uniform(bar_left, bar_right)`, `spawn_y = uniform(0, region_y_max)`. No mask sampling.
  - Capture life, size, hue jitter, **velocity (with jitter)** at spawn — same pattern as flash_particles' captured params.
  - Per frame on alive particles: `pos += velocity * dt`. If `respect_position_bounds`, check for out-of-bounds → kill.
- Render shader: standard `flash_particles` color path. Motion-vector pass emits per-particle `velocity` when non-zero.

When we have flash_particles + block_dehance + tingle_top all sharing the lifecycle + capture-at-spawn pattern, extract the `fx::MaskSeekingParticles` helper noted earlier in the doc.

#### Preset: downward sparkle

`gen.downward_sparkle` was originally specced as a separate effect — now folded in as a parameter preset of tingle_top, since the only differences were "particles move" + "always-on" (which are just specific param values).

Recipe:
```
default_gate_state      = true      // lock region at top_band_height forever (no gate input needed)
top_band_height         = 0.05      // tight spawn band right at the top
particle_velocity_y     = 1.0       // ~1 second to traverse the bar at the median rate
velocity_y_jitter       = 0.3       // some fast, some slow
velocity_x_jitter       = 0.1       // light horizontal spray
respawn_delay_ms        = 0         // continuous fountain, no gaps
particle_life_ms        = 1500      // long enough to traverse at slow rates
respect_position_bounds = true      // kill cleanly when particle reaches the bottom
```

For more "spray" / "comet" feel, increase `velocity_y_jitter` and chain `motion.blur` downstream — the per-particle motion vectors (now emitted automatically when velocity > 0) will streak each particle into a trail. Live performer can fade `density` up/down on a knob.

If a user wants the cinematic "spawns at top, releases downward on gate-off" gesture INSTEAD, they leave `default_gate_state = false` and wire `gate` from MIDI; velocity stays 0. Same effect, different preset.

#### Open questions (defer)

- **Bar-independent envelopes**. Today, gate-on/gate-off applies uniformly across all targeted bars. A future variant could stagger the per-bar envelope phases so the release cascade rolls left-to-right across the 4 bars (chain-reaction feel). Defer; the linked model is the simpler default.
- **Reset region on idle**. After release completes, the region stays at 1.0 indefinitely until next gate-on. Alternative: slowly contract back to `top_band_height` over an "auto-recharge" period so the effect "rearms" for the next trigger. The held-then-release pattern is the cleaner default; auto-recharge can come later if the show wants it.
- **`flash_particles` mask-falloff option** (the previous open question). Lifted from the previous draft: instead of building `source.light.tingle_top` separately, we could extend `flash_particles` with a `mask_falloff_shape` param that synthesizes a top-bias mask internally. Decision stays the same: build separately for the focused param surface; revisit when we feel duplication pain.

---

### source.light.bounce_resonator

Per-bar vertical mass-on-spring **with cross-bar diffusion**. An impulse trigger kicks one bar's band; the energy then bleeds into other bars via a seeded coupling matrix and rings out across all four. Each cross-bar "send" passes through a non-linear filter that can be cranked to add harmonics or pick out specific frequencies. Performer can boost Q toward self-resonance for sustained shimmer.

#### State per instance

- 4 oscillators, each `(y_i, vy_i)`.
- 4 send-filter biquad states, each `(z1_i, z2_i)`. Coefficients are shared across bars; state is per-bar.
- Total: 16 scalars.

#### Continuous-time physics

```
ÿ_i = -ω_i² · y_i                         // per-bar spring
      - 2 · ζ · ω_i · vy_i                 // damping (Q)
      + Σ_{j≠i} K_ij · (f_j(y_j) - y_i)   // diffusion via send filter
```

- `ω_i` = bar i's natural frequency. Set by `base_freq_hz` and `bar_freq_spread` (log spread across the 4 bars) plus per-bar offsets.
- `ζ` = damping ratio. Performer's `Q` knob maps to ζ.
- `K_ij` = seeded coupling matrix entry (below).
- `f_j(·)` = bar j's send filter (non-linear, biquad — below).

#### Coupling matrix (seeded random)

```
for each pair (i, j) with i < j:
  raw = pcg_hash(coupling_seed, i*4 + j)              // → uniform [-1, +1]
  K_ij = K_ji = raw * coupling_strength
K_ii = -Σ_{j≠i} K_ij                                   // row sums to zero
normalize so that max(|K_ij|) = coupling_strength       // bounds spectral radius
```

Negative entries are valid — pairs can push apart (anti-phase) just as much as they can pull together. Row-sum-zero ensures pure diffusion (no global gain). Bumping `coupling_seed` re-rolls the matrix; the performer can audition seeds until they find one that "sings."

#### Per-bar send filter (non-linear biquad)

Each bar's outgoing diffusion signal passes through a saturated biquad before being added to other bars' acceleration:

```
f_i(y) = biquad(tanh(pregain * y), coeffs[filter_type, filter_freq, filter_q])
```

- `pregain` boosts the signal into the tanh, giving the filter character (audio-style — small pregain = clean, large pregain = saturated harmonics).
- `filter_type` (LPF / BPF / HPF, select) shapes which oscillation frequencies couple.
- `filter_freq`, `filter_q` define the biquad coefficients (standard RBJ formulas).
- The biquad is run at the SUB-STEP rate (not the frame rate) so its frequency response is sampling-rate-stable as `sub_steps` changes.

Filter coefficients are shared across all 4 bars (saves params); each bar gets its own `(z1, z2)` state so its filter has its own memory of what it sent.

#### Self-resonance cap

When `Q` is cranked past the slider's 95% mark, `ζ` is allowed to go slightly negative — the system would diverge, but a soft velocity cap rescues it:

```
vy_i = tanh(vy_i / velocity_cap) * velocity_cap     // applied at end of each sub-step
```

Performer hears a steady self-oscillation rather than a runaway scream. Pre-95% slider, `ζ ≥ 0` and the cap is never hit.

#### Integration

Velocity Verlet with **8 sub-steps per frame** (default; tunable). Energy-preserving, handles high `ω` without explicit Euler instability, and the cost is negligible (8 × 16 scalars + a 4×4 matvec + 4 biquads per sub-step = sub-microsecond on CPU).

#### Impulse modes

`impulse_mode` (select: `velocity` / `position`, default `velocity`):
- **velocity**: add `±impulse_strength` to `vy_target`. Smooth physical-pendulum start. The natural bounce feel.
- **position**: snap `y_target` to `±impulse_strength`. Discontinuous "click" start, then bounces back. Sharper, more percussive.

#### Render

Per pixel, per bar:
- Compute distance from `uv.y` to `y_i + 0.5` (band centered on bar mid).
- Gaussian-like brightness: `exp(-(d / band_width)² * 2 / band_softness²)`.
- Output `band_color * intensity * brightness`, additively over `tex_in`.

Soft band edges are allowed here (unlike plasma's hard-step rule) — physical resonance reads better with the gradient.

Motion vectors emitted per pixel: `(0, vy_i * motion_scale)`. Downstream `motion.blur` will streak the bands during fast bounces.

#### Params

**Standard**
- `trigger` (event) — kick the resonator.
- `bar_target` (0..3, default 0).
- `bar_target_all` (bool, default false). Single-bar is the natural performance gesture; the diffusion does the rest. We may end up driving `bar_target` from an external texture sample (audio-energy-per-bar) — keep this knob exposed and animatable.
- `Q` (0..1, default 0.3) — perceptual log mapping; bottom 80% = "normal" range (heavy → long ring); top 5% enters self-resonance with the velocity cap engaged.
- `coupling` (0..1, default 0.3) — overall strength of the diffusion matrix.
- `coupling_seed` (int, default 0) — re-roll the per-pair coupling pattern.
- `cross_pregain` (0..1, default 0.5) — log-mapped to 0..+24 dB into the send filter's tanh.
- `cross_filter_type` (select: LPF / BPF / HPF, default BPF).
- `cross_filter_freq` (0..1, default 0.5) — log-mapped (0.5 Hz .. 30 Hz at the sub-step rate).
- `cross_filter_q` (0..1, default 0.5) — log-mapped (0.5 .. 20).
- `impulse_strength` (0..1, default 0.7).
- `impulse_mode` (select: velocity / position, default velocity).
- `band_color` (rgb, default warm white).
- `intensity` (0..2, default 1) + `intensity_mod` tap.

**Tuning**
- `base_freq_hz` (0.5..20, default 4).
- `bar_freq_spread` (0..1, default 0.2) — log spread across the 4 bars.
- `per_bar_freq_offsets[4]` (-1..1 each, default symmetric small values).
- `band_width` (0..0.5 uv, default 0.1).
- `band_softness` (0..1, default 0.3).
- `position_range` (0..1, default 0.7) — max swing from center as fraction of bar height (`y_i` is clamped before render so the band never leaves the bar).
- `velocity_cap` (0..2, default 0.5) — `vmax` for the tanh self-osc limiter.
- `motion_scale` (0..2, default 1) — multiplier on motion-vector output.
- `sub_steps` (1..32, default 8).
- `auto_rate` (0..1) — Poisson auto-trigger.

**Debug**
- `debug_show_coupling_matrix` (bool) — overlay a 4×4 grid in a corner showing K_ij magnitudes color-coded (red = negative, green = positive).
- `debug_show_state` (bool) — overlay per-bar `(y, vy)` as small text.

**I/O**
- `tex_in` PrimaryInput, `tex_out` PrimaryOutput.
- `render_outputs/motion` PrimaryOutput.
- `bar_y_0..3`, `bar_vy_0..3` floats as PrimaryOutput rails (cheap to expose; lets other effects react to the resonator's state).

#### Implementation sketch

- CPU-side simulation. ~16 scalars × 8 sub-steps × 60 fps = negligible.
- On `coupling_seed` change: rebuild the 4×4 matrix.
- On filter coefficient param change: recompute biquad coefficients once.
- Per frame: 8 sub-steps of Verlet + biquads + self-osc cap.
- Per frame: pack `(y_0..3, vy_0..3)` + render params into a tiny uniform buffer.
- One compute shader for color + motion (rgba8 / rgba16f via the now-standard split-shader approach).

#### Open questions (defer)

- **Per-bar filter offsets.** Shared coefficients now; could spread `cross_filter_freq` per-bar later (each bar sends a slightly different frequency band) for more modal interest. Add only if shared coefficients feel monotonous on hardware.
- **Color from impulse instead of bar.** Currently a kick into bar 2 paints bar 2's local color, and as energy diffuses into bar 0 it picks up bar 0's color. Alternative: impulse carries a color that travels with the energy. Adds state (per-bar `accumulated_color`) but might look more "spatial". Defer.
- **External-texture-driven kicks.** The user noted future plan to sample per-bar energy from an external texture and kick the corresponding bars. Out of scope for v1; just keep `bar_target` and `impulse_strength` exposed as animatable inputs.

---

### source.light.strobe_channel

**Logistic-map-driven single-bar selector.** Exactly one bar lit at any time; which bar is determined by iterating the logistic map a few steps from a slowly-oscillating seed value. With `r` in the chaotic regime (~3.57..4.0), tiny changes in the seed produce wildly different final values — so a smooth ping-pong of the seed yields a chaotically-jumpy bar pattern that visually reads as strobing. Aesthetically adjacent to `source.light.orthomod` (per-bar one-hot lighting) but driven by deterministic chaos instead of Hadamard codes.

#### The math

Each frame:
```
// 1. ping-pong seed: triangle wave between seed_low and seed_high
phase = frac(elapsed_seconds * ping_pong_rate_hz)         // [0, 1)
tri   = abs(phase * 2 - 1)                                 // [0, 1] tent
x0    = lerp(seed_low, seed_high, tri)

// 2. iterate the logistic map from x0
x = x0
for i in 0..iterations:
  x = r * x * (1 - x)

// 3. final x maps to bar index
active_bar = clamp(floor(x * 4), 0, 3)
```

Restarting from `x0` each frame (not accumulating across frames) is the key — it keeps the result a deterministic function of `(elapsed, r, iterations, seeds)`, no hidden state that drifts. The "chaos" comes from the iterated map's sensitivity within a single frame, not from compounding error across frames.

`r` near 4.0 → tent map → maximally chaotic; the active bar will jump erratically as the smooth ping-pong nudges `x0`. Lower `r` (3.5..3.7) → period-doubling regime; the pattern becomes quasi-periodic with discernible structure. `r < 3` → the iteration converges to a fixed point and the active bar barely changes — boring but useful as a "freeze" knob.

`iterations` controls sensitivity:
- 1–2 iterations: nearly smooth (the chaos hasn't kicked in yet).
- 4–8 iterations: classic chaotic look — strobing bar selection.
- 16+ iterations: essentially indistinguishable noise; finer than the eye can track.

#### Params

**Standard**
- `r` (0..4, default 3.95) — logistic map parameter. 4 = max chaos; 3.7 = quasi-periodic structure; <3 = fixed-point / boring.
- `iterations` (1..16, default 6) — how many logistic iterations per frame.
- `ping_pong_rate_hz` (0.05..20, default 0.5) — speed of the seed's triangle-wave oscillation. Drive this with a tap (LFO / envelope) to modulate the strobe tempo.
- `seed_low` (0..1, default 0.1) — lower bound of the seed ping-pong.
- `seed_high` (0..1, default 0.9) — upper bound.
- `flash_color` (rgb, default 1,1,1).
- `intensity` (0..2, default 1) + `intensity_mod` tap.

**Tuning**
- `bar_count` (2..16, default 4) — number of regions the [0, 1] output is split into. Default 4 matches the show's bar count; set to N to use this as a generic chaotic-N-way selector.
- `region_smoothness` (0..0.1, default 0) — when > 0, near a region boundary the active bar's brightness falls off and the next bar's comes up. Default 0 = strict hard-step selection (classic strobe).
- `transition_ms` (0..50, default 0) — optional brief temporal soft-fade when the active bar changes. Default 0 = instant switching (true strobe). Non-zero softens flicker.
- `bar_hue_offsets[4]` (each -0.5..+0.5, default `(0, 0, 0, 0)`) — additive hue offset per bar. Default zero = `flash_color` for all bars; set non-zero per-bar to give the strobe a chord-of-colors palette as it jumps between channels.
- `mode` (select: `full_bar` / `centered_band`, default `full_bar`) — what shape to draw within the lit bar.
- `band_height` (0..1, default 1.0) — used only when `mode = centered_band`.

**Beat sync** (optional)
- `use_beat_sync` (bool, default false).
- `beat_multiplier` (select: 1/4, 1/2, 1, 2, 4 — default 1) — when `use_beat_sync = true`, the ping-pong rate is derived from `host::barPhase()` instead of seconds. Useful for keeping the chaotic strobe locked to the track tempo.

**Debug**
- `debug_show_value` (bool) — overlay the current `x` value as a thin horizontal line across the canvas (helpful for tuning `r` and `iterations` until the look feels right).

**I/O**
- `tex_in` PrimaryInput, `tex_out` PrimaryOutput (additive over input).
- `render_outputs/motion` PrimaryOutput, `render_outputs_in` PrimaryInput — passthrough.

#### Implementation sketch

- All math is CPU-side. Per frame: 1 triangle-wave eval + ≤16 logistic-map iterations + 1 floor → produces `active_bar`. Effectively free.
- Push uniforms: `active_bar` (int), `flash_color`, `intensity`, optional bar hue offsets, `region_smoothness`, `transition_state` (the smoothed previous bar's contribution if transition_ms > 0).
- One compute shader: per pixel, compute `bar_index = floor(uv.x * bar_count)`, output `flash_color * intensity` if `bar_index == active_bar`, else black. Additively blend over `tex_in`.

That's it. The whole effect is a few lines of CPU logic + a trivial shader.

#### Open questions (defer)

- **Multiple simultaneous bars near boundaries.** Could split contribution between adjacent bars (e.g. `weight_bar_n = max(0, 1 - |x*bar_count - (n + 0.5)| / spread)`) for a less-strict, more-glow-y look. `region_smoothness` partially addresses this. Decide on hardware.
- **Map choice.** Logistic is the textbook chaos generator; the tent map (`x' = (x < 0.5) ? 2x : 2-2x`) or the Lorenz attractor would give different textures of randomness. Logistic is the easiest to reason about, so default; expose `map_kind` as a tuning select later if we want others.
- **Initial value as a tap target.** Right now the ping-pong is internal. Could expose `seed_x` as a continuous input that overrides the ping-pong — performer wires an audio-onset envelope and the bar selection chaotically reacts to hits. Probably worth adding (`use_external_seed` bool + `external_seed` float). Defer to hardware test.
- **Per-bar gates.** Could let the user mute specific bars (e.g. only allow bars 0 and 2 to light, treating 1 and 3 as "skip"). Easy `bar_enable[4]` bool array. Defer.

---

### source.light.side_jet

Horizontal jet trail crossing the canvas left → right (or reversed). **JPL-style, not 90s game fire** — diverging cone with shock-diamond pulsation along the axis and fast-evolving turbulent edges. The lo-fi LED projection will only carry a hint of structure (4 bars × 13 segments crushes most detail), but designing for the full-canvas view auto-degrades to a satisfying sweep at LED resolution.

#### Visual model

Each active jet is a 2D shape rendered procedurally, no particles. At any pixel `(px, py)`:

```
dx = head_x - px                            // distance behind the head (canvas-uv)
dy = py - centerline_y                       // vertical offset from jet axis
if dx < 0 or dx > trail_length: skip         // outside trail extent

cone_half_width   = head_width + tan(half_angle) * dx
if abs(dy) > cone_half_width: skip           // outside cone

radial_falloff = exp(-pow(dy / cone_half_width, 2) * radial_sharpness)
axial_decay    = pow(1 - dx/trail_length, axial_decay_curve)
diamonds       = 1 + diamond_amp * cos(TAU * (dx / diamond_period + shimmer_phase))
turbulence     = 1 + turb_amp * (fbm2(dx * turb_scale, dy * turb_scale + turb_phase) - 0.5)

brightness     = radial_falloff * axial_decay * diamonds * turbulence
contribution   = brightness * color
```

Per-frame `shimmer_phase` and `turb_phase` advance at their own rates (`diamond_shimmer_rate_hz`, `turbulence_rate_hz`), giving the "very quickly evolving shape" — the diamonds slide along the axis and the turbulent edges churn independently. No pre-baked LUT, no LOD — pure procedural so motion is smooth at any sampling rate.

Multiple jets can be on screen at once (the trigger model decouples spawning from transit time). Each jet contributes additively into an `rgba16f` intermediate; final tone-map to surface format.

#### Trigger / spawn / cull

- `trigger` event → allocate the next free slot in a fixed pool of up to `pool_size` jets (default 4).
- Each slot stores `(start_time, direction, centerline_y, color_seed, transit_seconds)`. `head_x` is computed live from `(now - start_time) * dir / transit_seconds`.
- Jet is culled when `head_x` is past the opposite edge by `trail_length` (whole trail has cleared).
- `auto_rate` Poisson-fires `trigger` events.
- `direction = random` picks each spawn independently — so bursty triggers can give alternating- or same-direction salvos depending on the dice.

#### Params

**Standard**
- `trigger` (event)
- `auto_rate` (0..1) — Poisson auto-fire.
- `transit_seconds` (0.05..3, default 0.4) — time for the head to traverse the full canvas. Default tuned for "fast pan".
- `direction` (select: `LtoR` / `RtoL` / `random`, default `random`)
- `centerline_y` (0..1, default 0.5)
- `centerline_y_jitter` (0..0.5, default 0.1) — per-jet random Y offset.
- `color_core` (rgb, default warm white `1.0, 0.95, 0.85`).
- `color_edge` (rgb, default cool blue `0.4, 0.6, 1.0`) — mixed in at low brightness regions for the schlieren / shock-front feel.
- `intensity` (0..2) + `intensity_mod` tap.

**Tuning — shape**
- `head_width` (0..0.1 canvas-uv, default 0.015) — jet width at the head (nozzle).
- `cone_half_angle_deg` (0..30, default 8) — divergence angle.
- `trail_length` (0..2 canvas-uv, default 0.6) — how far behind the head the trail extends.
- `axial_decay_curve` (0.25..4, default 2) — power on trail intensity falloff.
- `radial_sharpness` (1..16, default 4) — exponent inside the radial gaussian. Higher = harder edges.

**Tuning — evolution**
- `diamond_amp` (0..1, default 0.5) — strength of Mach diamond pulsation along the axis.
- `diamond_period` (0..0.3 canvas-uv, default 0.05) — wavelength of the diamonds.
- `diamond_shimmer_rate_hz` (0..30, default 10) — how fast the diamond phase slides along the axis (apparent motion of the diamonds).
- `turbulence_amp` (0..1, default 0.3) — strength of fbm noise modulation on the cross-section.
- `turbulence_scale` (1..32, default 12) — spatial scale of the noise (higher = finer chaos).
- `turbulence_rate_hz` (0..30, default 8) — temporal evolution rate of the turbulence.

**Tuning — pool**
- `pool_size` (1..16, default 4) — max concurrent jets.
- `seed` (int) — RNG for centerline jitter, direction-random, turbulence offsets.

**Debug**
- `debug_show_axis` (bool) — draws a 1-px line at each active jet's centerline.

**I/O**
- `tex_in` PrimaryInput, `tex_out` PrimaryOutput (additive over input).
- `render_outputs/motion` — for each pixel inside an active jet's cone, emit a motion vector `(jet_dx_per_sec, 0)` in canvas-uv-per-second. Downstream `motion.blur` will streak the head naturally without needing per-effect blur logic.
- `render_outputs_in` PrimaryInput (chain upstream motion).

#### Implementation sketch

- CPU-side jet pool. On trigger, allocate slot; on cull, free.
- Per frame, push active jets to a small storage buffer (16 × 32 bytes = 512 B max).
- One compute shader. Per pixel: iterate active jets, sum contributions. Tone-map. Output.
- Motion vector emission in the same shader (writes both `tex_out` and `motion_tex` — same pattern as `flash_particles` with split-shader format substitution).

#### Open questions (low priority)

- **Two-color (core/edge) mix mechanism.** I sketched it as a brightness-driven lerp (`final = lerp(color_edge, color_core, brightness)`). Alternative: position-along-axis-driven (`color = lerp(color_edge, color_core, 1 - dx/trail_length)` so head is core color, tail is edge color). The brightness-driven feel matches schlieren imagery better; the position-driven feel matches chemical-rocket plume photos. Default brightness-driven.
- **Per-jet color randomness.** Could vary each jet's hue slightly via `color_seed` for visual variety in salvos. Easy to add; not in the default param set yet.
- **Coupling with `bar_target`.** Other cut-ins use `bar_target` to fire just one bar. This effect spans bars by design, so `bar_target` is omitted. If a single-bar use case appears (jet stays within one bar's slice), we'd add `transit_axis` (full / single_bar) but it's a different effect spec at that point.

---

### source.light.plasma_beam_cannon

90s anime power-up — per-bar vertical beam that snaps from a focused seed to a fully-lit bar, holds, then crumbles via simulated "breaks" eating the beam, with a flickering tail. **Strict rule: every visual transition is a hard on/off step. No alpha, no soft edges, no fades.**

The arc is the gestalt: small seed → SNAP big → solid → crumble + flicker → black.

#### ADSR timeline

All 4 bars share one ADSR clock when `bar_target_all = true` (the default) — only the per-bar break particle positions differ. Per-bar independent triggering is deferred (out of scope for v1).

| Phase | Default duration | What renders |
|---|---|---|
| Attack | 0.15s | Small focused beam at `seed_y` (height = `seed_height`), full brightness from frame 1. Small Y wobble (`attack_wobble_amp` at `attack_wobble_rate_hz`). |
| Decay  | 0.10s | Beam height ramps linearly from `seed_height` → 1.0 (full bar). Hard ramp — no easing. |
| Sustain | gated, or `sustain_s` for one-shot | Full bar solid `beam_color`. Particles don't exist yet. |
| Release | `release_s` (default 1.5s) | Break particles spawn + simulate. Solid breaks paint black over their range. Last 30% adds duty-cycle flicker. |

#### Trigger / sustain model

Two ways to drive the envelope, both wired simultaneously:

- **`gate` (bool, PrimaryInput)** — rising edge starts attack→decay→sustain; the system *holds in sustain* as long as `gate` stays true. Falling edge starts release. Use this for held-note style cues (MIDI note-on/note-off, performer holding a knob, etc).
- **`trigger` (event, PrimaryInput)** — rising edge synthesizes a `(attack + decay + sustain_s)` gate pulse internally, then auto-falls into release. Use this for instantaneous cues (MIDI button taps, audio onsets) that don't carry a duration.

Conflict resolution: if `gate` is high while a `trigger` event fires, the trigger is ignored (gate is already driving the envelope). Once gate falls, future triggers are honored.

#### Break particles (release phase only)

Per-bar pool of `break_count` particles (default 12 per bar), 1D physics in beam-local coords `y ∈ [0, 1]`. Three types:

- **Solid attractor** — paints black over `[y - size/2, y + size/2]`. Pulls neighbors toward itself.
- **Solid repellor** — same eating behavior. Pushes neighbors away.
- **Spacer repellor** — invisible (no painting). Pushes neighbors away. Keeps the solid breaks from collapsing onto each other into a single big void.

Particle state: `(y, vy, size, type, alive, respawn_delay)`. On entering release, the pool is freshly initialized: types assigned by `attractor_fraction` / `spacer_fraction` (remainder = solid repellors), Y positions seeded uniformly, sizes from the length controller (below), velocities zero.

Force model — pairwise, O(N²) but N≤32:
```
for each pair (i, j) within interaction_radius:
  dy = y[j] - y[i]
  d  = max(|dy|, min_dist)
  sign_term = (type[j] == attractor) ? +sign(dy) : -sign(dy)
  force[i] += sign_term * force_strength / (d * d)
vy[i] = vy[i] * exp(-damping_per_s * dt) + force[i] * dt
y[i]  = clamp(y[i] + vy[i] * dt, 0, 1)
```

Particles that die (size shrinks below `min_size`, or they hit `respawn_delay`-driven cycling) respawn at a uniformly random Y with size driven by the length controller.

Run on the CPU. With N=12 per bar × 4 bars = 48 particles total, GPU sim is overkill.

#### Length-target controller

Drives total break length toward a target curve over the release phase. Stochastic — nudges, doesn't force.

```
target_length(release_t) = lerp(length_target_start, length_target_end, pow(release_t, length_target_curve))
current_length           = sum(size for alive solid particles in this bar)
error                    = target_length - current_length

# On respawn:
new_size = clamp(base_size + error * spawn_response + noise, min_size, max_size)

# Per-frame on alive particles:
size += error * grow_response * dt
size = clamp(size, min_size, max_size)
```

Default curve: linear from 0.1 → 0.7 of bar height across the release.

#### Flicker (release tail)

Starts at `release_t = flicker_start_t` (default 0.7). Hard on/off — never partial.

```
flicker_t   = clamp01((release_t - flicker_start_t) / (1 - flicker_start_t))
duty        = lerp(flicker_duty_start, flicker_duty_end, flicker_t)
flicker_phase += dt * flicker_freq_hz                  // accumulator (style guide §2.1)
flicker_on  = frac(flicker_phase) < duty               // hard step
```

When `flicker_on == false`, the entire bar is forced to black for that period — overrides beam, breaks, everything.

#### Render

Per pixel, per bar, in order:
1. Not in attack/decay/sustain/release at all (idle) → black, done.
2. `flicker_on == false` (release-tail flicker is currently off) → black, done.
3. Pixel `uv.y` outside beam extent (computed from current ADSR phase) → black, done.
4. For each alive solid particle in this bar: if `|uv.y - particle.y| < particle.size / 2` → black, done.
5. Otherwise → `beam_color` at full brightness.

Hard tests at every step. No alpha computed anywhere.

#### Params

Standard:
- `gate` (bool, PrimaryInput) — held-gate trigger.
- `trigger` (event, PrimaryInput) — one-shot trigger; auto-runs the full envelope.
- `auto_rate` (0..1) — Poisson auto-trigger (synthesizes `trigger` events).
- `bar_target` (0..3, default 0).
- `bar_target_all` (bool, default true).
- `beam_color` (rgb, default warm white — `1.0, 0.95, 0.8`).
- `seed_y` (0..1, default 0.5) — vertical position of the attack seed.

Tuning — ADSR shape:
- `attack_s` (0..1, default 0.15)
- `decay_s` (0..0.5, default 0.10)
- `sustain_s` (0..4, default 0.40) — used only by `trigger` (one-shot); gated mode ignores this.
- `release_s` (0.1..5, default 1.5)
- `seed_height` (0..0.3, default 0.06) — beam height during attack (uv).
- `attack_wobble_amp` (0..0.1, default 0.015) — Y wobble amplitude during attack (uv).
- `attack_wobble_rate_hz` (5..40, default 18).

Tuning — break particles:
- `break_count` (4..32, default 12) — per bar.
- `attractor_fraction` (0..1, default 0.25).
- `spacer_fraction` (0..1, default 0.25). Remainder = solid repellors.
- `min_size` (0..0.1, default 0.015) — beam-local units.
- `max_size` (0..0.5, default 0.12).
- `force_strength` (0..2, default 0.4).
- `damping_per_s` (0.1..10, default 4).
- `interaction_radius` (0.05..1, default 0.3).
- `respawn_delay_s` (0..0.5, default 0.05).

Tuning — length controller:
- `length_target_start` (0..1, default 0.1).
- `length_target_end` (0..1, default 0.7).
- `length_target_curve` (0.5..4, default 1).
- `spawn_response` (0..2, default 1) — error sensitivity at respawn.
- `grow_response` (0..4, default 0.5) — error sensitivity per frame.

Tuning — flicker:
- `flicker_start_t` (0..1, default 0.7) — fraction of release at which flicker begins.
- `flicker_duty_start` (0..1, default 0.8) — duty cycle when flicker begins.
- `flicker_duty_end` (0..1, default 0.05) — duty cycle at end of release.
- `flicker_freq_hz` (1..60, default 24).

Tuning — RNG:
- `seed` (int).

Debug:
- `debug_show_particles` (bool) — overlay particle centers (1px) colored by type (attractor=red, repellor=blue, spacer=gray).
- `debug_show_envelope` (bool) — overlay a tiny horizontal bar at the top showing current ADSR phase + envelope position.

I/O:
- `tex_in` PrimaryInput (additive over input — beam blows out atmosphere during sustain, atmosphere reads during the breaks and quiet moments).
- `tex_out` PrimaryOutput.
- `env` float PrimaryOutput — current envelope position (0 in idle, climbing in attack, full in sustain, decaying in release). Useful for downstream effects to react.

#### Implementation sketch

- CPU state: ADSR phase (idle/attack/decay/sustain/release), `time_in_phase`, `gate_prev` (edge detection), `trigger_pulse_remaining` (synthetic gate for one-shot), per-bar particle arrays.
- On phase entry (attack): reset wobble phase. On phase entry (release): freshly init all particles per bar.
- Per tick: advance `time_in_phase`; advance to next phase when timer expires (or gate falls during sustain). Update particles only in release. Update flicker accumulator.
- Pack uniforms: ADSR phase enum, beam extent (computed from phase + decay_t + wobble), particle data (storage buffer), flicker_on bool, beam color, per-pixel constants.
- One compute shader: per pixel, runs the 5-step decision tree above and writes to `tex_out`.

#### Open questions (low priority — defer)

- **Per-bar independent ADSRs.** Out of scope for v1; revisit if the show wants single-bar cannon fires.
- **Spacer / solid-repellor distinction.** Right now they're identical mechanically (just paint vs no-paint). If the simulation feels off we may want spacers to have different force strength / interaction radius — split the params then.
- **Particle lifecycle.** Today's model: particles never "die" during release, only respawn cycles via the length controller. Alternative: hard lifetimes per particle so the pool churns visibly. Defer until we see the simpler version on hardware.

---

## Layer³ (three-floor event)

### source.mesh.three_planes

Three isometric quads stacked like a 3D chess board, shaded as VCR-era neon.
The signature visual for **Layer³**, an event at a three-floor venue. The 4 LED
bars mirror the three levels, but that mapping is done externally — as is the
peak-holding EV meter, the camera-orbit envelope, and any image composited above
the top plane. The effect renders the stack and publishes the rails.

**Why a fullscreen SDF pass, not three additive draws.** Rasterising works in
principle (`dst = E + dst*(1-A)` is premultiplied alpha-over, and the stack
order is known), but a single plane needs its halo *additive* over what's
beneath and its black body *multiplicative* — two blend equations, so six draws.
The grade then forces a fullscreen resolve pass over an HDR accumulator anyway.
Against that, one compute pass with no intermediate wins outright, and it also
gets: a halo radius that is a free uniform rather than baked geometry, exact
Euclidean distance at corners (the "morphological smoothness" requirement), and
a three-offset chroma split that costs three cheap re-evaluations instead of
three passes.

**The resolve** — per pixel, bottom-to-top, and the whole reason for the design:

```
acc = tex_in * input_opacity
for i in 1..3:
    acc *= (1 - A_i)   // black body eats lower planes AND their halos
    acc += E_i         // its own outline + halo still emits, over that black
out = nano_vcr_grade(acc, uv, grade)
```

`fill` is signed, so a plane is neon-filled (`+`) or a mask (`-`), never both.
`core_whiten` blows the line core toward white and leaves the colour in the halo
— that single knob is what makes it read as neon rather than vector art.

**Two fields, not one.** The outline and interior coverage use the exact signed
distance, but the halo must NOT: `min()` over the four edges is only C0 where two
are equidistant, so `exp(-|sd|/r)` inherits the field's medial axis and Mach-bands
a visible crease out of every corner. The halo folds the edge distances with a
log-sum-exp soft minimum instead (`halo_smooth`, as a fraction of the halo
radius). It is C-infinity, and its bias is self-limiting — where one edge
dominates it returns the true minimum exactly, so it rounds the crease and leaves
the tuned falloff alone. A plain sum of per-edge kernels is the same softmin with
the temperature pinned to `r`: equally smooth, but it under-reads distance by
`r*ln(4)` and washes the figure out.

**Camera.** Squares in the model XZ plane at `y = -spacing, 0, +spacing`; the
origin is the middle plane's centre, so orbit is about the right point for free.
Orthographic, so projection is one affine map — no divide, no near plane.
Elevation defaults to 35.264° (true isometric). Because the plane centres sit on
the orbit axis, `planeN_y` is **azimuth-independent**; the silhouette half-height
is not, hence both rails.

**Rails.** `plane{1,2,3}_y` (cover-square, signed) and `plane{1,2,3}_half_h`.
Both are viewport-free closed forms, so they are computed and published without
any GPU readback — from `tick()` (so taps see this frame's value) and from
`render()` (so a host that renders without ticking still gets live rails).

**Look.** The grade lives in the shared `shaders_common/nano_vcr.hlsl`:
HDR highlight bleach → **highlight tint** → warmth → asymmetric per-channel soft
clip (which *is* the tone map) → filmic toe/shoulder → scanlines + grain.
Factored out so the post-process sibling below wears the identical look; only
the halo generation differs.

**Highlight tint** is the bleach run backwards, on the same region: the cores
that `core_whiten` + `highlight_desat` just flattened to white get pulled to a
chosen colour instead. Two things about it are non-obvious and both were found
the hard way:

- The tint is an **absolute** colour, not a multiplier and not scaled by the
  pixel's own peak. `nano_vcr_softclip` at any real drive saturates by about
  1.1, so anything still above that resolves to white whatever hue it nominally
  carries — a multiplicative tint keeps every channel hot and therefore keeps
  clipping. The only way to make a blown core read as a colour is to bring its
  weak channels *below* the knee, so the tint replaces rather than scales. The
  useful side effect is that the swatch reads as the result: pick the colour you
  want the tube to be, and dim it for a deeper one.
- The ramp is a smoothstep over `pivot → 1.5*pivot`, not a full stop. Past ~1.1
  the tone map has already flattened everything, so a wide ramp spends its range
  on pixels that are indistinguishable from each other and leaves a white fringe
  between the tinted core and the halo.

---

### `filter.glow.vcr_halo` — "VCR Halo" *(the post-process sibling)*

The same look pointed at an arbitrary image. Everything downstream of the halo is
literally the same code — one `nano_vcr_grade()` call over a linear HDR
accumulator, with the channel split done by sampling the accumulator at three
horizontal offsets. What differs is where the halo comes from, and that
difference is the whole design.

**Convolution, as a weighted octave stack.** Four passes: prefilter (full res →
half res, 13-tap down + soft-knee threshold + chroma boost + tint), a down chain,
a progressive up chain that folds each octave in on the way back, and a fullscreen
composite. That gives

```
S[N-1] = w[N-1]·D[N-1]
S[k]   = tent(S[k+1]) + w[k]·band(k)
glow   = S[0]  ==  Σ_k w[k]·kernel_k(emitter)
```

which is the same shape as the analytic `Σ_j a_j·exp(-d/(R·s_j))` — a weighted
stack of kernels at geometrically spaced radii. The analytic version places the
radii exactly; here they are pinned to octaves and **Halo Radius slides weight
between neighbours instead**, as a Gaussian lobe in `log2(radius)`. That is what
keeps the radius continuously modulatable: weight leaves one octave exactly as
fast as it arrives at the next, so no level ever pops in. The radius curve is
deliberately the same `0.006·40^t` as three_planes', converted to pixels, so the
two sliders agree on what "0.4" means.

**Radius compensation — the one thing the convolution has to be told.**
three_planes evaluates `exp(-d/r)` against distance, so the halo is at full
strength on the line however wide it is. A convolution spreads a fixed amount of
light, and for a thin line an isotropic blur's peak falls as `1/r` — so plain
energy normalisation makes the glow *fade out* exactly as you ask for more of it.
Scaling the gain by `r/r_ref` restores it. It is not universal (it over-brightens
a large bright field, whose blurred peak was already radius-independent), so it is
a knob at 1.0 rather than a constant.

**Outline is a Laplacian pyramid, not a fixed high-pass.** A convolution has no
notion of where an outline *is*, so a filled shape blooms as a soft lump. The fix
is to band-pass the emitter — but band-pass *at what scale*? Doing it in the
prefilter (centre tap vs the 13-tap average) measures a ~2px rim at every
resolution and every radius, which is invisible the moment the halo is wide. So
the subtraction happens per-octave in the up chain instead:
`band(k) = max(D[k] − stretch(D[k+1]), 0)`. The detected edge is then as wide as
the octave, and therefore tracks Halo Radius, since that is what decides which
octaves carry weight. At full outline the coarsest level's low-pass residue is
dropped as well, so a flat field yields exactly zero.

**Cost.** `benchmark_barrel` grew `--module` / `--bundle` / `--state` so any
effect can be measured; the numbers below are the marginal ms per *added*
instance (slope over n = 1, 2, 4, 8, so the fixed GL-blit harness cost cancels):

| effect | 1080p | 4K |
| --- | --- | --- |
| `filter.blur.fast` | 0.118 | 0.469 |
| `filter.local_contrast` | 0.145 | 0.657 |
| **`filter.glow.vcr_halo`** | **0.219** | **0.941** |
| `source.mesh.three_planes` | 0.440 | 1.925 |
| `filter.blur.gaussian` | 1.353 | 10.208 |

So ~2× a dual-filter blur, ~6–11× cheaper than a gaussian, and — the surprise —
**half the cost of the analytic sibling**, whose one fullscreen pass evaluates
twelve segment distances three times over for the chroma split.

Cost is essentially **flat in radius** (0.237 at radius 0, 0.196 at radius 1).
Levels below a thousandth of the peak weight are skipped, but a geometric
pyramid's tail was nearly free to begin with. Flat is the property that matters
for a live rig: sweeping the halo from an envelope cannot spike a frame. `outline`
adds one tap per up pass (0.252). Halo Gain at 0 skips the chain for ~45% off.

**Known limits, all inherent to convolving instead of measuring distance:**
- It shimmers under motion where the analytic version is rock steady (accepted).
- It cannot recover colour the source already bleached out: feed it a white-hot
  neon core and the halo comes back white. Feed it the *un*-bleached image, use
  Glow Tint on the emitter, or use the shared **Highlight Tint** to re-colour the
  clipped cores after the fact (usually with Pivot pulled under 1.0, since a
  tone-mapped input never gets far above it).
- On thick, already-bright content the default gain clips. Threshold up or
  Outline up is the answer, and both are on the front page of the group help.
- Threshold is in HDR units but the input usually is not: a tone-mapped image
  tops out at 1.0, so a "sensible" bloom threshold of 0.55 leaves almost nothing
  to glow. Default is 0.25 for that reason.

---

## Complicator FX (post-process)

*"fx.wave_traveling_down" from the original list was a duplicate of motion-rain — already absorbed into `source.light.motion_blobs` (with `spawn_edge=top, traverse_speed > 0`).*

---

### source.light.motion_blobs *(absorbs the old fx.directional_blur, fx.zoom_blur, and fx.shadow_flyover placeholders)*

**A small pool of traveling soft blobs that can emit motion vectors AND/OR darken what's underneath.** With `motion_strength > 0, shadow_darkness = 0` it's pure "motion rain" — invisible blobs injecting velocity into `render_outputs/motion`, with the visible effect coming from a downstream `motion.blur`. With `motion_strength = 0, shadow_darkness > 0` it's "shadow flyover" — soft dark blobs sweeping across the canvas like passing objects. Both at once gives moving shadows that also blur what's around them.

Renamed from `gen.motion_rain` once we realized shadow_flyover slots cleanly into the same blob-pool machinery — the two modes differ only in which output (motion field vs darkening) the blobs drive.

#### Visual model (configurably visible blobs)

Per blob: `(x, y, vx, vy, radius)`. CPU-managed pool.

- Blobs spawn on a configurable **edge** (`spawn_edge`: top / bottom / left / right). Position on the edge is uniform random; position into the canvas starts at `-radius` (just outside) so blobs enter cleanly.
- Velocity has two components: **`traverse_speed`** (perpendicular to the spawn edge, INTO the canvas) and **`drift`** (parallel to the edge, jittered per-blob). For the original "motion rain" preset, `spawn_edge = top` makes traverse vertical and drift horizontal. For "shadow flyover", `spawn_edge = left` makes traverse horizontal and drift vertical.
- `radius` is `blob_size * (1 + size_jitter * rand)`, captured at spawn.

Per tick: integrate `(x, y) += velocity * dt`. When the blob's position exits the canvas on the OPPOSITE side from where it spawned, it dies and respawns on the original edge with new randomized params.

`density` (0..1) controls target alive-count = `density * pool_max`. When a blob dies, it respawns only if current alive < target.

Optionally: `spawn_edge_random` (bool) picks a fresh random edge per spawn — useful for "shadows zooming through from multiple directions" feel.

#### Outputs (configurable mix)

The blob pool drives two independent outputs:

**Motion vectors** — per pixel, gaussian-weighted sum of contributing blobs' velocities, blended with upstream motion. Same code as before:
```
for each alive blob:
  d = distance(pixel_uv, blob_pos) / blob.radius
  w = exp(-d² * softness_curve)
  total_v += blob.velocity * motion_strength * w
  total_w += w
local_mask    = saturate(total_w)
local_motion  = total_v / max(total_w, 1e-5)
out_motion    = lerp(upstream_motion, local_motion, local_mask)
```
`motion_strength = 0` zeros the contribution → motion passthrough.

**Color darkening** — per pixel, gaussian-weighted sum of blob coverage drives a darkening (or tinting) of the input:
```
shadow_w = sum over blobs of exp(-d² * softness_curve)
shadow_strength = saturate(shadow_w) * shadow_darkness
out_color = lerp(tex_in, shadow_tint, shadow_strength)
```
`shadow_darkness = 0` → tex_in passthrough. `shadow_darkness = 1` and `shadow_tint = (0,0,0)` → blobs paint full black at their centers, soft fade at edges.

Both outputs use the same blob field — set whichever combo of `motion_strength` and `shadow_darkness` you want. Common combinations:

| Use case | `motion_strength` | `shadow_darkness` |
|---|---|---|
| Motion rain (visible smear via downstream motion_blur) | 1.0 | 0 |
| Shadow flyover (soft dark sweeps) | 0 | 0.7 |
| Cinematic "shadow with motion blur" | 0.5 | 0.5 |
| Visible blob preview for tuning | 0 (or whatever) | 0.3 |

#### Params

**Standard**
- `density` (0..1, default 0.4) — target fraction of `blob_count_max` alive.
- `traverse_speed` (0..3 canvas-uv/sec, default 0.7) — base velocity perpendicular to the spawn edge.
- `traverse_speed_jitter` (0..1, default 0.5) — per-blob log-stops variance.
- `drift` (-1..1, default 0) — bias on velocity parallel to the spawn edge.
- `motion_strength` (0..2, default 1.0) — scale on emitted motion vectors. Independent of `traverse_speed`.
- `shadow_darkness` (0..1, default 0) — strength of the color-darkening pass. 0 = invisible blobs (pure motion mode).
- `shadow_tint` (rgb, default 0,0,0) — what color the shadow is. Black = real shadow; non-black = colored fog.
- `blob_size` (0..0.4 cover-square, default 0.12) — aspect-aware radius.
- `spawn_edge` (select: top / bottom / left / right, default top).

**Tuning**
- `blob_count_max` (1..32, default 8) — small enough that the eye can track individual blobs.
- `blob_size_jitter` (0..1, default 0.3).
- `drift_jitter` (0..0.5, default 0.1) — random walk on the parallel-velocity component.
- `softness_curve` (1..16, default 4) — gaussian exponent. Lower = softer/wider; higher = harder edges. Used by BOTH motion and shadow output passes (same blob footprint).
- `spawn_offset` (-0.5..0, default -0.05) — how far OUTSIDE the spawn edge blobs initially sit. Negative = just outside (clean entry).
- `spawn_edge_random` (bool, default false) — when true, picks a fresh random edge per spawn. Useful for chaotic flyover from all sides.
- `seed` (int).

**Debug**
- `debug_show_blobs` (bool) — overlay faint colored disks at blob positions for tuning.

**I/O**
- `tex_in` PrimaryInput, `tex_out` PrimaryOutput.
- `render_outputs/motion` PrimaryOutput.
- `render_outputs_in` PrimaryInput (chain upstream motion).

#### Implementation sketch

- CPU blob pool (struct array, ≤32 entries). Per frame: tick velocities + positions, respawn dead blobs up to `target_alive`.
- Push pool to small storage buffer.
- Single compute shader, per pixel:
  - Iterate blobs once, accumulate both motion contribution and shadow coverage in the same loop.
  - Output `tex_out` (lerp of tex_in toward shadow_tint by `shadow_strength`) and `motion_tex` (lerp of upstream toward local motion by `local_mask`).
  - Use the same split-shader format-substitution trick from `flash_particles` to handle the two different storage texture formats (rgba8 + rgba16f). Or two compute passes — equally fine at this scale.

#### Open questions

- **Per-blob direction variation.** All blobs share the same `traverse_speed` / `drift` (with jitter on speed magnitude, not direction). A future "swarm" variant could give each blob a randomized direction vector. With `spawn_edge_random` covering "random entry side", explicit per-blob direction randomization is probably overkill. Defer.
- **Visible color injection (non-shadow).** `shadow_tint` is currently a single color and uses lerp-toward (darkening). A "highlight" mode that lerps toward a bright color instead would give a "passing spotlight" feel. Could ship as a `tint_mode` select (`darken` / `lighten` / `replace`), but `shadow_darkness` already covers the most common case. Defer.
- **Blob lifetime envelope.** Currently each blob is at full size + intensity for its whole transit. The old shadow_flyover spec had `width_envelope` (constant / wide_then_narrow / narrow_then_wide) to suggest distance-of-object. With our blobs already having size_jitter, the lifetime-envelope adds limited new variety. Defer.

---

### warp.dispersion

**Not chromatic aberration** (the obvious interpretation that was here previously) — instead, a **block-quantized UV-jitter sampler**: tiles the canvas into blocks, picks a stable random offset per block, samples the input at `(block_center + offset)`, fills the block with that single color. With small blocks → crunchy grain / fast blur. With large blocks → mosaic downres. Random offsets can be large enough to cross bar boundaries (pulls in colors from neighboring bars — the cross-pollination is the point).

#### Block layout — the "no sweeping" rule

The slider for vertical (and horizontal) block size is **quantized internally to discrete steps**. The block layout is parameterized by `(block_size, start_offset)`:
- `block_size`: integer pixels (e.g. 1, 2, 3, ..., up to `block_max_pixels`).
- `start_offset`: integer in `[0, block_size)`, says where the first block boundary sits.

Block boundaries at `y = start_offset + k * block_size`.

When the slider lands on a different quantized `block_size` value (e.g. user sweeps slider crossing the threshold from "4 px tall blocks" to "5 px tall blocks"), we ALSO reroll `start_offset` to a fresh random integer. Both axes get this treatment independently.

The reason: if `block_size` slid continuously, block boundaries would visibly sweep up the canvas as the slider moved — a distracting drift that reads as "movement" the user didn't ask for. With discrete steps + re-roll, the layout instead jumps to a fresh arrangement. Performer perceives the change as "the mosaic just reshuffled" rather than "something is sliding". Much friendlier live.

#### Per-block sample

```
block_ix = floor((pixel.x - start_offset_x) / block_size_x)
block_iy = floor((pixel.y - start_offset_y) / block_size_y)
block_center_px = (block_ix * block_size_x + start_offset_x) + block_size_x * 0.5
block_center_py = (block_iy * block_size_y + start_offset_y) + block_size_y * 0.5

hash = pcg_hash3(block_ix + 100000, block_iy + 100000, tick_index ^ seed)
angle = (hash & 0xFFFF) * (TAU / 65536)
mag   = ((hash >> 16) & 0xFFFF) * (1.0 / 65536) * offset_max
offset = float2(cos(angle), sin(angle)) * mag

sample_uv = block_center / viewport + offset
sample_uv = wrap_or_clamp_or_mirror(sample_uv, wrap_mode)

color = tex_in.SampleLevel(sampler, sample_uv, 0)
output = lerp(tex_in[pixel], color, intensity)
```

Every pixel in the same block gets the same color. Small blocks (1–2 px) read as grain; large blocks (1/4 of a bar to whole bar) read as chunky mosaic. Same effect, same shader, knob-driven feel.

#### Temporal rate

`temporal_rate_hz` controls how often the random offsets re-roll:
- 60 Hz (default): new offsets every frame → the grain shimmers / boils continuously.
- 10 Hz: re-rolls 10 times per second → stuttery, "low FPS" feel.
- 0 Hz: frozen offsets → a static grain texture, useful as a stylized still.

Implementation: an accumulator advances `tick_accum += dt * temporal_rate_hz`, increments `tick_index` on each crossing, hash uses `tick_index` instead of frame number.

#### Params

**Standard**
- `vertical_block_norm` (0..1, default 0.1) — quantized internally to one of N discrete vertical block heights.
- `horizontal_block_norm` (0..1, default 0.1) — same for horizontal.
- `offset_max` (0..0.5 canvas-uv, default 0.08) — max distance the random sample can pull from. Above ~0.25 it routinely crosses bar boundaries.
- `intensity` (0..1, default 1) — lerp between original and dispersed output.
- `temporal_rate_hz` (0..60, default 60) — how often new offsets are sampled.

**Tuning**
- `quantization_levels_vertical` (4..64, default 16) — number of discrete block heights the slider quantizes to.
- `quantization_levels_horizontal` (4..64, default 16).
- `block_max_pixels_vertical` (1..512, default ≈ viewport height) — biggest possible block height. Together with `quantization_levels` defines the discrete ladder.
- `block_max_pixels_horizontal` (1..512, default ≈ viewport width).
- `wrap_mode` (select: `mirror` / `wrap` / `clamp`, default `mirror`) — how out-of-bounds sample UVs are handled. Mirror gives the most natural cross-edge feel; wrap is more glitchy; clamp is safest.
- `seed` (int).

**Debug**
- `debug_show_blocks` (bool) — overlay block boundaries as 1-px lines in a contrasting color.

**I/O**
- `tex_in` PrimaryInput, `tex_out` PrimaryOutput.
- `render_outputs/motion` PrimaryOutput, `render_outputs_in` PrimaryInput — pure passthrough of upstream motion vectors. The dispersion shuffles pixel colors; we don't shuffle motion semantics.

#### Implementation sketch

- CPU state: `last_block_size_x/y`, `current_start_offset_x/y`, `tick_accum`, `tick_index`.
- Per frame: quantize the sliders to integer block sizes via `round((1 + (norm * (max - 1))) * step)`. Compare to last frame's value. If either changed → re-roll start_offset for that axis via a CPU RNG (no need for GPU determinism here since the new offset is one-shot).
- Advance tick accumulator; bump `tick_index` on crossings.
- Pack `(block_size_x, block_size_y, start_offset_x, start_offset_y, tick_index, offset_max, intensity, wrap_mode, seed)` into a uniform buffer.
- One compute shader, per pixel: implement the per-block sample above. Linear sampler with the chosen wrap address mode.

#### Open questions (low priority)

- **Quantization ladder shape.** Linear (1, 2, 3, …, max) gets very chunky at high end. Maybe perceptually-curved (1, 2, 3, 5, 8, 13, …) so the slider feels even across its range. Likely needs hardware tuning to decide.
- **Independent x/y temporal rates.** Right now one `temporal_rate_hz` governs both axes' offsets. Could split — would let the user freeze one axis and shimmer the other. Defer.
- **Sample falloff.** Currently the random offset is uniform within a disk of radius `offset_max`. A non-uniform distribution (Gaussian peaked at 0, or doughnut peaked at `offset_max/2`) could change the texture feel — Gaussian = softer, doughnut = more chaotic. Default uniform; let hardware decide if we want to add a `offset_distribution` select.

---

### fx.chrome_wave

> **SHIPPED (reimagined) as `source.light.chroma_wave`.** The spec below (chromatic
> aberration of the *input*) was superseded: the built effect GENERATES a
> prismatic charge/burst "wave bloom" graded from its own density field and
> composites it additively over the input — no input distortion. It is
> polyphonic (up to 32 CPU-managed charge→burst voices that interact in the
> band-phase domain so overlapping hues rotate further), and emits
> `render_outputs/motion` as the optical flow of the band field (with
> perceptual `motion_warp` / `motion_edge_mask` shaping). See
> `native/wasm_modules/chroma_wave/`.

**Charge-and-burst chroma-distortion bloom.** A large soft Gaussian-ish blob grows while gated, then expands rapidly and fades out on release. The blob's curve field drives radial chromatic aberration on the underlying input — gentle classical R/G/B separation while held, rainbow-banded foldback chaos during the release expansion. The blob itself is rendered as a soft light-leak overlay (semi-transparent additive bloom).

The original "recolor input to a chrome ramp" interpretation was wrong — kept the slot, completely retitled.

#### Phases

| Phase | Trigger | Blob size | Chroma chaos | Overlay alpha |
|---|---|---|---|---|
| Idle | gate off, no recent trigger | 0 | 0 | 0 |
| Attack | gate-on rising edge | 0 → `attack_size` over `attack_s` | low | low |
| Charge | gate held | grows at `charge_rate` cap'd at `charge_max_size` | low (classical chroma sep.) | low-to-mid (`overlay_alpha_hold`) |
| Release | gate-off falling edge | rapidly expands ×`release_expand` | ramps high (rainbow foldback) | spike then decay |
| → Idle | when `release_s` elapses | 0 | 0 | 0 |

Trigger event synthesizes the standard `(attack + brief_sustain + release)` gate pulse for one-shot use.

#### Blob curve

At each pixel:
```
center_uv  = position (cover-square)
d2         = distance²(pixel_uv, center_uv) / current_radius²
g          = exp(-d2 * gaussian_sharpness)     // [0, 1], peaks at center, smooth falloff
```
`current_radius` is the live size driven by the phase machine.

#### Chroma distortion (the wave part)

`g` drives a per-channel radial offset. Direction = unit vector from `center_uv` to pixel (radial-outward); magnitude per channel = a wave function of `g`:

```
dir = normalize(pixel_uv - center_uv)    // radial direction
freq = lerp(wave_freq_hold, wave_freq_release, release_progress)
amp  = lerp(shift_max_hold, shift_max_release, release_progress)

phase_r = freq * g + 0.0
phase_g = freq * g + 2.094              // 2π/3
phase_b = freq * g + 4.189              // 4π/3

shift_r = sin(phase_r) * amp * dir
shift_g = sin(phase_g) * amp * dir
shift_b = sin(phase_b) * amp * dir

r = sample(tex_in, pixel_uv + shift_r).r
g_c = sample(tex_in, pixel_uv + shift_g).g
b = sample(tex_in, pixel_uv + shift_b).b
out_color = float3(r, g_c, b)
```

Two regimes:
- **Hold phase**: `freq ≈ 1` (one sub-cycle across the gaussian) and `amp ≈ small`. The sin-phases stay in the early part of the curve — R/G/B shifts grow together as you approach center but never wrap. Result: classical centered chromatic aberration; gentle, focused.
- **Release phase**: `freq` ramps to ~12 and `amp` doubles. The sin-phases complete multiple cycles across the gaussian → R/G/B shifts oscillate, creating concentric **rainbow bands** of color separation. The folding behavior is intrinsic to the sin wave at high frequency — no special foldback math needed; the cyclical sin function gives the rainbow look for free as long as `freq` is high enough.

`gaussian_sharpness` (default 4) controls how localized the blob is — high values give a tight peak with fast falloff; low values are wide and soft.

#### Overlay (the light-leak part)

Additive bloom on top of the chroma-distorted base, weighted by `g`:
```
overlay_alpha = lerp(overlay_alpha_hold, overlay_alpha_release, release_progress)
out_color += blob_color * g * overlay_alpha
```

Gives the "light leak" feel — a soft warm/colored bloom co-located with the chroma distortion. Color is configurable; default warm white.

#### Params

**Standard**
- `gate` (bool) — held = attack + ongoing charge; released = expand + fade.
- `trigger` (event) — one-shot synthesizes `(attack + min_sustain + release)`.
- `level` (0..1) — continuous alternative; `level >= 0.5` treated as gated.
- `auto_rate` (0..1) — Poisson auto-trigger.
- `position` (cover-square vec2, default `(0, 0)`) — blob center.
- `blob_color` (rgb, default warm white `1.0, 0.92, 0.78`).
- `intensity` (0..2) + `intensity_mod`.

**Tuning — phase shape**
- `attack_s` (0..1, default 0.1) — time from gate-on to `attack_size`.
- `attack_size` (0..0.5 canvas-uv, default 0.08) — radius at end of attack.
- `charge_rate` (0..2 canvas-uv/sec, default 0.4) — how fast radius grows while held.
- `charge_max_size` (0..2, default 0.6) — cap on hold-growth.
- `release_s` (0.05..3, default 0.7) — release phase duration.
- `release_expand` (1..6, default 3.0) — radius multiplier reached at end of release.
- `release_curve` (0.25..4, default 2.0) — power on release expansion ramp (higher = slow-start fast-end).
- `min_sustain_s` (0..1, default 0.2) — for `trigger` one-shots, sustain duration before auto-release.

**Tuning — chroma wave**
- `shift_max_hold` (0..0.03 canvas-uv, default 0.005) — base per-channel offset during hold.
- `shift_max_release` (0..0.08, default 0.025) — peak per-channel offset during release.
- `wave_freq_hold` (0..4, default 1.0) — sin cycles across the gaussian during hold (low = classical).
- `wave_freq_release` (0..40, default 12) — peak frequency during release (high = rainbow banding).
- `gaussian_sharpness` (1..20, default 4) — gaussian curve tightness.

**Tuning — overlay**
- `overlay_alpha_hold` (0..1, default 0.2) — light-leak alpha during hold.
- `overlay_alpha_release` (0..1, default 0.5) — peak alpha during release.

**Tuning — RNG**
- `seed` (int) — for any per-frame jitter we end up wanting.

**Debug**
- `debug_show_phase` (bool) — tiny overlay showing current phase + radius + release_progress for tuning.

**I/O**
- `tex_in` PrimaryInput, `tex_out` PrimaryOutput.
- `render_outputs/motion` PrimaryOutput, `render_outputs_in` PrimaryInput — passthrough (the chroma distortion is "where the pixel comes from" not "where it's going," so we don't synthesize motion).

#### Implementation sketch

- CPU phase machine: `(phase enum, time_in_phase, current_radius, release_progress)`. Updates per frame.
- Push uniforms: `position, current_radius, freq, amp, gaussian_sharpness, blob_color, overlay_alpha`.
- One compute shader, per pixel:
  - Compute `g` from gaussian at distance.
  - Compute per-channel `(shift_r, shift_g, shift_b)` from `sin(freq * g + channel_phase) * amp * radial_dir`.
  - Triple-sample tex_in with the three offsets, take per-channel result.
  - Add overlay (`blob_color * g * overlay_alpha`) additively.
  - Output.
- HDR intermediate (rgba16f) recommended so the overlay can accumulate cleanly; tone-map at the end (or skip if the surface format and overlay amounts are tame enough that clipping isn't an issue).

#### Open questions

- **Linear vs radial shift direction.** Currently radial. A `shift_mode` select (radial / linear-along-axis / spiral) would unlock different feels. Spiral is just radial + a tangential rotation, basically the same math we proposed for the old zoom_blur. Defer until we want it.
- **Per-channel phase offsets.** Defaulted to evenly-spaced thirds for that "rainbow ROYGBIV" feel during release. Could expose as 3 tuning params (`phase_r`, `phase_g`, `phase_b`) for asymmetric color schemes — e.g. pure red/blue separation by setting two phases identical and one offset by π. Defer.
- **Damping on `current_radius` after release ends**. Right now the radius drops to 0 when the release phase elapses — sharp termination. Could ease the final return-to-zero for a softer end. Defer (the overlay alpha fades to 0 anyway, so it should read OK).
- **Multiple simultaneous blobs.** Right now one phase machine = one blob. If we ever want overlapping blobs (chord-like trigger stacks), we'd need a pool with separate per-blob phase state. Defer.

---

### filter.glitch.block_dehance

**Cinema/game-glitch rectangles that "dehance" the input** in one of several modes — black-out (the original dropout), mosaic / downres, noise. Each rectangle's mode is sampled probabilistically at spawn, so a single instance can mix all three. Same lifecycle + bright-seeking spawn machinery as `flash_particles` / what `fx.dropout` was originally specced as. The original two-effect plan (`fx.dropout` + a separate full-canvas `filter.glitch.block_dehance`) collapses into this one effect: dropout is just `mode_black_weight = 1`, and the full-canvas macroblock look is already covered by `warp.dispersion`.

#### Lifecycle (mirrors flash_particles)

Each rectangle: `(x, y, w, h, life_remain, life_total, respawn_remain, respawn_total, mode, mode_seed, mode_param_captured...)`. Three states:

- `life_remain > 0` → visible; tick life down.
- `life_remain <= 0 && respawn_remain > 0` → invisible; tick respawn delay.
- both `<= 0` → respawn:
  - **K random uv samples on the mask texture, pick the brightest** (softmax-weighted by `mask_temperature` — same code as flash_particles).
  - **Sample a mode** based on `mode_*_weight` params (see below). Capture mode + per-mode jittered params at spawn.
  - Capture position, width, height, life, respawn totals — same as flash_particles.
  - Capture a per-rect `mode_seed` for any randomization the mode needs (noise pattern, mosaic offset, etc.).

`mask_in` is an optional secondary input; falls back to `tex_in` when not connected.

#### Mode sampling

Three modes with independent weight knobs. At spawn time:
```
weights = [black, mosaic, noise]
total = sum(weights)
r = rand() * total
cumulative = 0
for i, w in enumerate(weights):
  cumulative += w
  if r < cumulative: return mode i
```

Adjust weights to taste — `(1, 0, 0)` reproduces classic dropout; `(0, 0.5, 0.5)` is pure dehance-without-blackout; `(0.33, 0.33, 0.33)` mixes all three. Weights need not normalize manually — the effect normalizes internally.

#### Per-mode render

Per pixel, after determining "covered by alive rect `i`":

**Mode 0 — black/fill** (the original dropout):
```
return fill_color    // rgba; alpha blends with tex_in for partial darken
```

**Mode 1 — mosaic** (downres the covered region):
```
cell = mosaic_cell_size_captured       // per-rect cell size in canvas-uv, captured at spawn
local_uv = pixel_uv - rect_corner_uv
cell_uv  = floor(local_uv / cell) * cell + cell * 0.5    // snap to cell center
sample_uv = rect_corner_uv + cell_uv
return tex_in.SampleLevel(sampler, sample_uv, 0)
```
Every pixel in the same mosaic cell within the rect samples the same input uv → blocky downsampled look constrained to the rect.

**Mode 2 — noise** (replace with random):
```
hash = pcg(pixel.x, pixel.y, rect.mode_seed, noise_temporal ? tick_index : 0)
n    = unpack_rgb(hash)
if noise_color_mode == grayscale: n = vec3(luminance(n))
if noise_color_mode == luma_preserve:
  src_luma = luminance(tex_in[pixel])
  n = n * src_luma / max(luminance(n), 1e-3)
return lerp(tex_in[pixel], n, noise_intensity)
```
`noise_temporal = false` freezes the noise pattern for the rect's lifetime (one captured snapshot of static). `noise_temporal = true` re-rolls every frame (TV-static hiss).

#### Optional per-rect flicker

(Carried over from the old dropout spec — same hard-step duty-cycle flicker overlay on top of any mode.)

`flicker_rate_hz` (default 0 = continuous on) overlays a hard duty-cycle flicker on each rectangle while it's alive. At 0 Hz the rectangle is statically present for its lifetime. At 30 Hz it stutters on/off, giving the more aggressive glitch feel. Stable per-rect seeds so different rectangles flicker out of phase.

#### Params

**Standard**
- `count` (0..64, default 6) — number of rectangle slots in active rotation. 0 = pure passthrough.
- `life_s` (0.05..10, default 1.5).
- `respawn_delay_s` (0..10, default 1.0).
- `life_jitter` (0..1, default 0.3).
- `rect_width` (0.005..1.0 canvas-uv, default 0.18).
- `rect_height` (0.005..1.0 canvas-uv, default 0.06) — wide-and-short default for the classic glitch-bar look.
- `rect_size_jitter` (0..1, default 0.4).
- `mask_temperature` (0..4, default 0.5) — softmax on bright-seeking spawn.
- `mode_black_weight` (0..1, default 0.33) — probability weight for the black-fill mode.
- `mode_mosaic_weight` (0..1, default 0.33).
- `mode_noise_weight` (0..1, default 0.33).

**Tuning — black mode**
- `fill_color` (rgba, default 0,0,0,1) — what to paint inside black-mode rectangles. Default opaque black; alpha < 1 partially darkens instead of full blackout.

**Tuning — mosaic mode**
- `mosaic_cell_size` (0.001..0.2 canvas-uv, default 0.02) — base cell size.
- `mosaic_cell_size_jitter` (0..1, default 0.5) — per-rect captured variance on cell size. With jitter, each mosaic rect downsamples at a different scale.

**Tuning — noise mode**
- `noise_temporal` (bool, default true) — true = TV-static-hiss; false = frozen snapshot per rect.
- `noise_color_mode` (select: rgb / grayscale / luma_preserve, default rgb).
- `noise_intensity` (0..1, default 1.0) — lerp between source and noise. 1 = full noise replacement; <1 keeps some of the original showing through.

**Tuning — common**
- `pool_max` (8..128, default 32) — hard cap; `count` is clamped to this.
- `mask_samples` (4..16, default 8) — K random samples per respawn.
- `flicker_rate_hz` (0..60, default 0).
- `flicker_duty` (0..1, default 0.5).
- `seed` (int).

**Debug**
- `debug_show_rects` (bool) — draw 1-pixel outline around each alive rectangle, color-coded by mode (red=black, green=mosaic, blue=noise).

**I/O**
- `tex_in` PrimaryInput.
- `mask_in` SecondaryInput — optional, falls back to `tex_in`.
- `tex_out` PrimaryOutput.
- `render_outputs/motion` PrimaryOutput, `render_outputs_in` PrimaryInput — passthrough of upstream motion.

#### Implementation sketch

Same scaffolding as the old fx.dropout spec, with two additions:
- **Update shader**: on respawn, also sample mode + capture per-mode jittered params (currently just `mosaic_cell_size` with jitter).
- **Render shader**: branch on the captured mode in the covered-pixel path. Mosaic needs a sampler on `tex_in`; noise needs the `tick_index` uniform when temporal; black needs nothing beyond `fill_color`.

Per-pixel iteration over ≤128 rectangles with a small per-mode branch is still cheap. No need for the vertex/fragment route.

#### Open questions (defer)

- **Per-rect mosaic cell aspect.** Currently `mosaic_cell_size` is isotropic. Tall cells (cell_w ≪ cell_h) give scanline-pixelated feel; wide cells the opposite. Could expose `mosaic_cell_aspect` (-1..+1) for variety. Defer.
- **Captured mode per rect vs per-tick re-roll.** Mode is sticky for a rect's lifetime — feels intentional and readable. If we want chaotic mode-flipping during a rect's life, that's an optional `mode_reroll_rate_hz` knob. Defer.
- **Mode-specific lifetime tuning.** Noise mode at high `flicker_rate` is the most chaotic; black mode lives longer; mosaic somewhere between. Could expose `mode_*_life_mult` per mode if defaults don't read well on hardware. Defer.
- **Additional dehance modes worth adding later**: chroma_strip (luminance-only / desaturate), invert, color_quantize (palette reduction), source_offset (sample from a jittered uv — essentially `warp.dispersion` constrained to the rect). All easy to slot in as new mode IDs + weights.

---

## Build order

Atmosphere first — establishes the bed everything else lives on:
1. `source.light.soft_glow` (single biggest visual win; informs the random-walk LFO helper)
2. `source.light.orthomod` (Hadamard-driven beat-synced visual; also exposes per-bar `ch1..ch4` rails downstream effects can consume)

Then 2–3 cut-ins to validate the trigger model + per-bar conventions:
3. `source.light.plasma_beam_cannon` (most dramatic — proves the trigger / charge-release pattern)
4. `source.light.bounce_resonator` (proves the resonator core)
5. `source.light.side_jet` (proves the per-bar traversal + motion-vector emission)

Then complicators we know we'll lean on:
6. `source.light.motion_blobs` (absorbs three old placeholders: `fx.directional_blur`, `fx.zoom_blur`, and `fx.shadow_flyover`. Same blob pool drives both motion vectors and color darkening — pick modes via `motion_strength` + `shadow_darkness`. Stack `motion.blur` downstream when in motion mode for the visible smear.)
7. `warp.dispersion` (cheap, broadly useful)
8. `filter.glitch.block_dehance` (absorbs the original `fx.dropout` — black is just one of the three dehance modes; weights control mode mix per spawn)

Then the rest as time permits:
9. `source.light.tingle_top` (covers both the original tingle and the downward_sparkle preset via velocity params; bench against `flash_particles` to see if all three end up sharing enough machinery to extract a helper)
10. `source.light.strobe_channel`
11. `fx.chrome_wave` → **shipped as `source.light.chroma_wave`** (reimagined: a self-rendered, polyphonic prismatic "wave bloom" generator rather than an input chroma-distorter — see the spec note below). All 11 effects above are now built.

(`fx.bounce_resonator` was dropped: `source.light.bounce_resonator` became a GPU diffusion network, so the shared spring-oscillator helper no longer exists, and its `impulse_mode = tex_in` already samples the input image per bar — covering the audio-energy-driven gesture this slot was for.)

## Shared helpers we'll likely extract during this work

- `fx::RandomLfo` (effect_random_lfo.h) — used by soft_glow. (strobe_channel was previously listed here but the revised spec doesn't need it — pure logistic-map math.)
- `fx::BeatTick` (effect_beat_tick.h) — wraps the `barPhase` → dphase → tick-counter → trigger pattern with a `beat_multiplier` knob. Used by every beat-synced effect (orthomod first).
- Bar layout helpers in `nano_bars.hlsl` — used by every bar-aware effect.
- Possibly `fx::ParticleSystem` if tingle_top + flash_particles + block_dehance end up with too much duplicated structure.

---

## Open meta-questions

1. **Trigger event field type.** Style guide §0 mentions `eventField`. Need to confirm the existing one supports a clean "rising edge fired once" semantic plus a `level` continuous alternative on the same conceptual input. May need a small refactor.
2. **Auto-trigger Poisson plumbing.** Each cut-in should be able to self-fire when no upstream trigger is wired. Worth standardizing the `auto_rate → Poisson → internal trigger` plumbing into a helper rather than reimplementing in every cut-in.
3. **`source.light.orthomod`'s float outputs.** Today, struct-rail outputs require the canonical schema. The 4 per-bar channel envelopes + 1 global env need either a custom struct schema or 5 separate float fields. Suggest 5 separate float fields (`ch1..ch4`, `env`) — simpler, no schema-shape proliferation. Mirrors the original Repatch node.
4. **Resolume cue routing.** How do MIDI triggers from Resolume reach the effect schema? Out of scope for this doc but worth a note: we'll need a sketch-input or rail mapping that translates Resolume MIDI → float rail values, which the effects' `trigger` event fields read.
