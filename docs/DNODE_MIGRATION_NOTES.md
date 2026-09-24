# Migrating dnode / NanoGraph effects to nano-modules

Field notes from porting shipped **dnode (NanoGraph)** effects into the
nano-modules `com.nano.legacy` bundle — `BicolorGrad`, `Glisten`, and (the big
one) `DoubleChamber`. This is the practical companion to `EFFECTS_STYLE_GUIDE.md`
and `EFFECTS_CATALOG.md`: what tripped us up, where we accidentally dropped
behavior the team relied on, and how to not do that next time.

The two systems are close cousins (dnode is the direct ancestor), but the
porting is a *reimplementation*, not a translation — dnode is a Unity-Bolt graph
that code-generates Metal kernels; nano-modules is hand-written C++ + HLSL
compute/render stages. So the source is a *spec*, and the gaps are subtle.

---

## 1. Read the source as a spec, up front

dnode effects are graphs. Reconstructing behavior from the generated Metal
kernels (`*.txt`) **alone is not enough** — and actively misleading. (In the
catalogue pass, a kernel-only read had `MadScanner` as a PCA blob-fitter; the
decoded graph showed it's a trigger/rate flash-beam generator. Same trap waits
in any port.)

Decode the `.asset` graph first:

- `MonoBehaviour._data._json` is a (lightly wrapped) JSON string holding the full
  graph. Strip the Unity header lines (`%…`, `--- !u!…`), `yaml.safe_load`, then
  `json.loads(...)['graph']`. Scripts live in the catalogue's scratchpad.
- Node types that matter: **`ValueInputNode.Name`** = the exposed parameter
  surface (the control surface to match for parity). **`ExpressionNode.InlineExpression`**
  = embedded math (recurrences, force fields, stop conditions). **`LatchNode`** =
  one-frame feedback register = cross-frame state (particle pools, trails) — the
  "weird hacky buffer persistence". Node *defaults* live in the `$ref` wiring
  (LiteralNodes), not on the node.
- Produce a **complete parameter inventory** + the key expressions **before
  writing code**. The thing you skim past in the graph is exactly the param the
  performer reaches for live (see §4).

For DoubleChamber we had an agent dump all 97 params + the force terms + the L
tracer loop + the spawn selector + the bridger fire logic + the coloring. That
inventory was the difference between "looks plausible" and "behaves right."

---

## 2. Gotchas that cost us time

### 2.1 Never clamp a *simulation* position to the viewport
The single nastiest bug. Spawn positions were wrapped in `saturate(uv)` to keep
them in `[0,1]` for color sampling. But `uv` **is the stored sim position**, so
any particle spawned on a disc larger than the frame got **clamped onto the
viewport edge and then simulated from there** — a hard rectangular pile-up on
the borders (top/bottom first on 16:9, because the aspect factor makes `y` clamp
at a smaller radius than `x`).

- Fix: spawn at the *true* (possibly off-screen) position. Off-screen is fine —
  off-screen particles are just invisible.
- Color/texture sampling tolerates out-of-range uv via the **ClampToEdge
  sampler** — you don't need to `saturate` for that.
- Rule: `saturate`/`clamp` is for *sampling coordinates*, never for a value you
  write back as state.

### 2.2 Work in an aspect-corrected space, not raw uv
A circular boundary / radial field computed in raw uv is an **ellipse on screen**
on any non-square viewport, and a 4-tap gradient sampled with a uniform step is
anisotropic. We moved the whole sim into **"s-space"**: `s = (uv - 0.5) / aspect`
with `aspect = (min(W,H)/W, min(W,H)/H)`; velocity lives in s-space, position is
kept in uv for rendering/sampling. A circle in s is a circle on screen, and
gradient taps at `e * aspect` are equal pixels on both axes.

This also fixed image coupling "not working": the gradient was being sampled in
the wrong space, and far too weak (image gradients are tiny — amplify them).

### 2.3 The IDE clips float sliders to 2 decimals
A `0.001 … 0.05` range gives ~5 usable steps — "the param isn't sensitive
enough." The convention (see `flow_swarm`): expose a **`[0,1]` slider** and scale
to the real range internally (`size = slider * SIZE_SCALE`). 100 usable steps,
fine control at the small end.

### 2.4 Lockstep seeding → everything pulses
If every agent is seeded with the **same initial life/phase**, they all expire
and respawn on the same frame → a visible synchronized pulse. We hit this twice:

- Tracers seeded `time = 1.0` for all → the whole line field blinked in unison.
  Fix: random initial life per tracer.
- Particles seeded **alive at full brightness** (`life_total == life_remain` →
  `life_norm == 1` for all) and scattered full-screen → the whole pool popped in
  at once on load/HMR/count-change, then migrated to the steady state.
  Fix: **seed to match the steady state** — positions on the respawn
  distribution, and a *staggered remaining life against a full total* so
  brightness and death times start spread.

General rule: **seed the pool to look like a frame of the running effect**, not a
uniform initial condition.

### 2.5 A "source" must render with no input wired
Generators that read `tex_in` (for color capture / optional compositing) must
still render when nothing is connected — both because that's correct generator
behavior in the IDE and because the GPU test harness gives a generator **no
input**. We added a `1×1` black fallback texture + a clear-to-black base path.
Without it, `render()` bailed on `!in.valid()` and you saw a stale/garbage
buffer (which is *also* how we first mis-saw the "white screen").

### 2.6 Bound the dynamics — codegen fields blow up
The PetriDish field has a cubic term; combined with a curl term whose factor we'd
left too hot, particles got flung where `pow(|x|,3)` went to infinity → NaNs →
additive white everywhere. Clamp the field input, clamp the per-step force and
velocity, and kill escapees past a safety radius. Codegen kernels often have
implicit bounds (step caps, energy thresholds) — port those too.

### 2.7 Registered shader *names* are bundle-global
`registerShaderSPV("compute", …)` collides if two effects in the same bundle both
use `"compute"`. Prefix them (`dc_trace`, `glisten_vs`). The generated SPV C++
symbols (`TRACE_SPV`) are TU-local statics, so those don't collide — only the
string names do.

### 2.8 Bundle bootstrap is not checked in
A freshly-rebased workspace has an empty `build/wasm` and no fonts. Symptoms that
look scary but are just bootstrap:

- `WebAssembly.instantiate(): BufferSource argument is empty` → a bundle `.wasm`
  is missing. Run `cmake -B build -S .` (fetches deps) then
  `wasm_modules/build_all.sh`.
- `text-engine: te_set_font failed` → `web/public/fonts/*.ttf` missing
  (`scripts/fetch_fonts.sh`).
- The `.aot` files are a **native** speed sidecar; the web path never loads them.
  Don't chase them.

### 2.9 Baking a 2D-strip LUT? Use the GL texel-centre convention, not `u*(N-1)`
The nastiest bug of the `lut_collection` (Wire "LUT 2") port, and invisible for a
long time. The source LUTs are the classic **GPUImage 8×8-tile strips** (a 64³
cube packed into 512×512: blue selects the 64px tile, red/green index within it
— the iOS CoreImage CIPhotoEffect look-up tables). Re-baking them into a real
32³/64³ 3D texture means *resampling the strip*, and the sampler convention is
load-bearing:

- GL / ISF `IMG_NORM_PIXEL` (and the strip's `0.5/512` + `(0.125 - 1/512)`
  addressing math) assume **texel centres at `(i+0.5)/N`** — i.e. a normalized
  `u` maps to texel-space `u*N - 0.5`. Bake with `u*(N-1)` instead and every
  sample lands **~half a texel off**.
- Inside a tile that half-texel is harmless. **At the 64px tile boundaries it
  bleeds the *adjacent blue-tile* into the result.** Dark / low-blue inputs sit
  right at tile 0's edges, so they pull in higher-blue tiles → wrong, posterized
  colours (our Process LUT turned dark-blue **red**; `(0,0,0.1)` gave `(74,17,59)`
  instead of `(0,30,221)`). It corrupts **every preset**, worst in the shadows.
- The fix is one line in the bake's bilinear: `fx = u*W - 0.5` (then clamp),
  never `u*(W-1)`. The *runtime* 3D-texture sample (`uvw=(coord*(N-1)+0.5)/N` +
  hardware trilinear) is correct as-is — a 3D texture has no tiles to bleed
  across; the bug only ever lives in the **2D-strip → cube bake**.

Two lessons that generalise beyond LUTs (both cost real time here):

- **"We match the reference shader" can be false confidence.** Our Python/E2E
  reference model had the *identical* `u*(N-1)` bug, so "GPU output == our model"
  proved nothing against the real Resolume output. Validate against the *external*
  ground truth (probe the source asset directly, or render the user's own frame
  and compare), not against a model that shares your assumptions.
- **A symmetric validator can't catch an asymmetric bug.** "Mono LUT → grayscale"
  passed the whole time — but *every* cell of a mono LUT is gray regardless of
  axis orientation or boundary bleed, so it validates nothing about addressing.
  Add a **chromatic** check: a pure-channel input must stay on-hue (we assert
  Process pure-blue keeps `B > R`). When a port has an orientation/layout degree
  of freedom, test the dimension that the freedom actually moves.

Debugging method that worked (per §5 "render and look"): dump the source strip's
structure directly (tile-grid base values `qx×qy` + within-tile gradients) to
learn its real layout, then render the user's actual frame through candidate
models and compare the images — don't reason about it in the abstract. And rule
things out cheaply: the user noting *"the comp is 8bpp"* killed the bit-depth
theory, and *"harsh on all presets, none in the original"* proved it was
systematic (the bake) rather than per-LUT.

---

## 3. Where we missed features from the original

Every one of these was caught by the person who used the effect **live** — they
remembered the *feel* and the *knobs*, not the code. That's the most important
tool in the box (see §4). What we dropped, and why:

| Missed feature | Why we dropped it | Root cause class |
|---|---|---|
| Tracers follow the **tangent / level curves** (with a small gradient-descent blend), not straight down-gradient | Re-architected tracers as raw field-followers; didn't port the `tangentDir = perp(grad)` + `GradientDescentFactor` blend | **Design liberty** silently changed behavior |
| Tracer **stop conditions** (`value_stop`, `grad_stop`, `time_stop_decay`) — lines die based on how "off" the gradient/luma is | Simplified the trace loop to a fixed length + uniform decay | **Dropped a rule** that wasn't obvious from a quick read |
| Spawn is a **uniform-area disc** (`r = size·√rand`), not a square box | Used the naive box; didn't check the spawn shape in the graph | **Didn't inventory** a "shape" detail |
| **Spawn-on-line** (`spawn_on_line` / "P To Line Rate") — particles respawn onto existing tracer vertices | Deferred; nearly forgotten | **Cross-system coupling** is easy to miss |
| **Image smoothing** stages feeding the gradient | Sampled the raw image (noisy, lazy) | **Quality stage** that reads as optional but isn't |

Two recurring shapes here:

- **Design liberties drop behavior silently.** "Tracers as field streamlines" is
  a *nicer* idea than "image-contour tracer" — but it quietly removed the
  level-curve feel the team's muscle memory expected. Re-architecting is fine
  (and welcome — these ports are "v2"s), but each liberty needs to be a
  *conscious, flagged* decision checked against the original's intent, not an
  accident of reimplementation.
- **The boring details are load-bearing.** Spawn *shape*, stop *thresholds*,
  smoothing *stages* — none of them jump out of the kernels, all of them change
  how the effect reads live.

---

## 4. Strategies to avoid parity gaps

1. **Inventory before you implement.** From the decoded graph, list *every*
   `ValueInputNode` and what it does, and every `ExpressionNode` rule
   (recurrences, force terms, stop/spawn/recycle conditions, coloring). Port the
   list, not your mental model of the effect. A param you can't explain is a
   param you're about to drop.

2. **Keep the original's knobs even if defaulted off.** When porting a control
   you're unsure about, include it with a safe default (`0`/off) rather than
   omitting it. `value_stop`/`grad_stop`/`time_stop_decay` all default off, so the
   new-default look is clean *and* the parity behavior is one slider away. A
   present-but-off param is recoverable; an absent one is a re-discovery later.

3. **Flag every re-architecture explicitly.** When you deviate from the source
   (we trace the field instead of image contours; we recycle instead of PONK
   output), write it down in the effect header *and* tell the reviewer. The team
   can then say "actually we needed that" before it ships, not after.

4. **Treat the live user as the parity oracle.** Far more reliable than the code
   for *feel*: which sub-features are actually used, which are dead weight, what
   "looks wrong." Demo early and specifically ("does the spawn shape / line
   travel / edge behavior match what you remember?"). Every gap in §3 came from
   them, not from us re-reading the graph.

5. **Build a per-effect parity checklist** (template below) and walk it before
   calling an effect done.

6. **Mind the "used subset."** The catalogue records which parts of multi-feature
   effects are actually used (e.g. DoubleChamber's "particle accelerator" is
   almost always off; PONK output is unused). Port the used subset first and
   deliberately — but still *inventory* the rest so a dropped feature is a choice,
   not an oversight.

---

## 5. Validation workflow that worked

- **Build is the first gate.** DXC compiling HLSL→SPV catches binding/type/syntax
  errors; clang→wasm catches the C++ side. Most mistakes die here.
- **Render and *look*.** Don't reason about pixels — dump a PNG and view it. The
  test harness writes PNGs to `/tmp/gpu-test-dumps/`; `runGpuTest` dumps full
  frames. Half the bugs in §2 were invisible in reasoning and obvious in a frame
  (the white screen, the edge rectangle, the lockstep brightness, the
  gradient-vs-tangent line direction).
- **Render on black to inspect a generator** — a uniform input hides nothing, and
  the harness gives generators no input anyway (which is why the black fallback
  in §2.5 matters for testing too).
- **Reproduce before fixing.** We twice "fixed" the edge pile-up from a wrong
  hypothesis (crescent boundary, then image-edge trap) before the user's
  isolation (*all forces off, oversized spawn disc*) pinned it to the `saturate`.
  A repro that you can toggle is worth more than a plausible diagnosis.
- **Know the harness limits.** The GPU test runner loads **one** bundle and feeds
  **solid-color** inputs, so it can't exercise cross-bundle chains or
  textured-input paths (image coupling, `to_image` steering). Those need IDE
  confirmation with real video — flag them as unverified rather than assuming.
- **Commit incrementally**, one fix per commit, with the diagnosis in the message
  — so a regression bisects to a cause, not a pile.

---

## 6. Per-effect parity checklist

Before calling a port "at parity":

- [ ] Every `ValueInputNode` from the source is accounted for (ported, or
      consciously cut with a note).
- [ ] Every `ExpressionNode` rule is ported: recurrences, force terms, **stop /
      spawn / recycle conditions**, coloring.
- [ ] **Spawn shape** matches (disc vs box vs ring; area-uniform vs clustered;
      anchor/center).
- [ ] **Cross-system couplings** ported (e.g. spawn-on-line, attractor pull).
- [ ] **Cross-frame state** (LatchNode) maps to a clean persistent buffer; no
      lockstep seeding (seed = a frame of the running state).
- [ ] No `saturate`/clamp on a stored **position**; sim runs in aspect-corrected
      space; dynamics are bounded (force/vel clamp, escape kill).
- [ ] Renders correctly with **no input** wired.
- [ ] Sliders are `[0,1]`-normalized with internal scaling (IDE 2-decimal clip).
- [ ] Quality/smoothing stages from the original are present.
- [ ] Baked/resampled assets (LUT cubes, atlases) sampled with the GL
      texel-centre convention (`u*N - 0.5`, not `u*(N-1)`); validated against the
      *external* ground truth, not a model that shares the bake's assumptions
      (§2.9). A chromatic/asymmetric check, not just a symmetric one.
- [ ] Re-architecture decisions are documented in the effect header.
- [ ] Demoed to the live user; "does this match what you remember?" answered yes.

---

*Written after the BicolorGrad / Glisten / DoubleChamber ports. Update it as the
next ports teach us new ways to drop a feature.*
