#!/bin/bash
# legacy — ports of shipped dnode/NanoGraph effects (com.nano.legacy).
#
# Built against the Nano effect SDK (see ../../README.md):
#   NANO_SDK=/path/to/sdk ./build.sh        (or ../../build.sh for every bundle)
# OUT_DIR / TMP_DIR override where the .wasm and the generated headers go.
set -e
cd "$(dirname "$0")"
EXTRAS_ROOT="$(cd ../.. && pwd)"
: "${NANO_SDK:?set NANO_SDK to the Nano effect SDK directory (the one holding scripts/ and include/)}"
NANO_SDK="$(cd "$NANO_SDK" && pwd)"
OUT_DIR="${OUT_DIR:-$EXTRAS_ROOT/out}"
TMP_DIR="${TMP_DIR:-$OUT_DIR/tmp}"
mkdir -p "$OUT_DIR" "$TMP_DIR"
OUT_DIR="$(cd "$OUT_DIR" && pwd)"
TMP_DIR="$(cd "$TMP_DIR" && pwd)"
MODULE_NAME=legacy

# Effects live in ../../effects; this repo's shared HLSL in ../../shaders.
NANO_EFFECTS_ROOT="$EXTRAS_ROOT/effects"
NANO_SHADER_INCLUDE_DIRS="$EXTRAS_ROOT/shaders${NANO_SHADER_INCLUDE_DIRS:+:$NANO_SHADER_INCLUDE_DIRS}"

echo "=== Compiling shaders (legacy) ==="
source "$NANO_SDK/scripts/wasm_build_env.sh"

# bicolor_grad — content-adaptive two-colour gradient.
#   hist    (compute) — RGB→YIQ hue histogram, atomic scatter into 64 bins.
#   analyze (compute, 1 thread) — pick major/minor/off hues + colours, locate
#                                 spatial centroids, temporally smooth.
#   render  (compute) — paint the bicolor gradient + composite over input.
compile_shaders_compute_var_spv bicolor_grad hist
compile_shaders_compute_var_spv bicolor_grad analyze
compile_shaders_compute_var_spv bicolor_grad render
_emit_spv_header_var bicolor_grad hist analyze render
echo "  bicolor_grad shaders compiled (SPV: hist + analyze + render)"

# glisten — image-anchored sparkle fans (faithful NanoGraph pipeline).
#   downsample (compute) — input → 64² search grid.
#   blur       (compute) — separable weighted blur w/ per-pass gain (used on
#                          the search grid AND the sparkle layer; the flicker
#                          pulses the layer gain).
#   findanchor (compute, 1 thread) — coarse/fine argmax on the blurred grid +
#                                    luma/colour gradient extraction.
#   vs/fs      (vert/frag) — rim-vertex polygon-fan discs, additive, half-res.
#   composite  (compute) — out = in × input_alpha + layer × tint.
compile_shaders_compute_var_spv glisten downsample
compile_shaders_compute_var_spv glisten blur
compile_shaders_compute_var_spv glisten findanchor
compile_shaders_compute_var_spv glisten composite
dxc -T vs_6_0 -E main -spirv -fspv-target-env=vulkan1.1 \
  "${NANO_SHADER_INCLUDES[@]}" \
  $NANO_EFFECTS_ROOT/glisten/vs.hlsl -Fo "$TMP_DIR/glisten_vs.spv"
dxc -T ps_6_0 -E main -spirv -fspv-target-env=vulkan1.1 \
  "${NANO_SHADER_INCLUDES[@]}" \
  $NANO_EFFECTS_ROOT/glisten/fs.hlsl -Fo "$TMP_DIR/glisten_fs.spv"
_emit_spv_header_var glisten downsample blur findanchor composite vs fs
echo "  glisten shaders compiled (SPV: downsample + blur + findanchor + composite + vs + fs)"

# double_chamber — P field-particles + Big attractors (DoubleChamber v2).
compile_shaders_compute_var_spv double_chamber big_update
compile_shaders_compute_var_spv double_chamber p_update
compile_shaders_compute_var_spv double_chamber prefill
compile_shaders_compute_var_spv double_chamber trace
compile_shaders_compute_var_spv double_chamber bridger
compile_shaders_compute_var_spv double_chamber motion_prefill
dxc -T vs_6_0 -E main -spirv -fspv-target-env=vulkan1.1 \
  "${NANO_SHADER_INCLUDES[@]}" \
  $NANO_EFFECTS_ROOT/double_chamber/vs.hlsl -Fo "$TMP_DIR/double_chamber_vs.spv"
dxc -T ps_6_0 -E main -spirv -fspv-target-env=vulkan1.1 \
  "${NANO_SHADER_INCLUDES[@]}" \
  $NANO_EFFECTS_ROOT/double_chamber/fs.hlsl -Fo "$TMP_DIR/double_chamber_fs.spv"
dxc -T vs_6_0 -E main -spirv -fspv-target-env=vulkan1.1 \
  "${NANO_SHADER_INCLUDES[@]}" \
  $NANO_EFFECTS_ROOT/double_chamber/line_vs.hlsl -Fo "$TMP_DIR/double_chamber_line_vs.spv"
dxc -T ps_6_0 -E main -spirv -fspv-target-env=vulkan1.1 \
  "${NANO_SHADER_INCLUDES[@]}" \
  $NANO_EFFECTS_ROOT/double_chamber/line_fs.hlsl -Fo "$TMP_DIR/double_chamber_line_fs.spv"
dxc -T vs_6_0 -E main -spirv -fspv-target-env=vulkan1.1 \
  "${NANO_SHADER_INCLUDES[@]}" \
  $NANO_EFFECTS_ROOT/double_chamber/motion_vs.hlsl -Fo "$TMP_DIR/double_chamber_motion_vs.spv"
dxc -T ps_6_0 -E main -spirv -fspv-target-env=vulkan1.1 \
  "${NANO_SHADER_INCLUDES[@]}" \
  $NANO_EFFECTS_ROOT/double_chamber/motion_fs.hlsl -Fo "$TMP_DIR/double_chamber_motion_fs.spv"
dxc -T vs_6_0 -E main -spirv -fspv-target-env=vulkan1.1 \
  "${NANO_SHADER_INCLUDES[@]}" \
  $NANO_EFFECTS_ROOT/double_chamber/line_motion_vs.hlsl -Fo "$TMP_DIR/double_chamber_line_motion_vs.spv"
dxc -T ps_6_0 -E main -spirv -fspv-target-env=vulkan1.1 \
  "${NANO_SHADER_INCLUDES[@]}" \
  $NANO_EFFECTS_ROOT/double_chamber/line_motion_fs.hlsl -Fo "$TMP_DIR/double_chamber_line_motion_fs.spv"
dxc -T vs_6_0 -E main -spirv -fspv-target-env=vulkan1.1 \
  "${NANO_SHADER_INCLUDES[@]}" \
  $NANO_EFFECTS_ROOT/double_chamber/density_vs.hlsl -Fo "$TMP_DIR/double_chamber_density_vs.spv"
dxc -T ps_6_0 -E main -spirv -fspv-target-env=vulkan1.1 \
  "${NANO_SHADER_INCLUDES[@]}" \
  $NANO_EFFECTS_ROOT/double_chamber/density_fs.hlsl -Fo "$TMP_DIR/double_chamber_density_fs.spv"
compile_shaders_compute_var_spv double_chamber density_debug
_emit_spv_header_var double_chamber big_update p_update prefill trace bridger motion_prefill \
  vs fs line_vs line_fs motion_vs motion_fs line_motion_vs line_motion_fs \
  density_vs density_fs density_debug
echo "  double_chamber shaders compiled (SPV: + motion_prefill/vs/fs + line_motion vs/fs + density/debug)"

# d_wave — Darkburst's polar radial-ripple distortion field ("D wave").
#   field (compute) — stateful wave field: inject grain at centre, advect outward, decay.
#   particles (compute) — forward-integrate the dampening-flash pool (mid-radius band).
#   blob_vs/fs (vert/frag) — splat flashes into the RGBA16F damp texture (subtracted at warp).
#   warp  (compute) — polar lookup (wave − damp) + radial UV warp + composite.
compile_shaders_compute_var_spv d_wave field
compile_shaders_compute_var_spv d_wave particles
compile_shaders_compute_var_spv d_wave warp
compile_shaders_compute_var_spv d_wave motion
dxc -T vs_6_0 -E main -spirv -fspv-target-env=vulkan1.1 \
  "${NANO_SHADER_INCLUDES[@]}" \
  $NANO_EFFECTS_ROOT/d_wave/blob_vs.hlsl -Fo "$TMP_DIR/d_wave_blob_vs.spv"
dxc -T ps_6_0 -E main -spirv -fspv-target-env=vulkan1.1 \
  "${NANO_SHADER_INCLUDES[@]}" \
  $NANO_EFFECTS_ROOT/d_wave/blob_fs.hlsl -Fo "$TMP_DIR/d_wave_blob_fs.spv"
_emit_spv_header_var d_wave field particles warp motion blob_vs blob_fs
echo "  d_wave shaders compiled (SPV: field + particles + warp + motion + blob_vs/fs)"

# lut_collection — "LUT Collection 1": baked preset colour LUTs (Wire "LUT 2").
#   fill  (compute) — copy a baked 32^3 rgba8 cube (storage buffer) into a 3D texture.
#   apply (compute) — pow pregain curve + single hardware-trilinear 3D LUT sample + mix.
compile_shaders_compute_var_spv lut_collection fill
compile_shaders_compute_var_spv lut_collection apply
_emit_spv_header_var lut_collection fill apply
echo "  lut_collection shaders compiled (SPV: fill + apply)"

# zoom_scroller — procedural pan/zoom sequence camera (Wire "ZoomScroller").
#   apply (compute) — scale+translate sample of tex_in + analytic gizmo box.
#   All sequencing/state-machine logic lives in main.cpp (tick).
compile_shaders_compute_var_spv zoom_scroller apply
_emit_spv_header_var zoom_scroller apply
echo "  zoom_scroller shaders compiled (SPV: apply)"

# subtle_blur — light Gaussian blur + drifting chromatic offset (Wire "Subtle Blur").
#   chroma (compute) — resample R/G/B at a slowly-rotating 120°-split basis.
#   Blur stage uses the shared fx::GaussianBlur (effect_blur.h / blur shader below).
compile_shaders_compute_var_spv subtle_blur chroma
_emit_spv_header_var subtle_blur chroma
echo "  subtle_blur shaders compiled (SPV: chroma)"

# sphr_blur — sphere-aware (latitude-dependent) horizontal blur + Gaussian
#   (Wire "SPHR Blur"). Useful off-sphere too.
#   expand (compute) — SPHR Expand ISF: lat-dependent horizontal weighted blur.
#   Gaussian stage uses the shared fx::GaussianBlur (blur shader below).
compile_shaders_compute_var_spv sphr_blur expand
_emit_spv_header_var sphr_blur expand
echo "  sphr_blur shaders compiled (SPV: expand)"

# burn_out — AR-envelope-driven exposure-blowout grade (Wire "Burn Out").
#   burn (compute) — per-pixel saturation/contrast lift + exposure + white fade.
#   Envelope lives in main.cpp (tick).
compile_shaders_compute_var_spv burn_out burn
_emit_spv_header_var burn_out burn
echo "  burn_out shaders compiled (SPV: burn)"

# chroma_wobble — triggered fbm-noise UV wobble + chromatic split (Wire "ChromaWobble").
#   wobble (compute) — analytic fbm displacement → warp + nano_chroma_offset split.
#   Envelope/drift live in main.cpp (tick). Uses shared nano_chroma.hlsl + nano_hash.hlsl.
compile_shaders_compute_var_spv chroma_wobble wobble
_emit_spv_header_var chroma_wobble wobble
echo "  chroma_wobble shaders compiled (SPV: wobble)"

# wobble_master — beat-pulsed radial-ripple wobble + chromatic dispersion
#   (Wire "Wobble Master 2", v2). Uses shared nano_chroma.hlsl.
compile_shaders_compute_var_spv wobble_master wobble
_emit_spv_header_var wobble_master wobble
echo "  wobble_master shaders compiled (SPV: wobble)"

# stutter_scale — beat-stutter scale/flip/hue/invert glitch (Wire "Stutter Scale 2").
#   stutter (compute) — per-step transform + grade; scheduler in main.cpp.
compile_shaders_compute_var_spv stutter_scale stutter
_emit_spv_header_var stutter_scale stutter
echo "  stutter_scale shaders compiled (SPV: stutter)"

# freeze_pulse — frame-freeze + randomized-blend stutter pulse (Wire "Freeze Pulse").
#   capture (compute) — copy the live frame into the freeze buffer on trigger.
#   pulse   (compute) — scale-pop/jitter/grade the frozen frame, blend over live.
compile_shaders_compute_var_spv freeze_pulse capture
compile_shaders_compute_var_spv freeze_pulse pulse
_emit_spv_header_var freeze_pulse capture pulse
echo "  freeze_pulse shaders compiled (SPV: capture + pulse)"

# pixulant — scatter-cascade + Difference "dive" (Wire "Pixulant").
#   pixulant (compute) — 3-deep Radial Stretch Sample cascade composed into one
#                        pass + abs-difference + exposure.
compile_shaders_compute_var_spv pixulant pixulant
_emit_spv_header_var pixulant pixulant
echo "  pixulant shaders compiled (SPV: pixulant)"

# Shared Gaussian blur helper (effect_blur.h) — double_chamber's image smoothing,
# subtle_blur's blur stage.
# (blur_shaders.h ships pre-built in the SDK)
echo "  blur shader compiled (SPV) for effect_blur.h"

echo "=== Building WASM (legacy) ==="

WASM_COMMON_EXPORTS=(
  -Wl,--export=nano_module_main
  -Wl,--export=malloc
  -Wl,--export=free
  -Wl,--export=__indirect_function_table
)

wasm_build \
  -I"$TMP_DIR" \
  -I"$NANO_INCLUDE_DIR" \
  -I"$EXTRAS_ROOT/include" \
  bundle.cpp \
  $NANO_EFFECTS_ROOT/bicolor_grad/main.cpp \
  $NANO_EFFECTS_ROOT/glisten/main.cpp \
  $NANO_EFFECTS_ROOT/double_chamber/main.cpp \
  $NANO_EFFECTS_ROOT/d_wave/main.cpp \
  $NANO_EFFECTS_ROOT/lut_collection/main.cpp \
  $NANO_EFFECTS_ROOT/zoom_scroller/main.cpp \
  $NANO_EFFECTS_ROOT/subtle_blur/main.cpp \
  $NANO_EFFECTS_ROOT/sphr_blur/main.cpp \
  $NANO_EFFECTS_ROOT/burn_out/main.cpp \
  $NANO_EFFECTS_ROOT/chroma_wobble/main.cpp \
  $NANO_EFFECTS_ROOT/wobble_master/main.cpp \
  $NANO_EFFECTS_ROOT/stutter_scale/main.cpp \
  $NANO_EFFECTS_ROOT/freeze_pulse/main.cpp \
  $NANO_EFFECTS_ROOT/pixulant/main.cpp

echo "Built: $OUT_DIR/$MODULE_NAME.wasm ($(wc -c < "$OUT_DIR/$MODULE_NAME.wasm")B)"
