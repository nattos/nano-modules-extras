#!/bin/bash
# Lights bundle build — show effects for the LED-bar performance.
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
MODULE_NAME=lights

# Effects live in ../../effects; this repo's shared HLSL in ../../shaders.
NANO_EFFECTS_ROOT="$EXTRAS_ROOT/effects"
NANO_SHADER_INCLUDE_DIRS="$EXTRAS_ROOT/shaders${NANO_SHADER_INCLUDE_DIRS:+:$NANO_SHADER_INCLUDE_DIRS}"

echo "=== Compiling shaders (lights) ==="
source "$NANO_SDK/scripts/wasm_build_env.sh"

# Each call compiles one .hlsl file and emits a <effect>_shaders.h with
# the variant baked in. All four v1 effects use a single `render.hlsl`
# compute pass.
compile_shaders_compute_spv strobe_channel     render
compile_shaders_compute_spv dispersion         render
compile_shaders_compute_spv plasma_beam_cannon render
compile_shaders_compute_spv orthomod           render
compile_shaders_compute_spv lights_sim           render
# The two neon-quad instruments each carry a second, tiny pass: the LED-bar
# pixel map. Its source is one line — the whole shader lives in
# shaders_common/nano_led_bars.hlsl and is shared verbatim — but it still has
# to be compiled and named per effect, because registered shader names are
# module-global.
compile_shaders_compute_var_spv three_planes render
compile_shaders_compute_var_spv three_planes led
compile_shaders_compute_var_spv three_planes wall
_emit_spv_header_var three_planes render led wall
echo "  three_planes shaders compiled (SPV: render + led + wall)"

compile_shaders_compute_var_spv three_walls render
compile_shaders_compute_var_spv three_walls led
_emit_spv_header_var three_walls render led
echo "  three_walls shaders compiled (SPV: render + led)"

# vcr_halo: prefilter -> down chain -> progressive up chain -> composite.
compile_shaders_compute_var_spv vcr_halo prefilter
compile_shaders_compute_var_spv vcr_halo down
compile_shaders_compute_var_spv vcr_halo up
compile_shaders_compute_var_spv vcr_halo composite
_emit_spv_header_var vcr_halo prefilter down up composite
echo "  vcr_halo shaders compiled (SPV: prefilter + down + up + composite)"

# soft_glow has two compute shaders — color (rgba8) and motion
# (rgba16f). Separate variants because naga substitutes one storage-
# texture format per shader module.
compile_shaders_compute_var_spv soft_glow color
compile_shaders_compute_var_spv soft_glow motion
_emit_spv_header_var soft_glow color motion
echo "  soft_glow shaders compiled (SPV: color + motion)"

# bounce_resonator: GPU-resident sim + color + motion passes.
compile_shaders_compute_var_spv bounce_resonator sim
compile_shaders_compute_var_spv bounce_resonator color
compile_shaders_compute_var_spv bounce_resonator motion
_emit_spv_header_var bounce_resonator sim color motion
echo "  bounce_resonator shaders compiled (SPV: sim + color + motion)"

compile_shaders_compute_var_spv side_jet sim
compile_shaders_compute_var_spv side_jet color
compile_shaders_compute_var_spv side_jet motion
_emit_spv_header_var side_jet sim color motion
echo "  side_jet shaders compiled (SPV: sim + color + motion)"

compile_shaders_compute_var_spv motion_blobs color
compile_shaders_compute_var_spv motion_blobs motion
_emit_spv_header_var motion_blobs color motion
echo "  motion_blobs shaders compiled (SPV: color + motion)"

compile_shaders_compute_var_spv block_dehance update
compile_shaders_compute_var_spv block_dehance render
compile_shaders_compute_var_spv block_dehance motion
_emit_spv_header_var block_dehance update render motion
echo "  block_dehance shaders compiled (SPV: update + render + motion)"

compile_shaders_compute_var_spv chroma_wave render
compile_shaders_compute_var_spv chroma_wave motion
_emit_spv_header_var chroma_wave render motion
echo "  chroma_wave shaders compiled (SPV: render + motion)"

# flicker_grid: per-cell reduce + persistent per-column sim + render.
compile_shaders_compute_var_spv flicker_grid reduce
compile_shaders_compute_var_spv flicker_grid sim
compile_shaders_compute_var_spv flicker_grid render
_emit_spv_header_var flicker_grid reduce sim render
echo "  flicker_grid shaders compiled (SPV: reduce + sim + render)"

compile_shaders_compute_var_spv tingle_top update
compile_shaders_compute_var_spv tingle_top prefill
compile_shaders_compute_var_spv tingle_top motion
dxc -T vs_6_0 -E main -spirv -fspv-target-env=vulkan1.1 \
  "${NANO_SHADER_INCLUDES[@]}" \
  $NANO_EFFECTS_ROOT/tingle_top/vs.hlsl -Fo "$TMP_DIR/tingle_top_vs.spv"
dxc -T ps_6_0 -E main -spirv -fspv-target-env=vulkan1.1 \
  "${NANO_SHADER_INCLUDES[@]}" \
  $NANO_EFFECTS_ROOT/tingle_top/fs.hlsl -Fo "$TMP_DIR/tingle_top_fs.spv"
_emit_spv_header_var tingle_top update prefill vs fs motion
echo "  tingle_top shaders compiled (SPV: update + prefill + vs + fs + motion)"

echo "=== Building WASM (lights) ==="

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
  $NANO_EFFECTS_ROOT/strobe_channel/main.cpp \
  $NANO_EFFECTS_ROOT/soft_glow/main.cpp \
  $NANO_EFFECTS_ROOT/dispersion/main.cpp \
  $NANO_EFFECTS_ROOT/plasma_beam_cannon/main.cpp \
  $NANO_EFFECTS_ROOT/orthomod/main.cpp \
  $NANO_EFFECTS_ROOT/bounce_resonator/main.cpp \
  $NANO_EFFECTS_ROOT/side_jet/main.cpp \
  $NANO_EFFECTS_ROOT/motion_blobs/main.cpp \
  $NANO_EFFECTS_ROOT/lights_sim/main.cpp \
  $NANO_EFFECTS_ROOT/block_dehance/main.cpp \
  $NANO_EFFECTS_ROOT/tingle_top/main.cpp \
  $NANO_EFFECTS_ROOT/chroma_wave/main.cpp \
  $NANO_EFFECTS_ROOT/flicker_grid/main.cpp \
  $NANO_EFFECTS_ROOT/three_planes/main.cpp \
  $NANO_EFFECTS_ROOT/three_walls/main.cpp \
  $NANO_EFFECTS_ROOT/vcr_halo/main.cpp

echo "Built: $OUT_DIR/$MODULE_NAME.wasm ($(wc -c < "$OUT_DIR/$MODULE_NAME.wasm")B)"
