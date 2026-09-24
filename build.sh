#!/bin/bash
# Build the extras bundles (nano, lights, legacy) into $OUT_DIR/<bundle>.wasm.
#
#   NANO_SDK=/path/to/nano-sdk ./build.sh              # every bundle
#   NANO_SDK=/path/to/nano-sdk ./build.sh nano lights  # just these
#   ./build.sh --list                                   # the bundle names
#
#   OUT_DIR   where the .wasm files go        (default: ./out)
#   TMP_DIR   generated shader headers, SPIR-V (default: $OUT_DIR/tmp)
#
# NANO_SDK is the Nano effect SDK — nano-modules' native/sdk/stage_sdk.sh
# makes one. It is the ONLY link back to nano-modules: nothing here reaches
# into that tree, which is what lets it build these bundles as a package input
# (nano-modules' build_all.sh --extras <this dir>).
set -euo pipefail
cd "$(dirname "$0")"

ALL=(nano lights legacy)
if [ "${1:-}" = "--list" ]; then
  printf '%s\n' "${ALL[@]}"
  exit 0
fi

: "${NANO_SDK:?set NANO_SDK to the Nano effect SDK directory (the one holding scripts/ and include/)}"
export NANO_SDK
export OUT_DIR="${OUT_DIR:-$PWD/out}"
export TMP_DIR="${TMP_DIR:-$OUT_DIR/tmp}"

bundles=("$@")
[ ${#bundles[@]} -eq 0 ] && bundles=("${ALL[@]}")

for b in "${bundles[@]}"; do
  if [ ! -x "bundles/$b/build.sh" ]; then
    echo "error: no bundle '$b' (have: ${ALL[*]})" >&2
    exit 1
  fi
  echo "--- Building $b ---"
  "bundles/$b/build.sh"
done
