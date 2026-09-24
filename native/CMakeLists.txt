# nano-modules-extras' native targets, built INSIDE nano-modules' CMake tree:
#
#   cmake -S nano-modules/native -B build -DNANO_EXTRAS_DIR=/path/to/nano-modules-extras
#
# nano-modules adds this directory at the end of its own CMakeLists.txt, so the
# targets and variables below come from there: NANO_NATIVE_DIR (its native/),
# NANO_WASM_DIR (its build/wasm, where build_all.sh --extras puts our bundles),
# ffgl::sdk, Catch2, bridge_server, effect_runtime, sketch_executor, wasm_host,
# gpu_backend, NANO_INTEROP_TEXTURE. It is not a standalone project.

if(NOT DEFINED NANO_NATIVE_DIR)
  message(FATAL_ERROR "nano-modules-extras/native is added by nano-modules' CMake "
                      "(-DNANO_EXTRAS_DIR=...), not configured on its own")
endif()

set(EXTRAS_ROOT "${CMAKE_CURRENT_SOURCE_DIR}/..")
set(EXTRAS_TESTS "${EXTRAS_ROOT}/tests/native")

# --- NanoLooper (NLPR) — the dedicated looper FFGL plugin (macOS) -----------
#
# A thin shell over nano-modules' shared effect runtime (the same architecture
# as NanoBarrel): it dlopens the sibling libbridge_server.dylib and drives ONE
# fixed effect (control.nanolooper, in our nano bundle) through the shared
# executor, reading the bundle from wherever the barrel loads bundles. Links NO
# engine — only the FFGL shell, the GL↔Metal InteropTexture pair, the bridge
# loader, and a native Synth plucked by the shared runtime's audio_bus fan-out.
# Live-only (nothing persisted).
if(APPLE AND TARGET bridge_server)
  add_library(NanoLooper MODULE
    plugin/looper_plugin.mm
    plugin/synth.mm
    ${NANO_NATIVE_DIR}/${NANO_INTEROP_TEXTURE}
    ${NANO_NATIVE_DIR}/src/plugin/bridge_loader.cpp
  )
  # The dylib is dlopen'd at runtime, not linked — but build it first so the
  # sibling copy (staged by NanoBarrelDeploy) exists for a from-scratch build.
  add_dependencies(NanoLooper bridge_server)
  target_include_directories(NanoLooper PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/plugin
    ${NANO_NATIVE_DIR}/src
    ${NANO_NATIVE_DIR}/src/plugin               # "nano_barrel/interop_texture.h"
    ${NANO_NATIVE_DIR}/src/plugin/nano_barrel
  )
  target_link_libraries(NanoLooper PRIVATE
    nlohmann_json::nlohmann_json
    ffgl::sdk
    "-framework OpenGL"
    "-framework Metal"
    "-framework MetalKit"
    "-framework CoreVideo"
    "-framework AppKit"
    "-framework Foundation"
    "-framework AVFoundation"
    "-framework AudioToolbox"
  )
  target_compile_options(NanoLooper PRIVATE -Wno-attributes -Wno-deprecated-declarations)
  set_target_properties(NanoLooper PROPERTIES
    BUNDLE TRUE
    BUNDLE_EXTENSION "bundle"
    MACOSX_BUNDLE_INFO_PLIST "${NANO_NATIVE_DIR}/src/plugin/Info.plist.in"
    LINK_FLAGS "-undefined dynamic_lookup"
  )
  # Codesign so Resolume/Gatekeeper loads it. No payload of its own → a plain
  # relink-time sign suffices.
  if(NOT DEFINED NANOBARREL_CODESIGN_IDENTITY)
    set(NANOBARREL_CODESIGN_IDENTITY "-")
  endif()
  add_custom_command(TARGET NanoLooper POST_BUILD
    COMMAND bash ${NANO_NATIVE_DIR}/tools/codesign_bundle.sh
            "${NANOBARREL_CODESIGN_IDENTITY}" $<TARGET_BUNDLE_DIR:NanoLooper>
    COMMENT "Codesigning NanoLooper.bundle (identity=${NANOBARREL_CODESIGN_IDENTITY})"
    VERBATIM)
endif()

if(NOT TARGET Catch2::Catch2WithMain)
  return()
endif()

# --- Pure-logic tests: host-free headers, no wasm, no GPU --------------------

# The looper's sequencer core (records gate duration + optional quantize),
# compiled natively straight from the effect source.
add_executable(test_looper_core ${EXTRAS_TESTS}/test_looper_core.cpp
               ${EXTRAS_ROOT}/effects/nanolooper/core.cpp)
target_include_directories(test_looper_core PRIVATE ${EXTRAS_ROOT}/effects)
target_link_libraries(test_looper_core PRIVATE Catch2::Catch2WithMain)
catch_discover_tests(test_looper_core)

# The spectral-morph curve generator shared by mod.source.spectral_lfo and
# mod.shaper.spectral. Pulls in the 1.2MB atlas.
add_executable(test_spectral_curve ${EXTRAS_TESTS}/test_spectral_curve.cpp
               ${EXTRAS_ROOT}/effects/spectral_lfo/spectral_curve.cpp)
target_include_directories(test_spectral_curve PRIVATE ${EXTRAS_ROOT}/effects/spectral_lfo)
target_link_libraries(test_spectral_curve PRIVATE Catch2::Catch2WithMain)
catch_discover_tests(test_spectral_curve)

# The SDF-field provider descriptor (Desc + validate outside __wasm__).
add_executable(test_sdf_field ${EXTRAS_TESTS}/test_sdf_field.cpp)
target_include_directories(test_sdf_field PRIVATE ${EXTRAS_ROOT}/include)
target_link_libraries(test_sdf_field PRIVATE Catch2::Catch2WithMain)
catch_discover_tests(test_sdf_field)

# The LED-bar show logic: three_planes' glints and Strobe throw, the house
# rig's pixel map, three_walls' moves. Each is driven at an exact dt.
foreach(t test_three_planes_glints test_three_planes_strobe test_led_bars
          test_three_walls_show)
  add_executable(${t} ${EXTRAS_TESTS}/${t}.cpp)
  # Ours first (<sketch/led_bars.h>), then nano-modules' src for the shared
  # utilities the SDK also ships (<sketch/knob_rate.h>).
  target_include_directories(${t} PRIVATE ${EXTRAS_ROOT}/include ${NANO_NATIVE_DIR}/src)
  target_link_libraries(${t} PRIVATE Catch2::Catch2WithMain)
  catch_discover_tests(${t})
endforeach()

# --- GPU tests: our bundles through nano-modules' executor on Metal ----------

if(TARGET effect_runtime AND TARGET gpu_backend)
  set(EXTRAS_WASM_DEFS
    CORE_WASM_PATH="${NANO_WASM_DIR}/core.wasm"
    TESTONLY_WASM_PATH="${NANO_WASM_DIR}/testonly.wasm"
    NANO_WASM_PATH="${NANO_WASM_DIR}/nano.wasm"
    LIGHTS_WASM_PATH="${NANO_WASM_DIR}/lights.wasm"
    LEGACY_WASM_PATH="${NANO_WASM_DIR}/legacy.wasm")
  foreach(t test_plane_shear test_tri_shear test_recompose test_envelope_warp
            test_extras_render)
    add_executable(${t} ${EXTRAS_TESTS}/${t}.cpp)
    target_include_directories(${t} PRIVATE ${NANO_NATIVE_DIR}/src ${NANO_NATIVE_DIR}/tests)
    target_link_libraries(${t} PRIVATE
      effect_runtime sketch_executor wasm_host gpu_backend
      Catch2::Catch2WithMain)
    target_compile_definitions(${t} PRIVATE ${EXTRAS_WASM_DEFS})
    target_compile_options(${t} PRIVATE -Wno-attributes)
    catch_discover_tests(${t})
  endforeach()
endif()
