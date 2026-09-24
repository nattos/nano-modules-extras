# nano-modules-extras

Effect bundles for [nano-modules](https://github.com/nattos/nano-modules) that
live outside it:

| Bundle | Id | What |
|---|---|---|
| `nano` | `com.nano.nano` | the flagship generators and processors — SDF volumes, particles, triangulation, lens, line reconstruction, spectral LFOs — and `control.nanolooper` |
| `lights` | `com.nano.lights` | show effects for the LED-bar performance |
| `legacy` | `com.nano.legacy` | ports of shipped dnode/NanoGraph and Resolume Wire effects |

The apps and the Resolume plugin load them from the per-user modules folder
(`~/Library/Application Support/Nano Modules/Modules`, `%APPDATA%\Nano Modules\Modules`),
which a packaged app seeds with these bundles on first launch.

## Layout

```
build.sh              build every bundle (or the ones named)
bundles/<bundle>/     build.sh + bundle.cpp (registers the bundle's effects)
effects/<effect>/     one folder per effect: main.cpp + its HLSL
include/              C++ headers shared by these effects (<sketch/led_bars.h>, overlay.h, ...)
shaders/              HLSL includes shared by these effects
```

It follows the layout of the SDK's template (`<sdk>/template`): effects are
folders, a bundle's `build.sh` lists which it compiles and links.

## Building

Needs the Nano effect SDK and its toolchain (wasi-sdk or a wasm-capable
clang++, `dxc`, python 3 — see the SDK's `template/README.md`).

```bash
# In nano-modules: stage an SDK
native/sdk/stage_sdk.sh /tmp/nano-sdk

# Here:
NANO_SDK=/tmp/nano-sdk ./build.sh          # -> out/{nano,lights,legacy}.wasm
NANO_SDK=/tmp/nano-sdk ./build.sh lights   # one bundle
```

To try a build in the apps, map `out/` in Settings → Modules (a mapped folder
replaces the same bundle from anywhere else, and reloads on rebuild), or copy
the `.wasm` files into the modules folder.

The SDK is the only link back to nano-modules. Nothing here includes anything
from that tree directly, so a bundle builds anywhere the SDK is.

## From nano-modules

nano-modules builds, packages and tests these bundles when told where this
checkout is. It never finds it on its own:

```bash
native/wasm_modules/build_all.sh --extras ../nano-modules-extras
```

That stages the SDK, runs this repo's `build.sh` into nano-modules'
`build/wasm/`, and records what it built in `build/wasm/extras.json`. See
nano-modules' `DESKTOP.md` for packaging, and the notes there on running this
repo's tests.
