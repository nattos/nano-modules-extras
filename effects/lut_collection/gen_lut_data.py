#!/usr/bin/env python3
"""
Bake the LUT 2 (-> "LUT Collection 1") preset strips into a C++ header.

The shipped Resolume Wire "LUT 2" patch applies each preset via the ISF
`LUTShader` — a 64x64x64 colour cube packed into a 512x512 PNG as an 8x8 grid
of 64x64 tiles (blue selects the tile, red/green index within it; trilinear
across the two adjacent blue tiles). See LUTShader.isf.

This script replays that exact strip addressing to resample each preset onto a
LUT_DIM^3 grid and emits `lut_data.h`:
    LUT_DIM, LUT_COUNT, LUT_NAMES[], LUT_DATA[]   (RGBA8, x + y*N + z*N*N order)

A real 3D texture + hardware trilinear at runtime is far cheaper than the
original's hand-rolled 8x8 strip math (the team's "implement more efficiently"
ask). LUT_DIM=32 is a deliberate downsample of the 64^3 source: these are smooth
global grades, and 32^3 trilinear-upsampled is visually indistinguishable while
keeping the baked table at ~1.7 MB total instead of ~13.6 MB. Re-run with a
larger LUT_DIM if a preset ever shows banding.

The header is committed — the build does NOT depend on the Google Drive source
path. Re-run this only to regenerate (e.g. add/reorder presets).

Usage:  python3 gen_lut_data.py [src_image_dir]
"""
import os, sys, math
from PIL import Image

LUT_DIM = 64  # native resolution of the source 64^3 strips -> direct copy, no resampling loss

# (display name, lookup-<file>.png)  — order is the selectField option order.
# All 13 baked presets from the patch's Resource/Image are included (the live
# patch wired 10 of these into its switch; keeping all 13 is strictly more
# useful and the team is happy to keep the baked resources as-is).
PRESETS = [
    ("Process",                 "lookup-process.png"),
    ("Instant",                 "lookup-instant.png"),
    ("Fade",                    "lookup-fade.png"),
    ("Chrome",                  "lookup-chrome.png"),
    ("Transfer",                "lookup-transfer.png"),
    ("Tonal",                   "lookup-tonal.png"),
    ("Mono",                    "lookup-mono.png"),
    ("Noir",                    "lookup-noir.png"),
    ("Saturation + Contrast",   "lookup-saturation+contrast.png"),
    ("Saturation/Contrast (More)", "lookup-saturation-contrast-more.png"),
    ("Hue Rotate 90",           "lookup-rotate-90.png"),
    ("Hue Rotate 180",          "lookup-rotate-180.png"),
    ("Hue Rotate 270",          "lookup-rotate-270.png"),
]

DEFAULT_SRC = os.path.expanduser(
    "~/Library/CloudStorage/GoogleDrive-natnyo@gmail.com/My Drive/Resolume/Wire/Patches/LUT 2/Resource/Image"
)


def sample_bilinear(px, W, H, u, v):
    """Bilinear sample of a top-origin RGB image at normalized (u,v) in [0,1],
    matching GL/IMG_NORM_PIXEL convention: texel centres at (i+0.5)/N, so the
    sample position in texel space is u*N - 0.5 (NOT u*(N-1)). This matters at
    the strip's 64px tile boundaries — using u*(N-1) shifts ~half a texel and
    bleeds the adjacent blue-tile in, which corrupts the LUT (harsh/red darks)."""
    fx = min(max(u, 0.0), 1.0) * W - 0.5
    fy = min(max(v, 0.0), 1.0) * H - 0.5
    x0 = int(math.floor(fx)); y0 = int(math.floor(fy))
    tx = fx - x0; ty = fy - y0
    x0 = max(0, min(x0, W - 1)); y0 = max(0, min(y0, H - 1))
    x1 = min(x0 + 1, W - 1);     y1 = min(y0 + 1, H - 1)
    def g(x, y):
        return px[y * W + x]
    c00 = g(x0, y0); c10 = g(x1, y0); c01 = g(x0, y1); c11 = g(x1, y1)
    out = []
    for k in range(3):
        a = c00[k] * (1 - tx) + c10[k] * tx
        b = c01[k] * (1 - tx) + c11[k] * tx
        out.append(a * (1 - ty) + b * ty)
    return out


def isf_lookup(px, W, H, r, g, b):
    """Replicate LUTShader.isf addressing: 64^3 cube in an 8x8 grid of 64x64.

    The shader's unconditional `texPos.y = 1 - texPos.y` cancels the GL
    bottom-origin vs file top-origin flip, so sampling the top-origin PIL
    image directly at (tx, ty) is the faithful result.
    """
    blue = b * 63.0
    lo = int(math.floor(blue)); hi = int(math.ceil(blue)); frac = blue - lo

    def tile(bidx):
        bidx = min(max(bidx, 0), 63)
        qy = math.floor(bidx / 8.0)
        qx = bidx - qy * 8.0
        tx = (qx * 0.125) + 0.5 / W + ((0.125 - 1.0 / W) * r)
        ty = (qy * 0.125) + 0.5 / H + ((0.125 - 1.0 / H) * g)
        return sample_bilinear(px, W, H, tx, ty)

    c1 = tile(lo)
    c2 = tile(hi)
    return [c1[k] * (1 - frac) + c2[k] * frac for k in range(3)]


def bake_one(path):
    im = Image.open(path).convert("RGB")
    W, H = im.size
    px = list(im.getdata())
    N = LUT_DIM
    data = bytearray(N * N * N * 4)
    inv = 1.0 / (N - 1)
    for z in range(N):
        b = z * inv
        for y in range(N):
            g = y * inv
            for x in range(N):
                r = x * inv
                c = isf_lookup(px, W, H, r, g, b)
                idx = (x + y * N + z * N * N) * 4
                data[idx + 0] = int(round(min(max(c[0], 0.0), 255.0)))
                data[idx + 1] = int(round(min(max(c[1], 0.0), 255.0)))
                data[idx + 2] = int(round(min(max(c[2], 0.0), 255.0)))
                data[idx + 3] = 255
    return data


def main():
    src = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_SRC
    out_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "lut_data.h")

    luts = []
    for name, fname in PRESETS:
        p = os.path.join(src, fname)
        if not os.path.exists(p):
            print(f"!! missing {p}", file=sys.stderr); sys.exit(1)
        print(f"baking {name:30s} <- {fname}")
        luts.append((name, bake_one(p)))

    # --- validation: Mono must be grayscale (R==G==B) everywhere AND the axes
    # must be oriented right. Mono-gray alone is insufficient (every cell of a
    # mono LUT is gray even with scrambled axes), so also assert a chromatic LUT
    # keeps a pure-channel input on-hue: Process pure-blue stays blue-dominant
    # (the tile-boundary bake bug turned it red — this guards against that).
    N = LUT_DIM
    def cell(data, x, y, z):
        i = (x + y * N + z * N * N) * 4
        return tuple(data[i:i + 3])
    for name, data in luts:
        if name == "Mono":
            worst = max(max(c) - min(c) for c in
                       [cell(data, 8, 24, 16), cell(data, 31, 0, 7), cell(data, 4, 28, 20)])
            print(f"   [validate] Mono max channel spread = {worst} (must be ~0)")
            assert worst <= 2, "Mono not grayscale -> axis/orientation bug"
        if name == "Process":
            pb = cell(data, 0, 0, N - 1)   # pure blue input
            print(f"   [validate] Process pure-blue -> {pb} (must be blue-dominant, B>R)")
            assert pb[2] > pb[0], "Process pure-blue not blue -> tile-boundary bake bug"

    n_total = sum(len(d) for _, d in luts)
    flat = bytearray()
    for _, d in luts:
        flat += d

    # Emit the blob as a chunked C string literal (every byte as \xNN, so the
    # "\x is greedy" rule never bites — each escape is followed by a backslash).
    # clang parses adjacent string literals far faster than a multi-million
    # element integer initializer, and the file stays plain ASCII / git-clean.
    CHUNK = 2048
    tbl = [f"\\x{b:02x}" for b in range(256)]
    with open(out_path, "w") as f:
        f.write("// AUTO-GENERATED by gen_lut_data.py — do not edit by hand.\n")
        f.write("// Baked LUT presets from the Resolume Wire \"LUT 2\" patch (ISF LUTShader),\n")
        f.write(f"// baked to {LUT_DIM}^3 RGBA8 cubes (native source resolution). Order x + y*N + z*N*N.\n")
        f.write("#pragma once\n\n")
        f.write(f"static constexpr int LUT_DIM = {LUT_DIM};\n")
        f.write(f"static constexpr int LUT_COUNT = {len(luts)};\n")
        f.write(f"static constexpr int LUT_BYTES = {LUT_DIM*LUT_DIM*LUT_DIM*4}; // per LUT\n\n")
        names = ", ".join('"%s"' % n for n, _ in luts)
        f.write(f"static const char* const LUT_NAMES[LUT_COUNT] = {{ {names} }};\n\n")
        # Unsized array: string adds an implicit trailing '\0' (ignored — we
        # index via LUT_BYTES). Embedded \x00 bytes are fine in the literal.
        f.write("static const unsigned char LUT_DATA[] =\n")
        for i in range(0, n_total, CHUNK):
            chunk = flat[i:i + CHUNK]
            f.write('"' + "".join(tbl[b] for b in chunk) + '"\n')
        f.write(";\n")
        f.write(f"static_assert(sizeof(LUT_DATA) >= LUT_BYTES * LUT_COUNT, \"LUT blob size\");\n")
    print(f"wrote {out_path}  ({n_total} bytes, {len(luts)} LUTs, {os.path.getsize(out_path)} bytes of source)")


if __name__ == "__main__":
    main()
