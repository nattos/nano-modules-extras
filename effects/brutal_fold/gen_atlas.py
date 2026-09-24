"""Generate brutal_fold_atlas.h from the brutal-fold research atlas.json.

The atlas is the baked control surface (complexity × order × liveliness) of a
brutalist axonometric-prism field. brutal_fold's CPU `build_params` reads it
directly (it never touches the GPU), so we embed it as a committed C++ header of
flat float arrays — mirroring the prototype's web/src/field.ts + atlas.ts layout:

  term fields    [G,G,B]    row-major (gi*G+gj)*B + ti
  scene scalars  [G,G]      row-major gi*G+gj
  script fields  [G,G,Z,B]  row-major ((gi*G+gj)*Z+z)*B + ti
  b_* = structure 2 (co-folded), b_tilt = its screen shear.

Only the fields the native renderer actually reads are baked: the 8 term fields
and 19 scene scalars (for BOTH structures, so the shader's scene-index offsets
line up), the 4 animation script channels (structure 1 only — structure 2 has no
folded trajectory), and b_tilt. The unused rot/slew script channels are skipped.

Run once and commit the output (no build-time python dependency):

    python3 native/wasm_modules/brutal_fold/gen_atlas.py \
        [--atlas /path/to/brutal-fold/web/public/atlas.json]
"""

import argparse
import json
import os

# Per the prototype atlas.ts: 8 term fields, 19 scene scalars (order MATTERS —
# the shader indexes scene scalars by position), 4 animation script channels.
TERM_FIELDS = ["theta", "mtheta", "shear", "freq", "phase", "h", "amp", "mix"]
SCENE_FIELDS = [
    "sev", "gx", "gy", "form_scale", "thresh", "back_len", "back_ang", "extrude",
    "layers", "sep", "front_detail", "win_dark", "fog", "face", "sky_val", "dc",
    "bold_gain", "rot", "slew",
]
SCRIPT_FIELDS = ["h_amp", "h_om", "h_psi", "phase_drift"]

HERE = os.path.dirname(os.path.abspath(__file__))
_PROTO = os.path.join(HERE, "..", "..", "..", "..", "nano-fx-prototypes", "brutal-fold", "web", "public")
DEFAULT_ATLAS = os.path.join(_PROTO, "atlas.json")        # explore atlas (XY × liveliness)
DEFAULT_KM_ATLAS = os.path.join(_PROTO, "atlas_km.json")  # key-moment atlas (its own scenes + windows)
# Editor asset: the pad's grayed-cell overlay needs the KM atlas grid + per-cell sky.
DEFAULT_KM_EDITOR_JSON = os.path.join(HERE, "..", "..", "..", "web", "public", "brutal-fold-km.json")


def fmt(v):
    # Compact float literal: drop trailing zeros, keep enough precision. Ensure a
    # decimal point/exponent so the `f` suffix is valid C++ ("1" -> "1.0", not "1f").
    s = f"{float(v):.5g}"
    if "." not in s and "e" not in s and "E" not in s:
        s += ".0"
    return s


def emit_array(fh, name, values):
    fh.write(f"static const float {name}[{len(values)}] = {{\n")
    line = "  "
    for v in values:
        tok = fmt(v) + "f,"
        if len(line) + len(tok) > 100:
            fh.write(line + "\n")
            line = "  "
        line += tok
    if line.strip():
        fh.write(line + "\n")
    fh.write("};\n\n")


def emit_scene_atlas(fh, atlas, p, *, with_keymoments):
    """Emit one atlas's scene data (term + scene + script + b_tilt) under C++ prefix
    `p` (e.g. "BF_" or "BFKM_"), plus its dims. With `with_keymoments`, also emit the
    per-cell key-moment window (t1 center-peak + score + covmax) and the end-of-loop
    `sky` fraction used by the sky-threshold filter."""
    G = int(atlas["grid"]); B = int(atlas["n_terms"]); Z = int(atlas["n_z"])
    co_fold = int(atlas.get("co_fold", 0))
    h_act = float(atlas.get("constants", {}).get("H_ACT", 0.08))
    tc = [float(v) for v in atlas["tc_centers"]]
    fh.write(f"static const int   {p}GRID    = {G};\n")
    fh.write(f"static const int   {p}NTERMS  = {B};\n")
    fh.write(f"static const int   {p}NZ      = {Z};\n")
    fh.write(f"static const int   {p}CO_FOLD = {co_fold};\n")
    fh.write(f"static const float {p}H_ACT   = {fmt(h_act)}f;\n")
    fh.write(f"static const float {p}TC_CENTERS[{Z}] = {{ {', '.join(fmt(v) + 'f' for v in tc)} }};\n\n")
    # Structure 1 + structure 2 (b_) term + scene fields.
    for src_prefix, cpp_infix in (("", ""), ("b_", "B_")):
        for name in TERM_FIELDS + SCENE_FIELDS:
            emit_array(fh, p + cpp_infix + name.upper(), atlas[src_prefix + name])
    # Animation script channels — structure 1 only.
    for name in SCRIPT_FIELDS:
        emit_array(fh, p + name.upper(), atlas[name])
    # Structure 2 screen shear.
    emit_array(fh, p + "B_TILT", atlas["b_tilt"])
    if with_keymoments:
        km = atlas.get("keymoments") or {}
        n = G * G * Z
        sky = km.get("sky") or [1.0] * n     # no sky data → every cell reachable
        fh.write(f"static const int   {p}NFRAMES = {int(km.get('n_frames', 0))};\n\n")
        emit_array(fh, p + "T1", km.get("t1", [0.0] * n))
        emit_array(fh, p + "SCORE", km.get("score", [0.0] * n))
        emit_array(fh, p + "COVMAX", km.get("covmax", [1.0] * n))
        emit_array(fh, p + "SKY", sky)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--atlas", default=DEFAULT_ATLAS, help="explore atlas (XY × liveliness)")
    ap.add_argument("--km-atlas", default=DEFAULT_KM_ATLAS,
                    help="key-moment atlas (its own scenes + curated windows + sky)")
    ap.add_argument("--km-editor-json", default=DEFAULT_KM_EDITOR_JSON,
                    help="write {grid, sky} for the inspector's grayed-cell overlay")
    args = ap.parse_args()

    with open(args.atlas) as f:
        atlas = json.load(f)
    with open(args.km_atlas) as f:
        km_atlas = json.load(f)

    out_path = os.path.join(HERE, "brutal_fold_atlas.h")
    with open(out_path, "w") as fh:
        fh.write("// AUTO-GENERATED by gen_atlas.py from the brutal-fold research atlases. Do not edit.\n")
        fh.write("// Two baked control surfaces (CPU-only, read by build_params):\n")
        fh.write("//   BF_*   explore atlas — complexity (x) × order (y) × liveliness (z), continuous loop.\n")
        fh.write("//   BFKM_* key-moment atlas — XY only; each cell has a curated inflow→peak window\n")
        fh.write("//          (T1/SCORE/COVMAX) + an end-of-loop SKY fraction for the reachability filter.\n")
        fh.write("#ifndef BRUTAL_FOLD_ATLAS_H\n#define BRUTAL_FOLD_ATLAS_H\n\n")
        fh.write("namespace brutal_fold {\n\n")
        fh.write("// --- Explore atlas -------------------------------------------------------\n")
        emit_scene_atlas(fh, atlas, "BF_", with_keymoments=False)
        fh.write("// --- Key-moment atlas ----------------------------------------------------\n")
        emit_scene_atlas(fh, km_atlas, "BFKM_", with_keymoments=True)
        fh.write("} // namespace brutal_fold\n\n")
        fh.write("#endif // BRUTAL_FOLD_ATLAS_H\n")

    # Editor asset: grid + per-cell sky so the pad can gray unreachable cells.
    kmm = km_atlas.get("keymoments") or {}
    Gk = int(km_atlas["grid"])
    sky = kmm.get("sky") or [1.0] * (Gk * Gk)
    ed_path = os.path.abspath(args.km_editor_json)
    os.makedirs(os.path.dirname(ed_path), exist_ok=True)
    with open(ed_path, "w") as fh:
        json.dump({"grid": Gk, "sky": [round(float(v), 4) for v in sky]}, fh, separators=(",", ":"))

    print(f"wrote {out_path}: explore {atlas['grid']}³-ish ({atlas['grid']}x{atlas['grid']}x{atlas['n_z']}), "
          f"km {Gk}x{Gk}x{km_atlas['n_z']} ({sum(1 for s in kmm.get('score', []) if s > 0)} cells with a window)")
    print(f"wrote {ed_path} (grid {Gk}, sky[{len(sky)}])")


if __name__ == "__main__":
    main()
