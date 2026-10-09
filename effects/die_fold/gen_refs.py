"""Python reference frames for source.die_fold's parity test.

Renders the prototype's REAL `render.render` at atlas cells (from atlas.json --
the same floats die_fold_atlas.h bakes), and writes gzipped raw RGB u8 plus a
JSON manifest of the settings, for nano-modules-extras/tests/web/die_fold.test.ts.

    <die-fold>/.venv/bin/python -I effects/die_fold/gen_refs.py \
        ../nano-fx-prototypes/die-fold tests/web/fixtures/die_fold

Re-run whenever the prototype or the atlas changes (and regenerate the atlas
header with gen_atlas.py first).
"""
import gzip
import json
import math
import os
import sys

proto, out = sys.argv[1], sys.argv[2]
sys.path.insert(0, proto)

import numpy as np
import torch

from diefold import fold as FD
from diefold import render as RD

RES = 128
G = 15
CASES = [
    # name, x, y, t (depth), pan_x, pan_y, seed
    ("center",   7 / 14, 7 / 14, 0.0,  0.0,    0.0,    11),
    ("fine_net", 2 / 14, 12 / 14, 0.37, 0.0,   0.0,    11),
    ("far",      12 / 14, 3 / 14, 0.63, 37.31, -12.62, 11),
    ("seed",     10 / 14, 9 / 14, 0.83, 0.0,   0.0,    5),
]

with open(os.path.join(proto, "web", "public", "atlas.json")) as fh:
    aj = json.load(fh)
K = len(aj["fold_names"])
atlas = {
    "grid": aj["grid"],
    "names": aj["fold_names"],
    "cell_v": np.asarray(aj["cells"], np.float64).reshape(G, G, K),
}

os.makedirs(out, exist_ok=True)
manifest = {"res": RES, "cases": []}
for name, x, y, t, panx, pany, seed in CASES:
    cx, cy = math.floor(panx), math.floor(pany)
    fx, fy = panx - cx, pany - cy
    P = FD.sample_atlas(atlas, [(x, y)], seed=seed, cam_cell=(cx, cy),
                        cam_frac=(fx, fy), device="cpu")
    with torch.no_grad():
        img = RD.render(P, RES, t=t, supersample=1)[0].cpu().numpy()
    u8 = (np.clip(img, 0.0, 1.0) * 255.0 + 0.5).astype(np.uint8)
    assert u8.shape == (RES, RES, 3), u8.shape
    with gzip.open(os.path.join(out, f"{name}.rgb.gz"), "wb") as fh:
        fh.write(u8.tobytes())
    manifest["cases"].append({"name": name, "grain": x, "netness": y, "depth": t,
                              "pan_x": panx, "pan_y": pany, "seed": seed})
    print(name, "mean", float(u8.mean()))
with open(os.path.join(out, "manifest.json"), "w") as fh:
    json.dump(manifest, fh, indent=2)
