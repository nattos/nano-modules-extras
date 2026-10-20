"""Dump the poster scene's block layout for Lantern's Layout 0.

    Blender -b ~/Code/blender/comp1.blend --python blender_dump_layout.py -- layout_ref.json
    python3 gen_layout_ref.py layout_ref.json ../layout_ref.h

Re-runs part_structure.py's layout with its own RNG (sculpting and the halo
skipped), recording each block's rest cell, jut axis and distance, tip, size
and material, and the fragments' drift and tumble.
"""
import sys, json, math
sys.path.insert(0, "/Users/nattos/Code/blender/scripts")
import bpy
from mathutils import Vector, Euler
import part_structure as ps

rec = {}
orig_place = ps._place
def place(ob, rng, rest, axis_n, jut):
    strain = jut / (ps.CELL * ps.JUT_MAX)
    st = rng.getstate()
    orig_place(ob, rng, rest, axis_n, jut)
    rec[ob.name] = dict(kind="block", rest=list(rest - ps.CENTRE), axis=list(axis_n), jut=jut,
                        glow=strain, size=list(ob.scale), rot=list(ob.rotation_euler))
ps._place = place
orig_frag = ps._add_fragments
def add_frag(coll, mesh, rng):
    obs = orig_frag(coll, mesh, rng)
    for ob in obs:
        rec[ob.name] = dict(kind="fragment", pos=list(ob.location - ps.CENTRE), glow=1.0,
                            size=list(ob.scale), rot=list(ob.rotation_euler))
    return obs
ps._add_fragments = add_frag
ps._sculpt = lambda blocks, mat: None
ps._add_halo = lambda coll: None
ps.build()
for ob in bpy.data.collections["STRUCTURE"].objects:
    if ob.name in rec:
        r = rec[ob.name]
        r["material"] = ob.material_slots[0].material.name if ob.material_slots else "Shell"
        if r["kind"] == "fragment":
            # velocity/spin from keyframes
            fcs = ps._fcurves(ob)
            vel = [0, 0, 0]; spin = [0, 0, 0]
            for f in fcs:
                v1 = f.evaluate(2) - f.evaluate(1)
                if f.data_path == "location": vel[f.array_index] = v1
                if f.data_path == "rotation_euler": spin[f.array_index] = v1
            r["vel"] = vel; r["spin"] = spin
out = dict(
    centre=list(ps.CENTRE), tilt_deg=list(ps.TILT_DEG), cell=ps.CELL, cube_fill=ps.CUBE_FILL,
    blocks=[rec[k] for k in sorted(rec)],
)
cam = bpy.context.scene.camera
out["camera"] = dict(loc=list(cam.location), rot=list(cam.rotation_euler), lens=cam.data.lens,
                     sensor=cam.data.sensor_width, matrix=[list(r) for r in cam.matrix_world])
lights = {}
for o in bpy.data.objects:
    if o.type == 'LIGHT':
        d = o.data
        fwd = o.matrix_world.to_3x3() @ Vector((0, 0, -1))
        lights[o.name] = dict(type=d.type, energy=d.energy, color=list(d.color), loc=list(o.matrix_world.translation),
                              dir=list(fwd.normalized()), size=getattr(d, "size", 0), radius=getattr(d, "shadow_soft_size", 0),
                              angle=getattr(d, "angle", 0))
out["lights"] = lights
json.dump(out, open(sys.argv[-1], "w"), indent=1)
print("DUMPED", len(rec))
