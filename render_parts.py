"""Parts wall: every STEP part file from mechanical/ASV1/*/FABRICATION, meshed with gmsh,
normalised to a common cell size and grouped by fabrication class. Camera pans across the groups.
"""
import os, glob, json, argparse
os.environ.setdefault("MUJOCO_GL", "egl")
import numpy as np, trimesh, fast_simplification, mujoco
import imageio.v2 as imageio
from render_model import styled_spec

from render_model import REPO
STL = os.path.join(os.path.dirname(os.path.abspath(__file__)), "cad", "stl")
NORM = os.path.join(os.path.dirname(os.path.abspath(__file__)), "cad", "norm")
CLASSES = [("ALU_7075", "Machined aluminium 7075", [0.42, 0.44, 0.47, 1], 0.45, 0.6),
           ("SML_316L", "3D-printed stainless 316L", [0.52, 0.53, 0.55, 1], 0.7, 0.6),
           ("MJF_PA12", "3D-printed nylon PA12", [0.22, 0.23, 0.25, 1], 0.3, 0.4),
           ("OFF_THE_SHELF", "Off-the-shelf", [0.50, 0.36, 0.19, 1], 0.5, 0.6)]
CELL = 0.2


def prepare():
    os.makedirs(NORM, exist_ok=True)
    parts = []
    for cls, *_ in CLASSES:
        files = sorted(glob.glob(f"{REPO}/mechanical/ASV1/*/FABRICATION/{cls}/*.STEP"))
        for f in files:
            name = os.path.basename(f)[:-5]
            src = os.path.join(STL, name + ".stl")
            entry = dict(name=name, cls=cls, sub=f.split("/")[-4], meshed=os.path.exists(src))
            if entry["meshed"]:
                out = os.path.join(NORM, name + ".stl")
                if not os.path.exists(out):
                    m = trimesh.load(src, force="mesh")
                    m.apply_scale(0.001)
                    # principal orientation: longest extent along x, shortest along z
                    try:
                        T, _ = trimesh.bounds.oriented_bounds(m)
                        m.apply_transform(T)
                    except Exception:
                        pass
                    ext = m.extents
                    order = np.argsort(-ext)
                    m.vertices = m.vertices[:, [order[0], order[1], order[2]]]
                    m.fix_normals()
                    s = 0.85 * CELL / max(m.extents[:2].max(), 1e-6)
                    s = min(s, 0.12 / max(m.extents[2], 1e-6))
                    m.apply_scale(s)
                    b = m.bounds
                    m.apply_translation([-(b[0][0] + b[1][0]) / 2, -(b[0][1] + b[1][1]) / 2, -b[0][2]])
                    if len(m.faces) > 9000:
                        v, fc = fast_simplification.simplify(m.vertices.astype(np.float32), m.faces.astype(np.int32),
                                                             target_reduction=1 - 9000 / len(m.faces))
                        m = trimesh.Trimesh(v, fc)
                    m.export(out)
                entry["mesh"] = out
            parts.append(entry)
    return parts


def build(parts, width, height):
    spec = styled_spec(width=width, height=height, mocap_parts=True)
    spec.delete(spec.body("pelvis_link"))
    for t in list(spec.sensors): spec.delete(t)
    for ex in list(spec.excludes): spec.delete(ex)
    for l in spec.worldbody.lights:
        if l.name == "key":
            l.diffuse = [0, 0, 0]; l.castshadow = False
        if l.name == "rim":
            l.diffuse = [0.35, 0.2, 0.1]
    spec.visual.headlight.diffuse = [0.15, 0.15, 0.15]
    spec.worldbody.add_light(name="table_key", pos=[0, -2, 4], dir=[0.0, 0.45, -0.9],
                             type=mujoco.mjtLightType.mjLIGHT_DIRECTIONAL, diffuse=[0.62, 0.6, 0.57],
                             specular=[0.35, 0.35, 0.35], castshadow=True)
    for cls, label, rgba, spec_, shin in CLASSES:
        spec.add_material(name="m_" + cls, rgba=rgba, specular=spec_, shininess=shin)
    x0 = 0.0
    layout = {}
    for cls, label, *_ in CLASSES:
        group = [p for p in parts if p["cls"] == cls]
        n = len(group); cols = int(np.ceil(np.sqrt(n * 1.6))); rows = int(np.ceil(n / cols))
        for i, p in enumerate(group):
            r, c = divmod(i, cols)
            pos = [x0 + c * CELL, (rows - 1 - r) * CELL - rows * CELL / 2, 0.0]
            p["pos"] = pos
            if p["meshed"]:
                spec.add_mesh(name="mesh_" + p["name"], file=p["mesh"])
                b = spec.worldbody.add_body(name="part_" + p["name"], pos=pos, mocap=True)
                b.add_geom(type=mujoco.mjtGeom.mjGEOM_MESH, meshname="mesh_" + p["name"], material="m_" + cls,
                           contype=0, conaffinity=0, density=0, rgba=[0.5, 0.5, 0.5, 1])
        layout[cls] = dict(label=label, count=n, meshed=sum(p["meshed"] for p in group),
                           x_min=x0, x_max=x0 + (cols - 1) * CELL, rows=rows, cols=cols)
        x0 += cols * CELL + 0.6
    return spec.compile(), layout


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="clips/parts_wall.mp4")
    ap.add_argument("--width", type=int, default=1920); ap.add_argument("--height", type=int, default=1080)
    ap.add_argument("--fps", type=int, default=30); ap.add_argument("--seconds", type=float, default=14.0)
    ap.add_argument("--max_frames", type=int, default=0)
    a = ap.parse_args()
    parts = prepare()
    m, layout = build(parts, a.width, a.height)
    d = mujoco.MjData(m)
    ren = mujoco.Renderer(m, a.height, a.width)
    cam = mujoco.MjvCamera(); opt = mujoco.MjvOption()
    xs = [v["x_min"] for v in layout.values()] + [v["x_max"] for v in layout.values()]
    x_start, x_end = min(xs) + 0.35, max(xs) - 0.35
    n = int(a.seconds * a.fps)
    if a.max_frames: n = min(n, a.max_frames)
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    w = imageio.get_writer(a.out, fps=a.fps, codec="libx264", quality=None,
                           ffmpeg_params=["-crf", "17", "-pix_fmt", "yuv420p", "-preset", "medium"])
    cam_track = []
    for k in range(n):
        u = k / max(n - 1, 1); e = u * u * (3 - 2 * u)
        cam.lookat[:] = [x_start + (x_end - x_start) * e, 0, 0.0]
        cam.distance = 1.55; cam.azimuth = 100; cam.elevation = -48
        mujoco.mj_forward(m, d)
        ren.update_scene(d, cam, opt)
        ren.scene.flags[mujoco.mjtRndFlag.mjRND_SHADOW] = True
        ren.scene.flags[mujoco.mjtRndFlag.mjRND_REFLECTION] = False
        w.append_data(ren.render())
        cam_track.append(float(cam.lookat[0]))
        if k % 30 == 0: print("parts frame", k, n, flush=True)
    w.close()
    json.dump(dict(layout=layout, fps=a.fps, cam_x=cam_track,
                   parts=[{k: v for k, v in p.items() if k != "mesh"} for p in parts]),
              open(os.path.splitext(a.out)[0] + "_layout.json", "w"), indent=1)
    print("wrote", a.out, {k: (v["count"], v["meshed"]) for k, v in layout.items()})


if __name__ == "__main__":
    main()
