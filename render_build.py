"""Virtual build of Asimov 1: the 26 link meshes of the official sim model fly into
place in the assembly order of the Asimov 1 manual:
  left leg -> right leg -> pelvis -> left arm -> right arm -> torso -> head (finish)
Output: an MP4 clip plus a JSON timeline of when each module completes.
"""
import os, sys, json, argparse
os.environ.setdefault("MUJOCO_GL", "egl")
import numpy as np
import mujoco
import imageio.v2 as imageio
from render_model import styled_spec, SUBASSEMBLY, GOLD_LINKS
from asimov_sim import home_pose

ORDER = [  # (module label, subassembly code, links bottom-up)
    ("Left leg", "600", ["left_ankle_roll_link", "left_ankle_pitch_link", "left_knee_link",
                         "left_hip_yaw_link", "left_hip_roll_link", "left_hip_pitch_link"]),
    ("Right leg", "500", ["right_ankle_roll_link", "right_ankle_pitch_link", "right_knee_link",
                          "right_hip_yaw_link", "right_hip_roll_link", "right_hip_pitch_link"]),
    ("Pelvis", "200", ["pelvis_link"]),
    ("Left arm", "400", ["left_shoulder_pitch_link", "left_shoulder_roll_link", "left_shoulder_yaw_link",
                         "left_elbow_link", "left_wrist_yaw_link"]),
    ("Right arm", "300", ["right_shoulder_pitch_link", "right_shoulder_roll_link", "right_shoulder_yaw_link",
                          "right_elbow_link", "right_wrist_yaw_link"]),
    ("Torso", "100", ["waist_yaw_link"]),
    ("Head", "700", ["neck_yaw_link", "neck_pitch_link"]),
]


def assembled_poses():
    spec = styled_spec()
    m = spec.compile(); d = mujoco.MjData(m)
    for n, v in home_pose().items():
        d.qpos[m.jnt_qposadr[m.joint(n).id]] = v
    d.qpos[2] = 0.0
    mujoco.mj_forward(m, d)
    # lift so the lowest sole touches the floor
    zmin = min(d.geom_xpos[g][2] - m.geom_size[g][0] for g in range(m.ngeom) if "foot" in (m.geom(g).name or ""))
    poses = {}
    for b in range(1, m.nbody):
        name = m.body(b).name
        poses[name] = (d.xpos[b].copy() - np.array([0, 0, zmin]), d.xquat[b].copy())
    meshes = {}
    for g in range(m.ngeom):
        n = m.geom(g).name
        if n.endswith("_visual"):
            meshes[m.body(m.geom_bodyid[g]).name] = m.mesh(m.geom_dataid[g]).name
    return poses, meshes


def build_scene(width, height):
    spec = styled_spec(width=width, height=height, mocap_parts=True)
    poses, meshes = assembled_poses()
    spec.delete(spec.body("pelvis_link"))
    for t in list(spec.sensors):
        spec.delete(t)
    for ex in list(spec.excludes):
        spec.delete(ex)
    for link, mesh in meshes.items():
        b = spec.worldbody.add_body(name="part_" + link, mocap=True, pos=poses[link][0], quat=poses[link][1])
        b.add_geom(name="vis_" + link, type=mujoco.mjtGeom.mjGEOM_MESH, meshname=mesh,
                   material="gold" if link in GOLD_LINKS else "shell", contype=0, conaffinity=0,
                   rgba=[0.5, 0.5, 0.5, 1], density=0)
    m = spec.compile()
    return m, poses


def ease_out(s):
    s = np.clip(s, 0, 1)
    return 1 - (1 - s) ** 3


def quat_slerp(q0, q1, s):
    q = np.zeros(4)
    mujoco.mju_mulQuat(q, q1, np.array([q0[0], -q0[1], -q0[2], -q0[3]]))  # q1 * q0^-1
    v = np.zeros(3); mujoco.mju_quat2Vel(v, q, 1.0)
    dq = np.zeros(4); mujoco.mju_axisAngle2Quat(dq, v / (np.linalg.norm(v) + 1e-12), np.linalg.norm(v) * s)
    out = np.zeros(4); mujoco.mju_mulQuat(out, dq, q0)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="clips/build.mp4")
    ap.add_argument("--width", type=int, default=1920)
    ap.add_argument("--height", type=int, default=1080)
    ap.add_argument("--fps", type=int, default=30)
    ap.add_argument("--module_time", type=float, default=2.1)
    ap.add_argument("--lead", type=float, default=0.8)
    ap.add_argument("--tail", type=float, default=2.5)
    ap.add_argument("--max_frames", type=int, default=0)
    a = ap.parse_args()
    m, poses = build_scene(a.width, a.height)
    d = mujoco.MjData(m)
    hl = m.material("highlight").id
    base_mat = {}
    for g in range(m.ngeom):
        n = m.geom(g).name or ""
        if n.startswith("vis_"):
            base_mat[n[4:]] = (g, m.geom_matid[g])
    center = np.array([0, 0, 0.6])
    rng = np.random.default_rng(3)
    sched = {}
    timeline = []
    t = a.lead
    for label, code, links in ORDER:
        n = len(links)
        stagger = min(0.18, 1.0 / max(n, 1))
        for i, l in enumerate(links):
            p, q = poses[l]
            out = p - center; out[2] *= 0.3
            side = np.sign(p[1]) if abs(p[1]) > 0.02 else 0.0
            dirv = out / (np.linalg.norm(out) + 1e-9)
            start = p + 0.9 * dirv + np.array([0.25 * (1 if code in ("100", "200", "700") else 0), 0.35 * side, 0.55])
            ax = rng.normal(size=3); ax /= np.linalg.norm(ax)
            q0 = np.zeros(4); mujoco.mju_axisAngle2Quat(q0, ax, 1.4)
            qs = np.zeros(4); mujoco.mju_mulQuat(qs, q0, q)
            sched[l] = (t + i * stagger, 1.0, start, qs, p, q)
        t_done = t + (n - 1) * stagger + 1.0
        timeline.append(dict(module=label, code=code, start=round(t, 2), done=round(t_done, 2), links=links))
        t = t_done + max(0.0, a.module_time - (n - 1) * stagger - 1.0)
    total = t + a.tail
    nframes = int(total * a.fps)
    if a.max_frames:
        nframes = min(nframes, a.max_frames)
    ren = mujoco.Renderer(m, a.height, a.width)
    cam = mujoco.MjvCamera()
    opt = mujoco.MjvOption()
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    writer = imageio.get_writer(a.out, fps=a.fps, codec="libx264", quality=None,
                                ffmpeg_params=["-crf", "17", "-pix_fmt", "yuv420p", "-preset", "medium"])
    for k in range(nframes):
        tt = k / a.fps
        for l, (t0, dur, start, qs, p, q) in sched.items():
            mid = m.body("part_" + l).mocapid[0]
            s = (tt - t0) / dur
            if s <= 0:
                d.mocap_pos[mid] = [0, 0, -5]  # hidden below the floor
            else:
                e = ease_out(s)
                d.mocap_pos[mid] = start + (p - start) * e
                d.mocap_quat[mid] = quat_slerp(qs, q, e)
            g, mat = base_mat[l]
            m.geom_matid[g] = hl if 0.75 < s < 1.45 else mat
        mujoco.mj_forward(m, d)
        # slow orbit, slightly rising
        u = np.clip(tt / (total - a.tail * 0.6), 0, 1)
        e = u * u * (3 - 2 * u)
        cam.lookat[:] = [0, 0, 0.36 + 0.26 * e]
        cam.distance = 1.55 + 0.75 * e
        cam.azimuth = 215 - 75 * e
        cam.elevation = -3 - 9 * e
        ren.update_scene(d, cam, opt)
        ren.scene.flags[mujoco.mjtRndFlag.mjRND_SHADOW] = True
        ren.scene.flags[mujoco.mjtRndFlag.mjRND_REFLECTION] = True
        writer.append_data(ren.render())
        if k % 30 == 0:
            print(f"frame {k}/{nframes}", flush=True)
    writer.close()
    json.dump(dict(fps=a.fps, total=total, modules=timeline), open(os.path.splitext(a.out)[0] + "_timeline.json", "w"), indent=1)
    print("wrote", a.out)


if __name__ == "__main__":
    main()
