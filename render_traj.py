"""Render a recorded trajectory (traj/<name>.npz) with the styled Asimov 1 model.

Camera presets per clip live in CLIPS below. Optional overlays drawn into the 3D scene:
  push arrow (during the push), footstep trail, centre-of-mass ground marker.
"""
import os, sys, json, argparse
os.environ.setdefault("MUJOCO_GL", "egl")
import numpy as np
import mujoco
import imageio.v2 as imageio
from render_model import styled_model

CLIPS = {
    "collapse": dict(mode="fixed", lookat=[-0.1, 0, 0.45], distance=2.0, azimuth=125, elevation=-10,
                     drift=(-10, 0.0)),
    "pd_push": dict(mode="fixed", lookat=[-0.3, 0, 0.55], distance=2.6, azimuth=100, elevation=-8, push=True),
    "wbc_push": dict(mode="fixed", lookat=[-0.3, 0, 0.55], distance=2.6, azimuth=100, elevation=-8, push=True),
    "first_attempt": dict(mode="track", distance=2.2, azimuth=100, elevation=-10, z=0.48),
    "walk_straight": dict(mode="track", distance=2.1, azimuth=125, elevation=-12, z=0.52, az_drift=-40, trail=True),
    "walk_turn": dict(mode="track", distance=2.6, azimuth=210, elevation=-20, z=0.45, az_drift=45, trail=True),
    "walk_xray": dict(mode="track", distance=1.9, azimuth=145, elevation=-16, z=0.42, xray=True, src="walk_straight"),
    "hook": dict(mode="fixed", lookat=[-0.05, 0, 0.6], distance=2.5, azimuth=150, elevation=-5, src="collapse_long",
                 dist_drift=-0.55, drift=(-18, 0), drift_until=6.0),
    "pd_push_slow": dict(mode="fixed", lookat=[-0.3, 0, 0.55], distance=2.6, azimuth=100, elevation=-8, push=True,
                         src="pd_push60", speed=0.5),
    "wbc_push_slow": dict(mode="fixed", lookat=[-0.3, 0, 0.55], distance=2.6, azimuth=100, elevation=-8, push=True,
                          src="wbc_push60", speed=0.5),
    "pd_springs": dict(mode="fixed", lookat=[-0.15, 0, 0.55], distance=2.2, azimuth=140, elevation=-8, push=True,
                       src="pd_push_long", joints_until=8.8, drift=(-12, 0)),
    "hero": dict(mode="orbit", lookat=[0, 0, 0.62], distance=1.9, azimuth=150, elevation=-6, src="collapse",
                 frame=0, seconds=14.0, orbit=110),
    "collapse_slow": dict(mode="fixed", lookat=[-0.2, 0, 0.45], distance=2.4, azimuth=95, elevation=-6, src="collapse60",
                          speed=0.5),
    "first_attempt_long": dict(mode="track", distance=2.2, azimuth=100, elevation=-10, z=0.48),
    "feet_closeup": dict(mode="fixed", lookat=[0.0, 0.0, 0.06], distance=0.75, azimuth=145, elevation=-22,
                         src="collapse_long", xray=True, drift=(25, 0), max_t=5.5),
}


def add_arrow(scn, start, end, rgba, width=0.025):
    if scn.ngeom >= scn.maxgeom:
        return
    g = scn.geoms[scn.ngeom]
    mujoco.mjv_initGeom(g, mujoco.mjtGeom.mjGEOM_ARROW, np.zeros(3), np.zeros(3), np.zeros(9), np.array(rgba, np.float32))
    mujoco.mjv_connector(g, mujoco.mjtGeom.mjGEOM_ARROW, width, np.array(start, float), np.array(end, float))
    g.emission = 0.6
    scn.ngeom += 1


def add_disc(scn, pos, radius, rgba, height=0.002):
    if scn.ngeom >= scn.maxgeom:
        return
    g = scn.geoms[scn.ngeom]
    mujoco.mjv_initGeom(g, mujoco.mjtGeom.mjGEOM_CYLINDER, np.array([radius, radius, height]),
                        np.array(pos, float), np.eye(3).flatten(), np.array(rgba, np.float32))
    g.emission = 0.5
    scn.ngeom += 1


def add_box(scn, pos, size, yaw, rgba):
    if scn.ngeom >= scn.maxgeom:
        return
    g = scn.geoms[scn.ngeom]
    c, s = np.cos(yaw), np.sin(yaw)
    R = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])
    mujoco.mjv_initGeom(g, mujoco.mjtGeom.mjGEOM_BOX, np.array(size, float), np.array(pos, float), R.flatten(),
                        np.array(rgba, np.float32))
    g.emission = 0.04
    scn.ngeom += 1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("clip")
    ap.add_argument("--out", default=None)
    ap.add_argument("--width", type=int, default=1920)
    ap.add_argument("--height", type=int, default=1080)
    ap.add_argument("--fps", type=int, default=30)
    ap.add_argument("--max_frames", type=int, default=0)
    ap.add_argument("--traj_dir", default="traj")
    a = ap.parse_args()
    cfg = CLIPS[a.clip]
    src = cfg.get("src", a.clip)
    data = np.load(os.path.join(a.traj_dir, src + ".npz"))
    meta = json.load(open(os.path.join(a.traj_dir, src + ".json")))
    Q = data["qpos"]; T = data["t"]; fps_src = int(data["fps"])
    spec, m = styled_model(width=a.width, height=a.height)
    d = mujoco.MjData(m)
    ren = mujoco.Renderer(m, a.height, a.width)
    pass  # default scene
    cam = mujoco.MjvCamera(); opt = mujoco.MjvOption()
    if cfg.get("xray"):
        m.mat_rgba[m.material("shell").id][3] = 0.35
        m.mat_rgba[m.material("gold").id][3] = 0.5
        opt.flags[mujoco.mjtVisFlag.mjVIS_COM] = False
        m.vis.map.force = 0.004
        m.vis.scale.forcewidth = 0.025
        m.vis.scale.contactwidth = 0.03
        m.vis.scale.contactheight = 0.01
        m.vis.rgba.contactforce = np.array([1.0, 0.45, 0.1, 1.0], np.float32)
        for g in range(m.ngeom):
            n = m.geom(g).name or ""
            if "foot" in n and "collision" in n:
                m.geom_group[g] = 0
                m.geom_rgba[g] = [1.0, 0.45, 0.1, 1.0]
    speed = cfg.get("speed", 1.0)
    if cfg["mode"] == "orbit":
        nframes = int(cfg["seconds"] * a.fps)
    else:
        nframes = int(min(T[-1], cfg.get("max_t", 1e9)) / speed * a.fps) + 1
    if a.max_frames:
        nframes = min(nframes, a.max_frames)
    out = a.out or os.path.join("clips", a.clip + ".mp4")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    writer = imageio.get_writer(out, fps=a.fps, codec="libx264", quality=None,
                                ffmpeg_params=["-crf", "17", "-pix_fmt", "yuv420p", "-preset", "medium"])
    pel = m.body("pelvis_link").id
    torso = m.body("waist_yaw_link").id
    lf = m.body("left_ankle_roll_link").id; rf = m.body("right_ankle_roll_link").id
    track = None
    trail = []
    foot_prev, foot_z0 = {}, {}
    for k in range(nframes):
        tt = k / a.fps
        if cfg["mode"] == "orbit":
            q = Q[cfg.get("frame", 0)]
        else:
            idx = min(int(round(tt * speed * fps_src)), len(Q) - 1)
            q = Q[idx]
        d.qpos[:] = q
        mujoco.mj_kinematics(m, d); mujoco.mj_comPos(m, d); mujoco.mj_camlight(m, d)
        u = k / max(nframes - 1, 1)
        if cfg["mode"] == "fixed":
            if cfg.get("drift_until"):
                uu = min(tt / cfg["drift_until"], 1.0); u = uu * uu * (3 - 2 * uu)
            cam.lookat[:] = cfg["lookat"]; cam.distance = cfg["distance"] + cfg.get("dist_drift", 0) * u
            cam.azimuth = cfg["azimuth"] + cfg.get("drift", (0, 0))[0] * u
            cam.elevation = cfg["elevation"]
        elif cfg["mode"] == "orbit":
            cam.lookat[:] = cfg["lookat"]; cam.distance = cfg["distance"]
            cam.azimuth = cfg["azimuth"] + cfg["orbit"] * u; cam.elevation = cfg["elevation"]
        else:
            p = d.xpos[pel].copy()
            target = np.array([p[0], p[1], cfg.get("z", 0.5)])
            track = target if track is None else track + 0.08 * (target - track)
            cam.lookat[:] = track; cam.distance = cfg["distance"]
            cam.azimuth = cfg["azimuth"] + cfg.get("az_drift", 0) * u
            cam.elevation = cfg["elevation"]
        ren.update_scene(d, cam, opt)
        scn = ren.scene
        scn.flags[mujoco.mjtRndFlag.mjRND_SHADOW] = True
        scn.flags[mujoco.mjtRndFlag.mjRND_REFLECTION] = not cfg.get("xray", False)
        if cfg.get("joints_until"):
            ghost = tt * speed < cfg["joints_until"] - 0.6
            fade = min(1.0, max(0.0, (tt * speed - (cfg["joints_until"] - 0.6)) / 0.6))
            alpha_ = 0.42 + 0.58 * fade if not ghost else 0.42
            m.mat_rgba[m.material("shell").id][3] = alpha_
            m.mat_rgba[m.material("gold").id][3] = alpha_
        if cfg.get("joints_until") and tt * speed < cfg["joints_until"]:
            for j in range(1, m.njnt):
                if scn.ngeom < scn.maxgeom:
                    g = scn.geoms[scn.ngeom]
                    mujoco.mjv_initGeom(g, mujoco.mjtGeom.mjGEOM_SPHERE, np.array([0.03, 0, 0]), d.xanchor[j].copy(),
                                        np.eye(3).flatten(), np.array([1.0, 0.38, 0.08, 1.0], np.float32))
                    g.emission = 0.35
                    scn.ngeom += 1
        if cfg.get("push"):
            t_src = tt * speed
            if meta["push_at"] - 0.05 <= t_src <= meta["push_at"] + 0.35:
                dirv = np.array(meta["direction"], float)
                tip = d.xpos[torso] + np.array([0, 0, 0.18]) - dirv * 0.04
                add_arrow(scn, tip - dirv * 0.55, tip, [1.0, 0.25, 0.15, 1], width=0.05)
        if cfg.get("xray"):
            ex = meta["extra"][min(int(round(tt * fps_src)), len(meta["extra"]) - 1)]
            for (pt, fv) in ex.get("contacts", []):
                fv = np.array(fv)
                if fv[2] > 3:
                    add_arrow(scn, pt, np.array(pt) + fv * 0.0016, [1.0, 0.45, 0.12, 1], width=0.012)
            if ex.get("com"):
                cp = np.array(ex["com"])
                add_disc(scn, [cp[0], cp[1], 0.002], 0.025, [0.3, 0.85, 1.0, 1])
                add_arrow(scn, cp, [cp[0], cp[1], 0.03], [0.3, 0.85, 1.0, 0.8], width=0.006)
        if cfg.get("trail"):
            # one print per stance phase: the foot is planted when it is low and hasn't moved since the last frame
            for side, f in (("l", lf), ("r", rf)):
                pos = d.xpos[f].copy()
                if foot_z0.get(side) is None:
                    foot_z0[side] = pos[2]
                prev = foot_prev.get(side)
                foot_prev[side] = pos
                planted = prev is not None and np.linalg.norm(pos[:2] - prev[:2]) < 0.002 and pos[2] < foot_z0[side] + 0.006
                if planted:
                    R = d.xmat[f].reshape(3, 3); yaw = np.arctan2(R[1, 0], R[0, 0])
                    c = pos + R @ np.array([0.038, 0, 0])
                    mine = [t_ for t_ in trail if t_[2] == side]
                    if not mine or np.linalg.norm(mine[-1][0][:2] - c[:2]) > 0.05:
                        trail.append((np.array([c[0], c[1], 0.0012]), yaw, side))
            for (c, yaw, _) in trail:
                add_box(scn, c, [0.072, 0.028, 0.0008], yaw, [0.85, 0.40, 0.16, 0.45])
        writer.append_data(ren.render())
        if k % 30 == 0:
            print(f"{a.clip}: frame {k}/{nframes}", flush=True)
    writer.close()
    print("wrote", out)


if __name__ == "__main__":
    main()
