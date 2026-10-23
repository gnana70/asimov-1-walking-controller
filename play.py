"""Watch Asimov 1 live in MuJoCo's viewer.

    python play.py walk        # 20 steps straight ahead with the whole-body controller
    python play.py turn        # curved walk (about 80 degrees)
    python play.py stand-push  # balance controller standing, shoved from the front at t=2 s
    python play.py pd-push     # joint springs only, same shove (falls)
    python play.py collapse    # no controller at all

Needs a display (Windows/macOS/Linux desktop). Close the window to quit.
"""
import sys, time
import numpy as np
import mujoco, mujoco.viewer
from asimov_sim import Robot, home_pose
from wbc import WBC, path_plan
from scenarios import WBC_KW, PLAN_KW

mode = sys.argv[1] if len(sys.argv) > 1 else "walk"
r = Robot()
ctrl = None
if mode in ("walk", "turn"):
    cmds = [(0.15, 0.0)] * 20 if mode == "walk" else [(0.12, 0.0)] * 4 + [(0.12, 0.1)] * 14 + [(0.12, 0.0)] * 4
    ctrl = WBC(r, path_plan(cmds, **PLAN_KW), **WBC_KW); r.gain_scale = 0
elif mode == "stand-push":
    ctrl = WBC(r, path_plan([], t_ds0=0.25, apa=0.0, end_brake=0.0), **WBC_KW); r.gain_scale = 0
elif mode == "pd-push":
    r.set_pose(home_pose()); r.gain_scale = 1.0
elif mode == "collapse":
    r.set_pose(home_pose()); r.power = False
else:
    sys.exit(__doc__)
if ctrl is not None:
    ctrl.t0 = r.d.time

with mujoco.viewer.launch_passive(r.m, r.d) as v:
    v.cam.distance = 3.0; v.cam.elevation = -12; v.cam.azimuth = 140
    pushed = False
    while v.is_running():
        t_wall = time.time()
        for _ in range(8):  # 8 x 2 ms = 16 ms of simulation per frame
            if mode in ("stand-push", "pd-push") and not pushed and r.d.time > 2.0:
                r.ext_force = (r.torso, np.array([-80.0, 0, 0]), r.d.time + 0.1); pushed = True
            if ctrl is not None:
                ctrl.update()
            r.step(1)
        v.cam.lookat[:] = [r.d.qpos[0], r.d.qpos[1], 0.55]
        v.sync()
        time.sleep(max(0.0, 0.016 - (time.time() - t_wall)))
