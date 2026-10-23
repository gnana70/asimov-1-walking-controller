import sys, numpy as np, mujoco
from asimov_sim import Robot, home_pose
from wbc import WBC, path_plan
def trial(mode, F, direction=(1,0,0), dur=0.1, T=3.0):
    r = Robot()
    if mode == "pd":
        r.set_pose(home_pose(knee=0.5, hip=0.3)); r.gain_scale = 1.0
        step = lambda: r.step(1)
    else:
        c = WBC(r, path_plan([], t_ds0=0.25, apa=0.0, end_brake=0.0), kp_sw=1600, w_swing=100, kp_post=30, w_com=600, adapt=0)
        r.gain_scale = 0; c.t0 = r.d.time
        def step(): c.update(); r.step(1)
    for _ in range(500): step()
    r.ext_force = (r.torso, F*np.array(direction, float), r.d.time + dur)
    for _ in range(int(T/0.002)):
        step()
        if r.upright() < 0.7: return False
    return True
for mode in ["pd", "wbc"]:
    for d in [(1,0,0),(-1,0,0),(0,1,0)]:
        best = 0
        for F in [50,100,150,200,250,300,400,500]:
            if trial(mode, F, d): best = F
            else: break
        print(mode, d, "max survived push (N, 0.1 s):", best)
