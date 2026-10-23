"""Simulation scenarios for the video. Each returns a dict with qpos trajectory sampled at fps.

All physics: MuJoCo, official Asimov 1 25-DOF MJCF, 2 ms timestep, actuator torque limits
from the MJCF (actuatorfrcrange), joint friction from the Asimov 1 actuator table.
"""
import numpy as np, mujoco, json, os, sys
from asimov_sim import Robot, home_pose
from walking import WalkController, straight_steps
from wbc import WBC, path_plan, walk_plan

FPS = 30
WBC_KW = dict(kp_sw=1600, w_swing=100, kp_post=30, w_com=600, adapt=0)
PLAN_KW = dict(t_ss=0.28, t_ds=0.08, lift=0.05, zmp_in=0.02, apa=0.7, apa_t=0.15, end_brake=0.5,
               end_hold=0.1, t_ds0=0.25, swing_pitch=0.0)


class Recorder:
    def __init__(self, r, fps=FPS):
        self.r = r; self.fps = fps; self.q = []; self.t = []; self.next_t = r.d.time
        self.extra = []
    def tick(self, extra=None):
        if self.r.d.time + 1e-9 >= self.next_t:
            self.q.append(self.r.d.qpos.copy()); self.t.append(self.r.d.time)
            self.extra.append(extra if extra is not None else {})
            self.next_t += 1.0 / self.fps
    def result(self, **meta):
        return dict(qpos=np.array(self.q), t=np.array(self.t) - self.t[0], fps=self.fps, meta=meta, extra=self.extra)


def collapse(hold=1.0, after=2.5, fps=FPS):
    r = Robot(); r.set_pose(home_pose()); rec = Recorder(r, fps)
    while r.d.time < hold:
        r.step(1); rec.tick({"power": 1})
    r.power = False
    t_off = r.d.time
    while r.d.time < hold + after:
        r.step(1); rec.tick({"power": 0})
    return rec.result(power_off_at=t_off, head_start=1.10)


def pd_stand_push(F=80.0, direction=(-1, 0, 0), t_push=1.5, dur=0.1, T=4.5, fps=FPS):
    r = Robot(); r.set_pose(home_pose()); r.gain_scale = 1.0; rec = Recorder(r, fps)
    pushed = False
    while r.d.time < T:
        if not pushed and r.d.time >= t_push:
            r.ext_force = (r.torso, F * np.array(direction, float), r.d.time + dur); pushed = True
        r.step(1)
        rec.tick({"push": float(t_push <= r.d.time < t_push + dur)})
    return rec.result(push_at=t_push, force=F, direction=list(direction), fell=bool(r.upright() < 0.7))


def wbc_push(F=80.0, direction=(-1, 0, 0), t_push=1.5, dur=0.1, T=4.5, fps=FPS):
    r = Robot()
    c = WBC(r, path_plan([], t_ds0=0.25, apa=0.0, end_brake=0.0), **WBC_KW)
    r.gain_scale = 0; c.t0 = r.d.time; rec = Recorder(r, fps)
    pushed = False
    while r.d.time < T:
        if not pushed and r.d.time >= t_push:
            r.ext_force = (r.torso, F * np.array(direction, float), r.d.time + dur); pushed = True
        c.update(); r.step(1)
        rec.tick({"push": float(t_push <= r.d.time < t_push + dur)})
    return rec.result(push_at=t_push, force=F, direction=list(direction), fell=bool(r.upright() < 0.7))


def first_attempt(T=4.5, fps=FPS):
    """Development history: the first walking controller (IK + joint PD, no torso/momentum control)."""
    r = Robot()
    wc = WalkController(r, straight_steps(10, 0.05), knee=0.5, hip=0.3, k_tilt=1.0)
    wc.t0 = r.d.time; rec = Recorder(r, fps)
    fell_at = None
    while r.d.time - wc.t0 < T:
        wc.update(); r.step(5)
        rec.tick()
        if fell_at is None and r.upright() < 0.7:
            fell_at = r.d.time - wc.t0
    return rec.result(fell_at=fell_at)


def walk(cmds, settle=0.5, tail=1.5, record_forces=False):
    r = Robot()
    c = WBC(r, path_plan(cmds, **PLAN_KW), **WBC_KW)
    r.gain_scale = 0; c.t0 = r.d.time; rec = Recorder(r)
    done = False; steps_t = []
    prev = None
    while not done:
        done = c.update(); r.step(1)
        sw = c.log[-1][5] if c.log else None
        if sw != prev and sw is not None:
            steps_t.append(round(r.d.time - c.t0, 3))
        prev = sw
        rec.tick({"swing": sw, "contacts": getattr(c, "last_contacts", []), "com": getattr(c, "last_com", None)})
        if r.upright() < 0.7:
            break
    t_end = r.d.time
    while r.d.time < t_end + tail:
        c.update(); r.step(1); rec.tick({"swing": None})
    res = rec.result(steps=len(steps_t), step_times=steps_t, fell=bool(r.upright() < 0.7),
                     distance=float(np.linalg.norm(r.com()[:2] - rec.q[0][:2])))
    return res


def save(name, res, outdir="traj"):
    os.makedirs(outdir, exist_ok=True)
    np.savez_compressed(os.path.join(outdir, name + ".npz"), qpos=res["qpos"], t=res["t"], fps=res["fps"])
    json.dump(dict(res["meta"], frames=len(res["qpos"]), duration=float(res["t"][-1]),
                   extra=res["extra"]), open(os.path.join(outdir, name + ".json"), "w"))
    print(name, {k: v for k, v in res["meta"].items() if k != "step_times"}, "frames", len(res["qpos"]))


if __name__ == "__main__":
    which = sys.argv[1:] or ["collapse", "pd_push", "wbc_push", "first_attempt", "walk_straight", "walk_turn"]
    if "collapse" in which: save("collapse", collapse())
    if "pd_push" in which: save("pd_push", pd_stand_push(F=80.0))
    if "wbc_push" in which: save("wbc_push", wbc_push(F=80.0))
    if "first_attempt" in which: save("first_attempt", first_attempt())
    if "walk_straight" in which: save("walk_straight", walk([(0.15, 0.0)] * 20))
    if "first_attempt_long" in which: save("first_attempt_long", first_attempt(T=6.0))
    if "collapse60" in which: save("collapse60", collapse(hold=1.5, after=2.6, fps=60))
    if "pd_push_long" in which: save("pd_push_long", pd_stand_push(F=80.0, t_push=8.5, T=11.0))
    if "collapse_long" in which: save("collapse_long", collapse(hold=6.0, after=3.0))
    if "pd_push60" in which: save("pd_push60", pd_stand_push(F=80.0, fps=60, T=6.5))
    if "wbc_push60" in which: save("wbc_push60", wbc_push(F=80.0, fps=60, T=6.5))
    if "walk_turn" in which: save("walk_turn", walk([(0.12, 0.0)] * 4 + [(0.12, 0.1)] * 14 + [(0.12, 0.0)] * 4))
