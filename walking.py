"""Model-based walking controller for Asimov 1 (MuJoCo).

Pipeline (classic Kajita-style humanoid walking):
  1. Footstep plan  -> ZMP reference (centre of each support foot)
  2. ZMP preview control on the linear inverted pendulum -> CoM trajectory
  3. Swing-foot trajectories
  4. Whole-body inverse kinematics (floating base + 12 leg joints) so the
     *whole-body* centre of mass and both feet follow the plan
  5. Joint-level PD (in asimov_sim.Robot) tracks the IK joint angles
  6. Feedback: CoM / DCM error shifts the IK CoM target, torso tilt feeds
     back into the hip joints.
"""
import numpy as np
import mujoco
from scipy.linalg import solve_discrete_are
from asimov_sim import Robot, LEG_JOINTS, ALL_JOINTS, home_pose

G = 9.81


# ---------------------------------------------------------------- preview control
def preview_gains(zc, dt, n_preview, qe=1.0, r=1e-6):
    A = np.array([[1, dt, dt * dt / 2], [0, 1, dt], [0, 0, 1]])
    B = np.array([[dt ** 3 / 6], [dt * dt / 2], [dt]])
    C = np.array([[1, 0, -zc / G]])
    At = np.block([[np.eye(1), C @ A], [np.zeros((3, 1)), A]])
    Bt = np.vstack([C @ B, B])
    It = np.vstack([np.eye(1), np.zeros((3, 1))])
    Q = np.zeros((4, 4)); Q[0, 0] = qe
    R = np.array([[r]])
    P = solve_discrete_are(At, Bt, Q, R)
    S = np.linalg.inv(R + Bt.T @ P @ Bt)
    K = S @ Bt.T @ P @ At
    Gi = K[0, 0]
    Gx = K[0, 1:]
    Ac = At - Bt @ K
    X = -Ac.T @ P @ It
    Gd = np.zeros(n_preview)
    Gd[0] = -Gi
    for i in range(1, n_preview):
        Gd[i] = (S @ Bt.T @ X)[0, 0]
        X = Ac.T @ X
    return A, B, C, Gi, Gx, Gd


def preview_com(zmp_ref, zc, dt, n_preview=160, x0=None):
    """zmp_ref: (T,) for one axis. returns com pos, vel, acc and zmp (T,)"""
    A, B, C, Gi, Gx, Gd = preview_gains(zc, dt, n_preview)
    T = len(zmp_ref)
    ref = np.concatenate([zmp_ref, np.full(n_preview, zmp_ref[-1])])
    x = np.array([zmp_ref[0], 0, 0.0]) if x0 is None else np.array(x0, float)
    esum = 0.0
    out = np.zeros((T, 3)); zmp = np.zeros(T)
    for k in range(T):
        p = (C @ x)[0]
        esum += p - ref[k]
        u = -Gi * esum - Gx @ x - Gd @ ref[k + 1:k + 1 + n_preview]
        out[k] = x; zmp[k] = p
        x = A @ x + B[:, 0] * u
    return out, zmp


# ---------------------------------------------------------------- gait plan
def min_jerk(s):
    s = np.clip(s, 0, 1)
    return 10 * s ** 3 - 15 * s ** 4 + 6 * s ** 5


def pitch_mat(a):
    c, s = np.cos(a), np.sin(a)
    return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])


def yaw_mat(yaw):
    c, s = np.cos(yaw), np.sin(yaw)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])


class GaitPlan:
    """Generates a walk: list of footsteps (x, y, yaw) alternating feet."""

    def __init__(self, foot0_l, foot0_r, steps, t_ss=0.55, t_ds=0.2, t_start=1.2, t_end=1.5,
                 dt=0.01, zc=0.6, lift=0.05, zmp_off=np.array([0.035, 0.0]), zmp_in=0.015, apa=0.0, apa_t=0.15, t_ds0=0.35, end_brake=0.0, end_hold=0.1, swing_pitch=0.0):
        self.dt = dt
        self.zc = zc
        self.lift = lift
        self.swing_pitch = swing_pitch
        self.zmp_off = zmp_off
        # footsteps: sequence of (foot, x, y, yaw); first swing foot given by steps
        self.events = []  # (t0, t1, kind, data)
        L = np.array(foot0_l, float); Rf = np.array(foot0_r, float)  # x,y,yaw
        t = 0.0
        zmp_pts = []  # (time, zmp xy)

        def foot_zmp(f, which=None):
            off = zmp_off.copy()
            if which is not None:
                off[1] += -zmp_in if which == "L" else zmp_in
            return f[:2] + yaw_mat(f[2])[:2, :2] @ off

        mid = 0.5 * (foot_zmp(L) + foot_zmp(Rf))
        zmp_pts.append((0.0, mid))
        t = t_start
        zmp_pts.append((t, mid))
        if apa:
            # anticipatory postural adjustment: briefly load the first swing foot
            first = steps[0][0]
            sw0 = L if first == "L" else Rf
            zmp_pts.append((t + apa_t, mid + apa * (foot_zmp(sw0) - mid)))
            zmp_pts.append((t + 2 * apa_t, mid + apa * (foot_zmp(sw0) - mid)))
            t += 2 * apa_t
        feet_timeline = []  # (t0, t1, swing_foot, start, end, stance)
        for (foot, x, y, yaw) in steps:
            stance = Rf if foot == "L" else L
            # double support: shift zmp to stance foot
            t_ds_now = t_ds if len(feet_timeline) else t_ds0
            zmp_pts.append((t + t_ds_now, foot_zmp(stance, 'R' if foot == 'L' else 'L')))
            t += t_ds_now
            target = np.array([x, y, yaw])
            start = (L if foot == "L" else Rf).copy()
            feet_timeline.append((t, t + t_ss, foot, start, target))
            zmp_pts.append((t + t_ss, foot_zmp(stance, 'R' if foot == 'L' else 'L')))
            t += t_ss
            if foot == "L":
                L = target
            else:
                Rf = target
        # final double support back to the middle
        mid = 0.5 * (foot_zmp(L) + foot_zmp(Rf))
        if end_brake:
            last = steps[-1][0]
            landed = L if last == "L" else Rf
            zmp_pts.append((t + t_ds, mid + end_brake * (foot_zmp(landed) - mid)))
            zmp_pts.append((t + t_ds + end_hold, mid + end_brake * (foot_zmp(landed) - mid)))
            t += t_ds + end_hold
        zmp_pts.append((t + 0.35, mid))
        t += t_end
        zmp_pts.append((t, mid))
        self.T = t
        self.feet_timeline = feet_timeline
        self.foot0 = {"L": np.array(foot0_l, float), "R": np.array(foot0_r, float)}
        # sample zmp ref
        times = np.arange(0, t, dt)
        tp = np.array([p[0] for p in zmp_pts])
        xp = np.array([p[1][0] for p in zmp_pts]); yp = np.array([p[1][1] for p in zmp_pts])
        self.times = times
        self.zmp_ref = np.stack([np.interp(times, tp, xp), np.interp(times, tp, yp)], 1)
        cx, zx = preview_com(self.zmp_ref[:, 0], zc, dt)
        cy, zy = preview_com(self.zmp_ref[:, 1], zc, dt)
        self.com = np.stack([cx[:, 0], cy[:, 0]], 1)
        self.com_vel = np.stack([cx[:, 1], cy[:, 1]], 1)
        self.zmp = np.stack([zx, zy], 1)

    def feet_at(self, t):
        """returns dict foot -> (pos xyz of ankle-roll frame offset (z = lift), yaw), and swing flag"""
        res = {}
        res_pitch = {"L": 0.0, "R": 0.0}
        swing = None
        for f in ("L", "R"):
            # last known placement before t
            pose = self.foot0[f].copy()
            z = 0.0
            for (t0, t1, foot, start, target) in self.feet_timeline:
                if foot != f:
                    continue
                if t >= t1:
                    pose = target.copy()
                elif t0 <= t < t1:
                    s = (t - t0) / (t1 - t0)
                    w = min_jerk((s - 0.1) / 0.8)
                    pose = start + (target - start) * w
                    sz = min(s / 0.85, 1.0)
                    z = self.lift * np.sin(np.pi * sz) ** 1.5 if s < 0.85 else -0.008 * (s - 0.85) / 0.15
                    self.last_s = s
                    pitch = self.swing_pitch * np.sin(np.pi * min(1.0, s * 1.15)) if s < 0.87 else 0.0
                    swing = f
                    res_pitch[f] = pitch
                    break
                else:
                    break
            res[f] = (np.array([pose[0], pose[1], z]), pose[2])
        self.last_pitch = res_pitch
        return res, swing

    def com_at(self, t):
        k = min(int(t / self.dt), len(self.times) - 1)
        k2 = min(k + 1, len(self.times) - 1)
        a = (t - k * self.dt) / self.dt
        return (1 - a) * self.com[k] + a * self.com[k2], (1 - a) * self.com_vel[k] + a * self.com_vel[k2]


# ---------------------------------------------------------------- whole-body IK
class LegIK:
    def __init__(self, model, q_upper):
        self.m = model
        self.d = mujoco.MjData(model)
        self.leg_dofs = np.array([model.jnt_dofadr[model.joint(n).id] for n in LEG_JOINTS])
        self.leg_qadr = np.array([model.jnt_qposadr[model.joint(n).id] for n in LEG_JOINTS])
        self.cols = np.concatenate([np.arange(6), self.leg_dofs])
        self.feet = {"L": model.body("left_ankle_roll_link").id, "R": model.body("right_ankle_roll_link").id}
        self.pelvis = model.body("pelvis_link").id
        self.q_upper = q_upper  # dict
        self.lo = model.jnt_range[[model.joint(n).id for n in LEG_JOINTS], 0]
        self.hi = model.jnt_range[[model.joint(n).id for n in LEG_JOINTS], 1]

    def init_from(self, qpos):
        self.d.qpos[:] = qpos
        mujoco.mj_kinematics(self.m, self.d)
        mujoco.mj_comPos(self.m, self.d)

    def foot_pose(self, f):
        b = self.feet[f]
        return self.d.xpos[b].copy(), self.d.xmat[b].reshape(3, 3).copy()

    def com(self):
        return self.d.subtree_com[self.pelvis].copy()

    def solve(self, feet_targets, com_xy, pelvis_z, pelvis_R, iters=6, w_com=1.0):
        m, d = self.m, self.d
        nv = m.nv
        jp = np.zeros((3, nv)); jr = np.zeros((3, nv))
        for it in range(iters):
            mujoco.mj_kinematics(m, d)
            mujoco.mj_comPos(m, d)
            rows = []; errs = []
            for f, (p_des, R_des) in feet_targets.items():
                b = self.feet[f]
                mujoco.mj_jacBody(m, d, jp, jr, b)
                rows.append(jp[:, self.cols].copy()); errs.append(p_des - d.xpos[b])
                R = d.xmat[b].reshape(3, 3)
                rows.append(jr[:, self.cols].copy()); errs.append(rot_err(R_des, R))
            mujoco.mj_jacSubtreeCom(m, d, jp, self.pelvis)
            rows.append(w_com * jp[:2, self.cols].copy()); errs.append(w_com * (com_xy - d.subtree_com[self.pelvis][:2]))
            mujoco.mj_jacBody(m, d, jp, jr, self.pelvis)
            rows.append(jp[2:3, self.cols].copy()); errs.append(np.array([pelvis_z - d.xpos[self.pelvis][2]]))
            rows.append(jr[:, self.cols].copy()); errs.append(rot_err(pelvis_R, d.xmat[self.pelvis].reshape(3, 3)))
            J = np.vstack(rows); e = np.concatenate(errs)
            lam = 1e-4
            dx = J.T @ np.linalg.solve(J @ J.T + lam * np.eye(J.shape[0]), e)
            dq = np.zeros(nv); dq[self.cols] = dx
            mujoco.mj_integratePos(m, d.qpos, dq, 1.0)
            d.qpos[self.leg_qadr] = np.clip(d.qpos[self.leg_qadr], self.lo, self.hi)
            if np.abs(e).max() < 1e-5:
                break
        return d.qpos[self.leg_qadr].copy(), np.abs(e).max()


def rot_err(R_des, R):
    Re = R_des @ R.T
    q = np.zeros(4); mujoco.mju_mat2Quat(q, Re.flatten())
    v = np.zeros(3); mujoco.mju_quat2Vel(v, q, 1.0)
    return v


# ---------------------------------------------------------------- controller
class WalkController:
    def __init__(self, robot: Robot, steps, knee=0.6, t_ss=0.28, t_ds=0.08, lift=0.04,
                 k_com=0.0, k_tilt=0.0, arm_swing=0.25, zmp_in=0.02, apa=0.7, apa_t=0.15,
                 end_brake=0.5, end_hold=0.1, t_ds0=0.25, hip=None, swing_pitch=0.25, ff=1.0):
        self.r = robot
        m = robot.m
        pose = home_pose(knee=knee, hip=hip)
        robot.set_pose(pose)
        self.pose = pose
        d = robot.d
        self.ik = LegIK(m, pose)
        self.ik.init_from(d.qpos)
        fl, Rl = self.ik.foot_pose("L"); fr, Rr = self.ik.foot_pose("R")
        self.foot_z = {"L": fl[2], "R": fr[2]}
        self.pelvis_z = d.xpos[robot.pelvis][2]
        com = self.ik.com()
        self.zc = com[2]
        self.foot_R0 = {"L": Rl, "R": Rr}
        foot0_l = (fl[0], fl[1], 0.0); foot0_r = (fr[0], fr[1], 0.0)
        # steps given relative to the start: list of (foot, dx, dy, yaw)
        abs_steps = []
        for (f, x, y, yaw) in steps:
            base = foot0_l if f == "L" else foot0_r
            abs_steps.append((f, base[0] + x, base[1] + y if False else y_abs(f, y, foot0_l, foot0_r), yaw))
        self.plan = GaitPlan(foot0_l, foot0_r, abs_steps, t_ss=t_ss, t_ds=t_ds, zc=self.zc, lift=lift, zmp_in=zmp_in,
                             apa=apa, apa_t=apa_t, end_brake=end_brake, end_hold=end_hold, t_ds0=t_ds0,
                             swing_pitch=swing_pitch)
        # CoM reference comes straight from the plan (above the support centre).
        self.com_offset = np.zeros(2)
        # Re-pose the robot to the t=0 IK solution so the start is consistent.
        feet0, _ = self.plan.feet_at(0.0)
        tg = {f: (np.array([feet0[f][0][0], feet0[f][0][1], self.foot_z[f]]), self.foot_R0[f]) for f in ("L", "R")}
        q0, _ = self.ik.solve(tg, self.plan.com[0], self.pelvis_z, np.eye(3), iters=30)
        pose = dict(pose)
        for n, v in zip(LEG_JOINTS, q0):
            pose[n] = v
        robot.set_pose(pose)
        base = self.ik.d.qpos[:7].copy()
        d.qpos[0:2] = base[0:2]
        mujoco.mj_forward(m, d)
        robot.q_des[:] = d.qpos[robot.qadr]
        self.ik.init_from(d.qpos)
        self.ik.d.qpos[2] = d.qpos[2]
        self.k_com = k_com
        self.ff = ff
        self.mass = float(m.body_subtreemass[robot.pelvis])
        self.k_tilt = k_tilt
        self.arm_swing = arm_swing
        self.t0 = d.time
        self.last_q = None
        self.ik_err = 0
        self.log = []

    def update(self):
        r = self.r; d = r.d
        t = d.time - self.t0
        t = min(t, self.plan.T - 1e-3)
        feet, swing = self.plan.feet_at(t)
        com_ref, comv_ref = self.plan.com_at(t)
        com_ref = com_ref + self.com_offset
        # feedback: measured com vs reference (in world)
        com_meas = r.com()[:2]
        comv_meas = (d.subtree_linvel[r.pelvis][:2]).copy()
        com_cmd = com_ref.copy()
        if self.k_com:
            # Divergent component of motion feedback
            w = np.sqrt(G / self.zc)
            dcm_err = (com_meas + comv_meas / w) - (com_ref + comv_ref / w)
            com_cmd = com_ref - self.k_com * dcm_err
        targets = {}
        for f in ("L", "R"):
            p, yaw = feet[f]
            pr = pitch_mat(self.plan.last_pitch[f])
            targets[f] = (np.array([p[0], p[1], self.foot_z[f] + p[2]]), yaw_mat(yaw) @ pr @ self.foot_R0[f])
        yaw_p = 0.5 * (feet["L"][1] + feet["R"][1])
        q_leg, err = self.ik.solve(targets, com_cmd, self.pelvis_z, yaw_mat(yaw_p))
        self.ik_err = err
        idx = [r.idx(n) for n in LEG_JOINTS]
        r.q_des[idx] = q_leg
        if self.ff:
            self.feedforward(t, swing, feet)
        # torso tilt feedback on hips (stance)
        if self.k_tilt:
            R = d.xmat[r.pelvis].reshape(3, 3)
            pitch = np.arctan2(-R[2, 0], R[2, 2])  # forward lean +
            roll = np.arctan2(R[2, 1], R[2, 2])
            for f, sgn in (("L", 1), ("R", -1)):
                if swing != f:
                    hp = r.idx(("left" if f == "L" else "right") + "_hip_pitch_joint")
                    hr = r.idx(("left" if f == "L" else "right") + "_hip_roll_joint")
                    r.q_des[hp] += sgn * self.k_tilt * pitch
                    r.q_des[hr] += self.k_tilt * roll
        # arm swing (opposite to legs)
        if self.arm_swing:
            dx = feet["L"][0][0] - feet["R"][0][0]
            sp_l = r.idx("left_shoulder_pitch_joint"); sp_r = r.idx("right_shoulder_pitch_joint")
            r.q_des[sp_l] = self.arm_swing * 2.5 * dx
            r.q_des[sp_r] = self.arm_swing * 2.5 * dx
        self.log.append((d.time, com_ref.copy(), com_meas.copy(), r.pelvis_height(), r.upright(), swing))
        return t >= self.plan.T - 2e-3


def _ff(self, t, swing, feet):
    """Static/LIPM contact-force feedforward: tau = h - J_c^T F_c (actuated dofs)."""
    r = self.r; m = r.m; d = r.d
    k = min(int(t / self.plan.dt), len(self.plan.times) - 1)
    zmp = self.plan.zmp_ref[k]
    com_ref = self.plan.com[k]
    w2 = G / self.zc
    acc = w2 * (com_ref - zmp)
    M = self.mass
    F = np.array([M * acc[0], M * acc[1], M * G])
    # weight split
    pl = feet["L"][0][:2]; pr = feet["R"][0][:2]
    if swing == "L":
        alpha = {"L": 0.0, "R": 1.0}
    elif swing == "R":
        alpha = {"L": 1.0, "R": 0.0}
    else:
        v = pl - pr
        a = np.clip(np.dot(zmp - pr, v) / max(np.dot(v, v), 1e-9), 0, 1)
        alpha = {"L": a, "R": 1 - a}
    tau = d.qfrc_bias[r.dadr].copy()
    jp = np.zeros((3, m.nv))
    for f in ("L", "R"):
        if alpha[f] <= 1e-3:
            continue
        b = self.ik.feet[f]
        R = d.xmat[b].reshape(3, 3)
        pt = d.xpos[b] + R @ np.array([self.plan.zmp_off[0], 0, -0.034])
        mujoco.mj_jac(m, d, jp, None, pt, b)
        tau -= (jp[:, r.dadr].T @ (alpha[f] * F))
    r.tau_ff[:] = self.ff * tau


WalkController.feedforward = _ff


def y_abs(f, y, fl, fr):
    return (fl[1] if f == "L" else fr[1]) + y


def straight_steps(n, length=0.12):
    """n steps forward, alternating starting with left. returns relative (foot, x, dy, yaw)"""
    steps = []
    xL = 0.0; xR = 0.0
    for i in range(n):
        f = "L" if i % 2 == 0 else "R"
        last = (i == n - 1)
        if f == "L":
            xL = xR + length if not last else xR
            steps.append(("L", xL, 0.0, 0.0))
        else:
            xR = xL + length if not last else xL
            steps.append(("R", xR, 0.0, 0.0))
    return steps
