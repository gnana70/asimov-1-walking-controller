"""Whole-body torque controller (task-space inverse dynamics, TSID) for Asimov 1.

At every control tick a quadratic program chooses joint accelerations and
contact forces that
  * respect the floating-base equations of motion (Newton-Euler, 6 rows),
  * keep stance feet fixed (contact constraint),
  * keep contact forces inside a friction pyramid with positive normal force
    at the 4 sole corners of each stance foot (this bounds the CoP to the foot),
  * keep joint torques inside the actuator limits,
while tracking (soft, weighted):
  * the centre-of-mass trajectory from the ZMP preview planner,
  * an upright pelvis (this is what regulates angular momentum),
  * the swing-foot trajectory,
  * a joint posture (IK solution for legs, nominal pose for arms/waist/neck).
Torques are then  tau = M qdd + h - Jc^T f  for the actuated joints.
"""
import numpy as np
import mujoco
import quadprog
from asimov_sim import Robot, LEG_JOINTS, ALL_JOINTS
from walking import GaitPlan, LegIK, yaw_mat, pitch_mat, rot_err, home_pose, G, min_jerk

# sole corner points in the ankle-roll-link frame (centres of the MJCF contact spheres, minus radius)
CORNERS = np.array([[-0.045, 0.020, -0.034], [-0.045, -0.020, -0.034],
                    [0.122, 0.028, -0.034], [0.122, -0.028, -0.034]])


class WBC:
    def __init__(self, robot: Robot, plan_fn, knee=0.5, hip=0.3, mu=0.5,
                 w_com=30.0, w_ori=40.0, w_swing=40.0, w_post=1.0, w_force=1e-4,
                 kp_com=60.0, kp_ori=200.0, kp_sw=400.0, kp_post=100.0, arm_swing=0.25,
                 dcm_gain=0.0, pd_scale=0.0, adapt=True, k_adapt=(1.0, 0.5)):
        self.r = robot
        m = robot.m
        pose = home_pose(knee=knee, hip=hip)
        robot.set_pose(pose)
        d = robot.d
        self.ik = LegIK(m, pose)
        self.ik.init_from(d.qpos)
        fl, Rl = self.ik.foot_pose("L"); fr, Rr = self.ik.foot_pose("R")
        self.foot_z = {"L": fl[2], "R": fr[2]}
        self.foot_R0 = {"L": Rl, "R": Rr}
        self.pelvis_z = d.xpos[robot.pelvis][2]
        self.zc = self.ik.com()[2]
        self.plan = plan_fn((fl[0], fl[1], 0.0), (fr[0], fr[1], 0.0), self.zc)
        # start from the t=0 IK pose (CoM over the support centre)
        feet0, _ = self.plan.feet_at(0.0)
        tg = {f: (np.array([feet0[f][0][0], feet0[f][0][1], self.foot_z[f]]), self.foot_R0[f]) for f in "LR"}
        q0, _ = self.ik.solve(tg, self.plan.com[0], self.pelvis_z, np.eye(3), iters=30)
        pose = dict(pose)
        for n, v in zip(LEG_JOINTS, q0):
            pose[n] = v
        robot.set_pose(pose)
        d.qpos[0:2] = self.ik.d.qpos[0:2]
        mujoco.mj_forward(m, d)
        robot.q_des[:] = d.qpos[robot.qadr]
        self.ik.init_from(d.qpos)
        self.q_nom = d.qpos[robot.qadr].copy()
        self.mass = float(m.body_subtreemass[robot.pelvis])
        self.feet_b = {"L": m.body("left_ankle_roll_link").id, "R": m.body("right_ankle_roll_link").id}
        self.nv = m.nv
        self.na = len(ALL_JOINTS)
        self.mu = mu
        self.w = dict(com=w_com, ori=w_ori, swing=w_swing, post=w_post, force=w_force)
        self.kp = dict(com=kp_com, ori=kp_ori, sw=kp_sw, post=kp_post)
        self.arm_swing = arm_swing
        self.dcm_gain = dcm_gain
        self.pd_scale = pd_scale
        self.t0 = d.time
        self.M = np.zeros((m.nv, m.nv))
        self.log = []
        self.last_status = 0
        self.yaw_ref = 0.0
        self.landed = {"L": True, "R": True}
        self.stance_now = ["L", "R"]
        self.off = np.zeros(2)
        self.foot_off = {"L": np.zeros(2), "R": np.zeros(2)}
        self._plan_swing_prev = None
        self.adapt = adapt
        self.k_adapt = np.array(k_adapt, float)
        self.max_step_adj = np.array([0.06, 0.03])
        self.max_lat_off = 0.05

    # ------------------------------------------------------------------ helpers
    def foot_forces(self):
        m, d = self.r.m, self.r.d
        fz = {"L": 0.0, "R": 0.0}
        inv = {b: f for f, b in self.feet_b.items()}
        f6 = np.zeros(6)
        for i in range(d.ncon):
            c = d.contact[i]
            for g in (c.geom1, c.geom2):
                b = m.geom_bodyid[g]
                if b in inv:
                    mujoco.mj_contactForce(m, d, i, f6)
                    fz[inv[b]] += abs(f6[0])
        return fz

    def _jac_point(self, body, point):
        m, d = self.r.m, self.r.d
        jp = np.zeros((3, m.nv)); jr = np.zeros((3, m.nv))
        mujoco.mj_jac(m, d, jp, jr, point, body)
        jpd = np.zeros((3, m.nv)); jrd = np.zeros((3, m.nv))
        mujoco.mj_jacDot(m, d, jpd, jrd, point, body)
        return jp, jr, jpd, jrd

    def ref(self, t):
        p = self.plan
        t = min(max(t, 0.0), p.T - 1e-3)
        dt = 0.005
        feet, swing = p.feet_at(t)
        pitch = dict(p.last_pitch)
        feet_p, _ = p.feet_at(max(t - dt, 0)); pitch_p = dict(p.last_pitch)
        feet_n, _ = p.feet_at(min(t + dt, p.T - 1e-3)); pitch_n = dict(p.last_pitch)
        out = {}
        for f in "LR":
            pos = np.array([feet[f][0][0], feet[f][0][1], self.foot_z[f] + feet[f][0][2]])
            pp = np.array([feet_p[f][0][0], feet_p[f][0][1], feet_p[f][0][2]])
            pn = np.array([feet_n[f][0][0], feet_n[f][0][1], feet_n[f][0][2]])
            pc = np.array([feet[f][0][0], feet[f][0][1], feet[f][0][2]])
            vel = (pn - pp) / (2 * dt)
            acc = (pn - 2 * pc + pp) / (dt * dt)
            R = yaw_mat(feet[f][1]) @ pitch_mat(pitch[f]) @ self.foot_R0[f]
            pr_rate = (pitch_n[f] - pitch_p[f]) / (2 * dt)
            yr_rate = (feet_n[f][1] - feet_p[f][1]) / (2 * dt)
            out[f] = (pos, vel, acc, R, np.array([0.0, pr_rate, yr_rate]), feet[f][1])
        k = min(int(t / p.dt), len(p.times) - 1)
        c = p.com[k]; cv = p.com_vel[k]
        ca = (G / self.zc) * (p.com[k] - p.zmp_ref[k])
        return out, swing, c, cv, ca, feet

    # ------------------------------------------------------------------ main
    def update(self):
        r = self.r; m = r.m; d = r.d
        t = d.time - self.t0
        feet_ref, swing, c_ref, cv_ref, ca_ref, feet_raw = self.ref(t)
        # ---- discrete capture-point step adjustment: at each lift-off, absorb the
        #      measured DCM error into the plan (shifts the CoM reference and every
        #      future footstep, so the next step lands where the robot can catch itself)
        if self.adapt and swing is not None and swing != self._plan_swing_prev:
            jc = np.zeros((3, m.nv)); mujoco.mj_jacSubtreeCom(m, d, jc, r.pelvis)
            c_m = d.subtree_com[r.pelvis][:2]; v_m = (jc @ d.qvel)[:2]
            wn = np.sqrt(G / self.zc)
            e = (c_m + v_m / wn) - (c_ref + self.off + cv_ref / wn)
            step = np.clip(self.k_adapt * e, -self.max_step_adj, self.max_step_adj)
            self.off = self.off + step
            self.off[1] = np.clip(self.off[1], -self.max_lat_off, self.max_lat_off)
        if swing is None and self._plan_swing_prev is not None:
            self.foot_off[self._plan_swing_prev] = self.off.copy()
        self._plan_swing_prev = swing
        ws = 0.0
        if swing is not None:
            ws = min_jerk(np.clip((getattr(self.plan, "last_s", 0.0) - 0.1) / 0.8, 0, 1))
        for f in "LR":
            pos, vel, acc, Rr, wr, yw = feet_ref[f]
            o = self.foot_off[f] + ((self.off - self.foot_off[f]) * ws if f == swing else 0)
            pos = pos.copy(); pos[:2] += o
            feet_ref[f] = (pos, vel, acc, Rr, wr, yw)
        c_ref = c_ref + self.off
        # ---- contact-aware phase: early touchdown -> stance, late touchdown -> keep descending
        fz = self.foot_forces()
        if swing is not None:
            if self.landed.get(swing, True) and getattr(self.plan, "last_s", 0) < 0.5:
                self.landed[swing] = False
            s_sw = getattr(self.plan, "last_s", 0.0)
            if s_sw > 0.85 and fz[swing] > 40:
                self.landed[swing] = True
        swing_set = set()
        descend = set()
        for f in "LR":
            if f == swing and not self.landed.get(f, True):
                swing_set.add(f)
            elif f != swing and not self.landed.get(f, True):
                if fz[f] > 40:
                    self.landed[f] = True
                else:
                    swing_set.add(f); descend.add(f)
        stance = [f for f in "LR" if f not in swing_set]
        if not stance:  # never leave the robot with no support in the QP
            stance = [max("LR", key=lambda f: fz[f])]
            swing_set.discard(stance[0]); descend.discard(stance[0])
        self.stance_now = stance
        nv = self.nv; nc = 4 * len(stance)
        nx = nv + 3 * nc
        # IK for posture reference of the legs (keeps the QP away from joint limits)
        targets = {f: (feet_ref[f][0], feet_ref[f][3]) for f in "LR"}
        yaw_p = 0.5 * (feet_raw["L"][1] + feet_raw["R"][1])
        q_leg, _ = self.ik.solve(targets, c_ref, self.pelvis_z, yaw_mat(yaw_p), iters=3)
        q_post = self.q_nom.copy()
        q_post[[r.idx(n) for n in LEG_JOINTS]] = q_leg
        if self.arm_swing:
            dx = feet_raw["L"][0][0] - feet_raw["R"][0][0]
            q_post[r.idx("left_shoulder_pitch_joint")] = self.arm_swing * 2.5 * dx
            q_post[r.idx("right_shoulder_pitch_joint")] = self.arm_swing * 2.5 * dx
        r.q_des[:] = q_post

        mujoco.mj_fullM(m, d, self.M)
        h = d.qfrc_bias.copy()
        A_rows = []; b_rows = []; w_rows = []

        def task(J, b, w):
            A_rows.append(np.hstack([J, np.zeros((J.shape[0], 3 * nc))]))
            b_rows.append(b); w_rows.append(np.full(len(b), w))

        # --- CoM task
        jcom = np.zeros((3, nv))
        mujoco.mj_jacSubtreeCom(m, d, jcom, r.pelvis)
        com = d.subtree_com[r.pelvis].copy()
        comv = jcom @ d.qvel
        kp = self.kp["com"]; kd = 2 * np.sqrt(kp)
        a_com = np.zeros(3)
        a_com[:2] = ca_ref + kp * (c_ref - com[:2]) + kd * (cv_ref - comv[:2])
        a_com[2] = kp * (self.zc - com[2]) + kd * (0 - comv[2])
        task(jcom, a_com, self.w["com"])
        # --- pelvis orientation
        jp, jr, jpd, jrd = self._jac_point(r.pelvis, d.xpos[r.pelvis])
        Rp = d.xmat[r.pelvis].reshape(3, 3)
        e = rot_err(yaw_mat(yaw_p), Rp)
        wv = jr @ d.qvel
        kp = self.kp["ori"]; kd = 2 * np.sqrt(kp)
        task(jr, kp * e - kd * wv - jrd @ d.qvel, self.w["ori"])
        # --- feet
        Jc_list = []
        for f in "LR":
            b = self.feet_b[f]
            jp, jr, jpd, jrd = self._jac_point(b, d.xpos[b])
            v = jp @ d.qvel; w_ = jr @ d.qvel
            if f in swing_set:
                pos, vel, acc, Rref, wref, _ = feet_ref[f]
                if f in descend:
                    pos = pos.copy(); pos[2] = self.foot_z[f] - 0.015
                    vel = np.array([0, 0, -0.15]); acc = np.zeros(3); wref = np.zeros(3)
                kp = self.kp["sw"]; kd = 2 * np.sqrt(kp)
                a_lin = acc + kp * (pos - d.xpos[b]) + kd * (vel - v) - jpd @ d.qvel
                a_ang = kp * rot_err(Rref, d.xmat[b].reshape(3, 3)) + kd * (wref - w_) - jrd @ d.qvel
                task(jp, a_lin, self.w["swing"]); task(jr, a_ang, self.w["swing"])
            else:
                # contact: foot acceleration = -D * velocity (equality below)
                Jc_list.append((f, jp, jr, -jpd @ d.qvel - 20 * v, -jrd @ d.qvel - 20 * w_))
        # --- posture (actuated joints)
        Jq = np.zeros((self.na, nv)); Jq[np.arange(self.na), r.dadr] = 1
        kp = self.kp["post"]; kd = 2 * np.sqrt(kp)
        a_post = kp * (q_post - d.qpos[r.qadr]) - kd * d.qvel[r.dadr]
        wpost = np.full(self.na, self.w["post"])
        task(Jq, a_post, self.w["post"])

        A = np.vstack(A_rows); bvec = np.concatenate(b_rows); W = np.concatenate(w_rows)
        H = A.T @ (W[:, None] * A)
        g = A.T @ (W * bvec)
        H[np.diag_indices(nv)] += 1e-5
        H[nv:, nv:] += np.eye(3 * nc) * self.w["force"]

        # contact jacobian at corners (for forces)
        Jcorner = []  # list of (3, nv)
        for f in stance:
            b = self.feet_b[f]
            R = d.xmat[b].reshape(3, 3)
            for cpt in CORNERS:
                jp = np.zeros((3, nv))
                mujoco.mj_jac(m, d, jp, None, d.xpos[b] + R @ cpt, b)
                Jcorner.append(jp)
        Jc = np.vstack(Jcorner) if Jcorner else np.zeros((0, nv))

        # equality: floating base dynamics  M_b qdd - Jc_b^T f = -h_b
        Eq = [np.hstack([self.M[:6], -Jc[:, :6].T])]; eq = [-h[:6]]
        for (f, jp, jr, al, aa) in Jc_list:
            Eq.append(np.hstack([jp, np.zeros((3, 3 * nc))])); eq.append(al)
            Eq.append(np.hstack([jr, np.zeros((3, 3 * nc))])); eq.append(aa)
        Eq = np.vstack(Eq); eq = np.concatenate(eq)
        # inequalities C x >= d
        Ci = []; di = []
        for i in range(nc):
            row = np.zeros((5, nx)); base = nv + 3 * i
            row[0, base + 2] = 1                           # fz >= 0
            row[1, base + 2] = self.mu; row[1, base + 0] = -1   # mu fz - fx >= 0
            row[2, base + 2] = self.mu; row[2, base + 0] = 1
            row[3, base + 2] = self.mu; row[3, base + 1] = -1
            row[4, base + 2] = self.mu; row[4, base + 1] = 1
            Ci.append(row); di.append(np.array([2.0, 0, 0, 0, 0]))
        # torque limits: tau = Ma qdd + ha - Jca^T f
        Ma = self.M[r.dadr]
        T = np.hstack([Ma, -Jc[:, r.dadr].T]) if nc else Ma
        lim = r.tau_lim
        Ci.append(T); di.append(-lim - h[r.dadr])
        Ci.append(-T); di.append(-lim + h[r.dadr])
        Ci = np.vstack(Ci); di = np.concatenate(di)
        Cmat = np.vstack([Eq, Ci]).T
        dvec = np.concatenate([eq, di])
        try:
            sol = quadprog.solve_qp(H, g, Cmat, dvec, Eq.shape[0])[0]
            self.last_status = 0
        except ValueError:
            # fall back: drop torque limits
            Cmat = np.vstack([Eq, Ci[:5 * nc]]).T
            dvec = np.concatenate([eq, di[:5 * nc]])
            try:
                sol = quadprog.solve_qp(H, g, Cmat, dvec, Eq.shape[0])[0]
                self.last_status = 1
            except ValueError:
                self.last_status = 2
                r.tau_ff[:] = 0
                return t >= self.plan.T - 2e-3
        qdd = sol[:nv]; fc = sol[nv:]
        pts = []
        for f in stance:
            b = self.feet_b[f]; R = d.xmat[b].reshape(3, 3)
            for cpt in CORNERS:
                pts.append(d.xpos[b] + R @ cpt)
        self.last_contacts = [(pts[i].tolist(), fc[3 * i:3 * i + 3].tolist()) for i in range(len(pts))]
        self.last_com = com.tolist()
        tau = Ma @ qdd + h[r.dadr] - (Jc[:, r.dadr].T @ fc if nc else 0)
        r.tau_ff[:] = tau
        self.log.append((d.time, c_ref.copy(), com[:2].copy(), r.pelvis_height(), r.upright(), swing if swing_set else None))
        return t >= self.plan.T - 2e-3


def walk_plan(steps, **kw):
    def fn(fl, fr, zc):
        abs_steps = []
        for (f, x, dy, yaw) in steps:
            base = fl if f == "L" else fr
            abs_steps.append((f, base[0] + x, base[1] + dy, yaw))
        return GaitPlan(fl, fr, abs_steps, zc=zc, **kw)
    return fn


def path_plan(cmds, first="L", **kw):
    """cmds: list of (advance, dyaw) per step. Feet are placed around a body path."""
    def fn(fl, fr, zc):
        W = abs(fl[1] - fr[1])
        p = 0.5 * (np.array(fl[:2]) + np.array(fr[:2]))
        th = 0.0
        steps = []
        foot = first
        for (adv, dth) in cmds:
            th_mid = th + 0.5 * dth
            th += dth
            p = p + np.array([np.cos(th_mid), np.sin(th_mid)]) * adv
            side = 1 if foot == "L" else -1
            fpos = p + np.array([-np.sin(th), np.cos(th)]) * side * W / 2
            steps.append((foot, fpos[0], fpos[1], th))
            foot = "R" if foot == "L" else "L"
        # closing step
        side = 1 if foot == "L" else -1
        fpos = p + np.array([-np.sin(th), np.cos(th)]) * side * W / 2
        steps.append((foot, fpos[0], fpos[1], th))
        return GaitPlan(fl, fr, steps, zc=zc, **kw)
    return fn
