"""Asimov 1 simulation core.

Loads the official 25-DOF MJCF from the menloresearch/asimov-1 repo, adds the
actuator properties published in the Asimov 1 manual (armature already in the
MJCF; friction from the actuator table) and exposes a torque-level PD
interface that mimics the joint-level PD loop running in the real actuators.
"""
import os
import numpy as np
import mujoco

HERE_ = os.path.dirname(os.path.abspath(__file__))
REPO = os.environ.get("ASIMOV_REPO") or next((p for p in (os.path.join(HERE_, "asimov-1"), os.path.join(HERE_, "..", "asimov-1")) if os.path.isdir(p)), os.path.join(HERE_, "asimov-1"))
MJCF = os.path.join(REPO, "sim-model", "asimov_1_25dof", "xml", "asimov_1_25dof.xml")

LEG_JOINTS = [
    "left_hip_pitch_joint", "left_hip_roll_joint", "left_hip_yaw_joint",
    "left_knee_joint", "left_ankle_pitch_joint", "left_ankle_roll_joint",
    "right_hip_pitch_joint", "right_hip_roll_joint", "right_hip_yaw_joint",
    "right_knee_joint", "right_ankle_pitch_joint", "right_ankle_roll_joint",
]
UPPER_JOINTS = [
    "waist_yaw_joint", "neck_yaw_joint", "neck_pitch_joint",
    "right_shoulder_pitch_joint", "right_shoulder_roll_joint", "right_shoulder_yaw_joint",
    "right_elbow_joint", "right_wrist_yaw_joint",
    "left_shoulder_pitch_joint", "left_shoulder_roll_joint", "left_shoulder_yaw_joint",
    "left_elbow_joint", "left_wrist_yaw_joint",
]
ALL_JOINTS = LEG_JOINTS + UPPER_JOINTS

# Actuator families from the Asimov 1 manual (mechanical system tour):
# family: (rated Nm, peak Nm, velocity limit rad/s, static friction Nm)
FAMILIES = {
    "EC-A6416-P2-25": (40, 120, 12.57, 0.70),
    "EC-A5013-H17-100": (30, 90, 3.98, 0.20),
    "EC-A3814-H14-107": (20, 60, 5.45, 0.70),
    "EC-A4315-P2-36": (25, 75, 12.25, 0.70),
    "EC-A4310-P2-36-ankle": (40, 145.4, 9.32, 0.40),
    "EC-A4310-P2-36": (12, 36, 9.32, 0.40),
}
JOINT_FAMILY = {
    "hip_pitch": "EC-A6416-P2-25", "waist_yaw": "EC-A6416-P2-25",
    "hip_roll": "EC-A5013-H17-100", "shoulder_pitch": "EC-A5013-H17-100",
    "hip_yaw": "EC-A3814-H14-107", "shoulder_yaw": "EC-A3814-H14-107",
    "knee": "EC-A4315-P2-36", "shoulder_roll": "EC-A4315-P2-36",
    "ankle_pitch": "EC-A4310-P2-36-ankle", "ankle_roll": "EC-A4310-P2-36-ankle",
    "elbow": "EC-A4310-P2-36", "wrist_yaw": "EC-A4310-P2-36",
    "neck_yaw": "EC-A4310-P2-36", "neck_pitch": "EC-A4310-P2-36",
}

# Joint-space PD gains (Nm/rad, Nm s/rad).
GAINS = {
    "hip_pitch": (220, 8), "hip_roll": (260, 8), "hip_yaw": (150, 5),
    "knee": (260, 8), "ankle_pitch": (140, 4), "ankle_roll": (90, 3),
    "waist_yaw": (180, 6), "neck_yaw": (20, 1), "neck_pitch": (20, 1),
    "shoulder_pitch": (60, 2.5), "shoulder_roll": (60, 2.5), "shoulder_yaw": (40, 1.5),
    "elbow": (40, 1.5), "wrist_yaw": (20, 1),
}


def joint_kind(name):
    n = name.replace("left_", "").replace("right_", "").replace("_joint", "")
    return n


# Home pose: slightly bent knees (left knee +, right knee - because the right
# leg axes are mirrored in the MJCF), arms relaxed at the sides.
def home_pose(knee=0.55, hip=None):
    hip = knee * 0.5 if hip is None else hip
    ank = knee - hip
    q = {
        "left_hip_pitch_joint": -hip, "left_knee_joint": knee, "left_ankle_pitch_joint": -ank,
        "right_hip_pitch_joint": hip, "right_knee_joint": -knee, "right_ankle_pitch_joint": ank,
        "left_shoulder_roll_joint": -0.12, "right_shoulder_roll_joint": 0.12,
        "left_elbow_joint": 1.05, "right_elbow_joint": -1.05,
        "left_wrist_yaw_joint": 1.57, "right_wrist_yaw_joint": -1.57,
    }
    return q


def build_model(timestep=0.002, with_motors=True, floor_rgba=None):
    spec = mujoco.MjSpec.from_file(MJCF)
    spec.option.timestep = timestep
    for jn in ALL_JOINTS:
        j = spec.joint(jn)
        fam = FAMILIES[JOINT_FAMILY[joint_kind(jn)]]
        j.frictionloss = fam[3] * 0.5  # half of static friction as Coulomb loss
        if with_motors:
            a = spec.add_actuator()
            a.name = jn.replace("_joint", "_motor")
            a.target = jn
            a.trntype = mujoco.mjtTrn.mjTRN_JOINT
            a.set_to_motor()
            lim = float(abs(j.actfrcrange[1])) if j.actfrcrange[1] > 0 else fam[1]
            a.ctrlrange = [-lim, lim]
            a.ctrllimited = 1
    model = spec.compile()
    return spec, model


class Robot:
    def __init__(self, model=None, timestep=0.002):
        if model is None:
            _, model = build_model(timestep)
        self.m = model
        self.d = mujoco.MjData(model)
        m = self.m
        self.jid = np.array([m.joint(n).id for n in ALL_JOINTS])
        self.qadr = m.jnt_qposadr[self.jid]
        self.dadr = m.jnt_dofadr[self.jid]
        self.aid = np.array([m.actuator(n.replace("_joint", "_motor")).id for n in ALL_JOINTS]) if m.nu else None
        self.kp = np.array([GAINS[joint_kind(n)][0] for n in ALL_JOINTS], float)
        self.kd = np.array([GAINS[joint_kind(n)][1] for n in ALL_JOINTS], float)
        self.tau_lim = np.array([m.actuator_ctrlrange[a][1] for a in self.aid]) if m.nu else None
        self.q_des = np.zeros(len(ALL_JOINTS))
        self.qd_des = np.zeros(len(ALL_JOINTS))
        self.tau_ff = np.zeros(len(ALL_JOINTS))
        self.power = True
        self.gain_scale = 1.0
        self.pelvis = m.body("pelvis_link").id
        self.torso = m.body("waist_yaw_link").id
        self.ext_force = None  # (body_id, force xyz, until_time)

    def idx(self, name):
        return ALL_JOINTS.index(name)

    def set_pose(self, qdict, base_z=None, settle=True):
        d = self.d
        mujoco.mj_resetData(self.m, d)
        for n, v in qdict.items():
            d.qpos[self.m.jnt_qposadr[self.m.joint(n).id]] = v
        self.q_des[:] = d.qpos[self.qadr]
        if base_z is None:
            # place the robot so the lowest foot contact touches the floor
            mujoco.mj_forward(self.m, d)
            zmin = min(d.geom_xpos[g][2] - self.m.geom_size[g][0]
                       for g in range(self.m.ngeom) if "foot" in (self.m.geom(g).name or ""))
            d.qpos[2] -= zmin - 0.0005
        else:
            d.qpos[2] = base_z
        mujoco.mj_forward(self.m, d)

    def q(self):
        return self.d.qpos[self.qadr].copy()

    def qd(self):
        return self.d.qvel[self.dadr].copy()

    def control(self):
        d = self.d
        if not self.power:
            d.ctrl[:] = 0
            return
        e = self.q_des - d.qpos[self.qadr]
        ed = self.qd_des - d.qvel[self.dadr]
        tau = self.gain_scale * (self.kp * e + self.kd * ed) + self.tau_ff
        tau = np.clip(tau, -self.tau_lim, self.tau_lim)
        d.ctrl[self.aid] = tau

    def step(self, n=1):
        for _ in range(n):
            self.control()
            self.d.xfrc_applied[:] = 0
            if self.ext_force is not None:
                b, f, t_end = self.ext_force
                if self.d.time < t_end:
                    self.d.xfrc_applied[b, :3] = f
            mujoco.mj_step(self.m, self.d)

    def com(self):
        return self.d.subtree_com[self.pelvis].copy()

    def pelvis_height(self):
        return float(self.d.xpos[self.pelvis][2])

    def head_height(self):
        return float(self.d.xpos[self.m.body("neck_pitch_link").id][2])

    def upright(self):
        # z component of pelvis z-axis
        return float(self.d.xmat[self.pelvis].reshape(3, 3)[2, 2])
