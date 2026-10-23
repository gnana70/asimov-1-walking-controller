# Asimov 1 in MuJoCo — a walking controller for the open-source humanoid

Code from the Singularity Feed video **"I Built a $20,000 Open-Source Humanoid Robot for Free (in Simulation)"**.

It loads the **official** Asimov 1 simulation model from
[menloresearch/asimov-1](https://github.com/menloresearch/asimov-1) (25-DOF MJCF, tested at commit `96a6807`)
and adds what the public model doesn't ship with: something that keeps the robot standing and walking.

> This is an independent, classic model-based controller written for a simulation experiment.
> It is **not** Menlo's controller, and it has only been tested in simulation. Menlo trains
> reinforcement-learning walking policies and documents a careful sim-to-real process; do not
> run this on hardware.

## Quick start

```bash
git clone https://github.com/gnana70/asimov-1-walking-controller
cd asimov-1-walking-controller
git clone https://github.com/menloresearch/asimov-1        # the robot model, cloned inside this folder
pip install mujoco numpy scipy quadprog                     # Python 3.10+ (renders need requirements.txt)
python play.py walk                                         # opens MuJoCo's viewer: the 21-step walk from the video
```

Other modes: `python play.py turn | stand-push | pd-push | collapse`.
Or just look at the bare model: `python -m mujoco.viewer --mjcf=asimov-1/sim-model/asimov_1_25dof/xml/asimov_1_25dof.xml`.

## What's inside

| File | What it does |
|---|---|
| `asimov_sim.py` | Loads the MJCF, adds 25 torque motors (limits from the MJCF `actuatorfrcrange`), joint friction from the Asimov 1 actuator table, optional joint PD ("stiff springs"). |
| `wbc.py` | **The walking controller.** ZMP preview planner (linear inverted pendulum, 1.6 s look-ahead) + a whole-body quadratic program solved every 2 ms: floating-base dynamics, friction pyramids at the 4 sole corners of each stance foot, torque limits, CoM / pelvis-orientation / swing-foot / posture tasks. Output: `tau = M qdd + h - Jc^T f`. |
| `walking.py` | Footstep + ZMP preview planner and the inverse-kinematics solver (also the "first attempt" IK + PD walker shown falling in the video). |
| `scenarios.py` | Reproduces every simulation in the video and saves trajectories (`traj/*.npz`). |
| `push_eval.py` | Push-recovery test (largest 0.1 s shove survived, PD vs controller). |
| `render_*.py`, `decimate.py`, `step_to_stl.py` | Offline renders used for the video (MuJoCo EGL; set `MUJOCO_GL=egl` on Linux). |

## Results in the video (MuJoCo 3.16, 2 ms step)

| Test | Result |
|---|---|
| Model mass | 35.55 kg (spec: 35 kg) |
| Motors off | head 1.10 m → floor in ≈ 1.0 s |
| Joint PD only, 0.1 s shove from the front | survives up to 40 N |
| Whole-body controller, same shove | survives up to 90 N |
| Straight walk, 0.15 m steps | 21 steps, ≈ 3 m, no fall |
| Curved walk | 23 steps, 80° turn, no fall |

## Why it was hard (things the model taught me)

* Each foot touches the ground at only 4 small spheres: 16.7 cm long, 4.0–5.6 cm wide.
* Ankle roll is limited to ±0.1 rad (5.7°) in the MJCF, so the pelvis can only shift ≈ 4.5 cm sideways
  with both feet flat — short, quick steps (0.28 s single support) keep the sideways sway small enough.
* An IK + joint-PD walker ignores angular momentum and tips over backwards; regulating pelvis orientation
  inside the whole-body QP (and giving the CoM task high priority) fixed it.

## License

This controller code: MIT (see `LICENSE`). The Asimov 1 model and CAD are **not** included here; they are
Menlo Research's, licensed CERN-OHL-S-2.0 (hardware) and GPL-2.0 (software) in their repository.
