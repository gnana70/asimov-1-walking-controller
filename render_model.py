"""Builds a styled copy of the Asimov 1 MJCF for rendering (same kinematics/qpos layout).

Visual meshes are decimated copies of the repo STLs; colours follow the real robot:
charcoal MJF PA12 nylon shells with gold anodised aluminium accents.
"""
import os
import mujoco
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
HERE_ = os.path.dirname(os.path.abspath(__file__))
REPO = os.environ.get("ASIMOV_REPO") or next((p for p in (os.path.join(HERE_, "asimov-1"), os.path.join(HERE_, "..", "asimov-1")) if os.path.isdir(p)), os.path.join(HERE_, "asimov-1"))
MJCF = os.path.join(REPO, "sim-model", "asimov_1_25dof", "xml", "asimov_1_25dof.xml")
RENDER_MESHES = os.path.join(HERE, "render_meshes")

GOLD_LINKS = {"left_shoulder_pitch_link", "right_shoulder_pitch_link", "left_elbow_link", "right_elbow_link",
              "left_hip_roll_link", "right_hip_roll_link", "left_ankle_pitch_link", "right_ankle_pitch_link",
              "neck_yaw_link"}

# subassembly codes from the repo naming convention (mechanical/naming_convention.png)
SUBASSEMBLY = {
    "pelvis_link": "200", "waist_yaw_link": "100",
    "neck_yaw_link": "700", "neck_pitch_link": "700",
}
for side, arm, leg in (("right", "300", "500"), ("left", "400", "600")):
    for l in ("shoulder_pitch", "shoulder_roll", "shoulder_yaw", "elbow", "wrist_yaw"):
        SUBASSEMBLY[f"{side}_{l}_link"] = arm
    for l in ("hip_pitch", "hip_roll", "hip_yaw", "knee", "ankle_pitch", "ankle_roll"):
        SUBASSEMBLY[f"{side}_{l}_link"] = leg


def styled_spec(floor="studio", width=1920, height=1080, mocap_parts=False):
    spec = mujoco.MjSpec.from_file(MJCF)
    spec.meshdir = RENDER_MESHES
    spec.visual.global_.offwidth = width
    spec.visual.global_.offheight = height
    spec.visual.quality.shadowsize = 4096
    spec.visual.quality.offsamples = 8
    spec.visual.headlight.ambient = [0.16, 0.16, 0.17]
    spec.visual.headlight.diffuse = [0.30, 0.30, 0.31]
    spec.visual.headlight.specular = [0.1, 0.1, 0.1]
    spec.visual.rgba.haze = [0.03, 0.035, 0.045, 1]
    spec.visual.map.znear = 0.01
    spec.visual.map.zfar = 60
    spec.visual.map.haze = 0.6
    # remove default textures/materials we replace, keep mesh assets
    for t in list(spec.textures):
        spec.delete(t)
    for mtl in list(spec.materials):
        if mtl.name in ("groundplane", "visual_grey"):
            pass
    sky = spec.add_texture(name="sky", type=mujoco.mjtTexture.mjTEXTURE_SKYBOX,
                           builtin=mujoco.mjtBuiltin.mjBUILTIN_GRADIENT,
                           rgb1=[0.11, 0.12, 0.15], rgb2=[0.01, 0.01, 0.015], width=800, height=800)
    grid = spec.add_texture(name="grid", type=mujoco.mjtTexture.mjTEXTURE_2D,
                            builtin=mujoco.mjtBuiltin.mjBUILTIN_CHECKER, mark=mujoco.mjtMark.mjMARK_EDGE,
                            rgb1=[0.15, 0.155, 0.165], rgb2=[0.15, 0.155, 0.165], markrgb=[0.27, 0.28, 0.30],
                            width=512, height=512)
    for mtl in spec.materials:
        if mtl.name == "groundplane":
            mtl.textures = ["", "grid"] + [""] * 8 if False else mtl.textures
    gp = [m_ for m_ in spec.materials if m_.name == "groundplane"][0]
    tex = list(gp.textures)
    tex[mujoco.mjtTextureRole.mjTEXROLE_RGB] = "grid"
    gp.textures = tex
    gp.texrepeat = [8, 8]
    gp.reflectance = 0.07
    gp.rgba = [1, 1, 1, 1]
    shell = spec.add_material(name="shell", rgba=[0.20, 0.21, 0.235, 1], specular=0.45, shininess=0.55)
    gold = spec.add_material(name="gold", rgba=[0.74, 0.54, 0.28, 1], specular=0.9, shininess=0.8)
    hi = spec.add_material(name="highlight", rgba=[1.0, 0.42, 0.12, 1], specular=0.5, shininess=0.5,
                           emission=0.35)
    for g in spec.geoms:
        if g.name and g.name.endswith("_visual"):
            body = g.name[:-len("_visual")]
            if body == "pelvis":
                body = "pelvis_link"
            g.material = "gold" if body in GOLD_LINKS else "shell"
            g.rgba = [0.5, 0.5, 0.5, 1]  # default -> material colour is used
        if g.name == "floor":
            g.size = [30, 30, 0.05]
    # lights: replace the defaults with a key / fill / rim setup
    wb = spec.worldbody
    for l in list(wb.lights):
        spec.delete(l)
    pel = spec.body("pelvis_link") if not mocap_parts else None
    key_parent = pel if pel is not None else wb
    key_parent.add_light(name="key", pos=[1.6, -1.9, 3.0], dir=[-0.45, 0.5, -0.74],
                         type=mujoco.mjtLightType.mjLIGHT_SPOT, mode=mujoco.mjtCamLight.mjCAMLIGHT_TRACKCOM,
                         diffuse=[1.15, 1.1, 1.04], specular=[0.5, 0.5, 0.5], castshadow=True,
                         cutoff=55, exponent=2, attenuation=[1, 0, 0])
    wb.add_light(name="top", pos=[0, 0, 5], dir=[0, 0, -1], type=mujoco.mjtLightType.mjLIGHT_DIRECTIONAL,
                 diffuse=[0.28, 0.28, 0.3], specular=[0.05, 0.05, 0.05], castshadow=False)
    wb.add_light(name="fill", pos=[-3.0, 2.0, 3.0], dir=[0.6, -0.4, -0.6], type=mujoco.mjtLightType.mjLIGHT_DIRECTIONAL,
                 diffuse=[0.36, 0.40, 0.50], specular=[0.12, 0.12, 0.15], castshadow=False)
    wb.add_light(name="rim", pos=[-2.0, 0.0, 2.5], dir=[0.8, 0.2, -0.35], type=mujoco.mjtLightType.mjLIGHT_DIRECTIONAL,
                 diffuse=[0.8, 0.40, 0.17], specular=[0.9, 0.5, 0.25], castshadow=False)
    return spec


def styled_model(**kw):
    spec = styled_spec(**kw)
    return spec, spec.compile()


def set_link_material(model, link_names, material):
    mid = model.material(material).id
    for i in range(model.ngeom):
        n = model.geom(i).name
        if n.endswith("_visual") and n[:-7] in link_names:
            model.geom_matid[i] = mid
