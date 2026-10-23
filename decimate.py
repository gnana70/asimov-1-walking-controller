import trimesh, fast_simplification, numpy as np, glob, os
import os as _os
from render_model import REPO, RENDER_MESHES
src = _os.path.join(REPO, "sim-model", "asimov_1_25dof", "meshes")
out = RENDER_MESHES
_os.makedirs(out, exist_ok=True)
tot0=tot1=0
for f in sorted(glob.glob(src+"/*.STL")):
    m = trimesh.load(f)
    n = len(m.faces); tot0+=n
    budget = 40000 if n > 150000 else 15000
    target = min(1.0, max(0.0, 1 - budget/n)) if n > budget else 0.0
    if target > 0:
        v, fc = fast_simplification.simplify(m.vertices.astype(np.float32), m.faces.astype(np.int32), target_reduction=target)
        m2 = trimesh.Trimesh(v, fc, process=False)
    else:
        m2 = m
    tot1 += len(m2.faces)
    m2.export(os.path.join(out, os.path.basename(f)))
print(tot0, "->", tot1)
