import gmsh, sys, os, json
f, out = sys.argv[1], sys.argv[2]
gmsh.initialize(); gmsh.option.setNumber("General.Terminal", 0)
gmsh.option.setNumber("Mesh.Algorithm", 6)
gmsh.option.setNumber("Mesh.MeshSizeFromCurvature", 10)
gmsh.model.occ.importShapes(f); gmsh.model.occ.synchronize()
bb = gmsh.model.getBoundingBox(-1, -1)
diag = ((bb[3]-bb[0])**2 + (bb[4]-bb[1])**2 + (bb[5]-bb[2])**2) ** 0.5
gmsh.option.setNumber("Mesh.MeshSizeMax", max(diag / 20, 1.0))
gmsh.option.setNumber("Mesh.MeshSizeMin", max(diag / 300, 0.3))
gmsh.model.mesh.generate(2)
gmsh.write(out)
json.dump(dict(bbox=bb), open(out + ".json", "w"))
