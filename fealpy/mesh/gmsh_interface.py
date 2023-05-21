import gmsh
import numpy as np
from fealpy.mesh import TetrahedronMesh

def tra(mol):
    mol.occ.translate([(2, 1)], 0, 0, 1.59)

# Initialize gmsh
gmsh.initialize()

# Create a new model
model = gmsh.model

# Set the model's dimension to 3D
model.add("3D")

# Define the parameters
w = [1.3, 3.93, 89.6]
l = [12.32, 18.48, 1.3]

poly = [(2, 1)]
#ends = [(-89.6, 0)]
ends = [(0, 0)]
ends0 = []
for i in range(4):
    w[2] = 89.6/2**i + 1.3
    for p in ends:
        rec = model.occ.add_rectangle(p[0]-w[0]/2, p[1],           0, w[0], -l[0])
        model.occ.fuse(poly, [(2, rec)])
        if i < 3:
            rec = model.occ.add_rectangle(p[0]-w[1]/2, p[1]-l[0],      0, w[1], -l[1])
            model.occ.fuse(poly, [(2, rec)])
            rec = model.occ.add_rectangle(p[0]-w[2]/2, p[1]-l[0]-l[1], 0, w[2], -l[2])
            model.occ.fuse(poly, [(2, rec)])
            ends0.append((p[0]-89.6/2**(i+1), p[1]-l[0]-l[1]-l[2]))
            ends0.append((p[0]+89.6/2**(i+1), p[1]-l[0]-l[1]-l[2]))

    ends = ends0.copy()
    ends0 = []

#model.occ.translate([(2, 1)], 0, 0, 1.59)
tra(model)
model.occ.synchronize()
gmsh.fltk.initialize()
gmsh.fltk.run()

points = np.random.rand(10, 3)
cv, pv = model.getClosestPoint(2, 1, points.flatten())
print(cv)


# Finalize gmsh
gmsh.finalize()


