
import numpy as np
import sympy as sp

from fealpy.mesh import TriangleMesh
from fealpy.functionspace import BernsteinFiniteElementSpace
from fealpy.symcom import BernsteinFEMSpace 

def sp_to_np_function(f_sp, GD):
    l = ['l'+str(i) for i in range(GD+1)]
    return sp.lambdify(l, f_sp, "numpy")

mesh = TriangleMesh.from_box([0, 1, 0, 1], nx=1, ny=1)
space =  BernsteinFiniteElementSpace(mesh, p=5)
space_sym = BernsteinFEMSpace(2)

bc = np.array([[1/3, 1/6, 1/2]])
phi = space.basis(bc)
phi_sym = space_sym.basis(5)
phi_sym_list = []
for _phi in phi_sym:
    ff = sp_to_np_function(_phi, 2)
    phi_sym_list.append(ff(1/3, 1/6, 1/2))
print(phi)
print(phi_sym_list)



