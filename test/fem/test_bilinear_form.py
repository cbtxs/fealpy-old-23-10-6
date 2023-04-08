import numpy as np
import matplotlib.pyplot as plt
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import spsolve
import pytest

from fealpy.fem import BilinearForm

def test_interval_mesh():
    from fealpy.pde.elliptic_1d import SinPDEData as PDE
    from fealpy.mesh import IntervalMesh
    from fealpy.functionspace import LagrangeFESpace as Space
    from fealpy.fem import DiffusionIntegrator
    from fealpy.fem import LinearForm
    from fealpy.fem import DirichletBC


    pde = PDE()
    domain = pde.domain()
    mesh = IntervalMesh.from_interval_domain(domain, nx=10)
    space = LagrangeFESpace(mesh, p=1)
    
    bform = BilinearForm(space)
    bform.add_domain_integrator(DiffusionIntegrator())
    bform.assembly()

    K = bform.M

    bc = DirichletBC(space, disp, threshold=idx)


def test_truss_structure():

    from fealpy.mesh import EdgeMesh
    from fealpy.functionspace import LagrangeFESpace as Space
    from fealpy.fem import TrussStructureIntegrator
    from fealpy.fem import DirichletBC
    
    mesh = EdgeMesh.from_tower()
    GD = mesh.geo_dimension()
    space = Space(mesh, p=1, doforder='vdims')

    bform = BilinearForm(GD*(space,))

    E = 1500 # 杨氏模量
    A = 2000 # 横截面积
    bform.add_domain_integrator(TrussStructureIntegrator(E, A))
    bform.assembly()

    K = bform.M

    uh = space.function(dim=GD)
    
    # 加载力的条件 
    F = np.zeros((uh.shape[0], GD), dtype=np.float64)
    idx, f = mesh.meshdata['force_bc']
    F[idx] = f 

    idx, disp = mesh.meshdata['disp_bc']
    bc = DirichletBC(space, disp, threshold=idx)
    A, F = bc.apply(K, F.flat, uh)

    uh.flat[:] = spsolve(A, F)
    print('uh:', uh)
    fig = plt.figure()
    axes = fig.add_subplot(1, 1, 1, projection='3d') 
    mesh.add_plot(axes)

    mesh.node += uh
    mesh.add_plot(axes, nodecolor='b', cellcolor='m')
    plt.show()


if __name__ == '__main__':
    test_truss_structure()



