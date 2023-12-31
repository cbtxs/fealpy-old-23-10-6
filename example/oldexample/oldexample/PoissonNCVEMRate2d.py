#!/usr/bin/env python3
#

import numpy as np

from fealpy.pde.poisson_2d import CosCosData
from fealpy.vem import PoissonNCVEMModel
from fealpy.tools.show import showmultirate
from fealpy.mesh import MeshFactory

import matplotlib.pyplot as plt


pde = CosCosData()
maxit = 5
n = 2
mesh = MeshFactory.boxmesh2d([0, 1, 0, 1], nx=n, ny=n, meshtype='poly')

Ndof = np.zeros((maxit,), dtype=np.int)
errorType = ['$|| u - \Pi^\\nabla u_h||_0$ with p=1',
             '$|| u - \Pi^\\nabla u_h||_0$ with p=2',
             '$|| u - \Pi^\\nabla u_h||_0$ with p=5',
             '$||\\nabla u - \\nabla \Pi^\\nabla u_h||_0$ with p=1',
             '$||\\nabla u - \\nabla \Pi^\\nabla u_h||_0$ with p=2',
             '$||\\nabla u - \\nabla \Pi^\\nabla u_h||_0$ with p=5',
             ]

ps = [1, 2, 5]

errorMatrix = np.zeros((len(errorType), maxit), dtype=np.float)

for i in range(maxit):
    for j, p in enumerate(ps):
        vem = PoissonNCVEMModel(pde, mesh, p, q=7)
        vem.solve()
        Ndof[i] = mesh.number_of_cells()
        errorMatrix[j, i] = vem.L2_error()
        errorMatrix[j+3, i] = vem.H1_semi_error()
    if i < maxit - 1:
        n *= 2
        mesh = MeshFactory.boxmesh2d([0, 1, 0, 1], nx=n, ny=n, meshtype='poly')

print(errorMatrix)
mesh.add_plot(plt, cellcolor='w')
showmultirate(plt, 0, Ndof, errorMatrix, errorType)
plt.show()


