#!/usr/bin/env python3
# 

import argparse
import numpy as np
import matplotlib.pyplot as plt
from scipy.sparse.linalg import spsolve

from fealpy.pde.poisson_1d import CosData
from fealpy.mesh.interval_mesh import IntervalMesh 
from fealpy.functionspace import LagrangeFESpace
from fealpy.fem import DiffusionIntegrator # 计算单元矩阵 
from fealpy.fem import ScalarSourceIntegrator # 计算单元向量
from fealpy.fem import BilinearForm # 组装总矩阵
from fealpy.fem import LinearForm # 组装总向量
from fealpy.fem import DirichletBC # 处理边界条件

import ipdb

## 参数解析
parser = argparse.ArgumentParser(description=
        """
        IntervalMesh 上任意次有限元方法
        """)

parser.add_argument('--degree',
        default=1, type=int,
        help='Lagrange 有限元空间的次数, 默认为 1 次.')

parser.add_argument('--nx',
        default=4, type=int,
        help='初始网格剖分段数.')

parser.add_argument('--maxit',
        default=4, type=int,
        help='默认网格加密求解的次数, 默认加密求解 4 次')

args = parser.parse_args()

p = args.degree
nx =  args.nx
maxit = args.maxit


pde = CosData()
domain = pde.domain()

mesh = IntervalMesh.from_interval(domain, nx=nx)

errorType = ['$|| u - u_h||_{\Omega,0}$', 
        '$||\\nabla u - \\nabla u_h||_{\Omega, 0}$']
errorMatrix = np.zeros((2, maxit), dtype=np.float64)
NDof = np.zeros(maxit, dtype=np.int_)

for i in range(maxit):
    print("The {}-th computation:".format(i))
    space = LagrangeFESpace(mesh, p=p)
    NDof[i] = space.number_of_global_dofs()

    #ipdb.set_trace()
    bform = BilinearForm(space)
    bform.add_domain_integrator(DiffusionIntegrator(q=p+3))
    A = bform.assembly()

    lform = LinearForm(space)
    lform.add_domain_integrator(ScalarSourceIntegrator(pde.source, q=p+3))
    F = lform.assembly()

    bc = DirichletBC(space, pde.dirichlet) 
    uh = space.function() 
    A, F = bc.apply(A, F, uh)
    uh[:] = spsolve(A, F)

    errorMatrix[0, i] = mesh.error(pde.solution, uh, q=p+3)
    errorMatrix[1, i] = mesh.error(pde.gradient, uh.grad_value, q=p+3)

    if i < maxit-1:
        mesh.uniform_refine()

print(errorMatrix)
print(errorMatrix[:, 0:-1]/errorMatrix[:, 1:])
