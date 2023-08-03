import numpy as np
import pyamg
import matplotlib.pyplot as plt
from scipy.sparse.linalg import spsolve

from fealpy.mesh import TriangleMesh 

from fealpy.geometry import SquareWithCircleHoleDomain
from fealpy.functionspace import LagrangeFESpace
from fealpy.fem import BilinearForm
from fealpy.fem import ProvidesSymmetricTangentOperatorIntegrator
from fealpy.fem import VectorSourceIntegrator
from fealpy.fem import LinearForm
from fealpy.fem import DirichletBC
from fealpy.fem import VectorNeumannBCIntegrator

import mgis.behaviour as mgis_bv
from fealpy.csm.mfront import compile_mfront_file

class ramberg_osgood_model():
    def __init__(self):
        self.E = 70000 # 杨氏模量
        self.nu = 0.2 # 泊松比
        self.n = 4.8
        self.alpha = 3/7
        self.sigma0 = 243 # 屈服应力 MPa

        self.mu = self.E / (1 + self.nu) / 2.0 # 剪切模量
        self.lam = self.E * self.mu / (1 + self.nu) / (1- 2*self.nu) # 拉梅常数

    def right_boundary_force(self):
        """
        @brief 受力情况
        Notes
        -----
        这里向量的第 i 个值表示第 i 个时间步力的大小
        """
        return np.concatenate((np.linspace(0, 133.65, 11)[1:], np.linspace(133.65,
            0, 11)[1:]))

    def right_force_direction(self):
        """
        @brief 右边界力的方向
        Notes
        -----
        位移方向沿 (1, 0) 方向，即向 x 的正向拉伸
        """
        return np.array([1, 0], np.float_)

    def is_right_boundary(self, p):
        """
        @brief 标记右边界, x = 1 时的边界点
        """
        return np.abs(p[..., 0] - 1) < 1e-12 

    def is_dirichlet_boundary(self, p):
        """
        @brief 标记 dirchlet 边界
        Notes
        -----
        DirichletBC，位移为 0
        """
        return (np.abs(p[..., 0]) < 1e-12) | (np.abs(p[..., 1]) < 1e-12)

class ramberg_osgood_integrator():
    def __init__(self, mesh, model):
        self.E = model.E # 杨氏模量
        self.nu = model.nu # 泊松比
        self.n = model.n
        self.alpha = model.alpha
        self.sigma0 = model.sigma0 # 屈服应力 MPa
        self.mesh = mesh

    def build_mfront(self):
        file = compile_mfront_file('material/nonlinear_elasticity_ramberg_osgood_type.mfront')

        lib = "./libnonlinear_elasticity_ramberg_osgood_type.so"  
        h = mgis_bv.Hypothesis.PlaneStrain # 平面

        # 加载行为模型
        self._b = mgis_bv.load(lib, "RambergOsgoodNonLinearElasticity", h)

    def update_mfront(self, uh):
        b = self._b
        NC = self.mesh.number_of_cells()
        m = mgis_bv.MaterialDataManager(b, NC)
        self._m = m

        mgis_bv.setMaterialProperty(m.s1, "YoungModulus", self.E) # 设置材料属性
        mgis_bv.setMaterialProperty(m.s1, "PoissonRatio", self.nu)
        mgis_bv.setMaterialProperty(m.s1, "n", self.n)
        mgis_bv.setMaterialProperty(m.s1, "alpha", self.alpha)
        mgis_bv.setMaterialProperty(m.s1, "YieldStrength", self.sigma0)
        mgis_bv.setExternalStateVariable(m.s1, "Temperature", 293.15)
        mgis_bv.update(m) # 更新材料数据
        eto = self.strain(uh)
        for i in range(NC):
            m.s1.gradients[i:] = eto[i:, :]

    def tangent_matrix(self):
        m = self._m
        it = mgis_bv.IntegrationType.IntegrationWithTangentOperator
        dt = 0.1 
        mgis_bv.integrate(m, it, dt, 0, m.n)
        sigma = m.s1.thermodynamic_forces
        M = m.K
        print("MMMM:", M)

        #将切算子矩阵转化为felapy中默认的储存顺序
        M = np.delete(M, 1, axis=1)
        M = np.delete(M, 1, axis=2)
        M[:, -1, -1] = M[:, -1, -1]/2
#        M[:, [1, 2], :] = M[:, [2, 1], :]
#        M[:, :, [1, 2]] = M[:, :, [2, 1]]
        return M

    def source(self):
        m = self._m
        sigma = m.s1.thermodynamic_forces
        n = sigma.shape[0]
        
        #将应力转化为felapy中默认的储存顺序, 即二维情况应力默认储存为 NC*2*2 的矩阵
        s = np.zeros((n, 2, 2), dtype=np.float_)
        s[:, 0, 0] = sigma[:, 0]
        s[:, 0, 1] = sigma[:, 3]
        s[:, 1, 0] = sigma[:, 3]
        s[:, 1, 1] = sigma[:, 1]
        return s

    def strain(self, uh):
        """
        @brief 给定一个位移，计算相应的应变，这里假设是线性元
        
        ____
        Notes:
        mfront 二维平面应变的默认保存顺序为[0, 0] [0, 1] [1, 0] [1, 1]
        ___
        """
        cell = self.mesh.entity('cell')
        NC = self.mesh.number_of_cells()
        gphi = self.mesh.grad_lambda()  # NC x 3 x 2

        s = np.zeros((NC, 4), dtype=np.float64)
        s[:, 0] = np.sum(uh[:, 0][cell] * gphi[:, :, 0], axis=-1)
        s[:, 1] = np.sum(uh[:, 1][cell] * gphi[:, :, 1], axis=-1)

        val = np.sum(uh[:, 0][cell] * gphi[:, :, 1], axis=-1)
        val += np.sum(uh[:, 1][cell] * gphi[:, :, 0], axis=-1)
        val /= 2.0
        s[:, 3] = val
        s[:, 0] = 0
        return s

model = ramberg_osgood_model()

domain = SquareWithCircleHoleDomain() 
mesh = TriangleMesh.from_domain_distmesh(domain, 0.05, maxit=100)

GD = mesh.geo_dimension()
space = LagrangeFESpace(mesh, p=1, doforder='vdims')
vspace  = GD*(space, )

uh = space.function(dim=GD)

integrator = ramberg_osgood_integrator(mesh, model)
integrator.build_mfront()
bform = BilinearForm(vspace)

integrator.update_mfront(uh)
D = integrator.tangent_matrix()
bform.add_domain_integrator(ProvidesSymmetricTangentOperatorIntegrator(D))
bform.assembly()
A = bform.get_matrix()

val = np.array([0.0, 0.0], dtype=np.float64)

# 构建单线性型，表示问题的源项
lform = LinearForm(vspace)
lform.add_domain_integrator(VectorSourceIntegrator(-integrator.source(), q=1))

neumann = model.right_force_direction()[0]*model.right_force_direction()
bi = VectorNeumannBCIntegrator(neumann, threshold=model.is_right_boundary, q=1)
lform.add_boundary_integrator(bi)
lform.assembly()

A = bform.get_matrix()
F = lform.get_vector()

bc = DirichletBC(vspace, val, threshold=model.is_dirichlet_boundary)
A, F = bc.apply(A, F, uh)
uh.flat[:] = spsolve(A, F)

fig = plt.figure()
axes = fig.add_subplot(111)
mesh.add_plot(axes)
plt.show()

