import numpy as np
from scipy.sparse import csr_matrix
from numpy.linalg import inv
from fealpy.functionspace.Function import Function
from fealpy.decorator import cartesian, barycentric
from fealpy.common import ranges

from fealpy.functionspace.femdof import multi_index_matrix2d, multi_index_matrix1d
from fealpy.functionspace.LagrangeFiniteElementSpace import LagrangeFiniteElementSpace
from fealpy.functionspace.scaled_monomial_space_2d import ScaledMonomialSpace2d 
class VMDof2d():
    """
    缩放单项式空间自由度管理类
    """
    def __init__(self, mesh, p):
        self.mesh = mesh
        self.p = p # 默认的空间次数
        self.multiIndex = self.multi_index_matrix() # 默认的多重指标

    def multi_index_matrix(self, p=None):
        """
        Compute the natural correspondence from the one-dimensional index
        starting from 0.

        Notes
        -----

        0<-->(0, 0), 1<-->(1, 0), 2<-->(0, 1), 3<-->(2, 0), 4<-->(1, 1),
        5<-->(0, 2), .....

        """
        ldof = self.number_of_local_dofs(p=p, doftype='cell')
        idx = np.arange(0, ldof)
        idx0 = np.floor((-1 + np.sqrt(1 + 8*idx))/2)
        multiIndex = np.zeros((ldof, 2), dtype=np.int_)
        multiIndex[:, 1] = idx - idx0*(idx0 + 1)/2
        multiIndex[:, 0] = idx0 - multiIndex[:, 1]
        return multiIndex

    def number_of_local_dofs(self, p=None, doftype='cell'):
        p = self.p if p is None else p
        if doftype in {'cell', 2}:
            return (p+1)*(p+2)
        elif doftype in {'face', 'edge', 1}:
            return (p+1) #需要改
        elif doftype in {'node', 0}:
            return 0 


class VectorMonomialSpace2d():
    def __init__(self, mesh, p, q=None, bc=None):
        """
        The Scaled Momomial Space in R^2
        """

        self.mesh = mesh
        self.cellbarycenter = mesh.entity_barycenter('cell') if bc is None else bc
        self.p = p
        self.cellmeasure = mesh.entity_measure('cell')
        self.cellsize = np.sqrt(self.cellmeasure)
        self.dof = VMDof2d(mesh, p)
        self.GD = 2
        self.smspace = ScaledMonomialSpace2d(mesh, p, q=q, bc=bc)

        q = q if q is not None else p+3

        mtype = mesh.meshtype
        self.itype = self.mesh.itype
        self.ftype = self.mesh.ftype


    def diff_index_1(self, p=None):
        """

        Notes
        -----
        对基函数求一阶导后非零项的编号，及系数
        """
        p = self.p if p is None else p
        index = multi_index_matrix2d(p)

        x, = np.nonzero(index[:, 1] > 0) # 关于 x 求导非零的缩放单项式编号
        y, = np.nonzero(index[:, 2] > 0) # 关于 y 求导非零的缩放单项式编号

        return {'x':(x, index[x, 1]),
                'y':(y, index[y, 2]),
                }

    def diff_index_2(self, p=None):
        """

        Notes
        -----
        对基函数求二阶导后非零项的编号，及系数
        """
        p = self.p if p is None else p
        index = multi_index_matrix2d(p)

        xx, = np.nonzero(index[:, 1] > 1)
        yy, = np.nonzero(index[:, 2] > 1)

        xy, = np.nonzero((index[:, 1] > 0) & (index[:, 2] > 0))

        return {'xx':(xx, index[xx, 1]*(index[xx, 1]-1)),
                'yy':(yy, index[yy, 2]*(index[yy, 2]-1)),
                'xy':(xy, index[xy, 1]*index[xy, 2]),
                }


    def geo_dimension(self):
        return self.GD



    @cartesian
    def basis(self, point, index=np.s_[:], p=None):

        smphi = self.smspace.basis(point, index=index, p=p)
        ldof = self.number_of_local_dofs(p=p, doftype='cell')
        shape = point.shape[:-1] + (ldof, 2)
        phi = np.zeros(shape, dtype=self.ftype)
        phi[..., :ldof//2, 0] = smphi
        phi[..., -ldof//2:, 1] = smphi
        return phi

    @cartesian
    def grad_basis(self, point, index=np.s_[:], p=None, scaled=True):
        """

        p >= 0
        """
        smgphi = self.smspace.grad_basis(point, index=index, p=p)
        ldof = self.number_of_local_dofs(p=p, doftype='cell')
        shape = point.shape[:-1]+(ldof, 2, 2) 
        gphi = np.zeros(shape, dtype=self.ftype)
        gphi[..., :ldof//2, : , 0] = smgphi
        gphi[..., -ldof//2:, : , 1] = smgphi
        return gphi

    @cartesian
    def laplace_basis(self, point, index=np.s_[:], p=None, scaled=True):
        smlphi = self.smspace.laplace_basis(point, index=index, p=p)
        ldof = self.number_of_local_dofs(p=p, doftype='cell')
        shape = point.shape[:-1] + (ldof, 2)
        lphi = np.zeros(shape, dtype=self.ftype)
        lphi[..., :ldof//2, 0] = smlphi
        lphi[..., -ldof//2:, 1] = smlphi
        return lphi


    def number_of_local_dofs(self, p=None, doftype='cell'):
        return self.dof.number_of_local_dofs(p=p, doftype=doftype)

