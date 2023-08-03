import numpy as np
from scipy.sparse import coo_matrix, csc_matrix, csr_matrix, spdiags, eye

from ..functionspace import ConformingScalarVESpace2d
from ..functionspace import NonConformingScalarVESpace2d

class ConformingScalarVEMLaplaceIntegrator2d():
    def __init__(self, PI1, G, D, c=None):
        self.coef = c
        self.PI1 = PI1
        self.G = G
        self.D = D

    def assembly_cell_matrix(self, space: ConformingScalarVESpace2d):
        p = space.p
        mesh = space.mesh
        coef = self.coef

        def f(x):
            x[0, :] = 0
            return x

        if p == 1:
            tG = np.array([(0, 0, 0), (0, 1, 0), (0, 0, 1)])
            if coef is None:
                f1 = lambda x: x[1].T@tG@x[1] + (np.eye(x[1].shape[1]) - x[0]@x[1]).T@(np.eye(x[1].shape[1]) - x[0]@x[1])
                K = list(map(f1, zip(self.D, self.PI1)))
            else:
                pass
            
        else:
            tG = list(map(f, self.G))
            if coef is None:
                f1 = lambda x: x[1].T@x[2]@x[1] + (np.eye(x[1].shape[1]) - x[0]@x[1]).T@(np.eye(x[1].shape[1]) - x[0]@x[1])
                K = list(map(f1, zip(self.D, self.PI1, tG)))
            else:
                pass

        return K
       

class NonConformingScalarVEMLaplaceIntegrator2d():
    def __init__(self, PI1, G, D, c=None):
        self.coef = c
        self.PI1 = PI1
        self.G = G 
        self.D = D

    def assembly_cell_matrix(self, space: NonConformingScalarVESpace2d):
        """
        """

        def f(x):
            x[0, :] = 0
            return x
        tG = list(map(f, self.G)) # 注意这里把 G 修改掉了
        f1 = lambda x: x[1].T@x[2]@x[1] + (np.eye(x[1].shape[1]) - x[0]@x[1]).T@(np.eye(x[1].shape[1]) - x[0]@x[1])
        K = list(map(f1, zip(self.D, self.PI1, tG)))
        return K
